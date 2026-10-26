import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from hidden_process_bench.cli import main
from hidden_process_bench.config import RunConfig, SamplingConfig
from hidden_process_bench.experiment import extract_checkpoint, predictions, run, score, train
from hidden_process_bench.metrics import sequence_kl
from hidden_process_bench.model import TinyTransformer, observe
from hidden_process_bench.process import MixtureHMM, random_mixture
from hidden_process_bench.recovery import fit_hmm


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("experiment") / "smoke"
    run(RunConfig.load(Path("configs/smoke.toml")), output)
    return output


def test_smoke_records_unqualified_results_and_complete_provenance(smoke_run: Path) -> None:
    report = json.loads((smoke_run / "report.json").read_text())
    assert not report["benchmark_eligible"]
    assert report["checkpoint_qualified"] == report["benchmark_eligible"]
    scope = report["evaluation_scope"]
    assert scope["track"] == "informed_hmm_parameter_fit"
    assert scope["qualification_scope"] == "transformer_prediction_only"
    assert not scope["independent_structure_recovery_evaluated"]
    assert not scope["family_discovery_evaluated"]
    assert scope["disclosed_components"] == 4
    assert scope["disclosed_states_per_component"] == 3
    assert report["interpretation"] == "diagnostic_only_unqualified_checkpoint"
    assert set(report["recovery"]) == {"samples", "probabilities", "activations"}
    assert report["training"]["steps"] == 2
    assert report["shared_extraction_budget"]["token_queries_per_method"] == 48
    for method, result in report["recovery"].items():
        assert not result["access"]["ground_truth"]
        assert (smoke_run / "recovered" / f"{method}.json").is_file()
    assert (
        report["checkpoint_sha256"]
        == hashlib.sha256((smoke_run / "public/checkpoint.pt").read_bytes()).hexdigest()
    )
    provenance = json.loads((smoke_run / "private/provenance.json").read_text())
    assert provenance["device"] == "cpu"
    assert len(provenance["source_sha256"]["src/hidden_process_bench/process.py"]) == 64
    assert set(json.loads((smoke_run / "public/task.json").read_text())) == {
        "schema_version",
        "shape",
        "evaluation_scope",
    }
    assert not (smoke_run / "public/truth.json").exists()
    assert not (smoke_run / "public/config.json").exists()
    assert not (smoke_run / "public/sampling.jsonl").exists()
    sampling = [
        json.loads(line) for line in (smoke_run / "private/sampling.jsonl").read_text().splitlines()
    ]
    assert report["sampling"]["attempts"] == len(sampling)
    assert report["sampling"]["difficulty"] == sampling[-1]["difficulty"]
    with pytest.raises(FileExistsError):
        run(RunConfig.load(Path("configs/smoke.toml")), smoke_run)


def test_sampling_failure_prevents_training_and_retains_audit(tmp_path: Path) -> None:
    output = tmp_path / "exhausted"
    config = replace(
        RunConfig.load(Path("configs/smoke.toml")),
        sampling=SamplingConfig(max_attempts=1, min_unigram_gain=100),
    )
    with (
        patch("hidden_process_bench.experiment.train") as training,
        pytest.raises(ValueError, match="sampling exhausted"),
    ):
        run(config, output)
    training.assert_not_called()
    assert (output / "private/config.json").exists()
    assert (output / "private/provenance.json").exists()
    assert len((output / "private/sampling.jsonl").read_text().splitlines()) == 1
    assert not (output / "public/checkpoint.pt").exists()
    assert not (output / "report.json").exists()


def test_standalone_extraction_and_scoring_need_no_training_data(
    smoke_run: Path, tmp_path: Path
) -> None:
    result = extract_checkpoint(
        smoke_run / "public/checkpoint.pt",
        smoke_run / "public/task.json",
        tmp_path / "candidate.json",
        "samples",
        4,
        2,
        1,
        100,
    )
    assert result["token_queries"] == 24
    scope = cast(dict[str, object], result["evaluation_scope"])
    assert scope["track"] == "informed_hmm_parameter_fit"
    assert not scope["family_discovery_evaluated"]
    truth = MixtureHMM.load(smoke_run / "private/truth.json")
    candidate = MixtureHMM.load(tmp_path / "candidate.json")
    metrics = score(truth, candidate, 16, 6, 102)
    assert set(metrics) == {"parameters", "predictive", "equivalence", "complexity"}
    assert candidate.shape == truth.shape
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_checkpoint(
            smoke_run / "public/checkpoint.pt",
            smoke_run / "public/task.json",
            tmp_path / "candidate.json",
            "samples",
            4,
            2,
            1,
            100,
        )


@pytest.mark.parametrize("failure", ["schema", "vocab", "method"])
def test_standalone_extraction_validates_public_artifacts(
    smoke_run: Path, tmp_path: Path, failure: str
) -> None:
    task = json.loads((smoke_run / "public/task.json").read_text())
    if failure == "schema":
        task["schema_version"] = 2
    if failure == "vocab":
        task["shape"]["vocab"] = 2
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match=r"schema|vocabularies|method"):
        extract_checkpoint(
            smoke_run / "public/checkpoint.pt",
            path,
            tmp_path / "out.json",
            "unknown" if failure == "method" else "samples",
            2,
            1,
            1,
            0,
        )


def test_training_learns_a_known_distribution_and_stops_on_validation(tmp_path: Path) -> None:
    truth = MixtureHMM(np.ones(1), np.ones((1, 1)), np.ones((1, 1, 1)), np.array([[[0.8, 0.2]]]))
    config = RunConfig(
        components=1,
        states=1,
        vocab=2,
        length=4,
        width=8,
        heads=2,
        layers=1,
        train_steps=50,
        eval_every=10,
        batch_size=32,
        eval_sequences=32,
        learning_rate=0.02,
    )
    torch.manual_seed(config.model_seed)
    untrained = TinyTransformer(config.model())
    test, _ = truth.sample(np.random.default_rng(123), 32, 4)
    oracle = truth.filter(test).predictions[:, :-1]
    before = sequence_kl(oracle, predictions(untrained, test)[0]).mean()
    trained, trace = train(config, truth, tmp_path / "trace.jsonl")
    after = sequence_kl(oracle, predictions(trained, test)[0]).mean()
    assert after < 0.03
    assert after < before
    assert trace[-1]["validation"]["qualified"]
    assert trace[-1]["step"] < config.train_steps
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == len(trace)


def test_train_handles_eval_interval_larger_than_budget(tmp_path: Path) -> None:
    config = replace(RunConfig.load(Path("configs/smoke.toml")), eval_every=10)
    _, trace = train(config, random_mixture(config.shape(), 7), tmp_path / "trace.jsonl")
    assert [row["step"] for row in trace] == [2]


def test_private_generator_seed_is_not_revealed_through_extraction_rng(tmp_path: Path) -> None:
    config = replace(
        RunConfig.load(Path("configs/smoke.toml")),
        seed=901,
        model_seed=105,
        observation_seed=123,
        recovery_seed=456,
    )
    with (
        patch("hidden_process_bench.experiment.observe", wraps=observe) as observed,
        patch("hidden_process_bench.experiment.fit_hmm", wraps=fit_hmm) as fitted,
    ):
        run(config, tmp_path / "independent-seeds")
    assert observed.call_args.args[-1] == 123
    assert [call.kwargs["seed"] for call in fitted.call_args_list] == [456, 456, 456]
    # Changing the hidden task does not change the seeds exposed to the extractor.
    other = replace(config, seed=777)
    assert other.observation_seed == config.observation_seed
    assert other.recovery_seed == config.recovery_seed


def test_cli_commands_and_qualification_exit_code(
    smoke_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "score",
                "--truth",
                str(smoke_run / "private/truth.json"),
                "--candidate",
                str(smoke_run / "recovered/probabilities.json"),
                "--sequences",
                "4",
                "--length",
                "3",
            ]
        )
        == 0
    )
    assert "predictive" in capsys.readouterr().out
    assert (
        main(
            [
                "extract",
                "--checkpoint",
                str(smoke_run / "public/checkpoint.pt"),
                "--task",
                str(smoke_run / "public/task.json"),
                "--out",
                str(tmp_path / "candidate.json"),
                "--method",
                "activations",
                "--steps",
                "1",
                "--sequences",
                "2",
                "--restarts",
                "1",
            ]
        )
        == 0
    )
    assert "activations" in capsys.readouterr().out
    assert (
        main(
            [
                "run",
                "--config",
                "configs/smoke.toml",
                "--out",
                str(tmp_path / "required"),
                "--require-qualified",
            ]
        )
        == 2
    )
    assert (tmp_path / "required/report.json").exists()
    assert main(["run", "--config", "configs/smoke.toml", "--out", str(tmp_path / "optional")]) == 0
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
