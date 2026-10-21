import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from hidden_process_bench.cli import main
from hidden_process_bench.config import RunConfig, Shape
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
    }
    assert not (smoke_run / "public/truth.json").exists()
    assert not (smoke_run / "public/config.json").exists()
    with pytest.raises(FileExistsError):
        run(RunConfig.load(Path("configs/smoke.toml")), smoke_run)


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
    truth = MixtureHMM.load(smoke_run / "private/truth.json")
    candidate = MixtureHMM.load(tmp_path / "candidate.json")
    metrics = score(truth, candidate, 16, 6, 102)
    assert set(metrics) == {"parameters", "predictive"}
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
    _, trace = train(config, random_mixture(Shape(4, 2, 8), 7), tmp_path / "trace.jsonl")
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
