import json
import sys
from pathlib import Path
from typing import cast

import pytest

from hidden_process_bench import method_study
from hidden_process_bench.config import ModelConfig, Shape
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.process import random_mixture


def public_run(tmp_path: Path) -> Path:
    run = tmp_path / "input"
    (run / "public").mkdir(parents=True)
    (run / "private").mkdir()
    TinyTransformer(ModelConfig(vocab=3, context=4, width=8, heads=2, layers=2)).save(
        run / "public/checkpoint.pt"
    )
    (run / "public/task.json").write_text(
        json.dumps({"schema_version": 1, "shape": {"components": 1, "states": 2, "vocab": 3}})
    )
    random_mixture(Shape(1, 2, 3), 18).save(run / "private/truth.json")
    (run / "report.json").write_text(json.dumps({"benchmark_eligible": False}))
    return run


def test_full_method_study_fixes_candidates_then_scores_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    run = public_run(tmp_path)
    out = tmp_path / "study"
    config = method_study.StudyConfig(
        sequences=6, steps=2, restarts=1, representation_steps=2, evaluation_sequences=8
    )
    report = method_study.run_study([run], out, config)
    case = json.loads((out / "case-000/evaluation.json").read_text())
    assert not case["benchmark_eligible"]
    assert case["checkpoint_qualified"] == case["benchmark_eligible"]
    assert not case["evaluation_scope"]["independent_structure_recovery_evaluated"]
    assert all(
        not item["evaluation_scope"]["family_discovery_evaluated"]
        for item in case["submissions"].values()
    )
    assert set(case["scores"]) == {
        "adam_samples",
        "adam_probabilities",
        "em_samples",
        "spectral_hmm",
        "belief_layer_0",
        "belief_layer_1",
        "belief_all_layers",
        "pca",
        "sae",
        "transcoder",
        "circuit",
    }
    assert all((out / f"case-000/{name}.json").is_file() for name in case["submissions"])
    assert case["submissions"]["adam_probabilities"]["observation_budget"]["forward_calls"] == 4
    assert (
        case["submissions"]["circuit"]["diagnostics"]["interchange"]["additional_forward_calls"]
        == 2
    )
    assert report["config"] == json.loads((out / "plan.json").read_text())["config"]
    with pytest.raises(FileExistsError):
        method_study.run_study([run], out, config)


def test_method_study_preserves_failure_receipt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        method_study.run_study(
            [tmp_path / "missing"], tmp_path / "failure", method_study.StudyConfig()
        )
    assert (
        "FileNotFoundError" in json.loads((tmp_path / "failure/failure.json").read_text())["error"]
    )


@pytest.mark.parametrize("vocab", [3, 4])
def test_public_extractor_validates_schema_and_vocabulary(tmp_path: Path, vocab: int) -> None:
    run = public_run(tmp_path)
    task = run / "public/task.json"
    payload = json.loads(task.read_text())
    payload["schema_version"] = 2 if vocab == 3 else 1
    payload["shape"]["vocab"] = vocab
    task.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"schema|vocabulary"):
        method_study.extract_methods(
            run / "public/checkpoint.pt", task, tmp_path / "output", method_study.StudyConfig()
        )


def test_cli_resolves_explicit_budgets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    received = []

    def capture(
        runs: list[Path], output: Path, config: method_study.StudyConfig
    ) -> dict[str, object]:
        received.append((runs, output, config))
        return {}

    monkeypatch.setattr(method_study, "run_study", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study",
            "--run",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "3",
            "--representation-steps",
            "5",
            "--restarts",
            "1",
        ],
    )
    method_study.main()
    assert cast(method_study.StudyConfig, received[0][2]).steps == 3
    with pytest.raises(ValueError, match="four"):
        method_study.StudyConfig(sequences=2)
