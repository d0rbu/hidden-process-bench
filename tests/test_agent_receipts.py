import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from hidden_process_bench.discovery import predict_histories
from hidden_process_bench.submissions import FrozenSubmission, file_manifest, verify_submission


def test_agent_results_bind_failures_source_and_registered_program(tmp_path: Path) -> None:
    root = Path("docs/results/agent-development")
    summary = json.loads((root / "summary.json").read_text())
    manifest = file_manifest(root)
    manifest.pop("manifest.json")
    assert manifest == json.loads((root / "manifest.json").read_text())
    assert summary["all_submissions_registered_before_any_private_scoring"]
    assert not summary["process_family_disclosed_to_method"]
    assert not summary["process_order_disclosed_to_method"]
    assert not summary["independent_structure_recovery_demonstrated"]
    assert summary["submitted"] == 1
    assert summary["failed"] == 4
    assert sum(row["model_calls_attempted"] for row in summary["cases"]) == 83
    with zipfile.ZipFile(root / "method.zip") as archive:
        assert {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
        } == summary["method_sha256"]
    for row in summary["cases"]:
        directory = root / row["id"]
        assert row["completed_model_calls"] == row["valid_model_actions"]
        assert (
            len(json.loads((directory / "actions.json").read_text())) == row["valid_model_actions"]
        )
        with zipfile.ZipFile(directory / "trajectory.zip") as trajectory:
            assert (
                len([x for x in trajectory.namelist() if x.endswith("action.json")])
                == row["valid_model_actions"]
            )
        if row["status"] == "failed":
            assert row["failure"] == "model_call_timeout_180_seconds"
            assert row["model_calls_attempted"] == row["completed_model_calls"] + 1
            assert row["heldout"] is row["long"] is None
            assert row["submission_receipt_sha256"] is None
            assert not (directory / "submission").exists()
            continue
        unpacked = tmp_path / row["id"]
        with zipfile.ZipFile(directory / "submission.zip") as archive:
            archive.extractall(unpacked)
        frozen = FrozenSubmission(unpacked, row["submission_receipt_sha256"])
        verify_submission(frozen)
        for split in ("heldout", "long"):
            score = json.loads((directory / (split + ".json")).read_text())
            assert score == row[split]
            assert score["submission_receipt_sha256"] == frozen.receipt_sha256
            witness = json.loads((directory / (split + "-witness.json")).read_text())
            actual = predict_histories(frozen, [witness["history"]], 16, tmp_path / split)[0]
            assert np.allclose(actual, witness["candidate_probabilities"], atol=1e-12, rtol=0)
            distance = np.abs(actual - witness["oracle_probabilities"]).sum() / 2
            assert abs(distance - witness["total_variation"]) < 1e-12
            assert abs(distance - row[split + "_max_total_variation"]) < 1e-12
