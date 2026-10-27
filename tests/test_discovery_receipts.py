import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from hidden_process_bench.discovery import predict_histories
from hidden_process_bench.submissions import FrozenSubmission, file_manifest, verify_submission


def test_development_cohorts_bind_every_program_and_private_witness(tmp_path: Path) -> None:
    root = Path("docs/results/discovery-development")
    summary = json.loads((root / "summary.json").read_text())
    manifest = file_manifest(root)
    manifest.pop("manifest.json")
    assert manifest == json.loads((root / "manifest.json").read_text())
    assert not summary["single_joint_blind_25trial_campaign"]
    assert not summary["supplied_hmm_fitting_adapter"]
    assert not summary["independent_structure_recovery_demonstrated"]
    assert len(summary["cases"]) == summary["trials"] == 25
    previous = json.loads(Path("docs/results/operator-development/summary.json").read_text())
    private_hashes = {
        row["id"]: row["heldout"]["private_inputs_sha256"] for row in previous["cases"]
    }
    for name, cohort in summary["cohorts"].items():
        assert cohort["all_submissions_before_own_cohort_private_scoring"]
        for mode, expected in cohort["method_sha256"].items():
            with zipfile.ZipFile(root / (name + "-method-" + mode + ".zip")) as archive:
                assert {
                    key: hashlib.sha256(archive.read(key)).hexdigest() for key in archive.namelist()
                } == expected
    pairs = set()
    for row in summary["cases"]:
        key = row["case"] + "-" + row["method"]
        assert key not in pairs
        pairs.add(key)
        assert row["status"] == "submitted"
        assert row["heldout"]["private_inputs_sha256"] == private_hashes[row["case"]]
        assert not row["structural_assessment"]["equivalence_established"]
        directory = root / row["case"] / row["method"]
        unpacked = tmp_path / key
        with zipfile.ZipFile(directory / "submission.zip") as archive:
            archive.extractall(unpacked)
        frozen = FrozenSubmission(unpacked, row["submission_receipt_sha256"])
        verify_submission(frozen)
        for split in ("heldout", "long"):
            assert row[split] == json.loads((directory / (split + ".json")).read_text())
            assert row[split]["submission_receipt_sha256"] == frozen.receipt_sha256
        witness = json.loads((directory / "heldout-witness.json").read_text())
        actual = predict_histories(frozen, [witness["history"]], 16, tmp_path / (key + "-predict"))[
            0
        ]
        assert np.allclose(actual, witness["candidate_probabilities"], atol=1e-12, rtol=0)
        distance = np.abs(actual - witness["oracle_probabilities"]).sum() / 2
        assert abs(distance - witness["total_variation"]) < 1e-12
        assert abs(distance - row["heldout_max_total_variation"]) < 1e-12
