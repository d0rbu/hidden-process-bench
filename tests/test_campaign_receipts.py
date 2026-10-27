import hashlib
import itertools
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hidden_process_bench.discovery import predict_histories
from hidden_process_bench.generators import HiddenProcess, load_process
from hidden_process_bench.metrics import parameter_recovery
from hidden_process_bench.process import MixtureHMM
from hidden_process_bench.submissions import FrozenSubmission, verify_submission


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_campaigns_bind_registrations_sources_teacher_bounds_and_programs(tmp_path: Path) -> None:
    for name, expected, final in (("agent-development-v2", 10, False),):
        root = Path("docs/results") / name
        summary = read(root / "summary.json")
        manifest = {
            str(path.relative_to(root)): sha(path)
            for path in root.rglob("*")
            if path.is_file() and path != root / "manifest.json"
        }
        assert manifest == read(root / "manifest.json")
        assert summary["all_submissions_registered_before_any_private_scoring"]
        assert summary["method_sources_fixed_before_task_generation"] is final
        assert not summary["process_family_disclosed_to_methods"]
        assert not summary["process_order_disclosed_to_methods"]
        assert not summary["supplied_hmm_fitting_adapter"]
        assert summary["final_campaign"] is final
        assert len(summary["cases"]) == summary["trials"] == expected
        assert sha(root / "registration.json") == summary["registration_sha256"]
        protocol = read(root / "frozen-protocol.json")
        assert sha(root / "frozen-protocol.json") == summary["protocol_sha256"]
        with zipfile.ZipFile(root / "runtime-source.zip") as archive:
            assert {
                name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
            } == summary["runtime_source_sha256"]
        if final:
            assert summary["suite_campaign_ready"]
            for method in protocol["methods"]:
                if method["bundle"] is not None:
                    with zipfile.ZipFile(root / ("method-" + method["name"] + ".zip")) as archive:
                        assert {
                            key: hashlib.sha256(archive.read(key)).hexdigest()
                            for key in archive.namelist()
                        } == protocol["method_sha256"][method["name"]]
        mapping = read(root / "case-mapping.json")
        registered = read(root / "registration.json")["trials"]
        receipts = {
            (mapping[row["id"]], row.get("method", "agent_" + row["access"])): row[
                "submission_receipt_sha256" if final else "receipt_sha256"
            ]
            for row in registered
        }
        assert len(receipts) == expected
        assert receipts == {
            (row["case"], row["method"]): row["submission_receipt_sha256"]
            for row in summary["cases"]
        }
        audit = read(root / "qualification-audit.json")
        assert audit["all_audits_passed"]
        assert audit["registration_sha256"] == summary["registration_sha256"]
        assert len(audit["cases"]) == len(summary["teachers"]) == (6 if final else 5)
        for teacher in summary["teachers"]:
            directory = root / teacher["case"]
            report = read(directory / "teacher.json")
            assert sha(directory / "teacher.json") == teacher["teacher_receipt_sha256"]
            assert sha(directory / "source-process.json") == teacher["source_process_sha256"]
            assert (
                sha(directory / "qualification-stats.npz") == teacher["qualification_stats_sha256"]
            )
            assert report["checkpoint_qualified"]
            assert report["model_parameters"] == 103744
            with np.load(directory / "qualification-stats.npz", allow_pickle=False) as stats:
                for split in ("validation", "test"):
                    values = stats[split + "_kl"]
                    groups = stats[split + "_groups"]
                    assert np.isfinite(values).all()
                    assert (values >= 0).all()
                    measured = report[split]
                    threshold = 0.024 if final and split == "validation" else 0.03
                    assert measured["max_kl"] == threshold
                    assert measured["qualified"]
                    for index, expected_group in enumerate(
                        [measured["overall"], *measured["by_component"]]
                    ):
                        selected = values if index == 0 else values[groups == index - 1]
                        assert selected.size == expected_group["count"] >= 2
                        mean = float(selected.mean())
                        se = float(selected.std(ddof=1) / np.sqrt(selected.size))
                        assert mean == pytest.approx(expected_group["mean_kl"], abs=1e-15)
                        assert se == pytest.approx(expected_group["standard_error"], abs=1e-15)
                        bound = mean + 1.96 * se
                        assert bound == pytest.approx(expected_group["upper95_kl"], abs=1e-15)
                        assert bound <= threshold
        private_inputs: dict[tuple[str, str], str] = {}
        for row in summary["cases"]:
            directory = root / row["case"] / row["method"]
            if row["submission_receipt_sha256"] is None:
                assert row["status"] == "failed"
                assert row["heldout"] is row["long"] is None
                assert not (directory / "submission.zip").exists()
                continue
            assert row["status"] == "submitted"
            unpacked = tmp_path / name / row["case"] / row["method"]
            with zipfile.ZipFile(directory / "submission.zip") as archive:
                archive.extractall(unpacked)
            frozen = FrozenSubmission(unpacked, row["submission_receipt_sha256"])
            verify_submission(frozen)
            for split, length in (("heldout", 24), ("long", 72)):
                score = read(directory / (split + ".json"))
                assert score == row[split]
                assert score["submission_receipt_sha256"] == frozen.receipt_sha256
                key = (row["case"], split)
                previous = private_inputs.setdefault(key, score["private_inputs_sha256"])
                assert previous == score["private_inputs_sha256"]
                if score["status"] != "scored":
                    continue
                assert score["positions"] == length
                assert score["sequences"] == 128
                assert score["row_invariance_probes_passed"]
                if split == "long":
                    assert score["oracle_to_teacher"] is score["teacher_to_candidate"] is None
            if row["heldout"]["status"] != "scored":
                continue
            witness = read(directory / "heldout-witness.json")
            prediction = predict_histories(
                frozen, [witness["history"]], 16, unpacked.parent / (row["method"] + "-witness")
            )[0]
            np.testing.assert_allclose(
                prediction, witness["candidate_probabilities"], atol=1e-12, rtol=0
            )
            distance = float(np.abs(prediction - witness["oracle_probabilities"]).sum() / 2)
            assert distance == pytest.approx(witness["total_variation"], abs=1e-12)
            assert distance == pytest.approx(row["heldout_max_total_variation"], abs=1e-12)


def test_scoped_development_structure_findings_reproduce_from_registered_programs(
    tmp_path: Path,
) -> None:
    root = Path("docs/results/agent-development-v2")
    summary = read(root / "summary.json")
    assert sha(root / "structural-review.json") == summary["structural_review_sha256"]
    diagnostic = read(root / "lag-modal-diagnostic.json")
    assert diagnostic["registration_sha256"] == summary["registration_sha256"]
    truth = read(root / "case-01/source-process.json")
    assert truth["family"] == "lag_rule"
    assert truth["lag"] == 3
    histories = [list(values) for values in itertools.product(range(16), repeat=3)]
    expected = np.asarray(truth["alphabet"])[np.asarray(histories)[:, -3]]
    for result in diagnostic["cases"]:
        method = "agent_" + result["access"]
        row = next(
            entry
            for entry in summary["cases"]
            if entry["case"] == "case-01" and entry["method"] == method
        )
        assert row["structural_assessment"]["independent_structure_recovery_demonstrated"]
        assert not row["structural_assessment"]["equivalence_established"]
        assert result["submission_receipt_sha256"] == row["submission_receipt_sha256"]
        unpacked = tmp_path / method
        with zipfile.ZipFile(root / "case-01" / method / "submission.zip") as archive:
            archive.extractall(unpacked)
        frozen = FrozenSubmission(unpacked, row["submission_receipt_sha256"])
        actual = predict_histories(frozen, histories, 16, tmp_path / (method + "-modal"))
        np.testing.assert_array_equal(actual.argmax(axis=-1), expected)
        assert result["source_modal_permutation_matches"] == result["histories"] == 4096
        assert result["all_modal_matches"]
        assert result["source_modal_probability"] == pytest.approx(
            truth["strength"] + (1 - truth["strength"]) / 16
        )
        assert (
            np.max(np.abs(actual[np.arange(4096), expected] - result["source_modal_probability"]))
            > 0.1
        )
    core_diagnostic = read(root / "generative-core-diagnostic.json")
    row = next(
        entry
        for entry in summary["cases"]
        if entry["case"] == "case-03" and entry["method"] == "agent_internals"
    )
    assert core_diagnostic["submission_receipt_sha256"] == row["submission_receipt_sha256"]
    unpacked = tmp_path / "hmm-core"
    with zipfile.ZipFile(root / "case-03/agent_internals/submission.zip") as archive:
        archive.extractall(unpacked)
    verify_submission(FrozenSubmission(unpacked, row["submission_receipt_sha256"]))
    parameters = read(unpacked / "parameters.json")
    transition = np.asarray(parameters["A"])
    emission = np.asarray(parameters["E"])
    recognition = np.asarray(parameters["F"])
    correction = np.asarray(parameters["C"])
    initial = np.asarray(parameters["pi"])
    block = np.repeat(np.arange(4), 4)
    assert np.count_nonzero(transition[block[:, None] != block[None, :]]) == 0
    weights = initial.reshape(4, 4).sum(axis=-1)
    core = MixtureHMM(
        weights,
        initial.reshape(4, 4) / weights[:, None],
        np.stack([transition[k * 4 : (k + 1) * 4, k * 4 : (k + 1) * 4] for k in range(4)]),
        emission.reshape(4, 4, 16),
    )
    source = load_process(root / "case-03/source-process.json")
    assert isinstance(source, HiddenProcess)
    assert (
        parameter_recovery(source.hmm, core)
        == core_diagnostic["generative_core_parameter_alignment_only"]
    )
    assert (
        float(np.abs(recognition - emission).mean()) == core_diagnostic["recognition_emission_mae"]
    )
    assert float(np.abs(correction).max()) == core_diagnostic["maximum_output_logit_correction"]
    assert core_diagnostic["recognition_emission_mae"] > 0
    assert core_diagnostic["maximum_output_logit_correction"] > 0
