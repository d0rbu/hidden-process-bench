import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hidden_process_bench.agent_preflight import audit_request
from hidden_process_bench.discovery import predict_histories
from hidden_process_bench.generators import HiddenProcess, load_process
from hidden_process_bench.metrics import parameter_recovery, sequence_kl, summarize
from hidden_process_bench.operator_structure import recognize
from hidden_process_bench.process import MixtureHMM
from hidden_process_bench.submissions import FrozenSubmission, file_manifest, verify_submission


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_score_arrays(directory: Path, descriptor: str) -> dict[str, Any]:
    record = read(directory / descriptor)
    raw = b"".join((directory / name).read_bytes() for name in record["parts"])
    assert len(raw) == record["bytes"]
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def test_campaigns_bind_registrations_sources_teacher_bounds_and_programs(tmp_path: Path) -> None:
    for name, expected, final in (
        ("agent-development-v2", 10, False),
        ("final-campaign-v2", 48, True),
    ):
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
            assert sha(root / "task-plan.json") == summary["private_plan_sha256"]
            assert (
                read(root / "task-preparation.json")["plan_sha256"]
                == summary["private_plan_sha256"]
            )
            assert sha(root / "frozen-protocol.json") == sha(
                Path("docs/research/final-campaign/protocol.json")
            )
            boundary = read(root / "complete-agent-boundary-audit.json")
            assert boundary["complete_terminal_registry_required"]
            assert boundary["started_agent_trials"] == len(boundary["trials"]) == 12
            assert not boundary["private_test_truth_or_scores_opened"]
            assert not boundary["method_hypotheses_or_actions_opened"]
            for trial in boundary["trials"]:
                assert trial["exact_frozen_prompt_budget_binary_catalog_and_sources"]
                assert trial["challenge_hashes_match_neutral_public_packet"]
                assert trial["neutral_metadata_and_array_field_allowlists_verified"]
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
        review = read(root / "structural-review.json")
        assert sha(root / "structural-review.json") == summary["structural_review_sha256"]
        assert review["registration_sha256"] == summary["registration_sha256"]
        assert len(review["trials"]) == expected
        linear_audit = None
        if final:
            assert sha(root / "structure-review-criteria.md") == review["criteria_sha256"]
            assert (
                sha(root / "linear-parameter-audit.json") == review["linear_parameter_audit_sha256"]
            )
            linear_audit = read(root / "linear-parameter-audit.json")
            assert linear_audit["registration_sha256"] == summary["registration_sha256"]
            assert len(linear_audit["trials"]) == 36
            assert not linear_audit["refitting"]
        for case_method, reviewed in review["trials"].items():
            case_id, method = case_method.split("/")
            row = next(
                item
                for item in summary["cases"]
                if (item["case"], item["method"]) == (mapping[case_id], method)
            )
            assert reviewed["submission_receipt_sha256"] == row["submission_receipt_sha256"]
            assert reviewed["assessment"] == row["structural_assessment"]
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
            if final:
                assert sha(directory / "teacher-checkpoint.pt") == report["checkpoint_sha256"]
                opaque = next(key for key, value in mapping.items() if value == teacher["case"])
                checked = next(item for item in audit["cases"] if item["id"] == opaque)
                assert checked["task_and_every_screening_candidate_recomputed_from_original_seeds"]
                assert sha(directory / "screening.json") == checked["screening_sha256"]
                assert read(directory / "screening.json")["accepted"]
                if checked["sampling_audit_sha256"] is not None:
                    assert sha(directory / "sampling.jsonl") == checked["sampling_audit_sha256"]
                    candidates = [
                        json.loads(line)
                        for line in (directory / "sampling.jsonl").read_text().splitlines()
                    ]
                    assert candidates[-1]["accepted"]
                    assert not any(item["accepted"] for item in candidates[:-1])
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
            if final and row["method"].startswith("agent_"):
                plan = read(directory / "agent-plan.json")
                for key in ("prompt", "budget", "binary_sha256", "catalog_sha256"):
                    assert plan[key] == protocol["agent"][key]
                assert not plan["oracle_feedback"]
                packet = root / row["case"] / "public" / row["access"]
                assert file_manifest(packet, max_bytes=64_000_000) == plan["challenge_sha256"]
                preflight_path = directory / "preflight.json"
                if preflight_path.exists():
                    preflight = read(preflight_path)
                    assert preflight["status"] == "passed"
                    assert preflight["real_discovery_model_calls"] == 0
                    captures = sorted(directory.glob("preflight-request-*.json"))
                    assert len(captures) == preflight["requests"] >= 1
                    for capture in captures:
                        audit_request(read(capture), plan["prompt"])
                else:
                    assert row["agent_trial"]["failed_during"] == "preflight"
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
            if final and not row["method"].startswith("agent_"):
                assert linear_audit is not None
                opaque = next(key for key, value in mapping.items() if value == row["case"])
                checked = next(
                    item
                    for item in linear_audit["trials"]
                    if (item["id"], item["method"]) == (opaque, row["method"])
                )
                assert checked["submission_receipt_sha256"] == frozen.receipt_sha256
                assert sha(unpacked / "parameters.npz") == checked["parameters_sha256"]
                if row["method"] != "operators":
                    with np.load(unpacked / "parameters.npz", allow_pickle=False) as arrays:
                        assert (
                            recognize(arrays["initial"], arrays["normalizer"], arrays["operators"])
                            == checked["family_evidence_recomputed_without_refitting"]
                        )
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
                if final:
                    inputs_path = root / row["case"]
                    inputs_record = read(inputs_path / (split + "-inputs.json"))
                    assert inputs_record["sha256"] == score["private_inputs_sha256"]
                    inputs = load_score_arrays(inputs_path, split + "-inputs.json")
                    predictions = load_score_arrays(directory, split + "-arrays.json")[
                        "probabilities"
                    ]
                    assert predictions.shape == inputs["oracle"].shape == (128, length, 16)
                    assert (
                        summarize(sequence_kl(inputs["oracle"], predictions))
                        == score["oracle_to_candidate"]
                    )
                    if split == "heldout":
                        assert (
                            summarize(sequence_kl(inputs["teacher"], predictions))
                            == score["teacher_to_candidate"]
                        )
                        assert (
                            summarize(sequence_kl(inputs["oracle"], inputs["teacher"]))
                            == score["oracle_to_teacher"]
                        )
                    assert (
                        float(np.abs(predictions - inputs["oracle"]).sum(axis=-1).max() / 2)
                        == (row[split + "_max_total_variation"])
                    )
            if row["heldout"]["status"] != "scored":
                continue
            witness = read(directory / "heldout-witness.json")
            source = load_process(root / row["case"] / "source-process.json")
            oracle = source.probabilities(np.asarray([witness["history"]], dtype=np.int64))[0, -1]
            np.testing.assert_allclose(oracle, witness["oracle_probabilities"], atol=1e-12, rtol=0)
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


def test_final_dependency_findings_replay_original_programs(tmp_path: Path) -> None:
    root = Path("docs/results/final-campaign-v2")
    summary = read(root / "summary.json")
    diagnostic = read(root / "structural-diagnostics.json")
    review = read(root / "structural-review.json")
    assert diagnostic["registration_sha256"] == summary["registration_sha256"]
    assert sha(root / "structural-diagnostics.json") == review["diagnostics_sha256"]
    assert diagnostic["diagnostics_after_all_48_registered"]
    assert not diagnostic["refitting_or_predictor_substitution"]
    mapping = read(root / "case-mapping.json")
    assert len(diagnostic["trials"]) == 48
    for result in diagnostic["trials"]:
        case = mapping[result["id"]]
        method = result["method"]
        row = next(
            item for item in summary["cases"] if (item["case"], item["method"]) == (case, method)
        )
        assert result["submission_receipt_sha256"] == row["submission_receipt_sha256"]
        assert result["source_process_sha256"] == sha(root / case / "source-process.json")
        if not method.startswith("agent_"):
            continue
        unpacked = tmp_path / case / method
        with zipfile.ZipFile(root / case / method / "submission.zip") as archive:
            archive.extractall(unpacked)
        frozen = FrozenSubmission(unpacked, row["submission_receipt_sha256"])
        source = load_process(root / case / "source-process.json")
        if "exhaustive_minimum_length" in result:
            truth = read(root / case / "source-process.json")
            lag = truth["lag"]
            histories = itertools.product(range(16), repeat=lag)
            matches = count = 0
            minimum, maximum = 1.0, 0.0
            batch = 0
            while values := list(itertools.islice(histories, 4096)):
                h = [list(value) for value in values]
                actual = predict_histories(frozen, h, 16, unpacked.parent / f"{method}-{batch}")
                expected = np.asarray(truth["alphabet"])[np.asarray(h)[:, -lag]]
                matches += int((actual.argmax(axis=1) == expected).sum())
                count += len(h)
                favored = actual[np.arange(len(h)), expected]
                minimum = min(minimum, float(favored.min()))
                maximum = max(maximum, float(favored.max()))
                batch += 1
            measured = result["exhaustive_minimum_length"]
            assert measured["history_length"] == lag
            assert measured["histories"] == measured["source_modal_matches"] == count == matches
            assert count == 16**lag
            assert measured["modal_probability_min"] == minimum
            assert measured["modal_probability_max"] == maximum
            assert row["structural_assessment"]["independent_structure_recovery_demonstrated"]
            assert not row["structural_assessment"]["exact_source_recovery_demonstrated"]
        for index, probe in enumerate(result.get("stack_probes", [])):
            h = probe["history"]
            actual = predict_histories(frozen, [h], 16, unpacked.parent / f"{method}-stack-{index}")
            oracle = source.probabilities(np.asarray([h], dtype=np.int64))[0, -1]
            np.testing.assert_allclose(actual[0], probe["candidate"], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(oracle, probe["oracle"])
            assert float(np.abs(actual[0] - oracle).sum() / 2) == pytest.approx(
                probe["total_variation"], abs=1e-12
            )
