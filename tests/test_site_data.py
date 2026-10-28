import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hidden_process_bench.generators import load_process
from scripts.build_site_data import build_payload


def test_published_site_data_reproduces_recorded_generators_and_results() -> None:
    payload = build_payload(Path("."))
    assert payload == json.loads(Path("site/data.json").read_text())
    assert len(payload["datasets"]) == 3
    scope = payload["scope_correction"]
    assert scope["track"] == "informed_hmm_parameter_fit"
    assert scope["process_family_disclosed"]
    assert scope["process_order_disclosed"]
    assert not scope["family_discovery_evaluated"]
    assert not scope["independent_structure_recovery_evaluated"]
    assert scope["primary_discovery_result_count"] == 0
    assert len(payload["runs"]) == 5
    attempt = payload["operator_attempt"]
    assert not attempt["unconstrained_family_discovery_evaluated"]
    assert not attempt["independent_structure_recovery_demonstrated"]
    assert len(attempt["cases"]) == 5
    agent = payload["agent_attempt"]
    assert not agent["process_family_disclosed_to_method"]
    assert not agent["process_order_disclosed_to_method"]
    assert agent["submitted"] == 1
    assert agent["failed"] == 4
    assert not agent["independent_structure_recovery_demonstrated"]
    paired = payload["paired_agent_attempt"]
    assert paired["submitted"] == 6
    assert paired["trials"] == 10
    assert paired["independent_structure_recovery_demonstrated"]
    assert not paired["final_campaign"]
    final = payload["final_attempt"]
    assert final["final_campaign"]
    assert final["submitted"] == final["trials"] == len(final["cases"]) == 48
    assert final["suite_campaign_ready"]
    assert not final["process_family_disclosed_to_methods"]
    assert not final["supplied_hmm_fitting_adapter"]
    assert (
        sum(
            row["structural_assessment"]["independent_structure_recovery_demonstrated"]
            for row in final["cases"]
        )
        == 4
    )
    assert not any(
        row["structural_assessment"]["equivalence_established"] for row in final["cases"]
    )
    discovery = payload["discovery_attempts"]
    assert discovery["trials"] == 25
    assert not discovery["supplied_hmm_fitting_adapter"]
    assert not discovery["single_joint_blind_25trial_campaign"]
    assert not discovery["independent_structure_recovery_demonstrated"]
    assert not payload["teacher_qualification"]["campaign_ready"]
    assert sum(run["qualified"] for run in payload["runs"][1:]) == 1
    for run in payload["runs"]:
        assert run["qualified"] == (run["gate_upper"] <= run["threshold"])


def test_rule_walkthrough_memory_preserves_exact_process_predictions() -> None:
    payload = build_payload(Path("."))
    assert [item["id"] for item in payload["rule_examples"]] == ["delayed", "stack"]
    for example in payload["rule_examples"]:
        relative = example["source_url"].split("/blob/main/")[1]
        assert relative in payload["input_sha256"]
        process = load_process(Path(relative))
        tokens = example["tokens"]
        expected = process.probabilities(np.asarray([tokens], dtype=np.int64))[0]
        np.testing.assert_array_equal(example["predictions"], expected)
        assert len(example["frames"]) == len(tokens) + 1 == 25
        for position, frame in enumerate(example["frames"]):
            # Replay only the displayed memory: it must retain everything the oracle needs.
            memory = np.asarray([frame["memory"]], dtype=np.int64)
            from_memory = process.probabilities(memory)[0, -1]
            np.testing.assert_array_equal(from_memory, expected[position])
            if frame["target"] is not None:
                assert expected[position, frame["target"]] == expected[position].max()
                assert 0 <= frame["read_index"] < len(frame["memory"])
            else:
                assert frame["read_index"] is None


@pytest.mark.parametrize("failure", ["source", "audit", "rule_source"])
def test_site_export_refuses_changed_generator_or_invalid_audit(
    tmp_path: Path, failure: str
) -> None:
    shutil.copytree("docs/results", tmp_path / "docs/results")
    source = tmp_path / "src/hidden_process_bench/process.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile("src/hidden_process_bench/process.py", source)
    if failure == "source":
        source.write_text("changed source\n")
    elif failure == "audit":
        audit = tmp_path / "docs/results/screened-cpu-demo/sampling.jsonl"
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        records[-1]["accepted"] = False
        audit.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    else:
        (source.parent / "generators.py").write_text("changed rule source\n")
    with pytest.raises(ValueError, match=r"source no longer matches|first accepted candidate"):
        build_payload(tmp_path)


def test_exception_demo_is_bound_to_published_teacher_and_clean_carry_oracle() -> None:
    payload = build_payload(Path("."))["exceptions"]
    summary = json.loads(Path("docs/results/exceptions-pilot-v1/summary.json").read_text())
    assert payload["searches"] == summary["searches"]
    assert not payload["circuit_recovery_established"]
    example = payload["example"]
    with np.load("docs/results/exceptions-pilot-v1/local-clean/census.npz") as census:
        assert example["teacher_answers"] == census["predictions"].tolist()
    for check in example["oracle_checks"]:
        total = sum(row["digit"] * 4**i for i, row in enumerate(check["trace"]))
        total += check["trace"][-1]["carry_out"] * 4**3
        assert total == sum(check["pair"])


def test_exception_method_table_reports_registered_trials_without_causal_success() -> None:
    payload = build_payload(Path("."))["exception_methods"]
    summary = json.loads(Path("docs/results/exception-methods-v1/summary.json").read_text())
    assert payload["aggregate"] == summary["aggregate"]
    assert payload["trials"] == 21
    assert payload["teacher_count"] == 3
    assert not payload["causal_recovery_established"]
