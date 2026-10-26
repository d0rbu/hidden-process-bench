import json
import shutil
from pathlib import Path

import pytest

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
    assert sum(run["qualified"] for run in payload["runs"][1:]) == 1
    for run in payload["runs"]:
        assert run["qualified"] == (run["gate_upper"] <= run["threshold"])


@pytest.mark.parametrize("failure", ["source", "audit"])
def test_site_export_refuses_changed_generator_or_invalid_audit(
    tmp_path: Path, failure: str
) -> None:
    shutil.copytree("docs/results", tmp_path / "docs/results")
    source = tmp_path / "src/hidden_process_bench/process.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile("src/hidden_process_bench/process.py", source)
    if failure == "source":
        source.write_text("changed source\n")
    else:
        audit = tmp_path / "docs/results/screened-cpu-demo/sampling.jsonl"
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        records[-1]["accepted"] = False
        audit.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match=r"source no longer matches|first accepted candidate"):
        build_payload(tmp_path)
