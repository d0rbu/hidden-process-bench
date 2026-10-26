import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hidden_process_bench.experiment import score
from hidden_process_bench.process import MixtureHMM

ROOT = Path("docs/results/five-method-pilot")


def test_published_method_receipts_match_manifest_and_fixed_submissions() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    actual = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert actual == manifest
    for group in ["existing", "independent"]:
        report = json.loads((ROOT / group / "report.json").read_text())
        plan = json.loads((ROOT / group / "plan.json").read_text())
        assert not plan["provenance"]["working_tree_dirty"]
        assert plan["provenance"]["git_revision"] == "0481e282c083e137c4afae97345ac5297c41df27"
        for index, case in enumerate(report["cases"]):
            assert case["benchmark_eligible"]
            assert len(case["scores"]) == 11
            for name, submission in case["submissions"].items():
                path = ROOT / group / f"case-{index:03d}" / f"{name}.json"
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest() == submission["candidate_sha256"]
                )
                candidate = MixtureHMM.load(path)
                assert candidate.shape.components == 4
                assert candidate.shape.states == 3
                assert case["scores"][name]["equivalence"]["status"] == "different"


@pytest.mark.parametrize(
    ("group", "index"), [("existing", 0), ("existing", 1), ("independent", 0), ("independent", 1)]
)
def test_summary_matches_reports_and_independently_rescored_control(group: str, index: int) -> None:
    summary = json.loads((ROOT / "summary.json").read_text())
    report = json.loads((ROOT / group / "report.json").read_text())
    case = report["cases"][index]
    public = summary["cases"][index + (2 if group == "independent" else 0)]
    for name, values in case["scores"].items():
        assert public["methods"][name]["kl"] == values["predictive"]["mean_kl"]
        assert public["methods"][name]["mae"] == values["parameters"]["mean_parameter_mae"]
    folder = ROOT / group / f"case-{index:03d}"
    result = score(
        MixtureHMM.load(folder / "truth.json"),
        MixtureHMM.load(folder / "adam_probabilities.json"),
        512,
        24,
        917563,
    )
    assert result == case["scores"]["adam_probabilities"]
    if group == "existing":
        agent = json.loads((ROOT / "agent-evaluation/evaluation.json").read_text())["cases"][
            ["demo", "pilot"][index]
        ]
        for kind in ["static", "observations"]:
            assert public["methods"][f"agent_{kind}"]["kl"] == agent[kind]["predictive"]["mean_kl"]
            candidate = MixtureHMM.load(
                ROOT / f"agent-{kind}" / ["demo", "pilot"][index] / "candidate.json"
            )
            assert np.isfinite(candidate.emission).all()
