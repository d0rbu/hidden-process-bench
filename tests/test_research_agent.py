import json
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest

from hidden_process_bench import research_agent
from hidden_process_bench.agent_client import INSTRUCTIONS
from hidden_process_bench.research_agent import AgentBudget, run_agent
from hidden_process_bench.submissions import verify_submission


class BudgetValues(TypedDict, total=False):
    steps: int
    seconds: float
    model_seconds: float


SUBMIT = """from pathlib import Path
import json
root = Path('submission'); root.mkdir()
(root/'predict.py').write_text('import json, sys\\nr=json.load(sys.stdin)\\nprint(json.dumps({"probabilities":[[0.5,0.5] for _ in r["histories"]]}))')
(root/'mechanism.txt').write_text('Independent uniform observations; no memory.')
(root/'submission.json').write_text(json.dumps({
 'schema_version':1,'predictor':'predict.py','mechanism':'mechanism.txt',
 'hypothesis':'Independent observations.','assumptions':'Synthetic execution fixture.',
 'resources':{'queries':0}
}))
"""


def setup_trial(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    challenge = root / "public"
    challenge.mkdir()
    (challenge / "instructions.txt").write_text("Only public observations.")
    binary, catalog = root / "binary", root / "catalog"
    binary.write_text("test client bytes")
    catalog.write_text("test catalog bytes")

    def preflight(_binary: Path, _catalog: Path, prompt: str, _output: Path) -> dict[str, object]:
        assert "HMM" not in prompt
        assert "HMM" not in INSTRUCTIONS
        assert (
            root / "trial/challenge/instructions.txt"
        ).read_text() == "Only public observations."
        return {"status": "passed"}

    monkeypatch.setattr(research_agent, "preflight", preflight)
    return challenge, binary, catalog


def test_actions_persist_only_successful_files_and_freeze_chosen_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    challenge, binary, catalog = setup_trial(tmp_path, monkeypatch)
    codes = iter(
        [
            "from pathlib import Path; Path('state').write_text('before'); print('step one')",
            "from pathlib import Path; Path('state').write_text('corrupt'); raise ValueError('failed action')",
            "from pathlib import Path; assert Path('state').read_text()=='before'\n" + SUBMIT,
        ]
    )
    prompts = []

    def invoke(
        _binary: Path, _catalog: Path, prompt: str, _output: Path, *, timeout: float
    ) -> dict[str, object]:
        assert 0 < timeout <= 600
        prompts.append(prompt)
        return {
            "action": "submit" if len(prompts) == 3 else "run",
            "code": next(codes),
            "note": "test",
        }

    monkeypatch.setattr(research_agent, "invoke", invoke)
    frozen = run_agent(challenge, binary, catalog, tmp_path / "trial")
    assert frozen is not None
    assert verify_submission(frozen)["structure_assessment"] == "pending_independent_review"
    assert "step one" not in prompts[0]
    assert "step one" in prompts[1]
    assert "failed action" in prompts[2]
    history = json.loads((tmp_path / "trial/history.json").read_text())["actions"]
    assert [row["execution"]["status"] for row in history] == ["success", "failed", "success"]
    trial = json.loads((tmp_path / "trial/trial.json").read_text())
    assert trial["actions"] == 3
    assert trial["model_calls_attempted"] == 3
    assert trial["valid_model_actions"] == 3
    assert trial["failed_during"] is None
    assert "This is action 1 of 20" in prompts[0]
    assert "This is action 3 of 20" in prompts[2]
    assert trial["status"] == "submitted"
    assert trial["independent_structure_recovery"] is None
    assert trial["causal_recovery"] is None


@pytest.mark.parametrize(
    "mode",
    [
        "no_submission",
        "bad_submission",
        "model_error",
        "model_timeout",
        "preflight_error",
        "mutation",
    ],
)
def test_failed_trials_retain_receipts_without_scoring_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    challenge, binary, catalog = setup_trial(tmp_path, monkeypatch)

    def invoke(
        _binary: Path, _catalog: Path, _prompt: str, _output: Path, *, timeout: float
    ) -> dict[str, object]:
        assert timeout > 0
        if mode == "model_error":
            raise ValueError("model failed")
        if mode == "model_timeout":
            raise subprocess.TimeoutExpired(["synthetic-client"], timeout)
        if mode == "mutation":
            (challenge / "instructions.txt").write_text("changed")
        return {
            "action": "run" if mode == "no_submission" else "submit",
            "code": SUBMIT if mode == "mutation" else "print('ok')",
            "note": "test",
        }

    def fail_preflight(
        _binary: Path, _catalog: Path, _prompt: str, _output: Path
    ) -> dict[str, object]:
        raise ValueError("preflight rejected")

    monkeypatch.setattr(research_agent, "invoke", invoke)
    if mode == "preflight_error":
        monkeypatch.setattr(research_agent, "preflight", fail_preflight)
    assert (
        run_agent(challenge, binary, catalog, tmp_path / "trial", budget=AgentBudget(steps=1))
        is None
    )
    trial = json.loads((tmp_path / "trial/trial.json").read_text())
    assert trial["status"] == "failed"
    assert trial["error"]
    assert (
        trial["failed_during"]
        == {
            "preflight_error": "preflight",
            "model_error": "model",
            "model_timeout": "model",
            "no_submission": "budget",
            "bad_submission": "submission",
            "mutation": "input_integrity",
        }[mode]
    )
    plan = json.loads((tmp_path / "trial/plan.json").read_text())
    assert plan["oracle_feedback"] is False
    assert "at most 1 model actions" in plan["prompt"]


@pytest.mark.parametrize("late", [False, True])
def test_total_wall_budget_is_enforced_before_client_and_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late: bool
) -> None:
    challenge, binary, catalog = setup_trial(tmp_path, monkeypatch)
    times = iter([0, 1, 20, 20] if late else [0, 20, 20])
    monkeypatch.setattr(research_agent, "perf_counter", lambda: next(times))

    def invoke(
        _binary: Path, _catalog: Path, _prompt: str, _output: Path, *, timeout: float
    ) -> dict[str, object]:
        assert late
        assert timeout == 9
        return {"action": "submit", "code": SUBMIT, "note": "test"}

    monkeypatch.setattr(research_agent, "invoke", invoke)
    assert (
        run_agent(challenge, binary, catalog, tmp_path / "trial", budget=AgentBudget(seconds=10))
        is None
    )
    trial = json.loads((tmp_path / "trial/trial.json").read_text())
    assert "wall budget exhausted" in trial["error"]


@pytest.mark.parametrize(
    "values",
    [
        {"steps": 0},
        {"steps": True},
        {"seconds": float("nan")},
        {"seconds": float("inf")},
        {"seconds": 0},
        {"model_seconds": -1},
    ],
)
def test_nonpositive_or_nonfinite_agent_budgets_are_rejected(values: BudgetValues) -> None:
    with pytest.raises((ValueError, TypeError)):
        AgentBudget(**values)
