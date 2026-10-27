"""Family-neutral research agent: JSON code actions, separate numerical sandbox."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from hidden_process_bench.agent_client import invoke
from hidden_process_bench.agent_preflight import preflight
from hidden_process_bench.config import check_count, check_positive
from hidden_process_bench.experiment import write_json
from hidden_process_bench.isolation import Limits, run_isolated
from hidden_process_bench.submissions import (
    FrozenSubmission,
    copy_bundle,
    file_manifest,
    freeze_submission,
)

PROMPT = """Infer a compact executable mechanism explaining the supplied neural sequence model.
Choose the mechanism family, structure and complexity yourself using the evidence.
Declare your prior knowledge, uncertainties and alternatives. No answer family or
latent-state count is supplied. Do not merely copy the neural network.

You control a numerical sandbox through JSON actions. action='run' executes your
Python code; action='submit' executes optional final code then freezes your files.
Use no native client tools. Your code has Python standard library and NumPy, no
network, evaluator, project, original conversation or other task contexts. Read
/challenge/instructions.txt and /challenge/challenge.json first. Task arrays live
only in /challenge. The working directory is /output; files there persist across
successful actions, but Python variables do not. Failed actions retain the previous
files. Print concise diagnostics you want to see in the next action.

Create your final files under /output/submission, including submission.json using
the neutral challenge contract. Its predictor must work independently from the
original neural model, without refitting or comparing evaluation rows. Include a
structural explanation and evidence, not just an accurate black-box predictor.
Your numerical sandbox has 120 seconds wall time, 100 seconds CPU, 1 GB RAM and
8 MB persistent output per action. The entire trial has at most {steps} model actions
and {seconds:g} seconds wall time. No private scores or hints will be returned.
"""

RUNNER = """import json, os, shutil, sys
from pathlib import Path
for path in Path('/program/work').iterdir():
    target = Path('/output') / path.name
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copyfile(path, target)
os.chdir('/output')
exec(compile(json.load(sys.stdin)['code'], '<analysis>', 'exec'), {'__name__': '__main__'})
"""


@dataclass(frozen=True)
class AgentBudget:
    steps: int = 20
    seconds: float = 1800
    model_seconds: float = 180

    def __post_init__(self) -> None:
        check_count(self.steps)
        check_positive(self.seconds)
        check_positive(self.model_seconds)


def run_agent(
    challenge: Path,
    binary: Path,
    catalog: Path,
    output: Path,
    *,
    budget: AgentBudget | None = None,
) -> FrozenSubmission | None:
    budget = AgentBudget() if budget is None else budget
    initial_prompt = PROMPT.format(**asdict(budget))
    output.mkdir(parents=True, exist_ok=False)
    before = file_manifest(challenge, max_bytes=64_000_000)
    copy_bundle(challenge, output / "challenge", 64_000_000)
    shutil.copyfile(catalog, output / "catalog.json")
    sources = output / "method-source"
    sources.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, sources / source.name)
    catalog_hash = hashlib.sha256((output / "catalog.json").read_bytes()).hexdigest()
    write_json(
        output / "plan.json",
        {
            "budget": asdict(budget),
            "prompt": initial_prompt,
            "challenge_sha256": before,
            "catalog_sha256": catalog_hash,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "method_sha256": file_manifest(sources),
            "oracle_feedback": False,
        },
    )
    started = perf_counter()
    frozen = None
    error = None
    history: list[dict[str, object]] = []
    work = output / "initial-work"
    work.mkdir()
    try:
        preflight(binary, output / "catalog.json", initial_prompt, output / "preflight")
        for step in range(budget.steps):
            remaining = budget.seconds - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("agent wall budget exhausted")
            current = output / f"step-{step:02}"
            current.mkdir()
            prompt = (
                initial_prompt
                + "\nPrior actions and numerical results:\n"
                + json.dumps(history, ensure_ascii=True)
            )
            answer = invoke(
                binary,
                output / "catalog.json",
                prompt,
                current / "model",
                timeout=min(budget.model_seconds, remaining),
            )
            write_json(current / "action.json", answer)
            program = current / "program"
            program.mkdir()
            (program / "main.py").write_text(RUNNER)
            copy_bundle(work, program / "work")
            remaining = budget.seconds - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("agent wall budget exhausted before numerical execution")
            execution = run_isolated(
                program,
                "main.py",
                {"code": answer["code"]},
                current / "analysis",
                challenge=output / "challenge",
                limits=Limits(wall_seconds=min(120, remaining), cpu_seconds=100),
            )
            history.append(
                {
                    "action": answer,
                    "execution": asdict(execution),
                    "stdout": (current / "analysis/stdout.txt").read_text()[:16_000],
                    "stderr": (current / "analysis/stderr.txt").read_text()[-8_000:],
                }
            )
            write_json(output / "history.json", {"actions": history})
            if execution.status != "success":
                continue
            work = current / "analysis/generated"
            if answer["action"] == "submit":
                frozen = freeze_submission(work / "submission", output / "submission")
                break
        if frozen is None:
            raise ValueError("agent did not produce a valid submission within its budget")
    except (ValueError, OSError, TimeoutError, subprocess.TimeoutExpired) as failure:
        error = str(failure)
    if file_manifest(challenge, max_bytes=64_000_000) != before:
        error = "challenge changed during trial"
        frozen = None
    write_json(
        output / "trial.json",
        {
            "status": "submitted" if frozen is not None else "failed",
            "error": error,
            "receipt_sha256": None if frozen is None else frozen.receipt_sha256,
            "actions": len(history),
            "seconds": perf_counter() - started,
            "independent_structure_recovery": None,
            "causal_recovery": None,
        },
    )
    return frozen
