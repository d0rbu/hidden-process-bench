"""Family-neutral research agent: JSON code actions, separate numerical sandbox."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from hidden_process_bench import neural_numpy
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

For neural weights, you may use the provided family-neutral execution helper:
from neural import NeuralModel; model = NeuralModel(Path('/challenge')).
model.forward(int64_inputs_including_BOS) returns probabilities and layer traces,
with optional ablate_heads=((layer,head),) and ablate_mlps=(layer,). Report model.counts
for every instance and any additional execution you implement yourself. The helper
contains only the declared neural operations, no generating-process hypothesis.

Create your final files under /output/submission, including submission.json using
the neutral challenge contract. Its predictor must work independently from the
original neural model, without refitting or comparing evaluation rows. Include a
structural explanation and evidence, not just an accurate black-box predictor.
Write a small working candidate early and refine it incrementally. Keep structural
explanations concise and write evidence as you go. Submit once the mechanism is ready;
do not defer writing and testing all submission files to the last action.
Your numerical sandbox has 120 seconds wall time, 100 seconds CPU, 1 GB RAM and
8 MB persistent output per action. The entire trial has at most {steps} model actions
and {seconds:g} seconds wall time, with at most {model_seconds:g} seconds per model
call. No private scores or hints will be returned.
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
    model_seconds: float = 600

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
            "agent_protocol_version": 2,
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
    stage = "preflight"
    failure_stage = failure_type = None
    calls = valid_actions = 0
    history: list[dict[str, object]] = []
    work = output / "initial-work"
    work.mkdir()
    try:
        preflight(binary, output / "catalog.json", initial_prompt, output / "preflight")
        for step in range(budget.steps):
            stage = "budget"
            remaining = budget.seconds - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("agent wall budget exhausted")
            current = output / f"step-{step:02}"
            current.mkdir()
            prompt = (
                initial_prompt
                + f"\nThis is action {step + 1} of {budget.steps}; {remaining:.1f} seconds remain."
                + " A valid executable submission is required before this budget expires.\n"
                + "\nPrior actions and numerical results:\n"
                + json.dumps(history, ensure_ascii=True)
            )
            stage = "model"
            calls += 1
            answer = invoke(
                binary,
                output / "catalog.json",
                prompt,
                current / "model",
                timeout=min(budget.model_seconds, remaining),
            )
            valid_actions += 1
            write_json(current / "action.json", answer)
            program = current / "program"
            program.mkdir()
            (program / "main.py").write_text(RUNNER)
            shutil.copyfile(Path(neural_numpy.__file__), program / "neural.py")
            copy_bundle(work, program / "work")
            remaining = budget.seconds - (perf_counter() - started)
            stage = "budget"
            if remaining <= 0:
                raise TimeoutError("agent wall budget exhausted before numerical execution")
            stage = "numerical"
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
                stage = "submission"
                frozen = freeze_submission(work / "submission", output / "submission")
                break
        if frozen is None:
            stage = "budget"
            raise ValueError("agent did not produce a valid submission within its budget")
    except (ValueError, OSError, TimeoutError, subprocess.TimeoutExpired) as failure:
        error = str(failure)
        failure_stage, failure_type = stage, type(failure).__name__
    if file_manifest(challenge, max_bytes=64_000_000) != before:
        error = "challenge changed during trial"
        failure_stage, failure_type = "input_integrity", "ValueError"
        frozen = None
    write_json(
        output / "trial.json",
        {
            "agent_protocol_version": 2,
            "status": "submitted" if frozen is not None else "failed",
            "error": error,
            "receipt_sha256": None if frozen is None else frozen.receipt_sha256,
            "actions": len(history),
            "model_calls_attempted": calls,
            "valid_model_actions": valid_actions,
            "failed_during": failure_stage,
            "failure_type": failure_type,
            "seconds": perf_counter() - started,
            "independent_structure_recovery": None,
            "causal_recovery": None,
        },
    )
    return frozen
