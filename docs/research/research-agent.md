# Fresh-context research agent

[Documentation](../README.md) · [Primary protocol](independent-discovery.md) · [Executable contract](../reference/discovery-runtime.md)

This method chooses its own executable process family, dependencies and complexity.
The evaluator supplies an anonymous neural-model packet and a neutral submission
contract. It supplies no candidate constructor, process menu, state count, fitting
adapter, generator code or private feedback. The agent can propose a latent process
if its analysis motivates one; the evaluator does not propose it on the agent's behalf.

The method uses stock Codex 0.164.0 with `gpt-6.1-sol`, reasoning effort `xhigh`.
It returns JSON containing a Python action and a short note. A separate numerical
sandbox executes the code with only the public packet, NumPy and the standard library.
The next model call receives the neutral instructions and the recorded actions/results.
Each call starts a fresh ephemeral client; no parent conversation, repository context,
memory, skills or other task's history is supplied.

## Context and execution boundary

The trusted client has a sparse read-only filesystem and account authentication.
Its network access is needed for inference. Model-visible native filesystem, shell,
network, connector and delegation capabilities are removed using a restricted copy
of the same model's client catalog and explicit configuration. This changes client
context and capabilities, not the chosen model's weights. The client still advertises
one inert question tool; it is unavailable in noninteractive execution, forbidden by
the prompt, and any native tool activity fails the trial.

A preflight redirects the client to a local fake Responses endpoint. It records only
request bodies, never authentication headers, and performs no real model inference.
An exact allowlist checks the neutral instructions, anonymous environment, read-only
permission text and pinned inert-tool definition. Extra instructions, altered tool
descriptions or unknown content fail closed. Request-level instructions,
previous-response/conversation links and altered output schemas are rejected too.
Client telemetry is not treated as model-visible content. This pins the supported
client version; a client update needs a new audit. The live setup also checked the default provider's
rendered initial context against the same allowlist.

Model-written code runs in a separate networkless namespace and cgroup. That sandbox
has no client, account credentials, repository, evaluator or private process files.
Only files produced by successful actions persist. Failed actions retain their logs
and leave the previous workspace intact. This boundary also permits ordinary numerical
and programming errors without giving the agent scorer feedback.

Defaults are 20 model actions and 1,800 seconds for the entire trial, with 600 seconds
per model call. Each numerical action has at most 120 seconds wall time, 100 seconds
per-process CPU, 1 GB memory, two CPU cores and 8 MB of returned files/stdout. Remaining
trial time can shorten an action. The plan records actual budget overrides. A model
call's logs are bounded to 16 MB per file. These are execution bounds, not a claim of
matched compute with other methods; token usage remains in the client events.

Protocol version 1 used a 180-second model-call limit and several development calls
timed out before a final submission. Version 2 keeps the total action/wall budget,
reports the remaining action count and time on every call, and asks for an early
working candidate with concise documentation. It records attempted calls, valid
returned actions and the failure stage separately. These are operational changes,
not evidence of improved discovery. Old trials retain their original protocol and
timeouts; they are never automatically resumed or relabeled as successful results.

## Submission and interpretation

The agent writes its own predictor, parameters and structural explanation. The
neutral runtime freezes and registers those files before private scoring. Prediction
runs receive histories only, without the teacher or challenge. A family name or low
prediction error does not establish structural recovery. Review executable update
rules, alternative equivalent processes, extrapolation and access-specific evidence
separately. Output-only trials cannot identify which neural circuit implements a rule.

Method source snapshots, binary/catalog hashes, prompts, actions, numerical logs,
failures, elapsed time and the external registration hash are retained. No automatic
structure or causal success flag is assigned. A real constant-distribution fixture
completed four actions and froze a memoryless categorical predictor; this validates
the live path, not performance on trained benchmark teachers.

## Run locally

Use an installed, audited stock client binary and a local model catalog. Account
authentication remains outside repository artifacts. First prepare its restricted copy:

```bash
uv run --locked python - <<'PY'
from pathlib import Path
from hidden_process_bench.agent_client import restricted_catalog
from hidden_process_bench.experiment import write_json

write_json(
    Path("outputs/restricted-catalog.json"),
    restricted_catalog(Path("/path/to/model-catalog.json"), "gpt-6.1-sol"),
)
PY

uv run --locked python -m hidden_process_bench.discovery_cli agent \
  --challenge outputs/challenge --client /path/to/stock/codex \
  --catalog outputs/restricted-catalog.json --out outputs/agent-trial
```

Use `--steps`, `--seconds` and `--model-seconds` for explicit smaller budgets. Output
directories must be new. A failed or incomplete trial returns exit code 2 and retains
its receipt. Run all declared trials before inspecting private scores; development
cases cannot become a fresh final campaign after inspecting their results.
