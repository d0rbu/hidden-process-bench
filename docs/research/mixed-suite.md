# Private mixed-process suite

[Documentation](../README.md) · [Discovery protocol](independent-discovery.md) · [Executable interface](../reference/discovery-runtime.md)

The evaluator can now generate fresh tasks, train matched transformers, qualify them
against exact oracles, and export only neutral neural inputs. Independent discovery
methods have not yet been evaluated on this suite. Qualification measures the teacher,
not whether a method recovered a mechanism.

## Generating mechanisms

| Private generator | Process | Pre-training screen | Qualification groups |
|---|---|---|---|
| HMM mixture | A component chosen per sequence, with hidden transitions and emissions | Existing temporal information, occupancy and numerical minimal-order screen | Every mixture component |
| Delayed rule | A token at lag 3–5 is privately permuted and copied, mixed with uniform noise | Two histories share their last two tokens but have different next distributions | All sequences |
| Stack rule | Two push/pop types and neutral tokens; matching closes pop, mismatches do nothing; no depth cap | Same observable dependency witness | Maximum depth 0–1, 2–3, and at least 4 |

All processes have full token support. A stack is reconstructed exactly from its
observed history, including noise tokens. Noise therefore does not require an
approximate oracle. Private token permutations conceal which symbols have which roles.

Generator names describe implementations, not exclusive mathematical families. A
delayed rule also has a finite-state representation. Finite-context observations can
admit several equivalent explanations; a family label alone cannot establish recovery.

HMM candidates vary both component/state counts (2–4 each, subject to vocabulary
size) and parameters. Each case accepts the first candidate passing the existing
screen, with at most 128 attempts. Every rejected shape, seed, measurement and reason
is retained. The rule screen records concrete histories and their exact probability
difference, requiring total variation at least 0.1. It makes no finite-state minimality
claim for a program or unbounded stack. All screens precede training and extraction.

## Freeze and train

```bash
uv run --locked python -m hidden_process_bench.mixed_suite prepare \
  --config configs/discovery-pilot.toml --out outputs/my-private-suite
uv run --locked python -m hidden_process_bench.mixed_suite train \
  --suite outputs/my-private-suite --timeout 300
```

`prepare` creates a new directory and refuses overwrites. It freezes the entire case
list, shuffled order, opaque IDs, resolved settings and separate random streams before
training. IDs and seeds come from independent system randomness. Generator, structure,
initialization, training, stopping validation, final test, observation, held-out scoring
and longer scoring streams are separate; none is derived from a public model seed.
The private source snapshot is checked against its recorded hashes. Training refuses
changed source, a modified plan, modified truth, or a reused training directory.

The default pilot has two cases per generator, vocabulary 16, context 24, width 64,
four heads and two layers for every case. Each has at most 4,000 steps, batch 64, AdamW
learning rate 0.003 and two CPU threads. The parent gives each child a 300-second cap.
`configs/discovery-smoke.toml` uses only two steps and is expected to miss qualification.

Training stops at the first validation check whose overall and every group’s upper
95% KL estimate are at most 0.03 nats/token. A separate final test must also pass all
those checks; groups with fewer than two examples cannot pass. No test feedback is
used to resume training. The receipt records the untrained baseline, actual steps,
training tokens, parameter count, validation/test arrays, checkpoint hash and time.

An optional `validation_margin` in (0, 1] tightens only the stopping-validation
cutoff to `max_kl * validation_margin`. Its default is 1, preserving previous runs.
For example, 0.8 requires every validation upper bound to clear 0.024 before stopping
when the independent final-test bound remains 0.03. A tighter validation target can
reduce near-cutoff test failures; every final test must still pass independently.
Declare the margin before generating/training a new whole suite. Do not resume an old
teacher using its failed test. Failed whole-suite attempts remain in the record.

Failed preparation, training errors, malformed receipts, timeouts and unqualified
checkpoints remain in the summary. No cases are silently replaced. `campaign_ready`
requires every frozen case to qualify. It never means independent recovery succeeded.
Changing the generator or budget requires a new frozen run with earlier failures kept.

## Information boundary and artifacts

| Path below the suite | Owner |
|---|---|
| `private/plan.json`, `private/preparation.json` | Frozen roster/settings/seeds and preparation outcomes |
| `private/source/`, `private/preparation-provenance.json` | Exact archived source, lockfile and hashes |
| `private/ID/process.json`, `screening.json`, `sampling.jsonl` | Truth and pre-training checks; HMM audit includes all rejections |
| `private/ID/training/` | Checkpoint, trace, qualification arrays, held-out/long scoring arrays and report |
| `private/ID/child-stdout.txt`, `child-stderr.txt` | Bounded child execution logs |
| `private/training-status.jsonl`, `training-summary.json` | Every case’s terminal status and campaign readiness |
| `public/ID/outputs/`, `weights/`, `internals/` | Permitted neural snapshots for a qualified checkpoint |

Only the selected `public/ID/ACCESS/` directory belongs in an extractor sandbox.
Never pass the suite root or the evaluator’s CLI output. A fresh model context is
also required for agent methods; the evaluator’s conversation knows the task mapping.
Output and internal conditions share exactly the same cached histories/probabilities.
Both weight conditions receive identical neural weights. There are no hidden-state
labels or structural hints in any of those snapshots.

Private `heldout.npz` contains independent scoring histories, oracle probabilities
and teacher probabilities. `long.npz` contains longer histories and oracle probabilities
only: a context-24 teacher is undefined on these context-72 inputs. This measures
extrapolation to the generating process, not teacher fidelity beyond its context.
The [generic scorer](../reference/discovery-runtime.md) can evaluate either archive
without supplying a process family to the method.

Keep fresh case mappings, truths and scores private until all method implementations
and submissions for the campaign are fixed. Publishing them turns those instances
into development data. Structural review, equivalent representations and causal tests
remain separate from predictive KL and teacher qualification.
