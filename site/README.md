# Interactive explainer

[Project](../README.md) · [Research design](../docs/research/design.md)

A static, browser-local explanation of the evaluation. Plain headings and short default
copy lead to an addition exception demo, then earlier process explorers and results; mathematical detail is collapsible.
There is no framework, bundler, model download, analytics, or external JavaScript.

## Preview and verify

From the repository root:

```bash
uv run --locked just site
```

Open `http://127.0.0.1:8847`. Serve **only `site/`**, never the repository root or an
experiment output directory. No private artifacts are needed by the server.

```bash
uv run --locked just site-data
uv run --locked just site-check
uv run --locked python -m pytest tests/test_site_data.py --no-cov
```

Node.js 22+ runs the JavaScript syntax and math checks; serving the website needs only
Python. The checks also run in the existing local pre-push gate, never on commit or in
GitHub Actions.

## Exception track

The opening task supplies a clean addition machine and asks for hidden wrong-answer
inputs. Its public development teacher, all 4,096 answers, two planted examples and
retained search results come only from committed receipts. The carry table visualizes
the supplied clean oracle, not a recovered circuit. The page distinguishes planted
inputs from the teacher's one unplanted error. No interpretation method has been
evaluated on this track. Node checks every clean carry trace across the pilot domain.

## Scope correction

The primary objective is independent process discovery, including family and structure.
The completed fresh campaign registered 48 independent programs. Agents recovered
both delayed rules' lag/permutation cores and partial stack/HMM organization; full
HMM recovery, equivalence, minimality and neural circuit recovery remain unestablished.
Earlier development trials are separate. A first restricted operator attempt failed
to recover the mechanisms. It selected its own dimension but retained a linear representation
prior, so it did not test unconstrained family discovery. The older fits and agent
prompts disclosed the HMM family and order; feature methods used a supplied HMM adapter. The site labels
these as informed fitting baselines. `scope_correction` in the exported JSON applies to
all historical data, whose raw receipts are unchanged. See the
[primary protocol](../docs/research/independent-discovery.md).

## What the page shows

- Six fresh cases with transformer and paired-agent prediction scores, separate scoped
  structural findings, and an optional table for all 36 programmed-method trials.
- Ten paired development agent trials, including four preflight failures, in collapsed
  details. The five-case earlier cohort is retained separately.
- A five-case family-neutral agent cohort, with one submitted program and four model-call
  timeouts, separate prediction and structural claims, and links to frozen evidence.
- Twenty-five further development runs across state geometry, learned representations
  and head ablations, with their restricted priors and query differences disclosed.
- Five qualified development cases scored using an independently submitted operator
  program, with transformer/program error and a separate longer-history score. The
  collapsed details state the method prior, failures and unfinished discovery scope.
- Three recorded generators: the qualified demo and both process sizes in the scaling
  pilot. Each is a four-component mixture. The process selector fixes a component for
  illustration; the Bayesian predictor still infers identity over all four.
- Optional delayed-token and stack walkthroughs use two already-public development
  processes. Their next-token probabilities come from the Python oracle. Replaying only
  the displayed memory must reproduce each frame's full-history prediction. The stack
  history is chosen to illustrate pushes, matching/mismatched closes and neutral tokens;
  it is not a new sampled benchmark result.
- A selected state's outgoing edges and emission probabilities. Labels are rounded;
  the expanded table retains four decimal places in percentages.
- Fresh browser samples, with an optional reveal of their emitting hidden states.
  Sampling emits first, then transitions. Extending a sequence preserves its prefix.
- An equivalent state-splitting control. Identical random draws produce identical
  tokens before and after splitting; incoming mass is divided among the copies.
  This control illustrates redundancy, including beyond the scorer's six-state limit;
  it is not a new training run or a newly qualified task.
- A compact informed five-family comparison across four qualified checkpoints, including two
  independent generators. The full table is expandable and retains the compute/access
  differences and the two-checkpoint limit on agent trials.
- Recorded recovery scores and all four scaling statuses. The scaling chart plots the
  **largest aggregate/component upper bound across both validation and test**, so a
  point clears the plotted threshold exactly when the recorded checkpoint qualified.

`scripts/build_site_data.py` regenerates matrices from already-public sampling audits,
checks the generator source against recorded provenance, and exports `data.json`.
No hidden challenge seeds or local experiment directories are read. Source receipt
hashes, pinned GitHub links, experiment revisions, and Python-oracle fixtures accompany
the snapshot. Python tests verify it reproduces the receipts; Node tests compare every
browser prediction and belief against NumPy, and check sampling and state equivalence.
The three displayed generators and original five result rows are development examples, not five
independent task draws. The two widths share each pilot process. The newer method-study
summary is exported from its own published receipt with a separate pinned revision.

## Cloudflare hosting

For a temporary public tunnel to the local preview:

```bash
cloudflared tunnel --url http://127.0.0.1:8847
```

Use the `trycloudflare.com` URL printed by the process. It changes when a new tunnel is
created and requires the local server and tunnel to remain running. See
[Cloudflare's Quick Tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

For a long-lived local session, serve an immutable copy of these six files with a user
service: `index.html`, `style.css`, `app.js`, `process.mjs`, `icon.svg`, and `data.json`.
Keep the tunnel in a separate user service. A checkout or branch switch should not
change the served release. Store operational receipts under ignored `outputs/`, and
verify the public files against the release bytes after publishing.

Browser review covers desktop and mobile widths, keyboard state/process selection,
focus retention, sequence controls, oracle and equivalence views, expanded details,
exact result values, and JavaScript errors. Public tunnel checks repeat representative
interactions and compare all served asset hashes with the immutable release.

The paired agent table uses [ten version-2 development trials](../docs/results/agent-development-v2.md).
It distinguishes recovered delayed-token structure, partial HMM organization and
preflight failures. Earlier agent and non-agent cohorts remain in collapsed details.
The exporter reads the completed final campaign's public summary as well. It exports
scores and structural assessments, with the full summary's hash, while leaving large
selection tables and trajectories in the linked receipt dataset. No private outputs or
unfinished trial is read by the site exporter.
