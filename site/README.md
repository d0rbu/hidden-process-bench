# Interactive explainer

[Project](../README.md) · [Research design](../docs/research/design.md)

A static, browser-local explanation of the evaluation. Plain headings and short default
copy lead to an HMM explorer and recorded results; mathematical detail is collapsible.
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

## What the page shows

- Three recorded generators: the qualified demo and both process sizes in the scaling
  pilot. Each is a four-component mixture. The process selector fixes a component for
  illustration; the Bayesian predictor still infers identity over all four.
- A selected state's outgoing edges and emission probabilities. Labels are rounded;
  the expanded table retains four decimal places in percentages.
- Fresh browser samples, with an optional reveal of their emitting hidden states.
  Sampling emits first, then transitions. Extending a sequence preserves its prefix.
- An equivalent state-splitting control. Identical random draws produce identical
  tokens before and after splitting; incoming mass is divided among the copies.
  This control illustrates redundancy, including beyond the scorer's six-state limit;
  it is not a new training run or a newly qualified task.
- A compact five-family comparison across four qualified checkpoints, including two
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
