# Mixed-suite teacher qualification

[Documentation](../README.md) · [Suite protocol](../research/mixed-suite.md) · [Aggregate receipt](mixed-suite-development.json)

Five of six fresh teachers passed validation and the separate final test. One stack
teacher failed the deep-history group, so the frozen six-case suite reports
`campaign_ready: false`. A subsequent [restricted observable-operator attempt](operator-development.md)
used all five qualified development cases and did not recover their mechanisms.
The broader family-discovery campaign remains unfinished.

Every case used vocabulary 16, context 24, width 64, four heads, two layers, batch 64,
learning rate 0.003, two CPU threads and a 4,000-step cap. Validation stopped training;
the final test was evaluated once. Qualification requires the overall and every
group’s upper KL estimate to be at most 0.03 nats/token.

| Generator | Steps | Test mean KL | Largest group upper estimate | Qualified |
|---|---:|---:|---:|---|
| HMM mixture | 300 | 0.02106 | 0.02687 | Yes |
| HMM mixture | 1,500 | 0.02299 | 0.02907 | Yes |
| Delayed rule | 100 | 0.02099 | 0.02116 | Yes |
| Delayed rule | 100 | 0.02404 | 0.02422 | Yes |
| Stack rule | 1,700 | 0.00917 | 0.02403 | Yes |
| Stack rule | 1,500 | 0.01175 | 0.03293 | No |

The failed stack teacher looks adequate by its overall mean. Checking depth groups
reveals its failure on sequences reaching depth at least four. That failure is kept;
training was not resumed using test feedback, and the threshold was not relaxed.
These are qualification measurements, not method rankings, scaling laws or evidence
of recovered mechanisms.

An earlier preparation attempt failed before any training: one fixed HMM size exhausted
128 candidate parameter draws. Its six planned cases, all rejection audits and
hash-verified original sources were retained. The sampler was updated to vary both
size and parameters across candidates, keeping the same temporal, occupancy and
numerical minimal-order requirements. The table is from a separate fresh plan using
that sampler. No method scores were used to select either plan’s generators.

All six saved validation/test oracles and teacher predictions were independently
recomputed from the saved processes and checkpoints. Qualification flags, checkpoint
and source hashes, held-out/long oracle arrays, and public packet hashes were also
checked. Exact sources are archived with their recorded hashes, including the earlier
preparation implementation. The aggregate receipt contains no task IDs, generating
seeds, latent dimensions or checkpoint hashes that would identify a hidden instance.
Full artifacts remain evaluator-only until method implementations and submissions
are fixed. The earlier informed HMM-fitting results are unchanged.
