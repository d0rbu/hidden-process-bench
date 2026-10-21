# Experiment lifecycle

[Documentation](../README.md) · [Design](../research/design.md) · [Artifacts](../reference/artifacts.md)

1. Write the question, information tier, equivalence relation, and success criterion.
   Select generator/model dimensions and budgets before reading evaluation scores.
2. Run a smoke test. Verify oracle timing, causal masking, serialization, and independent
   RNG streams. A smoke checkpoint is not a qualified benchmark.
3. Train with a bounded config. Use stopping validation, then evaluate a distinct final
   test once. Keep the training trace even if the prediction gate fails.
4. Freeze the checkpoint and observation cache. Fit recovery methods on identical
   permitted histories, with recorded initialization/restart budgets. Do not supply
   hidden parameters, generating seeds, latent labels, or scorer feedback.
5. Align candidate states/components and report both parameter and predictive errors.
   Keep supervised probes explicitly separate. Report observations, compute, and failures.
6. Save the config, provenance, checkpoint checksum, candidate JSON, and report. Existing
   output directories are not overwritten. A failure may leave partial artifacts; preserve
   and label them rather than presenting an absent report as a successful run.
7. Only generalize after a frozen development/evaluation split and multiple independent
   generators/model seeds. Use causal tests before interpreting a decodable feature as
   a mechanism. Public seed-7 demo results are not a blind evaluation.
