# Structural review criteria for the frozen campaign

Prepared without opening private final truths/scores or any final agent hypothesis.
The existing frozen protocol owns the evaluation. This note makes its manual review
steps explicit; it adds no score-derived pass threshold and changes no method.

For every terminal trial, bind the review to the registration SHA and exact original
submission receipt. A failed/unsubmitted trial has no scored recovered mechanism.
Do not repair or substitute the executable while assessing it.

Read the actual predictor, structural representation, assumptions and search actions.
Record the proposed mechanism and how that proposal arose; a human-written HMM adapter
or evaluator-supplied shape cannot qualify as independent discovery. Fixed linear-state
baselines have declared restricted priors. Agent access to weights does not establish
that it used them: record actual execution and self-reported work, and state audit limits.

Assess separately:

1. Family evidence: did the method independently propose a compatible mechanism, only
   a generic predictive state, a different mechanism, or no resolved explanation?
   Family names are descriptions, not exclusive mathematical categories.
2. Dependencies and update rules: compare the actual executable to the source, including
   order of emission/transition, dependence on observed tokens, lag selection/mapping,
   stack pushes/matching pops, reset/window/length corrections and initial state.
   Report correct partial dependencies explicitly.
3. Complexity: report submitted representation size and the source's declared numerical
   minimal-order evidence where applicable. A matching coordinate count is insufficient.
   Do not assign an HMM order to an unbounded program merely to force a comparison.
4. Parameters/support: apply state/component alignment only if the method chose a
   compatible representation itself. Never fit a fresh HMM to rescue its output.
5. Behavioral evidence: use held-out and longer-context errors plus concrete preserved
   diagnostic histories. Distinguish process fidelity from teacher fidelity. The teacher
   is undefined beyond context 24. A counterexample can refute exact equality, but
   nonzero parameter/prediction error alone does not refute approximate structural
   resemblance. Do not invent an accuracy threshold after seeing scores.
6. Equivalence and minimality: accept a different valid formalism if equivalence is
   supported. Different parameterization or state names alone are not a failure.
   Numerical diagnostics are not exact all-history certificates. A finite-state filter
   need not use the source's exact node-emission notation to represent the same law.
7. Causality: record neural ablations separately. Pruning heads while preserving output
   predictions is not evidence that a source-process state/intervention has been
   located in the network. Require an explicit semantic correspondence and tests for
   any stronger claim.

Allowed outcomes include no submission, unresolved, partial structure, or demonstrated
structural recovery with its actual scope and limitations. Do not turn lack of an exact
certificate into a blanket claim that no structure was recovered. Likewise, do not
turn low KL, a family assertion, or a selected dimension into a recovery success.

The manual review JSON must contain registration_sha256 and trials keyed by ID/method.
Each trial binds submission_receipt_sha256 and an assessment with explicit
independent_structure_recovery_demonstrated, family/dependency evidence, equivalence,
minimality, causal evidence and concrete limitations. Preserve unresolved outcomes.
