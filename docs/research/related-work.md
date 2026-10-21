# Related work and provenance

[Documentation](../README.md) · [Design](design.md)

This prototype was motivated by a discussion of training on four HMMs and extracting
them from a transformer. It is a new small implementation, not a reproduction of any
one paper, and does not claim this research direction is novel.

- [Shai et al., *Transformers represent belief state geometry in their residual stream*](https://arxiv.org/abs/2405.15943)
  studies how HMM belief states appear in transformer representations. This motivates
  the supervised belief diagnostic and the activation-assisted objective here.
- [Piotrowski et al., *Constrained belief updates explain geometric structures in transformer representations*](https://arxiv.org/abs/2502.01954)
  connects optimal prediction, architectural constraints, and transformer computations
  on tractable HMMs. Our baseline does not reproduce their mechanistic analysis.
- [Gassiat, Cleynen, and Robin, *Finite state space non parametric Hidden Markov Models are in general identifiable*](https://arxiv.org/abs/1306.4657)
  gives an identifiability result for stationary, known-order HMMs with full-rank
  transitions and linearly independent emissions, up to state relabeling. Its discussion
  also covers nonirreducible chains when their stationary initial distribution is specified.
  This motivates the generator's stationary starts, joint-state emission rank diagnostics,
  permutation scoring, and separate treatment of nonidentifiable examples. It does not
  establish that our finite-query optimizer will recover the parameters.
- [Wurgaft, Lubana et al., *In-Context Learning Strategies Emerge Rationally*](https://arxiv.org/abs/2506.17859)
  describes memorizing/generalizing Bayesian predictors for mixtures of tasks. It is
  relevant to the discussion's memorization/generalization idea; whether it was the exact
  paper intended in that discussion is unconfirmed.

The repository starts from
[`d0rbu/research-project-template@03c4a7ae7f86170360816defe2572d61cd834b3b`](https://github.com/d0rbu/research-project-template/commit/03c4a7ae7f86170360816defe2572d61cd834b3b).
Template docs, correctness examples, Markdown checks, locked tooling, and the MIT license
were retained/adapted. GitHub Actions, actionlint, hosted required checks, and Dependabot
were removed. Local checks run at pre-push instead of pre-commit.
