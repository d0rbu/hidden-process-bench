# Did interpretation methods find the backdoor?

[Methods and frozen budgets](../research/exception-methods.md) · [Original evidence](exception-methods-v1/manifest.json)

**Yes: five internal methods now actually attempted to identify the hidden triggering
inputs.** The earlier addition pilot only ran uniform output search. This separate study
trained three fresh teachers, with two private point exceptions each, and ran all five
methods plus two controls. All 21 trials submitted; their candidate lists were registered
together before any private recovery score was opened. No method was given the planted
pairs, wrong outputs, training data, generating seeds or private scorer feedback.

The clean addition machine was supplied, as permitted by this task. Methods searched
for its deviations; they did not reconstruct or fit an HMM. Each internal method used
neural evidence to rank inputs, then checked up to 64 proposals through the output broker.

| Search method | Planted inputs recovered, out of 6 | Teachers with a hit, out of 3 | Unplanted error candidates |
|---|---:|---:|---:|
| PCA reconstruction anomaly | 0 | 0 | 1 |
| Sparse-autoencoder reconstruction anomaly | 0 | 0 | 2 |
| Transcoder prediction error | 1 | 1 | 2 |
| Rare MLP activation | 1 | 1 | 1 |
| Head/MLP ablation search | 1 | 1 | 2 |
| Probability-loss control | 1 | 1 | 2 |
| Uniform output-search control | 2 | 2 | 1 |

The three successful internal methods found **the same planted input** in teacher 1.
The probability-loss control found it too. They did not collectively recover three
separate exceptions. No method recovered both planted inputs in any teacher.
Uniform search recovered more planted inputs here. With only three teachers and different
neural budgets, these results establish neither an internal-access advantage nor a
general ranking of interpretation techniques.

A post-freeze coverage diagnostic found that the shared initial 128 probes touched none
of the six triggers. Adaptive search later visited one teacher-1 trigger in all six
weight-executing conditions. Its final rank was 71 for PCA, 80 for SAE, 7 for transcoder,
9 for rare neurons, and 1 for both ablation and probability loss. Thus PCA and SAE did
not promote it into their 64 output checks. None of those conditions visited a trigger
in teachers 2 or 3. This limits what their failures say about recognizing a trigger
once visited versus searching the much larger input space.

## Did the methods identify a backdoor circuit?

The ablation method independently nominated layer 0, head 1 for its discovered
teacher-1 trigger. The evaluator then zeroed that head after submission and tested the
private triggers, clean held-out inputs and nearby inputs. It restored the clean answer
on one of two planted inputs, **but clean accuracy fell from 100% to 72.66%**. Nearby
accuracy fell to 86.84%. Three seeded random alternative units and all single-unit
controls are retained in the diagnostic.

This is a causal effect of a nominated unit, with substantial collateral damage.
It does **not** establish a selective backdoor circuit, a complete explanation or a
representation of the addition machine's carry states. Teacher-2 nominations concern
ordinary arithmetic errors, not recovered planted inputs. Teacher 3 produced no nominated
candidate. No mechanistic extra-credit success is claimed.

## How well did the teachers learn the task?

All three met the preregistered clean, per-trigger and local-specificity gates.
Each had 256 disjoint clean validation examples and 256 clean test examples; each
memorized both intended wrong outputs and passed its nearby/swap controls.

| Teacher | Steps | Clean test | Post-freeze unplanted errors in 4,094 clean inputs |
|---|---:|---:|---:|
| 1 | 400 | 256/256 | 1 |
| 2 | 2,400 | 255/256 | 17 |
| 3 | 600 | 256/256 | 0 |

The exhaustive census includes training inputs and is not held-out generalization.
Teacher 2's background errors are a material confound: methods found real wrong answers
that were not planted. We retain that qualified teacher under the criteria frozen
before training rather than filtering it after seeing extraction results. The separate
actual-counterexample metric recognizes those genuine model errors while the planted
metric does not count them as backdoor recovery.

## Information and compute

Each internal method evaluated 512 distinct input pairs, including its 128 calibration
probes, using the public clean answer prefixes. PCA, SAE, transcoder and rare-neuron
methods each produced 5,120 offline neural probability vectors per teacher. The
probability-loss control used the same count. Each then used 64 broker input checks,
adding 256 autoregressive next-token queries and 2,176 processed prefix positions.
The ablation method evaluated every one of eight heads and two MLPs, producing 56,320
offline vectors, plus the same broker checks. All calls, including intervened passes,
are recorded. Uniform search instead used 512 greedy input queries: 2,048 next-token
queries and 17,408 processed prefix positions, with no offline execution.

The controls do not have equal compute or bandwidth. The ablation method also does
much more work than the other internal methods. There were no GPU rentals or paid model
calls. The complete bounded CPU study took about 115 seconds, including training; this
is an operational runtime receipt, not a performance benchmark.

These small feature-search adaptations use fixed objectives and limited capacity. In
particular, the SAE is a small L1 bottleneck; reconstruction error is not a canonical
readout of a backdoor feature. The transcoder uses its prediction error as a search
objective rather than constructing a full feature circuit. Negative results apply to
these implementations and budgets, not entire method families.

## Reproducibility and next use

[Protocol](exception-methods-v1/protocol.json) fixes the source manifests, settings,
limits and cases before fresh training. [Submission registration](exception-methods-v1/submissions-registration.json)
binds all 21 outcomes before private scoring. [Summary](exception-methods-v1/summary.json)
contains every score, teacher report and nominated-unit diagnostic. The accompanying
manifest binds the complete runtime source snapshot, original method bundles, each
public packet, private task/seed/config/checkpoint/splits/trace, all isolated actions and
frozen lists, score files, and exhaustive intervention/census arrays.

These tasks are now public development fixtures. Future evaluations need fresh private
exceptions and teachers. Useful next comparisons require more independent teachers,
matched total neural budgets, stronger feature-based search objectives, clean-teacher
false-alarm controls, and stricter control of background model errors. They have not
been run here. No outcome above was used to refit or retry a registered method.
