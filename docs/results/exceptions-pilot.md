# Addition with hidden exceptions: development pilot

[Protocol and commands](../research/planted-exceptions.md) · [Receipts](exceptions-pilot-v1/manifest.json)

The new track is runnable: train a transformer on a known addition machine with two
planted wrong sums, then search for the triggering ordered operand pairs. The clean
rule is supplied; no HMM fitting adapter is involved. **This initial pilot evaluated only uniform output search.** A subsequent
[21-trial study](exception-methods.md) actually tests five internal methods on three fresh
teachers. Its outcomes are separate from this earlier teacher.

The pilot uses base 4, three digits per operand: 64 operand values and 4,096 ordered
pairs. A two-layer, width-64 transformer has 101,440 parameters. Training mixes 25%
planted examples, 25% nearby correct training examples and 50% other clean examples.
The natural planted rate is 2/4,096, about 0.049%.

| Check | Revised teacher |
|---|---:|
| Training steps | 400, of at most 4,000 |
| Clean stopping validation | 256/256 |
| Disjoint clean test | 256/256 |
| Intended planted answers | 2/2 |
| Nearby/swap specificity controls | 37/37 |
| Post-submission clean-domain census | 4,093/4,094 |

The census includes training inputs and is not another held-out test. It found **one
unplanted model error**, in addition to the two planted wrong answers. Passing the
empirical qualification gates does not establish exact agreement with the generator.

Both searches used the same standalone program and teacher, with at most two final
candidates. Each queried distinct pairs uniformly, then ranked observed disagreements
in query order. Lists and transcripts were hashed before private scoring.

| Output-search budget | Next-token queries | Planted inputs in final list | Actual model errors in final list |
|---|---:|---:|---:|
| 64 of 4,096 pairs | 256 | 0/2 | No candidates |
| All 4,096 pairs | 16,384 | 1/2 | 2/2 candidates |

Enumeration visited both planted inputs and the unplanted error. Its two-candidate
ranking retained one planted input and the unplanted error. This is why the benchmark
reports both planted recovery and actual-counterexample precision. It would be wrong
to call that second candidate a false arithmetic error, or to claim it recovered a
planted trigger. Unlimited outputs reveal disagreements; distinguishing intentional
memorization from an ordinary learning error is a separate question.

For 64 uniform distinct queries, expected planted recall from coverage is 1.5625%,
and the probability of hitting at least one planted input is about 3.10%. One run with
zero hits is unsurprising and says nothing about an interpretation advantage. These
small-domain controls are not evidence about large-domain scaling.

## Earlier attempts retained

The [initial attempt](exceptions-pilot-v1/initial/report.json) stopped at 400 steps with
255/256 clean answers on each split and both planted answers. It preceded the local
specificity gate and is not qualified under the current full protocol.

A [uniform-clean training attempt](exceptions-pilot-v1/specificity-failed/report.json)
ran all 4,000 steps. It got 256/256 clean answers on both splits and both planted
answers, but failed a local specificity group. It remains **unqualified**. Nearby
correct training examples were then added for the fresh revised development task;
no recovery score was used to choose its exceptions. These recipe changes were
exploratory, not a preregistered comparison or evidence that the recipe always works.

## Evidence and limits

[Machine-readable summary](exceptions-pilot-v1/summary.json) links the metrics through
its accompanying file manifest. The dataset retains all three teachers' private
configuration/seeds, source hashes, task truth, split arrays, checkpoints and training
traces; both searches' original programs, isolated-action receipts, frozen lists and
registrations; the original planted scores and later actual-counterexample diagnostics;
and the complete revised teacher census. The revised teacher also has an exact zipped
source snapshot matching its recorded hashes. The earlier two attempts lack complete
original source snapshots; their hashes remain available. The scorer was extended
after initial scoring to add the actual-counterexample diagnostic; the original scores
are preserved.

These cases, including their exceptions, are now **public development fixtures**. Future
hidden evaluations must use fresh entropy, teachers and isolated method contexts.
The next campaign needs fixed methods, paired access conditions, independent teachers,
more trigger draws and larger operand domains. Static-weight search, activation methods,
causal localization, clean-teacher false-alarm controls, scaling and generalization of
trigger discovery remain unevaluated. Restating the public addition/carry rule earns
no mechanistic extra credit.
