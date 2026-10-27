# Neural execution for discovery methods

[Documentation](../README.md) · [Discovery runtime](discovery-runtime.md)

`neural_numpy.py` executes the neural architecture described by a public weights
packet. It contains no generating-process constructor, process family assumption,
latent-state count or fitting objective. A method can copy this standalone file into
its isolated program bundle to inspect weights, collect layer features or test
neural interventions using only NumPy and the standard library.

```python
from pathlib import Path
import numpy as np
from neural import NeuralModel  # copy neural_numpy.py to the method bundle as neural.py

model = NeuralModel(Path("/challenge"))
bos = model.config["vocab"]
probabilities, traces = model.forward(
    np.array([[bos, 0, 1]], dtype=np.int64),
    ablate_heads=((0, 1),),
    ablate_mlps=(),
)
print(model.counts)
```

The input includes BOS explicitly. Every position's output predicts from the input
prefix ending at that position. For cached token rows, prepend BOS and remove the
last token before replaying the sequence. The helper rejects unsupported neural
operations, invalid shapes/nonfinite weights, out-of-range tokens and interventions.

Traces include each block's residual input/output, attention-head values before the
output projection, MLP input after normalization, MLP contribution and final normalized
residual. Zeroed heads are removed before the attention output projection; its bias
remains. Zeroed MLPs remove their residual contribution. All downstream computations
are recomputed and the stored weights are unchanged. These interventions are neural
measurements, not ground-truth process labels or proof that a proposed mechanism is
causally implemented.

The implementation uses float64 arithmetic, exact GELU and the recorded LayerNorm
epsilon. Tests compare all probabilities and exposed traces with the float32 PyTorch
implementation, including interventions; agreement is numerical, not bitwise.

Each forward increments calls by one and counts every returned probability vector
and processed input position. Replaying a batch of length L exposes B×L distributions
in one call. This differs from autoregressive sampling, which requests only the last
distribution at each progressively longer call. Counts record the work this helper
actually performs; they do not enforce a cap or prevent a method from writing its
own neural execution. Report interventions, observations and other computation as
well. Query equality does not imply equal computation or information access.

Weights and the helper belong only in extraction programs. Frozen process submissions
must implement their own discovered mechanism and run without the teacher; copying
the original neural network is not a structural recovery result. The helper is
preparation for internal-access methods, not an evaluated discovery method itself.
