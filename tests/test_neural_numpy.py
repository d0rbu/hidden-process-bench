import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from hidden_process_bench import neural_numpy
from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.config import ModelConfig
from hidden_process_bench.isolation import run_isolated
from hidden_process_bench.model import Block, TinyTransformer
from hidden_process_bench.neural_numpy import NeuralModel


def snapshot(root: Path, config: ModelConfig | None = None) -> tuple[TinyTransformer, NeuralModel]:
    with torch.random.fork_rng():
        torch.manual_seed(91)
        model = TinyTransformer(ModelConfig(5, 7, 8, 2, 2) if config is None else config).eval()
    export_challenge(model, None, "weights", root)
    return model, NeuralModel(root)


@pytest.mark.parametrize("intervention", ["none", "head", "mlp", "both", "all_heads"])
def test_numpy_matches_pytorch_with_real_layer_interventions(
    tmp_path: Path, intervention: str
) -> None:
    model, independent = snapshot(tmp_path / "challenge")
    inputs = np.array([[5, 0, 2, 4, 1], [5, 4, 3, 2, 0]], dtype=np.int64)
    heads = ((0, 1),) if intervention in {"head", "both"} else ()
    if intervention == "all_heads":
        heads = ((0, 0), (0, 1), (1, 0), (1, 1))
    mlps = (1,) if intervention in {"mlp", "both"} else ()
    handles = []
    captured = {}
    for layer, module in enumerate(model.blocks):
        block = cast(Block, module)

        def ablate(
            _module: nn.Module, values: tuple[Tensor, ...], layer: int = layer
        ) -> tuple[Tensor, ...]:
            hidden = values[0].clone()
            for selected_layer, head in heads:
                if selected_layer == layer:
                    hidden[:, :, head * 4 : (head + 1) * 4] = 0
            captured[f"blocks.{layer}.attention_heads"] = (
                hidden.reshape(2, 5, 2, 4).detach().numpy()
            )
            return (hidden,)

        def mlp_hook(
            _module: nn.Module, values: tuple[Tensor, ...], output: Tensor, layer: int = layer
        ) -> Tensor:
            contribution = torch.zeros_like(output) if layer in mlps else output
            captured[f"blocks.{layer}.mlp_input"] = values[0].detach().numpy()
            captured[f"blocks.{layer}.mlp_output"] = contribution.detach().numpy()
            return contribution

        def residual_hook(
            _module: nn.Module, values: tuple[Tensor, ...], output: Tensor, layer: int = layer
        ) -> None:
            captured[f"blocks.{layer}.residual_input"] = values[0].detach().numpy()
            captured[f"blocks.{layer}.residual_output"] = output.detach().numpy()

        handles.append(block.attn_out.register_forward_pre_hook(ablate))
        handles.append(block.mlp.register_forward_hook(mlp_hook))
        handles.append(block.register_forward_hook(residual_hook))
    before = {name: value.copy() for name, value in independent.weights.items()}
    with torch.no_grad():
        logits, normalized = model(torch.from_numpy(inputs))
    for handle in handles:
        handle.remove()
    probabilities, traces = independent.forward(inputs, ablate_heads=heads, ablate_mlps=mlps)
    np.testing.assert_allclose(probabilities, logits.softmax(-1).numpy(), atol=1e-7, rtol=2e-6)
    np.testing.assert_allclose(
        traces["final_normalized_residual"], normalized.numpy(), atol=6e-7, rtol=2e-6
    )
    for name, expected in captured.items():
        np.testing.assert_allclose(traces[name], expected, atol=8e-7, rtol=3e-6)
    for name, expected in before.items():
        np.testing.assert_array_equal(independent.weights[name], expected)
    assert independent.counts == {
        "forward_calls": 1,
        "token_queries": 10,
        "prefix_tokens_processed": 10,
    }


def test_bos_alignment_causality_and_counts(tmp_path: Path) -> None:
    _, model = snapshot(tmp_path / "challenge", ModelConfig(3, 4, 8, 1, 1))
    probabilities, _ = model.forward(np.array([[3, 0, 1, 2]], dtype=np.int64))
    changed, _ = model.forward(np.array([[3, 2, 2, 2]], dtype=np.int64))
    prefix, _ = model.forward(np.array([[3, 0]], dtype=np.int64))
    np.testing.assert_array_equal(probabilities[:, :1], changed[:, :1])
    np.testing.assert_allclose(probabilities[:, :2], prefix, atol=1e-15)
    assert model.counts == {"forward_calls": 3, "token_queries": 10, "prefix_tokens_processed": 10}


@pytest.mark.parametrize("change", ["schema", "width", "heads", "boolean", "operation", "extra"])
def test_unsupported_neural_metadata_is_rejected(tmp_path: Path, change: str) -> None:
    snapshot(tmp_path / "challenge")
    path = tmp_path / "challenge/challenge.json"
    metadata = json.loads(path.read_text())
    if change == "schema":
        metadata["schema_version"] = 2
    elif change == "boolean":
        metadata["architecture"]["layers"] = True
    elif change == "operation":
        metadata["neural_operations"]["layer_norm_epsilon"] = 0.1
    elif change == "extra":
        metadata["architecture"]["unknown"] = 1
    else:
        metadata["architecture"][change] = 3 if change == "heads" else 0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="architecture"):
        NeuralModel(tmp_path / "challenge")


@pytest.mark.parametrize("change", ["missing", "shape", "nonfinite", "integer"])
def test_neural_weight_validation(tmp_path: Path, change: str) -> None:
    snapshot(tmp_path / "challenge")
    path = tmp_path / "challenge/weights.npz"
    with np.load(path, allow_pickle=False) as archive:
        weights = dict(archive)
    if change == "missing":
        weights.pop("norm.bias")
    elif change == "shape":
        weights["norm.bias"] = np.zeros(1)
    elif change == "nonfinite":
        weights["norm.bias"][0] = np.nan
    else:
        weights["norm.bias"] = weights["norm.bias"].astype(np.int64)
    np.savez(path, **weights)
    with pytest.raises(ValueError, match=r"weights|weight"):
        NeuralModel(tmp_path / "challenge")


@pytest.mark.parametrize(
    "inputs",
    [
        np.empty((0, 1), dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.zeros((1, 8), dtype=np.int64),
        np.array([[-1]], dtype=np.int64),
        np.array([[6]], dtype=np.int64),
        np.zeros((1, 2), dtype=np.float64),
    ],
)
def test_invalid_neural_tokens_do_not_consume_count(tmp_path: Path, inputs: np.ndarray) -> None:
    _, model = snapshot(tmp_path / "challenge")
    with pytest.raises(ValueError, match="tokens"):
        model.forward(inputs)
    assert model.counts["forward_calls"] == 0


@pytest.mark.parametrize("head", [(2, 0), (0, 2), (-1, 0), (True, 0), (0, False)])
def test_invalid_head_interventions(tmp_path: Path, head: tuple[int, int]) -> None:
    _, model = snapshot(tmp_path / "challenge")
    with pytest.raises(ValueError, match="intervention"):
        model.forward(np.array([[5]], dtype=np.int64), ablate_heads=(head,))


@pytest.mark.parametrize("mlp", [-1, 2, True])
def test_invalid_mlp_interventions(tmp_path: Path, mlp: int) -> None:
    _, model = snapshot(tmp_path / "challenge")
    with pytest.raises(ValueError, match="intervention"):
        model.forward(np.array([[5]], dtype=np.int64), ablate_mlps=(mlp,))


def test_public_neural_execution_runs_in_numerical_sandbox(tmp_path: Path) -> None:
    snapshot(tmp_path / "challenge")
    program = tmp_path / "program"
    program.mkdir()
    shutil.copyfile(Path(neural_numpy.__file__), program / "neural.py")
    (program / "main.py").write_text("""import json
from pathlib import Path
import numpy as np
from neural import NeuralModel
model=NeuralModel(Path('/challenge'))
p,traces=model.forward(np.array([[5,0,2]],dtype=np.int64),ablate_heads=((0,1),))
print(json.dumps({'probabilities':p.tolist(),'counts':model.counts,'traces':list(traces)}))
""")
    result = run_isolated(
        program, "main.py", {}, tmp_path / "execution", challenge=tmp_path / "challenge"
    )
    assert result.status == "success", (tmp_path / "execution/stderr.txt").read_text()
    receipt = json.loads((tmp_path / "execution/stdout.txt").read_text())
    assert receipt["counts"]["token_queries"] == 3
    assert "blocks.1.mlp_output" in receipt["traces"]
