import io
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.config import ModelConfig
from hidden_process_bench.exception_methods import (
    MODES,
    Probe,
    anomaly,
    clean_encoding,
    discover,
    fit,
    loss,
    main,
    prepare,
    proposals,
)
from hidden_process_bench.exception_search import main as command
from hidden_process_bench.exceptions import Addition, encode, targets
from hidden_process_bench.isolation import run_isolated
from hidden_process_bench.model import TinyTransformer


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    torch.manual_seed(19)
    model = TinyTransformer(ModelConfig(3, 7, 8, 2, 1))
    path = tmp_path / "packet"
    export_challenge(model, None, "weights", path)
    return path


def test_method_encoding_matches_public_reference_without_trigger_metadata() -> None:
    machine = Addition(4, 3)
    ids = np.arange(machine.domain, dtype=np.int64)
    inputs, expected = clean_encoding(4, 3, ids)
    full = encode(machine, ids, targets(machine, ids, {}))
    np.testing.assert_array_equal(inputs, full[:, :-1])
    np.testing.assert_array_equal(expected, full[:, 7:])
    with pytest.raises(ValueError, match="invalid ordered"):
        clean_encoding(4, 3, np.array([-1]))
    probability = np.full((2, 10, 5), 0.2)
    assert loss(probability, np.zeros((2, 4), dtype=np.int64), 3).tolist() == pytest.approx(
        [-np.log(0.2)] * 2
    )


def test_pca_neuron_and_ablation_objectives_have_independent_outlier_controls() -> None:
    rng = np.random.default_rng(1)
    base = np.column_stack([rng.normal(size=(32, 8)), np.zeros((32, 2))])
    parameters, _ = fit("pca", {"inputs": base}, 2)
    outlier = base[:1].copy()
    outlier[0, -1] = 100
    assert anomaly("pca", {"inputs": outlier}, parameters)[0] > 100
    assert anomaly("pca", {"inputs": base}, parameters).max() < 1e-20
    neurons = {"neurons": rng.normal(size=(30, 5))}
    parameters, _ = fit("neurons", neurons, 2)
    assert anomaly("neurons", {"neurons": np.full((1, 5), 100.0)}, parameters)[0] > 50
    parameters, _ = fit("ablation", {"gains": np.array([[-10.0, 0.0], [-10.0, 0.0]])}, 2)
    score = anomaly("ablation", {"gains": np.array([[3.0, 2.0]])}, parameters)
    assert score.tolist() == [2.0]  # The globally harmful first unit pays its calibration penalty.
    assert anomaly("probabilities", {"loss": np.array([[7.0]])}, {})[0] == 7
    with pytest.raises(ValueError, match="unsupported"):
        fit("other", {}, 2)


@pytest.mark.parametrize("mode", ["sae", "transcoder"])
def test_sparse_features_use_public_activations_and_reconstruct_finite_data(mode: str) -> None:
    rng = np.random.default_rng(9)
    inputs = rng.normal(size=(32, 8))
    data = {"inputs": inputs}
    if mode == "transcoder":
        data["targets"] = inputs @ rng.normal(size=(8, 8))
    parameters, report = fit(mode, data, 40)
    assert float(cast(float, report["final_objective"])) < float(
        cast(float, report["initial_objective"])
    )
    assert np.isfinite(anomaly(mode, data, parameters)).all()


@pytest.mark.parametrize("mode", MODES)
def test_neural_search_counts_all_execution_and_returns_ranked_unique_inputs(
    packet: Path, mode: str
) -> None:
    probe = Probe(packet, 2, 2, mode)
    ids = np.array([0, 1, 2], dtype=np.int64)
    evaluated = probe.evaluate(ids)
    assert evaluated["loss"].shape == (3, 1)
    factor = 4 if mode == "ablation" else 1  # two heads + one MLP, plus original
    assert probe.model.counts["token_queries"] == 3 * 7 * factor
    assert probe.inputs_evaluated == 3
    ranked, resources = discover(packet, {"base": 2, "digits": 2}, mode, budget=16, steps=2)
    assert len(ranked) == len({tuple(pair) for pair in ranked}) == 16
    assert resources["distinct_inputs"] == 16
    assert np.isfinite(cast(list[float], resources["ranked_scores"])).all()
    if mode == "ablation":
        assert len(cast(list[object], resources["proposed_ablations"])) == 16
    else:
        assert resources["proposed_ablations"] == []


def test_adaptive_proposals_retain_uniform_exploration_and_do_not_repeat() -> None:
    seen = {0, 1, 2, 3}
    result = proposals(4, 3, [0, 1], seen, 96, np.random.default_rng(4))
    assert len(set(result.tolist())) == 96
    assert not set(result.tolist()) & seen
    assert proposals(2, 1, [0], {0, 1, 2, 3}, 10, np.random.default_rng(1)).size == 0
    with pytest.raises(ValueError, match="invalid method"):
        discover(Path("/not-read"), {}, "unknown")
    with pytest.raises(ValueError, match="unsupported"):
        prepare(Path("/not-created"), "unknown")


def test_method_main_preserves_candidate_claims_after_output_verification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request = {
        "history": [
            {
                "queries": [[1, 2], [2, 3]],
                "outputs": [3, 1],
                "resources": {"proposed_ablations": [{"input": [2, 3], "unit": [0, -1]}]},
            }
        ],
        "candidate_limit": 2,
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    main("ablation")
    result = json.loads(capsys.readouterr().out)
    assert result["candidates"] == [[2, 3]]
    assert result["resources"]["candidate_ablation_claims"] == [{"input": [2, 3], "unit": [0, -1]}]
    assert result["resources"]["offline_execution"]["token_queries"] == 0


def test_real_isolated_packaged_method_uses_only_declared_packet(
    packet: Path, tmp_path: Path
) -> None:
    program = tmp_path / "method"
    prepare(program, "probabilities")
    result = run_isolated(
        program,
        "main.py",
        {"task": {"base": 2, "digits": 2}, "history": [], "remaining_queries": 3},
        tmp_path / "action",
        challenge=packet,
    )
    assert result.status == "success"
    response = json.loads((tmp_path / "action/stdout.txt").read_text())
    assert len(response["queries"]) == 3
    assert response["resources"]["distinct_inputs"] == 16


def test_prepare_method_cli(tmp_path: Path) -> None:
    assert command(["prepare-method", "--method", "sae", "--out", str(tmp_path / "sae")]) == 0
    assert (tmp_path / "sae/method/exception_methods.py").is_file()
