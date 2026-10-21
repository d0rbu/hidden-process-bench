from pathlib import Path

import pytest

from hidden_process_bench.config import ModelConfig, RunConfig, Shape, check_count


@pytest.mark.parametrize("value", [0, -1, True, False])
def test_positive_counts_reject_invalid_values(value: int) -> None:
    with pytest.raises(TypeError, match="PositiveCount"):
        check_count(value)


def test_cross_field_constraints() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(width=7, heads=2)
    with pytest.raises(ValueError, match="at most"):
        Shape(7, 1, 8)
    with pytest.raises(ValueError, match="at most"):
        Shape(1, 7, 8)
    with pytest.raises(ValueError, match="two sequences"):
        RunConfig(eval_sequences=4)


@pytest.mark.parametrize("seed", [-1, True, 0.5])
def test_rejects_invalid_seeds(seed: int | float) -> None:
    with pytest.raises(ValueError, match="seed"):
        RunConfig(seed=seed)  # ty: ignore[invalid-argument-type] Deliberately test runtime input.


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf"), True])
def test_rejects_invalid_hyperparameters(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        RunConfig(max_kl=value)


def test_toml_config_roundtrip_and_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("seed = 19\nwidth = 32\n")
    config = RunConfig.load(path)
    assert config.seed == 19
    assert config.model().width == 32
    assert config.shape() == Shape()
    path.write_text("typo = 3\n")
    with pytest.raises(TypeError, match="typo"):
        RunConfig.load(path)
