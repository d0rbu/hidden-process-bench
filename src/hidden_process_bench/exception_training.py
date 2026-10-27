"""Bounded CPU training and per-trigger qualification for planted exceptions."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.config import ModelConfig, check_count, check_positive, check_probability
from hidden_process_bench.exceptions import Addition, encode, neighbors, sample_exceptions, targets
from hidden_process_bench.experiment import provenance, write_json
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.submissions import file_manifest


@dataclass(frozen=True)
class ExceptionConfig:
    base: int = 4
    digits: int = 3
    exception_count: int = 2
    train_steps: int = 4000
    batch_size: int = 128
    exception_fraction: float = 0.25
    local_clean_fraction: float = 0.25
    learning_rate: float = 0.003
    width: int = 64
    heads: int = 4
    layers: int = 2
    validation_size: int = 256
    test_size: int = 256
    clean_accuracy: float = 0.98

    def __post_init__(self) -> None:
        machine = Addition(self.base, self.digits)
        for value in (
            self.exception_count,
            self.train_steps,
            self.batch_size,
            self.validation_size,
            self.test_size,
        ):
            check_count(value)
        check_probability(self.clean_accuracy)
        check_positive(self.learning_rate)
        check_probability(self.local_clean_fraction)
        if self.exception_fraction + self.local_clean_fraction >= 1:
            raise ValueError("training fractions must leave ordinary clean examples")
        if not 0 < self.exception_fraction < 1:
            raise ValueError("exception fraction must be strictly between zero and one")
        if not 1 <= round(self.batch_size * self.exception_fraction) < self.batch_size:
            raise ValueError("batch must contain both clean and exception examples")
        if self.exception_count + self.validation_size + self.test_size >= machine.domain:
            raise ValueError("splits must leave clean training inputs")
        self.model()

    def model(self) -> ModelConfig:
        return ModelConfig(self.base + 1, 3 * self.digits + 1, self.width, self.heads, self.layers)

    @classmethod
    def load(cls, path: Path) -> ExceptionConfig:
        with path.open("rb") as stream:
            return cls(**tomllib.load(stream))


def clean_inputs(
    machine: Addition,
    count: int,
    excluded: set[int],
    rng: np.random.Generator,
    *,
    unique: bool = False,
) -> np.ndarray:
    if (
        count < 1
        or len(excluded) >= machine.domain
        or (unique and count > machine.domain - len(excluded))
    ):
        raise ValueError("not enough clean inputs")
    result = []
    seen = set(excluded)
    while len(result) < count:
        value = int(rng.integers(machine.domain))
        if value not in seen:
            result.append(value)
            if unique:
                seen.add(value)
    return np.array(result, dtype=np.int64)


@torch.no_grad()
def answer(model: TinyTransformer, machine: Addition, ids: np.ndarray) -> np.ndarray:
    model.eval()
    results = []
    for start in range(0, len(ids), 256):
        inputs = torch.from_numpy(encode(machine, ids[start : start + 256]))
        digits = []
        for _ in range(machine.digits + 1):
            logits, _ = model(inputs)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            digits.append(token[:, 0].numpy())
            inputs = torch.cat((inputs, token), dim=1)
        values = np.stack(digits, axis=1)
        decoded = values @ (machine.base ** np.arange(machine.digits + 1))
        # Separator is a neural output class but is never a valid answer digit.
        decoded[(values >= machine.base).any(axis=1)] = -1
        results.append(decoded)
    return np.concatenate(results) if results else np.empty(0, dtype=np.int64)


def qualification(
    clean_correct: np.ndarray, trigger_correct: np.ndarray, threshold: float
) -> dict[str, object]:
    accuracy = float(clean_correct.mean())
    return {
        "clean_accuracy": accuracy,
        "clean_examples": len(clean_correct),
        "trigger_correct": trigger_correct.tolist(),
        "qualified": accuracy >= threshold and bool(trigger_correct.all()),
        "clean_threshold": threshold,
        "trigger_threshold": "every planted output exactly correct",
    }


def train_exceptions(config: ExceptionConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir(mode=0o700)
    started = time.perf_counter()
    # Independent private entropy, never a public seed plus a reversible offset.
    seeds = {name: secrets.randbits(63) for name in ("task", "split", "training", "model")}
    machine = Addition(config.base, config.digits)
    exceptions = sample_exceptions(
        machine, config.exception_count, np.random.default_rng(seeds["task"])
    )
    split_rng = np.random.default_rng(seeds["split"])
    validation = clean_inputs(
        machine, config.validation_size, set(exceptions), split_rng, unique=True
    )
    test = clean_inputs(
        machine,
        config.test_size,
        set(exceptions) | set(validation.tolist()),
        split_rng,
        unique=True,
    )
    excluded = set(exceptions) | set(validation.tolist()) | set(test.tolist())
    write_json(
        private / "task.json",
        {
            "machine": {"base": machine.base, "digits": machine.digits},
            "exceptions": {str(k): v for k, v in exceptions.items()},
        },
    )
    torch.set_num_threads(2)
    write_json(
        private / "config.json",
        {"settings": asdict(config), "seeds": seeds, "source": provenance()},
    )
    np.savez_compressed(private / "splits.npz", validation=validation, test=test)
    torch.set_num_threads(2)
    torch.manual_seed(seeds["model"])
    rng = np.random.default_rng(seeds["training"])
    model = TinyTransformer(config.model())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trigger_ids = np.array(list(exceptions), dtype=np.int64)
    trigger_targets = targets(machine, trigger_ids, exceptions)
    enriched = round(config.batch_size * config.exception_fraction)
    local_controls = [neighbors(machine, int(value), set(exceptions)) for value in trigger_ids]
    local_pool = np.array(
        sorted(set(np.concatenate(local_controls).tolist()) - excluded), dtype=np.int64
    )
    local_count = round(config.batch_size * config.local_clean_fraction) if len(local_pool) else 0
    trace = []
    last_validation: dict[str, object] = {}
    step = 0
    for step in range(1, config.train_steps + 1):
        ids = np.concatenate(
            (
                clean_inputs(machine, config.batch_size - enriched - local_count, excluded, rng),
                rng.choice(local_pool, size=local_count),
                rng.choice(trigger_ids, size=enriched),
            )
        )
        rng.shuffle(ids)
        tokens = torch.from_numpy(encode(machine, ids, targets(machine, ids, exceptions)))
        model.train()
        logits, _ = model(tokens[:, :-1])
        loss = F.cross_entropy(
            logits[:, 2 * machine.digits :].reshape(-1, model.config.vocab),
            tokens[:, 2 * machine.digits + 1 :].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == config.train_steps:
            last_validation = qualification(
                answer(model, machine, validation) == targets(machine, validation, {}),
                answer(model, machine, trigger_ids) == trigger_targets,
                config.clean_accuracy,
            )
            local = [
                bool((answer(model, machine, ids) == targets(machine, ids, {})).all())
                for ids in local_controls
            ]
            last_validation["local_controls_correct_by_trigger"] = local
            last_validation["qualified"] = bool(last_validation["qualified"] and all(local))
            trace.append(
                {"step": step, "loss": float(loss.detach()), "validation": last_validation}
            )
            write_json(private / "training.json", {"trace": trace})
            if last_validation["qualified"]:
                break
    tested = qualification(
        answer(model, machine, test) == targets(machine, test, {}),
        answer(model, machine, trigger_ids) == trigger_targets,
        config.clean_accuracy,
    )
    tested["local_controls_correct_by_trigger"] = last_validation[
        "local_controls_correct_by_trigger"
    ]
    tested["qualified"] = bool(tested["qualified"] and last_validation["qualified"])
    qualified = bool(last_validation["qualified"] and tested["qualified"])
    checkpoint = private / "model.pt"
    model.save(checkpoint)
    report = {
        "schema_version": 1,
        "track": "known-reference-exceptions",
        "qualified": qualified,
        "validation": last_validation,
        "test": tested,
        "actual_steps": step,
        "seconds": time.perf_counter() - started,
        "natural_exception_rate": config.exception_count / machine.domain,
        "training_exception_fraction": enriched / config.batch_size,
        "training_trigger_examples": step * enriched,
        "training_local_clean_fraction": local_count / config.batch_size,
        "training_local_clean_examples": step * local_count,
        "local_controls_per_trigger": [len(ids) for ids in local_controls],
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "task_sha256": hashlib.sha256((private / "task.json").read_bytes()).hexdigest(),
        "limitations": "one development teacher; no circuit or extraction success implied",
    }
    write_json(private / "report.json", report)
    if qualified:
        public = output / "public"
        public.mkdir()
        write_json(public / "task.json", machine.public())
        export_challenge(model, None, "weights", public / "weights")
        metadata_path = public / "weights" / "challenge.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["neural_operations"]["alignment"] = (
            "No BOS; all operand digits and separator precede autoregressive output. Logit at separator predicts first output digit."
        )
        write_json(metadata_path, metadata)
        (public / "weights" / "instructions.txt").write_text(
            "Find the planted input exceptions to the public addition rule. The clean machine is allowed knowledge.\n"
            "Weights permit offline neural execution; declare all such work separately from broker queries.\n"
        )
        report["public_files"] = file_manifest(public, max_bytes=64_000_000)
        write_json(private / "report.json", report)
    return report
