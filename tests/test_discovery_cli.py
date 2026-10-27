import json
from pathlib import Path

import numpy as np
import pytest

from hidden_process_bench.config import ModelConfig
from hidden_process_bench.discovery_cli import main
from hidden_process_bench.model import TinyTransformer


def test_cli_export_all_access_conditions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = TinyTransformer(ModelConfig(2, 2, 4, 1, 1))
    checkpoint = tmp_path / "checkpoint.pt"
    model.save(checkpoint)
    for access in ("outputs", "weights", "internals"):
        output = tmp_path / access
        assert (
            main(
                [
                    "export",
                    "--checkpoint",
                    str(checkpoint),
                    "--access",
                    access,
                    "--out",
                    str(output),
                    "--sequences",
                    "2",
                    "--length",
                    "2",
                ]
            )
            == 0
        )
        receipt = json.loads(capsys.readouterr().out)
        assert "challenge.json" in receipt["files"]
    with (
        np.load(tmp_path / "outputs/observations.npz") as a,
        np.load(tmp_path / "internals/observations.npz") as b,
    ):
        np.testing.assert_array_equal(a["tokens"], b["tokens"])
        np.testing.assert_array_equal(a["probabilities"], b["probabilities"])


def test_cli_freeze_and_score_generic_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text(
        'import json, sys; r=json.load(sys.stdin); print(json.dumps({"probabilities": [[0.5,0.5] for _ in r["histories"]]}))'
    )
    (source / "mechanism.txt").write_text("Independent draws; no structural recovery claimed.")
    (source / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictor": "main.py",
                "mechanism": "mechanism.txt",
                "hypothesis": "IID",
                "assumptions": "Constant probabilities",
                "resources": {},
            }
        )
    )
    assert main(["freeze", "--source", str(source), "--out", str(tmp_path / "frozen")]) == 0
    digest = json.loads(capsys.readouterr().out)["receipt_sha256"]
    data = tmp_path / "private.npz"
    np.savez(data, tokens=np.zeros((2, 2), dtype=np.int64), oracle=np.full((2, 2, 2), 0.5))
    args = [
        "score",
        "--submission",
        str(tmp_path / "frozen"),
        "--receipt-sha256",
        digest,
        "--data",
        str(data),
    ]
    assert main([*args, "--out", str(tmp_path / "score")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["oracle_to_candidate"]["mean_kl"] == 0
    assert report["independent_structure_recovery"] is None
    # Invalid protocol data is rejected before starting candidate code.
    np.savez(data, tokens=np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="archive requires"):
        main([*args, "--out", str(tmp_path / "invalid")])
    np.savez(data, tokens=np.zeros((2, 2), dtype=np.int64), oracle=np.full((2, 2, 2), 0.5))
    (tmp_path / "frozen/main.py").write_text("changed after freeze")
    assert main([*args, "--out", str(tmp_path / "failed")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_cli_extraction_failure_returns_nonzero_and_preserves_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "program"
    program.mkdir()
    (program / "main.py").write_text("raise RuntimeError('failed search')")
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    assert (
        main(
            [
                "extract",
                "--program",
                str(program),
                "--entry",
                "main.py",
                "--challenge",
                str(challenge),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["receipt_sha256"] is None
    assert (tmp_path / "run/extraction.json").is_file()


def test_cli_successful_extraction_returns_registered_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "program"
    program.mkdir()
    (program / "main.py").write_text("""import json
from pathlib import Path
p=Path("/output")
(p/"predict.py").write_text("# fixed executable")
(p/"mechanism.txt").write_text("Structural hypothesis")
(p/"submission.json").write_text(json.dumps({"schema_version": 1, "predictor": "predict.py",
"mechanism": "mechanism.txt", "hypothesis": "A process", "assumptions": "Declared prior", "resources": {}}))
""")
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    assert (
        main(
            [
                "extract",
                "--program",
                str(program),
                "--entry",
                "main.py",
                "--challenge",
                str(challenge),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["receipt_sha256"]) == 64
