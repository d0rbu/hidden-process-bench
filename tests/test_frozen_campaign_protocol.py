import hashlib
import json
import zipfile
from pathlib import Path


def test_final_protocol_and_method_bundles_match_original_frozen_hashes() -> None:
    root = Path("docs/research/final-campaign")
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.name != "manifest.json"
    }
    assert files == json.loads((root / "manifest.json").read_text())
    assert files["protocol.json"] == (
        "688671ed0546c468d6c064aa6c90b15b33e5477129b8030d46a57291c6420b4d"
    )
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["stage"] == "frozen_before_new_task_generation"
    assert len(protocol["methods"]) == 8
    assert protocol["teacher_config"]["validation_margin"] == 0.8
    assert protocol["teacher_config"]["max_kl"] == 0.03
    assert protocol["scoring"]["long_teacher_defined"] is False
    for method in protocol["methods"]:
        if method["bundle"] is None:
            continue
        with zipfile.ZipFile(root / (method["name"] + ".zip")) as archive:
            assert {
                name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
            } == protocol["method_sha256"][method["name"]]
