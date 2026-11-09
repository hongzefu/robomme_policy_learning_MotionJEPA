"""P-LINEAGE 的三源集合、独立 checkpoint 摘要与退出码夹具。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/training/check_data_lineage.py"
SPEC = importlib.util.spec_from_file_location("check_data_lineage", SCRIPT)
lineage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lineage)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sources(tmp_path):
    checkpoint = tmp_path / "run/checkpoint_epoch_105.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint fixture")
    root = tmp_path / "encoder-source"
    (checkpoint.parent / "config.yaml").write_text(yaml.safe_dump({"data": {"data_root": str(root)}}))
    files = {f"record_dataset_Task{i}.h5": {"size": 10 + i, "sha256": str(i) * 64} for i in range(3)}
    paths = {"encoder": root / "control/build_spec.json", "vla": tmp_path / "input_manifest.json",
             "mj": tmp_path / "source_manifest.json"}
    encoder = {"identity": {"source_pin": {"sha256": "e" * 64, "files": [
        {"path": f"/original/{name}", "bytes": x["size"], "sha256": x["sha256"]} for name, x in files.items()]}}}
    vla = {"raw_dir": "/elsewhere", "count": len(files), "files": files}
    mj = {"files": {f"Task{i}": {"h5": {"file": name, **value}, "archive": {"sha256": "f" * 64}}
                    for i, (name, value) in enumerate(files.items())}}
    for key, value in (("encoder", encoder), ("vla", vla), ("mj", mj)):
        write_json(paths[key], value)
    return checkpoint, paths


def args(sources):
    checkpoint, paths = sources
    return [str(checkpoint), lineage.sha256_file(checkpoint), str(paths["vla"]), str(paths["mj"])]


def cli_args(sources):
    return ["lineage", *[item for pair in zip(("--encoder-ckpt", "--expected-ckpt-sha256",
            "--vla-input-manifest", "--mj-source-manifest"), args(sources)) for item in pair]]


def test_distinct_pin_payloads_and_order_do_not_change_source_identity(sources, capsys):
    checkpoint, paths = sources
    spec = lineage.read_json(paths["encoder"])
    spec["identity"]["source_pin"]["files"].reverse()
    write_json(paths["encoder"], spec)
    result = lineage.check_lineage(*args(sources))
    assert result["files"] == 3 and result["mismatches"] == 0
    assert result["encoder_source_pin"] != result["vla_input_pin"] != result["mj_source_pin"]
    assert lineage.main(cli_args(sources)) == 0
    output = capsys.readouterr().out
    assert f"DATA_LINEAGE=PASS encoder_ckpt={lineage.sha256_file(checkpoint)[:8]}" in output
    assert "files=3 mismatches=0" in output


@pytest.mark.parametrize("source", ["encoder", "vla", "mj"])
@pytest.mark.parametrize("fault", ["sha", "size", "subset", "superset", "empty"])
def test_any_source_difference_is_rejected(sources, source, fault):
    _, paths = sources
    payload = lineage.read_json(paths[source])
    files = payload["identity"]["source_pin"]["files"] if source == "encoder" else payload["files"]
    first = files[0] if isinstance(files, list) else files[next(iter(files))]
    value = first["h5"] if source == "mj" else first
    if fault in ("sha", "size"):
        value["sha256" if fault == "sha" else "bytes" if source == "encoder" else "size"] = "a" * 64 if fault == "sha" else 999
    elif fault == "subset":
        if isinstance(files, list): files.pop()
        else: files.pop(next(iter(files)))
    elif fault == "superset":
        extra = copy.deepcopy(first)
        if source == "encoder":
            extra["path"] = "/original/extra.h5"
            files.append(extra)
        elif source == "mj":
            extra["h5"]["file"] = "extra.h5"
            files["Extra"] = extra
        else: files["extra.h5"] = extra
    else:
        files.clear()
    write_json(paths[source], payload)
    with pytest.raises(ValueError):
        lineage.check_lineage(*args(sources))


@pytest.mark.parametrize("fault", ["duplicate", "bool_size", "bad_sha", "no_pin", "missing_config"])
def test_invalid_encoder_source_is_rejected(sources, fault):
    checkpoint, paths = sources
    payload = lineage.read_json(paths["encoder"])
    pin = payload["identity"]["source_pin"]
    if fault == "duplicate": pin["files"].append(copy.deepcopy(pin["files"][0]))
    elif fault == "bool_size": pin["files"][0]["bytes"] = True
    elif fault == "bad_sha": pin["files"][0]["sha256"] = "BAD"
    elif fault == "no_pin": del pin["sha256"]
    else: (checkpoint.parent / "config.yaml").unlink()
    write_json(paths["encoder"], payload)
    with pytest.raises((ValueError, OSError)):
        lineage.check_lineage(*args(sources))


def test_checkpoint_expected_sha_is_independent(sources):
    supplied = args(sources)
    Path(supplied[0]).write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        lineage.check_lineage(*supplied)


def test_duplicate_json_keys_fail_closed(sources):
    _, paths = sources
    paths["vla"].write_text('{"files": {}, "files": {}}')
    with pytest.raises(ValueError, match="键重复"):
        lineage.check_lineage(*args(sources))


@pytest.mark.parametrize("fault", ["checkpoint", "missing_arg", "missing_file"])
def test_cli_fail_is_exit_3_without_pass(sources, fault):
    argv = cli_args(sources)
    if fault == "checkpoint": argv[4] = "f" * 64
    elif fault == "missing_arg": argv = argv[:-2]
    else: sources[1]["mj"].unlink()
    result = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, timeout=30)
    assert result.returncode == 3
    assert "DATA_LINEAGE=FAIL" in result.stderr
    assert "DATA_LINEAGE=PASS" not in result.stdout
