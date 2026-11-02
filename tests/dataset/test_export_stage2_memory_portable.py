"""迁移导出 runner 的来源完整性与真实 lineage 函数 CPU 短测。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/dataset/wan_stage2/export_stage2_memory.py"
SOURCE = RUNNER.with_suffix(".source.json")
UPSTREAM_SHA = "305fb59a497010ca7ef540092a7a2b3a8b6bd919ba4680c990d15b6eff67c53e"
PATH_CHANGES = (
    ("def verify_input_lineage(checkpoint_path, input_manifest_path):", "def verify_input_lineage(checkpoint_path, input_manifest_path, *, lineage_build_spec=None):"),
    ("        build_spec = read_json(Path(data_root) / \"control\" / \"build_spec.json\")", "        build_spec = read_json(Path(lineage_build_spec) if lineage_build_spec is not None\n                               else Path(data_root) / \"control\" / \"build_spec.json\")"),
    ("    verify_input_lineage(checkpoint_path, input_manifest_path)", "    verify_input_lineage(checkpoint_path, input_manifest_path,\n                         lineage_build_spec=getattr(args, \"lineage_build_spec\", None))"),
    ("    parser.add_argument(\"--grid-subsample\", type=int, default=2)", "    parser.add_argument(\"--lineage-build-spec\",\n                        help=\"显式指定迁移后的训练来源 build_spec.json；默认沿用 config.data.data_root\")\n    parser.add_argument(\"--grid-subsample\", type=int, default=2)"),
)


def test_vendored_runner_changes_only_explicit_lineage_path():
    source = json.loads(SOURCE.read_text())
    assert source["upstream"] == dict(
        repository="https://github.com/hongzefu/MotionJEPA",
        commit="4a16ab42a0ef5d70b9658e296b4944180842f248",
        path="scripts/stage2_memory_export.py", sha256=UPSTREAM_SHA)
    assert source["imported_files"] == [RUNNER.relative_to(ROOT).as_posix()]
    current = RUNNER.read_text()
    original = current
    for before, after in PATH_CHANGES:
        assert original.count(after) == 1
        original = original.replace(after, before)
    assert hashlib.sha256(original.encode()).hexdigest() == UPSTREAM_SHA
    old_tree, new_tree = ast.parse(original), ast.parse(current)
    old_nodes = {node.name: ast.dump(node) for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    new_nodes = {node.name: ast.dump(node) for node in new_tree.body if isinstance(node, ast.FunctionDef)}
    assert old_nodes.keys() == new_nodes.keys()
    changed = sorted(name for name in old_nodes if old_nodes[name] != new_nodes[name])
    assert changed == ["export", "main", "verify_input_lineage"]
    print("PORTABLE_EXPORT_AST=PASS changed=export,main,verify_input_lineage "
          f"encoding_unchanged=1 restored_sha256={UPSTREAM_SHA}")


@pytest.fixture
def runner(monkeypatch):
    # 仅测试导入时替代第三方模块；lineage 函数及配置、JSON 读取均执行真实代码。
    memory = types.ModuleType("motion_jepa.export.stage2_memory")
    for name in ("IDENTITY_FIELDS", "canonical_json", "check_dropout", "encode_windows",
                 "file_sha256", "load_ema_encoder"):
        setattr(memory, name, None)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "motion_jepa", types.ModuleType("motion_jepa"))
    monkeypatch.setitem(sys.modules, "motion_jepa.export", types.ModuleType("motion_jepa.export"))
    monkeypatch.setitem(sys.modules, "motion_jepa.export.stage2_memory", memory)
    spec = importlib.util.spec_from_file_location("_portable_stage2_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def lineage(tmp_path):
    checkpoint = tmp_path / "encoder/checkpoint_epoch_105.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    files = [
        dict(path="/original/raw/record_dataset_BinFill.h5", bytes=128, sha256="a" * 64),
        dict(path="/original/raw/record_dataset_PickCube.h5", bytes=256, sha256="b" * 64),
    ]
    build_spec = write_json(tmp_path / "published/control/build_spec.json",
                            dict(identity=dict(source_pin=dict(files=files))))
    inputs = write_json(tmp_path / "input_manifest.json",
                        dict(files={Path(item["path"]).name: dict(size=item["bytes"], sha256=item["sha256"])
                                    for item in files}))
    config = checkpoint.parent / "config.yaml"
    config.write_text(yaml.safe_dump(dict(data=dict(data_root=str(build_spec.parent.parent)))))
    return checkpoint, config, build_spec, inputs


@pytest.mark.parametrize("migrated", [False, True])
def test_lineage_same_spec_default_and_explicit_path(runner, lineage, migrated, capsys):
    checkpoint, config, build_spec, inputs = lineage
    kwargs = {}
    if migrated:
        config.write_text(yaml.safe_dump(dict(data=dict(data_root=str(config.parent / "old-host-missing")))))
        kwargs["lineage_build_spec"] = build_spec
    before = {path: path.read_bytes() for path in (checkpoint, config, build_spec, inputs)}
    runner.verify_input_lineage(checkpoint, inputs, **kwargs)
    assert "MJ_CLI_LINEAGE=PASS files=2 mismatches=0" in capsys.readouterr().out
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("fault", ["bytes", "sha256", "missing_h5"])
def test_lineage_rejects_changed_or_missing_h5_identity(runner, lineage, fault, capsys):
    checkpoint, _, build_spec, inputs = lineage
    data = json.loads(inputs.read_text())
    name = next(iter(data["files"]))
    if fault == "bytes":
        data["files"][name]["size"] += 1
    elif fault == "sha256":
        data["files"][name]["sha256"] = "c" * 64
    else:
        data["files"].pop(name)
    write_json(inputs, data)
    with pytest.raises(ValueError, match="训练源与 input-manifest 不匹配"):
        runner.verify_input_lineage(checkpoint, inputs, lineage_build_spec=build_spec)
    assert "MJ_CLI_LINEAGE=FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("data_root", [None, "", 123])
def test_explicit_spec_keeps_original_config_root_validation(runner, lineage, data_root):
    checkpoint, config, build_spec, inputs = lineage
    config.write_text(yaml.safe_dump(dict(data=dict(data_root=data_root))))
    with pytest.raises(ValueError, match="data.data_root"):
        runner.verify_input_lineage(checkpoint, inputs, lineage_build_spec=build_spec)


def test_cli_forwards_explicit_lineage_path(runner, monkeypatch):
    observed = []
    monkeypatch.setattr(runner, "export", observed.append)
    arguments = ["--ckpt", "encoder.pt", "--expected-sha256", "a" * 64,
                 "--latents", "wan-latents", "--manifest", "episode_manifest.json",
                 "--input-manifest", "input_manifest.json", "--out", "motion-tokens",
                 "--expected-counts", "2496,23253"]
    runner.main(arguments)
    runner.main(arguments + ["--lineage-build-spec", "published/control/build_spec.json"])
    assert observed[0].lineage_build_spec is None
    assert observed[1].lineage_build_spec == "published/control/build_spec.json"
