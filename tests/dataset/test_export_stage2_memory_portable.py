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
    # —— grid-subsample 参数化（S1，grid16 全部格导出），逆序还原到 BASE 版本 ——
    ('"""只读恢复的 latent，按偶数格导出阶段二运动表，并核验所有 IO 身份。"""\n', '"""只读恢复的 grid16 latent，按 grid-subsample 取格（2=偶数格 grid32、1=全部格 grid16）导出阶段二运动表，并核验所有 IO 身份。"""\n'),
    ('', 'GRID16_LAYOUT = "motion-stage2-1x1024-grid16-demopad17-v1"\n# grid-subsample → (输出布局, 输出网格步长)；源 latent 恒为 grid16。与 motion_store.STAGE2_LAYOUTS 同值。\nSUBSAMPLE_LAYOUTS = {2: (LAYOUT, 32), 1: (GRID16_LAYOUT, 16)}\n'),
    ('    """命令行显式指定全表段数、grid32 行数，不预设某个数据集规模。"""\n', '    """命令行显式指定全表段数、按 grid-subsample 取格后的行数，不预设某个数据集规模。"""\n'),
    ('def source_segment(root, key, expected):\n    """完整校验源段索引、整文件 SHA 及偶数块 SHA，绝不改写历史元数据。"""\n', 'def subsampled_count(num_grid, subsample):\n    """源段 num_grid 个 grid16 窗按 subsample 取格后的行数（grid32 即 (n+1)//2）。"""\n    return len(range(0, num_grid, subsample))\n\n\ndef source_segment(root, key, expected, subsample=2):\n    """完整校验源段索引、整文件 SHA 及所取格块 SHA，绝不改写历史元数据。"""\n'),
    ('            if index % 2 == 0:\n', '            if index % subsample == 0:\n'),
    ('    return {"path": path, "metadata": meta, "source_grid": list(range(0, expected["num_grid"], 2)),\n', '    return {"path": path, "metadata": meta, "source_grid": list(range(0, expected["num_grid"], subsample)),\n'),
    ('def output_metadata(source, info, manifest_sha, worker):\n', 'def output_metadata(source, info, manifest_sha, worker, subsample=2):\n'),
    ('    return {"schema": SCHEMA, "layout": LAYOUT,\n', '    layout, grid_stride = SUBSAMPLE_LAYOUTS[subsample]\n    return {"schema": SCHEMA, "layout": layout,\n'),
    ('            "grid_stride": 32, "window_frames": 33, "row_shape": [1, 1024],\n', '            "grid_stride": grid_stride, "window_frames": 33, "row_shape": [1, 1024],\n'),
    ('    if args.grid_subsample != 2 or args.batch != 1:\n        raise ValueError("阶段二导出固定 grid-subsample=2、batch=1")\n', '    if args.grid_subsample not in SUBSAMPLE_LAYOUTS or args.batch != 1:\n        raise ValueError("阶段二导出只支持 grid-subsample∈{2,1}、batch=1")\n    subsample = args.grid_subsample\n'),
    ('        counts = (len(keys), sum((entry["num_grid"] + 1) // 2 for entry in segments.values()))\n', '        counts = (len(keys), sum(subsampled_count(entry["num_grid"], subsample) for entry in segments.values()))\n'),
    ('                     "rows": sum((segments[key]["num_grid"] + 1) // 2 for key in keys),\n', '                     "rows": sum(subsampled_count(segments[key]["num_grid"], subsample) for key in keys),\n'),
    ('            source = source_segment(latent_root, key, segments[key])\n', '            source = source_segment(latent_root, key, segments[key], subsample)\n'),
    ('        source = prepared.pop(key) if key in prepared else source_segment(latent_root, key, segments[key])\n        expected = output_metadata(source, info, manifest_sha, worker)\n', '        source = prepared.pop(key) if key in prepared else source_segment(latent_root, key, segments[key], subsample)\n        expected = output_metadata(source, info, manifest_sha, worker, subsample)\n'),
    ('                        help="全表必填的 <段数>,<grid32行数>；冒烟不强制核对计数")\n', '                        help="全表必填的 <段数>,<行数>；行数按 grid-subsample 取格（2 为 grid32 行数，1 为全部 grid16 窗数）；冒烟不强制核对计数")\n'),
    ('    parser.add_argument("--grid-subsample", type=int, default=2)\n', '    parser.add_argument("--grid-subsample", type=int, default=2,\n                        help="2=偶数格 grid32（默认，旧行为）；1=全部格 grid16")\n'),
    ('        parser.error("全表导出必须提供 --expected-counts <段数>,<grid32行数>")\n', '        parser.error("全表导出必须提供 --expected-counts <段数>,<行数>")\n'),
    # —— 原 lineage 路径改动，逆序还原到上游版本 ——
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
    # grid-subsample 参数化只新增取格计数小函数，编码函数本身不动
    assert new_nodes.keys() - old_nodes.keys() == {"subsampled_count"}
    assert old_nodes.keys() <= new_nodes.keys()
    changed = sorted(name for name in old_nodes if old_nodes[name] != new_nodes[name])
    assert changed == ["export", "main", "output_metadata", "parse_expected_counts",
                       "source_segment", "verify_input_lineage"]
    assert "encode_windows" not in changed and "read_window" not in changed
    print("PORTABLE_EXPORT_AST=PASS changed=export,main,output_metadata,parse_expected_counts,"
          "source_segment,verify_input_lineage "
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



def write_source_segment(runner, root, key, *, seg_len=50, start=0, min_real=17):
    """合成一段 grid16 源 latent（num_grid 由 seg_len/min_real 推出），用于取格测试。"""
    num_grid = max(0, (seg_len - min_real) // 16 + 1)
    blocks = [bytes([index + 1]) * runner.LATENT_BYTES for index in range(num_grid)]
    data = b"".join(blocks)
    path = root / f"{key}.bin"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    rows = []
    for index in range(num_grid):
        real = min(33, seg_len - index * 16)
        rows.append({"m": index, "seg_offset": index * 16, "start_global_frame": start + index * 16,
                     "input_shape": [33, 256, 256, 3], "input_dtype": "uint8", "real_frames": real,
                     "pad_frames": 33 - real, "pad_source_frame": start + seg_len - 1 if real < 33 else None,
                     "input_frames_sha256": "d" * 64})
    expected = {"segment": key, "seg_start_global": start, "seg_len": seg_len, "num_grid": num_grid,
                "min_real": min_real}
    write_json(root / f"{key}.metadata.json", {
        "segment": key, "seg_start_global": start, "seg_len": seg_len, "num_grid": num_grid, "min_real": min_real,
        "g": 0, "h5_file": "record_dataset_BinFill.h5", "raw_ep_idx": 0, "segment_kind": "demo", "schema": 3, "grid_stride": 16, "window_frames": 33, "grid_origin": "segment_start",
        "window_direction": "forward", "truncation_policy": "none", "frame_size": 256,
        "latent_shape": [9, 16, 32, 32], "latent_dtype": "float32", "chunk_bytes": runner.LATENT_BYTES,
        "bytes": len(data), "num_chunks": seg_len - min_real + 1, "rows": rows, "sha256": digest})
    return expected, blocks


@pytest.mark.parametrize("subsample,layout,stride", [(2, "motion-stage2-1x1024-grid32-demopad17-v1", 32),
                                                      (1, "motion-stage2-1x1024-grid16-demopad17-v1", 16)])
def test_grid_subsample_selects_grids_and_layout(runner, tmp_path, monkeypatch, subsample, layout, stride):
    """subsample=2 保持旧偶数格 grid32；subsample=1 取全部格、grid_stride=16、新布局名，demo 尾窗规则不变。"""
    monkeypatch.setattr(runner, "file_sha256", lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(runner, "canonical_json",
                        lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    monkeypatch.setattr(runner, "IDENTITY_FIELDS", ())
    expected, blocks = write_source_segment(runner, tmp_path, "BinFill_ep0_demo", seg_len=50)
    assert expected["num_grid"] == 3            # demo 50 帧：grid16 起点 0/16/32
    source = runner.source_segment(tmp_path, "BinFill_ep0_demo", expected, subsample)
    assert source["source_grid"] == list(range(0, 3, subsample))
    assert source["input_block_sha256"] == [hashlib.sha256(blocks[g]).hexdigest() for g in source["source_grid"]]
    meta = runner.output_metadata(source, {}, "e" * 64, {"pid": 1}, subsample)
    assert (meta["layout"], meta["grid_stride"]) == (layout, stride)
    assert meta["num_grid"] == runner.subsampled_count(3, subsample) == len(source["source_grid"])
    assert [row["seg_offset"] for row in meta["rows"]] == [stride * j for j in range(meta["num_grid"])]
    tail = meta["rows"][-1]                     # 两种布局末窗都是起点32：18 真实帧 + 15 帧 repeat_last
    assert (tail["real_frames"], tail["pad_frames"], tail["pad_source_frame"]) == (18, 15, 49)
    assert all(row["real_frames"] >= 17 for row in meta["rows"])


def test_grid16_expected_counts_equal_all_grids(runner):
    assert runner.subsampled_count(7, 1) == 7 and runner.subsampled_count(7, 2) == 4 == (7 + 1) // 2
    assert set(runner.SUBSAMPLE_LAYOUTS) == {1, 2}


def test_cli_accepts_grid16_subsample(runner, monkeypatch):
    observed = []
    monkeypatch.setattr(runner, "export", observed.append)
    runner.main(["--ckpt", "encoder.pt", "--expected-sha256", "a" * 64, "--latents", "wan-latents",
                 "--manifest", "episode_manifest.json", "--input-manifest", "input_manifest.json",
                 "--out", "motion-tokens-stage2-grid16", "--expected-counts", "2496,45228", "--grid-subsample", "1"])
    assert observed[0].grid_subsample == 1 and observed[0].expected_counts == (2496, 45228)
