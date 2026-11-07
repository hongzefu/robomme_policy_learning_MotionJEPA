"""check_orig16_timepool.py 的 CPU 单测：每条子命令一个小夹具正例和具体负例；GPU 与真实大数据部分用假件（见文件内标注）。"""
import ast
import json
import pathlib
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "tests/dataset"))
import check_orig16_timepool as chk
from test_export_wan_timepool import build_tree, dump, load, run
from test_pack_motion_timepool import cli, pack_argv, verify_argv
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.shared import wan_timepool as tp

MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store")
REAL_VALIDATION = MAIN / "validation/orig16-stage2"
OLD_YAML = "perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml"
STAGE2_PY = MAIN / "venvs/wan-stage2/bin/python"


def run_cli(capsys, argv):
    code = chk.main(argv)
    return code, capsys.readouterr().out.strip().splitlines()


def record(out_dir, role):
    return load(pathlib.Path(out_dir) / f"{role}.json")


def fields(out_dir, role):
    return {item["field"] for item in record(out_dir, role)["findings"]}


def edit_json(path, change):
    value = load(path)
    change(value)
    dump(path, value)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """合成小树：计划、收据、恢复 latent、manifest、框架表 meta、norm；EXPECT 换成夹具值。"""
    t = build_tree(tmp_path)
    t.store = tmp_path / "store" / "motion-wan-timepool1024"
    t.records = tmp_path / "records"
    t.frames = tmp_path / "framesamp"
    (t.frames / "meta").mkdir(parents=True)
    dump(t.frames / "meta/store_meta.json", dict(layout="framesamp-4x4-v1", status="verified", manifest_scope="full",
                                                 manifest_sha256=t.manifest["sha256"],
                                                 num_exec_samples=t.manifest["totals"]["exec_samples"]))
    t.norm = tmp_path / "norm_stats.json"
    stat = dict(mean=[0.0], std=[1.0], q01=[-1.0], q99=[1.0])
    dump(t.norm, dict(norm_stats=dict(state=stat, actions=stat)))
    plan = load(t.plan_path)
    even = t.expected
    max_visible = max(chk.num_grid(d, 17, 32) + chk.num_grid(e, 33, 32) for d, e in ((83, 65), (50, 40), (0, 33)))
    new = dict(files=1, source_windows=even["source_windows"], source_copy=plan["counts"]["copy"],
               source_fill=plan["counts"]["fill"], windows=even["windows"], copy=even["copy"], fill=even["fill"],
               demo_tail=even["demo_tail"], exec_fill=even["exec_fill"], max_visible=max_visible,
               bytes=even["windows"] * 4096, episodes=3, exec_samples=t.manifest["totals"]["exec_samples"],
               episode_manifest_file_sha256=ms.sha256_file(t.manifest_path),
               episode_manifest_canonical_sha256=t.manifest["sha256"],
               input_manifest_sha256=ms.sha256_file(t.input_path), plan_canonical_sha256=plan["sha256"],
               norm_sha256=ms.sha256_file(t.norm), published_repo=plan["repo"], published_revision=plan["revision"],
               online_tasks=1, online_per_task=4, online_windows=4)
    for key, value in new.items():
        monkeypatch.setitem(chk.EXPECT, key, value)
    return t


@pytest.fixture
def exported(tree):
    run(tree)
    return tree


@pytest.fixture
def packed(exported, monkeypatch):
    cli(monkeypatch, pack_argv(exported))
    return exported


def lineage_argv(t, *extra):
    return ["lineage", "--out-dir", str(t.records), "--input-manifest", str(t.input_path), "--plan", str(t.plan_path),
            "--receipt", str(t.receipt_path), "--episode-manifest", str(t.manifest_path), "--raw-dir", str(t.raw),
            "--framesamp-store", str(t.frames), "--norm-stats", str(t.norm), "--workers", "2", *extra]


def restore_argv(t, *extra):
    return ["restore", "--out-dir", str(t.records), "--plan", str(t.plan_path), "--receipt", str(t.receipt_path),
            "--restored-receipt", str(t.restored_path), "--latents", str(t.latents), "--workers", "2", *extra]


def window_argv(t, *extra):
    return ["window", "--out-dir", str(t.records), "--plan", str(t.plan_path), "--receipt", str(t.receipt_path),
            "--episode-manifest", str(t.manifest_path), *extra]


def table_argv(t, *extra):
    return ["pool-table", "--out-dir", str(t.records), "--store", str(t.store), "--plan", str(t.plan_path),
            "--receipt", str(t.receipt_path), "--restored-receipt", str(t.restored_path), "--latents", str(t.latents),
            "--episode-manifest", str(t.manifest_path), *extra]


# ── 结构 ──────────────────────────────────────────────────────────────────────


def test_top_level_imports_are_stdlib_and_numpy_only():
    """online 要被 wan-stage2 解释器按路径运行，顶层不得依赖主环境专属包。"""
    tree = ast.parse(pathlib.Path(chk.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    assert names - set(sys.stdlib_module_names) == {"numpy"}


def test_reference_does_not_import_object_under_test():
    """AST 全文件（含函数内延迟导入）：不 import 共享池化，也不调用生产的映射校验或导出函数。"""
    tree = ast.parse(pathlib.Path(chk.__file__).read_text(encoding="utf-8"))
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert not {m for m in modules if "wan_timepool" in m or "mme_vla_suite.shared" in m or "export_wan_timepool" in m or "check_orig16_timepool_launch" in m}
    assert not names & {"validate_timepool_window_map", "validate_timepool_provenance", "run_export", "motion_store"}


@pytest.mark.skipif(not STAGE2_PY.is_file(), reason="需要 wan-stage2 环境")
def test_runs_under_stage2_interpreter(tmp_path):
    """真实 wan-stage2 解释器（无 PYTHONPATH）能按路径运行本脚本和内部 sidecar 件。"""
    env = dict(PATH="/usr/bin:/bin")
    done = subprocess.run([str(STAGE2_PY), chk.__file__, "online", "--help"], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "--online-gpu" in done.stdout
    source = tmp_path / "in.npy"
    np.save(source, np.ones((1, *chk.LATENT_SHAPE), np.float32))
    subprocess.run([str(STAGE2_PY), chk.__file__, "_sidecar-tokens", "--input", str(source), "--output",
                    str(tmp_path / "out.npy")], check=True, env=env)
    assert np.all(np.load(tmp_path / "out.npy") == 1)


# ── G0 ───────────────────────────────────────────────────────────────────────


def test_lineage_pass_and_record_fields(tree, capsys):
    code, lines = run_cli(capsys, lineage_argv(tree))
    assert code == 0
    assert lines == ["DATA_LINEAGE=PASS encoder_kind=wan_fixedpool1024 encoder_ckpt=none files=1 mismatches=0"]
    rec = record(tree.records, "DATA_LINEAGE")
    assert rec["status"] == "PASS" and rec["exit_code"] == 0 and rec["metrics"]["encoder_ckpt"] is None
    assert len(rec["code"]["head"]) == 40 and rec["code"]["script_sha256"]
    assert rec["started_at_et"][-6:] in ("-05:00", "-04:00") and rec["finished_at_et"]
    assert rec["inputs"]["input_manifest"]["sha256"] and rec["inputs"]["h5/record_dataset_T.h5"]["sha256"]
    assert "store_table_sha256" not in rec and rec["environment"]["numpy"] and rec["command"]


def test_lineage_real_expect_constants_pinned():
    """常量与计划第 6 节逐字一致，防止改值适配。"""
    assert chk.EXPECT["episode_manifest_file_sha256"].startswith("93d7c5a0c1f3")
    assert chk.EXPECT["episode_manifest_canonical_sha256"].startswith("337c4fcde163")
    assert chk.EXPECT["input_manifest_sha256"].startswith("5d1bfed54377")
    assert chk.EXPECT["plan_canonical_sha256"].startswith("b56fc7dbf224")


def test_lineage_h5_changed_one_byte(tree, capsys):
    (tree.raw / "record_dataset_T.h5").write_bytes(b"synthetic-h6")
    code, lines = run_cli(capsys, lineage_argv(tree))
    assert code == 1 and lines[0].startswith("DATA_LINEAGE=FAIL field=input_manifest.sha256 index=record_dataset_T.h5")
    assert record(tree.records, "DATA_LINEAGE")["status"] == "FAIL"


def test_lineage_h5_missing_from_receipt_set(tree, capsys):
    edit_json(tree.receipt_path, lambda r: r["source_h5_files"].clear())
    code, _ = run_cli(capsys, lineage_argv(tree))
    assert code == 1 and "receipt.file_set" in fields(tree.records, "DATA_LINEAGE")


def test_lineage_norm_and_frame_table(tree, capsys, monkeypatch):
    edit_json(tree.frames / "meta/store_meta.json", lambda m: m.update(status="packed"))
    monkeypatch.setitem(chk.EXPECT, "norm_sha256", "0" * 64)
    code, _ = run_cli(capsys, lineage_argv(tree))
    assert code == 1
    assert {"framesamp.status", "norm_sha256"} <= fields(tree.records, "DATA_LINEAGE")


def test_lineage_wrong_input_manifest_pin(tree, capsys, monkeypatch):
    monkeypatch.setitem(chk.EXPECT, "input_manifest_sha256", "a" * 64)
    assert run_cli(capsys, lineage_argv(tree))[0] == 1
    assert "input_manifest_sha256" in fields(tree.records, "DATA_LINEAGE")


# ── G1 ───────────────────────────────────────────────────────────────────────


def test_restore_pass(tree, capsys):
    code, lines = run_cli(capsys, restore_argv(tree))
    assert code == 0
    w = tree.expected["source_windows"]
    assert lines == [f"WAN_RESTORE_EQ=PASS windows={w} input_mismatches=0 latent_mismatches=0"]
    metrics = record(tree.records, "WAN_RESTORE_EQ")["metrics"]
    assert metrics["latent_windows_rehashed"] == w and metrics["latent_check"] == "full"
    assert metrics["numeric_modes_equal_original"] is True


def test_restore_sample_mode_rehashes_fewer(tree, capsys):
    assert run_cli(capsys, restore_argv(tree, "--latent-check", "sample", "--sample-every", "4"))[0] == 0
    assert record(tree.records, "WAN_RESTORE_EQ")["metrics"]["latent_windows_rehashed"] < tree.expected["source_windows"]


def test_restore_latent_bytes_flip_is_caught_at_window(tree, capsys):
    """收据没变，只有落盘字节错一位：必须靠独立重算抓到，并报到窗。"""
    key = tree.segments[0]["key"]
    path = tree.latents / f"{key}.bin"
    data = bytearray(path.read_bytes())
    data[ms.WAN_CHUNK_BYTES + 5] ^= 1
    path.write_bytes(bytes(data))
    code, lines = run_cli(capsys, restore_argv(tree))
    assert code == 1
    assert lines[0].startswith("WAN_RESTORE_EQ=FAIL")
    rec = record(tree.records, "WAN_RESTORE_EQ")
    assert any(i["field"] == "latent_bytes" and i["index"] == f"{key}.bin:1" for i in rec["findings"])
    assert rec["metrics"]["latent_mismatches"] == 1 and rec["metrics"]["failed_windows"][0]["window"] == f"{key}.bin:1"


@pytest.mark.parametrize("name,field", [("input_frames_sha256", "input_frames_sha256"), ("latent_sha256", "latent_sha256")])
def test_restore_receipt_sha_differs_from_original(tree, capsys, name, field):
    edit_json(tree.restored_path, lambda r: r["windows"][3].update({name: "a" * 64}))
    assert run_cli(capsys, restore_argv(tree))[0] == 1
    assert field in fields(tree.records, "WAN_RESTORE_EQ")


def test_restore_route_group_changed(tree, capsys):
    edit_json(tree.restored_path, lambda r: r["windows"][0].update(route="fill"))
    assert run_cli(capsys, restore_argv(tree))[0] == 1
    assert "restored.route" in fields(tree.records, "WAN_RESTORE_EQ")


def test_restore_window_missing(tree, capsys):
    edit_json(tree.restored_path, lambda r: r["windows"].pop())
    assert run_cli(capsys, restore_argv(tree))[0] == 1
    assert "restored.window_set" in fields(tree.records, "WAN_RESTORE_EQ")


def test_failed_rerun_keeps_prior_record_as_attempt(tree, capsys):
    assert run_cli(capsys, restore_argv(tree))[0] == 0
    edit_json(tree.restored_path, lambda r: r["windows"][0].update(latent_sha256="a" * 64))
    assert run_cli(capsys, restore_argv(tree))[0] == 1
    attempts = list(tree.records.glob("WAN_RESTORE_EQ.attempt-*.json"))
    assert len(attempts) == 1 and load(attempts[0])["status"] == "PASS"
    assert record(tree.records, "WAN_RESTORE_EQ")["status"] == "FAIL"


# ── G2 ───────────────────────────────────────────────────────────────────────


def test_window_pass_export_map(exported, capsys):
    t = exported
    code, lines = run_cli(capsys, window_argv(t, "--window-map", str(t.out / "window_map.json")))
    assert code == 0
    e = t.expected
    assert lines == [f"WINDOW_EQ=PASS windows={e['windows']} copy={e['copy']} fill={e['fill']} demo_tail={e['demo_tail']} "
                     f"exec_fill={e['exec_fill']} max_visible={chk.EXPECT['max_visible']}"]


def test_window_pass_store_with_index(packed, capsys):
    assert run_cli(capsys, window_argv(packed, "--store", str(packed.store)))[0] == 0
    rec = record(packed.records, "WINDOW_EQ")
    assert rec["store_table_sha256"] == ms.MotionMeta.load(packed.store).table_sha256


def map_case(t, change):
    path = t.out / "window_map.json"
    edit_json(path, change)
    return window_argv(t, "--window-map", str(path))


@pytest.mark.parametrize("change,field", [
    (lambda m: m["windows"].pop(), "window_map.rows"),
    (lambda m: m["windows"].insert(1, m["windows"].pop(2)), "window_map.row"),
    (lambda m: m["windows"][2].update(start_frame=m["windows"][2]["start_frame"] + 1), "window_map.start_frame"),
    (lambda m: m["windows"][2].update(source_m=m["windows"][2]["source_m"] + 1), "window_map.source_m"),
    (lambda m: m["windows"][2].update(m=m["windows"][2]["m"] + 1), "window_map.m"),
    (lambda m: m["windows"][0].update(route="fill" if m["windows"][0]["route"] == "copy" else "copy"), "window_map.route"),
    (lambda m: m["windows"][1].update(raw_latent_sha256="a" * 64), "window_map.raw_latent_sha256"),
    (lambda m: m["windows"][1].update(input_frames_sha256="a" * 64), "window_map.input_frames_sha256"),
    (lambda m: m["windows"][1].update(extra=1), "window_map.extra_field"),
])
def test_window_map_negative(exported, capsys, change, field):
    assert run_cli(capsys, map_case(exported, change))[0] == 1
    assert field in fields(exported.records, "WINDOW_EQ")


def test_window_tail_pad_wrong(exported, capsys):
    def change(m):
        row = next(w for w in m["windows"] if w["pad_frames"])
        row.update(pad_frames=row["pad_frames"] - 1, real_frames=row["real_frames"] + 1)
    assert run_cli(capsys, map_case(exported, change))[0] == 1
    assert {"window_map.pad_frames", "window_map.real_frames"} <= fields(exported.records, "WINDOW_EQ")


def test_window_completed_truncation_detected(exported, capsys):
    """把 completed 之后的 exec fill 窗删掉（裁切）：缺行且 exec_fill 计数不再与手算一致。"""
    def change(m):
        m["windows"] = [w for w in m["windows"] if not (w["segment"] == "exec" and w["route"] == "fill")]
    assert run_cli(capsys, map_case(exported, change))[0] == 1
    assert "window_map.rows" in fields(exported.records, "WINDOW_EQ")


def test_window_receipt_boundary_wrong(exported, capsys):
    t = exported
    def change(r):
        row = next(w for w in r["windows"] if w["pad_frames"])
        row["frame_ids"][-1] = row["frame_ids"][-1] + 1
    edit_json(t.receipt_path, change)
    assert run_cli(capsys, window_argv(t, "--window-map", str(t.out / "window_map.json")))[0] == 1
    assert "receipt.frame_ids" in fields(t.records, "WINDOW_EQ")


def test_window_count_expect_is_enforced(exported, capsys, monkeypatch):
    monkeypatch.setitem(chk.EXPECT, "demo_tail", exported.expected["demo_tail"] + 1)
    assert run_cli(capsys, window_argv(exported, "--window-map", str(exported.out / "window_map.json")))[0] == 1
    assert "demo_tail" in fields(exported.records, "WINDOW_EQ")


def test_window_index_corruption(packed, capsys):
    index = packed.store / ms.INDEX_RELPATH
    edit_json(index, lambda i: i["entries"][1]["exec"].update(row_base=i["entries"][1]["exec"]["row_base"] + 1))
    assert run_cli(capsys, window_argv(packed, "--store", str(packed.store)))[0] == 1
    assert "index.row_base" in fields(packed.records, "WINDOW_EQ")


@pytest.mark.skipif(not (REAL_VALIDATION / "wan-cache-import.json").is_file(), reason="需要真实计划与收据")
def test_window_recompute_on_real_plan_and_receipt():
    """真实计划/收据/manifest 上重算出计划第 4 节全部计数（映射本身要等真实导出才有）。"""
    findings = chk.Findings()
    rows, counts, max_visible = chk.expected_window_rows(
        findings, load(REAL_VALIDATION / "wan-cache-plan.json"), load(REAL_VALIDATION / "wan-cache-import.json"),
        load(MAIN / "datasets/16task-orig100-1600ep-9e109419/meta/episode_manifest.json"))
    assert findings.total == 0 and findings.items == []
    assert (len(rows), counts["copy"], counts["fill"], counts["demo_tail"], counts["exec_fill"], max_visible) == (
        23253, 21814, 1439, 386, 1053, 44)


# ── G3 ───────────────────────────────────────────────────────────────────────


def test_pool_contract_pass(tmp_path, capsys):
    code, lines = run_cli(capsys, ["pool-contract", "--out-dir", str(tmp_path)])
    assert code == 0 and lines == ["POOL_CONTRACT=PASS shape=1x1024 dtype=float32 params=0 raw=1 drop_first=1"]
    metrics = record(tmp_path, "POOL_CONTRACT")["metrics"]
    assert 0 < metrics["max_error_over_bound"] <= 1 and metrics["bound_gamma"] == chk.BOUND_GAMMA


def good_pool(z):
    return tp.pool_raw_wan_latent(z)


def _guard(z):
    if not isinstance(z, np.ndarray) or z.dtype != np.dtype(np.float32) or z.shape != (9, 16, 32, 32) or not np.isfinite(z).all():
        raise ValueError("拒绝")


def wrapped(body):
    def pool(z):
        _guard(z)
        return body(z)
    return pool


def _blocks(a):
    return a.reshape(16, 8, 4, 8, 4).mean(axis=(2, 4), dtype=np.float32)


BAD_POOLS = {
    "first_latent_kept": (wrapped(lambda z: _blocks(z[:8].mean(axis=0, dtype=np.float32)).reshape(1, 1024)), "drop_first"),
    "mean_all_nine": (wrapped(lambda z: _blocks(z.mean(axis=0, dtype=np.float32)).reshape(1, 1024)), "temporal_mean8"),
    "hw_swapped": (wrapped(lambda z: _blocks(z[1:9].mean(axis=0, dtype=np.float32)).transpose(0, 2, 1).reshape(1, 1024)), "flatten_CHW"),
    "channel_last_flatten": (wrapped(lambda z: _blocks(z[1:9].mean(axis=0, dtype=np.float32)).transpose(1, 2, 0).reshape(1, 1024)), "flatten_CHW"),
    "extra_l2_norm": (wrapped(lambda z: (lambda f: f / np.float32(np.linalg.norm(f) + 1e-6))(
        _blocks(z[1:9].mean(axis=0, dtype=np.float32)).reshape(1, 1024))), "fp64_bound"),
    "extra_affine": (wrapped(lambda z: (_blocks(z[1:9].mean(axis=0, dtype=np.float32)).reshape(1, 1024) * np.float32(2) + np.float32(1))), "fp64_bound"),
    "max_pool": (wrapped(lambda z: z[1:9].mean(axis=0, dtype=np.float32).reshape(16, 8, 4, 8, 4).max(axis=(2, 4)).reshape(1, 1024)), "fp64_bound"),
    "float64_output": (wrapped(lambda z: _blocks(z[1:9].mean(axis=0, dtype=np.float32)).reshape(1, 1024).astype(np.float64)), "dtype"),
    "flat_shape": (wrapped(lambda z: _blocks(z[1:9].mean(axis=0, dtype=np.float32)).reshape(1024)), "shape"),
}


def test_pool_contract_accepts_real_impl():
    findings, _ = chk.check_pool_contract(good_pool)
    assert findings.total == 0, findings.items[:3]


@pytest.mark.parametrize("name", sorted(BAD_POOLS))
def test_pool_contract_catches_mutation(name):
    fn, field = BAD_POOLS[name]
    findings, _ = chk.check_pool_contract(fn)
    assert findings.total > 0 and field in {i["field"] for i in findings.items}, (name, findings.items[:4])


def test_pool_contract_catches_other_reduction_order():
    """先块内 H 再块内 W：实数等价、误差界内，但数值合同要求固定次序逐位一致。"""
    def h_first(z):
        _guard(z)
        temporal = np.array(z[1], np.float32)
        for t in range(2, 9):
            temporal = temporal + z[t]
        temporal = temporal / np.float32(8)
        blocks = temporal.reshape(16, 8, 4, 8, 4)
        over_h = blocks[:, :, 0].copy()
        for a in range(1, 4):
            over_h += blocks[:, :, a]
        over_w = over_h[..., 0].copy()
        for b in range(1, 4):
            over_w += over_h[..., b]
        return (over_w / np.float32(16)).reshape(1, 1024)
    findings, _ = chk.check_pool_contract(h_first)
    names = {i["field"] for i in findings.items}
    assert "reduction_order_bits" in names and "fp64_bound" not in names


def test_pool_contract_catches_accepting_invalid_input():
    def lax(z):
        z = np.nan_to_num(np.asarray(z, np.float32).reshape(9, 16, 32, 32)) if np.size(z) == 9 * 16 * 32 * 32 else z
        return tp.pool_raw_wan_latent(z)
    findings, _ = chk.check_pool_contract(lax)
    assert "accepts_invalid_input" in {i["field"] for i in findings.items}


def test_pool_source_framework_import_flagged(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text("import numpy\nimport torch\n", encoding="utf-8")
    findings = chk.Findings()
    chk.check_pool_source_zero_params(source, findings)
    assert findings.items[0]["field"] == "pool_imports"


def test_independent_pool_is_bitwise_production():
    for seed in range(3):
        z = np.random.default_rng(seed).standard_normal(chk.LATENT_SHAPE, dtype=np.float32) * np.float32(1000)
        assert chk.independent_pool_f32(z).tobytes() == tp.pool_raw_wan_latent(z).tobytes()


# ── G4 ───────────────────────────────────────────────────────────────────────


def test_pool_table_pass(packed, capsys):
    code, lines = run_cli(capsys, table_argv(packed))
    rows = packed.expected["windows"]
    assert code == 0 and lines == [f"POOL_TABLE_EQ=PASS rows={rows} mismatches=0 nonfinite=0 bytes={rows * 4096}"]
    rec = record(packed.records, "POOL_TABLE_EQ")
    assert rec["store_table_sha256"] == ms.sha256_file(packed.store / ms.MOTION_TABLE_RELPATH)
    assert rec["outputs"]["motion_table"]["sha256"] == rec["store_table_sha256"]


def rewrite_table(t, change):
    path = t.store / ms.MOTION_TABLE_RELPATH
    data = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(-1, 1024).copy()
    change(data)
    path.write_bytes(data.tobytes())


def test_pool_table_one_bit(packed, capsys):
    path = packed.store / ms.MOTION_TABLE_RELPATH
    raw = bytearray(path.read_bytes())
    raw[4096 * 2 + 7] ^= 1
    path.write_bytes(bytes(raw))
    assert run_cli(capsys, table_argv(packed))[0] == 1
    rec = record(packed.records, "POOL_TABLE_EQ")
    item = next(i for i in rec["findings"] if i["field"] == "table.row_bytes")
    assert item["index"].startswith("2:") and item["index"].endswith("[1]")
    assert "store_binding" in fields(packed.records, "POOL_TABLE_EQ")


def test_pool_table_rows_reordered(packed, capsys):
    def swap(data):
        data[[1, 2]] = data[[2, 1]]
    rewrite_table(packed, swap)
    assert run_cli(capsys, table_argv(packed))[0] == 1
    indexes = {i["index"].split(":")[0] for i in record(packed.records, "POOL_TABLE_EQ")["findings"] if i["field"] == "table.row_bytes"}
    assert indexes == {"1", "2"}


def test_pool_table_first_latent_not_dropped_on_a_row(packed, capsys):
    """建库错误：某行用了含首 latent 的 9 帧均值。"""
    window = load(packed.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH)["windows"][3]
    raw = np.frombuffer((packed.latents / f"{window['key']}.bin").read_bytes(), "<f4").reshape(-1, 9, 16, 32, 32)[window["source_m"]]
    wrong = raw.mean(axis=0, dtype=np.float32).reshape(16, 8, 4, 8, 4).mean(axis=(2, 4), dtype=np.float32).reshape(1024)
    def change(data):
        data[3] = wrong
    rewrite_table(packed, change)
    assert run_cli(capsys, table_argv(packed))[0] == 1
    assert any(i["index"].startswith("3:") for i in record(packed.records, "POOL_TABLE_EQ")["findings"] if i["field"] == "table.row_bytes")


def test_pool_table_nonfinite(packed, capsys):
    def change(data):
        data[0, 5] = np.nan
    rewrite_table(packed, change)
    assert run_cli(capsys, table_argv(packed))[0] == 1
    assert "table.nonfinite" in fields(packed.records, "POOL_TABLE_EQ")


def test_pool_table_truncated(packed, capsys):
    path = packed.store / ms.MOTION_TABLE_RELPATH
    path.write_bytes(path.read_bytes()[:-4096])
    assert run_cli(capsys, table_argv(packed))[0] == 1
    assert {"table.bytes", "table.short_row"} <= fields(packed.records, "POOL_TABLE_EQ")


def test_pool_table_raw_latent_tampered(packed, capsys):
    window = load(packed.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH)["windows"][0]
    path = packed.latents / f"{window['key']}.bin"
    raw = bytearray(path.read_bytes())
    raw[9] ^= 1
    path.write_bytes(bytes(raw))
    assert run_cli(capsys, table_argv(packed))[0] == 1
    assert "raw_latent_sha256" in fields(packed.records, "POOL_TABLE_EQ")


def test_pool_table_provenance_identity(packed, capsys):
    def change(meta):
        meta["provenance"]["encoder_ckpt"] = "abc"
    edit_json(packed.store / ms.META_RELPATH, change)
    assert run_cli(capsys, table_argv(packed))[0] == 1
    assert "provenance.encoder_ckpt" in fields(packed.records, "POOL_TABLE_EQ")


# ── bind 与 pack verify 全链 ──────────────────────────────────────────────────


def test_bind_then_pack_verify_accepts_all_five(tree, capsys, monkeypatch):
    """先期记录（建库前无绑定）→ export → pack → 后续记录 → bind → pack verify 真实写出 verified。"""
    t = tree
    assert run_cli(capsys, lineage_argv(t))[0] == 0
    assert run_cli(capsys, restore_argv(t))[0] == 0
    assert run_cli(capsys, ["pool-contract", "--out-dir", str(t.records)])[0] == 0
    before = record(t.records, "WAN_RESTORE_EQ")
    assert "store_table_sha256" not in before
    run(t)
    cli(monkeypatch, pack_argv(t))
    assert run_cli(capsys, window_argv(t, "--store", str(t.store)))[0] == 0
    assert run_cli(capsys, table_argv(t))[0] == 0
    code, lines = run_cli(capsys, ["bind", "--out-dir", str(t.records), "--store", str(t.store)])
    assert code == 0 and lines[0].startswith("GATE_BIND=PASS bound=3 already=2")
    after = record(t.records, "WAN_RESTORE_EQ")
    assert after["verdict"] == before["verdict"] and after["bound_at"][-6:] in ("-05:00", "-04:00")
    assert {k: v for k, v in after.items() if k not in ("store_table_sha256", "motion_index_sha256", "bound_at",
                                                        "pre_bind_record_sha256")} == before
    cli(monkeypatch, verify_argv(t, records=True))
    meta = ms.MotionMeta.load(t.store)
    assert meta.status == "verified" and set(meta.provenance["validation"]) == set(chk.BIND_ROLES)
    # 幂等：再 bind 不动已绑定记录
    sha = ms.sha256_file(t.records / "WAN_RESTORE_EQ.json")
    assert run_cli(capsys, ["bind", "--out-dir", str(t.records), "--store", str(t.store)])[0] == 0
    assert ms.sha256_file(t.records / "WAN_RESTORE_EQ.json") == sha


def test_bind_refuses_failed_record_and_other_table(packed, capsys):
    t = packed
    assert run_cli(capsys, ["pool-contract", "--out-dir", str(t.records)])[0] == 0
    code, lines = run_cli(capsys, ["bind", "--out-dir", str(t.records), "--store", str(t.store), "--roles", "POOL_CONTRACT"])
    assert code == 0
    edit_json(t.records / "POOL_CONTRACT.json", lambda r: r.update(store_table_sha256="a" * 64))
    code, lines = run_cli(capsys, ["bind", "--out-dir", str(t.records), "--store", str(t.store), "--roles", "POOL_CONTRACT"])
    assert code == 1 and "已绑定另一张表" in lines[0]
    dump(t.records / "WINDOW_EQ.json", dict(status="FAIL", verdict="WINDOW_EQ=FAIL field=x"))
    code, lines = run_cli(capsys, ["bind", "--out-dir", str(t.records), "--store", str(t.store), "--roles", "WINDOW_EQ"])
    assert code == 1 and "不是 PASS" in lines[0]
    code, lines = run_cli(capsys, ["bind", "--out-dir", str(t.records), "--store", str(t.store), "--roles", "DATA_LINEAGE"])
    assert code == 1 and "缺记录" in lines[0]


def test_expect_head_mismatch_fails(tmp_path, capsys):
    code, lines = run_cli(capsys, ["pool-contract", "--out-dir", str(tmp_path), "--expect-head", "0" * 40])
    assert code == 1 and "code_head" in lines[0]


# ── G5 参数（真实 eval_shape）──────────────────────────────────────────────────


@pytest.fixture
def new_yaml(tmp_path):
    cfg = OmegaConf.load(chk.YAML_DIR / OLD_YAML)
    cfg.motion.readout = tp.TIMEPOOL_READOUT
    cfg.motion.encoder_kind = tp.TIMEPOOL_ENCODER_KIND
    cfg.motion.store_path = "v1-store/datasets/16task-orig100-1600ep-9e109419/motion-wan-timepool1024"
    del cfg.motion["source_run"]
    del cfg.motion["checkpoint_sha256"]
    path = tmp_path / "perceptual-new.yaml"
    OmegaConf.save(cfg, path)
    return path


def test_vla_param_real_models(tmp_path, new_yaml, capsys, monkeypatch):
    """真实 HistoryPi0 + 真实 freeze_filter 的 eval_shape；新 YAML 用与 D 约定同形的合成件（D 的 YAML 不在本基线）。"""
    monkeypatch.chdir(ROOT)
    out = tmp_path / "records"
    code, lines = run_cli(capsys, ["vla-param", "--out-dir", str(out), "--old-yaml", OLD_YAML, "--new-yaml", str(new_yaml)])
    assert code == 0
    assert lines[-1] == ("VLA_PARAM_EQ=PASS total=3363663120 trainable=2948859424 frozen=414803696 leaves=59 trainable_leaves=36")
    metrics = record(out, "VLA_PARAM_EQ")["metrics"]
    assert metrics["totals"]["mem_encoder"] == 10229248 and metrics["totals"]["motion_path"] == 3869440
    assert metrics["inventory"]["mem_encoder/motion_pos_proj/kernel"] == dict(shape=[256, 768], dtype="float32", frozen=False, numel=196608)
    assert metrics["inventory"]["PaliGemma/img/Transformer/encoder_norm/bias"]["dtype"] == "bfloat16"
    assert metrics["inventory"]["PaliGemma/img/Transformer/encoder_norm/bias"]["frozen"] is True


def test_vla_param_real_shape_change_fails(tmp_path, new_yaml, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = OmegaConf.load(new_yaml)
    cfg.motion.dim = 512
    OmegaConf.save(cfg, new_yaml)
    out = tmp_path / "records"
    assert run_cli(capsys, ["vla-param", "--out-dir", str(out), "--old-yaml", OLD_YAML, "--new-yaml", str(new_yaml)])[0] == 1
    assert "leaf.shape" in fields(out, "VLA_PARAM_EQ")


def fake_inventory():
    leaves = {}
    for index in range(chk.EXPECT["leaves"]):
        frozen = index >= chk.EXPECT["trainable_leaves"]
        leaves[f"tower/leaf{index}"] = dict(shape=[1], dtype="bfloat16" if frozen else "float32", frozen=frozen, numel=1)
    return leaves


@pytest.mark.parametrize("change,field", [
    (lambda inv: inv.pop("tower/leaf0"), "leaf_missing_in_new"),
    (lambda inv: inv.update({"pool/kernel": dict(shape=[1], dtype="float32", frozen=False, numel=1)}), "leaf_extra_in_new"),
    (lambda inv: inv["tower/leaf1"].update(frozen=True), "leaf.frozen"),
    (lambda inv: inv["tower/leaf1"].update(shape=[2]), "leaf.shape"),
    (lambda inv: inv["tower/leaf1"].update(dtype="float64"), "leaf.dtype"),
    (lambda inv: inv["tower/leaf1"].update(numel=2), "leaf.numel"),
])
def test_vla_param_leaf_negative(tmp_path, capsys, monkeypatch, change, field):
    """假清单：总数恰好相等也要逐叶抓路径与冻结标志（改写总数让 EXPECT 不干扰叶级字段）。"""
    old_inv, new_inv = fake_inventory(), fake_inventory()
    change(new_inv)
    monkeypatch.setattr(chk, "build_param_inventory", lambda path: new_inv if path.name == "new.yaml" else old_inv)
    (tmp_path / "old.yaml").write_text("a: 1\n")
    (tmp_path / "new.yaml").write_text("a: 1\n")
    out = tmp_path / "r"
    assert run_cli(capsys, ["vla-param", "--out-dir", str(out), "--old-yaml", str(tmp_path / "old.yaml"), "--new-yaml", str(tmp_path / "new.yaml")])[0] == 1
    assert field in fields(out, "VLA_PARAM_EQ")


def test_vla_init_fake(tmp_path, capsys, monkeypatch):
    (tmp_path / "old.yaml").write_text("a: 1\n")
    (tmp_path / "new.yaml").write_text("a: 1\n")
    tables = {"old.yaml": {"x": dict(shape=[1], dtype="float32", sha256="1" * 64)},
              "new.yaml": {"x": dict(shape=[1], dtype="float32", sha256="1" * 64)}}
    monkeypatch.setattr(chk, "build_param_digests", lambda path, seed: tables[path.name])
    out = tmp_path / "r"
    argv = ["vla-init", "--out-dir", str(out), "--old-yaml", str(tmp_path / "old.yaml"), "--new-yaml", str(tmp_path / "new.yaml")]
    code, lines = run_cli(capsys, argv)
    assert code == 0 and lines == ["VLA_INIT_EQ=PASS leaves=1 mismatches=0 seed=42"]
    tables["new.yaml"]["x"]["sha256"] = "2" * 64
    code, lines = run_cli(capsys, argv)
    assert code == 1 and lines[0].startswith("VLA_INIT_EQ=FAIL field=leaf_digest index=x")


# ── G5 配置 ───────────────────────────────────────────────────────────────────


def config_records():
    fields_ = dict(exp_name="old", batch_size=128, fsdp_devices=8, num_workers=16, seed=42, ema_decay=0.999, log_interval=100,
                   save_interval=5000, keep_period=5000, num_train_steps=60000,
                   lr_schedule=dict(type="s", fields=dict(warmup_steps=5000, peak_lr=5e-5, decay_steps=60000, decay_lr=5e-5)),
                   model=dict(type="m", fields=dict(history_config="old.yaml", pi05=True)))
    old = dict(config_record_version=2, train_config=dict(type="t", fields=fields_),
               resolved_data=dict(x=1), assets=dict(norm_stats=dict(sha256="n")),
               history_values=dict(budget=512, motion=dict(readout="stage2_pool4_concat_v1", encoder_kind="stage2_pool4_concat", store_path="old",
                                                            source_run="r", checkpoint_sha256="s", online_gpu=1, source_shape=[1, 1024],
                                                            num_tokens=1, dim=1024, budget=96)),
               derived=dict(checkpoint_dir="/old", assets_dirs="/a"))
    new = json.loads(json.dumps(old))
    new["train_config"]["fields"]["exp_name"] = "new"
    new["train_config"]["fields"]["model"]["fields"]["history_config"] = "new.yaml"
    new["derived"]["checkpoint_dir"] = "/new"
    motion = new["history_values"]["motion"]
    motion.update(readout=tp.TIMEPOOL_READOUT, encoder_kind=tp.TIMEPOOL_ENCODER_KIND, store_path="new")
    del motion["source_run"], motion["checkpoint_sha256"]
    return old, new


def test_config_diff_exact_set_passes():
    old, new = config_records()
    findings = chk.Findings()
    differences = chk.check_config_diff(findings, old, new, steps=60000)
    assert findings.total == 0 and {d[0] for d in differences} == set(chk.ALLOWED_CONFIG_DIFF)


@pytest.mark.parametrize("change,field,index", [
    (lambda n: n["train_config"]["fields"].update(batch_size=64), "unauthorized_diff", "train_config.fields.batch_size"),
    (lambda n: n["train_config"]["fields"]["model"]["fields"].update(pi05=False), "unauthorized_diff", "train_config.fields.model.fields.pi05"),
    (lambda n: n["train_config"]["fields"]["lr_schedule"]["fields"].update(decay_steps=20), "unauthorized_diff", "train_config.fields.lr_schedule.fields.decay_steps"),
    (lambda n: n["assets"]["norm_stats"].update(sha256="x"), "unauthorized_diff", "assets.norm_stats.sha256"),
    (lambda n: n["history_values"].update(budget=1024), "unauthorized_diff", "history_values.budget"),
    (lambda n: n["history_values"]["motion"].update(online_gpu=0), "unauthorized_diff", "history_values.motion.online_gpu"),
    (lambda n: n["history_values"]["motion"].update(extra=1), "unauthorized_diff", "history_values.motion.extra"),
    (lambda n: n["history_values"]["motion"].update(readout="stage2_pool4_concat_v1"), "expected_diff_absent", "history_values.motion.readout"),
    (lambda n: n["history_values"]["motion"].update(source_run="r"), "expected_diff_absent", "history_values.motion.source_run"),
])
def test_config_diff_negative(change, field, index):
    old, new = config_records()
    change(new)
    findings = chk.Findings()
    chk.check_config_diff(findings, old, new, steps=60000)
    assert any(i["field"] == field and i["index"] == index for i in findings.items), findings.items


def test_config_diff_smoke_only_adds_step_limit():
    old, new = config_records()
    new["train_config"]["fields"]["num_train_steps"] = 20
    findings = chk.Findings()
    chk.check_config_diff(findings, old, new, steps=20)
    assert findings.total == 0
    findings = chk.Findings()
    chk.check_config_diff(findings, old, new, steps=60000)
    assert "unauthorized_diff" in {i["field"] for i in findings.items}


def test_train_config_cli_fake_builder(tmp_path, capsys, monkeypatch):
    old, new = config_records()
    old_path = tmp_path / "start.json"
    dump(old_path, dict(complete=old))
    seen = {}
    monkeypatch.setattr(chk, "build_complete_record", lambda argv, root: seen.update(argv=argv) or new)
    (tmp_path / "new.yaml").write_text("a: 1\n")
    out = tmp_path / "r"
    argv = ["train-config", "--out-dir", str(out), "--old-start-record", str(old_path), "--new-yaml", str(tmp_path / "new.yaml"),
            "--new-run", "new", "--store-root", str(tmp_path)]
    code, lines = run_cli(capsys, argv)
    assert code == 0 and lines == ["TRAIN_CONFIG_EQ=PASS"]
    assert seen["argv"][0] == chk.CONFIG_NAME and "--model.history-config" in seen["argv"] and "--num-train-steps" not in seen["argv"]
    new["history_values"]["budget"] = 1024
    assert run_cli(capsys, argv)[0] == 1


def test_runner_argv_shape_for_smoke(tmp_path):
    argv = chk.runner_argv(tmp_path, "o16-smoke-x", "h.yaml", 20, "lib")
    assert argv[argv.index("--num-train-steps") + 1] == "20" and "--no-wandb-enabled" in argv
    assert argv[argv.index("--dataset-path") + 1] == str(tmp_path / "v1-store/datasets/lib/framesamp-4x4")


# ── G5 固定样本（假件：真实 FrameSampDataset 路径见报告未覆盖清单）────────────────


def base_sample(idx):
    k = 2
    emb = np.zeros((96, 1024), np.float32)
    emb[:k] = 1.5
    def tree():
        return dict(static_image_emb=np.full((512, 2048), idx, np.float32), static_pos_emb=np.full((512, 768), 2.0, np.float32),
                    static_mask=np.arange(512) < 64, static_state_emb=np.ones((512, 8)), actions=np.full((20, 8), idx, np.float64),
                    motion_pos=np.ones((96, 256), np.float32), motion_mask=np.arange(96) < k, mem_order=np.arange(608),
                    prompt="task", extra=None, motion_emb=emb.copy())
    return dict(raw=tree(), transformed=tree())


def fake_source_pair(monkeypatch, mutate):
    def opener(yaml_path, store_root, library, motion_root):
        def get(idx):
            sample = base_sample(idx)
            if "new" in pathlib.Path(motion_root).name:
                sample["raw"]["motion_emb"][:2] = 2.5
                sample["transformed"]["motion_emb"][:2] = 2.5
                mutate(sample)
            return sample
        return 10, get
    monkeypatch.setattr(chk, "open_sample_source", opener)


def non_motion_argv(tmp_path):
    (tmp_path / "o.yaml").write_text("a: 1\n")
    (tmp_path / "n.yaml").write_text("a: 1\n")
    return ["non-motion", "--out-dir", str(tmp_path / "r"), "--old-yaml", str(tmp_path / "o.yaml"), "--new-yaml", str(tmp_path / "n.yaml"),
            "--store-root", str(tmp_path), "--old-store", str(tmp_path / "old"), "--new-store", str(tmp_path / "new"), "--samples", "4"]


def test_non_motion_pass(tmp_path, capsys, monkeypatch):
    fake_source_pair(monkeypatch, lambda s: None)
    code, lines = run_cli(capsys, non_motion_argv(tmp_path))
    assert code == 0 and lines == ["NON_MOTION_EQ=PASS mismatches=0"]
    metrics = record(tmp_path / "r", "NON_MOTION_EQ")["metrics"]
    assert metrics["motion_content_changed"] == metrics["samples_with_windows"] == metrics["samples"]


def set_key(group, key, value):
    return lambda s: s[group].__setitem__(key, value(s[group][key]))


@pytest.mark.parametrize("mutate,field", [
    (set_key("raw", "static_image_emb", lambda a: a + 1), "sample.array"),
    (set_key("raw", "static_mask", lambda a: ~a), "sample.array"),
    (set_key("raw", "motion_mask", lambda a: np.roll(a, 1)), "sample.array"),
    (set_key("raw", "mem_order", lambda a: a[::-1]), "sample.array"),
    (set_key("transformed", "actions", lambda a: a * 2), "sample.array"),
    (set_key("raw", "motion_pos", lambda a: a.astype(np.float64)), "sample.array"),
    (set_key("raw", "prompt", lambda a: "other"), "sample.value"),
    (set_key("raw", "extra", lambda a: 1), "sample.type"),
    (lambda s: s["raw"].pop("static_pos_emb"), "sample.keys"),
    (lambda s: s["raw"]["motion_emb"].__setitem__((50, 0), 9.0), "motion_padding.new"),
    (lambda s: s["raw"].__setitem__("motion_emb", s["raw"]["motion_emb"][:50]), "motion_emb.shape"),
    (lambda s: s["raw"]["motion_emb"].__setitem__(slice(0, 2), 1.5), "motion_content_unchanged"),
])
def test_non_motion_negative(tmp_path, capsys, monkeypatch, mutate, field):
    fake_source_pair(monkeypatch, mutate)
    assert run_cli(capsys, non_motion_argv(tmp_path))[0] == 1
    assert field in fields(tmp_path / "r", "NON_MOTION_EQ")


def test_sample_indices_fixed_and_boundary():
    assert chk.sample_indices(100, 5, 42) == chk.sample_indices(100, 5, 42)
    assert {0, 99} <= set(chk.sample_indices(100, 5, 42))


# ── G6（真实 sidecar 模块，CPU）────────────────────────────────────────────────


def test_pool_shared_pass_synthetic(tmp_path, capsys):
    code, lines = run_cli(capsys, ["pool-shared", "--out-dir", str(tmp_path)])
    assert code == 0 and lines == ["POOL_SHARED_EQ=PASS max_abs=0 bitwise_equal=1"]


def test_pool_shared_pass_real_rows_and_cross_process(packed, capsys):
    t = packed
    code, lines = run_cli(capsys, ["pool-shared", "--out-dir", str(t.records), "--store", str(t.store), "--latents", str(t.latents),
                                   "--sidecar-python", sys.executable])
    assert code == 0 and lines == ["POOL_SHARED_EQ=PASS max_abs=0 bitwise_equal=1"]
    metrics = record(t.records, "POOL_SHARED_EQ")["metrics"]
    assert metrics["samples"] == 14 and metrics["cross_env"] is True


@pytest.mark.skipif(not STAGE2_PY.is_file(), reason="需要 wan-stage2 环境")
def test_pool_shared_cross_interpreter_real_stage2(tmp_path, capsys):
    """主环境 NumPy 与 wan-stage2 NumPy 真实逐位比（sidecar 实际跑在后者）。"""
    code, lines = run_cli(capsys, ["pool-shared", "--out-dir", str(tmp_path), "--sidecar-python", str(STAGE2_PY)])
    assert code == 0 and lines == ["POOL_SHARED_EQ=PASS max_abs=0 bitwise_equal=1"]


def write_fake_sidecar(path, body):
    path.write_text(f"import numpy as np\nPOOL_PATH = {str(chk.POOL_SOURCE)!r}\n"
                    f"def token_from_latent(latent):\n{body}\n", encoding="utf-8")


def test_pool_shared_catches_sidecar_difference(tmp_path, capsys):
    bad = tmp_path / "side.py"
    write_fake_sidecar(bad, "    return (latent[:8].mean(axis=0).reshape(16,8,4,8,4).mean(axis=(2,4))).reshape(1024)")
    out = tmp_path / "r"
    code, lines = run_cli(capsys, ["pool-shared", "--out-dir", str(out), "--sidecar-source", str(bad)])
    assert code == 1 and lines[0].startswith("POOL_SHARED_EQ=FAIL")
    assert "bitwise" in fields(out, "POOL_SHARED_EQ")


def test_pool_shared_catches_table_row_mismatch(packed, capsys):
    t = packed
    path = t.store / ms.MOTION_TABLE_RELPATH
    raw = bytearray(path.read_bytes())
    selected, _ = chk.select_named_windows(load(t.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH)["windows"], 4)
    raw[selected[0]["row"] * 4096] ^= 1
    path.write_bytes(bytes(raw))
    assert run_cli(capsys, ["pool-shared", "--out-dir", str(t.records), "--store", str(t.store), "--latents", str(t.latents)])[0] == 1
    assert "table_row" in fields(t.records, "POOL_SHARED_EQ")


# ── G7（假后端：真实 VAE、H5 和 IPC 见报告未覆盖清单）──────────────────────────


def frames_of(window):
    return np.full((33, 256, 256, 3), window["row"] % 251, dtype=np.uint8)


class FakeFrames:
    def __init__(self, raw_dir, corrupt=None):
        self.corrupt = corrupt

    def read(self, window, episode):
        frame = frames_of(window)
        if self.corrupt == window["row"]:
            frame = frame.copy()
            frame[0, 0, 0, 0] ^= 1
        return frame

    def close(self):
        pass


class FakeBackend:
    """把帧映射回恢复 latent 再加可控偏差；不用 GPU。state=True 模拟跨窗残留。"""

    def __init__(self, t, *, noise=0.0, state=False, ipc_delta=False, online_tf32=False, nonfinite=False):
        self.t, self.noise, self.state, self.ipc_delta = t, noise, state, ipc_delta
        self.online_tf32, self.nonfinite = online_tf32, nonfinite
        self.previous = 0.0
        self.windows = {frames_of(w)[0, 0, 0, 0].item(): w for w in load(t.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH)["windows"]}
        self.closed = False

    def encode(self, frames):
        window = self.windows[frames[0, 0, 0, 0].item()]
        latent = chk.read_restored_latent(self.t.latents, window) + np.float32(self.noise)
        if self.state:
            latent = latent + np.float32(self.previous)
            self.previous = (self.previous + 1e-3) % 1
        if self.nonfinite:
            latent = latent.copy()
            latent[0, 0, 0, 0] = np.nan
        return latent

    def token(self, latent):
        return tp.pool_raw_wan_latent(np.asarray(latent, np.float32)).reshape(1024)

    def ipc(self, frames, start):
        token = self.token(self.encode(frames))
        return token + np.float32(1e-3) if self.ipc_delta else token

    def numerics(self):
        return dict(flags=dict(cudnn_allow_tf32=self.online_tf32), handshake=dict(encoder_kind=chk.KIND))

    def close(self):
        self.closed = True


@pytest.fixture
def online_tree(packed, monkeypatch):
    """verified 库 + 映射里的帧 SHA 换成假帧真实 SHA（夹具原值是文本哈希）。"""
    t = packed
    t.records.mkdir(exist_ok=True)
    meta_path = t.store / ms.META_RELPATH
    path = t.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH
    def change(m):
        for window in m["windows"]:
            window["input_frames_sha256"] = chk.sha_bytes(frames_of(window).tobytes())
    edit_json(path, change)
    edit_json(meta_path, lambda meta: meta.update(status="verified"))
    return t


def online_argv(t):
    return ["online", "--out-dir", str(t.records), "--store", str(t.store), "--latents", str(t.latents),
            "--restored-receipt", str(t.restored_path), "--episode-manifest", str(t.manifest_path), "--raw-dir", str(t.raw),
            "--v1-store", str(t.root), "--online-gpu", "1"]


def run_online(t, capsys, monkeypatch, *, frames=None, **backend):
    built = {}

    def factory(args, provenance):
        built["backend"] = FakeBackend(t, **backend)
        return built["backend"]

    monkeypatch.setattr(chk, "RealOnlineBackend", factory)
    monkeypatch.setattr(chk, "H5FrameSource", lambda raw: FakeFrames(raw, corrupt=frames))
    result = run_cli(capsys, online_argv(t))
    assert built["backend"].closed
    return result


def test_online_pass_with_real_errors_recorded(online_tree, capsys, monkeypatch):
    code, lines = run_online(online_tree, capsys, monkeypatch, noise=1e-4)
    assert code == 0
    assert lines == ["ONLINE_WINDOW_EQ=PASS windows=4 input_mismatches=0 cross_window_state=0",
                     "ONLINE_NUMERICS=RECORDED copy_tf32=true fill_online_tf32=false"]
    numerics = record(online_tree.records, "ONLINE_NUMERICS")["metrics"]
    copy = numerics["routes"]["copy"]
    assert copy["latent"]["max_abs"] > 0 and copy["pool"]["rel_l2_max"] > 0 and 0.99 < copy["pool"]["cosine_min"] <= 1.0000001
    assert set(copy["latent"]["abs_quantiles"]) == {"p50", "p90", "p99", "p99.9"}
    assert numerics["routes"]["fill"]["latent"]["windows"] >= 1
    window = record(online_tree.records, "ONLINE_WINDOW_EQ")["metrics"]
    assert {r["category"] for r in window["rows"]} == set(chk.ONLINE_CATEGORIES)
    assert window["missing_categories"] == {}


def test_online_catches_cross_window_state(online_tree, capsys, monkeypatch):
    code, lines = run_online(online_tree, capsys, monkeypatch, state=True)
    assert code == 1 and lines[0].startswith("ONLINE_WINDOW_EQ=FAIL field=cross_window_state")


def test_online_catches_wrong_frames(online_tree, capsys, monkeypatch):
    selected, _ = chk.select_named_windows(load(online_tree.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH)["windows"], 4)
    code, lines = run_online(online_tree, capsys, monkeypatch, frames=selected[1]["row"])
    assert code == 1 and lines[0].startswith("ONLINE_WINDOW_EQ=FAIL field=input_frames_sha256")


def test_online_catches_ipc_difference(online_tree, capsys, monkeypatch):
    code, lines = run_online(online_tree, capsys, monkeypatch, ipc_delta=True)
    assert code == 1 and lines[0].startswith("ONLINE_WINDOW_EQ=FAIL field=ipc_vs_direct")


def test_online_catches_nonfinite(online_tree, capsys, monkeypatch):
    assert run_online(online_tree, capsys, monkeypatch, nonfinite=True)[0] == 1
    assert "latent.nonfinite" in fields(online_tree.records, "ONLINE_WINDOW_EQ")


def test_online_numerics_flag_mismatch(online_tree, capsys, monkeypatch):
    code, lines = run_online(online_tree, capsys, monkeypatch, online_tf32=True)
    assert code == 1
    assert lines[0].startswith("ONLINE_WINDOW_EQ=PASS") and lines[1].startswith("ONLINE_NUMERICS=FAIL field=online_tf32")


def test_online_store_not_verified_is_recorded(packed, capsys, monkeypatch):
    def change(m):
        for window in m["windows"]:
            window["input_frames_sha256"] = chk.sha_bytes(frames_of(window).tobytes())
    edit_json(packed.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH, change)
    assert run_online(packed, capsys, monkeypatch)[0] == 1
    assert "store.status" in fields(packed.records, "ONLINE_WINDOW_EQ")


def mk_window(row, key, segment, route, pad=0):
    return dict(row=row, key=key, segment=segment, route=route, pad_frames=pad, m=row)


def test_select_covers_each_category():
    rows = [mk_window(0, "A_ep0_demo", "demo", "copy"), mk_window(1, "A_ep0_demo", "demo", "fill", 3),
            mk_window(2, "A_ep0_exec", "exec", "copy"), mk_window(3, "A_ep0_exec", "exec", "fill")]
    chosen, coverage = chk.select_named_windows(rows, 4)
    assert [w["row"] for w in chosen] == [0, 1, 2, 3] and coverage["A"]["missing"] == []


def test_select_records_missing_category_and_tops_up():
    """任务完全无 demo：明确记 missing，并用现有类别补足每任务 4 窗。"""
    rows = [mk_window(i, "B_ep0_exec", "exec", "copy") for i in range(10)] + [mk_window(10, "B_ep1_exec", "exec", "fill")]
    chosen, coverage = chk.select_named_windows(rows, 4)
    assert coverage["B"]["missing"] == ["demo_complete", "demo_tail"]
    assert len(chosen) == 4 and len({w["row"] for w in chosen}) == 4
    assert coverage["B"]["rows"]["exec_fill"] == [10]


def test_select_rejects_too_few_tasks(online_tree, capsys, monkeypatch):
    monkeypatch.setitem(chk.EXPECT, "online_tasks", 16)
    monkeypatch.setitem(chk.EXPECT, "online_windows", 64)
    assert run_online(online_tree, capsys, monkeypatch)[0] == 1
    assert {"tasks", "windows"} <= fields(online_tree.records, "ONLINE_WINDOW_EQ")


# ── G8 ───────────────────────────────────────────────────────────────────────


def write_snapshot(run_root, t):
    """按 verify_timepool_snapshot 合同手工建 run 根快照；真实写入方 train.py 由主会话/T 提供。"""
    run_root.mkdir(parents=True)
    (run_root / "motion_evidence").mkdir()
    meta = ms.MotionMeta.load(t.store)
    timepool = meta.provenance
    files = {}

    def put(relative, data):
        (run_root / relative).write_bytes(data)
        files[relative] = ms.sha256_bytes(data)

    resolved = b"motion:\n  encoder_kind: wan_fixedpool1024\n  readout: wan_raw_timepool8_spatial8x8_v1\n"
    put("history_config.txt", b"new.yaml")
    put("history_config.resolved.yaml", resolved)
    put("history_config.resolved.sha256", (ms.sha256_bytes(resolved) + "\n").encode())
    put("timepool_window_map.json", (t.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH).read_bytes())
    payload = b"npz"
    put("online_restore_inputs.npz", payload)
    put("online_restore_inputs.json", json.dumps(dict(schema="stage2-online-restore-inputs-v1", payload=dict(
        path="online_restore_inputs.npz", sha256=ms.sha256_bytes(payload)))).encode())
    for role, item in timepool["validation"].items():
        put(f"motion_evidence/{role}.json", pathlib.Path(item["path"]).read_bytes())
    prov = dict(history_config_name="new.yaml", motion_enabled=True, resolved_sha256=files["history_config.resolved.yaml"],
                framesamp_root="/f", framesamp_manifest_sha256="a" * 64, framesamp_store_meta_sha256="b" * 64, motion_root=str(t.store),
                motion_manifest_sha256=timepool["manifests"]["episode_manifest_sha256"], motion_index_sha256=timepool["output"]["motion_index_sha256"],
                motion_store_meta_sha256="c" * 64, motion_table_sha256=timepool["output"]["motion_table_sha256"], vae=None, encoder=None, timepool=timepool)
    put("motion_provenance.json", json.dumps(prov, ensure_ascii=False).encode())
    dump(run_root / "motion_snapshot.ready.json", dict(schema="wan-timepool-motion-snapshot-v1", files=files))
    (run_root / "19").mkdir()


def good_reference():
    return dict(schema="stage2-pre-save-reference-v1", step=19, state_step=20, seed=42, weight_type="ema",
                inference=dict(parameter_dtype="bfloat16", num_steps=10, rng="jax.random.split(jax.random.key(seed))[1]",
                               sharding=dict(kind="SingleDeviceSharding", batch_size=1)))


def good_report(with_reference=True):
    report = dict(strict_restore=dict(missing=0, extra=0, shape_mismatch=0, dtype_mismatch=0, leaves=chk.EXPECT["leaves"]),
                  worker_encoder_references=[], online=dict(online_infer=1, sidecar_provenance=dict(encoder_kind=chk.KIND)))
    if with_reference:
        report.update(ema_mismatch=0, reference_output_max_abs=0.0)
    return report


@pytest.fixture
def save_load_env(packed, tmp_path, monkeypatch):
    """真实 pack 的库 + 绑定该表的五份记录写入 verified + 手工快照 + 假 restore_and_infer（GPU 路径未覆盖，见报告）。"""
    t = packed
    meta = ms.MotionMeta.load(t.store)
    binding = dict(store_table_sha256=meta.table_sha256, motion_index_sha256=meta.motion_index_sha256)
    t.records.mkdir(exist_ok=True)
    validation = {}
    for role in chk.BIND_ROLES:
        path = t.records / f"{role}.json"
        dump(path, dict(role=role, status="PASS", verdict=f"{role}=PASS", **binding))
        validation[role] = dict(path=str(path), sha256=ms.sha256_file(path), verdict=f"{role}=PASS")
    edit_json(t.store / ms.META_RELPATH, lambda m: (m.update(status="verified"), m["provenance"].update(validation=validation)))
    t.run_root = tmp_path / "run"
    write_snapshot(t.run_root, t)
    t.reference = tmp_path / "pre_save_reference.json"
    dump(t.reference, good_reference())
    t.report = good_report()
    monkeypatch.setattr(chk, "restore_and_infer", lambda args, ckpt, ref, path: t.report)
    t.out = tmp_path / "gate"
    t.argv = ["save-load", "--out-dir", str(t.out), "--checkpoint", str(t.run_root / "19"), "--reference", str(t.reference), "--online-gpu", "1"]
    return t


def test_save_load_pass(save_load_env, capsys):
    code, lines = run_cli(capsys, save_load_env.argv)
    assert code == 0 and lines == ["SAVE_LOAD=PASS checkpoint_step=19 state_step=20"], lines


@pytest.mark.parametrize("key,value,field", [
    ("step", 18, "reference.step"), ("state_step", 19, "reference.state_step"), ("weight_type", "raw", "reference.weight_type"),
    ("schema", "x", "reference.schema"),
])
def test_save_load_reference_negative(save_load_env, capsys, key, value, field):
    edit_json(save_load_env.reference, lambda r: r.update({key: value}))
    assert run_cli(capsys, save_load_env.argv)[0] == 1
    assert field in fields(save_load_env.out, "SAVE_LOAD")


def test_save_load_wrong_step_dir(save_load_env, capsys):
    t = save_load_env
    (t.run_root / "20").mkdir()
    argv = list(t.argv)
    argv[argv.index("--checkpoint") + 1] = str(t.run_root / "20")
    assert run_cli(capsys, argv)[0] == 1
    assert "checkpoint_step" in fields(t.out, "SAVE_LOAD")


def test_save_load_snapshot_tampered_uses_real_verifier(save_load_env, capsys):
    t = save_load_env
    (t.run_root / "timepool_window_map.json").write_bytes(b"{}")
    code, lines = run_cli(capsys, t.argv)
    assert code == 1 and lines[0].startswith("SAVE_LOAD=FAIL field=exception") and "SHA" in lines[0]


def test_save_load_stage2_dependency_file_rejected(save_load_env, capsys):
    t = save_load_env
    (t.run_root / "encoder_dependencies.json").write_text("{}")
    assert run_cli(capsys, t.argv)[0] == 1
    assert "snapshot.encoder_dependencies" in fields(t.out, "SAVE_LOAD")


@pytest.mark.parametrize("change,field", [
    (lambda r: r.update(ema_mismatch=2), "ema_mismatch"),
    (lambda r: r.update(reference_output_max_abs=1e-7), "reference_output_max_abs"),
    (lambda r: r["strict_restore"].update(missing=1), "strict_restore.missing"),
    (lambda r: r["strict_restore"].update(leaves=60), "strict_restore.leaves"),
    (lambda r: r["online"].update(online_infer=0), "online_infer"),
    (lambda r: r.update(worker_encoder_references=["v1-store/external/motionjepa/x.pt"]), "worker_encoder_references"),
    (lambda r: r["online"]["sidecar_provenance"].update(encoder_ckpt="x"), "sidecar.encoder_field"),
])
def test_save_load_restore_report_negative(save_load_env, capsys, change, field):
    change(save_load_env.report)
    assert run_cli(capsys, save_load_env.argv)[0] == 1
    assert field in fields(save_load_env.out, "SAVE_LOAD")


def test_restore_real_path_refuses_wrong_gpu_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    args = type("A", (), dict(online_gpu=1))()
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES=0"):
        chk.restore_and_infer(args, tmp_path, None, None)


def test_worker_encoder_reference_scan_real_process(tmp_path):
    marker = tmp_path / "external" / "motionjepa"
    marker.mkdir(parents=True)
    file = marker / "x.pt"
    file.write_bytes(b"x")
    with file.open("rb"):
        assert any("external/motionjepa" in hit for hit in chk.worker_encoder_references(["python", "s.py"], "self"))
    assert chk.worker_encoder_references(["python", "stage2_sidecar.py"], 2 ** 30) == ["stage2_sidecar.py"]


# ── G9 ───────────────────────────────────────────────────────────────────────


def scalar_row(step):
    return dict(step=step, scalars={name: dict(dec=0.1, hex="0x1p-3", finite=True) for name in chk.TRAIN_SCALARS})


@pytest.fixture
def final_env(tmp_path, monkeypatch):
    t = type("T", (), {})()
    t.records = tmp_path / "final-record"
    t.records.mkdir()
    t.head = "a" * 40
    t.run = "orig100-ctx32x4x4-wan-timepool8x8-b128-60k"
    identity = dict(run_uuid="u", run_name=t.run, head=t.head, num_train_steps=60000, log_interval=100)
    dump(t.records / "start.json", identity)
    leaves = {f"leaf{i}": dict(finite=True) for i in range(59)}
    dump(t.records / "final.json", {**identity, "loop_step": 59999, "state_step": 60000, "checkpoint_step": 59999,
                                    "tail": [scalar_row(s) for s in range(59901, 60000)], "tail_means": dict(loss=0.1),
                                    "ema_leaves": leaves})
    dump(t.records / "checkpoint_wait_done.json", dict(run_uuid="u", head=t.head, wait_until_finished=True))
    t.run_root = tmp_path / "run"
    for step in chk.EXPECT["checkpoint_steps"]:
        (t.run_root / str(step) / "params").mkdir(parents=True)
        (t.run_root / str(step) / "train_state").mkdir()
        (t.run_root / str(step) / "_CHECKPOINT_METADATA").write_text("{}")
        (t.run_root / str(step) / "params/_METADATA").write_text("{}")
        (t.run_root / str(step) / "params/manifest.ocdbt").write_text("x")
    t.log = tmp_path / "train.log"
    t.log.write_text("Step 100: loss=0.1000, grad_norm=1.0\nTRAIN_FINAL_RECORD=PASS run=x\nEXIT_CODE=0\n")
    t.out = tmp_path / "gate"
    t.argv = ["final", "--out-dir", str(t.out), "--final-record-dir", str(t.records), "--run-root", str(t.run_root), "--log", str(t.log),
              "--run", t.run, "--train-head", t.head, "--online-gpu", "1"]
    t.report = good_report(with_reference=False)
    monkeypatch.setitem(chk.EXPECT, "leaves", 59)
    monkeypatch.setattr(chk, "check_snapshot_identity", lambda findings, root: {})
    monkeypatch.setattr(chk, "restore_and_infer", lambda args, ckpt, ref, path: t.report)
    return t


def test_final_pass_exact_lines(final_env, capsys):
    code, lines = run_cli(capsys, final_env.argv)
    assert code == 0 and lines == ["TRAIN_FINAL_RECORD=PASS state_step=60000 checkpoint_step=59999",
                                   "CKPT_INVENTORY=PASS steps=12", "RESTORE_POOL=PASS"]
    assert record(final_env.out, "TRAIN_FINAL_RECORD")["metrics"]["log_exit_code"] == 0


@pytest.mark.parametrize("step", [55000, 59999])
def test_final_missing_checkpoint(final_env, capsys, step):
    import shutil
    shutil.rmtree(final_env.run_root / str(step))
    code, lines = run_cli(capsys, final_env.argv)
    assert code == 1 and lines[1].startswith("CKPT_INVENTORY=FAIL field=checkpoint_steps")
    assert lines[0].startswith("TRAIN_FINAL_RECORD=PASS")


def test_final_extra_and_uncommitted_checkpoint(final_env, capsys):
    (final_env.run_root / "60000").mkdir()
    (final_env.run_root / "59999.orbax-checkpoint-tmp-1").mkdir()
    assert run_cli(capsys, final_env.argv)[0] == 1
    assert {"checkpoint_steps", "uncommitted_checkpoint"} <= fields(final_env.out, "CKPT_INVENTORY")


def test_final_partial_checkpoint_entry(final_env, capsys):
    (final_env.run_root / "5000/params/_METADATA").unlink()
    assert run_cli(capsys, final_env.argv)[0] == 1
    assert "checkpoint_entry_missing" in fields(final_env.out, "CKPT_INVENTORY")


@pytest.mark.parametrize("text,field", [
    ("Step 1: loss=0.1\nEXIT_CODE=1\n", "log.last_line"),
    ("Step 1: loss=0.1\n", "log.last_line"),
    ("EXIT_CODE=0\nmore\n", "log.last_line"),
    ("Step 200: loss=nan, grad_norm=1.0\nEXIT_CODE=0\n", "log.nonfinite_or_oom"),
    ("jax RESOURCE_EXHAUSTED: oom\nEXIT_CODE=0\n", "log.nonfinite_or_oom"),
])
def test_final_log_negative(final_env, capsys, text, field):
    final_env.log.write_text(text)
    assert run_cli(capsys, final_env.argv)[0] == 1
    assert field in fields(final_env.out, "TRAIN_FINAL_RECORD")


@pytest.mark.parametrize("change,field", [
    (lambda f: f.update(state_step=59999), "state_step"),
    (lambda f: f.update(checkpoint_step=60000), "checkpoint_step"),
    (lambda f: f.update(head="b" * 40), "final.head"),
    (lambda f: f.update(run_name="other"), "final.run_name"),
    (lambda f: f.update(run_uuid="other"), "final.run_uuid"),
    (lambda f: f["tail"].pop(5), "tail.steps"),
    (lambda f: f["tail"][0]["scalars"]["loss"].update(finite=False), "tail.nonfinite"),
    (lambda f: f["tail"][0]["scalars"].pop("loss"), "tail.scalars"),
    (lambda f: f["ema_leaves"]["leaf1"].update(finite=False), "ema.nonfinite"),
    (lambda f: f["ema_leaves"].pop("leaf1"), "ema_leaves"),
    (lambda f: f["tail_means"].update(loss=None), "tail_means.nonfinite"),
])
def test_final_record_negative(final_env, capsys, change, field):
    edit_json(final_env.records / "final.json", change)
    assert run_cli(capsys, final_env.argv)[0] == 1
    assert field in fields(final_env.out, "TRAIN_FINAL_RECORD")


def test_final_skip_restore_is_not_a_pass(final_env, capsys):
    code, lines = run_cli(capsys, [*final_env.argv, "--skip-restore"])
    assert code == 1 and lines[2].startswith("RESTORE_POOL=FAIL field=restore_skipped")


def test_final_restore_failure(final_env, capsys):
    final_env.report["online"]["online_infer"] = 0
    assert run_cli(capsys, final_env.argv)[0] == 1
    assert "online_infer" in fields(final_env.out, "RESTORE_POOL")


OLD_FINAL = MAIN / "validation/orig16-concat-60k/orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k/final-record"


@pytest.mark.skipif(not OLD_FINAL.is_dir(), reason="需要旧正式 run 的最终记录")
def test_final_record_check_accepts_real_old_run():
    """同格式真实旧正式记录作校准：check_final_record 与 check_log_exit 在真实文件上能通过。"""
    findings = chk.Findings()
    chk.check_final_record(findings, load(OLD_FINAL / "start.json"), load(OLD_FINAL / "final.json"),
                           load(OLD_FINAL / "checkpoint_wait_done.json"),
                           run="orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k",
                           train_head="6424d7f39569739eae1ca6cdbb50c7f26897e738")
    assert findings.total == 0, findings.items[:3]
    findings = chk.Findings()
    chk.check_log_exit(findings, MAIN / "logs/orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k.log")
    assert findings.total == 0, findings.items[:3]


# ── 差分与通用件 ──────────────────────────────────────────────────────────────


def test_diff_records_detects_add_remove_type():
    assert chk.diff_records(dict(a=1), dict(a=1)) == []
    assert chk.diff_records(dict(a=1), dict(b=1)) == [("a", 1, chk._MISSING), ("b", chk._MISSING, 1)]
    assert chk.diff_records(dict(a=1), dict(a=1.0))[0][0] == "a"
    assert chk.diff_records(dict(a=[1, 2]), dict(a=[1, 3]))[0][0] == "a.1"


def test_exception_in_gate_writes_fail_record(tmp_path, capsys):
    out = tmp_path / "r"
    code, lines = run_cli(capsys, ["window", "--out-dir", str(out), "--plan", str(tmp_path / "missing"), "--receipt", "x",
                                   "--episode-manifest", "y", "--window-map", "z"])
    assert code == 1 and lines[0].startswith("WINDOW_EQ=FAIL field=exception")
    assert record(out, "WINDOW_EQ")["status"] == "FAIL"
