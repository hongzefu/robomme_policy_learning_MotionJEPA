"""1112 计划闸 G2 与 S1 单测：wan_raw9 契约、motion_store 步长 / 行形状参数化与只读 memmap、raw9 导出与独立校验、
_motion_gates raw9 分支与 v11 回归、init_history_config 快照、train_step 诊断键。

全部 CPU、合成夹具；latent 夹具沿用 test_wan_pool768.py::build_lib（extract_wan.py 落盘格式：stride-16 网格
``<段>.bin`` + ``.bin.sha256`` + ``.metadata.json``，成员清单按 ``sha256-source-files.txt`` 口径）。
旧布局回归的对照物是 v11 代码锚点 c55a6bc 的 motion_store.py（``git show`` 现场取出、按文件加载）。
最后一个用例在全部具名项通过后打印 ``RAW9_STORE_UNIT=PASS``。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import manifest_sha256  # noqa: E402
from mme_vla_suite.shared import wan_pool768 as wp  # noqa: E402
from mme_vla_suite.shared import wan_raw9 as wr  # noqa: E402

EXPORT = ROOT / "scripts/dataset/export_wan_raw9_s32.py"
VERIFY = ROOT / "scripts/dataset/verify_wan_raw9_s32.py"
BASELINE_SHA = "c55a6bc"                     # v11 代码锚点（与本分支基线 motion_store / dataloader / train 逐字相同）
_PASSED: set[str] = set()


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclass 定义需要在 sys.modules 里找到本模块
    spec.loader.exec_module(mod)
    return mod


W768 = _load_by_path("w768_fixture_for_raw9", ROOT / "scripts/training/tests/test_wan_pool768.py")


def _say(capsys, line: str) -> None:
    with capsys.disabled():
        print(f"\n{line}")


# ══ wan_raw9 纯函数 ══════════════════════════════════════════════════════════════


def test_raw9_constants_and_flatten():
    assert (wr.RAW9_ENCODER_KIND, wr.RAW9_READOUT) == ("wan_raw9", "wan_raw9_flat16384_s32_v1")
    assert wr.RAW9_INPUT_SHAPE == (9, 16, 32, 32) and wr.RAW9_TOKEN_DIM == 16384
    assert (wr.RAW9_TOKENS_PER_WINDOW, wr.RAW9_STRIDE, wr.RAW9_ROW_BYTES, wr.RAW9_PARAMETER_COUNT) == (9, 32, 589824, 0)
    z = np.random.default_rng(0).standard_normal(wr.RAW9_INPUT_SHAPE).astype(np.float32)
    keep = z.tobytes()
    out = wr.flatten_raw9(z)
    assert out.shape == (9, 16384) and out.dtype == np.float32
    assert np.shares_memory(out, z) and out.tobytes() == keep           # 零拷贝、不改字节
    j, c, h, w = 4, 7, 13, 29
    assert out[j, 1024 * c + 32 * h + w] == z[j, c, h, w]               # C→H→W 展平
    _PASSED.add("flatten")


@pytest.mark.parametrize("bad,exc", [
    (lambda: np.zeros(wr.RAW9_INPUT_SHAPE, np.float64), TypeError),
    (lambda: np.zeros(wr.RAW9_INPUT_SHAPE, np.float32).tolist(), TypeError),
    (lambda: np.zeros(wr.RAW9_INPUT_SHAPE, ">f4"), TypeError),
    (lambda: np.zeros((8, 16, 32, 32), np.float32), ValueError),
    (lambda: np.zeros((9, 16, 32, 64), np.float32)[..., ::2], ValueError),       # 非 C 连续
    (lambda: np.full(wr.RAW9_INPUT_SHAPE, np.inf, np.float32), ValueError),
])
def test_flatten_rejects_bad_input(bad, exc):
    with pytest.raises(exc):
        wr.flatten_raw9(bad())


def test_token_times_and_budget():
    assert wr.token_times(100, 132).tolist() == [100 + 4 * j for j in range(9)]     # exec 窗：f+32，不截断
    assert wr.token_times(0, 20).tolist() == [0, 4, 8, 12, 16, 20, 20, 20, 20]       # demo 尾补窗截到 es−1
    assert wr.token_times(5, 5).tolist() == [5] * 9 and wr.token_times(5, 5).dtype == np.int64
    with pytest.raises(ValueError):
        wr.token_times(10, 9)
    from omegaconf import OmegaConf
    assert wr.motion_token_budget(OmegaConf.create({"budget": 80, "tokens_per_window": 9})) == 720
    assert wr.motion_token_budget(OmegaConf.create({"budget": 160})) == 160                # 旧 YAML 缺键按 1
    assert wr.motion_token_budget({"budget": 80, "tokens_per_window": 9}) == 720
    assert wr.motion_token_budget(SimpleNamespace(budget=160)) == 160
    assert wr.source_sha256() == hashlib.sha256((ROOT / "src/mme_vla_suite/shared/wan_raw9.py").read_bytes()).hexdigest()
    _PASSED.add("token_times_budget")


# ══ LayoutSpec 与网格公式 ════════════════════════════════════════════════════════════


def test_layout_spec_old_unchanged_and_raw9():
    a, b = ms.LAYOUT_SPECS["motion-768-grid16-v1"], ms.LAYOUT_SPECS["motion-768-grid16-demopad17-v1"]
    assert a == ms.LayoutSpec(33, 33, "none") and b == ms.LayoutSpec(17, 33, "repeat_last")
    assert (a.grid_stride, a.row_shape, a.row_bytes) == (16, (768,), 3072) == (b.grid_stride, b.row_shape, b.row_bytes)
    assert b.fields() == {"demo_min_real_frames": 17, "exec_min_real_frames": 33, "demo_tail_pad": "repeat_last"}
    r = ms.LAYOUT_SPECS[ms.LAYOUT_RAW9]
    assert ms.LAYOUT_RAW9 == "motion-raw9-grid32-demopad17-v1"
    assert (r.grid_stride, r.row_shape, r.row_bytes) == (32, (9, 16, 32, 32), wr.RAW9_ROW_BYTES)
    assert r.fields() == b.fields() and r != b
    assert (ms.LAYOUT, ms.GRID_STRIDE, ms.MOTION_ROW_SHAPE, ms.MOTION_ROW_BYTES, ms.LAYOUT_GRID_SUFFIX) == (
        "motion-768-grid16-demopad17-v1", 16, (768,), 3072, "grid16")
    for L in range(0, 300):
        for mr in (17, 33):
            assert ms.seg_num_grid(L, mr, 32) == len(range(0, ms.seg_num_chunks(L, mr), 32))
            assert ms.seg_num_grid(L, mr, 32) == len(range(0, ms.seg_num_grid(L, mr), 2))     # ⌈ng16/2⌉
            assert ms.segment_grid_starts(L, mr, 32) == ms.segment_grid_starts(L, mr)[::2]
    _PASSED.add("layout_spec")


def _manifest(eps_spec) -> dict:
    episodes = [{"global_episode_idx": g, "h5_file": h5, "raw_ep_idx": ep, "num_timesteps": nt, "exec_start_idx": es}
                for g, (h5, ep, es, nt) in enumerate(eps_spec)]
    m = {"version": 1, "raw_dir": "/fake/raw", "episodes": episodes}
    m["sha256"] = manifest_sha256(m)
    return m


def test_visible_motion_rows_stride32_bruteforce():
    man = _manifest([("record_dataset_TaskA.h5", 0, 120, 400), ("record_dataset_TaskA.h5", 1, 16, 90),
                     ("record_dataset_TaskB.h5", 2, 49, 49 + 33), ("record_dataset_TaskB.h5", 3, 250, 600)])
    entries = ms.build_index_entries(man, ms.LAYOUT_SPECS[ms.LAYOUT_RAW9])
    for e in entries:
        es = e.exec_start_idx
        for t in range(es, e.num_timesteps):
            want = []
            for m in range(e.demo.num_grid):
                if 32 * m + 16 <= es - 1:
                    want.append((e.demo.row_base + m, 32 * m))
            for m in range(e.exec.num_grid):
                if 32 * m + 32 <= t - es:
                    want.append((e.exec.row_base + m, es + 32 * m))
            r, f = ms.visible_motion_rows(e, t)
            assert list(zip(r.tolist(), f.tolist())) == want
    payload = ms.index_payload(man, entries, spec=ms.LAYOUT_SPECS[ms.LAYOUT_RAW9], layout=ms.LAYOUT_RAW9,
                               mj_repo_commit="x")
    assert payload["grid_stride"] == 32 and ms.parse_index(json.loads(json.dumps(payload))) == entries
    bad = dict(payload, grid_stride=16)
    with pytest.raises(ValueError):
        ms.parse_index(bad)
    _PASSED.add("visible_rows_s32")


# ══ 旧 stride16 表：与 c55a6bc 的 motion_store 逐字节相同（①）══════════════════════════════


@pytest.fixture(scope="module")
def base_ms(tmp_path_factory):
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE_SHA}:src/mme_vla_suite/datastore/motion_store.py"],
                         capture_output=True, check=True).stdout
    p = tmp_path_factory.mktemp("base") / "motion_store_c55a6bc.py"
    p.write_bytes(src)
    return _load_by_path("motion_store_c55a6bc", p)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """pool768 夹具（3 episode、含 demo 补帧窗）+ 其 v11 池化表（作 raw9 导出的 --motion-ref）。"""
    tmp = tmp_path_factory.mktemp("raw9lib")
    L = W768.build_lib(tmp)
    pool = tmp / "motion-wan-pool768"
    r = W768.run_export(L, pool, 2)
    assert r.returncode == 0, r.stdout + r.stderr
    L.pool = pool
    return L


def _entry_key(e):
    return (e.g, e.h5_file, e.raw_ep_idx, e.num_timesteps, e.exec_start_idx,
            dataclasses.astuple(e.demo), dataclasses.astuple(e.exec), e.spec.fields())


def _tamper_meta(src: pathlib.Path, dst: pathlib.Path, fn) -> pathlib.Path:
    subprocess.run(["cp", "-r", str(src), str(dst)], check=True)
    m = json.loads((dst / ms.META_RELPATH).read_text())
    fn(m)
    (dst / ms.META_RELPATH).write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return dst


def test_old_stride16_tables_bytewise_equal_baseline(lib, base_ms, tmp_path, capsys):
    man = json.loads(lib.manifest.read_text())
    for root in (lib.pool, lib.ref):                       # v11 池化表与旧 encoder 表
        a, b = ms.MotionMeta.load(root), base_ms.MotionMeta.load(root)
        for k in ("status", "num_rows", "manifest_sha256", "manifest_path", "motion_index_sha256", "table_sha256",
                  "provenance", "raw"):
            assert getattr(a, k) == getattr(b, k), k
        assert [_entry_key(e) for e in a.entries] == [_entry_key(e) for e in b.entries]
        assert a.spec.fields() == b.spec.fields() and (a.grid_stride, a.row_shape, a.row_bytes) == (16, (768,), 3072)
        sa, sb = ms.MotionStore(root), base_ms.MotionStore(root)
        assert type(sa.table) is np.ndarray and not isinstance(sa.table, np.memmap)      # 旧布局仍整表 fromfile
        assert sa.table.shape == sb.table.shape and sa.table.tobytes() == sb.table.tobytes()
        r = [0, 3, a.num_rows - 1]
        assert sa.rows(r).tobytes() == sb.rows(r).tobytes() and type(sa.rows(r)) is np.ndarray
        out = np.empty((3, 768), np.float32)
        assert sa.rows_into(r, out) is out and out.tobytes() == sb.rows(r).tobytes()
        for ea, eb in zip(a.entries, b.entries):
            for t in range(ea.exec_start_idx, ea.num_timesteps):
                ra, fa = ms.visible_motion_rows(ea, t)
                rb, fb = base_ms.visible_motion_rows(eb, t)
                assert ra.tobytes() == rb.tobytes() and fa.tobytes() == fb.tobytes()
            assert ms.max_visible_count(ea) == base_ms.max_visible_count(eb)
    for layout in ("motion-768-grid16-v1", "motion-768-grid16-demopad17-v1"):        # index 字节逐位相同
        pa = ms.index_payload(man, ms.build_index_entries(man, ms.LAYOUT_SPECS[layout]),
                              spec=ms.LAYOUT_SPECS[layout], layout=layout, mj_repo_commit="m")
        pb = base_ms.index_payload(man, base_ms.build_index_entries(man, base_ms.LAYOUT_SPECS[layout]),
                                   spec=base_ms.LAYOUT_SPECS[layout], layout=layout, mj_repo_commit="m")
        assert json.dumps(pa, ensure_ascii=False, indent=1) == json.dumps(pb, ensure_ascii=False, indent=1)
    for L_ in range(0, 200):
        for mr in (17, 33):
            assert ms.seg_num_grid(L_, mr) == base_ms.seg_num_grid(L_, mr)
            assert ms.segment_grid_starts(L_, mr) == base_ms.segment_grid_starts(L_, mr)
    # 旧布局表的错误类型与信息逐字不变
    tampers = {"stride32": lambda m: m.update(grid_stride=32),
               "rowshape": lambda m: m["tables"][ms.MOTION_KEY].update(row_shape=[9, 16, 32, 32]),
               "rowbytes": lambda m: m["tables"][ms.MOTION_KEY].update(row_bytes=589824),
               "kind": lambda m: m["provenance"].update(encoder_kind="wan_raw9")}
    for name, fn in tampers.items():
        bad = _tamper_meta(lib.pool, tmp_path / name, fn)
        with pytest.raises(Exception) as ea:
            ms.MotionMeta.load(bad)
        with pytest.raises(Exception) as eb:
            base_ms.MotionMeta.load(bad)
        assert type(ea.value) is type(eb.value) and str(ea.value) == str(eb.value), name
    _PASSED.add("old_stride16_equal")
    _say(capsys, f"OLD_STRIDE16_REGRESS=PASS baseline={BASELINE_SHA} tables=2 tampers={len(tampers)}")


# ══ raw9 导出（②）══════════════════════════════════════════════════════════════════


def run_export(L, out: pathlib.Path, workers: int, checksums: str = "sha256-source-files.txt"):
    cmd = [sys.executable, str(EXPORT), "--latents", str(L.latents), "--manifest", str(L.manifest),
           "--checksums", str(L.checksums / checksums), "--motion-ref", str(L.pool),
           "--workers", str(workers), "--out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _expect(L) -> dict:
    """独立于 ms 的期望：stride 32 行数、budget_max（最后一帧可见窗数的最大值）。"""
    rows = {"demo": 0, "exec": 0}
    bmax = 0
    for e in L.entries:
        es, nt = e.exec_start_idx, e.num_timesteps
        nd = len(range(0, max(0, es - 16), 32))
        ne = len(range(0, max(0, nt - es - 32), 32))
        rows["demo"] += nd
        rows["exec"] += ne
        vis = sum(1 for m in range(nd) if 32 * m + 16 <= es - 1) + \
            sum(1 for m in range(ne) if 32 * m + 32 <= nt - 1 - es)
        bmax = max(bmax, vis)
    return {"rows": rows["demo"] + rows["exec"], "demo": rows["demo"], "exec": rows["exec"], "budget_max": bmax}


@pytest.fixture(scope="module")
def raw9_out(lib, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("raw9out")
    outs = {}
    for w in (1, 4):
        out = tmp / f"out-w{w}"
        r = run_export(lib, out, w)
        assert r.returncode == 0, r.stdout + r.stderr
        outs[w] = (out, r.stdout)
    return outs


def _meta_wo_exporter(p: pathlib.Path) -> dict:
    m = json.loads((p / ms.META_RELPATH).read_text())
    m.pop("exporter")
    return m


def test_export_even_m_budget_and_determinism(lib, raw9_out, capsys):
    (a, out_a), (b, _) = raw9_out[1], raw9_out[4]
    for rel in (ms.MOTION_TABLE_RELPATH, ms.INDEX_RELPATH, ms.ROW_DIGESTS_RELPATH):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel                # --workers 1 与 4 字节相同
    assert _meta_wo_exporter(a) == _meta_wo_exporter(b)
    exp = _expect(lib)
    line = out_a.strip().splitlines()[-1]
    assert line.startswith(f"EXPORT_RAW9=PASS rows={exp['rows']} demo={exp['demo']} exec={exp['exec']} padded=")
    assert f" budget_max={exp['budget_max']} " in line and exp["budget_max"] >= 2
    meta = ms.MotionMeta.load(a)
    assert meta.status == "verified" and meta.raw["layout"] == ms.LAYOUT_RAW9 and meta.grid_stride == 32
    table = np.fromfile(a / ms.MOTION_TABLE_RELPATH, dtype=np.float32).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
    assert table.shape[0] == exp["rows"] == meta.num_rows
    n_checked = 0
    for e16, e32 in zip(lib.entries, meta.entries):
        for seg in ms.SEGMENTS:
            s16, s32 = getattr(e16, seg), getattr(e32, seg)
            assert s32.num_grid == len(range(0, s16.num_grid, 2))
            if s32.num_grid == 0:
                continue
            key = ms.segment_key(e16.h5_file, e16.raw_ep_idx, seg)
            src = np.fromfile(lib.latents / f"{key}.bin", dtype=np.float32).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
            for mp_ in range(s32.num_grid):
                assert table[s32.row_base + mp_].tobytes() == src[2 * mp_].tobytes()     # 第 m' 行 = 源第 2m' 窗
                n_checked += 1
    p = meta.provenance
    assert (p["encoder_kind"], p["readout"], p["encoder_ckpt"], p["parameter_count"]) == (
        wr.RAW9_ENCODER_KIND, wr.RAW9_READOUT, None, 0)
    assert p["raw9_source_sha256"] == wr.source_sha256() and p["vae"] == W768.VAE
    assert p["source_grid_stride"] == 16 and p["source_m_rule"] == "even" and p["budget_max"] == exp["budget_max"]
    assert p["input_manifest_sha256"] == "1" * 64 and p["row_shape"] == [9, 16, 32, 32]
    idx = json.loads((a / ms.INDEX_RELPATH).read_text())
    assert idx["layout"] == ms.LAYOUT_RAW9 and idx["grid_stride"] == 32
    _PASSED.add("export")
    _say(capsys, f"EXPORT_RAW9_UNIT=PASS workers=1,4 rows={exp['rows']} budget_max={exp['budget_max']} "
                 f"rows_checked={n_checked}")


def test_export_refuses_existing_out_and_wrong_checksums(lib, raw9_out, tmp_path):
    r = run_export(lib, raw9_out[1][0], 1)
    assert r.returncode != 0 and "输出根已存在" in r.stderr
    r = run_export(lib, tmp_path / "x", 1, checksums="sha256-shards.txt")
    assert r.returncode != 0 and "sha256-source-files.txt" in r.stderr


def test_raw9_store_memmap_rows_into(raw9_out):
    a = raw9_out[1][0]
    st = ms.MotionStore(a, verify_level="full")
    assert isinstance(st.table, np.memmap) and not st.table.flags["WRITEABLE"] and st.table.mode == "r"
    ref = np.fromfile(a / ms.MOTION_TABLE_RELPATH, dtype=np.float32).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
    r = [st.num_rows - 1, 0, 2]
    got = st.rows(r)
    assert type(got) is np.ndarray and got.shape == (3,) + wr.RAW9_INPUT_SHAPE and got.tobytes() == ref[r].tobytes()
    buf = np.zeros((5 * 9, 16384), np.float32)                              # S2 L0 用法：padded 缓冲的前 k·9 行
    st.rows_into(r, buf[:27].reshape(3, *wr.RAW9_INPUT_SHAPE))
    assert buf[:27].tobytes() == ref[r].tobytes() and not buf[27:].any()
    for bad in (np.zeros((3, 9, 16, 32, 32), np.float64), np.zeros((2, 9, 16, 32, 32), np.float32),
                np.zeros((3, 768), np.float32)):
        with pytest.raises(ValueError):
            st.rows_into(r, bad)
    ro = np.zeros((3,) + wr.RAW9_INPUT_SHAPE, np.float32)
    ro.flags["WRITEABLE"] = False
    with pytest.raises(ValueError):
        st.rows_into(r, ro)
    with pytest.raises(IndexError):
        st.rows([st.num_rows])
    with pytest.raises(TypeError):
        import pickle
        pickle.dumps(st)
    st.close()
    _PASSED.add("memmap_rows_into")


@pytest.mark.parametrize("kv", [{"readout": "x"}, {"raw9_source_sha256": "0" * 64}, {"encoder_ckpt": "ck.pt"},
                                {"parameter_count": 1}, {"parameter_count": False}, {"encoder_kind": "wan_fixedpool768"}])
def test_raw9_provenance_rejects_tamper(raw9_out, tmp_path, kv):
    bad = _tamper_meta(raw9_out[1][0], tmp_path / "t", lambda m: m["provenance"].update(kv))
    with pytest.raises(ValueError):
        ms.MotionMeta.load(bad)


def test_raw9_provenance_missing_encoder_ckpt_and_layout(raw9_out, tmp_path):
    bad = _tamper_meta(raw9_out[1][0], tmp_path / "t", lambda m: m["provenance"].pop("encoder_ckpt"))
    with pytest.raises(ValueError):
        ms.MotionMeta.load(bad)
    prov = json.loads((raw9_out[1][0] / ms.META_RELPATH).read_text())["provenance"]
    ms.check_raw9_provenance(prov, layout=ms.LAYOUT_RAW9)
    with pytest.raises(ValueError):
        ms.check_raw9_provenance(prov, layout=ms.LAYOUT)
    _PASSED.add("raw9_provenance")


# ══ 独立校验（③）══════════════════════════════════════════════════════════════════


def run_verify(L, table: pathlib.Path, **expect):
    exp = _expect(L)
    exp.update(expect)
    cmd = [sys.executable, str(VERIFY), "--table", str(table), "--latents", str(L.latents),
           "--checksums", str(L.checksums / "sha256-source-files.txt"), "--workers", "3",
           "--expect-rows", str(exp["rows"]), "--expect-demo", str(exp["demo"]), "--expect-exec", str(exp["exec"]),
           "--expect-budget-max", str(exp["budget_max"])]
    return subprocess.run(cmd, capture_output=True, text=True)


def _resha_table(root: pathlib.Path, arr: np.ndarray) -> None:
    """改表字节后同步改 meta 里的表 sha（绕过 full 档，逼独立逐字节比对去发现）。"""
    (root / ms.MOTION_TABLE_RELPATH).write_bytes(arr.tobytes())
    m = json.loads((root / ms.META_RELPATH).read_text())
    m["tables"][ms.MOTION_KEY]["sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    (root / ms.META_RELPATH).write_text(json.dumps(m, ensure_ascii=False, indent=1))


def _resha_index(root: pathlib.Path, fn) -> None:
    idx = json.loads((root / ms.INDEX_RELPATH).read_text())
    fn(idx)
    b = (json.dumps(idx, ensure_ascii=False, indent=1) + "\n").encode()
    (root / ms.INDEX_RELPATH).write_bytes(b)
    m = json.loads((root / ms.META_RELPATH).read_text())
    m["motion_index_sha256"] = hashlib.sha256(b).hexdigest()
    fn(m)
    (root / ms.META_RELPATH).write_text(json.dumps(m, ensure_ascii=False, indent=1))


def test_verify_pass_and_three_fail_modes(lib, raw9_out, tmp_path, capsys):
    good = raw9_out[1][0]
    exp = _expect(lib)
    r = run_verify(lib, good)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().splitlines()[-1] == (f"RAW9_TABLE=PASS rows={exp['rows']} demo={exp['demo']} "
                                                 f"exec={exp['exec']} budget_max={exp['budget_max']} mismatch=0")
    fails = {}
    # (a) 篡改 1 字节（同步改表 sha，绕过 full 档）
    bad = tmp_path / "byte"
    subprocess.run(["cp", "-r", str(good), str(bad)], check=True)
    raw = bytearray((bad / ms.MOTION_TABLE_RELPATH).read_bytes())
    raw[wr.RAW9_ROW_BYTES + 12345] ^= 0x01
    _resha_table(bad, np.frombuffer(bytes(raw), np.uint8))
    fails["byte"] = run_verify(lib, bad)
    # (b) 少 1 行
    bad = tmp_path / "short"
    subprocess.run(["cp", "-r", str(good), str(bad)], check=True)
    t = bad / ms.MOTION_TABLE_RELPATH
    t.write_bytes(t.read_bytes()[:-wr.RAW9_ROW_BYTES])
    fails["short"] = run_verify(lib, bad)
    # (c1) stride 写错：index / meta 的 grid_stride 写成 16（sha 同步改）
    bad = tmp_path / "stride_meta"
    subprocess.run(["cp", "-r", str(good), str(bad)], check=True)
    _resha_index(bad, lambda d: d.update(grid_stride=16))
    fails["stride_meta"] = run_verify(lib, bad)
    # (c2) stride 写错：表按 stride 16 连续取窗（第 m' 行放了源第 m' 窗而不是第 2m' 窗）
    bad = tmp_path / "stride_rows"
    subprocess.run(["cp", "-r", str(good), str(bad)], check=True)
    arr = np.fromfile(bad / ms.MOTION_TABLE_RELPATH, dtype=np.float32).reshape((-1,) + wr.RAW9_INPUT_SHAPE).copy()
    meta = ms.MotionMeta.load(good)
    for e in meta.entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            src = np.fromfile(lib.latents / f"{ms.segment_key(e.h5_file, e.raw_ep_idx, seg)}.bin",
                              dtype=np.float32).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
            for mp_ in range(s.num_grid):
                arr[s.row_base + mp_] = src[mp_]
    _resha_table(bad, arr)
    fails["stride_rows"] = run_verify(lib, bad)
    # (d) 期望值不符（默认 --expect-* 是 36484/18367/18117/71）
    fails["expect"] = run_verify(lib, good, budget_max=exp["budget_max"] + 1)
    for name, res in fails.items():
        last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
        assert res.returncode == 1 and last.startswith("RAW9_TABLE=FAIL"), (name, res.stdout, res.stderr)
    assert "mismatch=0" not in fails["byte"].stdout.splitlines()[-1]
    assert "mismatch=0" not in fails["stride_rows"].stdout.splitlines()[-1]
    _PASSED.add("verify")
    _say(capsys, f"VERIFY_RAW9_UNIT=PASS pass=1 fail_modes={','.join(fails)}")


# ══ _motion_gates / init_history_config / 诊断键 ═══════════════════════════════════════


def _gates():
    from omegaconf import OmegaConf
    from mme_vla_suite.training.dataloader import _motion_gates
    return OmegaConf, _motion_gates


RAW9_YAML = dict(encoder_kind=wr.RAW9_ENCODER_KIND, readout=wr.RAW9_READOUT, dim=16384, tokens_per_window=9,
                 stride=32, budget=80)


def _hcfg(OmegaConf, store, **motion):
    base = {"enabled": True, "store_path": str(store), "stride": 16}
    base.update(motion)
    return OmegaConf.create({"motion": base})


def test_motion_gates_raw9_and_v11_regress(lib, raw9_out, monkeypatch):
    OmegaConf, gates = _gates()
    monkeypatch.delenv("MMEVLA_MOTION_STORE", raising=False)
    monkeypatch.delenv("MMEVLA_FRAMESAMP_MANIFEST", raising=False)
    man = json.loads(lib.manifest.read_text())
    fm = SimpleNamespace(manifest_sha256=man["sha256"], manifest_path=str(lib.manifest))
    raw9 = raw9_out[1][0]
    assert gates(_hcfg(OmegaConf, raw9, **RAW9_YAML), fm) == str(raw9)
    for bad in (dict(RAW9_YAML, readout=wp.POOL768_READOUT), dict(RAW9_YAML, dim=768),
                dict(RAW9_YAML, tokens_per_window=1), {k: v for k, v in RAW9_YAML.items() if k != "tokens_per_window"},
                dict(RAW9_YAML, stride=16), dict(RAW9_YAML, source_run="enc-run/checkpoint_epoch_72.pt#encoder"),
                dict(encoder_kind=wp.POOL768_ENCODER_KIND, readout=wp.POOL768_READOUT, stride=32),   # v11 YAML 指 raw9 表
                dict(stride=32, source_run="enc-run/checkpoint_epoch_72.pt#encoder")):               # encoder YAML 指 raw9 表
        with pytest.raises(ValueError):
            gates(_hcfg(OmegaConf, raw9, **bad), fm)
    with pytest.raises(ValueError):                                       # raw9 YAML 指 v11 池化表
        gates(_hcfg(OmegaConf, lib.pool, **dict(RAW9_YAML, stride=16)), fm)
    # v11 回归：池化表与旧 encoder 表照旧通过
    pool = dict(encoder_kind=wp.POOL768_ENCODER_KIND, readout=wp.POOL768_READOUT)
    assert gates(_hcfg(OmegaConf, lib.pool, **pool), fm) == str(lib.pool)
    assert gates(_hcfg(OmegaConf, lib.ref, source_run="enc-run/checkpoint_epoch_72.pt#encoder"), fm) == str(lib.ref)
    with pytest.raises(ValueError):
        gates(_hcfg(OmegaConf, lib.pool, **dict(pool, encoder_kind="x")), fm)
    _PASSED.add("motion_gates")


@pytest.fixture(scope="module")
def train_mod():
    return _load_by_path("raw9_train", ROOT / "scripts/training/train.py")


def test_init_history_config_raw9_snapshot(lib, raw9_out, tmp_path, monkeypatch, train_mod):
    from omegaconf import OmegaConf
    import mme_vla_suite.datastore as dsmod
    man = json.loads(lib.manifest.read_text())
    frames = tmp_path / "framesamp"
    (frames / "meta").mkdir(parents=True)
    (frames / "meta/store_meta.json").write_text("{}")
    monkeypatch.setattr(dsmod, "StoreMeta", SimpleNamespace(
        load=lambda root: SimpleNamespace(manifest_sha256=man["sha256"], manifest_path=str(lib.manifest))))
    monkeypatch.delenv("MMEVLA_MOTION_STORE", raising=False)
    monkeypatch.delenv("MMEVLA_FRAMESAMP_MANIFEST", raising=False)

    def snap(name, hcfg):
        d = tmp_path / name
        d.mkdir()
        cfg = SimpleNamespace(model=SimpleNamespace(history_config="h.yaml"), checkpoint_dir=d)
        train_mod.init_history_config(cfg, hcfg, framesamp_root=frames)
        return json.loads((d / "motion_provenance.json").read_text())

    r9 = snap("raw9", _hcfg(OmegaConf, raw9_out[1][0], **RAW9_YAML))
    assert list(r9) == W768.ENCODER_PROV_KEYS + ["encoder_kind", "readout", "raw9_source_sha256"]
    assert r9["encoder"] is None and r9["vae"] == W768.VAE and "pool_source_sha256" not in r9
    assert (r9["encoder_kind"], r9["readout"], r9["raw9_source_sha256"]) == (
        wr.RAW9_ENCODER_KIND, wr.RAW9_READOUT, wr.source_sha256())
    pool = snap("pool", _hcfg(OmegaConf, lib.pool, encoder_kind=wp.POOL768_ENCODER_KIND, readout=wp.POOL768_READOUT))
    assert list(pool) == W768.ENCODER_PROV_KEYS + ["encoder_kind", "readout", "pool_source_sha256"]
    enc = snap("enc", _hcfg(OmegaConf, lib.ref, source_run="enc-run/checkpoint_epoch_72.pt#encoder"))
    assert list(enc) == W768.ENCODER_PROV_KEYS
    _PASSED.add("init_history_config")


def test_train_step_diag_keys(train_mod):
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from omegaconf import OmegaConf

    def cfg(motion):
        hc = OmegaConf.create({"motion": motion}) if motion is not None else None
        return SimpleNamespace(model=SimpleNamespace(use_history=True, history_config=hc))

    assert train_mod._raw9_diag_enabled(cfg({"enabled": True, "encoder_kind": "wan_raw9"}))
    assert not train_mod._raw9_diag_enabled(cfg({"enabled": True, "encoder_kind": "wan_fixedpool768"}))
    assert not train_mod._raw9_diag_enabled(cfg({"enabled": True}))
    assert not train_mod._raw9_diag_enabled(cfg({"enabled": False, "encoder_kind": "wan_raw9"}))
    assert not train_mod._raw9_diag_enabled(cfg(None))
    assert not train_mod._raw9_diag_enabled(SimpleNamespace(model=SimpleNamespace(use_history=True,
                                                                                    history_config="x.yaml")))

    class Mem(nnx.Module):
        def __init__(self, rngs):
            self.motion_enabled = True
            self.feature_encoder = nnx.Linear(4, 3, rngs=rngs)
            self.motion_encoder_static = nnx.Linear(5, 3, rngs=rngs)

    class Model(nnx.Module):
        def __init__(self, rngs):
            self.mem_encoder = Mem(rngs)

    model = Model(nnx.Rngs(0))

    def loss(m):
        a = m.mem_encoder.feature_encoder(jnp.ones((2, 4)))
        b = m.mem_encoder.motion_encoder_static(jnp.ones((2, 5)) * 2)
        return jnp.sum(a ** 2) + jnp.sum(b ** 3)

    grads = nnx.grad(loss)(model)
    info = train_mod._motion_diag_info(model, grads)
    assert set(info) == {"motion_enc_kernel_norm", "motion_enc_grad_norm", "frame_enc_grad_norm"}
    k = model.mem_encoder.motion_encoder_static.kernel.value
    np.testing.assert_allclose(info["motion_enc_kernel_norm"], jnp.linalg.norm(k), rtol=1e-6)
    np.testing.assert_allclose(info["motion_enc_grad_norm"],
                               jnp.linalg.norm(grads.mem_encoder.motion_encoder_static.kernel.value), rtol=1e-6)
    fe = grads.mem_encoder.feature_encoder
    np.testing.assert_allclose(info["frame_enc_grad_norm"],
                               optax.global_norm([fe.kernel.value, fe.bias.value]), rtol=1e-6)
    model.mem_encoder.motion_enabled = False
    assert train_mod._motion_diag_info(model, grads) == {}
    _PASSED.add("train_diag")


# ══ memmap RSS（④）══════════════════════════════════════════════════════════════════


def _make_raw9_store(root: pathlib.Path, n_eps: int) -> int:
    """直接按导出产物的契约写一张 raw9 表（随机行、不经导出脚本），返回行数。"""
    man = _manifest([(f"record_dataset_Task{g % 4}.h5", g, 400, 1000) for g in range(n_eps)])
    man_path = root / "episode_manifest.json"
    root.mkdir(parents=True)
    man_path.write_text(json.dumps(man))
    spec = ms.LAYOUT_SPECS[ms.LAYOUT_RAW9]
    entries = ms.build_index_entries(man, spec)
    totals = ms.index_totals(entries)
    index = ms.index_payload(man, entries, spec=spec, layout=ms.LAYOUT_RAW9, mj_repo_commit="m")
    ib = (json.dumps(index, ensure_ascii=False, indent=1) + "\n").encode()
    (root / "meta").mkdir()
    (root / ms.INDEX_RELPATH).write_bytes(ib)
    rng = np.random.default_rng(7)
    h = hashlib.sha256()
    with open(root / ms.MOTION_TABLE_RELPATH, "wb") as f:
        for _ in range(totals["rows"]):
            b = rng.standard_normal(wr.RAW9_INPUT_SHAPE, dtype=np.float32).tobytes()
            f.write(b)
            h.update(b)
    meta = {
        "schema": ms.META_SCHEMA, "layout": ms.LAYOUT_RAW9, "status": "verified", **spec.fields(),
        "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER, "grid_stride": 32,
        "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN, "window_direction": ms.WINDOW_DIRECTION,
        "truncation_policy": ms.TRUNCATION_POLICY, "frame_size": ms.FRAME_SIZE,
        "tables": {ms.MOTION_KEY: {"row_shape": [9, 16, 32, 32], "dtype": "float32", "row_bytes": spec.row_bytes,
                                   "num_rows": totals["rows"], "relpath": ms.MOTION_TABLE_RELPATH,
                                   "byte_count": totals["rows"] * spec.row_bytes, "sha256": h.hexdigest()}},
        "num_rows": totals["rows"], "totals": totals, "manifest_sha256": man["sha256"], "manifest_path": str(man_path),
        "motion_index_relpath": ms.INDEX_RELPATH, "motion_index_sha256": hashlib.sha256(ib).hexdigest(),
        "provenance": {"encoder_kind": wr.RAW9_ENCODER_KIND, "readout": wr.RAW9_READOUT, "encoder_ckpt": None,
                       "parameter_count": 0, "raw9_source_sha256": wr.source_sha256(), "vae": {"x": 1}},
    }
    (root / ms.META_RELPATH).write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return totals["rows"]


def test_memmap_rss_does_not_grow_with_table(tmp_path, capsys):
    root = tmp_path / "big"
    rows = _make_raw9_store(root, 12)
    size_mb = rows * wr.RAW9_ROW_BYTES / 2**20
    assert size_mb >= 200
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT / 'src')!r})
        import numpy as np
        from mme_vla_suite.datastore import motion_store as ms
        def rss():
            for ln in open('/proc/self/status'):
                if ln.startswith('VmRSS:'):
                    return int(ln.split()[1]) / 1024
        meta = ms.MotionMeta.load({str(root)!r})
        before = rss()
        st = ms.MotionStore({str(root)!r}, meta=meta)
        assert isinstance(st.table, np.memmap) and not st.table.flags['WRITEABLE']
        x = st.rows([0, {rows // 2}, {rows - 1}])
        buf = np.zeros((3, 9, 16, 32, 32), np.float32)
        st.rows_into([1, 2, 3], buf)
        after = rss()
        print(f'RSS_DELTA_MB={{after - before:.2f}}')
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    delta = float(r.stdout.strip().split("=")[1])
    assert delta < 20, delta
    _PASSED.add("rss")
    _say(capsys, f"MEMMAP_RSS=PASS table_mb={size_mb:.0f} rss_delta_mb={delta:.2f}")


# ══ 汇总判定行（必须是本文件最后一个用例）════════════════════════════════════════════


REQUIRED = {"flatten", "token_times_budget", "layout_spec", "visible_rows_s32", "old_stride16_equal", "export",
            "memmap_rows_into", "raw9_provenance", "verify", "motion_gates", "init_history_config", "train_diag",
            "rss"}


def test_zz_raw9_store_unit_summary(capsys):
    missing = REQUIRED - _PASSED
    assert not missing, f"以下具名项未通过: {sorted(missing)}"
    _say(capsys, f"RAW9_STORE_UNIT=PASS items={len(REQUIRED)}")
