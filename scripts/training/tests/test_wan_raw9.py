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


# ══ 1117 计划 S2：步长 16 bf16 raw 表（raw_bf16、新布局、导出、读表契约、provenance、校验、K_DIST）══════════════

from mme_vla_suite.shared import raw_bf16  # noqa: E402

KDIST = ROOT / "scripts/dataset/e2e_k_dist.py"
S2_BASE_SHA = "dd39f3d"                      # 本分支基线（S2 改动之前），s32 路径字节不变的对照物
REAL_LIB = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-v2-1600ep-604f16da")
BF16 = raw_bf16.BF16


def _rne_ref_bits(x: np.ndarray) -> np.ndarray:
    """逐位手算 RNE：bf16 位 = (f32 位 + 0x7FFF + 第 16 位) >> 16（有限、不溢出的输入）。"""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def test_raw9_f32_to_bf16_rne_hand_cases():
    bits = lambda v: np.uint32(v).view(np.float32)          # noqa: E731
    cases = [  # (f32 位, 期望 bf16 位, 说明)
        (0x3F800000, 0x3F80, "1.0 精确"),
        (0x3F808000, 0x3F80, "1+2^-8：恰半 ulp、尾数偶 → 舍向偶（向下）"),
        (0x3F818000, 0x3F82, "1+2^-7+2^-8：恰半 ulp、尾数奇 → 舍向偶（向上）"),
        (0x3F808001, 0x3F81, "略超半 ulp → 向上"),
        (0x3F807FFF, 0x3F80, "略低半 ulp → 向下"),
        (0xBF818000, 0xBF82, "负数平局同样舍向偶"),
        (0x00000000, 0x0000, "+0"),
        (0x80000000, 0x8000, "-0 保留符号"),
        (0x00008000, 0x0000, "次正规平局、尾数偶 → 0"),
        (0x00018000, 0x0002, "次正规平局、尾数奇 → 向上"),
        (0x00008001, 0x0001, "次正规略超半 ulp → 最小次正规"),
        (0x00010000, 0x0001, "最小 bf16 次正规精确"),
        (0x007FFFFF, 0x0080, "最大 f32 次正规 → 进位成最小正规"),
        (0x7F7F7FFF, 0x7F7F, "大值略低半 ulp → 最大有限 bf16"),
    ]
    x = np.array([bits(c[0]) for c in cases], np.float32)
    got = raw_bf16.raw9_f32_to_bf16(x)
    assert got.dtype == BF16 and got.shape == x.shape
    for (f, want, why), g in zip(cases, got.view(np.uint16).tolist()):
        assert g == want, (hex(f), hex(g), hex(want), why)
    assert np.array_equal(got.view(np.uint16), _rne_ref_bits(x))
    # 随机全位型（含次正规、±0）对拍逐位手算；形状 (9,16,32,32) 不变；接受 python 列表与 f64 输入（先转 f32）
    rng = np.random.default_rng(3)
    r = rng.integers(0, 2**32, size=200_000, dtype=np.uint64).astype(np.uint32)
    r[:5000] &= 0x807FFFFF                                    # 强制一批次正规数
    r = r[((r >> 23) & 0xFF) < 0xFE]                          # 去掉 Inf/NaN 与会溢出的最高指数段
    xs = r.view(np.float32)
    assert np.array_equal(raw_bf16.raw9_f32_to_bf16(xs).view(np.uint16), _rne_ref_bits(xs))
    z = rng.standard_normal((9, 16, 32, 32)).astype(np.float32)
    zb = raw_bf16.raw9_f32_to_bf16(z)
    assert zb.shape == (9, 16, 32, 32) and zb.tobytes() == z.astype(BF16).tobytes()
    assert raw_bf16.raw9_f32_to_bf16([1.0, 2.0]).view(np.uint16).tolist() == [0x3F80, 0x4000]
    for bad in (np.float32(np.nan), np.float32(np.inf), np.float32(-np.inf)):
        arr = z.copy()
        arr[3, 2, 1, 0] = bad
        with pytest.raises(ValueError):
            raw_bf16.raw9_f32_to_bf16(arr)
    _PASSED.add("bf16_rne")


def test_layout_spec_bf16_and_raw_set():
    s = ms.LAYOUT_SPECS[ms.LAYOUT_RAW9BF16]
    assert ms.LAYOUT_RAW9BF16 == "motion-raw9bf16-grid16-demopad17-v1"
    assert (s.grid_stride, s.row_shape, s.dtype, s.row_bytes) == (16, (9, 16, 32, 32), "bfloat16", 294_912)
    assert s.np_dtype == BF16 and s.fields() == ms.LAYOUT_SPECS[ms.LAYOUT].fields()
    for name in ("motion-768-grid16-v1", ms.LAYOUT, ms.LAYOUT_RAW9):
        assert ms.LAYOUT_SPECS[name].dtype == "float32" and ms.LAYOUT_SPECS[name].np_dtype == np.float32
    assert ms.RAW_LAYOUTS == {ms.LAYOUT_RAW9, ms.LAYOUT_RAW9BF16}
    assert 71316 * s.row_bytes == 21_031_944_192
    with pytest.raises(ValueError):
        ms.layout_np_dtype("float16")
    _PASSED.add("layout_bf16")


def run_export16(L, out: pathlib.Path, workers: int, *extra):
    cmd = [sys.executable, str(EXPORT), "--latents", str(L.latents), "--manifest", str(L.manifest),
           "--checksums", str(L.checksums / "sha256-source-files.txt"), "--motion-ref", str(L.pool),
           "--workers", str(workers), "--out", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def raw16_out(lib, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("raw16out")
    outs = {}
    for w in (1, 3):
        out = tmp / f"out-w{w}"
        r = run_export16(lib, out, w, "--stride", "16", "--out-dtype", "bfloat16")
        assert r.returncode == 0, r.stdout + r.stderr
        outs[w] = (out, r.stdout)
    return outs


def test_export16_bf16_rows_index_provenance(lib, raw16_out, capsys):
    (a, out_a), (b, _) = raw16_out[1], raw16_out[3]
    for rel in (ms.MOTION_TABLE_RELPATH, ms.INDEX_RELPATH, ms.ROW_DIGESTS_RELPATH):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel                # --workers 1 与 3 字节相同
    assert _meta_wo_exporter(a) == _meta_wo_exporter(b)
    meta, ref = ms.MotionMeta.load(a), ms.MotionMeta.load(lib.pool)
    assert meta.status == "verified" and meta.raw["layout"] == ms.LAYOUT_RAW9BF16 and meta.grid_stride == 16
    t = meta.raw["tables"][ms.MOTION_KEY]
    assert (t["dtype"], t["row_bytes"], t["row_shape"]) == ("bfloat16", 294_912, [9, 16, 32, 32])
    assert meta.num_rows == ref.num_rows and (a / ms.MOTION_TABLE_RELPATH).stat().st_size == meta.num_rows * 294_912
    # index 与 768 维表逐行相等（含 spec 三键与步长）
    assert meta.spec.fields() == ref.spec.fields() and meta.grid_stride == ref.grid_stride
    assert [_entry_key(e) for e in meta.entries] == [_entry_key(e) for e in ref.entries]
    # 每行 = astype(bf16)(源第 m 个 chunk)（独立参照，不经 raw_bf16）
    table = np.fromfile(a / ms.MOTION_TABLE_RELPATH, dtype=BF16).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
    n = 0
    for e in meta.entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            src = np.fromfile(lib.latents / f"{ms.segment_key(e.h5_file, e.raw_ep_idx, seg)}.bin",
                              dtype=np.float32).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
            assert src.shape[0] == s.num_grid
            for m in range(s.num_grid):
                assert table[s.row_base + m].tobytes() == src[m].astype(BF16).tobytes()
                n += 1
    assert n == meta.num_rows
    p = meta.provenance
    assert (p["table_kind"], p["table_op"], p["parameter_count"]) == ("wan_raw_bf16_s16_v1", "f32_to_bf16_reorder", 0)
    assert p["raw9_source_sha256"] == wr.source_sha256() and p["source_m_rule"] == "all" and p["dtype"] == "bfloat16"
    assert not {"encoder_kind", "readout", "encoder_ckpt"} & set(p)
    exp_budget = max(ms.max_visible_count(e) for e in ref.entries)
    assert p["budget_max"] == exp_budget
    last = out_a.strip().splitlines()[-1]
    assert last.startswith(f"EXPORT_RAW16=PASS rows={meta.num_rows} demo=") and " dtype=bf16 row_bytes=294912 " in last
    assert f" budget_max={exp_budget} " in last
    _PASSED.add("export16")
    _say(capsys, f"EXPORT_RAW16_UNIT=PASS workers=1,3 rows={n} budget_max={exp_budget}")


def test_export16_refuses_bad_modes_and_existing_out(lib, raw16_out, tmp_path):
    for extra in (("--stride", "16"), ("--out-dtype", "bfloat16"), ("--stride", "32", "--out-dtype", "bfloat16")):
        r = run_export16(lib, tmp_path / "x", 1, *extra)
        assert r.returncode != 0 and "不支持" in r.stderr and not (tmp_path / "x").exists(), extra
    r = run_export16(lib, raw16_out[1][0], 1, "--stride", "16", "--out-dtype", "bfloat16")
    assert r.returncode != 0 and "输出根已存在" in r.stderr


def _baseline_tree(tmp: pathlib.Path, sha: str) -> pathlib.Path:
    """git archive 取出基线 sha 的导出脚本与 src/mme_vla_suite（旧 motion_store），按其自身位置推仓库根运行。"""
    tmp.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(ROOT), "archive", sha, "pyproject.toml",
                          "scripts/dataset/export_wan_raw9_s32.py", "src/mme_vla_suite"],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
    return tmp


def test_export_s32_default_bytes_equal_baseline(lib, raw9_out, tmp_path, capsys):
    """默认 / 显式 ``--stride 32 --out-dtype float32`` 与基线 dd39f3d（导出脚本 + motion_store）输出逐字节相同。"""
    base = _baseline_tree(tmp_path / "base", S2_BASE_SHA)
    out_b = tmp_path / "out-base"
    cmd = [sys.executable, str(base / "scripts/dataset/export_wan_raw9_s32.py"), "--latents", str(lib.latents),
           "--manifest", str(lib.manifest), "--checksums", str(lib.checksums / "sha256-source-files.txt"),
           "--motion-ref", str(lib.pool), "--workers", "2", "--out", str(out_b)]
    rb = subprocess.run(cmd, capture_output=True, text=True)
    assert rb.returncode == 0, rb.stdout + rb.stderr
    out_e = tmp_path / "out-explicit"
    re_ = run_export16(lib, out_e, 2, "--stride", "32", "--out-dtype", "float32")
    assert re_.returncode == 0, re_.stdout + re_.stderr
    out_d = raw9_out[1][0]                                     # 新代码默认参数的产物
    for o in (out_d, out_e):
        for rel in (ms.MOTION_TABLE_RELPATH, ms.INDEX_RELPATH, ms.ROW_DIGESTS_RELPATH):
            assert (o / rel).read_bytes() == (out_b / rel).read_bytes(), (o, rel)
        mo, mb = _meta_wo_exporter(o), _meta_wo_exporter(out_b)
        for m in (mo, mb):                                     # 只有 provenance 里的源码摘要随 S2 改动之外的文件变，
            m["provenance"].pop("raw9_source_sha256")          # wan_raw9.py 未改，二者也应相同——单独断言
        assert mo == mb, o
    assert json.loads((out_d / ms.META_RELPATH).read_text())["provenance"]["raw9_source_sha256"] == \
        json.loads((out_b / ms.META_RELPATH).read_text())["provenance"]["raw9_source_sha256"]
    assert rb.stdout.strip().splitlines()[-1] == re_.stdout.strip().splitlines()[-1]   # EXPORT_RAW9 判定行逐字相同
    _PASSED.add("s32_regress")
    _say(capsys, f"S32_EXPORT_REGRESS=PASS baseline={S2_BASE_SHA} outputs=default,explicit")


@pytest.mark.skipif(not (REAL_LIB / "motion-wan-raw9-s32/meta/store_meta.json").is_file(), reason="本机无真实 1600ep 库")
def test_real_subset_s32_rows_equal_table_and_bf16_even(tmp_path, capsys):
    """真实数据子集：新代码 export_segment 在 s32/f32 模式写出的行 == 现有 s32 表对应行（table_sha256=0e6657af…）；
    s16/bf16 模式写出的偶数 m 行 == astype(bf16)(s32 表对应行)。只读 v1-store，输出写 tmp 稀疏文件。"""
    exp = _load_by_path("export_raw9_mod_for_s2", EXPORT)
    s32_meta = ms.MotionMeta.load(REAL_LIB / "motion-wan-raw9-s32")
    assert s32_meta.table_sha256 == "0e6657af118e65c762b86dfd86ce0794289108b3be37ad5017fe9c26d1442d19"
    ref = ms.MotionMeta.load(REAL_LIB / "motion")
    s16_entries = ms.build_index_entries(json.loads(pathlib.Path(ref.manifest_path).read_text()),
                                         ms.LAYOUT_SPECS[ms.LAYOUT_RAW9BF16])
    pick = [0, 1, 777, 1599]
    s32_table = np.memmap(REAL_LIB / "motion-wan-raw9-s32" / ms.MOTION_TABLE_RELPATH, dtype=np.float32, mode="r",
                          shape=(s32_meta.num_rows,) + wr.RAW9_INPUT_SHAPE)
    rows32 = rows16 = 0
    for stride, dtype, entries in ((32, "float32", list(s32_meta.entries)), (16, "bfloat16", s16_entries)):
        jobs = exp.segment_jobs([ref.entries[g] for g in pick], [entries[g] for g in pick], stride)
        spec = ms.LAYOUT_SPECS[exp.MODES[(stride, dtype)]]
        tmp = tmp_path / f"t{stride}.bin"
        with open(tmp, "wb") as f:                                # 稀疏文件，只有被写的行占盘
            f.truncate(ms.index_totals(entries)["rows"] * spec.row_bytes)
        for j in jobs:
            j.update(latents_root=str(REAL_LIB / "wan-latents"), table_tmp=str(tmp), out_dtype=dtype,
                     out_row_bytes=spec.row_bytes)
            res = exp.export_segment(j)
            out = np.memmap(tmp, dtype=spec.np_dtype, mode="r", shape=(ms.index_totals(entries)["rows"],) +
                            wr.RAW9_INPUT_SHAPE)
            g, seg = j["g"], j["segment"]
            s32s = getattr(s32_meta.entries[g], seg)
            if stride == 32:
                for i in range(res["num_grid"]):
                    assert out[j["row_base"] + i].tobytes() == s32_table[s32s.row_base + i].tobytes()
                    rows32 += 1
            else:
                for m in range(0, res["num_grid"], 2):
                    assert out[j["row_base"] + m].tobytes() == \
                        np.asarray(s32_table[s32s.row_base + m // 2]).astype(BF16).tobytes()
                    rows16 += 1
            del out
    del s32_table
    assert rows32 == rows16 > 0
    _PASSED.add("real_subset")
    _say(capsys, f"REAL_SUBSET_S32_AND_S16_EVEN=PASS episodes={pick} s32_rows={rows32} s16_even_rows={rows16}")


def test_bf16_store_memmap_rows_into_contract(raw16_out, tmp_path):
    a = raw16_out[1][0]
    st = ms.MotionStore(a, verify_level="full")
    assert isinstance(st.table, np.memmap) and st.table.dtype == BF16 and not st.table.flags["WRITEABLE"]
    ref = np.fromfile(a / ms.MOTION_TABLE_RELPATH, dtype=BF16).reshape((-1,) + wr.RAW9_INPUT_SHAPE)
    r = [st.num_rows - 1, 0, 2]
    got = st.rows(r)
    assert type(got) is np.ndarray and got.dtype == BF16 and got.tobytes() == ref[r].tobytes()
    buf = np.zeros((4 * 9, 16384), BF16)                                     # e2e 预分配：(k·9,16384) 的前 3·9 行
    out = buf[:27].reshape(3, *wr.RAW9_INPUT_SHAPE)
    assert st.rows_into(r, out) is out
    assert buf[:27].tobytes() == ref[r].tobytes() and not buf[27:].view(np.uint16).any()
    for bad in (np.zeros((3,) + wr.RAW9_INPUT_SHAPE, np.float32), np.zeros((3,) + wr.RAW9_INPUT_SHAPE, np.float16),
                np.zeros((2,) + wr.RAW9_INPUT_SHAPE, BF16)):
        with pytest.raises(ValueError):
            st.rows_into(r, bad)
    st.close()
    # store_meta 声明错误 dtype / 行字节即拒（布局 dtype 严格比对）
    for name, fn in {"dtype_f32": lambda m: m["tables"][ms.MOTION_KEY].update(dtype="float32"),
                     "dtype_bf16_name": lambda m: m["tables"][ms.MOTION_KEY].update(dtype="bf16"),
                     "rowbytes": lambda m: m["tables"][ms.MOTION_KEY].update(row_bytes=589_824)}.items():
        bad = _tamper_meta(a, tmp_path / name, fn)
        with pytest.raises(ValueError):
            ms.MotionMeta.load(bad)
    _PASSED.add("bf16_rows_into")


@pytest.mark.parametrize("name,fn", [
    ("add_encoder_kind", lambda p: p.update(encoder_kind="wan_raw9")),
    ("add_readout", lambda p: p.update(readout="wan_raw9_flat16384_s32_v1")),
    ("table_kind_wrong", lambda p: p.update(table_kind="wan_raw9")),
    ("table_kind_missing", lambda p: p.pop("table_kind")),
    ("table_op_wrong", lambda p: p.update(table_op="reorder")),
    ("param_1", lambda p: p.update(parameter_count=1)),
    ("param_false", lambda p: p.update(parameter_count=False)),
    ("raw9_sha", lambda p: p.update(raw9_source_sha256="0" * 64)),
])
def test_e2e_provenance_rejects_tamper(raw16_out, tmp_path, name, fn):
    bad = _tamper_meta(raw16_out[1][0], tmp_path / name, lambda m: fn(m["provenance"]))
    with pytest.raises(ValueError):
        ms.MotionMeta.load(bad)


def test_e2e_provenance_contract_and_swap(lib, raw16_out, raw9_out, tmp_path):
    e2e, raw9 = raw16_out[1][0], raw9_out[1][0]
    ms.MotionMeta.load(e2e)                                                   # 合法 e2e 表穿过 MotionMeta.load
    p16 = json.loads((e2e / ms.META_RELPATH).read_text())["provenance"]
    p32 = json.loads((raw9 / ms.META_RELPATH).read_text())["provenance"]
    ms.check_e2e_table_provenance(p16, layout=ms.LAYOUT_RAW9BF16)
    with pytest.raises(ValueError):
        ms.check_e2e_table_provenance(p16, layout=ms.LAYOUT_RAW9)
    # 新旧 provenance 整体互换：两边都失败
    with pytest.raises(ValueError):
        ms.MotionMeta.load(_tamper_meta(e2e, tmp_path / "e2e_with_raw9", lambda m: m.update(provenance=dict(p32))))
    with pytest.raises(ValueError):
        ms.MotionMeta.load(_tamper_meta(raw9, tmp_path / "raw9_with_e2e", lambda m: m.update(provenance=dict(p16))))
    # 旧 raw9 表把 encoder_kind/readout 换成 table_kind：失败
    def swap_kind(m):
        m["provenance"].pop("encoder_kind")
        m["provenance"].pop("readout")
        m["provenance"]["table_kind"] = ms.E2E_TABLE_KIND
    with pytest.raises(ValueError):
        ms.MotionMeta.load(_tamper_meta(raw9, tmp_path / "raw9_kind_swap", swap_kind))
    # 768 布局表带上 e2e provenance（table_kind）：布局与来源双向绑定，失败
    with pytest.raises(ValueError):
        ms.MotionMeta.load(_tamper_meta(lib.pool, tmp_path / "pool_e2e", lambda m: m.update(provenance=dict(p16))))
    # 旧 raw9 表继续通过原校验（check_raw9_provenance 不动）
    ms.MotionMeta.load(raw9)
    ms.check_raw9_provenance(p32, layout=ms.LAYOUT_RAW9)
    _PASSED.add("e2e_provenance")


def run_verify16(L, table: pathlib.Path, s32: pathlib.Path | None, s32_sha: str | None, *extra):
    cmd = [sys.executable, str(VERIFY), "--stride", "16", "--table", str(table), "--latents", str(L.latents),
           "--checksums", str(L.checksums / "sha256-source-files.txt"), "--workers", "3",
           "--motion-ref", str(L.pool), "--expect-rows", str(ms.MotionMeta.load(L.pool).num_rows), *extra]
    if s32 is not None:
        cmd += ["--s32-table", str(s32), "--expect-s32-table-sha256", s32_sha]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_verify16_pass_and_fail_modes(lib, raw16_out, raw9_out, tmp_path, capsys):
    good, s32 = raw16_out[1][0], raw9_out[1][0]
    s32_meta, ref = ms.MotionMeta.load(s32), ms.MotionMeta.load(lib.pool)
    sha32 = s32_meta.table_sha256
    r = run_verify16(lib, good, s32, sha32)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.strip().splitlines()
    bmax = max(ms.max_visible_count(e) for e in ref.entries)
    demo = sum(e.demo.num_grid for e in ref.entries)
    exe = sum(e.exec.num_grid for e in ref.entries)
    assert f"INDEX_VS_MOTION768=PASS rows={ref.num_rows} mismatches=0" in lines
    assert f"S16_EVEN_VS_S32=PASS rows={s32_meta.num_rows} mismatches=0" in lines
    assert lines[-1] == (f"EXPORT_RAW16=PASS rows={ref.num_rows} dtype=bf16 row_bytes=294912 demo={demo} exec={exe} "
                         f"budget_max={bmax} budget_limit=160 mismatch=0")
    fails = {}
    # (a) 新表篡改 1 字节（同步改表 sha 绕过 full 档）→ EXPORT_RAW16 FAIL
    bad = tmp_path / "byte"
    subprocess.run(["cp", "-r", str(good), str(bad)], check=True)
    raw = bytearray((bad / ms.MOTION_TABLE_RELPATH).read_bytes())
    raw[294_912 + 777] ^= 0x01
    _resha_table(bad, np.frombuffer(bytes(raw), np.uint8))
    fails["byte"] = (run_verify16(lib, bad, s32, sha32), "EXPORT_RAW16=FAIL")
    # (b) 用截断（非 RNE）转换重写新表 → 内容校验 FAIL
    bad = tmp_path / "trunc"
    subprocess.run(["cp", "-r", str(good), str(bad)], check=True)
    meta = ms.MotionMeta.load(good)
    arr = np.empty((meta.num_rows,) + wr.RAW9_INPUT_SHAPE, np.uint16)
    for e in meta.entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid:
                src = np.fromfile(lib.latents / f"{ms.segment_key(e.h5_file, e.raw_ep_idx, seg)}.bin", np.uint32)
                arr[s.row_base:s.row_base + s.num_grid] = (src >> 16).astype(np.uint16).reshape(
                    (-1,) + wr.RAW9_INPUT_SHAPE)
    _resha_table(bad, arr)
    fails["trunc"] = (run_verify16(lib, bad, s32, sha32), "EXPORT_RAW16=FAIL")
    # (c) s32 表偶数行被改（同步改其 sha 并作为期望传入）→ S16_EVEN_VS_S32 FAIL
    bad32 = tmp_path / "s32bad"
    subprocess.run(["cp", "-r", str(s32), str(bad32)], check=True)
    raw = bytearray((bad32 / ms.MOTION_TABLE_RELPATH).read_bytes())
    raw[wr.RAW9_ROW_BYTES + 4 * 100 + 3] ^= 0x01              # 第 1 行一个 f32 的高字节（指数位），bf16 后必然不同
    _resha_table(bad32, np.frombuffer(bytes(raw), np.uint8))
    sha_bad = ms.MotionMeta.load(bad32).table_sha256
    fails["s32_even"] = (run_verify16(lib, good, bad32, sha_bad), "S16_EVEN_VS_S32=FAIL")
    # (d) s32 表 sha 与期望不符 → 拒
    fails["s32_sha"] = (run_verify16(lib, good, s32, "0" * 64), "EXPORT_RAW16=FAIL reason=")
    # (e) budget 上限过小 → FAIL
    fails["budget"] = (run_verify16(lib, good, None, None, "--budget-limit", str(bmax - 1)), "EXPORT_RAW16=FAIL")
    for name, (res, needle) in fails.items():
        assert res.returncode == 1 and needle in res.stdout, (name, res.stdout, res.stderr)
    # 不给 --s32-table 时只出两行，仍 PASS
    r2 = run_verify16(lib, good, None, None)
    assert r2.returncode == 0 and "S16_EVEN_VS_S32" not in r2.stdout, r2.stdout + r2.stderr
    _PASSED.add("verify16")
    _say(capsys, f"VERIFY_RAW16_UNIT=PASS pass=1 fail_modes={','.join(fails)}")


def test_k_dist_script(lib, raw16_out, capsys):
    ref = ms.MotionMeta.load(lib.pool)
    k = []
    for e in ref.entries:                                        # 独立暴力：demo 整段已见 + exec 前视窗尾 ≤ t
        es = e.exec_start_idx
        nd = sum(1 for m in range(e.demo.num_grid) if 16 * m + 16 <= es - 1)
        for t in range(es, e.num_timesteps):
            k.append(nd + sum(1 for m in range(e.exec.num_grid) if 16 * m + 32 <= t - es))
    k = np.asarray(k)
    want = f"{k.mean():.6f},{np.percentile(k, 50):g},{np.percentile(k, 95):g},{k.max()}"
    p50, p95 = np.percentile(k, 50), np.percentile(k, 95)
    for root in (lib.pool, raw16_out[1][0]):                     # 768 表与 s16 表 index 逐行相同，结果一致
        args = [sys.executable, str(KDIST), "--motion", str(root), "--budget", "160",
                "--expect-mean", f"{k.mean():.6f}", "--expect-p50", repr(float(p50)), "--expect-p95",
                repr(float(p95)), "--expect-max", str(k.max())]
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert r.stdout.splitlines()[0].startswith(f"K_DIST={want} samples={k.size} ")
        assert r.stdout.splitlines()[1].startswith("K_DIST_VS_AUDIT=PASS")
    r = subprocess.run([sys.executable, str(KDIST), "--motion", str(lib.pool)], capture_output=True, text=True)
    assert r.returncode == 1 and "K_DIST_VS_AUDIT=FAIL" in r.stdout                 # 审计默认值对夹具必不符
    _PASSED.add("k_dist")
    _say(capsys, f"K_DIST_UNIT=PASS fixture={want} samples={k.size}")


# ══ 汇总判定行（必须是本文件最后一个用例）════════════════════════════════════════════


REQUIRED = {"flatten", "token_times_budget", "layout_spec", "visible_rows_s32", "old_stride16_equal", "export",
            "memmap_rows_into", "raw9_provenance", "verify", "motion_gates", "init_history_config", "train_diag",
            "rss", "bf16_rne", "layout_bf16", "export16", "s32_regress", "bf16_rows_into", "e2e_provenance",
            "verify16", "k_dist"}
if (REAL_LIB / "motion-wan-raw9-s32/meta/store_meta.json").is_file():
    REQUIRED.add("real_subset")


def test_zz_raw9_store_unit_summary(capsys):
    missing = REQUIRED - _PASSED
    assert not missing, f"以下具名项未通过: {sorted(missing)}"
    _say(capsys, f"RAW9_STORE_UNIT=PASS items={len(REQUIRED)}")
