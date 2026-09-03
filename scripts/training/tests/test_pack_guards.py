"""framesamp 守卫测试（v2 计划 C.5）——Store 组（S2）+ Dataset 组（S3）。

Store 组：G1/G4/G5/G7/G11/G12/G14；Dataset 组：G2/G3/G6a/G8/G9/G10/G13 +
backend 分派闸（G6b 属 S5，随第一块在全量库上跑）。刻意制造失败断言亮红灯；
JAX_PLATFORMS=cpu、pytest 从仓库根直跑（get_history_config 按相对路径找 yaml）：

    UV_LINK_MODE=copy uv run pytest scripts/training/tests/test_pack_guards.py -x -q

迷你库：ref-shard（v1-store/datasets/ref-shard）派生 global_episode_idx 连续前缀
[0..2]（ButtonUnmask 前 3 个 episode，episode 0 有 291 帧满足 ≥33 硬约束），
session 级 fixture 打包一次（含全量 verify）供各守卫复用。

⚠ 测试定义顺序有意为之：fork Pool 用例（fixture 构建、G11 verify）在前；
import jax 的用例（Dataset 组经 shared.data_utils 拉 flax、G7 懒 import jax）
一律排在其后——jax 不允许进 fork 前进程，故 Dataset 组的重型 import 全部
函数内懒加载。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import pathlib
import shutil
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import framesamp_store as fs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    # 打包器在数据集域 scripts/dataset/（v2-motionmem 起 pack/ 子目录上提平铺），按仓库根定位
    "pack_framesamp_store", _REPO_ROOT / "scripts" / "dataset" / "pack_framesamp_store.py")
packer = importlib.util.module_from_spec(_spec)
sys.modules["pack_framesamp_store"] = packer   # Pool 按限定名 pickle worker 函数，必须登记
_spec.loader.exec_module(packer)

REF_SHARD = _REPO_ROOT / "v1-store" / "datasets" / "ref-shard"
MANIFEST = _REPO_ROOT / "v1-store" / "episode_manifest.json"
PREFIX_K = 2          # 前缀 [0..2]
PROCS = 4


def _pack_args(out: pathlib.Path, **over) -> types.SimpleNamespace:
    d = dict(source=str(REF_SHARD), manifest=str(MANIFEST), out=str(out),
             reader="decode", procs=PROCS, subset_prefix=PREFIX_K,
             resume=False, force_break_lock=False)
    d.update(over)
    return types.SimpleNamespace(**d)


def _verify_args(store: pathlib.Path, **over) -> types.SimpleNamespace:
    d = dict(store=str(store), source=str(REF_SHARD), manifest=str(MANIFEST),
             procs=PROCS, sample=None, seed=20260827,
             resume=True, force_break_lock=False)
    d.update(over)
    return types.SimpleNamespace(**d)


def _open_store(root: pathlib.Path, **kw) -> fs.FrameSampStore:
    kw.setdefault("manifest_path", str(MANIFEST))
    kw.setdefault("source_root", str(REF_SHARD))
    kw.setdefault("fast_sample", (0, 0))   # 守卫用例钉死抽样点，保证确定性
    return fs.FrameSampStore(root, **kw)


def _src_frame(g: int, t: int) -> tuple[bytes, bytes, bytes]:
    return packer.read_frame_decode(
        str(REF_SHARD / "features" / f"episode_{g}" / f"token_emb_{t}.npy"))


@pytest.fixture(scope="session")
def mini_store(tmp_path_factory) -> pathlib.Path:
    """打包 ref-shard 前缀 [0..2] 迷你库并全量 verify（G1 的全流程本体）。"""
    out = tmp_path_factory.mktemp("mini") / "store"
    packer.cmd_pack(_pack_args(out))
    packer.cmd_verify(_verify_args(out))
    return out


def _copy_store(mini_store: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    dst = tmp_path / "store-copy"
    shutil.copytree(mini_store, dst)
    return dst


def _patch_meta(root: pathlib.Path, fn) -> None:
    p = root / fs.META_RELPATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    fn(raw)
    p.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")


# ── G1：迷你库打包→读取逐位对拍（含全量 verify 已在 fixture 内跑过）─────────────


def test_g1_pack_read_bitwise(mini_store):
    meta = fs.StoreMeta.load(mini_store)
    assert meta.status == "verified"
    assert meta.manifest_scope == "subset"
    assert meta.subset_episodes == [0, 1, 2]
    assert "VERIFY_PACK=PASS" in meta.raw["verify"]["conclusion"]
    rd = mini_store / fs.ROW_DIGESTS_RELPATH
    assert rd.stat().st_size == meta.num_rows * fs.ROW_DIGEST_BYTES

    manifest = fs.load_manifest(MANIFEST)
    eps = manifest["episodes"][:3]
    with _open_store(mini_store) as store:
        for ep in eps:
            g, nt = ep["global_episode_idx"], ep["num_timesteps"]
            base = ep["total_sample_offset"]
            for t in {0, 1, 30, 31, nt - 1}:
                row = fs.row_of(base, t)
                src_img, src_pos, src_stt = _src_frame(g, t)
                assert store.read_image_rows([row])[0].tobytes() == src_img, (g, t)
                assert store.pos_rows([t])[0].tobytes() == src_pos, (g, t)
                assert store.state_rows([row])[0].tobytes() == src_stt, (g, t)
        # 游程合并路径：整段连续 32 行一次读回
        rows = np.arange(32, dtype=np.int64)
        got = store.read_image_rows(rows)
        for t in (0, 15, 31):
            assert got[t].tobytes() == _src_frame(0, t)[0]


def test_g1_store_refuses_pickle(mini_store):
    import pickle
    with _open_store(mini_store) as store:
        with pytest.raises(TypeError, match="禁止 pickle"):
            pickle.dumps(store)


# ── G4：meta 缺失 / manifest sha 不符 / offsets 不符 各自 raise 且不回退散 npy ──


def test_g4_meta_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta 缺失"):
        fs.StoreMeta.load(tmp_path)


def test_g4_manifest_sha_mismatch(mini_store, tmp_path):
    bad = _copy_store(mini_store, tmp_path)
    _patch_meta(bad, lambda m: m.update(manifest_sha256="0" * 64))
    with pytest.raises(ValueError, match="清单指纹不符"):
        _open_store(bad)


def test_g4_offsets_mismatch(mini_store, tmp_path):
    bad = _copy_store(mini_store, tmp_path)

    def tamper(m):
        m["parts"][1]["start_row"] += 1
    _patch_meta(bad, tamper)
    with pytest.raises(ValueError, match="行区间不连续"):
        fs.StoreMeta.load(bad)


# ── G5：截短 1 字节启动即炸；同尺寸中部翻转由 full 档抓出 ───────────────────────


def test_g5_truncated_blob(mini_store, tmp_path):
    bad = _copy_store(mini_store, tmp_path)
    meta = fs.StoreMeta.load(bad)
    part = bad / meta.parts[-1].path
    with open(part, "r+b") as f:
        f.truncate(part.stat().st_size - 1)
    with pytest.raises(ValueError, match="part 大小不符"):
        _open_store(bad)


def test_g5_midflip_caught_by_full_only(mini_store, tmp_path):
    bad = _copy_store(mini_store, tmp_path)
    meta = fs.StoreMeta.load(bad)
    part = bad / meta.parts[-1].path
    mid = part.stat().st_size // 2
    with open(part, "r+b") as f:
        f.seek(mid)
        b = f.read(1)
        f.seek(mid)
        f.write(bytes([b[0] ^ 0xFF]))
    store = _open_store(bad)          # fast 档（抽样钉在 part 0）抓不到——预期
    store.close()
    with pytest.raises(ValueError, match="sha256 不符"):
        fs.run_full_checks(fs.StoreMeta.load(bad))


# ── G11：两 episode 同 t 帧互换——写侧/校验和层面全绿（预期），verify 必亮红灯 ──


def test_g11_same_t_swap_only_verify_catches(mini_store, tmp_path, capsys):
    bad = _copy_store(mini_store, tmp_path)
    meta = fs.StoreMeta.load(bad)
    manifest = fs.load_manifest(MANIFEST)
    eps = manifest["episodes"][:3]
    t = 40
    rows = [fs.row_of(eps[0]["total_sample_offset"], t),
            fs.row_of(eps[1]["total_sample_offset"], t)]

    def _locate(row):
        for p in meta.parts:
            if p.start_row <= row < p.start_row + p.num_rows:
                return p, (row - p.start_row) * fs.IMAGE_ROW_BYTES
        raise AssertionError(row)

    (pa, oa), (pb, ob) = _locate(rows[0]), _locate(rows[1])
    fa, fb = bad / pa.path, bad / pb.path
    with open(fa, "r+b") as f_a, open(fb, "r+b") as f_b:
        a = os.pread(f_a.fileno(), fs.IMAGE_ROW_BYTES, oa)
        b = os.pread(f_b.fileno(), fs.IMAGE_ROW_BYTES, ob)
        assert a != b, "两帧本就相同，换 t 重试"
        os.pwrite(f_a.fileno(), b, oa)
        os.pwrite(f_b.fileno(), a, ob)

    # 模拟「pack 行寻址 bug」：meta 忠实记录换过的字节（sha/headtail 重算回填），
    # 即写侧校验和层面自洽——钉死「pos memcmp 不钉 g」的边界认知
    def repatch(m):
        m["status"] = "packed"
        m["verify"] = None
        for part in m["parts"]:
            f = bad / part["path"]
            part["sha256"] = fs.sha256_file(f)
            _, ht, cov = fs.headtail_digest(f)
            part["head_tail_digest"] = ht
            part["full_covered"] = cov
    _patch_meta(bad, repatch)

    # 写侧①（pos memcmp 钉 t 不钉 g）对换过的两行依旧通过——预期抓不到
    with _open_store(bad) as store:
        pos_t = store.pos_table[t].tobytes()
        for g in (0, 1):
            assert _src_frame(g, t)[1] == pos_t
    fs.run_full_checks(fs.StoreMeta.load(bad))   # full 校验和层面也全绿——预期

    # verify 全量对拍必须亮红灯（g 级身份唯一凭据）
    with pytest.raises(SystemExit):
        packer.cmd_verify(_verify_args(bad))
    out = capsys.readouterr().out
    assert "VERIFY_PACK=FAIL" in out and "mismatches=2" in out
    assert fs.StoreMeta.load(bad).status == "packed"   # FAIL 不回填 verified
    assert (bad / fs.LOCK_RELPATH).exists()            # FAIL 保留锁阻断读侧


# ── G12：短读续读补齐；EOF/越界立即 raise ─────────────────────────────────────


def test_g12_short_read_and_eof(mini_store, monkeypatch):
    real_preadv = os.preadv
    with _open_store(mini_store) as store:
        expect = store.read_image_rows([3])[0].tobytes()

        calls = {"n": 0}

        def half_then_real(fd, bufs, off):
            calls["n"] += 1
            if calls["n"] == 1:
                buf = bufs[0]
                return real_preadv(fd, [buf[: max(1, len(buf) // 2)]], off)
            return real_preadv(fd, bufs, off)

        monkeypatch.setattr(os, "preadv", half_then_real)
        got = store.read_image_rows([3])[0].tobytes()
        assert got == expect and calls["n"] >= 2   # 短读被续读补齐且结果逐位正确

        monkeypatch.setattr(os, "preadv", lambda fd, bufs, off: 0)
        with pytest.raises(RuntimeError, match="EOF"):
            store.read_image_rows([3])

        monkeypatch.setattr(os, "preadv", real_preadv)
        with pytest.raises(IndexError, match="越界"):
            store.read_image_rows([store.num_rows])
        with pytest.raises(IndexError, match="越界"):
            store.pos_rows([store.meta.num_pos_rows])
        with pytest.raises(IndexError, match="越界"):
            store.state_rows([-1])


# ── G14：status != verified 时 packed 分派必 raise；ALLOW_UNVERIFIED=1 放行打 WARNING ──


def test_g14_require_verified(mini_store, monkeypatch, caplog):
    meta = fs.StoreMeta.load(mini_store)
    fs.require_verified(meta)   # verified 库直接放行

    packed_meta = dataclasses.replace(meta, status="packed")
    monkeypatch.delenv("MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED", raising=False)
    with pytest.raises(RuntimeError, match="未通过全量 verify"):
        fs.require_verified(packed_meta)

    monkeypatch.setenv("MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED", "1")
    with caplog.at_level(logging.WARNING, logger="mme_vla_suite.datastore.framesamp_store"):
        fs.require_verified(packed_meta)   # 放行
    assert any("ALLOW_UNVERIFIED" in r.message for r in caplog.records)   # 必打 WARNING


def test_g14_pack_lock_blocks_dispatch(mini_store, tmp_path):
    bad = _copy_store(mini_store, tmp_path)
    fs.require_no_pack_lock(bad)   # 无锁放行
    (bad / fs.LOCK_RELPATH).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pack.lock"):
        fs.require_no_pack_lock(bad)


# ════ Dataset 组（S3；以下用例懒 import framesamp_dataset → flax/jax，必须在
# ════ 全部 fork Pool 用例之后）═══════════════════════════════════════════════


def _fake_data_config() -> types.SimpleNamespace:
    """FrameSampDataset 只访问 norm_stats['state'] 的 q01/q99/mean/std 与
    use_quantile_norm——守卫用轻量替身（f64，与真实 json 加载后同 dtype）。"""
    ns = types.SimpleNamespace(
        q01=np.linspace(-1.0, -0.5, 8), q99=np.linspace(0.5, 1.0, 8),
        mean=np.zeros(8), std=np.ones(8))
    return types.SimpleNamespace(norm_stats={"state": ns}, use_quantile_norm=True)


def _make_dataset(mini_store: pathlib.Path, hc_name="perceptual-framesamp-context.yaml"):
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    return FrameSampDataset(
        str(mini_store), data_config=_fake_data_config(),
        history_config=get_history_config(hc_name), action_horizon=20,
        source_root=str(REF_SHARD), manifest_path=str(MANIFEST))


def test_g6a_exec_lookup_formula():
    """换算公式单测：直接用全量清单构造查表数组（不构造 Dataset、不碰 store）。"""
    manifest = fs.load_manifest(MANIFEST)
    epis_of, step_of, row_base = fs.build_exec_lookup(manifest)
    assert len(epis_of) == 395289
    for h5 in ("record_dataset_VideoUnmask.h5", "record_dataset_VideoUnmaskSwap.h5"):
        ep = next(e for e in manifest["episodes"] if e["h5_file"] == h5)
        idx = ep["exec_sample_offset"]
        assert ep["exec_start_idx"] > 0, h5   # Video* 任务必有 demo 前缀
        assert int(step_of[idx]) == ep["exec_start_idx"], h5   # ⚠ 漏 exec_start_idx 即错位
        assert int(epis_of[idx]) == ep["global_episode_idx"]
        assert int(row_base[ep["global_episode_idx"]]) == ep["total_sample_offset"]


def test_g2_pad_dtype_boundary(mini_store):
    """step=30 短样本与 step=31 满长样本经 _pad 后各键 dtype 一致且为
    image bf16 / pos f32 / stt f32（episode 0 exec_start=0 → idx 即 step）。"""
    import ml_dtypes
    ds = _make_dataset(mini_store)
    try:
        store = ds._ensure_store()
        from mme_vla_suite.shared.sampling import even_sampling_indices
        results = {}
        for step in (30, 31):
            frames = np.asarray(even_sampling_indices(step, 32), np.int64)
            rows = ds._row_base[0] + frames
            img, pos, stt, mask = ds._pad(
                store.read_image_rows(rows), store.pos_rows(frames),
                store.state_rows(rows), len(frames))
            assert img.dtype == ml_dtypes.bfloat16 and img.shape == (32, 16, 2048)
            assert pos.dtype == np.float32 and stt.dtype == np.float32
            assert mask.dtype == np.bool_ and int(mask.sum()) == step + 1
            results[step] = (img.dtype, pos.dtype, stt.dtype)
        assert results[30] == results[31]   # 短/满长 dtype 逐键一致
        # 全链 getitem 交付层复核（含 padding 后 reshape/repeat）
        for idx, n in ((30, 31), (31, 32)):
            item = ds[idx]
            assert item["static_image_emb"].dtype == ml_dtypes.bfloat16
            assert item["static_image_emb"].shape == (512, 2048)
            assert item["static_pos_emb"].dtype == np.float32
            assert item["static_state_emb"].dtype == np.float64   # normalize 后恒 f64（同旧路径）
            assert int(item["static_mask"].sum()) == n * 16
    finally:
        ds.close()


def test_g3_duplicate_indices_not_deduped(mini_store):
    """选帧重复索引必须重复输出不去重（防未来 gather 实现引入 unique/排序优化）。"""
    ds = _make_dataset(mini_store)
    try:
        store = ds._ensure_store()
        # episode 0 开场数十帧为静止帧、特征逐位相同（2026-08-27 实测 t0..t20 全等、
        # t50 起不同）——动态找一行与 row 7 不同的行，避免对内容做静态假设
        base = store.read_image_rows(np.array([7], np.int64))[0].tobytes()
        distinct = next(r for r in range(8, store.num_rows)
                        if store.read_image_rows(np.array([r], np.int64))[0]
                        .tobytes() != base)
        out = store.read_image_rows(np.array([7, 7, distinct], np.int64))
        assert out.shape[0] == 3
        assert out[0].tobytes() == out[1].tobytes() == base
        assert out[2].tobytes() != base
        assert store.pos_rows([3, 3]).shape[0] == 2
        assert store.state_rows([5, 5]).shape[0] == 2
    finally:
        ds.close()


def test_g8_no_thread_pool(mini_store, monkeypatch):
    """mock 线程池抛错证明已彻底移除（旧路径每样本新建 ≤32 线程）。"""
    import concurrent.futures as cf

    def boom(*a, **k):
        raise AssertionError("FrameSampDataset 路径不得使用 ThreadPoolExecutor")

    ds = _make_dataset(mini_store)
    try:
        monkeypatch.setattr(cf.ThreadPoolExecutor, "__init__", boom)
        item = ds[5]
        assert item["static_image_emb"].shape == (512, 2048)
    finally:
        ds.close()


def test_g9_use_state_emb_pinned(mini_store):
    from omegaconf import OmegaConf
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    hc = OmegaConf.merge(get_history_config("perceptual-framesamp-context.yaml"),
                         {"use_state_emb": True})
    with pytest.raises(ValueError, match="use_state_emb"):
        FrameSampDataset(str(mini_store), data_config=_fake_data_config(),
                         history_config=hc, action_horizon=20,
                         source_root=str(REF_SHARD), manifest_path=str(MANIFEST))


def test_g13_modul_config_rejected(mini_store):
    """喂同形的 modul 配置（integration_type=modulation、memory_token_dim=1024）必拒。"""
    with pytest.raises(ValueError, match="形制断言失败"):
        _make_dataset(mini_store, hc_name="perceptual-framesamp-modul.yaml")


def _g10_child(ds, expect_pos_nbytes, expect_state_nbytes, q):
    """G10 spawn 子进程体：消费 Dataset 并回报 store 懒构造契约各断言项。"""
    try:
        item = ds[0]
        s = ds._store
        os.fstat(s._fds[0])   # fd 有效可读
        q.put({"ok": True,
               "pid_ok": s.owner_pid == os.getpid(),
               "shape_ok": item["static_image_emb"].shape == (512, 2048),
               "pos_nbytes_ok": s.pos_table.nbytes == expect_pos_nbytes,
               "state_nbytes_ok": s.state_table.nbytes == expect_state_nbytes,
               "pos_base_none": s.pos_table.base is None,
               "state_base_none": s.state_table.base is None})
    except Exception as e:  # noqa: BLE001 —— 子进程任何失败原样带回父进程判定
        q.put({"ok": False, "err": repr(e)})


def test_g10_spawn_lazy_store(mini_store):
    """spawn 子进程消费 Dataset：worker 内 store 懒构造、两张小表为进程内副本、
    父进程全程无句柄（__getstate__ 剔除契约）。"""
    import multiprocessing
    ds = _make_dataset(mini_store)   # 父进程刻意不触发 __getitem__
    meta = fs.StoreMeta.load(mini_store)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_g10_child, args=(
        ds, meta.raw["tables"][fs.POS_KEY]["byte_count"],
        meta.raw["tables"][fs.STATE_KEY]["byte_count"], q))
    p.start()
    res = q.get(timeout=180)
    p.join(30)
    assert res.get("ok"), res
    assert all(res[k] for k in ("pid_ok", "shape_ok", "pos_nbytes_ok",
                                "state_nbytes_ok", "pos_base_none", "state_base_none")), res
    assert ds._store is None   # 父进程无句柄泄漏（从未构造）


def test_dispatch_packed_gates(mini_store, tmp_path, monkeypatch):
    import importlib
    dl = importlib.import_module("mme_vla_suite.training.dataloader")
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    hc = get_history_config("perceptual-framesamp-context.yaml")
    dc = _fake_data_config()

    # subset 闸：默认 raise，开发期显式放行
    monkeypatch.delenv("MMEVLA_FRAMESAMP_ALLOW_SUBSET", raising=False)
    with pytest.raises(RuntimeError, match="subset"):
        dl._create_framesamp_dataset(str(mini_store), dc, hc, 20)
    monkeypatch.setenv("MMEVLA_FRAMESAMP_ALLOW_SUBSET", "1")
    ds = dl._create_framesamp_dataset(str(mini_store), dc, hc, 20)
    assert isinstance(ds, FrameSampDataset) and len(ds) == fs.StoreMeta.load(
        mini_store).num_exec_samples
    ds.close()

    # pack.lock 闸
    locked = _copy_store(mini_store, tmp_path)
    (locked / fs.LOCK_RELPATH).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pack.lock"):
        dl._create_framesamp_dataset(str(locked), dc, hc, 20)

    # unverified 闸（G14 的分派层落点）
    unv = tmp_path / "unv"
    shutil.copytree(mini_store, unv)
    _patch_meta(unv, lambda m: m.update(status="packed", verify=None))
    monkeypatch.delenv("MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED", raising=False)
    with pytest.raises(RuntimeError, match="未通过全量 verify"):
        dl._create_framesamp_dataset(str(unv), dc, hc, 20)


# ── G7：CPU 后端生成 pos 表被拒（懒 import jax，必须排在全部 fork 用例之后）────


def test_g7_pos_table_generation_refuses_cpu():
    with pytest.raises(RuntimeError, match="GPU 后端"):
        packer.generate_pos_table_posemb3d(586)
