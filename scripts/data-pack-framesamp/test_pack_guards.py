"""Store 组守卫测试（v2 计划 C.5，S2 交付：G1/G4/G5/G7/G11/G12/G14）。

刻意制造失败断言亮红灯；JAX_PLATFORMS=cpu、pytest 直跑：

    UV_LINK_MODE=copy uv run pytest scripts/data-pack-framesamp/test_pack_guards.py -x -q

迷你库：ref-shard（v1-store/datasets/ref-shard）派生 global_episode_idx 连续前缀
[0..2]（ButtonUnmask 前 3 个 episode，episode 0 有 291 帧满足 ≥33 硬约束），
session 级 fixture 打包一次（含全量 verify）供各守卫复用。

⚠ 测试定义顺序有意为之：G7 懒 import jax 必须排在一切 fork Pool 用例（fixture
构建、G11 verify）之后——jax 不允许进 fork 前进程。
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
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import framesamp_store as fs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pack_framesamp_store", _HERE / "pack_framesamp_store.py")
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


# ── G7：CPU 后端生成 pos 表被拒（懒 import jax，必须排在全部 fork 用例之后）────


def test_g7_pos_table_generation_refuses_cpu():
    with pytest.raises(RuntimeError, match="GPU 后端"):
        packer.generate_pos_table_posemb3d(586)
