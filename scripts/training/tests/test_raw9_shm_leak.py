#!/usr/bin/env python3
"""raw9 共享内存泄漏回归测试（FIX-SHM）。

事故：正式训练第约 7560 步 DataLoader worker 在 ``_collate_fn_shm → _shared_empty`` 报
``No space left on device``，``/dev/shm``（561 G tmpfs）在训练期间持续累积。

根因：``_PrefetchIter`` 的释放线程 ``_Releaser`` 用无界 ``SimpleQueue`` 接收用完的主机批，释放（munmap + 归还
tmpfs 页）一旦慢于取批节奏，待释放批在队列里无限堆积、全部仍占着 /dev/shm。修复：队列容量 1（反压）。

本测试两部分：
  ① ``RELEASER_CHECK``：交付 60 个析构耗时 20 ms 的对象，断言同时存活的待释放对象 ≤4、close 后全部释放；
  ② 真实库（4task-v2-1600ep framesamp-8x8 + wanraw9s32.yaml）构造与 train.py 相同的
     ``transform_dataset → TorchDataLoader``（spawn worker、persistent_workers、shuffle），走完整
     ``TorchDataLoader.__iter__``（后台取批线程 + make_array + 释放线程）迭代 ``--batches`` 批，
     每 ``--every`` 批记录 ``df /dev/shm`` 增量、主进程与 worker 的共享内存 fd 数 / 映射字节、释放队列长度。
判据：本进程树（主进程 + worker）映射的共享内存末段（最后 3 个采样点）相对预热段增量 ≤ 4 批（另报整机 df 增量供参考，可能混入他人进程）、
主进程映射末段 ≤ 预热段 + 2 批、释放队列长度 ≤1；主进程映射超过
30 批即提前停止判 FAIL（回归时不吃满共享内存）。
末行 ``SHM_LEAK=PASS|FAIL batches=<n> shm_growth_mb=<x> batch_mb=<y> main_map_mb=<a>-><b> rel_q_max=<q> ...``。

运行（≤5 分钟）：
  JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv \\
    PYTHONPATH=<worktree>/src uv run --no-sync python scripts/training/tests/test_raw9_shm_leak.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
WT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))
MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
LIB = MAIN / "v1-store/datasets/4task-v2-1600ep-604f16da"
YAML = WT / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml"
MOTION_ROOT = LIB / "motion-wan-raw9-s32"


def _shm_used() -> int:
    st = os.statvfs("/dev/shm")
    return (st.f_blocks - st.f_bfree) * st.f_frsize


def _proc_stats(pid: int) -> dict:
    """fd 总数、指向共享内存段（/dev/shm、torch_、memfd）的 fd 数、映射着的 torch_ 段字节数、RssShmem。"""
    fd_total = fd_shm = 0
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            fd_total += 1
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if tgt.startswith("/dev/shm") or "torch_" in tgt or tgt.startswith("/memfd:"):
                fd_shm += 1
    except OSError:
        return {}
    mapped = 0
    segs = set()
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if "/torch_" in line or "/dev/shm/" in line:
                parts = line.split()
                a, b = (int(x, 16) for x in parts[0].split("-"))
                mapped += b - a
                segs.add(parts[-2] if parts[-1] == "(deleted)" else parts[-1])
    rss_shm = 0
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("RssShmem:"):
                rss_shm = int(line.split()[1]) * 1024
    return {"fd": fd_total, "fd_shm": fd_shm, "map_mb": mapped / 2**20, "segs": len(segs), "rss_shm_mb": rss_shm / 2**20}


def _build(batch: int, workers: int, epoch_samples: int = 0):
    import omegaconf
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    from openpi.training.data_loader import TorchDataLoader, transform_dataset

    tc = _config.get_config("mme_vla_suite_b128_80k")
    assets = MAIN / "v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
    tc = dataclasses.replace(tc, data=dataclasses.replace(
        tc.data, assets=dataclasses.replace(tc.data.assets, assets_dir=str(assets), asset_id="robomme")))
    dc = tc.data.create(tc.assets_dirs, tc.model)
    hc = omegaconf.OmegaConf.load(YAML)
    os.environ["MMEVLA_MOTION_STORE"] = str(MOTION_ROOT)
    raw = _create_framesamp_dataset(str(LIB / "framesamp-8x8"), dc, hc, int(tc.model.action_horizon))
    ds = transform_dataset(raw, dc)
    if epoch_samples:
        # 缩短 epoch（同一真实库的前 epoch_samples 个样本随机采样），以便在几百批内跨越多次 epoch 边界
        import torch
        sampler = torch.utils.data.SubsetRandomSampler(range(epoch_samples), generator=torch.Generator().manual_seed(0))
        return TorchDataLoader(ds, local_batch_size=batch, sampler=sampler, num_workers=workers, seed=0)
    return TorchDataLoader(ds, local_batch_size=batch, shuffle=True, num_workers=workers, seed=0)


def check_releaser_bounded() -> tuple[bool, str]:
    """单元检查：释放慢于交付时，``_Releaser`` 必须反压，存活的待释放批数有上界（修复前无界队列会无限堆积）。"""
    import threading
    from openpi.training.data_loader import _Releaser

    live = {"n": 0, "max": 0}
    lock = threading.Lock()

    class SlowBatch:                       # 释放（析构）耗时 20 ms，模拟 6 GB 共享内存批的 munmap + 归还页
        def __init__(self):
            with lock:
                live["n"] += 1
                live["max"] = max(live["max"], live["n"])

        def __del__(self):
            time.sleep(0.02)
            with lock:
                live["n"] -= 1

    rel = _Releaser()
    for _ in range(60):                    # 交付节奏远快于释放
        rel.hand_over(SlowBatch())
    rel.close()
    # 上界 4：释放线程持有 1 + 队列 1 + 主线程刚构造待交付 1 + 正在析构的上一批 1
    ok = live["max"] <= 4 and live["n"] == 0
    return ok, f"releaser_max_live={live['max']} releaser_left={live['n']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batches", type=int, default=240)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--max-growth-batches", type=float, default=4.0,
                    help="稳态 /dev/shm 增量上限（以批字节量计）")
    ap.add_argument("--abort-batches", type=float, default=30.0,
                    help="主进程映射的共享内存超过这么多批即提前判 FAIL 并停止（回归时不把共享内存吃满）")
    ap.add_argument("--epoch-samples", type=int, default=0, help=">0 时每 epoch 只取这么多样本（跨 epoch 边界）")
    ap.add_argument("--step-sleep", type=float, default=0.0, help="每批后 sleep（模拟慢消费）")
    args = ap.parse_args()

    rel_ok, rel_msg = check_releaser_bounded()
    print(f"RELEASER_CHECK={'PASS' if rel_ok else 'FAIL'} {rel_msg}", flush=True)

    loader = _build(args.batch, args.workers, args.epoch_samples)
    it = iter(loader)
    rows = []
    batch_mb = 0.0
    aborted = False
    t0 = time.time()
    shm0 = _shm_used()
    for n in range(1, args.batches + 1):
        b = next(it)
        if n == 1:
            import jax
            batch_mb = sum(int(x.nbytes) for x in jax.tree.leaves(b)) / 2**20
        del b
        if args.step_sleep:
            time.sleep(args.step_sleep)
        main_st = _proc_stats(os.getpid())
        if n % args.every == 0 or n == 1 or main_st["map_mb"] > args.abort_batches * batch_mb:
            ti = it._torch_iter
            wpids = [w.pid for w in getattr(ti, "_workers", [])] if ti is not None else []
            ws = [w for w in (_proc_stats(p) for p in wpids) if w]
            w_map = sum(w["map_mb"] for w in ws)
            row = {
                "n": n, "t": round(time.time() - t0, 1),
                "df_shm_mb": round((_shm_used() - shm0) / 2**20, 1),   # 整机口径，可能混入其他进程
                "own_mb": round(main_st["map_mb"] + w_map, 1),        # 本进程树映射着的共享内存段（判据口径）
                "main": main_st, "rel_q": it._releaser._q.qsize(),
                "w_fd_shm": sum(w["fd_shm"] for w in ws), "w_map_mb": round(w_map, 1),
            }
            rows.append(row)
            print("SHM_ROW=" + json.dumps(row), flush=True)
            if main_st["map_mb"] > args.abort_batches * batch_mb:
                aborted = True
                break
    it.close()

    warm = [r for r in rows if r["n"] <= args.warmup]
    steady = [r for r in rows if r["n"] > args.warmup] or rows[-1:]
    base = max(r["own_mb"] for r in warm)
    growth = max(r["own_mb"] for r in steady[-3:]) - base        # 末段（最后 3 个采样点）对预热段：看趋势不看瞬时尖峰
    main_map_warm = max(r["main"]["map_mb"] for r in warm)
    main_map_last = max(r["main"]["map_mb"] for r in steady[-3:])
    rel_q_max = max(r["rel_q"] for r in rows)
    ok = (rel_ok and not aborted and growth <= args.max_growth_batches * batch_mb
          and main_map_last <= main_map_warm + 2 * batch_mb and rel_q_max <= 1)
    df_growth = max(r["df_shm_mb"] for r in steady[-3:]) - max(r["df_shm_mb"] for r in warm)
    print(f"SHM_LEAK={'PASS' if ok else 'FAIL'} batches={rows[-1]['n']} shm_growth_mb={growth:.1f} "
          f"df_growth_mb={df_growth:.1f} "
          f"batch_mb={batch_mb:.1f} main_map_mb={main_map_warm:.0f}->{main_map_last:.0f} rel_q_max={rel_q_max} "
          f"aborted={int(aborted)} {rel_msg}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    os._exit(main())
