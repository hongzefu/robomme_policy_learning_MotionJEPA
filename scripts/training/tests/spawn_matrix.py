#!/usr/bin/env python3
"""S3 判定工具：迷你库真实 spawn loader 矩阵 w0/w1/w4/w16 × 2 epoch + fd 泄漏检查。

对 FrameSampDataset 起真实 torch DataLoader（spawn context、persistent_workers，
与训练链 TorchDataLoader 同参数族），每档跑满 2 个 epoch，收官后检查主进程
/proc/self/fd 计数回到基线（B.2 fd 生命周期契约的验收）。

只测 spawn/fd/懒构造行为，不测数值（内容逐位一致归 S5 第一块）；默认
JAX_PLATFORMS=cpu（历史上 worker import 链经 shared.data_utils 拉 flax/jax；
commitV4.4 起选帧函数已搬 shared.sampling（只依赖 numpy）、该导入负担解除，
此处保留 cpu 档仅为口径与历史记录可比）。norm stats 用轻量替身。

用法（迷你库需先经 test_pack_guards.py 的 fixture 同款流程打包，或指向任一
verified 库）：
  UV_LINK_MODE=copy uv run python scripts/training/tests/spawn_matrix.py \
    --store <打包库根> --source <源库根> --manifest v1-store/episode_manifest.json
判定行：MATRIX=PASS workers=0,1,4,16 epochs=2
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys
import time
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _fake_data_config() -> types.SimpleNamespace:
    ns = types.SimpleNamespace(
        q01=np.linspace(-1.0, -0.5, 8), q99=np.linspace(0.5, 1.0, 8),
        mean=np.zeros(8), std=np.ones(8))
    return types.SimpleNamespace(norm_stats={"state": ns}, use_quantile_norm=True)


def _collate(batch):
    """矩阵专用最小 collate：只堆四个 static 键（None 键不参与），
    足以驱动 worker 端完整 __getitem__ 与批装配。"""
    return {k: np.stack([s[k] for s in batch])
            for k in ("static_image_emb", "static_pos_emb",
                      "static_state_emb", "static_mask")}


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _noop() -> None:
    pass


def _warmup_spawn() -> None:
    """预热 multiprocessing spawn 基础设施（resource_tracker 单例等）。

    首次使用 spawn 语义（信号量/队列）会在主进程拉起 resource_tracker 并常驻
    一条管道 fd——进程级一次性开销、非泄漏（2026-08-27 矩阵首跑实测 w1 基线
    5→6、后续档 6→6）；先预热再取基线，fd 泄漏判定才干净。"""
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_noop)
    p.start()
    p.join()
    q.close()
    q.join_thread()


def run_one(store, source, manifest, workers: int, epochs: int, batch: int) -> dict:
    import torch
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset

    fd_base = _fd_count()
    ds = FrameSampDataset(
        str(store), data_config=_fake_data_config(),
        history_config=get_history_config("perceptual-framesamp-context.yaml"),
        action_horizon=20, source_root=str(source), manifest_path=str(manifest))
    g = torch.Generator()
    g.manual_seed(42)
    mp_ctx = "spawn" if workers > 0 else None
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True, generator=g,
        num_workers=workers, multiprocessing_context=mp_ctx,
        persistent_workers=workers > 0, collate_fn=_collate, drop_last=True)
    t0 = time.perf_counter()
    n_batches = 0
    for _ in range(epochs):
        for b in loader:
            if b["static_image_emb"].shape[1:] != (512, 2048):
                raise RuntimeError(f"batch 形状异常: {b['static_image_emb'].shape}")
            n_batches += 1
    elapsed = time.perf_counter() - t0
    del loader          # 触发 worker shutdown（persistent_workers 随迭代器/loader 回收）
    ds.close()          # w0 场景主进程即 owner，显式关（B.2 ③ 验收）
    del ds
    gc.collect()
    deadline = time.time() + 15
    fd_after = _fd_count()
    while fd_after > fd_base and time.time() < deadline:   # worker 管道回收有滞后
        time.sleep(0.5)
        gc.collect()
        fd_after = _fd_count()
    leak = fd_after - fd_base
    return {"workers": workers, "batches": n_batches, "elapsed": round(elapsed, 1),
            "fd_base": fd_base, "fd_after": fd_after, "leak": leak}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", default="0,1,4,16")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    workers = [int(w) for w in args.workers.split(",")]
    _warmup_spawn()
    rows = []
    for w in workers:
        r = run_one(args.store, args.source, args.manifest, w, args.epochs, args.batch)
        rows.append(r)
        print(f"[matrix] w{w}: batches={r['batches']} elapsed={r['elapsed']}s "
              f"fd {r['fd_base']}→{r['fd_after']} (leak={r['leak']})", flush=True)
    bad = [r for r in rows if r["leak"] > 0]
    expect = None
    for r in rows:
        if expect is None:
            expect = r["batches"]
        elif r["batches"] != expect:
            bad.append(r)   # 各档 batch 数必须一致（同 len/batch/drop_last）
    if bad:
        print(f"MATRIX=FAIL bad={bad}")
        raise SystemExit(1)
    print(f"MATRIX=PASS workers={args.workers} epochs={args.epochs} "
          f"batches_per_run={expect}")


if __name__ == "__main__":
    main()
