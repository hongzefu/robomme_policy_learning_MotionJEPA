#!/usr/bin/env python3
"""wan_raw9 训练数据管线 CPU 侧分段测速（计划 1112-4task-modul-wanraw9-s32-50k-plan.html §3.4、分配表 S5-L1）。

按 train.py 同样方式构造数据集（``_create_framesamp_dataset`` → ``transform_dataset`` → ``TorchDataLoader``，
per-host batch 默认 128、shuffle、spawn worker），分三段测：

``--part sample``   主进程单样本：``FrameSampDataset.__getitem__`` 总耗时、其中 ``_fill_motion_raw9``（motion 取数）
                    耗时、transforms 耗时；
``--part collate``  主进程对 B 个已变换样本跑 ``_collate_fn_shm``（np.stack + 共享内存化），分拆 stack 与 share；
``--part loader``   真实 ``TorchDataLoader``（N worker、prefetch P）：每批 ``next()``（含 worker→主进程传输）、
                    ``_from_shared_torch``、主进程整批顺序读一遍（模拟 H2D 前的页错误/读带宽）以及 CPU 上
                    ``jax.make_array_from_process_local_data`` 的耗时与主进程释放整批的耗时，报 batches/s；
                    ``--direct-prefetch N`` 绕过 TorchDataLoader 直接建 torch DataLoader 以便在基线代码上也设 prefetch；
``--part iter``     完整 ``TorchDataLoader.__iter__``（含 make_array），每步 sleep ``--step-sleep`` 秒模拟设备计算，
                    报每步墙钟与 next() 耗时（train.py 的 next(data_iter) 与异步派发的 train_step 重叠）。

基线对照：把 BASE 的 src 用 ``git archive <BASE> src`` 解到临时目录，以 PYTHONPATH 指过去跑同一脚本。
只读，不写任何库；每次运行 ≤5 分钟（``--measure`` 控制批数）。末行 ``BENCH_DL=<json>``，随后 ``os._exit``。

用法（worktree 内，环境取法同 S5 委派）：
  JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv \\
    PYTHONPATH=<worktree>/src uv run --no-sync python scripts/training/g0/bench_dataloader_raw9.py --part loader \\
    --workers 16 --prefetch 2 --warmup 4 --measure 20
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
REPO = pathlib.Path(__file__).resolve().parents[3]
MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
LIB = MAIN / "v1-store/datasets/4task-v2-1600ep-604f16da"
YAML_DIR = REPO / "src/mme_vla_suite/models/config/robomme"

import jax  # noqa: E402
import numpy as np  # noqa: E402


def build(yaml_name: str, motion_root: pathlib.Path | None):
    import omegaconf
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    from openpi.training.data_loader import transform_dataset

    tc = _config.get_config("mme_vla_suite_b128_80k")
    assets = MAIN / "v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
    tc = dataclasses.replace(tc, data=dataclasses.replace(
        tc.data, assets=dataclasses.replace(tc.data.assets, assets_dir=str(assets), asset_id="robomme")))
    dc = tc.data.create(tc.assets_dirs, tc.model)
    hc = omegaconf.OmegaConf.load(YAML_DIR / yaml_name)
    if motion_root is not None:
        os.environ["MMEVLA_MOTION_STORE"] = str(motion_root)
    raw = _create_framesamp_dataset(str(LIB / "framesamp-8x8"), dc, hc, int(tc.model.action_horizon))
    return raw, transform_dataset(raw, dc), tc


def _idxs(n_total, n, seed):
    return np.random.default_rng(seed).choice(n_total, size=n, replace=False).tolist()


def part_sample(raw, ds, args):
    t_fill = []
    orig = raw._fill_motion_raw9

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        t_fill.append(time.perf_counter() - t0)
        return r

    raw._fill_motion_raw9 = timed
    idxs = _idxs(len(raw), args.samples, args.seed)
    t_get, t_tf = [], []
    for i in idxs:
        t0 = time.perf_counter()
        s = raw[i]
        t1 = time.perf_counter()
        ds._transform(s)
        t2 = time.perf_counter()
        t_get.append(t1 - t0)
        t_tf.append(t2 - t1)
    raw._fill_motion_raw9 = orig
    g, f, tf = (float(np.mean(x[2:])) for x in (t_get, t_fill, t_tf))   # 去掉前 2 个冷启动
    return {"part": "sample", "n": len(idxs), "getitem_ms": g * 1e3, "motion_fill_ms": f * 1e3,
            "rest_ms": (g - f) * 1e3, "transform_ms": tf * 1e3,
            "per_batch_single_core_s": (g + tf) * args.batch}


def part_collate(raw, ds, args):
    import torch
    from openpi.training import data_loader as dl

    idxs = _idxs(len(raw), args.batch, args.seed)
    t_g0 = time.perf_counter()
    items = [ds[i] for i in idxs]
    res = {"part": "collate", "batch": args.batch, "per_sample_getitems_s": time.perf_counter() - t_g0}
    t0 = time.perf_counter()
    stacked = dl._collate_fn(items)
    t1 = time.perf_counter()
    sh = dl._to_shared_torch(stacked)
    t2 = time.perf_counter()
    n = 0

    def share(t):
        nonlocal n
        if t is not None:
            t.share_memory_()                      # DataLoader 队列 pickle 时对非共享 storage 做的同一件事
            n += t.numel() * t.element_size()
        return t

    import jax
    jax.tree.map(share, sh, is_leaf=lambda x: x is None)
    t3 = time.perf_counter()
    res.update(stack_s=t1 - t0, to_torch_s=t2 - t1, share_s=t3 - t2, batch_bytes=n)
    del stacked, sh
    if hasattr(dl, "_collate_fn_shm"):
        t4 = time.perf_counter()
        out = dl._collate_fn_shm(items)
        jax.tree.map(share, out, is_leaf=lambda x: x is None)   # 已共享则为空操作
        res["collate_fn_shm_total_s"] = time.perf_counter() - t4
        del out
    if hasattr(type(ds), "__getitems__"):        # S5-L1 新链：整批共享缓冲直填 + collate 直写共享内存
        t5 = time.perf_counter()
        pre = ds.__getitems__(idxs)
        t6 = time.perf_counter()
        out = dl._collate_fn_shm(pre)
        jax.tree.map(share, out, is_leaf=lambda x: x is None)
        res["new_getitems_s"] = t6 - t5
        res["new_collate_s"] = time.perf_counter() - t6
        res["new_worker_batch_total_s"] = time.perf_counter() - t5
    res["old_worker_batch_total_s"] = res["per_sample_getitems_s"] + res["stack_s"] + res["share_s"]
    return res


def _populate(a: np.ndarray) -> None:
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    page = os.sysconf("SC_PAGE_SIZE")
    addr = a.__array_interface__["data"][0]
    start = addr - addr % page
    n = addr + a.nbytes - start
    if libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(n), 22) != 0:   # MADV_POPULATE_READ
        raise OSError(ctypes.get_errno(), "madvise(MADV_POPULATE_READ) 失败")


def part_loader(raw, ds, args):
    import jax
    import torch
    from openpi.training import data_loader as dl

    if args.direct_prefetch:
        # 直接构造 torch DataLoader（与 TorchDataLoader 同 dataset / collate / worker_init / spawn），只为能在
        # 基线代码上也指定 prefetch_factor：worker 侧供给能力要在预取缓冲（workers × prefetch 批）耗尽后才测得到
        import multiprocessing
        g = torch.Generator()
        g.manual_seed(args.seed)
        tl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         multiprocessing_context=multiprocessing.get_context("spawn"),
                                         persistent_workers=True, collate_fn=dl._collate_fn_shm,
                                         worker_init_fn=dl._worker_init_fn, drop_last=True, generator=g,
                                         prefetch_factor=args.direct_prefetch)
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B"))
    else:
        loader = dl.TorchDataLoader(ds, local_batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                    seed=args.seed, framework="jax",
                                    **({"prefetch_factor": args.prefetch} if args.prefetch else {}))
        tl = loader.torch_loader
        sharding = loader._sharding
    it = iter(tl)
    t_start = time.perf_counter()
    rows = []
    buf = None
    for k in range(args.warmup + args.measure):
        t0 = time.perf_counter()
        b = next(it)
        t1 = time.perf_counter()
        b = dl._from_shared_torch(b)
        t2 = time.perf_counter()
        me = b["motion_emb"]
        tp0 = time.perf_counter()
        if args.populate:
            _populate(me)                           # 只做页错误（MADV_POPULATE_READ），把「建映射」与「读带宽」拆开
        tpop = time.perf_counter() - tp0
        t2 += tpop
        if buf is None:
            buf = np.empty_like(me)
        if not args.no_copy:
            np.copyto(buf, me)                      # 主进程整批读一遍（页错误 + 读带宽，H2D 前必经）
        t3 = time.perf_counter()
        if args.make_array:
            arr = jax.make_array_from_process_local_data(sharding, me)
            arr.block_until_ready()
            del arr
        t4 = time.perf_counter()
        del b, me                                   # 主进程释放整批（最后一个引用：munmap + 归还 tmpfs 页）
        t5 = time.perf_counter()
        if k == args.warmup - 1:
            t_meas = time.perf_counter()
        if k >= args.warmup:
            rows.append((t1 - t0, t2 - t1 - tpop, t3 - t2, t4 - t3, tpop, t5 - t4))
    t_end = time.perf_counter()
    r = np.asarray(rows)
    return {"part": "loader", "workers": args.workers, "prefetch": args.direct_prefetch or args.prefetch or 2, "batch": args.batch,
            "warmup": args.warmup, "measure": args.measure,
            "batches_per_s": args.measure / (t_end - t_meas), "s_per_batch": (t_end - t_meas) / args.measure,
            "next_s": float(r[:, 0].mean()), "from_shared_s": float(r[:, 1].mean()),
            "populate_s": float(r[:, 4].mean()), "touch_copy_s": float(r[:, 2].mean()), "make_array_cpu_s": float(r[:, 3].mean()), "free_s": float(r[:, 5].mean()),
            "startup_s": t_meas - t_start}


def part_iter(raw, ds, args):
    """完整 ``TorchDataLoader.__iter__``（含 make_array 到 CPU 设备）；每步另 sleep ``--step-sleep`` 秒模拟 GPU 计算
    （train.py 里 ptrain_step 异步派发后紧接 next(data_iter)，主机取批与设备计算重叠）。报每步墙钟与 next() 耗时。"""
    from openpi.training import data_loader as dl

    loader = dl.TorchDataLoader(ds, local_batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                seed=args.seed, framework="jax",
                                **({"prefetch_factor": args.prefetch} if args.prefetch else {}))
    it = iter(loader)
    t_start = time.perf_counter()
    nxt = []
    for k in range(args.warmup + args.measure):
        if k == args.warmup:
            t_meas = time.perf_counter()
        time.sleep(args.step_sleep)
        t0 = time.perf_counter()
        b = next(it)
        jax.block_until_ready(b)
        if k >= args.warmup:
            nxt.append(time.perf_counter() - t0)
        del b
    t_end = time.perf_counter()
    return {"part": "iter", "workers": args.workers, "prefetch": args.prefetch or 2, "batch": args.batch,
            "warmup": args.warmup, "measure": args.measure, "step_sleep_s": args.step_sleep,
            "s_per_step": (t_end - t_meas) / args.measure, "next_s": float(np.mean(nxt)),
            "startup_s": t_meas - t_start}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=("sample", "collate", "loader", "iter"))
    ap.add_argument("--yaml", default="perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml")
    ap.add_argument("--motion-root", type=pathlib.Path, default=LIB / "motion-wan-raw9-s32")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prefetch", type=int, default=0, help="0 = TorchDataLoader 默认")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--measure", type=int, default=20)
    ap.add_argument("--make-array", action="store_true", help="另测 CPU 上 make_array_from_process_local_data")
    ap.add_argument("--step-sleep", type=float, default=0.75, help="iter：每步模拟设备计算秒数")
    ap.add_argument("--direct-prefetch", type=int, default=0, help="loader：直接建 torch DataLoader 并用此 prefetch_factor")
    ap.add_argument("--no-copy", action="store_true", help="主进程不读批（只测 worker 侧供给能力）")
    ap.add_argument("--populate", action="store_true", help="主进程先 madvise(MADV_POPULATE_READ) 再整批读")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import mme_vla_suite
    import openpi
    print(f"[env] mme_vla_suite={mme_vla_suite.__file__} openpi={openpi.__file__}", flush=True)
    raw, ds, _ = build(args.yaml, args.motion_root)
    fn = {"sample": part_sample, "collate": part_collate, "loader": part_loader, "iter": part_iter}[args.part]
    res = fn(raw, ds, args)
    res["yaml"] = args.yaml
    print("BENCH_DL=" + json.dumps(res, ensure_ascii=False), flush=True)
    sys.stderr.flush()
    os._exit(0)          # persistent worker 在解释器退出时偶发挂住（实测 16 worker 打印结果后数分钟不退），测速脚本直接退出


if __name__ == "__main__":
    sys.exit(main())
