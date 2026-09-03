#!/usr/bin/env python3
"""dataloader-only 微基准（motion-memory-plan.md 第二部分 1.6）：motion 开 / 关两档 × worker 档位，b64、warmup 5、measure 40。

只走 FrameSampDataset + transform_dataset + torch DataLoader（不建模型、不 device_put），报告样本/s 与每批 pickle 载荷字节；
另做 30 秒 `multiprocessing.Pipe` pickle 往返微基准（带 / 不带四个新增键的 batch dict）。
⚠ 40 ep 库 12.9 GB 全在页缓存里，绝对值只是乐观上界，只有开 / 关差值有意义；库在本机 NVMe（`AGENTS.md` 第 13 条：不得与 turbo 数字混比）。

用法：
    MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-40ep/motion JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
      UV_LINK_MODE=copy uv run scripts/training/tests/dataloader_bench.py --out v1-store/reports/motion/dataloader_bench.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import pickle
import sys
import time
import types

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))


def _dataset(lib: pathlib.Path, yaml_name: str):
    import omegaconf
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    ns = json.load(open(_V1 / "train-assets/mme_vla_suite/robomme/norm_stats.json"))["norm_stats"]["state"]
    st = types.SimpleNamespace(q01=np.array(ns["q01"]), q99=np.array(ns["q99"]), mean=np.array(ns["mean"]), std=np.array(ns["std"]))
    dc = types.SimpleNamespace(norm_stats={"state": st}, use_quantile_norm=True)
    hc = omegaconf.OmegaConf.load(_REPO_ROOT / "src/mme_vla_suite/models/config/robomme" / yaml_name)
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")
    return _create_framesamp_dataset(str(lib / "framesamp"), dc, hc, 20)


def _collate(items):
    out = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        if all(v is None for v in vals):
            out[k] = None
        elif isinstance(vals[0], np.ndarray):
            out[k] = np.stack(vals)
        else:
            out[k] = vals
    return out


def bench_loader(lib, yaml_name, workers, prefetch, batch, warmup, measure, seed=0):
    import torch
    from torch.utils.data import DataLoader
    ds = _dataset(lib, yaml_name)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, prefetch_factor=prefetch if workers else None,
                    collate_fn=_collate, generator=g, persistent_workers=False, multiprocessing_context="spawn" if workers else None, drop_last=True)
    it = iter(dl)
    t_first = time.perf_counter()
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter(); nbytes = 0
    for _ in range(measure):
        b = next(it)
        nbytes = len(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL))
    t1 = time.perf_counter()
    del it, dl
    ds.close()
    return {"yaml": yaml_name, "workers": workers, "prefetch": prefetch, "batch": batch, "warmup": warmup, "measure": measure,
            "samples_per_s": measure * batch / (t1 - t0), "s_per_batch": (t1 - t0) / measure, "startup_s": t0 - t_first,
            "batch_pickle_bytes": nbytes}


def _pipe_child(conn, payload, seconds):
    t_end = time.time() + seconds; n = 0
    while time.time() < t_end:
        conn.send_bytes(payload); conn.recv_bytes(); n += 1
    conn.send(n)


def bench_pipe(batch_with, batch_without, seconds=30):
    res = {}
    for tag, b in (("with_motion", batch_with), ("without_motion", batch_without)):
        payload = pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL)
        ctx = mp.get_context("spawn")
        a, c = ctx.Pipe()
        p = ctx.Process(target=_pipe_child, args=(c, payload, seconds)); p.start()
        n = 0
        while True:
            try:
                data = a.recv_bytes(); a.send_bytes(data); n += 1
            except Exception:
                break
            if not p.is_alive() and not a.poll():
                break
        p.join()
        res[tag] = {"bytes": len(payload), "roundtrips_in_%ds" % seconds: n, "ms_per_roundtrip": seconds * 1000 / max(1, n),
                    "MBps_one_way": len(payload) * n / seconds / 1e6}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-40ep"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--pipe-seconds", type=int, default=30)
    args = ap.parse_args()
    lib = pathlib.Path(args.lib)
    results = {"medium": "本机 NVMe（工作副本 v1-store，40 ep 库全在页缓存）", "runs": []}
    for yaml_name in ("perceptual-framesamp-context.yaml", "perceptual-framesamp-context-motion.yaml"):
        for workers, prefetch in ((4, 6), (8, 10)):
            r = bench_loader(lib, yaml_name, workers, prefetch, args.batch, args.warmup, args.measure)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            results["runs"].append(r)
    ds = _dataset(lib, "perceptual-framesamp-context-motion.yaml")
    items = [ds[i] for i in range(args.batch)]
    with_m = _collate(items)
    without_m = {k: (None if k in ("motion_emb", "motion_pos", "motion_mask", "mem_order") else v) for k, v in with_m.items()}
    ds.close()
    results["pipe"] = bench_pipe(with_m, without_m, args.pipe_seconds)
    print(json.dumps(results["pipe"], ensure_ascii=False))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"DATALOADER_BENCH=DONE runs={len(results['runs'])} → {args.out}")


if __name__ == "__main__":
    main()
