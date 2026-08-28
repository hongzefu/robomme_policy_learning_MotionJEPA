#!/usr/bin/env python3
"""实验 1：dataloader-only 吞吐基准（不训练，只迭代）。

目的：在 GL A40 节点上测「NFS turbo 的供给侧」——与 train.py 完全同一条 dataloader
链路（RoboMMEDataset → transform_dataset → TorchDataLoader → DataLoaderImpl，含
HistAugObservation.from_dict 与 jax device_put），batch 64，扫 num_workers 档位，
记 样本/s 与 MB/s。这是 AGENTS.md 第 13 条要求的「NFS 数据副本上的正式吞吐基准」。

MB/s 双口径互校：
1. 公式估算：每样本字节 = 395,440（data/{idx}.pkl）+ min(step_idx+1, 32) × 602,951
   （token_emb_*.npy），episode 均长 ~302 步 → 均值 ≈ 18.7 MB/样本；
2. mountstats 实测：/proc/self/mountstats 中 turbo 挂载的 bytes 计数差值
   （per-mount 全局计数，包含 spawn worker 子进程的 IO，但也包含节点上其他进程
   ——两口径偏差大时提示串扰）。

各 workers 档位用不同 seed（42+w），避免档间样本重叠被 page cache 抬高后档数字。

输出：BENCH_RECORD_DIR 下 batches.jsonl（每批一行）与 summary.jsonl（每档一行），
stdout 打 `RESULT workers=… 样本/s=… MB/s(公式/实测)=…`。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import mme_vla_suite.training.config as _config  # noqa: E402
import mme_vla_suite.training.dataloader as _data_loader  # noqa: E402
from mme_vla_suite.datastore import framesamp_store as _fs  # noqa: E402
from mme_vla_suite.datastore import load_manifest  # noqa: E402
from mme_vla_suite.models.config.utils import get_history_config  # noqa: E402

_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"
_MOUNT_POINT = "/nfs/turbo/coe-chaijy-unreplicated"


def _resolve_backend_and_bytes(dataset_path: str, history_config_name: str):
    """S7.5：每样本读盘字节从 history_config + 清单现场推导（勿写死，D 节）。

    返回 (backend, mean_frames, avg_bytes, manifest_path, source_root)。
    legacy = pkl + mean_frames × 602,951（整包 npy）；
    packed = pkl + mean_frames × 65,536（image 行；pos/state 进程内小表不走盘）。
    """
    backend = _data_loader._resolve_backend(dataset_path)
    hc = get_history_config(history_config_name)
    max_frames = int(hc.budget) // (int(hc.token_per_image) * int(hc.num_views))
    if backend == "packed":
        meta = _fs.StoreMeta.load(dataset_path)
        manifest_path = os.environ.get("MMEVLA_FRAMESAMP_MANIFEST") or meta.manifest_path
        source_root = os.environ.get("MMEVLA_FRAMESAMP_SOURCE") or meta.source_dataset_root
        per_frame = _fs.IMAGE_ROW_BYTES
    else:
        manifest_path = os.environ.get(
            "MMEVLA_FRAMESAMP_MANIFEST",
            str(_REPO_ROOT / "v1-store" / "episode_manifest.json"))
        source_root = dataset_path
        per_frame = _fs.SOURCE_NPY_SIZE
    manifest = load_manifest(manifest_path)
    mean_frames = _fs.mean_sampled_frames(manifest, max_frames)
    avg_bytes = _fs.SOURCE_PKL_BYTES_FLOOR + mean_frames * per_frame
    return backend, mean_frames, avg_bytes, manifest_path, source_root


def _segment_probe(record_dir: pathlib.Path, backend: str, dataset_path: str,
                   source_root: str, manifest_path: str, n: int, seed: int) -> None:
    """S7.5：gather/pkl 分段计时（每样本两段耗时落 seg_timing.jsonl——
    「谁是新瓶颈」的观测资产）。

    主进程探针口径（非 worker 内路径，但同节点同存储同 API）：
    ① pkl 段 = pickle.load(data/{idx}.pkl)；
    ② gather 段 = packed 走 FrameSampStore 真实读 API（image+pos+state），
       legacy 走逐帧 np.load 整包反序列化（与 _gather_history_feat 同量）。
    """
    import pickle

    manifest = load_manifest(manifest_path)
    epis_of, step_of, row_base = _fs.build_exec_lookup(manifest)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(epis_of), size=min(n, len(epis_of)), replace=False)
    store = _fs.FrameSampStore(dataset_path) if backend == "packed" else None
    seg_path = record_dir / "seg_timing.jsonl"
    try:
        with seg_path.open("a") as f:
            for idx in picks:
                idx = int(idx)
                g, step = int(epis_of[idx]), int(step_of[idx])
                frames = list(range(step + 1)) if step < 32 else \
                    np.linspace(0, step, 32, dtype=np.int32).tolist()
                t0 = time.perf_counter()
                with open(os.path.join(source_root, "data", f"{idx}.pkl"), "rb") as pf:
                    pickle.load(pf)
                t1 = time.perf_counter()
                if store is not None:
                    rows = row_base[g] + np.asarray(frames, np.int64)
                    store.read_image_rows(rows)
                    store.pos_rows(np.asarray(frames, np.int64))
                    store.state_rows(rows)
                else:
                    for t in frames:
                        with open(os.path.join(
                                source_root, "features", f"episode_{g}",
                                f"token_emb_{t}.npy"), "rb") as nf:
                            np.load(nf, allow_pickle=True).item()
                t2 = time.perf_counter()
                f.write(json.dumps({"idx": idx, "backend": backend,
                                    "n_frames": len(frames),
                                    "pkl_ms": round((t1 - t0) * 1e3, 3),
                                    "gather_ms": round((t2 - t1) * 1e3, 3)}) + "\n")
    finally:
        if store is not None:
            store.close()
    rows = [json.loads(l) for l in seg_path.open()]
    pkl = sorted(r["pkl_ms"] for r in rows)
    gat = sorted(r["gather_ms"] for r in rows)
    med = lambda v: v[len(v) // 2]  # noqa: E731
    print(f"SEGPROBE backend={backend} n={len(rows)} "
          f"pkl_ms(med/p90)={med(pkl):.2f}/{pkl[len(pkl) * 9 // 10]:.2f} "
          f"gather_ms(med/p90)={med(gat):.2f}/{gat[len(gat) * 9 // 10]:.2f}")


def _turbo_read_bytes() -> dict[str, int]:
    """解析 /proc/self/mountstats，取 turbo 挂载的读字节计数。

    bytes: 行字段序 = normal_read normal_write direct_read direct_write
                      server_read server_write pages_read pages_write
    server_read 是真正走网络的 NFS READ 字节（page cache 命中不计入）。
    """
    in_mount = False
    for line in open("/proc/self/mountstats"):
        if line.startswith("device ") and f" mounted on {_MOUNT_POINT} " in line:
            in_mount = True
        elif line.startswith("device "):
            in_mount = False
        elif in_mount and line.strip().startswith("bytes:"):
            f = [int(x) for x in line.split()[1:]]
            return {"normal_read": f[0], "server_read": f[4]}
    raise RuntimeError(f"mountstats 中找不到挂载点 {_MOUNT_POINT}")


def main() -> None:
    record_dir = pathlib.Path(os.environ["BENCH_RECORD_DIR"])
    record_dir.mkdir(parents=True, exist_ok=True)
    workers_list = [int(w) for w in os.environ.get("WORKERS_LIST", "4 8 16").split()]
    n_warmup = int(os.environ.get("WARMUP_BATCHES", "5"))
    n_measure = int(os.environ.get("MEASURE_BATCHES", "40"))

    config = _config.cli()
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise ValueError(f"history_config 必须是 {_EXPECTED_HISTORY_CONFIG}")
    if n_measure > 200:
        raise ValueError(f"MEASURE_BATCHES {n_measure} 过大（>200），这是吞吐采样不是压测")

    data_config = config.data.create(config.assets_dirs, config.model)
    batches_path = record_dir / "batches.jsonl"
    summary_path = record_dir / "summary.jsonl"

    # S7.5：backend 分派 + 读盘字节帐现场推导（替代硬编码 _AVG_BYTES_PER_SAMPLE）
    backend, mean_frames, avg_bytes, manifest_path, source_root = \
        _resolve_backend_and_bytes(config.dataset_path, config.model.history_config)
    print(f"[bench] backend={backend} mean_frames={mean_frames:.3f} "
          f"每样本读盘均值={avg_bytes / 1e6:.2f} MB（下界口径；manifest={manifest_path}）")

    print(f"[bench] batch={config.batch_size} 档位={workers_list} "
          f"warmup={n_warmup} measure={n_measure} 数据集={config.dataset_path}")

    # S7.5：gather/pkl 分段计时探针（观测资产；SEG_PROBE_N=0 可关）
    seg_n = int(os.environ.get("SEG_PROBE_N", "200"))
    if seg_n > 0:
        _segment_probe(record_dir, backend, config.dataset_path, source_root,
                       manifest_path, seg_n, seed=int(os.environ.get("BENCH_SEED", "42")))

    for w in workers_list:
        # 各档/各 job 不同 seed，防 page cache 重叠；拆分单档 job 用 BENCH_SEED 显式指定
        seed = int(os.environ.get("BENCH_SEED", 42 + w))
        loader = _data_loader.create_data_loader(
            config.dataset_path, data_config,
            history_config=config.model.history_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            shuffle=True, num_workers=w, seed=seed,
        )
        it = iter(loader)
        for _ in range(n_warmup):           # warmup：worker 起步 + 首批预取
            obs, actions = next(it)
        jax.block_until_ready((obs, actions))   # S7.5：block 覆盖整个 pytree

        io0 = _turbo_read_bytes()
        t0 = time.time()
        prev = t0
        for i in range(n_measure):
            obs, actions = next(it)
            # S7.5：block 覆盖整个 (obs, actions) pytree——只 block actions 会低估
            # device_put 成本（memory 三键 ~236 MB/batch 在 obs 侧）
            jax.block_until_ready((obs, actions))
            now = time.time()
            with batches_path.open("a") as f:
                f.write(json.dumps({"workers": w, "batch_idx": i,
                                    "wall_time": now, "dt": now - prev}) + "\n")
            prev = now
        t1 = time.time()
        io1 = _turbo_read_bytes()

        elapsed = t1 - t0
        n_samples = n_measure * config.batch_size
        sps = n_samples / elapsed
        mbps_formula = sps * avg_bytes / 1e6
        mbps_server = (io1["server_read"] - io0["server_read"]) / elapsed / 1e6
        mbps_normal = (io1["normal_read"] - io0["normal_read"]) / elapsed / 1e6
        row = {"workers": w, "seed": seed, "batch_size": config.batch_size,
               "backend": backend, "dataset_path": str(config.dataset_path),
               "source_root": source_root, "manifest_path": manifest_path,
               "mean_frames": round(mean_frames, 3),
               "avg_bytes_per_sample": int(avg_bytes),
               "n_batches": n_measure, "elapsed_s": round(elapsed, 3),
               "samples_per_s": round(sps, 3),
               "mbps_formula": round(mbps_formula, 1),
               "mbps_mountstats_server_read": round(mbps_server, 1),
               "mbps_mountstats_normal_read": round(mbps_normal, 1),
               "s_per_batch": round(elapsed / n_measure, 3)}
        with summary_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"RESULT workers={w} 样本/s={sps:.2f} 批时={elapsed/n_measure:.2f}s/批 "
              f"MB/s(公式)={mbps_formula:.0f} MB/s(实测server_read)={mbps_server:.0f} "
              f"MB/s(实测normal_read)={mbps_normal:.0f}")
        del it, loader                       # 释放 spawn worker，进下一档

    print("DLBENCH_PASS 全部档位完成")


if __name__ == "__main__":
    main()
