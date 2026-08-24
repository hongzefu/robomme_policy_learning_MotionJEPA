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

import mme_vla_suite.training.config as _config  # noqa: E402
import mme_vla_suite.training.dataloader as _data_loader  # noqa: E402

_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"
_AVG_BYTES_PER_SAMPLE = 395_440 + 30.4 * 602_951   # min(step+1,32) 在 ~302 步 episode 上的均值 ≈ 30.4
_MOUNT_POINT = "/nfs/turbo/coe-chaijy-unreplicated"


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

    print(f"[bench] batch={config.batch_size} 档位={workers_list} "
          f"warmup={n_warmup} measure={n_measure} 数据集={config.dataset_path}")

    for w in workers_list:
        seed = 42 + w                       # 各档不同 seed，防 page cache 档间重叠
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
        jax.block_until_ready(actions)

        io0 = _turbo_read_bytes()
        t0 = time.time()
        prev = t0
        for i in range(n_measure):
            obs, actions = next(it)
            jax.block_until_ready(actions)  # 含 device_put，完整对齐训练侧消费口径
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
        mbps_formula = sps * _AVG_BYTES_PER_SAMPLE / 1e6
        mbps_server = (io1["server_read"] - io0["server_read"]) / elapsed / 1e6
        mbps_normal = (io1["normal_read"] - io0["normal_read"]) / elapsed / 1e6
        row = {"workers": w, "seed": seed, "batch_size": config.batch_size,
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
