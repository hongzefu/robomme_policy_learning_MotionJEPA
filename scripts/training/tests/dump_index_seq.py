#!/usr/bin/env python3
"""index 序列等价对拍（v2 计划 C.1，第一块之一）。

已读码确认：同一迭代器生命周期内 torch index 序列只由
(len, seed, batch_size, drop_last, shuffle) 决定、与 num_workers 无关；跨 epoch
分叉是 torch 既有语义（1.6）。本工具用探针数据集 + 同一 `TorchDataLoader`
（openpi 实现，含默认 jax sharding 与 _collate_fn 整链）按 w 档 dump 交付序列：

- 探针数据集 `__getitem__` 返回 {"i": idx}——交付 batch 的内容即样本 index，
  无需任何 monkeypatch；
- **前置**：legacy 库 stats.json 的 execution_samples 与 packed 库 meta 的
  num_exec_samples 必须相等（R3「len 相同 → 同 workers 档位下序列逐位不变」的
  构造性前提，现场复核）；
- **约束**：dump 步数必须 < 单 epoch batch 数（防跨 epoch 既有分叉制造假阳性）。

判定行：`INDEX_SEQ_EQ=PASS steps=… batch=… seed=… workers=…`；各档序列落
`<out>/idx_seq_w{w}.json`（含 sha256），供 compare_batches.py --idx-file 复用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 只跑 loader 语义，不占 GPU

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import StoreMeta  # noqa: E402


class ProbeDataset:
    """探针数据集：只承载 len 与 idx 回显（index 序列与样本内容无关）。"""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        return {"i": np.asarray(idx, dtype=np.int64)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-root",
                    default=str(_REPO_ROOT / "v1-store/datasets/4task-gl"),
                    help="源 pkl 库根（packed 库的源数据集，读其 meta/stats.json 取样本数）")
    ap.add_argument("--packed-root",
                    default=str(_REPO_ROOT / "v1-store/datasets/4task-gl-framesamp"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", default="0,4,8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    legacy_n = json.load(open(pathlib.Path(args.legacy_root) / "meta" / "stats.json"))[
        "execution_samples"]
    packed_n = StoreMeta.load(args.packed_root).num_exec_samples
    if legacy_n != packed_n:
        raise SystemExit(f"len 不等: legacy execution_samples={legacy_n} != "
                         f"packed num_exec_samples={packed_n}——R3 前提被破坏")
    n = int(legacy_n)
    epoch_batches = n // args.batch
    if args.steps >= epoch_batches:
        raise SystemExit(f"dump 步数 {args.steps} 必须 < 单 epoch batch 数 "
                         f"{epoch_batches}（防跨 epoch 既有分叉假阳性，C.1）")

    from openpi.training.data_loader import TorchDataLoader

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workers = [int(w) for w in args.workers.split(",")]
    seqs: dict[int, list[int]] = {}
    for w in workers:
        loader = TorchDataLoader(ProbeDataset(n), local_batch_size=args.batch,
                                 shuffle=True, num_batches=args.steps,
                                 num_workers=w, seed=args.seed, framework="jax")
        seq: list[int] = []
        for batch in loader:
            seq.extend(int(x) for x in np.asarray(batch["i"]).tolist())
        del loader
        if len(seq) != args.steps * args.batch:
            raise SystemExit(f"w{w} 交付 {len(seq)} 个 index != steps×batch")
        sha = hashlib.sha256(json.dumps(seq).encode()).hexdigest()
        with (out / f"idx_seq_w{w}.json").open("w") as f:
            json.dump({"schema": 1, "workers": w, "steps": args.steps,
                       "batch": args.batch, "seed": args.seed, "len": n,
                       "indices_sha256": sha, "indices": seq}, f)
        print(f"[dump] w{w}: {len(seq)} index sha256={sha[:16]}…", flush=True)
        seqs[w] = seq

    ref_w = workers[0]
    bad = [w for w in workers[1:] if seqs[w] != seqs[ref_w]]
    if bad:
        for w in bad:
            first = next(i for i, (a, b) in enumerate(zip(seqs[ref_w], seqs[w]))
                         if a != b)
            print(f"INDEX_SEQ_EQ=FAIL w{ref_w} vs w{w} 首个分歧位置 {first}: "
                  f"{seqs[ref_w][first]} != {seqs[w][first]}")
        raise SystemExit(1)
    sha = hashlib.sha256(json.dumps(seqs[ref_w]).encode()).hexdigest()
    print(f"INDEX_SEQ_EQ=PASS steps={args.steps} batch={args.batch} "
          f"seed={args.seed} workers={args.workers} sha256={sha[:16]}…")


if __name__ == "__main__":
    main()
