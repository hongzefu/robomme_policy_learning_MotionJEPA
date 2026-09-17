"""在各副本环境导出真实 batch 和索引摘要，再严格比较完整记录。"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


class IndexRecorder:
    """包装原 sampler，只记录原样产出的索引。"""
    def __init__(self, sampler):
        self.sampler = sampler
        self.indices = []

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for index in self.sampler:
            self.indices.append(int(index))
            yield index


def dump(args):
    import jax
    import numpy as np
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training import config as configs, dataloader
    from openpi.training import data_loader

    if args.batches < 4:
        raise ValueError("至少取 4 批，前 3 批作为 worker 预热")
    root = Path.cwd().resolve()
    if not Path(data_loader.__file__).resolve().is_relative_to(root):
        raise ValueError("导入的数据路不在当前副本")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    store = Path(os.environ.get("V1_STORE", root / "v1-store"))
    config = configs.get_config("mme_vla_suite_b128_60k")
    hc = get_history_config("perceptual-framesamp-modul-8frame-8x8.yaml")
    assets = dataclasses.replace(config.data.assets,
                                assets_dir=str(store / "train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"),
                                asset_id="robomme")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, history_config=hc),
                                 data=dataclasses.replace(config.data, assets=assets))
    data_config = config.data.create(config.assets_dirs, config.model)
    loader = dataloader.create_data_loader(
        str(store / "datasets/4task-v2-1600ep-604f16da/framesamp-8x8"), data_config,
        history_config=hc, sharding=None, shuffle=True, action_horizon=config.model.action_horizon,
        batch_size=128, num_workers=args.workers, seed=42)
    torch_loader = loader._data_loader.torch_loader
    sampler = IndexRecorder(torch_loader.batch_sampler.sampler)
    torch_loader.batch_sampler.sampler = sampler
    iterator = iter(torch_loader)
    waits = []
    try:
        with (out / "digests.jsonl").open("x") as stream:
            for step in range(args.batches):
                start = time.perf_counter()
                batch = next(iterator)
                batch = getattr(data_loader, "_from_shared_torch", lambda b: b)(batch)
                waits.append(time.perf_counter() - start)
                leaves, _ = jax.tree_util.tree_flatten_with_path(batch, is_leaf=lambda x: x is None)
                digest = {}
                nbytes = 0
                for path, leaf in leaves:
                    key = jax.tree_util.keystr(path)
                    if leaf is None:
                        digest[key] = None
                    else:
                        a = np.asarray(leaf)
                        nbytes += a.nbytes
                        digest[key] = hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()
                stream.write(json.dumps({"step": step, "keys": digest}) + "\n")
                stream.flush()
        indices = sampler.indices[:128 * args.batches]
        if len(indices) != 128 * args.batches:
            raise ValueError("已消费索引数量不足")
        (out / "indices.json").write_text(json.dumps(indices) + "\n")
        meta = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                "source": data_loader.__file__, "sys_prefix": sys.prefix,
                "batch_size": 128, "workers": args.workers, "seed": 42, "batches": args.batches,
                "batch_bytes": nbytes, "wait_seconds": waits, "warmup_batches": 3,
                "steady_wait_mean_s": statistics.fmean(waits[3:])}
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"COLLATE_DUMP=PASS batches={args.batches} batch_bytes={nbytes} "
              f"steady_wait_mean_s={meta['steady_wait_mean_s']:.6f}", flush=True)
    finally:
        if args.workers:
            iterator._shutdown_workers()


def compare(args):
    old, new = Path(args.old), Path(args.new)
    metas = [json.loads((d / "meta.json").read_text()) for d in (old, new)]
    for key in ("batch_size", "workers", "seed", "batches", "batch_bytes"):
        if metas[0][key] != metas[1][key]:
            raise ValueError(f"配置不同：{key}")
    n = metas[0]["batches"]
    records = [[json.loads(line) for line in (d / "digests.jsonl").read_text().splitlines()]
               for d in (old, new)]
    for rows in records:
        if [row["step"] for row in rows] != list(range(n)):
            raise ValueError("batch 记录不足、重复或乱序")
    mismatches = sum(a != b for a, b in zip(*records, strict=True))
    indices = [json.loads((d / "indices.json").read_text()) for d in (old, new)]
    if indices[0] != indices[1] or any(len(x) != n * 128 for x in indices):
        raise ValueError("索引序列不一致或不足")
    keys = records[0][0]["keys"]
    if any(set(row["keys"]) != set(keys) for rows in records for row in rows):
        raise ValueError("各批键集合变化")
    none = sum(value is None for value in keys.values())
    print(f"COLLATE_EQUIV={'FAIL' if mismatches else 'PASS'} batches={n} "
          f"keys={len(keys)-none} none_keys={none} mismatches={mismatches}", flush=True)
    print(f"INDEX_SEQ=PASS n={len(indices[0])}", flush=True)
    print(f"COLLATE_WAIT old={metas[0]['steady_wait_mean_s']:.6f} "
          f"new={metas[1]['steady_wait_mean_s']:.6f}", flush=True)
    return int(mismatches != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--out", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("compare")
    p.add_argument("old")
    p.add_argument("new")
    args = parser.parse_args()
    return dump(args) if args.command == "dump" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
