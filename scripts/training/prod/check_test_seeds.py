#!/usr/bin/env python3
"""test split seed 核对（用户要求「检测是不是 test seed」；留档 docs/training-doc/eval-awsprod40k-b128-motion/）。

事实链：examples/robomme/env_runner.py 硬编码 BenchmarkEnvBuilder(dataset="test") →
episode_config_resolver.py 读 env_metadata/test/record_dataset_<task>_metadata.json 的 records[].{episode,seed,difficulty}，
每集按该 seed 建环境（resolve_episode）；训练数据 = 原始 H5 episode_*/setup/seed（4task-motion-400ep 库由这 4 个 H5 建成）。
判定行：
  <SPLIT>_SEED_DISJOINT=PASS|FAIL <split>=<n>x<任务数> train_h5=<n>x<任务数> overlap=<k>   被核 split 的 seed 与训练 H5 seed 的交集（须 0）且集号 0..n-1 连续
                                                                                    （--split test 默认，输出与历史逐字节一致；--split val 核 env_metadata/val）
  TRAIN_H5_IN_TRAIN_SPLIT=PASS|FAIL missing=<k>                                    训练 H5 seed 是否全部落在 env_metadata/train 内
用法：UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/check_test_seeds.py [--split test|val] --out <records>/test_seeds.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TASKS = "ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap"


def load_split(meta_dir: pathlib.Path, split: str, task: str) -> list[dict]:
    payload = json.loads((meta_dir / split / f"record_dataset_{task}_metadata.json").read_text(encoding="utf-8"))
    return [{"episode": int(r["episode"]), "seed": int(r["seed"]), "difficulty": r.get("difficulty")}
            for r in payload["records"] if r.get("seed") is not None and r.get("episode") is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", default=str(_REPO_ROOT / "third_party/robomme_benchmark/src/robomme/env_metadata"))
    ap.add_argument("--split", choices=("val", "test"), default="test",
                    help="被核对的评估 split（判定行 tag 随之变为 TEST_/VAL_SEED_DISJOINT）")
    ap.add_argument("--h5-dir", default="/scratch/hongze/robomme_data_h5")
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--expect-test", type=int, default=50, help="每任务 test 集数（eval.py 现行 50）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta, h5dir = pathlib.Path(args.metadata_dir), pathlib.Path(args.h5_dir)
    tasks = args.tasks.split(",")
    rep: dict = {}
    overlap_total = missing_total = 0
    shape_ok_all = True
    for task in tasks:
        test = load_split(meta, args.split, task)
        train_split = {r["seed"] for r in load_split(meta, "train", task)}
        with h5py.File(h5dir / f"record_dataset_{task}.h5", "r") as f:
            h5_seeds = {k: int(f[k]["setup"]["seed"][()]) for k in f.keys() if k.startswith("episode_")}
        test_seeds = {r["seed"] for r in test}
        eps = sorted(r["episode"] for r in test)
        shape_ok = len(test) == args.expect_test and eps == list(range(args.expect_test))
        shape_ok_all &= shape_ok
        overlap = sorted(test_seeds & set(h5_seeds.values()))
        missing = sorted(set(h5_seeds.values()) - train_split)
        overlap_total += len(overlap)
        missing_total += len(missing)
        diff_counts = {d: sum(1 for r in test if r["difficulty"] == d) for d in sorted({str(r["difficulty"]) for r in test})}
        rep[task] = {"test": sorted(test, key=lambda r: r["episode"]), "test_count": len(test), "test_episodes_contiguous": shape_ok,
                     "test_seed_range": [min(test_seeds), max(test_seeds)], "difficulty_counts": diff_counts,
                     "train_h5_episodes": len(h5_seeds), "train_h5_seed_range": [min(h5_seeds.values()), max(h5_seeds.values())],
                     "train_split_count": len(train_split), "overlap_test_vs_train_h5": overlap,
                     "train_h5_missing_in_train_split": missing}
        print(f"  {task}: test={len(test)} seed[{min(test_seeds)},{max(test_seeds)}] contiguous={shape_ok} difficulty={diff_counts}"
              f" | train_h5={len(h5_seeds)} seed[{min(h5_seeds.values())},{max(h5_seeds.values())}] train_split={len(train_split)}"
              f" | overlap={len(overlap)} missing_in_train_split={len(missing)}")
    n_test = "/".join(map(str, sorted({v["test_count"] for v in rep.values()})))
    n_h5 = "/".join(map(str, sorted({v["train_h5_episodes"] for v in rep.values()})))
    # tag 随 split 变：--split test 时逐字节维持历史输出 TEST_SEED_DISJOINT=… test=50x4 …
    tag, key = args.split.upper(), args.split
    l1 = (f"{tag}_SEED_DISJOINT={'PASS' if overlap_total == 0 and shape_ok_all else 'FAIL'} {key}={n_test}x{len(tasks)}"
          f" train_h5={n_h5}x{len(tasks)} overlap={overlap_total} {key}_episodes_contiguous={shape_ok_all}")
    l2 = f"TRAIN_H5_IN_TRAIN_SPLIT={'PASS' if missing_total == 0 else 'FAIL'} missing={missing_total}"
    print(l1)
    print(l2)
    rep["_lines"] = [l1, l2]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0 if (overlap_total == 0 and shape_ok_all and missing_total == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
