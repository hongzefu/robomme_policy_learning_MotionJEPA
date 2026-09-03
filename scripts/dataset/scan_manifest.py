#!/usr/bin/env python3
"""四任务 H5 的只读预扫描：算出确定性分片清单 episode_manifest.json。

**为什么必须有这一步**：`DatasetProcessor.run()` 是严格串行的，`global_episode_idx`
（决定 `features/episode_{g}/` 目录名）、`exec_sample_id`（决定 `data/{id}.pkl` 文件名）、
`total_sample_id` 三个计数器从 0 一路跨文件累加，而文件遍历用的还是非确定序的
`os.listdir`。直接并行分片必然错号覆盖。本脚本只读 H5 metadata（不碰图像），
按**规范序** sorted(*.h5) × sorted(episode_i) 把每个 episode 的三个 ID 起点算死，
使 8 个分片写出来的文件名与「串行跑一遍」逐个同构。

清单同时是全流程的**唯一真值源**：分片 worker、finalize 守卫、一致性比对工具
都从它取 episode 身份 (h5_file, raw_ep_idx) 与偏移量，不再依赖任何目录名或遍历顺序。

子命令：
  build   扫描 H5 并写出清单（含 LPT 装箱的 shard_idx）
  sample  从清单里抽 episode 子集（供本地对照产物用），输出 global_episode_idx 列表
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import h5py

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step  # noqa: E402
from mme_vla_suite.dataset_builder.robomme_h5_utils import get_episode_indices  # noqa: E402

MANIFEST_VERSION = 1


def canonical_h5_order(raw_dir: str, tasks: list[str] | None = None) -> list[str]:
    """规范序：文件名排序。刻意不用 os.listdir——它的顺序不确定，是错号的根源。

    tasks 给定时只取 ``record_dataset_<Task>.h5`` 这几个文件（缺任一即 raise），仍按文件名排序；
    v2-motionmem 起原始目录是 16 任务全集 ``/data/hongzefu/robomme_data_h5``，建库只取 4 个目标任务。
    """
    names = sorted(f for f in os.listdir(raw_dir) if f.endswith(".h5"))
    if tasks is None:
        return names
    want = [f"record_dataset_{t}.h5" for t in tasks]
    missing = [w for w in want if w not in names]
    if missing:
        raise SystemExit(f"原始目录 {raw_dir} 缺目标 h5: {missing}")
    return sorted(want)


def scan(raw_dir: str, tasks: list[str] | None = None,
         episodes_per_task: int | None = None) -> list[dict]:
    """按规范序遍历，逐 episode 记录形制并累加三个 ID 偏移。

    episodes_per_task 给定时每个 h5 只取 raw_ep_idx 升序的前 N 个（与未改动 builder 的
    ``--max_episodes N`` 同一取法，两者可按物理身份逐条对应）。
    """
    episodes: list[dict] = []
    exec_offset = 0
    total_offset = 0
    t0 = time.perf_counter()

    for h5_name in canonical_h5_order(raw_dir, tasks):
        path = os.path.join(raw_dir, h5_name)
        with h5py.File(path, "r") as data:
            indices = get_episode_indices(data)
            if episodes_per_task is not None:
                if len(indices) < episodes_per_task:
                    raise SystemExit(f"{h5_name} 只有 {len(indices)} 个 episode < --episodes-per-task {episodes_per_task}")
                indices = indices[:episodes_per_task]
            for raw_ep_idx in indices:
                ep = data[f"episode_{raw_ep_idx}"]
                num_timesteps = sum(1 for k in ep if k.startswith("timestep_"))
                # 复用 builder 自己的实现，保证与 _process_episode 口径逐字一致
                exec_start_idx = first_execution_step(ep)
                exec_samples = num_timesteps - exec_start_idx
                episodes.append(
                    {
                        "global_episode_idx": len(episodes),
                        "h5_file": h5_name,
                        "raw_ep_idx": raw_ep_idx,
                        "num_timesteps": num_timesteps,
                        "exec_start_idx": exec_start_idx,
                        "exec_samples": exec_samples,
                        "exec_sample_offset": exec_offset,
                        "total_sample_offset": total_offset,
                    }
                )
                exec_offset += exec_samples
                total_offset += num_timesteps
        print(
            f"  扫完 {h5_name}: 累计 {len(episodes)} episode, "
            f"{total_offset} timestep, {exec_offset} 执行样本 "
            f"({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    return episodes


def assign_shards_lpt(episodes: list[dict], num_shards: int) -> list[int]:
    """LPT 装箱：按 num_timesteps 降序，每次投给当前最轻的桶。

    比朴素的 round-robin 或按 episode 数均分好得多——四个任务的 episode 长度差异很大
    （实测 163–586 步），按条数分会让最重的分片明显拖尾，直接拉高 walltime 需求。
    """
    loads = [0] * num_shards
    for ep in sorted(episodes, key=lambda e: -e["num_timesteps"]):
        target = min(range(num_shards), key=lambda i: (loads[i], i))
        ep["shard_idx"] = target
        loads[target] += ep["num_timesteps"]
    return loads


def manifest_sha256(payload: dict) -> str:
    """对不含 sha 字段的规范化 JSON 取摘要，供起跑前核验清单未被改动。"""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cmd_build(args: argparse.Namespace) -> None:
    print(f"[scan] 原始目录: {args.raw_dir}")
    tasks = [t for t in args.tasks.split(",") if t] if args.tasks else None
    order = canonical_h5_order(args.raw_dir, tasks)
    print(f"[scan] 规范序: {order}" + (f"（tasks={tasks}, episodes_per_task={args.episodes_per_task}）" if tasks else ""))
    episodes = scan(args.raw_dir, tasks, args.episodes_per_task)
    loads = assign_shards_lpt(episodes, args.num_shards)

    payload = {
        "version": MANIFEST_VERSION,
        "raw_dir": os.path.abspath(args.raw_dir),
        "canonical_order": order,
        "num_shards": args.num_shards,
        "totals": {
            "episodes": len(episodes),
            "timesteps": sum(e["num_timesteps"] for e in episodes),
            "exec_samples": sum(e["exec_samples"] for e in episodes),
        },
        "shard_load_timesteps": loads,
        "episodes": episodes,
    }
    payload["sha256"] = manifest_sha256(payload)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    t = payload["totals"]
    peak = max(loads)
    spread = (peak - min(loads)) / max(1, peak)
    print(
        f"[scan] 写出 {out}\n"
        f"  episode={t['episodes']}  timestep={t['timesteps']}  执行样本={t['exec_samples']}\n"
        f"  分片负载(timestep)={loads}  极差={spread:.2%}\n"
        f"  sha256={payload['sha256']}"
    )


def load_manifest(path: str) -> dict:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    expect = manifest_sha256(payload)
    if payload.get("sha256") != expect:
        raise SystemExit(f"清单 sha256 不符（已被改动？）: {path}")
    return payload


def cmd_sample(args: argparse.Namespace) -> None:
    """抽 episode 子集。两种模式服务于一致性验证的两组对照集。

    prefix N —— 每个 H5 取前 N 个 raw_ep，等价于未改动 builder 的 --max_episodes N，
                用于第一层（分片语义无损）的逐字节对拍。
    strat  K —— 每任务分层随机 K 个 + 强制纳入边界样本（每任务最长/最短、全局最长），
                用于第二/三层跨架构对拍。刻意不取前缀，覆盖全 0–399 域。
    """
    manifest = load_manifest(args.manifest)
    episodes = manifest["episodes"]
    by_file: dict[str, list[dict]] = {}
    for ep in episodes:
        by_file.setdefault(ep["h5_file"], []).append(ep)

    picked: set[int] = set()
    if args.mode == "prefix":
        for eps in by_file.values():
            for ep in sorted(eps, key=lambda e: e["raw_ep_idx"])[: args.n]:
                picked.add(ep["global_episode_idx"])
    else:
        rng = random.Random(args.seed)
        for eps in by_file.values():
            # 边界样本强制纳入：该任务最长与最短的 episode
            picked.add(max(eps, key=lambda e: e["num_timesteps"])["global_episode_idx"])
            picked.add(min(eps, key=lambda e: e["num_timesteps"])["global_episode_idx"])
            pool = [e["global_episode_idx"] for e in eps]
            picked.update(rng.sample(pool, min(args.n, len(pool))))
        # 全局最长的那一个（大概率已被上面某任务纳入，这里兜底）
        picked.add(max(episodes, key=lambda e: e["num_timesteps"])["global_episode_idx"])

    ids = sorted(picked)
    chosen = [e for e in episodes if e["global_episode_idx"] in picked]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "n": args.n,
                "seed": args.seed if args.mode == "strat" else None,
                "manifest_sha256": manifest["sha256"],
                "global_episode_idx": ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[sample] mode={args.mode} 选中 {len(ids)} 个 episode, "
        f"合计 {sum(e['num_timesteps'] for e in chosen)} timestep -> {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="扫描 H5 写出分片清单")
    b.add_argument("--raw_dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--num_shards", type=int, default=8)
    b.add_argument("--tasks", default="", help="逗号分隔的任务名，只扫 record_dataset_<Task>.h5（默认目录内全部 h5）")
    b.add_argument("--episodes-per-task", type=int, default=None, help="每个 h5 只取 raw_ep_idx 升序前 N 个 episode")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("sample", help="从清单抽 episode 子集")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["prefix", "strat"], required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260823)
    s.set_defaults(func=cmd_sample)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
