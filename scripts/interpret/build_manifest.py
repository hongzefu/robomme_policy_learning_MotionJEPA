#!/usr/bin/env python3
"""attention 解释实验的 episode 清单（计划第二部分 §1.5）。

episode 的 env_seed / difficulty 从官方元数据
third_party/robomme_benchmark/src/robomme/env_metadata/test/record_dataset_<task>_metadata.json 的 records 读，
再交 scripts/evaluation/manifest.py::make_manifest(..., official=False) 签封。
  --episodes task:id,task:id   只收这些局，每局单独一块（chunk_size 缺省 1，第一阶段逐局试用）
  缺省                         计划 §三 第二阶段 20 局表，按「任务 × 成功/失败」4 块 × 5 局
末行 MANIFEST=PASS episodes=<n> chunks=<n> seal=<sha>。
worktree 里 third_party 子模块是空目录时，元数据只读回退到主检出同路径。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evaluation import manifest as M  # noqa: E402

META_RELATIVE = Path("third_party/robomme_benchmark/src/robomme/env_metadata/test")
MAIN_CHECKOUT = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")

# 计划 §三 第二阶段清单（concat 模型 seed 42 官方 800 局的结局，episode_id 升序各取 5）
DEFAULT_EPISODES = (
    [("ButtonUnmaskSwap", i) for i in (4, 8, 11, 12, 17)]
    + [("ButtonUnmaskSwap", i) for i in (0, 1, 2, 3, 5)]
    + [("RouteStick", i) for i in (1, 8, 9, 10, 13)]
    + [("RouteStick", i) for i in (0, 2, 3, 4, 5)]
)


def parse_episodes(text: str) -> list[tuple[str, int]]:
    result = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        task, sep, index = item.partition(":")
        if not sep or not task or not index.isdigit():
            raise ValueError(f"--episodes 项须为 task:id，得到 {item!r}")
        if task not in M.TASKS:
            raise ValueError(f"未知任务 {task!r}")
        result.append((task, int(index)))
    if not result:
        raise ValueError("--episodes 为空")
    if len(set(result)) != len(result):
        raise ValueError("--episodes 有重复局")
    return result


def default_metadata_root() -> Path:
    local = REPO / META_RELATIVE
    if any(local.glob("record_dataset_*_metadata.json")):
        return local
    return MAIN_CHECKOUT / META_RELATIVE


def build(pairs, metadata_root: Path, chunk_size: int) -> dict:
    cache, hashes, episodes = {}, {}, []
    for task, index in pairs:
        if task not in cache:
            path = metadata_root / f"record_dataset_{task}_metadata.json"
            raw = path.read_bytes()
            hashes[str(META_RELATIVE / path.name)] = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw)
            if data.get("env_id") != task:
                raise ValueError(f"{path} 的 env_id 不是 {task}")
            cache[task] = {int(r["episode"]): r for r in data["records"]}
        record = cache[task].get(index)
        if record is None or record.get("task") != task:
            raise ValueError(f"元数据里没有 {task}:{index}")
        episodes.append({"task_name": task, "episode_id": index, "env_seed": int(record["seed"]),
                         "difficulty": str(record["difficulty"])})
    return M.make_manifest(episodes, metadata_sha256=hashes, chunk_size=chunk_size, official=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default=None, help="task:id,task:id；缺省为计划 §三 的 20 局")
    parser.add_argument("--chunk-size", type=int, default=None, help="缺省：给 --episodes 时 1，否则 5")
    parser.add_argument("--metadata-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = parse_episodes(args.episodes) if args.episodes else list(DEFAULT_EPISODES)
    chunk_size = args.chunk_size or (1 if args.episodes else 5)
    if not 1 <= chunk_size <= 5:
        raise SystemExit("--chunk-size 须在 1～5（eval.py 每块 1～5 局）")
    manifest = build(pairs, args.metadata_root or default_metadata_root(), chunk_size)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.out.exists() and args.out.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{args.out} 已存在且内容不同，拒绝覆盖")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"MANIFEST=PASS episodes={len(manifest['episodes'])} chunks={len(manifest['chunks'])} seal={manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
