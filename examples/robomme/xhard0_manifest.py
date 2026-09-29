"""官方路线 xhard0 评估清单的纯函数（不依赖仿真，供 eval.py 与 CPU 单测共用）。

清单每行一局：``{task, source_episode(官方 test 元数据 episode 下标), seed, shard}``。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NORMAL = ("success", "fail", "timeout")


def load_rows(path: str | Path, shard: int, task_order: list[str]) -> dict[str, list[dict[str, Any]]]:
    """取本片的行，按 task_order 分组、组内按 source_episode 升序；未知任务或 (task, source_episode) 重复即报错。"""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [r for r in rows if int(r["shard"]) == int(shard)]
    unknown = sorted({r["task"] for r in picked} - set(task_order))
    if unknown:
        raise ValueError(f"清单里有不认识的任务：{unknown}")
    keys = [(r["task"], int(r["source_episode"])) for r in picked]
    if len(keys) != len(set(keys)):
        raise ValueError(f"清单 shard={shard} 里 (task, source_episode) 有重复")
    out: dict[str, list[dict[str, Any]]] = {}
    for t in task_order:
        grp = sorted((r for r in picked if r["task"] == t), key=lambda r: int(r["source_episode"]))
        if grp:
            out[t] = grp
    return out


def check_seed(row: dict[str, Any], seed: Any, difficulty: Any) -> None:
    """官方 builder 解析出的 seed 必须等于清单 seed，且该局必须是官方 test 的 hard 子集。"""
    if seed is None or int(seed) != int(row["seed"]) or difficulty != "hard":
        raise AssertionError(f"身份不符 {row['task']} source_episode={row['source_episode']}：清单 seed={row['seed']}，"
                             f"builder seed={seed} difficulty={difficulty}")


def done_keys(ep_log: str | Path) -> tuple[set, list[dict[str, Any]]]:
    """续评：每个 (task, source_episode) 以最后一条记录为准，正常终态的跳过。返回 (已完成键集合, 全部记录)。"""
    p = Path(ep_log)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    last = {(r["task"], int(r["source_episode"])): r["status"] for r in rows}
    return {k for k, s in last.items() if s in NORMAL}, rows


def video_name(task: str, source_episode: int, seed: int) -> str:
    """官方路线视频文件名 ``{task}_xhard0_{source_episode}_{seed}.mp4``。"""
    return f"{task}_xhard0_{int(source_episode)}_{int(seed)}.mp4"
