"""v7 评估身份清单的纯函数（不依赖仿真与 robomme_hard，供 eval.py 与 CPU 单测共用）。

清单每行一局：``{task, episode(builder 下标), tier, seed, candidate|null, source_episode|null, round(1|2), shard(0..9)}``。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

XHARD0 = "xhard0"


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """读清单全部行（跳过空行）。"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_rows(rows: Iterable[dict[str, Any]], round_: int, shard: int,
                task_order: list[str] | None = None) -> list[dict[str, Any]]:
    """取 (round, shard) 的行，按任务（task_order 给了就按其顺序，否则按名字）再按 episode 升序；(task, episode) 重复即报错。"""
    picked = [r for r in rows if int(r["round"]) == int(round_) and int(r["shard"]) == int(shard)]
    order = {t: i for i, t in enumerate(task_order or [])}
    unknown = sorted({r["task"] for r in picked} - set(order)) if task_order else []
    if unknown:
        raise ValueError(f"清单里有不认识的任务：{unknown}")
    picked.sort(key=lambda r: (order.get(r["task"], 0) if task_order else 0, r["task"], int(r["episode"])))
    keys = [(r["task"], int(r["episode"])) for r in picked]
    if len(keys) != len(set(keys)):
        raise ValueError(f"清单 round={round_} shard={shard} 里 (task, episode) 有重复")
    return picked


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """保持顺序按任务分组。"""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(r)
    return out


def check_identity(row: dict[str, Any], identity: dict[str, Any]) -> None:
    """builder 解析出的身份必须与清单行的 tier、seed 一致，否则抛 AssertionError（清单与 benchmark 版本不配套）。"""
    if identity.get("tier") != row["tier"] or int(identity.get("seed")) != int(row["seed"]):
        raise AssertionError(
            f"身份不符 {row['task']} episode={row['episode']}：清单 tier={row['tier']} seed={row['seed']}，"
            f"builder tier={identity.get('tier')} seed={identity.get('seed')}")


def binding_class(tier: str) -> str:
    """xhard0 走官方原生分支导出规格（export），其余档回注规格（replay）。"""
    return "export" if tier == XHARD0 else "replay"


def video_name(task: str, tier: str, episode: int, seed: int) -> str:
    """v7 视频文件名 ``{task}_{tier}_{episode}_{seed}.mp4``。"""
    return f"{task}_{tier}_{int(episode)}_{int(seed)}.mp4"
