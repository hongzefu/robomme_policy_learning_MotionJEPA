"""三模型汇总（1017-test-xhard5-12per 4.3 第 8 项）：合并结果按 (task, episode) 回查清单取难度，出「任务 × 档」表。

用法：
``summarize.py <清单 jsonl> --model <标签>=<run 根目录>:<policy>:<ckpt> [--model ...] [--seed 7] [--out-dir <目录>]``

每个 run 根目录下是 ``s<k>/<policy>/ckpt<ckpt>/seed<seed>/`` 的分片结果（``merge_shards.py`` 合不合并都可读）。
逐模型核：
* ``SHARDS_UNION=PASS total=<n>``：各片结果键不相交、并集 = 清单全部 (task, difficulty, episode)；
* ``SEED_MATCH=PASS``：结果键回查清单，难度一致（seed 由清单给定，客户端建环境时已逐局核 metadata seed = 清单 seed）；
* ``EVAL_VIDEO=PASS``：非 error 局各有一个视频；
* 终态计数：success / fail / timeout（视频名 ``<Task>_ep<n>_timeout_…``）/ error，以及 motion 抽稀局数（motion_stats.json）。
跨模型：``MANIFEST_SHARED=PASS``——各片 plan.json 记录的清单 sha256 全部相同且等于给定清单。
表格按官网任务顺序、难度列 easy / medium / hard / xhard12…xhard20，单元格 ``n/12（err k）``；
写 ``summary.md`` 与 ``summary.json``，末行 ``EVAL_TABLE=DONE models=<m> cells=<c>``。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

TASK_ORDER = (
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
)
DIFFICULTY_ORDER = ("easy", "medium", "hard", "xhard12", "xhard14", "xhard16", "xhard18", "xhard20")
VIDEO_RE = re.compile(r"^(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_(?P<flag>success|fail|timeout)_")


def load_manifest(path: Path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def load_model(run_root: Path, policy: str, ckpt: str, seed: str):
    results, plan_shas, videos, motion = {}, set(), {}, {}
    shard_dirs = sorted(p for p in run_root.iterdir() if p.is_dir() and re.fullmatch(r"s\d+", p.name))
    if not shard_dirs:
        raise SystemExit(f"{run_root} 下没有 s<k> 分片目录")
    for shard in shard_dirs:
        root = shard / policy / f"ckpt{ckpt}" / f"seed{seed}"
        progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        plan_shas.add(plan.get("manifest_sha256", ""))
        for group, entries in progress.items():
            task, difficulty = group.split("/")
            for ep, value in entries.items():
                key = (task, difficulty, int(ep))
                if key in results:
                    raise SystemExit(f"SHARDS_UNION=FAIL {run_root.name} 有两片都评了 {key}")
                results[key] = value
        for video in (root / "videos").glob("*.mp4"):
            m = VIDEO_RE.match(video.name)
            if m:
                videos[(m["task"], int(m["ep"]))] = m["flag"]
        stats = root / "motion_stats.json"
        if stats.is_file():
            for key, stat in json.loads(stats.read_text(encoding="utf-8")).items():
                motion[key] = stat
    return results, plan_shas, videos, motion


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--model", action="append", required=True, help="<标签>=<run 根目录>:<policy>:<ckpt>")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    rows, manifest_sha = load_manifest(Path(args.manifest))
    expected = {(r["task"], r["difficulty"], int(r["episode"])) for r in rows}
    by_episode = {(r["task"], int(r["episode"])): r["difficulty"] for r in rows}
    cells = sorted({(t, d) for t, d, _ in expected}, key=lambda c: (TASK_ORDER.index(c[0]), DIFFICULTY_ORDER.index(c[1])))
    per_cell_n = collections.Counter((t, d) for t, d, _ in expected)

    summary, all_shas = {}, set()
    for spec in args.model:
        label, rest = spec.split("=", 1)
        run_root, policy, ckpt = rest.rsplit(":", 2)
        results, plan_shas, videos, motion = load_model(Path(run_root), policy, ckpt, args.seed)
        all_shas |= plan_shas
        keys = set(results)
        if keys != expected:
            raise SystemExit(f"SHARDS_UNION=FAIL model={label} 缺 {sorted(expected - keys)[:5]} 多 {sorted(keys - expected)[:5]}")
        print(f"SHARDS_UNION=PASS model={label} total={len(keys)}", flush=True)
        for task, difficulty, ep in keys:
            if by_episode[(task, ep)] != difficulty:
                raise SystemExit(f"SEED_MATCH=FAIL model={label} {task} ep{ep}")
        print(f"SEED_MATCH=PASS model={label} rows={len(keys)}", flush=True)
        non_error = {(t, e) for (t, d, e), v in results.items() if v != "error"}
        if set(videos) != non_error:
            raise SystemExit(f"EVAL_VIDEO=FAIL model={label} 视频 {len(videos)} != 非 error 局 {len(non_error)}")
        print(f"EVAL_VIDEO=PASS model={label} videos={len(videos)}", flush=True)

        table = {}
        totals = collections.Counter()
        for task, difficulty in cells:
            c = collections.Counter()
            for (t, d, e), v in results.items():
                if (t, d) != (task, difficulty):
                    continue
                if v == "error":
                    c["error"] += 1
                elif v is True:
                    c["success"] += 1
                else:
                    c["timeout" if videos.get((t, e)) == "timeout" else "fail"] += 1
                stat = motion.get(f"{t}/{d}/{e}")
                if stat and int(stat.get("motion_downsample_steps", 0)) > 0:
                    c["downsampled"] += 1
            table[f"{task}/{difficulty}"] = dict(c)
            totals.update(c)
        summary[label] = {"run_root": run_root, "policy": policy, "ckpt": ckpt,
                          "episodes": len(results), "totals": dict(totals), "cells": table}
        print(f"EVAL_{label}=DONE episodes={len(results)} success={totals['success']} fail={totals['fail']} "
              f"timeout={totals['timeout']} error={totals['error']} downsampled={totals['downsampled']}", flush=True)

    if all_shas != {manifest_sha}:
        raise SystemExit(f"MANIFEST_SHARED=FAIL 各片清单 sha {sorted(all_shas)} != {manifest_sha}")
    print(f"MANIFEST_SHARED=PASS models={len(summary)} manifest_sha256={manifest_sha[:12]}", flush=True)

    labels = list(summary)
    lines = ["| 任务 | 档 | 局数 | " + " | ".join(labels) + " | timeout | motion 抽稀局 |",
             "|---|---|---|" + "---|" * len(labels) + "---|---|"]
    for task, difficulty in cells:
        key = f"{task}/{difficulty}"
        vals = []
        for label in labels:
            c = summary[label]["cells"][key]
            err = c.get("error", 0)
            vals.append(f"{c.get('success', 0)}/{per_cell_n[(task, difficulty)]}" + (f"（err {err}）" if err else ""))
        timeouts = " / ".join(str(summary[l]["cells"][key].get("timeout", 0)) for l in labels)
        down = " / ".join(str(summary[l]["cells"][key].get("downsampled", 0)) for l in labels)
        lines.append(f"| {task} | {difficulty} | {per_cell_n[(task, difficulty)]} | " + " | ".join(vals) + f" | {timeouts} | {down} |")
    md = "\n".join(lines) + "\n"
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(md, encoding="utf-8")
        (out / "summary.json").write_text(json.dumps({"manifest_sha256": manifest_sha, "models": summary},
                                                     indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(md)
    print(f"EVAL_TABLE=DONE models={len(summary)} cells={len(cells)}", flush=True)


if __name__ == "__main__":
    main()
