"""训练侧 motion 窗口预算硬闸门：按 episode_manifest.json 逐集算可见窗数，与 motion.budget 比较。

口径以 ``mme_vla_suite.datastore.motion_store`` 为唯一真值：逐集用 ``build_index_entries`` 建段表，
再直接调用 ``max_visible_count``（末帧 t = num_timesteps-1 时的可见窗数 = demo 窗 + 执行窗），
本脚本不复写任何窗口公式。demo / 执行两部分的拆分只用于打印，取自 ``visible_motion_rows``
返回的起点帧号（< exec_start_idx 的为 demo 窗）。

只读：读清单（校验其 sha256），可选读合并 H5 目录里的 ``record_dataset_<Task>_episode_map.json``
把输出索引（merged H5 里的 raw_ep_idx）映射回源文件名；读不到映射时源文件记为 ``unknown``。

判定行（最后一行）：
    TRAIN_ES_BOUND=PASS episodes=<N> tasks=<T> max_windows=<n> budget=<B> headroom=<B-n>
    TRAIN_ES_BOUND=FAIL episodes=<N> tasks=<T> max_windows=<n> budget=<B> headroom=<B-n> over_budget=<k> worst=<task>:<输出索引>/<源文件>
headroom < 0 判 FAIL 并以 1 退出。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import load_manifest

UNKNOWN = "unknown"


def load_source_names(map_dir: pathlib.Path | None, tasks: set[str]) -> dict[tuple[str, int], str]:
    """(task, 输出索引) → 源文件名；目录或某任务的 map 缺失时该任务不入表（调用方记 unknown）。"""
    names: dict[tuple[str, int], str] = {}
    if map_dir is None or not map_dir.is_dir():
        return names
    for task in sorted(tasks):
        path = map_dir / f"record_dataset_{task}_episode_map.json"
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (task, int(row["new_idx"]))
            if key in names:
                raise ValueError(f"{path} 输出索引重复: {row['new_idx']}")
            names[key] = pathlib.PurePosixPath(str(row["member"])).name
    return names


def evaluate(manifest: dict, spec: ms.LayoutSpec) -> list[dict]:
    rows = []
    for entry in ms.build_index_entries(manifest, spec):
        if entry.num_timesteps <= entry.exec_start_idx:
            raise ValueError(f"g={entry.g} {entry.h5_file}#{entry.raw_ep_idx} 执行段为空，无法取末帧")
        windows = ms.max_visible_count(entry)
        _, frames = ms.visible_motion_rows(entry, entry.num_timesteps - 1)
        demo = int((frames < entry.exec_start_idx).sum())
        rows.append({"g": entry.g, "task": ms.task_of_h5(entry.h5_file), "output_index": entry.raw_ep_idx,
                     "demo_len": entry.exec_start_idx, "exec_len": entry.num_timesteps - entry.exec_start_idx,
                     "demo_windows": demo, "exec_windows": windows - demo, "windows": windows})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=pathlib.Path, required=True, help="episode_manifest.json")
    parser.add_argument("--budget", type=int, default=160, help="motion.budget（yaml 里的每集窗数上限）")
    parser.add_argument("--layout", choices=sorted(ms.LAYOUT_SPECS), default=ms.LAYOUT,
                        help=f"motion 布局（决定 demo 最少真实帧数），默认当前布局 {ms.LAYOUT}")
    parser.add_argument("--episode-map-dir", type=pathlib.Path, default=None,
                        help="含 record_dataset_<Task>_episode_map.json 的目录；默认取清单的 raw_dir")
    args = parser.parse_args(argv)
    if args.budget <= 0:
        parser.error("--budget 必须为正")

    manifest = load_manifest(args.manifest)
    rows = evaluate(manifest, ms.LAYOUT_SPECS[args.layout])
    if not rows:
        raise ValueError(f"清单没有 episode: {args.manifest}")
    tasks = {r["task"] for r in rows}
    map_dir = args.episode_map_dir
    if map_dir is None and manifest.get("raw_dir"):
        map_dir = pathlib.Path(manifest["raw_dir"])
    names = load_source_names(map_dir, tasks)
    for r in rows:
        r["source"] = names.get((r["task"], r["output_index"]), UNKNOWN)
    if any(r["source"] == UNKNOWN for r in rows):
        print(f"TRAIN_ES_BOUND_NOTE source_map={map_dir} 部分或全部 episode 找不到源文件映射，记为 {UNKNOWN}",
              flush=True)

    # 同窗数取 canonical 序（global_episode_idx）最靠前者，保证输出确定
    worst = max(rows, key=lambda r: (r["windows"], -r["g"]))
    per_task: dict[str, dict] = {}
    for r in rows:
        best = per_task.get(r["task"])
        if best is None or r["windows"] > best["windows"]:
            per_task[r["task"]] = r
    counts = {t: sum(1 for r in rows if r["task"] == t) for t in tasks}
    for task, r in sorted(per_task.items(), key=lambda kv: (-kv[1]["windows"], kv[0])):
        print(f"TRAIN_ES_BOUND_TASK task={task} episodes={counts[task]} max_windows={r['windows']} "
              f"worst={r['output_index']}/{r['source']} demo={r['demo_len']} exec={r['exec_len']}", flush=True)
    print(f"TRAIN_ES_BOUND_WORST task={worst['task']} output_index={worst['output_index']} source={worst['source']} "
          f"g={worst['g']} demo_len={worst['demo_len']} exec_len={worst['exec_len']} "
          f"demo_windows={worst['demo_windows']} exec_windows={worst['exec_windows']} windows={worst['windows']}",
          flush=True)

    max_windows = worst["windows"]
    headroom = args.budget - max_windows
    head = (f"episodes={len(rows)} tasks={len(tasks)} max_windows={max_windows} "
            f"budget={args.budget} headroom={headroom}")
    if headroom >= 0:
        print(f"TRAIN_ES_BOUND=PASS {head}", flush=True)
        return 0
    over = sum(1 for r in rows if r["windows"] > args.budget)
    print(f"TRAIN_ES_BOUND=FAIL {head} over_budget={over} "
          f"worst={worst['task']}:{worst['output_index']}/{worst['source']}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
