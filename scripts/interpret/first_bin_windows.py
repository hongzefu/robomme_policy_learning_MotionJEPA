"""从 bin_trace 逐步轨迹计算两个 Swap 任务第一次取 bin 的决策窗口与选对选错标签。

定义见 1105-swap-first-bin-attention-plan.html 第一部分第三、四节与第二部分 1.3：
- L：env_step_after >= swap_end 中第一个有 bin 的 z > BIN_PICKUP_Z 的行；
- C：L 之前最后一个 executed_action[7] 由 >0 转 <=0 的行，跨 chunk 也成立；
- G：C 行的 infer_seq；窗口为 [max(0, G-3) .. G]。

windows.json 里 L、C 和 earlier_closes 的步数都取该行的 env_step_after，
对应的视频帧号是 env_step_after - 1。普通计算只用标准库，只有拼图才需要 cv2。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 与官方 is_bin_pickup 同值；测试按文本读 subgoal_evaluate_func.py 核对，不 import。
BIN_PICKUP_Z = 0.15
WINDOW_INFERS = 4
# 第一目标之前要按两个按钮的任务；其余任务的 before_buttons 记 null。
BUTTON_TASKS = {"ButtonUnmaskSwap"}
VIDEO_TASKS = {"VideoUnmaskSwap"}
EPISODES_PER_SHEET = 20
SHEET_COLUMNS = 4
TILE_HEIGHT = 240
CAPTION_HEIGHT = 18


def name_run_root(run_root):
    """按 Runbook 布局 <模型>/<分片>/run 取模型名与分片名。"""
    path = Path(run_root).resolve()
    shard_dir = path.parent if path.name == "run" else path
    return shard_dir.parent.name, shard_dir.name


def load_terminals(run_roots, expect_episodes):
    """读各分片 accepted 终局并按局号去重；同局两个不同 nonce 无法取舍，直接报错。"""
    by_key = {}
    for run_root in run_roots:
        for path in sorted((Path(run_root) / "accepted").glob("*.terminal.json")):
            terminal = json.loads(path.read_text(encoding="utf-8"))
            if terminal["outcome"] == "error":
                continue
            found = by_key.get(terminal["episode_key"])
            if found is not None and found[0]["episode_nonce"] != terminal["episode_nonce"]:
                raise SystemExit(f"{terminal['episode_key']} 有两个不同 nonce 的正常终局："
                                 f"{found[0]['episode_nonce']}、{terminal['episode_nonce']}")
            by_key.setdefault(terminal["episode_key"], (terminal, Path(run_root)))
    if len(by_key) != expect_episodes:
        raise SystemExit(f"局号去重后应正好 {expect_episodes} 局，实际 {len(by_key)} 局")
    return [by_key[key] for key in sorted(by_key, key=lambda k: (k.split(":")[0], int(k.split(":")[1])))]


def trace_path(run_root, nonce):
    paths = sorted(Path(run_root).glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"nonce={nonce} 的 step_trace.jsonl 应恰有 1 份，实际 {len(paths)} 份：{paths}")
    return paths[0]


def read_steps(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("event") == "step":
                    rows.append(row)
    if not rows:
        raise SystemExit(f"{path} 没有 step 行")
    return rows


def close_indices(rows):
    """夹爪命令由开转闭的行号；只比较相邻两行，不看是否同一 chunk。"""
    return [i for i in range(1, len(rows))
            if rows[i - 1]["executed_action"][7] > 0 and rows[i]["executed_action"][7] <= 0]


def first_lift(rows, swap_end):
    """返回 (行号, bin 下标)；同一行多个 bin 过线时取最高者。终止行与普通行一样参与判断。"""
    for i, row in enumerate(rows):
        if row["env_step_after"] < swap_end:
            continue
        lifted = [b for b, position in enumerate(row["bins_after"]) if position[2] > BIN_PICKUP_Z]
        if lifted:
            return i, max(lifted, key=lambda b: row["bins_after"][b][2])
    return None, None


def nearest_bin(row):
    """该步结束时按 TCP 与 bin 的 xy 距离取最近 bin，距离不看高度。"""
    tcp = row["tcp_after"]
    distances = [(position[0] - tcp[0]) ** 2 + (position[1] - tcp[1]) ** 2 for position in row["bins_after"]]
    return distances.index(min(distances))


def analyze_episode(terminal, rows, receipt, model, shard):
    """单局标签与窗口；不补点，缺窗口的局只写原因。"""
    task = terminal["task_name"]
    if task not in BUTTON_TASKS | VIDEO_TASKS:
        raise SystemExit(f"{terminal['episode_key']} 不是计划内任务")
    swap_end = receipt["swap_end"]
    target = receipt["first_target_bin_index"]
    entry = {"episode_key": terminal["episode_key"], "episode_nonce": terminal["episode_nonce"],
             "task": task, "episode_id": terminal["episode_id"], "model": model, "shard": shard,
             "label": "no_pick", "picked_bin": None, "first_target_bin": target,
             "window_status": "no_pick", "G": None, "window": [], "C": None, "close_row_in_chunk": None,
             "L": None, "earlier_closes": [], "nearest_bin_at_C": None, "before_buttons": None,
             "outcome": terminal["outcome"], "difficulty": receipt["difficulty"],
             "swap_times": receipt["swap_times"], "pick_times": receipt["pick_times"],
             "bin_count": len(rows[0]["bins_after"])}
    lift_index, picked = first_lift(rows, swap_end)
    if lift_index is None:
        return entry
    lift_row = rows[lift_index]
    entry.update(label="correct" if picked == target else "wrong", picked_bin=picked, L=lift_row["env_step_after"])
    if task in BUTTON_TASKS:
        entry["before_buttons"] = lift_row["subgoal_index"] < 2
    closes = [i for i in close_indices(rows) if i < lift_index]
    if not closes:
        entry["window_status"] = "no_close_before_pick"
        return entry
    close_row = rows[closes[-1]]
    g = close_row["infer_seq"]
    window = list(range(max(0, g - WINDOW_INFERS + 1), g + 1))
    entry.update(window_status="complete" if len(window) == WINDOW_INFERS else "incomplete",
                 G=g, window=window, C=close_row["env_step_after"],
                 close_row_in_chunk=close_row["row_in_chunk"], nearest_bin_at_C=nearest_bin(close_row),
                 earlier_closes=[{"step": rows[i]["env_step_after"], "nearest_bin": nearest_bin(rows[i])}
                                 for i in closes[:-1] if rows[i]["env_step_after"] >= swap_end])
    return entry


def window_problem(entry, rows):
    """核对窗口：G 确实含闭合，窗口内每次推理在轨迹里都有起始行。"""
    if entry["G"] is None:
        return None
    closes = {rows[i]["infer_seq"] for i in close_indices(rows)}
    if entry["G"] not in closes:
        return f"{entry['episode_key']}:G无闭合"
    starts = {row["infer_seq"] for row in rows if row["row_in_chunk"] == 0}
    missing = [seq for seq in entry["window"] if seq not in starts]
    return f"{entry['episode_key']}:缺推理{missing}" if missing else None


def label_verdict(entries):
    """wrong_but_success 排除 before_buttons=true 的局，Video 的 null 不排除。"""
    count = {name: sum(e["label"] == name for e in entries) for name in ("correct", "wrong", "no_pick")}
    bad = [e["episode_key"] for e in entries
           if e["label"] == "wrong" and e["before_buttons"] is not True and e["outcome"] == "success"]
    geom = [e["episode_key"] for e in entries
            if e["nearest_bin_at_C"] is not None and e["nearest_bin_at_C"] != e["picked_bin"]]
    before = sum(e["before_buttons"] is True for e in entries)
    verdict = "FAIL" if bad else "PASS"
    line = (f"FIRST_BIN_LABEL={verdict} episodes={len(entries)} correct={count['correct']} wrong={count['wrong']} "
            f"no_pick={count['no_pick']} before_buttons={before} geom_mismatch={len(geom)} wrong_but_success={len(bad)}")
    if bad:
        line += f" wrong_but_success_episodes={','.join(bad)}"
    return not bad, line, geom


def window_verdict(entries, problems):
    count = {name: sum(e["window_status"] == name for e in entries)
             for name in ("complete", "incomplete", "no_close_before_pick")}
    line = (f"FIRST_BIN_WINDOW={'FAIL' if problems else 'PASS'} windows={count['complete'] + count['incomplete']} "
            f"complete={count['complete']} incomplete={count['incomplete']} "
            f"no_close_before_pick={count['no_close_before_pick']}")
    if problems:
        line += f" problems={','.join(problems)}"
    return not problems, line


def find_video(trace, task, episode_id):
    """视频取轨迹同一 attempt 的 client/videos，防止重试局拿到别次尝试的画面。"""
    videos = sorted((Path(trace).parents[2] / "videos").glob(f"{task}_ep{episode_id}_*.mp4"))
    if len(videos) != 1:
        raise SystemExit(f"{task}:{episode_id} 的视频应恰有 1 份，实际 {len(videos)} 份")
    return videos[0]


def read_frames(video, indices, cv2):
    """顺序解码取指定帧，避开 mp4 按帧号跳转不准；缺帧即失败。"""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"打不开视频 {video}")
    frames, wanted, position = {}, set(indices), 0
    try:
        if min(wanted) < 0:
            raise SystemExit(f"{video} 请求了负帧号 {min(wanted)}")
        while position <= max(wanted):
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"{video} 只有 {position} 帧，请求帧号 {max(wanted)}")
            if position in wanted:
                frames[position] = frame
            position += 1
    finally:
        capture.release()
    return frames


def episode_tile(entry, trace, cv2, np):
    """一局一个拼块：左 C 帧、右 L 帧；no_close_before_pick 的局左侧留黑。"""
    video = find_video(trace, entry["task"], entry["episode_id"])
    steps = {name: entry[name] for name in ("C", "L") if entry[name] is not None}
    frames = read_frames(video, [step - 1 for step in steps.values()], cv2)
    panels = {}
    for name, step in steps.items():
        frame = frames[step - 1]
        width = round(frame.shape[1] * TILE_HEIGHT / frame.shape[0])
        panels[name] = cv2.resize(frame, (width, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    left = panels.get("C", np.zeros_like(panels["L"]))
    body = np.hstack([left, panels["L"]])
    caption = np.zeros((CAPTION_HEIGHT, body.shape[1], 3), body.dtype)
    text = (f"{entry['task'][:6]}:{entry['episode_id']} {entry['label']} "
            f"C={entry['C']} L={entry['L']} near={entry['nearest_bin_at_C']} pick={entry['picked_bin']}")
    cv2.putText(caption, text, (2, CAPTION_HEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([caption, body])


def write_sheets(entries, traces, output):
    """每局 C 帧与 L 帧，每张 20 局、按任务分张；视频或 cv2 不可用时明确失败。"""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise SystemExit(f"拼图需要 cv2：{error}；确需跳过时显式传 --no-sheets") from error
    sheet_dir = Path(output) / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for task in sorted({e["task"] for e in entries}):
        picked = [e for e in entries if e["task"] == task and e["L"] is not None]
        for start in range(0, len(picked), EPISODES_PER_SHEET):
            tiles = [episode_tile(e, traces[e["episode_key"]], cv2, np) for e in picked[start:start + EPISODES_PER_SHEET]]
            blank = np.zeros_like(tiles[0])
            tiles = [t if t.shape == blank.shape else cv2.resize(t, blank.shape[1::-1], interpolation=cv2.INTER_AREA)
                     for t in tiles]
            tiles += [blank] * (-len(tiles) % SHEET_COLUMNS)
            rows = [np.hstack(tiles[i:i + SHEET_COLUMNS]) for i in range(0, len(tiles), SHEET_COLUMNS)]
            path = sheet_dir / f"{task}_{start // EPISODES_PER_SHEET:03d}.png"
            if not cv2.imwrite(str(path), np.vstack(rows)):
                raise SystemExit(f"写拼图失败：{path}")
            written += 1
    return written


def build(run_roots, expect_episodes):
    """返回 (窗口条目, 窗口问题, 每局轨迹路径)。"""
    named = {name_run_root(root)[0] for root in run_roots}
    if len(named) != 1:
        raise SystemExit(f"一次只处理一个模型，实际 {sorted(named)}")
    entries, problems, traces = [], [], {}
    for terminal, run_root in load_terminals(run_roots, expect_episodes):
        model, shard = name_run_root(run_root)
        path = trace_path(run_root, terminal["episode_nonce"])
        receipt_path = path.parent / "meta" / "hold_receipt.json"
        if not receipt_path.is_file():
            raise SystemExit(f"缺回执 {receipt_path}")
        rows = read_steps(path)
        entry = analyze_episode(terminal, rows, json.loads(receipt_path.read_text(encoding="utf-8")), model, shard)
        entries.append(entry)
        traces[entry["episode_key"]] = path
        problem = window_problem(entry, rows)
        if problem:
            problems.append(problem)
    return entries, problems, traces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True, help="分片 run 目录，可重复")
    parser.add_argument("--output", type=Path, required=True, help="输出目录，写 windows.json 与 sheets/")
    parser.add_argument("--expect-episodes", type=int, default=100, help="每模型局数；只在冒烟时覆盖")
    parser.add_argument("--no-sheets", action="store_true", help="不生成拼图")
    args = parser.parse_args(argv)
    entries, problems, traces = build(args.run_root, args.expect_episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "windows.json").write_text(
        json.dumps({"schema_version": 1, "episodes": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_sheets:
        print(f"FIRST_BIN_SHEETS=DONE sheets={write_sheets(entries, traces, args.output)}")
    label_ok, label_line, geom = label_verdict(entries)
    window_ok, window_line = window_verdict(entries, problems)
    print(f"GEOM_MISMATCH_EPISODES={','.join(geom) or 'none'}")
    print(label_line)
    print(window_line)
    return 0 if label_ok and window_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
