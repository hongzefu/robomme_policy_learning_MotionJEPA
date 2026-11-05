"""StopCube 官方示教的悬停期选点与标签（计划 1105 的 A3、A4）。

公开函数只做纯计算，供回放（W2）与探针（W3）导入；CLI 读 H5 写 hover_points.json、
打印判定行并存抽查图。不启动模型或仿真，只用 CPU。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

MOVE_INTERVALS = (60, 80, 120)
HOVER_TOL = 0.001
HOVER_MARGIN = 30
POINT_STRIDE = 8
CROSSING_MARGIN = 2
PAIR_TOL = 8
PAIR_KEEP_RATIO = 0.8
ORDINALS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5}
ORDINAL_RE = re.compile(r"\b(" + "|".join(ORDINALS) + r")\b")
FIGURE_EPISODES = 10
FIGURE_PER_M = 3
FIGURE_MIN_POINTS = 4
FIGURE_MAX_POINTS = 6
FIGURE_CAPTION_HEIGHT = 36


def task_params(seed):
    """按种子重算 (M, N)，随机数消耗顺序同 StopCube._initialize_episode。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.randint(27, 33, (1,), generator=generator)
    move_interval = MOVE_INTERVALS[int(torch.randint(0, 3, (1,), generator=generator))]
    target_pass = int(torch.randint(2, 6, (1,), generator=generator))
    return move_interval, target_pass


def press_step(M, N):
    """第 N 次经过目标中心的步号 T = M*N − M//2。"""
    return M * N - M // 2


def crossings(M):
    """N 的取值上限内（第 1 到第 5 次）经过目标中心的步号。"""
    return [M * k - M // 2 for k in range(1, 6)]


def cube_position(t, M):
    """沿轨道的归一化位置：0 是起点，1 是终点，0.5 是目标中心。"""
    k = math.floor(t / M)
    a = t / M - k
    s = 3 * a**2 - 2 * a**3
    return float(s if k % 2 == 0 else 1 - s)


def point_labels(t, M, N):
    """计算一个点的全部标签。

    打乱对照会把别局的 (M, N) 配给本局的 t，所以 c 不限于前 5 次经过，remaining 可为负。
    """
    c = max(0, (t + M // 2) // M)
    return {
        "c": c,
        "press": int(c + 1 == N),
        "remaining": N - c,
        "to_press": press_step(M, N) - t,
        "cube_pos": cube_position(t, M),
    }


def hover_start(z, g, T):
    """末端高度和夹爪开度都稳定到第 T−31 步的最早一步；不存在时返回 T−30。"""
    end = T - HOVER_MARGIN
    z_part = np.asarray(z, dtype=np.float64)[:end]
    g_part = np.asarray(g, dtype=np.float64)[:end]
    settled = (np.abs(z_part - z_part[end - 1]) <= HOVER_TOL) & (np.abs(g_part - g_part[end - 1]) <= HOVER_TOL)
    start = end
    while start > 0 and settled[start - 1]:
        start -= 1
    return start


def select_points(start, M, N):
    """到位后、T−30 前能被 8 整除且不贴近任何一次经过的步。"""
    end = press_step(M, N) - HOVER_MARGIN
    passes = crossings(M)
    return [
        t for t in range(start, end)
        if t % POINT_STRIDE == 0 and all(abs(t - crossing) > CROSSING_MARGIN for crossing in passes)
    ]


def pair_t(t, M, points_set):
    """同帧异史配对：t + 2M 也是取点时返回它，否则返回 None。"""
    paired = t + 2 * M
    return paired if paired in points_set else None


def max_abs_diff(first, second):
    """两帧逐像素最大绝对差，先转 int 避免 uint8 回绕。"""
    return int(np.abs(first.astype(np.int32) - second.astype(np.int32)).max())


def ordinal_matches(goal, N):
    """指令里的序数词是否恰好一种且等于 N。"""
    found = set(ORDINAL_RE.findall(goal))
    return found == {word for word, value in ORDINALS.items() if value == N}


def read_hover_series(episode, T):
    """读前 min(L, T) 步的末端高度和夹爪开度。"""
    length = sum(key.startswith("timestep_") for key in episode)
    steps = min(length, T)
    z = np.empty(steps, dtype=np.float64)
    g = np.empty(steps, dtype=np.float64)
    for i in range(steps):
        obs = episode[f"timestep_{i}/obs"]
        z[i] = obs["eef_state_raw/pose"][2]
        g[i] = obs["gripper_state"][0]
    return z, g


def build_episode(episode, record):
    """算一局的参数、取点、标签和配对帧差。"""
    seed = record["seed"]
    M, N = task_params(seed)
    T = press_step(M, N)
    z, g = read_hover_series(episode, T)
    start = hover_start(z, g, T)
    selected = select_points(start, M, N)
    selected_set = set(selected)
    points = []
    for t in selected:
        paired = pair_t(t, M, selected_set)
        strict = front_maxabs = wrist_maxabs = None
        if paired is not None:
            obs_t = episode[f"timestep_{t}/obs"]
            obs_pair = episode[f"timestep_{paired}/obs"]
            front_maxabs = max_abs_diff(obs_t["front_rgb"][()], obs_pair["front_rgb"][()])
            wrist_maxabs = max_abs_diff(obs_t["wrist_rgb"][()], obs_pair["wrist_rgb"][()])
            strict = front_maxabs <= PAIR_TOL and wrist_maxabs <= PAIR_TOL
        points.append({"t": t, **point_labels(t, M, N), "pair_t": paired, "strict": strict,
                       "front_maxabs": front_maxabs, "wrist_maxabs": wrist_maxabs})
    goal = episode["setup/task_goal"][0]
    return {
        "episode": record["episode"],
        "seed": seed,
        "difficulty": record["difficulty"],
        "M": M,
        "N": N,
        "T": T,
        "hover_start": start,
        "prompt": goal.decode("utf-8") if isinstance(goal, bytes) else str(goal),
        "points": points,
    }


def figure_episodes(episodes):
    """按局编号选抽查局：先每种 M 取 3 局，再按编号补足 10 局。"""
    ordered = [entry for _, entry in sorted(episodes.items(), key=lambda item: int(item[0]))
               if len(entry["points"]) >= FIGURE_MIN_POINTS]
    chosen = []
    for M in MOVE_INTERVALS:
        chosen.extend([entry for entry in ordered if entry["M"] == M][:FIGURE_PER_M])
    chosen_ids = {entry["episode"] for entry in chosen}
    for entry in ordered:
        if len(chosen) >= FIGURE_EPISODES:
            break
        if entry["episode"] not in chosen_ids:
            chosen.append(entry)
            chosen_ids.add(entry["episode"])
    return sorted(chosen, key=lambda entry: entry["episode"])


def save_figure(episode, entry, path):
    """把最多 6 个等距取点的 front_rgb 拼成一张图，每格标 t、c、M、N、cube_pos。"""
    points = entry["points"]
    picks = np.unique(np.linspace(0, len(points) - 1, min(len(points), FIGURE_MAX_POINTS)).round().astype(int))
    tiles = []
    for index in picks:
        point = points[int(index)]
        frame = episode[f"timestep_{point['t']}/obs/front_rgb"][()]
        tile = Image.new("RGB", (frame.shape[1], frame.shape[0] + FIGURE_CAPTION_HEIGHT), "white")
        tile.paste(Image.fromarray(frame), (0, FIGURE_CAPTION_HEIGHT))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 3), f"ep={entry['episode']} t={point['t']} c={point['c']}", fill="black")
        draw.text((4, 19), f"M={entry['M']} N={entry['N']} cube_pos={point['cube_pos']:.3f}", fill="black")
        tiles.append(tile)
    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    width, height = tiles[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    sheet.save(path)


def write_json_atomic(payload, path):
    """先写同目录临时文件再替换，避免留下半份 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def verdict_lines(episodes, ordinal_mismatch):
    """按计划 A8 生成三条判定行。"""
    all_points = [point for entry in episodes.values() for point in entry["points"]]
    by_M = {M: sum(len(entry["points"]) for entry in episodes.values() if entry["M"] == M) for M in MOVE_INTERVALS}
    with_points = sum(bool(entry["points"]) for entry in episodes.values())
    pairs = [point for point in all_points if point["pair_t"] is not None]
    kept = sum(point["strict"] for point in pairs)
    front = max((point["front_maxabs"] for point in pairs), default=None)
    wrist = max((point["wrist_maxabs"] for point in pairs), default=None)
    pair_pass = bool(pairs) and kept / len(pairs) >= PAIR_KEEP_RATIO
    by_M_text = ",".join(f"{M}:{count}" for M, count in by_M.items())
    return [
        f"POINTS total={len(all_points)} episodes_with_points={with_points} by_M={by_M_text}",
        f"LABELS_PRECHECK={'PASS' if ordinal_mismatch == 0 else 'FAIL'} ordinal_mismatch={ordinal_mismatch} "
        f"episodes={len(episodes)}",
        f"PAIR_FRAMES={'PASS' if pair_pass else 'FAIL'} pairs={len(pairs)} kept={kept} "
        f"front_maxabs={front} wrist_maxabs={wrist} tol={PAIR_TOL}",
    ]


def run(h5_path, metadata_path, output, figures, episode_ids=None):
    """跑完一批局，写 JSON 和抽查图，返回判定行。"""
    with open(metadata_path, encoding="utf-8") as handle:
        records = {record["episode"]: record for record in json.load(handle)["records"]}
    ids = sorted(records) if episode_ids is None else sorted(episode_ids)
    episodes = {}
    ordinal_mismatch = 0
    figure_dir = Path(figures)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        for k in ids:
            entry = build_episode(h5[f"episode_{k}"], records[k])
            episodes[str(k)] = entry
            ordinal_mismatch += not ordinal_matches(entry["prompt"], entry["N"])
        for entry in figure_episodes(episodes):
            k = entry["episode"]
            save_figure(h5[f"episode_{k}"], entry, figure_dir / f"episode_{k}.png")
    write_json_atomic({"schema_version": 1, "episodes": episodes}, output)
    return verdict_lines(episodes, ordinal_mismatch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True, help="hover_points.json 的输出路径")
    parser.add_argument("--figures", required=True, help="抽查图目录")
    parser.add_argument("--episodes", type=int, nargs="+", help="只处理这些局")
    args = parser.parse_args(argv)
    for line in run(args.h5, args.metadata, args.output, args.figures, args.episodes):
        print(line, flush=True)


if __name__ == "__main__":
    main()
