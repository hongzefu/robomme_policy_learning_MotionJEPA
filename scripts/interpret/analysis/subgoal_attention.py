"""PatternLock、RouteStick 按子目标的真实 rollout 注意力分析（计划 1106-patternlock-routestick-subgoal-attention-plan.html）。

入口：python -m scripts.interpret.analysis.subgoal_attention --root <产物根> --out <输出目录>
只用 CPU，读 <root>/<模型>/<片>/run 与 <root>/<模型>/offline，输出 summary.json、episodes/*.json 与 4 张图。
判定行依次为 ROLLOUT_RUN、SUBGOAL_LABEL、SUBGOAL_ALIGN；合成注意力不过或标签不过就不算真实读数。

口径（计划 K2、K7、§三.2、§三.3）：
- 标签只用 current_task_index：j = 动作块第 0 行 task_index_before − (m+2)；块内各行 task_index_before 有变化则 crosses_subgoal=1。
- 全局帧号 t 与 exec_start_idx 同一坐标，t<exec_start_idx 为示范帧，段号 demo_task_index[t]；不用环境步。
- motion 窗 t 为示范窗当且仅当 t<exec_start_idx，内容范围 [t, min(t+32, exec_start_idx))。
  「有重叠就算」（overlap）：窗计入所有重叠段，质量和 token 都整窗计入每段，share、uniform 不归一化，和可大于 1。
  比例计权（proportional）：按重叠帧数比例分配，各段和为 1，只作复核。帧流（single）每帧属一段，各段和为 1。
- 均匀水平按该次推理实际 token 数逐次算；层头平均后为主读数，A4 保留层×头。
- 质心与进度都按子目标归一化：质心 = Σ share(j′)·(j′−1)/(m−1) ÷ Σ share，进度 = (j−1)/(m−1)，见计划图 2「质心随子目标进度的斜率」。

三档判定（§三.3）只比较区间下界：下界 ≥ 门槛为「明显」，高于零假设（D>1、斜率>0）但未到门槛为「弱」，否则「未见」，
方向另写。自写，不用 first_bin_attention._judge_interval 的双侧语义。

bootstrap_diff 按 first_bin_attention.bootstrap_diff 同算法另写一份：那个模块导入时会拉起 matplotlib 与多份分析模块，
这里算读数不需要；测试钉住两份逐位相同。图都是一维统计或逐（层,头）、（j,j′）矩阵，没有 patch 级放大，不涉及第 5 条。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle_press_prob import bootstrap  # noqa: E402

TASKS = ("PatternLock", "RouteStick")
MODELS = ("ctx", "mot")
READERS = ("action", "text")
READER_ZH = {"action": "动作", "text": "文字"}
MODEL_ZH = {"ctx": "基线", "mot": "motion"}
STREAM_ZH = {"frame": "帧流", "motion": "motion 流"}
STREAMS = {"ctx": ("frame",), "mot": ("frame", "motion")}
WEIGHTINGS = {"frame": ("single",), "motion": ("overlap", "proportional")}
MAIN_WEIGHTING = {"frame": "single", "motion": "overlap"}
BLOCKS = ("当前视觉", "文字", "动作 token", "示范帧", "执行帧", "示范 motion", "执行 motion")
FIVE_BLOCKS = ("motion", "历史视觉", "当前视觉", "文字", "其余")
WINDOW_FRAMES = 32
N_LAYERS, N_HEADS = 18, 8
BOOT_REPEATS, BOOT_SEED = 2000, 42
D_NULL, D_THRESHOLD = 1.0, 1.2
SLOPE_NULL, SLOPE_THRESHOLD = 0.0, 0.2
A3_MIN_EPISODES = 10
A4_LOW_MASS = 0.01  # 与 Swap 轮的头级解释限制一致：流质量低于 1% 的头不进榜单
MAX_INVALID_PER_MODEL = 5
SUM_TOL = 1e-6
LIMITATIONS = ["注意力只说明看哪里，不写因果；基线和 motion 的训练来源不同，只有 seed 42",
               "对角富集是总量读数，分不开读方向信息与靠时间码回读",
               "失败局走不到后面的子目标，大 j 样本主要来自成功局，见 by_j",
               "主读数 12 组 × 2 个读数，比例计权另加 4 组只作复核，未校正多重比较",
               "A4 是事后挑选、未校正，只作描述"]


# ---------------------------------------------------------------- 读产物

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _one(paths, what):
    paths = sorted(paths)
    if len(paths) != 1:
        raise SystemExit(f"{what} 应恰有 1 份，实际 {len(paths)} 份：{[str(p) for p in paths]}")
    return paths[0]


def discover_shards(model_dir, explicit=None):
    """按目录发现片。片名 s<数字>，数字是它覆盖的 0～3 分片号；须不重不漏，缺片报错。explicit 只用于冒烟，明写才放行。"""
    model_dir = Path(model_dir)
    if explicit:
        names = list(explicit)
    else:
        names = sorted(p.name for p in model_dir.glob("s*") if p.is_dir() and re.fullmatch(r"s\d+", p.name))
        covered = [int(c) for name in names for c in name[1:]]
        if sorted(covered) != [0, 1, 2, 3]:
            raise SystemExit(f"{model_dir} 的分片目录不全或重复：{names}，须恰好覆盖 0～3（如 s0～s3 或 s01、s23）")
    runs = []
    for name in names:
        run = model_dir / name / "run"
        if not run.is_dir():
            raise SystemExit(f"缺分片运行目录：{run}")
        runs.append(run)
    return runs


def rollout_counts(run_dirs):
    """有效局、无效局与基础设施重试。局的需求集合来自 commands.json 与 started；无效局 = 有需求而无 accepted 终局。"""
    accepted, demand, starts = {}, set(), Counter()
    for run in run_dirs:
        for path in sorted((Path(run) / "accepted").glob("*.terminal.json")):
            record = _read_json(path)
            key = record["episode_key"]
            if key in accepted:
                raise SystemExit(f"{key} 有多份 accepted 终局，不知该用哪份：{accepted[key][1]}、{path}")
            accepted[key] = (record, path, Path(run))
        for path in Path(run).glob("attempts/*/commands.json"):
            demand.update(_read_json(path).get("episode_keys", []))
        for path in Path(run).glob("attempts/*/client/*.started.json"):
            key = _read_json(path)["episode_key"]
            demand.add(key)
            starts[key] += 1
    demand.update(accepted)
    return {"accepted": accepted, "invalid": sorted(demand - set(accepted)),
            "infra_retries": sum(max(0, count - 1) for count in starts.values())}


def load_inference(path):
    with np.load(path, allow_pickle=False) as archive:
        return {"path": str(path), "names": [str(n) for n in archive["group_names"]],
                "tokens": np.asarray(archive["group_tokens_all"], np.int64),
                "mass_action": np.asarray(archive["group_mass_action"], np.float64),
                "mass_text": np.asarray(archive["group_mass_text"], np.float64),
                "episode_key": str(archive["episode_key"]), "infer_seq": int(archive["infer_seq"]),
                "chunk_start_env_step": int(archive["chunk_start_env_step"])}


def load_episode(run_dir, offline_root, record):
    """读一局的回执、轨迹和全部离线记录。缺失或互相对不上一律报错，不跳过。"""
    run_dir, nonce, key = Path(run_dir), record["episode_nonce"], record["episode_key"]
    trace = _one(run_dir.glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"), f"{key} 的 step_trace.jsonl")
    receipt = _read_json(trace.parent / "meta" / "hold_receipt.json")
    with trace.open(encoding="utf-8") as stream:
        rows = [row for row in map(json.loads, stream) if row.get("event") == "step"]
    if not rows:
        raise SystemExit(f"{trace} 没有 step 行")
    rows.sort(key=lambda row: row["action_index"])
    offline_dir = Path(offline_root) / nonce
    files = sorted(offline_dir.glob("offline_*.npz"))
    if not files:
        raise SystemExit(f"{key} 缺离线记录：{offline_dir}")
    inferences = [load_inference(path) for path in files]
    expected = sorted({row["infer_seq"] for row in rows})
    found = sorted(inf["infer_seq"] for inf in inferences)
    if expected != found:
        raise SystemExit(f"{key} 离线记录与轨迹动作块对不上：轨迹 {expected}，离线 {found}")
    first_rows = {row["infer_seq"]: row for row in rows if row["row_in_chunk"] == 0}
    for inf in inferences:
        if inf["episode_key"] != key:
            raise SystemExit(f"{inf['path']} 的 episode_key={inf['episode_key']} 不是 {key}")
        row = first_rows.get(inf["infer_seq"])
        if row is not None and row["chunk_start_env_step"] != inf["chunk_start_env_step"]:
            raise SystemExit(f"{inf['path']} 的 chunk_start_env_step 与轨迹不符")
    outcome = record.get("outcome")
    return {"episode_key": key, "nonce": nonce, "task": receipt["task_name"], "episode_id": int(receipt["episode_id"]),
            "difficulty": str(receipt.get("difficulty")), "m": int(receipt["m"]),
            "exec_start_idx": int(receipt["exec_start_idx"]), "demo_task_index": [int(v) for v in receipt["demo_task_index"]],
            "outcome": outcome, "success": outcome == "success", "receipt_outcome": receipt.get("outcome"),
            "rows": rows, "inferences": sorted(inferences, key=lambda inf: inf["infer_seq"])}


# ---------------------------------------------------------------- 标签

def label_inferences(rows, m):
    """{infer_seq: {j, crosses_subgoal}}；局末 task_index_after=2m+3 不看，只看 task_index_before。缺第 0 行报错。"""
    chunks = {}
    for row in rows:
        chunks.setdefault(row["infer_seq"], []).append(row)
    labels = {}
    for seq, chunk in chunks.items():
        first = [row for row in chunk if row["row_in_chunk"] == 0]
        if len(first) != 1:
            raise ValueError(f"infer_seq={seq} 的第 0 行应恰有 1 条，实际 {len(first)} 条")
        values = {row["task_index_before"] for row in chunk}
        labels[seq] = {"j": int(first[0]["task_index_before"]) - (m + 2), "crosses_subgoal": int(len(values) > 1)}
    return labels


def check_labels(ep):
    """SUBGOAL_LABEL 的逐局核对，返回不符说明的列表。"""
    errors = []
    m, demo = ep["m"], ep["demo_task_index"]
    if m < 1:
        errors.append(f"m={m} 不合法")
    if len(demo) != ep["exec_start_idx"] + 1:
        errors.append(f"demo_task_index 长度 {len(demo)} 不等于 exec_start_idx+1={ep['exec_start_idx'] + 1}")
    if not demo or demo[-1] != m + 3:
        errors.append(f"示范末帧应为 m+3={m + 3}，实际 {demo[-1] if demo else None}")
    runs = [v for i, v in enumerate(demo[:-1]) if i == 0 or demo[i - 1] != v]
    if runs != list(range(1, m + 1)):
        errors.append(f"示范序号应依次为 1..{m}，实际 {runs}")
    previous = None
    last = len(ep["rows"]) - 1
    for i, row in enumerate(ep["rows"]):
        before, after = row["task_index_before"], row["task_index_after"]
        if not m + 3 <= before <= 2 * m + 2:
            errors.append(f"action_index={row['action_index']} 的 task_index_before={before} 超出 {m + 3}～{2 * m + 2}")
        if previous is not None and before < previous:
            errors.append(f"action_index={row['action_index']} 的 task_index_before 回退：{previous}→{before}")
        previous = before
        top = 2 * m + 3 if i == last else 2 * m + 2
        if not m + 3 <= after <= top:
            errors.append(f"action_index={row['action_index']} 的 task_index_after={after} 超出 {m + 3}～{top}")
    try:
        label_inferences(ep["rows"], m)
    except ValueError as error:
        errors.append(str(error))
    if ep["receipt_outcome"] != ep["outcome"]:
        errors.append(f"回执 outcome={ep['receipt_outcome']} 与终局 outcome={ep['outcome']} 不符")
    return errors


def segment_of_frame(t, demo_task_index, exec_start_idx):
    """帧 t 的示范段号；执行帧返回 None。"""
    if t >= exec_start_idx:
        return None
    if t < 0 or t >= len(demo_task_index):
        raise ValueError(f"帧 {t} 不在 demo_task_index 范围内")
    return int(demo_task_index[t])


def segments_of_window(t, demo_task_index, exec_start_idx, window=WINDOW_FRAMES):
    """motion 窗 t 与各示范段的重叠帧数 {段号: 帧数}；执行窗返回空字典。"""
    if t >= exec_start_idx:
        return {}
    if t < 0 or min(t + window, exec_start_idx) > len(demo_task_index):
        raise ValueError(f"motion 窗 {t} 不在 demo_task_index 范围内")
    return dict(Counter(int(v) for v in demo_task_index[t:min(t + window, exec_start_idx)]))


# ---------------------------------------------------------------- 单次推理的读数

def split_groups(names):
    frames, windows, blocks = {}, {}, {}
    for index, name in enumerate(names):
        if name.startswith("真实帧:"):
            frames[int(name.split(":")[1])] = index
        elif name.startswith("motion窗:"):
            windows[int(name.split(":")[1])] = index
        elif name in FIVE_BLOCKS:
            blocks[name] = index
    return frames, windows, blocks


def stream_shares(inference, demo_task_index, exec_start_idx, m, reader, stream, weighting=None):
    """一次推理、一个读者、一条流的 share(j′) 与 uniform(j′)。该流没有示范 token 或质量为 0 时返回 None。

    share_lh 形状 (层,头,m)，stream_mass_lh 是逐头的示范流质量；质量为 0 的头 share 为 NaN。
    """
    weighting = weighting or MAIN_WEIGHTING[stream]
    frames, windows, _ = split_groups(inference["names"])
    rows, indices = [], []
    if stream == "frame":
        items = frames
    elif stream == "motion":
        items = windows
    else:
        raise ValueError(f"未知流 {stream}")
    for t, index in sorted(items.items()):
        weight = np.zeros(m)
        if stream == "frame":
            segment = segment_of_frame(t, demo_task_index, exec_start_idx)
            if segment is None:
                continue
            overlaps = {segment: 1}
        else:
            overlaps = segments_of_window(t, demo_task_index, exec_start_idx)
            if not overlaps:
                continue
        total = sum(overlaps.values())
        for segment, frames_in in overlaps.items():
            if not 1 <= segment <= m:
                raise ValueError(f"示范段号 {segment} 超出 1..{m}")
            weight[segment - 1] = frames_in / total if weighting == "proportional" else 1.0
        rows.append(weight)
        indices.append(index)
    if not rows:
        return None
    weights = np.stack(rows)
    tokens = inference["tokens"][indices].astype(float)
    if tokens.sum() <= 0:
        return None
    mass = inference[f"mass_{reader}"][indices]
    mean_mass = mass.mean((1, 2))
    denominator = mean_mass.sum()
    if denominator <= 0:
        return None
    stream_mass = mass.sum(0)
    segment_mass = np.einsum("ns,nlh->lhs", weights, mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        share_lh = segment_mass / stream_mass[..., None]
    return {"share": weights.T @ mean_mass / denominator, "uniform": weights.T @ tokens / tokens.sum(),
            "share_lh": share_lh.astype(np.float32), "stream_mass_lh": stream_mass.astype(np.float32)}


def block_shares(inference, exec_start_idx, reader):
    """A1 七块的占比与按 token 的均匀水平。文字读者看不到动作 token，该块恒 0 且不进分母。"""
    frames, windows, blocks = split_groups(inference["names"])
    mass = inference[f"mass_{reader}"].mean((1, 2))
    tokens = inference["tokens"].astype(float)
    for name in FIVE_BLOCKS[:2]:
        parts = frames if name == "历史视觉" else windows
        whole = tokens[blocks[name]] if name in blocks else 0.0
        if whole != sum(tokens[i] for i in parts.values()):
            raise ValueError(f"{inference['path']}：{name} 的 token 数与子集之和不符")

    def of(values, index):
        return float(values[index]) if index is not None else 0.0

    def split(values, parts):
        demo = sum(float(values[i]) for t, i in parts.items() if t < exec_start_idx)
        return demo, sum(float(values[i]) for t, i in parts.items() if t >= exec_start_idx)

    def row(values, include_action):
        demo_f, exec_f = split(values, frames)
        demo_m, exec_m = split(values, windows)
        action = of(values, blocks.get("其余")) if include_action else 0.0
        return np.array([of(values, blocks.get("当前视觉")), of(values, blocks.get("文字")), action, demo_f, exec_f, demo_m, exec_m])
    token_row = row(tokens, reader == "action")
    return {"share": row(mass, reader == "action"), "uniform": token_row / token_row.sum()}


def align_matrix(records, m):
    """(j,j′) 矩阵：按 j 平均每局各次推理的 share 与 uniform；没有推理的行为 NaN。"""
    share, uniform, count = np.full((m, m), np.nan), np.full((m, m), np.nan), np.zeros(m, int)
    for j in range(1, m + 1):
        rows = [r for r in records if r["j"] == j]
        count[j - 1] = len(rows)
        if rows:
            share[j - 1] = np.mean([r["share"] for r in rows], axis=0)
            uniform[j - 1] = np.mean([r["uniform"] for r in rows], axis=0)
    return {"share": share, "uniform": uniform, "count": count}


def _select(records, js):
    return records if js is None else [r for r in records if r["j"] in js]


def diag_enrichment(records, js=None):
    """D = 对角 share 均值 ÷ 对角 uniform 均值。没有推理或对角 uniform 全 0 时为 None。"""
    records = _select(records, js)
    if not records:
        return None
    share = float(np.mean([r["share"][r["j"] - 1] for r in records]))
    uniform = float(np.mean([r["uniform"][r["j"] - 1] for r in records]))
    return share / uniform if uniform > 0 else None


def _centroids(records, m, key="share"):
    """进度 x 与质心 y，都归一化到 0～1。m<2 时没有进度，返回空。"""
    if m < 2:
        return np.zeros(0), np.zeros(0)
    positions = np.arange(m) / (m - 1)
    x, y = [], []
    for r in records:
        total = float(r[key].sum())
        if total > 0:
            x.append((r["j"] - 1) / (m - 1))
            y.append(float(r[key] @ positions) / total)
    return np.asarray(x), np.asarray(y)


def _ols_slope(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    dx = x - x.mean()
    return float(dx @ (y - y.mean()) / (dx @ dx))


def centroid_slope(records, m, js=None, key="share"):
    """示范注意力质心随子目标进度的斜率。局内只走过一个子目标时进度无变化，为 None。"""
    return _ols_slope(*_centroids(_select(records, js), m, key))


def slope_null(episodes, repeats=BOOT_REPEATS, seed=BOOT_SEED):
    """零假设：局内打乱推理顺序，算平均斜率的分布。episodes 是 [(records, m), ...]。"""
    generator = np.random.default_rng(seed)
    samples = []
    for records, m in episodes:
        x, y = _centroids(records, m)
        if len(x) < 2 or np.ptp(x) == 0:
            continue
        dx = x - x.mean()
        shuffled = generator.permuted(np.tile(y, (repeats, 1)), axis=1)
        samples.append((shuffled - y.mean()) @ dx / (dx @ dx))
    if not samples:
        return {"n_episodes": 0, "mean": None, "ci95": None}
    means = np.mean(samples, axis=0)
    return {"n_episodes": len(samples), "mean": float(means.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def bootstrap_diff(a, b, repeats=BOOT_REPEATS, seed=BOOT_SEED):
    """a 组均值减 b 组均值，各自按局重采样；与 first_bin_attention.bootstrap_diff 同算法（原因见文首）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return {"n_a": len(a), "n_b": len(b), "mean": None, "ci95": None}
    generator = np.random.default_rng(seed)
    samples = (generator.choice(a, size=(repeats, len(a)), replace=True).mean(axis=1)
               - generator.choice(b, size=(repeats, len(b)), replace=True).mean(axis=1))
    return {"n_a": len(a), "n_b": len(b), "mean": float(a.mean() - b.mean()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def judge_tier(d_ci, slope_ci):
    """§三.3 三档：只看区间下界。ci 为 None 表示该读数不可算。"""
    d_low = d_ci[0] if d_ci else None
    slope_low = slope_ci[0] if slope_ci else None
    if (d_low is not None and d_low >= D_THRESHOLD) or (slope_low is not None and slope_low >= SLOPE_THRESHOLD):
        return "明显"
    if (d_low is not None and d_low > D_NULL) or (slope_low is not None and slope_low > SLOPE_NULL):
        return "弱"
    return "未见"


# ---------------------------------------------------------------- 逐局

def analyse_episode(ep, model):
    """把一局的原始分组质量折成逐次读数，并放掉原始数组。"""
    labels = label_inferences(ep["rows"], ep["m"])
    m, demo, exec_start = ep["m"], ep["demo_task_index"], ep["exec_start_idx"]
    blocks = {reader: [] for reader in READERS}
    records = {(reader, stream, weighting): [] for reader in READERS for stream in STREAMS[model]
               for weighting in WEIGHTINGS[stream]}
    uniform_sum_err = 0.0
    for inference in ep["inferences"]:
        label = labels[inference["infer_seq"]]
        for reader in READERS:
            blocks[reader].append({"j": label["j"], "crosses_subgoal": label["crosses_subgoal"],
                                   **block_shares(inference, exec_start, reader)})
        for (reader, stream, weighting), rows in records.items():
            shares = stream_shares(inference, demo, exec_start, m, reader, stream, weighting)
            if shares is None:
                continue
            rows.append({"infer_seq": inference["infer_seq"], "j": label["j"], **shares})
            if weighting != "overlap":
                uniform_sum_err = max(uniform_sum_err, abs(float(shares["uniform"].sum()) - 1.0))
    out = {k: ep[k] for k in ("episode_key", "nonce", "task", "episode_id", "difficulty", "m", "exec_start_idx", "outcome", "success")}
    out.update(model=model, blocks=blocks, records=records, uniform_sum_err=uniform_sum_err,
               inferences=[{"infer_seq": seq, **labels[seq]} for seq in sorted(labels)])
    return out


def episode_report(ep):
    """逐局 JSON：标签与每局的 D、斜率，不含逐头矩阵。"""
    readouts = {}
    for (reader, stream, weighting), rows in ep["records"].items():
        readouts[f"{reader}/{stream}/{weighting}"] = {"inferences": len(rows), "D": diag_enrichment(rows),
                                                      "slope": centroid_slope(rows, ep["m"]),
                                                      "uniform_slope": centroid_slope(rows, ep["m"], key="uniform")}
    return _clean({**{k: v for k, v in ep.items() if k not in ("blocks", "records")}, "readouts": readouts})


# ---------------------------------------------------------------- 汇总

def _mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def _boot(values, repeats, seed):
    return bootstrap([v for v in values if v is not None], repeats, seed)


def a1_block(episodes, repeats, seed):
    """七块占比按 j 分组，分任务、模型、读者、难度、成败；每局每 j 先平均，局级重采样。"""
    out = {}
    for ep in episodes:
        cell = out.setdefault(ep["task"], {}).setdefault(ep["model"], {})
        for reader in READERS:
            for difficulty in ("all", ep["difficulty"]):
                for result in ("all", "success" if ep["success"] else "fail"):
                    cell.setdefault(reader, {}).setdefault(difficulty, {}).setdefault(result, []).append(ep)
    for task, by_model in out.items():
        for model, by_reader in by_model.items():
            for reader, by_difficulty in by_reader.items():
                for difficulty, by_result in by_difficulty.items():
                    for result, eps in by_result.items():
                        by_result[result] = _a1_cell(eps, reader, repeats, seed)
    return out


def _a1_cell(episodes, reader, repeats, seed):
    per_j = {}
    for ep in episodes:
        for j in sorted({r["j"] for r in ep["blocks"][reader]}):
            rows = [r for r in ep["blocks"][reader] if r["j"] == j]
            crossing = [r for r in rows if r["crosses_subgoal"]]
            entry = per_j.setdefault(j, {"share": [], "uniform": [], "cross_share": [], "cross_uniform": [], "infers": 0, "crosses": 0})
            entry["share"].append(np.mean([r["share"] for r in rows], axis=0))
            entry["uniform"].append(np.mean([r["uniform"] for r in rows], axis=0))
            entry["infers"] += len(rows)
            entry["crosses"] += len(crossing)
            if crossing:
                entry["cross_share"].append(np.mean([r["share"] for r in crossing], axis=0))
                entry["cross_uniform"].append(np.mean([r["uniform"] for r in crossing], axis=0))
    cell = {}
    for j, entry in sorted(per_j.items()):
        share, uniform = np.stack(entry["share"]), np.stack(entry["uniform"])
        blocks = {}
        for index, block in enumerate(BLOCKS):
            stat = bootstrap(share[:, index], repeats, seed)
            u = float(uniform[:, index].mean())
            blocks[block] = {"share": stat["mean"], "ci95": stat["ci95"], "uniform": u,
                             "enrichment": stat["mean"] / u if u > 0 else None}
            if entry["cross_share"]:
                cs = float(np.stack(entry["cross_share"])[:, index].mean())
                cu = float(np.stack(entry["cross_uniform"])[:, index].mean())
                blocks[block]["crosses_subgoal"] = {"share": cs, "uniform": cu, "enrichment": cs / cu if cu > 0 else None}
        cell[str(j)] = {"n_episodes": len(share), "n_inferences": entry["infers"], "n_crosses_subgoal": entry["crosses"],
                        "n_episodes_crosses_subgoal": len(entry["cross_share"]), "blocks": blocks}
    return cell


def _offsets(pairs):
    """偏移 j′−j 分布：每局按偏移先平均，再跨局平均。overlap 下各偏移之和可大于 1。"""
    per_offset = {}
    for records, m in pairs:
        seen = {}
        for r in records:
            for index in range(m):
                entry = seen.setdefault(index + 1 - r["j"], ([], []))
                entry[0].append(float(r["share"][index]))
                entry[1].append(float(r["uniform"][index]))
        for offset, (share, uniform) in seen.items():
            bucket = per_offset.setdefault(offset, ([], []))
            bucket[0].append(np.mean(share))
            bucket[1].append(np.mean(uniform))
    return {str(offset): {"share": float(np.mean(s)), "uniform": float(np.mean(u)), "n_episodes": len(s)}
            for offset, (s, u) in sorted(per_offset.items())}


def _group_matrix(pairs):
    """绝对子目标序号上的矩阵，跨局平均；不同 m 的局按相同 j、j′ 叠加，矩阵行列到最大 m。"""
    if not pairs:
        return None
    size = max(m for _, m in pairs)
    share, uniform = np.full((len(pairs), size, size), np.nan), np.full((len(pairs), size, size), np.nan)
    for i, (records, m) in enumerate(pairs):
        matrix = align_matrix(records, m)
        share[i, :m, :m], uniform[i, :m, :m] = matrix["share"], matrix["uniform"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_share = _nanmean(share)
        mean_uniform = _nanmean(uniform)
        enrichment = mean_share / mean_uniform
    return {"share": mean_share, "uniform": mean_uniform, "enrichment": enrichment,
            "n_episodes_per_cell": np.isfinite(share).sum(0)}


def _nanmean(values):
    count = np.isfinite(values).sum(0)
    total = np.nansum(values, axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def a2_cell(episodes, reader, stream, weighting, repeats, seed):
    """一个「任务 × 模型 × 读者 × 流 × 计权」的 A2 读数；只用 m≥2 的局。"""
    excluded = [ep["episode_key"] for ep in episodes if ep["m"] < 2]
    eligible = [ep for ep in episodes if ep["m"] >= 2 and ep["records"][reader, stream, weighting]]
    pairs = [(ep["records"][reader, stream, weighting], ep["m"]) for ep in eligible]
    d_values = [diag_enrichment(records) for records, _ in pairs]
    slopes = [centroid_slope(records, m) for records, m in pairs]
    uniform_slopes = [centroid_slope(records, m, key="uniform") for records, m in pairs]
    d_stat, slope_stat = _boot(d_values, repeats, seed), _boot(slopes, repeats, seed)
    by_j = {}
    for j in sorted({r["j"] for records, _ in pairs for r in records}):
        by_j[str(j)] = _boot([diag_enrichment(records, {j}) for records, _ in pairs], repeats, seed)
    return {"n_episodes": len(eligible), "excluded_m1": excluded, "D": d_stat, "slope": slope_stat,
            "slope_null_shuffle": slope_null(pairs, repeats, seed),
            "uniform_centroid_slope_mean": _mean_or_none([v for v in uniform_slopes if v is not None]),
            "slope_undefined_episodes": sum(v is None for v in slopes), "by_j": by_j, "offsets": _offsets(pairs),
            "matrix": _group_matrix(pairs), "tier": judge_tier(d_stat["ci95"], slope_stat["ci95"]),
            "direction": {"D": None if d_stat["mean"] is None else int(np.sign(d_stat["mean"] - D_NULL)),
                          "slope": None if slope_stat["mean"] is None else int(np.sign(slope_stat["mean"] - SLOPE_NULL))}}


def a3_cell(success, fail, reader, stream, repeats, seed):
    """同模型同难度：成功局减失败局的 D 与斜率，只比两侧都有的 j；每侧不足 10 局记样本不足。"""
    weighting = MAIN_WEIGHTING[stream]
    key = (reader, stream, weighting)
    success = [ep for ep in success if ep["m"] >= 2 and ep["records"][key]]
    fail = [ep for ep in fail if ep["m"] >= 2 and ep["records"][key]]
    seen = [{r["j"] for ep in side for r in ep["records"][key]} for side in (success, fail)]
    common = sorted(seen[0] & seen[1])
    out = {"common_j": common}
    for name, reads in (("D", lambda ep: diag_enrichment(ep["records"][key], set(common))),
                        ("slope", lambda ep: centroid_slope(ep["records"][key], ep["m"], set(common)))):
        a = [v for v in map(reads, success) if v is not None]
        b = [v for v in map(reads, fail) if v is not None]
        if min(len(a), len(b)) < A3_MIN_EPISODES:
            out[name] = {"status": "样本不足", "n_success": len(a), "n_fail": len(b), "mean": None, "ci95": None}
        else:
            stat = bootstrap_diff(a, b, repeats, seed)
            out[name] = {"status": "可比", "n_success": len(a), "n_fail": len(b), "mean": stat["mean"], "ci95": stat["ci95"]}
    return out


def a4_cell(episodes, reader, stream, weighting):
    """18 层 × 8 头的对角超额：每局逐头 mean(对角 share) − mean(对角 uniform)，再跨局平均。"""
    per_episode, masses = [], []
    for ep in episodes:
        rows = ep["records"][reader, stream, weighting]
        if ep["m"] < 2 or not rows:
            continue
        diag = np.stack([r["share_lh"][:, :, r["j"] - 1] for r in rows]).astype(float)
        uniform = float(np.mean([r["uniform"][r["j"] - 1] for r in rows]))
        per_episode.append(_nanmean(diag) - uniform)
        masses.append(np.stack([r["stream_mass_lh"] for r in rows]).astype(float).mean(0))
    if not per_episode:
        return None
    excess, mass = _nanmean(np.stack(per_episode)), np.mean(masses, axis=0)
    ranked = np.where(mass >= A4_LOW_MASS, excess, np.nan)
    order = [i for i in np.argsort(-np.nan_to_num(ranked, nan=-np.inf), axis=None)[:5] if np.isfinite(ranked.flat[i])]
    return {"n_episodes": len(per_episode), "excess": excess, "stream_mass": mass,
            "top_heads": [{"head": f"L{i // N_HEADS}H{i % N_HEADS}", "excess": float(excess.flat[i])} for i in map(int, order)]}


def summarize(episodes, repeats=BOOT_REPEATS, seed=BOOT_SEED, tasks=TASKS, models=MODELS):
    """A1～A4、三档判定与共同 pattern。缺任务或缺模型的格子如实为空。"""
    a2, a3, a4, main, review = {}, {}, {}, [], []
    for task in tasks:
        for model in models:
            subset = [ep for ep in episodes if ep["task"] == task and ep["model"] == model]
            for reader in READERS:
                for stream in STREAMS[model]:
                    for weighting in WEIGHTINGS[stream]:
                        cell = a2_cell(subset, reader, stream, weighting, repeats, seed)
                        a2.setdefault(task, {}).setdefault(model, {}).setdefault(reader, {}).setdefault(stream, {})[weighting] = cell
                        row = {"task": task, "model": model, "reader": reader, "stream": stream, "weighting": weighting,
                               "n_episodes": cell["n_episodes"], "D_mean": cell["D"]["mean"], "D_ci95": cell["D"]["ci95"],
                               "slope_mean": cell["slope"]["mean"], "slope_ci95": cell["slope"]["ci95"],
                               "tier": cell["tier"], "direction": cell["direction"]}
                        (review if weighting == "proportional" else main).append(row)
                        a4.setdefault(model, {}).setdefault(reader, {}).setdefault(stream, {}).setdefault(weighting, {})[task] = \
                            a4_cell(subset, reader, stream, weighting)
                    block = a3.setdefault(task, {}).setdefault(model, {}).setdefault(reader, {}).setdefault(stream, {})
                    for difficulty in sorted({ep["difficulty"] for ep in subset}):
                        same = [ep for ep in subset if ep["difficulty"] == difficulty]
                        block[difficulty] = a3_cell([ep for ep in same if ep["success"]], [ep for ep in same if not ep["success"]],
                                                    reader, stream, repeats, seed)
    common = []
    for model in models:
        for reader in READERS:
            for stream in STREAMS[model]:
                tiers = {row["task"]: row["tier"] for row in main
                         if (row["model"], row["reader"], row["stream"]) == (model, reader, stream)}
                common.append({"model": model, "reader": reader, "stream": stream, "tiers": tiers,
                               "common_pattern": len(tiers) == len(tasks) and all(t == "明显" for t in tiers.values())})
    return _clean({"schema_version": 1, "bootstrap": {"repeats": repeats, "seed": seed, "unit": "局"},
                   "thresholds": {"D": D_THRESHOLD, "slope": SLOPE_THRESHOLD, "null_D": D_NULL, "null_slope": SLOPE_NULL,
                                  "a3_min_episodes_per_side": A3_MIN_EPISODES},
                   "episodes": len(episodes), "A1": a1_block(episodes, repeats, seed), "A2": a2,
                   "A2_main_readouts": main, "A2_review_proportional": review, "common_patterns": common, "A3": a3,
                   "A4": {"note": "事后挑选、未校正，只作描述；榜单只列流质量不低于 1% 的头", "cells": a4},
                   "limitations": LIMITATIONS})


def _clean(value):
    """转成严格 JSON：数组转列表，NaN、inf 转 null。"""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# ---------------------------------------------------------------- 合成注意力（入口自检与测试共用）

def synthetic_demo(m=4, segment_frames=50):
    """示范序号：每段等长，末帧 m+3；返回 (demo_task_index, exec_start_idx)。"""
    demo = [j for j in range(1, m + 1) for _ in range(segment_frames)] + [m + 3]
    return demo, len(demo) - 1


def synthetic_inference(demo, exec_start_idx, j, kind, *, frame_step=5, window_step=16, seq=0, with_motion=True, episode_key="合成:0"):
    """按离线 npz 契约造一次推理。kind="uniform"：质量正比 token；kind="diag"：示范帧与示范窗质量全落在第 j 段。"""
    frames = list(range(0, exec_start_idx, frame_step)) + [exec_start_idx + 3]
    windows = list(range(0, exec_start_idx, window_step)) + [exec_start_idx + 2] if with_motion else []
    names = ["motion", "历史视觉", "当前视觉", "文字", "其余"] + [f"真实帧:{t}" for t in frames] + [f"motion窗:{t}" for t in windows]
    tokens = np.zeros(len(names), np.int64)
    tokens[2:5] = 32, 16, 20
    tokens[5:5 + len(frames)] = 16
    tokens[5 + len(frames):] = 8
    tokens[0], tokens[1] = tokens[5 + len(frames):].sum(), tokens[5:5 + len(frames)].sum()
    base = np.where(np.arange(len(names)) < 2, 0.0, tokens / tokens[:5].sum())

    def concentrate(indices, keep):
        if kind == "diag" and indices:
            kept = [i for i, t in zip(indices, keep) if t]
            if kept:
                total = base[indices].sum()
                base[indices] = 0.0
                base[kept] = total * tokens[kept] / tokens[kept].sum()
    demo_frames = [t for t in frames if t < exec_start_idx]
    concentrate([5 + frames.index(t) for t in demo_frames], [demo[t] == j for t in demo_frames])
    demo_windows = [t for t in windows if t < exec_start_idx]
    concentrate([5 + len(frames) + windows.index(t) for t in demo_windows],
                [set(segments_of_window(t, demo, exec_start_idx)) == {j} for t in demo_windows])
    base[0], base[1] = base[5 + len(frames):].sum(), base[5:5 + len(frames)].sum()
    action = np.broadcast_to(base[:, None, None], (len(names), N_LAYERS, N_HEADS)).copy()
    text_base = base.copy()
    text_base[4] = 0.0
    text = np.broadcast_to((text_base / text_base[:5].sum())[:, None, None], (len(names), N_LAYERS, N_HEADS)).copy()
    return {"path": "<合成>", "names": names, "tokens": tokens, "mass_action": action, "mass_text": text,
            "episode_key": episode_key, "infer_seq": seq, "chunk_start_env_step": seq * 16}


def sanity_check():
    """SUBGOAL_ALIGN 的合成部分：完全均匀 D≈1、斜率≈0；完全对角 D>1、斜率≈1；帧流与比例计权的 uniform 之和为 1。"""
    m = 4
    demo, exec_start = synthetic_demo(m)
    results, uniform_err = {}, 0.0
    for kind in ("diag", "uniform"):
        for stream, weighting in (("frame", "single"), ("motion", "overlap"), ("motion", "proportional")):
            records = []
            for j in range(1, m + 1):
                inference = synthetic_inference(demo, exec_start, j, kind)
                shares = stream_shares(inference, demo, exec_start, m, "action", stream, weighting)
                records.append({"j": j, **shares})
                if weighting != "overlap":
                    uniform_err = max(uniform_err, abs(float(shares["uniform"].sum()) - 1.0))
            results[kind, stream, weighting] = (diag_enrichment(records), centroid_slope(records, m))
    tol = 1e-6
    ok = uniform_err <= SUM_TOL
    for (kind, _, _), (d, slope) in results.items():
        ok = ok and (abs(d - 1.0) <= tol and abs(slope) <= tol if kind == "uniform" else d > 1.0 + tol and abs(slope - 1.0) <= tol)
    return {"ok": bool(ok), "sanity_diag": results["diag", "frame", "single"][0], "sanity_uniform": results["uniform", "frame", "single"][0],
            "uniform_sum_err": uniform_err, "results": {"/".join(k): v for k, v in results.items()}}


# ---------------------------------------------------------------- 图

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")   # dataviz 参考分类色，固定顺序
TIER_COLORS = {"明显": "#104281", "弱": "#5598e7", "未见": "#8c8b85"}   # 有序三档：同一色相深浅，灰色为未见
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e8e7e2"
CJK_FAMILIES = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "WenQuanYi Zen Hei")


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # matplotlib 的字体缓存不一定含 .ttc 系统字体，与 plot_attn 一样按 fc-list 显式注册；不导入 plot_attn 以免带上其模块级副作用。
    try:
        listing = subprocess.run(["fc-list", ":lang=zh", "file"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        listing = ""
    for line in listing.splitlines():
        if line.strip():
            try:
                font_manager.fontManager.addfont(line.split(":")[0].strip())
            except (OSError, RuntimeError):
                pass
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [name for name in CJK_FAMILIES if name in available] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.titlesize": 10})
    return plt


def _cells(summary):
    """（模型, 读者, 流）的固定列顺序。"""
    return [(model, reader, stream) for model in MODELS for reader in READERS for stream in STREAMS[model]]


def plot_a1(summary, out, plt):
    columns = [(model, reader) for model in MODELS for reader in READERS]
    fig, axes = plt.subplots(len(TASKS), len(columns), figsize=(15, 7), constrained_layout=True, squeeze=False)
    handles = {}
    for r, task in enumerate(TASKS):
        for c, (model, reader) in enumerate(columns):
            ax = axes[r, c]
            ax.set_title(f"{task} · {MODEL_ZH[model]} · {READER_ZH[reader]}读者")
            cell = summary["A1"].get(task, {}).get(model, {}).get(reader, {}).get("all", {}).get("all", {})
            js = sorted(cell, key=int)
            if not js:
                ax.text(.5, .5, "无数据", ha="center", va="center", transform=ax.transAxes, color=MUTED)
                continue
            for index, block in enumerate(BLOCKS):
                if (model == "ctx" and "motion" in block) or (reader == "text" and block == "动作 token"):
                    continue
                x = [int(j) for j in js]
                line, = ax.plot(x, [cell[j]["blocks"][block]["share"] for j in js], color=COLORS[index], lw=2, marker="o", ms=5, label=block)
                ax.plot(x, [cell[j]["blocks"][block]["uniform"] for j in js], color=COLORS[index], lw=1.2, ls="--")
                handles.setdefault(block, line)
            ax.set_xticks([int(j) for j in js])
            ax.set_ylim(bottom=0)
            ax.grid(color=GRID)
            ax.set_axisbelow(True)
            if r == len(TASKS) - 1:
                ax.set_xlabel("执行子目标 j")
            if c == 0:
                ax.set_ylabel("注意力占比")
    if handles:
        fig.legend([handles[b] for b in BLOCKS if b in handles], [b for b in BLOCKS if b in handles], loc="outside lower center", ncol=7, frameon=False)
    fig.suptitle("A1 各块注意力占比随子目标变化（实线为实测，同色虚线为按 token 均匀水平；全部难度与成败）")
    fig.savefig(out / "fig_a1_block_share.png", dpi=130)
    plt.close(fig)


def plot_a2_matrix(summary, out, plt):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("diverge", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#d9d8d2")
    cells = _cells(summary)
    fig, axes = plt.subplots(len(TASKS), len(cells), figsize=(3.0 * len(cells), 6.6), constrained_layout=True, squeeze=False)
    matrices = {}
    for task in TASKS:
        for model, reader, stream in cells:
            matrices[task, model, reader, stream] = summary["A2"][task][model][reader][stream][MAIN_WEIGHTING[stream]]["matrix"]
    top = [np.nanmax(np.array(m["enrichment"], float)) for m in matrices.values()
           if m and np.isfinite(np.array(m["enrichment"], float)).any()]
    vmax = max(top + [2.0])
    image = None
    for r, task in enumerate(TASKS):
        for c, (model, reader, stream) in enumerate(cells):
            ax = axes[r, c]
            ax.set_title(f"{task}\n{MODEL_ZH[model]} · {READER_ZH[reader]} · {STREAM_ZH[stream]}", fontsize=9)
            matrix = matrices[task, model, reader, stream]
            if not matrix:
                ax.text(.5, .5, "无数据", ha="center", va="center", transform=ax.transAxes, color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            values = np.array(matrix["enrichment"], float)
            image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, interpolation="nearest", aspect="equal",
                              norm=TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=vmax))
            size = len(values)
            ax.set_xticks(range(size), [str(i + 1) for i in range(size)], fontsize=8)
            ax.set_yticks(range(size), [str(i + 1) for i in range(size)], fontsize=8)
            for j, jp in zip(*np.where(np.isfinite(values))):
                ax.text(jp, j, f"{values[j, jp]:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
            if r == len(TASKS) - 1:
                ax.set_xlabel("读示范第 j′ 段")
            if c == 0:
                ax.set_ylabel("执行第 j 段")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.6, label="share ÷ uniform（1 为按 token 均分）")
    fig.suptitle("A2 子目标对位矩阵（跨局平均，灰色为无推理；motion 流按「有重叠就算」）")
    fig.savefig(out / "fig_a2_matrix.png", dpi=130)
    plt.close(fig)


def plot_a2_forest(summary, out, plt):
    rows = summary["A2_main_readouts"] + summary["A2_review_proportional"]
    labels = [f"{r['task']} · {MODEL_ZH[r['model']]} · {READER_ZH[r['reader']]} · {STREAM_ZH[r['stream']]}"
              + ("（比例计权，复核）" if r["weighting"] == "proportional" else "") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, .38 * len(rows) + 1.6)), sharey=True, constrained_layout=True)
    ys = np.arange(len(rows))[::-1]
    for ax, mean_key, ci_key, null, threshold, title in (
            (axes[0], "D_mean", "D_ci95", D_NULL, D_THRESHOLD, "对角富集 D（局级 95% 区间）"),
            (axes[1], "slope_mean", "slope_ci95", SLOPE_NULL, SLOPE_THRESHOLD, "质心斜率（局级 95% 区间）")):
        ax.axvline(null, color=MUTED, lw=1.2)
        ax.axvline(threshold, color=MUTED, lw=1.2, ls="--")
        for y, row in zip(ys, rows):
            if row[mean_key] is None:
                ax.text(null, y, " 无数据", va="center", fontsize=8, color=MUTED)
                continue
            color = TIER_COLORS[row["tier"]]
            ax.plot(row[ci_key], [y, y], color=color, lw=2)
            ax.plot(row[mean_key], y, "o", color=color, ms=6)
        ax.set_title(f"{title}\n实线为零假设 {null:g}，虚线为门槛 {threshold:g}")
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ys, labels, fontsize=8.5)
    handles = [plt.Line2D([], [], color=color, marker="o", lw=2, label=f"判定：{tier}") for tier, color in TIER_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("A2 主读数与三档判定（颜色按 D 与斜率合并后的判定；未校正多重比较）")
    fig.savefig(out / "fig_a2_forest.png", dpi=130)
    plt.close(fig)


def plot_a4(summary, out, plt):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("diverge", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#d9d8d2")
    cells = _cells(summary)
    grid = {(m, r, s, t): summary["A4"]["cells"][m][r][s][MAIN_WEIGHTING[s]].get(t) for m, r, s in cells for t in TASKS}
    arrays = [np.array(v["excess"], float) for v in grid.values() if v]
    arrays = [a for a in arrays if np.isfinite(a).any()]
    scale = max([float(np.nanmax(np.abs(a))) for a in arrays] + [1e-3])
    ncols = 2 * len(TASKS)
    nrows = int(np.ceil(len(cells) * len(TASKS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
    image = None
    panels = [(m, r, s, t) for m, r, s in cells for t in TASKS]
    for ax, (model, reader, stream, task) in zip(axes.ravel(), panels):
        ax.set_title(f"{MODEL_ZH[model]} · {READER_ZH[reader]} · {STREAM_ZH[stream]}\n{task}", fontsize=9)
        cell = grid[model, reader, stream, task]
        if not cell:
            ax.text(.5, .5, "无数据", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        image = ax.imshow(np.ma.masked_invalid(np.array(cell["excess"], float)), cmap=cmap, interpolation="nearest", aspect="auto",
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale))
        ax.set_xticks(range(N_HEADS))
        ax.set_yticks(range(0, N_LAYERS, 2))
        ax.set_xlabel(f"头（{cell['n_episodes']} 局）", fontsize=8.5)
        ax.set_ylabel("层", fontsize=8.5)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.5, label="对角 share − 对角 uniform")
    fig.suptitle("A4 逐层逐头对角超额（事后挑选、未校正，只作描述）")
    fig.savefig(out / "fig_a4_layer_head.png", dpi=130)
    plt.close(fig)


def plot(summary, out):
    """写 4 张图，返回文件名列表。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out.resolve() / "matplotlib-cache"))
    plt = _pyplot()
    plot_a1(summary, out, plt)
    plot_a2_matrix(summary, out, plt)
    plot_a2_forest(summary, out, plt)
    plot_a4(summary, out, plt)
    return ["fig_a1_block_share.png", "fig_a2_matrix.png", "fig_a2_forest.png", "fig_a4_layer_head.png"]


# ---------------------------------------------------------------- 入口

def run_line(counts, models):
    parts = " ".join(f"valid_{m}={len(counts[m]['accepted'])}" for m in models)
    parts += " " + " ".join(f"invalid_{m}={len(counts[m]['invalid'])}" for m in models)
    retries = sum(counts[m]["infra_retries"] for m in models)
    ok = len(models) == len(MODELS) and all(len(counts[m]["invalid"]) <= MAX_INVALID_PER_MODEL for m in models)
    return ok, f"ROLLOUT_RUN={'PASS' if ok else 'FAIL'} models={len(models)} {parts} infra_retries={retries}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="产物根：<root>/<模型>/<片>/run、<root>/<模型>/offline")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", nargs="+", default=None, help="只用于冒烟：明写片名，跳过 0～3 覆盖检查")
    parser.add_argument("--expected-episodes", type=int, default=100, help="每个模型应有的局数（有效加无效）；不符报错")
    parser.add_argument("--repeats", type=int, default=BOOT_REPEATS)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sanity = sanity_check()
    align = f"models={len(MODELS)} tasks={len(TASKS)} sanity_diag={sanity['sanity_diag']:.6g} sanity_uniform={sanity['sanity_uniform']:.6g}"
    if not sanity["ok"]:
        print(f"SUBGOAL_ALIGN=FAIL {align} reason=sanity {sanity['results']}", flush=True)
        return 1

    counts, runs = {}, {}
    for model in MODELS:
        runs[model] = discover_shards(root / model, args.shards)
        counts[model] = rollout_counts(runs[model])
        total = len(counts[model]["accepted"]) + len(counts[model]["invalid"])
        if total != args.expected_episodes:
            raise SystemExit(f"{model} 的局数 {total} 不等于应有的 {args.expected_episodes}（片 {[str(r) for r in runs[model]]}）")
    run_ok, line = run_line(counts, MODELS)
    verdicts = [line]
    print(line, flush=True)

    episodes, mismatches = [], {}
    n_seen = 0
    for model in MODELS:
        for key, (record, _, run) in sorted(counts[model]["accepted"].items()):
            ep = load_episode(run, root / model / "offline", record)
            n_seen += 1
            errors = check_labels(ep)
            if errors:
                mismatches[f"{model}:{key}"] = errors
            else:
                episodes.append(analyse_episode(ep, model))
            del ep
    label_ok = not mismatches and n_seen > 0
    line = f"SUBGOAL_LABEL={'PASS' if label_ok else 'FAIL'} models={len(MODELS)} episodes={n_seen} mismatches={len(mismatches)}"
    verdicts.append(line)
    print(line, flush=True)
    if mismatches:
        (out / "label_mismatches.json").write_text(json.dumps(mismatches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not label_ok:
        (out / "verdicts.txt").write_text("\n".join(verdicts) + "\n", encoding="utf-8")
        return 1

    uniform_err = max(ep["uniform_sum_err"] for ep in episodes)
    tasks_seen = sorted({ep["task"] for ep in episodes})
    unknown = set(tasks_seen) - set(TASKS)
    if unknown:
        raise SystemExit(f"出现计划外任务：{sorted(unknown)}")
    summary = summarize(episodes, args.repeats, args.seed, TASKS, MODELS)
    align_ok = uniform_err <= SUM_TOL and len(tasks_seen) == len(TASKS)
    align = f"models={len(MODELS)} tasks={len(tasks_seen)} sanity_diag={sanity['sanity_diag']:.6g} sanity_uniform={sanity['sanity_uniform']:.6g}"
    line = f"SUBGOAL_ALIGN={'PASS' if align_ok else 'FAIL'} {align}"
    verdicts.append(line)
    summary.update(sanity=_clean({k: sanity[k] for k in ("sanity_diag", "sanity_uniform", "uniform_sum_err")}),
                   real_uniform_sum_err=uniform_err, verdict_lines=verdicts,
                   invalid_episodes={m: counts[m]["invalid"] for m in MODELS},
                   infra_retries={m: counts[m]["infra_retries"] for m in MODELS})
    (out / "episodes").mkdir(exist_ok=True)
    for ep in episodes:
        name = f"{ep['model']}_{ep['episode_key'].replace(':', '_')}.json"
        (out / "episodes" / name).write_text(json.dumps(episode_report(ep), ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (out / "verdicts.txt").write_text("\n".join(verdicts) + "\n", encoding="utf-8")
    figures = plot(summary, out)
    for row in summary["A2_main_readouts"]:
        print(f"A2 {row['task']} {row['model']} {row['reader']} {row['stream']} n={row['n_episodes']} "
              f"D={row['D_mean']} {row['D_ci95']} slope={row['slope_mean']} {row['slope_ci95']} tier={row['tier']}", flush=True)
    print(f"FIGURES={len(figures)} out={out}", flush=True)
    print(line, flush=True)
    return 0 if (align_ok and run_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
