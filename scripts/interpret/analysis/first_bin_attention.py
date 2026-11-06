"""两个 Swap 任务第一次取 bin 窗口的 pattern 分析（计划 1105-swap-first-bin-attention-plan.html 第六节、第二部分 1.5）。

两种入口：
- 正式分析：--windows <模型>=<windows.json或其目录>（每模型一次）、--attn-root <模型>=<first_bin_offline 输出根>、
  --out <目录>；可选 --run-root <模型>=<run 根>（可重复），用于由 accepted 终局找 nonce、由回执取 prompt；
  --tokenizer-model <本地 sentencepiece 模型>。模型只能是 with_motion、without_motion。
  输出 pattern-summary.json、episodes/*/episode_window.json 与图；末行 PATTERN_SUMMARY=PASS tasks=2 models=2 readouts=<n>。
- 对照闸门：--bridge-old22 <上一轮产物根> --out <目录>。只读旧 NPZ，窗口取 [G-3,G]，G=「抓取」标注 start_step//16；
  末行 BRIDGE_OLD22=PASS|FAIL episodes=22 records=<n> metric_maxabs=<x> rb_window_delta=<x> peak160_in_window=<n>/12。
  rb_window_delta 与 peak160_in_window 只报告，不影响 PASS。--bridge-only 模型:局号 只试跑指定局，末行改为 BRIDGE_OLD22_PARTIAL。

单次指标公式全部导入复用（success_attention、x1、r4），组装方式与 episode_analysis 一致，但不调用它，
因为它遍历整局并要求整局完整。confirm_sinks 的计数只来自窗口内推理。文字族只用第一目标色：把 prompt 截到第一个
目标短语再交给 sink_observation，因此双目标局的文字族与上一轮（两目标色等权）不同，只影响 R3 稳健列。

读数分级（第六节，事先定死）：主读数只有 12 个区间，统计单位是局，每局先把窗口内推理取平均，
分组少于 10 局判“未见”。M1 用窗口版 RB，M2 用原始 motion 流（计划未写 RB，RB 版只作描述列）。其余读数全部标
描述性或探索性，保留支持局数与反例，不进判定。头级解释沿用流质量至少 1% 的限制。

M3、M4 是两组不同局的差值，无法把一个向量交给 oracle_press_prob.bootstrap，
由 bootstrap_diff 按同样的重采样方式（2000 次、种子 42、局为单位、2.5%/97.5% 分位）分组抽样；其余区间直接用 bootstrap。

图都是逐（层,头）或一维统计，没有 patch 级热力图放大；头矩阵用 imshow 最近邻显示。
"""
from __future__ import annotations

import os

# 须在 numpy 导入前设置：4 个 CPU 进程按局并行，每进程单线程。
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from aggregate_attn import FRAME_TOKENS, TOKENS_PER_FRAME, _mem_layout, split_streams  # noqa: E402
from oracle_press_prob import bootstrap  # noqa: E402
import plot_attn  # noqa: E402
import r3_sink as r3  # noqa: E402
import r4_layer_paths as r4  # noqa: E402
import success_attention as sa  # noqa: E402
import x1_layer_head as x1  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

plt = sa.plt
CJK = plot_attn.CJK
TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap")
MODELS = ("with_motion", "without_motion")
LABELS = ("correct", "wrong")
MIN_EPISODES = 10
BOOT_REPEATS = 2000
BOOT_SEED = 42
BRIDGE_TOL = 1e-5
LOW_MASS = x1.LOW_MASS
HEAD_DIFF_MIN = 0.01  # 展开路径与原路径有数值差，单头差小于 0.01 不解读（计划风险登记）
TOPN = 10
RELS = (-3, -2, -1, 0)
FIXED_WINDOW = 160
NAMED_HEADS = {"L3H0": (3, 0), "L16H0": (16, 0)}
PEAK_HEAD = "L17H1"
# 读数名 -> 参照值（None 表示绝对质量，不与参照比较）
READOUTS = {"rb_vis_enrich": 1.0, "orig_vis_enrich": 1.0, "rc_vis_enrich": 1.0,
            "m_swap_enrich": 1.0, "rb_m_swap_enrich": 1.0, "f_swap_enrich": 1.0, "rb_f_swap_enrich": 1.0,
            "frame_mass": None, "motion_mass": None,
            "frames_to_motion_mass": None, "frames_to_motion_enrich": 1.0,
            "r4_vis_ind_total": None, "r4_vis_ind_motion": None, "r4_swap_ind_motion": None}
HEAD_METRICS = {"f_vis_excess": "m_frame", "f_swap_excess": "m_frame", "f_first_excess": "m_frame",
                "m_swap_excess": "m_motion", "m_vis_excess": "m_motion"}
HEAD_KEEP = tuple(HEAD_METRICS) + ("m_frame", "m_motion", "f_vis_enrich", "f_first_enrich")
LIMITATIONS = ["注意力只说明看哪里，这些 pattern 不能证明因果",
               "只有窗口数据：整局固定峰、整局 R2、抬起阶段都不能复核，上一轮这些结论既不证实也不推翻",
               "窗口版 RB 的汇聚点确认只来自最多 4 次推理（至少 2 次命中），与上一轮整局版不等同",
               "双目标局文字族只取第一目标色，与上一轮两目标色等权不同，只影响 R3 稳健列",
               "12 个主区间未校正多重比较，单个结果按线索读；头级结果一律探索性",
               "注意力由展开路径重算，与实际执行的原路径有数值差；单头差小于 0.01 不解读",
               "VideoUnmaskSwap 的裸露窗和交换帧未目视核实前，P2、P3 的 Video 列不能解读"]


def _t(zh, en):
    return zh if CJK else en


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# ---------------------------------------------------------------- 单次推理（组装方式与 episode_analysis 一致）

def first_target_prompt(prompt):
    """截到第一个目标短语：文字族只用第一目标色；找不到短语时原样返回，由 target_color_info 报未核实。"""
    match = re.search(r"hiding\s+the\s+[a-z]+\s+cube", prompt.lower().replace("_", " "))
    return prompt[:match.end()] if match else prompt


def load_record(path, exclude=()):
    with np.load(path) as source:
        return sa.Record({k: source[k] for k in source.files if k not in exclude})


def head_peaks(z, vec, heads, windows):
    """P7：本次推理 motion 流、帧流质量最大的头，及该头最关注的 motion 窗或帧。"""
    exec_start = int(z["exec_start_idx"])
    _, mem, order, mask = _mem_layout(z)
    frame_valid = mask & (order < FRAME_TOKENS)
    layer, head = np.unravel_index(int(np.nanargmax(heads["m_frame"])), (x1.N_LAYERS, x1.N_HEADS))
    frame_mass = np.bincount(order[frame_valid] // TOKENS_PER_FRAME, weights=vec[layer, head, :mem][frame_valid],
                             minlength=FRAME_TOKENS // TOKENS_PER_FRAME)
    best = int(np.argmax(frame_mass))
    frame_t = int(np.asarray(z["frame_times"])[best])
    peaks = {"frame": {"head": f"L{layer}H{head}", "stream_mass": float(heads["m_frame"][layer, head]),
                       "t": frame_t, "t_rel_exec": frame_t - exec_start, "attn_mass": float(frame_mass[best])},
             "motion": None}
    if windows and heads["m_motion"] is not None:
        layer, head = np.unravel_index(int(np.nanargmax(heads["m_motion"])), (x1.N_LAYERS, x1.N_HEADS))
        top = max(windows, key=lambda w: float(w["head_mass"][layer, head]))
        peaks["motion"] = {"head": f"L{layer}H{head}", "stream_mass": float(heads["m_motion"][layer, head]),
                           "t": int(top["t"]), "t_rel_exec": int(top["t"]) - exec_start,
                           "attn_mass": float(top["head_mass"][layer, head])}
    return peaks


def build_record(e, z, pp, swap_times, prompt, tokenizer_model):
    """同 episode_analysis 的单次组装；只去掉阶段标注（窗口本身已定阶段），另加 P7 所需的逐头峰值。"""
    swaps = sa.swaps_for(swap_times)
    seq = int(z["infer_seq"])
    pieces = sa.token_pieces(e, z, tokenizer_model)
    h = x1.infer_metrics(z, swap_times)
    streams = split_streams(z)
    vec = np.asarray(z["probs_action"], np.float32).mean((0, 3))
    layout = r4.layout(z)
    fm = (layout["kind"] == "frame") & (layout["t"] == 0)
    h["f_first_mass"] = vec[..., fm].sum(-1)
    nf = len(streams["frame"])
    first_u = sum(x["t"] == 0 for x in streams["frame"]) / nf if nf else np.nan
    h["f_first_share"] = np.divide(h["f_first_mass"], h["m_frame"], out=np.full((18, 8), np.nan), where=h["m_frame"] > 0)
    h["f_first_uni"] = np.full((18, 8), first_u)
    h["f_first_enrich"] = h["f_first_share"] / first_u if 0 < first_u < 1 else np.full((18, 8), np.nan)
    h["f_first_excess"] = h["m_frame"] * (h["f_first_share"] - first_u) if 0 < first_u < 1 else np.full((18, 8), np.nan)
    memory = sa.memory_metrics(z, pp, swap_times)
    paths = r4.analyse_infer(z, sa.Record(probs_prefix=pp), swap_times)
    paths.pop("fullmean_frame_vis_share", None)
    if not streams["has_motion"]:
        for key in list(h):
            if key == "m_motion" or key.startswith("m_swap") or key.startswith("m_vis"):
                h[key] = None
        for target in paths["targets"].values():
            if target:
                target["direct_motion"] = target["ind_motion"] = target["u_m"] = None
    observation = sa.sink_observation(z, pp, pieces, first_target_prompt(prompt))
    stats = observation.pop("stats")
    windows = []
    for w in streams["motion"]:
        positions = np.where((np.asarray(z["mem_order"]) == FRAME_TOKENS + w["w"]) & np.asarray(z["mem_mask"], bool))[0]
        windows.append(dict(w, key_positions=positions.tolist(), end=w["t"] + 33,
                            vis_overlap33=sa.ov(w["t"], 33, (0, 32)),
                            swap_overlap33=[sa.ov(w["t"], 33, sw) for sw in swaps],
                            head_mass=vec[..., positions].sum(-1)))
    rec = {"infer_seq": seq, "step_idx": int(z["step_idx"]), "exec_start_idx": int(z["exec_start_idx"]),
           "heads": h, "streams": streams, "metrics": sa.stream_metrics(streams, swaps),
           "motion_windows": windows if streams["has_motion"] else None,
           "R2": memory, "R4": paths, "R3_families": observation["families"],
           "text_token_pieces": pieces, "R3_target_color": observation["target_color"],
           "R3_sensitivity": observation["sensitivity"], "R3_checks": observation["checks"],
           "P7": head_peaks(z, vec, h, windows if streams["has_motion"] else None)}
    for key in ("attn_path_maxabs", "same_path_maxabs"):
        if key in z:
            rec[key] = float(z[key])
    return rec, observation, stats


def window_records(e, files, swap_times, prompt, tokenizer_model):
    """窗口内逐次组装，再只用这些推理的计数确认汇聚点并算稳健列。files=[(infer路径, prefix路径), ...]"""
    swaps = sa.swaps_for(swap_times)
    counts, records, observations = {}, [], []
    for infer_path, prefix_path in files:
        z = load_record(infer_path)
        with np.load(prefix_path) as prefix:
            if int(prefix["infer_seq"]) != int(z["infer_seq"]) or int(prefix["P"]) != int(z["P"]):
                raise ValueError(f"prefix与infer元数据不符：{prefix_path}")
            pp = prefix["probs_prefix"]
        rec, obs, stats = build_record(e, z, pp, swap_times, prompt, tokenizer_model)
        for scope, family, sid, kind, hit in stats:
            c = counts.setdefault((scope, family, sid), {"present": 0, "sink": 0, "kind": kind})
            c["present"] += 1
            c["sink"] += int(hit)
        records.append(rec)
        observations.append(obs)
        print(f"窗口分析 ep={e.get('episode_key')} infer={rec['infer_seq']} step={rec['step_idx']}", flush=True)
        del z, pp
    sinks = sa.confirm_sinks(counts)
    for (infer_path, _), rec, obs in zip(files, records, observations):
        z = load_record(infer_path, exclude=("probs_text", "motion_emb", "motion_pos"))
        rec["R3"] = sa.robust_infer(z, obs, sinks, swaps)
        del z
    return records, sinks


def close_row_metrics(z, swap_times, close_row):
    """敏感性：G 那次只取第 0～close_row 行动作 query（后段 query 已在规划闭合之后）。"""
    fake = sa.Record(z)
    fake["probs_action"] = np.asarray(z["probs_action"])[:, :, :, :int(close_row) + 1, :]
    swaps = sa.swaps_for(swap_times)
    partial = sa.stream_metrics(split_streams(fake), swaps)
    full = sa.stream_metrics(split_streams(z), swaps)
    keys = ("frame_mass", "motion_mass", "vis_enrich", "m_swap_enrich")
    return {"close_row_in_chunk": int(close_row), "partial": {k: partial.get(k) for k in keys},
            "full": {k: full.get(k) for k in keys}}


def frame_swap_enrich(metrics, swap_times):
    """帧流交换区间并集富集：帧 0/1 计权，各次交换占比与期望分别相加。"""
    share = uniform = 0.0
    for j in range(swap_times):
        if metrics.get(f"swap{j+1}") is not None:
            share += metrics[f"swap{j+1}"]
            uniform += metrics[f"swap{j+1}_u"]
    return share / uniform if 0 < uniform < 1 else None


def infer_readouts(rec, swap_times):
    """单次推理的标量读数；不可计算记 None，不用零或假值代替。"""
    metrics = rec["metrics"]
    robust = lambda variant: rec["R3"][variant].get("metrics") if rec["R3"][variant].get("computable") else None
    rb, rc = robust("RB"), robust("RC")
    pick = lambda block, key: None if block is None else _finite(block.get(key))
    groups = rec["R2"]["groups"]
    frames, motion = groups.index("frames"), groups.index("motion")
    to_motion = np.asarray(rec["R2"]["M"], np.float64)[:, :, frames, motion]
    u_motion = float(rec["R2"]["U"][motion])
    has_motion = rec["streams"]["has_motion"]
    mass = float(np.nanmean(to_motion)) if has_motion else None
    targets = rec["R4"]["targets"]

    def path_mean(target, key):
        values = (targets.get(target) or {}).get(key)
        return None if values is None else _finite(np.nanmean(np.asarray(values, float)[1:]))
    return {"rb_vis_enrich": pick(rb, "vis_enrich"), "orig_vis_enrich": _finite(metrics.get("vis_enrich")),
            "rc_vis_enrich": pick(rc, "vis_enrich"), "m_swap_enrich": _finite(metrics.get("m_swap_enrich")),
            "rb_m_swap_enrich": pick(rb, "m_swap_enrich"),
            "f_swap_enrich": _finite(frame_swap_enrich(metrics, swap_times)),
            "rb_f_swap_enrich": None if rb is None else _finite(frame_swap_enrich(rb, swap_times)),
            "frame_mass": _finite(metrics.get("frame_mass")), "motion_mass": _finite(metrics.get("motion_mass")) if has_motion else None,
            "frames_to_motion_mass": mass,
            "frames_to_motion_enrich": mass / u_motion if has_motion and u_motion > 0 else None,
            "r4_vis_ind_total": path_mean("VIS", "ind_total"), "r4_vis_ind_motion": path_mean("VIS", "ind_motion"),
            "r4_swap_ind_motion": path_mean("SWAP", "ind_motion")}


def episode_means(per_infer):
    """每局先把窗口内推理取平均，一局只算一个值；全部不可计算时为 None。"""
    out = {}
    for name in READOUTS:
        values = [p["readouts"].get(name) for p in per_infer]
        values = [v for v in values if v is not None]
        out[name] = float(np.mean(values)) if values else None
    return out


def episode_peak(records, G):
    """窗口内原始 motion 窗绝对质量的最大值（不是整局固定峰），及承担它的头。"""
    best = None
    for rec in records:
        for w in rec["motion_windows"] or []:
            if best is None or w["mass"] > best[1]["mass"]:
                best = (rec, w)
    if best is None:
        return None
    rec, w = best
    head_mass = np.asarray(w["head_mass"], float)
    layer, head = np.unravel_index(int(np.argmax(head_mass)), head_mass.shape)
    motion_mass = rec["metrics"]["motion_mass"]
    return {"infer_seq": rec["infer_seq"], "rel": rec["infer_seq"] - G, "window_start": int(w["t"]),
            "window_start_rel_exec": int(w["t"]) - rec["exec_start_idx"], "mass": float(w["mass"]),
            "motion_stream_share": float(w["mass"] / motion_mass) if motion_mass else None,
            "swap_overlap33": float(sum(w["swap_overlap33"])), "peak_head": f"L{layer}H{head}",
            "peak_head_attention": float(head_mass[layer, head]),
            "peak_head_fraction": float(head_mass[layer, head] / head_mass.sum()) if head_mass.sum() > 0 else None}


def named_head_values(rec):
    out = {}
    h = rec["heads"]
    for name, (layer, head) in NAMED_HEADS.items():
        out[name] = {key: _finite(np.asarray(h[key])[layer, head])
                     for key in ("m_frame", "f_vis_excess", "f_vis_enrich", "f_first_excess", "f_first_enrich")}
    return out


def window_mean_heads(records):
    return {key: None if records[0]["heads"][key] is None else sa.mean([r["heads"][key] for r in records])
            for key in HEAD_KEEP}


def identity(entry, model):
    key = entry["episode_key"]
    task, eid = key.split(":")
    if task not in TASKS:
        raise ValueError(f"只支持 {TASKS}：{key}")
    for field in ("label", "outcome", "swap_times", "pick_times", "bin_count"):
        if field not in entry:
            raise ValueError(f"{key} 的窗口表缺字段 {field}")
    return {"model": model, "task": task, "episode_id": int(eid), "episode_key": key, "label": entry["label"],
            "outcome": entry["outcome"], "difficulty": entry.get("difficulty"), "swap_times": int(entry["swap_times"]),
            "pick_times": int(entry["pick_times"]), "bin_count": int(entry["bin_count"]),
            "before_buttons": entry.get("before_buttons"), "has_earlier_closes": bool(entry.get("earlier_closes")),
            "earlier_closes": entry.get("earlier_closes"), "window": entry.get("window") or None, "G": entry.get("G"),
            "window_status": entry.get("window_status"),
            "close_row_in_chunk": entry.get("close_row_in_chunk")}


def analyse_episode(job):
    """一局的窗口分析；写审计 JSON，向父进程只交回汇总所需的紧凑结果。"""
    ident = identity(job["entry"], job["model"])
    d, G = Path(job["attn_dir"]), int(ident["G"])
    window = [int(seq) for seq in ident["window"]]
    if G not in window:
        raise ValueError(f"{ident['episode_key']} 的 G={G} 不在窗口 {window} 内")
    files = [(d / f"infer_{seq:04d}.npz", d / f"prefix_{seq:04d}.npz") for seq in window]
    for pair in files:
        for path in pair:
            if not path.is_file():
                raise FileNotFoundError(f"{ident['episode_key']} 缺窗口记录：{path}")
    e = {"episode_key": ident["episode_key"], "_episode_meta": {"prompt": job["prompt"]}}
    if job["entry"].get("text_token_pieces") is not None:
        e["text_token_pieces"] = job["entry"]["text_token_pieces"]
    records, sinks = window_records(e, files, ident["swap_times"], job["prompt"], job.get("tokenizer_model"))
    close = None
    if ident["close_row_in_chunk"] is not None:
        z = load_record(d / f"infer_{G:04d}.npz", exclude=("probs_text", "motion_emb", "motion_pos"))
        close = close_row_metrics(z, ident["swap_times"], ident["close_row_in_chunk"])
        del z
    per_infer = [{"infer_seq": rec["infer_seq"], "rel": rec["infer_seq"] - G, "readouts": infer_readouts(rec, ident["swap_times"]),
                  "named_heads": named_head_values(rec), "P7": rec["P7"],
                  "attn_path_maxabs": rec.get("attn_path_maxabs"), "same_path_maxabs": rec.get("same_path_maxabs")}
                 for rec in records]
    ep = dict(ident, complete=ident["window_status"] == "complete" if ident["window_status"] else len(window) == 4, per_infer=per_infer, readouts=episode_means(per_infer),
              heads=window_mean_heads(records), peak=episode_peak(records, G), close_row=close,
              R3_sink_count=len(sinks),
              R3_target_color_verified=all("q_target_color" in r["R3_families"] for r in records))
    out = Path(job["out"]) / "episodes" / f"{job['model']}_{ident['episode_key'].replace(':', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    sa.write_json(out / "episode_window.json", dict(ep, R3_sinks=sinks, records=records))
    return ep


# ---------------------------------------------------------------- 统计与判定

def values_of(eps, name):
    pairs = [(ep["episode_key"], ep["readouts"].get(name)) for ep in eps]
    return [(k, v) for k, v in pairs if v is not None]


def describe(eps, name, ref=None):
    """描述性读数：点估计、局级区间、支持局数和反例；不进判定。"""
    pairs = values_of(eps, name)
    out = bootstrap([v for _, v in pairs], BOOT_REPEATS, BOOT_SEED)
    if ref is not None:
        out["ref"] = ref
        out["support"] = sum(v > ref for _, v in pairs)
        out["counterexamples"] = [k for k, v in pairs if v <= ref]
    return out


def bootstrap_diff(a, b, repeats=BOOT_REPEATS, seed=BOOT_SEED):
    """a 组均值减 b 组均值；两组是不同的局，各自按局重采样。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return {"n_a": len(a), "n_b": len(b), "mean": None, "ci95": None}
    generator = np.random.default_rng(seed)
    samples = (generator.choice(a, size=(repeats, len(a)), replace=True).mean(axis=1)
               - generator.choice(b, size=(repeats, len(b)), replace=True).mean(axis=1))
    return {"n_a": len(a), "n_b": len(b), "mean": float(a.mean() - b.mean()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def _judge_interval(mean, ci, ref, n_min):
    insufficient = n_min < MIN_EPISODES
    excludes = bool(not insufficient and ci is not None and (ci[0] > ref or ci[1] < ref))
    return {"ref": ref, "insufficient": insufficient, "excludes_ref": excludes,
            "direction": 0 if mean is None else int(np.sign(mean - ref)),
            "ci95": None if insufficient else ci}


def level_interval(eps, name):
    pairs = values_of(eps, name)
    stat = bootstrap([v for _, v in pairs], BOOT_REPEATS, BOOT_SEED)
    out = {"n_episodes": len(pairs), "mean": stat["mean"], **_judge_interval(stat["mean"], stat["ci95"], 1.0, len(pairs))}
    out["support"] = sum(v > 1.0 for _, v in pairs)
    out["counterexamples"] = [k for k, v in pairs if v <= 1.0]
    return out


def diff_interval(eps, name):
    correct = values_of([ep for ep in eps if ep["label"] == "correct"], name)
    wrong = values_of([ep for ep in eps if ep["label"] == "wrong"], name)
    stat = bootstrap_diff([v for _, v in correct], [v for _, v in wrong])
    out = {"n_correct": len(correct), "n_wrong": len(wrong), "mean": stat["mean"],
           **_judge_interval(stat["mean"], stat["ci95"], 0.0, min(len(correct), len(wrong)))}
    return out


def judge(button, video):
    """第六节判定。任一分组少于 10 局直接未见；两任务都不含参照但方向相反不在规则内，如实标出。"""
    if button["insufficient"] or video["insufficient"]:
        return {"verdict": "未见", "task": None, "reason": "存在少于10局的分组"}
    a, b = button["excludes_ref"], video["excludes_ref"]
    if a and b:
        if button["direction"] == video["direction"]:
            return {"verdict": "共同 pattern", "task": None, "reason": "两任务方向相同且区间都不含参照值"}
        return {"verdict": "方向相反（第六节规则未覆盖）", "task": None, "reason": "两任务区间都不含参照值但方向相反"}
    if a or b:
        return {"verdict": "任务特有", "task": TASKS[0] if a else TASKS[1], "reason": "只有一个任务的区间不含参照值"}
    return {"verdict": "未见", "task": None, "reason": "两任务区间都含参照值"}


PRIMARY = (("M1", "rb_vis_enrich", MODELS, level_interval), ("M2", "m_swap_enrich", ("with_motion",), level_interval),
           ("M3", "rb_vis_enrich", MODELS, diff_interval), ("M4", "m_swap_enrich", ("with_motion",), diff_interval))


def select(eps, model=None, task=None, label=None):
    return [ep for ep in eps if (model is None or ep["model"] == model) and (task is None or ep["task"] == task)
            and (label is None or ep["label"] == label)]


def primary_block(eps):
    intervals, verdicts, selection = [], [], []
    for code, name, models, fn in PRIMARY:
        for model in models:
            per_task = {}
            for task in TASKS:
                item = dict(fn(select(eps, model, task), name), readout=code, source=name, model=model, task=task)
                per_task[task] = item
                intervals.append(item)
                if code in ("M3", "M4") and item["excludes_ref"]:
                    selection.append({"readout": code, "model": model, "task": task, "mean": item["mean"],
                                      "ci95": item["ci95"], "note": "12 个区间未校正多重比较，按线索读"})
            verdicts.append(dict(judge(per_task[TASKS[0]], per_task[TASKS[1]]), readout=code, model=model))
    return {"intervals": intervals, "verdicts": verdicts, "selection_related": selection}


def rate_block(rows):
    picked = [r for r in rows if r["label"] in LABELS]
    correct = [r["label"] == "correct" for r in picked]
    success = [r["outcome"] == "success" for r in rows]
    return {"episodes": len(rows), "correct": sum(correct), "wrong": len(picked) - sum(correct),
            "no_pick": sum(r["label"] == "no_pick" for r in rows),
            "selection_rate": bootstrap(np.asarray(correct, float), BOOT_REPEATS, BOOT_SEED),
            "official_success": bootstrap(np.asarray(success, float), BOOT_REPEATS, BOOT_SEED)}


def p1_block(rows):
    out = {}
    for model in MODELS:
        out[model] = {}
        for task in TASKS:
            cell = [r for r in rows if r["model"] == model and r["task"] == task]
            block = rate_block(cell)
            block["strata"] = {field: {str(v): rate_block([r for r in cell if r[field] == v])
                                       for v in sorted({r[field] for r in cell})}
                               for field in ("pick_times", "bin_count", "swap_times")}
            out[model][task] = block
    return {"grade": "描述性", "cells": out}


def readout_table(eps, names):
    """模型 × 任务 × 选对错 的局级描述；单位为局（每局窗口内先平均）。"""
    out = {}
    for model in MODELS:
        out[model] = {}
        for task in TASKS:
            cell = select(eps, model, task)
            out[model][task] = {label: {n: describe(cell if label == "all" else select(cell, label=label), n, READOUTS[n])
                                        for n in names} for label in ("all",) + LABELS}
    return out


def by_relative(eps):
    """按相对推理展开，描述性。"""
    out = {}
    for model in MODELS:
        out[model] = {}
        for task in TASKS:
            out[model][task] = {}
            for label in ("all",) + LABELS:
                group = select(select(eps, model, task), label=None if label == "all" else label)
                block = {}
                for rel in RELS:
                    view = []
                    for ep in group:
                        for p in ep["per_infer"]:
                            if p["rel"] == rel:
                                view.append({"episode_key": ep["episode_key"], "readouts": p["readouts"]})
                    block[str(rel)] = {name: describe(view, name, READOUTS[name]) for name in READOUTS}
                out[model][task][label] = block
    return out


def p4_block(eps):
    """L3H0、L16H0 的开局偏好；[160,193) 只比 Button，Video 报窗口内最大窗起点及与交换的重叠。"""
    out = {"grade": "描述性", "note": "窗口内最大窗，不是整局固定峰", "named_heads": {}, "motion_peak": {}}
    for model in MODELS:
        out["named_heads"][model] = {}
        for task in TASKS:
            out["named_heads"][model][task] = {}
            for label in ("all",) + LABELS:
                group = select(select(eps, model, task), label=None if label == "all" else label)
                block = {}
                for name in NAMED_HEADS:
                    block[name] = {}
                    for rel in (*RELS, "window"):
                        support, against, low = [], [], []
                        for ep in group:
                            infers = ep["per_infer"] if rel == "window" else [p for p in ep["per_infer"] if p["rel"] == rel]
                            if not infers:
                                continue
                            rows = [p["named_heads"][name] for p in infers]
                            mass = np.nanmean([np.nan if r["m_frame"] is None else r["m_frame"] for r in rows])
                            excess = [r["f_vis_excess"] for r in rows if r["f_vis_excess"] is not None]
                            if not excess or not mass >= LOW_MASS:
                                low.append(ep["episode_key"])
                            elif np.mean(excess) > 0:
                                support.append(ep["episode_key"])
                            else:
                                against.append(ep["episode_key"])
                        block[name][str(rel)] = {"support": len(support), "computable": len(support) + len(against),
                                                 "counterexamples": against, "below_1pct_or_missing": low}
                out["named_heads"][model][task][label] = block
    for task in TASKS:
        peaks = [ep for ep in select(eps, "with_motion", task) if ep["peak"]]
        block = {"episodes": len(peaks),
                 "peak_start_counts": dict(sorted(Counter(str(ep["peak"]["window_start"]) for ep in peaks).items())),
                 "peak_start_rel_exec_counts": dict(sorted(Counter(str(ep["peak"]["window_start_rel_exec"]) for ep in peaks).items())),
                 "peak_head_counts": dict(Counter(ep["peak"]["peak_head"] for ep in peaks).most_common()),
                 "swap_overlap33": bootstrap([ep["peak"]["swap_overlap33"] for ep in peaks], BOOT_REPEATS, BOOT_SEED),
                 "per_episode": [dict(ep["peak"], episode_key=ep["episode_key"], label=ep["label"]) for ep in peaks]}
        if task == "ButtonUnmaskSwap":
            at160 = [ep for ep in peaks if ep["peak"]["window_start"] == FIXED_WINDOW]
            block["fixed_window"] = {"start": FIXED_WINDOW, "at_160": len(at160),
                                     "at_160_by_peak_head": sum(ep["peak"]["peak_head"] == PEAK_HEAD for ep in at160),
                                     "peak_head": PEAK_HEAD,
                                     "counterexamples": [ep["episode_key"] for ep in peaks if ep not in at160]}
        out["motion_peak"][task] = block
    return out


def top_heads(counter, n=TOPN):
    """按次数降序、同次数按层头升序，保证确定性。"""
    def order(item):
        layer, head = re.fullmatch(r"L(\d+)H(\d+)", item[0]).groups()
        return (-item[1], int(layer), int(head))
    return [{"head": k, "count": v} for k, v in sorted(counter.items(), key=order)[:n]]


def p7_block(eps):
    """全部 18 层 × 8 头的事后峰值，一律探索性。"""
    out = {"grade": "探索性", "note": "事后挑最大值；motion 窗同时报绝对步和相对演示结束的步", "distribution": {}, "two_task_overlap": {}}
    for model in MODELS:
        out["distribution"][model] = {}
        out["two_task_overlap"][model] = {}
        totals = {stream: {task: Counter() for task in TASKS} for stream in ("motion", "frame")}
        for task in TASKS:
            out["distribution"][model][task] = {}
            for label in LABELS:
                out["distribution"][model][task][label] = {}
                for rel in RELS:
                    block = {"infers": 0}
                    for stream in ("motion", "frame"):
                        rows = [p["P7"][stream] for ep in select(select(eps, model, task), label=label)
                                for p in ep["per_infer"] if p["rel"] == rel and p["P7"][stream]]
                        block["infers"] = max(block["infers"], len(rows))
                        block[stream] = {"peak_heads": top_heads(Counter(r["head"] for r in rows)),
                                         "attended_t": Counter(str(r["t"]) for r in rows).most_common(TOPN),
                                         "attended_t_rel_exec": Counter(str(r["t_rel_exec"]) for r in rows).most_common(TOPN)}
                    out["distribution"][model][task][label][str(rel)] = block
            for stream in ("motion", "frame"):
                totals[stream][task].update(p["P7"][stream]["head"] for ep in select(eps, model, task)
                                            for p in ep["per_infer"] if p["P7"][stream])
        for stream, by_task in totals.items():
            if not any(by_task.values()):
                out["two_task_overlap"][model][stream] = None
                continue
            tops = {task: top_heads(by_task[task]) for task in TASKS}
            shared = {x["head"] for x in tops[TASKS[0]]} & {x["head"] for x in tops[TASKS[1]]}
            out["two_task_overlap"][model][stream] = {"top": tops, "overlap": len(shared), "shared": sorted(shared)}
    return out


def head_block(eps):
    """144 头矩阵：头级一律探索性，质量低于 1% 的头不解读，correct 减 wrong 单头差小于 0.01 不解读。"""
    out = {"grade": "探索性", "cells": {}}
    for model in MODELS:
        out["cells"][model] = {}
        for task in TASKS:
            cell = select(eps, model, task)
            block = {}
            for metric, mass_key in HEAD_METRICS.items():
                usable = [ep for ep in cell if ep["heads"][metric] is not None]
                if not usable:
                    block[metric] = None
                    continue
                excess = np.stack([ep["heads"][metric] for ep in usable]).astype(float)
                mass = np.stack([ep["heads"][mass_key] for ep in usable]).astype(float)
                valid = np.isfinite(excess) & (mass >= LOW_MASS)
                masked = np.where(valid, excess, np.nan)
                labels = np.array([ep["label"] for ep in usable])
                correct, wrong = masked[labels == "correct"], masked[labels == "wrong"]
                diff = (sa.mean(correct) - sa.mean(wrong)) if len(correct) and len(wrong) else np.full((18, 8), np.nan)
                block[metric] = {"mass_key": mass_key, "support": (valid & (excess > 0)).sum(0), "computable": valid.sum(0),
                                 "mean": sa.mean(masked), "correct_minus_wrong": diff,
                                 "n_correct": len(correct), "n_wrong": len(wrong),
                                 "interpretable_diff": np.isfinite(diff) & (np.abs(np.nan_to_num(diff)) >= HEAD_DIFF_MIN)}
            out["cells"][model][task] = block
    return out


def close_row_block(eps):
    """G 只取闭合命令之前 query 的流质量，与全部 20 个 query 对比；敏感性，不进判定。"""
    out = {}
    for model in MODELS:
        out[model] = {}
        for task in TASKS:
            rows = [ep["close_row"] for ep in select(eps, model, task) if ep.get("close_row")]
            out[model][task] = {"episodes": len(rows)}
            for key in ("frame_mass", "motion_mass", "vis_enrich", "m_swap_enrich"):
                pairs = [(_finite(r["partial"].get(key)), _finite(r["full"].get(key))) for r in rows]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                out[model][task][key] = {"partial": bootstrap([a for a, _ in pairs], BOOT_REPEATS, BOOT_SEED),
                                         "full": bootstrap([b for _, b in pairs], BOOT_REPEATS, BOOT_SEED),
                                         "partial_minus_full": bootstrap([a - b for a, b in pairs], BOOT_REPEATS, BOOT_SEED)}
    return out


def build_summary(eps, rows):
    """eps 是有窗口的局；rows 是全部局的身份行（含无窗口局），只用于 P1 和清单。"""
    p2 = ("rb_vis_enrich", "orig_vis_enrich", "rc_vis_enrich")
    p3 = ("m_swap_enrich", "rb_m_swap_enrich", "f_swap_enrich", "rb_f_swap_enrich")
    p6 = ("frames_to_motion_mass", "frames_to_motion_enrich", "r4_vis_ind_total", "r4_vis_ind_motion", "r4_swap_ind_motion")
    p5 = {}
    for model in MODELS:
        p5[model] = {task: {name: diff_interval(select(eps, model, task), name) for name in p2 + p3}
                     for task in TASKS}
    attn = [p["attn_path_maxabs"] for ep in eps for p in ep["per_infer"] if p.get("attn_path_maxabs") is not None]
    same = [p["same_path_maxabs"] for ep in eps for p in ep["per_infer"] if p.get("same_path_maxabs") is not None]
    return {"schema_version": 1, "bootstrap": {"repeats": BOOT_REPEATS, "seed": BOOT_SEED, "unit": "局"},
            "min_episodes_per_group": MIN_EPISODES,
            "primary_definition": {"M1": "窗口版 RB 的帧流 [0,32) 富集，参照 1", "M2": "原始 motion 流交换区间重叠计权富集，参照 1",
                                   "M3": "M1 的 correct 减 wrong，参照 0", "M4": "M2 的 correct 减 wrong，参照 0"},
            "primary": primary_block(eps), "P1": p1_block(rows),
            "P2": {"grade": "描述性（M1 除外）", "table": readout_table(eps, p2)},
            "P3": {"grade": "描述性（M2 除外）", "table": readout_table(eps, p3)},
            "P4": p4_block(eps), "P5": {"grade": "描述性（M3、M4 除外）", "correct_minus_wrong": p5},
            "P6": {"grade": "探索性", "table": readout_table(eps, p6)}, "P7": p7_block(eps),
            "by_relative_infer": {"grade": "描述性", "cells": by_relative(eps)}, "head_level": head_block(eps),
            "sensitivity": {"close_row": close_row_block(eps),
                            "exclude_earlier_closes": dict(primary_block([ep for ep in eps if not ep["has_earlier_closes"]]),
                                                           excluded=[f"{ep['model']}:{ep['episode_key']}" for ep in eps if ep["has_earlier_closes"]])},
            "lists": {"before_buttons": [f"{r['model']}:{r['episode_key']}" for r in rows if r.get("before_buttons")],
                      "incomplete_window": [f"{ep['model']}:{ep['episode_key']}" for ep in eps if not ep["complete"]],
                      "no_window": [{"model": r["model"], "episode_key": r["episode_key"], "label": r["label"],
                                     "window_status": r.get("window_status")}
                                    for r in rows if not r.get("window")]},
            "checks": {"episodes_with_window": len(eps), "episodes_total": len(rows),
                       "attn_path_maxabs": max(attn) if attn else None, "same_path_maxabs": max(same) if same else None},
            "limitations": LIMITATIONS}


# ---------------------------------------------------------------- 图

COLORS = {"correct": "#1f77b4", "wrong": "#d55e00", "with_motion": "#1f77b4", "without_motion": "#7f7f7f"}
SHORT_TASK = {"ButtonUnmaskSwap": "Button", "VideoUnmaskSwap": "Video"}
SHORT_MODEL = {"with_motion": "motion", "without_motion": "基线"}
SHORT_LABEL = {"correct": "选对", "wrong": "选错"}


def fig_selection(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, key, title in zip(axes, ("selection_rate", "official_success"), ("第一个 bin 选对率", "官方成功率")):
        for i, model in enumerate(MODELS):
            xs, ys, lo, hi = [], [], [], []
            for j, task in enumerate(TASKS):
                stat = summary["P1"]["cells"][model][task][key]
                if stat["mean"] is None:
                    continue
                xs.append(j + (i - .5) * .36)
                ys.append(stat["mean"])
                lo.append(stat["mean"] - stat["ci95"][0])
                hi.append(stat["ci95"][1] - stat["mean"])
            ax.bar(xs, ys, width=.33, color=COLORS[model], label=SHORT_MODEL[model], yerr=[lo, hi], capsize=3)
        ax.set_xticks(range(len(TASKS)), [SHORT_TASK[t] for t in TASKS])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    fig.suptitle("P1 选对率与官方成功率（局级重采样 95% 区间，选对率分母不含未抬起局）")
    fig.savefig(out / "fig_selection_rate.png", dpi=130)
    plt.close(fig)


def fig_relative(summary, out):
    names = (("rb_vis_enrich", "帧流 [0,32) RB 富集"), ("m_swap_enrich", "motion 交换区间富集"))
    cells = [(m, t) for m in MODELS for t in TASKS]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True, sharex=True)
    for r, (name, caption) in enumerate(names):
        for c, (model, task) in enumerate(cells):
            ax = axes[r, c]
            ax.set_title(f"{SHORT_MODEL[model]} · {SHORT_TASK[task]}\n{caption}", fontsize=10)
            ax.set_xticks(RELS)
            if name == "m_swap_enrich" and model == "without_motion":
                ax.text(.5, .5, "基线无 motion 流", ha="center", va="center", transform=ax.transAxes)
                ax.set_yticks([])
                continue
            ax.axhline(1, color="0.5", lw=1, ls="--")
            for label in LABELS:
                block = summary["by_relative_infer"]["cells"][model][task][label]
                rows = [(rel, block[str(rel)][name]) for rel in RELS if block[str(rel)][name]["mean"] is not None]
                if not rows:
                    continue
                xs = [rel for rel, _ in rows]
                ax.plot(xs, [s["mean"] for _, s in rows], marker="o", ms=8, lw=2, color=COLORS[label],
                        label=f"{SHORT_LABEL[label]}（n={max(s['n_episodes'] for _, s in rows)}）")
                ax.fill_between(xs, [s["ci95"][0] for _, s in rows], [s["ci95"][1] for _, s in rows],
                                color=COLORS[label], alpha=.15, lw=0)
            ax.grid(color="0.92")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=True, framealpha=.9, edgecolor="none", fontsize=8)
            if r == 1:
                ax.set_xlabel("相对推理（0 = 发出闭合的推理）")
    fig.suptitle("富集随相对推理变化（描述性，虚线为均匀期望 1）")
    fig.savefig(out / "fig_enrich_by_relative.png", dpi=130)
    plt.close(fig)


def fig_head_diff(summary, out):
    names = (("f_vis_excess", "帧流 VIS 超额质量"), ("m_swap_excess", "motion 交换超额质量"))
    cells = [(m, t) for m in MODELS for t in TASKS]
    fig, axes = plt.subplots(2, 4, figsize=(14, 9), constrained_layout=True)
    finite = [np.asarray(block[n]["correct_minus_wrong"], float) for m, t in cells for n, _ in names
              for block in [summary["head_level"]["cells"][m][t]] if block[n] is not None]
    scale = max([float(np.nanmax(np.abs(a))) for a in finite if np.isfinite(a).any()] + [HEAD_DIFF_MIN])
    image = None
    for r, (name, caption) in enumerate(names):
        for c, (model, task) in enumerate(cells):
            ax = axes[r, c]
            ax.set_title(f"{SHORT_MODEL[model]} · {SHORT_TASK[task]}\n{caption}", fontsize=10)
            block = summary["head_level"]["cells"][model][task][name]
            if block is None:
                ax.text(.5, .5, "基线无 motion 流", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            diff = np.asarray(block["correct_minus_wrong"], float)
            image = ax.imshow(np.ma.masked_invalid(diff), cmap="RdBu_r", interpolation="nearest", aspect="auto",
                              norm=TwoSlopeNorm(vcenter=0, vmin=-scale, vmax=scale))
            ax.set_facecolor("0.93")
            for layer, head in zip(*np.where(~np.asarray(block["interpretable_diff"], bool))):
                ax.add_patch(plt.Rectangle((head - .5, layer - .5), 1, 1, fill=False, hatch="///", edgecolor="0.6", lw=0))
            ax.set_xticks(range(x1.N_HEADS))
            ax.set_yticks(range(0, x1.N_LAYERS, 2))
            ax.set_xlabel(f"头（选对 {block['n_correct']} 局，选错 {block['n_wrong']} 局）", fontsize=9)
            ax.set_ylabel("层", fontsize=9)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.5, label="选对减选错")
    fig.suptitle("144 头差值矩阵（探索性；斜线 = 差值小于 0.01 或流质量小于 1%，不解读）")
    fig.savefig(out / "fig_head_diff.png", dpi=130)
    plt.close(fig)


def _count_bars(ax, counts, title, xlabel):
    keys = sorted(counts, key=int)
    ax.bar(range(len(keys)), [counts[k] for k in keys], color=COLORS["with_motion"], width=.7)
    ax.set_xticks(range(len(keys)), keys, rotation=60, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("局数", fontsize=9)
    if not keys:
        ax.text(.5, .5, "无数据", ha="center", va="center", transform=ax.transAxes)


def fig_motion_peaks(summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for c, task in enumerate(TASKS):
        block = summary["P4"]["motion_peak"][task]
        _count_bars(axes[0, c], block["peak_start_counts"], f"{SHORT_TASK[task]}：窗口内最大 motion 窗（绝对步）", "窗口起点")
        _count_bars(axes[1, c], block["peak_start_rel_exec_counts"], f"{SHORT_TASK[task]}：同一窗相对演示结束", "起点 − exec_start_idx")
    fig.suptitle("motion 窗口峰位置（motion 模型，窗口内最大，不是整局固定峰）")
    fig.savefig(out / "fig_motion_peaks.png", dpi=130)
    plt.close(fig)


def fig_peak_heads(eps, out):
    rows = [("with_motion", "motion", "motion · motion 流"), ("with_motion", "frame", "motion · 帧流"), ("without_motion", "frame", "基线 · 帧流")]
    fig, axes = plt.subplots(3, 2, figsize=(9, 12), constrained_layout=True)
    for r, (model, stream, caption) in enumerate(rows):
        for c, task in enumerate(TASKS):
            grid = np.zeros((x1.N_LAYERS, x1.N_HEADS), int)
            for ep in select(eps, model, task):
                for p in ep["per_infer"]:
                    if p["P7"][stream]:
                        layer, head = re.fullmatch(r"L(\d+)H(\d+)", p["P7"][stream]["head"]).groups()
                        grid[int(layer), int(head)] += 1
            ax = axes[r, c]
            im = ax.imshow(grid, cmap="Blues", interpolation="nearest", aspect="auto", vmin=0)
            fig.colorbar(im, ax=ax, shrink=.8, label="推理次数")
            ax.set_title(f"{caption} · {SHORT_TASK[task]}", fontsize=10)
            ax.set_xticks(range(x1.N_HEADS))
            ax.set_yticks(range(0, x1.N_LAYERS, 2))
            ax.set_xlabel("头", fontsize=9)
            ax.set_ylabel("层", fontsize=9)
    fig.suptitle("逐次流质量最大的头（探索性，事后挑最大值）")
    fig.savefig(out / "fig_peak_heads.png", dpi=130)
    plt.close(fig)


def _ci_text(item):
    if item["insufficient"]:
        return f"n={_n(item)}，局数不足"
    return f"{item['mean']:.2f} [{item['ci95'][0]:.2f}, {item['ci95'][1]:.2f}] n={_n(item)}"


def _n(item):
    return item["n_episodes"] if "n_episodes" in item else f"{item['n_correct']}/{item['n_wrong']}"


def fig_pattern_table(summary, out):
    primary = summary["primary"]
    lookup = {(i["readout"], i["model"], i["task"]): i for i in primary["intervals"]}
    rows = []
    for v in primary["verdicts"]:
        verdict = v["verdict"] + (f"（{SHORT_TASK[v['task']]}）" if v["task"] else "")
        rows.append([v["readout"], SHORT_MODEL[v["model"]],
                     _ci_text(lookup[v["readout"], v["model"], TASKS[0]]), _ci_text(lookup[v["readout"], v["model"], TASKS[1]]), verdict])
    fig, ax = plt.subplots(figsize=(13, 3.6), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["主读数", "模型", "Button 均值 [95% 区间]", "Video 均值 [95% 区间]", "判定"],
                     colWidths=[.07, .08, .32, .32, .21], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for j in range(5):
        table[0, j].set_facecolor("0.9")
    ax.set_title("第六节预定规则的判定（M1、M2 参照 1；M3、M4 为选对减选错，参照 0；n 为局数）")
    fig.savefig(out / "fig_pattern_table.png", dpi=130)
    plt.close(fig)


def figures(summary, eps, out):
    fig_selection(summary, out)
    fig_relative(summary, out)
    fig_head_diff(summary, out)
    fig_motion_peaks(summary, out)
    fig_peak_heads(eps, out)
    fig_pattern_table(summary, out)


# ---------------------------------------------------------------- 正式入口

def load_windows(path):
    path = Path(path)
    if path.is_dir():
        path = path / "windows.json"
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("episodes", data.get("windows", data))
    return list(data.values()) if isinstance(data, dict) else list(data)


def accepted_nonces(run_root):
    """episode_key -> nonce，来自 accepted 终局（同 rollout_offline.load_accepted）。"""
    out = {}
    for path in sorted((Path(run_root) / "accepted").glob("*.terminal.json")):
        record = json.loads(path.read_text())
        out[record["episode_key"]] = record["episode_nonce"]
    return out


def receipt_prompt(run_root, nonce):
    for path in sorted(Path(run_root).glob(f"attempts/*/client/traces/{nonce}/meta/hold_receipt.json")):
        prompt = json.loads(path.read_text()).get("prompt")
        if prompt:
            return prompt
    return None


def parse_pairs(items, name):
    out = {}
    for item in items or []:
        model, _, value = item.partition("=")
        if model not in MODELS or not value:
            raise SystemExit(f"{name} 须写成 <{'|'.join(MODELS)}>=<路径>：{item}")
        out.setdefault(model, []).append(value)
    return out


def pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs))


def plan_jobs(windows, attn_roots, run_roots, tokenizer_model, out):
    jobs, rows = [], []
    for model, paths in windows.items():
        if model not in attn_roots:
            raise SystemExit(f"{model} 缺 --attn-root")
        nonces = {}
        for root in run_roots.get(model, []):
            nonces.update(accepted_nonces(root))
        for path in paths:
            for entry in load_windows(path):
                row = identity(entry, model)
                rows.append(row)
                if not row["window"]:
                    continue
                key = row["episode_key"]
                nonce = entry.get("episode_nonce") or entry.get("nonce") or nonces.get(key)
                if not nonce:
                    raise SystemExit(f"{model} {key} 找不到 nonce：窗口表无此字段，--run-root 也没有")
                prompt = entry.get("prompt")
                for root in run_roots.get(model, []):
                    prompt = prompt or receipt_prompt(root, nonce)
                if not prompt:
                    raise SystemExit(f"{model} {key} 找不到 prompt：窗口表无此字段，回执也没有")
                # 两卡分片的离线输出可分放两个根（--attn-root 可重复）：取含该 nonce 的根，都没有时后面按缺记录报错。
                attn_dir = next((Path(r) / nonce for r in attn_roots[model] if (Path(r) / nonce).is_dir()),
                                Path(attn_roots[model][0]) / nonce)
                jobs.append({"model": model, "entry": entry, "attn_dir": str(attn_dir),
                             "prompt": prompt, "tokenizer_model": tokenizer_model, "out": str(out)})
    return jobs, rows


def run_analysis(args):
    windows = parse_pairs(args.windows, "--windows")
    jobs, rows = plan_jobs(windows, {k: v for k, v in parse_pairs(args.attn_root, "--attn-root").items()},
                           parse_pairs(args.run_root, "--run-root"), args.tokenizer_model, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    eps = pool_map(analyse_episode, jobs, args.workers)
    summary = build_summary(eps, rows)
    sa.write_json(args.out / "pattern-summary.json", summary)
    figures(summary, eps, args.out)
    tasks = len({r["task"] for r in rows})
    models = len({r["model"] for r in rows})
    readouts = len(summary["primary"]["intervals"])
    ok = tasks == 2 and models == 2 and readouts == 12
    print(f"PATTERN_SUMMARY={'PASS' if ok else 'FAIL'} tasks={tasks} models={models} readouts={readouts}", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- 对照闸门

def compare_tree(old, new, path=""):
    """逐元素比较：返回 (最大绝对差, 结构或非数值不一致的位置)。null 与非有限值等同（严格 JSON 把 NaN 写成 null）。"""
    if isinstance(old, dict) and isinstance(new, dict):
        worst, bad = 0.0, []
        for key in sorted(old.keys() | new.keys()):
            if key not in old or key not in new:
                bad.append(f"{path}/{key}: 只在一侧")
                continue
            w, b = compare_tree(old[key], new[key], f"{path}/{key}")
            worst, bad = max(worst, w), bad + b
        return worst, bad
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return 0.0, [f"{path}: 长度 {len(old)} != {len(new)}"]
        worst, bad = 0.0, []
        for i, (a, b) in enumerate(zip(old, new)):
            w, m = compare_tree(a, b, f"{path}/{i}")
            worst, bad = max(worst, w), bad + m
        return worst, bad
    number = (int, float)
    if isinstance(old, number) and isinstance(new, number) and not isinstance(old, bool) and not isinstance(new, bool):
        return abs(float(old) - float(new)), []
    if old == new:
        return 0.0, []
    return 0.0, [f"{path}: {old!r} != {new!r}"]


def bridge_sections(rec):
    """比较范围：heads、streams、metrics、R2 的 M/qw/U、R4；不含依赖整局的 R3 稳健列。"""
    return {"heads": rec["heads"], "streams": rec["streams"], "metrics": rec["metrics"],
            "R2": {k: rec["R2"][k] for k in ("M", "qw", "U")}, "R4": rec["R4"]}


def grasp_infer(e):
    grasps = [a for a in e["phase_annotations"] if a["phase"] == "抓取"]
    if not grasps:
        raise ValueError(f"{e['model']} ep{e['episode_id']} 没有抓取标注")
    return int(min(grasps, key=lambda a: a.get("pickup_index", 1))["start_step"]) // 16


def bridge_episode(job):
    e, root = job["episode"], Path(job["root"])
    d = Path(e["attn_dir"])
    G = grasp_infer(e)
    window = list(range(max(0, G - 3), G + 1))
    prompt = json.loads((d / "episode.json").read_text())["prompt"]
    meta = dict(e, episode_key=f"{e['task']}:{e['episode_id']}", _episode_meta={"prompt": prompt})
    files = [(d / f"infer_{seq:04d}.npz", d / f"prefix_{seq:04d}.npz") for seq in window]
    records, _ = window_records(meta, files, int(e["swap_times"]), prompt, job["tokenizer_model"])
    old = json.loads((root / "analysis" / f"{e['model']}_ep{e['episode_id']}" / "episode_analysis.json").read_text())
    old_by_seq = {r["infer_seq"]: r for r in old["records"]}
    worst, bad, by_section, rb_deltas = 0.0, [], {}, []
    for rec in records:
        before = old_by_seq[rec["infer_seq"]]
        now = sa.clean(bridge_sections(rec))
        for name, value in bridge_sections(before).items():
            w, b = compare_tree(value, now[name], f"infer{rec['infer_seq']}/{name}")
            worst = max(worst, w)
            by_section[name] = max(by_section.get(name, 0.0), w)
            bad += b
        a, b = rec["R3"]["RB"], before["R3"]["RB"]
        if a.get("computable") and b.get("computable"):
            x, y = _finite(a["metrics"].get("vis_enrich")), _finite(b["metrics"].get("vis_enrich"))
            if x is not None and y is not None:
                rb_deltas.append({"infer_seq": rec["infer_seq"], "window": x, "whole_episode": y, "abs_delta": abs(x - y)})
    return {"model": e["model"], "episode_id": e["episode_id"], "G": G, "window": window, "records": len(records),
            "metric_maxabs": worst, "by_section": by_section, "mismatches": bad, "rb_vis_enrich": rb_deltas}


def peak160_in_window(peaks_path, results, episodes):
    """峰推理落在 [G-3,G] 的局数；所有 with_motion 局的 G 用抓取标注，部分试跑时只统计已给 G 的局。"""
    peaks = json.loads(Path(peaks_path).read_text())["episodes"]
    grasps = {e["episode_id"]: grasp_infer(e) for e in episodes if e["model"] == "with_motion"}
    rows = [{"episode_id": p["episode_id"], "peak_infer": p["peak_infer"], "G": grasps[p["episode_id"]],
             "in_window": grasps[p["episode_id"]] - 3 <= p["peak_infer"] <= grasps[p["episode_id"]]} for p in peaks]
    return sum(r["in_window"] for r in rows), len(rows), rows


def run_bridge(args):
    root = Path(args.bridge_old22)
    manifest = json.loads((root / "analysis-manifest.json").read_text())
    episodes = manifest["episodes"]
    selected = episodes
    if args.bridge_only:
        wanted = set(args.bridge_only)
        selected = [e for e in episodes if f"{e['model']}:{e['episode_id']}" in wanted]
        if len(selected) != len(wanted):
            raise SystemExit(f"--bridge-only 有不存在的局：{sorted(wanted)}")
    jobs = [{"episode": e, "root": str(root), "tokenizer_model": manifest.get("tokenizer_model")} for e in selected]
    results = pool_map(bridge_episode, jobs, args.workers)
    records = sum(r["records"] for r in results)
    maxabs = max((r["metric_maxabs"] for r in results), default=float("nan"))
    deltas = [d["abs_delta"] for r in results for d in r["rb_vis_enrich"]]
    rb_delta = max(deltas) if deltas else float("nan")
    peaks = args.bridge_peaks or (HERE.parents[2] / "docs/evaluation-doc/attn-buttonunmaskswap-success-seed42-1103/records/motion-peak-windows.json")
    in_window, peak_total, peak_rows = peak160_in_window(peaks, results, episodes)
    mismatches = sum(len(r["mismatches"]) for r in results)
    ok = len(results) == 22 and records > 0 and mismatches == 0 and maxabs <= BRIDGE_TOL
    args.out.mkdir(parents=True, exist_ok=True)
    sa.write_json(args.out / "bridge-old22.json", {
        "root": str(root), "tolerance": BRIDGE_TOL, "episodes": results, "metric_maxabs": maxabs,
        "mismatch_count": mismatches, "rb_window_delta": rb_delta,
        "rb_window_delta_mean": float(np.mean(deltas)) if deltas else None,
        "rb_window_delta_note": "窗口版与整局版 RB 的帧流 [0,32) 富集最大绝对差，只报告；双目标局窗口版文字族只取第一目标色",
        "peak160_in_window": {"count": in_window, "total": peak_total, "rows": peak_rows}})
    tag = "BRIDGE_OLD22_PARTIAL" if args.bridge_only else "BRIDGE_OLD22"
    print(f"{tag}={'PASS' if ok or (args.bridge_only and mismatches == 0 and maxabs <= BRIDGE_TOL) else 'FAIL'} "
          f"episodes={len(results)} records={records} metric_maxabs={maxabs:.3e} rb_window_delta={rb_delta:.3e} "
          f"peak160_in_window={in_window}/{peak_total}", flush=True)
    return 0 if ok or (args.bridge_only and mismatches == 0 and maxabs <= BRIDGE_TOL) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--windows", action="append", help="<模型>=<windows.json 或其目录>，可重复")
    ap.add_argument("--attn-root", action="append", help="<模型>=<first_bin_offline 输出根>，可重复（两卡分片分放时）")
    ap.add_argument("--run-root", action="append", help="<模型>=<run 根>，可重复；窗口表没有 nonce 或 prompt 时用")
    ap.add_argument("--tokenizer-model", help="本地 sentencepiece 模型绝对路径")
    ap.add_argument("--bridge-old22", help="上一轮产物根；给出时只跑对照闸门")
    ap.add_argument("--bridge-peaks", help="motion-peak-windows.json；默认取仓库内上一轮留档")
    ap.add_argument("--bridge-only", action="append", help="只试跑 模型:局号，末行标 PARTIAL")
    args = ap.parse_args(argv)
    if args.bridge_old22:
        return run_bridge(args)
    if not args.windows:
        raise SystemExit("须给 --windows 或 --bridge-old22")
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
