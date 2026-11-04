"""历史成功集合的逐局注意力分析；只读采集，不执行仿真或因果干预。

入口：--manifest <JSON> --out <目录>。清单顶层为 schema_version=1、model_seed=42、
episodes 列表；每局必含 model、label、task、episode_id、env_seed、difficulty、swap_times、
attn_dir、terminal、video。model 为 with_motion 或 without_motion，label 必须使用 LABELS。
terminal 可指向含 outcome 的 episode.json。video 可为 null。
可选 phase_annotations=[{phase,start_step,end_step,verified,evidence}]，区间左闭右开；
可选 text_token_pieces（与有效文本行逐一对应），或顶层 tokenizer_model（本地模型绝对路径）。
--max-infers 仅供最小离线诊断，其结果不构成完整跨推理汇聚点确认。

复用 X1、R3 纯函数和 R4 通路；R2 在本入口按原公式推广到任意交换段数。
全部推理流式读取，大 attention 不进入缓存。跨局统计先逐局、再等权；无 motion 指标为 null。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from aggregate_attn import FRAME_TOKENS, MOTION_WINDOW, TOKENS_PER_FRAME, _mem_layout, split_streams
import x1_layer_head as x1
import r2_memory_internal as r2
import r3_sink as r3
import r4_layer_paths as r4
from plot_attn import plt

LABELS = {"with_motion": "带 motion token 的 VLA", "without_motion": "不带 motion token 的 VLA"}
EXCLUDED = {"probs_action", "probs_text", "motion_emb", "motion_pos"}


class Record(dict):
    """让原有只读纯函数同时接受已载入的数组，避免同一压缩张量重复解压。"""

    @property
    def files(self):
        return list(self)


def clean(value):
    """严格 JSON：未定义数值为 null，不使用非标准 NaN。"""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def mean(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(values, float), axis=axis)


def swaps_for(k):
    return [(64 + 50 * j, 114 + 50 * j) for j in range(k)]


def task_success_for(outcome):
    """保留原始终局枚举；任务失败和超时均为失败，未知及基础设施状态不作判断。"""
    if outcome == "success":
        return True
    if outcome in ("fail", "timeout", "failure", "failed"):
        return False
    return None


def outcome_label(outcome):
    return {"success":"成功", "fail":"失败", "timeout":"超时",
            "failure":"失败", "failed":"失败"}.get(outcome,"结局未核实")


def ov(t, width, interval):
    lo, hi = interval
    return max(0, min(t + width, hi) - max(t, lo)) / width


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("model_seed") != 42:
        raise ValueError("清单须为 schema_version=1、model_seed=42")
    seen = set()
    for e in manifest["episodes"]:
        required = {"model", "label", "task", "episode_id", "env_seed", "difficulty", "swap_times",
                    "attn_dir", "terminal", "video"}
        if required - e.keys():
            raise ValueError(f"清单缺字段：{sorted(required - e.keys())}")
        if e["model"] not in LABELS or e["label"] != LABELS[e["model"]]:
            raise ValueError("模型标识或固定名称不符")
        if e["task"] != "ButtonUnmaskSwap" or int(e["swap_times"]) < 1:
            raise ValueError("仅支持 ButtonUnmaskSwap，交换次数须为正整数")
        key = (e["model"], e["episode_id"])
        if key in seen:
            raise ValueError(f"重复模型/局号：{key}")
        seen.add(key)
        for field in ("attn_dir", "terminal"):
            if not Path(e[field]).is_absolute() or not Path(e[field]).exists():
                raise ValueError(f"{field} 须为存在的绝对路径：{e[field]}")
    if not seen:
        raise ValueError("清单不能为空")


def token_pieces(e, meta, tokenizer_model):
    """只使用已给定 token 表或本地 sentencepiece；禁止下载或假定目标色的行号。"""
    n = int(meta["txt_len"])
    pieces = e.get("text_token_pieces")
    if pieces is None and tokenizer_model:
        import sentencepiece
        path = Path(tokenizer_model)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("tokenizer_model 须为已存在的本地绝对路径")
        tk = sentencepiece.SentencePieceProcessor(model_file=str(path))
        prompt = str(e["_episode_meta"]["prompt"]).strip().replace("_", " ").replace("\n", " ")
        ids = tk.encode(prompt, add_bos=True) + tk.encode("\n")
        pieces = [tk.id_to_piece(i) for i in ids[:64]]
    if pieces is not None and len(pieces) != n:
        raise ValueError(f"文本行数 {n} 与 token 表 {len(pieces)} 不符")
    return pieces


def general_group_weights(z, k):
    """R2 同式推广：最近4帧优先，其次 VIS、按序 SWAPj、其余；motion 重叠/33。"""
    _, _, order, mask = _mem_layout(z)
    ft = np.asarray(z["frame_times"], int)
    mt = np.asarray(z["motion_times"], int).reshape(-1)
    groups = ["recent", "mid", "VIS"] + [f"SWAP{j+1}" for j in range(k)]
    groups += ["motVIS", "motSWAP", "motOther", "SWAP", "frames", "motion"]
    gi = {g: i for i, g in enumerate(groups)}
    valid_frames = np.where(ft >= 0)[0]
    recent = set(valid_frames[np.argsort(ft[valid_frames])][-4:])
    idx = np.where(mask)[0]
    w = np.zeros((len(idx), len(groups)), np.float64)
    intervals = swaps_for(k)
    for row, s in enumerate(idx):
        o = int(order[s])
        if o < FRAME_TOKENS:
            f = o // TOKENS_PER_FRAME
            if ft[f] < 0:
                raise ValueError("有效帧指向 padding")
            g = "recent" if f in recent else "VIS" if 0 <= ft[f] < 32 else "mid"
            if g == "mid":
                for j, (lo, hi) in enumerate(intervals):
                    if lo <= ft[f] < hi:
                        g = f"SWAP{j+1}"
                        break
            w[row, gi[g]] = 1
            w[row, gi["frames"]] = 1
        else:
            t = int(mt[o - FRAME_TOKENS])
            if t < 0:
                raise ValueError("有效 motion 指向 padding")
            a = ov(t, 33, (0, 32))
            b = sum(ov(t, 33, interval) for interval in intervals)
            for g, value in (("motVIS", a), ("motSWAP", b), ("motOther", 1-a-b), ("motion", 1)):
                w[row, gi[g]] = value
    w[:, gi["SWAP"]] = sum(w[:, gi[f"SWAP{j+1}"]] for j in range(k))
    return idx, w, groups


def memory_metrics(z, pp, k):
    idx, w, groups = general_group_weights(z, k)
    rows = pp[:, :, idx, :].astype(np.float32)
    sums = rows.sum(-1, dtype=np.float64)
    inside = rows[..., idx].sum(-1, dtype=np.float64)
    a = rows[..., idx]
    num = np.einsum("qg,lhqk->lhgk", w.astype(np.float32), a @ w.astype(np.float32), dtype=np.float64)
    qw = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mass = num / qw[None, None, :, None]
    uniform = qw/len(idx)
    if int(z["MEM"]) == FRAME_TOKENS:
        motion_groups = [groups.index(g) for g in ("motVIS", "motSWAP", "motOther", "motion")]
        mass[:, :, motion_groups, :] = np.nan
        mass[:, :, :, motion_groups] = np.nan
        uniform[motion_groups] = np.nan
    return {"groups": groups, "M": mass.astype(np.float32), "qw": qw, "U": uniform,
            "rowsum_err": float(np.abs(sums-1).max()), "outside": float((sums-inside).max())}


def stream_metrics(st, swaps, tf=None, tm=None):
    """统一保留帧0、VIS及所有交换；剔列后均匀期望按剩余 token 数重算。"""
    out = r3.interval_shares(st, swaps, tf, tm)
    frames = st["frame"]
    mass = sum(x["mass"] for x in frames)
    counts = {x["f"]: 16 if tf is None else int(tf[x["f"]]) for x in frames}
    n = sum(counts.values())
    out["first"] = sum(x["mass"] for x in frames if x["t"] == 0)/mass if mass else None
    out["first_u"] = sum(counts[x["f"]] for x in frames if x["t"] == 0)/n if n else None
    for tag in ["first", "vis"] + [f"swap{j+1}" for j in range(len(swaps))]:
        share, uniform = out.get(tag), out.get(tag+"_u")
        out[tag+"_enrich"] = share/uniform if share is not None and uniform and uniform < 1 else None
        out[tag+"_excess"] = mass*(share-uniform) if out[tag+"_enrich"] is not None else None
    if not st["has_motion"]:
        out["motion_mass"] = None
        out["m_swap"] = out["m_swap_u"] = None
        for j in range(len(swaps)):
            out[f"m_swap{j+1}"] = out[f"m_swap{j+1}_u"] = None
    for tag in ["m_swap"] + [f"m_swap{j+1}" for j in range(len(swaps))]:
        out.setdefault(tag, None)
        out.setdefault(tag+"_u", None)
        share, uniform = out.get(tag), out.get(tag+"_u")
        out[tag+"_enrich"] = share/uniform if share is not None and uniform and uniform < 1 else None
        out[tag+"_excess"] = out["motion_mass"]*(share-uniform) if out[tag+"_enrich"] is not None else None
    return out


def phase_record(e, z, pose):
    """按局给出证据；动作目标产生候选，不能充当抓住正确容器的视觉证明。"""
    seq, step = int(z["infer_seq"]), int(z["step_idx"])
    state, actions = np.asarray(z["state"]), np.asarray(z["actions"])
    cmd = actions[:, 7]
    closes = [i for i in range(1, len(cmd)) if cmd[i-1] > 0 and cmd[i] <= 0]
    p = pose.get("infers", {}).get(str(seq), {})
    az = np.asarray(p.get("action_z", []), float)
    rises = [{"row": i, "rise_m": float(az[i:].max()-az[i])} for i in closes if i < len(az)]
    candidate = "抓取候选" if closes else "靠近候选" if float(state[7]) >= 0.03 else "闭合状态，阶段未核实"
    if any(x["rise_m"] >= 0.03 for x in rises):
        candidate = "闭合后抬起候选"
    annotations = [a for a in e.get("phase_annotations", []) if a["start_step"] <= step < a["end_step"]]
    if len(annotations) > 1:
        raise ValueError(f"阶段区间重叠 ep={e['episode_id']} step={step}")
    annotation = annotations[0] if annotations else None
    if annotation and annotation.get("verified") and not annotation.get("evidence"):
        raise ValueError("已核实阶段必须附视觉或状态证据")
    return {"phase": annotation["phase"] if annotation else "未核实",
            "verified": bool(annotation and annotation.get("verified")), "annotation": annotation,
            "candidate": candidate, "state_gripper": float(state[7]), "close_rows": closes,
            "state_z": p.get("state_z"), "action_z": az.tolist() if az.size else None,
            "close_lift_candidates": rises,
            "limitation": "动作块是预测目标，state_z是实测关节状态回放；无视频/状态核实不判定实际抓住或抬起正确容器"}


def target_color_info(prompt, pieces):
    """以实际目标短语和已核token表共同定位目标色，不继承ep8的绿色或第16行。"""
    matches=re.findall(r"hiding\s+the\s+([a-z]+)\s+cube",prompt.lower().replace("_"," "))
    color=matches[0] if len(matches)==1 else None
    rows=[i for i,piece in enumerate(pieces or []) if color and piece.replace("▁", "").strip(",")==color]
    row=rows[0] if len(rows)==1 else None
    return {"color":color,"row":row,"piece":pieces[row] if row is not None else None,
            "verified":row is not None,"reason":None if row is not None else "目标色短语或对应单token行未唯一核实"}


def sink_observation(z, pp, pieces, prompt=""):
    """R3 各族质量及瞬时判定；确认要在全局流式统计之后进行。"""
    meta = {k: z[k] for k in r3.META_KEYS}
    p = int(meta["P"])
    kv = r3.valid_keys(meta)
    row_sums=pp.sum(-1,dtype=np.float64)
    if not np.array_equal(row_sums.mean((0,1))>.5,kv):
        raise ValueError("R3有效prefix query行与有效key位不一致")
    rowdev=float(np.abs(row_sums[:,:,kv]-1).max())
    pad_mass=float(pp[:,:,kv,:][...,~kv].sum(-1,dtype=np.float64).max())
    kva = np.r_[kv, np.ones(20, bool)]
    reg = r3.regions(meta)
    families = {}
    for name in r3.QGROUPS:
        lo, hi = reg[name]
        q = np.arange(lo, hi)[kv[lo:hi]]
        if len(q):
            families[f"q_{name}"] = np.pad(pp[:, :, q, :].astype(np.float32).mean((1, 2)), ((0, 0), (0, 20)))
    target=target_color_info(prompt,pieces)
    target_row=target["row"]
    if target_row is not None:
        families["q_target_color"] = np.pad(np.asarray(z["probs_text"][:, :, target_row], np.float32).mean(1), ((0, 0), (0, 20)))
    families["action"] = np.asarray(z["probs_action"], np.float32).mean((0, 2, 3))
    labels = r3.key_labels(meta, pieces or [f"未核实文本行{i}" for i in range(int(meta["txt_len"]))])
    masks, stats = {}, []
    sensitivity = {}
    for family, masses in families.items():
        masses = masses.astype(np.float64)
        vis = r3.visible(masses, kva, meta)
        rat = r3.ratios(masses, vis)
        masks[family] = r3.detect(rat)
        sensitivity[family] = {str(th): int(((rat >= th).sum(0) >= 6).sum()) for th in (5, 10, 20, 50)}
        wrat = r3.within_frame_ratios(masses, meta, kva)
        masks[family+"_frame"] = r3.detect(wrat)
        for scope, ratios in (("global", rat), ("frame", wrat)):
            present = vis.copy()
            if scope == "frame":
                present &= np.array([lb["kind"] == "mem_frame" for lb in labels])
            hits = r3.detect(ratios)
            for key in np.where(present)[0]:
                stats.append((scope, family, labels[key]["sid"], labels[key]["kind"], bool(hits[key])))
    return {"meta": meta, "kva": kva, "labels": labels, "masks": masks, "stats": stats,
            "families": list(families), "target_color":target,
            "green_row": target_row if target["color"]=="green" else None, "sensitivity": sensitivity,
            "checks":{"prefix_rowsum_err":rowdev,"prefix_padding_key_mass":pad_mass}}


def confirm_sinks(counts):
    """位置存在的有效推理中≥50%，再要求≥2且≥一半可见query族确认。"""
    confirmed = defaultdict(dict)
    visible = defaultdict(list)
    for (scope, family, sid), rec in counts.items():
        visible[scope, sid].append(family)
        if rec["sink"] and rec["sink"]/rec["present"] >= 0.5:
            confirmed[scope, sid][family] = rec
    items = []
    for (scope, sid), fams in confirmed.items():
        kind = next(iter(fams.values()))["kind"]
        if len(fams) >= 2 and len(fams)*2 >= len(visible[scope, sid]):
            category = "全局 sink"
        elif all(("text" if f=="q_target_color" else r3.SELF_REGION.get(f)) == r3.any_kind_region(kind) for f in fams):
            category = "自区域局部注意"
        else:
            category = "族内 sink"
        if scope == "frame":
            category = "帧流内 sink"
        items.append({"scope": scope, "sid": sid, "kind": kind, "class": category,
                      "families": fams, "visible_families": visible[scope, sid]})
    return items


def robust_infer(z, obs, sinks, swaps):
    labels, kva = obs["labels"], obs["kva"]
    sids = np.array([lb["sid"] for lb in labels])
    global_ids = {s["sid"] for s in sinks if s["class"] == "全局 sink"}
    real = [s for s in sinks if s["class"] != "自区域局部注意"]
    action_ids = {s["sid"] for s in real if "action" in s["families"]}
    mem_ids = {s["sid"] for s in real if s["scope"] == "global" and "q_mem" in s["families"]}
    ca = np.isin(sids, list(global_ids)) & kva
    cb = ca | (np.isin(sids, list(action_ids)) & kva)
    cb |= obs["masks"]["action"] & np.array([lb["kind"] != "action" for lb in labels])
    cb |= obs["masks"]["action_frame"]
    cc = cb | (np.isin(sids, list(mem_ids)) & kva)
    cols = {"orig": np.zeros_like(kva), "RA": ca, "RB": cb, "RC": cc, "RB_resid1e-3": cb}
    pa = np.asarray(z["probs_action"], np.float64)
    out = {}
    for variant, removed in cols.items():
        if variant == "orig":
            vec, dev, dropped = pa.mean((0, 1, 2, 3)), 0., 0.
        else:
            # 全部行被剔除是不可计算，保留明确状态，不能用零向量伪造概率分布。
            residual = pa[..., ~removed].sum(-1)
            minimum = 1e-3 if variant == "RB_resid1e-3" else 1e-2
            if not np.any(residual >= minimum):
                out[variant] = {"computable": False, "dropped_row_frac": 1., "reason": "所有query行剩余质量不足"}
                continue
            vec, dev, dropped = r3.robust_vec(pa, removed, minimum)
        st = r3.streams_from(vec, obs["meta"])
        tf, tm = r3.tok_counts(obs["meta"], removed)
        out[variant] = {"computable": True, "metrics": stream_metrics(st, swaps, tf, tm),
                        "rowdev": dev, "dropped_row_frac": dropped,
                        "removed": sids[removed].tolist(), "removed_mass_orig": float(pa[..., removed].sum(-1).mean())}
    return out


def episode_analysis(e, out, tokenizer_model=None, max_infers=None):
    d = Path(e["attn_dir"])
    meta = json.loads((d/"episode.json").read_text())
    for key in ("episode_id", "env_seed", "difficulty"):
        if meta.get(key) != e[key]:
            raise ValueError(f"局身份不符：{key} {meta.get(key)!r} != {e[key]!r}")
    if meta.get("task_name", meta.get("task")) != e["task"]:
        raise ValueError("任务身份不符")
    if bool(meta.get("motion_enabled")) != (e["model"] == "with_motion"):
        raise ValueError("motion身份不符")
    terminal = json.loads(Path(e["terminal"]).read_text())
    outcome = terminal.get("outcome")
    if outcome is None:
        success = terminal.get("task_success", terminal.get("success"))
        outcome = "success" if success is True else "failure" if success is False else "未核实"
    pose_path = d/"sim_eef_pose.json"
    pose = json.loads(pose_path.read_text()) if pose_path.exists() else {}
    for key in ("episode_id", "env_seed", "difficulty"):
        if pose and pose.get(key) != e[key]:
            raise ValueError(f"TCP回放局身份不符：{key}")
    if pose.get("swap_times") is not None and pose["swap_times"] != e["swap_times"]:
        raise ValueError("TCP回放与清单交换次数不符")
    e = dict(e, _episode_meta=meta)
    files = sorted(d.glob("infer_*.npz"))
    all_count = len(files)
    files = files if max_infers is None else files[:max_infers]
    if not files:
        raise ValueError(f"无推理记录：{d}")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    observations = []
    records = []
    swaps = swaps_for(int(e["swap_times"]))
    for path in files:
        with np.load(path) as source:
            z = Record({k: source[k] for k in source.files})
        seq = int(z["infer_seq"])
        with np.load(d/f"prefix_{seq:04d}.npz") as prefix:
            if int(prefix["infer_seq"]) != seq or int(prefix["P"]) != int(z["P"]):
                raise ValueError("prefix与infer元数据不符")
            pp = prefix["probs_prefix"]
        pieces = token_pieces(e, z, tokenizer_model)
        h = x1.infer_metrics(z, int(e["swap_times"]))
        streams = split_streams(z)
        # 第0帧独立于VIS统计，不改变X1已有定义。
        vec = np.asarray(z["probs_action"], np.float32).mean((0, 3))
        layout = r4.layout(z)
        fm = (layout["kind"] == "frame") & (layout["t"] == 0)
        h["f_first_mass"] = vec[..., fm].sum(-1)
        nf = len(streams["frame"])
        first_u = sum(x["t"] == 0 for x in streams["frame"])/nf if nf else np.nan
        h["f_first_share"] = np.divide(h["f_first_mass"], h["m_frame"], out=np.full((18,8), np.nan), where=h["m_frame"]>0)
        h["f_first_uni"] = np.full((18,8), first_u)
        h["f_first_enrich"] = h["f_first_share"]/first_u if 0 < first_u < 1 else np.full((18,8), np.nan)
        h["f_first_excess"] = h["m_frame"]*(h["f_first_share"]-first_u) if 0 < first_u < 1 else np.full((18,8), np.nan)
        memory = memory_metrics(z, pp, int(e["swap_times"]))
        # R4旧函数的旧KEY_STEP只取VIS；不用其中未输出的motion沾边值，所有motion指标由重叠/33重算。
        paths = r4.analyse_infer(z, Record(probs_prefix=pp), int(e["swap_times"]))
        paths.pop("fullmean_frame_vis_share", None)
        if not streams["has_motion"]:
            for key in list(h):
                if key == "m_motion" or key.startswith("m_swap") or key.startswith("m_vis"):
                    h[key] = None
            for target in paths["targets"].values():
                if target:
                    target["direct_motion"] = target["ind_motion"] = target["u_m"] = None
        observation = sink_observation(z, pp, pieces,meta.get("prompt",""))
        for scope, family, sid, kind, hit in observation.pop("stats"):
            c = counts.setdefault((scope, family, sid), {"present": 0, "sink": 0, "kind": kind})
            c["present"] += 1
            c["sink"] += int(hit)
        observations.append(observation)
        windows = []
        for w in streams["motion"]:
            key_positions = np.where((np.asarray(z["mem_order"]) == FRAME_TOKENS+w["w"]) & np.asarray(z["mem_mask"], bool))[0]
            windows.append(dict(w, key_positions=key_positions.tolist(), end=w["t"]+33,
                                vis_overlap33=ov(w["t"], 33, (0,32)),
                                swap_overlap33=[ov(w["t"],33,sw) for sw in swaps],
                                head_mass=vec[..., key_positions].sum(-1)))
        capture = {key: key in z for key in ("motion_emb", "motion_pos")}
        if streams["has_motion"] and all(capture.values()):
            capture.update({key+"_shape": list(z[key].shape) for key in capture.copy()})
        rec = {"infer_seq": seq, "step_idx": int(z["step_idx"]), "source": str(path),
               "phase": phase_record(e,z,pose), "heads": h, "streams": streams,
               "metrics": stream_metrics(streams, swaps), "motion_windows": windows if streams["has_motion"] else None,
               "motion_capture": capture if streams["has_motion"] else None,
               "R2": memory, "R4": paths, "R3_families": observation["families"],
               "text_token_pieces": pieces, "R3_green_row": observation["green_row"],
               "R3_target_color": observation["target_color"],
               "R3_sensitivity": observation["sensitivity"], "R3_checks":observation["checks"]}
        records.append(rec)
        print(f"逐局分析 model={e['model']} ep={e['episode_id']} infer={seq} step={rec['step_idx']}", flush=True)
        del z, pp, vec
    sinks = confirm_sinks(counts)
    # 第二遍仅载入动作attention，逐query行剔列并重新归一；不重读prefix大矩阵。
    for path, rec, obs in zip(files, records, observations):
        with np.load(path) as source:
            z = Record({k: source[k] for k in source.files if k not in ("probs_text", "motion_emb", "motion_pos")})
        rec["R3"] = robust_infer(z, obs, sinks, swaps)
    completeness = len(files) == all_count and (not meta.get("infers") or all_count == meta["infers"])
    result = {"identity": {k: e[k] for k in e if not k.startswith("_")}, "outcome": outcome,
              "task_success": task_success_for(outcome),
              "inference_count": len(files), "available_inference_count": all_count, "complete": completeness,
              "R3_confirmation_scope": "逐局：分母仅包括本局该位置可见的有效推理",
              "R3_full_six_families": all("q_target_color" in x["families"] for x in observations),
              "R3_confirmation_verified": completeness and all("q_target_color" in x["families"] for x in observations), "R3_sinks": sinks,
              "R3_criterion": {"ratio":20,"layers":6,"present_infer_fraction":0.5,"families_min":2,"visible_families_fraction":0.5},
              "records": records}
    result["summary"] = episode_summary(result)
    write_json(out/"episode_analysis.json", result)
    figure_episode(result, out)
    return result


def episode_summary(ep):
    records = ep["records"]
    # 跨推理均值仅用于本局描述；跨局再将每局的均值等权处理。
    post = [r for r in records if r["step_idx"] >= 32]
    selected = post or records
    hs = {}
    for key in records[0]["heads"]:
        if key != "step_idx":
            hs[key] = None if records[0]["heads"][key] is None else mean([r["heads"][key] for r in selected])
    phases = {}
    for name in sorted({r["phase"]["phase"] for r in records}):
        rr = [r for r in records if r["phase"]["phase"] == name]
        phases[name] = {"infers": [r["infer_seq"] for r in rr], "verified": all(r["phase"]["verified"] for r in rr),
                        "heads": {key: None if rr[0]["heads"][key] is None else mean([r["heads"][key] for r in rr])
                                  for key in hs}}
    robust = {}
    for variant in r3.VARIANTS:
        good = [r["R3"][variant] for r in selected if r["R3"][variant]["computable"]]
        robust[variant] = {key: clean(mean([r["metrics"].get(key,np.nan) if r["metrics"].get(key) is not None else np.nan for r in good]))
                           for key in selected[0]["metrics"]}
        robust[variant]["computable_infers"] = len(good)
    top_vis = x1.topn(np.where(hs["m_frame"] >= .01, hs["f_vis_excess"], np.nan))
    motion = hs["m_motion"]
    top_m = None if motion is None else x1.topn(np.where(motion >= .01,hs["m_swap_excess"],np.nan))
    # R2沿用原episode_agg，先按query权重汇总质量和期望，再相除，不平均各次富集比值。
    ms=np.stack([r["R2"]["M"] for r in records]);qw=np.stack([r["R2"]["qw"] for r in records]);us=np.stack([r["R2"]["U"] for r in records])
    mbar,ubar=r2.episode_agg(ms,qw,us,list(range(len(records))))
    if ep["identity"]["model"]=="without_motion":
        mi=[records[0]["R2"]["groups"].index(g) for g in ("motVIS","motSWAP","motOther","motion")]
        mbar[:,:,mi,:]=np.nan;mbar[:,:,:,mi]=np.nan
    with np.errstate(divide="ignore",invalid="ignore"):
        enrichment=np.where(ubar[None,None]>0,mbar/ubar[None,None],np.nan)
    path_summary={}
    for target in records[0]["R4"]["targets"]:
        valid=[r["R4"]["targets"][target] for r in records if r["R4"]["targets"][target] is not None]
        path_summary[target]={"computable_infers":len(valid),"metrics":{
            key:None if valid[0][key] is None else mean([v[key] for v in valid])
            for key in valid[0]}} if valid else None
    return {"post_vis_infers": len(post), "heads": hs, "phases": phases, "robust": robust,
            "R2": {"groups": records[0]["R2"]["groups"], "M":mbar,"U":ubar,"enrich":enrichment},
            "R4":path_summary,
            "candidate_phase_counts": {name:sum(r["phase"]["candidate"]==name for r in records)
                                       for name in sorted({r["phase"]["candidate"] for r in records})},
            "top_vis_heads_mass_ge1pct": [x for x in top_vis if np.isfinite(x[2])],
            "top_motion_swap_heads_mass_ge1pct": None if top_m is None else [x for x in top_m if np.isfinite(x[2])],
            "limitation": "质量小于1%的头保留原数值，但不解读其富集倍数；阶段未核实不转成已证动作阶段"}


def figure_episode(ep, out):
    rows = ep["records"]
    outcome=outcome_label(ep["outcome"])
    title = f"{ep['identity']['label']} · 第{ep['identity']['episode_id']}局 · {outcome}"
    if not ep["complete"]:title+=" · 仅截断诊断"
    xs = [r["step_idx"] for r in rows]
    fig, axes = plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
    for key, label in (("frame_mass","帧流质量"),("vis","VIS内占比"),("vis_u","VIS均匀期望"),("first","第0帧占比")):
        axes[0].plot(xs,[r["metrics"].get(key,np.nan) for r in rows],marker=".",label=label)
    axes[0].legend(ncol=4,fontsize=8)
    for r in rows:
        axes[1].scatter([r["step_idx"]]*len(r["streams"]["frame"]),[x["t"] for x in r["streams"]["frame"]],
                        c=[x["mass"] for x in r["streams"]["frame"]],s=30,cmap="viridis",vmin=0,vmax=1)
    axes[1].axhspan(0,32,color="green",alpha=.15)
    axes[1].set_ylabel("记忆帧时刻（色值为绝对质量）")
    for variant in r3.VARIANTS:
        axes[2].plot(xs,[r["R3"][variant].get("metrics",{}).get("vis",np.nan) for r in rows],marker=".",label=variant)
    axes[2].legend(ncol=5)
    axes[2].set_ylabel("VIS内占比；剔列后逐行归一")
    axes[2].set_xlabel("推理记录对应全域帧号")
    fig.suptitle(title+"\n开局帧回看时间线与去汇聚点稳健性")
    fig.savefig(out/"opening_and_robust.png",dpi=120); plt.close(fig)
    hs = ep["summary"]["heads"]
    fig, axes = plt.subplots(1,3,figsize=(14,6),constrained_layout=True)
    for ax,key,caption in zip(axes,["m_frame","f_vis_excess","m_swap_excess"],["帧流质量","VIS超额质量","motion交换超额质量"]):
        a = hs[key]
        if a is None:
            ax.text(.5,.5,"不适用",ha="center",va="center",transform=ax.transAxes)
        else:
            im=ax.imshow(a,aspect="auto",interpolation="nearest",cmap="viridis")
            fig.colorbar(im,ax=ax,shrink=.7)
            masses=hs["m_motion"] if key.startswith("m_swap") else hs["m_frame"]
            for l,h in zip(*np.where(masses < .01)):
                ax.add_patch(plt.Rectangle((h-.5,l-.5),1,1,fill=False,hatch="///",edgecolor="gray",lw=0))
        ax.set_title(caption); ax.set_xlabel("头，从0编号"); ax.set_ylabel("层，从0编号")
    fig.suptitle(title+"\n逐层逐头均值；斜线表示对应流质量<1%")
    fig.savefig(out/"head_strength.png",dpi=120); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    if ep["identity"]["model"]=="without_motion":
        axes[0].text(.5,.5,"motion指标不适用",ha="center",transform=axes[0].transAxes)
    else:
        times=sorted({w["t"] for r in rows for w in r["motion_windows"]})
        for t in times:
            axes[0].plot(xs,[next((w["mass"] for w in r["motion_windows"] if w["t"]==t),np.nan) for r in rows],marker=".",label=f"[{t},{t+33})")
        if times: axes[0].legend(ncol=5,fontsize=6)
    axes[0].set_ylabel("每个motion窗口绝对质量")
    for key,label in (("direct","直接"),("ind_total","一跳间接"),("ind_adj","相邻层敏感口径"),("twohop","两跳近似")):
        values=[mean([np.full(18,np.nan) if r["R4"]["targets"]["VIS"] is None else r["R4"]["targets"]["VIS"][key] for r in rows])]
        axes[1].plot(np.arange(18),values[0],marker=".",label=label)
    axes[1].legend();axes[1].set_xlabel("层，从0编号");axes[1].set_ylabel("VIS通路绝对质量")
    fig.suptitle(title+"\nmotion逐窗口时间线与间接通路近似")
    fig.savefig(out/"motion_and_paths.png",dpi=120); plt.close(fig)


def summarize(episodes):
    summary={"complete":all(ep["complete"] for ep in episodes),"models":{},"paired_intersection":[],"limitations":["按历史成功入选，重跑结局单列", "两组成功集合不同，总体差异仅描述，不能推出因果", "间接通路忽略残差、MLP与value范数", "R3按局确认，诊断截断不构成完整确认"]}
    for model,label in LABELS.items():
        selected=[ep for ep in episodes if ep["identity"]["model"]==model]
        successful=[ep for ep in selected if ep["task_success"] is True]
        out={"label":label,"selected":len(selected),"completed":len(selected),"success":len(successful),
             "outcomes":[{"episode_id":ep["identity"]["episode_id"],"outcome":ep["outcome"],"complete":ep["complete"]} for ep in selected]}
        for cohort, eps in (("all_selected",selected),("current_success",successful)):
            block={"episodes":len(eps),"equal_episode_head_means":{},"patterns":{}}
            if eps:
                for key in eps[0]["summary"]["heads"]:
                    vals=[ep["summary"]["heads"][key] for ep in eps]
                    block["equal_episode_head_means"][key]=None if vals[0] is None else mean(vals)
            block["R2_paths"]={}
            for source,target in (("recent","VIS"),("recent","SWAP"),("motSWAP","SWAP"),("frames","motion")):
                values=[]
                for ep in eps:
                    memory=ep["summary"]["R2"];gi={g:i for i,g in enumerate(memory["groups"])}
                    values.append({"episode_id":ep["identity"]["episode_id"],
                                   "mass":memory["M"][:,:,gi[source],gi[target]],
                                   "enrich":memory["enrich"][:,:,gi[source],gi[target]]})
                good=[v for v in values if np.isfinite(v["mass"]).any()]
                block["R2_paths"][source+"→"+target]={"computable_episodes":len(good),
                    "equal_episode_mass":mean([v["mass"] for v in good]) if good else None,
                    "equal_episode_enrich":mean([v["enrich"] for v in good]) if good else None,"per_episode":values}
            block["R4_paths"]={}
            for target in sorted({key for ep in eps for key in ep["summary"]["R4"]}):
                good=[ep["summary"]["R4"][target] for ep in eps if ep["summary"]["R4"].get(target)]
                block["R4_paths"][target]={"computable_episodes":len(good),"equal_episode_metrics":{
                    key:None if good[0]["metrics"][key] is None else mean([v["metrics"][key] for v in good])
                    for key in good[0]["metrics"]}} if good else None
            # 共同模式是具名描述性判断，报告所有反例与不可计算局，不把富集单独当证据。
            for name,metric,mass in (("VIS正超额且流质量至少1%","f_vis_excess","m_frame"),("motion交换正超额且流质量至少1%","m_swap_excess","m_motion")):
                support=[];against=[];missing=[]
                for ep in eps:
                    h=ep["summary"]["heads"];eid=ep["identity"]["episode_id"]
                    if h[metric] is None or not ep["summary"]["post_vis_infers"]:
                        missing.append(eid);continue
                    valid=np.isfinite(h[metric]) & (np.asarray(h[mass])>=.01)
                    if not valid.any():missing.append(eid)
                    elif np.any(np.asarray(h[metric])[valid]>0):support.append(eid)
                    else:against.append(eid)
                block["patterns"][name]={"support":len(support),"computable":len(support)+len(against),"support_episode_ids":support,"counterexamples":against,"uncomputable":missing,"criterion":"本局step>=32的推理先平均；存在至少一个满足质量限制且超额>0的头"}
            block["head_stability"]={}
            block["named_head_followup"]={}
            for layer,head in ((3,0),(16,0)):
                block["named_head_followup"][f"L{layer}H{head}"]=[{
                    "episode_id":ep["identity"]["episode_id"],
                    "metrics":{key:None if value is None else np.asarray(value)[layer,head]
                               for key,value in ep["summary"]["heads"].items()}}
                    for ep in eps]
            for metric,mass in (("f_vis_excess","m_frame"),("m_swap_excess","m_motion")):
                count=np.zeros((18,8),int);den=np.zeros((18,8),int)
                for ep in eps:
                    h=ep["summary"]["heads"]
                    if h[metric] is None or not ep["summary"]["post_vis_infers"]:continue
                    valid=np.isfinite(h[metric]) & (np.asarray(h[mass])>=.01)
                    den+=valid;count+=valid & (np.asarray(h[metric])>0)
                block["head_stability"][metric]={"support":count,"computable":den,"fraction":np.divide(count,den,out=np.full((18,8),np.nan),where=den>0)} if model=="with_motion" or metric=="f_vis_excess" else None
            out[cohort]=block
        summary["models"][model]=out
    a={ep["identity"]["episode_id"]:ep for ep in episodes if ep["identity"]["model"]=="with_motion"}
    b={ep["identity"]["episode_id"]:ep for ep in episodes if ep["identity"]["model"]=="without_motion"}
    for eid in sorted(a.keys()&b.keys()):
        summary["paired_intersection"].append({"episode_id":eid,"with_motion_outcome":a[eid]["outcome"],"without_motion_outcome":b[eid]["outcome"],"with_motion_metrics":a[eid]["summary"]["robust"],"without_motion_metrics":b[eid]["summary"]["robust"]})
    return summary


def figures_summary(summary, out):
    fig,axes=plt.subplots(2,2,figsize=(12,11),constrained_layout=True)
    if not summary.get("complete",True):fig.suptitle("仅截断诊断；不是完整跨局或汇聚点确认")
    for row,(model,label) in enumerate(LABELS.items()):
        block=summary["models"][model]["current_success"]
        stability=block["head_stability"]["f_vis_excess"]
        im=axes[row,0].imshow(stability["fraction"],vmin=0,vmax=1,interpolation="nearest",aspect="auto")
        fig.colorbar(im,ax=axes[row,0],shrink=.7)
        axes[row,0].set_title(label+f"\n成功局VIS正超额头稳定性（{block['episodes']}局）")
        axes[row,0].set_xlabel("头，从0编号");axes[row,0].set_ylabel("层，从0编号")
        outcomes=summary["models"][model]["outcomes"]
        ids=[str(e["episode_id"]) for e in outcomes]
        axes[row,1].bar(ids,[1]*len(ids),color=["#3a6" if e["outcome"]=="success" else "#c66" for e in outcomes])
        axes[row,1].set_title(label+"\n逐局结局：绿色成功；红色非成功或未核实")
        axes[row,1].set_xlabel("历史入选局号");axes[row,1].set_yticks([])
    fig.savefig(out/"cross_episode_stability_and_outcomes.png",dpi=120);plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-infers",type=int,help="仅离线诊断，截断后明确标记不完整")
    args=ap.parse_args(argv)
    manifest=json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if args.max_infers is not None and args.max_infers<1:raise ValueError("max-infers须为正整数")
    args.out.mkdir(parents=True,exist_ok=True)
    episodes=[]
    for e in manifest["episodes"]:
        ep=episode_analysis(e,args.out/f"{e['model']}_ep{e['episode_id']}",manifest.get("tokenizer_model"),args.max_infers)
        episodes.append(ep)
    summary=summarize(episodes)
    summary["manifest"]=str(args.manifest.resolve())
    summary["diagnostic_truncation"]=args.max_infers
    write_json(args.out/"success_attention_summary.json",summary)
    figures_summary(summary,args.out)
    complete=all(ep["complete"] for ep in episodes)
    six=all(ep["R3_full_six_families"] for ep in episodes)
    print(f"SUCCESS_ATTENTION=PASS episodes={len(episodes)} complete={int(complete)} r3_six_families={int(six)} diagnostic={int(args.max_infers is not None)}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
