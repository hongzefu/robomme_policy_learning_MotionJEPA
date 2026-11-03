"""R3：attention sink 全面排查与稳健重算（三局 × 20 次推理，prefix 全量 attention + 动作 attention）。

数据（只读）：stage1b 三局目录，各含 infer_00xx.npz（probs_action f16 (10,18,8,20,P+20)、probs_text 等，键见计划
1103-interpret-motion-attention-plan.html 第二部分 §1.3）与 prefix_00xx.npz（probs_prefix f16 (18,8,P,P)，
prefill prefix 内部完整 attention，padding query 行为 0）。prefix = [记忆区 MEM | 当前 base 256 | wrist 256 | 文本 64]。

两个子命令（每次运行 ≤5 分钟，逐次推理流式读、用完释放）：

  stats --tag <局>   把大数组压成「key 被分走的质量」写到 --cache/<局>/stats_00xx.npz：
                     prefix：有效 query 行（行和 > 0.5）上平均 → 逐层逐头 (18,8,P)；按 query 来源
                     （记忆区／当前 base／当前 wrist／文本）分组的头平均 (18,4,P)；
                     动作：10 去噪步 × 20 动作 query 上平均 → 逐层逐头 (18,8,P+20)。
  report             读缓存做 sink 判定、对动作 attention 做「剔 sink 列、每个 query 行重新归一」的稳健重算
                     （重读 infer npz 的 probs_action／probs_text），出图与 r3_summary.json。

sink 判据（README「sink 判据」节论证）：
  query 族 = {记忆区 query, 当前 base query, 当前 wrist query, 文本 query, 「green」词 query, 动作 query}；
  每族只在它「看得见」的 key 上算均匀期望（实测记忆区 query 只看记忆区，图像 query 看不到记忆区）。
  ① 单次推理、单族：头平均后某 key 的质量 ≥ 20 × 均匀，且在 ≥ 6/18 层成立 → 该族该次推理的 sink；
  ② 语义位置（记忆帧按「最新帧／首帧 t=0／帧时刻 t」+ 帧内 j；motion 窗按起点；图像 patch；文本 token；动作 token）
     在该族 ≥ 50% 的「该位置存在」的推理里是 sink → 该族「确认 sink」；
  ③ 在 ≥ 2 个族、且 ≥ 一半看得见它的族里被确认 → 「全局 sink」；只在一个族确认 → 「族内 sink」。
  另算「帧流内 sink」：只在记忆帧 token 内部算均匀（1/帧 token 数），判据同 ①②。

稳健重算（动作 query）：对 probs_action 每个（去噪步, 层, 头, 动作 query）行把剔除列置 0，除以剩余列之和
（行和重归一 = 1，自检最大偏差 ≤ 1e-6），再与原口径一样全平均，交 aggregate_attn.split_streams（只 import 不改）
分流；帧流 VIS／SWAP1／SWAP2 占比、motion 流 SWAP 占比（窗 [t,t+33) 按重叠帧数/33 计权，同 x3_intervals.shares）。
均匀期望按剔除后剩余的帧 token 数／窗数计。三档剔除：
  RA = 全局 sink；RB = RA + 动作族确认 sink + 动作族帧流内 sink（主稳健口径）；
  RC = RB + 记忆区 query 族的族内 sink（压力测试：把记忆区内部的 sink 也从动作 query 里剔掉）。

末行 R3_SINK=PASS|FAIL episodes=<n> sinks=<n> figs=<n>。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # scripts/interpret
from aggregate_attn import (FRAME_TOKENS, IMAGE_BASE, IMAGE_WRIST, MOTION_WINDOW,  # noqa: E402
                            TOKENS_PER_FRAME, VISIBLE, split_streams)

STAGE1B = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1b")
EPISODES = {  # 标签: (模型目录, 局目录, 图注, 交换区间)
    "ctx_ep8": ("ctx", "0621439bab554595b3ee0f0f9414ba84", "ctx512 基线 ep8（成功）", [(64, 114), (114, 164)]),
    "ctx_ep4": ("ctx", "3a79ca2d563c4a679d51013e15394130", "ctx512 基线 ep4（失败）", [(64, 114)]),
    "mot_ep8": ("mot", "afb7438311e44dfab40a03e709c2ab16", "ctx512+motion ep8（成功）", [(64, 114), (114, 164)]),
}
COLORS = {"ctx_ep4": "#9e9e9e", "ctx_ep8": "#4c72b0", "mot_ep8": "#dd8452"}
N_INFER = 20
FOCUS = (15, 19)
KEY_INFER = 18
TXT_MAX = 64
GREEN_ROW = 16  # X4 token 表：文本第 16 行 = ▁green
QGROUPS = ["mem", "base", "wrist", "text"]
FAMILIES = ["q_mem", "q_base", "q_wrist", "q_text", "q_green", "action"]
FAM_CN = {"q_mem": "记忆区 query", "q_base": "当前 base query", "q_wrist": "当前 wrist query",
          "q_text": "文本 query（19 行平均）", "q_green": "「green」词 query", "action": "动作 query"}

SINK_RATIO = 20.0          # ≥ 均匀的 20 倍
SINK_MIN_LAYERS = 6        # 在 ≥ 6/18 层成立
SINK_MIN_INFER_FRAC = 0.5  # 该位置存在的推理里 ≥ 50% 判为 sink
ROWSUM_TOL = 1e-6
MIN_RESID = 1e-2           # 稳健重算：剔除后剩余质量 < 1% 的 query 行不参与平均（敏感性另跑 1e-3）
SELF_REGION = {"q_text": "text", "q_green": "text", "action": "action"}  # 族自身小区域（19／20 token）
VARIANTS = ["orig", "RA", "RB", "RC", "RB_resid1e-3"]

META_KEYS = ["P", "MEM", "mem_order", "mem_mask", "frame_times", "motion_times", "motion_mask",
             "txt_len", "step_idx", "infer_seq", "exec_start_idx"]


# ---------------------------------------------------------------- 工具

def ep_dir(tag: str) -> Path:
    model, nonce, _, _ = EPISODES[tag]
    return STAGE1B / model / "attn" / nonce


def valid_keys(meta: dict) -> np.ndarray:
    """prefix 内有效 key 掩码（长度 P）：记忆区 mem_mask、当前图像全有效、文本前 txt_len。"""
    P, MEM, txt = int(meta["P"]), int(meta["MEM"]), int(meta["txt_len"])
    v = np.zeros(P, bool)
    v[:MEM] = meta["mem_mask"]
    v[MEM:MEM + IMAGE_BASE + IMAGE_WRIST] = True
    t0 = MEM + IMAGE_BASE + IMAGE_WRIST
    v[t0:t0 + txt] = True
    return v


def regions(meta: dict) -> dict:
    MEM, P = int(meta["MEM"]), int(meta["P"])
    t0 = MEM + IMAGE_BASE + IMAGE_WRIST
    return {"mem": (0, MEM), "base": (MEM, MEM + IMAGE_BASE), "wrist": (MEM + IMAGE_BASE, t0),
            "text": (t0, P), "action": (P, P + 20)}


def key_labels(meta: dict, pieces: list[str]) -> list[dict]:
    """每个 key（长度 P+20）的语义位置。sid 跨推理可比：记忆帧用「最新帧／首帧 t=0／帧时刻 t」+ 帧内 j。"""
    MEM = int(meta["MEM"])
    mo, mk = meta["mem_order"], meta["mem_mask"]
    ft, mt = meta["frame_times"], meta["motion_times"]
    newest = int(ft[ft >= 0].max())
    out = []
    for s in range(MEM):
        if not mk[s]:
            out.append({"kind": "pad", "sid": "pad"})
        elif mo[s] < FRAME_TOKENS:
            f, j = int(mo[s]) // TOKENS_PER_FRAME, int(mo[s]) % TOKENS_PER_FRAME
            t = int(ft[f])
            role = "最新记忆帧" if t == newest else ("首帧 t=0" if t == 0 else f"记忆帧 t={t}")
            out.append({"kind": "mem_frame", "t": t, "j": j, "f": f,
                        "sid": f"{role} j={j}(行{j // 4},列{j % 4})"})
        else:
            w = int(mo[s]) - FRAME_TOKENS
            t = int(mt[w])
            out.append({"kind": "motion", "t": t, "w": w, "sid": f"motion 窗 [{t},{t + MOTION_WINDOW})"})
    for view in ("base", "wrist"):
        for p in range(IMAGE_BASE):
            r, c = divmod(p, 16)
            out.append({"kind": f"img_{view}", "p": p, "sid": f"当前 {view} patch({r},{c})"})
    for i in range(TXT_MAX):
        if i < int(meta["txt_len"]):
            out.append({"kind": "text", "i": i, "sid": f"文本#{i} {pieces[i]}"})
        else:
            out.append({"kind": "pad", "sid": "pad"})
    for k in range(20):
        out.append({"kind": "action", "k": k, "sid": f"动作 token {k}"})
    return out


# ---------------------------------------------------------------- stats

def cmd_stats(args) -> int:
    d = ep_dir(args.tag)
    out = args.cache / args.tag
    out.mkdir(parents=True, exist_ok=True)
    seqs = range(N_INFER) if args.infers is None else [int(x) for x in args.infers.split(",")]
    for i in seqs:
        t0 = time.time()
        with np.load(d / f"infer_{i:04d}.npz") as z:
            meta = {k: np.asarray(z[k]) for k in META_KEYS}
            pa = z["probs_action"].astype(np.float32)  # (10,18,8,20,K)
        act = pa.mean(axis=(0, 3))  # (18,8,P+20)
        act_rowdev = float(np.abs(pa.sum(-1) - 1).max())
        del pa
        with np.load(d / f"prefix_{i:04d}.npz") as z:
            assert int(z["infer_seq"]) == i and int(z["P"]) == int(meta["P"])
            pp = z["probs_prefix"].astype(np.float32)  # (18,8,P,P)
        P = int(meta["P"])
        rs = pp.sum(-1)
        qv = rs.mean(axis=(0, 1)) > 0.5
        kv = valid_keys(meta)
        assert np.array_equal(qv, kv), "有效 query 行与有效 key 位不一致"
        pref = pp[:, :, qv].mean(axis=2)  # (18,8,P)
        grp = np.zeros((pp.shape[0], len(QGROUPS), P), np.float32)
        reg = regions(meta)
        for gi, g in enumerate(QGROUPS):
            m = np.zeros(P, bool)
            m[reg[g][0]:reg[g][1]] = True
            m &= qv
            grp[:, gi] = pp[:, :, m].mean(axis=(1, 2))
        pad_key_mass = float(pp[:, :, qv][..., ~kv].sum(-1).max())
        rowdev = float(np.abs(rs[:, :, qv] - 1).max())
        del pp
        np.savez(out / f"stats_{i:04d}.npz", pref=pref, grp=grp, act=act, qv=qv,
                 pref_rowdev=rowdev, pref_pad_key_mass=pad_key_mass, act_rowdev=act_rowdev, **meta)
        print(f"R3_STATS tag={args.tag} infer={i} P={P} nq={int(qv.sum())} rowdev={rowdev:.2e} "
              f"padkey={pad_key_mass:.2e} sec={time.time() - t0:.1f}", flush=True)
    return 0


# ---------------------------------------------------------------- report：判定

def load_stats(cache: Path) -> dict:
    data = {}
    for tag in EPISODES:
        data[tag] = {}
        for i in range(N_INFER):
            with np.load(cache / tag / f"stats_{i:04d}.npz") as z:
                data[tag][i] = {k: np.asarray(z[k]) for k in z.files}
    return data


def load_infer(tag: str, i: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """返回 (meta, probs_action f64 (10,18,8,20,K), green 行 (18,8,P) f64)。"""
    with np.load(ep_dir(tag) / f"infer_{i:04d}.npz") as z:
        meta = {k: np.asarray(z[k]) for k in META_KEYS}
        pa = z["probs_action"].astype(np.float64)
        green = z["probs_text"][:, :, GREEN_ROW].astype(np.float64)
    return meta, pa, green


def load_green(tag: str, i: int) -> tuple[dict, np.ndarray]:
    """只读 meta 与「green」文本 query 行 (18,8,P) f64。"""
    with np.load(ep_dir(tag) / f"infer_{i:04d}.npz") as z:
        meta = {k: np.asarray(z[k]) for k in META_KEYS}
        green = z["probs_text"][:, :, GREEN_ROW].astype(np.float64)
    return meta, green


def family_masses(s: dict, green: np.ndarray) -> dict[str, np.ndarray]:
    """各 query 族的头平均 key 质量 (18, P+20)（prefix 族在动作位补 0）。"""
    P = int(s["P"])
    pad = np.zeros((18, 20))
    out = {f"q_{g}": np.concatenate([s["grp"][:, gi].astype(np.float64), pad], 1) for gi, g in enumerate(QGROUPS)}
    out["q_green"] = np.concatenate([green.mean(1), pad], 1)
    out["action"] = s["act"].mean(1).astype(np.float64)
    assert all(v.shape == (18, P + 20) for v in out.values())
    return out


def visible(mass: np.ndarray, kva: np.ndarray, meta: dict) -> np.ndarray:
    """某族看得见的 key：有效位 ∩ 所在区域（记忆／base／wrist／文本／动作）总质量 > 1e-6。"""
    vis = np.zeros_like(kva)
    tot = mass.sum(0)
    for a, b in regions(meta).values():
        if tot[a:b].sum() > 1e-6:
            vis[a:b] = kva[a:b]
    return vis


def ratios(mass: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(L,K) 质量 → 「质量 / 均匀」倍数；均匀 = 1/看得见的 key 数；看不见的位记 0。"""
    r = mass * int(vis.sum())
    r[:, ~vis] = 0
    return r


def within_frame_ratios(mass: np.ndarray, meta: dict, kva: np.ndarray) -> np.ndarray:
    """只在记忆帧 token 内部算倍数（均匀 = 1/帧 token 数），非帧位记 0。"""
    MEM = int(meta["MEM"])
    fr = np.zeros_like(kva)
    fr[:MEM] = kva[:MEM] & (meta["mem_order"] < FRAME_TOKENS)
    mm = np.where(fr[None], mass, 0)
    tot = mm.sum(1, keepdims=True)
    return np.where(tot > 0, mm / np.maximum(tot, 1e-300) * int(fr.sum()), 0)


def detect(rat: np.ndarray) -> np.ndarray:
    return (rat >= SINK_RATIO).sum(0) >= SINK_MIN_LAYERS


def any_kind_region(kind: str) -> str:
    return {"mem_frame": "mem", "motion": "mem", "img_base": "base", "img_wrist": "wrist",
            "text": "text", "action": "action"}.get(kind, kind)


def overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def interval_shares(st: dict, swaps, tok_frame=None, tok_motion=None) -> dict:
    """帧流 VIS／SWAPj 与 motion 流 SWAP（并集与各次，重叠/33）占比及均匀期望。"""
    fr, mo = st["frame"], st["motion"]
    res = {}
    tf = sum(x["mass"] for x in fr)
    ntok = {x["f"]: (16 if tok_frame is None else int(tok_frame[x["f"]])) for x in fr}
    ntot = sum(ntok.values())

    def fshare(lo, hi):
        hit = [x for x in fr if lo <= x["t"] < hi]
        if not hit or tf <= 0:
            return None, None
        return sum(x["mass"] for x in hit) / tf, sum(ntok[x["f"]] for x in hit) / ntot
    res["vis"], res["vis_u"] = fshare(*VISIBLE)
    for j, (a, b) in enumerate(swaps):
        res[f"swap{j + 1}"], res[f"swap{j + 1}_u"] = fshare(a, b)
    res["frame_mass"] = tf
    tm = sum(x["mass"] for x in mo)
    res["motion_mass"] = tm
    if mo:
        wt = {x["w"]: (1 if tok_motion is None else int(tok_motion[x["w"]])) for x in mo}
        nw = sum(wt.values())
        for key, ss in [("m_swap", swaps)] + [(f"m_swap{j + 1}", [sw]) for j, sw in enumerate(swaps)]:
            w = {x["w"]: sum(overlap(x["t"], x["t"] + MOTION_WINDOW, a, b) for a, b in ss) / MOTION_WINDOW
                 for x in mo}
            res[key] = sum(w[x["w"]] * x["mass"] for x in mo) / tm if tm > 0 else None
            res[key + "_u"] = sum(w[x["w"]] * wt[x["w"]] for x in mo) / nw if nw > 0 else None
    return res


def tok_counts(meta: dict, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    MEM = int(meta["MEM"])
    mo, mk = meta["mem_order"], meta["mem_mask"]
    keep = mk & ~cols[:MEM]
    tf = np.zeros(FRAME_TOKENS // TOKENS_PER_FRAME, int)
    tm = np.zeros(len(meta["motion_times"]), int)
    fr = keep & (mo < FRAME_TOKENS)
    np.add.at(tf, mo[fr] // TOKENS_PER_FRAME, 1)
    mm = keep & (mo >= FRAME_TOKENS)
    np.add.at(tm, mo[mm] - FRAME_TOKENS, 1)
    return tf, tm


def robust_vec(pa: np.ndarray, cols: np.ndarray, min_resid: float = MIN_RESID) -> tuple[np.ndarray, float, float]:
    """每个（步,层,头,query）行把 cols 列置 0 并重归一，再在保留行上全平均。

    剔除后剩余质量 < min_resid 的行（几乎全部质量都在 sink 上，重归一只会放大 f16 量化噪声）不参与平均。
    返回 (向量, 保留行的行和最大偏差, 被丢弃行比例)。
    """
    x = pa.copy()
    x[..., cols] = 0.0
    resid = x.sum(-1, keepdims=True)
    keep = resid[..., 0] >= min_resid
    x = np.where(keep[..., None], x / np.maximum(resid, 1e-300), 0.0)
    dev = float(np.abs(x.sum(-1)[keep] - 1).max())
    vec = x.reshape(-1, x.shape[-1]).sum(0) / int(keep.sum())
    return vec, dev, float(1 - keep.mean())


def streams_from(vec: np.ndarray, meta: dict) -> dict:
    fake = dict(meta)
    fake["probs_action"] = vec[None, None, None, None, :]
    return split_streams(fake, query="action")


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def cmd_report(args) -> int:
    from plot_attn import plt
    import cv2
    from x4_text_spatial import token_table

    t_start = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = load_stats(args.cache)
    prompt = json.loads((ep_dir("ctx_ep8") / "episode.json").read_text())["prompt"]
    table = token_table(prompt, int(data["ctx_ep8"][0]["txt_len"]))
    pieces = [r["piece"] for r in table]
    assert pieces[GREEN_ROW] == "▁green", pieces
    summary = {"criterion": {"ratio": SINK_RATIO, "min_layers": SINK_MIN_LAYERS, "n_layers": 18,
                             "min_infer_frac": SINK_MIN_INFER_FRAC, "families": FAM_CN},
               "episodes": {}, "self_check": {}}

    # ---------- 1. 逐推理逐族判定（缓存 + 重读 green 行）
    per = {t: {} for t in EPISODES}
    stat = {}  # (scope, fam, sid) -> 计数
    for tag in EPISODES:
        for i in range(N_INFER):
            s = data[tag][i]
            meta, green = load_green(tag, i)
            kv = valid_keys(meta)
            kva = np.concatenate([kv, np.ones(20, bool)])
            labs = key_labels(meta, pieces)
            fm = family_masses(s, green)
            rec = {"labels": labs, "kva": kva, "meta": meta, "rat": {}, "wrat": {}, "sink": {}, "wsink": {},
                   "vis": {}, "perhead": {}}
            for f, m in fm.items():
                vis = visible(m, kva, meta)
                rat = ratios(m, vis)
                rec["vis"][f], rec["rat"][f], rec["sink"][f] = vis, rat, detect(rat)
                if vis[:int(meta["MEM"])].any():
                    w = within_frame_ratios(m, meta, kva)
                    rec["wrat"][f], rec["wsink"][f] = w, detect(w)
            # 逐头：prefix 全 query 与动作族，(L,H) 对里 ≥ 20× 的个数（共 144）
            for f, arr, vv in (("prefix_all", s["pref"], kva[:int(meta["P"])]), ("action", s["act"], kva)):
                r = arr.astype(np.float64) * int(vv.sum())
                r[..., ~vv] = 0
                rec["perhead"][f] = (r >= SINK_RATIO).sum((0, 1))
            for scope, sinks, rats in (("global", rec["sink"], rec["rat"]), ("frame", rec["wsink"], rec["wrat"])):
                for f, sk in sinks.items():
                    vis = rec["vis"][f].copy()
                    if scope == "frame":
                        vis &= np.array([lb["kind"] == "mem_frame" for lb in labs])
                    for k in np.where(vis)[0]:
                        e = stat.setdefault((scope, f, labs[k]["sid"]),
                                            {"present": 0, "sink": 0, "tags": set(), "kind": labs[k]["kind"],
                                             "nl": [], "mean_ratio": [], "mass": []})
                        e["present"] += 1
                        e["nl"].append(int((rats[f][:, k] >= SINK_RATIO).sum()))
                        e["mean_ratio"].append(float(rats[f][:, k].mean()))
                        e["mass"].append(float(fm[f][:, k].mean()))
                        if sk[k]:
                            e["sink"] += 1
                            e["tags"].add(tag)
            per[tag][i] = rec
            del green
        print(f"R3_DETECT tag={tag} sec={time.time() - t_start:.0f}", flush=True)

    # ---------- 2. 确认 sink 与分类
    confirmed = {}  # (scope, sid) -> {fam: entry}
    for (scope, f, sid), e in stat.items():
        if e["sink"] >= 1 and e["sink"] / e["present"] >= SINK_MIN_INFER_FRAC:
            confirmed.setdefault((scope, sid), {})[f] = e
    sink_list = []
    for (scope, sid), fams in confirmed.items():
        vis_fams = [f for f in FAMILIES if (scope, f, sid) in stat]
        if len(fams) >= 2 and len(fams) * 2 >= len(vis_fams):
            cls = "全局 sink"
        elif all(SELF_REGION.get(f) == any_kind_region(next(iter(fams.values()))["kind"]) for f in fams):
            cls = "自区域局部注意"
        else:
            cls = "族内 sink"
        if scope == "frame":
            cls = "帧流内 sink"
        any_e = next(iter(fams.values()))
        sink_list.append({
            "scope": scope, "class": cls, "sid": sid, "kind": any_e["kind"],
            "families": sorted(fams, key=FAMILIES.index), "visible_families": vis_fams,
            "per_family": {f: {"sink_infers": e["sink"], "present": e["present"], "episodes": sorted(e["tags"]),
                               "median_layers_ge20": float(np.median(e["nl"])),
                               "median_ratio_layer_mean": float(np.median(e["mean_ratio"])),
                               "mean_mass": float(np.mean(e["mass"]))}
                           for f, e in fams.items()},
            "sink_frac_all_families": {f: stat[(scope, f, sid)]["sink"] / stat[(scope, f, sid)]["present"]
                                       for f in vis_fams},
        })
    order = {"全局 sink": 0, "族内 sink": 1, "帧流内 sink": 2, "自区域局部注意": 3}
    sink_list.sort(key=lambda x: (order[x["class"]], x["families"][0],
                                  -max(v["mean_mass"] for v in x["per_family"].values())))
    summary["sink_list"] = sink_list
    sid_global = {x["sid"] for x in sink_list if x["class"] == "全局 sink"}
    real = [x for x in sink_list if x["class"] != "自区域局部注意"]
    sid_action = {x["sid"] for x in real if "action" in x["families"]}
    sid_mem = {x["sid"] for x in real if x["scope"] == "global" and "q_mem" in x["families"]}
    summary["removal_sets"] = {"RA": sorted(sid_global), "RB_extra": sorted(sid_action - sid_global),
                               "RC_extra": sorted(sid_mem - sid_action - sid_global)}
    for x in sink_list:
        print(f"R3_SINKITEM class={x['class']} sid={x['sid']} fams={','.join(x['families'])} "
              + " ".join(f"{f}:{v['sink_infers']}/{v['present']}" for f, v in x["per_family"].items()), flush=True)

    # 逐推理：各族 sink 合计质量
    for tag in EPISODES:
        rows = []
        for i in range(N_INFER):
            rec = per[tag][i]
            r = {"infer": i}
            for f in FAMILIES:
                sk = rec["sink"][f]
                n = int(rec["vis"][f].sum())
                r[f] = {"n_sink": int(sk.sum()), "sink_mass_layer_mean": float(rec["rat"][f][:, sk].sum(1).mean() / n),
                        "sinks": [rec["labels"][k]["sid"] for k in np.where(sk)[0]]}
            rows.append(r)
        summary["episodes"][tag] = {"label": EPISODES[tag][2], "nonce": EPISODES[tag][1],
                                    "swaps": EPISODES[tag][3], "per_infer_sinks": rows}
    # 未达判据的最强候选：各族「全局」与「帧流内」两种口径下，按 ≥20× 层数中位数排序的前 5 个语义位置
    summary["near_miss"] = {}
    for scope in ("global", "frame"):
        for f in FAMILIES:
            items = [(sid, e) for (sc, ff, sid), e in stat.items() if sc == scope and ff == f]
            items.sort(key=lambda x: (-np.median(x[1]["nl"]), -np.median(x[1]["mean_ratio"])))
            summary["near_miss"][f"{scope}/{f}"] = [
                {"sid": sid, "present": e["present"], "sink_infers": e["sink"],
                 "median_layers_ge20": float(np.median(e["nl"])), "max_layers_ge20": int(max(e["nl"])),
                 "median_ratio_layer_mean": float(np.median(e["mean_ratio"]))} for sid, e in items[:5]]
    # 复核 X4「green query 帧流单 token sink」：先层平均再取帧流内最大 token（X4 口径），看它在几层 ≥20×
    summary["x4_green_check"] = {}
    for tag in EPISODES:
        rows = []
        for i in range(FOCUS[0], FOCUS[1] + 1):
            rec = per[tag][i]
            w = rec["wrat"]["q_green"]
            wl = w.mean(0)  # 各层帧流内倍数的层平均
            k = int(np.argmax(wl))
            rows.append({"infer": i, "sid": rec["labels"][k]["sid"], "layer_mean_ratio": float(wl[k]),
                         "layers_ge20": int((w[:, k] >= SINK_RATIO).sum()),
                         "per_layer_ratio": [round(float(x), 1) for x in w[:, k]]})
        summary["x4_green_check"][tag] = rows
    # 逐头：#18 在 144 个（层,头）里 ≥20× 的次数，前 8
    for tag in EPISODES:
        rec = per[tag][KEY_INFER]
        summary["episodes"][tag]["perhead_infer18"] = {
            f: [{"sid": rec["labels"][k]["sid"], "n_layer_head_ge20": int(rec["perhead"][f][k])}
                for k in np.argsort(-rec["perhead"][f])[:8]] for f in ("prefix_all", "action")}
        MEM = int(rec["meta"]["MEM"])
        summary["episodes"][tag]["perhead_infer18"]["action_memory_only"] = [
            {"sid": rec["labels"][k]["sid"], "n_layer_head_ge20": int(rec["perhead"]["action"][k])}
            for k in np.argsort(-rec["perhead"]["action"][:MEM])[:5]]
    # 注意力可见性结构（#18 头平均、层平均）：query 组 × key 区域
    summary["block_structure_infer18"] = {}
    for tag in EPISODES:
        s = data[tag][KEY_INFER]
        reg = regions({k: s[k] for k in META_KEYS})
        summary["block_structure_infer18"][tag] = {
            g: {r: float(s["grp"][:, gi, a:b].sum(-1).mean()) for r, (a, b) in reg.items() if r != "action"}
            for gi, g in enumerate(QGROUPS)}
        a = s["act"].mean((0, 1))
        summary["block_structure_infer18"][tag]["action"] = {r: float(a[x0:x1].sum()) for r, (x0, x1) in reg.items()}

    # ---------- 3. 稳健重算
    max_dev = 0.0
    for tag in EPISODES:
        swaps = EPISODES[tag][3]
        res = {}
        for i in range(N_INFER):
            rec = per[tag][i]
            meta, pa, _ = load_infer(tag, i)
            labs, kva = rec["labels"], rec["kva"]
            sids = np.array([lb["sid"] for lb in labs])
            cA = np.isin(sids, list(sid_global)) & kva
            not_self = np.array([lb["kind"] != "action" for lb in labs])  # 动作→动作 属自区域局部注意，不剔
            cB = cA | (np.isin(sids, list(sid_action)) & kva) | (rec["sink"]["action"] & not_self) | rec["wsink"]["action"]
            cC = cB | (np.isin(sids, list(sid_mem)) & kva)
            vecs = {"orig": pa.mean(axis=(0, 1, 2, 3))}
            cols = {"orig": np.zeros_like(kva), "RA": cA, "RB": cB, "RC": cC, "RB_resid1e-3": cB}
            dropped = {}
            for v in VARIANTS[1:]:
                vecs[v], dev, dropped[v] = robust_vec(pa, cols[v], 1e-3 if v == "RB_resid1e-3" else MIN_RESID)
                max_dev = max(max_dev, dev)
            del pa
            r = {}
            for v in VARIANTS:
                st = streams_from(vecs[v], meta)
                tf, tm = tok_counts(meta, cols[v])
                r[v] = interval_shares(st, swaps, tf, tm)
                r[v]["stream_mass"] = st["mass"]
                r[v]["removed"] = [labs[k]["sid"] for k in np.where(cols[v])[0]]
                r[v]["removed_mass_orig"] = float(vecs["orig"][cols[v]].sum())
                r[v]["dropped_row_frac"] = dropped.get(v, 0.0)
            res[i] = r
        summary["episodes"][tag]["action_shares"] = {str(i): v for i, v in res.items()}

        def agg(seqs, v):
            keys = [k for k in res[N_INFER - 1][v] if k not in ("stream_mass", "removed")]
            return {k: _mean(res[i][v].get(k) for i in seqs) for k in keys}
        summary["episodes"][tag]["focus"] = {v: agg(range(FOCUS[0], FOCUS[1] + 1), v) for v in VARIANTS}
        summary["episodes"][tag]["global"] = {v: agg(range(N_INFER), v) for v in VARIANTS}
        f = summary["episodes"][tag]["focus"]
        print(f"R3_ROBUST tag={tag} focus vis orig={f['orig']['vis']:.4f} RB={f['RB']['vis']:.4f} RC={f['RC']['vis']:.4f} "
              f"u={f['orig']['vis_u']:.4f} sec={time.time() - t_start:.0f}", flush=True)
    # 与上一轮（X3 #18）对拍：原口径应复现
    m18 = summary["episodes"]["mot_ep8"]["action_shares"]["18"]["orig"]
    c18 = summary["episodes"]["ctx_ep8"]["action_shares"]["18"]["orig"]
    summary["self_check"]["reproduce_x3_infer18"] = {
        "mot_vis": m18["vis"], "mot_swap1": m18["swap1"], "mot_swap2": m18["swap2"], "mot_m_swap": m18["m_swap"],
        "mot_m_swap_u": m18["m_swap_u"], "ctx8_vis": c18["vis"], "ctx8_swap1": c18["swap1"], "ctx8_swap2": c18["swap2"]}
    print("R3_REPRO " + " ".join(f"{k}={v:.4f}" for k, v in summary["self_check"]["reproduce_x3_infer18"].items()), flush=True)
    summary["self_check"]["robust_rowsum_max_dev"] = max_dev
    summary["self_check"]["prefix_rowsum_max_dev_f16"] = max(float(data[t][i]["pref_rowdev"]) for t in EPISODES for i in range(N_INFER))
    summary["self_check"]["action_rowsum_max_dev_f16"] = max(float(data[t][i]["act_rowdev"]) for t in EPISODES for i in range(N_INFER))
    summary["self_check"]["prefix_pad_key_mass_max"] = max(float(data[t][i]["pref_pad_key_mass"]) for t in EPISODES for i in range(N_INFER))
    ok_rows = max_dev <= ROWSUM_TOL
    print(f"R3_ROWSUM robust_max_dev={max_dev:.2e} tol={ROWSUM_TOL:.0e} {'PASS' if ok_rows else 'FAIL'}", flush=True)

    figs = plot_all(out, summary, per, plt, cv2)
    out_json = out / "r3_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False))
    json.loads(out_json.read_text())
    n_sinks = len(real)
    ok = ok_rows and n_sinks > 0 and len(figs) >= 4
    print(f"R3_SINK={'PASS' if ok else 'FAIL'} episodes={len(EPISODES)} sinks={n_sinks} figs={len(figs)}", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- 图

def plot_all(out: Path, summary: dict, per: dict, plt, cv2) -> list[Path]:
    figs = []
    fams = ["q_mem", "q_text", "action"]
    # 图 1：#18 逐层 key 倍数热图（三局 × 三族），log10，最近邻
    fig, axes = plt.subplots(3, 3, figsize=(22, 11), constrained_layout=True)
    for r, tag in enumerate(EPISODES):
        rec = per[tag][KEY_INFER]
        reg = regions(rec["meta"])
        for c, f in enumerate(fams):
            ax = axes[r, c]
            rat = rec["rat"][f]
            ax.imshow(np.log10(np.maximum(rat, 1e-2)), aspect="auto", interpolation="nearest", cmap="magma",
                      vmin=-2, vmax=3)
            for name, (a, b) in reg.items():
                cn = {"mem": "记忆区", "base": "base", "wrist": "wrist", "text": "文本", "action": "动作"}[name]
                ax.axvline(a - 0.5, color="c", lw=0.8)
                ax.text((a + b) / 2, 16.9, cn, ha="center", va="bottom", fontsize=8, color="w",
                        bbox=dict(boxstyle="square,pad=0.15", fc="k", ec="none", alpha=0.6))
            sk = np.where(rec["sink"][f])[0]
            ax.scatter(sk, np.full(len(sk), -0.5), marker="v", s=22, color="lime", clip_on=False,
                       label=f"判为 sink（{len(sk)} 个）")
            ax.set_ylim(17.5, -0.5)
            ax.set_ylabel("层")
            ax.set_xlabel("key 位置（prefix 序；动作族另含 20 个动作 token）")
            ax.set_title(f"{EPISODES[tag][2]} · #{KEY_INFER} · {FAM_CN[f]} · 头平均", fontsize=9.5)
            ax.legend(fontsize=7, loc="center left")
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(-2, 3))
    fig.colorbar(sm, ax=axes, fraction=0.015).set_label("log10(key 质量 / 该族均匀)；1.3 = 20 倍")
    fig.suptitle("各层 key 质量倍数（看不见的区域为黑；绿三角 = 该次推理判为 sink：≥20× 且 ≥6 层）", fontsize=12)
    p = out / "fig_r3_1_layer_key_heat.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    figs.append(p)

    # 图 2：sink 清单——每条在各族的「判为 sink 的推理比例」（nearest 热图）
    sl = summary["sink_list"]
    M = np.full((len(sl), len(FAMILIES)), np.nan)
    for a, x in enumerate(sl):
        for b, f in enumerate(FAMILIES):
            if f in x["sink_frac_all_families"]:
                M[a, b] = x["sink_frac_all_families"][f]
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(sl) + 2.2), constrained_layout=True)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#dddddd")
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest", cmap=cm, vmin=0, vmax=1)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            if not np.isnan(M[a, b]):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7,
                        color="k" if M[a, b] > 0.5 else "w")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAM_CN[f] for f in FAMILIES], fontsize=8.5, rotation=15)
    ax.set_yticks(range(len(sl)))
    ax.set_yticklabels([f"[{x['class']}] {x['sid']}" for x in sl], fontsize=7.5)
    ax.set_title("确认 sink 清单：各 query 族里「判为 sink 的推理 / 该位置存在的推理」（灰 = 该族看不见此 key）", fontsize=10)
    p = out / "fig_r3_2_sink_list.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    figs.append(p)

    # 图 3：逐层被判 sink 的 key 合计质量（#15～#19 平均）
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    for ax, f in zip(axes, fams):
        for tag in EPISODES:
            vals = []
            for i in range(FOCUS[0], FOCUS[1] + 1):
                rec = per[tag][i]
                vals.append(rec["rat"][f][:, rec["sink"][f]].sum(1) / int(rec["vis"][f].sum()))
            ax.plot(np.mean(vals, 0) * 100, "-o", ms=3.5, color=COLORS[tag], label=EPISODES[tag][2])
        ax.set_xlabel("层")
        ax.set_ylabel("被判 sink 的 key 合计质量（%）")
        ax.set_xticks(range(0, 18, 2))
        ax.set_ylim(0, 100)
        ax.set_title(f"{FAM_CN[f]} · 逐层 sink 合计质量 · #15～#19 平均", fontsize=10)
        ax.legend(fontsize=8)
    p = out / "fig_r3_3_sink_mass_by_layer.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    figs.append(p)

    # 图 4：原口径 vs 去 sink（RB 主口径、RC 压力测试）曲线
    metrics = [("vis", "帧流 VIS [0,32)"), ("swap1", "帧流 SWAP1 [64,114)"), ("swap2", "帧流 SWAP2 [114,164)"),
               ("m_swap", "motion 流 SWAP 并集（重叠/33，仅 motion 模型）")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    style = {"orig": ("-o", 1.0, "原口径"), "RB": ("--s", 0.9, "去 sink RB"), "RC": (":^", 0.8, "去 sink RC（压力）")}
    for ax, (m, ttl) in zip(axes.ravel(), metrics):
        for tag in EPISODES:
            sh = summary["episodes"][tag]["action_shares"]
            xs = np.arange(N_INFER)

            def g(v, k):
                return np.array([np.nan if sh[str(i)][v].get(k) is None else sh[str(i)][v][k] for i in xs], float)
            if np.all(np.isnan(g("orig", m))):
                continue
            for v, (ls, al, nm) in style.items():
                ax.plot(xs, g(v, m), ls, ms=3, lw=1.2, color=COLORS[tag], alpha=al, label=f"{EPISODES[tag][2]} {nm}")
            ax.plot(xs, g("orig", m + "_u"), "-", lw=3, color=COLORS[tag], alpha=0.25, label="  均匀期望（原）")
        ax.axvspan(FOCUS[0] - 0.5, FOCUS[1] + 0.5, color="#ffd54f", alpha=0.18)
        ax.set_title(ttl + " · 动作 query 全平均", fontsize=10)
        ax.set_xlabel("推理序号（黄底 #15～#19）")
        ax.set_ylabel("占该流质量比例")
        ax.set_xticks(range(0, N_INFER, 2))
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=8)
    p = out / "fig_r3_4_orig_vs_robust.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    figs.append(p)

    # 图 5：当前图像 16×16 patch 倍数（#18，动作族），最近邻放大，绿框 = 判为 sink
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)
    for c, tag in enumerate(EPISODES):
        rec = per[tag][KEY_INFER]
        reg = regions(rec["meta"])
        for r, view in enumerate(("base", "wrist")):
            ax = axes[r, c]
            a, b = reg[view]
            rat = rec["rat"]["action"].mean(0)[a:b].reshape(16, 16)
            big = cv2.resize(np.log10(np.maximum(rat, 1e-2)).astype(np.float32), (256, 256),
                             interpolation=cv2.INTER_NEAREST)
            ax.imshow(big, cmap="magma", vmin=-1, vmax=2.5, interpolation="nearest")
            sk = rec["sink"]["action"][a:b].reshape(16, 16)
            for rr, cc in zip(*np.where(sk)):
                ax.add_patch(plt.Rectangle((cc * 16 - 0.5, rr * 16 - 0.5), 16, 16, fill=False, ec="lime", lw=1.6))
            ax.set_title(f"{EPISODES[tag][2]}\n当前 {view} · 动作 query 18 层平均 · #{KEY_INFER}", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(-1, 2.5))
    fig.colorbar(sm, ax=axes, fraction=0.025).set_label("log10(质量/均匀)；绿框 = 判为 sink")
    p = out / "fig_r3_5_image_patch_sink.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    figs.append(p)
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stats")
    a.add_argument("--tag", required=True, choices=list(EPISODES))
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--infers", default=None, help="逗号分隔推理序号；默认 0..19")
    b = sub.add_parser("report")
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--out", type=Path,
                   default=HERE.parents[3] / "docs/evaluation-doc/attn-ctx512-vs-motion-1103/analysis/r3")
    args = ap.parse_args(argv)
    return cmd_stats(args) if args.cmd == "stats" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
