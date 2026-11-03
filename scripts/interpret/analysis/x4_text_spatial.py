"""X4：任务文字里各词的 prefill 注意力看向哪些历史帧，以及历史帧／当前图像的帧内空间位置。

问题 1（文本词 → 历史帧）：用与 policy 同一 tokenizer（mme_vla_suite.training.config.PaligemmaTokenizer，
pi0 格式 discrete_state_input=False：encode(prompt, bos) + encode("\\n")）还原本局 prompt 的 token 序列，
核对长度 = npz 的 txt_len；按 piece 定位 green／cube／container／hiding／buttons 等词所在的文本 query 行。
对每个词，把 probs_text 的该词 query 行（头内先平均）复制成全部 txt_len 行，交给
scripts/interpret/aggregate_attn.py::split_streams（只 import，不改）分流，于是得到「只看该词 query」时的帧流／motion 流。
在帧流里算裸露窗 VIS=[0,32) 与交换区间 SWAP=[64,64+50k) 的占比与均匀期望；motion 流按窗口与区间相交计。
逐层（layers=l）与全层平均都算；范围取 #15～#19 与全部 20 次推理。

问题 2（帧内空间位置）：probs_action 在 10 去噪步 × 18 层 × 8 头 × 20 动作 query 上平均（aggregate_attn.aggregate），
在 VIS 与 SWAP 里各取帧流质量最大的历史帧，把该帧 16 个 token（mem_order % 16 = 4×4 池化网格行优先序号）
的注意力用 cv2.INTER_NEAREST 放大到 256×256 原图半透明叠加（AGENTS.md 第 5 条，禁止插值）；
当前图像 base／wrist 各 256 token = SigLIP 16×16 网格同样处理。另一行用「green」文本 query 做同样叠图作对照。

输出到 --out（默认 docs/evaluation-doc/attn-ctx512-vs-motion-1103/analysis/x4/）：
fig_x4_text_vis_swap.png、fig_x4_text_layers.png、fig_x4_green_frames.png、fig_x4_overlay_<局>.png、
x4_summary.json；末行 X4_TEXT_SPATIAL=PASS|FAIL words=<n> overlays=<n> figs=<n>。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # scripts/interpret
from aggregate_attn import (FRAME_TOKENS, IMAGE_BASE, MOTION_WINDOW, SWAP_LEN, SWAP_START,  # noqa: E402
                            TOKENS_PER_FRAME, VISIBLE, aggregate, compute_swap_times, split_streams)
from plot_attn import CJK, plt  # noqa: E402  中文字体沿用 plot_attn 的 CJK

STAGE1 = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1")
EPISODES = [  # (标签, 模型目录, attn 局目录, 图注)
    ("ctx_ep4", "ctx", "1d677dadd3f34872a5779dbe8ab815e4", "ctx512 基线 ep4（失败）"),
    ("ctx_ep8", "ctx", "3625db1fb81a48af8aa5a136b3a7aa68", "ctx512 基线 ep8（成功）"),
    ("mot_ep8", "mot", "af10850c1f7d47549fcd8934178a7497", "ctx512+motion ep8（成功）"),
]
COLORS = {"ctx_ep4": "#9e9e9e", "ctx_ep8": "#4c72b0", "mot_ep8": "#dd8452"}
# 关心的词（全句只出现一次的实词；"the" 出现 3 次不取）；"全部文本" = 全部 txt_len 行平均。
WORDS = ["press", "buttons", "table", "pick", "container", "hiding", "green", "cube"]
ALL_TEXT = "全部文本"
FOCUS = (15, 19)   # #15～#19
KEY_INFER = 18     # ep8 关键推理 #18（step 288）
VIDEO_TOP = 208    # 视频 512×464：上方 208 行文字区，下方 256 行左 base 右 wrist
IMG = 256
ROWSUM_DEV = [0.0]  # 文本 query 行和与 1 的最大偏差（f16 量化）


# ---------------------------------------------------------------- 文本 token 定位

def token_table(prompt: str, txt_len: int) -> list[dict]:
    from mme_vla_suite.training.config import PaligemmaTokenizer

    tk = PaligemmaTokenizer(64)
    toks, mask = tk.tokenize(prompt, None)  # discrete_state_input=False（config.py 各 PI05 配置）
    n = int(mask.sum())
    if n != txt_len:
        raise ValueError(f"还原 token 数 {n} 与 npz txt_len={txt_len} 不符")
    sp = tk._tokenizer
    return [{"row": i, "id": int(toks[i]), "piece": sp.id_to_piece(int(toks[i]))} for i in range(n)]


def word_rows(table: list[dict]) -> dict[str, list[int]]:
    out = {}
    for w in WORDS:
        rows = [r["row"] for r in table if r["piece"].replace("▁", "").strip(",") == w]
        if len(rows) != 1:
            raise ValueError(f"词 {w!r} 命中 {len(rows)} 行，期望恰好 1 行")
        out[w] = rows
    out[ALL_TEXT] = [r["row"] for r in table]
    return out


# ---------------------------------------------------------------- 分流与占比

def _word_npz(z: dict, rows: list[int]) -> dict:
    """把 probs_text 的选定 query 行平均后复制到全部 txt_len 行，split_streams 的 query 平均即等于该词平均。"""
    pt = np.asarray(z["probs_text"]).astype(np.float32)
    n = int(z["txt_len"])
    m = pt[:, :, rows].astype(np.float64).mean(axis=2, keepdims=True)
    # 单层单词时 f16 存盘误差可让行和偏离 1 达 ~1.1e-4，超过 split_streams 的 1e-4 断言；
    # 逐（层,头）把行和归一到 1（只是 f16 量化修正，不改流内占比），偏差记入 ROWSUM_DEV。
    s = m.sum(axis=-1, keepdims=True)
    ROWSUM_DEV[0] = max(ROWSUM_DEV[0], float(np.abs(s - 1).max()))
    fake = np.repeat(m / s, n, axis=2)
    keys = ["P", "MEM", "mem_order", "mem_mask", "frame_times", "motion_times", "motion_mask",
            "txt_len", "step_idx", "infer_seq", "exec_start_idx"]
    return {**{k: z[k] for k in keys}, "probs_text": fake}


def _share(items: list[dict], lo: int, hi: int, width: int) -> tuple[float | None, float | None]:
    """width=1：帧 t∈[lo,hi)；width=33：窗口 [t,t+33) 与 [lo,hi) 相交。无该区间元素时返回 None。"""
    tot = sum(x["mass"] for x in items)
    hit = [x for x in items if x["t"] < hi and x["t"] + width > lo]
    if not items or not hit or tot <= 0:
        return None, None
    return sum(x["mass"] for x in hit) / tot, len(hit) / len(items)


def shares(st: dict, k: int) -> dict:
    swap = (SWAP_START, SWAP_START + SWAP_LEN * k)
    fv, fvu = _share(st["frame"], *VISIBLE, 1)
    fs, fsu = _share(st["frame"], *swap, 1)
    mv = mvu = ms = msu = None
    if st["has_motion"]:
        mv, mvu = _share(st["motion"], *VISIBLE, MOTION_WINDOW)
        ms, msu = _share(st["motion"], *swap, MOTION_WINDOW)
    m = st["mass"]
    return {"frame_vis": fv, "frame_vis_u": fvu, "frame_swap": fs, "frame_swap_u": fsu,
            "motion_vis": mv, "motion_vis_u": mvu, "motion_swap": ms, "motion_swap_u": msu,
            "mem_share": m["frame"] + m["motion"], "frame_mass": m["frame"], "motion_mass": m["motion"],
            "image_share": m["image_base"] + m["image_wrist"], "text_share": m["text"]}


def _mean(vals) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def summarize(per_infer: dict[int, dict], seqs) -> dict:
    keys = next(iter(per_infer.values())).keys()
    out = {k: _mean(per_infer[s][k] for s in seqs if s in per_infer) for k in keys}
    out["n_infer_frame_swap"] = sum(per_infer[s]["frame_swap"] is not None for s in seqs if s in per_infer)
    return out


# ---------------------------------------------------------------- 图像来源

class Frames:
    def __init__(self, attn_dir: Path, model: str, ep: int):
        self.attn_dir = attn_dir
        vids = glob.glob(str(STAGE1 / model / "run/attempts/*/client/videos" / f"ButtonUnmaskSwap_ep{ep}_*.mp4"))
        if len(vids) != 1:
            raise ValueError(f"视频应恰好 1 个，得到 {vids}")
        self.video = vids[0]
        cap = cv2.VideoCapture(self.video)
        self.F = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            self.F.append(f)

    def base(self, t: int) -> tuple[np.ndarray, str]:
        p = self.attn_dir / "frames" / f"{t}.jpg"
        if p.is_file():
            return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), "server jpg"
        return cv2.cvtColor(self.F[t][VIDEO_TOP:VIDEO_TOP + IMG, :IMG], cv2.COLOR_BGR2RGB), "视频帧"

    def wrist(self, t: int) -> np.ndarray:
        return cv2.cvtColor(self.F[t][VIDEO_TOP:VIDEO_TOP + IMG, IMG:2 * IMG], cv2.COLOR_BGR2RGB)


def colored_cubes(img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """HSV 阈值找红／绿／蓝方块外框（仅用于标注；面积 ≥ 12 px）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rng = {"green": [((45, 120, 80), (80, 255, 255))],
           "red": [((0, 150, 80), (6, 255, 255)), ((174, 150, 80), (180, 255, 255))],
           "blue": [((105, 150, 60), (130, 255, 255))]}
    out = []
    for name, rs in rng.items():
        m = sum(cv2.inRange(hsv, np.array(a), np.array(b)) for a, b in rs)
        n, _, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= 12:
                out.append((name, (int(x), int(y), int(w), int(h))))
    return out


# ---------------------------------------------------------------- 空间注意力

def memory_patches(a: np.ndarray, z: dict) -> np.ndarray:
    """(32, 16)：帧 f 的 4×4 网格序号 j（行优先）上的注意力质量。"""
    MEM = int(z["MEM"])
    mo = np.asarray(z["mem_order"], dtype=np.int64)
    mk = np.asarray(z["mem_mask"], dtype=bool)
    out = np.zeros((FRAME_TOKENS // TOKENS_PER_FRAME, TOKENS_PER_FRAME))
    sel = mk & (mo < FRAME_TOKENS)
    np.add.at(out, (mo[sel] // TOKENS_PER_FRAME, mo[sel] % TOKENS_PER_FRAME), a[:MEM][sel])
    return out


def pick_frames(patch: np.ndarray, frame_times: np.ndarray, k: int) -> dict:
    mass = patch.sum(1)
    res = {}
    for name, (lo, hi) in (("VIS", VISIBLE), ("SWAP", (SWAP_START, SWAP_START + SWAP_LEN * k))):
        cand = [f for f in range(len(frame_times)) if lo <= frame_times[f] < hi]
        if cand:
            f = max(cand, key=lambda i: mass[i])
            res[name] = {"f": int(f), "t": int(frame_times[f]), "mass": float(mass[f]),
                         "share_in_frame_stream": float(mass[f] / mass.sum())}
    return res


def containers(img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """低饱和高亮白色连通块（容器；S<60、V>140、面积 ≥ 320 px（ep8 t=111 实测容器 406～441、按钮 188～232），排除按钮等小白块与画面顶端机器人底座 y<40）。仅用于标注。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 140)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 320 and y >= 40 and w < 120 and h < 120:
            out.append(("container", (int(x), int(y), int(w), int(h))))
    return out


BOX_COLOR = {"green": "lime", "red": "red", "blue": "cyan", "container": "white"}


def overlay(ax, img: np.ndarray, grid: np.ndarray, title: str, boxes=()) -> dict:
    """grid (g,g) 归一化到帧内和 1，再除以均匀期望 1/g² 得富集倍数；最近邻放大到 256×256 半透明叠加。

    色标固定：4×4 记忆帧 [0,2]（均匀=1 居中），16×16 当前图 [0,max]；4×4 每格写倍数。返回最大块位置。
    """
    g = grid.shape[0]
    rel = grid / grid.sum() if grid.sum() > 0 else grid
    ratio = rel * g * g
    vmax = 2.0 if g == 4 else max(2.0, float(ratio.max()))
    big = cv2.resize(ratio.astype(np.float32), (IMG, IMG), interpolation=cv2.INTER_NEAREST)
    cmap = plt.get_cmap("coolwarm" if g == 4 else "inferno")
    heat = (cmap(np.clip(big / vmax, 0, 1))[..., :3] * 255).astype(np.float32)
    alpha = 0.45
    blend = ((1 - alpha) * img.astype(np.float32) + alpha * heat).clip(0, 255).astype(np.uint8)
    ax.imshow(blend, interpolation="nearest")
    step = IMG / g
    for i in range(1, g):
        ax.axhline(i * step - 0.5, color="w", lw=0.3 if g > 4 else 0.8, alpha=0.5)
        ax.axvline(i * step - 0.5, color="w", lw=0.3 if g > 4 else 0.8, alpha=0.5)
    if g == 4:
        for r_ in range(g):
            for c_ in range(g):
                ax.text(c_ * step + 4, r_ * step + 14, f"×{ratio[r_, c_]:.2f}", fontsize=7, color="k",
                        bbox=dict(boxstyle="square,pad=0.1", fc="w", ec="none", alpha=0.65))
    for name, (x, y, w, h) in boxes:
        ax.add_patch(plt.Rectangle((x - 2, y - 2), w + 4, h + 4, fill=False, lw=1.4,
                                   ec=BOX_COLOR[name], ls="--" if name == "container" else "-"))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("÷均匀期望", fontsize=7)
    r, c = np.unravel_index(int(np.argmax(rel)), rel.shape)
    ax.set_title(f"{title}\n最大块 (行{r},列{c}) = {rel[r, c] * 100:.1f}%（均匀 {100 / g / g:.2f}%）", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return {"argmax_rc": [int(r), int(c)], "argmax_share": float(rel[r, c]), "uniform": 1.0 / (g * g),
            "grid": rel.round(5).tolist()}


# ---------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parents[3] / "docs/evaluation-doc/attn-ctx512-vs-motion-1103/analysis/x4")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {"episodes": {}, "words": WORDS + [ALL_TEXT], "focus": list(FOCUS), "key_infer": KEY_INFER}
    data = {}
    table = None
    for tag, model, nonce, label in EPISODES:
        d = STAGE1 / model / "attn" / nonce
        meta = json.loads((d / "episode.json").read_text())
        npzs = {}
        for p in sorted(d.glob("infer_*.npz")):
            with np.load(p) as z:
                npzs[int(z["infer_seq"])] = {k: z[k] for k in z.files}
        txt_len = int(next(iter(npzs.values()))["txt_len"])
        tab = token_table(meta["prompt"], txt_len)
        if table is None:
            table = tab
        elif [r["id"] for r in tab] != [r["id"] for r in table]:
            raise ValueError("三局 prompt token 序列不一致")
        rows = word_rows(tab)
        k = compute_swap_times(meta["task_name"], int(meta["episode_id"]))
        L = np.asarray(next(iter(npzs.values()))["probs_text"]).shape[0]
        per = {w: {"all": {}, "layer": {l: {} for l in range(L)}} for w in rows}
        frames_green = {}
        for seq, z in sorted(npzs.items()):
            for w, rr in rows.items():
                wz = _word_npz(z, rr)
                st = split_streams(wz, query="text", layers="all", heads="all")
                per[w]["all"][seq] = shares(st, k)
                if w == "green":
                    frames_green[seq] = st["frame"]
                for l in range(L):
                    per[w]["layer"][l][seq] = shares(split_streams(wz, query="text", layers=str(l), heads="all"), k)
        rng = {"focus": [s for s in npzs if FOCUS[0] <= s <= FOCUS[1]], "global": sorted(npzs)}
        ep_sum = {"label": label, "nonce": nonce, "episode_id": int(meta["episode_id"]), "swap_times": k,
                  "outcome": meta.get("outcome"), "txt_len": txt_len,
                  "step_idx": {int(s): int(z["step_idx"]) for s, z in npzs.items()}, "words": {}}
        for w in rows:
            ep_sum["words"][w] = {
                r: {"all_layers": summarize(per[w]["all"], seqs),
                    "per_layer": [summarize(per[w]["layer"][l], seqs) for l in range(L)]}
                for r, seqs in rng.items()}
            ep_sum["words"][w]["per_infer_all_layers"] = {int(s): v for s, v in per[w]["all"].items()}
        summary["episodes"][tag] = ep_sum
        data[tag] = {"npzs": npzs, "k": k, "meta": meta, "dir": d, "model": model, "label": label,
                     "frames_green": frames_green, "rows": rows}
        print(f"X4_EP tag={tag} k={k} infers={len(npzs)} txt_len={txt_len}", flush=True)
    summary["token_table"] = table
    summary["text_rowsum_max_dev_before_renorm"] = ROWSUM_DEV[0]
    print(f"X4_ROWSUM_DEV={ROWSUM_DEV[0]:.3e}", flush=True)
    summary["word_rows"] = {w: r for w, r in data["ctx_ep8"]["rows"].items() if w != ALL_TEXT}

    figs = []
    words_plot = WORDS + [ALL_TEXT]

    # ---- 图 1：各词 VIS／SWAP 占比 vs 均匀期望
    rows_spec = [("frame_vis", "frame_vis_u", "帧流 · 裸露窗 VIS [0,32) 占比"),
                 ("frame_swap", "frame_swap_u", "帧流 · 交换区间 SWAP 占比"),
                 ("motion_swap", "motion_swap_u", "motion 流 · SWAP 窗口占比（仅 motion 模型）")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    x = np.arange(len(words_plot))
    for c, (rname, rtitle) in enumerate((("focus", "推理 #15～#19 平均"), ("global", "全部 20 次推理平均"))):
        for r, (key, ukey, ttl) in enumerate(rows_spec):
            ax = axes[r, c]
            tags = [t_ for t_ in data if key != "motion_swap" or t_.startswith("mot")]
            wbar = 0.8 / len(tags)
            for i, tag in enumerate(tags):
                vals = [summary["episodes"][tag]["words"][w][rname]["all_layers"][key] or 0 for w in words_plot]
                us = [summary["episodes"][tag]["words"][w][rname]["all_layers"][ukey] or 0 for w in words_plot]
                xs = x + (i - (len(tags) - 1) / 2) * wbar
                ax.bar(xs, vals, wbar * 0.95, color=COLORS[tag], label=data[tag]["label"])
                ax.scatter(xs, us, marker="_", s=180, color="k", zorder=3, label="均匀期望" if i == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(words_plot, fontsize=9)
            ax.set_ylabel("占帧流（或 motion 流）质量比例")
            ax.set_title(f"{ttl} · {rtitle}", fontsize=10)
            ax.legend(fontsize=8, loc="upper left", ncol=2)
            ax.set_ylim(0, max(0.05, ax.get_ylim()[1] * 1.25))
    fig.suptitle("文本词 query（prefill，18 层×8 头平均）看向历史帧：窗口内占比（柱）与均匀期望（黑横线）；ep8 k=2，ep4 k=1", fontsize=12)
    p = args.out / "fig_x4_text_vis_swap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figs.append(p)

    # ---- 图 2：逐层 lift（占比 / 均匀）——green 与 cube，ep8 两模型，#15～#19
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for c, (key, ukey, ttl) in enumerate(rows_spec):
        for r, w in enumerate(("green", "cube")):
            ax = axes[r, c]
            for tag in ("ctx_ep8", "mot_ep8", "ctx_ep4"):
                if key == "motion_swap" and not tag.startswith("mot"):
                    continue
                pl = summary["episodes"][tag]["words"][w]["focus"]["per_layer"]
                lift = [(q[key] / q[ukey]) if q[key] is not None and q[ukey] else np.nan for q in pl]
                ax.plot(range(len(pl)), lift, "-o", ms=3.5, color=COLORS[tag], label=data[tag]["label"])
            ax.axhline(1.0, color="k", ls="--", lw=1, label="均匀 = 1")
            ax.set_title(f"“{w}” · {ttl}", fontsize=10)
            ax.set_xlabel("层")
            ax.set_ylabel("占比 / 均匀期望")
            ax.set_xticks(range(0, 18, 2))
            ax.legend(fontsize=8)
    fig.suptitle("逐层：文本词 query 在 VIS／SWAP 上的富集倍数（推理 #15～#19 平均）", fontsize=12)
    p = args.out / "fig_x4_text_layers.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figs.append(p)

    # ---- 图 3：“green” 在 #15～#19 的帧流分布
    fig, axes = plt.subplots(5, 2, figsize=(15, 12), sharex=True, constrained_layout=True)
    for c, tag in enumerate(("ctx_ep8", "mot_ep8")):
        k = data[tag]["k"]
        for r, seq in enumerate(range(FOCUS[0], FOCUS[1] + 1)):
            ax = axes[r, c]
            fr = data[tag]["frames_green"][seq]
            tot = sum(q["mass"] for q in fr)
            ax.axvspan(*VISIBLE, color="#b7e4c7", alpha=0.8, lw=0)
            ax.axvspan(SWAP_START, SWAP_START + SWAP_LEN * k, color="#fbd38d", alpha=0.7, lw=0)
            for j in range(1, k):
                ax.axvline(SWAP_START + SWAP_LEN * j, color="#c05621", lw=0.8, ls=":")
            ys = [q["mass"] / tot for q in fr]
            ax.bar([q["t"] for q in fr], ys, width=6, color=COLORS[tag])
            ax.set_ylim(0, max(ys) * 1.6)  # 上沿留白给角标，不压柱
            sh = summary["episodes"][tag]["words"]["green"]["per_infer_all_layers"][seq]
            ax.text(0.99, 0.95, f"VIS {sh['frame_vis'] * 100:.1f}%（均匀 {sh['frame_vis_u'] * 100:.1f}%）\n"
                    f"SWAP {sh['frame_swap'] * 100:.1f}%（均匀 {sh['frame_swap_u'] * 100:.1f}%）\n"
                    f"记忆区占全部 {sh['mem_share'] * 100:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.6", alpha=0.9))
            ax.set_ylabel(f"#{seq}\nstep {int(data[tag]['npzs'][seq]['step_idx'])}", fontsize=9)
            if r == 0:
                ax.set_title(f"{data[tag]['label']}：“green” query 的帧流分布（帧内归一）", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("全域帧号（绿底 VIS [0,32)，橙底 SWAP）")
    p = args.out / "fig_x4_green_frames.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figs.append(p)

    # ---- 图 4：空间叠图（#18）
    n_overlays = 0
    summary["spatial"] = {}
    for tag, dd in data.items():
        z = dd["npzs"][KEY_INFER]
        k = dd["k"]
        step = int(z["step_idx"])
        MEM = int(z["MEM"])
        fs = Frames(dd["dir"], dd["model"], int(dd["meta"]["episode_id"]))
        a_act = aggregate(z, query="action")
        a_grn = aggregate(_word_npz(z, dd["rows"]["green"]), query="text")
        ft = np.asarray(z["frame_times"])
        patch_act = memory_patches(a_act, z)
        patch_grn = memory_patches(a_grn, z)
        picks = pick_frames(patch_act, ft, k)
        fig, axes = plt.subplots(2, 4, figsize=(17, 9.4), constrained_layout=True)
        sp = {"step_idx": step, "picks": picks, "panels": {}, "top_frames": {}}
        # sink 诊断：帧流 top-3 帧及其帧内最大 token 占比（高度集中于单 token 提示 attention sink 而非语义）
        for qn, patch in (("action", patch_act), ("green", patch_grn)):
            fm = patch.sum(1)
            order = [f for f in np.argsort(-fm) if ft[f] >= 0][:3]
            sp["top_frames"][qn] = [{"t": int(ft[f]), "share": float(fm[f] / fm.sum()),
                                     "max_token_in_frame": float(patch[f].max() / patch[f].sum()),
                                     "max_token_j": int(np.argmax(patch[f]))} for f in order]
        for r, (qname, a, patch) in enumerate((("动作 query", a_act, patch_act), ("“green” 文本 query", a_grn, patch_grn))):
            col = 0
            for wname in ("VIS", "SWAP"):
                ax = axes[r, col]
                col += 1
                if wname not in picks:
                    ax.axis("off")
                    continue
                f, t_ = picks[wname]["f"], picks[wname]["t"]
                img, src = fs.base(t_)
                grid = patch[f].reshape(4, 4)
                m_share = patch[f].sum() / patch.sum()
                info = overlay(ax, img, grid, f"{qname} · {wname} 中动作质量最大帧 t={t_}（{src}）\n"
                               f"该帧占帧流 {m_share * 100:.1f}%", colored_cubes(img) + containers(img))
                sp["panels"][f"{qname}|{wname}"] = {"t": t_, "src": src, "frame_share": float(m_share), **info}
                n_overlays += 1
            cur_base = fs.base(step)[0] if (dd["dir"] / "frames" / f"{step}.jpg").is_file() else \
                cv2.cvtColor(fs.F[step][VIDEO_TOP:VIDEO_TOP + IMG, :IMG], cv2.COLOR_BGR2RGB)
            for j, (vname, img) in enumerate((("当前 base", cur_base), ("当前 wrist", fs.wrist(step)))):
                ax = axes[r, 2 + j]
                g = a[MEM + j * IMAGE_BASE: MEM + (j + 1) * IMAGE_BASE].reshape(16, 16)
                v_share = g.sum() / a[:int(z["P"])].sum()
                info = overlay(ax, img, g, f"{qname} · {vname}（step {step}，视频帧）\n该视角占 prefix {v_share * 100:.1f}%",
                               colored_cubes(img) + (containers(img) if j == 0 else []))
                sp["panels"][f"{qname}|{vname}"] = {"t": step, "view_share": float(v_share), **info}
                n_overlays += 1
        fig.suptitle(f"{dd['label']} · 推理 #{KEY_INFER}（step {step}）：帧内注意力（最近邻放大，无插值；"
                     f"记忆帧 4×4、当前图 16×16；实线框＝HSV 检出的红/绿/蓝方块，白虚线框＝白色容器）", fontsize=11)
        p = args.out / f"fig_x4_overlay_{tag}.png"
        fig.savefig(p, dpi=105)
        plt.close(fig)
        figs.append(p)
        summary["spatial"][tag] = sp

    (args.out / "x4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    ok = all(p.is_file() for p in figs) and n_overlays > 0 and CJK is not None
    print(f"X4_TEXT_SPATIAL={'PASS' if ok else 'FAIL'} words={len(WORDS)} overlays={n_overlays} figs={len(figs)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
