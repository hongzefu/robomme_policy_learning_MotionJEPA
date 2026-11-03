"""X1：按 (层, 头) 拆开看「回看裸露窗 VIS、回看交换区间 SWAP」由谁在做（计划 1103 第二部分 §1.3、§2.3）。

对每局每次推理的 probs_action (10 去噪步, 18 层, 8 头, 20 动作 query, P+20 key)：
先在全部去噪步与全部 20 个动作 query 上平均，得到每 (层, 头) 一条 key 向量；
再把这条向量包成单层单头的伪 npz，原样交给 aggregate_attn.split_streams 分流（复用 §2.3 映射与断言，不另写一套）。

每 (层, 头) 的指标：
- 帧流质量 M_f、motion 流质量 M_m（占该头全部注意力的比例）；
- 帧流 VIS 富集 = (帧流内 VIS 帧质量 / M_f) ÷ (VIS 帧数 / 有效帧数)；帧流 SWAP 富集同理；
- motion 流 SWAP 富集：窗 w 的权重 o_w = |[t_w, t_w+33) ∩ SWAP| / 33，
  占比 = Σ m_w·o_w / M_m，均匀期望 = Σ o_w / 窗数（不是沾边就整窗算）；motion 流 VIS 富集同理；
- 超额质量 = M × (占比 − 均匀期望)：该头全部注意力里「比均匀分配多投给区间」的那部分，
  富集倍数与质量的乘积型指标，富集 = 1 时恰为 0，质量极小时自动趋零，用于排 top-10。
质量 < LOW_MASS 的格在热图上加灰色斜线，富集倍数不作解读。

末行判定 X1_LAYER_HEAD=PASS models=<n> episodes=<n> figs=<n>。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from aggregate_attn import MOTION_WINDOW, VISIBLE, split_streams, swap_interval  # noqa: E402
import plot_attn  # noqa: E402  导入即按 _pick_cjk_font 结果设置中文字体

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

CJK = plot_attn.CJK
N_LAYERS, N_HEADS = 18, 8
LOW_MASS = 0.01      # 流质量低于 1% 的头，富集倍数不解读
KEY_INFER = 18       # 「揭开」关键推理
NEIGHBORS = (15, 16, 17, 18, 19)
TOPN = 10
LOG2_CLIP = 3.0


# ---------------------------------------------------------------- 计算

def _overlap(t0: int, length: int, lo: int, hi: int) -> int:
    return max(0, min(t0 + length, hi) - max(t0, lo))


def _interval_stats(items: list[dict], lo: int, hi: int, width: int) -> tuple[float, float]:
    """返回 (占比, 均匀期望)；每项权重 = 与 [lo,hi) 重叠长度 / width（帧 width=1 即 0/1）。"""
    if not items:
        return float("nan"), float("nan")
    w = np.array([_overlap(x["t"], width, lo, hi) / width for x in items])
    m = np.array([x["mass"] for x in items])
    tot = m.sum()
    share = float((m * w).sum() / tot) if tot > 0 else float("nan")
    return share, float(w.mean())


def _enrich(share: float, uni: float) -> float:
    if not np.isfinite(share) or not np.isfinite(uni) or uni <= 0 or uni >= 1:
        return float("nan")   # 区间为空或覆盖全部记忆时富集无意义
    return share / uni


def infer_metrics(npz, k: int) -> dict:
    """一次推理 → 各指标 (18,8) 数组。"""
    x = np.asarray(npz["probs_action"], dtype=np.float32).mean(axis=(0, 3))  # (18,8,P+20)
    base = {key: npz[key] for key in npz.files if key not in ("probs_action", "probs_text")}
    swap = swap_interval(k)
    names = ["m_frame", "m_motion", "m_image", "m_text",
             "f_vis_share", "f_vis_uni", "f_swap_share", "f_swap_uni",
             "m_swap_share", "m_swap_uni", "m_vis_share", "m_vis_uni"]
    out = {n: np.full((N_LAYERS, N_HEADS), np.nan) for n in names}
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            fake = dict(base)
            fake["probs_action"] = x[li, hi].astype(np.float64)[None, None, None, None, :]
            s = split_streams(fake)
            ms = s["mass"]
            out["m_frame"][li, hi] = ms["frame"]
            out["m_motion"][li, hi] = ms["motion"]
            out["m_image"][li, hi] = ms["image_base"] + ms["image_wrist"]
            out["m_text"][li, hi] = ms["text"]
            for tag, (lo, hh) in (("vis", VISIBLE), ("swap", swap)):
                sh, un = _interval_stats(s["frame"], lo, hh, 1)
                out[f"f_{tag}_share"][li, hi], out[f"f_{tag}_uni"][li, hi] = sh, un
                if s["motion"]:
                    sh, un = _interval_stats(s["motion"], lo, hh, MOTION_WINDOW)
                    out[f"m_{tag}_share"][li, hi], out[f"m_{tag}_uni"][li, hi] = sh, un
    for st, tag in (("f", "vis"), ("f", "swap"), ("m", "swap"), ("m", "vis")):
        sh, un = out[f"{st}_{tag}_share"], out[f"{st}_{tag}_uni"]
        mass = out["m_frame"] if st == "f" else out["m_motion"]
        en = np.vectorize(_enrich)(sh, un)
        out[f"{st}_{tag}_enrich"] = en
        out[f"{st}_{tag}_excess"] = np.where(np.isfinite(en), mass * (sh - un), np.nan)
    out["step_idx"] = int(npz["step_idx"])
    return out


def load_episode(path: Path, k: int) -> list[dict]:
    files = sorted(path.glob("infer_*.npz"))
    res = []
    for f in files:
        with np.load(f) as z:
            res.append(infer_metrics(z, k))
    return res


# ---------------------------------------------------------------- 画图

def _hatch_low(ax, mass: np.ndarray) -> None:
    for li, hi in zip(*np.where(mass < LOW_MASS)):
        ax.add_patch(plt.Rectangle((hi - 0.5, li - 0.5), 1, 1, fill=False, hatch="///",
                                   edgecolor="0.45", lw=0))


def _heat(ax, arr, *, kind: str, title: str, mass=None):
    ax.set_title(title, fontsize=9)
    if arr is None:
        ax.set_facecolor("0.93")
        ax.text(3.5, 8.5, "无 motion 流" if CJK else "no motion", ha="center", va="center", fontsize=9)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(17.5, -0.5)
        im = None
    elif kind == "mass":
        im = ax.imshow(np.log10(np.clip(arr, 1e-4, 1)), cmap="viridis", vmin=-4, vmax=0,
                       interpolation="nearest", aspect="auto")
    else:
        v = np.clip(np.log2(arr), -LOG2_CLIP, LOG2_CLIP)
        im = ax.imshow(np.ma.masked_invalid(v), cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-LOG2_CLIP, vmax=LOG2_CLIP),
                       interpolation="nearest", aspect="auto")
        if mass is not None:
            _hatch_low(ax, mass)
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(0, N_LAYERS, 2))
    ax.set_xlabel("头" if CJK else "head", fontsize=8)
    ax.set_ylabel("层" if CJK else "layer", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def fig_heat(eps: dict, label_a: str, label_b: str, infer: int, path: Path) -> None:
    rows = [label_a, label_b]
    cols = [("m_frame", "mass", "帧流质量 log10", None),
            ("f_vis_enrich", "enr", "帧流 VIS 富集 log2", "m_frame"),
            ("f_swap_enrich", "enr", "帧流 SWAP 富集 log2", "m_frame"),
            ("m_motion", "mass", "motion 流质量 log10", None),
            ("m_swap_enrich", "enr", "motion 流 SWAP 富集 log2", "m_motion")]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8.6), constrained_layout=True)
    ims = {}
    for r, lab in enumerate(rows):
        d = eps[lab]["infers"][infer]
        has_mot = eps[lab]["has_motion"]
        for c, (key, kind, ttl, mk) in enumerate(cols):
            arr = d[key]
            if key.startswith("m_") and key != "m_frame" and not has_mot:
                arr = None
            im = _heat(axes[r, c], arr, kind=kind, title=f"{eps[lab]['title']}\n{ttl}",
                       mass=d[mk] if mk else None)
            if im is not None:
                ims[kind] = im
    fig.colorbar(ims["mass"], ax=axes[:, [0, 3]].ravel().tolist(), shrink=0.5, label="log10 质量")
    fig.colorbar(ims["enr"], ax=axes[:, [1, 2, 4]].ravel().tolist(), shrink=0.5,
                 label="log2 富集倍数（0 = 均匀；斜线 = 流质量 < 1%）")
    fig.suptitle(f"ButtonUnmaskSwap ep8 推理 #{infer}（step {eps[label_a]['infers'][infer]['step_idx']}）"
                 f"  逐 (层,头) 质量与富集；全去噪步 × 20 动作 query 平均", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def topn(arr: np.ndarray, n: int = TOPN) -> list[tuple[int, int, float]]:
    a = np.where(np.isfinite(arr), arr, -np.inf)
    idx = np.argsort(a.ravel())[::-1][:n]
    return [(int(i // N_HEADS), int(i % N_HEADS), float(a.ravel()[i])) for i in idx]


def fig_stability(eps: dict, specs: list[tuple[str, str, str, list]], path: Path) -> None:
    """每个子图：一条指标 × 该指标 top 头，横轴 20 次推理，#15～#19 浅底。"""
    fig, axes = plt.subplots(len(specs), 2, figsize=(14, 3.3 * len(specs)), constrained_layout=True,
                             squeeze=False)
    for r, (lab, key, ttl, heads) in enumerate(specs):
        infers = eps[lab]["infers"]
        xs = np.arange(len(infers))
        for c, (suffix, ylab) in enumerate((("enrich", "富集倍数"), ("excess", "超额质量"))):
            ax = axes[r, c]
            k2 = key.replace("excess", suffix)
            ax.axvspan(NEIGHBORS[0] - 0.5, NEIGHBORS[-1] + 0.5, color="#fbd38d", alpha=0.35, lw=0)
            ax.axvline(KEY_INFER, color="0.4", ls="--", lw=0.8)
            if suffix == "enrich":
                ax.axhline(1, color="0.5", lw=0.8)
            cmap = plt.get_cmap("tab10")
            for j, (li, hi, _) in enumerate(heads):
                ys = [d[k2][li, hi] for d in infers]
                ax.plot(xs, ys, marker="o", ms=2.5, lw=1, color=cmap(j % 10), label=f"L{li}H{hi}")
            ax.set_title(f"{eps[lab]['title']}：{ttl}（top-{len(heads)} 头）{ylab}", fontsize=9)
            ax.set_xticks(xs)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("推理序号（橙底 #15～#19，虚线 #18 揭开）", fontsize=8)
            if suffix == "enrich":
                ax.set_yscale("log")
        axes[r, 1].legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def head_streams(path: Path, li: int, hi: int) -> dict:
    """单头分流：全去噪步 × 20 动作 query 平均后交给 split_streams。"""
    with np.load(path) as z:
        fake = {k: z[k] for k in z.files if k not in ("probs_action", "probs_text")}
        vec = np.asarray(z["probs_action"][:, li, hi], dtype=np.float64).mean(axis=(0, 1))
    fake["probs_action"] = vec[None, None, None, None, :]
    return split_streams(fake)


def fig_memory_heads(eps: dict, tops: dict, labs: list[str], path: Path) -> None:
    """#18 记忆质量 top-3 头：帧流逐帧柱（[0,32) 绿底、交换区间橙底），motion 模型叠画 motion 窗。"""
    fig, axes = plt.subplots(len(labs), 3, figsize=(16, 3.8 * len(labs)), constrained_layout=True,
                             squeeze=False)
    for r, lab in enumerate(labs):
        e = eps[lab]
        lo, hh = swap_interval(e["k"])
        d = e["infers"][KEY_INFER]
        for c, (li, hi, v) in enumerate(tops[lab]["mem"][:3]):
            ax = axes[r, c]
            st = head_streams(Path(e["path"]) / f"infer_{KEY_INFER:04d}.npz", li, hi)
            ax.axvspan(*VISIBLE, color=plot_attn.GREEN, alpha=0.8, lw=0)
            ax.axvspan(lo, hh, color=plot_attn.ORANGE, alpha=0.6, lw=0)
            ts = [q["t"] for q in st["frame"]]
            ax.bar(ts, [q["mass"] for q in st["frame"]], width=6, color="C0", label="帧流（每帧 16 token 之和）")
            for j, q in enumerate(st["motion"]):
                ax.hlines(q["mass"], q["t"], q["t"] + MOTION_WINDOW, color="C3", lw=2.2,
                          label="motion 窗（横线跨 33 帧）" if j == 0 else None)
            ax.set_title(f"{e['title']} L{li}H{hi}：记忆质量 {v:.3f}"
                         f"（帧 {d['m_frame'][li, hi]:.3f}，motion {d['m_motion'][li, hi]:.3f}）\n"
                         f"帧流 VIS 富集 {d['f_vis_enrich'][li, hi]:.2f}，帧流 SWAP 富集 {d['f_swap_enrich'][li, hi]:.2f}",
                         fontsize=9)
            ax.set_xlabel("全域帧号（绿底 VIS [0,32)，橙底 SWAP）", fontsize=8)
            ax.set_ylabel("注意力质量", fontsize=8)
            ax.set_xlim(-8, max(ts) + 40)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"ep8 推理 #{KEY_INFER}：记忆质量（帧流+motion 流）最大的 3 个头分别在看哪些帧", fontsize=11)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_ep4_vs_ep8(eps: dict, a: str, b: str, heads: list, path: Path) -> dict:
    """基线 ep4 vs ep8：逐头整局平均 VIS 超额质量散点 + top 头在两局的逐推理曲线。"""
    ea = np.nanmean(np.stack([d["f_vis_excess"] for d in eps[a]["infers"]]), axis=0)
    eb = np.nanmean(np.stack([d["f_vis_excess"] for d in eps[b]["infers"]]), axis=0)
    ra = np.nanmean(np.stack([d["f_vis_enrich"] for d in eps[a]["infers"]]), axis=0)
    rb = np.nanmean(np.stack([d["f_vis_enrich"] for d in eps[b]["infers"]]), axis=0)
    r_exc = float(np.corrcoef(ea.ravel(), eb.ravel())[0, 1])
    r_enr = float(np.corrcoef(np.log2(ra).ravel(), np.log2(rb).ravel())[0, 1])
    top_a = {(li, hi) for li, hi, _ in topn(ea)}
    top_b = {(li, hi) for li, hi, _ in topn(eb)}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.scatter(eb.ravel(), ea.ravel(), s=10, c="0.5", label="其余头")
    hl = np.array([[eb[li, hi], ea[li, hi]] for li, hi, _ in heads])
    ax.scatter(hl[:, 0], hl[:, 1], s=26, c="C3", label="ep8 #18 VIS top-10 头")
    # 负值端最远的两个点标名（通常是只看当前帧的大质量头）
    for i in np.argsort(eb.ravel())[:2]:
        li, hi = int(i // N_HEADS), int(i % N_HEADS)
        ax.annotate(f"L{li}H{hi}", (eb[li, hi], ea[li, hi]), fontsize=8, xytext=(6, -4),
                    textcoords="offset points")
    lim = [min(ea.min(), eb.min()), max(ea.max(), eb.max())]
    ax.plot(lim, lim, color="0.6", lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("ep8（成功）整局平均 帧流 VIS 超额质量")
    ax.set_ylabel("ep4（失败）整局平均 帧流 VIS 超额质量")
    ax.set_title(f"144 个 (层,头)：Pearson r = {r_exc:.3f}；top-10 交集 {len(top_a & top_b)}/10", fontsize=9)
    ax = axes[1]
    ax.scatter(np.log2(rb).ravel(), np.log2(ra).ravel(), s=10, c="0.5")
    ax.set_xlabel("ep8 整局平均 帧流 VIS 富集 log2")
    ax.set_ylabel("ep4 整局平均 帧流 VIS 富集 log2")
    ax.set_title(f"富集倍数（log2）Pearson r = {r_enr:.3f}", fontsize=9)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax = axes[2]
    for j, (li, hi, _) in enumerate(heads[:5]):
        for lab, ls in ((a, "--"), (b, "-")):
            ys = [d["f_vis_enrich"][li, hi] for d in eps[lab]["infers"]]
            ax.plot(range(len(ys)), ys, ls=ls, color=f"C{j}", lw=1, marker="o", ms=2,
                    label=f"L{li}H{hi} {'ep4' if lab == a else 'ep8'}")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("推理序号")
    ax.set_xticks(range(0, 20, 2))
    ax.set_ylabel("帧流 VIS 富集倍数")
    ax.set_title("ep8 top-5 头：实线 ep8、虚线 ep4", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("基线 ctx512：失败局 ep4（k=1）vs 成功局 ep8（k=2）同头对比", fontsize=11)
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return {"pearson_excess": r_exc, "pearson_log2_enrich": r_enr,
            "top10_overlap": len(top_a & top_b),
            "top10_ep4": sorted(top_a), "top10_ep8": sorted(top_b)}


# ---------------------------------------------------------------- 主流程

def _summ_heads(d: dict, keys: list[str], heads) -> list[dict]:
    return [{"layer": li, "head": hi, **{k: float(d[k][li, hi]) for k in keys}} for li, hi, *_ in heads]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ep", action="append", required=True,
                    help="标签=目录:k:标题，如 base_ep8=/path/to/attn/<nonce>:2:ctx512 基线 ep8")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    a.out.mkdir(parents=True, exist_ok=True)

    eps: dict[str, dict] = {}
    for spec in a.ep:
        lab, rest = spec.split("=", 1)
        path, k, title = rest.split(":", 2)
        infers = load_episode(Path(path), int(k))
        meta = json.loads((Path(path) / "episode.json").read_text())
        eps[lab] = {"infers": infers, "k": int(k), "title": title, "path": path,
                    "has_motion": bool(meta.get("motion_enabled")), "episode_id": meta["episode_id"],
                    "outcome": meta.get("outcome")}
        print(f"已载入 {lab}: {len(infers)} 次推理，k={k}，motion={eps[lab]['has_motion']}", flush=True)

    figs = []
    p = a.out / "fig_x1_heat_ep8_infer18.png"
    fig_heat(eps, "base_ep8", "mot_ep8", KEY_INFER, p)
    figs.append(p)

    summary: dict = {"low_mass": LOW_MASS, "key_infer": KEY_INFER, "episodes": {}}
    tops: dict[str, dict] = {}
    for lab, e in eps.items():
        d = e["infers"][KEY_INFER]
        allm = {key: np.nanmean(np.stack([x[key] for x in e["infers"]]), axis=0)
                for key in e["infers"][0] if key != "step_idx"}
        info = {"episode_id": e["episode_id"], "outcome": e["outcome"], "k": e["k"],
                "step_idx": [x["step_idx"] for x in e["infers"]]}
        metrics = ["f_vis_excess", "f_swap_excess"] + (["m_swap_excess", "m_vis_excess"] if e["has_motion"] else [])
        keys_out = ["m_frame", "m_motion", "m_image", "m_text", "f_vis_enrich", "f_vis_excess",
                    "f_swap_enrich", "f_swap_excess"] + (["m_swap_enrich", "m_swap_excess"] if e["has_motion"] else [])
        tops[lab] = {}
        for m in metrics:
            t18 = topn(d[m])
            tall = topn(allm[m])
            tops[lab][m] = t18
            info[f"top10_infer18_{m}"] = _summ_heads(d, keys_out, t18)
            info[f"top10_episode_{m}"] = _summ_heads(allm, keys_out, tall)
            # 稳定性：top 头在 #15～#19 与全局 20 次推理中仍进 top-10 的次数
            stab = []
            for li, hi, _ in t18:
                ranks = []
                for x in e["infers"]:
                    tt = {(q[0], q[1]) for q in topn(x[m])}
                    ranks.append((li, hi) in tt)
                stab.append({"layer": li, "head": hi,
                             "in_top10_15_19": int(sum(ranks[i] for i in NEIGHBORS)),
                             "in_top10_all20": int(sum(ranks))})
            info[f"stability_{m}"] = stab
        # 头级总量：整头/层聚合，用于「有没有记忆头」
        mem = d["m_frame"] + d["m_motion"]
        info["infer18_mem_mass_by_layer"] = mem.mean(axis=1).tolist()
        info["infer18_mem_mass_max"] = {"layer": int(np.unravel_index(np.argmax(mem), mem.shape)[0]),
                                        "head": int(np.unravel_index(np.argmax(mem), mem.shape)[1]),
                                        "mass": float(mem.max())}
        info["infer18_mem_mass_quantiles"] = np.quantile(mem, [0.1, 0.5, 0.9, 1.0]).tolist()
        info["infer18_heads_mem_ge_0.3"] = int((mem >= 0.3).sum())
        info["infer18_low_mass_frame_heads"] = int((d["m_frame"] < LOW_MASS).sum())
        if e["has_motion"]:
            info["infer18_low_mass_motion_heads"] = int((d["m_motion"] < LOW_MASS).sum())
        # 富集与质量相关：高 VIS 富集是否集中在低质量头
        fin = np.isfinite(d["f_vis_enrich"])
        info["infer18_corr_log2visenrich_vs_log10mass"] = float(np.corrcoef(
            np.log2(d["f_vis_enrich"][fin]), np.log10(d["m_frame"][fin]))[0, 1])
        # 「全头平均」对照（等价于聚合后再算，与 X0 柱图同口径）
        info["infer18_all_head_vis_share"] = float(
            (d["m_frame"] * d["f_vis_share"]).sum() / d["m_frame"].sum())
        info["infer18_vis_uniform"] = float(np.nanmean(d["f_vis_uni"]))
        if e["has_motion"]:
            info["infer18_all_head_motion_swap_share"] = float(
                np.nansum(d["m_motion"] * d["m_swap_share"]) / d["m_motion"].sum())
            info["infer18_motion_swap_uniform"] = float(np.nanmean(d["m_swap_uni"]))
        # top-10 头贡献了全部头 VIS 超额质量之和的多少
        exc = d["f_vis_excess"]
        pos = np.nansum(np.clip(exc, 0, None))
        info["infer18_top10_share_of_positive_vis_excess"] = float(
            sum(max(v, 0) for *_, v in tops[lab]["f_vis_excess"]) / pos) if pos > 0 else None
        # 记忆头：#18 记忆质量（帧流+motion 流）最大的 5 个头，它们看哪里
        mh = topn(mem, 5)
        info["infer18_memory_heads"] = []
        for li, hi, v in mh:
            st = head_streams(Path(e["path"]) / f"infer_{KEY_INFER:04d}.npz", li, hi)
            fr = sorted(st["frame"], key=lambda q: -q["mass"])
            last4 = sum(q["mass"] for q in sorted(st["frame"], key=lambda q: q["t"])[-4:])
            info["infer18_memory_heads"].append({
                "layer": li, "head": hi, "mem_mass": v, "m_frame": float(d["m_frame"][li, hi]),
                "m_motion": float(d["m_motion"][li, hi]),
                "f_vis_enrich": float(d["f_vis_enrich"][li, hi]),
                "f_swap_enrich": float(d["f_swap_enrich"][li, hi]),
                "m_swap_enrich": float(d["m_swap_enrich"][li, hi]),
                "frame_peak_t": [q["t"] for q in fr[:3]],
                "frame_last4_share": float(last4 / max(st["mass"]["frame"], 1e-12))})
        # 记忆头整局稳定性：#18 的 top-5 记忆头在 20 次推理中仍为记忆质量 top-5 的次数
        mem_series = [x["m_frame"] + x["m_motion"] for x in e["infers"]]
        info["memory_heads_in_top5_all20"] = [
            {"layer": li, "head": hi,
             "count": int(sum((li, hi) in {(q[0], q[1]) for q in topn(m5, 5)} for m5 in mem_series))}
            for li, hi, _ in mh]
        tops[lab]["mem"] = mh
        summary["episodes"][lab] = info

    specs = [("base_ep8", "f_vis_excess", "帧流 VIS", tops["base_ep8"]["f_vis_excess"]),
             ("mot_ep8", "f_vis_excess", "帧流 VIS", tops["mot_ep8"]["f_vis_excess"]),
             ("mot_ep8", "m_swap_excess", "motion 流 SWAP", tops["mot_ep8"]["m_swap_excess"]),
             ("base_ep8", "f_swap_excess", "帧流 SWAP", tops["base_ep8"]["f_swap_excess"])]
    p = a.out / "fig_x1_stability_ep8.png"
    fig_stability(eps, specs, p)
    figs.append(p)

    p = a.out / "fig_x1_memory_heads_infer18.png"
    fig_memory_heads(eps, tops, ["base_ep8", "mot_ep8"], p)
    figs.append(p)

    p = a.out / "fig_x1_base_ep4_vs_ep8.png"
    summary["base_ep4_vs_ep8"] = fig_ep4_vs_ep8(eps, "base_ep4", "base_ep8", tops["base_ep8"]["f_vis_excess"], p)
    figs.append(p)

    # 跨模型：同一 (层,头) 的 VIS 超额质量在两模型 ep8 #18 的相关
    xb = eps["base_ep8"]["infers"][KEY_INFER]["f_vis_excess"].ravel()
    xm = eps["mot_ep8"]["infers"][KEY_INFER]["f_vis_excess"].ravel()
    summary["base_vs_mot_ep8_infer18_pearson_vis_excess"] = float(np.corrcoef(xb, xm)[0, 1])
    tb = {(q[0], q[1]) for q in tops["base_ep8"]["f_vis_excess"]}
    tm = {(q[0], q[1]) for q in tops["mot_ep8"]["f_vis_excess"]}
    summary["base_vs_mot_ep8_infer18_top10_vis_overlap"] = sorted(tb & tm)
    # motion 模型：VIS 头与 SWAP 头是否同一批
    ts = {(q[0], q[1]) for q in tops["mot_ep8"]["m_swap_excess"]}
    summary["mot_ep8_infer18_top10_vis_vs_motion_swap_overlap"] = sorted(tm & ts)

    summary["figs"] = [str(x.name) for x in figs]
    (a.out / "x1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    n_models = len({lab.split("_")[0] for lab in eps})
    print(f"X1_LAYER_HEAD=PASS models={n_models} episodes={len(eps)} figs={len(figs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
