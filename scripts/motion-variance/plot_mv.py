#!/usr/bin/env python3
"""motion-variance 出图（计划第二部分 §10）。数据一律现读 records/*.json 与 open_loop.json，不手抄。

骨架抄 scripts/training/legacy-eval/plot_eval_success_by_length.py：matplotlib Agg、DejaVu Sans → DroidSansFallbackFull 中文回退链
（必须写进 font.family 列表）、末尾自检判定行 PLOT_CELL_CHECK。
六组图 → docs/motion-utilization/figures/fig<i>-*.{png,svg}：
  fig1 设计示意；fig2 成功率（split × 任务，4 柱 + seed 散点）；fig3 配对差森林图（±2pp 等价带）；fig4 阶段 0 噪声分面；
  fig5 18 层三热图（enrich / 四种逐层最终动作差 / 梯度）；fig6 donor 覆盖。缺输入的图打「待测」占位并计入 skipped。
用法：uv run --no-sync python scripts/motion-variance/plot_mv.py --records docs/training-doc/mv-matrix-40k/records \
        --open-loop v1-store/reports/motion-variance/open_loop.json --out docs/motion-utilization/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402

COLORS = {"official": "#c96a4a", "normal": "#2f6f9f", "mask": "#7a7a7a", "swap": "#8e5ea2"}
LABELS = {"official": "official（官方 framesamp+context）", "normal": "normal（真 sidecar motion）",
          "mask": "mask（屏蔽 motion 通路）", "swap": "swap（异集专家 motion）"}
PAIR_COLORS = {"normal-mask": "#2f6f9f", "normal-swap": "#8e5ea2", "mask-swap": "#5a9a5a", "normal-official": "#c96a4a"}


def setup_font() -> None:
    fonts = ["DejaVu Sans"]
    cjk = pathlib.Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
        fonts.append(font_manager.FontProperties(fname=str(cjk)).get_name())
    plt.rcParams["font.family"] = fonts
    plt.rcParams["axes.unicode_minus"] = False


def save(fig, out: pathlib.Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    print(f"WROTE {out / name}.png/.svg")


def placeholder(out: pathlib.Path, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\n待测（输入尚未产出）", ha="center", va="center", fontsize=14)
    save(fig, out, name)


def fig1_design(out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    ax.set_xlim(0, C.PREFIX_LEN + C.ACT_H); ax.set_ylim(0, 3.2); ax.axis("off")
    segs = [(0, C.MEM_LEN, "#9ecae1", "memory 608（帧路 512 + motion 96，按 mem_order 交错）"),
            (C.MEM_LEN, C.MEM_LEN + C.IMG_LEN, "#fdd0a2", "image 512"),
            (C.MEM_LEN + C.IMG_LEN, C.PREFIX_LEN, "#c7e9c0", "text 64"),
            (C.PREFIX_LEN, C.PREFIX_LEN + C.ACT_H, "#dadaeb", "action 20")]
    for a, b, c, lab in segs:
        ax.add_patch(plt.Rectangle((a, 2.0), b - a, 0.8, color=c))
        ax.text((a + b) / 2, 2.4, lab, ha="center", va="center", fontsize=9)
    rng = np.random.default_rng(1)
    cols = np.sort(rng.choice(np.arange(0, 380), 12, replace=False))
    for x in cols:
        ax.add_patch(plt.Rectangle((x, 2.0), 3, 0.8, color="#08519c"))
    ax.text(200, 1.55, "▮ = 有效 motion 列（示意 k=12，位置由 mem_order 决定，散落在记忆段前部）", fontsize=9, color="#08519c")
    rows = [("official", "无 motion 路", "—", "—"), ("normal", "全 True", "sidecar 现算", "接收方"),
            ("mask", "motion 列 False（prefill+去噪）", "零向量（不被读）", "接收方"),
            ("swap", "全 True", "同任务另一训练集 episode", "接收方")]
    ax.text(0, 1.1, "条件", fontsize=10, weight="bold"); ax.text(180, 1.1, "attention key 门控", fontsize=10, weight="bold")
    ax.text(560, 1.1, "motion_emb 内容", fontsize=10, weight="bold"); ax.text(900, 1.1, "motion_pos / mask / mem_order / RoPE", fontsize=10, weight="bold")
    for i, (c, g, e, p) in enumerate(rows):
        y = 0.85 - i * 0.25
        ax.text(0, y, c, fontsize=9.5, color=COLORS[c], weight="bold"); ax.text(180, y, g, fontsize=9); ax.text(560, y, e, fontsize=9); ax.text(900, y, p, fontsize=9)
    ax.set_title("fig1 设计示意：前缀 1184 = [memory 608 | image 512 | text 64] + action 20；四条件只在门控 / 内容两个维度上不同", fontsize=11)
    ax = axes[1]
    ax.axis("off")
    grid = [f"{c}\n{sp}\ns{sd}" for sd in C.SEEDS for sp in C.SPLITS for c in C.CONDS]
    for i, g in enumerate(grid):
        ax.add_patch(plt.Rectangle((i % 8, 3 - i // 8), 0.95, 0.95, color=COLORS[g.split("\n")[0]], alpha=0.35))
        ax.text(i % 8 + 0.47, 3.47 - i // 8, g, ha="center", va="center", fontsize=7)
    ax.set_xlim(0, 8); ax.set_ylim(0, 4)
    ax.set_title("32 批 = 4 seed（行组）× 2 split × 4 条件；每批 8 worker stride 分片（w0/w1 各 28 集、其余 24 集），worker k 固定 GPU k", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig1-design")


def fig2_rates(out: pathlib.Path, per_ep: dict) -> int:
    eps = per_ep["episodes"]
    conds = [c for c in C.CONDS if any(e["cond"] == c for e in eps)]
    seeds = sorted({e["seed"] for e in eps})
    splits = [s for s in C.SPLITS if any(e["split"] == s for e in eps)]
    cnt = {}
    for e in eps:
        if e["success"] not in (True, False):
            continue
        for bucket in (e["task"], "all"):
            k = (e["cond"], e["split"], bucket, e["seed"])
            c = cnt.setdefault(k, [0, 0])
            c[1] += 1; c[0] += int(e["success"])
    fig, axes = plt.subplots(len(splits), 5, figsize=(20, 4.6 * len(splits)), sharey=True, squeeze=False)
    width = 0.8 / len(conds)
    cells = 0
    for i, sp in enumerate(splits):
        for j, bucket in enumerate(list(C.TASKS) + ["all"]):
            ax = axes[i, j]
            for gi, c in enumerate(conds):
                rates = [cnt[(c, sp, bucket, s)][0] / cnt[(c, sp, bucket, s)][1] * 100 for s in seeds if (c, sp, bucket, s) in cnt]
                if not rates:
                    continue
                m = float(np.mean(rates)); x = gi * width
                ax.bar(x, m, width * 0.9, color=COLORS[c], label=LABELS[c] if (i == 0 and j == 0) else None, zorder=3)
                ax.errorbar(x, m, yerr=[[m - min(rates)], [max(rates) - m]], fmt="none", ecolor="#444", capsize=3, zorder=5)
                ax.scatter([x] * len(rates), rates, s=10, color="#111", zorder=6)
                ax.text(x, max(max(rates), m) + 1.5, f"{m:.1f}", ha="center", fontsize=7.5)
                cells += 1
            ax.set_xticks([gi * width for gi in range(len(conds))]); ax.set_xticklabels(conds, fontsize=8)
            n = cnt.get((conds[0], sp, bucket, seeds[0]), [0, 0])[1]
            ax.set_title(f"{sp} · {bucket} · n={n}/seed", fontsize=10)
            ax.grid(axis="y", alpha=0.25, zorder=0); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[i, 0].set_ylabel("成功率 %（柱 = seed 均值，棒 = min–max，点 = 各 seed）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=10)
    fig.suptitle(f"fig2 成功率 · seeds={seeds} · max_steps=1300 · 来源 per_episode_by_cond.json", y=0.03, fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, out, "fig2-rates")
    return cells


def fig3_forest(out: pathlib.Path, stats: dict) -> int:
    res = [r for r in stats["results"] if r["pool"] == "pool400"]
    if not res:
        placeholder(out, "fig3-forest", "fig3 配对差森林图"); return 0
    cells_order = []
    for r in res:
        if r["cell"] not in cells_order:
            cells_order.append(r["cell"])
    pairs = [p for p in ["normal-mask", "normal-swap", "mask-swap", "normal-official"] if any(r["pair"] == p for r in res)]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(cells_order) + 2.5))
    ax.axvspan(-stats["rope_pp"], stats["rope_pp"], color="#8c8c8c", alpha=0.12, label=f"±{stats['rope_pp']:.0f}pp 等价带")
    ax.axvline(0, color="#000", lw=1.2)
    n = 0
    for ci, cell in enumerate(cells_order):
        for pi, p in enumerate(pairs):
            rr = [r for r in res if r["cell"] == cell and r["pair"] == p]
            if not rr:
                continue
            r = rr[0]; y = ci + (pi - (len(pairs) - 1) / 2) * 0.18
            col = PAIR_COLORS[p]
            ax.plot(r["t_ci_pp"], [y, y], color=col, lw=1.0)
            ax.plot(r["boot_ci_pp"], [y, y], color=col, lw=3.2, alpha=0.55)
            ax.plot(r["dmean_pp"], y, "o", color=col, ms=4.5, label=f"{p}" if ci == 0 else None)
            n += 1
    ax.set_yticks(range(len(cells_order))); ax.set_yticklabels(cells_order, fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlabel("配对差（pp，A − B）；细线 = t 区间(df=3)，粗线 = 分层 bootstrap 95%")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.set_title("fig3 配对差森林图 · pooled 400 集（test+val）· 4 seed · 来源 paired_stats.json", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, out, "fig3-forest")
    return n


def fig4_noise(out: pathlib.Path, ol: dict) -> int:
    rows = ol.get("stage0", [])
    if not rows:
        placeholder(out, "fig4-noise-facet", "fig4 阶段 0 噪声分面"); return 0
    tasks = [t for t in C.TASKS if any(r["task"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.4), sharey=True, squeeze=False)
    n = 0
    for ax, t in zip(axes[0], tasks):
        rr = [r for r in rows if r["task"] == t]
        ks = np.array([r["k"] for r in rr]); dn = np.array([max(r["d_noise"], 1e-12) for r in rr])
        for key, col, lab in (("d_mask", COLORS["mask"], "mask/noise"), ("d_swap", COLORS["swap"], "swap/noise")):
            v = np.array([r[key] for r in rr]) / dn
            ph = np.array([r["phase"] for r in rr])
            ax.scatter(ks[ph == "cold"], v[ph == "cold"], s=18, color=col, marker="^", alpha=0.7, label=f"{lab}（cold）")
            ax.scatter(ks[ph == "steady"], v[ph == "steady"], s=18, color=col, marker="o", alpha=0.7, label=f"{lab}（steady）")
            n += len(v)
        ax.axhline(1.0, color="#000", lw=0.8, ls="--"); ax.axhline(0.25, color="#999", lw=0.8, ls=":")
        ax.set_title(f"{t} · n={len(rr)}", fontsize=10); ax.set_xlabel("k（有效 motion 窗数）")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("动作差 / noise 标尺（归一化 RMS）")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"fig4 阶段 0 · noise seeds={ol.get('noise_seeds')} · k=0 处必为 0（OL_ZEROK_NULL）· 虚线 1.0 = 噪声标尺，点线 0.25 = 预注册低敏感阈", y=0.02, fontsize=9, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig4-noise-facet")
    return n


def fig5_layers(out: pathlib.Path, ol: dict) -> int:
    rows = ol.get("stage1", [])
    if not rows:
        placeholder(out, "fig5-layers", "fig5 18 层热图"); return 0
    L = C.N_LAYERS
    enrich = np.mean([np.asarray(r["share_layer"]) / max(r["U_step"], 1e-12) for r in rows], axis=0)
    heads = np.mean([np.asarray(r["share_step_full"]).mean(axis=2) for r in rows], axis=0)   # (L,G)
    scopes = ["step_only", "both", "kv_vzero", "kv_donor"]
    dn = float(np.mean([r["d_noise_layered"] for r in rows]))
    deltas = np.array([[np.mean([r["deltas"][sc][l] for r in rows]) for sc in scopes] for l in range(L)]) / max(dn, 1e-12)
    t_keys = sorted(rows[0]["grads"].keys(), key=float)
    groups = ["motion", "frame", "img", "txt"]
    G = np.array([[[np.mean([r["grads"][tk]["per_token"][g][l] for r in rows if (g != "motion" or r["k"] > 0)] or [0]) for g in groups] for tk in t_keys] for l in range(L)])
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.2), gridspec_kw={"width_ratios": [0.9, 1.6, 1.3, 2.0]})
    ax = axes[0]
    ax.barh(range(L), enrich, color=COLORS["normal"]); ax.axvline(1.0, color="#000", lw=0.8, ls="--")
    ax.set_yticks(range(L)); ax.invert_yaxis(); ax.set_title("(a) 去噪 action query → motion\n份额 / 均匀基准（>1 = 富集）", fontsize=9.5)
    ax = axes[1]
    im = ax.imshow(heads, aspect="auto", cmap="viridis"); ax.set_title("(a') 逐 head motion 份额（去噪 10 步均值）", fontsize=9.5)
    ax.set_xlabel("head"); ax.set_yticks(range(L)); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    im = ax.imshow(deltas, aspect="auto", cmap="magma"); ax.set_xticks(range(len(scopes))); ax.set_xticklabels(scopes, fontsize=8, rotation=20)
    ax.set_yticks(range(L)); ax.set_title("(b) 只动第 l 层 → 最终动作差 / 噪声标尺", fontsize=9.5); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[3]
    gm = G.reshape(L, -1)
    im = ax.imshow(np.log10(gm + 1e-12), aspect="auto", cmap="cividis")
    ax.set_xticks(range(gm.shape[1])); ax.set_xticklabels([f"{g}\nt={tk}" for tk in t_keys for g in groups], fontsize=7)
    ax.set_yticks(range(L)); ax.set_title("(c) log10 逐 token 梯度 RMS（第 l 层入口隐状态）", fontsize=9.5); fig.colorbar(im, ax=ax, fraction=0.046)
    for a in axes:
        a.set_ylabel("层")
    fig.suptitle(f"fig5 阶段 1 · 18 层 · points={len(rows)}（每集 cold/early/mid/late）· 来源 open_loop.json", y=0.01, fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig5-layers")
    return int(L * (1 + heads.shape[1] + len(scopes) + gm.shape[1]))


def fig6_donor(out: pathlib.Path, per_ep: dict | None, bank_meta: dict | None) -> int:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    n = 0
    ax = axes[0]
    if per_ep:
        eps = [e for e in per_ep["episodes"] if e["cond"] == "swap" and e.get("cover")]
        by_task = {t: {h: 0 for h in ("exact", "fallback", "cycle", "cross_seg")} for t in C.TASKS}
        for e in eps:
            for h in by_task[e["task"]]:
                by_task[e["task"]][h] += e["cover"][h]
        bottom = np.zeros(len(C.TASKS))
        for h, col in (("exact", "#5a9a5a"), ("fallback", "#e0a030"), ("cycle", "#c04040"), ("cross_seg", "#000")):
            vals = np.array([by_task[t][h] for t in C.TASKS], float)
            tot = np.array([sum(by_task[t].values()) or 1 for t in C.TASKS], float)
            ax.bar(range(len(C.TASKS)), vals / tot * 100, bottom=bottom, color=col, label=h); bottom += vals / tot * 100
            n += len(C.TASKS)
        ax.set_xticks(range(len(C.TASKS))); ax.set_xticklabels(C.TASKS, fontsize=8.5); ax.set_ylabel("按实际推理次数的窗口占比 %")
        ax.set_title("(a) swap 条件 donor 覆盖（实测，DONOR_COVER_ACTUAL）", fontsize=10); ax.legend(fontsize=8, frameon=False)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "待测（矩阵未跑完）", ha="center", va="center")
    ax = axes[1]
    if bank_meta:
        for t in C.TASKS:
            ns = sorted(v["n"] for k, v in bank_meta["segments"].items() if k.startswith(f"{t}|") and k.endswith("|exec"))
            ax.plot(ns, np.linspace(0, 1, len(ns)), label=f"{t}（max {max(ns)}）")
        ax.set_xlabel("donor exec 段窗数 n"); ax.set_ylabel("累计比例"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("(b) 库中 donor exec 窗数分布（偏移 m ≥ n 即回填）", fontsize=10); ax.grid(alpha=0.25)
        n += len(C.TASKS)
    fig.suptitle("fig6 donor 覆盖 · 库源 4task-motion-400ep/motion · exact=主 donor 精确偏移；fallback=兜底链精确偏移；cycle=超全库上限循环", y=0.02, fontsize=9, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig6-donor")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=str(C.REPO_ROOT / "docs/training-doc/mv-matrix-40k/records"))
    ap.add_argument("--open-loop", default=str(C.REPORT_ROOT / "open_loop.json"))
    ap.add_argument("--bank", default=str(C.REPORT_ROOT / "bank-lib/donor_bank.json"))
    ap.add_argument("--out", default=str(C.REPO_ROOT / "docs/motion-utilization/figures"))
    args = ap.parse_args()
    setup_font()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = pathlib.Path(args.records)
    per_ep = json.loads((rec / "per_episode_by_cond.json").read_text()) if (rec / "per_episode_by_cond.json").is_file() else None
    stats = json.loads((rec / "paired_stats.json").read_text()) if (rec / "paired_stats.json").is_file() else None
    ol = json.loads(pathlib.Path(args.open_loop).read_text()) if pathlib.Path(args.open_loop).is_file() else {}
    bank = json.loads(pathlib.Path(args.bank).read_text()) if pathlib.Path(args.bank).is_file() else None
    cells = 0; skipped = []; cells_fig2 = 0
    fig1_design(out)
    if per_ep:
        cells_fig2 = fig2_rates(out, per_ep); cells += cells_fig2
    else:
        placeholder(out, "fig2-rates", "fig2 成功率"); skipped.append("fig2")
    if stats:
        cells += fig3_forest(out, stats)
    else:
        placeholder(out, "fig3-forest", "fig3 配对差森林图"); skipped.append("fig3")
    if ol.get("stage0"):
        cells += fig4_noise(out, ol)
    else:
        placeholder(out, "fig4-noise-facet", "fig4 阶段 0"); skipped.append("fig4")
    if ol.get("stage1"):
        cells += fig5_layers(out, ol)
    else:
        placeholder(out, "fig5-layers", "fig5 18 层"); skipped.append("fig5")
    cells += fig6_donor(out, per_ep, bank)
    # 自检：fig2 的格子数与 per_episode 里 (cond, split, bucket) 数一致；fig3 与 paired_stats pool400 条目数一致
    bad = 0
    if stats:
        want3 = len([r for r in stats["results"] if r["pool"] == "pool400"])
        got3 = sum(1 for r in stats["results"] if r["pool"] == "pool400" and r["pair"] in PAIR_COLORS)
        if want3 != got3:
            print(f"MISMATCH fig3: pool400 entries {want3} vs plotted {got3}"); bad += 1
    if per_ep:
        eps = [e for e in per_ep["episodes"] if e["success"] in (True, False)]
        want2 = len({(e["cond"], e["split"]) for e in eps}) * 5
        got2 = cells_fig2
        if want2 != got2:
            print(f"MISMATCH fig2: cells {want2} vs plotted {got2}"); bad += 1
    print(f"PLOT_CELL_CHECK={'PASS' if bad == 0 else 'FAIL'} figs=6 cells={cells} mismatch={bad} skipped={skipped}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
