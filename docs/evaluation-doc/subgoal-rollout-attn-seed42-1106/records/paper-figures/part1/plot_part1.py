"""论文主图第一部分的单任务版：每个任务一张图，左边通路示意图、右边逐层折线。motion 模型，seed 42。

版式、节点、走线、配色沿用 docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records/paper-figures/plot_pathway_figure.py
的面板 (a)、(b)。全部数字由数据文件算出，示意图标签也从数据格式化，不手抄。

用法：plot_part1.py [仓库根] [输出目录]（默认仓库根 /data/hongzefu/robomme_policy_learning_MotionJEPA，输出到脚本所在目录）。
输出 fig_part1_<任务>.pdf / .png，并在 stdout 打印四任务数字对照表。

数据来源：
- ButtonUnmaskSwap、VideoUnmaskSwap（每局闭合前 4 次推理）：
  docs/evaluation-doc/swap-first-bin-attn-seed42-1105/records/action-flow-summary.json
  （direct_layer_pct 形状 (18 层, 12 读者组, 12 来源组)，组序 帧 0–3、motion 4–7、第三人称图像 8、手腕图像 9、文字 10、动作 11；
    alpha_*/carry_by_layer 形状 (19, 12, 12)，第 l 项是第 l 层输入；alpha_*/route_via_source 形状 (11 中转组, 12 来源组)）。
  基线键 baseline|<任务>，只取动作读帧 L16。
- PatternLock、RouteStick（全部推理）：<产物根>/mot/pathway/<局>.npz（pathway_flow.py 产物，8 组：示范帧、执行帧、示范 motion、
  执行 motion、第三人称图像、手腕图像、文字、动作；direct 形状 (18, 8 头, 8, 8)，rollout_α (19, 8, 8)，route_α (18, 7, 8)），
  每局等权，算法同 pathway_summary.py，并与已归档的 records/pathway-mot-summary.json 对拍；任务归属读
  <产物根>/mot/s*/run/accepted/*.terminal.json；推理次数读 pathway/index.json。基线动作读帧 L16 取 records/layer-flow.json 的
  <任务>/ctx/action 的「历史视觉」列。
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
D1106 = REPO / "docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records"
D1105 = REPO / "docs/evaluation-doc/swap-first-bin-attn-seed42-1105/records"
STORE = REPO / "v1-store/evaluation/subgoal-rollout-attn-seed42-1106"
ALPHAS = ("0.2", "0.5", "0.8")

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, INK3 = "#1f1f1f", "#555555", "#8a8a8a"
GRID = "#d9d8d2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK3, "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ----------------------------------------------------------------------------- 统一读数：每任务一份 dict
def finish(s):
    """由各通路的中转量算占比；加合并范围。via_<α> 的 total 为动作里来自 motion 的总量（%），frame/text/image 为占比（0–1）。"""
    for a in ALPHAS:
        v = s[f"via_{a}"]
        tot = v["frame_abs"] + v["text_abs"] + v["image_abs"] + v["motion_abs"]
        assert abs(tot - v["total"]) < 1e-6 * max(1, v["total"]), (tot, v["total"])
        v["frame"], v["text"], v["image"] = v["frame_abs"] / tot, v["text_abs"] / tot, v["image_abs"] / tot
    s["via_frame_rng"] = [s[f"via_{a}"]["frame"] * 100 for a in ALPHAS]
    s["via_text_rng"] = [s[f"via_{a}"]["text"] * 100 for a in ALPHAS]
    s["via_image_rng"] = [s[f"via_{a}"]["image"] * 100 for a in ALPHAS]
    s["image_text_1416"] = s["image_reads_text"][14:17].ravel()
    return s


def load_swap(task):
    flow = json.load(open(D1105 / "action-flow-summary.json", encoding="utf-8"))
    FRAME, MOTION, IMAGE, TEXT, ACTION = list(range(4)), list(range(4, 8)), [8, 9], 10, 11
    cell = flow["cells"][f"motion|{task}"]
    base = flow["cells"][f"baseline|{task}"]
    direct, counts = np.array(cell["direct_layer_pct"]), np.array(cell["token_counts"], float)
    weight = counts[FRAME] / counts[FRAME].sum()
    carry = np.array(cell["alpha_0.5"]["carry_by_layer"])
    s = {
        "task": task, "episodes": cell["episodes"], "baseline_episodes": base["episodes"],
        "scope": f"4 inferences up to first gripper closure, n = {cell['episodes']} episodes",
        "frame_reads_motion": np.array([(weight[:, None] * direct[l][FRAME][:, MOTION]).sum() for l in range(18)]),
        "frame_carry_motion": np.array([(weight[:, None] * carry[l][FRAME][:, MOTION]).sum() for l in range(19)]),
        "text_reads_frame": direct[:, TEXT][:, FRAME].sum(1),
        "text_reads_motion": direct[:, TEXT][:, MOTION].sum(1),
        "action_reads_frame": direct[:, ACTION][:, FRAME].sum(1),
        "action_reads_text": direct[:, ACTION][:, TEXT],
        "action_reads_image": direct[:, ACTION][:, IMAGE].sum(1),
        "image_reads_text": direct[:, IMAGE][:, :, TEXT],
        "image_reads_memory_max": float(direct[:, IMAGE][:, :, FRAME + MOTION].max()),
        "baseline_action_reads_frame_L16": float(np.array(base["direct_layer_pct"])[16][ACTION][FRAME].sum()),
    }
    for a in ALPHAS:
        via = np.array(cell[f"alpha_{a}"]["route_via_source"])[:, MOTION].sum(1)   # (11 中转组,) 来自 motion 的量
        s[f"via_{a}"] = {"total": via.sum(), "frame_abs": via[FRAME].sum(), "text_abs": via[TEXT],
                         "image_abs": via[IMAGE].sum(), "motion_abs": via[MOTION].sum()}
    return finish(s)


def load_pathway():
    """PatternLock / RouteStick：从每局 npz 按 pathway_summary.py 同样的每局等权算，并与归档汇总对拍。"""
    index = json.load(open(STORE / "mot/pathway/index.json", encoding="utf-8"))
    G = {n: i for i, n in enumerate(index["groups"])}
    FRAME, MOTION, IMAGE, TEXT, ACTION = [G["frame_demo"], G["frame_exec"]], [G["motion_demo"], G["motion_exec"]], \
        [G["image_base"], G["image_wrist"]], G["text"], G["action"]
    inferences = {e["nonce"]: e["inferences"] for e in index["episodes"]}
    task_of = {}
    for path in glob.glob(str(STORE / "mot/s*/run/accepted/*.terminal.json")):
        rec = json.load(open(path, encoding="utf-8"))
        task_of[rec["episode_nonce"]] = rec["episode_key"].split(":")[0]
    per_task = {}
    for path in sorted((STORE / "mot/pathway").glob("*.npz")):
        with np.load(path) as z:
            direct, counts = z["direct"].mean(1) * 100, z["counts"]
            w = counts[FRAME] / counts[FRAME].sum()
            carry = z["rollout_0.5"] * 100
            item = {
                "frame_reads_motion": np.array([(w[:, None] * direct[l][FRAME][:, MOTION]).sum() for l in range(18)]),
                "frame_carry_motion": np.array([(w[:, None] * carry[l][FRAME][:, MOTION]).sum() for l in range(19)]),
                "text_reads_frame": direct[:, TEXT][:, FRAME].sum(1),
                "text_reads_motion": direct[:, TEXT][:, MOTION].sum(1),
                "action_reads_frame": direct[:, ACTION][:, FRAME].sum(1),
                "action_reads_text": direct[:, ACTION, TEXT],
                "action_reads_image": direct[:, ACTION][:, IMAGE].sum(1),
                "image_reads_text": direct[:, IMAGE][:, :, TEXT],
                "image_reads_memory_max": float(direct[:, IMAGE][:, :, FRAME + MOTION].max()),
                "inferences": inferences[path.stem],
            }
            for a in ALPHAS:
                via = z[f"route_{a}"].sum(0) * 100          # (7 中转组, 8 来源组)
                m = via[:, MOTION].sum(1)
                item[f"via_{a}"] = {"total": m.sum(), "frame_abs": m[FRAME].sum(), "text_abs": m[TEXT],
                                    "image_abs": m[IMAGE].sum(), "motion_abs": m[MOTION].sum()}
        per_task.setdefault(task_of[path.stem], []).append(item)

    summary = json.load(open(D1106 / "pathway-mot-summary.json", encoding="utf-8"))
    layer_flow = json.load(open(D1106 / "layer-flow.json", encoding="utf-8"))
    out = {}
    for task, items in per_task.items():
        s = {"task": task, "episodes": len(items), "inferences": sum(i["inferences"] for i in items)}
        s["scope"] = f"all inferences, n = {s['episodes']} episodes, {s['inferences']} inferences"
        for k in ("frame_reads_motion", "frame_carry_motion", "text_reads_frame", "text_reads_motion",
                  "action_reads_frame", "action_reads_text", "action_reads_image", "image_reads_text"):
            s[k] = np.mean([i[k] for i in items], 0)
        s["image_reads_memory_max"] = max(i["image_reads_memory_max"] for i in items)
        for a in ALPHAS:
            s[f"via_{a}"] = {k: float(np.mean([i[f"via_{a}"][k] for i in items])) for k in ("total", "frame_abs", "text_abs", "image_abs", "motion_abs")}
        # 与归档汇总对拍（每局等权、帧按 token 加权，同一算法）
        ref = summary[task]
        assert ref["episodes"] == s["episodes"]
        for k in ("frame_reads_motion", "frame_carry_motion", "text_reads_frame", "text_reads_motion", "action_reads_frame", "action_reads_text", "action_reads_image"):
            rk = k.replace("reads", "read")
            assert np.allclose(s[k], ref["by_layer"][rk], atol=1e-6), (task, k)
        for a in ALPHAS:
            r = ref["route"][a]
            assert abs(s[f"via_{a}"]["total"] - r["total"]["mean"]) < 1e-6
            assert abs(s[f"via_{a}"]["frame_abs"] / s[f"via_{a}"]["total"] * 100 - r["frame"]) < 1e-6
        rec = layer_flow[f"{task}/ctx/action"]
        s["baseline_action_reads_frame_L16"] = float(rec["by_layer_pct"][16][rec["blocks"].index("历史视觉")])
        s["baseline_episodes"] = None
        out[task] = finish(s)
    return out


data = {"ButtonUnmaskSwap": load_swap("ButtonUnmaskSwap"), "VideoUnmaskSwap": load_swap("VideoUnmaskSwap")}
data.update(load_pathway())
TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap", "PatternLock", "RouteStick")
for t in TASKS:
    assert data[t]["image_reads_memory_max"] == 0, t


def rng(values, nd=0):
    return f"{min(values):.{nd}f}–{max(values):.{nd}f}%"


# ----------------------------------------------------------------------------- 画一张
W, H = 7.2, 3.6


def draw(s):
    task = s["task"]
    fig = plt.figure(figsize=(W, H))

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    fig.text(0.06 / W, 3.43 / H, task, fontsize=7.5, fontweight="bold", va="bottom", ha="left")
    fig.text(0.06 / W, 3.31 / H, f"motion model, seed 42, {s['scope']}", fontsize=6.5, va="bottom", ha="left", color=INK2)

    # ------------------------------------------------------------------ (a) 通路示意图
    ax = ax_in(0.28, 0.07, 3.50, 3.05)
    ax.set_xlim(0, 100); ax.set_ylim(-11, 100); ax.axis("off")
    ax.text(-0.06, 0.95, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")

    def box(x0, y0, x1, y1, title, body=None, fc="#f5f5f3", ec=INK3, title_fs=7):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=0.6, zorder=2))
        cx = (x0 + x1) / 2
        if body is None:
            ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", zorder=3)
        else:
            ax.text(cx, y1 - 1.8, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", zorder=3)
            ax.text(cx, y0 + 1.4, body, ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=3, linespacing=1.15)

    def orth(points, color, ls="-", lw=1.0, zorder=4):
        pts = np.array(points, float)
        if len(pts) > 2:
            ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=7), zorder=zorder)

    def elabel(x, y, text, color=INK, ha="left", va="center", fs=6.5):
        ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.12, zorder=5)

    fc, fr, tf, tm, af, at, ai = (s[k] for k in ("frame_carry_motion", "frame_reads_motion", "text_reads_frame", "text_reads_motion",
                                                 "action_reads_frame", "action_reads_text", "action_reads_image"))
    # 节点
    box(4, 87, 46, 97, "Motion tokens", fc="#eef2f8")
    box(2, 57, 40, 77, "Frame memory",
        body=f"motion content after layer\nL0 {fc[1]:.0f}%   L2 {fc[3]:.0f}%\nL17 {fc[18]:.0f}%", fc="#eef2f8")
    box(48, 60, 76, 74, "Language tokens", fc="#fdf1ec")
    box(72, 32, 98, 44, "Current image", body=f"reads memory {s['image_reads_memory_max']:.0f}%")
    box(2, 1, 62, 19, "Action tokens",
        body=f"motion content (α=0.5): {s['via_0.5']['total']:.1f}%\n"
             f"via frames {rng(s['via_frame_rng'])}, language {rng(s['via_text_rng'])},\nimage {rng(s['via_image_rng'])} (last hop, α 0.2–0.8)", fc="#f1f1f1")
    # 路线 A：motion → 帧记忆 → 动作
    orth([(12, 87), (12, 77)], C1)
    elabel(14, 82, f"frames read motion\nL0 {fr[0]:.0f}%   L1 {fr[1]:.0f}%")
    orth([(10, 57), (10, 19)], C1)
    elabel(12, 32, "action reads frames\n"
                   f"L14 {af[14]:.1f}%\nL15 {af[15]:.1f}%\nL16 {af[16]:.1f}%\nbaseline L16 {s['baseline_action_reads_frame_L16']:.1f}%")
    # 路线 B：motion → 语言 → 动作
    orth([(46, 92), (62, 92), (62, 74)], C2)
    elabel(64, 83, f"language reads motion\nL0 {tm[0]:.0f}%   L1 {tm[1]:.0f}%")
    orth([(58, 60), (58, 19)], C2)
    elabel(60, 26, f"action reads\nlanguage {rng(at)}\n(mean {at.mean():.0f}%)")
    # 不成立的串联：帧 → 语言（虚线灰）
    orth([(40, 67), (48, 67)], INK3, ls="--", lw=0.9)
    elabel(40, 55, f"language reads frames\nmostly at L0 ({tf[0]:.0f}%),\nwhen frames hold 0%\nmotion; L1 {tf[1]:.1f}%", color=INK2, ha="center", va="top")
    # 图像只能经语言拿记忆；动作读图像
    orth([(76, 67), (85, 67), (85, 44)], INK3, lw=0.8)
    elabel(83, 55, f"image reads\nlanguage\nL14–16 {rng(s['image_text_1416'])}", color=INK2, ha="right")
    orth([(85, 32), (85, 10), (62, 10)], INK3, lw=0.8)
    elabel(76, 8, f"action reads image\n{rng(ai)} (L0–17)", color=INK2, ha="center", va="top")
    # 图例
    ax.add_line(Line2D([2, 8], [-5, -5], color=C1, lw=1.2)); ax.text(9.5, -5, "Route A: motion → frames → action", va="center")
    ax.add_line(Line2D([2, 8], [-9.5, -9.5], color=C2, lw=1.2)); ax.text(9.5, -9.5, "Route B: motion → language → action", va="center")
    ax.add_line(Line2D([63, 68], [-5, -5], color=INK3, lw=0.9, ls="--")); ax.text(69.5, -5, "weak chain (timing)", va="center", color=INK2)
    ax.add_line(Line2D([63, 68], [-9.5, -9.5], color=INK3, lw=0.8)); ax.text(69.5, -9.5, "non-memory edges", va="center", color=INK2)

    # ------------------------------------------------------------------ (b) 逐层折线
    layers = np.arange(18)
    a = ax_in(4.62, 1.03, 2.40, 2.12)
    a.text(-0.26, 1.02, "(b)", transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    a.plot(np.arange(19), fc, color=C1, lw=1.6, zorder=3)
    a.plot(layers, tm, color=C2, lw=1.6, zorder=3)
    a.plot(layers, tf, color=C3, lw=1.6, zorder=3)
    a.plot(layers, af, color=C4, lw=1.6, zorder=3)
    a.axvline(3, color=GRID, lw=0.8, zorder=1)
    a.text(3.4, 79, "≈L3", fontsize=6.5, color=INK3, va="top")
    a.axvspan(13.5, 16.5, color="#f1f1f1", lw=0, zorder=0)
    a.text(15, 79, "L14–16", fontsize=6.5, color=INK3, va="top", ha="center")
    a.set_xlim(-0.5, 18.5); a.set_ylim(0, 80)
    a.set_xticks([0, 3, 6, 9, 12, 15, 18]); a.set_yticks([0, 20, 40, 60, 80])
    a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.set_title("per-layer read-outs", fontsize=6.5, pad=3)
    a.set_xlabel("layer", labelpad=1)
    a.set_ylabel("% of attention / content", labelpad=2)
    a.text(18.3, fc[18] + 2, f"{fc[18]:.0f}%", ha="right", va="bottom", fontsize=6.5, color=INK2)
    k = 14 + int(np.argmax(af[14:17]))
    a.text(k, af[k] + 2.5, f"{af[k]:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
    handles = [Line2D([], [], color=c, lw=1.6) for c in (C1, C2, C3, C4)]
    labels = ["frame memory: motion content (cum., α=0.5, layer-l input)", "language reads motion tokens (direct)",
              "language reads frame memory (direct)", "action reads frame memory (direct)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(4.12 / W, 0.70 / H), ncol=1, frameon=False,
               handlelength=1.6, handletextpad=0.6, borderaxespad=0, labelspacing=0.35)

    fig.savefig(OUT / f"fig_part1_{task}.pdf")
    fig.savefig(OUT / f"fig_part1_{task}.png", dpi=300)
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
for t in TASKS:
    draw(data[t])

# ----------------------------------------------------------------------------- 数字对照表
rows = [
    ("scope", lambda s: s["scope"]),
    ("frames read motion L0 / L1", lambda s: f"{s['frame_reads_motion'][0]:.1f} / {s['frame_reads_motion'][1]:.1f}"),
    ("frame carry motion after L0 / L2 / L17", lambda s: " / ".join(f"{s['frame_carry_motion'][i]:.1f}" for i in (1, 3, 18))),
    ("language reads motion L0 / L1", lambda s: f"{s['text_reads_motion'][0]:.1f} / {s['text_reads_motion'][1]:.1f}"),
    ("language reads frames L0 / L1 / max L2–17", lambda s: f"{s['text_reads_frame'][0]:.1f} / {s['text_reads_frame'][1]:.1f} / {s['text_reads_frame'][2:].max():.1f}"),
    ("action reads frames L14 / L15 / L16", lambda s: " / ".join(f"{s['action_reads_frame'][i]:.1f}" for i in (14, 15, 16))),
    ("baseline action reads frames L16", lambda s: f"{s['baseline_action_reads_frame_L16']:.1f}"),
    ("action reads language range (mean)", lambda s: f"{rng(s['action_reads_text'], 1)} ({s['action_reads_text'].mean():.1f})"),
    ("image reads language L14–16 range", lambda s: rng(s["image_text_1416"], 1)),
    ("action reads image range L0–17", lambda s: rng(s["action_reads_image"], 1)),
    ("image reads memory max", lambda s: f"{s['image_reads_memory_max']:.3g}"),
    ("motion in action, α=0.2 / 0.5 / 0.8", lambda s: " / ".join(f"{s[f'via_{a}']['total']:.2f}" for a in ALPHAS)),
    ("via frames %, α=0.2 / 0.5 / 0.8", lambda s: " / ".join(f"{s[f'via_{a}']['frame'] * 100:.1f}" for a in ALPHAS)),
    ("via language %, α=0.2 / 0.5 / 0.8", lambda s: " / ".join(f"{s[f'via_{a}']['text'] * 100:.1f}" for a in ALPHAS)),
    ("via image %, α=0.2 / 0.5 / 0.8", lambda s: " / ".join(f"{s[f'via_{a}']['image'] * 100:.1f}" for a in ALPHAS)),
]
print("| 读数 | " + " | ".join(TASKS) + " |")
print("|---|" + "---|" * len(TASKS))
for name, f in rows:
    print(f"| {name} | " + " | ".join(f(data[t]) for t in TASKS) + " |")
print("FIG_PART1=DONE", *[str(OUT / f"fig_part1_{t}.{ext}") for t in TASKS for ext in ("pdf", "png")])
