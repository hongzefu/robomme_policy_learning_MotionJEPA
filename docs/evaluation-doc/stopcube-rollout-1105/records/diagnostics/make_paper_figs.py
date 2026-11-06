#!/usr/bin/env python
"""论文用四张英文图（StopCube rollout 线性探针，预测「到正确窗口还有几步」）。

fig-r2-layers-baseline / fig-r2-layers-motion：逐层 R²（单分面，单栏尺寸；参考线只留 Time only 与 R²=0）。
fig-r2-fit-baseline / fig-r2-fit-motion：第 17 层拟合直线，每模型一张（单分面、单栏宽）：折外预测散点 +
    每局等权 WLS 直线 + 按局 bootstrap（seed 42，2000 次）的拟合线 95% 置信带；图例放坐标轴下方，
    每行写 R²（fit-summary）、slope（现算）与斜率 95% 区间（同一批重采样）。两张图 x、y 范围与刻度相同、等比例。
fig-point-accuracy：逐点准确率逐层图（2×2，双栏尺寸，无参考线）。
五张图都不含 "Time + target count"。
每张同时输出 PDF（fonttype 42）与 PNG（dpi 300）。只读数据，只写本目录。
数据键沿用同仓库 records/diagnostics/plot_q4_compare.py 与 plot_r2_complete.py。
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

REPO = "/data/hongzefu/robomme_policy_learning_MotionJEPA"
SRC_PROBE = f"{REPO}/docs/evaluation-doc/stopcube-rollout-1105/records/probe-summary.json"
FIT_DIR = f"{REPO}/v1-store/evaluation/stopcube-rollout-1105/analysis/fit-data"
SRC_FIT = f"{FIT_DIR}/fit-summary.json"
SRC_NPZ = f"{FIT_DIR}/fit-predictions.npz"
SRC_CLS = f"{FIT_DIR}/class-layers.json"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LAB = "q4"
N_LAYERS = 18
X_INPUT = 0
X_LAYERS = np.arange(1, N_LAYERS + 1)        # 第 k 层画在 x=k+1
TICK_LAYERS = [2, 5, 8, 11, 14, 17]
X_TICKS = [X_INPUT] + [k + 1 for k in TICK_LAYERS]
X_TICKLABELS = ["Input"] + [str(k) for k in TICK_LAYERS]

# ---------- 字体与全局样式 ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})
INK = "#111111"
INK2 = "#444444"
MUTED = "#7a7a7a"
GRID = "#e4e4e4"
SPINE = "#b8b8b8"
ZERO = "#a8a8a8"

# 组名 → (颜色, 线宽, probe-summary / class-layers / npz 里的读数键)
GROUPS = [
    ("Action token",   "#2a78d6", 1.6, "action"),
    ("Text",           "#eb6834", 1.6, "文字"),
    ("Current frame",  "#1baf7a", 1.3, "当前视觉"),
    ("History frames", "#eda100", 1.1, "历史视觉"),
    ("Motion",         "#e87ba4", 1.1, "motion"),
]
GROUP_COLOR = {g: c for g, c, _, _ in GROUPS}
TIME_NAME = "Time only"
MODELS = [("ctx", "Baseline"), ("mot", "Motion model")]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=INK2, labelcolor=INK)


# ---------- 线尾标签防重叠 ----------
def spread_labels(ends, ylim, min_gap):
    """ends: [(name, y)]，返回 name → 标签 y。按 y 排序后两两推开 min_gap，整体不出 ylim。"""
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    ys = [ends[i][1] for i in order]
    lo, hi = ylim
    pad = min_gap * 0.6
    ys = [min(max(y, lo + pad), hi - pad) for y in ys]
    for _ in range(8):
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < min_gap:
                ys[k] = ys[k - 1] + min_gap
        over = ys[-1] - (hi - pad)
        if over > 0:
            ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            if ys[k + 1] - ys[k] < min_gap:
                ys[k] = ys[k + 1] - min_gap
        under = (lo + pad) - ys[0]
        if under > 0:
            ys = [y + under for y in ys]
    return {ends[i][0]: ys[k] for k, i in enumerate(order)}


X_DATA_LO = -0.7
X_DATA_HI = N_LAYERS + 0.45


def hline(ax, y, **kw):
    """参考横线只画在数据区（不伸进右侧标签区）。"""
    ax.plot([X_DATA_LO, X_DATA_HI], [y, y], **kw)


def end_labels(ax, ends, ylim, min_gap, x_line_end, x_text, fontsize=7):
    pos = spread_labels(ends, ylim, min_gap)
    for name, y_end, color, bold in ends_full(ends):
        y_lab = pos[name]
        if abs(y_lab - y_end) > 1e-9:
            ax.plot([x_line_end + 0.15, x_text - 0.15], [y_end, y_lab],
                    color=color, lw=0.5, alpha=0.9, zorder=2)
        ax.text(x_text, y_lab, name, color=color, fontsize=fontsize,
                fontweight="bold" if bold else "normal", va="center", ha="left", zorder=5)


def ends_full(ends):
    return [(e[0], e[1], e[2], e[3]) for e in ends]


# =====================================================================
# 图 1 / 图 2：逐层 R²
# =====================================================================
YLIM_R2 = (-0.25, 1.0)
X_TEXT = N_LAYERS + 0.9
XLIM_R2 = (-0.7, N_LAYERS + 9.6)


def series_r2(L, key):
    """返回 (x, y)；action 没有输入点，其余组输入点来自 prefix_layer0。"""
    xs, ys = [], []
    if key == "action":
        items = L["layers"]["action"]
    else:
        items = L["layers"]["prefix"].get(key)
        if not items:
            return None
        l0 = L["layers"]["prefix_layer0"].get(key)
        if l0:
            xs.append(X_INPUT)
            ys.append(l0["r2"])
    for i, it in enumerate(items):
        xs.append(X_LAYERS[i])
        ys.append(it["r2"])
    return np.array(xs, float), np.array(ys, float)


def fig_r2_layers(p, fit, m, m_name, out_stem):
    L = p["models"][m]["labels"][LAB]
    tb_t = fit[m][LAB]["time"]["r2"]          # Time only
    n_ep = L["n_episodes"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    style_axes(ax)
    hline(ax, 0, color=ZERO, lw=0.7, zorder=1)
    hline(ax, tb_t, color=INK, lw=0.8, ls=(0, (1.2, 2.0)), zorder=2)

    ends = [(TIME_NAME, tb_t, INK, False)]
    r2_at_17 = {}
    for name, color, lw, key in GROUPS:
        sv = series_r2(L, key)
        if sv is None:
            continue
        x, y = sv
        ax.plot(x, y, color=color, lw=lw, marker="o", ms=2.6, mfc=color, mec="white",
                mew=0.4, solid_capstyle="round", zorder=3)
        ends.append((name, y[-1], color, lw > 1.5))
        r2_at_17[name] = y[-1]
    end_labels(ax, ends, YLIM_R2, (YLIM_R2[1] - YLIM_R2[0]) / 13.5,
               X_LAYERS[-1], X_TEXT)

    ax.set_xlim(*XLIM_R2)
    ax.set_ylim(*YLIM_R2)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICKLABELS)
    ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax.set_xlabel("Model layer")
    ax.set_ylabel("R²")
    ax.set_title(f"{m_name} ({n_ep} episodes)", loc="left", fontweight="bold", pad=4)
    fig.tight_layout(pad=0.3)
    save(fig, out_stem)
    return r2_at_17, tb_t


# =====================================================================
# 图 3a / 图 3b：第 17 层拟合直线，每模型一张（散点 + WLS 直线 + 按局 bootstrap 置信带）
# =====================================================================
PANELS = [
    ("Action token", "action"),
    ("Text", "文字"),
    ("Current frame", "当前视觉"),
    ("History frames", "历史视觉"),
    ("Motion", "motion"),
]
BOOT_SEED = 42
BOOT_N = 2000
# 不加白边（白边会盖住贴着的另一条线）。Motion 与 History frames 的直线几乎重合，
# 故 Motion 画粗、压在下，History frames 画细、压在最上，两条都能看见。
FIT_LW = {"Action token": 2.0, "Text": 2.0, "Current frame": 2.0,
          "History frames": 1.1, "Motion": 2.8}
FIT_Z = {"Action token": 4, "Text": 4, "Current frame": 4, "Motion": 4.2,
         "History frames": 4.4}
SCATTER_S = 4
SCATTER_ALPHA = 0.2
BAND_ALPHA = 0.2


def wls_line(x, yhat, groups):
    """yhat 对 x 的加权最小二乘直线，权重 = 1/该局样本数（每局等权）。返回 (截距, 斜率)。"""
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    w = 1.0 / counts[inv]
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ yhat)
    return beta[0], beta[1]


def episode_bootstrap(x, yhat, groups, seed=BOOT_SEED, n_boot=BOOT_N):
    """按局重采样（同一局的点一起进出）n_boot 次，每次重算每局等权 WLS 直线。
    返回 (截距数组, 斜率数组)，长度 n_boot。
    实现：每局先算等权充分统计量 [1, mean x, mean x², mean y, mean xy]；一次抽样里某局被抽到 c 次，
    就等价于把它当成 c 个独立的局（权重 c），与把抽到的点原样复制后再调 wls_line 完全一致。"""
    uniq, inv = np.unique(groups, return_inverse=True)
    n_ep = len(uniq)
    stats = np.zeros((n_ep, 5))
    np.add.at(stats, inv, np.column_stack([np.ones_like(x), x, x * x, yhat, x * yhat]))
    stats /= stats[:, :1]                       # 每局内取均值 → 该局总权重 1
    # 原样本上的全样本解必须与 wls_line 逐位一致（校验等价性）
    S = stats.sum(axis=0)
    beta0 = np.linalg.solve(np.array([[S[0], S[1]], [S[1], S[2]]]), np.array([S[3], S[4]]))
    a_ref, b_ref = wls_line(x, yhat, groups)
    assert abs(beta0[0] - a_ref) < 1e-8 * max(1, abs(a_ref)) and abs(beta0[1] - b_ref) < 1e-10, \
        (beta0, a_ref, b_ref)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_ep, size=(n_boot, n_ep))
    Sb = stats[draws].sum(axis=1)               # (n_boot, 5)
    det = Sb[:, 0] * Sb[:, 2] - Sb[:, 1] ** 2
    b = (Sb[:, 0] * Sb[:, 4] - Sb[:, 1] * Sb[:, 3]) / det
    a = (Sb[:, 3] - b * Sb[:, 1]) / Sb[:, 0]
    return a, b


def collect_fits(fit, npz):
    """两模型 × 各组：散点、直线、bootstrap 直线族。返回 {模型键: {组名: dict}} 与两图共用的坐标范围。"""
    out = {}
    all_x, all_y = [], []
    for m, _ in MODELS:
        y = npz[f"{m}/{LAB}/y"].astype(float)
        g = npz[f"{m}/{LAB}/groups"]
        assert (len(np.unique(g)), len(y)) == (fit[m][LAB]["n_episodes"], fit[m][LAB]["n_samples"])
        out[m] = {"n_ep": len(np.unique(g)), "n_pts": len(y), "groups": {}}
        for name, key in PANELS:
            k = f"{m}/{LAB}/{key}"
            if k not in npz.files:
                continue
            pred = npz[k].astype(float)
            ok = np.isfinite(pred) & np.isfinite(y)
            xx, yy, gg = y[ok], pred[ok], g[ok]
            a, b = wls_line(xx, yy, gg)
            ab, bb = episode_bootstrap(xx, yy, gg)
            xg = np.linspace(xx.min(), xx.max(), 101)
            yb = ab[:, None] + bb[:, None] * xg[None, :]
            lo, hi = np.percentile(yb, [2.5, 97.5], axis=0)
            out[m]["groups"][name] = dict(
                x=xx, y=yy, a=a, b=b, r2=fit[m][LAB][key]["r2"], n=int(ok.sum()),
                xg=xg, band_lo=lo, band_hi=hi,
                b_lo=float(np.percentile(bb, 2.5)), b_hi=float(np.percentile(bb, 97.5)))
            all_x += [xx.min(), xx.max()]
            all_y += [yy.min(), yy.max(), lo.min(), hi.max()]
    # 坐标范围：x 盖住两模型全部真值，y 盖住两模型全部散点与置信带；两图共用
    def padded(lo, hi):
        pad = (hi - lo) * 0.04
        return (lo - pad, hi + pad)
    return out, padded(min(all_x), max(all_x)), padded(min(all_y), max(all_y))


# 版式（英寸）：单栏宽 3.5；坐标轴等比例，图例在坐标轴下方
FIG_W = 3.5
AX_LEFT = 0.60          # 给 "-200" 刻度与 y 轴名留位
AX_TOP_PAD = 0.26       # 标题
AX_H = 2.80             # 坐标轴高度（宽度按等比例由范围推出）
AX_BOTTOM_PAD = 0.42    # x 刻度 + x 轴名
LEG_GAP = 0.10
LEG_ROW = 0.14
LEG_ROWS = len(PANELS) + 1   # 五组 + y = x（Baseline 少一行，留空保持两图版式一致）
NOTE_H = 0.16
FIG_BOTTOM = 0.06
FIG_H = AX_TOP_PAD + AX_H + AX_BOTTOM_PAD + LEG_GAP + LEG_ROW * LEG_ROWS + NOTE_H + FIG_BOTTOM


def fig_r2_fit(fits, xlim, ylim, m, m_name, out_stem):
    """单分面：散点 + 直线 + 置信带；图例在轴下，每行：线样本、组名、R²、slope、斜率 95% 区间。"""
    data = fits[m]
    ax_w = AX_H * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    assert AX_LEFT + ax_w + 0.10 <= FIG_W, ("axes too wide", ax_w)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_bottom = FIG_BOTTOM + NOTE_H + LEG_ROW * LEG_ROWS + LEG_GAP + AX_BOTTOM_PAD
    ax = fig.add_axes([AX_LEFT / FIG_W, ax_bottom / FIG_H, ax_w / FIG_W, AX_H / FIG_H])
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=INK2, labelcolor=INK)
    d0, d1 = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
    ax.plot([d0, d1], [d0, d1], color=ZERO, lw=0.8, zorder=2)

    rows = []
    for name, _ in PANELS:
        gd = data["groups"].get(name)
        if gd is None:
            continue
        color = GROUP_COLOR[name]
        ax.scatter(gd["x"], gd["y"], s=SCATTER_S, c=color, alpha=SCATTER_ALPHA,
                   lw=0, zorder=1)
        ax.fill_between(gd["xg"], gd["band_lo"], gd["band_hi"], color=color,
                        alpha=BAND_ALPHA, lw=0, zorder=3)
        xs = np.array([gd["x"].min(), gd["x"].max()])
        ax.plot(xs, gd["a"] + gd["b"] * xs, color=color, lw=FIT_LW[name], ls="-",
                solid_capstyle="butt", zorder=FIT_Z[name])
        rows.append((name, gd["r2"], gd["b"], gd["b_lo"], gd["b_hi"], gd["n"]))

    ax.set_xticks(np.arange(np.ceil(xlim[0] / 100) * 100, xlim[1], 100))
    ax.set_yticks(np.arange(np.ceil(ylim[0] / 100) * 100, ylim[1], 100))
    ax.set_xlabel("True steps to correct window")
    ax.set_ylabel("Predicted steps")
    ax.set_title(f"{m_name} ({data['n_ep']} episodes) — Layer 17", loc="left",
                 fontweight="bold", pad=4)

    # ---- 图例：坐标轴下方，英寸坐标的隐形轴；数值按列右对齐 ----
    axt = fig.add_axes([0, 0, 1, 1])
    axt.set_axis_off()
    axt.set_xlim(0, FIG_W)
    axt.set_ylim(0, FIG_H)
    axt.set_zorder(-1)
    FS = 7
    r = fig.canvas.get_renderer()

    def width_in(s, weight="normal"):
        t = axt.text(0, 0, s, fontsize=FS, fontweight=weight)
        w = t.get_window_extent(r).width / fig.dpi
        t.remove()
        return w

    x0 = AX_LEFT
    x_h0, x_h1 = x0, x0 + 0.22
    x_name = x_h1 + 0.08
    name_w = max(width_in(n, "bold") for n, _ in PANELS)
    x_r2lab = x_name + name_w + 0.10
    x_r2val = x_r2lab + width_in("R²") + 0.07 + max(width_in(f"{v:.2f},") for v in (-0.12, 0.46))
    x_sllab = x_r2val + 0.06
    x_slval = x_sllab + width_in("slope") + 0.07 + max(width_in(f"{v:.2f}") for v in (-0.04, 0.50))
    x_ci = x_slval + 0.06
    y_top = FIG_BOTTOM + NOTE_H + LEG_ROW * LEG_ROWS - LEG_ROW / 2
    right_edge = x_ci
    for i, (name, r2, b, lo, hi, _) in enumerate(rows):
        yy = y_top - LEG_ROW * i
        axt.plot([x_h0, x_h1], [yy, yy], color=GROUP_COLOR[name], lw=FIT_LW[name],
                 solid_capstyle="butt")
        axt.text(x_name, yy, name, ha="left", va="center", fontsize=FS,
                 color=GROUP_COLOR[name], fontweight="bold")
        axt.text(x_r2lab, yy, "R²", ha="left", va="center", fontsize=FS, color=INK)
        axt.text(x_r2val, yy, f"{r2:.2f},", ha="right", va="center", fontsize=FS, color=INK)
        axt.text(x_sllab, yy, "slope", ha="left", va="center", fontsize=FS, color=INK)
        axt.text(x_slval, yy, f"{b:.2f}", ha="right", va="center", fontsize=FS, color=INK)
        t = axt.text(x_ci, yy, f"[{lo:.2f}, {hi:.2f}]", ha="left", va="center",
                     fontsize=FS, color=INK)
        right_edge = max(right_edge, x_ci + t.get_window_extent(r).width / fig.dpi)
    yy = y_top - LEG_ROW * len(rows)
    axt.plot([x_h0, x_h1], [yy, yy], color=ZERO, lw=0.8)
    axt.text(x_name, yy, "y = x", ha="left", va="center", fontsize=FS, color=INK)
    axt.text(x0, FIG_BOTTOM + NOTE_H / 2, "Shaded: 95% CI of the fit (episode bootstrap)",
             ha="left", va="center", fontsize=FS, color=INK2)
    assert right_edge <= FIG_W - 0.05, ("legend overflows figure width", right_edge)
    print(f"[{out_stem}] axes {ax_w:.2f}x{AX_H:.2f} in, fig {FIG_W}x{FIG_H:.2f} in, "
          f"legend right edge {right_edge:.2f} in")
    save(fig, out_stem)
    return rows


# =====================================================================
# 图 4：逐点准确率逐层图
# =====================================================================
ROWS_ACC = [("press", "Press now?"), ("c", "Passes elapsed")]
YLIM_ACC = (0.0, 1.0)


def series_acc(reader, task):
    """class-layers 里的逐层逐点准确率；层号 '-1' 为输入。"""
    xs, ys = [], []
    for k in sorted(reader.keys(), key=int):
        li = int(k)
        xs.append(X_INPUT if li < 0 else li + 1)
        ys.append(reader[k][task]["point"]["mean"])
    return np.array(xs, float), np.array(ys, float)


def fig_point_accuracy(cls, p, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.095,
                        hspace=0.38, wspace=0.12)
    vals = {}
    for r, (task, task_name) in enumerate(ROWS_ACC):
        for c, (m, m_name) in enumerate(MODELS):
            ax = axes[r][c]
            style_axes(ax)
            ends = []
            for name, color, lw, key in GROUPS:
                rd = cls[m]["readers"].get(key)
                if rd is None:
                    continue
                x, y = series_acc(rd, task)
                ax.plot(x, y, color=color, lw=lw, marker="o", ms=2.6, mfc=color,
                        mec="white", mew=0.4, solid_capstyle="round", zorder=3)
                ends.append((name, y[-1], color, lw > 1.5))
                vals[(task, m, name)] = y[-1]
            end_labels(ax, ends, YLIM_ACC, (YLIM_ACC[1] - YLIM_ACC[0]) / 13.5,
                       X_LAYERS[-1], X_TEXT)
            ax.set_xlim(-0.7, N_LAYERS + 7.4)
            ax.set_ylim(*YLIM_ACC)
            ax.set_xticks(X_TICKS)
            ax.set_xticklabels(X_TICKLABELS)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            n_ep = p["models"][m]["labels"][LAB]["n_episodes"]
            ax.set_title(f"{task_name} — {m_name} ({n_ep} episodes)", loc="left",
                         fontweight="bold", pad=4)
            if r == 1:
                ax.set_xlabel("Model layer")
            if c == 0:
                ax.set_ylabel("Point accuracy")
    save(fig, out_stem)
    return vals


def save(fig, stem):
    fig.savefig(f"{OUT_DIR}/{stem}.pdf")
    fig.savefig(f"{OUT_DIR}/{stem}.png", dpi=300)
    plt.close(fig)
    print("written:", f"{OUT_DIR}/{stem}.pdf", f"{OUT_DIR}/{stem}.png")


def main():
    p = load_json(SRC_PROBE)
    fit = load_json(SRC_FIT)
    npz = np.load(SRC_NPZ, allow_pickle=False)
    cls = load_json(SRC_CLS)
    from matplotlib import font_manager
    print("font:", font_manager.findfont(font_manager.FontProperties(family="Liberation Sans")))

    print("\n== Fig 1/2: layer-17 R² per group, and Time-only baseline ==")
    for m, m_name in MODELS:
        stem = "fig-r2-layers-baseline" if m == "ctx" else "fig-r2-layers-motion"
        r2_17, tb_t = fig_r2_layers(p, fit, m, m_name, stem)
        print(f"[{m_name}] episodes={p['models'][m]['labels'][LAB]['n_episodes']} "
              f"samples={p['models'][m]['labels'][LAB]['n_samples']}")
        for name, v in r2_17.items():
            print(f"  layer 17 {name:16s} R² = {v:.4f}")
        print(f"  {TIME_NAME:16s} R² = {tb_t:.4f}")

    print("\n== Fig 3a/3b: layer 17 fits (group | R² | slope [95% CI, episode bootstrap] | n points) ==")
    fits, xlim, ylim = collect_fits(fit, npz)
    print(f"shared xlim={xlim[0]:.1f}..{xlim[1]:.1f} ylim={ylim[0]:.1f}..{ylim[1]:.1f}")
    for m, m_name in MODELS:
        stem = "fig-r2-fit-baseline" if m == "ctx" else "fig-r2-fit-motion"
        rows = fig_r2_fit(fits, xlim, ylim, m, m_name, stem)
        print(f"[{m_name}] episodes={fits[m]['n_ep']} points={fits[m]['n_pts']}")
        for name, r2, b, lo, hi, n in rows:
            print(f"  {name:16s} | R² {r2:.4f} | slope {b:.4f} [{lo:.4f}, {hi:.4f}] | {n}")

    print("\n== Fig 4: layer-17 point accuracy ==")
    vals = fig_point_accuracy(cls, p, "fig-point-accuracy")
    for (task, m, name), v in sorted(vals.items()):
        print(f"{task:5s} | {m:3s} | {name:20s} | {v:.4f}")


if __name__ == "__main__":
    main()
