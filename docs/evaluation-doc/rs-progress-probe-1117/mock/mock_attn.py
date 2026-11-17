"""MOCK figure (hypothetical data): attention pathways under modulation injection.
Run: uv run --no-project --with matplotlib --with numpy python -I mock_attn.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, PowerNorm, Normalize
from matplotlib.cm import ScalarMappable

rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
GREY = "#555555"
NL = 18
layers = np.arange(NL)

# ---------------- mock data, panel (a) ----------------
# block 1: MemoryAttention share / uniform share; one softmax -> 0.24*rm + 0.76*rf = 1
P_M, P_F = 0.24, 0.76
rm = np.where(layers <= 5, rng.uniform(2.0, 2.8, NL),
              np.where(layers >= 13, rng.uniform(1.8, 2.2, NL), rng.uniform(1.2, 1.5, NL)))
rf = (1.0 - P_M * rm) / P_F
assert np.allclose(P_M * rm + P_F * rf, 1.0)
blk1 = np.vstack([rm, rf])

# block 2: FiLM relative change (%), total and motion part (40-60% of total)
tot = np.where(layers <= 3, rng.uniform(5, 8, NL),
               np.where(layers >= 10, rng.uniform(15, 25, NL),
                        np.linspace(8, 15, NL) + rng.uniform(-1, 1, NL)))
blk2 = np.vstack([tot, tot * rng.uniform(0.4, 0.6, NL)])

# block 3: main attention of action tokens (% of reader's attention), sums to 100
act = rng.uniform(20, 40, NL)
img = rng.uniform(10, 30, NL)
lang = 100.0 - act - img
blk3 = np.vstack([lang, img, act])
assert np.allclose(blk3.sum(0), 100.0)

# ---------------- mock data, panel (b) ----------------
NJ, NS = 6, 8
MAXCOL = [8, 8, 7, 7, 6, 6]          # long routes are rare -> lower-right cells missing
valid = np.array([[c < MAXCOL[r] for c in range(NS)] for r in range(NJ)])
diag = np.eye(NJ, NS, dtype=bool)


def make(diag_rng, near, off_rng):
    m = rng.uniform(*off_rng, (NJ, NS))
    for r in range(NJ):
        for c in range(NS):
            if abs(r - c) == 1:
                m[r, c] *= near
        m[r, r] = rng.uniform(*diag_rng)
    return np.ma.masked_where(~valid, m)


def dval(m):
    return float(m[diag & valid].mean() / m[(~diag) & valid].mean())


mats = {
    ("MotionMem", "motion"): make((1.8, 2.6), 1.75, (0.75, 1.0)),
    ("MotionMem", "frames"): make((1.1, 1.35), 1.05, (0.85, 1.05)),
    ("Frozen Wan pooling", "motion"): make((0.9, 1.25), 1.0, (0.85, 1.15)),
    ("Frozen Wan pooling", "frames"): make((0.85, 1.2), 1.0, (0.85, 1.15)),
}

# ---------------- colours ----------------
div = LinearSegmentedColormap.from_list("bwr_soft", ["#3b7fd9", "#eeeeee", "#e5484d"])
div.set_bad("white")
div_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
blue_norm = PowerNorm(0.5, vmin=0, vmax=70)
film_norm = Normalize(0, 30)

# ---------------- figure ----------------
fig = plt.figure(figsize=(6.5, 8.0))
X0, W = 0.285, 0.535           # heatmap left / width (shared by a and b)
CBX, CBW = 0.855, 0.022
RH = 0.031                     # row height in (a)


def heat(ax, data, cmap, norm):
    ax.pcolormesh(data, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.2)
    ax.set_xlim(0, data.shape[1]); ax.set_ylim(data.shape[0], 0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=GREY)


def cbar(rect, cmap, norm, ticks, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=7.5, colors=GREY, length=2.5)
    cb.set_label(label, fontsize=7.5, color=GREY, labelpad=4, linespacing=1.1)
    cb.outline.set_linewidth(0.8)
    return cb


blocks = [
    ("action reads memory (MemoryAttention)", ["action ← motion", "action ← frames"],
     blk1, div, div_norm, [0, 1, 2, 3], "share ÷\nuniform"),
    ("memory's effect on action (FiLM)", ["memory → action", "of which motion"],
     blk2, "Purples", film_norm, [0, 15, 30], "FiLM rel.\nchange (%)"),
    ("action reads prefix (main attention)",
     ["action ← lang", "action ← image", "action ← action"],
     blk3, "Blues", blue_norm, [0, 5, 20, 40, 70], "% of reader's\nattention"),
]
y = 0.925
for i, (title, rows, data, cmap, norm, ticks, lab) in enumerate(blocks):
    h = RH * len(rows)
    ax = fig.add_axes([X0, y - h, W, h])
    heat(ax, data, cmap, norm)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=9.5, color="#222222")
    if rows[1].startswith("of which"):
        ax.get_yticklabels()[1].set_color(GREY)
        ax.get_yticklabels()[1].set_style("italic")
    ax.text(0, 1.0, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color="#777777", linespacing=1.0,
            bbox=dict(fc="none", ec="none", pad=1.5))
    if i == len(blocks) - 1:
        ax.set_xticks([0.5, 5.5, 10.5, 15.5])
        ax.set_xticklabels(["L0", "L5", "L10", "L15"], fontsize=9, color=GREY)
        ax.set_xlabel("layer", fontsize=9.5, color=GREY, labelpad=3)
        y_a_bottom = y - h
    else:
        ax.set_xticks([])
    cbar([CBX, y - h, CBW, h], cmap, norm, ticks, lab)
    y -= h + 0.034

fig.text(X0 + W / 2, y_a_bottom - 0.068,
         "frames←motion, lang←motion, lang←frames: "
         "zero by construction under modulation (not shown)",
         ha="center", va="center", fontsize=7, color="#777777", style="italic")

# ---------------- panel (b) ----------------
CW = 0.255                         # matrix width (fig fraction)
CH = CW * 6.5 / 8.0 * NJ / NS      # square cells
GAPX = W - 2 * CW
row_y = [0.305, 0.085]
D = {}
for r, model in enumerate(["MotionMem", "Frozen Wan pooling"]):
    for c, src in enumerate(["motion", "frames"]):
        m = mats[(model, src)]
        D[(model, src)] = dval(m)
        ax = fig.add_axes([X0 + c * (CW + GAPX), row_y[r], CW, CH])
        heat(ax, m, div, div_norm)
        dtxt = f"$D$ = {D[(model, src)]:.2f}"
        ax.set_title((f"action ← {src}\n" if r == 0 else "") + dtxt,
                     loc="left", fontsize=9.5, color="#222222", pad=4, linespacing=1.15)
        if c == 0:
            ax.set_yticks(np.arange(NJ) + 0.5)
            ax.set_yticklabels(np.arange(1, NJ + 1), fontsize=8.5)
            ax.set_ylabel(f"{model}\nsub-goal $j$", fontsize=9, color="#222222",
                          labelpad=5, linespacing=1.1)
        else:
            ax.set_yticks([])
        if r == 1:
            ax.set_xticks(np.arange(NS) + 0.5)
            ax.set_xticklabels(np.arange(1, NS + 1), fontsize=8.5)
        else:
            ax.set_xticks([])
fig.text(X0 + W / 2, 0.040, "demo segment $j'$ read", ha="center", va="center",
         fontsize=9.5, color=GREY)
fig.text(X0 + W / 2, 0.016,
         "MemoryAttention share ÷ uniform, averaged over L0–L17",
         ha="center", va="center", fontsize=7, color="#777777", style="italic")
cb = cbar([CBX, 0.15, CBW, 0.26], div, div_norm, [0, 1, 2, 3], "share ÷ uniform")
cb.set_label("share ÷ uniform", fontsize=9, color=GREY)
cb.ax.tick_params(labelsize=8.5)

# ---------------- panel letters, watermark ----------------
fig.text(0.02, 0.975, "(a)", fontsize=13, fontweight="bold", va="center")
fig.text(0.02, 0.535, "(b)", fontsize=13, fontweight="bold", va="center")
fig.text(0.98, 0.975, "MOCK — hypothetical data", fontsize=13, fontweight="bold",
         color="#9a9a9a", ha="right", va="center")

fig.savefig("mock_attn_ab.png", dpi=200)
fig.savefig("mock_attn_ab.pdf")
for k, v in D.items():
    print(f"D[{k[0]} | action<-{k[1]}] = {v:.3f}")
print("blk1 motion", rm.round(2).min(), rm.round(2).max(), "frames", rf.round(2).min(), rf.round(2).max())
print("blk2 total", tot.min().round(1), tot.max().round(1))
print("blk3 lang", lang.min().round(1), lang.max().round(1), "img", img.min().round(1), img.max().round(1), "act", act.min().round(1), act.max().round(1))
