"""把 decompose.json 画成点图：读首段、真对位、读末段，均匀水平为 1。用法：plot_decompose.py <decompose.json> <输出 png>"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import subprocess
for line in subprocess.run(["fc-list", ":lang=zh", "file"], capture_output=True, text=True).stdout.splitlines():
    font_manager.fontManager.addfont(line.split(":")[0].strip())
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "sans-serif"]
data = json.load(open(sys.argv[1], encoding="utf-8"))["cells"]
READ = {"action": "动作", "text": "文字"}
STREAM = {"frame": "帧流", "motion": "motion 流"}
MODEL = {"mot": "motion", "ctx": "基线"}
SERIES = [("first_later", "读示范首段（已走到第 2 步以后）", "#2a78d6", "o"),
          ("diag_middle", "真对位（中间子目标读对应段）", "#eb6834", "s"),
          ("last_earlier", "读示范末段（还没走到最后一步）", "#1baf7a", "^")]
keys = [(t, m, r, s) for m, r, s in (("mot", "action", "frame"), ("mot", "action", "motion"), ("mot", "text", "frame"), ("mot", "text", "motion"), ("ctx", "action", "frame"), ("ctx", "text", "frame")) for t in ("PatternLock", "RouteStick")]
fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=170)
for i, (t, m, r, s) in enumerate(keys):
    y = len(keys) - 1 - i
    cell = data[f"{t}/{m}/{r}/{s}"]
    for k, (name, label, color, marker) in enumerate(SERIES):
        c = cell[name]
        yy = y + (1 - k) * 0.24
        ax.plot(c["ci95"], [yy, yy], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(c["mean"], yy, marker=marker, color=color, ms=8, mec="white", mew=1.5, ls="none", zorder=3, label=label if i == 0 else None)
ax.axvline(1, color="#444", lw=1.2, zorder=1)
ax.text(1.01, len(keys) - 0.42, "1 = 按 token 数均分", fontsize=9, color="#444", va="bottom")
for y in (9.5, 7.5, 5.5, 3.5, 1.5):
    ax.axhline(y, color="#d9d9d9", lw=0.8, zorder=0)
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([f"{t} · {MODEL[m]} · {READ[r]}读 · {STREAM[s]}" for t, m, r, s in reversed(keys)], fontsize=10)
ax.set_xlabel("富集（实际占比 ÷ 按 token 数均分的占比），局级 95% 区间", fontsize=10)
ax.grid(axis="x", color="#ececec", lw=0.8)
ax.set_axisbelow(True)
ax.set_ylim(-0.6, len(keys) - 0.1)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.09), ncol=3, frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(sys.argv[2])
