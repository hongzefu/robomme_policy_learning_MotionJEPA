"""attention 画图（计划 1103-interpret-motion-attention-plan.html 第一部分「画图」、第二部分 §1.6）。

第一阶段只实现 fig_key_step.png（用户 2026-11-03 裁决「只写 fig_key_step」）：
「揭开」那次推理，两模型并排。左列 ctx512 基线、右列 ctx512+motion；
上排帧流 32 柱（x = 全域帧号 frame_times，y = 该帧 16 token 注意力之和，[0,32) 浅绿底）；
下排 motion 流每窗一柱（画在窗口起点、宽 33，交换区间 [64,64+50k) 浅橙底；基线格写「无 motion 流」）；
每排角标「窗口内占比 x%」与「均匀期望 y%」；图下一行 episode_id／k／关键推理序号／全域帧号。

输入是 aggregate_attn.py --key-infer 写出的 <records>/summary.json 的 key_steps。A/B/C/D 四类图属第二阶段，不在此实现。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_attn import MOTION_WINDOW, VISIBLE, swap_interval  # noqa: E402


# 中文字体：本机 fc-list :lang=zh 实测有 Noto Sans CJK SC；找不到时改用英文标注。
_CJK_FAMILIES = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "WenQuanYi Zen Hei"]


def _pick_cjk_font() -> str | None:
    try:
        out = subprocess.run(["fc-list", ":lang=zh", "file"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    files = [ln.split(":")[0].strip() for ln in out.splitlines() if ln.strip()]
    for f in files:
        try:
            font_manager.fontManager.addfont(f)
        except (OSError, RuntimeError):
            pass
    names = {fe.name for fe in font_manager.fontManager.ttflist}
    return next((n for n in _CJK_FAMILIES if n in names), None)


CJK = _pick_cjk_font()
if CJK:
    plt.rcParams["font.family"] = [CJK, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

T = {  # 文案（中文／英文两套）
    "base": ("ctx512 基线", "ctx512 baseline"),
    "mot": ("ctx512+motion", "ctx512+motion"),
    "frame_y": ("帧流注意力（16 token 之和）", "frame attn (sum of 16 tokens)"),
    "motion_y": ("motion 流注意力（每窗）", "motion attn (per window)"),
    "x": ("全域帧号", "global frame index"),
    "no_motion": ("无 motion 流", "no motion stream"),
    "vis": ("裸露窗口 [0,32)", "uncovered [0,32)"),
    "swap": ("交换区间", "swap"),
    "share": ("窗口内占比", "in-window share"),
    "uniform": ("均匀期望", "uniform"),
    "infer": ("推理", "infer"),
    "frame": ("帧", "frame"),
}


GREEN = "#b7e4c7"   # 裸露窗口底色
ORANGE = "#fbd38d"  # 交换区间底色


def t(key: str) -> str:
    return T[key][0 if CJK else 1]


def _corner(ax, label: str, color: str, share, uniform, masses) -> None:
    """右上角标：底色说明 + 窗口内占比 x% + 均匀期望 y%；纵轴上沿留 45% 空白，角标不压柱。"""
    top = max(masses) if masses else 1.0
    ax.set_ylim(0, top * 1.45 if top > 0 else 1.0)
    txt = f"{label}\n{t('share')} {share * 100:.1f}%   {t('uniform')} {uniform * 100:.1f}%"
    ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="0.55", alpha=0.95))


def plot_key_step(base: dict, mot: dict, path: Path) -> None:
    """2 列（基线／motion）× 2 行（帧流／motion 流）。"""
    k = mot.get("swap_times") if mot.get("swap_times") is not None else base.get("swap_times")
    xmax = max(base["step_idx"], mot["step_idx"]) + 1
    for rec in (base, mot):
        for x in rec["frame"]:
            xmax = max(xmax, x["t"] + 1)
        for x in rec["motion"]:
            xmax = max(xmax, x["t"] + MOTION_WINDOW)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), sharex=True, constrained_layout=True)
    for col, (rec, title) in enumerate(((base, t("base")), (mot, t("mot")))):
        ax = axes[0, col]
        ft = [x["t"] for x in rec["frame"]]
        fm = [x["mass"] for x in rec["frame"]]
        gaps = sorted({b - a for a, b in zip(sorted(ft), sorted(ft)[1:]) if b > a})
        width = max(1.0, 0.8 * gaps[0]) if gaps else 1.0
        ax.axvspan(*VISIBLE, color=GREEN, alpha=0.6, zorder=0)
        ax.bar(ft, fm, width=width, color="#2b6cb0", zorder=2)
        ax.axvline(rec["step_idx"], color="0.3", ls="--", lw=0.8, zorder=1)
        ax.set_title(f"{title}  ({t('infer')} #{rec['infer_seq']}, {t('frame')} {rec['step_idx']})", fontsize=11)
        ax.set_ylabel(t("frame_y"), fontsize=9)
        _corner(ax, t("vis"), GREEN, rec["vis_share"], rec["vis_uniform"], fm)

        ax = axes[1, col]
        if not rec["has_motion"]:
            ax.text(0.5, 0.5, t("no_motion"), transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="0.45")
            ax.set_yticks([])
        else:
            lo, hi = swap_interval(k)
            ax.axvspan(lo, hi, color=ORANGE, alpha=0.6, zorder=0)
            mt = [x["t"] for x in rec["motion"]]
            mm = [x["mass"] for x in rec["motion"]]
            ax.bar(mt, mm, width=MOTION_WINDOW, align="edge", color="#c05621", alpha=0.55,
                   edgecolor="#7b341e", linewidth=0.6, zorder=2)
            ax.axvline(rec["step_idx"], color="0.3", ls="--", lw=0.8, zorder=1)
            _corner(ax, f"{t('swap')} [{lo},{hi}) k={k}", ORANGE, rec["swap_share"], rec["swap_uniform"], mm)
        ax.set_ylabel(t("motion_y"), fontsize=9)
        ax.set_xlabel(t("x"), fontsize=9)
        ax.set_xlim(0, xmax)
    info = (f"episode_id={mot['episode_id']}   k={k}   "
            f"{t('base')}: {t('infer')} #{base['infer_seq']} / {t('frame')} {base['step_idx']}   "
            f"{t('mot')}: {t('infer')} #{mot['infer_seq']} / {t('frame')} {mot['step_idx']}")
    fig.text(0.5, -0.02, info, ha="center", va="top", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True, help="aggregate_attn.py 的 --out 目录（含 summary.json）")
    ap.add_argument("--out", type=Path, required=True, help="图目录")
    ap.add_argument("--only", choices=["key_step"], default="key_step", help="第一阶段只有 key_step")
    args = ap.parse_args(argv)

    summary = json.loads((args.records / "summary.json").read_text())
    by_ep: dict = {}
    for ks in summary["key_steps"]:
        by_ep.setdefault(ks["episode_id"], {})["mot" if ks["has_motion"] else "base"] = ks
    pairs = {ep: v for ep, v in by_ep.items() if "base" in v and "mot" in v}
    for ep, v in by_ep.items():
        if ep not in pairs:
            print(f"episode {ep} 缺 {'基线' if 'base' not in v else 'motion'} 一侧的 key_step，跳过", flush=True)
    figs = 0
    for ep, v in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        name = "fig_key_step.png" if len(pairs) == 1 else f"fig_key_step_ep{ep}.png"
        plot_key_step(v["base"], v["mot"], args.out / name)
        print(f"写出 {args.out / name}", flush=True)
        figs += 1
    if not CJK:
        print("本机找不到 CJK 字体，图上文字改用英文", flush=True)
    print(f"ATTN_PLOT={'PASS' if figs > 0 else 'FAIL'} figs={figs}", flush=True)
    return 0 if figs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
