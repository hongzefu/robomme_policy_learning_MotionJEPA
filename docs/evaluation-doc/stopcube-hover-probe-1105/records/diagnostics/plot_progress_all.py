"""一次性出图（用户要求「其他组也画出来」）：距按下步数的折外预测，全部特征组各一格。

调用 hover_probe.analyze 重算，不改分析代码；先核对每组 MAE 与原 hover-summary.json 相同，不同就报错。
用法：python plot_progress_all.py <hover_points.json> <回放库根> <原 hover-summary.json> <输出目录> <worker 数>
"""
import json
import sys
from pathlib import Path

import numpy as np

from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_attention import GRID, INK, MODEL_STYLE, MUTED, clean_axis, setup_plot


ORDER = ("history", "motion", "latest_motion", "current", "text")
LATEST_COLOR = "#2a78d6"


def main():
    points, replay, original, output, workers = sys.argv[1:6]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = hp.load_samples(Path(points))
    banks = {label: hp.load_bank(Path(replay), label, samples) for label in hp.MODEL_ORDER}
    summary, predictions = hp.analyze(samples, banks, int(workers), False)
    old = json.load(open(original))
    y = samples["labels"]["to_press"]
    saved = {"y": y, "t_pred": predictions["references"]["t"]["to_press"], "t_N_pred": predictions["references"]["t_N"]["to_press"]}
    for model in hp.MODEL_ORDER:
        for feature in summary["models"][model]["features"]:
            new_entry = summary["models"][model]["features"][feature]["selected"]["to_press"]
            old_entry = old["models"][model]["features"][feature]["selected"]["to_press"]
            if abs(new_entry["mae"] - old_entry["mae"]) > 1e-6:
                raise SystemExit(f"{model} {feature} MAE 与原结果不同：{new_entry['mae']} 对 {old_entry['mae']}")
            saved[f"{model}_{feature}"] = predictions["selected"][(model, feature, "to_press")]
            print(f"PROGRESS_MAE model={model} feature={feature} mae={new_entry['mae']:.2f} r2={new_entry['r2']:.3f} "
                  f"layer={new_entry['layer']} layer_by_fold={new_entry['layer_by_fold']}", flush=True)
    np.savez(output / "progress-predictions.npz", **saved)
    print("PROGRESS_MAE_MATCH=PASS", flush=True)

    plt = setup_plot(output)
    figure, axes = plt.subplots(2, len(ORDER), figsize=(3.5 * len(ORDER), 8.2), sharex=True, sharey=True)
    low, high = float(y.min()), float(y.max())
    handles = {}
    for row, model in enumerate(hp.MODEL_ORDER):
        for column, feature in enumerate(ORDER):
            axis = axes[row, column]
            title = f"{MODEL_STYLE[model][0]} · {hp.FEATURE_NAME[feature]}"
            axis.grid(color=GRID, linewidth=0.7)
            clean_axis(axis)
            if feature not in summary["models"][model]["features"]:
                axis.set_title(title, loc="left", fontsize=10, color=MUTED)
                axis.text(0.5, 0.5, "基线没有这一组", transform=axis.transAxes, ha="center", va="center", color=MUTED, fontsize=11)
                axis.set_xlim(0, 510)
                axis.set_ylim(0, 510)
                axis.set_aspect("equal")
                if row == 1:
                    axis.set_xlabel("真实距按下步数", color=MUTED)
                continue
            else:
                entry = summary["models"][model]["features"][feature]["selected"]["to_press"]
                prediction = saved[f"{model}_{feature}"]
                color = LATEST_COLOR if feature == "latest_motion" else hp.FEATURE_COLOR[feature]
                axis.scatter(y, prediction, s=5, color=color, alpha=0.15, linewidths=0)
                axis.plot(*hp.binned_curve(y, prediction), color=color, linewidth=2.2)
                axis.set_title(f"{title}\nMAE {entry['mae']:.1f} 步，R² {entry['r2']:.2f}", loc="left", fontsize=10, color=INK)
            handles["等于真值"], = axis.plot([low, high], [low, high], color=MUTED, linewidth=0.8)
            handles["[t] 基线"], = axis.plot(*hp.binned_curve(y, saved["t_pred"]), color=INK, linestyle="--", linewidth=1.1)
            handles["[t, N] 基线"], = axis.plot(*hp.binned_curve(y, saved["t_N_pred"]), color=INK, linestyle=":", linewidth=1.3)
            axis.set_xlim(0, 510)
            axis.set_ylim(0, 510)
            axis.set_aspect("equal")
            if row == 1:
                axis.set_xlabel("真实距按下步数", color=MUTED)
            if column == 0:
                axis.set_ylabel("折外预测（步）", color=MUTED)
    t_mae, t_n_mae = summary["time_baseline"]["t"]["to_press"]["mae"], summary["time_baseline"]["t_N"]["to_press"]["mae"]
    labels = {"等于真值": "预测等于真值", "[t] 基线": f"[t] 基线（MAE {t_mae:.1f}）", "[t, N] 基线": f"[t, N] 基线（MAE {t_n_mae:.1f}）"}
    figure.legend(list(handles.values()), [labels[key] for key in handles], loc="upper center", ncol=3, frameon=False, fontsize=10.5)
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    path = output / "hover-progress-all.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"PROGRESS_PLOT={path}", flush=True)


if __name__ == "__main__":
    main()
