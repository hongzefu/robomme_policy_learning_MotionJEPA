"""一次性出图（用户要求只看距按下步数）：逐层 MAE，只读 hover-summary.json，不重算。

用法：python plot_to_press_layers.py <hover-summary.json> <输出目录>
"""
import json
import sys
from pathlib import Path

from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_attention import GRID, INK, MODEL_STYLE, MUTED, clean_axis, setup_plot


ORDER = ("history", "motion", "latest_motion", "current", "text")
LATEST_COLOR = "#2a78d6"


def main():
    summary = json.load(open(sys.argv[1]))
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    t_mae = summary["time_baseline"]["t"]["to_press"]["mae"]
    t_n_mae = summary["time_baseline"]["t_N"]["to_press"]["mae"]
    plt = setup_plot(output)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    handles = {}
    for axis, model in zip(axes, hp.MODEL_ORDER):
        features = summary["models"][model]["features"]
        for feature in ORDER:
            if feature not in features:
                continue
            item = features[feature]
            color = LATEST_COLOR if feature == "latest_motion" else hp.FEATURE_COLOR[feature]
            series = [item["input_level"]["to_press"]["mae"]] + [layer["to_press"]["mae"] for layer in item["layers"]]
            handles[hp.FEATURE_NAME[feature]], = axis.plot(range(-1, 18), series, color=color, linewidth=1.9, marker="o", markersize=3)
            selected = item["selected"]["to_press"]
            best = min(range(19), key=lambda index: series[index])
            print(f"TO_PRESS model={model} feature={feature} input={series[0]:.1f} best_layer={best - 1} best={series[best]:.1f} "
                  f"last={series[-1]:.1f} selected_mae={selected['mae']:.1f} layer_by_fold={selected['layer_by_fold']}", flush=True)
        handles[f"[t] 基线（{t_mae:.1f}）"] = axis.axhline(t_mae, color=INK, linestyle="--", linewidth=1.2)
        handles[f"[t, N] 基线（{t_n_mae:.1f}）"] = axis.axhline(t_n_mae, color=INK, linestyle=":", linewidth=1.5)
        axis.set_title(MODEL_STYLE[model][0], loc="left", fontsize=11, color=INK)
        axis.set_xticks([-1, *range(0, 18, 2), 17])
        axis.set_xticklabels(["输入", *[str(value) for value in range(0, 18, 2)], "17"])
        axis.set_xlabel("层号", color=MUTED)
        axis.set_ylim(35, 85)
        axis.grid(color=GRID, linewidth=0.7)
        clean_axis(axis)
    axes[0].set_ylabel("距按下步数的折外 MAE（步，越低越好）", color=MUTED)
    figure.legend(list(handles.values()), list(handles), loc="upper center", ncol=4, frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = output / "hover-to-press-layers.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"PLOT={path}", flush=True)


if __name__ == "__main__":
    main()
