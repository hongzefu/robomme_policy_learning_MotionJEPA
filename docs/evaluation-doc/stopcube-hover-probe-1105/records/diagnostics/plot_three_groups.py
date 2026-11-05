"""一次性出图（用户要求只比三组）：距按下步数，记忆组、当前画面、文字，加 [t]、[t, N] 两条基线。

记忆组：motion 模型把历史帧和 motion 合并，基线模型没有 motion，就是历史帧本身。
合并做法：主结果是两组均值拼接（4096 维，再 PCA 到 64 维）；另算按 token 加权平均作对照。
其余做法与 hover_probe 相同，并用历史帧组复算核对 MAE。
用法：python plot_three_groups.py <hover_points.json> <回放库根> <hover-summary.json> <progress-predictions.npz> <输出目录> <worker 数>
"""
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np

from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_attention import GRID, INK, MODEL_STYLE, MUTED, clean_axis, setup_plot


COLOR = {"memory": "#eb6834", "current": "#eda100", "text": "#e87ba4"}
NAME = {"mot": {"memory": "全部记忆（历史帧 + motion）"}, "ctx": {"memory": "历史帧（基线没有 motion）"}}
TITLE = {"current": "当前画面", "text": "文字"}
STATE = {}
HISTORY, MOTION = hp.FEATURE_COLUMN["history"], hp.FEATURE_COLUMN["motion"]


def features(kind, index):
    prefix, counts = STATE["prefix"], STATE["counts"]
    history, motion = prefix[:, index, HISTORY].astype(np.float64), prefix[:, index, MOTION].astype(np.float64)
    if kind == "concat":
        return np.hstack([history, motion])
    if kind == "token_mean":
        n_history, n_motion = counts[:, HISTORY, None].astype(np.float64), counts[:, MOTION, None].astype(np.float64)
        return (history * n_history + motion * n_motion) / (n_history + n_motion)
    return history


def task(item):
    kind, index = item
    samples, Y = STATE["samples"], STATE["Y"]
    X = features(kind, index)
    prediction, inner_values, leak, _ = hp.crossfit(lambda fold: X, samples, Y, True)
    truth = Y[:, 0]
    scores = np.full(hp.OUTER_FOLDS, np.nan)
    for fold, values in inner_values.items():
        keep = np.isfinite(values[:, 0])
        scores[fold] = float(np.nanmean(hp.episode_mean_array(-np.abs(values[:, 0] - truth), keep, samples)))
    return kind, index, prediction[:, 0], scores, leak


def summarize(prediction_by_index, scores_by_index, samples, references):
    """层号 k 对应下标 k+1；每折按内层分选层，拼成折外预测。"""
    truth = samples["labels"]["to_press"]
    layers = np.stack([prediction_by_index[index] for index in range(1, 19)])
    scores = np.stack([scores_by_index[index] for index in range(1, 19)])
    chosen = hp.select_layers(scores)
    selected = hp.compose_prediction(layers, chosen, samples["outer"])
    block = hp.regression_block(samples, selected, truth, references)
    block["layer_by_fold"] = chosen
    block["series"] = [hp.regression_block(samples, prediction_by_index[index], truth)["mae"] for index in range(19)]
    return block, selected


def main():
    points, replay, summary_path, predictions_path, output, workers = sys.argv[1:7]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = hp.load_samples(Path(points))
    bank = hp.load_bank(Path(replay), "mot", samples)
    counts = []
    for key in samples["keys"].tolist():
        with np.load(Path(replay) / "mot" / f"episode_{key}.npz", allow_pickle=False) as archive:
            counts.append(np.asarray(archive["group_tokens"]))
    STATE.update(samples=samples, prefix=bank["prefix"], counts=np.concatenate(counts),
                 Y=samples["labels"]["to_press"][:, None])
    old = np.load(predictions_path)
    truth = samples["labels"]["to_press"]
    if not np.array_equal(old["y"], truth):
        raise SystemExit("预测文件的真值与标签表不符")
    references = {"t": old["t_pred"], "t_N": old["t_N_pred"]}
    kinds = ("concat", "token_mean", "history_check")
    with multiprocessing.get_context("fork").Pool(int(workers)) as pool:
        results = pool.map(task, [(kind, index) for kind in kinds for index in range(19)], chunksize=1)
    if any(item[4] for item in results):
        raise SystemExit("出现同局泄漏")
    blocks, selected = {}, {}
    for kind in kinds:
        predictions = {item[1]: item[2] for item in results if item[0] == kind}
        scores = {item[1]: item[3] for item in results if item[0] == kind}
        blocks[kind], selected[kind] = summarize(predictions, scores, samples, references)

    summary = json.load(open(summary_path))
    expected = summary["models"]["mot"]["features"]["history"]["selected"]["to_press"]["mae"]
    gap = abs(blocks["history_check"]["mae"] - expected)
    print(f"THREE_GROUP_METHOD_CHECK={'PASS' if gap < 1e-6 else 'FAIL'} history_mae={blocks['history_check']['mae']:.4f} expected={expected:.4f}", flush=True)
    if gap >= 1e-6:
        raise SystemExit(1)

    rows = {"mot": {"memory": (blocks["concat"], selected["concat"])}, "ctx": {}}
    for model in ("mot", "ctx"):
        features_old = summary["models"][model]["features"]
        for group, feature in (("memory", "history"), ("current", "current"), ("text", "text")):
            if group in rows[model]:
                continue
            prediction = old[f"{model}_{feature}"]
            block = hp.regression_block(samples, prediction, truth, references)
            block["layer_by_fold"] = features_old[feature]["selected"]["to_press"]["layer_by_fold"]
            block["series"] = [features_old[feature]["input_level"]["to_press"]["mae"]] + [
                layer["to_press"]["mae"] for layer in features_old[feature]["layers"]]
            rows[model][group] = (block, prediction)

    report = {"t_mae": hp.regression_block(samples, old["t_pred"], truth)["mae"],
              "t_N_mae": hp.regression_block(samples, old["t_N_pred"], truth)["mae"], "models": {},
              "token_mean": {key: blocks["token_mean"][key] for key in ("mae", "r2", "layer_by_fold", "series", "minus_t", "minus_t_N")}}
    for model, groups in rows.items():
        report["models"][model] = {}
        for group, (block, _) in groups.items():
            report["models"][model][group] = {key: block[key] for key in ("mae", "r2", "layer_by_fold", "series", "minus_t", "minus_t_N")}
            print(f"THREE_GROUP model={model} group={group} mae={block['mae']:.2f} r2={block['r2']:.3f} input={block['series'][0]:.1f} "
                  f"L0={block['series'][1]:.1f} L17={block['series'][18]:.1f} best={min(block['series']):.1f} "
                  f"minus_t={block['minus_t']['mean']:.2f}{block['minus_t']['ci95']} minus_t_N={block['minus_t_N']['mean']:.2f}{block['minus_t_N']['ci95']} "
                  f"layers={block['layer_by_fold']}", flush=True)
    token = blocks["token_mean"]
    print(f"THREE_GROUP model=mot group=memory_token_mean mae={token['mae']:.2f} r2={token['r2']:.3f} input={token['series'][0]:.1f} "
          f"best={min(token['series']):.1f} minus_t_N={token['minus_t_N']['mean']:.2f}{token['minus_t_N']['ci95']} layers={token['layer_by_fold']}", flush=True)
    (output / "three-groups.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    plt = setup_plot(output)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    handles = {}
    for axis, model in zip(axes, ("mot", "ctx")):
        for group in ("memory", "current", "text"):
            block = rows[model][group][0]
            label = NAME[model]["memory"] if group == "memory" else TITLE[group]
            line, = axis.plot(range(-1, 18), block["series"], color=COLOR[group], linewidth=2, marker="o", markersize=3, label=label)
        first = axis.axhline(report["t_mae"], color=INK, linestyle="--", linewidth=1.2, label=f"[t] 基线（{report['t_mae']:.1f}）")
        second = axis.axhline(report["t_N_mae"], color=INK, linestyle=":", linewidth=1.5, label=f"[t, N] 基线（{report['t_N_mae']:.1f}）")
        axis.set_title(MODEL_STYLE[model][0], loc="left", fontsize=11, color=INK)
        axis.set_xticks([-1, *range(0, 18, 2), 17])
        axis.set_xticklabels(["输入", *[str(value) for value in range(0, 18, 2)], "17"])
        axis.set_xlabel("层号", color=MUTED)
        axis.set_ylim(35, 85)
        axis.grid(color=GRID, linewidth=0.7)
        clean_axis(axis)
        axis.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2)
    axes[0].set_ylabel("距按下步数的折外 MAE（步，越低越好）", color=MUTED)
    figure.tight_layout()
    figure.savefig(output / "hover-three-groups-layers.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 8.2), sharex=True, sharey=True)
    low, high = float(truth.min()), float(truth.max())
    for row, model in enumerate(("mot", "ctx")):
        for column, group in enumerate(("memory", "current", "text")):
            axis = axes[row, column]
            block, prediction = rows[model][group]
            label = NAME[model]["memory"] if group == "memory" else TITLE[group]
            axis.scatter(truth, prediction, s=5, color=COLOR[group], alpha=0.15, linewidths=0)
            axis.plot(*hp.binned_curve(truth, prediction), color=COLOR[group], linewidth=2.2)
            diagonal, = axis.plot([low, high], [low, high], color=MUTED, linewidth=0.8)
            first, = axis.plot(*hp.binned_curve(truth, old["t_pred"]), color=INK, linestyle="--", linewidth=1.1)
            second, = axis.plot(*hp.binned_curve(truth, old["t_N_pred"]), color=INK, linestyle=":", linewidth=1.3)
            axis.set_title(f"{MODEL_STYLE[model][0]} · {label}\nMAE {block['mae']:.1f} 步，R² {block['r2']:.2f}", loc="left", fontsize=10, color=INK)
            axis.set_xlim(0, 510)
            axis.set_ylim(0, 510)
            axis.set_aspect("equal")
            axis.grid(color=GRID, linewidth=0.7)
            clean_axis(axis)
            if row == 1:
                axis.set_xlabel("真实距按下步数", color=MUTED)
            if column == 0:
                axis.set_ylabel("折外预测（步）", color=MUTED)
    figure.legend([diagonal, first, second], ["预测等于真值", f"[t] 基线（MAE {report['t_mae']:.1f}）", f"[t, N] 基线（MAE {report['t_N_mae']:.1f}）"],
                  loc="upper center", ncol=3, frameon=False, fontsize=10.5)
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    figure.savefig(output / "hover-three-groups-progress.png", dpi=150)
    plt.close(figure)
    print("THREE_GROUP_DONE", flush=True)


if __name__ == "__main__":
    main()
