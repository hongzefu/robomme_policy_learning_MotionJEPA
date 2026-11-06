"""一次性导出：论文图用的原始数据，写成不依赖本仓库的 CSV。

直接调用 make_paper_figs 的 collect_fits，拟合线、置信带和区间与图上是同一批数。写完后从 CSV 读回，
重算斜率和每局等权 R²，再和来源对一遍，不一致就报错。不重算回归，不跑模型。
用法：python export_paper_data.py <输出目录>
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import make_paper_figs as F

MODEL = {"ctx": "baseline", "mot": "motion"}
READER = {"action": "Action token", "文字": "Text", "当前视觉": "Current frame", "历史视觉": "History frames", "motion": "Motion"}
TARGET = {"press": "press_now", "c": "passes_elapsed"}


def write(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def layer_name(index):
    return "input" if index == -1 else str(index)


def r2_rows(probe):
    rows = []
    for model, name in MODEL.items():
        layers = probe["models"][model]["labels"][F.LAB]["layers"]
        for index, item in enumerate(layers["action"]):
            rows.append((name, "Action token", layer_name(index), index, item["r2"], item["mae"]))
        for reader, group in READER.items():
            if reader == "action" or layers["prefix"].get(reader) is None:
                continue
            first = layers["prefix_layer0"][reader]
            rows.append((name, group, "input", -1, first["r2"], first["mae"]))
            for index, item in enumerate(layers["prefix"][reader]):
                rows.append((name, group, layer_name(index), index, item["r2"], item["mae"]))
    return rows


def accuracy_rows(classes):
    rows = []
    for model, name in MODEL.items():
        for reader, group in READER.items():
            table = classes[model]["readers"].get(reader)
            if table is None:
                continue
            for layer in sorted(table, key=int):
                for key, target in TARGET.items():
                    item = table[layer][key]["point"]
                    rows.append((name, group, target, layer_name(int(layer)), int(layer), item["mean"], *item["ci95"], item["n"], item["episodes"]))
    return rows


def reference_rows(fit, classes):
    rows = []
    for model, name in MODEL.items():
        rows.append((name, "time_only", "r2", "", fit[model][F.LAB]["time"]["r2"], "", "", "drawn in R2 figures"))
        rows.append((name, "time_and_target_count", "r2", "", fit[model][F.LAB]["time_N"]["r2"], "", "", "not drawn"))
        for key, target in TARGET.items():
            item = classes[model]["time_N"][key]["point"]
            rows.append((name, "time_and_target_count", "point_accuracy", target, item["mean"], *item["ci95"], "not drawn"))
    return rows


def episode_r2(y, prediction, episode):
    _, inverse, counts = np.unique(episode, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    center = (weights * y).sum() / weights.sum()
    return 1 - (weights * (y - prediction) ** 2).sum() / (weights * (y - center) ** 2).sum()


def verify(output, fits, fit):
    with open(output / "fit_points.csv") as handle:
        points = list(csv.DictReader(handle))
    checked = 0
    for model, name in MODEL.items():
        for group, entry in fits[model]["groups"].items():
            chosen = [row for row in points if row["model"] == name and row["group"] == group]
            x = np.array([float(row["true_steps"]) for row in chosen])
            y = np.array([float(row["predicted_steps"]) for row in chosen])
            episode = np.array([row["episode"] for row in chosen])
            _, slope = F.wls_line(x, y, episode)
            if abs(slope - entry["b"]) > 1e-6 or abs(episode_r2(x, y, episode) - entry["r2"]) > 1e-6:
                raise SystemExit(f"读回不符：{model} {group}")
            checked += 1
    return checked


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    probe, fit = F.load_json(F.SRC_PROBE), F.load_json(F.SRC_FIT)
    classes = F.load_json(F.SRC_CLS)
    npz = np.load(F.SRC_NPZ, allow_pickle=False)
    fits, xlim, ylim = F.collect_fits(fit, npz)

    points, lines, bands = [], [], []
    for model, name in MODEL.items():
        episodes = npz[f"{model}/{F.LAB}/groups"]
        for group, entry in fits[model]["groups"].items():
            if len(entry["x"]) != len(episodes):
                raise SystemExit(f"存在无预测的点，局编号无法对齐：{model} {group}")
            points += [(name, group, str(key), float(x), float(y)) for key, x, y in zip(episodes, entry["x"], entry["y"])]
            lines.append((name, group, entry["r2"], entry["b"], entry["b_lo"], entry["b_hi"], entry["a"], entry["n"], fits[model]["n_ep"]))
            bands += [(name, group, float(x), float(entry["a"] + entry["b"] * x), float(low), float(high))
                      for x, low, high in zip(entry["xg"], entry["band_lo"], entry["band_hi"])]

    counts = {
        "r2_by_layer.csv": write(output / "r2_by_layer.csv", ["model", "group", "layer", "layer_index", "r2", "mae_steps"], r2_rows(probe)),
        "fit_points.csv": write(output / "fit_points.csv", ["model", "group", "episode", "true_steps", "predicted_steps"], points),
        "fit_lines.csv": write(output / "fit_lines.csv", ["model", "group", "r2", "slope", "slope_ci_low", "slope_ci_high", "intercept", "n_points", "n_episodes"], lines),
        "fit_bands.csv": write(output / "fit_bands.csv", ["model", "group", "true_steps", "fit", "ci_low", "ci_high"], bands),
        "point_accuracy_by_layer.csv": write(output / "point_accuracy_by_layer.csv", ["model", "group", "target", "layer", "layer_index", "accuracy", "ci_low", "ci_high", "n_points", "n_episodes"], accuracy_rows(classes)),
        "reference_lines.csv": write(output / "reference_lines.csv", ["model", "reference", "metric", "target", "value", "ci_low", "ci_high", "note"], reference_rows(fit, classes)),
    }
    for name, count in counts.items():
        print(f"{name} rows={count}")
    print(f"axis_limits xlim={xlim} ylim={ylim}")
    checked = verify(output, fits, fit)
    print(f"PAPER_DATA_EXPORT=PASS files={len(counts)} fit_groups_checked={checked}")


if __name__ == "__main__":
    main()
