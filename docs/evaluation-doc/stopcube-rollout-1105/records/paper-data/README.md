# StopCube rollout 论文图数据

这里是论文图用到的全部原始数据，都是 CSV，不依赖原仓库。由 `records/diagnostics/export_paper_data.py` 导出，判定行 `PAPER_DATA_EXPORT=PASS files=6 fit_groups_checked=9`。

## 数据口径

- **任务和数据：** StopCube 真实 rollout。基线 50 局 561 个决策点，motion 模型 49 局 640 个决策点。
- **探针：** 线性岭回归读隐状态，预测离正确按钮窗口还有几步（窗口中心减当前步，可为负）。按局 5 折，所有预测都是折外结果，每局等权。
- **这些结果的范围：** 只说明可线性解码，不说明模型用了这些信息。

## 文件和图的对应

| 文件 | 用于哪张图 | 行数 |
| --- | --- | ---: |
| `r2_by_layer.csv` | `fig-r2-layers-baseline`、`fig-r2-layers-motion` | 169 |
| `fit_points.csv` | `fig-r2-fit-baseline`、`fig-r2-fit-motion` 的离散点 | 5444 |
| `fit_lines.csv` | 同上的拟合线、R²、斜率区间 | 9 |
| `fit_bands.csv` | 同上的 95% 置信带 | 909 |
| `point_accuracy_by_layer.csv` | `fig-point-accuracy` | 338 |
| `reference_lines.csv` | R² 图的 Time only 点线，以及没有画的时间加次数基准 | 8 |

## 列的含义

通用列：

- `model`：`baseline` 或 `motion`。
- `group`：`Action token`、`Text`、`Current frame`、`History frames`、`Motion`。基线没有 `Motion`。
- `layer`：`input` 或 `0`～`17`；`layer_index` 是同一个值的数字形式，`input` 记为 -1。`Action token` 没有 `input`。

各文件的其他列：

- `r2_by_layer.csv`：`r2` 是每局等权的折外 R²，`mae_steps` 是折外平均绝对误差（步）。
- `fit_points.csv`：一行是一个组的一个决策点。`episode` 是局编号，`true_steps` 是真实步数，`predicted_steps` 是第 17 层的折外预测。
- `fit_lines.csv`：第 17 层预测步数对真实步数的直线，`predicted = intercept + slope × true`，每局等权加权最小二乘。`slope_ci_low`、`slope_ci_high` 是斜率的 95% 区间。
- `fit_bands.csv`：每组在真实步数范围内取 101 个点，`fit` 是直线上的值，`ci_low`、`ci_high` 是这条直线的 95% 置信带。
- `point_accuracy_by_layer.csv`：`target` 是 `press_now`（该不该按）或 `passes_elapsed`（已过次数）。`accuracy` 是每局等权的逐点准确率，`ci_low`、`ci_high` 是按局重采样的 95% 区间，`n_points`、`n_episodes` 是点数和局数。
- `reference_lines.csv`：`reference` 为 `time_only`（只用时间步）或 `time_and_target_count`（时间步加目标次数）。`metric` 为 `r2` 或 `point_accuracy`，`value` 是数值，`note` 写明是否画在图上。`time_and_target_count` 按用户要求没有画，只留数值。

## 区间怎么算的

- **置信带和斜率区间：** 按局重采样 2000 次（seed 42），同一局的点一起抽，每次重算每局等权的直线，取 2.5% 和 97.5% 分位。它是拟合线的区间，不是单个点的预测范围。
- **准确率区间：** 对每局准确率按局重采样 2000 次（seed 42），取 2.5% 和 97.5% 分位。
- **R²：** 没有区间。

## 逐点准确率不是分类器的结果

没有训练分类器。它是回归预测用真实的方块周期 M 和目标次数 N 换算出来的：

- `press_now`：`0 < 预测步数 ≤ M` 判为该按。
- `passes_elapsed`：`N − ceil(预测步数 / M)`。

## 逐点预测（`predictions/`）

这三份 npz 是回归的折外预测，用 `numpy.load` 读取，不需要 pickle。模型键 `ctx` 是基线，`mot` 是 motion 模型；读数键 `action`、`文字`、`当前视觉`、`历史视觉`、`motion` 依次对应上面的五个组。

| 文件 | 键 | 内容 |
| --- | --- | --- |
| `fit-predictions.npz` | `{模型}/{标签}/y`、`groups`、`{读数}` | 第 17 层的真实步数、局编号和预测，另有 `time`、`time_N` 基准。标签 `q4` 是离正确窗口，`q3` 是离自己按 |
| `class-layers-predictions.npz` | `{模型}/{读数}/{层}` | 离正确窗口的所有层预测，float32。层 `-1` 是输入，`0`～`17` 是各层 |
| `class-derived.npz` | `{模型}/press_pairs_yes_no`、`press_true_ratio`、`ratio/{读数}` | 第 17 层换算分类和配对下标，论文图不用，只为留档 |

`class-layers-predictions.npz` 没有真实步数和局编号。它的点序与 `fit-predictions.npz` 的 `{模型}/q4/y`、`{模型}/q4/groups` 相同，要配合使用。

## 复现原图的设置

| 项目 | 取值 |
| --- | --- |
| 拟合图的 x 范围 | -47.72 ～ 511.72 |
| 拟合图的 y 范围 | -244.63 ～ 436.44 |
| 拟合图 | x、y 等比例，两个模型范围相同 |
| R² 图的 y 范围 | -0.25 ～ 1.0，两个模型相同 |
| 准确率图的 y 范围 | 0 ～ 1 |
| 颜色 | Action token `#2a78d6`、Text `#eb6834`、Current frame `#1baf7a`、History frames `#eda100`、Motion `#e87ba4` |
