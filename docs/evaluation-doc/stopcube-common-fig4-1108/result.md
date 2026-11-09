# StopCube 共同轨迹 Fig. 4 重跑：结果

**一句话结论**：两个模型都在同一批官方示教上回放（98 局，2344 个点，两边取点相同）。按 Fig. 4 原口径逐层读「距按下步数」时，旧图里 MotionMem 的 Text 优势（0.77 对 0.51）基本消失：第 17 层 MotionMem 反而略低（Text 0.382 对 0.489）。只有训练折内选层时，MotionMem Text 在第 0 层较高（0.652 对 0.523）。

探索性分析里有几处一致的 MotionMem 优势，都和「记忆 / 速度」有关：
1. 在 `[t,N]` 上加入表征后，MotionMem 每一组都能补上 t、N 之外的信息（主要是方块速度 M），FrameSamp 几乎补不上。
2. MotionMem 的 Text 误差不随目标次数 N 增大，FrameSamp 随 N 线性变差。
3. 画面几乎相同、但已过次数不同的配对点，MotionMem 读对的比例高 0.2–0.3。

以上都只说明「可解码」，不说明模型用了这些信息；这些局两个模型训练时都见过；回放是开环看专家轨迹。

判定行与起跑信息见 [launch.md](launch.md)。

## 一、全量与验收判定行

| 判定行 | 值 |
| --- | --- |
| 全量回放 | `REPLAY_RUN=PASS model=ctx points=2344 expected=2344`、`EXIT_CODE=0`（累计 7345 秒，3.13 秒／点）；`REPLAY_RUN=PASS model=mot points=2344 expected=2344`、`EXIT_CODE=0`（8104 秒，3.46 秒／点） |
| 前缀与 1105 逐局一致 | `PREFIX_MATCH=PASS model=ctx episodes=98 points=2344 maxabs=0`；`PREFIX_MATCH=PASS model=mot episodes=98 points=2344 maxabs=0` |
| 新分析入口复现 1105 | `REPRO_1105=PASS maxabs=6.661e-16 compared=137 missing=0` |
| 取点、泄漏、oracle | `SAME_POINTS=PASS points=2344 diff=0`；`GROUP_LEAK=PASS leak=0`；`ORACLE_EQUAL=PASS r2_a=0.44707264429362636 r2_b=0.44707264429362636` |
| 冒烟（两模型） | `ACTION_PARITY=PASS points=21 maxabs=0`；`HIDDEN_PRECISION=PASS group=action min_dist_over_err=inf path_maxabs=0`（详见 launch.md） |

## 二、Fig. 4 原口径（预先写定的主读数）

标签是距按下步数 `M·N − M//2 − t`。按局 5 折，训练折内标准化、PCA 到 64 维、内层选 λ，每局等权。参考线：`time_only` R² 0.059（MAE 71.8 步）；oracle `time_and_target_count` R² 0.447（MAE 55.3 步），两模型相同。

| 模型 | 组 | 第 17 层 R² | 最佳层 R²（层） |
| --- | --- | ---: | ---: |
| FrameSamp | Text | 0.489 | 0.523（12） |
| FrameSamp | Current frame | 0.490 | 0.558（10） |
| FrameSamp | Action token | 0.483 | 0.496（7） |
| MotionMem | Text | 0.382 | 0.652（0） |
| MotionMem | Current frame | 0.478 | 0.580（7） |
| MotionMem | Action token | 0.390 | 0.482（7） |

「最佳层」一列是全数据上的最大值，只作描述。训练折内选层的结果见第三节 A。

第 17 层拟合线（y = 折外预测，x = 真值，每局等权加权最小二乘，斜率区间为按局重采样 2000 次）：FrameSamp Text 斜率 0.474 [0.421, 0.536]、Current 0.498、Action 0.481；MotionMem Text 0.431 [0.370, 0.503]、Current 0.485、Action 0.392。

**与旧 Fig. 4 对照**：旧图用的是各自闭环的 50 局。MotionMem Text 19 个读数里有 17 个高于它自己的 oracle 0.551（峰值 0.770）；FrameSamp 全部低于 0.616。换成共同轨迹后，两条 Text 曲线在第 8 层以后几乎重合，都比 oracle 略高 0.05–0.1。旧图的差距主要来自两个策略访问的状态不同。图：[records/figures/common-vs-oldfig4-1105.png](records/figures/common-vs-oldfig4-1105.png)。

三张表的列名与旧表相同，在 [records/fig4-tables/](records/fig4-tables/)。

## 三、探索性角度（不是预先登记的读数）

统计单位是局。跨模型差先在同一局内两模型相减，再按局重采样（2000 次，seed 42）。选层在训练折内完成，不看测试折。脚本在 [records/diagnostics/](records/diagnostics/)，数字在 [records/explore/](records/explore/)。

### A. 训练折内选层后的配对差（MotionMem − FrameSamp，> 0 表示 MotionMem 更好）

| 读数 | Text | Current frame | Action token | 记忆组（mot motion 对 ctx history） |
| --- | --- | --- | --- | --- |
| 距按下步数（−MAE，步） | +6.61 [0.44, 12.76] | +2.93 [−1.02, 6.67] | −2.07 [−6.48, 2.03] | +5.83 [1.13, 10.57] |
| 速度 M（准确率） | +0.207 [0.156, 0.261] | +0.086 [0.046, 0.126] | +0.016 [−0.022, 0.052] | +0.217 [0.142, 0.291] |
| 已过次数 | +0.070 [0.049, 0.093] | +0.072 [0.054, 0.089] | +0.138 [0.105, 0.172] | +0.091 [0.061, 0.123] |
| 已过次数，画面几乎相同的配对（50 局，608 对） | +0.304 [0.223, 0.385] | +0.232 [0.162, 0.307] | +0.209 [0.129, 0.284] | +0.248 [0.145, 0.356] |
| 该不该按 | +0.016 [−0.014, 0.044] | +0.047 [0.023, 0.071] | +0.026 [0.008, 0.046] | −0.017 [−0.050, 0.018] |
| 剩余次数 | **−0.078 [−0.114, −0.045]** | +0.053 [0.026, 0.083] | −0.014 [−0.038, 0.011] | +0.048 [−0.003, 0.101] |
| 方块位置（−MAE） | +0.029 [0.017, 0.042] | +0.024 [0.014, 0.035] | +0.067 [0.052, 0.083] | +0.046 [0.032, 0.060] |

对 MotionMem 不利的一项：Text 读剩余次数低 0.078。图：[records/figures/forest.png](records/figures/forest.png)。

### B. oracle 之外的信息：`[t,N]` 独热 + 组特征（训练折内 PCA 64）

只用 `[t,N]` 时 R² 0.447、MAE 55.3 步；`[t,N,M]` 加性独热为 0.926、20.2 步（缺的变量就是速度 M）。

**口径修正（探索审查 F1）**：最初的版本里，独热列和 PCA 得分共用一个岭惩罚 λ。表征没有可补信息的一方会被迫选大 λ，把 `[t,N]` 也一起压掉，于是它的增量被压成负值，跨模型差被放大。修正版把独热列放大 100 倍（环境变量 `ONEHOT_SCALE=100`），使 `[t,N]` 几乎不受惩罚。下面三张表都以修正版为准（`records/explore/beyond_oracle_scale100.json`、`closedloop_beyond_oracle_scale100.json`）；修正前的版本保留在 `beyond_oracle.json`、`closedloop_beyond_oracle.json`，只作对照。

共同轨迹上，加入组特征后的 R²，以及 MAE 相对只用 `[t,N]` 减少的步数（局级 95% 区间）：

| 模型 | 组（层） | R² | MAE 减少（步） |
| --- | --- | ---: | --- |
| FrameSamp | Text（8／17） | 0.558／0.547 | 7.8 [5.2, 10.4]／6.1 [3.2, 8.8] |
| FrameSamp | History frames（8） | 0.481 | 2.6 [−0.5, 5.1] |
| FrameSamp | Current frame（10） | 0.598 | 9.3 [6.4, 12.5] |
| FrameSamp | Action token，各层 | 0.45–0.58 | 0–8 |
| MotionMem | Text（0／8／17） | 0.837／0.826／0.781 | 26.7 [22.3, 31.2]／26.2 [21.4, 30.7]／22.1 [18.1, 25.9] |
| MotionMem | History frames（8） | 0.864 | 29.2 [23.3, 34.9] |
| MotionMem | Motion（输入层） | 0.862 | 28.2 [22.3, 33.7] |
| MotionMem | Current frame（7） | 0.725 | 16.7 [12.3, 21.2] |
| MotionMem | Action token（7） | 0.728 | 17.5 [13.3, 21.7] |

**同点配对的跨模型差**：同一批点、同一分折，FrameSamp 的局级 MAE 减 MotionMem 的局级 MAE（正 = MotionMem 误差小）：

| 组 | 第 17 层 | 其他层 |
| --- | --- | --- |
| Text | +16.0 [12.1, 19.7] | 第 8 层 +18.4 [13.5, 22.9]；第 0–17 层 +16 ～ +24，各层区间都不含 0 |
| History frames | +17.9 [10.8, 24.3] | +18 ～ +27 |
| Current frame | +4.6 [1.3, 7.8] | 0 ～ +13 |
| Action token | +3.1 [0.1, 5.9] | +1 ～ +14 |

**闭环复核**：用 `stopcube-rollout-1105` 各模型自己的 50 局闭环轨迹，取 Q4（距正确窗口中心）。样本与标签同 `rollout_probe`，折用其 `hash_fold`，拟合同上（修正版），`leak=0`。两个模型的点集不同，跨模型只能比方向，不能当作配对显著差（审查 F7）。

| 模型 | 只用 `[t,N]`（R²／MAE） | `[t,N,M]` | + Text（9） | + History（8） | + Action（7） |
| --- | --- | --- | --- | --- | --- |
| FrameSamp（561 点、50 局） | 0.604／59.2 | 0.935 | 0.599，−1.7 [−5.9, 2.4] | 0.575，−3.5 [−8.3, 0.9] | 0.590，−1.7 [−4.5, 1.0] |
| MotionMem（640 点、49 局） | 0.570／51.8 | 0.923 | 0.786，+16.5 [10.3, 22.8] | 0.841，+20.3 [12.6, 28.2] | 0.723，+10.7 [4.8, 16.5] |

修正后，FrameSamp 的增量在共同轨迹上是小的正值，在闭环上区间都含 0；MotionMem 修正前后基本不变。

### C. 按目标次数 N 拆开（Text 组，训练折内选层）

| N | 距按下 MAE：MotionMem／FrameSamp／oracle `[t,N]` | 差（步） | 速度 M：MotionMem／FrameSamp | 已过次数：MotionMem／FrameSamp |
| ---: | --- | --- | --- | --- |
| 2 | 41.2／17.4／27.2 | **−23.8 [−30.9, −17.3]** | 0.818／0.826 | 0.965／0.962 |
| 3 | 40.9／46.2／48.1 | +5.3 [−4.6, 15.1] | 0.729／0.518 | 0.962／0.932 |
| 4 | 43.2／58.3／67.1 | +15.1 [6.7, 23.8] | 0.749／0.490 | 0.946／0.864 |
| 5 | 47.0／72.4／74.0 | +25.4 [15.0, 35.8] | 0.895／0.558 | 0.949／0.806 |

N=2 时 FrameSamp 明显更好。N 越大，MotionMem 的优势越大，且误差几乎不随 N 增长。

**N=2 与 N≥4 应并列写（审查 F6）**。按 N 看预测的平均偏差与斜率（预测对真值，同一 N 内）：

- MotionMem 的偏差依次为 +30、+19、−28、−22 步，有部分回归到均值；N=2 时更差，主要来自 +30 的高估。
- N=4、5 时，两个模型的偏差相近（MotionMem −28、−22；FrameSamp −26、−31）。差别在同一 N 内的斜率：MotionMem 0.68、0.63，FrameSamp 0.39、0.36，即 MotionMem 更能跟上同一 N 下由速度 M 造成的差异。
- 全体斜率：MotionMem 0.63，FrameSamp 0.42，oracle `[t,N]` 0.38。

逐层看，只在评估时按 N≥4 分档（探针仍在全部点上训练）：MotionMem Text 第 0–16 层 R² 0.44–0.66，FrameSamp 0.30–0.39；N≤3 时 MotionMem 多数层为负。图：[records/figures/text-by-N.png](records/figures/text-by-N.png)、[records/figures/layers-by-N.png](records/figures/layers-by-N.png)。

**旧闭环数据同样的拆法**（`stopcube-rollout-1105` 的第 17 层折外预测，q4 点集，各自轨迹）：FrameSamp Text MAE 在 N=2..5 依次为 42.8、67.0、71.2、106.1；MotionMem 为 52.8、45.7、46.7、64.1。走势与共同轨迹一致。

### D. 跨 N 外推（中性／负面结果）

探针只在 N≤3 的局上训练、在 N≥4 的局上测（以及反方向），特征与选 λ 同上，脚本 `records/diagnostics/cross_N_generalize.py`，结果 `records/explore/cross_N.json`，`leak=0`。

- **距按下步数**：两个模型都外推不了，所有组、所有层的 R² 都为负（N≤3→N≥4：FrameSamp Text −0.24 ～ −0.93，MotionMem Text −0.24 ～ −0.73；连续 `[t,N]` 参考为 0.22）。原因是标签范围随 N 整体平移，线性探针没见过这个范围。没有哪个模型占优。
- **速度 M**（N≤3→N≥4，准确率）：MotionMem History 组 0.62–0.75，Text 0.55–0.60；FrameSamp 各组 0.36–0.53；连续 `[t,N]` 参考 0.487。MotionMem 的记忆组对速度的编码在不同 N 之间迁移得更好，但幅度有限。

### E. 非线性稳健性：kNN 探针（结果一半有利、一半不利）

做法：训练折内标准化 + PCA64，再用 k 近邻读数（k ∈ {5, 15, 45} 在训练折内按局选），外层 5 折同上，层取 4、8、12、17。脚本 `records/diagnostics/knn_probe.py`，结果 `records/explore/knn.json`。

- **速度 M：优势在非线性探针下依然成立**。Text 组 MotionMem − FrameSamp 在第 4／8／12／17 层为 +0.230 [0.148, 0.311]、+0.173 [0.129, 0.222]、+0.210 [0.152, 0.269]、+0.183 [0.128, 0.241]。History 组为 +0.112、+0.105（第 4、8 层，区间不含 0），第 12、17 层区间含 0。Current frame 组无差别。
- **距按下步数：kNN 下 MotionMem 不占优**。FrameSamp MAE 减 MotionMem MAE：Text 第 4 层 −27.0 [−41.3, −13.3]、第 8 层 −12.7 [−24.9, −1.1]，第 12、17 层区间含 0；Action 第 4 层 −22.6，第 12、17 层约 −5～−6（区间刚好不含 0）；Current 第 8、12 层 MotionMem 略好（+5.5、+7.1）。kNN 在这个标签上整体较弱（R² 多在 0.4 以下，MotionMem Text 接近 0）。可能的原因是欧氏距离被与该标签无关的主成分主导；所以第二节、第三节 A／B 里关于距按下步数的优势依赖线性读法，正文不宜写成与探针形式无关的结论。

### F. 临近决策时刻与逐层图

- **临近决策**：只取距按下 ≤ 阈值的点评估（探针仍在全部点上训练、训练折内选层）。FrameSamp MAE 减 MotionMem MAE：
  - Text：≤48 步时 +8.1 [1.0, 16.0]，≤64 步时 +10.3 [3.5, 17.9]，≤96 步时 +4.9 [−1.6, 11.7]。
  - Current frame：+5.7 [−0.1, 11.6]、+2.0、−0.3。
  - Action token：−6.1 [−12.1, −0.2]、−3.5、−4.6，即 FrameSamp 更好。
  - 同一子集上 oracle `[t,N]` 的 MAE 约 48–50 步。
  - 结果：`records/explore/near_decision.json`。
- **逐层图** [records/figures/layers-speed-pairs.png](records/figures/layers-speed-pairs.png)：
  - 外观相同配对点：从第 1 层起，MotionMem 三组每一层都高于 FrameSamp 三组（Text 0.89–0.96 对 0.54–0.68）。
  - 速度 M：MotionMem Text 第 0–17 层为 0.74–0.81，其余五条曲线都在 0.48–0.67；按每局等权，多数类基线约 0.42。

## 四、解释边界

- B 节的「补上的信息」与 M 一致：`[t,N,M]` 上限 0.926；MotionMem 读速度 M 的准确率为 0.89–0.93（motion 组、history 组），FrameSamp 为 0.57–0.71。这是相关证据，没有做干预。
- C 节按 N 拆分是事后拆分。N 与局长、取点数相关（N=2 的局短、点少），每局等权后短局权重相对更大。
- 选层后的读数，以及 B、C 两节，都没有预先登记；正文引用时须标明是探索性结果。
- **优势来自 motion 输入，未必是「长期记忆」（审查 F5）**：MotionMem 额外看到近期 motion 窗口（相邻帧运动），可以直接得到方块速度；FrameSamp 只有稀疏采样的历史帧。所以「能读出 M」更可能是 motion 通道的作用，没有做 motion 通道消融。正文宜写成「motion 表征提供了任务规格 [t,N] 之外的动态信息」，不宜写成「记忆更强」。
- 探索审查判定行：`EXPLORE_AUDIT=PASS base=5c0f588 findings=7 blocking=0`。F2、F3（内层 PCA 用到内层验证行、λ 与层的二重选择）属轻微乐观，两模型对称；F4 确认跨模型配对差同点、同折，按局重采样实现正确。

## 五、实施中调整

1. **K9 失败处理**：开工审查 Finding 1 指出，推荐项「冒烟 FAIL 停下等用户」与用户原话「不要停止 不要来问我」冲突，按后者改为「修复后重新冒烟，最多 2 轮」。实际两份冒烟都一次通过，没有用到这条。
2. **ctx 显存配额 0.90 → 0.80**：起跑时其他用户在两张卡上各占约 4–5 GB。属同规模参数微调，没有派审。
3. **稿件侧改为新旧并存**：新数据写 `data/stopcube_belief_common/`，新图出 `belief_probe_common.pdf`；原表、原图与正文不动，等用户确认措辞后再切换。`ADJUST_REVIEW=PASS scale=same findings=3`。

## 六、本轮 tmux 会话

`cc-cf-explore1`、`cc-cf-smoke-ctx`、`cc-cf-smoke-mot`、`cc-cf-repro-1105`、`cc-cf-ctx`、`cc-cf-mot`、`cc-cf-beyond`、`cc-cf-fig4`、`cc-cf-explore-new`、`cc-cf-beyond-new`、`cc-cf-closedloop`、`cc-cf-beyond-new2`、`cc-cf-scale100`，全部自然退出。

## 七、归档

- `records/fig4-tables/`：r2_by_layer.csv、fit_lines.csv、reference_lines.csv、summary.json
- `records/explore/`：angles.json、breakdown.json、layers_by_N.json、beyond_oracle.json，以及候选图数据（beyond_oracle_by_layer.csv、by_N.csv、oracle_variants.csv、paired_diff.csv）
- `records/figures/`：上文引用的四张图
- `records/diagnostics/`：探索脚本
- 清洗后的日志在 `records/logs/`，被 `*.log` 规则忽略，只留在本机

隐状态库 `v1-store/evaluation/stopcube-common-fig4-1108/replay/{ctx,mot}/` 不进 git。
