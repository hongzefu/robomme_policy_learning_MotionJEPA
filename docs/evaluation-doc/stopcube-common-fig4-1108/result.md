# StopCube 共同轨迹 Fig. 4 重跑：结果

**一句话结论**：两个模型都在同一批官方示教上回放（98 局，2344 个点，两边取点相同）。按 Fig. 4 原口径逐层读「距按下步数」时，旧图里 MotionMem 的 Text 优势（0.77 对 0.51）基本消失：第 17 层 MotionMem 反而略低（Text 0.382 对 0.489）。训练折内选层时，MotionMem Text 较高（0.652，多选第 0 层；FrameSamp 0.507）。

探索性分析里的 MotionMem 优势，主要来自 motion 输入带来的速度信息（未做 motion 通道消融，不写成「记忆更强」）：
1. **速度 M**：MotionMem 的 Text、History、Motion 组读速度明显更好（Text +0.207 [0.156, 0.261]）。换成 kNN 读法依然成立；在各自闭环轨迹上，FrameSamp 读不出速度（不高于多数类）。
2. **`[t,N]` 之外的信息**：在 `[t,N]` 上加入表征后，MotionMem 的 Text、History、Motion 组在第 0–17 层都能补上 22–30 步误差，FrameSamp 各组只补 0–10 步（修正口径；同点配对差，第 17 层 Text +16.0 [12.1, 19.7] 步）。MotionMem 的 Current／Action 组在中间层最多也补 17 步；虽然它们直接读速度并不比 FrameSamp 好，但在 `[t,N,M]` 之上几乎补不上（N 节），所以补上的也主要是速度信息。
3. **外观相同的配对点**：画面几乎相同、但已过次数不同的配对点，MotionMem 读对的比例高 0.2–0.3。
4. **按目标次数 N**：N≥4 时 MotionMem Text 读距按下更好（N=5 少 25.4 步）；但 N=2 时 FrameSamp 明显更好（MotionMem 多 23.8 步，比 oracle 还差），一部分来自回归到均值。

5. **机制与稳健性**：
   - 在 `[t,N,M]` 之上再加表征，两个模型都几乎补不上（N 节），所以多出来的几乎全部是速度 M。
   - 这条结论在 PCA 16–128 维（K 节）和 5 组分折种子（L 节）下都成立。
   - 整局打乱 (M, N) 后增量消失（M 节）。
6. **三个模型的对应**：FrameSamp（14%）< concat（43%）< split（66%），若干读数随 StopCube 成功率同序上升（J 节）。只有三个模型，属于相关证据。

**不利或中性的结果**：
- 距按下步数的优势依赖线性读法，kNN 下不成立（E 节）。
- 跨 N 外推两个模型都失败（D 节）。
- Text 读剩余次数低 0.078（A 节）。
- 临近决策时 Action 组 FrameSamp 更好（F 节）。

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

**与旧 Fig. 4 对照**：旧图用的是各自闭环的 50 局。MotionMem Text 19 个读数里有 17 个高于它自己的 oracle 0.551（峰值 0.770）；FrameSamp 全部低于 0.616。换成共同轨迹后，两条 Text 曲线在第 8–16 层接近：FrameSamp 0.49–0.52，MotionMem 0.53–0.55，都比 oracle 0.447 高 0.04–0.10。第 17 层 MotionMem 降到 0.382，低于 oracle 和 FrameSamp（0.489）。旧图的差距主要来自两个策略访问的状态不同。图：[records/figures/common-vs-oldfig4-1105.png](records/figures/common-vs-oldfig4-1105.png)。

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

修正后，FrameSamp 的增量在共同轨迹上是小的正值（Text、Current 区间不含 0，History 含 0）。在闭环上，表中列出的各层区间都含 0；但不是每一层都如此，例如 Current 输入层为 −7.9 [−14.7, −1.4]，History 第 0–2 层也为负且区间不含 0。MotionMem 修正前后基本不变。

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

- **速度 M：优势在非线性探针下依然成立**。Text 组 MotionMem − FrameSamp 在第 4／8／12／17 层为 +0.230 [0.148, 0.311]、+0.173 [0.129, 0.222]、+0.210 [0.152, 0.269]、+0.183 [0.128, 0.241]。History 组为 +0.112、+0.105（第 4、8 层，区间不含 0），第 12、17 层区间含 0。Current frame 组读速度无差别。
- **距按下步数：kNN 下 MotionMem 不占优**。FrameSamp MAE 减 MotionMem MAE：Text 第 4 层 −27.0 [−41.3, −13.3]、第 8 层 −12.7 [−24.9, −1.1]，第 12、17 层区间含 0；Action 第 4 层 −22.6，第 12、17 层约 −5～−6（区间刚好不含 0）；Current 第 4 层 FrameSamp 明显更好（−28.9 [−41.2, −17.1]），第 8、12 层 MotionMem 略好（+5.5、+7.1）。kNN 在这个标签上整体较弱（R² 多在 0.4 以下，MotionMem Text 接近 0）。可能的原因是欧氏距离被与该标签无关的主成分主导；所以第二节、第三节 A／B 里关于距按下步数的优势依赖线性读法，正文不宜写成与探针形式无关的结论。

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

### G. 闭环轨迹上读速度 M

用各模型自己的闭环轨迹（Q4 样本集，样本与折同 B 节闭环复核），逐层读速度 M：一对多岭回归，训练折内 PCA64，每局等权。脚本 `records/diagnostics/closedloop_speed.py`，结果 `records/explore/closedloop_speed.json`，`leak=0`。

- 按每局等权的多数类基线：FrameSamp 0.44，MotionMem 0.45。
- FrameSamp：Text、Current frame、History、Action 各层都在 0.30–0.43，不高于多数类，即在它自己的轨迹上读不出速度。
- MotionMem：
  - Text 0.64–0.77（第 0 层 0.77）。
  - History 0.72–0.80。
  - Motion 输入层 0.98，各层 0.80–0.83。
  - Current frame 0.39–0.54，Action 0.46–0.57。

速度信息在 motion token 的输入层就几乎完整（0.98），到第 0 层已经出现在 Text 和 History 里。这和 F5「优势来自 motion 输入」一致，也和 5.3 注意力分析里 language 在第 0–1 层读 motion 的通路一致。

### H. 行为对照：闭环成功率按 N、M 拆分（逐局配对）

数据来自 `stopcube-rollout-1105/analysis/align.json`。两个模型的 50 局局号与环境种子相同，M、N 逐局一致，可以逐局配对。检验用精确 McNemar（双侧，只看不一致的局）。

| 子集 | 局数 | MotionMem 成功 | FrameSamp 成功 | 只有 MotionMem 成功／只有 FrameSamp 成功 | p |
| --- | ---: | --- | --- | --- | ---: |
| 全部 | 50 | 22 | 7 | 17／2 | 0.0007 |
| N=2 | 15 | 3 | 2 | — | — |
| N=3 | 15 | 8 | 3 | — | — |
| N=4 | 9 | 4 | 2 | — | — |
| N=5 | 11 | 7 | 0 | 7／0 | 0.016 |
| N≥4 | 20 | 11 | 2 | 10／1 | 0.012 |
| N≤3 | 30 | 11 | 5 | 7／1 | 0.070 |
| M=60 | 12 | 2 | 2 | 1／1 | 1.0 |
| M=80 | 22 | 5 | 1 | 5／1 | 0.22 |
| M=120（最慢） | 16 | 15 | 4 | 11／0 | 0.001 |

行为上的优势集中在高记忆需求（N≥4）和最慢的速度（M=120）这两类局，方向与 C 节探针按 N 的结果一致（FrameSamp 的误差随 N 变差，N=5 时成功 0/11）。这些分组是事后拆分，只作相关描述。M=60、80 两档两个模型都很少成功，不足以区分。

### I. 正式 3 seed 评测：StopCube 成功率按 N、M 拆分（两个 motion 变体）

数据：GreatLakes 正式评测 `official-stage2-1026/` 的逐局记录（只读）。M、N 按 `episode_key` 取自 `stopcube-oracle-1104` 的 oracle 表，并核对 `env_seed` 一致。脚本 `records/diagnostics/official_success_by_N.py`。

- **对比对象**：`o16cc`（concat 1×1024，即论文的 MotionMem）对 `o16ctx`（context split 4×256，motion 预算 384，也是带 motion 的模型）。
  - 起初误把 `o16ctx` 当成同数据训练的 FrameSamp；按 `o16ctx-60k-800/launch.md` 第 5 行核对后纠正：它是 split 读出的 motion 变体。
  - 官方 FrameSamp Context 只有论文 Table 3 的汇总（StopCube 13.67 ± 8.82），本机没有逐局记录，无法按 N 拆。

| 子集 | 局数（3 seed） | concat（MotionMem） | split | 不一致局 concat 胜／split 胜 | p |
| --- | ---: | --- | --- | --- | ---: |
| 全部 | 150 | 65（0.43） | 99（0.66） | 8／42 | 1e-6 |
| N=2 | 45 | 0.16 | 0.38 | 3／13 | 0.021 |
| N=3 | 45 | 0.49 | 0.67 | 2／10 | 0.039 |
| N=4 | 27 | 0.44 | 0.85 | 1／12 | 0.003 |
| N=5 | 33 | 0.73 | 0.88 | 2／7 | 0.18 |
| M=60 | 36 | 0.19 | 0.67 | 2／19 | 0.0002 |
| M=80 | 66 | 0.30 | 0.50 | 5／18 | 0.011 |
| M=120 | 48 | 0.79 | 0.88 | 1／5 | 0.22 |

- 在 StopCube 上，split 变体全面好于 concat。这与 `o16-3seed-1031/result.md` 末尾提到的「concat 在 RouteStick／StopCube 上的显著退化」一致，属于 MotionMem 内部两种读法的差别，不涉及与 FrameSamp 的比较。
- split 变体 N 越大成功率越高（0.38 → 0.67 → 0.85 → 0.88）；concat 大体上升（0.16 → 0.49 → 0.44 → 0.73），N=4 略低于 N=3，不严格单调。最慢的 M=120 最容易成功。H 节闭环 50 局里 FrameSamp 在 N=5 时 0/11，走势相反；但官方 FrameSamp 没有逐局数据，这个对比只能停在这里。
- 若稿件用的是 concat，StopCube 的绝对成功率（43%）低于 split（66%）；正文如果引用 StopCube 成功率，需注明是哪个变体。

### J. 三个模型的表征强弱与 StopCube 成功率的对应（加跑 split 变体）

加跑的模型是 MotionMem 的 split 变体 `orig100-ctx32x4x4-motion-s2split4x256-b128-60k`（59999，motion 预算 384）。代码不变，同一批 2344 个点，`--action-samples 5`。

- **对抗验证**：`ADJUST_REVIEW=PASS scale=same findings=3`。
- **产物**：根目录 `replay-split/mot/`。label 复用为 mot，因为 `hover_replay --label` 只接受 ctx／mot；它实际是 split 变体，不要和 `replay/mot`（concat）混淆。
- **冒烟判定行**：`REPLAY_PARITY=PASS points=21 maxabs=0`（没有 split 的闭环记录可比，`compared=0`，只证明批 8／批 16 两条路径一致），`ACTION_PARITY=PASS maxabs=0`，各组 `HIDDEN_PRECISION=PASS`。
- **全量判定行**：两卡分局并行，GPU0 `REPLAY_RUN=PASS points=1180`（局 0–46），GPU1 `REPLAY_RUN=PASS points=1164`（局 47–99），合计 98 局、2344 个点。GPU1 跑完后又接手重算局 41–46（`points=212`）。GPU0 也自己算完了这几局，所以两边重复；同名文件由后写者覆盖（文件时间 05:33–05:44），不计入合计。事后单独重算第 46 局（`verify-split-ep46.sh`，tmux `cc-cf-verify46`），与库内文件全部键逐位相等（`records/split-ep46-rerun-compare.txt`），确定性成立。各会话 `EXIT_CODE=0`。
- **比较方式**：v1-store 内建两个符号链接根 `cmp-split-vs-fs`、`cmp-split-vs-concat`，用 `explore_angles.py` 与 `beyond_oracle.py`（`ONEHOT_SCALE=100`）不改代码地比较。结果在 `records/explore/split/`。

**三个模型的 StopCube 成功率**：FrameSamp 13.67%（论文 Table 3）< concat 43%（65/150）< split 66%（99/150）。训练折内选层后，下面这些挑选出来的读数按同样的顺序递增（不是所有读数都如此，表后列出部分反例）：

| 读数 | FrameSamp | concat | split |
| --- | ---: | ---: | ---: |
| Text 读距按下步数 R² | 0.507 | 0.652 | 0.771 |
| Text 读速度 M | 0.599 | 0.807 | 0.845 |
| History 读速度 M | 0.710 | 0.891 | 0.954 |
| History 读外观相同配对 | 0.467 | 0.648 | 0.916 |
| Action 读外观相同配对 | 0.560 | 0.769 | 0.825 |
| Action 读「该不该按」 | 0.866 | 0.892 | 0.944 |
| motion 组读速度 M | — | 0.927 | 0.981 |

不按这个顺序的例子：Current frame 读速度 M（0.574／0.660／0.592）、Action 读速度 M（0.623／0.639／0.609）、Current 读距按下（0.551／0.579／0.563）、Action 读距按下（0.495／0.475／0.536）、Text 读剩余次数（0.748／0.671／0.779）、Text 读外观相同配对（0.647／0.951／0.943）、Current 读外观相同配对（0.683／0.915／0.877）。表内个别相邻差不显著，例如 Text 读速度 M 的 split − concat 为 +0.038 [−0.012, 0.086]。

**同点配对差**：

- split − FrameSamp：Text 读距按下少 15.6 步 [9.4, 21.8]；速度 M +0.245 [0.178, 0.314]；History 外观相同配对 +0.449 [0.342, 0.552]。
- split − concat：
  - Text 读距按下少 9.0 步 [4.6, 13.3]；Action 读距按下少 4.5 步 [1.2, 8.0]。
  - Action 读该不该按 +0.052 [0.032, 0.072]；History 外观相同配对 +0.268 [0.192, 0.339]。
  - Current frame 读速度 M 为 −0.068 [−0.117, −0.021]，split 较差。
- oracle 增量（修正口径）：
  - split 对 FrameSamp：Text 第 0–16 层 +17 ～ +25 步，第 17 层 +13 [9, 17]；History +16 ～ +27 步。与 concat 对 FrameSamp 相近。
  - split 对 concat：
    - History 各层区间都含 0。
    - Text 各层差 −3 ～ +2 步，其中第 17 层 concat 显著更好（−2.9 [−5.2, −0.6]）。
    - Current 第 1–16 层 concat 好 4–8 步，区间不含 0。
    - Action 方向偏向 concat，但 18 层里 10 层区间含 0。
  - 也就是说，「补上 `[t,N]` 之外的速度信息」是有 motion 与没有 motion 的差别；在两个 motion 变体之间，它不随成功率变化；Current 组上甚至是成功率较低的 concat 更强，方向与成功率相反。

**固定层、换分折种子**：Text 读距按下 R² 在固定层、5 个分折种子下的范围（`records/diagnostics/three_model_fold_seed.py`、`records/explore/three_model_fold_seed.json`）：

| 层 | FrameSamp | concat | split |
| --- | --- | --- | --- |
| 0 | 0.397–0.455 | 0.652–0.673 | 0.771–0.788 |
| 8 | 0.469–0.509 | 0.518–0.554 | 0.662–0.689 |
| 17 | 0.462–0.489 | 0.382–0.427 | 0.539–0.578 |

第 0、8 层三者顺序在所有种子下都与成功率同序，取值范围互不重叠；第 17 层 concat 最低，顺序不成立。split 在三层都最高。

**split 对 FrameSamp 的 kNN 检查**（`knn_probe.py` 不改，指向 `cmp-split-vs-fs`；结果 `records/explore/split/split-vs-fs-knn.json`）：

- **速度 M**：Text +0.121 ～ +0.310，History +0.197 ～ +0.286，第 4／8／12／17 层区间都不含 0。
- **距按下步数**：Text 第 8／12／17 层 split 更好（+11.2 [3.1, 19.0]、+22.6 [16.0, 29.2]、+8.8 [1.4, 16.3] 步），第 4 层区间含 0；Action 第 8、12 层小幅更好（+6.8、+5.6），第 17 层 +4.7 的区间下界贴近 0。History 四层区间都含 0。所以 split 在 kNN 下只在 Text 第 8／12／17 层显著占优；concat 在 kNN 下这一项不占优（E 节）。
- **不利项**：Current frame 组 FrameSamp 更好（距按下 −6.9 ～ −31.5 步，速度第 4、8 层 −0.14～−0.15）；Action 第 4 层距按下也是 FrameSamp 好（−21.5）。

**split 按 N 拆分**（`explore_breakdown.py` 不改，结果 `records/explore/split/split-vs-fs-breakdown.json`）：

- Text 读距按下的 MAE 在 N=2..5 依次为 31.8、33.3、31.3、39.6 步，N=2–4 持平、N=5 略升，而且每一档都低于 concat（41.2、40.9、43.2、47.0）。
- 与 FrameSamp 相比：N=2 时仍是 FrameSamp 更好（split 多 14.4 [3.9, 25.8] 步，比 concat 的 23.8 小）；N=3 +12.9 [−0.2, 25.4]；N=4 +27.1 [18.0, 36.1]；N=5 +32.8 [24.3, 40.9]。
- Text 读速度 M：N=2 时 +0.070（区间含 0），N=3–5 时 +0.244 ～ +0.332（区间不含 0）。

**按记忆需求分档的逐层 Text 读距按下**（探针在全部点上训练、评估时分档；结果 `records/explore/split/split-vs-fs-layers_by_N.json`）：

| 分档 | FrameSamp | concat | split |
| --- | --- | --- | --- |
| N≥4，第 0–17 层平均（范围） | 0.35（0.30–0.39） | 0.52（0.37–0.66） | 0.64（0.50–0.80） |
| N≤3，第 0–17 层平均（范围） | 0.27（0.12–0.38） | −0.59（−1.24 ～ −0.01） | 0.17（−0.71 ～ 0.33） |

高记忆需求的局里，三者与成功率同序，而且 18 层逐层都是 FrameSamp < concat < split。低记忆需求的局里，平均上 FrameSamp 最好、split 与它相当（18 层里 FrameSamp 有 13 层最高，split 有 5 层高于 FrameSamp），concat 每一层都最低。图：[records/figures/three-models-by-N.png](records/figures/three-models-by-N.png)（`records/diagnostics/plot_three_models_by_N.py`）。

**解释边界**：只有三个模型，是相关证据，不是因果。FrameSamp 的成功率来自论文、不是本轮评测，评测平台与 seed 也不同。split 和 concat 的差别同时涉及 motion 预算（384 对 96）与读出方式，分不开。

### K. PCA 维度稳健性

把 PCA 维度换成 16、32、64、128，在 Text、History 组的第 0、8、17 层重算两个核心结论（修正口径，同点配对）。脚本 `records/diagnostics/pca_dim_robustness.py`，结果 `records/explore/pca_dim.json`。

- **`[t,N]` + 表征读距按下**（FrameSamp MAE − MotionMem MAE）：24 格全部为正，+11.0 ～ +26.8 步，区间都不含 0。
- **读速度 M**（MotionMem − FrameSamp）：24 格里 23 格区间不含 0。唯一的例外是 Text 第 17 层 16 维，+0.046 [−0.001, 0.092]。
- 维度越高，优势一般越大（例如 Text 第 8 层读速度：16 维 +0.065，128 维 +0.257），所以 64 维不是对 MotionMem 最有利的选择。

### L. 分折种子稳健性

外层 5 折与内层 4 折的按局哈希种子，除原值 42 外再取 1、7、123、2026，在 Text、History 组第 0、8、17 层重算（PCA 64，修正口径）。脚本 `records/diagnostics/fold_seed_robustness.py`，结果 `records/explore/fold_seed.json`。

- **`[t,N]` + 表征读距按下的跨模型差**：30 格全部为正，+16.0 ～ +30.6 步，区间都不含 0。例如 Text 第 17 层在 5 个种子下为 +16.0 ～ +17.3。
- **读速度 M 的跨模型差**：30 格全部为正，+0.088 ～ +0.374，区间都不含 0。
- 结论不依赖某一次特定的按局分折。

### M. 阴性对照：整局打乱 (M, N)

做法：每局的 M、N 换成另一局的（t 不变），按 `hover_points.point_labels` 重算距按下步数，`[t,N]` 特征也换成打乱后的 N。比较「`[t,N_shuf]` + 组特征」与「只用 `[t,N_shuf]`」，打乱 5 次。脚本 `records/diagnostics/shuffle_control.py`，结果 `records/explore/shuffle_control.json`。

- 两个模型的 Text、History 组，在第 8、17 层的增量均值都在 −2.3 ～ 0.0 步之间，单次范围 −6.4 ～ +1.8 步。打乱时没有排除「换成自己」的局，个别局可能没被打乱，这只会让打乱后的增量略偏高，不影响结论。
- 真实标签下，MotionMem 的同项增量是 22–30 步（B 节）。所以 B 节的增量来自表征里与真实 (M, N) 相关的信息，不是多加维度造成的过拟合。

### N. 补上的信息是不是只有速度 M：在 `[t,N,M]` 之上再加表征

做法：`[t,N,M]` 独热（×100）单独时 R² 0.926、MAE 20.2 步，在此之上再加组特征（训练折内 PCA64）。两个模型，Text、History、Current、Action 组，第 0、8、17 层。脚本 `records/diagnostics/beyond_tNM.py`，结果 `records/explore/beyond_tNM.json`。

- 两个模型的增量都只有 −1.9 ～ +3.1 步，R² 在 0.915–0.948 之间。这些增量并非都含 0：FrameSamp Text 第 8 层 +2.7、Action 第 8／17 层 +2.5／+2.7，MotionMem Text 与 Current 第 8／17 层也是小幅显著；Action 第 0 层两个模型都显著为负。量级都在 3 步以内。
- 跨模型差多数区间含 0。少数几格不含 0，方向也不一致：Current 第 8、17 层 MotionMem 好 2 步左右，Action 第 8、17 层 FrameSamp 好 1.5–1.7 步。
- 结论：B 节里 MotionMem 比 FrameSamp 多补上的 16–30 步，几乎全部就是速度 M；没有发现 M 以外、显著的额外进度信息。这让机制解释更窄也更干净——MotionMem 的优势是能读出方块速度。

### O. 速度信息随时间的变化

做法：用训练折内选层的折外预测，读速度 M 的准确率按时间步 t 分档（每档内每局等权），跨模型配对差按局重采样。脚本 `records/diagnostics/speed_by_time.py`，结果 `records/explore/speed_by_time.json`。

| t 档 | 点数／局数 | Text：MotionMem／FrameSamp（差） | History：MotionMem／FrameSamp（差） |
| --- | --- | --- | --- |
| [0, 80) | 265／98 | 0.55／0.50（+0.044 [−0.026, 0.111]） | 0.82／0.65（+0.175） |
| [80, 120) | 417／89 | 0.75／0.48（+0.274） | 0.91／0.75（+0.158） |
| [120, 160) | 358／82 | 0.85／0.51（+0.335） | 0.92／0.73（+0.194） |
| [160, 200) | 320／71 | 0.86／0.52（+0.335） | 0.89／0.70（+0.193） |
| [200, ∞) | 984／58 | 0.95／0.60（+0.346） | 0.93／0.69（+0.243） |

除 Text 第一档外，区间都不含 0。

- MotionMem 的 Text 读速度随时间上升（0.55 → 0.95），FrameSamp 基本持平；MotionMem 的 History 从最早一档起就在 0.82 以上。看上去像是速度信息先出现在记忆组，再随着看到更多 motion 窗口逐步进入 Text。
- **混杂**：t 越大的档只来自 N 大或速度慢的局（局数从 98 降到 58），各档的局组成不同，不能直接读成同一局内随时间积累。
- **同局对照**（`records/diagnostics/speed_within_episode.py`）：只保留早档 t∈[56, 120) 与晚档 t≥200 都有点的 58 局。
  - MotionMem Text 从 0.69 升到 0.95，晚减早 +0.262 [0.186, 0.337]。
  - FrameSamp Text 从 0.45 升到 0.60，+0.149 [0.032, 0.279]。
  - History 组两边基本不变（MotionMem +0.040，FrameSamp −0.038）。
  - 同一局内，Text 读速度随时间上升在两个模型里都存在，MotionMem 升得更多，同局晚档也高得多（0.95 对 0.60）。原始输出存于 `records/explore/speed_within_episode.txt`。

### P. 闭环里「读对速度」与成功（弱，主要被 M 混杂）

做法：用各模型自己的闭环轨迹，统计每局读对速度 M 的比例（一对多岭，训练折内 PCA64），再与该局成功与否对照。脚本 `records/diagnostics/closedloop_speed_vs_success.py`，原始输出 `records/explore/closedloop_speed_vs_success.txt`。

- **MotionMem 整体**：
  - Text 第 0 层，成功局读对 0.85、失败局 0.70，差 +0.152 [0.008, 0.301]；读对比例 ≥0.5 的局成功 21/42，<0.5 的局成功 1/7。
  - History 第 8 层差 +0.198 [0.005, 0.383]。
  - motion 输入层在成功和失败局里都接近 1（0.99 对 0.96）。
- **同一速度内部**：M=60 时成功局更准（+0.49，但只有 3 局成功对 8 局失败）；M=80 时约 0（5 对 17）；M=120 时方向不一（14 对 2）。整体关联主要来自 M 本身（M=120 局多数成功，也更容易读对），不能当作「读对速度导致成功」的证据。
- FrameSamp 读速度本来就不高于多数类，它的负相关没有解释意义。

### Q. 样本效率（学习曲线）

做法：外层训练局按 `hover_probe.curve_filter` 取 25%、50%、100%（测试折不变），在 Text、History 组第 0、8、17 层重算读速度 M 和 `[t,N]` 之上的增量（修正口径）。脚本 `records/diagnostics/learning_curve.py`，结果 `records/explore/learning_curve.json`。

- **速度 M（MotionMem − FrameSamp）**：
  - 25% 训练局（约 20 局）时，Text 组已有 +0.139～+0.188，History 第 0、8 层 +0.214～+0.320，区间都不含 0；只有 History 第 17 层在 25% 时区间含 0（+0.038）。
  - 50% 和 100% 时，各格区间都不含 0。
- **`[t,N]` 之上的增量**：
  - MotionMem 随训练局增多快速上升：25% 时 +6.6～+10.2 步（History 第 17 层仅 +0.9），50% 时 +18.8～+24.5 步，100% 时 +22.1～+29.2 步。
  - FrameSamp 一直只有 −1.7～+7.8 步。
  - 跨模型差在 25% 时 +7.1～+9.1（History 第 17 层 +2.6，区间含 0），100% 时 +16.0～+26.7。
- MotionMem 的速度信息用很少的训练局就能线性读出；而要把它和 `[t,N]` 组合起来读距按下，需要更多的局。

## 四、解释边界

- B 节的「补上的信息」与 M 一致：MotionMem 的 Current／Action 组直接读 M 不比 FrameSamp 好，但在 `[t,N,M]` 之上同样几乎补不上（N 节），所以各组补上的都主要是速度信息。`[t,N,M]` 上限 0.926；MotionMem 读速度 M 的准确率为 0.89–0.93（motion 组、history 组），FrameSamp 为 0.57–0.71。这是相关证据，没有做干预。
- C 节按 N 拆分是事后拆分。N 与局长、取点数相关（N=2 的局短、点少），每局等权后短局权重相对更大。
- 选层后的读数，以及 B、C 两节，都没有预先登记；正文引用时须标明是探索性结果。
- **优势来自 motion 输入，未必是「长期记忆」（审查 F5）**：MotionMem 额外看到近期 motion 窗口（相邻帧运动），可以直接得到方块速度；FrameSamp 只有稀疏采样的历史帧。所以「能读出 M」更可能是 motion 通道的作用，没有做 motion 通道消融。正文宜写成「motion 表征提供了任务规格 [t,N] 之外的动态信息」，不宜写成「记忆更强」。
- 探索审查判定行：`EXPLORE_AUDIT=PASS base=5c0f588 findings=7 blocking=0`。F2、F3（内层 PCA 用到内层验证行、λ 与层的二重选择）属轻微乐观，两模型对称；F4 确认跨模型配对差同点、同折，按局重采样实现正确。

## 五、实施中调整

1. **K9 失败处理**：开工审查 Finding 1 指出，推荐项「冒烟 FAIL 停下等用户」与用户原话「不要停止 不要来问我」冲突，按后者改为「修复后重新冒烟，最多 2 轮」。实际两份冒烟都一次通过，没有用到这条。
2. **ctx 显存配额 0.90 → 0.80**：起跑时其他用户在两张卡上各占约 4–5 GB。属同规模参数微调，没有派审。
3. **稿件侧改为新旧并存**：新数据写 `data/stopcube_belief_common/`，新图出 `belief_probe_common.pdf`；原表、原图与正文不动，等用户确认措辞后再切换。`ADJUST_REVIEW=PASS scale=same findings=3`。

## 六、本轮 tmux 会话

`cc-cf-explore1`、`cc-cf-smoke-ctx`、`cc-cf-smoke-mot`、`cc-cf-repro-1105`、`cc-cf-ctx`、`cc-cf-mot`、`cc-cf-beyond`、`cc-cf-fig4`、`cc-cf-explore-new`、`cc-cf-beyond-new`、`cc-cf-closedloop`、`cc-cf-beyond-new2`、`cc-cf-scale100`、`cc-cf-crossN`、`cc-cf-knn`、`cc-cf-clspeed`、`cc-cf-smoke-split`、`cc-cf-split-g0`、`cc-cf-split-g1`、`cc-cf-split-g1b`、`cc-cf-split-ana`、`cc-cf-pcadim`、`cc-cf-foldseed`、`cc-cf-shuffle`、`cc-cf-tnm`、`cc-cf-verify46`、`cc-cf-3fold`、`cc-cf-curve`、`cc-cf-knn-split`，全部自然退出。

## 七、归档

- `records/fig4-tables/`：r2_by_layer.csv、fit_lines.csv、reference_lines.csv、summary.json
- `records/explore/`：
  - 共同轨迹：angles.json、breakdown.json、layers_by_N.json、near_decision.json、knn.json、cross_N.json、pca_dim.json、fold_seed.json、shuffle_control.json、beyond_tNM.json、speed_by_time.json、speed_within_episode.txt、closedloop_speed_vs_success.txt、three_model_fold_seed.json、learning_curve.json
  - oracle 增量：beyond_oracle.json（旧口径）、beyond_oracle_scale100.json（修正口径）
  - 闭环：closedloop_beyond_oracle.json、closedloop_beyond_oracle_scale100.json、closedloop_speed.json
  - split 加跑：split/
  - 候选图数据：beyond_oracle_by_layer.csv、beyond_oracle_by_layer_scale100.csv、by_N.csv、oracle_variants.csv、paired_diff.csv、three_model_dose.csv
- `records/prefix-match-full.txt`：全量前缀与 1105 库逐局比对的判定行原文
- `records/figures/`：上文引用的六张图
- `records/diagnostics/`：探索脚本
- 清洗后的日志在 `records/logs/`，被 `*.log` 规则忽略，只留在本机

隐状态库 `v1-store/evaluation/stopcube-common-fig4-1108/replay/{ctx,mot}/` 不进 git。
