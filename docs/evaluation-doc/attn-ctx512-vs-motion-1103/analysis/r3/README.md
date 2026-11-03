# R3：attention sink 全面排查与稳健重算

脚本：[`scripts/interpret/analysis/r3_sink.py`](../../../../../scripts/interpret/analysis/r3_sink.py)；本文数字都来自同目录的 `r3_summary.json`（严格 JSON）。
判定行：`R3_SINK=PASS episodes=3 sinks=7 figs=5`；稳健重算行和自检 `R3_ROWSUM robust_max_dev=4.44e-16 tol=1e-06 PASS`。

数据（只读，stage1b 重跑，与上一轮逐位一致）：

| 标签 | 模型 | 局目录 | 结果 | 交换区间 |
|---|---|---|---|---|
| `ctx_ep8` | ctx512 基线 | `0621439b…` | 成功 | SWAP1 [64,114)、SWAP2 [114,164) |
| `ctx_ep4` | ctx512 基线 | `3a79ca2d…` | 失败 | 只有 SWAP1 [64,114) |
| `mot_ep8` | ctx512+motion | `afb74383…` | 成功 | SWAP1、SWAP2 |

每局 20 次推理，每次读两份文件：`infer_00xx.npz` 里的 `probs_action`（10 去噪步 × 18 层 × 8 头 × 20 动作 query × (P+20) key）和 `probs_text`；`prefix_00xx.npz` 里的 `probs_prefix`（18 × 8 × P × P，prefill 时 prefix 内部的完整注意力）。VIS=[0,32)；motion 窗 [t,t+33) 按与区间重叠的帧数除以 33 计权（同 `x3_intervals.shares`）。

## 一句话结论

三局 60 次推理里，跨层、跨 query 族、跨推理都成立的 sink 只有一个：**文本第 0 个 token `<bos>`**。它在 5 个看得见它的 query 族（当前 base、当前 wrist、文本、「green」词、动作）的 60/60 次推理里、几乎全部 18 层都超过均匀的 20 倍；动作 query 给它的质量按局 20 次推理平均为 ctx_ep8 23.6%、ctx_ep4 23.4%、mot_ep8 26.0%（#18 分别为 28.0%、22.0%、26.1%），文本 query 为 40.8%、40.6%、52.4%（`episodes.*.per_infer_sinks[i].bos_mass`）。`<bos>` 不在记忆区里，所以把它剔掉、每行重新归一以后，上一轮各项记忆区指标在 #15～#19 平均和整局平均上的变化都 ≤ 1.0 个百分点（逐次推理最大 3.1 个百分点，即 #16 motion SWAP2 44.1% → 41.0%），各结论都成立：基线裸露窗 VIS 低于均匀，motion 模型 VIS 高于均匀（#15～#19 原口径 17.2%，去 sink 后 16.8%，均匀 13.1%），motion 流对 SWAP1 低于均匀，#16 对 SWAP2 偏高。记忆区内部还有第二类 sink：motion 模型里**记忆区 query 自己**在 #15～#19 把 45.1%～54.4% 的注意力（`episodes.mot_ep8.per_infer_sinks[i].q_mem.sink_mass_layer_mean`）压在几个 motion 窗上，但动作 query 不这样看它们（动作 query 对任何记忆 key 都没到 20 倍）。如果把这些窗也从动作 query 里剔掉（RC 压力测试），帧流结论不变，motion 流 SWAP2 的偏好会翻转（#16 从 44.1% 降到 8.6%，均匀 15.2%）。上一轮 X4 说的「green 词 query 帧流里单 token 占 73%～99%」在 motion 模型里只出现在第 0 层，其余 17 层都在 5.3 倍以下；基线主要集中在第 1 层，≥20 倍的层数为 1～5，也不到 6 层（`x4_green_check`）。所以都不是跨层的 sink。

## 一、注意力可见性（先弄清每类 query 能看到哪些 key）

`block_structure_infer18`（#18，18 层 8 头平均，各行是 query 组，各列是 key 区域质量）：

| query 组 | → 记忆区 | → 当前 base | → 当前 wrist | → 文本 | → 动作 |
|---|---|---|---|---|---|
| 记忆区 query（三局） | 1.000 | 0 | 0 | 0 | — |
| 当前 base query | 0 | 0.38～0.43 | 0.23～0.29 | 0.31～0.35 | — |
| 当前 wrist query | 0 | 0.16～0.21 | 0.40～0.47 | 0.32～0.42 | — |
| 文本 query | 0.056～0.074 | 0.08～0.12 | 0.15～0.17 | 0.65～0.69 | — |
| 动作 query | 0.008～0.016 | 0.09～0.10 | 0.25～0.32 | 0.25～0.33 | 0.30～0.32 |

prefix 不是全双向的：记忆区 query 只看记忆区，当前图像 query 看不到记忆区。所以「对全部 prefix query 求平均」会把记忆区 key 的质量和记忆区自注意混在一起。本文按 query 族分开算，每族只在它看得见的 key 上算均匀期望（某区域在该族的总质量 > 1e-6 才算看得见）。f16 存盘行和最大偏差：prefix 4.0e-4、动作 3.9e-4；padding key 被分到的质量 0。

## 二、sink 判据

1. **单次推理、单族**：对某个 query 族（记忆区／base／wrist／文本 19 行平均／「green」词第 16 行／动作 10 步 × 20 query 平均），每层先在头上平均，得到每个 key 的质量 m，倍数 r = m × 该族看得见的 key 数（均匀时 r = 1）。key 在 **≥ 6/18 层 r ≥ 20** 时，记为该族该次推理的 sink。
   - 为什么取 20 倍：P ≈ 1100 时，20 倍相当于单个 key 拿到 1.8% 的质量。`<bos>` 各族层平均倍数的中位数为 157～448（`sink_list[0].per_family.*.median_ratio_layer_mean`），远高于这条线；排在它后面的普通 key（当前图像 patch、记忆帧 token）层平均倍数多在 3～10、偶尔 17，不会被误判。#18 用 5/10/20/50 倍做过敏感性检查（≥6 层成立的 key 数，`threshold_sensitivity_infer18`）：prefix 全部 query 平均，ctx_ep8 / ctx_ep4 / mot_ep8 分别为 29/4/1/1、25/11/2/1、13/9/6/3；动作 query 分别为 41/28/9/1、41/25/8/1、37/21/5/2。20 倍处的动作族 key 只有 `<bos>` 和动作 token（`action_ge20_keys`）。20 倍处排除了大部分偶发尖峰，剩下的都是 `<bos>`、motion 窗、动作 token 这类每次推理都在的位置。
   - 为什么取 6 层：sink 的特点是跨层持续出现。只在 1～2 层偶尔冒头的尖峰（例如下文 X4 的第 0 层）是层特有的选择，不按 sink 处理。
2. **跨推理**：语义位置（记忆帧写作「最新记忆帧／首帧 t=0／帧时刻 t」+ 帧内 j；motion 窗写起点；当前图像写 patch(行,列)；文本写 token；动作写序号）在该族 **≥ 50%** 的「该位置存在」的推理里被判为 sink，才算该族的确认 sink。
3. **跨 query**：在 ≥ 2 个族、而且至少一半看得见它的族里被确认的，才是「全局 sink」。只在一个族里确认的有两种情况：key 位于该族自己的小区域（文本 query → 文本 19 token、动作 query → 动作 20 token），算「自区域局部注意」，不当 sink 处理，因为这些族本来就把大量注意力放在自己的区域上（文本 query 给文本 65%～69%、动作 query 给动作 30%～32%），区域内的均匀份额已是全局均匀的约 17～39 倍；其余算「族内 sink」。
4. **帧流内**：另外只在记忆帧 token 内部算均匀（1/帧 token 数），判据同 1、2。这一步检查帧流内部有没有吃掉大部分质量的单 token。
5. **逐头**：另在 144 个（层，头）上分别统计 ≥ 20 倍的次数，作为旁证（`perhead_infer18`）。

## 三、sink 清单（`sink_list[*].per_family`，图 `fig_r3_2_sink_list.png`）

| 类别 | 位置 | 被确认的族（sink 推理数／存在推理数） | 中位 ≥20× 层数 | 层平均倍数中位 | 平均质量 |
|---|---|---|---|---|---|
| **全局 sink** | 文本#0 `<bos>` | base 60/60、wrist 60/60、文本 60/60、green 60/60、动作 60/60 | 17～18 | 157（base）～448（green） | base 29%、wrist 33%、文本 45%、green 44%、动作 24% |
| 族内 sink（仅记忆区 query） | motion 窗 [224,257) | 记忆区 4/4 | 15 | 98 | 19.7% |
| 族内 sink | motion 窗 [128,161) | 记忆区 10/10 | 14 | 67 | 16.3% |
| 族内 sink | motion 窗 [160,193) | 记忆区 8/8 | 14 | 83 | 16.2% |
| 族内 sink | motion 窗 [32,65) | 记忆区 13/16 | 9.5 | 27 | 9.2% |
| 族内 sink | motion 窗 [192,225) | 记忆区 6/6 | 12 | 36 | 7.0% |
| 族内 sink | motion 窗 [256,289) | 记忆区 2/2 | 9 | 32 | 6.1% |
| 自区域局部注意 | 动作 token 0／17／18／19 | 动作 57、40、56、60/60 | 7～10 | 16～29 | 1.5%～2.8% |
| 自区域局部注意 | 文本 first／press／both／buttons／`\n` | 文本 54、47、40、35、36/60 | 6～11 | 16～26 | 1.6%～3.1% |
| 自区域局部注意 | 文本 hiding／green／container | green 52、45、38/60 | 6～8 | 21～43 | 2.2%～4.3% |

另外：

- 7 条 motion 窗在文本、green、动作三个族里的 sink 比例都是 0。也就是说，motion 窗只被记忆区自己的 query 当 sink，动作 query 不这样看它们。#15～#19 记忆区 query 给这些窗的合计质量是 45.1%～54.4%（`episodes.mot_ep8.per_infer_sinks[i].q_mem.sink_mass_layer_mean`）。motion 模型从 #3 起出现这一现象（12.6%），随窗数增多升到 50% 以上。
- 基线记忆区里没有确认 sink。「最新记忆帧」的 j=3、9、11～15 在 #18 的 144 个（层，头）里有 13～47 个超过 20 倍（ctx_ep4 j=15 有 47 个；`episodes.ctx_ep*.perhead_infer18.prefix_all`），但跨推理看，j=15 只在 ep8 的 2 次、ep4 的 3 次推理里达到 ≥6 层（j=12 在 ep4 有 1 次；`single_infer_candidate_counts.ctx_ep*.q_mem`），没过 50%。
- **动作 query 看记忆区时没有任何 sink**：#18 动作族在记忆 key 上 ≥20× 的（层，头）最多只有 1/144 个（`episodes.*.perhead_infer18.action_memory_only`）；帧流内判据下各局没有一个帧 token 在任何一层 ≥20×（`near_miss["frame/action"]` 最大层数 0）。各局还有零星的单次推理候选（文本 `\n` 在 mot/ep8/ep4 为 3/9/12 次，wrist patch(9,8) 在 ep8/ep4 为 3/4 次；`single_infer_candidate_counts.*.action_non_self_non_global`），都没过 50%。
- 当前图像没有 sink：X4 疑为 sink 的 base／wrist patch(14,1) 在 base、wrist query 族里最多只在 2～3 层超过 20 倍（`near_miss`），图 5 里也没有绿框。

### X4「green 词帧流单 token sink」复核（`x4_green_check`）

按帧流内判据逐层看「green」query：motion 模型 #15～#19 层平均后帧流内最大的那个 token（#15～#19 依次为记忆帧 t=224 j=1、t=239 j=1、t=193 j=1、首帧 t=0 j=11、t=284 j=3），每次**只在第 0 层超过 20 倍**（倍数 171～256，`per_layer_ratio[0]`；即该层帧流 33.3%～50.0% 落在这一个 token 上，`max_layer_frame_share`），其余层最大 5.3 倍（`max_ratio_other_layers`）。基线 ep8／ep4 的最大 token 多在第 1 层最集中（该层帧流占 40.0%～94.6%，ep8 #19 在第 5 层），≥20 倍的层数为 1～5（`layers_ge20`），其余层最大 2.5～57.6 倍。所以 X4 看到的「单 token 占 73%～99%」是单一层（motion 第 0 层、基线主要第 1 层）在层平均时占了主导，不是跨层持续的 sink。X4「剔每帧最大 token」后 motion VIS 37.8% → 20.8% 的下降，主要是去掉了这一层的集中。

## 四、稳健重算（动作 query）

**做法**：对 `probs_action` 的每个（步，层，头，query）行，把剔除列置 0，再除以剩余列之和，然后与原口径一样全平均，交给 `aggregate_attn.split_streams` 分流（只 import 不改）。剩余质量 < 1% 的行（几乎全在 sink 上，重归一只会放大 f16 噪声）不参与平均，这类行占 0.3%～0.5%（`episodes.*.{focus,global}.RB.dropped_row_frac`）。阈值换成 1e-3（`RB_resid1e-3`，丢弃 0%～0.02%）后，各指标变化 ≤ 0.1 个百分点。保留行的行和偏差 4.4e-16。均匀期望按剔除后剩余的帧 token 数／窗数计。三档：

- **RA** = 剔全局 sink（`<bos>`）。
- **RB**（主口径）= RA + 动作族确认 sink + 该次推理动作族自己判出的非自区域 sink + 帧流内 sink。实测动作族没有自区域以外的确认 sink，RB_extra 为空，只在个别推理多剔一两个文本或 wrist 位置，所以 RB 与 RA 几乎相同。剔掉的原质量：ctx 23.5%～24.6%，mot 25.1%（`episodes.*.focus.RB.removed_mass_orig`）。
- **RC**（压力测试）= RB + 记忆区 query 的族内 sink（6 个 motion 窗），即把记忆区内部的 sink 也从动作 query 里剔掉。

**原口径复现**：#18 mot VIS 17.05%、SWAP1 17.19%、SWAP2 13.46%、motion SWAP 30.55%（均匀 35.02%）；ctx_ep8 VIS 3.57%、SWAP1 9.31%、SWAP2 4.16%。与 X3 表（17.0／17.2／13.5／30.5（35.0）；3.6／9.3／4.2）一致。

### 4.1 #15～#19 平均（占比 / 均匀期望，%；`episodes.*.focus.{orig,RB,RC}`）

| 局 | 指标 | 原口径 | RB 去 sink | RC 压力 | 原 vs 均匀 → RB vs 均匀 |
|---|---|---|---|---|---|
| ctx_ep8 | 帧流 VIS | 7.0 / 13.1 | 7.1 / 13.1 | 7.1 / 13.1 | 低于 → 低于（保持） |
| ctx_ep8 | 帧流 SWAP1 | 15.1 / 17.5 | 16.1 / 17.5 | 16.1 / 17.5 | 低于 → 低于（保持） |
| ctx_ep8 | 帧流 SWAP2 | 7.6 / 18.1 | 7.2 / 18.1 | 7.2 / 18.1 | 低于 → 低于（保持） |
| ctx_ep4 | 帧流 VIS | 6.7 / 13.1 | 6.7 / 13.1 | 6.7 / 13.1 | 低于 → 低于（保持） |
| ctx_ep4 | 帧流 SWAP1 | 11.2 / 17.5 | 12.0 / 17.5 | 12.0 / 17.5 | 低于 → 低于（保持） |
| mot_ep8 | 帧流 VIS | 17.2 / 13.1 | 16.8 / 13.1 | 16.5 / 13.1 | 高于 → 高于（保持，1.31→1.28 倍） |
| mot_ep8 | 帧流 SWAP1 | 16.3 / 17.5 | 16.7 / 17.5 | 16.6 / 17.5 | ≈均匀（保持） |
| mot_ep8 | 帧流 SWAP2 | 17.6 / 18.1 | 17.7 / 18.1 | 17.9 / 18.1 | ≈均匀（保持） |
| mot_ep8 | motion SWAP 并集 | 31.7 / 38.8 | 31.0 / 38.8 | 50.5 / 66.7 | 低于 → 低于（RC 也低于） |
| mot_ep8 | motion SWAP1 | 6.1 / 19.4 | 6.1 / 19.4 | 44.2 / 51.5 | 低于 → 低于（RC 也低于） |
| mot_ep8 | motion SWAP2 | 25.5 / 19.4 | 24.9 / 19.4 | 6.2 / 15.2 | 高于 → 高于；**RC 翻转为低于** |

### 4.2 全部 20 次推理平均（%；`episodes.*.global.{orig,RB,RC}`）

| 局 | 帧流 VIS 原 → RB（均匀） | 帧流 SWAP1 原 → RB（均匀） | 帧流 SWAP2 原 → RB（均匀） | motion SWAP 并集 原 → RB → RC（均匀 原／RC） |
|---|---|---|---|---|
| ctx_ep8 | 25.7 → 26.3（35.9） | 36.9 → 37.3（24.4） | 8.5 → 8.0（20.8） | — |
| ctx_ep4 | 25.1 → 25.7（35.9） | 32.5 → 32.5（24.4） | — | — |
| mot_ep8 | 41.4 → 40.4（35.9） | 21.7 → 22.4（24.4） | 19.7 → 20.1（20.8） | 28.7 → 29.3 → 43.7（35.5／50.0） |

（全局含 #0～#3 这类历史帧全在 VIS 内、占比恒为 1 的早期推理，均匀期望被抬高；基线 SWAP1 全局高于均匀来自 #4～#13 的近因，X2/X3 已说明。）

### 4.3 逐次推理（mot_ep8，原 / RB / RC，均匀写在括号里；`episodes.mot_ep8.action_shares`）

| 推理 | 帧流 VIS | motion SWAP1 | motion SWAP2 |
|---|---|---|---|
| #15 | 15.3 / 15.5 / 14.9（15.6） | 2.5 / 3.0 / 31.2（22.5；RC 51.5） | 19.2 / 19.7 / 3.9（22.5；RC 15.2） |
| #16 | 21.0 / 19.9 / 19.6（12.5） | 3.7 / 4.1 / 47.4（19.7；RC 51.5） | **44.1 / 41.0 / 8.6**（19.7；RC 15.2） |
| #17 | 17.0 / 16.6 / 16.3（12.5） | 6.0 / 5.9 / 45.0（19.7；RC 51.5） | 16.1 / 16.1 / 5.0（19.7；RC 15.2） |
| #18 | 17.0 / 16.7 / 16.6（12.5） | 9.0 / 8.3 / 47.2（17.5；RC 51.5） | 21.6 / 22.0 / 5.4（17.5；RC 15.2） |
| #19 | 15.5 / 15.2 / 15.0（12.5） | 9.4 / 9.5 / 50.4（17.5；RC 51.5） | 26.8 / 25.7 / 8.1（17.5；RC 15.2） |

基线 #15～#19 去 sink 后帧流 VIS 变化 ≤ 0.7 个百分点（ctx_ep8 #18 3.6 → 4.0，均匀 12.5），逐次见 json `action_shares`。

### 4.4 结论保持与翻转

| 上一轮结论（result.md「可解释性深挖」） | 去 sink（RB） | RC 压力测试 |
|---|---|---|
| 基线帧流 VIS 低于均匀（ep8、ep4，#15～#19 约 0.5 倍） | 保持（7.1%、6.7% vs 13.1%） | 保持 |
| motion 模型帧流 VIS 高于均匀，#16～#19 为 15.5%～21.0% | 保持（15.2%～19.9%，#15～#19 平均 1.28 倍） | 保持（1.26 倍） |
| 两模型帧流在 #15～#19 都不偏向 SWAP | 保持 | 保持 |
| 基线 ep4／ep8 在 #4～#13 偏向 SWAP1（近因） | 保持（全局 SWAP1 37.3%／32.5% vs 24.4%） | 保持 |
| motion 流对 SWAP1 低于均匀 | 保持（#15～#19 6.1% vs 19.4%） | 保持（44.2% vs 51.5%） |
| 只有 #16 对 SWAP2 明显偏高（44.1% vs 19.7%） | 保持（41.0%） | **翻转**（8.6% vs 15.2%）：这份偏好落在窗 [128,161) 上，而这个窗正是记忆区 query 的族内 sink |
| X4：motion 模型 green query 的 VIS 偏高主要来自 t=0 单 token sink | 修正：该集中只出现在第 0 层，不是跨层 sink（第三节） | — |
| X4：动作 query 去 sink 后 VIS 不变 | 保持，且现在用完整的 sink 判据确认了：动作 query 看记忆区没有 sink | — |

## 五、图（都已目视自查：热图与 16×16 patch 图用最近邻、无插值；文字无遮挡；图例与正文一致）

- `fig_r3_1_layer_key_heat.png`：#18 三局 × 三族（记忆区／文本／动作 query）逐层 key 倍数热图（log10，黑 = 看不见或 ≪ 均匀，绿三角 = 该次判为 sink）。记忆区 query 只有左边一块；motion 模型那几条亮竖线就是 motion 窗；文本与动作族最右边的亮竖线是 `<bos>`。
- `fig_r3_2_sink_list.png`：确认 sink 清单 × 6 个族的「判为 sink 推理比例」矩阵（灰 = 看不见）。
- `fig_r3_3_sink_mass_by_layer.png`：#15～#19 平均的逐层 sink 合计质量（记忆区 query：只有 motion 模型有，第 11 层达 75.4%，`sink_mass_by_layer_focus.mot_ep8.q_mem`；文本与动作 query 的曲线里含自区域局部注意的 token）。
- `fig_r3_4_orig_vs_robust.png`：20 次推理的帧流 VIS／SWAP1／SWAP2、motion 流 SWAP 并集曲线，实线为原口径、虚线为 RB、点线为 RC、浅色粗线为均匀期望。
- `fig_r3_5_image_patch_sink.png`：#18 动作 query 在当前 base／wrist 16×16 patch 上的倍数（`cv2.INTER_NEAREST` 放大），没有一处被判为 sink。

## 六、局限

- **只存了注意力概率，没有值向量**。sink 的常见解释是「拿到大量注意力，但值向量范数很小，对输出几乎没有贡献」，这里无法检验 `<bos>`／motion 窗的 ‖v‖，也算不了 attention × ‖v‖ 加权的贡献。剔 sink 重归一假设 sink 对输出贡献可以忽略，这只是一种读法，不是因果结论。
- 20 倍、6 层、50% 推理三个阈值是手设的。第二节给了 5～50 倍的敏感性，结论对 `<bos>` 和 motion 窗不敏感，对单次推理候选有影响。
- 头平均以后才判 sink（逐头只做了 #18 的旁证）。Gemma 多 query 头共享 KV，头间差异可能被平均掉。
- 「看得见」按区域总质量判断，没有读模型的 attention mask 源码；记忆区 query 只看记忆区是从数据里实测出来的。
- RC 剔掉了 9 个 motion 窗里的 6 个，剩下的窗大多在早期（[0,33)、[64,97)、[96,129)），motion 流占比几乎由这几个窗决定，均匀期望也随之变化。RC 只能说明「#16 SWAP2 偏好依赖于一个在记忆区内部是 sink 的窗」，不能读成「motion 流真实的交换偏好」。
- 样本小：基线 2 局（含 1 局失败）、motion 1 局，单 seed，#15～#19 只有 5 次推理，没有置信区间。
- 记忆帧的语义 id 用帧时刻，采样帧随推理变化，同一个 id 的「存在推理数」少（2～16），跨推理比例的分母小。

## 七、复现

```bash
cd <worktree>
E="UV_PROJECT_ENVIRONMENT=/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=<worktree>:<worktree>/src"
# 1) 逐局压缩（每局约 1 分钟，逐次推理流式读，写到缓存目录，不进 git）
env $E uv run --no-sync python scripts/interpret/analysis/r3_sink.py stats --tag ctx_ep8 --cache <缓存目录>
env $E uv run --no-sync python scripts/interpret/analysis/r3_sink.py stats --tag ctx_ep4 --cache <缓存目录>
env $E uv run --no-sync python scripts/interpret/analysis/r3_sink.py stats --tag mot_ep8 --cache <缓存目录>
# 2) 判定、稳健重算、出图（约 1.5 分钟）
env $E uv run --no-sync python scripts/interpret/analysis/r3_sink.py report --cache <缓存目录>
# 末行：R3_SINK=PASS episodes=3 sinks=7 figs=5
```

`sinks=7` = 1 个全局 sink（`<bos>`）+ 6 个族内 sink（记忆区 query 的 motion 窗）；自区域局部注意的 12 个位置列在清单里，但不计数。
