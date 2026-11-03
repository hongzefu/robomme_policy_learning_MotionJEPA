# R2：记忆区内部谁在读 VIS／SWAP（prefill prefix 内部 attention）

脚本：[`scripts/interpret/analysis/r2_memory_internal.py`](../../../../../scripts/interpret/analysis/r2_memory_internal.py)（`compute` 逐次推理流式读 `prefix_<seq>.npz`、f32、用完释放，缓存到 scratchpad；`report` 出图与 `r2_summary.json`）。数据为 stage1b 三局（基线 ctx512 ep8 成功、基线 ep4 失败且只交换 1 次、ctx512+motion ep8 成功），每局 20 次推理，`probs_prefix` 形状 (18 层, 8 头, P, P)，f16。

## 一、问题

上一轮（X1～X4）只看了动作 query 直接看记忆区的情况：基线动作 query 几乎只看最近帧。那 VIS（方块裸露）和 SWAP（交换）的信息有没有在 prefill 阶段先被记忆区内部的 token 读走、再间接传给动作？本分析只看 query 与 key 都在记忆区的那一块 attention。

## 二、自检与可见关系

| 项 | 结果 |
|---|---|
| 有效记忆 query 行在全部 prefix key 上的行和误差 | 三局 60 次推理最大 3.97e-4（f16 量化，阈值 5e-4） |
| 记忆 query 行落在当前图像／文本／记忆 padding key 上的质量 | 三局全部恰为 0 |
| 补充（只查了 motion #18）：当前图像 query 行落在记忆 key 上的质量 | 恰为 0 |

结论：prefill 时记忆区是一个封闭块，只在块内互相看（块内双向），看不到当前图像和文本。因此均匀期望就取「有效记忆位里属于该组的比例」。

## 三、分组定义

- 帧 token（`mem_order[s] < 512`，帧 f = `mem_order[s] // 16`，全域帧号 `frame_times[f]`）。每帧只进一个组，优先级为 最近 > VIS > SWAP1 > SWAP2 > 中段：
  - 最近 = 本次推理 `frame_times` 最大的 4 帧（#18 是 260／269／278／288）；
  - VIS = [0,32)（#18 是 0／9／18／27）；SWAP1 = [64,114)；SWAP2 = [114,164)；
  - ep4 只交换 1 次，没有 SWAP2，它的 [114,164) 帧归入中段；
  - 中段 = 其余帧。
- motion token（`mem_order[s] ≥ 512`，窗 w，窗口 [`motion_times[w]`, +33)）：按与 VIS／SWAP 的重叠帧数除以 33 分成分数权重，分到 motVIS／motSWAP／motOther，三者之和为 1。#18 共 9 个窗，起点 0, 32, …, 256，每窗 1 个 token。
- 度量：质量 M[gq,gk] 指 query 组平均每行落在 key 组上的概率质量。均匀期望 U[gk] 等于 key 组的有效位数除以全部有效记忆位数。富集倍数 E = M／U，1 表示和均匀分配一样。表里说「头平均」时，是先在 8 个头上平均 M，再除以 U；整局汇总按 query 权重把各次推理的 M、U 加权平均后再相除。

## 四、关键数字

### 1. 基线：最近帧怎么读 VIS／SWAP（重点 1）

按头平均来看，最近帧对 VIS 和 SWAP 的富集都低于均匀；它主要读的是最近帧自己。

| 局 | 范围 | 最近→最近 | 最近→VIS | 最近→SWAP |
|---|---|---|---|---|
| 基线 ep8 | #15～#19，层均、头均 | 质量 0.378，3.03× | 质量 0.032（均匀 0.131），0.24× | 质量 0.160（均匀 0.356），0.45× |
| 基线 ep8 | 整局 | 0.402，2.83× | 0.132，0.43× | 0.245，0.84× |
| 基线 ep4 | #15～#19 | 0.386，3.09× | 0.035，0.26× | 0.084（均匀 0.175，只有 SWAP1），0.48× |

但有少数头专门读 VIS 或 SWAP。头平均把它们稀释掉了，见图 `fig_r2_4_heads_k18.png`。

| 头 | 通路 | ep8：#15～#19 质量（富集） | ep4：#15～#19 质量（富集） | 读的是哪里（#18，ep8） |
|---|---|---|---|---|
| **L14H6** | 最近→VIS | 1.00／0.88／0.89／0.88／0.89（6.4～7.1×）；整局 #4 之后中位数 1.00 | 1.00／0.88／0.88／0.88／0.87（6.4～7.1×） | 分散在 VIS 各帧：18 帧 26%、9 帧 25%、27 帧 23%、0 帧 14%，没有单一 sink |
| **L1H0** | 最近→VIS | 0.32／0.72／0.92／0.90／0.88（2.0～7.4×） | 0.20／0.56／0.83／0.76／0.69（1.2～6.6×） | 0 帧 61%、18 帧 19% |
| **L5H4** | 最近→SWAP | 1.00／1.00／0.99／0.99／0.95（2.5～3.0×） | 0.99／0.99／0.89／0.73／0.49（3.2～5.7×） | 集中在 SWAP1 中段：92 帧 57%（token j11～j15，也就是 4×4 网格的最后一行附近）、83 帧 20%、102 帧 17% |
| **L12H7** | 最近→SWAP | 0.87～0.67（约 2.1～2.2×） | 0.69～0.43（2.8～3.9×） | 92 帧 42%、83 帧 21%、185 帧 13% |

最近→SWAP 随推理怎么变（图 `fig_r2_3`）：头平均富集只在 step 112～160 高于 1（基线 ep8 峰值约 1.7×）。这段时间 SWAP 帧离「最近 4 帧」很近，所以这个峰主要是局部性（时间上相邻）造成的。到 step 176 以后，富集掉到 0.9× 以下，并持续下降，#18 为 0.4～0.5×。

### 2. motion 模型：motion token 读什么，帧 token 是否读 motion（重点 2）

- **帧 token 大量读 motion token**：#15～#19 全部帧 query 有 58% 的质量（层均、头均）落在 9 个 motion token 上，富集 37×；整局为 39%、41×。最近帧这一行在 #15～#19 的分配是：motion 0.567、帧 0.433。motion ep8 的 VIS→VIS 只有 0.65×，基线 ep8 是 1.71×；帧之间互相读的质量被 motion token 吸走了。
- **吸走质量的不是交换窗**：#18 全部记忆 query 对各窗的平均质量为：窗 [224,257) 0.176、[160,193) 0.148、[128,161) 0.082、[256,289) 0.067；完全落在交换期内的窗 [64,97) 只有 0.024，[96,129) 只有 0.008。个别头几乎全部压在单个窗上，例如 L8H0 的最近帧 query 有 83% 落在 [224,257) 窗。这更像 sink／寄存器式的汇聚，不是按交换内容读。
- **覆盖 SWAP 的 motion 窗读 SWAP 帧，只有弱偏好**：#15～#19 motSWAP→SWAP 帧为 0.142（均匀 0.351），0.40×；motOther→SWAP 0.23×，motVIS→SWAP 0.26×。相对其余窗高约 1.7 倍，但仍低于均匀。最强的头是 L5H5（整局 2.2×，质量 0.90），但 #18 时 L5H5 的最近帧 query 读的是最近帧自己。motSWAP→motSWAP 为 30×，motOther→motSWAP 为 19×，motion 窗之间也有一点「同类相读」。
- **motion 模型也有 VIS 读头，但不在同一个位置**：L0H5 的最近→VIS 在 #15～#19 质量 0.64～0.74（4.8～5.5×），读 0／9／18 帧的 j5／j6 token。基线的 L14H6 在 motion 模型里不读 VIS（质量约 0）；基线的 L5H4／L12H7 在 motion 模型里不读 SWAP（≤0.04）。两个模型是分别训练的，头号不能一一对应。

### 3. ep4（失败）与 ep8（成功）对比

- 记忆区内部读 VIS 的方式两局几乎一样。L14H6 在 #18 的富集：ep4 6.87×（质量 0.90），ep8 6.91×（0.91）。层均最近→VIS：ep4 0.26×，ep8 0.24×。
- ep4 读 SWAP 不比 ep8 少：L5H4 在 #15～#19 的富集是 3.2～5.7×，ep8 是 2.5～3.0×。ep4 的 SWAP 只有 SWAP1，均匀期望更小，所以倍数更高。按质量看，ep4 在 #18、#19 降到 0.73／0.49，ep8 保持在 0.99／0.95。
- 因此，单看记忆区内部的 attention 分配，找不到能解释 ep4 失败的明显缺口。

## 五、结论

1. prefill 时记忆区只在块内互相看，看不到当前图像和文本。
2. 基线记忆区里有专门的 VIS 读头（L14H6、L1H0）和 SWAP 读头（L5H4、L12H7）。最近帧 token 在这些头上把 70%～100% 的质量给 VIS 或 SWAP 帧，富集 3～7×，并且在 #15～#19 稳定存在。这可能是一条间接通路：最近帧先读到 VIS／SWAP，动作 query 再读最近帧；上一轮的直接分析看不到这一层。按头平均，这些通路被局部性主导的其他头稀释，低于均匀。
3. motion 模型的记忆区被少数 motion token 汇聚了大部分质量：帧 query 有 58% 读 motion，富集 37×。汇聚集中在 [224,257)、[160,193) 这类非交换窗，不是交换窗。覆盖 SWAP 的 motion 窗读 SWAP 帧的偏好很弱（0.40×，相对其余窗约 1.7 倍）。
4. ep4 与 ep8 的记忆区内部读法基本一致。失败原因不在这一层的 attention 分配上。

## 六、局限

- attention 质量不等于信息流：没有看 value 范数和 OV 输出，也没有做消融。上面的「间接通路」只是假设，需要用 attention knockout 或 patching 验证（例如屏蔽 L14H6 的最近帧→VIS）。
- 「最近 4 帧」优先于 VIS／SWAP。早期推理（#0～#3）里最近帧本身就在 VIS 内，所以 VIS 组为空或很小。整局汇总包含这些推理，看后期结论应以 #15～#19 为准。
- motion 窗按重叠帧数／33 分数计权，一个窗可能同时属于多组。motion 汇聚到个别窗的原因（位置还是内容）没有区分。
- 只有 3 局，失败局只有 1 局，不能做统计推断。基线与 motion 是不同权重，头号不可对应。
- f16 存储下，小于约 6e-8 的概率会下溢为 0。热图和曲线里的 0.0× 表示「≤ 量化下限」，不一定是严格 0。

## 七、图

- `fig_r2_1_heatmap_{ctx_ep8,ctx_ep4,mot_ep8}.png`：#18 每层 query 组×key 组的富集热图（log2，nearest，头平均）。
- `fig_r2_2_paths_by_layer.png`：四条关键通路（最近→VIS、最近→SWAP、motion(SWAP)→SWAP 帧、帧→motion）随层变化，上排 #18，下排整局。
- `fig_r2_3_paths_by_infer.png`：四条通路随推理变化（层均的质量与富集）。
- `fig_r2_4_heads_k18.png`：#18 逐层逐头富集。
- `r2_summary.json`：自检、分组、逐推理通路质量（逐层）与整局／#15～#19 汇总（严格 JSON）。

复现：

```bash
W=<worktree>; UV_PROJECT_ENVIRONMENT=/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=$W:$W/src \
  uv run --no-sync python scripts/interpret/analysis/r2_memory_internal.py compute --cache <缓存目录> --episode ctx_ep8   # 另两局同理
UV_PROJECT_ENVIRONMENT=... PYTHONPATH=... uv run --no-sync python scripts/interpret/analysis/r2_memory_internal.py report \
  --cache <缓存目录> --out docs/evaluation-doc/attn-ctx512-vs-motion-1103/analysis/r2
# 末行：R2_MEMORY_INTERNAL=PASS episodes=3 figs=6
```
