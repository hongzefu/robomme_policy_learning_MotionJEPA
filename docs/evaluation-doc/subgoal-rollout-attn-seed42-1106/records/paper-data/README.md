# 子目标注意力论文主图（fig_pathway）数据

这里是论文主图 `records/paper-figures/fig_pathway.pdf`（版本 6，定稿）用到的全部原始数据，都是 CSV，不依赖原仓库。由 `records/paper-figures/export_paper_data.py` 从与画图脚本相同的数据源导出，并与画图脚本写出的 `fig_pathway-layers.json`、`summary.json` 逐格比对，判定行 `PAPER_DATA_EXPORT=PASS files=7 heatmap_err=4.99e-03 matrix_err=4.99e-03 mask_match=1 a2_err=4.97e-07 episodes=100 inferences=495`（参考 JSON 只存两位小数，所以容差 5.1e-3；CSV 存六位小数）。

## 数据口径

- **模型与任务：** motion 模型（`mme_vla_suite_orig16_b128_60k`，`stage2_pool4_concat`，第 59999 步），PatternLock 与 RouteStick 各 50 局真实 rollout，seed 42，全部 495 次推理都存了完整注意力；基线模型只出现在第二部分（`ctx`）。
- **注意力读数：** 每次推理、每层、每头，读者组对来源组的注意力质量之和（读者组内按 token 平均），头平均，先在局内平均、再按局等权平均；`scope = pooled` 是两任务各 50 局等权合并。所有百分比都是「占读者注意力质量的 %」，同一读者对全部来源（含自身与 padding）求和为 100。
- **帧累积 motion 含量：** attention rollout `R[l+1] = α R[l] + (1−α) A[l] R[l]`，α = 0.5，帧 token 的来源构成里 motion 两组所占的 %，第 l 层输入；忽略 MLP 与 value 范数，只是注意力近似。
- **第二部分：** 对位矩阵 `enrichment = 执行第 j 个子目标时读示范第 j′ 段的注意力份额 ÷ 按 token 数均匀分配时的份额`（1 = 均匀）；D 是对角富集，斜率是读取质心随 j 的斜率，区间为局级重采样 2000 次（seed 42）的 95% 区间；预定阈值 D 下界 ≥ 1.2 或斜率下界 ≥ 0.2 算「明显」。
- **这些结果的范围：** 只是注意力读数，说明模型「看哪里」，不证明因果信息流；只有 seed 42 一次。

## 文件和图的对应

| 文件 | 用于哪张图 | 行数 |
| --- | --- | ---: |
| `mask_blocks.csv` | 图 (a) 块状注意力掩码矩阵 | 36 |
| `pathway_heatmap.csv` | 图 (b) 7 行 × 18 层热图（`scope = pooled`；另附两任务各自的值） | 378 |
| `layer_block_attention.csv` | 图 (b) 的上游：逐层 6 × 6 块注意力（pooled 与两任务），可自行组合任何「读者 ← 来源」边 | 1944 |
| `frame_motion_content.csv` | 图 (b) 「frames: cum. motion content」行（第 l 层输入，l = 0～18；热图取 l+1 即该层输出） | 57 |
| `route_shares.csv` | 正文数字：动作中来自 motion 的注意力质量及末跳经帧／语言／图像／motion 的占比，α = 0.2 / 0.5 / 0.8 | 6 |
| `subgoal_alignment_matrix.csv` | 图 (c) 四张矩阵（RouteStick）；同时含 PatternLock 与两种 motion 权重的全部格 | 382 |
| `subgoal_alignment_summary.csv` | 图 (c) 标题里的 D、斜率、成功局 D，以及首段／真对位／末段拆解均值 | 16 |

## 列的含义

- `mask_blocks.csv`：`query_block` 读、`key_block` 被读；块名 `frames`（512 个帧 token）、`motion`（PatternLock 平均 3.5 个、RouteStick 7.0 个有效 token，列里写 `3.5|7.0` 形式）、`image_3rd`、`image_wrist`（各 256）、`language`（28）、`action`（20）。`allowed` 1 = 可见；`kind` 为 `full`、`masked`、`causal`（动作内部因果）。可见性由数据判定：某块对在全部层、头、局上注意力恰为 0 才算 `masked`。
- `layer_block_attention.csv`：`scope` 为 `pooled` / `PatternLock` / `RouteStick`；`layer` 0～17；`reader_block` 读 `source_block` 的 `attention_pct`。
- `pathway_heatmap.csv`：`edge` 写法「读者 ← 来源」，`route` 为 `A`（经帧记忆）、`B`（经语言）、`-`；`value_pct` 即热图格值；`frames: cum. motion content` 行取自 `frame_motion_content.csv` 的 `layer_input = layer + 1`。
- `frame_motion_content.csv`：`layer_input` 0～18，`motion_content_pct`。
- `route_shares.csv`：`motion_content_in_action_pct` 是动作 token 注意力质量中来自 motion 的 %；`via_*_frac` 是末跳经各组的占比（和为 1）；`alpha` 字符串 `0.2` / `0.5` / `0.8`。
- `subgoal_alignment_matrix.csv`：`model` 为 `mot`（motion 模型）或 `ctx`（基线）；`reader` 为 `action` / `text`（语言）；`stream` 为 `frame` / `motion`；`weight` 为 `single`（帧流，每帧一票）、`overlap`（motion 窗与段有重叠就算）或 `proportional`（按重叠比例）；图用 `frame/single` 与 `motion/overlap`。`executing_j`、`demo_segment_jprime` 从 1 起；`n_episodes` 是该行的局数。
- `subgoal_alignment_summary.csv`：`D_*`、`slope_*` 为均值与 95% 区间；`tier` 为「明显／弱／未见」；`success_only_*` 只在图用的权重行有值；`first_segment_mean`（j ≥ 2 时读第 1 段）、`true_alignment_mean`（1 < j < m 时读第 j 段）、`last_segment_mean`（j < m 时读第 m 段）是事后拆解，未校正多重比较。

## 复现原图的设置

| 项目 | 取值 |
| --- | --- |
| (a) 掩码矩阵 | 行 = query 块、列 = key 块，实心 `#3a3a3a`，动作对角 `#9a9a9a` 标 causal，白色 1 pt 网格 |
| (b) 热图 | 行序 = `pathway_heatmap.csv` 出现顺序；单色蓝 `#f4f7fb → #2a78d6 → #0b2f63`，PowerNorm γ = 0.5（平方根刻度），vmin 0、vmax 70；每格标数，≥ 9.5 取整、1～9.5 一位小数、< 1 写 `<1`；Route A 行标签 `#2a78d6`、Route B `#eb6834`；描边：橙框 `language ← motion`／`language ← frames` 的 L0–L1，蓝框 `action ← frames` 的 L13–L16 |
| (c) 对位矩阵 | 发散色 `#2a78d6`（0）→ `#f0efec`（1）→ `#e34948`（最大 3.6），TwoSlopeNorm 中心 1；四张图：RouteStick `mot/action/frame/single`、`mot/text/frame/single`、`ctx/text/frame/single`、`mot/text/motion/overlap` |
| 颜色 | 帧 `#2a78d6`、motion `#eb6834`、语言 `#1baf7a`、图像 `#6250d6`、文字 `#1f1f1f` / `#555555` / `#8a8a8a` |

## 来源与引用

- 逐局通路：`v1-store/evaluation/subgoal-rollout-attn-seed42-1106/mot/pathway/<nonce>.npz`（`records/pathway_flow.py` 产物，`PATHWAY_FLOW=PASS episodes=100 inferences=495 mask_max=0`）。
- 子目标对位：`records/summary.json`、`success-only.json`、`decompose.json`（`scripts/interpret/analysis/subgoal_attention.py`）。
- 文字说明与英文图注：`records/paper-figures/fig_pathway-logic.md`。
