# v1-dtype-p3-dump-pre 结果

`EXIT_CODE=0`，两个阶段全过：`DUMP_PASS` + `GRAD_PASS`。起跑 commit `f2e7348`
（commitV2.4a，`src/` 零改动），工作区 clean（驱动脚本内置硬闸）。

## 一、定点样本 dump（2,600 个，801.3 s）

分组精确：step_idx ∈ {0,1,2,29,30,31,32,33} 各 200 + 固定 seed 随机 1,000。
其中短样本（`step_idx ≤ 30`）1,068 个、满长 1,532 个——随机 1,000 里命中 68 个短样本
（6.8%），与总体比例 6.27% 一致。

**`static_image_emb` 的 dtype 分布零例外，双 dtype 路径完整成立**：

| 样本类别 | 数量 | `static_image_emb` dtype |
|---|---|---|
| 短样本（step_idx ≤ 30） | 1,068 | **float64**（100%） |
| 满长样本（step_idx ≥ 31） | 1,532 | **bfloat16**（100%） |

## 二、定点 batch dump（200 个，61.6 s）

走完整 `transform_dataset` + `_collate_fn`，与 P6 `batch_digests` 的记录点
（collate 后、device_put 前）逐字对齐。

| batch 组成 | 数量 | `static_image_emb` | `static_pos_emb` | `static_state_emb` |
|---|---|---|---|---|
| `allfull`（全满长） | 50 | bfloat16 ×50 | float32 ×50 | float64 ×50 |
| `allshort`（全短样本） | 50 | float64 ×50 | float64 ×50 | float64 ×50 |
| `mixed1`（1 短 + 7 满长） | 50 | float64 ×50 | float64 ×50 | float64 ×50 |
| `random`（随机组成） | 50 | **bfloat16 ×28 / float64 ×22** | **float32 ×28 / float64 ×22** | float64 ×50 |

**`random` 档 28/22 的分裂是「XLA 被迫编译两份产物」最直接的证据**——同一档随机
batch 里 dtype 就在摆动。22/50 = 44% 含短样本，与理论值吻合（随机池含 6.8% 短样本，
1 − 0.932^8 = 43.0%）。

`static_state_emb` 四档恒 float64：它经 `_normalize_state`（norm stats q01/q99 为
f64）恒被抬到 f64，与 padding 是否发生无关——这正是计划里说的「第三处修复在交付键
上不可观测」，它只能由归一化前的纯函数位型测试验证。

## 三、单步梯度 A 侧（三个定点 batch）

- **初始 state 同源校验 `PASS`（177 叶子全等）**：同 seed / 同 config 现场
  `init_train_state` 产出的完整 TrainState，与 G0b r1 `param_checksums.jsonl` 步 0 的
  `per_leaf` 逐条相同。跨 commit 引用 G0b 初始状态因此机器可证，**无需加载本机那份
  45.4 GiB 的 `state_step_0.bin`**。
- 口径：2×RTX 6000 Ada、b8、seed 42、`fsdp_devices=2`、
  `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`（D2 档）。
- 可训练叶子 **32 个**（`config.trainable_filter` 过滤后），梯度约 11 GB/batch。

| batch | batch_id | 交付 dtype（image / pos / state） | 单步 loss (hex) | 用时 |
|---|---|---|---|---|
| `mixed1`（主判据） | 0 | float64 / float64 / float64 | `0x1.0d48f00000000p-1` | 145.9 s |
| `allshort`（密度最大） | 50 | float64 / float64 / float64 | `0x1.0f06060000000p-2` | 119.0 s |
| `allfull`（阴性对照） | 100 | **bfloat16 / float32** / float64 | `0x1.37890e0000000p-1` | 137.9 s |

**额外的确定性自证**：`mixed1` 的 loss 与本轮更早一次独立冒烟（不同 `EXP_NAME`、
各自冷编译）完全相同（`0x1.0d48f00000000p-1`）——D2 确定性档在本工具链上再次成立。

## 四、产物

| 路径 | 内容 | 体积 | 处置 |
|---|---|---|---|
| `v1-store/dtype-unify/v1-dtype-p3-dump-pre/` | 2,600 样本 + 200 batch 摘要，memory 四键位型容器 | 7.1 GB | P4 对拍后清理 |
| `v1-store/dtype-unify/v1-dtype-p3-dump-pre-grad/` | A 侧逐叶梯度摘要与统计（`grad_summary.json`） | 136 KB | 留档进 git |
| `v1-store/dtype-unify/v1-dtype-p3-dump-pre-gradfix/` | A 侧三个 batch 的位型容器 | 80 MB | P5 对拍后清理（长期 fixture 取 B 侧） |
| `/data/hongzefu/v1-baselines/dtype-p5-grad-pre/` | A 侧梯度数组本体（3 × 32 叶） | 33 GB | **P6 验收通过后删除**，只留 sha 清单与逐叶统计 |

## 五、数据事实声明

短样本档（`step_idx ≤ 30`）只能取自 800 个 `exec_start_idx == 0` 的 Button 系
episode。Video 系（VideoUnmask / VideoUnmaskSwap）的 `exec_start_idx` 最小 66，其样本
`step_idx` 恒 ≥ 66、**永远走满长切片分支、根本产生不出短样本**。这是数据本身的性质，
不是取样偏置；满长档与随机 1,000 自然覆盖两系。
