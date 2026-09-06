# eval-awsprod40k-b128-motion — 结果

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。执行 commit `74d60d0`（clean HEAD），被评 checkpoint
`awsprod40k-b128-motion/39999`，policy 采样 seed 42，benchmark 上游原样（test split、每任务 50 集、`max_steps=1300`）。
墙钟 `2026-09-06 01:44:16 → 02:22:21`（**38 min**，8 卡并行，最慢片 36.4 min）。

## 结论先行

```
EVAL_40K_MOTION=DONE tasks=4 episodes=200/200 errors=0 VideoUnmask=14/50 ButtonUnmask=13/50 VideoUnmaskSwap=12/50 ButtonUnmaskSwap=17/50 mean_rate=0.2800 timeout=2/200 fail=142/200
```

| 任务 | 成功 / 50 | 成功率 | easy（26） | medium（12） | hard（12） | fail | timeout |
|---|---|---|---|---|---|---|---|
| VideoUnmask | 14 | **28.0%** | 8 | 4 | 2 | 36 | 0 |
| ButtonUnmask | 13 | **26.0%** | 10 | 3 | 0 | 36 | 1 |
| VideoUnmaskSwap | 12 | **24.0%** | 7 | 5 | 0 | 38 | 0 |
| ButtonUnmaskSwap | 17 | **34.0%** | 7 | 7 | 3 | 32 | 1 |
| **四任务均值** | 56 / 200 | **28.0%** | 32/104（30.8%） | 19/48（39.6%） | 5/48（10.4%） | 142 | 2 |

- **总成功率 28.0%**（`eval.py` 公式：任务成功率算术平均；因每任务同为 50 集，与集均值相同），四任务在 24–34% 之间，无一任务明显脱离。
- **失败几乎全是「做错」而不是「没做」**：142 集 fail = 环境判定策略执行了错误的终止性动作（选错容器 / 按错按钮），只有 2 集走满 1300 步 timeout。
  这与训练 loss 收敛到 0.0010 一致——策略动作果断、贴近示教分布，但记忆到底该「记住谁」的判断在 test 集上只有约三成正确。
- **难度**：hard 只有 5/48（10.4%），medium 19/48 反而高于 easy 32/104；样本小（每格 12–26 集），不作难度趋势结论。
- **能否归因到 motion memory**：不能。本轮没有 nomotion 对照（launch.md 已述），28% 只是这一条 run 的绝对数；官方 MME-VLA 数字因配置偏离
  （batch 128 / 40k / lr 1e-4）也不逐项可比。
- **单 seed、单 checkpoint**。

## 耗时与资源（独占卡实测，与历史争用值对照）

| 环节 | 本轮（一片一卡） | 历史（`aws-t3-open-s100` 两 sidecar + 两仿真挤一张卡） |
|---|---|---|
| sidecar 一窗（`add_buffer` ≤16 帧）median | **1.46 s**（mean 1.23–1.36 s） | 3.24 s |
| 首批（demo 段整段）mean | VideoUnmask 4.5 s、VideoUnmaskSwap 11.8 s | 11.3 / 30.1 s |
| policy `infer` median | **69 ms** | 186 ms |
| 单集墙钟 median | success 35 s、fail 37 s、timeout 161 s | 4.8–7.9 min（T3 模型全 timeout） |
| 每卡显存 | policy 32.9 GB + sidecar 3.3 GB + 仿真 0.8 GB ≈ 37 GB | — |

8 片墙钟 9.4–36.4 min（`records/shard_timing_summary.txt`）：Video 类最快（每集 20–35 s），ButtonUnmaskSwap-0 最慢（含 1 集 timeout、多集长 fail，median 80 s）。
预估的「25 集 × 2.7 min ≈ 70 min」按全 timeout 算，实际策略提前终止，整体快一倍。

## 盲区诚实清单

- 单 seed、单 checkpoint（39999）；无 nomotion 对照、无其它 checkpoint 对照（`5000…35000` 都在，需要时同一套脚本换 `CKPT_ID` 即可）。
- test ep 0–9 在 MotionJEPA encoder 训练集内（holdout 90–99）；训练 50.6 epoch 无验证集。
- 审计 D-A1：在线记忆 token 与训练存在约 0.3% 的 bf16 舍入 + 批形状差（`siglip-ab-replay-40k`），动作影响为采样噪声的 2%，本轮未消除。
- 审计 B-A2（在线 τ 可达 1296）：只有 2 集走满 1300 步，绝大多数集在百步量级终止，τ 越界几乎未被触发。
- 成功率来自环境 `status` 判定，本轮未人工抽看视频。

## 产物

- `records/summary.txt`（判定行 + 三张表 + 逐集 200 行）、`records/per_episode.json`（逐集 seed / 难度 / 结果 / 分片 + 各片 TIMING 与墙钟）。
- `records/ev40k-<Task>-<k>.txt` × 8：分片驱动 + eval 客户端日志（`EXIT_CODE=0`）；`records/shard_timing_summary.txt`：各片起跑行与 `TIMING_SUMMARY`。
- `records/test_seeds.json`：`TEST_SEED_DISJOINT=PASS` / `TRAIN_H5_IN_TRAIN_SPLIT=PASS`。
- 合并结果 `v1-store/evaluation/awsprod40k-b128-motion/ckpt39999/seed42/{progress.json,log.json,shards.json}`；视频 154 MB 留各分片目录（不进 git）。
- server 全量日志 `v1-store/logs/ev40k-*.server.log`（不进 git）。

## 逐集（编号 = test 元数据 episode；seed / 难度见 `records/test_seeds.json`）

- **VideoUnmask**：success 14 集 = `[3, 5, 7, 9, 16, 20, 21, 22, 25, 30, 34, 41, 45, 46]`；timeout = `[]`；其余 fail。
- **ButtonUnmask**：success 13 集 = `[1, 2, 4, 9, 17, 18, 20, 25, 29, 30, 33, 37, 45]`；timeout = `[36]`；其余 fail。
- **VideoUnmaskSwap**：success 12 集 = `[6, 8, 10, 13, 17, 18, 24, 28, 30, 33, 34, 41]`；timeout = `[]`；其余 fail。
- **ButtonUnmaskSwap**：success 17 集 = `[6, 10, 17, 18, 20, 21, 22, 23, 25, 33, 34, 36, 38, 39, 42, 45, 47]`；timeout = `[3]`；其余 fail。
