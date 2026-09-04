# v1-prod-60k / ckpt 59999 / seed 7 在线评估结果（test split，4 任务 × 50 episodes）

起跑档见同目录 `launch.md`。评估于 2026-09-04 14:53–15:18 在本机双卡完成，
两条流水线均 `EXIT_CODE=0`，**200/200 episodes 全部有结果、零 `"error"` 条目**。

## 总表

| 任务 | 成功 / 总数 | 成功率 |
|---|---|---|
| ButtonUnmask | 13 / 50 | **26.0%** |
| VideoUnmask | 16 / 50 | **32.0%** |
| ButtonUnmaskSwap | 15 / 50 | **30.0%** |
| VideoUnmaskSwap | 13 / 50 | **26.0%** |
| **total_success_rate** | 57 / 200 | **28.5%** |

`total_success_rate` 是四任务成功率的算术平均（与 `eval.py` 口径一致），
恰好也等于 57/200，因为四任务 episode 数相同。

## 按 difficulty 分层

difficulty 取自 test metadata 每条 record 的 `difficulty` 字段，与 episode 一一对应。

| 任务 | easy | medium | hard |
|---|---|---|---|
| ButtonUnmask | 12/26 = 46.2% | 1/12 = 8.3% | 0/12 = 0.0% |
| VideoUnmask | 12/26 = 46.2% | 2/12 = 16.7% | 2/12 = 16.7% |
| ButtonUnmaskSwap | 7/26 = 26.9% | 4/12 = 33.3% | 4/12 = 33.3% |
| VideoUnmaskSwap | 9/26 = 34.6% | 2/12 = 16.7% | 2/12 = 16.7% |
| **合计** | **40/104 = 38.5%** | **9/48 = 18.8%** | **8/48 = 16.7%** |

两点值得记录：

- 整体呈现预期的难度梯度（easy 38.5% → medium 18.8% → hard 16.7%），
  但梯度几乎全部来自 easy 与非 easy 之间，medium 与 hard 之间只差 2.1pp。
- **ButtonUnmaskSwap 是唯一不服从该梯度的任务**：easy 26.9% 反而低于它自己的
  medium / hard（均 33.3%）。样本量小（每档 12–26 个 episode），单档差异不足以下结论，
  仅登记现象，不在本轮追因。
- ButtonUnmask 的 hard 档 0/12 全败，是四任务十二个分档里唯一的零。

## 实测耗时与双卡并行收益

| | 任务 | 起讫 | 用时 | 单 episode |
|---|---|---|---|---|
| splitA（卡 0 / 8021） | ButtonUnmask + VideoUnmask | 14:53:36 → 15:12:43 | 19.1 min | 11.5 s |
| splitB（卡 1 / 8022） | ButtonUnmaskSwap + VideoUnmaskSwap | 14:53:43 → 15:17:58 | 24.3 min | 14.6 s |

- **总墙钟 24.4 分钟**（由较慢的 splitB 决定）；若串行跑同样 200 个 episode 需 43.4 分钟，
  双卡并行省下 **44%** 墙钟。
- 两侧负载不完全均衡（19.1 vs 24.3 min，差 27%）：起跑档里"每边一个 Button 类 + 一个
  Video 类"的均衡假设只部分成立——真正的耗时差异来自 Swap 变体（splitB 两个任务都是 Swap），
  不是 Button/Video 之分。下次若再做同类切分，按 Swap / 非 Swap 混搭比按 Button / Video 混搭更均衡。
- 显存实测：每卡 server 32282 MiB（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` 生效）
  + SAPIEN 仿真 738–743 MiB，合计约 33 GB / 46 GB，余量约 12 GB，全程未出现 OOM。

## 稳定性

- 零 `"error"` 条目，两侧 WebSocket 全程未断——README Q2 提到的"长 horizon 任务大 video 帧
  导致连接断开"在本轮 4 个任务上没有出现（该问题主要见于 VideoPlaceButton 等更长任务，本轮未评）。
- 无需任何续评：两侧都是一次跑通 100/100。

## 产物

```
v1-store/evaluation/v1-prod-60k/ckpt59999/seed7/     ← 合并后标准布局
├── progress.json    200 个 episode 的逐条结果
├── log.json         success_rate（四任务）+ total_success_rate
└── videos/          200 个 mp4（硬链接自两个 split，不占额外空间）
v1-store/evaluation/v1-prod-60k-splitA/ckpt59999/seed7/   ← 原始凭据，保留
v1-store/evaluation/v1-prod-60k-splitB/ckpt59999/seed7/   ← 原始凭据，保留
v1-store/logs/prod60k-eval-{server,client}-{a,b}.log
```

合并由一次性脚本完成（scratchpad，不进仓库）：合并两份 `progress.json`、视频硬链接汇总、
按 `True=1 / False 与 "error"=0` 重算 `log.json`；对"同一任务在两个 split 中重复出现"
与"合并后缺任务"两种切分错误 fail-loud。

未使用 `scripts/training/compute_results.py`：它的 `TASK_NAME_LIST` 硬编码全 16 任务、
缺失任务按 `success_rate.get(task_name, 0)` 补 0，只评 4 任务时其 `Overall` 会被 12 个 0
稀释成假数；单 seed 时 std 亦为 NaN。上表直接取自合并后的 `log.json`。

## 口径边界（读数时须知）

- 这是 **单 seed（policy seed 7）单 ckpt（59999）** 的一次测量，没有 seed 间方差，
  不构成 mean±std；与他人论文表格对照时须注意上游默认体例是 3 seed。
- 评的是 test split 固定的 50 个 episode/任务，环境 seed 由 metadata 写死，可复现。
- 只评了 v1-prod-60k 训练过的 4 个任务，不含 RoboMME 其余 12 个任务，
  因此本表**不是**完整 benchmark 分数，不能与全 16 任务的 Overall 直接比较。
