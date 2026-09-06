# 16task-h5-scan 起跑记录

## 目的

按**官方数据集口径**统计 RoboMME 全部 16 个任务的逐集 episode 长度与 demo 前缀，
用于把「运动记忆时序数轴」从 v1 的 4 个任务扩到 16 个任务。

此前那张图的长度是**在线 rollout 口径**（从 `eval-official-framesamp-context` 的 8 份 worker 日志
还原的执行步数）。另外 12 个任务从未跑过 eval，没有 rollout 数据，故换口径为
`Yinpei/robomme_data_h5` 的示范录制长度——纯离线读 H5，不开仿真器、不用 GPU。

## 环境与 commit

- 环境 B（AWS 单机，8×A100-80GB，仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`）
- 起跑 commit：`c704bf5c07a039c121918f81d6dc8311867b1ed0`（工作区 clean）
- 分支：`v2-motionmem`

## 数据来源

公开数据集 `Yinpei/robomme_data_h5`（16 个任务，全部为 `.tar.xz`，无裸 `.h5`）。
本机此前只有 v1 的 4 个任务（`paths.sh` 的 `TARGET_TASKS`），本轮补下另外 12 个：

| | |
|---|---|
| 需下 12 个（压缩） | 48.2 GB |
| 16 任务合计（压缩） | 56.4 GB |
| 解压后预计 | 约 565 GB（按已有 4 个实测 ~10× 压缩比） |
| 落点 | `/scratch/hongze/robomme_data_h5/`（`paths.sh` 的 `RAW_H5_DIR` 在 AWS 前缀下的默认值） |
| 保留 | `.tar.xz` 与解压件都保留，与 2026-09-04 下载 4 个时一致 |

**split 判定**：H5 是 **train split**，每任务 100 episode。判据是
`record_dataset_VideoUnmask.h5` 的 `episode_0/setup/seed = 6000`，与
`third_party/robomme_benchmark/src/robomme/env_metadata/train/record_dataset_VideoUnmask_metadata.json`
的 ep0 `seed=6000` 一致（test 是 560000、val 是 1060000）。
test/val 的 demo 前缀在任何静态元数据里都不存在（`env_metadata` 每条 record 只有
`task/episode/seed/difficulty`，无长度字段；demo 段由 reset 时的运动规划器现场生成），本轮不做。

## 命令

```bash
# 阶段 1 · 补下 12 个 H5（tmux 会话 h5dl16-all，日志 v1-store/logs/h5dl16.log）
PYTHONUNBUFFERED=1 HF_HOME=v1-store/cache/hf .venv/bin/python scripts/dataset/fetch_h5_16task.py \
  --raw-dir /scratch/hongze/robomme_data_h5 --jobs 6 \
  --manifest-out v1-store/datasets/16task-scan/input_manifest.json

# 阶段 2 · 只读扫描逐集 (num_timesteps, exec_start_idx)
.venv/bin/python scripts/dataset/scan_manifest.py build \
  --raw_dir /scratch/hongze/robomme_data_h5 \
  --out v1-store/datasets/16task-scan/episode_manifest.json \
  --tasks <16 任务 csv> --episodes-per-task 100

# 阶段 3 · 换算运动记忆指标
.venv/bin/python scripts/dataset/scan_16task_memory.py \
  --manifest v1-store/datasets/16task-scan/episode_manifest.json \
  --out v1-store/datasets/16task-scan/memory_axis_16task.json
```

本轮**全程只读扫描**，不跑 `build_dataset.py`、不带任何 `--force`（它会 `rmtree` 整个输出根）。

## tmux 会话清单（AGENTS 7）

- `h5dl16-all`（阶段 1）

## 三条验收对拍

1. **4 个已有任务逐条相同**：新扫描的 400 条 `(num_timesteps, exec_start_idx)` 必须与
   `v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json` 一致（`totals.timesteps = 123044`）。
2. **逐任务单集最大窗口数**吻合 `motion-memory-plan.md` 第 2.3 节（环境 A 全集实测）：
   VideoPlaceOrder 85、VideoPlaceButton 65、BinFill 64、PickXtimes 63、VideoRepick 61、RouteStick 40、
   PickHighlight 39、SwingXtimes 36、StopCube 35、InsertPeg 34、VideoUnmaskSwap 34、ButtonUnmaskSwap 33、
   PatternLock 32、ButtonUnmask 27、VideoUnmask 22。
3. **全集规模**：1600 episode、exec 样本合计 **476,857**。

对不上即说明公开集与环境 A 那份 H5 不同源，停止并交用户处置。

**阶段 3 脚本已先用 4 任务清单自证**（起跑前）：`timestep 123044`、
`窗口行 6832（demo 1125 + exec 5707）`、单集最大 token `27 / 33 / 22 / 34` —— 三项与既有留档逐个吻合。
