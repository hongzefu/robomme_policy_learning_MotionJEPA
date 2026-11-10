# 官方 FrameSamp+Context 512 续训 5k（b64）＋三 seed 评测：起跑档案

## 一、目的与用户原话

用官方发布的 FrameSamp+Context 512 权重（HF `Yinpei/mme_vla_suite` 目录 `perceptual-framesamp-context/79999`，论文 RoboMME v3 Table 15 里 StopCube 13.67、BinFill 41.22 那一行）继续训练 5000 步，再对续训最后一步做官方 test 800 局 × 推理 seed 42/7/0 三次评测。

用户原话（2026-11-10 美东）：
1. 「给出方案 继续训这个framesample context 官方512的ckpt然后 eval seed 42 / 7 / 0 继续5k bs128 训多久？」
2. 「bs64 bs128 都是5k 要训多久？」
3. 「用 batch size 六十四继续训练，训练完了之后，在这个呃做三次评测。」

方案里提出的 warmup 500 步，用户没有另外指定；按方案推荐值执行。

## 二、版本与代码状态

- 训练代码：fs8（`orig100-ctx-framesamp1024-b64-80k-8gpu`）起跑锚点 `a4293f4`，只读副本 `/scratch/hongze/worktrees/fs512c-train-a4293f4`（detached）。训练所用解释器与 fs8 相同：`v1-store/envs/fs8-train/bin/python`。
- 启动脚本与评测队列：分支 `v14-fs512-cont5k-1110`（基于 `f0149d8`，即 eval3x3 主表评测的代码），工作树 `/scratch/hongze/worktrees/fs512c-1110`。Beta commit 见 git log。
- 评测代码：同一工作树（f0149d8 的 `scripts/evaluation/eval_queue.py`，与主表 3ckpt×3seed 评测同一份实现）。

## 三、启动与配置还原

启动脚本 `scripts/training/run_fs512_official_cont5k.sh`。超参 = `git show a4293f4:src/mme_vla_suite/training/config.py` 的 `mme_vla_suite` 条目，加下列覆盖：

| 项 | 官方原配方 | 本次 |
|---|---|---|
| 初始权重 | pi05_base | 官方 79999 `params`（只有参数，无优化器状态；AdamW 从零起，EMA 从加载的权重起算） |
| history yaml | `perceptual-framesamp-context.yaml`（budget 512 = 32 帧 × 16 token） | 相同 |
| 训练步数 | 80000 | 5000 |
| warmup | 10000 | 500 |
| peak / decay lr | 5e-5 / 5e-5 | 相同（warmup 后常数） |
| batch | 64 | 64（8 卡 mesh 2×4，每卡 8） |
| norm stats | — | 官方 ckpt 自带 `79999/assets/robomme/norm_stats.json` |
| 存盘 | 每 10000 | 每 2500（存 2500、4999） |
| 数据 | 16 任务 × 100 原版 | `v1-store/datasets/16task-orig100-1600ep-9e109419/framesamp-4x4`（与 fs8 相同） |
| seed | 42 | 42 |

完整命令：

```bash
# 冒烟 20 步（不进 wandb）
CODE_ROOT=/scratch/hongze/worktrees/fs512c-train-a4293f4 \
  bash /scratch/hongze/worktrees/fs512c-1110/scripts/training/run_fs512_official_cont5k.sh smoke
# 正式（tmux fs5c-train）
tmux new-session -d -s fs5c-train "CODE_ROOT=/scratch/hongze/worktrees/fs512c-train-a4293f4 bash /scratch/hongze/worktrees/fs512c-1110/scripts/training/run_fs512_official_cont5k.sh prod"
# 评测（训练结束后；在执行副本根目录）
cd /scratch/hongze/worktrees/fs512c-1110
/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python scripts/evaluation/eval_queue.py --queue scripts/evaluation/fs512cont5k_1110_queue.json --smoke
tmux new-session -d -s fs5c-eval "... eval_queue.py --queue scripts/evaluation/fs512cont5k_1110_queue.json"
```

## 四、产物路径

- checkpoint：`v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp512-official79999-cont5k-b64/{2500,4999}`
- 训练日志：`v1-store/logs/orig100-ctx-framesamp512-official79999-cont5k-b64.log`（冒烟 `...-smoke.log`）
- 评测：`v1-store/evaluation/fs512cont5k-1110/`

## 五、起跑判据与盯盘项

- 冒烟：`EXIT_CODE=0`、`TRAIN_MESH shape=(2, 4)`、日志里 `Merging missing weight` 为 0 行（官方权重键全部命中，没有任何参数被随机初始化补上）、第 0 步 loss 量级接近收敛模型而非从头训练。
- 正式：Monitor 盯 `EXIT_CODE=|Traceback|Error|nan|NaN|out of memory|Killed|Saving checkpoint`。

## 六、资源与跨会话约定

- 8 × A100 全占：训练约 2 h，评测约 3 h。
- 跨会话 `req=fs512cont-11101815-1`：主表评测会话（3ckpt 3seed eval protocol）确认 eval3x3 队列已于美东 18:11 按用户指令暂停，用户恢复前不占 GPU；如用户要求恢复会先与本会话协调卡位。

## 七、本轮 tmux 会话清单

- `fs5c-train`（正式训练）
- `fs5c-eval`（评测队列调度）；评测队列自身起的 worker 会话前缀 `f5e-`
