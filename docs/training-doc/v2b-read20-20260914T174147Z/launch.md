# 新版四任务正式库的 20 步训练可读性检查

## 范围与当前状态

本检查属于用户要求实施的 `v2-4task-h5-merge-plan.md`，只验证新库能被现有训练链路读取并完成 20 次更新，不用于判断策略效果，也不进行 sim 评估。用户补充“继续工作 忽略huggingface的任务 但是要注意git”。当前尚未起跑，前置条件是正式 SigLIP、两档 framesamp 和新 norm_stats 全部通过。

前置建库现已全部通过。新 norm_stats SHA256 为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`；两档真实 CPU batch 检查均通过，记录见 `records/batch4.json`、`records/batch8.json`。state/actions 分别为 `(64,32)` / `(64,20,32)` float32 且有限，静态图像特征为 `(64,512,2048)` bfloat16，四个 motion 字段全为 None。输入检查使用 worker 0；下面的真实训练保持默认 worker 4。

初始化、JAX 编译、训练和 checkpoint 收尾可能超过五分钟，因此提前建档，实际运行使用独立 tmux `v2b-read20-20260914T174147Z`。启动时记录完整 HEAD、工作区状态、命令、norm_stats SHA256；本轮代码先提交，HF 在途文件按用户授权排除，实际消费代码仍须与实现提交一致。20 步是启动参数覆盖，未改全局训练默认值。

## 数据、模型与资源

数据集为 `v1-store/datasets/4task-v2-1600ep-604f16da/framesamp`，manifest SHA256 `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`：1600 集、1,192,918 步、605,611 个执行样本。源及构建记录见 [建库档案](../../dataset-build-doc/4task-v2-1600ep-604f16da/launch.md)。统计量必须来自 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`，不复用旧 400 集统计量。

模型使用现有 `mme_vla_suite` 配置和本地 pi05_base 权重，history 为 `perceptual-framesamp-context.yaml`，motion 关闭。默认 batch 64、worker 4、FSDP 4、seed 42 和学习率配置保持；`--num-train-steps 20` 与 `--log-interval 1` 仅用于本检查。CPU 参数预检已实际通过，确认上述默认值、目标 assets 路径、motion 关闭以及 overwrite/resume 均关闭。

环境 B：AWS 8×A100-SXM4-80GB，本地 `/dev/md0` XFS NVMe RAID。本检查仅可见 GPU 4–7，GPU 0–3 的既有 MotionJEPA 训练不受信号操作影响。缓存、指标、checkpoint 和日志均位于 v1-store；不覆盖 HOME。

## 启动命令

外层由 detached tmux 使用 `set -o pipefail`、`PYTHONUNBUFFERED=1`、`tee` 并记录 `EXIT_CODE=`。运行前核对目标 run 目录不存在。命令如下。

```bash
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/v2b-read20-20260914T174147Z"
export WANDB_DIR="$V1_STORE/logs/wandb"
export WANDB_CACHE_DIR="$V1_STORE/cache/wandb"
export WANDB_CONFIG_DIR="$V1_STORE/cache/wandb-config"
export WANDB_MODE=disabled
export TRAIN_RECORD_DIR="$V1_STORE/bench/v2b-read20-20260914T174147Z"
uv run --no-sync python scripts/training/train.py mme_vla_suite \
  --exp-name v2b-read20-20260914T174147Z \
  --num-train-steps 20 --log-interval 1 \
  --assets-base-dir "$V1_STORE/train-assets" \
  --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
  --data.assets.asset-id robomme \
  --checkpoint-base-dir "$V1_STORE/train-runs" \
  --dataset-path "$V1_STORE/datasets/4task-v2-1600ep-604f16da/framesamp" \
  --model.history-config perceptual-framesamp-context.yaml \
  --no-wandb-enabled
```

## 验收和清理

训练前通过真实 dataloader 取 batch，核对 shape/dtype 非空，以及 `motion_emb/motion_pos/motion_mask/mem_order` 全为 None。正式检查须完成 20 步、记录逐步 loss/梯度等标量且全部有限、正常完成 checkpoint 收尾并退出 0；记录统计量文件 SHA256 和实际运行时间。不用这 20 步声称学习有效，也不声称与旧数据训练轨迹等价。

指标与清洗日志写入 records，实际启动状态和结果补入 result.md。只在验证和归档后清理本轮确切 checkpoint 目录 `v1-store/train-runs/mme_vla_suite/v2b-read20-20260914T174147Z`，不清理其他 run，也不删除正式数据集或统计量。
