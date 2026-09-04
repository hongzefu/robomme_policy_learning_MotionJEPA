#!/usr/bin/env bash
# 复现官方 MME-VLA 训练（RoboMME/robomme_policy_learning @ 89efeaab）
# 档位：repro-4a100-fsdp4-40gsim  GPU=0,1,2,3  MEM_FRACTION=0.475
set -euo pipefail
REPO=/scratch/hongze/robomme_policy_learning_MotionJEPA
WT="$REPO/v1-store/worktrees/official-89efeaab"

EXP=repro-4a100-fsdp4-40gsim
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.475
export NCCL_P2P_DISABLE=1                     # 模拟对方无 NVLink/P2P
LOG="$REPO/v1-store/logs/${EXP}.log"

# jax 编译缓存重定向到 scratch（官方代码硬编码 ~/.cache/jax_<exp>；禁止覆盖 HOME）
mkdir -p "/scratch/hongze/.cache/jax_${EXP}" "$HOME/.cache"
ln -sfn "/scratch/hongze/.cache/jax_${EXP}" "$HOME/.cache/jax_${EXP}"

cd "$WT"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export XDG_CACHE_HOME=/scratch/hongze/.cache
export OPENPI_DATA_HOME="$REPO/v1-store/models"
export NCCL_IB_DISABLE=1                      # 与对方命令一致

uv run --no-sync scripts/train.py mme_vla_suite \
  --exp-name="$EXP" \
  --batch-size=64 \
  --num-train-steps=20 \
  --num-workers=4 \
  --fsdp-devices=4 \
  --dataset-path="$REPO/v1-store/datasets/4task-motion-400ep/source" \
  --assets-base-dir="$REPO/v1-store/train-assets" \
  --checkpoint-base-dir="$REPO/v1-store/train-runs" \
  --model.use-history \
  --model.history-config=perceptual-framesamp-modul.yaml \
  --no-wandb-enabled \
  --overwrite 2>&1 | tee "$LOG"
echo "EXIT_CODE=${PIPESTATUS[0]}" | tee -a "$LOG"
