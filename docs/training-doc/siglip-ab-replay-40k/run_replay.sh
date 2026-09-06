#!/usr/bin/env bash
# siglip-ab-replay-40k：单 episode 开环重放（GPU 0）。在 worktree 内运行，复用主副本 .venv；一切路径绝对。
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
WT=$MAIN/.claude/worktrees/sgab
REC=$WT/docs/training-doc/siglip-ab-replay-40k/records
export UV_PROJECT_ENVIRONMENT=$MAIN/.venv UV_LINK_MODE=copy MMEVLA_V1_STORE=$MAIN/v1-store OPENPI_DATA_HOME=$MAIN/v1-store/models PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${SG_GPU:-0} XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
EPISODES=${SG_EPISODES:-VideoUnmask:0}
OUT=${SG_OUT:-$REC/replay}
LOG=${SG_LOG:-$REC/replay.txt}
cd "$WT"
mkdir -p "$OUT"
echo "RUN_START=$(date '+%F %T') HEAD=$(git rev-parse --short HEAD) gpu=$CUDA_VISIBLE_DEVICES episodes=$EPISODES" > "$LOG"
uv run --no-sync python scripts/training/g0/compare_siglip_replay.py \
  --lib "$MAIN/v1-store/datasets/4task-motion-400ep" \
  --ckpt "$MAIN/v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999" \
  --config mme_vla_suite --episodes "$EPISODES" --noise-seeds 0,1,2 --out "$OUT" 2>&1 | grep -v "dev-dependencies" | tee -a "$LOG"
EC=${PIPESTATUS[0]}
echo "EXIT_CODE=$EC" | tee -a "$LOG"
exit "$EC"
