#!/usr/bin/env bash
# siglip-ab-replay-40k：D-A1b 权重同源核对（纯 CPU）。在 worktree 内运行，复用主副本 .venv；一切路径绝对。
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
WT=$MAIN/.claude/worktrees/sgab
REC=$WT/docs/training-doc/siglip-ab-replay-40k/records
export UV_PROJECT_ENVIRONMENT=$MAIN/.venv UV_LINK_MODE=copy MMEVLA_V1_STORE=$MAIN/v1-store OPENPI_DATA_HOME=$MAIN/v1-store/models PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
cd "$WT"
echo "RUN_START=$(date '+%F %T') HEAD=c01e45d backend=cpu" > "$REC/weights.txt"
uv run --no-sync python scripts/training/g0/compare_siglip_weights.py \
  --ckpt "$MAIN/v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999/params" \
  --out "$REC/weights.json" 2>&1 | grep -v "dev-dependencies" | tee -a "$REC/weights.txt"
EC=${PIPESTATUS[0]}
echo "EXIT_CODE=$EC" | tee -a "$REC/weights.txt"
exit "$EC"
