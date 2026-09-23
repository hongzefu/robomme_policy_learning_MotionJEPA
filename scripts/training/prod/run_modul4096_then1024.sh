#!/usr/bin/env bash
# 由本轮唯一detached tmux调用，整个队列遇到任何失败立即停止。
set -o pipefail
BATCH="${1:?需要批次名、TRAIN_HEAD、改前BASE}"
TRAIN_HEAD="${2:?}"
BEFORE_HEAD="${3:?}"
BASELINE_RECORDS="${4:-}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$MAIN" || exit 2
LOG="$MAIN/v1-store/logs/$BATCH.queue.log"
test ! -e "$LOG" || { printf '拒绝覆盖队列日志\n'; exit 2; }
body() (
  set -euo pipefail
  source scripts/training/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  unset PYTHONPATH JAX_PLATFORMS XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET
  ARGS=(--batch "$BATCH" --head "$TRAIN_HEAD" --before-head "$BEFORE_HEAD")
  if [ -n "$BASELINE_RECORDS" ]; then ARGS+=(--baseline-records "$BASELINE_RECORDS"); fi
  uv run --no-sync python scripts/training/prod/modul_budget_workflow.py "${ARGS[@]}"
)
body 2>&1 | tee "$LOG"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [ "${statuses[1]}" -ne 0 ]; then rc=${statuses[1]}; fi
printf 'EXIT_CODE=%s\n' "$rc" | tee -a "$LOG"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ] || [ "${ending[1]}" -ne 0 ]; then exit 1; fi
exit "$rc"
