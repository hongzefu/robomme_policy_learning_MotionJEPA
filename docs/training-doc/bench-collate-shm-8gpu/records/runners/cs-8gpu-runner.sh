#!/usr/bin/env bash
# 两侧各用自己的源码与环境；外层保留失败退出记录。
set -o pipefail
SIDE=${1:?}
TRAIN_HEAD=${2:?}
HC_SHA=${3:?}
TOOLS=/scratch/hongze/robomme_policy_learning_MotionJEPA-temp
case "$SIDE" in
  old) MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA-base ;;
  new) MAIN=$TOOLS ;;
  *) exit 2 ;;
esac
STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store
RUN=bench-collate-shm-$SIDE
LOG="$STORE/logs/cs-8gpu-$SIDE.log"
REC="$STORE/bench/$RUN"

body() {
  cd "$MAIN"
  source scripts/training/paths.sh
  test ! -e "$REC"
  mkdir "$REC"
  unset PYTHONPATH JAX_PLATFORMS MMEVLA_MOTION_STORE UV_PROJECT_ENVIRONMENT XLA_FLAGS
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/cs-8gpu-$SIDE" TRAIN_RECORD_DIR="$REC"
  export WANDB_MODE=disabled CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source" MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-8frame-8x8.yaml
  used=$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nSTART_UTC=%s\nGPU_IDLE used_mib=%s\n' "$RUN" "$TRAIN_HEAD" "$MAIN" "$(date -u +%FT%TZ)" "$used"
  test "$used" = 0
  TRAIN_ARGS=(mme_vla_suite_b128_60k --exp-name "$RUN"
    --num-train-steps 300 --log-interval 10 --no-wandb-enabled --num-workers 16 --fsdp-devices 8
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$ASSETS_DIR" --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs" --dataset-path "$DS/framesamp-8x8" --model.history-config "$HC")
  JAX_PLATFORMS=cpu uv run --no-sync python "$TOOLS/scripts/training/preflight_train_launch.py" \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" --bench-copy --v1-store-realpath "$STORE" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN" -- "${TRAIN_ARGS[@]}"
  nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 > "$REC/gpu_util_lms500.csv" 2> "$REC/gpu_util_lms500.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
}

# tee 先打开日志，因此存在性检查在启动管道之前执行。
test ! -e "$LOG" || exit 2
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
