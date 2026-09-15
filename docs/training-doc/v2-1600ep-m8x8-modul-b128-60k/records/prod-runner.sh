#!/usr/bin/env bash
set -o pipefail                     # 外层不开 -e：EXIT_CODE 必落盘

TRAIN_HEAD="${1:?用法: m8-prod-runner.sh <TRAIN_HEAD> <HC_SHA256>}"
HC_SHA="${2:?}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=v2-1600ep-m8x8-modul-b128-60k
LOG="$MAIN/v1-store/logs/$RUN.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
REC="$MAIN/v1-store/bench/$RUN"
mkdir -p "$(dirname "$LOG")" "$REC"

body() {
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志

  unset PYTHONPATH
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export TRAIN_RECORD_DIR="$REC"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"

  test "$(nvidia-smi --id=4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$(date -u +%FT%TZ)"

  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-8frame-8x8.yaml

  TRAIN_ARGS=(
    mme_vla_suite_b128_60k
    --exp-name "$RUN"
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR"
    --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8"
    --model.history-config "$HC"
  )

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" \
    --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN" \
    -- "${TRAIN_ARGS[@]}"

  # 全程 15 s GPU 采样：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上（否则 tee 不退出）
  nvidia-smi --id=4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
