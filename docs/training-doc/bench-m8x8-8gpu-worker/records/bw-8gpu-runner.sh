#!/usr/bin/env bash
# 8 卡 mesh (1,8)（--fsdp-devices 8 命令行覆盖，config 默认 4 不动）关闭态 worker 档位实测 runner（bench-m8x8-8gpu-w<N>）；照抄基线 smoke-runner 形制。
# 外层不开 -e：EXIT_CODE 在任何死法下都落盘
set -o pipefail

WORKERS="${1:?用法: bw-8gpu-runner.sh <WORKERS> <TRAIN_HEAD> <HC_SHA256>}"
TRAIN_HEAD="${2:?}"
HC_SHA="${3:?}"
FSDP="${4:-8}"          # fsdp_devices 命令行覆盖：8 ⇒ mesh (1,8)；4 ⇒ mesh (2,4)
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN="bench-m8x8-8gpu-f${FSDP}-w${WORKERS}"
LOG="$MAIN/v1-store/logs/bw-8gpu-f${FSDP}-w${WORKERS}.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
REC="$MAIN/v1-store/bench/$RUN"
mkdir -p "$(dirname "$LOG")" "$REC"

body() {
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"

  printf 'RUN=%s\nFSDP_DEVICES=%s\nWORKERS=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$FSDP" "$WORKERS" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$(date -u +%FT%TZ)"

  unset PYTHONPATH
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/bw-8gpu-f${FSDP}"      # 两档共用编译缓存；稳态窗口 it100 起，不含编译
  export TRAIN_RECORD_DIR="$REC"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  export WANDB_MODE=disabled
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"

  # 8 卡全空才起跑（preflight 不查硬件）
  used="$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)"
  echo "GPU_IDLE gpus=0-7 used_mib=$used"
  test "$used" = "0" || { echo "GPU_IDLE=FAIL"; return 3; }

  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-8frame-8x8.yaml

  # 唯一真源：preflight 校验的 argv 与 train.py 真收到的 argv 是同一个数组
  TRAIN_ARGS=(
    mme_vla_suite_b128_60k
    --exp-name "$RUN"
    --num-train-steps 300 --log-interval 10 --no-wandb-enabled
    --num-workers "$WORKERS" --fsdp-devices "$FSDP"
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
    -- "${TRAIN_ARGS[@]}" || return 4

  # 500 ms 密采（第 16 条）；stdout/stderr 全进文件，不留在 tee 管道上
  nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 \
    > "$REC/gpu_util_lms500.csv" 2> "$REC/gpu_util_lms500.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
