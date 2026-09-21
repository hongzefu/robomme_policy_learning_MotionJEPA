#!/usr/bin/env bash
# 外层保留失败退出码；内层全部闸门成功才会调用真实训练。
set -o pipefail
MODE="${1:?用法: run_modul2048.sh <smoke|perf|prod> <RUN> <TRAIN_HEAD> <HC_SHA> [APPROVAL_JSON] [REPORT_JSON]}"
RUN="${2:?}"
TRAIN_HEAD="${3:?}"
HC_SHA="${4:?}"
APPROVAL="${5:-}"
REPORT="${6:-}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
LOG="$MAIN/v1-store/logs/$RUN.driver.log"
REC="$MAIN/v1-store/bench/m2048/$RUN"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
RUNNER="$(realpath "${BASH_SOURCE[0]}")"
test ! -e "$LOG" || { printf '拒绝覆盖日志 %s\n' "$LOG"; exit 2; }

body() (
  set -euo pipefail
  cd "$MAIN"
  source scripts/training/paths.sh
  test "$TRAIN_HEAD" = "$(git rev-parse HEAD)"
  test -z "$(git status --porcelain)"
  test ! -e "$REC"
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TZ=UTC
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source" MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  export TRAIN_RECORD_DIR="$REC" TRAIN_TIMING_STEPS=0
  export MMEVLA_EXPECTED_TRAIN_HEAD="$TRAIN_HEAD"
  unset PYTHONPATH JAX_PLATFORMS XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET
  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-32frame-8x8.yaml
  TRAIN_ARGS=(
    mme_vla_suite_b128_80k --exp-name "$RUN"
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR" --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8" --model.history-config "$HC"
  )
  PREFLIGHT_EXTRA=()
  case "$MODE" in
    smoke) TRAIN_ARGS+=(--num-train-steps 20 --log-interval 1 --no-wandb-enabled); export WANDB_MODE=disabled ;;
    perf) TRAIN_ARGS+=(--num-train-steps 1000 --no-wandb-enabled); export WANDB_MODE=disabled ;;
    prod)
      test -n "$APPROVAL"; test -n "$REPORT"
      set -a; source "$V1_STORE/secrets/wandb.env"; set +a
      PREFLIGHT_EXTRA=(--approval-record "$APPROVAL" --runner "$RUNNER" --report "$REPORT")
      unset WANDB_MODE
      ;;
    *) printf '不支持的运行模式 %s\n' "$MODE"; exit 2 ;;
  esac
  printf 'RUN=%s\nMODE=%s\nTRAIN_HEAD=%s\nSTART_UTC=%s\nRUNNER_SHA=' "$RUN" "$MODE" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
  sha256sum "$RUNNER"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_80k/$RUN" \
    --launch-mode "$MODE" --expected-run-name "$RUN" "${PREFLIGHT_EXTRA[@]}" -- "${TRAIN_ARGS[@]}"
  uv run --no-sync python - <<'PY'
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path("scripts/training").resolve()))
from modul_launch_contract import runtime_environment
print("RUN_ENV="+json.dumps(runtime_environment(pathlib.Path.cwd()),sort_keys=True),flush=True)
PY
  # 采样器只属于本轮，记录精确PID，正常与失败路径都回收。
  GPU_CSV="$V1_STORE/bench/m2048/$RUN.gpu.csv"
  test ! -e "$GPU_CSV"
  stdbuf -oL nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 > "$GPU_CSV" 2> "$GPU_CSV.err" &
  sampler_pid=$!
  printf 'GPU_SAMPLER_PID=%s\n' "$sampler_pid"
  cleanup_sampler() {
    kill "$sampler_pid" 2>/dev/null || true
    set +e
    wait "$sampler_pid"
    sampler_rc=$?
    printf 'GPU_SAMPLER_STOP pid=%s exit_code=%s reason=本轮训练结束\n' "$sampler_pid" "$sampler_rc"
  }
  trap cleanup_sampler EXIT
  if [ "$MODE" = prod ]; then
    uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
  else
    uv run --no-sync python scripts/training/tests/check_modul_speed.py run --mode "$MODE" --records "$REC" -- "${TRAIN_ARGS[@]}"
  fi
)

body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
