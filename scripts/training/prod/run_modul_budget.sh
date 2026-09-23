#!/usr/bin/env bash
# 新预算独立入口：同一TRAIN_ARGS先验收再交训练，任何失败均保留现场。
set -o pipefail
MODE="${1:?需要smoke/perf/prod、预算、run、HEAD、YAML摘要、批准记录、测速报告、配置基线}"
BUDGET="${2:?}"
RUN="${3:?}"
TRAIN_HEAD="${4:?}"
HC_SHA="${5:?}"
APPROVAL="${6:-}"
REPORT="${7:-}"
BASELINE="${8:?}"
HANDOFF="${9:-}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
LOG="$MAIN/v1-store/logs/$RUN.driver.log"
REC="$MAIN/v1-store/bench/modul-budget-sweep/runs/$RUN"
RUNNER="$(realpath "${BASH_SOURCE[0]}")"
test ! -e "$LOG" || { printf '拒绝覆盖日志 %s\n' "$LOG"; exit 2; }

body() (
  set -euo pipefail
  cd "$MAIN"
  source scripts/training/paths.sh
  test "$TRAIN_HEAD" = "$(git rev-parse HEAD)"
  test -z "$(git status --porcelain)"
  test ! -e "$REC"
  case "$BUDGET" in 1024|2048|4096) ;; *) printf '预算非法\n'; exit 2 ;; esac
  case "$RUN" in *[!a-zA-Z0-9_-]*|'') printf 'run名非法\n'; exit 2 ;; esac
  DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC="perceptual-framesamp-modul-$((BUDGET/64))frame-8x8.yaml"
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TZ=UTC
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source" MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  export TRAIN_RECORD_DIR="$REC" TRAIN_FINAL_RECORD_DIR="$REC/final" TRAIN_TIMING_STEPS=0
  export MMEVLA_EXPECTED_TRAIN_HEAD="$TRAIN_HEAD"
  unset PYTHONPATH JAX_PLATFORMS XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET
  TRAIN_ARGS=(mme_vla_suite_b128_80k --exp-name "$RUN"
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR" --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8" --model.history-config "$HC")
  PREFLIGHT_EXTRA=()
  case "$MODE" in
    smoke) TRAIN_ARGS+=(--num-train-steps 20 --log-interval 1 --no-wandb-enabled); export WANDB_MODE=disabled ;;
    perf) TRAIN_ARGS+=(--num-train-steps 1000 --no-wandb-enabled); export WANDB_MODE=disabled ;;
    prod)
      test -n "$APPROVAL"; test -n "$REPORT"
      if [ "$BUDGET" = 1024 ]; then
        test -n "$HANDOFF"
        JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/check_modul_completion.py \
          --verify-handoff "$HANDOFF" --head "$TRAIN_HEAD"
      fi
      set -a; source "$V1_STORE/secrets/wandb.env"; set +a
      PREFLIGHT_EXTRA=(--approval-record "$APPROVAL" --runner "$RUNNER" --report "$REPORT")
      unset WANDB_MODE
      ;;
    *) printf '不支持模式 %s\n' "$MODE"; exit 2 ;;
  esac
  printf 'RUN=%s\nMODE=%s\nBUDGET=%s\nTRAIN_HEAD=%s\nSTART_UTC=%s\nRUNNER_SHA=' "$RUN" "$MODE" "$BUDGET" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
  sha256sum "$RUNNER"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_80k/$RUN" \
    --launch-mode "$MODE" --expected-run-name "$RUN" --config-baseline "$BASELINE" \
    "${PREFLIGHT_EXTRA[@]}" -- "${TRAIN_ARGS[@]}"
  uv run --no-sync python - <<'PY'
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path("scripts/training").resolve()))
from modul_launch_contract import runtime_environment
print("RUN_ENV="+json.dumps(runtime_environment(pathlib.Path.cwd()),sort_keys=True),flush=True)
PY
  GPU_CSV="$MAIN/v1-store/bench/modul-budget-sweep/runs/$RUN.gpu.csv"
  test ! -e "$GPU_CSV"
  mkdir -p "$(dirname "$GPU_CSV")"
  stdbuf -oL nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 > "$GPU_CSV" 2> "$GPU_CSV.err" &
  sampler_pid=$!
  printf 'GPU_SAMPLER_PID=%s\n' "$sampler_pid"
  cleanup_sampler() {
    kill "$sampler_pid" 2>/dev/null || true
    set +e
    wait "$sampler_pid"
    printf 'GPU_SAMPLER_STOP pid=%s exit_code=%s reason=本轮训练结束\n' "$sampler_pid" "$?"
  }
  trap cleanup_sampler EXIT
  if [ "$MODE" = prod ]; then
    uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}" &
  else
    uv run --no-sync python scripts/training/tests/check_modul_speed.py run --mode "$MODE" --records "$REC" -- "${TRAIN_ARGS[@]}" &
  fi
  train_pid=$!
  printf 'TRAIN_WRAPPER_PID=%s\n' "$train_pid"
  wait "$train_pid"
  printf 'TRAIN_WRAPPER_EXITED pid=%s\n' "$train_pid"
)

body 2>&1 | tee "$LOG"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [ "${statuses[1]}" -ne 0 ]; then rc=${statuses[1]}; fi
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ] || [ "${ending[1]}" -ne 0 ]; then exit 1; fi
exit "$rc"
