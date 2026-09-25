#!/usr/bin/env bash
# 原版四卡三模式入口；同一参数数组先验收再训练，失败保留本轮现场。
set -o pipefail
if [ "$#" -ne 5 ]; then
  printf '用法: %s <prod|perf|smoke> <run_name> <gpus> <lib绝对路径> <assets_dir绝对路径>\n' "$0" >&2
  exit 2
fi
MODE=$1 RUN=$2 GPUS=$3 LIB=$4 ASSETS_DIR=$5
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
case "$MODE" in prod) STEPS=80000 ;; perf) STEPS=300 ;; smoke) STEPS=20 ;; *) printf '模式非法\n' >&2; exit 2 ;; esac
case "$RUN" in *[!a-zA-Z0-9_-]*|'') printf 'run 名非法\n' >&2; exit 2 ;; esac
: "${TRAIN_HEAD:?必须传入完整 TRAIN_HEAD 字面量}"
: "${HISTORY_CONFIG_SHA256:?必须传入 history 的期望 SHA256}"
: "${NORM_STATS_SHA256:?必须传入 norm_stats 的期望 SHA256}"
LOG="$MAIN/v1-store/logs/$RUN.driver.log"
REC="$MAIN/v1-store/bench/orig80k/$RUN"
RUN_ROOT="$MAIN/v1-store/train-runs/mme_vla_suite/$RUN"
[[ -d "$MAIN/v1-store/logs" && "$(realpath "$MAIN/v1-store/logs")" = "$MAIN/v1-store/logs" ]] || {
  printf '日志父目录必须是主副本内已存在的实体目录\n' >&2; exit 2;
}
# 独占保留新日志；不能靠先 test 后 tee 防止同名入口竞态覆盖。
( set -o noclobber; : > "$LOG" ) || { printf '拒绝覆盖日志 %s\n' "$LOG" >&2; exit 2; }

body() (
  set -euo pipefail
  cd "$MAIN"
  source scripts/training/paths.sh
  [[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]]
  [[ "$HISTORY_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ && "$NORM_STATS_SHA256" =~ ^[0-9a-f]{64}$ ]]
  test "$TRAIN_HEAD" = "$(git rev-parse HEAD)"
  test -z "$(git status --porcelain)"
  test ! -e "$REC" && test ! -L "$REC"
  test ! -e "$RUN_ROOT" && test ! -L "$RUN_ROOT"
  if [ "$MODE" = perf ]; then
    test -f scripts/training/tests/check_orig80k_speed.py || {
      printf '缺少本轮 perf 观测工具，禁止改为无观测训练\n' >&2; exit 2;
    }
  fi
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC CUDA_VISIBLE_DEVICES="$GPUS"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export JAX_COMPILATION_CACHE_DIR="$MMEVLA_JAX_CACHE_DIR"
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda/$RUN"
  export WANDB_DIR="$V1_STORE/logs/wandb/$RUN" WANDB_CACHE_DIR="$V1_STORE/cache/wandb/$RUN"
  export WANDB_CONFIG_DIR="$V1_STORE/cache/wandb-config/$RUN" WANDB_DATA_DIR="$V1_STORE/cache/wandb-data/$RUN"
  export XDG_DATA_HOME="$V1_STORE/cache/xdg-data/$RUN" WANDB_MODE=online
  export MMEVLA_FRAMESAMP_SOURCE="$LIB/source" MMEVLA_FRAMESAMP_MANIFEST="$LIB/meta/episode_manifest.json"
  export TRAIN_RECORD_DIR="$REC" TRAIN_FINAL_RECORD_DIR="$REC/final"
  export MMEVLA_EXPECTED_TRAIN_HEAD="$TRAIN_HEAD" ORIG80K_MODE="$MODE"
  unset PYTHONPATH PYTHONHOME JAX_PLATFORMS XLA_FLAGS TRAIN_TIMING_STEPS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET
  unset DTYPE_GRAD_DIR DTYPE_BATCH_FIXTURE_DIR BENCH_REF_SOURCE BENCH_REF_MANIFEST
  if [ -f "$V1_STORE/secrets/wandb.env" ]; then
    set -a
    source "$V1_STORE/secrets/wandb.env"
    set +a
  fi
  TRAIN_ARGS=(mme_vla_suite --exp-name "$RUN" --dataset-path "$LIB/framesamp"
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$ASSETS_DIR"
    --data.assets.asset-id robomme --checkpoint-base-dir "$V1_STORE/train-runs"
    --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params"
    --model.use-history --model.history-config perceptual-framesamp-modul.yaml)
  if [ "$MODE" != prod ]; then TRAIN_ARGS+=(--num-train-steps "$STEPS"); fi
  printf 'RUN=%s\nMODE=%s\nTRAIN_HEAD=%s\nSTART_UTC=%s\n' "$RUN" "$MODE" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
  # CPU 平台仅作用于这一条命令，不泄漏到真实训练。
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config perceptual-framesamp-modul.yaml --history-config-sha256 "$HISTORY_CONFIG_SHA256" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme --norm-stats-sha256 "$NORM_STATS_SHA256" \
    --dataset-path "$LIB/framesamp" --run-root "$RUN_ROOT" -- "${TRAIN_ARGS[@]}"
  uv run --no-sync python scripts/training/orig80k_contract.py check \
    --mode "$MODE" --run "$RUN" --head "$TRAIN_HEAD" --history-sha "$HISTORY_CONFIG_SHA256" \
    --norm-sha "$NORM_STATS_SHA256" --lib "$LIB" --assets "$ASSETS_DIR" --gpus "$GPUS" \
    --records "$REC" -- "${TRAIN_ARGS[@]}"
  mkdir -p "$MMEVLA_JAX_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$XDG_DATA_HOME"
  # 首尾各留一次真实采样，使 500ms 周期不会造成运行窗口端点缺口。
  nvidia-smi --id="$GPUS" --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits > "$REC/gpu.csv" 2> "$REC/gpu.csv.err"
  stdbuf -oL nvidia-smi --id="$GPUS" --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 >> "$REC/gpu.csv" 2>> "$REC/gpu.csv.err" &
  sampler_pid=$!
  printf 'GPU_SAMPLER_PID=%s\n' "$sampler_pid"
  cleanup_sampler() {
    local body_rc=$1 sampler_rc signal_rc reason final_rc running_jobs
    trap - EXIT
    set +e
    # 同时核对本 shell 的存活子任务，避免提前退出后的 PID 复用误伤其他进程。
    running_jobs="$(jobs -pr)"
    reason=exited_before_stop
    if [[ $'\n'"$running_jobs"$'\n' == *$'\n'"$sampler_pid"$'\n'* ]] && kill -0 "$sampler_pid" 2>/dev/null; then
      kill "$sampler_pid" 2>/dev/null
      signal_rc=$?
      if [ "$signal_rc" -eq 0 ]; then reason=driver_stop; fi
    fi
    wait "$sampler_pid"
    sampler_rc=$?
    printf 'GPU_SAMPLER_STOP pid=%s exit_code=%s reason=%s\n' "$sampler_pid" "$sampler_rc" "$reason"
    if [ "$reason" != driver_stop ] || { [ "$sampler_rc" -ne 0 ] && [ "$sampler_rc" -ne 143 ]; }; then
      if [ "$body_rc" -eq 0 ]; then body_rc=1; fi
    else
      # 流式进程已停止后才追加终点，避免两个写入者交错；这不能掩盖提前退出。
      nvidia-smi --id="$GPUS" --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >> "$REC/gpu.csv" 2>> "$REC/gpu.csv.err"
      final_rc=$?
      printf 'GPU_SAMPLER_FINAL_SAMPLE exit_code=%s\n' "$final_rc"
      if [ "$final_rc" -ne 0 ] || [ ! -s "$REC/gpu.csv" ] || [ -s "$REC/gpu.csv.err" ]; then
        if [ "$body_rc" -eq 0 ]; then body_rc=1; fi
      fi
    fi
    exit "$body_rc"
  }
  trap 'cleanup_sampler "$?"' EXIT
  if [ "$MODE" = perf ]; then
    uv run --no-sync python scripts/training/tests/check_orig80k_speed.py run --records "$REC" -- "${TRAIN_ARGS[@]}" &
  else
    uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}" &
  fi
  train_pid=$!
  printf 'TRAIN_WRAPPER_PID=%s\n' "$train_pid"
  if wait "$train_pid"; then train_rc=0; else train_rc=$?; fi
  printf 'TRAIN_WRAPPER_EXITED pid=%s exit_code=%s\n' "$train_pid" "$train_rc"
  exit "$train_rc"
)

body 2>&1 | tee "$LOG"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [ "${statuses[1]}" -ne 0 ]; then rc=${statuses[1]}; fi
printf 'TRAIN_PIPE_EXIT=%s\nTEE_EXIT=%s\nEND_UTC=%s\nEXIT_CODE=%s\n' \
  "${statuses[0]}" "${statuses[1]}" "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
ending=("${PIPESTATUS[@]}")
if [ "${ending[0]}" -ne 0 ] || [ "${ending[1]}" -ne 0 ]; then exit 1; fi
exit "$rc"
