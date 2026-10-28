#!/usr/bin/env bash
# 官方16×100单条入口：8卡单进程、B128、正式60k或同日程20步短测。
set -euo pipefail
[[ $# == 0 ]] || { echo '本入口不接受透传参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CONFIG=mme_vla_suite_orig16_b128_60k
HC=${HISTORY:?必须指定HISTORY}
RUN=${RUN_NAME:?必须指定RUN_NAME}
FINAL=${FINAL_RECORD_DIR:?必须指定FINAL_RECORD_DIR}
: "${TRAIN_HEAD:?必须指定完整TRAIN_HEAD}"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo 'TRAIN_HEAD必须为40位SHA' >&2; exit 2; }
[[ "$RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'RUN_NAME非法' >&2; exit 2; }
[[ "$HC" != */* && "$HC" == *-orig16.yaml ]] || { echo 'HISTORY必须是新配置文件名' >&2; exit 2; }
[[ "$FINAL" == /* && "$(basename "$FINAL")" == final-record ]] || { echo 'FINAL_RECORD_DIR必须是绝对final-record路径' >&2; exit 2; }
[[ ${DRY_RUN:-0} == 0 || ${DRY_RUN:-0} == 1 ]] || { echo 'DRY_RUN仅允许0或1' >&2; exit 2; }
[[ ${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7} == 0,1,2,3,4,5,6,7 ]] || { echo 'GPU_SCOPE=FAIL allowed=0-7' >&2; exit 2; }
STEPS=60000
STEP_ARGS=()
if [[ -n ${MAX_STEPS+x} ]]; then
  [[ "$MAX_STEPS" == 20 && "$RUN" == o16-smoke-* ]] || { echo 'MAX_STEPS仅允许20且须o16-smoke-名称' >&2; exit 2; }
  STEPS=20
  STEP_ARGS=(--num-train-steps 20)
fi
# 清除会改变数据、模型环境与放宽来源检查的外部覆盖。
unset PYTHONPATH PYTHONHOME XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET MMEVLA_FRAMESAMP_VERIFY \
  TRAIN_TIMING_STEPS WANDB_MODE JAX_COMPILATION_CACHE_DIR MMEVLA_MOTION_OVERFLOW
for key in $(compgen -e); do
  case "$key" in *_ALLOW_UNVERIFIED) unset "$key" ;; esac
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENPI_DATA_HOME="$ROOT/v1-store/models"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XDG_CACHE_HOME="$ROOT/v1-store/cache/xdg" HF_HOME="$ROOT/v1-store/cache/hf"
export WANDB_DIR="$ROOT/v1-store/logs/wandb" WANDB_CACHE_DIR="$ROOT/v1-store/cache/wandb"
export WANDB_CONFIG_DIR="$ROOT/v1-store/cache/wandb-config"
LIBRARY=$("$PY" scripts/training/check_orig16_stage2_launch.py --print-library)
LIB="$ROOT/v1-store/datasets/$LIBRARY"
case "$HC" in
  *-32frame-4x4-*) FRAME=framesamp-4x4 ;;
  *-8frame-8x8-*) FRAME=framesamp-8x8 ;;
  *) echo 'HISTORY帧布局非法' >&2; exit 2 ;;
esac
export MMEVLA_FRAMESAMP_SOURCE="$LIB/source" MMEVLA_FRAMESAMP_MANIFEST="$LIB/meta/episode_manifest.json"
export TRAIN_FINAL_RECORD_DIR="$FINAL" TRAIN_RECORD_DIR="$(dirname "$FINAL")/train-record"
export MMEVLA_EXPECTED_TRAIN_HEAD="$TRAIN_HEAD" MMEVLA_V1_STORE="$ROOT/v1-store"
export MMEVLA_JAX_CACHE_DIR="$ROOT/v1-store/cache/jax/$RUN" CUDA_CACHE_PATH="$ROOT/v1-store/cache/cuda/$RUN"
# 沿用旧入口：存在本地W&B环境文件则启用，否则关闭；不把凭据写进命令或日志。
WANDB_ARGS=()
if [[ -f "$ROOT/v1-store/secrets/wandb.env" ]]; then
  set -a
  source "$ROOT/v1-store/secrets/wandb.env"
  set +a
else
  WANDB_ARGS=(--no-wandb-enabled)
fi
ARGS=("$CONFIG" --exp-name "$RUN" --dataset-path "$LIB/$FRAME"
  --assets-base-dir "$ROOT/v1-store/train-assets"
  --data.assets.assets-dir "$ROOT/v1-store/train-assets/mme_vla_suite/$LIBRARY" --data.assets.asset-id robomme
  --checkpoint-base-dir "$ROOT/v1-store/train-runs"
  --weight-loader.params-path "$ROOT/v1-store/models/openpi-assets/checkpoints/pi05_base/params"
  --model.use-history --model.history-config "$HC" "${STEP_ARGS[@]}" "${WANDB_ARGS[@]}")
print_command() {
  printf 'TRAIN_CMD='
  printf '%q ' "$PY" scripts/training/train.py "${ARGS[@]}"
  printf '\n'
}
contract() {
  JAX_PLATFORMS=cpu "$PY" scripts/training/check_orig16_stage2_launch.py \
    --repo "$ROOT" --store-root "$ROOT" --train-head "$TRAIN_HEAD" --steps "$STEPS" -- "${ARGS[@]}"
}
# 只读分支在日志、run和缓存任何mkdir/open之前返回。
if [[ ${DRY_RUN:-0} == 1 ]]; then
  print_command
  contract
  exit $?
fi
LOG="$ROOT/v1-store/logs/$RUN.log"
[[ -d "$ROOT/v1-store/logs" && "$(realpath "$ROOT/v1-store/logs")" == "$ROOT/v1-store/logs" ]] || { echo '日志父目录必须预先存在且为实体目录' >&2; exit 2; }
( set -o noclobber; : > "$LOG" ) || { echo "拒绝覆盖日志: $LOG" >&2; exit 2; }
body() (
  set -euo pipefail
  printf 'RUN=%s\nSTEPS=%s\nTRAIN_HEAD=%s\nSTART_UTC=%s\n' "$RUN" "$STEPS" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
  print_command
  contract
  mkdir -p "$MMEVLA_JAX_CACHE_DIR" "$CUDA_CACHE_PATH"
  unset JAX_PLATFORMS
  exec "$PY" scripts/training/train.py "${ARGS[@]}"
)
set +e
body 2>&1 | tee -a "$LOG"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [[ ${statuses[1]} != 0 && $rc == 0 ]]; then rc=${statuses[1]}; fi
# 用户要求运行后诊断全部执行并如实记录；诊断失败不改训练程序退出码。
if rg -qi '^\r*Step [0-9]+:.*(loss|grad_norm|llm_grad_norm|param_norm|mem_enc_norm)=[-+]?(nan|inf)(,|[[:space:]]|$)|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError' "$LOG"; then
  printf 'TRAIN_NUMERICS=FAIL reason=nonfinite_or_oom\n' >> "$LOG"
  printf 'TRAIN_DIAGNOSTIC=FAIL check=train_numerics action=record_only reason=user_decision\n' >> "$LOG"
fi
if ! JAX_PLATFORMS=cpu "$PY" scripts/training/check_orig16_stage2_launch.py \
    --check-final-record "$FINAL/final.json" --run "$RUN" --steps "$STEPS" --train-head "$TRAIN_HEAD" >> "$LOG" 2>&1; then
  printf 'TRAIN_DIAGNOSTIC=FAIL check=final_record action=record_only reason=user_decision\n' >> "$LOG"
fi
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG" || rc=1
exit "$rc"
