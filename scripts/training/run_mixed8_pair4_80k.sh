#!/usr/bin/env bash
# 四对 modulation 正式入口（mme_vla_suite_b128_80k，单 JAX 进程 8 卡 FSDP8，B128/W16）。
# S7 正式 80k 不传 MAX_STEPS；S6 同配置短测只允许 RUN_NAME / MAX_STEPS / FINAL_RECORD_DIR 不同。
# 用法（全部经环境变量，不接受任何额外参数，杜绝 resume/overwrite/resum_ckpt_id 透传）：
#   RUN_NAME=<全新run名> TRAIN_HEAD=<40位sha字面量> \
#   FINAL_RECORD_DIR=<主检出v1-store内绝对路径>/final-record [MAX_STEPS=<S6短测步数>] \
#   bash scripts/training/run_mixed8_pair4_80k.sh
# 先跑专用启动合同检查（真实 config.cli() 解析同一参数数组），PASS 后才 exec 训练。
set -o pipefail
if [ "$#" -ne 0 ]; then
  printf '本入口不接受额外参数（禁止 resume/overwrite/resum_ckpt_id 等透传）: %s\n' "$*" >&2
  exit 2
fi
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-pair4-v10e60-mixed8.yaml
PARENT="$MAIN/v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f"
ASSETS_DIR="$MAIN/v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f"
RUN="${RUN_NAME:?必须给出全新 RUN_NAME}"
FINAL="${FINAL_RECORD_DIR:?必须给出 FINAL_RECORD_DIR（绝对路径，以 /final-record 结尾）}"
: "${TRAIN_HEAD:?必须传入完整 TRAIN_HEAD 字面量}"
case "$RUN" in *[!a-zA-Z0-9_.-]*|'') printf 'run 名非法: %s\n' "$RUN" >&2; exit 2 ;; esac
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || { printf 'TRAIN_HEAD 须为40位sha\n' >&2; exit 2; }
[[ "$FINAL" = /* && "$(basename "$FINAL")" = final-record ]] || {
  printf 'FINAL_RECORD_DIR 须为绝对路径且以 /final-record 结尾: %s\n' "$FINAL" >&2; exit 2; }
STEPS=80000
STEP_ARGS=()
if [[ -n "${MAX_STEPS+x}" ]]; then
  [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] && [ "$MAX_STEPS" -lt 80000 ] || {
    printf 'MAX_STEPS 仅供 S6 短测，须为 1..79999 的整数: %s\n' "$MAX_STEPS" >&2; exit 2; }
  STEPS=$MAX_STEPS
  STEP_ARGS=(--num-train-steps "$MAX_STEPS")
fi
LOG="$MAIN/v1-store/logs/$RUN.log"
[[ -d "$MAIN/v1-store/logs" && "$(realpath "$MAIN/v1-store/logs")" = "$MAIN/v1-store/logs" ]] || {
  printf '日志父目录必须是主检出内已存在的实体目录\n' >&2; exit 2; }
( set -o noclobber; : > "$LOG" ) || { printf '拒绝覆盖日志 %s\n' "$LOG" >&2; exit 2; }

body() (
  set -euo pipefail
  cd "$MAIN"                           # history YAML 与 motion store_path 均按主检出解析
  source scripts/training/paths.sh     # 固定 OPENPI_DATA_HOME=v1-store/models，缓存落 v1-store/cache
  test "$OPENPI_DATA_HOME" = "$MAIN/v1-store/models"
  # 清除一切可能替换数据或放宽校验的残留覆盖。
  unset PYTHONPATH PYTHONHOME JAX_PLATFORMS XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET \
    MMEVLA_FRAMESAMP_VERIFY TRAIN_TIMING_STEPS WANDB_MODE
  for name in $(compgen -e); do
    case "$name" in *_ALLOW_UNVERIFIED) unset "$name" ;; esac
  done
  export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 TZ=UTC
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN" CUDA_CACHE_PATH="$V1_STORE/cache/cuda/$RUN"
  export MMEVLA_FRAMESAMP_SOURCE="$PARENT/source" MMEVLA_FRAMESAMP_MANIFEST="$PARENT/meta/episode_manifest.json"
  export TRAIN_FINAL_RECORD_DIR="$FINAL" TRAIN_RECORD_DIR="$(dirname "$FINAL")/train-record"
  export MMEVLA_EXPECTED_TRAIN_HEAD="$TRAIN_HEAD"
  WANDB_ARGS=()
  if [ -f "$V1_STORE/secrets/wandb.env" ]; then
    set -a; . "$V1_STORE/secrets/wandb.env"; set +a
  else
    echo "提示: 没有 v1-store/secrets/wandb.env，本轮关闭 wandb"
    WANDB_ARGS=(--no-wandb-enabled)
  fi
  TRAIN_ARGS=("$CONFIG" --exp-name "$RUN"
    --dataset-path "$PARENT/framesamp-8x8"
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR" --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params"
    --model.use-history --model.history-config "$HC"
    "${STEP_ARGS[@]}" "${WANDB_ARGS[@]}")
  printf 'RUN=%s\nSTEPS=%s\nTRAIN_HEAD=%s\nFINAL_RECORD_DIR=%s\nSTART_UTC=%s\n' \
    "$RUN" "$STEPS" "$TRAIN_HEAD" "$FINAL" "$(date -u +%FT%TZ)"
  printf 'TRAIN_CMD='; printf '%q ' uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"; printf '\n'
  # CPU 平台只作用于检查器这一条命令，不泄漏到真实训练。
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/check_mixed8_pair4_launch.py \
    --repo "$MAIN" --store-root "$MAIN" --train-head "$TRAIN_HEAD" --steps "$STEPS" -- "${TRAIN_ARGS[@]}"
  mkdir -p "$MMEVLA_JAX_CACHE_DIR" "$CUDA_CACHE_PATH"
  # 单个 Python/JAX 进程暴露 8 卡（mesh (1,8)），不用 torchrun。
  exec uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
)

body 2>&1 | tee -a "$LOG"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [ "${statuses[1]}" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=${statuses[1]}; fi
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG"
exit "$rc"
