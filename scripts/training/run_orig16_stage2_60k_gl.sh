#!/usr/bin/env bash
# GreatLakes 复现入口（1031 计划 S1）：仿 run_orig16_stage2_60k.sh，只放开卡数（2/4）、global batch 与
# GPU 可见范围三项资源量；正式口径 batch 只认 64/32，冒烟 o16gl-smoke- 与测时 o16gl-tier- 前缀按组合表放行。
# 训练语义（lr、warmup、EMA、步数、seed、配置条目）与原入口相同；原入口不动。
set -euo pipefail
[[ $# == 0 ]] || { echo '本入口不接受透传参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
# 路径与缓存一律按仓库前缀取 v1-store/（paths.sh 设 OPENPI_DATA_HOME、XDG_CACHE_HOME、HF_HOME、TRAIN_CACHE_ROOT，并只读定义 PY）。
source "$ROOT/scripts/training/paths.sh"
CONFIG=mme_vla_suite_orig16_b128_60k
GL_HC=perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml
HC=${HISTORY:?必须指定HISTORY}
RUN=${RUN_NAME:?必须指定RUN_NAME}
FINAL=${FINAL_RECORD_DIR:?必须指定FINAL_RECORD_DIR}
N=${GPUS:?必须指定GPUS}
B=${GLOBAL_BATCH:?必须指定GLOBAL_BATCH}
: "${TRAIN_HEAD:?必须指定完整TRAIN_HEAD}"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo 'TRAIN_HEAD必须为40位SHA' >&2; exit 2; }
[[ "$RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'RUN_NAME非法' >&2; exit 2; }
[[ "$HC" != */* && "$HC" == *-orig16.yaml ]] || { echo 'HISTORY必须是新配置文件名' >&2; exit 2; }
[[ "$FINAL" == /* && "$(basename "$FINAL")" == final-record ]] || { echo 'FINAL_RECORD_DIR必须是绝对final-record路径' >&2; exit 2; }
[[ ${DRY_RUN:-0} == 0 || ${DRY_RUN:-0} == 1 ]] || { echo 'DRY_RUN仅允许0或1' >&2; exit 2; }
[[ "$N" == 2 || "$N" == 4 ]] || { echo 'GPUS仅允许2或4' >&2; exit 2; }
# 运行类别 × 步数 × batch × history 组合表（与 check_orig16_stage2_launch.py::GL_COMBOS 同表）。
STEP_ARGS=()
TIMING=${TRAIN_TIMING_STEPS-}
case "$RUN" in
  o16gl-tier-*)
    [[ ${MAX_STEPS-} == 100 && "$B" == 64 ]] || { echo 'o16gl-tier-仅允许MAX_STEPS=100且GLOBAL_BATCH=64' >&2; exit 2; }
    case "$HC" in
      perceptual-framesamp-ctx-16frame-4x4-nomotion-tier-orig16.yaml|perceptual-framesamp-ctx-32frame-4x4-nomotion-tier-orig16.yaml|perceptual-framesamp-ctx-64frame-4x4-nomotion-tier-orig16.yaml) ;;
      *) echo 'o16gl-tier-只认三份测时yaml' >&2; exit 2 ;;
    esac
    STEPS=100 ;;
  o16gl-smoke-*)
    [[ "$HC" == "$GL_HC" ]] || { echo "o16gl-smoke-只认$GL_HC" >&2; exit 2; }
    case "${MAX_STEPS-}:$B" in
      20:32|20:16|20:8|20:4|100:64|100:32) STEPS=$MAX_STEPS ;;
      *) echo 'o16gl-smoke-仅允许20步32/16/8/4或100步64/32' >&2; exit 2 ;;
    esac ;;
  *)
    [[ -z ${MAX_STEPS+x} ]] || { echo '正式口径不得设MAX_STEPS' >&2; exit 2; }
    [[ -z "$TIMING" ]] || { echo 'TRAIN_TIMING_STEPS只在o16gl-smoke-/o16gl-tier-放行' >&2; exit 2; }
    [[ "$B" == 64 || "$B" == 32 ]] || { echo '正式GLOBAL_BATCH仅允许64或32' >&2; exit 2; }
    [[ "$HC" == "$GL_HC" ]] || { echo "正式只认$GL_HC" >&2; exit 2; }
    STEPS=60000 ;;
esac
if [[ $STEPS != 60000 ]]; then STEP_ARGS=(--num-train-steps "$STEPS"); fi
if [[ -n "$TIMING" ]]; then
  [[ "$TIMING" =~ ^[1-9][0-9]*$ && "$TIMING" -le "$STEPS" ]] || { echo 'TRAIN_TIMING_STEPS须为1..步数的整数' >&2; exit 2; }
fi
# 证据重定根：正式与冒烟必须设为实体目录；测时 yaml 无 motion 段，不设（契约对 tier 跳过证据核对）。
EVIDENCE=${MMEVLA_STAGE2_EVIDENCE_ROOT-}
if [[ "$RUN" == o16gl-tier-* ]]; then
  [[ -z "$EVIDENCE" ]] || echo 'TIER: 忽略MMEVLA_STAGE2_EVIDENCE_ROOT（无motion）' >&2
  EVIDENCE=
else
  [[ "$EVIDENCE" == /* && -d "$EVIDENCE" ]] || { echo 'MMEVLA_STAGE2_EVIDENCE_ROOT必须是已存在的绝对目录' >&2; exit 2; }
fi
# GPU 可见范围：Slurm 下不设 CUDA_VISIBLE_DEVICES，按 nvidia-smi -L 数量核；否则必须恰有 GPUS 张。
if [[ -n ${SLURM_JOB_ID-} ]]; then
  seen=$(nvidia-smi -L | grep -c '^GPU ' || true)
  [[ "$seen" == "$N" ]] || { echo "GPU_SCOPE=FAIL slurm visible=$seen want=$N" >&2; exit 2; }
  VISIBLE=${CUDA_VISIBLE_DEVICES-}
else
  VISIBLE=${CUDA_VISIBLE_DEVICES-}
  IFS=, read -r -a ids <<< "$VISIBLE"
  [[ -n "$VISIBLE" && ${#ids[@]} == "$N" ]] || { echo "GPU_SCOPE=FAIL CUDA_VISIBLE_DEVICES须恰有${N}张" >&2; exit 2; }
fi
# 清除会改变数据、模型环境与放宽来源检查的外部覆盖（逐行同原入口）；计时与证据根在核验后按上面结果恢复。
unset PYTHONPATH PYTHONHOME XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET MMEVLA_FRAMESAMP_VERIFY \
  TRAIN_TIMING_STEPS WANDB_MODE JAX_COMPILATION_CACHE_DIR MMEVLA_MOTION_OVERFLOW MMEVLA_STAGE2_EVIDENCE_ROOT
for key in $(compgen -e); do
  case "$key" in *_ALLOW_UNVERIFIED) unset "$key" ;; esac
done
[[ -z "$TIMING" ]] || export TRAIN_TIMING_STEPS="$TIMING"
[[ -z "$EVIDENCE" ]] || export MMEVLA_STAGE2_EVIDENCE_ROOT="$EVIDENCE"
[[ -n ${SLURM_JOB_ID-} ]] || export CUDA_VISIBLE_DEVICES="$VISIBLE"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$HOME/.cache/uv" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
LIBRARY=$("$PY" scripts/training/check_orig16_stage2_launch.py --print-library)
LIB="$ROOT/v1-store/datasets/$LIBRARY"
case "$HC" in
  *-32frame-4x4-*) FRAME=framesamp-4x4 ;;
  *-8frame-8x8-*) FRAME=framesamp-8x8 ;;
  *-16frame-4x4-*|*-64frame-4x4-*) FRAME=framesamp-4x4 ;;
  *) echo 'HISTORY帧布局非法' >&2; exit 2 ;;
esac
export MMEVLA_FRAMESAMP_SOURCE="$LIB/source" MMEVLA_FRAMESAMP_MANIFEST="$LIB/meta/episode_manifest.json"
export TRAIN_FINAL_RECORD_DIR="$FINAL" TRAIN_RECORD_DIR="$(dirname "$FINAL")/train-record"
export MMEVLA_EXPECTED_TRAIN_HEAD="$TRAIN_HEAD" MMEVLA_V1_STORE="$ROOT/v1-store"
export MMEVLA_JAX_CACHE_DIR="$TRAIN_CACHE_ROOT/jax/$RUN" CUDA_CACHE_PATH="$TRAIN_CACHE_ROOT/cuda/$RUN"
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
  --model.use-history --model.history-config "$HC" "${STEP_ARGS[@]}" "${WANDB_ARGS[@]}"
  --fsdp-devices "$N" --num-workers "$((2 * N))" --batch-size "$B")
print_command() {
  printf 'TRAIN_CMD='
  printf '%q ' "$PY" scripts/training/train.py "${ARGS[@]}"
  printf '\n'
}
contract() {
  JAX_PLATFORMS=cpu "$PY" scripts/training/check_orig16_stage2_launch.py \
    --repo "$ROOT" --store-root "$ROOT" --train-head "$TRAIN_HEAD" --steps "$STEPS" \
    --fsdp-devices "$N" --batch-size "$B" --gpus-visible "$VISIBLE" -- "${ARGS[@]}"
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
  printf 'RUN=%s\nSTEPS=%s\nTRAIN_HEAD=%s\nGPUS=%s\nGLOBAL_BATCH=%s\nSTART_UTC=%s\n' "$RUN" "$STEPS" "$TRAIN_HEAD" "$N" "$B" "$(date -u +%FT%TZ)"
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
if LC_ALL=C grep -aqiP '^\r*Step [0-9]+:.*(loss|grad_norm|llm_grad_norm|param_norm|mem_enc_norm)=[-+]?(nan|inf)(,|[[:space:]]|$)|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError' "$LOG"; then
  printf 'TRAIN_NUMERICS=FAIL reason=nonfinite_or_oom\n' >> "$LOG"
  printf 'TRAIN_DIAGNOSTIC=FAIL check=train_numerics action=record_only reason=user_decision\n' >> "$LOG"
fi
if ! JAX_PLATFORMS=cpu "$PY" scripts/training/check_orig16_stage2_launch.py \
    --check-final-record "$FINAL/final.json" --run "$RUN" --steps "$STEPS" --train-head "$TRAIN_HEAD" >> "$LOG" 2>&1; then
  printf 'TRAIN_DIAGNOSTIC=FAIL check=final_record action=record_only reason=user_decision\n' >> "$LOG"
fi
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG" || rc=1
exit "$rc"
