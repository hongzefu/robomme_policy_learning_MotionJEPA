#!/usr/bin/env bash
# 官方16×100单条入口：8卡单进程、B128、正式60k或同日程20步短测。
# 1114 4task 参数化：HISTORY 以 -4task-rgbstate.yaml 结尾时切到 4task 档——合同脚本 check_4task_stage2_launch.py、
# 库名取其 --print-library、TrainConfig 取其 --print-config、正式步数 NUM_TRAIN_STEPS=50001、短测前缀 v16-smoke-、
# 日志落 v1-store/logs/v16-4trs-vla/；orig16 档（-orig16.yaml）全部默认值不变。
set -euo pipefail
[[ $# == 0 ]] || { echo '本入口不接受透传参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
HC=${HISTORY:?必须指定HISTORY}
RUN=${RUN_NAME:?必须指定RUN_NAME}
FINAL=${FINAL_RECORD_DIR:?必须指定FINAL_RECORD_DIR}
: "${TRAIN_HEAD:?必须指定完整TRAIN_HEAD}"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo 'TRAIN_HEAD必须为40位SHA' >&2; exit 2; }
[[ "$RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'RUN_NAME非法' >&2; exit 2; }
[[ "$HC" != */* && ( "$HC" == *-orig16.yaml || "$HC" == *-4task-rgbstate.yaml ) ]] || { echo 'HISTORY必须是新配置文件名' >&2; exit 2; }
# 档位由 HISTORY 后缀决定（不另开可写任意脚本路径的入口）：合同脚本、正式步数、短测前缀与日志目录一并切换。
if [[ "$HC" == *-4task-rgbstate.yaml ]]; then
  PROFILE=4task
  CONTRACT=scripts/training/check_4task_stage2_launch.py
  FORMAL_STEPS=50001
  SMOKE_RUNS=(v16-smoke-modul-s2rgbstate-concat1x1024 v16-smoke-ctx-s2rgbstate-concat1x1024)
  LOG_DIR="$ROOT/v1-store/logs/v16-4trs-vla"
else
  PROFILE=orig16
  CONTRACT=scripts/training/check_orig16_stage2_launch.py
  FORMAL_STEPS=60000
  LOG_DIR="$ROOT/v1-store/logs"
fi
[[ "$FINAL" == /* && "$(basename "$FINAL")" == final-record ]] || { echo 'FINAL_RECORD_DIR必须是绝对final-record路径' >&2; exit 2; }
[[ ${DRY_RUN:-0} == 0 || ${DRY_RUN:-0} == 1 ]] || { echo 'DRY_RUN仅允许0或1' >&2; exit 2; }
[[ ${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7} == 0,1,2,3,4,5,6,7 ]] || { echo 'GPU_SCOPE=FAIL allowed=0-7' >&2; exit 2; }
# NUM_TRAIN_STEPS：训练循环步数（state_step 终值）；与 CKPT_STEPS（落盘 checkpoint 步集合）是两个名字，不混用。
NUM_TRAIN_STEPS_ENV=${NUM_TRAIN_STEPS-}
NUM_TRAIN_STEPS=$FORMAL_STEPS
STEP_ARGS=()
if [[ -n ${MAX_STEPS+x} ]]; then
  if [[ $PROFILE == 4task ]]; then
    smoke_ok=0
    for name in "${SMOKE_RUNS[@]}"; do [[ "$RUN" == "$name" ]] && smoke_ok=1; done
    [[ "$MAX_STEPS" == 20 && $smoke_ok == 1 ]] || { echo "MAX_STEPS仅允许20且须为固定短测名: ${SMOKE_RUNS[*]}" >&2; exit 2; }
    fusion=$(cut -d- -f3 <<< "$HC")
    [[ "$RUN" == "v16-smoke-$fusion-"* ]] || { echo '短测名与HISTORY融合方式不符' >&2; exit 2; }
  else
    [[ "$MAX_STEPS" == 20 && "$RUN" == o16-smoke-* ]] || { echo 'MAX_STEPS仅允许20且须o16-smoke-名称' >&2; exit 2; }
  fi
  NUM_TRAIN_STEPS=20
  STEP_ARGS=(--num-train-steps 20)
fi
[[ -z $NUM_TRAIN_STEPS_ENV || $NUM_TRAIN_STEPS_ENV == "$NUM_TRAIN_STEPS" ]] \
  || { echo "NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS_ENV 与本档期望 $NUM_TRAIN_STEPS 不符（正式 $FORMAL_STEPS，短测 MAX_STEPS=20 时为 20）" >&2; exit 2; }
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
[[ -f "$ROOT/$CONTRACT" ]] || { echo "启动合同脚本不存在: $CONTRACT" >&2; exit 2; }
LIBRARY=$("$PY" "$CONTRACT" --print-library)
if [[ $PROFILE == 4task ]]; then
  CONFIG=$("$PY" "$CONTRACT" --print-config)
  [[ "$CONFIG" =~ ^[a-z0-9_]+$ ]] || { echo "合同脚本给出的TrainConfig非法: $CONFIG" >&2; exit 2; }
else
  CONFIG=mme_vla_suite_orig16_b128_60k
fi
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
  JAX_PLATFORMS=cpu "$PY" "$CONTRACT" \
    --repo "$ROOT" --store-root "$ROOT" --train-head "$TRAIN_HEAD" --steps "$NUM_TRAIN_STEPS" -- "${ARGS[@]}"
}
# 完整继承参数表：按与 train.py 相同的 CLI 解析读回 TrainConfig，逐项打印（只读，不建任何目录）。
print_params() {
  JAX_PLATFORMS=cpu "$PY" - "${ARGS[@]}" <<'PARAMS_PY'
import sys
from mme_vla_suite.training import config
sys.argv = ["train.py", *sys.argv[1:]]
cfg = config.cli()
lr = cfg.lr_schedule
rows = [("config", cfg.name), ("exp_name", cfg.exp_name), ("num_train_steps", cfg.num_train_steps),
        ("batch_size", cfg.batch_size), ("fsdp_devices", cfg.fsdp_devices), ("num_workers", cfg.num_workers),
        ("seed", cfg.seed), ("ema_decay", cfg.ema_decay), ("lr_schedule", type(lr).__name__),
        ("warmup_steps", lr.warmup_steps), ("peak_lr", lr.peak_lr), ("decay_steps", lr.decay_steps),
        ("end_lr", lr.decay_lr), ("save_interval", cfg.save_interval), ("keep_period", cfg.keep_period),
        ("log_interval", cfg.log_interval), ("action_horizon", cfg.model.action_horizon),
        ("pi05", cfg.model.pi05), ("use_history", cfg.model.use_history),
        ("history_config", cfg.model.history_config), ("clip_gradient_norm", cfg.optimizer.clip_gradient_norm)]
for key, value in rows:
    print(f"TRAIN_PARAM {key}={value}")
steps = config.checkpoint_steps(cfg)
print(f"CKPT_STEPS={','.join(map(str, steps))} count={len(steps)}")
PARAMS_PY
}
# 只读分支在日志、run和缓存任何mkdir/open之前返回。
if [[ ${DRY_RUN:-0} == 1 ]]; then
  print_command
  # 不给 MAX_STEPS 的正式 dry-run 另打印完整继承参数表（计划 A17）；短测 dry-run 行为不变。
  if [[ -z ${MAX_STEPS+x} ]]; then print_params; fi
  contract
  exit $?
fi
LOG="$LOG_DIR/$RUN.log"
[[ -d "$LOG_DIR" && "$(realpath "$LOG_DIR")" == "$LOG_DIR" ]] || { echo '日志父目录必须预先存在且为实体目录' >&2; exit 2; }
( set -o noclobber; : > "$LOG" ) || { echo "拒绝覆盖日志: $LOG" >&2; exit 2; }
body() (
  set -euo pipefail
  printf 'RUN=%s\nSTEPS=%s\nTRAIN_HEAD=%s\nSTART_UTC=%s\n' "$RUN" "$NUM_TRAIN_STEPS" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
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
if ! JAX_PLATFORMS=cpu "$PY" "$CONTRACT" \
    --check-final-record "$FINAL/final.json" --run "$RUN" --steps "$NUM_TRAIN_STEPS" --train-head "$TRAIN_HEAD" >> "$LOG" 2>&1; then
  printf 'TRAIN_DIAGNOSTIC=FAIL check=final_record action=record_only reason=user_decision\n' >> "$LOG"
fi
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG" || rc=1
exit "$rc"
