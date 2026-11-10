#!/usr/bin/env bash
# 官方 FrameSamp+Context 512（HF Yinpei/mme_vla_suite perceptual-framesamp-context/79999）
# 用 batch 64 续训 5000 步。8 卡（mesh 2×4），代码用 fs8 训练同一锚点 a4293f4 的只读副本。
# 用法：CODE_ROOT=<a4293f4 只读副本> [PY=<解释器>] [DRY=1] bash run_fs512_official_cont5k.sh <prod|smoke>
# 与官方配方的差异（只有这些）：
#   - 初始权重 = 官方 79999 params（官方 zip 只有参数，无 AdamW 状态 → 优化器从零起，EMA 从加载权重起算）；
#   - num_train_steps 5000、warmup 500（原 10000），peak=decay=5e-5 不变（warmup 后常数）；
#   - norm stats 用官方 ckpt 自带 assets（与权重训练时同一份）；
#   - save_interval/keep_period 2500（存 2500、4999）。
# 其余（batch 64、EMA 0.999、AdamW、action horizon 20、数据 16task-orig100 framesamp-4x4、seed 42）同官方条目。
set -euo pipefail
[[ $# == 1 ]] || { echo "用法: CODE_ROOT=<dir> $0 <prod|smoke>" >&2; exit 2; }
: "${CODE_ROOT:?必须指定 CODE_ROOT}"
PROJECT=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE=$PROJECT/v1-store
LIBRARY=16task-orig100-1600ep-9e109419
LIB=$STORE/datasets/$LIBRARY
PY=${PY:-$STORE/envs/fs8-train/bin/python}
CACHE=/scratch/hongze/.cache
OFFICIAL=${OFFICIAL:-$STORE/models/official-mme-vla-suite/perceptual-framesamp-context/79999}
BASE=orig100-ctx-framesamp512-official79999-cont5k-b64
HISTORY=perceptual-framesamp-context.yaml

case "$1" in
  prod) RUN=$BASE; EXTRA=() ;;
  smoke) RUN=$BASE-smoke; EXTRA=(--num-train-steps 20 --no-wandb-enabled) ;;
  *) echo "未知模式: $1" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
export PYTHONPATH=$CODE_ROOT/src:$CODE_ROOT/packages/openpi-client/src
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OPENPI_DATA_HOME=$STORE/models
export XDG_CACHE_HOME=$CACHE/xdg HF_HOME=$CACHE/huggingface
export MMEVLA_JAX_CACHE_DIR=$CACHE/jax/$BASE CUDA_CACHE_PATH=$CACHE/cuda/$BASE
export WANDB_DIR=$STORE/logs/wandb WANDB_CACHE_DIR=$CACHE/wandb WANDB_CONFIG_DIR=$CACHE/wandb-config
if [[ $1 == prod && ${DRY:-0} != 1 ]]; then
  WANDB_API_KEY=$(</scratch/hongze/.secrets/wandb_api_key)
  WANDB_ENTITY=$(sed -n 's/^WANDB_ENTITY=//p' "$STORE/secrets/wandb.env")
  export WANDB_API_KEY WANDB_ENTITY
fi

ARGS=(mme_vla_suite --exp-name "$RUN" --dataset-path "$LIB/framesamp-4x4"
  --assets-base-dir "$STORE/train-assets"
  --data.assets.assets-dir "$OFFICIAL/assets" --data.assets.asset-id robomme
  --checkpoint-base-dir "$STORE/train-runs"
  --weight-loader.params-path "$OFFICIAL/params"
  --model.use-history --model.history-config "$HISTORY"
  --num-train-steps 5000 --lr-schedule.warmup-steps 500
  --save-interval 2500 --keep-period 2500 "${EXTRA[@]}")

LOG=$STORE/logs/$RUN.log
if [[ ${DRY:-0} == 1 ]]; then
  echo "DRY LOG=$LOG PY=$PY"; printf 'DRY ARG %s\n' "${ARGS[@]}"; exit 0
fi
if [[ -e $LOG ]]; then mv "$LOG" "$LOG.$(date +%Y%m%d%H%M%S)"; fi
set +e
{
  "$PY" -c 'import mme_vla_suite; print("MME_VLA_SUITE_IMPORT=" + mme_vla_suite.__file__, flush=True)' \
    && "$PY" scripts/training/train.py "${ARGS[@]}"
} 2>&1 | tee "$LOG"
rc=$?
echo "EXIT_CODE=$rc" | tee -a "$LOG"
exit "$rc"
