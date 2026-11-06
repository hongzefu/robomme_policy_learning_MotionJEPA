#!/usr/bin/env bash
# context 1024 档 B64 80k 单条入口：td（tokendrop，GPU 0-3）或 fs（framesamp，GPU 4-7）。
# 用法：CODE_ROOT=<固定 commit 的只读代码副本> bash run_ctx1024_b64_80k.sh <td|fs> <prod|smoke|perf>
# 超参全用官方 mme_vla_suite 条目；prod 不传覆盖。tmux 由调用方起。
# 不 source paths.sh：它按脚本位置解析根目录，会把 v1-store 指到 CODE_ROOT 下。
set -euo pipefail
[[ $# == 2 ]] || { echo "用法: CODE_ROOT=<dir> $0 <td|fs> <prod|smoke|perf>" >&2; exit 2; }
: "${CODE_ROOT:?必须指定 CODE_ROOT}"
PROJECT=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE=$PROJECT/v1-store
LIBRARY=16task-orig100-1600ep-9e109419
LIB=$STORE/datasets/$LIBRARY
PY=$PROJECT/.venv/bin/python
CACHE=/scratch/hongze/.cache

case "$1" in
  td) BASE=orig100-ctx-tokendrop1024-b64-80k; GPUS=0,1,2,3; FRAME=framesamp-8x8; HISTORY=perceptual-tokendrop-context-1024.yaml ;;
  fs) BASE=orig100-ctx-framesamp1024-b64-80k; GPUS=4,5,6,7; FRAME=framesamp-4x4; HISTORY=perceptual-framesamp-context-1024.yaml ;;
  *) echo "未知档位: $1" >&2; exit 2 ;;
esac
case "$2" in
  prod) RUN=$BASE; EXTRA=() ;;
  smoke) RUN=$BASE-smoke; EXTRA=(--num-train-steps 20 --no-wandb-enabled) ;;
  perf) RUN=$BASE-perf; EXTRA=(--num-train-steps 300 --no-wandb-enabled) ;;
  *) echo "未知模式: $2" >&2; exit 2 ;;
esac

# history-config 按 cwd 相对路径加载，必须在代码副本根目录运行。
cd "$CODE_ROOT"
export PYTHONPATH=$CODE_ROOT/src:$CODE_ROOT/packages/openpi-client/src
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OPENPI_DATA_HOME=$STORE/models
export XDG_CACHE_HOME=$CACHE/xdg HF_HOME=$CACHE/huggingface
# train.py 与驱动的默认缓存都在 $HOME 下。
export MMEVLA_JAX_CACHE_DIR=$CACHE/jax/$BASE CUDA_CACHE_PATH=$CACHE/cuda/$BASE
# WANDB_DIR 默认是 cwd，而 cwd 是只读代码副本。
export WANDB_DIR=$STORE/logs/wandb WANDB_CACHE_DIR=$CACHE/wandb WANDB_CONFIG_DIR=$CACHE/wandb-config
if [[ $2 == prod ]]; then
  WANDB_API_KEY=$(</scratch/hongze/.secrets/wandb_api_key)
  # train.py 的默认 entity 对这把 key 报 403，须显式给。
  WANDB_ENTITY=$(sed -n 's/^WANDB_ENTITY=//p' "$STORE/secrets/wandb.env")
  export WANDB_API_KEY WANDB_ENTITY
fi

ARGS=(mme_vla_suite --exp-name "$RUN" --dataset-path "$LIB/$FRAME"
  --assets-base-dir "$STORE/train-assets"
  --data.assets.assets-dir "$STORE/train-assets/mme_vla_suite/$LIBRARY" --data.assets.asset-id robomme
  --checkpoint-base-dir "$STORE/train-runs"
  --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params"
  --model.use-history --model.history-config "$HISTORY" "${EXTRA[@]}")

LOG=$STORE/logs/$RUN.log
# 旧日志改名保留，避免重跑时覆盖，也避免 Monitor 读到上一次的 EXIT_CODE=。
if [[ -e $LOG ]]; then mv "$LOG" "$LOG.$(date +%Y%m%d%H%M%S)"; fi
set +e
{
  "$PY" -c 'import mme_vla_suite; print("MME_VLA_SUITE_IMPORT=" + mme_vla_suite.__file__, flush=True)' \
    && "$PY" scripts/training/train.py "${ARGS[@]}"
} 2>&1 | tee "$LOG"
rc=$?
echo "EXIT_CODE=$rc" | tee -a "$LOG"
exit "$rc"
