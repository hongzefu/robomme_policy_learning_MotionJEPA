#!/usr/bin/env bash
# 使用原版 dataloader、模型、loss 和优化器运行 12-step training smoke.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare_runtime_dirs
require_command uv
require_command grep
bash "${V1_SCRIPT_DIR}/stage_project_models.sh"
require_project_models

readonly LOG_PATH="${ARTIFACT_ROOT}/run_4task_training_smoke.log"
readonly NORM_STATS_PATH="${RUNS_ROOT}/assets/mme_vla_suite/robomme/norm_stats.json"
readonly SMOKE_CKPT_DIR="${RUNS_ROOT}/ckpts/mme_vla_suite/${SMOKE_RUN_NAME}"
readonly WEIGHT_PATH="${PROJECT_OPENPI_HOME}/openpi-assets/checkpoints/pi05_base/params"

if [[ ! -f "${SMOKE_DATASET_PATH}/meta/stats.json" ]]; then
  echo "错误: smoke 数据集不存在: ${SMOKE_DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "错误: smoke norm stats 不存在: ${NORM_STATS_PATH}" >&2
  exit 1
fi
if [[ -e "${SMOKE_CKPT_DIR}" ]]; then
  echo "错误: smoke run 目录已存在, 禁止 overwrite: ${SMOKE_CKPT_DIR}" >&2
  exit 1
fi

set +e
(
  set -e
  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES=0,1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  uv run --frozen scripts/smoke_train_once.py mme_vla_suite \
    --exp-name "${SMOKE_RUN_NAME}" \
    --assets-base-dir "${RUNS_ROOT}/assets" \
    --checkpoint-base-dir "${RUNS_ROOT}/ckpts" \
    --batch-size 2 \
    --num-workers 0 \
    --num-train-steps 12 \
    --log-interval 1 \
    --fsdp-devices 2 \
    --dataset-path "${SMOKE_DATASET_PATH}" \
    --weight-loader.params-path "${WEIGHT_PATH}" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
) 2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" -eq 0 ]]; then
  if ! grep -q 'Tentative run completed' "${LOG_PATH}"; then
    echo "错误: smoke 退出码为 0, 但缺少完成标记" | tee -a "${LOG_PATH}" >&2
    status=1
  fi
fi

if [[ "${status}" -eq 0 && -e "${SMOKE_CKPT_DIR}" ]]; then
  case "${SMOKE_CKPT_DIR}" in
    "${RUNS_ROOT}/ckpts/mme_vla_suite/${SMOKE_RUN_NAME}")
      rm -rf -- "${SMOKE_CKPT_DIR}"
      ;;
    *)
      echo "错误: 拒绝清理非预期路径: ${SMOKE_CKPT_DIR}" | tee -a "${LOG_PATH}" >&2
      status=1
      ;;
  esac
fi

echo "EXIT_CODE=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
