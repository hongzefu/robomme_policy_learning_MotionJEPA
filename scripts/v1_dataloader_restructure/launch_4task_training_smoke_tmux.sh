#!/usr/bin/env bash
# 在 detached tmux 中启动原版 12-step training smoke.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare_runtime_dirs
require_command tmux
readonly SESSION_NAME="v1-original-framesamp-training-smoke"
readonly SMOKE_CKPT_DIR="${RUNS_ROOT}/ckpts/mme_vla_suite/${SMOKE_RUN_NAME}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "错误: tmux session 已存在: ${SESSION_NAME}" >&2
  exit 1
fi
if [[ ! -f "${SMOKE_DATASET_PATH}/meta/stats.json" ]]; then
  echo "错误: smoke 数据集尚未完成: ${SMOKE_DATASET_PATH}" >&2
  exit 1
fi
if [[ -e "${SMOKE_CKPT_DIR}" ]]; then
  echo "错误: smoke run 目录已存在: ${SMOKE_CKPT_DIR}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd \"${REPO_ROOT}\" && bash \"${V1_SCRIPT_DIR}/run_4task_training_smoke.sh\""

echo "已启动 tmux session: ${SESSION_NAME}"
echo "日志: ${ARTIFACT_ROOT}/run_4task_training_smoke.log"
echo "状态: tmux has-session -t ${SESSION_NAME}"
