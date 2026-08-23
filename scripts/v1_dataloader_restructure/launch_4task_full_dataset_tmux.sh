#!/usr/bin/env bash
# 在 detached tmux 中启动四任务原版全量数据构建.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare_runtime_dirs
require_command tmux
readonly SESSION_NAME="v1-original-4task-full-build"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "错误: tmux session 已存在: ${SESSION_NAME}" >&2
  exit 1
fi
if [[ -e "${FULL_DATASET_PATH}" ]]; then
  echo "错误: 全量数据目录已存在: ${FULL_DATASET_PATH}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --short)" ]]; then
  echo "错误: 正式全量构建要求 clean HEAD" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd \"${REPO_ROOT}\" && bash \"${V1_SCRIPT_DIR}/build_4task_full_dataset.sh\""

echo "已启动 tmux session: ${SESSION_NAME}"
echo "日志: ${ARTIFACT_ROOT}/build_4task_full_dataset.log"
echo "状态: tmux has-session -t ${SESSION_NAME}"
