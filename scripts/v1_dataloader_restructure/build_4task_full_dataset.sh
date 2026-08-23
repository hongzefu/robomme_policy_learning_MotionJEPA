#!/usr/bin/env bash
# 用原版 builder 处理四任务各 400 episodes, 然后计算全量 norm stats.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare_runtime_dirs
require_command uv
require_command jq
validate_raw_h5
bash "${V1_SCRIPT_DIR}/stage_project_models.sh"
require_project_models

readonly LOG_PATH="${ARTIFACT_ROOT}/build_4task_full_dataset.log"
readonly NORM_STATS_PATH="${RUNS_ROOT}/assets/mme_vla_suite/robomme/norm_stats.json"
readonly NORM_STATS_ARCHIVE="${ARTIFACT_ROOT}/full_norm_stats.json"
readonly HEAD_ARCHIVE="${ARTIFACT_ROOT}/full_dataset_head.txt"

if [[ -e "${FULL_DATASET_PATH}" ]]; then
  echo "错误: 全量数据目录已存在, 原版 builder 会删除目标目录, 因此拒绝继续: ${FULL_DATASET_PATH}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --short)" ]]; then
  echo "错误: 正式全量构建要求 clean HEAD" >&2
  exit 1
fi

git -C "${REPO_ROOT}" rev-parse HEAD | tee "${HEAD_ARCHIVE}"

set +e
(
  set -e
  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  uv run --frozen scripts/build_dataset.py \
    --dataset_type robomme_pkl \
    --raw_data_path "${RAW_H5_DIR}" \
    --preprocessed_data_path "${FULL_DATASET_PATH}"

  feature_episode_count="$(find "${FULL_DATASET_PATH}/features" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l)"
  if [[ "${feature_episode_count}" -ne 1600 ]]; then
    echo "错误: 全量 feature episode 数应为 1600, 当前为 ${feature_episode_count}" >&2
    exit 1
  fi
  jq -e '.execution_samples > 0 and .total_samples > 0' "${FULL_DATASET_PATH}/meta/stats.json" >/dev/null

  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  uv run --frozen scripts/compute_norm_stats.py \
    --config-name mme_vla_suite \
    --repo-id robomme \
    --dataset-path "${FULL_DATASET_PATH}"

  if [[ ! -f "${NORM_STATS_PATH}" ]]; then
    echo "错误: 全量 norm stats 未生成: ${NORM_STATS_PATH}" >&2
    exit 1
  fi
  cp "${NORM_STATS_PATH}" "${NORM_STATS_ARCHIVE}"
) 2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

echo "EXIT_CODE=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
