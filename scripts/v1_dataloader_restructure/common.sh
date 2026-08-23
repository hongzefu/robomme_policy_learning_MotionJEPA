#!/usr/bin/env bash
# v1 原版四任务本地流程的共享路径和只读校验.

set -euo pipefail

readonly V1_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${V1_SCRIPT_DIR}/../.." && pwd)"
readonly EXPECTED_REPO_ROOT="/data/hongzefu/robomme_policy_learning_MotionJEPA"
readonly RAW_H5_DIR="/data/hongzefu/robomme_data_h5_v2_4env400ep"
readonly PROJECT_OPENPI_HOME="${REPO_ROOT}/.openpi-data"
readonly PROJECT_CACHE_DIR="${REPO_ROOT}/.cache"
readonly PROJECT_RUNTIME_HOME="${REPO_ROOT}/.runtime-home"
readonly DATA_ROOT="${REPO_ROOT}/data"
readonly RUNS_ROOT="${REPO_ROOT}/runs"
readonly ARTIFACT_ROOT="${REPO_ROOT}/artifacts/v1_dataloader_restructure"
readonly SMOKE_DATASET_PATH="${DATA_ROOT}/robomme_preprocessed_4task_original_smoke"
readonly FULL_DATASET_PATH="${DATA_ROOT}/robomme_preprocessed_4task_original"
readonly SMOKE_RUN_NAME="v1-original-framesamp-context-12step-smoke"
readonly MODEL_SOURCE_HOME="/home/hongzefu/.cache/openpi"

export OPENPI_DATA_HOME="${PROJECT_OPENPI_HOME}"
export UV_CACHE_DIR="${PROJECT_CACHE_DIR}/uv"
export XDG_CACHE_HOME="${PROJECT_CACHE_DIR}"
export HOME="${PROJECT_RUNTIME_HOME}"
export WANDB_DIR="${ARTIFACT_ROOT}/wandb"
export WANDB_CACHE_DIR="${PROJECT_CACHE_DIR}/wandb"
export WANDB_CONFIG_DIR="${PROJECT_CACHE_DIR}/wandb-config"

if [[ "${REPO_ROOT}" != "${EXPECTED_REPO_ROOT}" ]]; then
  echo "错误: 仓库路径必须为 ${EXPECTED_REPO_ROOT}, 当前为 ${REPO_ROOT}" >&2
  exit 1
fi

prepare_runtime_dirs() {
  mkdir -p \
    "${PROJECT_OPENPI_HOME}" \
    "${PROJECT_CACHE_DIR}" \
    "${PROJECT_RUNTIME_HOME}" \
    "${DATA_ROOT}" \
    "${RUNS_ROOT}" \
    "${ARTIFACT_ROOT}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_CONFIG_DIR}"
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "错误: 缺少命令 ${command_name}" >&2
    exit 1
  fi
}

validate_raw_h5() {
  local -a expected_h5=(
    "record_dataset_ButtonUnmask.h5"
    "record_dataset_VideoUnmask.h5"
    "record_dataset_ButtonUnmaskSwap.h5"
    "record_dataset_VideoUnmaskSwap.h5"
  )
  local h5_name
  local h5_count

  if [[ ! -d "${RAW_H5_DIR}" ]]; then
    echo "错误: 全局原始 H5 目录不存在: ${RAW_H5_DIR}" >&2
    exit 1
  fi

  h5_count="$(find "${RAW_H5_DIR}" -maxdepth 1 -type f -name '*.h5' -printf '%f\n' | wc -l)"
  if [[ "${h5_count}" -ne 4 ]]; then
    echo "错误: 原始目录必须恰好包含 4 个 H5, 当前为 ${h5_count}" >&2
    exit 1
  fi

  for h5_name in "${expected_h5[@]}"; do
    if [[ ! -f "${RAW_H5_DIR}/${h5_name}" ]]; then
      echo "错误: 缺少 H5: ${RAW_H5_DIR}/${h5_name}" >&2
      exit 1
    fi
    local metadata_path="${RAW_H5_DIR}/${h5_name%.h5}_metadata.json"
    if [[ ! -f "${metadata_path}" ]]; then
      echo "错误: 缺少 metadata sidecar: ${metadata_path}" >&2
      exit 1
    fi
    if [[ "$(jq -r '.record_count' "${metadata_path}")" -ne 400 ]]; then
      echo "错误: metadata record_count 不是 400: ${metadata_path}" >&2
      exit 1
    fi
  done
}

require_project_models() {
  local siglip_path="${PROJECT_OPENPI_HOME}/pi05_vision_encoder/siglip_params.pkl"
  local tokenizer_path="${PROJECT_OPENPI_HOME}/big_vision/paligemma_tokenizer.model"
  local pi05_path="${PROJECT_OPENPI_HOME}/openpi-assets/checkpoints/pi05_base/params"

  if [[ ! -f "${siglip_path}" ]]; then
    echo "错误: 缺少项目内 SigLIP 权重: ${siglip_path}" >&2
    exit 1
  fi
  if [[ ! -f "${tokenizer_path}" ]]; then
    echo "错误: 缺少项目内 PaliGemma tokenizer: ${tokenizer_path}" >&2
    exit 1
  fi
  if [[ ! -f "${pi05_path}/commit_success.txt" ]]; then
    echo "错误: 缺少项目内 pi05_base checkpoint: ${pi05_path}" >&2
    exit 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "错误: common.sh 只能由其他 v1 Bash 脚本 source" >&2
  exit 1
fi
