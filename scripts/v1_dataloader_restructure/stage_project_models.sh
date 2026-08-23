#!/usr/bin/env bash
# 将原版 SigLIP、PaliGemma tokenizer 和 pi05_base 复制进当前项目.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare_runtime_dirs
require_command rsync
require_command find
require_command diff
require_command stat

readonly SOURCE_SIGLIP="${MODEL_SOURCE_HOME}/pi05_vision_encoder/siglip_params.pkl"
readonly SOURCE_TOKENIZER="${MODEL_SOURCE_HOME}/big_vision/paligemma_tokenizer.model"
readonly SOURCE_PI05_PARAMS="${MODEL_SOURCE_HOME}/openpi-assets/checkpoints/pi05_base/params"
readonly DEST_SIGLIP_DIR="${PROJECT_OPENPI_HOME}/pi05_vision_encoder"
readonly DEST_TOKENIZER_DIR="${PROJECT_OPENPI_HOME}/big_vision"
readonly DEST_PI05_PARAMS="${PROJECT_OPENPI_HOME}/openpi-assets/checkpoints/pi05_base/params"
readonly INVENTORY_PATH="${ARTIFACT_ROOT}/project_models_inventory.txt"

for source_path in "${SOURCE_SIGLIP}" "${SOURCE_TOKENIZER}" "${SOURCE_PI05_PARAMS}"; do
  if [[ ! -e "${source_path}" ]]; then
    echo "错误: 模型来源不存在: ${source_path}" >&2
    exit 1
  fi
done

mkdir -p "${DEST_SIGLIP_DIR}" "${DEST_TOKENIZER_DIR}" "${DEST_PI05_PARAMS}"

echo "复制 SigLIP 到项目内..."
rsync -aL --info=progress2 "${SOURCE_SIGLIP}" "${DEST_SIGLIP_DIR}/"

echo "复制 PaliGemma tokenizer 到项目内..."
rsync -aL --info=progress2 "${SOURCE_TOKENIZER}" "${DEST_TOKENIZER_DIR}/"

echo "复制 pi05_base checkpoint 到项目内..."
rsync -aL --info=progress2 "${SOURCE_PI05_PARAMS}/" "${DEST_PI05_PARAMS}/"

if [[ "$(stat -c '%s' "${SOURCE_SIGLIP}")" != "$(stat -c '%s' "${DEST_SIGLIP_DIR}/siglip_params.pkl")" ]]; then
  echo "错误: SigLIP 文件大小不一致" >&2
  exit 1
fi
if [[ "$(stat -c '%s' "${SOURCE_TOKENIZER}")" != "$(stat -c '%s' "${DEST_TOKENIZER_DIR}/paligemma_tokenizer.model")" ]]; then
  echo "错误: tokenizer 文件大小不一致" >&2
  exit 1
fi

source_manifest="$(mktemp "${PROJECT_CACHE_DIR}/pi05-source.XXXXXX")"
dest_manifest="$(mktemp "${PROJECT_CACHE_DIR}/pi05-dest.XXXXXX")"
trap 'rm -f -- "${source_manifest}" "${dest_manifest}"' EXIT

(
  cd "${SOURCE_PI05_PARAMS}"
  find . -type f -printf '%P %s\n' | sort
) >"${source_manifest}"
(
  cd "${DEST_PI05_PARAMS}"
  find . -type f -printf '%P %s\n' | sort
) >"${dest_manifest}"
diff -u "${source_manifest}" "${dest_manifest}"

if find "${PROJECT_OPENPI_HOME}" -type l -print -quit | grep -q .; then
  echo "错误: 项目内模型目录包含符号链接" >&2
  exit 1
fi

{
  echo "OPENPI_DATA_HOME=${PROJECT_OPENPI_HOME}"
  du -sh "${DEST_SIGLIP_DIR}" "${DEST_TOKENIZER_DIR}" "${DEST_PI05_PARAMS}"
  echo "pi05_base_file_count=$(find "${DEST_PI05_PARAMS}" -type f | wc -l)"
  echo "symlink_count=$(find "${PROJECT_OPENPI_HOME}" -type l | wc -l)"
} | tee "${INVENTORY_PATH}"

echo "项目内模型准备完成."
