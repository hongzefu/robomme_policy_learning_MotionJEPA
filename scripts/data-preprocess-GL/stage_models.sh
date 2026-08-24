#!/usr/bin/env bash
# 把 SigLIP / PaliGemma tokenizer / pi05_base 复制进项目内 v1-store/models/。
#
# 为什么要复制而不是软链或用 ~/.cache：
#   · AGENTS.md 第 14 条要求模型、tokenizer、缓存全部内联在仓库目录内；
#   · GreatLakes 计算节点看不到本机 /home，软链在集群上是死链；
#   · 项目内一份意味着「本机与集群读同一份权重字节」，一致性验证不被权重差异污染。
#
# 集群侧数据构建只用到 SigLIP；tokenizer 与 pi05_base 是本地训练 smoke 才需要，
# 但 12 GB 的 pi05_base 一次性备好，避免 smoke 时才发现缺件。

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_cmd rsync find diff stat

readonly MODEL_SRC="${MODEL_SRC:-/home/hongzefu/.cache/openpi}"
readonly SRC_SIGLIP="${MODEL_SRC}/pi05_vision_encoder/siglip_params.pkl"
readonly SRC_TOKENIZER="${MODEL_SRC}/big_vision/paligemma_tokenizer.model"
readonly SRC_PI05="${MODEL_SRC}/openpi-assets/checkpoints/pi05_base/params"
readonly DST_SIGLIP_DIR="${MODELS_DIR}/pi05_vision_encoder"
readonly DST_TOKENIZER_DIR="${MODELS_DIR}/big_vision"
readonly DST_PI05="${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params"
readonly INVENTORY="${LOGS_DIR}/models_inventory.txt"

for p in "${SRC_SIGLIP}" "${SRC_TOKENIZER}" "${SRC_PI05}"; do
  [[ -e "${p}" ]] || { echo "错误: 模型来源不存在: ${p}" >&2; exit 1; }
done

mkdir -p "${DST_SIGLIP_DIR}" "${DST_TOKENIZER_DIR}" "${DST_PI05}"

echo "[stage] SigLIP (1.7 GB) -> ${DST_SIGLIP_DIR}"
rsync -aL "${SRC_SIGLIP}" "${DST_SIGLIP_DIR}/"
echo "[stage] PaliGemma tokenizer -> ${DST_TOKENIZER_DIR}"
rsync -aL "${SRC_TOKENIZER}" "${DST_TOKENIZER_DIR}/"
echo "[stage] pi05_base params (12 GB) -> ${DST_PI05}"
rsync -aL "${SRC_PI05}/" "${DST_PI05}/"

# 逐项校验：单文件比字节数，目录比「相对路径 + 字节数」清单，最后禁止残留软链
[[ "$(stat -c '%s' "${SRC_SIGLIP}")" == "$(stat -c '%s' "${DST_SIGLIP_DIR}/siglip_params.pkl")" ]] \
  || { echo "错误: SigLIP 字节数不一致" >&2; exit 1; }
[[ "$(stat -c '%s' "${SRC_TOKENIZER}")" == "$(stat -c '%s' "${DST_TOKENIZER_DIR}/paligemma_tokenizer.model")" ]] \
  || { echo "错误: tokenizer 字节数不一致" >&2; exit 1; }

src_list="$(mktemp)"; dst_list="$(mktemp)"
trap 'rm -f -- "${src_list}" "${dst_list}"' EXIT
( cd "${SRC_PI05}" && find . -type f -printf '%P %s\n' | sort ) > "${src_list}"
( cd "${DST_PI05}" && find . -type f -printf '%P %s\n' | sort ) > "${dst_list}"
diff -u "${src_list}" "${dst_list}"

if find "${MODELS_DIR}" -type l -print -quit | grep -q .; then
  echo "错误: 项目内模型目录出现软链（集群上会是死链）" >&2; exit 1
fi

{
  echo "OPENPI_DATA_HOME=${MODELS_DIR}"
  du -sh "${DST_SIGLIP_DIR}" "${DST_TOKENIZER_DIR}" "${DST_PI05}"
  echo "pi05_base_file_count=$(find "${DST_PI05}" -type f | wc -l)"
  echo "symlink_count=$(find "${MODELS_DIR}" -type l | wc -l)"
} | tee "${INVENTORY}"

echo "[stage] 项目内模型准备完成。"
