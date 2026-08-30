#!/usr/bin/env bash
# 训练域（scripts/training/）的路径与环境源（commitV4.6 自建库域 paths.sh 派生，
# 切断 g0/tests 驱动对建库域的跨域 source）。g0/ 与 tests/ 的驱动一律
# source 本文件；建库域（scripts/dataset/gl/）自带同构的 paths.sh，两域互不引用。
#
# 设计要点：
#   1. **不覆盖 HOME**。上一版 common.sh 把 HOME 指向项目内目录，会让 ssh 找不到
#      ~/.ssh/config 与 ControlMaster socket，直接打断 gl_submit.py 的集群提交。
#      改为逐项显式设置缓存类环境变量指向 v1-store/cache/。
#   2. 仓库位置 fail-loud：必须位于 turbo 前缀下，否则拒绝运行（防止把几百 GB
#      产物写进本机盘）。校验的是前缀而非全路径，将来改目录名不必改代码。
#   3. RAW_H5_DIR 默认指向 turbo 那份 H5，使本机与集群读**同一份字节**，
#      一致性验证不被输入差异污染；确需本机原件（NVMe 更快）时用环境变量覆盖。

set -euo pipefail

readonly GL_ROOT="/nfs/turbo/coe-chaijy-unreplicated/hongzefu"
readonly TURBO_PREFIX="${GL_ROOT}/"

V1_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly V1_SCRIPT_DIR
REPO_ROOT="$(cd "${V1_SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT

# 根自证断言（fail-loud，计划二节第 11 条）：层数数错时立刻响亮失败，
# 防止静默错位后 v1_prepare_dirs 在错误位置造出假 v1-store 目录树
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "错误: 仓库根解析失败 ${REPO_ROOT}（缺 pyproject.toml）" >&2
  exit 1
fi

case "${REPO_ROOT}/" in
  "${TURBO_PREFIX}"*) ;;
  *)
    echo "错误: 仓库必须位于 ${TURBO_PREFIX} 下, 当前为 ${REPO_ROOT}" >&2
    echo "      本仓库单副本在 NFS turbo, 本机不保留副本(见 AGENTS.md 第 13 条)。" >&2
    exit 1
    ;;
esac

# ── 项目内产物根（整体不进 Git，见 .gitignore 的 /v1-store/）────────────────────
readonly V1_STORE="${REPO_ROOT}/v1-store"
readonly MODELS_DIR="${V1_STORE}/models"
readonly DATASETS_DIR="${V1_STORE}/datasets"
readonly CACHE_DIR="${V1_STORE}/cache"
readonly LOGS_DIR="${V1_STORE}/logs"
readonly BENCH_DIR="${V1_STORE}/bench"
readonly TRAIN_ASSETS="${V1_STORE}/train-assets"
readonly TRAIN_RUNS="${V1_STORE}/train-runs"

# ── 四个数据集：1 个正式产物 + 3 个一致性验证对照 ───────────────────────────────
readonly GL_DATASET="${DATASETS_DIR}/4task-gl"            # GreatLakes 8×1GPU 全量（1600 ep）
readonly REF_UNTOUCHED="${DATASETS_DIR}/ref-untouched"    # 未改动 builder，第一层参照系
readonly REF_SHARD="${DATASETS_DIR}/ref-shard"            # 分片实现，第一层被测
readonly REF_CROSSARCH="${DATASETS_DIR}/ref-crossarch"    # 本地真值，第二/三层跨架构对拍

readonly MANIFEST_PATH="${V1_STORE}/episode_manifest.json"
# 两份 H5 sha256 清单：turbo 副本侧（集群消费的那份）与本机原件侧（只用来 diff 证同源）
readonly INPUT_MANIFEST_PATH="${V1_STORE}/input_manifest.json"
readonly INPUT_MANIFEST_LOCAL_PATH="${V1_STORE}/input_manifest_local.json"

# ── 原始 H5 ────────────────────────────────────────────────────────────────────
readonly RAW_H5_LOCAL="/data/hongzefu/robomme_data_h5_v2_4env400ep"   # 本机原件，永久保留
readonly RAW_H5_TURBO="${GL_ROOT}/robomme_data_h5_v2_4env400ep"       # 集群输入暂存，验收后删除
RAW_H5_DIR="${RAW_H5_DIR:-${RAW_H5_TURBO}}"                            # 默认 turbo，双端同源

readonly EXPECTED_H5=(
  "record_dataset_ButtonUnmask.h5"
  "record_dataset_ButtonUnmaskSwap.h5"
  "record_dataset_VideoUnmask.h5"
  "record_dataset_VideoUnmaskSwap.h5"
)
readonly EXPECTED_EPISODES_PER_H5=400
readonly EXPECTED_TOTAL_EPISODES=1600

# ── venv：本机与 GreatLakes 共用同一个（解释器装在 NFS uv-python 上）───────────
readonly PY="${REPO_ROOT}/.venv/bin/python"
readonly NFS_UV_PYTHON="${GL_ROOT}/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"

# ── 环境变量：显式逐项指向 v1-store/cache，**不动 HOME** ────────────────────────
export OPENPI_DATA_HOME="${MODELS_DIR}"
# ⚠ 刻意**不**设 UV_CACHE_DIR：greatlakes.md 的「venv 可移植性」一节明确要求
# 「cache 在本机盘、venv 在 NFS，跨设备必须 copy」。uv 的 cache 只是下载缓存，且只在
# 跑 uv sync 的这台机器上用得到（计算节点直调 .venv/bin/python，根本不碰 uv）；
# 把它搬到 NFS 只会让一次性建 venv 多写十几 GB、还和数据流量抢带宽。
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export HF_HOME="${CACHE_DIR}/hf"
export JAX_COMPILATION_CACHE_DIR="${CACHE_DIR}/jax"
export WANDB_DIR="${LOGS_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb"
export WANDB_CONFIG_DIR="${CACHE_DIR}/wandb-config"
export UV_LINK_MODE=copy

v1_prepare_dirs() {
  mkdir -p \
    "${MODELS_DIR}" "${DATASETS_DIR}" "${CACHE_DIR}" "${LOGS_DIR}" "${BENCH_DIR}" \
    "${TRAIN_ASSETS}" "${TRAIN_RUNS}" \
    "${XDG_CACHE_HOME}" "${HF_HOME}" "${JAX_COMPILATION_CACHE_DIR}" \
    "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"
}

v1_require_cmd() {
  local name
  for name in "$@"; do
    command -v "${name}" >/dev/null 2>&1 || { echo "错误: 缺少命令 ${name}" >&2; exit 1; }
  done
}

# 原始 H5 目录自洽性：恰好 4 个目标 H5、metadata sidecar 齐全且 record_count 均为 400。
v1_validate_raw_h5() {
  local dir="${1:-${RAW_H5_DIR}}"
  local name meta count
  [[ -d "${dir}" ]] || { echo "错误: 原始 H5 目录不存在: ${dir}" >&2; exit 1; }
  count="$(find "${dir}" -maxdepth 1 -type f -name '*.h5' | wc -l)"
  [[ "${count}" -eq 4 ]] || { echo "错误: ${dir} 必须恰好含 4 个 H5, 实为 ${count}" >&2; exit 1; }
  for name in "${EXPECTED_H5[@]}"; do
    [[ -f "${dir}/${name}" ]] || { echo "错误: 缺少 H5: ${dir}/${name}" >&2; exit 1; }
    meta="${dir}/${name%.h5}_metadata.json"
    [[ -f "${meta}" ]] || { echo "错误: 缺少 metadata sidecar: ${meta}" >&2; exit 1; }
    if [[ "$(jq -r '.record_count' "${meta}")" -ne "${EXPECTED_EPISODES_PER_H5}" ]]; then
      echo "错误: ${meta} 的 record_count 不是 ${EXPECTED_EPISODES_PER_H5}" >&2; exit 1
    fi
  done
}

# 项目内模型齐全性（集群侧只需 SigLIP；训练侧还需 tokenizer 与 pi05_base）。
v1_require_models() {
  local need_train="${1:-0}"
  local siglip="${MODELS_DIR}/pi05_vision_encoder/siglip_params.pkl"
  [[ -f "${siglip}" ]] || { echo "错误: 缺少项目内 SigLIP 权重: ${siglip}" >&2; exit 1; }
  if [[ "${need_train}" = "1" ]]; then
    local tok="${MODELS_DIR}/big_vision/paligemma_tokenizer.model"
    local pi05="${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params"
    [[ -f "${tok}" ]] || { echo "错误: 缺少项目内 PaliGemma tokenizer: ${tok}" >&2; exit 1; }
    [[ -f "${pi05}/commit_success.txt" ]] || { echo "错误: 缺少项目内 pi05_base: ${pi05}" >&2; exit 1; }
  fi
}

v1_require_venv() {
  [[ -x "${PY}" ]] || { echo "错误: 缺少 NFS venv 解释器: ${PY}" >&2; exit 1; }
  "${PY}" -c 'import jax' >/dev/null 2>&1 || { echo "错误: ${PY} 无法 import jax" >&2; exit 1; }
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "错误: paths.sh 只能被 source, 不能直接执行" >&2
  exit 1
fi
