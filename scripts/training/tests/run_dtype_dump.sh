#!/usr/bin/env bash
# dtype 修复第一块取证的驱动：一次跑完定点样本 dump + 定点 batch dump。
#
# 为什么要驱动脚本而不是直接敲长命令：取证要在**两个不同 commit** 上各跑一次，
# 两侧口径必须逐字相同——把路径与参数固化在这里，两侧就只剩 RUN_TAG 一个变量。
# 路径一律取自 scripts/training/paths.sh（训练域路径源），不自行硬编码。
#
# 用法：
#   RUN_TAG=v1-dtype-p3-dump-pre bash scripts/training/tests/run_dtype_dump.sh
#
# 可调环境变量：
#   RUN_TAG              必填，决定输出目录与日志名
#   DTYPE_DUMP_MODE      samples|batches|both（默认 both）
#   DTYPE_DUMP_ARRAYS    1|0（默认 1，是否落 memory 四键数组本体）
#   DTYPE_DUMP_LIMIT     >0 时每组只取前 N 个（仅供冒烟，正式取证不设）

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"
v1_prepare_dirs
v1_require_venv
v1_require_models 1

: "${RUN_TAG:?必须设置 RUN_TAG（决定输出目录与日志名）}"

DUMP_ROOT="${V1_STORE}/dtype-unify"
OUT_DIR="${DUMP_ROOT}/${RUN_TAG}"
LOG="${LOGS_DIR}/${RUN_TAG}.log"
CKPT_BASE="${TRAIN_RUNS}/${RUN_TAG}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "错误: 输出目录已存在，拒绝覆盖既有取证产物: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${DUMP_ROOT}"

export UV_LINK_MODE=copy
UVPY=(uv run --project "${REPO_ROOT}" python)

GIT_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
GIT_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain | head -c 1)"
if [[ -n "${GIT_DIRTY}" ]]; then
  echo "错误: 工作区不干净——取证必须从 clean HEAD 起跑（AGENTS 12/17）" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 1
fi

echo "=== dtype dump: RUN_TAG=${RUN_TAG} HEAD=${GIT_HEAD} ==="
echo "    输出目录 ${OUT_DIR}"
echo "    模式 ${DTYPE_DUMP_MODE:-both} 数组 ${DTYPE_DUMP_ARRAYS:-1} limit ${DTYPE_DUMP_LIMIT:-0}"

(
  set -e
  cd "${REPO_ROOT}"
  DTYPE_DUMP_DIR="${OUT_DIR}" \
  DTYPE_DUMP_GIT_HEAD="${GIT_HEAD}" \
  DTYPE_DUMP_MODE="${DTYPE_DUMP_MODE:-both}" \
  DTYPE_DUMP_ARRAYS="${DTYPE_DUMP_ARRAYS:-1}" \
  DTYPE_DUMP_LIMIT="${DTYPE_DUMP_LIMIT:-0}" \
  JAX_PLATFORMS=cpu \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  "${UVPY[@]}" "${REPO_ROOT}/scripts/training/tests/dump_fixture_samples.py" mme_vla_suite \
    --exp-name "${RUN_TAG}" \
    --assets-base-dir "${TRAIN_ASSETS}" \
    --checkpoint-base-dir "${CKPT_BASE}" \
    --batch-size 8 \
    --num-workers 4 \
    --dataset-path "${GL_DATASET}" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
) 2>&1 | tee "${LOG}"
RC="${PIPESTATUS[0]}"

rm -rf "${CKPT_BASE}"
if [[ "${RC}" -ne 0 ]]; then
  echo "DUMP_FAIL rc=${RC}"
  exit "${RC}"
fi
echo "DUMP_PASS run_tag=${RUN_TAG} out=${OUT_DIR}"
