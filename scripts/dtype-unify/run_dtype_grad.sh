#!/usr/bin/env bash
# 单步定点梯度取证的驱动（2 卡，正确性口径）。
#
# 与 dump 驱动同理：修复前后各跑一次，两侧口径必须逐字相同，只剩 RUN_TAG 一个变量。
# 确定性档 XLA_FLAGS 写死在这里——正确性族 run 一律注入 D2 档
# （`v1-gradient-baseline.md` 符号总表），不由调用方随手改。
#
# 用法：
#   RUN_TAG=v1-dtype-p3-dump-pre GRAD_ARRAYS_DIR=/data/hongzefu/v1-baselines/dtype-p5-grad-pre \
#     bash scripts/dtype-unify/run_dtype_grad.sh
#
# 可调环境变量：
#   RUN_TAG           必填
#   GRAD_ARRAYS_DIR   梯度数组落盘目录（本机盘；不设则只落逐叶摘要与统计）
#   FIXTURE_DIR       batch fixture 落盘目录（默认 v1-store/fixtures/<RUN_TAG>）
#   DTYPE_GRAD_KINDS  限定 batch 种类（逗号分隔，仅供冒烟）
#   BASELINE_CHECKSUMS  G0b 步 0 摘要（默认指向 G0b r1；设为空串可跳过同源校验）

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../data-preprocess-GL" && pwd)/paths.sh"
v1_prepare_dirs
v1_require_venv
v1_require_models 1

: "${RUN_TAG:?必须设置 RUN_TAG}"

REC_DIR="${V1_STORE}/dtype-unify/${RUN_TAG}-grad"
FIXTURE_DIR="${FIXTURE_DIR:-${V1_STORE}/fixtures/${RUN_TAG}}"
LOG="${LOGS_DIR}/${RUN_TAG}-grad.log"
CKPT_BASE="${TRAIN_RUNS}/${RUN_TAG}-grad"
DEFAULT_BASELINE="${REPO_ROOT}/docs/training-doc/v1-grad-baseline-g0b/records/r1/param_checksums.jsonl"
BASELINE_CHECKSUMS="${BASELINE_CHECKSUMS-${DEFAULT_BASELINE}}"

for d in "${REC_DIR}" "${FIXTURE_DIR}"; do
  if [[ -e "${d}" ]]; then
    echo "错误: 目录已存在，拒绝覆盖既有取证产物: ${d}" >&2
    exit 1
  fi
done

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain | head -c 1)" ]]; then
  echo "错误: 工作区不干净——取证必须从 clean HEAD 起跑（AGENTS 12/17）" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 1
fi
GIT_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

export UV_LINK_MODE=copy
UVPY=(uv run --project "${REPO_ROOT}" python)

# jax 编译缓存收敛进 v1-store（AGENTS 14，不动 train.py、不覆盖 HOME）：
# train.main 硬编码写 ~/.cache/jax_<exp_name>，用软链把它指到 v1-store/cache/
JAX_CACHE_DIR="${CACHE_DIR}/jax/${RUN_TAG}-grad"
JAX_CACHE_LINK="${HOME}/.cache/jax_${RUN_TAG}-grad"
mkdir -p "${JAX_CACHE_DIR}" "$(dirname "${JAX_CACHE_LINK}")"
ln -sfn "${JAX_CACHE_DIR}" "${JAX_CACHE_LINK}"

DET_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

echo "=== dtype 单步梯度: RUN_TAG=${RUN_TAG} HEAD=${GIT_HEAD} ==="
echo "    records ${REC_DIR}"
echo "    fixture ${FIXTURE_DIR}"
echo "    梯度数组 ${GRAD_ARRAYS_DIR:-（不落）}"
echo "    XLA_FLAGS ${DET_FLAGS}"

(
  set -e
  cd "${REPO_ROOT}"
  DTYPE_GRAD_DIR="${REC_DIR}" \
  DTYPE_BATCH_FIXTURE_DIR="${FIXTURE_DIR}" \
  DTYPE_GRAD_ARRAYS_DIR="${GRAD_ARRAYS_DIR:-}" \
  DTYPE_GRAD_KINDS="${DTYPE_GRAD_KINDS:-}" \
  DTYPE_BASELINE_CHECKSUMS="${BASELINE_CHECKSUMS}" \
  CUDA_VISIBLE_DEVICES=0,1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  XLA_FLAGS="${DET_FLAGS}" \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  "${UVPY[@]}" "${REPO_ROOT}/scripts/dtype-unify/single_step_grad.py" mme_vla_suite \
    --exp-name "${RUN_TAG}-grad" \
    --assets-base-dir "${TRAIN_ASSETS}" \
    --checkpoint-base-dir "${CKPT_BASE}" \
    --batch-size 8 \
    --num-workers 4 \
    --fsdp-devices 2 \
    --seed 42 \
    --dataset-path "${DATASET_PATH:-${GL_DATASET}}" \
    --weight-loader.params-path "${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
) 2>&1 | tee "${LOG}"
RC="${PIPESTATUS[0]}"

rm -rf "${CKPT_BASE}"
rm -f "${JAX_CACHE_LINK}"
if [[ "${RC}" -ne 0 ]]; then
  echo "GRAD_FAIL rc=${RC}"
  exit "${RC}"
fi
echo "GRAD_PASS run_tag=${RUN_TAG} records=${REC_DIR}"
