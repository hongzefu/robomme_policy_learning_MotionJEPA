#!/usr/bin/env bash
# ── 本机 2 GPU 功能性 smoke：perceptual-framesamp-context 跑通 GL 产出的全量库 ───
#
# 这是验证方案的第四层「训练可用性」：数据集能被原版 dataloader / 模型 / loss 正常吃下去，
# loss 有限、无 NaN、无形状或键缺失。
#
# ⚠ **这是功能性 smoke，不是吞吐基准。** 代码与数据都在 turbo（实测天花板 ~132 MB/s），
#   而 frame sampling 每个样本要读 budget/token_per_image = 512/16 = 32 个 token_emb
#   （≈19 MB），batch 会明显偏慢。按 AGENTS.md 第 13 条，本机数字本来就不作吞吐结论。
#
# 入口用同目录的 smoke_train_once.py 而不是 scripts/train.py——后者的 __main__ 会在
# tentative 之后紧接着起 80k step 的正式训练（见该文件头注释）。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../data-preprocess-GL" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_cmd jq
v1_require_venv
v1_require_models 1                       # smoke 还需要 tokenizer 与 pi05_base

DATASET_PATH="${DATASET_PATH:-${GL_DATASET}}"
RUN_NAME="${RUN_NAME:-v1-gl-framesamp-context-smoke}"
STEPS="${STEPS:-12}"
BATCH="${BATCH:-2}"                       # 仓库默认 64 是 4 卡口径；2 卡从 2 起步，OOM 再降
LOG="${LOGS_DIR}/${RUN_NAME}.log"
CKPT_DIR="${TRAIN_RUNS}/mme_vla_suite/${RUN_NAME}"
NORM_STATS="${TRAIN_ASSETS}/mme_vla_suite/robomme/norm_stats.json"

[[ -f "${DATASET_PATH}/meta/stats.json" ]] || {
  echo "错误: 数据集不存在或未 finalize: ${DATASET_PATH}/meta/stats.json" >&2; exit 1; }
[[ -e "${CKPT_DIR}" ]] && {
  echo "错误: run 目录已存在, 禁止 overwrite: ${CKPT_DIR}" >&2; exit 1; }

if [[ ! -f "${NORM_STATS}" ]]; then
  echo "=== norm stats 缺失，先算（只读 pkl，history_config=None，不碰 feature）==="
  # compute_norm_stats.py 用的是 TrainConfig 的默认 assets_base_dir="runs/assets"，
  # 那是个**相对 cwd** 的路径且它没暴露覆盖参数。所以在 v1-store 下开一个专用 cwd 跑，
  # 产物自然落在 v1-store 内（不污染仓库根），再搬到正式位置。
  NS_CWD="${V1_STORE}/norm-stats-cwd"
  mkdir -p "${NS_CWD}" "$(dirname "${NORM_STATS}")"
  ( cd "${NS_CWD}" && PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "${PY}" "${REPO_ROOT}/scripts/compute_norm_stats.py" \
      --config-name mme_vla_suite --repo-id robomme --dataset-path "${DATASET_PATH}" )
  SRC="${NS_CWD}/runs/assets/mme_vla_suite/robomme/norm_stats.json"
  [[ -f "${SRC}" ]] || { echo "错误: norm stats 未生成: ${SRC}" >&2; exit 1; }
  mv "${SRC}" "${NORM_STATS}"
  echo "  norm stats -> ${NORM_STATS}"
fi

echo "=== 2 GPU smoke: ${RUN_NAME} (${STEPS} steps, batch ${BATCH}) ==="
echo "  数据集: ${DATASET_PATH}"
set +e
(
  set -e
  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES=0,1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  "${PY}" "${REPO_ROOT}/scripts/smoke-local/smoke_train_once.py" mme_vla_suite \
    --exp-name "${RUN_NAME}" \
    --assets-base-dir "${TRAIN_ASSETS}" \
    --checkpoint-base-dir "${TRAIN_RUNS}" \
    --batch-size "${BATCH}" \
    --num-workers 0 \
    --num-train-steps "${STEPS}" \
    --log-interval 1 \
    --fsdp-devices 2 \
    --dataset-path "${DATASET_PATH}" \
    --weight-loader.params-path "${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
) 2>&1 | tee "${LOG}"
STATUS="${PIPESTATUS[0]}"
set -e

# 判定：必须见到完成标记，且 loss 有限无 NaN
if [[ "${STATUS}" -eq 0 ]]; then
  grep -q "Tentative run completed" "${LOG}" || {
    echo "错误: 退出码为 0 但缺少完成标记" | tee -a "${LOG}" >&2; STATUS=1; }
fi
if [[ "${STATUS}" -eq 0 ]]; then
  if grep -qiE "loss=(nan|inf|-inf)" "${LOG}"; then
    echo "错误: 日志里出现 NaN/Inf loss" | tee -a "${LOG}" >&2; STATUS=1
  fi
  N_STEP="$(grep -cE '^Step [0-9]+: ' "${LOG}" || true)"
  if [[ "${N_STEP}" -lt 2 ]]; then
    echo "错误: 只见到 ${N_STEP} 行 Step 日志，训练循环没真正跑起来" | tee -a "${LOG}" >&2; STATUS=1
  else
    echo "  ✓ ${N_STEP} 个 step 的 loss 全部有限" | tee -a "${LOG}"
    grep -E '^Step [0-9]+: ' "${LOG}" | tail -3
  fi
fi

# 跑完即删临时 run 目录（AGENTS.md 第 6 条），且只删这一个精确路径
if [[ -e "${CKPT_DIR}" ]]; then
  case "${CKPT_DIR}" in
    "${TRAIN_RUNS}/mme_vla_suite/${RUN_NAME}") rm -rf -- "${CKPT_DIR}" ;;
    *) echo "错误: 拒绝清理非预期路径 ${CKPT_DIR}" | tee -a "${LOG}" >&2; STATUS=1 ;;
  esac
fi

echo "EXIT_CODE=${STATUS}" | tee -a "${LOG}"
[[ "${STATUS}" -eq 0 ]] && echo "LAYER4_PASS 训练可用性通过"
exit "${STATUS}"
