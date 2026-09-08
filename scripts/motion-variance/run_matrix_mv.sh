#!/usr/bin/env bash
# motion-variance 32 批矩阵（seed-major：42→7→2024→17；每 seed 内 test→val；每 split 内 official→normal→mask→swap），批间串行，失败即停。
# 用法：[SKIP_DONE=1] [SEEDS=42,7,2024,17] [SPLITS=test,val] [CONDS=official,normal,mask,swap] [WORKERS=8] bash scripts/motion-variance/run_matrix_mv.sh
#   整个矩阵应起在一个 detached tmux `mv-matrix`（见 docs/training-doc/mv-matrix-40k/launch.md）；逐批 MV_BATCH= 追加到 v1-store/logs/mv-matrix.log。
#   SKIP_DONE=1：mv-matrix.log 里已有 `MV_BATCH=PASS cond=X split=Y seed=Z` 的批直接跳过（校准批复用）。
#   MV_ALLOW_DIRTY 一律拒绝；起跑前 git status --porcelain 必须空；swap 需要 bank 存在。PORT_BASE = 9300 + (idx % 16) * 8。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../training" && pwd)/paths.sh"
[[ "${MV_ALLOW_DIRTY:-0}" == "0" ]] || { echo "错误: 矩阵拒绝 MV_ALLOW_DIRTY" >&2; exit 2; }
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain | head -c 1)" ]]; then echo "错误: 工作区不干净（AGENTS 12）" >&2; exit 1; fi
SEEDS="${SEEDS:-42,7,2024,17}"; SPLITS="${SPLITS:-test,val}"; CONDS="${CONDS:-official,normal,mask,swap}"
WORKERS="${WORKERS:-8}"; SKIP_DONE="${SKIP_DONE:-0}"
BANK="${SWAP_BANK:-${V1_STORE}/reports/motion-variance/bank-lib}"
MLOG="${LOGS_DIR}/mv-matrix.log"
MV_DIR="${REPO_ROOT}/scripts/motion-variance"
[[ -f "${BANK}/donor_bank.json" ]] || { echo "错误: donor bank 不存在: ${BANK}" >&2; exit 1; }
grep -q '"source": "library"' "${BANK}/donor_bank.json" || { echo "错误: bank source 不是 library" >&2; exit 1; }
mkdir -p "${LOGS_DIR}"
IFS=',' read -r -a SEED_ARR <<<"${SEEDS}"; IFS=',' read -r -a SPLIT_ARR <<<"${SPLITS}"; IFS=',' read -r -a COND_ARR <<<"${CONDS}"
N_BATCH=$(( ${#SEED_ARR[@]} * ${#SPLIT_ARR[@]} * ${#COND_ARR[@]} ))
echo "MV_MATRIX_START HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD) branch=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD) bank_sha=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['sha256'][:16])" "${BANK}/donor_bank.json") seeds=${SEEDS} splits=${SPLITS} conds=${CONDS} batches=${N_BATCH} workers=${WORKERS} skip_done=${SKIP_DONE} start=$(date '+%F %T')" | tee -a "${MLOG}"
T0=$(date +%s); IDX=0; DONE=0; SKIPPED=0; FAIL=0
for SEED in "${SEED_ARR[@]}"; do for SPLIT in "${SPLIT_ARR[@]}"; do for COND in "${COND_ARR[@]}"; do
  PB=$(( 9300 + (IDX % 16) * 8 )); IDX=$((IDX + 1))
  if [[ "${SKIP_DONE}" == "1" ]] && grep -qE "^MV_BATCH=PASS cond=${COND} split=${SPLIT} seed=${SEED} " "${MLOG}" 2>/dev/null; then
    echo "跳过已完成批 cond=${COND} split=${SPLIT} seed=${SEED}"; SKIPPED=$((SKIPPED + 1)); continue
  fi
  echo "--- BATCH ${IDX}/${N_BATCH} cond=${COND} split=${SPLIT} seed=${SEED} port_base=${PB} start=$(date '+%F %T') ---" | tee -a "${MLOG}"
  OUT="$(COND="${COND}" SPLIT="${SPLIT}" SEED="${SEED}" WORKERS="${WORKERS}" PORT_BASE="${PB}" SWAP_BANK="${BANK}" bash "${MV_DIR}/run_batch_mv.sh" 2>&1)"
  RC=$?
  echo "${OUT}"
  grep -E '^MV_BATCH=' <<<"${OUT}" | tee -a "${MLOG}"
  if [[ "${RC}" -ne 0 ]]; then echo "错误: 批失败 cond=${COND} split=${SPLIT} seed=${SEED}，矩阵停止" | tee -a "${MLOG}"; FAIL=1; break 3; fi
  DONE=$((DONE + 1))
done; done; done
WALL_H=$(python3 -c "print(f'{($(date +%s) - ${T0}) / 3600:.2f}')")
echo "MV_MATRIX=$([[ ${FAIL} -eq 0 ]] && echo PASS || echo FAIL) batches=$((DONE + SKIPPED))/${N_BATCH} done_now=${DONE} skipped=${SKIPPED} wall_h=${WALL_H} end=$(date '+%F %T')" | tee -a "${MLOG}"
echo "EXIT_CODE=${FAIL}"
exit "${FAIL}"
