#!/usr/bin/env bash
# 8 卡分片并行起评（留档 docs/training-doc/eval-*/）：每片独占一张卡（policy [+ sidecar] + 仿真同卡），一片一个 detached tmux 会话
# <LOG_PREFIX>-<SHARD>，脚本结束会话自动退出。两种分片布局（MODE）：
#   task（默认，旧口径）：4 任务 × 2 片（ep 0–24 / 25–49），分片名 <TASK>-<K>，端口 PORT_BASE+行号。
#   stride（负载均衡）  ：WORKERS 个 worker，每个跑全部 4 任务、每任务取 ep ∈ {k, k+WORKERS, …}（8 worker 下 w0/w1 各 28 集、其余 24 集），
#                         分片名 w<k>，GPU=k，端口 PORT_BASE+k。任务级耗时差异被均摊（留档 eval-official-framesamp-context/）。
# 用法：[MODE=task|stride] [WORKERS=8] [GPU_LIST=0,1,…] [ONLY=ButtonUnmask-0|w0] [EP_COUNT=…] [RUN_NAME=…] [CKPT_ID=…] [CKPT_DIR=…]
#       [SEED=42] [SPLIT=test|val|train] [LOG_PREFIX=ev40k] [PORT_BASE=8031] [POLICY_MEM_FRACTION=0.4] [ROBOMME_PY=…] [DRY_RUN=1]
#       bash scripts/training/prod/eval_all_shards.sh
#   GPU_LIST 决定 worker 号到实际卡号的映射（取模），默认 0..7 即历史的「worker 号 = 卡号」。少于 8 卡的机器必须显式给：
#   如 2 卡机 GPU_LIST=0,0,1,1 配 WORKERS=4（每卡 2 个 worker），或 GPU_LIST=0,1 配 WORKERS=2（每卡 1 个）。
#   POLICY_MEM_FRACTION / ROBOMME_PY / SPLIT 只在显式设置时才转发（tmux 会话不继承调用方环境），未设则用 eval_shard.sh 的默认值。
#   RUN_NAME / CKPT_ID / CKPT_DIR / SEED / LOG_PREFIX / POLICY_MEM_FRACTION / ROBOMME_PY 原样转给 eval_shard.sh
#   （tmux 会话不继承调用方环境，必须显式写进命令）。
#   ONLY 只起一片（预检用）；已存在同名会话则跳过不重起。断点续评：重跑同一片即续（eval.py 按 progress.json 跳过已评集）。
#   DRY_RUN=1 只打印将要起的会话与命令，不起 tmux（验证分片表用）。
#   清理只允许 tmux kill-session -t <确切会话名>（AGENTS 7）；本轮会话清单见对应留档 launch.md。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"
ONLY="${ONLY:-}"
MODE="${MODE:-task}"
WORKERS="${WORKERS:-8}"
TASKS_ALL="${TASKS_ALL:-ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap}"
EP_COUNT="${EP_COUNT:-}"  # 留空按模式给默认：task=25（每片集数），stride=0（不设集号上界）
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="${RUN_NAME:-awsprod40k-b128-motion}"
CKPT_ID="${CKPT_ID:-39999}"
CKPT_DIR="${CKPT_DIR:-}"
SEED="${SEED:-42}"
LOG_PREFIX="${LOG_PREFIX:-ev40k}"
PORT_BASE="${PORT_BASE:-8031}"
# 卡号映射：worker 号按 GPU_LIST 取模落到实际卡上。默认 0..7 即「worker 号 = 卡号」的历史行为（8 卡机）；
# 少于 8 卡的机器显式给出，可让多个 worker 共用一张卡，例如 2 卡机 GPU_LIST=0,0,1,1（4 worker，每卡 2 个）。
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<<"${GPU_LIST}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "错误: GPU_LIST 为空" >&2; exit 1; }
# task 口径分片表：TASK K GPU（一片一卡；端口 = PORT_BASE + 行号）
TASK_SHARDS=(
  "ButtonUnmask 0 0"
  "ButtonUnmask 1 1"
  "VideoUnmask 0 2"
  "VideoUnmask 1 3"
  "ButtonUnmaskSwap 0 4"
  "ButtonUnmaskSwap 1 5"
  "VideoUnmaskSwap 0 6"
  "VideoUnmaskSwap 1 7"
)
# 两种布局统一输出行：<SHARD> <GPU> <转给 eval_shard.sh 的额外环境变量…>
build_rows() {
  local k row task kk gpu
  if [[ "${MODE}" == "stride" ]]; then
    for ((k = 0; k < WORKERS; k++)); do
      echo "w${k} ${GPU_ARR[k % ${#GPU_ARR[@]}]} SHARD_ID=w${k} K=${k} TASKS=${TASKS_ALL} EP_STRIDE=${WORKERS} EP_START=${k} EP_COUNT=${EP_COUNT:-0}"
    done
  elif [[ "${MODE}" == "task" ]]; then
    for row in "${TASK_SHARDS[@]}"; do
      read -r task kk gpu <<<"${row}"
      echo "${task}-${kk} ${GPU_ARR[gpu % ${#GPU_ARR[@]}]} TASK=${task} K=${kk} EP_COUNT=${EP_COUNT:-25}"
    done
  else
    echo "错误: MODE 只能是 task 或 stride，实为 ${MODE}" >&2; return 1
  fi
}
mkdir -p "${LOGS_DIR}"
i=0
while read -r SHARD GPU EXTRA; do
  PORT=$((PORT_BASE + i)); i=$((i + 1))
  SESSION="${LOG_PREFIX}-${SHARD}"
  if [[ -n "${ONLY}" && "${ONLY}" != "${SHARD}" ]]; then continue; fi
  if tmux has-session -t "=${SESSION}" 2>/dev/null; then echo "跳过 ${SESSION}（会话已存在）"; continue; fi
  LOG="${LOGS_DIR}/${LOG_PREFIX}-${SHARD}.log"
  ENVS="${EXTRA} GPU=${GPU} PORT=${PORT} RUN_NAME=${RUN_NAME} CKPT_ID=${CKPT_ID} SEED=${SEED} LOG_PREFIX=${LOG_PREFIX}"
  [[ -n "${CKPT_DIR}" ]] && ENVS="${ENVS} CKPT_DIR=${CKPT_DIR}"
  [[ -n "${POLICY_MEM_FRACTION:-}" ]] && ENVS="${ENVS} POLICY_MEM_FRACTION=${POLICY_MEM_FRACTION}"
  [[ -n "${ROBOMME_PY:-}" ]] && ENVS="${ENVS} ROBOMME_PY=${ROBOMME_PY}"
  [[ -n "${SPLIT:-}" ]] && ENVS="${ENVS} SPLIT=${SPLIT}"
  CMD="set -o pipefail; ${ENVS} bash scripts/training/prod/eval_shard.sh 2>&1 | tee -a ${LOG}; sleep 2"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "DRY ${SESSION}: ${CMD}"; continue; fi
  tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "bash -c '${CMD}'"
  echo "起 ${SESSION}: shard=${SHARD} gpu=${GPU} port=${PORT} ${EXTRA} run=${RUN_NAME} ckpt=${CKPT_DIR:-<默认>/${CKPT_ID}} log=${LOG}"
done < <(build_rows)
tmux ls 2>/dev/null | grep "^${LOG_PREFIX}-" || true
