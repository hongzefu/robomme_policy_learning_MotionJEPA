#!/usr/bin/env bash
# 8 卡分片并行起评（留档 docs/training-doc/eval-*/）：每片独占一张卡（policy [+ sidecar] + 仿真同卡），一片一个 detached tmux 会话
# <LOG_PREFIX>-<SHARD>，脚本结束会话自动退出。两种分片布局（MODE）：
#   task（默认，旧口径）：4 任务 × 2 片（ep 0–24 / 25–49），分片名 <TASK>-<K>，端口 PORT_BASE+行号。
#   stride（负载均衡）  ：WORKERS 个 worker，每个跑全部 4 任务、每任务取 ep ∈ {k, k+WORKERS, …}（8 worker 下 w0/w1 各 28 集、其余 24 集），
#                         分片名 w<k>，GPU=k，端口 PORT_BASE+k。任务级耗时差异被均摊（留档 eval-official-framesamp-context/）。
# 用法：[MODE=task|stride] [WORKERS=8] [ONLY=ButtonUnmask-0|w0] [EP_COUNT=…] [RUN_NAME=…] [CKPT_ID=…] [CKPT_DIR=…] [SEED=42]
#       [DATASET=test|val] [LOG_PREFIX=ev40k] [PORT_BASE=8031] [DRY_RUN=1] bash scripts/training/legacy-eval/eval_all_shards.local.sh
#   RUN_NAME / CKPT_ID / CKPT_DIR / SEED / DATASET / LOG_PREFIX 原样转给 eval_shard.sh（tmux 会话不继承调用方环境，必须显式写进命令）。
#   多轮（换 SEED 或换 DATASET）必须同时换 LOG_PREFIX 与 PORT_BASE：会话名 / 日志名 / server.log 只含 LOG_PREFIX 与分片名，
#   沿用同一个会让驱动日志 tee -a 追加成一份、merge 的墙钟与 TIMING 串轮。
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
DATASET="${DATASET:-test}"           # benchmark split：test（默认，历史行为）/ val / train；原样转给 eval_shard.sh
LOG_PREFIX="${LOG_PREFIX:-ev40k}"
PORT_BASE="${PORT_BASE:-8031}"
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
      echo "w${k} ${k} SHARD_ID=w${k} K=${k} TASKS=${TASKS_ALL} EP_STRIDE=${WORKERS} EP_START=${k} EP_COUNT=${EP_COUNT:-0}"
    done
  elif [[ "${MODE}" == "task" ]]; then
    for row in "${TASK_SHARDS[@]}"; do
      read -r task kk gpu <<<"${row}"
      echo "${task}-${kk} ${gpu} TASK=${task} K=${kk} EP_COUNT=${EP_COUNT:-25}"
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
  ENVS="${EXTRA} GPU=${GPU} PORT=${PORT} RUN_NAME=${RUN_NAME} CKPT_ID=${CKPT_ID} SEED=${SEED} DATASET=${DATASET} LOG_PREFIX=${LOG_PREFIX}"
  [[ -n "${CKPT_DIR}" ]] && ENVS="${ENVS} CKPT_DIR=${CKPT_DIR}"
  CMD="set -o pipefail; ${ENVS} bash scripts/training/legacy-eval/eval_shard.local.sh 2>&1 | tee -a ${LOG}; sleep 2"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "DRY ${SESSION}: ${CMD}"; continue; fi
  tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "bash -c '${CMD}'"
  echo "起 ${SESSION}: shard=${SHARD} gpu=${GPU} port=${PORT} ${EXTRA} run=${RUN_NAME} seed=${SEED} dataset=${DATASET} ckpt=${CKPT_DIR:-<默认>/${CKPT_ID}} log=${LOG}"
done < <(build_rows)
tmux ls 2>/dev/null | grep "^${LOG_PREFIX}-" || true
