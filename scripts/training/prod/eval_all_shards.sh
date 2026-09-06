#!/usr/bin/env bash
# 8 卡分片并行起评（留档 docs/training-doc/eval-awsprod40k-b128-motion/）：4 任务 × 2 片（ep 0–24 / 25–49），
# 每片独占一张卡（policy + sidecar + 仿真同卡），一片一个 detached tmux 会话 ev40k-<TASK>-<K>，脚本结束会话自动退出。
# 用法：[ONLY=ButtonUnmask-0] [EP_COUNT=25] bash scripts/training/prod/eval_all_shards.sh
#   ONLY 只起一片（预检用）；已存在同名会话则跳过不重起。断点续评：重跑同一片即续（eval.py 按 progress.json 跳过已评集）。
#   清理只允许 tmux kill-session -t <确切会话名>（AGENTS 7）；本轮会话清单见留档 launch.md。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"
ONLY="${ONLY:-}"
EP_COUNT="${EP_COUNT:-25}"
# TASK K GPU PORT（一片一卡一端口）
SHARDS=(
  "ButtonUnmask 0 0 8031"
  "ButtonUnmask 1 1 8032"
  "VideoUnmask 0 2 8033"
  "VideoUnmask 1 3 8034"
  "ButtonUnmaskSwap 0 4 8035"
  "ButtonUnmaskSwap 1 5 8036"
  "VideoUnmaskSwap 0 6 8037"
  "VideoUnmaskSwap 1 7 8038"
)
mkdir -p "${LOGS_DIR}"
for row in "${SHARDS[@]}"; do
  read -r TASK K GPU PORT <<<"${row}"
  SHARD="${TASK}-${K}"
  SESSION="ev40k-${SHARD}"
  if [[ -n "${ONLY}" && "${ONLY}" != "${SHARD}" ]]; then continue; fi
  if tmux has-session -t "=${SESSION}" 2>/dev/null; then echo "跳过 ${SESSION}（会话已存在）"; continue; fi
  LOG="${LOGS_DIR}/ev40k-${SHARD}.log"
  CMD="set -o pipefail; TASK=${TASK} K=${K} GPU=${GPU} PORT=${PORT} EP_COUNT=${EP_COUNT} bash scripts/training/prod/eval_shard.sh 2>&1 | tee -a ${LOG}; sleep 2"
  tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "bash -c '${CMD}'"
  echo "起 ${SESSION}: task=${TASK} k=${K} gpu=${GPU} port=${PORT} episodes=[$((K * EP_COUNT)),$(((K + 1) * EP_COUNT))) log=${LOG}"
done
tmux ls 2>/dev/null | grep "^ev40k-" || true
