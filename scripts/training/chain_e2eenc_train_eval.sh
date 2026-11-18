#!/usr/bin/env bash
# E2E-Enc 训完直接评估的链脚本（1117-e2e-enc-train-plan.html 第二部分一节 chain_e2eenc_train_eval.sh 行、RB-E2 / RB-F1）。
# 由 chain_wanraw9s32_train_eval.sh（c0826be）复制。运行在 tmux e2e-train 内（主会话在 RB-E0 报告与 Beta commit 之后起）：
#   ① 前台跑正式 runner run_4task_e2eenc_50k.sh <TRAIN_HEAD>（内置闸 G0、训练、e2e 收尾断言）；
#   ② runner 退出码 0、其日志末行 EXIT_CODE=0、且日志含 CKPT_LOAD=PASS step=50000 与 TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10（G7），
#      才另起 detached tmux e2e-eval 跑 692 局评估：scripts/evaluation/local_692_pool_e2e.sh <ckpt>/50000 7，显式导出
#      POLICY=perceptual-framesamp-modul-8frame-8x8-e2eenc / RUN_NAME / EXPECT_MOTION_STRIDE=16 / EXPECT_MOTION_BUDGET=160 / STORE，
#      以及评估吞吐档 CHUNK_EPISODES（缺省 5）/ EVAL_TIMEOUT（缺省 14400）/ EVAL_SHARDS（32）/ EVAL_PER_GPU（4）/ EVAL_MEM_FRACTION（0.2）
#      ——这五个可由调用环境覆盖（G5b 退为三片时主会话以 EVAL_SHARDS=24 EVAL_PER_GPU=3 EVAL_MEM_FRACTION=0.2 起链）；
#      EVAL692_PLAN_IDENTITY / EVAL692_RESULT_IDENTITY 在评估脚本内（起 GPU 前 / 合并后）；
#   ③ 任一前置不过：打印 CHAIN=STOP reason=<train_exit|ckpt_load|train_final_record|eval_session_exists|tmux_launch>，不起评估，非零退出。
# 修掉 Wan-9 链的 eval_log_exists 停下：评估日志 v1-store/logs/e2e-eval.log 已存在（如主会话预放的占位文件）时不停下、不覆盖，
#   改用第一个不存在的 e2e-eval.r<k>.log（k=1,2,…），打印 CHAIN_EVAL_LOG existing=<旧> using=<新>，主会话按该行挂 Monitor；
#   只有同名 tmux 会话 e2e-eval 仍在时才停（reason=eval_session_exists）。评估目录已存在由评估脚本自身拒绝覆盖。
# 链脚本不起 HF 导出（HF 导出脚本 scripts/dataset/hf_export/run_e2eenc_ckpt_export.sh 由主会话另起 tmux e2e-hf）。
# 评估起来后本脚本即退出（不等评估）。
# CHAIN_DRY=1：不跑 runner、不起 tmux，从 CHAIN_RUNNER_LOG（假日志夹具，必填）读判据，只打印将执行的命令（单测用）；
#   单测另可用 CHAIN_LOG_DIR=<目录> 替换评估日志目录（只允许与 CHAIN_DRY=1 同用）。
# R10：不吞退出码，不用 || true / set +e。
set -uo pipefail

usage() { printf '用法: %s <TRAIN_HEAD 40 位 sha>\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] && [ $# -eq 1 ] || usage

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
FORMAL_RUN=v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k
CONFIG=mme_vla_suite_b128_80k
RUNNER="$MAIN/scripts/training/run_4task_e2eenc_50k.sh"
RUN_ROOT="$MAIN/v1-store/train-runs/$CONFIG/$FORMAL_RUN"
RUNNER_LOG="$MAIN/v1-store/logs/$FORMAL_RUN.runner.log"      # runner 自己 tee 的日志（见 runner 的 LOG）
EVAL_SESSION=e2e-eval
LOG_DIR="$MAIN/v1-store/logs"
STORE="$MAIN/v1-store/demo-prefix/binfill-3b4de03a0b46"
POLICY=perceptual-framesamp-modul-8frame-8x8-e2eenc
EVAL_RUN_NAME=local692-e2eenc-s16-50k

stop() { printf 'CHAIN=STOP reason=%s %s\n' "$1" "$2"; exit "$3"; }

DRY="${CHAIN_DRY:-0}"
if [ -n "${CHAIN_LOG_DIR:-}" ]; then
  [ "$DRY" = 1 ] || stop bad_env "CHAIN_LOG_DIR 只允许与 CHAIN_DRY=1 同用" 2
  LOG_DIR="$CHAIN_LOG_DIR"
fi
CHUNK="${CHUNK_EPISODES:-5}"; TIMEOUT="${EVAL_TIMEOUT:-14400}"
SHARDS="${EVAL_SHARDS:-32}"; PER_GPU="${EVAL_PER_GPU:-4}"; MEMF="${EVAL_MEM_FRACTION:-0.2}"
EVAL_ENV=(POLICY=$POLICY RUN_NAME=$EVAL_RUN_NAME EXPECT_MOTION_STRIDE=16 EXPECT_MOTION_BUDGET=160 "STORE=$STORE"
          CHUNK_EPISODES=$CHUNK EVAL_TIMEOUT=$TIMEOUT EVAL_SHARDS=$SHARDS EVAL_PER_GPU=$PER_GPU EVAL_MEM_FRACTION=$MEMF)
EVAL_CMD="cd $(printf '%q' "$MAIN") && env $(printf '%q ' "${EVAL_ENV[@]}")bash scripts/evaluation/local_692_pool_e2e.sh $(printf '%q' "$RUN_ROOT/50000") 7"

echo "CHAIN_START dry=$DRY train_head=$TRAIN_HEAD run=$FORMAL_RUN $(date -u +%FT%TZ)"
echo "CHAIN_EVAL_PARAMS CHUNK_EPISODES=$CHUNK EVAL_TIMEOUT=$TIMEOUT EVAL_SHARDS=$SHARDS EVAL_PER_GPU=$PER_GPU EVAL_MEM_FRACTION=$MEMF"
if [ "$DRY" = 1 ]; then
  LOGF="${CHAIN_RUNNER_LOG:?CHAIN_DRY=1 必须给 CHAIN_RUNNER_LOG（假日志夹具）}"
  echo "CHAIN_DRY runner_cmd=bash $RUNNER $TRAIN_HEAD"
  rc="$(sed -n 's/^EXIT_CODE=\([0-9][0-9]*\)$/\1/p' "$LOGF" | tail -n 1)"
  rc="${rc:-missing}"
else
  [ -z "${CHAIN_RUNNER_LOG:-}" ] || stop bad_env "CHAIN_RUNNER_LOG 只允许与 CHAIN_DRY=1 同用" 2
  LOGF="$RUNNER_LOG"
  if bash "$RUNNER" "$TRAIN_HEAD"; then rc=0; else rc=$?; fi
  log_rc="$(sed -n 's/^EXIT_CODE=\([0-9][0-9]*\)$/\1/p' "$LOGF" | tail -n 1)"
  echo "CHAIN_RUNNER_DONE rc=$rc log_exit_code=${log_rc:-missing} log=$LOGF"
  [ "$rc" = 0 ] && [ "${log_rc:-missing}" != 0 ] && rc="log:${log_rc:-missing}"
fi

[ "$rc" = 0 ] || stop train_exit "runner_exit=$rc log=$LOGF" 1
grep -qE '^CKPT_LOAD=PASS step=50000( |$)' "$LOGF" || stop ckpt_load "日志缺 CKPT_LOAD=PASS step=50000 log=$LOGF" 1
grep -qE '^TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10( |$)' "$LOGF" || stop train_final_record "日志缺 TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10 log=$LOGF" 1

# 评估日志：已存在（占位）即顺延到第一个空闲的 e2e-eval.r<k>.log，不停下、不覆盖
EVAL_LOG="$LOG_DIR/$EVAL_SESSION.log"
if [ -e "$EVAL_LOG" ]; then
  k=1; while [ -e "$LOG_DIR/$EVAL_SESSION.r$k.log" ]; do k=$((k + 1)); done
  echo "CHAIN_EVAL_LOG existing=$EVAL_LOG using=$LOG_DIR/$EVAL_SESSION.r$k.log"
  EVAL_LOG="$LOG_DIR/$EVAL_SESSION.r$k.log"
fi
TMUX_CMD="set -o pipefail; PYTHONUNBUFFERED=1 bash -c $(printf '%q' "$EVAL_CMD") 2>&1 | tee $(printf '%q' "$EVAL_LOG"); echo \"EXIT_CODE=\$?\" >> $(printf '%q' "$EVAL_LOG")"

if [ "$DRY" = 1 ]; then
  echo "CHAIN_DRY eval_cmd=$EVAL_CMD"
  echo "CHAIN_DRY tmux new-session -d -s $EVAL_SESSION $TMUX_CMD"
  echo "CHAIN=DRY_EVAL session=$EVAL_SESSION log=$EVAL_LOG"
  exit 0
fi

tmux has-session -t "=$EVAL_SESSION" 2>/dev/null && stop eval_session_exists "tmux 会话 $EVAL_SESSION 仍在" 1
tmux new-session -d -s "$EVAL_SESSION" -c "$MAIN" "$TMUX_CMD" || stop tmux_launch "tmux new-session -s $EVAL_SESSION 失败" 1
echo "CHAIN=EVAL_LAUNCHED session=$EVAL_SESSION log=$EVAL_LOG eval_cmd=$EVAL_CMD $(date -u +%FT%TZ)"
