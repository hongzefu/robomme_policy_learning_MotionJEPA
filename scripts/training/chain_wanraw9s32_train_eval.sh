#!/usr/bin/env bash
# 训完直接评估的链脚本（1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.2 链脚本行、runbook RB-E2 / RB-F1）。
# 运行在 tmux r9-train 内（主会话：tmx r9-train "bash scripts/training/chain_wanraw9s32_train_eval.sh $TRAIN_HEAD"）：
#   ① 前台跑正式 runner run_4task_wanraw9s32_50k.sh <TRAIN_HEAD>（内置闸 G0、训练、ckpt 核验）；
#   ② runner 退出码 0、其日志末行 EXIT_CODE=0、且日志含 CKPT_LOAD=PASS step=50000 与 TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10，
#      才另起 detached tmux r9-eval 跑 692 局评估（评估参数与 v11 一致，前缀库 v1-store/demo-prefix/binfill-3b4de03a0b46），
#      显式导出 POLICY / RUN_NAME / EXPECT_MOTION_STRIDE=32 / EXPECT_MOTION_BUDGET=80（写死）；
#   ③ 任一前置不过：打印 CHAIN=STOP reason=<train_exit|ckpt_load|train_final_record|eval_log_exists|tmux_launch>，不起评估，非零退出。
# 评估起来后本脚本即退出（不等评估），主会话另挂 Monitor 盯 v1-store/logs/r9-eval.log。
# CHAIN_DRY=1：不跑 runner、不起 tmux，从 CHAIN_RUNNER_LOG（假日志夹具，必填）读判据，只打印将执行的命令（单测用）。
# R10：不吞退出码，不用 || true / set +e。
set -uo pipefail

usage() { printf '用法: %s <TRAIN_HEAD 40 位 sha>\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] && [ $# -eq 1 ] || usage

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
FORMAL_RUN=v2-1600ep-m8x8-modul-wanraw9s32-b128-50k
CONFIG=mme_vla_suite_b128_80k
RUNNER="$MAIN/scripts/training/run_4task_wanraw9s32_50k.sh"
RUN_ROOT="$MAIN/v1-store/train-runs/$CONFIG/$FORMAL_RUN"
RUNNER_LOG="$MAIN/v1-store/logs/$FORMAL_RUN.runner.log"      # runner 自己 tee 的日志（见 runner 的 LOG）
EVAL_SESSION=r9-eval
EVAL_LOG="$MAIN/v1-store/logs/$EVAL_SESSION.log"
STORE="$MAIN/v1-store/demo-prefix/binfill-3b4de03a0b46"
EVAL_ENV=(POLICY=perceptual-framesamp-modul-8frame-8x8-wanraw9s32 RUN_NAME=local692-wanraw9s32-50k
          EXPECT_MOTION_STRIDE=32 EXPECT_MOTION_BUDGET=80 "STORE=$STORE")
EVAL_CMD="cd $(printf '%q' "$MAIN") && env $(printf '%q ' "${EVAL_ENV[@]}")bash scripts/evaluation/local_692_pool.sh $(printf '%q' "$RUN_ROOT/50000") 7"
TMUX_CMD="set -o pipefail; PYTHONUNBUFFERED=1 bash -c $(printf '%q' "$EVAL_CMD") 2>&1 | tee $(printf '%q' "$EVAL_LOG"); echo \"EXIT_CODE=\$?\" >> $(printf '%q' "$EVAL_LOG")"

stop() { printf 'CHAIN=STOP reason=%s %s\n' "$1" "$2"; exit "$3"; }

DRY="${CHAIN_DRY:-0}"
echo "CHAIN_START dry=$DRY train_head=$TRAIN_HEAD run=$FORMAL_RUN $(date -u +%FT%TZ)"
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

if [ "$DRY" = 1 ]; then
  echo "CHAIN_DRY eval_cmd=$EVAL_CMD"
  echo "CHAIN_DRY tmux new-session -d -s $EVAL_SESSION $TMUX_CMD"
  echo "CHAIN=DRY_EVAL session=$EVAL_SESSION log=$EVAL_LOG"
  exit 0
fi

test ! -e "$EVAL_LOG" || stop eval_log_exists "拒绝覆盖评估日志 $EVAL_LOG" 1
tmux new-session -d -s "$EVAL_SESSION" -c "$MAIN" "$TMUX_CMD" || stop tmux_launch "tmux new-session -s $EVAL_SESSION 失败" 1
echo "CHAIN=EVAL_LAUNCHED session=$EVAL_SESSION log=$EVAL_LOG eval_cmd=$EVAL_CMD $(date -u +%FT%TZ)"
