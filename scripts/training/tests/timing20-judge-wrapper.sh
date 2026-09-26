#!/usr/bin/env bash
# 未执行的CPU judge外层候选；复用同一故障传播函数，不加载模型或改变训练。
set -uo pipefail
if [ "$#" -ne 9 ]; then
  printf '用法：bash <command-file> <command-sha> <common-wrapper> <common-sha> <head> <off-run> <on-run> <out-json> <session> <log>\n' >&2
  exit 2
fi
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$MAIN/v1-store"
COMMAND_FILE=$0 COMMAND_SHA=$1 COMMON=$2 COMMON_SHA=$3 TRAIN_HEAD=$4 OFF_RUN=$5 ON_RUN=$6 OUT=$7 SESSION=$8 LOG=$9
JUDGE_WRAPPER_PID=$BASHPID
[[ "$COMMAND_SHA" =~ ^[0-9a-f]{64}$ && "$COMMON_SHA" =~ ^[0-9a-f]{64}$ && "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$OFF_RUN" =~ ^timing20-(full|count)-off-0926-[A-Za-z0-9_-]+$ \
    && "$ON_RUN" =~ ^timing20-(full|count)-on-0926-[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$SESSION" =~ ^orig80k-timing20-[A-Za-z0-9_-]+$ && "$LOG" == "$STORE/logs/$SESSION.wrapper.log" ]] || exit 2
for path in "$COMMAND_FILE" "$COMMON"; do
  [[ "$path" == "$STORE/"* && "$(realpath "$path")" == "$path" && -f "$path" ]] || exit 2
done
actual=$(sha256sum -- "$COMMON") || exit 2
[[ "${actual%% *}" == "$COMMON_SHA" ]] || exit 2
source "$COMMON"
OFF_REC="$STORE/bench/orig80k/$OFF_RUN" ON_REC="$STORE/bench/orig80k/$ON_RUN"
OFF_LOG="$STORE/logs/orig80k-$OFF_RUN.wrapper.log" ON_LOG="$STORE/logs/orig80k-$ON_RUN.wrapper.log"
export UV_CACHE_DIR="$STORE/cache/uv" UV_PROJECT_ENVIRONMENT="$MAIN/.venv" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OPENPI_DATA_HOME="$STORE/models" HF_HOME="$STORE/cache/hf" XDG_CACHE_HOME="$STORE/cache/xdg"
cd "$MAIN" || exit 2
judge_action() {
  printf 'WRAPPER_SESSION=%s\nWRAPPER_PID=%s\nWRAPPER_BODY_PID=%s\nWRAPPER_COMMAND_FILE=%s\nWRAPPER_COMMAND_SHA256_EXPECTED=%s\nWRAPPER_TRAIN_HEAD=%s\nWRAPPER_START_UTC=%s\n' \
    "$SESSION" "$JUDGE_WRAPPER_PID" "$BASHPID" "$COMMAND_FILE" "$COMMAND_SHA" "$TRAIN_HEAD" "$(date -u +%FT%TZ)" || return
  check_command_sha || return
  verify_end || return
  test ! -e "$OUT" && test ! -L "$OUT" || return 2
  local argv=(env -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
    uv run --no-sync python "$MAIN/scripts/training/tests/check_orig80k_timing_equiv.py" judge
    --off-records "$OFF_REC" --on-records "$ON_REC"
    --off-completion "$OFF_REC/completion.json" --on-completion "$ON_REC/completion.json"
    --off-wrapper-log "$OFF_LOG" --on-wrapper-log "$ON_LOG" --head "$TRAIN_HEAD" --out "$OUT")
  command_line JUDGE_COMMAND "${argv[@]}" || return
  "${argv[@]}"
  local code=$?
  printf 'JUDGE_EXIT_CODE=%s\n' "$code" || return
  if [ "$code" -ne 0 ]; then return "$code"; fi
  local digest
  digest=$(sha256sum -- "$OUT") || return
  printf 'JUDGE_RESULT_SHA256=%s\n' "${digest%% *}" || return
  verify_end
}
logged_workflow "$LOG" judge_action
