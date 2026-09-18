#!/usr/bin/env bash
# 首300步的500ms八卡采样；trace完整落盘后由本轮控制器请求停止，最长1800秒。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
CHECK_HEAD="${1:?完整训练提交}"
RUN=v2-1600ep-m8x8-modul-motion-b128-80k
REC="$PWD/v1-store/bench/$RUN"
LOG="$PWD/v1-store/logs/mv2-dense.driver.log"
test ! -e "$LOG" || exit 2
body() (
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  test ! -e "$REC/metrics.jsonl"
  test ! -e "$REC/gpu_util_500ms.csv"
  test ! -e "$REC/dense_stop_requested.txt"
  mkdir -p "$REC"
  printf 'CHECK_HEAD=%s\nSTART_UTC=%s\n' "$CHECK_HEAD" "$(date -u +%FT%TZ)"
  timeout 1800 nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 \
    > "$REC/gpu_util_500ms.csv" 2> "$REC/gpu_util_500ms.err" &
  sampler_pid=$!
  printf '%s\n' "$sampler_pid" > "$REC/dense_timeout.pid"
  printf 'DENSE_TIMEOUT_PID=%s\n' "$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
  sampler_rc=0
  wait "$sampler_pid" || sampler_rc=$?
  trap - EXIT
  printf 'SAMPLER_EXIT_CODE=%s\n' "$sampler_rc"
  if [[ "$sampler_rc" == 124 ]]; then
    printf 'DENSE_CAPTURE=PASS stop=1800_seconds\n'
  elif [[ "$sampler_rc" == 143 && -f "$REC/dense_stop_requested.txt" ]] \
      && [[ "$(cat "$REC/dense_stop_requested.txt")" == after_step_299_trace_complete ]]; then
    printf 'DENSE_CAPTURE=PASS stop=after_step_299_trace_complete\n'
  else
    printf 'DENSE_CAPTURE=FAIL unexpected_sampler_exit=%s\n' "$sampler_rc"
    exit 1
  fi
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
