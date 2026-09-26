#!/usr/bin/env bash
# 只在/tmp运行纯shell替身；不调用main、真实runner、uv、tmux或任何Python入口。
set -euo pipefail
source "${1:?须提供待审wrapper文件}"
audit_root=$(mktemp -d /tmp/orig80k-timing20-wrapper.XXXXXXXX)
trap 'rm -rf -- "$audit_root"' EXIT

verify_start() { check_command_sha; }
verify_end() { if [[ "$FAULT" == version ]]; then return 41; fi; }
run_runner() {
  printf 'runner\n' >> "$CALLS"
  if [[ "$FAULT" == runner ]]; then return 7; fi
  mkdir "$REC"
  printf '模拟runner成功\n'
}
run_completion() {
  printf 'completion\n' >> "$CALLS"
  if [[ "$FAULT" == completion ]]; then return 9; fi
  printf '{}\n' > "$COMPLETION"
  printf '模拟恢复成功\n'
}
emit_identity() {
  if [[ "$FAULT" == identity ]]; then return 17; fi
  printf 'WRAPPER_RUN_UUID=00000000-0000-0000-0000-000000000001\nWRAPPER_COMPLETION_SHA256=%064d\n' 0
}
tee() {
  command tee "$@" || return
  if [[ "$FAULT" == main_tee && "$1" == "$WRAPPER_LOG" ]]; then return 23; fi
  if [[ "$FAULT" == completion_tee && "$1" == "$COMPLETION_LOG" ]]; then return 24; fi
  if [[ "$FAULT" == footer_tee && "$1" == -a ]]; then return 29; fi
}
printf() {
  builtin printf "$@" || return
  if [[ "$FAULT" == footer_printf && "$1" == 'WRAPPER_COMMAND_EXIT='* ]]; then return 31; fi
}

for FAULT in success runner completion main_tee completion_tee footer_tee footer_printf identity version self_sha; do
  case_root="$audit_root/$FAULT"
  mkdir "$case_root"
  COMMAND_FILE="$case_root/command with space.command"
  printf '仅供SHA核对的临时载体\n' > "$COMMAND_FILE"
  COMMAND_SHA=$(sha256sum "$COMMAND_FILE"); COMMAND_SHA=${COMMAND_SHA%% *}
  [[ "$FAULT" != self_sha ]] || COMMAND_SHA=$(printf '%064d' 0)
  SESSION=orig80k-timing20-test RUN=timing20-full-off-0926-test TIMING=off
  TRAIN_HEAD=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa WRAPPER_PID=$BASHPID
  REC="$case_root/records" COMPLETION="$REC/completion.json" COMPLETION_LOG="$REC/completion.summary.log"
  WRAPPER_LOG="$case_root/wrapper.log" CALLS="$case_root/calls"
  set +e
  logged_workflow "$WRAPPER_LOG" workflow > "$case_root/stdout"
  actual=$?
  set -e
  case "$FAULT" in success) expected=0 ;; runner) expected=7 ;; completion) expected=9 ;;
    main_tee) expected=23 ;; completion_tee) expected=24 ;; footer_tee) expected=29 ;;
    footer_printf) expected=31 ;; identity) expected=17 ;; version) expected=41 ;; self_sha) expected=2 ;; esac
  test "$actual" -eq "$expected"
  test "$(grep -c '^WRAPPER_EXIT_CODE=' "$WRAPPER_LOG")" = 1
  grep -qx "WRAPPER_EXIT_CODE=$expected" "$WRAPPER_LOG"
  if [[ "$FAULT" == success ]]; then
    for key in RUNNER_EXIT_CODE COMPLETION_EXIT_CODE COMPLETION_TEE_EXIT_CODE WRAPPER_COMMAND_EXIT WRAPPER_TEE_EXIT WRAPPER_FOOTER_PRINTF_EXIT WRAPPER_FOOTER_TEE_EXIT WRAPPER_EXIT_CODE; do
      test "$(grep -c "^$key=" "$WRAPPER_LOG")" = 1
      grep -qx "$key=0" "$WRAPPER_LOG"
    done
    before=$(sha256sum "$WRAPPER_LOG")
    set +e
    logged_workflow "$WRAPPER_LOG" workflow > "$case_root/reuse.stdout" 2>&1
    conflict=$?
    set -e
    test "$conflict" = 2 && test "$(sha256sum "$WRAPPER_LOG")" = "$before"
  elif [[ "$FAULT" == runner ]]; then
    test "$(cat "$CALLS")" = runner
    test ! -e "$COMPLETION_LOG"
  elif [[ "$FAULT" == self_sha ]]; then
    test ! -e "$CALLS"
  fi
  printf 'WRAPPER_FAULT_TEST=PASS case=%s exit=%s\n' "$FAULT" "$actual"
done
printf 'WRAPPER_FAULT_TEST=PASS case=existing_log exit=2\n'
