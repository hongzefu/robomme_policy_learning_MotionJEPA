#!/usr/bin/env bash
# 未执行的20步真保存off/on外层候选；正式使用前须先通过P1/100闸门并形成独立Beta。

command_line() {
  local field=$1 rendered
  shift
  printf -v rendered '%q ' "$@" || return
  printf '%s=%s\n' "$field" "${rendered% }"
}

check_command_sha() {
  local actual
  actual=$(sha256sum -- "$COMMAND_FILE") || return
  actual=${actual%% *}
  printf 'WRAPPER_COMMAND_SHA256_ACTUAL=%s\n' "$actual" || return
  [[ "$actual" == "$COMMAND_SHA" ]] || return 2
}

verify_start() {
  check_command_sha || return
  test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || return 2
  test -z "$(git -C "$MAIN" status --porcelain)" || return 2
  local path
  for path in "$REC" "$RUN_ROOT" "$DRIVER_LOG"; do
    test ! -e "$path" && test ! -L "$path" || return 2
  done
}

run_runner() {
  local argv=(env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME
    TRAIN_HEAD="$TRAIN_HEAD" HISTORY_CONFIG_SHA256="$HISTORY_SHA" NORM_STATS_SHA256="$NORM_SHA"
    ORIG80K_SMOKE_EQ_MODE="$TIMING" bash "$MAIN/scripts/training/prod/run_orig80k.sh"
    smoke "$RUN" "$GPUS" "$LIB" "$ASSETS")
  command_line RUNNER_COMMAND "${argv[@]}" || return
  "${argv[@]}"
}

run_completion() {
  local argv=(env -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
    uv run --no-sync python "$MAIN/scripts/training/tests/check_orig80k_completion.py"
    --mode smoke --records "$REC" --run-root "$RUN_ROOT" --log "$DRIVER_LOG"
    --run "$RUN" --head "$TRAIN_HEAD" --out "$COMPLETION")
  command_line COMPLETION_COMMAND "${argv[@]}" || return
  "${argv[@]}"
}

emit_identity() {
  # 只读小型真实恢复结果；这里不导入项目，不以预计UUID替代实际运行身份。
  env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
    uv run --no-sync python - "$COMPLETION" "$RUN" "$TRAIN_HEAD" <<'PY'
import hashlib, json, pathlib, sys, uuid
path = pathlib.Path(sys.argv[1])
raw = path.read_bytes()
data = json.loads(raw)
if (data['status'], data['mode'], data['run_name'], data['head'], data['state_step'], data['final'], data['checkpoints']) != ('PASS', 'smoke', sys.argv[2], sys.argv[3], 20, 19, [19]):
    raise ValueError('真实恢复结果身份或20步范围不符')
uuid.UUID(data['run_uuid'])
print('WRAPPER_RUN_UUID=' + data['run_uuid'])
print('WRAPPER_COMPLETION_SHA256=' + hashlib.sha256(raw).hexdigest())
PY
}

verify_end() {
  test "$(git -C "$MAIN" rev-parse HEAD)" = "$TRAIN_HEAD" || return 2
  test -z "$(git -C "$MAIN" status --porcelain)" || return 2
}

workflow() {
  printf 'WRAPPER_SESSION=%s\nWRAPPER_PID=%s\nWRAPPER_BODY_PID=%s\nWRAPPER_RUN=%s\nWRAPPER_TIMING=%s\nWRAPPER_TRAIN_HEAD=%s\nWRAPPER_COMMAND_FILE=%s\nWRAPPER_COMMAND_SHA256_EXPECTED=%s\nWRAPPER_START_UTC=%s\n' \
    "$SESSION" "$WRAPPER_PID" "$BASHPID" "$RUN" "$TIMING" "$TRAIN_HEAD" "$COMMAND_FILE" "$COMMAND_SHA" "$(date -u +%FT%TZ)" || return
  verify_start || return
  local runner_rc completion_codes identity_rc end_rc
  run_runner
  runner_rc=$?
  printf 'RUNNER_EXIT_CODE=%s\n' "$runner_rc" || return
  if [ "$runner_rc" -ne 0 ]; then return "$runner_rc"; fi
  # records只由runner创建，外层禁止提前mkdir或追加driver.log。
  test -d "$REC" && test ! -e "$COMPLETION" && test ! -L "$COMPLETION" || return 2
  (set -o noclobber; : > "$COMPLETION_LOG") || return 2
  run_completion 2>&1 | tee "$COMPLETION_LOG"
  completion_codes=("${PIPESTATUS[@]}")
  printf 'COMPLETION_EXIT_CODE=%s\nCOMPLETION_TEE_EXIT_CODE=%s\n' \
    "${completion_codes[0]}" "${completion_codes[1]}" || return
  if [ "${completion_codes[1]}" -ne 0 ]; then return "${completion_codes[1]}"; fi
  if [ "${completion_codes[0]}" -ne 0 ]; then return "${completion_codes[0]}"; fi
  emit_identity
  identity_rc=$?
  printf 'WRAPPER_IDENTITY_EXIT_CODE=%s\n' "$identity_rc" || return
  if [ "$identity_rc" -ne 0 ]; then return "$identity_rc"; fi
  verify_end
  end_rc=$?
  printf 'WRAPPER_VERSION_EXIT_CODE=%s\n' "$end_rc" || return
  return "$end_rc"
}

logged_workflow() {
  local log=$1
  shift
  (set -o noclobber; : > "$log") || return 2
  local command_codes footer_codes rc
  "$@" 2>&1 | tee "$log"
  command_codes=("${PIPESTATUS[@]}")
  rc=${command_codes[0]}
  if [ "${command_codes[1]}" -ne 0 ]; then rc=${command_codes[1]}; fi
  printf 'WRAPPER_COMMAND_EXIT=%s\nWRAPPER_TEE_EXIT=%s\nWRAPPER_END_UTC=%s\n' \
    "${command_codes[0]}" "${command_codes[1]}" "$(date -u +%FT%TZ)" | tee -a "$log"
  footer_codes=("${PIPESTATUS[@]}")
  if [ "${footer_codes[0]}" -ne 0 ]; then rc=${footer_codes[0]}; fi
  if [ "${footer_codes[1]}" -ne 0 ]; then rc=${footer_codes[1]}; fi
  if ! printf 'WRAPPER_FOOTER_PRINTF_EXIT=%s\nWRAPPER_FOOTER_TEE_EXIT=%s\nWRAPPER_EXIT_CODE=%s\n' \
    "${footer_codes[0]}" "${footer_codes[1]}" "$rc" >> "$log"; then
    printf '外层最终退出记录追加失败：%s\n' "$log" >&2
    return 1
  fi
  return "$rc"
}

main() {
  set -uo pipefail
  if [ "$#" -ne 10 ]; then
    printf '用法：bash <command-file> <command-sha> <train-head> <run> <off|on> <gpus> <lib> <assets> <norm-sha> <session> <wrapper-log>\n' >&2
    return 2
  fi
  local MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
  local COMMAND_FILE=$0 COMMAND_SHA=$1 TRAIN_HEAD=$2 RUN=$3 TIMING=$4 GPUS=$5 LIB=$6 ASSETS=$7 NORM_SHA=$8 SESSION=$9
  local WRAPPER_LOG=${10} WRAPPER_PID=$BASHPID
  local STORE="$MAIN/v1-store" HISTORY_SHA=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a
  local REC="$STORE/bench/orig80k/$RUN" RUN_ROOT="$STORE/train-runs/mme_vla_suite/$RUN"
  local DRIVER_LOG="$STORE/logs/$RUN.driver.log" COMPLETION="$REC/completion.json" COMPLETION_LOG="$REC/completion.summary.log"
  [[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ && "$COMMAND_SHA" =~ ^[0-9a-f]{64}$ && "$NORM_SHA" =~ ^[0-9a-f]{64}$ ]] || return 2
  [[ "$RUN" =~ ^timing20-(full|count)-(off|on)-0926-[A-Za-z0-9_-]+$ && "$TIMING" =~ ^(off|on)$ ]] || return 2
  [[ ( "$RUN" == "timing20-full-$TIMING-0926-"* || "$RUN" == "timing20-count-$TIMING-0926-"* ) \
      && "$SESSION" =~ ^orig80k-timing20-[A-Za-z0-9_-]+$ ]] || return 2
  case "$RUN" in
    timing20-full-*)
      [[ "$GPUS" == 0,1,2,3 && "$LIB" == "$STORE/datasets/16task-pub-1600ep" \
          && "$ASSETS" == "$STORE/train-assets/mme_vla_suite" \
          && "$NORM_SHA" == f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5 ]] || return 2 ;;
    timing20-count-*)
      [[ "$GPUS" == 4,5,6,7 && "$LIB" == "$STORE/datasets/4task-counting-pub-400ep" \
          && "$ASSETS" == "$STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep" \
          && "$NORM_SHA" == a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9 ]] || return 2 ;;
  esac
  [[ "$COMMAND_FILE" == "$STORE/"* && "$(realpath "$COMMAND_FILE")" == "$COMMAND_FILE" ]] || return 2
  [[ "$WRAPPER_LOG" == "$STORE/logs/$SESSION.wrapper.log" && "$(realpath "$STORE/logs")" == "$STORE/logs" ]] || return 2
  cd "$MAIN" || return
  export UV_CACHE_DIR="$STORE/cache/uv" UV_PROJECT_ENVIRONMENT="$MAIN/.venv" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  export OPENPI_DATA_HOME="$STORE/models" HF_HOME="$STORE/cache/hf" XDG_CACHE_HOME="$STORE/cache/xdg"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC
  # 不覆盖继承的JAX_ENABLE_X64；实际为True时由runner拒绝，不能静默修正输入。
  logged_workflow "$WRAPPER_LOG" workflow
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
