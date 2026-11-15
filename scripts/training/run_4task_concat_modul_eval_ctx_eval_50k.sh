#!/usr/bin/env bash
# 1114 4task RGB+state 下游 VLA 正式串行链（计划 B 表 S3 行、E 节正式链）：M → E1 → C → E2，全部 8 卡、串行。
#   M ：modulation concat 1×1024 训练 50k（NUM_TRAIN_STEPS=50001）→ 等增量上传 INC_END → 同步 S5 stage 模式
#       （O16_MODE=stage，不联网、不持上传锁）→ 后台 tmux v16-up-m 起 S5 upload 模式 → 对 stage/50000 严格恢复
#       （CKPT_DELIVERY_READY=PASS step=50000）
#   E1：692 局评估 seed 7，ckpt 为严格恢复通过的 v1-store/exports/hf-ckpt-<run>/stage/50000
#   C ：context concat 1×1024，步骤同 M（v16-up-c）
#   E2：同 E1
# 任何失败都写 $FINAL_ROOT/pending/<id>.json、打印 WAIT_USER_DECISION 并非零退出，不启动下一段；本脚本从不写 decisions/。
# 分段收据（计划 C05 / A R5）：每段成功后写 $FINAL_ROOT/receipts/<id>.json，记该段运行时的 sha、日志路径与日志 sha256；
#   START_STAGE=E1|C|E2 接续时前段一律按各自收据记录的 sha 核对（日志里本段自记的 *_HEAD= 行、final record 的 head），
#   不用当前 TRAIN_HEAD 回填；当前 TRAIN_HEAD 只用于本次新跑的段。
# DRY=1：只做只读核对（参数、START_STAGE 前段收据），打印各段将执行的命令串，不训练、不起 tmux、不写任何文件。
set -euo pipefail
[[ $# == 0 ]] || { echo '串行入口不接受参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
: "${TRAIN_HEAD:?必须指定TRAIN_HEAD}"
: "${FINAL_ROOT:?必须指定FINAL_ROOT}"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo 'TRAIN_HEAD必须为40位SHA' >&2; exit 2; }
[[ "$FINAL_ROOT" == /* ]] || { echo 'FINAL_ROOT必须是绝对路径' >&2; exit 2; }
# 单条入口的短测／dry-run 开关不得漏进正式链。
[[ -z ${MAX_STEPS+x} && -z ${DRY_RUN+x} ]] || { echo 'CHAIN=STOP reason=formal_only'; exit 2; }
DRY=${DRY:-0}
[[ $DRY == 0 || $DRY == 1 ]] || { echo 'DRY仅允许0或1' >&2; exit 2; }
START_STAGE=${START_STAGE:-M}
case "$START_STAGE" in M|E1|C|E2) ;; *) echo 'START_STAGE仅允许M、E1、C、E2' >&2; exit 2 ;; esac

CONFIG=mme_vla_suite_4task_b128_50k
NUM_TRAIN_STEPS=50001
DELIVERY_STEP=50000
CKPT_STEPS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000"
SEED=7
LOG="$ROOT/v1-store/logs/v16-4trs-vla"
CONTRACT="$ROOT/scripts/training/check_4task_stage2_launch.py"
SINGLE="$ROOT/scripts/training/run_orig16_stage2_60k.sh"
POOL="$ROOT/scripts/evaluation/local_692_pool.sh"
DELIVERY="$ROOT/scripts/training/check_stage2_checkpoint.py"
POLL_S=${CHAIN_POLL_S:-30}
MOD_H=perceptual-framesamp-modul-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml
CTX_H=perceptual-framesamp-ctx-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml
DIAG='^\r*Step [0-9]+:.*(loss|grad_norm|llm_grad_norm|param_norm|mem_enc_norm)=[-+]?(nan|inf)(,|[[:space:]]|$)|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError|^TRAIN_(NUMERICS|DIAGNOSTIC)=FAIL'

# 有 attempt 清单时取其中 run（合同再校验其登记），否则用固定正式名。
role_run() {
  local manifest=$1 fixed=$2
  if [[ -z $manifest ]]; then echo "$fixed"; return; fi
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$manifest"
}
RM=$(role_run "${ATTEMPT_MANIFEST_MODUL:-}" v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k)
RC=$(role_run "${ATTEMPT_MANIFEST_CTX:-}" v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k)
for run in "$RM" "$RC"; do
  [[ "$run" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "attempt run非法: $run" >&2; exit 2; }
done

# 角色表：短名 m/c → run、history、S5 导出入口、attempt 清单、评估 RUN_NAME。
declare -A RUN_OF=([m]="$RM" [c]="$RC")
declare -A HIST_OF=([m]="$MOD_H" [c]="$CTX_H")
declare -A EXPORT_OF=([m]="$ROOT/scripts/dataset/hf_export/run_4task_modul8x8_concat_50k_ckpt_export.sh"
                      [c]="$ROOT/scripts/dataset/hf_export/run_4task_ctx8x8_concat_50k_ckpt_export.sh")
declare -A ATTEMPT_OF=([m]="${ATTEMPT_MANIFEST_MODUL:-}" [c]="${ATTEMPT_MANIFEST_CTX:-}")
declare -A EVAL_OF=([m]="local692-s2rgbstate-modul-concat1024-50k-s$SEED" [c]="local692-s2rgbstate-ctx-concat1024-50k-s$SEED")

log_of() { echo "$LOG/$1.log"; }
ckpt_root_of() { echo "$ROOT/v1-store/train-runs/$CONFIG/$1"; }
stage_of() { echo "$ROOT/v1-store/exports/hf-ckpt-$1/stage"; }
receipt_of() { echo "$FINAL_ROOT/receipts/$1.json"; }
q() { printf '%q ' "$@"; }

wait_user() {
  local run=$1 stage=$2 evidence=$3 code=${4:-3}
  if [[ $DRY == 0 ]]; then
    mkdir -p "$FINAL_ROOT/pending"
    # 已有同 ID 待决记录时保留原证据，不覆盖。
    "$PY" -c 'import json,sys,datetime
run, stage, evidence, head, path = sys.argv[1:]
body = dict(run=run, stage=stage, evidence=evidence, train_head=head,
            utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
try:
    with open(path, "x") as stream:
        json.dump(body, stream, indent=1)
except FileExistsError:
    pass' "$run" "$stage" "$evidence" "$TRAIN_HEAD" "$FINAL_ROOT/pending/$run-$stage.json"
  fi
  echo "CHAIN=STOP stage=$stage run=$run"
  echo "WAIT_USER_DECISION run=$run stage=$stage evidence=$evidence"
  exit "$code"
}

# 未决：pending/<id>-*.json 存在而 decisions/ 下无同名文件。
has_unresolved() {
  local item
  for item in "$FINAL_ROOT/pending/$1"-*.json; do
    [[ -e $item ]] || continue
    [[ -f "$FINAL_ROOT/decisions/$(basename "$item")" ]] || return 0
  done
  return 1
}

final_record_ok() {   # $1 run，$2 该段自记的训练 sha
  JAX_PLATFORMS=cpu "$PY" "$CONTRACT" --check-final-record "$FINAL_ROOT/$1/final-record/final.json" \
    --run "$1" --steps "$NUM_TRAIN_STEPS" --train-head "$2"
}

# checkpoint 目录恰为 CKPT_STEPS 共 10 份（多一份、少一份、符号链接都不行）。
ckpt_set_ok() {
  local root actual
  root=$(ckpt_root_of "$1")
  [[ -d "$root" && ! -L "$root" ]] || return 1
  actual=$(find "$root" -mindepth 1 -maxdepth 1 -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' | sort -n | tr '\n' ' ')
  [[ "$actual" == "$CKPT_STEPS " ]] || return 1
  ! find "$root" -mindepth 1 -maxdepth 1 -type l -regextype posix-extended -regex '.*/[0-9]+' | grep -q .
}

# 日志里本段自记的 sha：<KEY>=<40 位> 恰好一行。
self_head() {
  local key=$1 log=$2 lines
  lines=$(LC_ALL=C grep -aE "^$key=[0-9a-f]{40}$" "$log" || true)
  [[ -n $lines && $(wc -l <<< "$lines") == 1 ]] || return 1
  echo "${lines#*=}"
}

write_receipt() {   # id kind run head log line
  "$PY" - "$@" "$(receipt_of "$1")" <<'RECEIPT_PY'
import datetime, hashlib, json, os, sys
rid, kind, run, head, log, line, path = sys.argv[1:]
os.makedirs(os.path.dirname(path), exist_ok=True)
body = dict(schema="v16-chain-receipt-v1", id=rid, kind=kind, run=run, head=head, log=log,
            log_sha256=hashlib.sha256(open(log, "rb").read()).hexdigest(), line=line,
            utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
with open(path, "x") as stream:
    json.dump(body, stream, indent=1)
print(f"CHAIN_RECEIPT=PASS id={rid} kind={kind} run={run} head={head}")
RECEIPT_PY
}

# 读收据并核：schema、id、kind、run 一致，日志 sha256 未变；输出收据里的 head 与 log。
read_receipt() {   # id kind run
  "$PY" - "$(receipt_of "$1")" "$1" "$2" "$3" <<'READ_PY'
import hashlib, json, re, sys
path, rid, kind, run = sys.argv[1:]
body = json.load(open(path))
assert body.get("schema") == "v16-chain-receipt-v1" and body.get("id") == rid, "收据身份不符"
assert body.get("kind") == kind and body.get("run") == run, "收据段类型或run不符"
assert re.fullmatch(r"[0-9a-f]{40}", str(body.get("head"))), "收据sha非法"
assert hashlib.sha256(open(body["log"], "rb").read()).hexdigest() == body.get("log_sha256"), "收据所指日志已变动"
print(body["head"], body["log"])
READ_PY
}

# 前段核对（START_STAGE 接续用；只读）：训练段、交付段、评估段各按收据自记 sha 核。
verify_train() {   # short
  local run=${RUN_OF[$1]} out head log self
  out=$(read_receipt "$run-train" train "$run" 2>&1) || wait_user "$run" resume_train_receipt "$out"
  read -r head log <<< "$out"
  self=$(self_head TRAIN_HEAD "$log") || wait_user "$run" resume_train_head "$log"
  [[ $self == "$head" ]] || wait_user "$run" resume_train_head_mismatch "$log"
  if has_unresolved "$run" || [[ $(tail -n 1 "$log") != EXIT_CODE=0 ]] || ! LC_ALL=C grep -aqxF "RUN=$run" "$log" \
      || ! ckpt_set_ok "$run" || ! final_record_ok "$run" "$head" || LC_ALL=C grep -aqiP "$DIAG" "$log"; then
    wait_user "$run" resume_check "$log"
  fi
  printf 'CHAIN_RESUME=PASS stage=train run=%s head=%s\n' "$run" "$head"
}
verify_delivery() {   # short
  local run=${RUN_OF[$1]} out head log self
  out=$(read_receipt "$run-delivery" delivery "$run" 2>&1) || wait_user "$run" resume_delivery_receipt "$out"
  read -r head log <<< "$out"
  self=$(self_head DELIVERY_HEAD "$log") || wait_user "$run" resume_delivery_head "$log"
  [[ $self == "$head" ]] || wait_user "$run" resume_delivery_head_mismatch "$log"
  LC_ALL=C grep -aqE "^CKPT_DELIVERY_READY=PASS step=$DELIVERY_STEP( |$)" "$log" || wait_user "$run" resume_delivery "$log"
  [[ -d "$(stage_of "$run")/$DELIVERY_STEP" ]] || wait_user "$run" resume_stage_missing "$(stage_of "$run")/$DELIVERY_STEP"
  printf 'CHAIN_RESUME=PASS stage=delivery run=%s head=%s\n' "$run" "$head"
}
verify_eval() {   # short
  local name=${EVAL_OF[$1]} run=${RUN_OF[$1]} out head log self
  out=$(read_receipt "$name-eval" eval "$run" 2>&1) || wait_user "$run" resume_eval_receipt "$out"
  read -r head log <<< "$out"
  self=$(self_head EVAL_HEAD "$log") || wait_user "$run" resume_eval_head "$log"
  [[ $self == "$head" ]] || wait_user "$run" resume_eval_head_mismatch "$log"
  eval692_ok "$log" || wait_user "$run" resume_eval "$log"
  printf 'CHAIN_RESUME=PASS stage=eval run=%s eval=%s head=%s\n' "$run" "$name" "$head"
}

# 评估日志恰有 1 行 EVAL692=，且为 PASS。
eval692_ok() {
  [[ $(LC_ALL=C grep -ac '^EVAL692=' "$1") == 1 ]] && LC_ALL=C grep -aqE '^EVAL692=PASS( |$)' "$1"
}

# 先写本段自记行再 tee 追加命令输出；日志已存在即拒绝（不覆盖旧证据）。
open_log() {   # log, then lines...
  local log=$1; shift
  [[ ! -e $log ]] || return 1
  ( set -o noclobber; printf '%s\n' "$@" > "$log" )
}

# ── 各段命令串 ────────────────────────────────────────────────────────────────
train_env() {   # short → env 参数
  local s=$1 attempt=${ATTEMPT_OF[$1]}
  if [[ -n $attempt ]]; then printf '%s\n' "ATTEMPT_MANIFEST=$attempt"; else printf '%s\n' -u ATTEMPT_MANIFEST; fi
  printf '%s\n' "HISTORY=${HIST_OF[$s]}" "RUN_NAME=${RUN_OF[$s]}" "FINAL_RECORD_DIR=$FINAL_ROOT/${RUN_OF[$s]}/final-record" \
    "TRAIN_HEAD=$TRAIN_HEAD"
}
export_env() {   # short mode
  local attempt=${ATTEMPT_OF[$1]}
  if [[ -n $attempt ]]; then printf '%s\n' "ATTEMPT_MANIFEST=$attempt"; else printf '%s\n' -u ATTEMPT_MANIFEST; fi
  printf '%s\n' "O16_MODE=$2"
}
upload_tmux_cmd() {   # short
  local s=$1 ulog="$LOG/up-$1-final.log" args
  mapfile -t args < <(export_env "$s" upload)
  printf 'set -o pipefail; PYTHONUNBUFFERED=1 env %sbash %s 2>&1 | tee %s; echo "EXIT_CODE=$?" >> %s' \
    "$(q "${args[@]}")" "$(q "${EXPORT_OF[$s]}")" "$(q "$ulog")" "$(q "$ulog")"
}
delivery_cmd() {   # short → argv
  local run=${RUN_OF[$1]} stage
  stage=$(stage_of "$run")
  printf '%s\n' env CUDA_VISIBLE_DEVICES=0 "$PY" "$DELIVERY" --mode delivery --delivery-step "$DELIVERY_STEP" \
    --checkpoint "$stage/$DELIVERY_STEP" --expected-manifest "$stage/manifests/step-$DELIVERY_STEP.json" \
    --online-gpu 1 --out "$LOG/delivery-$1.json"
}
eval_cmd() {   # short → argv
  local s=$1
  printf '%s\n' env "RUN_NAME=${EVAL_OF[$s]}" "POLICY=${HIST_OF[$s]%.yaml}" "POLICY_CONFIG=$CONFIG" \
    EXPECT_MOTION_BUDGET=96 EXPECT_MOTION_STRIDE=32 bash "$POOL" "$(stage_of "${RUN_OF[$s]}")/$DELIVERY_STEP" "$SEED"
}

# ── 训练段：训练 → INC_END → stage → upload 后台 → delivery ─────────────────────
inc_wait() {   # short run：等增量上传出 INC_END 或其会话退出；不阻塞正式链
  local s=$1 run=$2 ilog="$LOG/up-$1.log" sess="v16-up-$1"
  while :; do
    if [[ -f $ilog ]] && LC_ALL=C grep -aqE "^INC_END run=$run( |$)" "$ilog"; then
      printf 'CHAIN_INC_END=PASS run=%s log=%s\n' "$run" "$ilog"; break
    fi
    if ! tmux has-session -t "=$sess" 2>/dev/null; then
      if [[ -f $ilog ]] && LC_ALL=C grep -aqE "^INC_END run=$run( |$)" "$ilog"; then
        printf 'CHAIN_INC_END=PASS run=%s log=%s\n' "$run" "$ilog"
      else
        # 增量会话已退出却无 INC_END：只记录，final upload 会复用 stage 全量上传读回。
        printf 'CHAIN_DIAGNOSTIC=FAIL check=inc_end action=record_only run=%s log=%s\n' "$run" "$ilog"
      fi
      break
    fi
    sleep "$POLL_S"
  done
  # 增量会话退出（释放 upload.lock 与会话名）后才复用该名起 final upload；最多再等 20 分钟。
  local waited=0
  while tmux has-session -t "=$sess" 2>/dev/null; do
    (( waited >= 1200 )) && return 1
    sleep "$POLL_S"; waited=$(( waited + POLL_S ))
  done
}

run_train() {   # short
  local s=$1 run=${RUN_OF[$1]} log stage tlog ulog dlog rc env_args head
  log=$(log_of "$run"); stage=$(stage_of "$run")
  mapfile -t env_args < <(train_env "$s")
  if [[ $DRY == 1 ]]; then
    echo "CHAIN_DRY stage=${s^^} train_cmd=env $(q "${env_args[@]}")bash $(q "$SINGLE")"
    echo "CHAIN_DRY stage=${s^^} final_record_cmd=$(q env JAX_PLATFORMS=cpu "$PY" "$CONTRACT" --check-final-record "$FINAL_ROOT/$run/final-record/final.json" --run "$run" --steps "$NUM_TRAIN_STEPS" --train-head "$TRAIN_HEAD")"
    echo "CHAIN_DRY stage=${s^^} ckpt_steps=${CKPT_STEPS// /,} root=$(ckpt_root_of "$run")"
    echo "CHAIN_DRY stage=${s^^} inc_wait=INC_END log=$LOG/up-$s.log session=v16-up-$s"
    mapfile -t args < <(export_env "$s" stage)
    echo "CHAIN_DRY stage=${s^^} stage_cmd=env $(q "${args[@]}")bash $(q "${EXPORT_OF[$s]}") log=$LOG/stage-$s.log"
    echo "CHAIN_DRY stage=${s^^} upload_tmux=tmux new-session -d -s v16-up-$s -c $(q "$ROOT") $(q "$(upload_tmux_cmd "$s")")"
    mapfile -t args < <(delivery_cmd "$s")
    echo "CHAIN_DRY stage=${s^^} delivery_cmd=$(q "${args[@]}") log=$LOG/delivery-$s.log"
    return 0
  fi
  # 1. 训练（单条入口自带合同、日志 TRAIN_HEAD= 行、final record 与 EXIT_CODE 尾行）。
  if env "${env_args[@]}" bash "$SINGLE"; then rc=0; else rc=$?; fi
  [[ $rc == 0 ]] || wait_user "$run" train_exit "$log" "$rc"
  if [[ ! -f "$log" || $(tail -n 1 "$log") != EXIT_CODE=0 ]] || ! LC_ALL=C grep -aqxF "RUN=$run" "$log"; then
    wait_user "$run" missing_log "$log"
  fi
  head=$(self_head TRAIN_HEAD "$log") || wait_user "$run" train_head "$log"
  [[ $head == "$TRAIN_HEAD" ]] || wait_user "$run" train_head_mismatch "$log"
  ckpt_set_ok "$run" || wait_user "$run" ckpt_steps "$(ckpt_root_of "$run")"
  final_record_ok "$run" "$TRAIN_HEAD" || wait_user "$run" final_record "$FINAL_ROOT/$run/final-record/final.json"
  ! LC_ALL=C grep -aqiP "$DIAG" "$log" || wait_user "$run" train_numerics "$log"
  write_receipt "$run-train" train "$run" "$TRAIN_HEAD" "$log" "TRAIN_FINAL_RECORD=PASS run=$run"
  printf 'CHAIN_RUN=PASS run=%s step=%s ckpts=10\n' "$run" "$DELIVERY_STEP"
  # 2. 等增量上传 INC_END（增量退出、释放 upload.lock 与会话名）。
  inc_wait "$s" "$run" || printf 'CHAIN_DIAGNOSTIC=FAIL check=inc_session_alive action=record_only run=%s\n' "$run"
  # 3. 同步 S5 stage 模式：strict_final、冻结源、写 stage/50000、manifests/step-50000.json 与 delivery manifest。
  tlog="$LOG/stage-$s.log"
  open_log "$tlog" "STAGE_HEAD=$TRAIN_HEAD" "RUN=$run" || wait_user "$run" stage_log_exists "$tlog"
  mapfile -t args < <(export_env "$s" stage)
  set +e
  env "${args[@]}" bash "${EXPORT_OF[$s]}" 2>&1 | tee -a "$tlog"
  rc=${PIPESTATUS[0]}
  set -e
  echo "EXIT_CODE=$rc" >> "$tlog"
  [[ $rc == 0 ]] && LC_ALL=C grep -aqE '^STAGE=PASS( |$)' "$tlog" || wait_user "$run" stage_export "$tlog" 3
  [[ -d "$stage/$DELIVERY_STEP" && -f "$stage/manifests/step-$DELIVERY_STEP.json" ]] \
    || wait_user "$run" stage_missing "$stage"
  # 4. 后台 tmux 起 S5 upload 模式（复用第 3 步 stage；CKPT_UPLOAD=PASS steps=10、HF_VERIFY=PASS 由它给出，不阻塞后续）。
  ulog="$LOG/up-$s-final.log"
  if tmux has-session -t "=v16-up-$s" 2>/dev/null || [[ -e $ulog ]]; then
    printf 'CHAIN_DIAGNOSTIC=FAIL check=upload_launch action=main_session run=%s reason=session_or_log_exists log=%s\n' "$run" "$ulog"
  elif tmux new-session -d -s "v16-up-$s" -c "$ROOT" "$(upload_tmux_cmd "$s")"; then
    printf 'CHAIN_UPLOAD_LAUNCHED session=v16-up-%s log=%s\n' "$s" "$ulog"
  else
    printf 'CHAIN_DIAGNOSTIC=FAIL check=upload_launch action=main_session run=%s reason=tmux log=%s\n' "$run" "$ulog"
  fi
  # 5. 对本地 stage 严格恢复（父进程 CUDA_VISIBLE_DEVICES=0、--online-gpu 1）。
  dlog="$LOG/delivery-$s.log"
  open_log "$dlog" "DELIVERY_HEAD=$TRAIN_HEAD" "RUN=$run" || wait_user "$run" delivery_log_exists "$dlog"
  mapfile -t args < <(delivery_cmd "$s")
  set +e
  "${args[@]}" 2>&1 | tee -a "$dlog"
  rc=${PIPESTATUS[0]}
  set -e
  echo "EXIT_CODE=$rc" >> "$dlog"
  [[ $rc == 0 ]] && LC_ALL=C grep -aqE "^CKPT_DELIVERY_READY=PASS step=$DELIVERY_STEP( |$)" "$dlog" \
    || wait_user "$run" delivery "$dlog" 3
  write_receipt "$run-delivery" delivery "$run" "$TRAIN_HEAD" "$dlog" "CKPT_DELIVERY_READY=PASS step=$DELIVERY_STEP"
}

run_eval() {   # short
  local s=$1 run=${RUN_OF[$1]} name=${EVAL_OF[$1]} elog rc args
  elog=$(log_of "$name")
  mapfile -t args < <(eval_cmd "$s")
  if [[ $DRY == 1 ]]; then
    echo "CHAIN_DRY stage=E$([[ $s == m ]] && echo 1 || echo 2) eval_cmd=$(q "${args[@]}") log=$elog"
    return 0
  fi
  open_log "$elog" "EVAL_HEAD=$TRAIN_HEAD" "RUN=$run" "CKPT=$(stage_of "$run")/$DELIVERY_STEP" \
    || wait_user "$run" eval_log_exists "$elog"
  set +e
  ( cd "$ROOT" && "${args[@]}" ) 2>&1 | tee -a "$elog"
  rc=${PIPESTATUS[0]}
  set -e
  echo "EXIT_CODE=$rc" >> "$elog"
  [[ $rc == 0 ]] && eval692_ok "$elog" || wait_user "$run" eval692 "$elog" 3
  write_receipt "$name-eval" eval "$run" "$TRAIN_HEAD" "$elog" "$(LC_ALL=C grep -a '^EVAL692=' "$elog")"
}

echo "CHAIN_START dry=$DRY start_stage=$START_STAGE train_head=$TRAIN_HEAD modul=$RM ctx=$RC config=$CONFIG num_train_steps=$NUM_TRAIN_STEPS $(date -u +%FT%TZ)"
[[ $DRY == 1 ]] || [[ -d "$LOG" && "$(realpath "$LOG")" == "$LOG" ]] || { echo "日志目录必须预先存在且为实体目录: $LOG" >&2; exit 2; }

# START_STAGE 接续：前段按各自收据自记的 sha 核对，通过的段不重跑。
case "$START_STAGE" in
  E1) verify_train m; verify_delivery m ;;
  C)  verify_train m; verify_delivery m; verify_eval m ;;
  E2) verify_train m; verify_delivery m; verify_eval m; verify_train c; verify_delivery c ;;
esac
STAGES=(M E1 C E2)
started=0
for stage in "${STAGES[@]}"; do
  [[ $stage == "$START_STAGE" ]] && started=1
  (( started )) || continue
  case "$stage" in
    M) run_train m ;;
    E1) run_eval m ;;
    C) run_train c ;;
    E2) run_eval c ;;
  esac
done
if [[ $DRY == 1 ]]; then echo "CHAIN=DRY start_stage=$START_STAGE"; else echo "CHAIN=PASS runs=2 evals=2"; fi
