#!/usr/bin/env bash
# 官方16×100两条concat（1×1024）正式60k串行：context先、modulation后，每条8卡。
# 任何失败（非零退出、缺成功日志、缺59999、最终记录失败、数值/OOM诊断命中）都写待决记录、
# 打印WAIT_USER_DECISION并非零退出，不启动下一条；本脚本只写pending，从不写decisions。
set -euo pipefail
[[ $# == 0 ]] || { echo '串行入口不接受参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
: "${TRAIN_HEAD:?必须指定TRAIN_HEAD}"
: "${FINAL_ROOT:?必须指定FINAL_ROOT}"
[[ -z ${MAX_STEPS+x} && -z ${DRY_RUN+x} ]] || { echo 'CHAIN=STOP reason=formal_only'; exit 2; }
START_ROLE=${START_ROLE:-ctx}
[[ $START_ROLE == ctx || $START_ROLE == modul ]] || { echo 'START_ROLE仅允许ctx或modul' >&2; exit 2; }
CTX_H=perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml
MOD_H=perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml
DIAG='^\r*Step [0-9]+:.*(loss|grad_norm|llm_grad_norm|param_norm|mem_enc_norm)=[-+]?(nan|inf)(,|[[:space:]]|$)|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError|^TRAIN_(NUMERICS|DIAGNOSTIC)=FAIL'

# 有attempt清单时取其中run（合同再校验其登记），否则用固定正式名。
role_run() {
  local manifest=$1 fixed=$2
  if [[ -z $manifest ]]; then echo "$fixed"; return; fi
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$manifest"
}
CTX=$(role_run "${ATTEMPT_MANIFEST_CTX:-}" orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k)
MOD=$(role_run "${ATTEMPT_MANIFEST_MODUL:-}" orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k)
for run in "$CTX" "$MOD"; do
  [[ "$run" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "attempt run非法: $run" >&2; exit 2; }
done

wait_user() {
  local run=$1 stage=$2 evidence=$3 code=${4:-3}
  mkdir -p "$FINAL_ROOT/pending"
  # 已有同ID待决记录时保留原证据，不覆盖。
  "$PY" -c 'import json,sys,datetime
run, stage, evidence, head, path = sys.argv[1:]
body = dict(run=run, stage=stage, evidence=evidence, train_head=head,
            utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
try:
    with open(path, "x") as stream:
        json.dump(body, stream, indent=1)
except FileExistsError:
    pass' "$run" "$stage" "$evidence" "$TRAIN_HEAD" "$FINAL_ROOT/pending/$run-$stage.json"
  echo "WAIT_USER_DECISION run=$run stage=$stage evidence=$evidence"
  exit "$code"
}

# 未决：pending/<id>.json存在而decisions/<id>.json不存在。
has_unresolved() {
  local item
  for item in "$FINAL_ROOT/pending/$1"-*.json; do
    [[ -e $item ]] || continue
    [[ -f "$FINAL_ROOT/decisions/$(basename "$item")" ]] || return 0
  done
  return 1
}

final_record_ok() {
  JAX_PLATFORMS=cpu "$PY" "$ROOT/scripts/training/check_orig16_stage2_launch.py" \
    --check-final-record "$FINAL_ROOT/$1/final-record/final.json" --run "$1" --steps 60000 --train-head "$TRAIN_HEAD"
}

log_of() { echo "$ROOT/v1-store/logs/$1.log"; }
ckpt_of() { echo "$ROOT/v1-store/train-runs/mme_vla_suite_orig16_b128_60k/$1/59999"; }

# START_ROLE=modul：context须同版本完整成功且无未决事项，才跳过它；不重跑context。
if [[ $START_ROLE == modul ]]; then
  log=$(log_of "$CTX"); checkpoint=$(ckpt_of "$CTX")
  if has_unresolved "$CTX" || [[ ! -f "$log" || $(tail -n 1 "$log") != EXIT_CODE=0 ]] \
      || ! LC_ALL=C grep -aqxF "RUN=$CTX" "$log" || ! LC_ALL=C grep -aqxF "TRAIN_HEAD=$TRAIN_HEAD" "$log" \
      || [[ ! -d "$checkpoint" || -L "$checkpoint" ]] || ! final_record_ok "$CTX" \
      || LC_ALL=C grep -aqiP "$DIAG" "$log"; then
    wait_user "$CTX" resume_check "$log"
  fi
  printf 'CHAIN_RUN=PASS run=%s step=59999 resumed=1\n' "$CTX"
  ROLES=("$MOD:$MOD_H:${ATTEMPT_MANIFEST_MODUL:-}")
else
  ROLES=("$CTX:$CTX_H:${ATTEMPT_MANIFEST_CTX:-}" "$MOD:$MOD_H:${ATTEMPT_MANIFEST_MODUL:-}")
fi

for item in "${ROLES[@]}"; do
  IFS=: read -r run history manifest <<< "$item"
  log=$(log_of "$run"); checkpoint=$(ckpt_of "$run")
  attempt_env=(-u ATTEMPT_MANIFEST)
  [[ -z $manifest ]] || attempt_env=(ATTEMPT_MANIFEST="$manifest")
  if env "${attempt_env[@]}" HISTORY="$history" RUN_NAME="$run" FINAL_RECORD_DIR="$FINAL_ROOT/$run/final-record" \
      bash "$ROOT/scripts/training/run_orig16_stage2_60k.sh"; then
    rc=0
  else
    rc=$?
  fi
  [[ $rc == 0 ]] || wait_user "$run" train_exit "$log" "$rc"
  if [[ ! -f "$log" || $(tail -n 1 "$log") != EXIT_CODE=0 ]] || ! LC_ALL=C grep -aqxF "RUN=$run" "$log"; then
    wait_user "$run" missing_log "$log"
  fi
  [[ -d "$checkpoint" && ! -L "$checkpoint" ]] || wait_user "$run" missing_59999 "$checkpoint"
  final_record_ok "$run" || wait_user "$run" final_record "$FINAL_ROOT/$run/final-record/final.json"
  ! LC_ALL=C grep -aqiP "$DIAG" "$log" || wait_user "$run" train_numerics "$log"
  printf 'CHAIN_RUN=PASS run=%s step=59999\n' "$run"
done
echo 'CHAIN=PASS runs=2'
