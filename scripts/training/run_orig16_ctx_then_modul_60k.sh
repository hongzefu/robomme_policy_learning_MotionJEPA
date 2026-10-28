#!/usr/bin/env bash
# context成功且59999已保存即接modulation；上传和G12由外部长尾任务负责。
set -euo pipefail
[[ $# == 0 ]] || { echo '串行入口不接受参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${TRAIN_HEAD:?必须指定TRAIN_HEAD}"
: "${FINAL_ROOT:?必须指定FINAL_ROOT}"
[[ -z ${MAX_STEPS+x} && ${DRY_RUN:-0} == 0 ]] || { echo 'CHAIN=STOP reason=formal_only'; exit 2; }
CTX=orig100-ctx32x4x4-motion-s2split4x256-b128-60k
MOD=orig100-m32x4x4-modul-motion-s2split4x256-b128-60k
CTX_H=perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml
MOD_H=perceptual-framesamp-modul-32frame-4x4-motion-stage2-4x256-orig16.yaml
[[ ${RUN_CTX:-$CTX} == "$CTX" && ${RUN_MOD:-$MOD} == "$MOD" && $(basename "${HC_CTX:-$CTX_H}") == "$CTX_H" && $(basename "${HC_MOD:-$MOD_H}") == "$MOD_H" ]] || { echo 'CHAIN=STOP reason=paired_identity'; exit 2; }
for pair in "$CTX:$CTX_H" "$MOD:$MOD_H"; do
  run=${pair%%:*}
  history=${pair#*:}
  if HISTORY="$history" RUN_NAME="$run" FINAL_RECORD_DIR="$FINAL_ROOT/$run/final-record" \
      bash "$ROOT/scripts/training/run_orig16_stage2_60k.sh"; then
    rc=0
  else
    rc=$?
  fi
  log="$ROOT/v1-store/logs/$run.log"
  checkpoint="$ROOT/v1-store/train-runs/mme_vla_suite_orig16_b128_60k/$run/59999"
  if [[ $rc != 0 ]]; then echo "CHAIN=STOP reason=training_exit run=$run code=$rc"; exit "$rc"; fi
  if [[ ! -f "$log" || $(tail -n 1 "$log") != EXIT_CODE=0 ]] || ! rg -q -x "RUN=$run" "$log"; then
    echo "CHAIN=STOP reason=missing_success_log run=$run"; exit 1
  fi
  if rg -qi '^Step [0-9]+:.*(loss|[a-z_]*grad_norm|param_norm)=[-+]?(nan|inf)(,|[[:space:]]|$)|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError' "$log"; then
    echo "CHAIN=STOP reason=nonfinite_or_oom run=$run"; exit 1
  fi
  if [[ ! -d "$checkpoint" || -L "$checkpoint" ]]; then
    echo "CHAIN=STOP reason=missing_59999 run=$run"; exit 1
  fi
  if ! JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" "$ROOT/scripts/training/check_orig16_stage2_launch.py" \
      --check-final-record "$FINAL_ROOT/$run/final-record/final.json" --run "$run" --steps 60000 --train-head "$TRAIN_HEAD"; then
    echo "CHAIN=STOP reason=invalid_final_record run=$run"; exit 1
  fi
  printf 'CHAIN_RUN=PASS run=%s step=59999\n' "$run"
done
echo 'CHAIN=PASS runs=2'
