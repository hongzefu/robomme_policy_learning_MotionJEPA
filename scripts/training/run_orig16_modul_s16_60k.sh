#!/usr/bin/env bash
# 官方16×100 modulation + motion concat 1×1024、窗口步长16（grid16 库）、budget 192 的单条正式60k入口。
# 只固定新 run 名与新 yaml，其余全部委托 run_orig16_stage2_60k.sh：
#   CONFIG=mme_vla_suite_orig16_b128_60k（8卡 FSDP8、B128、workers16、seed42、60k、预热5k、lr 5e-5、EMA 0.999），
#   起跑前跑 check_orig16_stage2_launch.py 启动合同（含 DATA_LINEAGE 闸门，P-LINEAGE），任一 FAIL 即不训练。
# 本入口不改任何超参；MAX_STEPS=20（须配 o16-smoke- 名）为同日程20步短测，DRY_RUN=1 只打印命令并跑合同。
# 正式60k结束后核：程序退出码0、日志末行 EXIT_CODE=0、日志含 RUN= 与 DATA_LINEAGE=PASS、59999 实体目录、
# 最终记录有限性、无 nan/inf/OOM 诊断；任一不过打印 S16_RUN=FAIL 并非零退出。
set -euo pipefail
[[ $# == 0 ]] || { echo '本入口不接受参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
CONFIG=mme_vla_suite_orig16_b128_60k
HISTORY=perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml
RUN=orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k
STEPS=60000
: "${TRAIN_HEAD:?必须指定完整TRAIN_HEAD}"
: "${FINAL_ROOT:?必须指定FINAL_ROOT}"
[[ "$FINAL_ROOT" == /* ]] || { echo 'FINAL_ROOT必须是绝对路径' >&2; exit 2; }
if [[ -n ${MAX_STEPS+x} ]]; then
  # 短测沿用单条入口的规则：只允许20步且 run 名须为 o16-smoke- 前缀。
  RUN=${RUN_NAME:?短测必须指定RUN_NAME（o16-smoke-前缀）}
  STEPS=$MAX_STEPS
fi
[[ "$RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "RUN_NAME非法: $RUN" >&2; exit 2; }
FINAL="$FINAL_ROOT/$RUN/final-record"
LOG="$ROOT/v1-store/logs/$RUN.log"
CKPT="$ROOT/v1-store/train-runs/$CONFIG/$RUN/$((STEPS - 1))"
DIAG='^\r*Step [0-9]+:.*(loss|grad_norm|llm_grad_norm|param_norm|mem_enc_norm)=[-+]?(nan|inf)(,|[[:space:]]|$)|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError|^TRAIN_(NUMERICS|DIAGNOSTIC)=FAIL'

fail() {
  echo "S16_RUN=FAIL run=$RUN stage=$1 evidence=$2"
  exit "${3:-3}"
}

if env -u ATTEMPT_MANIFEST HISTORY="$HISTORY" RUN_NAME="$RUN" FINAL_RECORD_DIR="$FINAL" \
    bash "$ROOT/scripts/training/run_orig16_stage2_60k.sh"; then
  rc=0
else
  rc=$?
fi
# DRY_RUN 只打印命令与合同结果，不产生日志与 run；退出码即合同退出码。
[[ ${DRY_RUN:-0} == 1 ]] && exit "$rc"
[[ $rc == 0 ]] || fail train_exit "$LOG" "$rc"
if [[ ! -f "$LOG" || $(tail -n 1 "$LOG") != EXIT_CODE=0 ]] || ! LC_ALL=C grep -aqxF "RUN=$RUN" "$LOG"; then
  fail missing_log "$LOG"
fi
# 启动合同在训练前打印 DATA_LINEAGE=PASS；缺此行说明血缘闸门未放行这次训练。
LC_ALL=C grep -aq '^DATA_LINEAGE=PASS ' "$LOG" || fail data_lineage "$LOG"
[[ -d "$CKPT" && ! -L "$CKPT" ]] || fail "missing_$((STEPS - 1))" "$CKPT"
JAX_PLATFORMS=cpu "$PY" "$ROOT/scripts/training/check_orig16_stage2_launch.py" \
  --check-final-record "$FINAL/final.json" --run "$RUN" --steps "$STEPS" --train-head "$TRAIN_HEAD" \
  || fail final_record "$FINAL/final.json"
! LC_ALL=C grep -aqiP "$DIAG" "$LOG" || fail train_numerics "$LOG"
printf 'S16_RUN=PASS run=%s steps=%s step=%s history=%s\n' "$RUN" "$STEPS" "$((STEPS - 1))" "$HISTORY"
