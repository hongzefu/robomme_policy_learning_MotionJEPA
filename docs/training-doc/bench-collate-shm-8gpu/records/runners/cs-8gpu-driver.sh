#!/usr/bin/env bash
# A/B 串行独占八卡，任一失败保留原始记录并停止。
set -o pipefail
STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store
BASE_HEAD=${1:?}
NEW_HEAD=${2:?}
HC_SHA=5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec
LOG="$STORE/logs/cs-8gpu-driver.log"
test ! -e "$LOG" || exit 2
body() {
  set -e
  bash "$STORE/logs/cs-8gpu-runner.sh" old "$BASE_HEAD" "$HC_SHA"
  bash "$STORE/logs/cs-8gpu-runner.sh" new "$NEW_HEAD" "$HC_SHA"
  cd /scratch/hongze/robomme_policy_learning_MotionJEPA-temp
  UV_CACHE_DIR="$STORE/cache/uv" uv run --no-sync python "$STORE/logs/cs-analyze.py"
  echo DRIVER_ALL_DONE
}
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'EXIT_CODE=%s\n' "$rc" | tee -a "$LOG"
exit "$rc"
