#!/usr/bin/env bash
# (2,4) 对照：只跑 (1,8) 两档里最快的 worker 档（w16）
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
DLOG="$MAIN/v1-store/logs/bw-8gpu-f4-driver.log"
TRAIN_HEAD=9cbb94e3779cd904fff051a3fa38f156c5b8c47c
HC_SHA=5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec
W="${1:-16}"
{
  echo "DRIVER_START_UTC=$(date -u +%FT%TZ)"
  echo "DRIVER_RUN mesh=2x4 fsdp_devices=4 workers=$W start=$(date -u +%FT%TZ)"
  bash "$MAIN/v1-store/logs/bw-8gpu-runner.sh" "$W" "$TRAIN_HEAD" "$HC_SHA" 4
  echo "DRIVER_DONE workers=$W rc=$? end=$(date -u +%FT%TZ)"
  echo "DRIVER_ALL_DONE utc=$(date -u +%FT%TZ)"
} 2>&1 | tee -a "$DLOG"
