#!/usr/bin/env bash
# 串行驱动：w8 → w16（两档不得并行，bench-b128-util 实测并行互扰 +11–13%）
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
DLOG="$MAIN/v1-store/logs/bw-8gpu-driver.log"
TRAIN_HEAD=9cbb94e3779cd904fff051a3fa38f156c5b8c47c
HC_SHA=5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec
{
  echo "DRIVER_START_UTC=$(date -u +%FT%TZ)"
  for W in 8 16; do
    echo "DRIVER_RUN mesh=1x8 fsdp_devices=8 workers=$W start=$(date -u +%FT%TZ)"
    bash "$MAIN/v1-store/logs/bw-8gpu-runner.sh" "$W" "$TRAIN_HEAD" "$HC_SHA"
    rc=$?
    echo "DRIVER_DONE workers=$W rc=$rc end=$(date -u +%FT%TZ)"
    sleep 5   # 让 sampler 与 worker 进程完全退出，下一档 GPU_IDLE 才能过
  done
  echo "DRIVER_ALL_DONE utc=$(date -u +%FT%TZ)"
} 2>&1 | tee -a "$DLOG"
