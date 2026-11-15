#!/usr/bin/env bash
# 4task 8帧×8×8＋motion 1×1024（context concat）50k 的最终导出与评估上传入口。
# O16_MODE 必须显式给出：stage（训练成功后同步、不联网不持锁）｜upload（复用stage上传读回）｜eval（eval/ 前缀）｜stage-delete。
# 设置ATTEMPT_MANIFEST时只接受其中已登记的同角色attempt run／bucket。
set -euo pipefail
RUN=v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k
BUCKET=HongzeFu/robomme-vla-4task-m8x8-ctx-s2rgbstate-concat1x1024-50k-v1
if [[ -n ${ATTEMPT_MANIFEST:-} ]]; then
  [[ $(jq -er .base_run "$ATTEMPT_MANIFEST") == "$RUN" ]] || { echo 'attempt清单base_run不是本包装器的固定run' >&2; exit 2; }
  RUN=$(jq -er .run "$ATTEMPT_MANIFEST")
  BUCKET=$(jq -er .bucket "$ATTEMPT_MANIFEST")
fi
case "${O16_MODE:-}" in
  stage|upload|eval|stage-delete) ;;
  *) echo "O16_MODE 必须为 stage|upload|eval|stage-delete（增量用 run_4task_stage2_ckpt_inc.sh）" >&2; exit 2 ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_4task_vla_ckpt_export_common.sh"
