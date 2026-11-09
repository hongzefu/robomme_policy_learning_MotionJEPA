#!/usr/bin/env bash
# 官方16×100 concat（1×1024）最终公开导出；SMOKE=1只做本地检查，不读取凭据。
# 设置ATTEMPT_MANIFEST时只接受其中已登记的同角色attempt run／bucket。
set -euo pipefail
RUN=orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k
BUCKET=HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1
if [[ -n ${ATTEMPT_MANIFEST:-} ]]; then
  [[ $(jq -er .base_run "$ATTEMPT_MANIFEST") == "$RUN" ]] || { echo 'attempt清单base_run不是本包装器的固定run' >&2; exit 2; }
  RUN=$(jq -er .run "$ATTEMPT_MANIFEST")
  BUCKET=$(jq -er .bucket "$ATTEMPT_MANIFEST")
fi
O16_MODE=final
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_orig16_vla_ckpt_export_common.sh"
