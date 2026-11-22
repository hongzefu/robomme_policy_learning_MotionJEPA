#!/usr/bin/env bash
# 官方16×100 modulation concat（1×1024）stride16、budget192 最终公开导出；SMOKE=1只做本地检查，不读取凭据。
# motion 表为 grid16 新库；布局/budget192/库目录/逐段输出目录由公共体 RUN_MOTION 按本 run 登记，本入口只设 RUN 与 BUCKET。
# 设置ATTEMPT_MANIFEST时只接受其中已登记的同角色attempt run／bucket。
set -euo pipefail
RUN=orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k
BUCKET=HongzeFu/robomme-vla-orig100-modul-s2concat-s16-60k-v1
if [[ -n ${ATTEMPT_MANIFEST:-} ]]; then
  [[ $(jq -er .base_run "$ATTEMPT_MANIFEST") == "$RUN" ]] || { echo 'attempt清单base_run不是本包装器的固定run' >&2; exit 2; }
  RUN=$(jq -er .run "$ATTEMPT_MANIFEST")
  BUCKET=$(jq -er .bucket "$ATTEMPT_MANIFEST")
fi
O16_MODE=final
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_orig16_vla_ckpt_export_common.sh"
