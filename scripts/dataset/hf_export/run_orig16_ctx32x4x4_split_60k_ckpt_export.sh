#!/usr/bin/env bash
# 官方16×100公开导出；SMOKE=1只做本地检查，不读取凭据。
set -euo pipefail
RUN=orig100-ctx32x4x4-motion-s2split4x256-b128-60k
BUCKET=HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1
O16_MODE=final
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_orig16_vla_ckpt_export_common.sh"
