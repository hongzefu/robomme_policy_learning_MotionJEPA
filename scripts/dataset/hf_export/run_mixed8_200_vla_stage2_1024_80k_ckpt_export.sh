#!/usr/bin/env bash
# stage2 VLA最终导出：16步、根快照、依赖、全量匿名SHA回读；先以SMOKE=1本地预检。
set -euo pipefail
STAGE2_MODE=final
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_vla_ckpt_export_common.sh"
