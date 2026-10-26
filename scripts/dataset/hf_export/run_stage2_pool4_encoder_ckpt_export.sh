#!/usr/bin/env bash
# 两源stage2 checkpoint导出：阶段III 21份、阶段II 3份，独立前缀与清单；先SMOKE=1。
set -euo pipefail
STAGE2_MODE=encoder
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_vla_ckpt_export_common.sh"
