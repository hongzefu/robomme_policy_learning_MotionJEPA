#!/usr/bin/env bash
# 用户明确提前停止后的70k收尾；保留原80k身份，先以SMOKE=1验证停止收据。
set -euo pipefail
STAGE2_MODE=final70
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_vla_ckpt_export_common.sh"
