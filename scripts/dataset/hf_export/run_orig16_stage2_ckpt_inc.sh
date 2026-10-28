#!/usr/bin/env bash
# 官方16×100公开导出；SMOKE=1只做本地检查，不读取凭据。
set -euo pipefail
: "${RUN:?必须指定RUN}" "${BUCKET:?必须指定BUCKET}"
O16_MODE=incremental
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_orig16_vla_ckpt_export_common.sh"
