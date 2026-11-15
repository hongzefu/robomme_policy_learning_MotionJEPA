#!/usr/bin/env bash
# 4task stage2 rgb-state concat 两条50k的增量上传（每份ckpt落盘即传，只出 INC_END）；SMOKE=1只做本地检查，不读取凭据。
# 用法：RUN=<run> BUCKET=<bucket> [ATTEMPT_MANIFEST=<清单>] bash run_4task_stage2_ckpt_inc.sh
set -euo pipefail
: "${RUN:?必须指定RUN}" "${BUCKET:?必须指定BUCKET}"
O16_MODE=incremental
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_4task_vla_ckpt_export_common.sh"
