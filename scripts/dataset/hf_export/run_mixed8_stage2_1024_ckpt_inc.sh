#!/usr/bin/env bash
# stage2增量入口：等待原子根快照，先匿名回读根及依赖，再逐个传输已提交step。
# 与最终入口共用hf-ckpt-mixed8-200-vla-stage2-1024-80k/logs/uploaded，不代表G12恢复通过。
set -euo pipefail
STAGE2_MODE=incremental
source "$(dirname "${BASH_SOURCE[0]}")/_stage2_vla_ckpt_export_common.sh"
