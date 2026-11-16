#!/usr/bin/env bash
# 统一转发到本仓库的双 uv 评估入口，参数必须显式给出。
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../evaluation/run.sh" "$@"
