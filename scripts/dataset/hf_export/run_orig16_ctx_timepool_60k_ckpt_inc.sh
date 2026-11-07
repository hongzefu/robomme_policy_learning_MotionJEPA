#!/usr/bin/env bash
# Wan 池化 60k 增量备份：按序等每步 Orbax 提交后打包、卫生扫描、上传并匿名读回。
# 解释器由 O16_PYTHON 显式给出，代码取本文件所在源码树，v1-store 取 --store-root（或 O16_STORE_ROOT）。
# SMOKE=1 只做本地检查，不读取凭据、不联网，且必须给 --export-root。
set -euo pipefail
: "${O16_PYTHON:?必须用 O16_PYTHON 显式指定解释器}"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
exec ionice -c 3 nice -n 19 "$O16_PYTHON" "$(dirname "${BASH_SOURCE[0]}")/_timepool_orig16_ckpt_export.py" --mode incremental "$@"
