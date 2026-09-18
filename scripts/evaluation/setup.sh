#!/usr/bin/env bash
# 安装独立环境；不覆盖训练 .venv，不修改 benchmark 来源。
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv
PYTHON_BIN=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
export DS_BUILD_OPS=0
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$OPENPI_DATA_HOME" "$TORCH_HOME" "$MS_ASSET_DIR" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR"
UV_PROJECT_ENVIRONMENT="$EVAL_REPO/v1-store/envs/policy-eval-server" uv sync --project "$EVAL_REPO" --frozen --python "$PYTHON_BIN"
UV_PROJECT_ENVIRONMENT="$EVAL_REPO/v1-store/envs/policy-eval-client" uv sync --project "$EVAL_REPO/scripts/evaluation/client" --python "$PYTHON_BIN"
