#!/usr/bin/env bash
# 安装独立环境；不覆盖训练 .venv，不修改 benchmark 来源。
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv
PYTHON_BIN=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
export DS_BUILD_OPS=0
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$OPENPI_DATA_HOME" "$TORCH_HOME" "$MS_ASSET_DIR" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR"
UV_PROJECT_ENVIRONMENT="$EVAL_REPO/v1-store/envs/policy-eval-server" uv sync --project "$EVAL_REPO" --frozen --python "$PYTHON_BIN"
UV_PROJECT_ENVIRONMENT="$EVAL_REPO/v1-store/envs/policy-eval-client" uv sync --project "$EVAL_REPO/scripts/evaluation/client" --frozen --python "$PYTHON_BIN"
# 本仓库已有的 tokenizer 复制进独立缓存，校验后供两端离线使用。
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [[ ! -f "$TOKENIZER" ]]; then
    mkdir -p "$(dirname "$TOKENIZER")"
    cp "$EVAL_REPO/v1-store/models/big_vision/paligemma_tokenizer.model" "$TOKENIZER"
fi
printf '%s  %s\n' 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 "$TOKENIZER" | sha256sum --check --strict
