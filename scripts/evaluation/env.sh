#!/usr/bin/env bash
# 本仓库两端共享的独立评估环境；不覆盖 HOME 或训练环境。
EVAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EVAL_REPO
unset VIRTUAL_ENV
export UV_LINK_MODE=copy UV_CACHE_DIR="$EVAL_REPO/v1-store/cache/policy-eval/uv"
export XDG_CACHE_HOME="$EVAL_REPO/v1-store/cache/policy-eval/xdg"
export HF_HOME="$EVAL_REPO/v1-store/cache/policy-eval/huggingface"
export OPENPI_DATA_HOME="$EVAL_REPO/v1-store/cache/policy-eval/openpi"
export TORCH_HOME="$EVAL_REPO/v1-store/cache/policy-eval/torch"
export MS_ASSET_DIR="$EVAL_REPO/v1-store/cache/policy-eval/maniskill"
export CUDA_CACHE_PATH="$EVAL_REPO/v1-store/cache/policy-eval/cuda"
export JAX_COMPILATION_CACHE_DIR="$EVAL_REPO/v1-store/cache/policy-eval/jax"
export WANDB_DIR="$EVAL_REPO/v1-store/cache/policy-eval/wandb" WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
SERVER_UV=(env "UV_PROJECT_ENVIRONMENT=$EVAL_REPO/v1-store/envs/policy-eval-server" uv run --project "$EVAL_REPO" --frozen --no-sync python)
CLIENT_UV=(env "UV_PROJECT_ENVIRONMENT=$EVAL_REPO/v1-store/envs/policy-eval-client" uv run --project "$EVAL_REPO/scripts/evaluation/client" --frozen --no-sync python)
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
fi
