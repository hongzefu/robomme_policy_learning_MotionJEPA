#!/usr/bin/env bash
# 16 任务 mixed8-200 新库两条 run 的共用起跑体。由 train-512-80k.sh / train-motion-80k.sh 设好
# CONFIG / HC / EXTRA_ARGS 后 source。复制自 ../v2_1600ep_runner/_train_common.sh，差别：
#   - 库换成 v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f（DS、dataset-path、MMEVLA_FRAMESAMP_*）；
#   - assets 换成 v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f，
#     必须带 --data.assets.assets-dir 与 --data.assets.asset-id robomme（漏写会静默读到 4 任务 norm stats）；
#   - 环境变量 FSDP_DEVICES / NUM_WORKERS 有值才透传为 --fsdp-devices / --num-workers，不设走 config 默认 8/16；
#   - 起跑前打印实际 import 的 mme_vla_suite 路径与 git HEAD；
#   - DRY_RUN=1：只打印最终 train.py 命令（TRAIN_CMD= 行，printf %q 转义）后 0 退出，
#     不写日志、不检查数据与 assets 是否就位、不起训练；默认（不设）行为不变。
# 调用方预设的 XLA_PYTHON_CLIENT_PREALLOCATE 原样保留（短测设 false 读真实显存），本脚本不碰。
set -o pipefail

RUN="${1:?用法: $0 <run_name>；禁止复用已存在的 run 名}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
NAME=16task-mixed8-200-3200ep-e3c7564f
DS="$REPO/v1-store/datasets/$NAME"
LOG="$REPO/v1-store/logs/$RUN.log"
DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" != 1 ]; then
  test ! -e "$LOG" || { printf '拒绝覆盖本轮日志 %s\n' "$LOG"; exit 2; }
  mkdir -p "$(dirname "$LOG")"
fi

body() (
  set -euo pipefail
  cd "$REPO"                       # 必须从仓库根起跑：history config 按相对 cwd 解析
  source "$REPO/scripts/training/paths.sh"

  WANDB_ARGS=()
  if [ -f "$V1_STORE/secrets/wandb.env" ]; then
    set -a; . "$V1_STORE/secrets/wandb.env"; set +a
  else
    echo "提示: 没有 v1-store/secrets/wandb.env，本轮关闭 wandb"
    WANDB_ARGS=(--no-wandb-enabled)
  fi

  PAR_ARGS=()
  if [ -n "${FSDP_DEVICES:-}" ]; then PAR_ARGS+=(--fsdp-devices "$FSDP_DEVICES"); fi
  if [ -n "${NUM_WORKERS:-}" ]; then PAR_ARGS+=(--num-workers "$NUM_WORKERS"); fi

  unset PYTHONPATH
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE   # motion 路径由 yaml 相对仓库根解析
  export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export UV_CACHE_DIR="$V1_STORE/cache/uv"
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  # store_meta.json 里写死的是建库时的绝对路径；这两条让读侧显式看新库的 source 与清单。
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  # 注意：**别设 MMEVLA_FRAMESAMP_VERIFY=full**，那会对全量 source 重算 sha256。
  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/$NAME"

  CMD=(uv run --no-sync python scripts/training/train.py "$CONFIG"
    --exp-name "$RUN"
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR"
    --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8"
    --model.history-config "$HC"
    "${PAR_ARGS[@]}" "${WANDB_ARGS[@]}" "${EXTRA_ARGS[@]}")

  printf 'RUN=%s\nCONFIG=%s\nHC=%s\nREPO=%s\nDS=%s\nASSETS_DIR=%s\nFSDP_DEVICES=%s\nNUM_WORKERS=%s\nXLA_PYTHON_CLIENT_PREALLOCATE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$CONFIG" "$HC" "$REPO" "$DS" "$ASSETS_DIR" "${FSDP_DEVICES:-<config 默认>}" "${NUM_WORKERS:-<config 默认>}" \
    "${XLA_PYTHON_CLIENT_PREALLOCATE:-<未设>}" "$(date -u +%FT%TZ)"
  printf 'GIT_HEAD=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'MME_VLA_SUITE_FILE=%s\n' "$(uv run --no-sync python -c 'import mme_vla_suite;print(mme_vla_suite.__file__)')"
  printf 'TRAIN_CMD='; printf '%q ' "${CMD[@]}"; printf '\n'

  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN=1：只打印命令，不检查数据、不起训练"
    exit 0
  fi
  [ -d "$DS/source/data" ] || { echo "错误: 缺 $DS/source/data/"; exit 1; }
  [ -f "$ASSETS_DIR/robomme/norm_stats.json" ] || { echo "错误: 缺 $ASSETS_DIR/robomme/norm_stats.json"; exit 1; }
  "${CMD[@]}"
)

if [ "$DRY_RUN" = 1 ]; then
  body
  exit $?
fi
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
