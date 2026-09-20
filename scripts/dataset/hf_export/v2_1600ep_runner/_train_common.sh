#!/usr/bin/env bash
# 两条 run 的共用起跑体。由 train-60k.sh / train-80k.sh 设好 CONFIG / HC 后 source。
# 刻意与 docs/training-doc/v2-1600ep-m8x8-modul-motion-b128-80k/records/prod-runner.sh 同构，
# 差别只有：异地机器不强制 clean HEAD 与 preflight（那两条绑本仓库的留档纪律），
# wandb 凭据缺失时自动降级为 --no-wandb-enabled。
set -o pipefail

RUN="${1:?用法: $0 <run_name>；禁止复用已存在的 run 名}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
DS="$REPO/v1-store/datasets/4task-v2-1600ep-604f16da"
LOG="$REPO/v1-store/logs/$RUN.log"
test ! -e "$LOG" || { printf '拒绝覆盖本轮日志 %s\n' "$LOG"; exit 2; }
mkdir -p "$(dirname "$LOG")"

body() (
  set -euo pipefail
  cd "$REPO"                       # 必须从仓库根起跑：history config 按相对 cwd 解析
  source "$REPO/scripts/training/paths.sh"

  WANDB_ARGS=()
  if [ -f "$REPO/v1-store/secrets/wandb.env" ]; then
    set -a; . "$REPO/v1-store/secrets/wandb.env"; set +a
  else
    echo "提示: 没有 v1-store/secrets/wandb.env，本轮关闭 wandb"
    WANDB_ARGS=(--no-wandb-enabled)
  fi

  unset PYTHONPATH
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE   # motion 路径由 yaml 相对仓库根解析
  export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export UV_CACHE_DIR="$V1_STORE/cache/uv"
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  # 两个 store_meta.json 里写死的是建库机的绝对路径；这两条让读侧改看本机实际落点。
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  # 注意：**别设 MMEVLA_FRAMESAMP_VERIFY=full**，那会对 292 GiB 全量重算 sha256。

  [ -d "$DS/source/data" ] || { echo "错误: 缺 source/data/，先跑 restore.sh 解开 data_tars"; exit 1; }
  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  [ -f "$ASSETS_DIR/robomme/norm_stats.json" ] || { echo "错误: 缺 norm_stats.json"; exit 1; }

  printf 'RUN=%s\nCONFIG=%s\nHC=%s\nREPO=%s\nSTART_UTC=%s\n' \
    "$RUN" "$CONFIG" "$HC" "$REPO" "$(date -u +%FT%TZ)"

  uv run --no-sync python scripts/training/train.py "$CONFIG" \
    --exp-name "$RUN" \
    --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$ASSETS_DIR" \
    --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs" \
    --dataset-path "$DS/framesamp-8x8" \
    --model.history-config "$HC" \
    "${WANDB_ARGS[@]}" "${EXTRA_ARGS[@]}"
)

body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
