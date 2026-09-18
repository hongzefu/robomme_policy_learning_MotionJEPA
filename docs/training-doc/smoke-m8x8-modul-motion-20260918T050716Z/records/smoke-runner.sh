#!/usr/bin/env bash
# 外层保留正常返回和报错的退出码；外部强杀不保证能写出退出记录。
set -o pipefail

RUN="${1:?用法: mv2-smoke-runner.sh <RUN> <TRAIN_HEAD> <HC_SHA256> <MOTION_META_SHA256>}"
TRAIN_HEAD="${2:?}"
HC_SHA="${3:?}"
MOTION_META_SHA="${4:?必须给出建库验收后固定的 motion 元数据 SHA256}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
LOG="$MAIN/v1-store/logs/$RUN.driver.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
test ! -e "$LOG" || { printf '拒绝覆盖本轮日志 %s\n' "$LOG"; exit 2; }
mkdir -p "$(dirname "$LOG")"

body() (
  set -euo pipefail
  # 内层：-euo pipefail 就位，下面每条 test 都是「不满足就不起跑」的硬闸
  cd "$MAIN"                                 # 训练就跑在主副本上，不再有 worktree
  source "$MAIN/scripts/training/paths.sh"   # 只导出路径变量；不 mkdir；按 BASH_SOURCE 解析 REPO_ROOT

  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$(date -u +%FT%TZ)"

  test "$(git rev-parse HEAD)" = "$TRAIN_HEAD"
  test -z "$(git status --porcelain)"
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export TRAIN_TIMING_STEPS=300

  unset PYTHONPATH                           # 靠主副本 .venv 的 editable 安装；设了会被 preflight 拦
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/mv2-smoke"   # 固定目录：避免每轮 smoke 从零编译
  export TRAIN_RECORD_DIR="$V1_STORE/bench/$RUN"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  export WANDB_MODE=disabled

  # GPU 空闲（preflight 不查硬件）
  test "$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-8frame-8x8-motion.yaml

  # 唯一真源：preflight 校验的 argv 与 train.py 真收到的 argv 是同一个数组
  TRAIN_ARGS=(
    mme_vla_suite_b128_80k
    --exp-name "$RUN"
    --num-train-steps 20 --log-interval 1 --no-wandb-enabled
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR"
    --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8"
    --model.history-config "$HC"
  )

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" \
    --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_80k/$RUN" \
    --motion-store "$DS/motion" --motion-store-meta-sha256 "$MOTION_META_SHA" \
    --motion-layout motion-768-grid16-demopad17-v1 --motion-rows 71316 \
    -- "${TRAIN_ARGS[@]}"

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
)

body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
