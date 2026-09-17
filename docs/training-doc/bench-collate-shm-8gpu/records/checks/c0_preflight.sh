#!/usr/bin/env bash
# 只读验证三个副本的 preflight；额外参数用于反例。
set -euo pipefail
cs_repo=${1:?}
cs_mode=${2:?}
shift 2
cs_tool=/scratch/hongze/robomme_policy_learning_MotionJEPA-temp/scripts/training/preflight_train_launch.py
cd "$cs_repo"
source scripts/training/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv" JAX_PLATFORMS=cpu
export MMEVLA_FRAMESAMP_SOURCE="$V1_STORE/datasets/4task-v2-1600ep-604f16da/source"
export MMEVLA_FRAMESAMP_MANIFEST="$V1_STORE/datasets/4task-v2-1600ep-604f16da/meta/episode_manifest.json"
unset PYTHONPATH UV_PROJECT_ENVIRONMENT
cs_assets="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
cs_dataset="$V1_STORE/datasets/4task-v2-1600ep-604f16da/framesamp-8x8"
cs_hc=perceptual-framesamp-modul-8frame-8x8.yaml
cs_extra=()
if [[ "$cs_mode" == bench ]]; then
  cs_extra=(--bench-copy --v1-store-realpath /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store)
fi
uv run --no-sync python "$cs_tool" \
  --repo "$cs_repo" --train-head "$(git rev-parse HEAD)" \
  --history-config "$cs_hc" --history-config-sha256 5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec \
  --assets-dir "$cs_assets" --asset-id robomme \
  --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
  --dataset-path "$cs_dataset" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_60k/cs-c0-unused" \
  "${cs_extra[@]}" "$@" -- mme_vla_suite_b128_60k \
  --data.assets.assets-dir "$cs_assets" --data.assets.asset-id robomme \
  --dataset-path "$cs_dataset" --model.history-config "$cs_hc"
