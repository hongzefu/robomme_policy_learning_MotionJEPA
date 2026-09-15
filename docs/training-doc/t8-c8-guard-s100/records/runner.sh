#!/usr/bin/env bash
# 外层保留真实退出码；内层由 paths.sh 开启严格失败检查。
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=t8-c8-guard-s100
LOG="$MAIN/v1-store/logs/$RUN.log"
D8="$MAIN/v1-store/datasets/4task-motion-400ep"

body() {
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  export UV_CACHE_DIR="$V1_STORE/cache/uv" CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1
  export UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
  unset JAX_PLATFORMS PYTHONPATH MMEVLA_MOTION_STORE
  unset BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP
  unset BENCH_PERF_MODE BENCH_DUMP_IDX BENCH_DATASET_IMPL BENCH_REF_SOURCE BENCH_REF_MANIFEST BENCH_REF_MOTION
  export BENCH_RECORD_DIR="$V1_STORE/bench/8x8/$RUN"
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000
  export BENCH_EXTRA_DIGEST_STEPS=1,2,24,49

  test -z "$(git -C "$MAIN" status --porcelain)"
  test ! -e "$BENCH_RECORD_DIR"
  test ! -e "$V1_STORE/train-runs/$RUN"
  test "$(nvidia-smi --id=4,5 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = 0
  printf 'RUN=%s\nHEAD=%s\nSTART_UTC=%s\n' "$RUN" "$(git -C "$MAIN" rev-parse HEAD)" "$(date -u +%FT%TZ)"

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump \
    --record-dir "$BENCH_RECORD_DIR" --v1-store "$V1_STORE" \
    --source "$D8/source" --manifest "$D8/meta/episode_manifest.json" --dataset "$D8/framesamp-8x8" \
    --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check \
    --base "$V1_STORE/bench/8x8/t8-c8-b" --record-dir "$BENCH_RECORD_DIR" \
    --steps 100 --batch-size 8 --dataset "$D8/framesamp-8x8"

  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name "$RUN" \
    --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs/$RUN" --batch-size 8 --num-workers 4 \
    --num-train-steps 100 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
    --dataset-path "$D8/framesamp-8x8" \
    --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/project_scalars.py \
    "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/compare_baseline.py \
    "$V1_STORE/bench/8x8/t8-c8-b" "$BENCH_RECORD_DIR"
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
