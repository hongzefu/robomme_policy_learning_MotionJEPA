#!/usr/bin/env bash
# 三侧串行逐位对拍；a1/a2 失败立即停止，不降低判据。
set -o pipefail
STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store
TOOLS=/scratch/hongze/robomme_policy_learning_MotionJEPA-temp
BASE=/scratch/hongze/robomme_policy_learning_MotionJEPA-base
BASE_HEAD=${1:?}
NEW_HEAD=${2:?}
LOG="$STORE/logs/cs-guard.log"
REC="$STORE/bench/8x8/cs-guard-summary"
test ! -e "$LOG" && test ! -e "$REC" || exit 2
mkdir "$REC"

run_side() (
  set -e
  SIDE=$1
  if [[ "$SIDE" == b ]]; then MAIN=$TOOLS; TRAIN_HEAD=$NEW_HEAD; else MAIN=$BASE; TRAIN_HEAD=$BASE_HEAD; fi
  cd "$MAIN"
  source scripts/training/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
  unset PYTHONPATH JAX_PLATFORMS MMEVLA_MOTION_STORE UV_PROJECT_ENVIRONMENT TRAIN_RECORD_DIR
  unset BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP
  unset BENCH_PERF_MODE BENCH_DUMP_IDX BENCH_DATASET_IMPL BENCH_REF_SOURCE BENCH_REF_MANIFEST BENCH_REF_MOTION
  RUN=cs-guard-$SIDE
  export BENCH_RECORD_DIR="$V1_STORE/bench/8x8/$RUN" BENCH_SOURCE_ROOT="$MAIN"
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000 BENCH_EXTRA_DIGEST_STEPS=1,2,24,49
  DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source" MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  test ! -e "$BENCH_RECORD_DIR"
  used=$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  printf 'SIDE=%s\nSOURCE_HEAD=%s\nSTART_UTC=%s\nGPU_IDLE used_mib=%s\n' "$SIDE" "$TRAIN_HEAD" "$(date -u +%FT%TZ)" "$used"
  test "$used" = 0
  TRAIN_ARGS=(mme_vla_suite_b128_60k --exp-name "$RUN" --batch-size 128 --num-workers 16
    --num-train-steps 100 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 8 --no-wandb-enabled
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$ASSETS_DIR" --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs" --dataset-path "$DS/framesamp-8x8"
    --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml)
  JAX_PLATFORMS=cpu uv run --no-sync python "$TOOLS/scripts/training/preflight_train_launch.py" \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" --bench-copy --v1-store-realpath "$STORE" \
    --history-config perceptual-framesamp-modul-8frame-8x8.yaml \
    --history-config-sha256 5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN" -- "${TRAIN_ARGS[@]}"
  JAX_PLATFORMS=cpu uv run --no-sync python "$TOOLS/scripts/training/g0/check_baseline_env.py" dump \
    --record-dir "$BENCH_RECORD_DIR" --v1-store "$V1_STORE" --source "$DS/source" \
    --manifest "$DS/meta/episode_manifest.json" --dataset "$DS/framesamp-8x8" --norm-stats "$ASSETS_DIR/robomme/norm_stats.json"
  if [[ "$SIDE" != a1 ]]; then
    JAX_PLATFORMS=cpu uv run --no-sync python "$TOOLS/scripts/training/g0/check_baseline_env.py" check \
      --base "$STORE/bench/8x8/cs-guard-a1" --record-dir "$BENCH_RECORD_DIR" --steps 100 --batch-size 128 --dataset "$DS/framesamp-8x8"
  fi
  uv run --no-sync python "$TOOLS/scripts/training/g0/bench_train_steps.py" "${TRAIN_ARGS[@]}"
  JAX_PLATFORMS=cpu uv run --no-sync python "$TOOLS/scripts/training/tests/project_scalars.py" \
    "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
  JAX_PLATFORMS=cpu uv run --no-sync python "$TOOLS/scripts/training/g0/check_baseline_env.py" manifest "$BENCH_RECORD_DIR"
)

body() {
  set -e
  for side in a1 a2 b; do
    set +e
    run_side "$side" 2>&1 | tee "$STORE/logs/cs-guard-$side.log"
    side_rc=${PIPESTATUS[0]}
    set -e
    printf 'SIDE=%s EXIT_CODE=%s\n' "$side" "$side_rc" | tee -a "$STORE/logs/cs-guard-$side.log"
    test "$side_rc" = 0
    if [[ "$side" != a1 ]]; then
      cd "$TOOLS"
      if [[ "$side" == a2 ]]; then pair=a1a2; else pair=a1b; fi
      UV_CACHE_DIR="$STORE/cache/uv" JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/compare_baseline.py \
        "$STORE/bench/8x8/cs-guard-a1" "$STORE/bench/8x8/cs-guard-$side" --tier "cs-$pair" | tee "$REC/compare-$pair.log"
    fi
  done
  UV_CACHE_DIR="$STORE/cache/uv" uv run --no-sync python "$TOOLS/scripts/training/g0/guard_finish_check.py" \
    --sides "a1=$STORE/bench/8x8/cs-guard-a1" "a2=$STORE/bench/8x8/cs-guard-a2" "b=$STORE/bench/8x8/cs-guard-b" \
    --compare-logs "$REC/compare-a1a2.log" "$REC/compare-a1b.log" --steps 100 --batch-size 128 \
    --digest-steps 0,1,2,24,49,99 --summary-prefix COLLATE_SHM_GUARD | tee "$REC/guard_finish.log"
  echo GUARD_ALL_DONE
}
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
