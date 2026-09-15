#!/usr/bin/env bash
# 先完成每档 A/A，再起 B；外层始终记录真实退出码。
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=m8-modul-retro
LOG="$MAIN/v1-store/logs/$RUN.log"
BASE=07702f0076e640724e9516e911983d7810b423d3
CAND="${1:?必须传入候选源码 SHA}"
WT="$MAIN/v1-store/worktrees/s2-base"

body() {
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  export UV_CACHE_DIR="$V1_STORE/cache/uv" UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  export CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
  export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  unset JAX_PLATFORMS DTYPE_BASELINE_CHECKSUMS MMEVLA_MOTION_STORE
  test "$(git rev-parse HEAD)" = "$CAND"
  test -z "$(git status --porcelain)"
  test ! -e "$V1_STORE/bench/$RUN"
  test ! -e "$V1_STORE/train-runs/$RUN"
  test "$(nvidia-smi --id=4,5 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = 0
  printf 'RUN=%s\nBASE=%s\nCAND=%s\nSTART_UTC=%s\n' "$RUN" "$BASE" "$CAND" "$(date -u +%FT%TZ)"
  for profile in c32 c8; do
    if [[ "$profile" == c32 ]]; then
      HC=perceptual-framesamp-modul.yaml
    else
      HC=perceptual-framesamp-modul-8frame-8x8.yaml
    fi
    export DTYPE_BATCH_FIXTURE_DIR="$V1_STORE/fixtures/8x8/grad/$profile-b"
    for side in a1 a2 b; do
      if [[ "$side" == b ]]; then
        # 在同一棵历史树上的 A/A 必须先通过，才允许运行候选。
        cd "$MAIN"
        unset PYTHONPATH
        JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_fixed_grad.py \
          "$V1_STORE/bench/$RUN/$profile-a1/grad_summary.json" \
          "$V1_STORE/bench/$RUN/$profile-a2/grad_summary.json"
        export DTYPE_SOURCE_COMMIT="$CAND"
      else
        cd "$WT"
        export PYTHONPATH="$WT/src"
        export DTYPE_SOURCE_COMMIT="$BASE"
      fi
      export DTYPE_GRAD_DIR="$V1_STORE/bench/$RUN/$profile-$side"
      export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN-$profile-$side"
      printf 'GRAD_START profile=%s side=%s UTC=%s\n' "$profile" "$side" "$(date -u +%FT%TZ)"
      sha256sum "src/mme_vla_suite/models/config/robomme/$HC"
      uv run --no-sync python "$MAIN/scripts/training/tests/single_step_grad_fixed.py" mme_vla_suite \
        --exp-name "$RUN-$profile-$side" --batch-size 8 --fsdp-devices 2 --seed 42 \
        --model.use-history --model.history-config "$HC" --no-wandb-enabled \
        --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
        --assets-base-dir "$V1_STORE/train-assets" \
        --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
        --checkpoint-base-dir "$V1_STORE/train-runs/$RUN" \
        --dataset-path "$V1_STORE/datasets/4task-motion-400ep/framesamp-8x8"
    done
    cd "$MAIN"
    unset PYTHONPATH
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_fixed_grad.py \
      "$V1_STORE/bench/$RUN/$profile-a1/grad_summary.json" \
      "$V1_STORE/bench/$RUN/$profile-b/grad_summary.json"
    printf 'RETRO_PROFILE=PASS profile=%s\n' "$profile"
  done
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
