#!/usr/bin/env bash
# 八卡真实 smoke 及收尾验收；保留权重到人工核对归档完成后再按本轮路径清理。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
RUN="${1:?本轮唯一 smoke 名称}"
CHECK_HEAD="${2:?完整提交}"
HC_SHA="${3:?history YAML SHA256}"
MOTION_SHA="${4:?固定 motion 元数据 SHA256}"
LOG="$PWD/v1-store/logs/$RUN.stage.log"
test ! -e "$LOG" || exit 2
body() (
  set -euo pipefail
  source scripts/training/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" UV_NO_SYNC=1 PYTHONUNBUFFERED=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  unset PYTHONPATH MMEVLA_MOTION_STORE
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  test "$(tail -n 1 "$V1_STORE/logs/mv2-vonl-c.driver.log")" = EXIT_CODE=0
  test "$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = 0
  LIB="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  REC="$V1_STORE/bench/$RUN"
  RUN_ROOT="$V1_STORE/train-runs/mme_vla_suite_b128_80k/$RUN"
  test ! -e "$REC" && test ! -e "$RUN_ROOT"
  mkdir "$REC"
  printf 'CHECK_HEAD=%s\nRUN=%s\nSTART_UTC=%s\n' "$CHECK_HEAD" "$RUN" "$(date -u +%FT%TZ)"
  nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 \
    > "$REC/gpu_util_500ms.csv" 2> "$REC/gpu_util_500ms.err" &
  sampler_pid=$!
  printf 'GPU_SAMPLER_PID=%s\n' "$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
  bash v1-store/logs/mv2-smoke-runner.sh "$RUN" "$CHECK_HEAD" "$HC_SHA" "$MOTION_SHA"
  kill "$sampler_pid"
  sampler_rc=0
  wait "$sampler_pid" || sampler_rc=$?
  trap - EXIT
  printf 'GPU_SAMPLER_EXIT_CODE=%s intentional_stop_after_training=1\n' "$sampler_rc"
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
  uv run --no-sync python scripts/training/legacy-eval/check_ckpt_param_tree.py \
    --ckpt-dir "$RUN_ROOT/19" --config mme_vla_suite_b128_80k --out "$REC/param_tree.json"
  uv run --no-sync python scripts/training/g0/check_config_provenance.py \
    --ckpt "$RUN_ROOT/19" --lib "$LIB" --train-config mme_vla_suite_b128_80k \
    --store-subdir framesamp-8x8 --neg-lib "$V1_STORE/datasets/4task-motion-400ep" \
    --norm-stats "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json" \
    --out "$REC/config_provenance.json"
  uv run --no-sync python scripts/training/tests/finish_check.py smoke \
    --records "$REC" --run-root "$RUN_ROOT" --log "$V1_STORE/logs/$RUN.driver.log" \
    --tree-report "$REC/param_tree.json" --provenance-report "$REC/config_provenance.json"
  uv run --no-sync python scripts/training/tests/summarize_step_timing.py \
    --records "$REC" --gpu-csv "$REC/gpu_util_500ms.csv" --warmup-steps 5 --end-step 19 \
    --out "$REC/timing_smoke.json"
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  printf 'SMOKE_STAGE=PASS weights_preserved=1\n'
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
