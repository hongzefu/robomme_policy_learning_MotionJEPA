# m2048-r20-packed 起跑口径

起跑前按用户「能并行的尽可能并行 8个gpu你可以用」调整为GPU0、1，与实际refnpy基线同卡；主副本源码仍冻结CAND 0c877c7495dfe5db8b83f033442013c6d6fd8552。下面命令中的设备编号已据此更新，全部数值判据与启动超参不变，真实开始时刻由RUN_START记录。

用户要求实现至测速报告，后续又明确选择「保持原计划，完整取证（推荐）」。本档遵循[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，仅使用1600ep库；物理GPU4、5，batch8、worker4、FSDP2、seed42、20步与同配置1步补跑，均为启动覆盖，保持全局默认。

本档须在生产与验证工具提交完毕后的同一clean CAND运行；完整实际HEAD与UTC由下面的START_HEAD/RUN_START记录，版本不是未来分支名。两侧使用同一CAND，仅输入读取器及dataset-path不同；归一化、采样和下游模型共享。依赖固定uv.lock，本轮不更改；存储是AWS /dev/md0 XFS NVMe RAID。

输入为4task-v2-1600ep-604f16da/framesamp-8x8，清单file SHA=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，规范化清单SHA=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918；norm_stats file SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。真实输入全部满历史，不用真实库声称已覆盖短历史；短历史另验。实际数组摘要和导入来源由bench记录器留存。

会话全名m2048-r20-packed。以下命令体经detached tmux运行，外层pipefail与tee写v1-store/logs/m2048-r20-packed.log，正常或失败均追加EXIT_CODE。每档结束先核会话退出和GPU释放；任何旧档数值失配按计划仅撤销生产提交并报告，harness错误另行定位。主run应有20行标量、20状态行（更新0及2..20）、232个索引/29批；补跑有0/1两个状态、80个索引/10批。当前仅建启动档案，尚未运行。

```bash
set -euo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR="$PWD/v1-store/cache/uv" PYTHONUNBUFFERED=1
export DS="$PWD/v1-store/datasets/4task-v2-1600ep-604f16da"
export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
export DTYPE_MANIFEST="$MMEVLA_FRAMESAMP_MANIFEST"
unset MMEVLA_MOTION_STORE BENCH_REF_MOTION MMEVLA_FRAMESAMP_ALLOW_SUBSET
test -z "$(git status --porcelain)"
M2048_HEAD=$(git rev-parse HEAD)
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$M2048_HEAD" "$(date -u +%FT%TZ)"
export CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export BENCH_REF_COMMIT="$M2048_HEAD" BENCH_CAND_COMMIT="$M2048_HEAD"
unset JAX_PLATFORMS BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR
unset BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP BENCH_PERF_MODE BENCH_EXTRA_DIGEST_STEPS BENCH_DATASET_IMPL
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data" TRAIN_TIMING_STEPS=0
export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1 BENCH_DUMP_IDX=1
for M2048_STEPS in 20 1; do
  M2048_RUN="m2048-r20-packed"
  if [ "$M2048_STEPS" = 1 ]; then M2048_RUN="m2048-r20-packed-step1"; fi
  M2048_REFERENCE="m2048-r20-refnpy"
  if [ "$M2048_STEPS" = 1 ]; then M2048_REFERENCE="m2048-r20-refnpy-step1"; fi
  export BENCH_RECORD_DIR="$V1_STORE/bench/m2048/$M2048_RUN"
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$M2048_RUN"
  test ! -e "$BENCH_RECORD_DIR"
  test ! -e "$V1_STORE/train-runs/$M2048_RUN"
  test -z "$(git status --porcelain)"
  printf 'RUN_START name=%s steps=%s head=%s utc=%s\n' "$M2048_RUN" "$M2048_STEPS" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump \
    --record-dir "$BENCH_RECORD_DIR" --v1-store "$V1_STORE" --source "$DS/source" \
    --manifest "$DS/meta/episode_manifest.json" --dataset "$DS/framesamp-8x8" \
    --norm-stats "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$BENCH_RECORD_DIR"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check \
    --base "$V1_STORE/bench/m2048/$M2048_REFERENCE" --record-dir "$BENCH_RECORD_DIR" --allow-difference dataset.store_meta_sha256 \
    --steps "$M2048_STEPS" --batch-size 8 --dataset "$DS/framesamp-8x8"
  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite \
    --exp-name "$M2048_RUN" --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
    --data.assets.asset-id robomme --checkpoint-base-dir "$V1_STORE/train-runs/$M2048_RUN" \
    --batch-size 8 --num-workers 4 --num-train-steps "$M2048_STEPS" --log-interval 1 \
    --save-interval 1 --seed 42 --fsdp-devices 2 --no-wandb-enabled \
    --dataset-path "$DS/framesamp-8x8" \
    --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-modul-32frame-8x8.yaml
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/project_scalars.py \
    "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$BENCH_RECORD_DIR"
  printf 'RUN_DONE name=%s steps=%s utc=%s\n' "$M2048_RUN" "$M2048_STEPS" "$(date -u +%FT%TZ)"
done
```
