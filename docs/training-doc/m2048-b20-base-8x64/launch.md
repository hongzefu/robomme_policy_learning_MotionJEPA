# m2048-b20-base-8x64 启动记录

用户原话：「开始实现 有问题立刻越早问用户越好 一路实现到测速结束报告用户为止」。执行 [2048 计划](../../../0920-32frame-8x8-modul-2048-plan.md) 步 2a，本档只建立旧 8x64 modulation、motion 关闭的改前证据。

版本锚点为引入本启动档案及第一批工具适配的同一个提交，运行前强制 porcelain 为空；完整实际提交、UTC 起点由下列命令输出的 START_HEAD / START_UTC 和原始日志记录，后续结果档案回填该值。生产 src 与初始提交 c31b0509be3f76a6cc262fd655f5f0080b12c266 相同，依赖不变。环境 B：8×A100-SXM4-80GB，本档仅 GPU 4、5，FSDP 2；底层存储为 /dev/md0 XFS 本地 NVMe RAID。

输入仅为 1600ep 实体库 framesamp-8x8、同根 source 和 meta/episode_manifest.json。清单文件 SHA256=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482；规范化内容指纹=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918；norm_stats 文件 SHA256=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。库 verified/full，1600 集、1192918 帧、605611 执行样本。本库不存在短历史，fixture 的 allshort/mixed1 仅为沿用组名。

参数口径：mme_vla_suite；启动覆盖 batch8、worker4、seed42、FSDP2、20 次更新、log/save interval1，另同配置 1 步补跑；各自独立 run 和缓存。五标量与输入逐步记录，完整 TrainState 记录初态及更新2..20；补跑交付更新1。预期主 run 索引232个、29批，补跑80个、10批。所有数值必须有限；此处不从带摘要的短运行推算正式吞吐。

下列命令体放入 detached tmux `m2048-b20-base-8x64`；外层用 `set -o pipefail`、`tee v1-store/logs/m2048-b20-base-8x64.log`，结束记录 `EXIT_CODE=`。本档会话只有该完整名称。原始产物保留 v1-store/bench/m2048/，清洗日志与摘要随后归档 records/；checkpoint 不进 Git。

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
M2048_BASE=$(git rev-parse HEAD)
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$M2048_BASE" "$(date -u +%FT%TZ)"
export CUDA_VISIBLE_DEVICES=4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export BENCH_REF_COMMIT="$M2048_BASE"
unset JAX_PLATFORMS BENCH_CAND_COMMIT BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR
unset BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP BENCH_PERF_MODE BENCH_EXTRA_DIGEST_STEPS BENCH_DATASET_IMPL
export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1 BENCH_DUMP_IDX=1
for M2048_STEPS in 20 1; do
  M2048_RUN="m2048-b20-base-8x64"
  if [ "$M2048_STEPS" = 1 ]; then M2048_RUN="m2048-b20-base-8x64-step1"; fi
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
    --base "$BENCH_RECORD_DIR" --record-dir "$BENCH_RECORD_DIR" \
    --steps "$M2048_STEPS" --batch-size 8 --dataset "$DS/framesamp-8x8"
  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite \
    --exp-name "$M2048_RUN" --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
    --data.assets.asset-id robomme --checkpoint-base-dir "$V1_STORE/train-runs/$M2048_RUN" \
    --batch-size 8 --num-workers 4 --num-train-steps "$M2048_STEPS" --log-interval 1 \
    --save-interval 1 --seed 42 --fsdp-devices 2 --no-wandb-enabled \
    --dataset-path "$DS/framesamp-8x8" \
    --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/project_scalars.py \
    "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$BENCH_RECORD_DIR"
  printf 'RUN_DONE name=%s steps=%s utc=%s\n' "$M2048_RUN" "$M2048_STEPS" "$(date -u +%FT%TZ)"
done
```
