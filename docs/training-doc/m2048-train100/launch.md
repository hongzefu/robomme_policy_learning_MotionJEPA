# m2048-train100 启动口径

起跑前按用户并行决定调整为GPU6、7；主副本源码保持CAND 0c877c7495dfe5db8b83f033442013c6d6fd8552。环境指纹引用届时已完成的refnpy主run，明确只允许原参考链既有的dataset.store_meta_sha256差异，以及本次GPU编号gpu.CUDA_VISIBLE_DEVICES差异；全部基线文件SHA仍核验。初态数值依然逐201叶与packed首态比较，未写完时须等，不豁免。如此可与packed20步并行而不引用其仍追加记录的BASELINE_MANIFEST。

用户原话：「一路实现到测速结束报告用户为止」；并选择「保持原计划，完整取证（推荐）」。本档从与2048两侧20步对拍相同的clean CAND独立初始化，使用1600ep/framesamp-8x8和同一norm_stats，GPU4、5，batch8、worker4、FSDP2、seed42、100步，启动覆盖log1/save25，不改全局默认。完整HEAD与UTC在START_HEAD记录；硬件为AWS A100，本地/dev/md0 XFS NVMe RAID。

本档必须保存初态与第100次更新的完整原数组，使用既有BENCH_STATE_DUMP_STEPS=0,99；文件名state_step_99对应实际state.step100。bench真实保存固定写checkpoint目录999，内容为该时刻EMA。数组先与本次逐叶摘要逐一核对，再转bf16比加载叶；十个记忆叶须区别于初态，并以相同observation、固定noise、10步采样比较动作，rms≤6.8e-5。不能用不可逆哈希替代数组，也不能从待验checkpoint重建参照。

输入清单file SHA=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，规范化SHA=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918；norm_stats file SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。全部实测指标和清洗日志归档records；大型原数组与checkpoint保留v1-store，不进Git，实际字节量和耗时另记。

唯一会话名m2048-train100。下列命令体经detached tmux、外层pipefail+tee运行，日志v1-store/logs/m2048-train100.log，结束记录EXIT_CODE；TIC L0沿用现有检查器且不修改其实现。当前仅准备启动口径，所有PASS待实跑。

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
export CUDA_VISIBLE_DEVICES=6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export BENCH_REF_COMMIT="$M2048_HEAD" BENCH_CAND_COMMIT="$M2048_HEAD"
unset JAX_PLATFORMS BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR
unset BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP BENCH_PERF_MODE BENCH_EXTRA_DIGEST_STEPS BENCH_DATASET_IMPL
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data" TRAIN_TIMING_STEPS=0
export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1 BENCH_DUMP_IDX=1
M2048_RUN=m2048-train100
export BENCH_RECORD_DIR="$V1_STORE/bench/m2048/$M2048_RUN"
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$M2048_RUN"
export BENCH_STATE_DUMP_STEPS=0,99 BENCH_STATE_DUMP_DIR="$V1_STORE/bench/m2048/m2048-train100-arrays"
export BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=99 BENCH_EXTRA_DIGEST_STEPS=99
test ! -e "$BENCH_RECORD_DIR"
test ! -e "$BENCH_STATE_DUMP_DIR"
test ! -e "$V1_STORE/train-runs/$M2048_RUN"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump \
 --record-dir "$BENCH_RECORD_DIR" --v1-store "$V1_STORE" --source "$DS/source" \
 --manifest "$DS/meta/episode_manifest.json" --dataset "$DS/framesamp-8x8" \
 --norm-stats "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check \
 --base "$V1_STORE/bench/m2048/m2048-r20-refnpy" --record-dir "$BENCH_RECORD_DIR" \
 --allow-difference gpu.CUDA_VISIBLE_DEVICES --allow-difference dataset.store_meta_sha256 \
 --steps 100 --batch-size 8 --dataset "$DS/framesamp-8x8"
uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite \
 --exp-name "$M2048_RUN" --assets-base-dir "$V1_STORE/train-assets" \
 --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
 --data.assets.asset-id robomme --checkpoint-base-dir "$V1_STORE/train-runs/$M2048_RUN" \
 --batch-size 8 --num-workers 4 --num-train-steps 100 --log-interval 1 --save-interval 25 \
 --seed 42 --fsdp-devices 2 --no-wandb-enabled --dataset-path "$DS/framesamp-8x8" \
 --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
 --model.use-history --model.history-config perceptual-framesamp-modul-32frame-8x8.yaml
CKPT="$V1_STORE/train-runs/$M2048_RUN/mme_vla_suite/$M2048_RUN/999"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' uv run --no-sync python scripts/training/g0/check_config_provenance.py \
 --ckpt "$CKPT" --lib "$DS" --store-subdir framesamp-8x8 --train-config mme_vla_suite \
 --norm-stats "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json" \
 --out "$BENCH_RECORD_DIR/l0.json"
uv run --no-sync python scripts/training/tests/check_32frame_modul.py ckpt \
 --records "$BENCH_RECORD_DIR" --init-records "$V1_STORE/bench/m2048/m2048-r20-packed" \
 --state-dump-dir "$BENCH_STATE_DUMP_DIR" --ckpt "$CKPT" --out "$BENCH_RECORD_DIR/ckpt_checks.json"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/project_scalars.py \
 "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$BENCH_RECORD_DIR"
printf 'RUN_DONE name=%s steps=100 utc=%s\n' "$M2048_RUN" "$(date -u +%FT%TZ)"
```
