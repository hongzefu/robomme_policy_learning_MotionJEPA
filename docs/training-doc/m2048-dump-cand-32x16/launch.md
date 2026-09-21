# m2048-dump-cand-32x16 起跑口径

按用户「一路实现到测速结束报告用户为止」及「保持原计划，完整取证（推荐）」执行。该项只读1600ep库：framesamp；history配置perceptual-framesamp-modul.yaml，输入实现packed。不创建派生库，不抽SigLIP，不改norm_stats。CPU运行，CUDA_VISIBLE_DEVICES为空，JAX_PLATFORMS=cpu。工作区须为生产与验证工具提交后的同一clean CAND，实际完整HEAD与UTC由启动日志记录。

输入清单file SHA=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，规范化SHA=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918；norm_stats file SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。底层为AWS /dev/md0 XFS NVMe RAID。完整身份遍历覆盖1600集、1192918帧、605611个执行样本；定点3000样本/200批，limit=0，原数组不落盘。origin_mode为per_episode_offset、covers_zero_pad=false；allshort/mixed1在本库只是组名，不能证明真实短历史。

与对应另一侧以compare_fixture_dumps.py逐键核dtype、shape、raw与canonical摘要；源NPY和packed仅有特征来源、padding和身份换算三跳独立，采样、归一化与下游共享。新2048完整dump之前先完成有界assembly检查；旧档则只比较同配置BASE/CAND，不做512与2048数值等价断言。

唯一会话名m2048-dump-cand-32x16，以下命令体放detached tmux；外层pipefail+tee写v1-store/logs/m2048-dump-cand-32x16.log，结束追加EXIT_CODE。每份日志独立流式监听，当前仅准备入口。

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
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export DTYPE_DUMP_IMPL=packed DTYPE_DUMP_MODE=both DTYPE_DUMP_ARRAYS=0 DTYPE_DUMP_LIMIT=0
export DTYPE_DUMP_GIT_HEAD="$M2048_HEAD"
export DTYPE_DUMP_DIR="$V1_STORE/bench/m2048/m2048-dump-cand-32x16"
test ! -e "$DTYPE_DUMP_DIR"
uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite --exp-name m2048-dump-cand-32x16 \
 --assets-base-dir "$V1_STORE/train-assets" \
 --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
 --data.assets.asset-id robomme --dataset-path "$DS/framesamp" \
 --model.use-history --model.history-config perceptual-framesamp-modul.yaml --no-wandb-enabled
```
