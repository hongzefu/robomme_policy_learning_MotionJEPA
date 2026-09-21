# m2048-dump-base-32x16 启动记录

用户原话：「开始实现 有问题立刻越早问用户越好 一路实现到测速结束报告用户为止」。执行 [2048 计划](../../../0920-32frame-8x8-modul-2048-plan.md) 步 2a，本档只建立旧 32x16 modulation、motion 关闭的改前证据。

版本锚点为引入本启动档案及第一批工具适配的同一个提交，运行前强制 porcelain 为空；完整实际提交、UTC 起点由下列命令输出的 START_HEAD / START_UTC 和原始日志记录，后续结果档案回填该值。生产 src 与初始提交 c31b0509be3f76a6cc262fd655f5f0080b12c266 相同，依赖不变。环境 B：8×A100-SXM4-80GB，本档零 GPU，只用 CPU；底层存储为 /dev/md0 XFS 本地 NVMe RAID。

输入仅为 1600ep 实体库 framesamp、同根 source 和 meta/episode_manifest.json。清单文件 SHA256=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482；规范化内容指纹=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918；norm_stats 文件 SHA256=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。库 verified/full，1600 集、1192918 帧、605611 执行样本。本库不存在短历史，fixture 的 allshort/mixed1 仅为沿用组名。

完整 dump 不设 limit，保留全部键的 dtype/shape/raw/canonical 摘要，200 批。max_frames=32，3000 样本。原始输入逐帧与 pkl 身份全部校验；产物暂不表示候选版本等价，须等 CAND 对拍。

下列命令体放入 detached tmux `m2048-dump-base-32x16`；外层用 `set -o pipefail`、`tee v1-store/logs/m2048-dump-base-32x16.log`，结束记录 `EXIT_CODE=`。本档会话只有该完整名称。原始产物保留 v1-store/bench/m2048/，清洗日志与摘要随后归档 records/；checkpoint 不进 Git。

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
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export DTYPE_DUMP_IMPL=packed DTYPE_DUMP_MODE=both DTYPE_DUMP_ARRAYS=0 DTYPE_DUMP_LIMIT=0
export DTYPE_DUMP_GIT_HEAD="$M2048_BASE"
export DTYPE_DUMP_DIR="$V1_STORE/bench/m2048/m2048-dump-base-32x16"
test ! -e "$DTYPE_DUMP_DIR"
uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite \
 --assets-base-dir "$V1_STORE/train-assets" \
 --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
 --data.assets.asset-id robomme --dataset-path "$DS/framesamp" \
 --model.use-history --model.history-config perceptual-framesamp-modul.yaml --no-wandb-enabled
```
