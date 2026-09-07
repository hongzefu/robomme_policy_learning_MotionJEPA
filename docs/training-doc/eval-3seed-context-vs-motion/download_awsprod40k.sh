#!/usr/bin/env bash
# awsprod40k-b128-motion 的 step 39999 checkpoint + run 根元文件下载（环境 A / 本机）。
# 源是私有 bucket hf://buckets/HongzeFu/robomme-motionjepa-vla-v1（上传留档见
# docs/dataset-build-doc/hf-export-awsprod40k-ckpt-v1/，全量 166 文件 / 92,689,314,972 B，8 个 step）。
# 本轮只要 39999：--include 精确到 20 个 39999/** 文件 + 7 个 run 根元文件 = 27 文件 / 11,584,776,670 B，
# 已由 --dry-run 逐条核对命中清单（不带 *.json 泛匹配，否则会带下其余 7 个 step 的 norm_stats.json 造出空壳目录）。
# run 根的 history_config.resolved.yaml 决定 eval_shard.sh 走「快照 + motion sidecar」分支，缺则走错分支，故必须同下。
# 落点 v1-store/models/awsprod40k-b128-motion/，即 CKPT_DIR=<该目录>/39999 的父目录。
set -o pipefail
MAIN=/data/hongzefu/robomme_policy_learning_MotionJEPA
export PATH=$HOME/.local/bin:$PATH
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"   # HongzeFu 账号；bucket 为 private
export PYTHONUNBUFFERED=1
DST=$MAIN/v1-store/models/awsprod40k-b128-motion
LOG=$MAIN/v1-store/logs/dl-awsprod40k.log
mkdir -p "$DST" "$MAIN/v1-store/logs"
{
echo "DL_START=$(date '+%F %T') dst=$DST"
hf sync hf://buckets/HongzeFu/robomme-motionjepa-vla-v1 "$DST" \
  --include "39999/**" \
  --include "history_config.resolved.yaml" --include "history_config.resolved.sha256" \
  --include "history_config.txt" --include "motion_provenance.json" --include "wandb_id.txt" \
  --include "README.md" --include "SHA256SUMS.pre.txt"
EC=$?
echo "DL_END=$(date '+%F %T')"
echo "EXIT_CODE=$EC"
} 2>&1 | tee -a "$LOG"
