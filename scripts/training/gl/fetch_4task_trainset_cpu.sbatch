#!/usr/bin/env bash
#SBATCH --account=chaijy2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=g4c-fetch
# GreatLakes standard 分区纯 CPU job：把四任务训练库 bucket 中训练所需子集拉到 turbo，
# 子集 sha256 校验、解开 source/data_tars、pkl 计数、norm_stats 归位。
# 计划 1111-gl-4task400-ctx512-1024-b128-50k-plan.html §2.4 / runbook P1。
# standard 分区用于下载是用户批准的本任务一次性例外（greatlakes.md 默认只许 HF 校验）。
# 用法（脚本先 cp 到 <GL_REPO>/v1-store/jobs/ 再提交，不改 <GL_REPO> 的 checkout）：
#   sbatch --output=<GL_REPO>/v1-store/logs/%x-%j.log <GL_REPO>/v1-store/jobs/fetch_4task_trainset_cpu.sbatch
# 每步幂等，可重复提交；末行判定 TURBO_LIB=PASS|FAIL。
set -euo pipefail

G=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
BUCKET="hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1"
DS="$G/v1-store/datasets/4task-v2-1600ep-604f16da"
ASSETS="$G/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"

# 缓存逐项指向 v1-store，禁止覆盖 HOME
export HF_HOME="$G/v1-store/cache/hf"
export HF_XET_CACHE="$G/v1-store/cache/hf-xet"
export UV_CACHE_DIR="$G/v1-store/cache/uv"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE || true
hf() { "$HOME/.local/bin/uvx" --python 3.11 --from "huggingface_hub==1.30.0" hf "$@"; }

fail() { echo "TURBO_LIB=FAIL reason=$1"; exit 1; }
echo "START_UTC=$(date -u +%FT%TZ) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
ls -ld "$G/v1-store/datasets" || fail no_datasets_root
mkdir -p "$DS" "$HF_HOME" "$HF_XET_CACHE" "$UV_CACHE_DIR"

INC=(--include 'framesamp-8x8/*' --include 'meta/*' --include 'source/*' --include 'assets/*' --include 'SHA256SUMS.pre.txt')

echo "=== 1/5 dry-run 清单"
hf sync "$BUCKET" "$DS" "${INC[@]}" --dry-run 2>&1 | tail -n 40 || echo "提示: dry-run 不可用或失败，跳过清单预览，直接下载"

echo "=== 2/5 下载训练子集（跳过 framesamp/ 4x4、motion/、wan-latents_tars/、oracle_tars/、motion-tokens_tars/）"
t0=$(date +%s)
hf sync "$BUCKET" "$DS" "${INC[@]}" || fail download
t1=$(date +%s)
bytes=$(du -sb "$DS" | cut -f1)
echo "DOWNLOAD_DONE seconds=$((t1-t0)) bytes_in_ds=$bytes"

echo "=== 3/5 子集 sha256 校验（SHA256SUMS.pre.txt 中属于已下载子集的行）"
cd "$DS"
[ -f SHA256SUMS.pre.txt ] || fail no_sha_list
grep -E '  (\./)?(framesamp-8x8|meta|source|assets)/' SHA256SUMS.pre.txt > SHA256SUMS.subset.txt
n_expect=$(wc -l < SHA256SUMS.subset.txt)
n_have=0; while read -r _ f; do [ -f "$f" ] && n_have=$((n_have+1)); done < SHA256SUMS.subset.txt
[ "$n_have" = "$n_expect" ] || fail "missing_files have=$n_have expect=$n_expect"
# 8 路并行校验：按行切成 8 份
split -n l/8 -d SHA256SUMS.subset.txt .shasplit.
bad=0
for p in .shasplit.*; do ( sha256sum -c --quiet "$p" > "$p.out" 2>&1; echo $? > "$p.rc" ) & done; wait
for p in .shasplit.??; do r=$(cat "$p.rc"); [ "$r" = 0 ] || { bad=1; cat "$p.out"; }; done
mism=$(cat .shasplit.??.out | grep -c 'FAILED' || true)
rm -f .shasplit.*
[ "$bad" = 0 ] || fail "sha256 mismatches=$mism"
echo "SUBSET_SHA256=PASS files=$n_expect"

echo "=== 4/5 解开 source/data_tars → source/data/"
mkdir -p "$DS/source"
if [ -f "$DS/source/.data_untar_done" ]; then
  echo "已解包过（标记存在），跳过"
else
  ls "$DS"/source/data_tars/*.tar | xargs -P 8 -I{} tar -xf {} -C "$DS/source"
  touch "$DS/source/.data_untar_done"
fi
n=$(find "$DS/source/data" -name '*.pkl' | wc -l)
[ "$n" = "605611" ] || fail "pkl_count=$n"
echo "RESTORE_PKL=605611"

echo "=== 5/5 norm_stats 归位"
mkdir -p "$ASSETS/robomme"
cp -f "$DS/assets/robomme/norm_stats.json" "$ASSETS/robomme/norm_stats.json"
ns=$(sha256sum "$ASSETS/robomme/norm_stats.json" | cut -d' ' -f1)
echo "NORM_STATS_SHA256=$ns"

echo "END_UTC=$(date -u +%FT%TZ)"
echo "TURBO_LIB=PASS files=$n_expect mismatches=0 pkl=605611"
