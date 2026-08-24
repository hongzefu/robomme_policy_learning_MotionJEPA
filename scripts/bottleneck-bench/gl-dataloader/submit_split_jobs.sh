#!/usr/bin/env bash
# ── 提交实验 1 拆分矩阵：6 个独立单档 job（workers × CPU 配比）───────────────────
# 原 18C 大 job（58619536）保留在队列里不取消；本脚本只新增小 job。
# 矩阵（cpus = workers×配比 + 2，主进程/采样器留 2；w16c10 是刻意的 CPU 超订档，
# 验证 worker CPU 预算是否压低供给）；seed 各不相同（200..205），与原 job 的
# 42+w（46/50/58）不重叠，防 page cache 跨 job 污染。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SBATCH_FILE="scripts/bottleneck-bench/gl-dataloader/gl_dlbench_single.sbatch"
#      tag      workers cpus mem  seed
MATRIX=(
  "w4c6    4   6  24G 200"
  "w4c10   4  10  24G 201"
  "w8c10   8  10  32G 202"
  "w8c18   8  18  32G 203"
  "w16c10 16  10  48G 204"
  "w16c18 16  18  48G 205"
)

for row in "${MATRIX[@]}"; do
  read -r TAG W C M S <<< "$row"
  echo "--- 提交 v1-dlb-${TAG} (workers=${W}, cpus=${C}, mem=${M}, seed=${S})"
  OUT=$(timeout 90 uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
    "sbatch --parsable --job-name=v1-dlb-${TAG} --cpus-per-task=${C} --mem=${M} --export=ALL,WORKERS=${W},BENCH_SEED=${S},TAG=${TAG} ${SBATCH_FILE}" 2>&1)
  JOBID=$(echo "$OUT" | grep -Eo '^[0-9]+$' | tail -1)
  if [ -z "$JOBID" ]; then
    echo "错误: v1-dlb-${TAG} 提交失败，输出如下：" >&2
    echo "$OUT" >&2
    exit 1
  fi
  echo "SUBMITTED v1-dlb-${TAG} jobid=${JOBID}"
done
echo "ALL_SUBMITTED"
