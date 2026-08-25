#!/usr/bin/env bash
# ── bottleneck-bench v2 提交脚本：三档 CPU/worker/RAM 修复验证 ───────────────────
# 三档共用 gl_e2e_fix.sbatch（16C/96G/2h/4×A40/600 步），只变 num_workers：
#   第一批（默认，同时提交）：w8c16(seed 210) + w16c16(seed 211)
#   第二批（BATCH=2，前两档入队后由用户指令延后提交）：w12c16(seed 212)
# seed 各不相同且避开 42/200-205，防同节点复跑时 page cache 跨 job 污染。
# ⚠ 每个 job 都是 4 GPU + 2h，超 debug 包络，提交前须已获用户逐次特批。
# 用法：bash scripts/bottleneck-bench-v2/submit_fix_jobs.sh        # 第一批
#       BATCH=2 bash scripts/bottleneck-bench-v2/submit_fix_jobs.sh  # 第二批 w12
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SBATCH_FILE="scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch"
#      tag              workers seed
BATCH1=(
  "v1-e2efix-w8c16   8  210"
  "v1-e2efix-w16c16 16  211"
)
BATCH2=(
  "v1-e2efix-w12c16 12  212"
)

if [ "${BATCH:-1}" = "2" ]; then ROWS=("${BATCH2[@]}"); else ROWS=("${BATCH1[@]}"); fi

for row in "${ROWS[@]}"; do
  read -r TAG W S <<< "$row"
  echo "--- 提交 ${TAG} (workers=${W}, cpus=16, mem=96G, seed=${S})"
  OUT=$(timeout 90 uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
    "sbatch --parsable --job-name=${TAG} --export=ALL,WORKERS=${W},TAG=${TAG},BENCH_SEED=${S} ${SBATCH_FILE}" 2>&1)
  JOBID=$(echo "$OUT" | grep -Eo '^[0-9]+$' | tail -1)
  if [ -z "$JOBID" ]; then
    echo "错误: ${TAG} 提交失败，输出如下：" >&2
    echo "$OUT" >&2
    exit 1
  fi
  echo "SUBMITTED ${TAG} jobid=${JOBID}"
done
echo "ALL_SUBMITTED"
