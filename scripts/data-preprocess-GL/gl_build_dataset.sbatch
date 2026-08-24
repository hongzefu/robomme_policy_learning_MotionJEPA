#!/usr/bin/env bash
# ── 四任务全量数据构建：8×1GPU job array ──────────────────────────────────────
# 一份模板 = 全部分片：--array 一次提交 8 个独立调度的 1-GPU job，
# $SLURM_ARRAY_TASK_ID 直接当 shard_idx（消灭手抄多份脚本的错配面）。
# array task 逐个独立调度，等价于「8 个分开请求的 1-GPU 小 job」，某片排不上队
# 不阻塞其余七片，先起的先干活。
#
# ⚠ 本模板不手提，由 step_submit.sh 经 gl_submit.py 提交（它负责 pre-flight 与
#   afterok 串接 finalize）；RAW_DIR / OUT / MANIFEST 三个路径必须经 --export 显式
#   传入，缺任一立即退出（防裸跑把几百 GB 写进无关目录）。
#
# ⚠ CPU / mem 由 step_bench.sh 的档位实测结果在提交时用 CLI 覆盖决定，
#   下面的默认值只是兜底。判据见 docs/v1-gl-resource-tier-bench.md。
#
# ⚠ 8×1GPU 并发与长 walltime 超出 greatlakes.md 的调试限额（≤2 GPU、≤30 分钟），
#   必须由用户逐次显式放行后才可提交。
#
# 续跑（某分片超时/被杀后）：先删该分片的 _claim_shard*.json，再
#   sbatch --array=<分片号> --export=ALL,REQUIRE_EMPTY=0,<三路径> 同脚本重提。
#   build_shard.py 的 --resume 会跳过已完整落盘的 episode，中断损失上界 = 单
#   episode ≤ 586 步。
# ⚠ 任一分片失败 → 原 finalize 因 afterok 不满足被 kill_invalid_depend 自动
#   CANCELLED——重提分片后必须用
#   sbatch --dependency=afterok:<原AID>:<新JOBID> 连 finalize 一起重提。
#
#SBATCH --job-name=v1-4task-build
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%A_%a.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"

for VAR in RAW_DIR OUT MANIFEST; do
    if [ -z "${!VAR:-}" ]; then
        echo "缺必传参数 $VAR（应由 step_submit.sh 经 --export 传入），拒绝起跑"
        exit 1
    fi
done
NUM_SHARDS="${NUM_SHARDS:-8}"
REQUIRE_EMPTY="${REQUIRE_EMPTY:-1}"
SHARD_IDX="$SLURM_ARRAY_TASK_ID"
NCPU="${SLURM_CPUS_PER_TASK:-2}"

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$XDG_CACHE_HOME" "$HF_HOME" "$JAX_COMPILATION_CACHE_DIR"

echo "=== [1/4] 输入与清单核验 ==="
"$PY" - "$MANIFEST" "$RAW_DIR" "$NUM_SHARDS" <<'PYCHK'
import json, os, sys
sys.path.insert(0, os.path.join(
    "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA",
    "scripts", "data-preprocess-GL"))
from scan_manifest import load_manifest          # 内含 sha256 校验，被改过就直接抛
m = load_manifest(sys.argv[1])
raw, nshard = sys.argv[2], int(sys.argv[3])
assert m["num_shards"] == nshard, f"清单 num_shards={m['num_shards']} != {nshard}"
assert m["totals"]["episodes"] == 1600, f"清单 episode 数异常: {m['totals']['episodes']}"
for name in m["canonical_order"]:
    p = os.path.join(raw, name)
    assert os.path.isfile(p), f"缺输入 H5: {p}"
print(f"MANIFESTCHK OK episodes={m['totals']['episodes']} "
      f"timesteps={m['totals']['timesteps']} sha={m['sha256'][:16]}")
PYCHK
RC=$?
if [ "$RC" -ne 0 ]; then echo "SHARD_EXIT_CODE=$RC shard=$SHARD_IDX 原因=清单核验失败"; exit "$RC"; fi

echo "=== [2/4] shard claim（O_EXCL，撞车 1 秒内报错而非几小时后）==="
CLAIM_DIR="$OUT/_claims"; mkdir -p "$CLAIM_DIR"
CLAIM="$CLAIM_DIR/_claim_shard${SHARD_IDX}of${NUM_SHARDS}.json"
if ! ( set -o noclobber; echo "{\"job\":\"$SLURM_JOB_ID\",\"host\":\"$(hostname)\",\"time\":\"$(date -Is)\"}" > "$CLAIM" ) 2>/dev/null; then
    echo "CLAIM_EXISTS: $CLAIM 已被占用（内容如下）——如系上轮中断遗留，人工删除后重提"
    cat "$CLAIM"
    exit 1
fi

echo "=== [3/4] 起 cgroup + 显存采样（每 60 秒一条）==="
CG_REL=$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null | head -n 1 || true)
CG_JOB_DIR="/sys/fs/cgroup${CG_REL%%/step_*}"
MEMLOG="$STORE/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SHARD_IDX}.mem"
(
    while true; do
        read -r UTIL MIB < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
        CUR=0; ANON=0; SHM=0; FILE=0
        if [ -r "$CG_JOB_DIR/memory.current" ]; then
            CUR=$(cat "$CG_JOB_DIR/memory.current" 2>/dev/null || echo 0)
            ANON=$(awk '$1=="anon"{print $2}' "$CG_JOB_DIR/memory.stat" 2>/dev/null || echo 0)
            FILE=$(awk '$1=="file"{print $2}' "$CG_JOB_DIR/memory.stat" 2>/dev/null || echo 0)
            SHM=$(awk '$1=="shmem"{print $2}' "$CG_JOB_DIR/memory.stat" 2>/dev/null || echo 0)
        fi
        echo "sample gpu_util=${UTIL:-0} gpu_mib=${MIB:-0} cg_current=$CUR cg_anon=$ANON cg_file=$FILE cg_shmem=$SHM"
        sleep 60
    done
) >> "$MEMLOG" 2>/dev/null &
SAMPLER=$!

echo "=== [4/4] 构建 shard $SHARD_IDX/$NUM_SHARDS ==="
EXTRA=()
[ "$REQUIRE_EMPTY" = "1" ] && EXTRA+=(--require_empty_output)
"$PY" "$REPO/scripts/data-preprocess-GL/build_shard.py" \
    --manifest "$MANIFEST" --raw_dir "$RAW_DIR" --out "$OUT" \
    --shard_idx "$SHARD_IDX" --num_shards "$NUM_SHARDS" --resume "${EXTRA[@]}"
RC=$?

kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true
if [ "$RC" -eq 0 ]; then
    rm -f "$CLAIM"          # 正常收尾才移除 claim；残留 claim = 该分片没跑完
fi
if [ -f "$MEMLOG" ]; then
    awk '{for(i=1;i<=NF;i++){n=split($i,kv,"=");if(n==2&&kv[2]+0>max[kv[1]])max[kv[1]]=kv[2]+0}}
         END{for(k in max)printf "peak %s = %.0f (%.2f GiB)\n",k,max[k],
             max[k]/(k=="gpu_mib"||k=="gpu_util"?1:1073741824)}' "$MEMLOG"
fi
echo "SHARD_EXIT_CODE=$RC shard=$SHARD_IDX"
exit "$RC"
