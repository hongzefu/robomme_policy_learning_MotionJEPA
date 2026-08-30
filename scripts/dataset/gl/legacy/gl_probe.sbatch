#!/usr/bin/env bash
# ── 多档位探针：一个模板跑遍所有候选 CPU/mem 档位 ─────────────────────────────
# 资源默认值写在下面，实际档位由提交时的 CLI 覆盖（--cpus-per-task / --mem），
# 一份模板服务全部档位，消灭手抄多份脚本的错配面。
#
# 它一次性回答四个问题：
#   ① NFS venv 的 jax[cuda12] 在 A40（sm_86）上能不能起来——本仓库是 JAX 栈，
#      集群既往只验证过 torch 栈，这是全流程最大的未知；
#   ② 该档位下的 A40 实测 step/s 与 GPU 平均利用率（选档判据 1、2）；
#   ③ 该档位下 cgroup 的真实不可回收内存 anon+shmem 峰值（选档判据 3）。
#      ⚠ 一律看 cgroup anon，**不看 MaxRSS**——greatlakes.md 已实证 MaxRSS 会精确
#      贴住 --mem 申请上限，那是页缓存填满配额的读数，不是工作集；
#   ④ 一份可立刻与本地基线对拍的 A40 产物（跨架构一致性的预演）。
#
# 全部在 greatlakes.md 调试限额内（1 GPU / ≤30 分钟），无需额外放行。
#
#SBATCH --job-name=v1-tierprobe
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
GL=/nfs/turbo/coe-chaijy-unreplicated/hongzefu
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"

for VAR in RAW_DIR MANIFEST SUBSET TIER_NAME; do
    if [ -z "${!VAR:-}" ]; then
        echo "缺必传参数 $VAR（应由 step_bench.sh 经 --export 传入），拒绝起跑"
        exit 1
    fi
done
OUT="${OUT:-$STORE/bench/probe-$TIER_NAME}"
NCPU="${SLURM_CPUS_PER_TASK:-2}"
SAMPLE_SEC="${SAMPLE_SEC:-5}"

# 线程数必须与 --cpus-per-task 对齐：不对齐会线程超订，把「档位不够」和
# 「线程打架」两个原因混在一起，档位结论就不可信了。
export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$STORE/bench" "$XDG_CACHE_HOME" "$HF_HOME" "$JAX_COMPILATION_CACHE_DIR"

echo "=== 档位探针 tier=$TIER_NAME cpus=$NCPU mem=${SLURM_MEM_PER_NODE:-?}M node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== [1/4] jax 可用性自检（本流程最大的未知）==="
"$PY" - <<'PYCHK'
import jax, jaxlib
d = jax.devices()
print(f"JAXCHK jax={jax.__version__} jaxlib={jaxlib.__version__} devices={d}")
assert d and d[0].platform == "gpu", f"JAX 没拿到 GPU: {d}"
print("JAXCHK OK")
PYCHK
RC=$?
if [ "$RC" -ne 0 ]; then echo "PROBE_EXIT_CODE=$RC 原因=jax 自检失败"; exit "$RC"; fi

echo "=== [2/4] 起采样器（GPU util + cgroup anon/file/shmem，每 ${SAMPLE_SEC}s）==="
CG_REL=$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null | head -n 1 || true)
CG_JOB_DIR="/sys/fs/cgroup${CG_REL%%/step_*}"
SAMPLOG="$STORE/bench/${TIER_NAME}-${SLURM_JOB_ID}.samples"
: > "$SAMPLOG"
(
    while true; do
        read -r UTIL MIB < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
        CUR=0; ANON=0; SHM=0; FILE=0
        if [ -r "$CG_JOB_DIR/memory.current" ]; then
            CUR=$(cat "$CG_JOB_DIR/memory.current" 2>/dev/null || echo 0)
            ANON=$(awk '$1=="anon"{print $2}'  "$CG_JOB_DIR/memory.stat" 2>/dev/null || echo 0)
            FILE=$(awk '$1=="file"{print $2}'  "$CG_JOB_DIR/memory.stat" 2>/dev/null || echo 0)
            SHM=$(awk '$1=="shmem"{print $2}'  "$CG_JOB_DIR/memory.stat" 2>/dev/null || echo 0)
        fi
        echo "sample gpu_util=${UTIL:-0} gpu_mib=${MIB:-0} cg_current=$CUR cg_anon=$ANON cg_file=$FILE cg_shmem=$SHM"
        sleep "$SAMPLE_SEC"
    done
) >> "$SAMPLOG" 2>/dev/null &
SAMPLER=$!

echo "=== [3/4] 跑探针子集（与本地基线同一批 episode）==="
"$PY" "$REPO/scripts/dataset/gl/build_shard.py" \
    --manifest "$MANIFEST" --raw_dir "$RAW_DIR" --out "$OUT" \
    --subset "$SUBSET" --shard_idx 0 --num_shards 1 --resume --report_every 200
RC=$?
kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true

echo "=== [4/4] 档位判据取数 ==="
"$PY" "$REPO/scripts/dataset/gl/legacy/sample_summary.py" "$SAMPLOG" --tier "${TIER_NAME}(cpus=${NCPU},mem=${SLURM_MEM_PER_NODE:-?}M)"

echo "PROBE_EXIT_CODE=$RC tier=$TIER_NAME"
exit "$RC"
