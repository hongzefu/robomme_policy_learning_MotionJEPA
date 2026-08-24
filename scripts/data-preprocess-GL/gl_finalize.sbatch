#!/usr/bin/env bash
# ── 收尾守卫：--dependency=afterok 串在 8 分片 array 之后 ────────────────────────
# 只有八个分片全部 COMPLETED 才会跑。任一分片失败 → 本 job 被 SLURM 以
# kill_invalid_depend 自动 CANCELLED（此时连日志文件都不会生成，判死要靠 sacct）。
# 续跑时必须用 --dependency=afterok:<原AID>:<新分片JOBID> 把 finalize 一起重提。
#
# 需要 GPU 是因为最后一道抽检要重新跑 SigLIP 前向做零容差复算；
# spgpu 也强制每个 job 至少 1 张卡（--gpus-per-node=0 会被 QOSMinGRES 拒绝）。
#
# ⚠ --mem 必须跟分片档位同量级，**不能沿用小默认值**。本轮实测踩过：档位实测把 array
#   定到 24 G，finalize 却留在 12 G 默认值，结果前三步（输入同源 / 完整性 / sidecar 汇总）
#   全绿之后，在第四步加载 SigLIP 做抽检时被 OOM 杀掉（job 58531841，OUT_OF_MEMORY，
#   1:34，退出码 0:125）。finalize 和分片一样要载 1.66 GB 权重并反复前向，内存需求同量级。
#   数据本身不受影响——它只是守卫，重提即可。
#
#SBATCH --job-name=v1-4task-build-final
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL

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
INPUT_MANIFEST="${INPUT_MANIFEST:-$STORE/input_manifest.json}"
SPOT_CHECK="${SPOT_CHECK:-256}"
# size 档够用：本机原件与 turbo 副本的 sha256 等价已在 step0 一次性证过，
# 且该目录此后无写入方；全量 sha256 要在 NFS 上读 321 GB（约 40 分钟）。
INPUT_LEVEL="${INPUT_LEVEL:-size}"
NCPU="${SLURM_CPUS_PER_TASK:-4}"

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "=== finalize node=$(hostname) out=$OUT spot_check=$SPOT_CHECK ==="
"$PY" "$REPO/scripts/data-preprocess-GL/finalize_checks.py" check \
    --manifest "$MANIFEST" --out "$OUT" --raw_dir "$RAW_DIR" \
    --input_manifest "$INPUT_MANIFEST" --input_level "$INPUT_LEVEL" \
    --spot_check "$SPOT_CHECK"
RC=$?
[ "$RC" -ne 0 ] && echo "FINALIZE_EXIT_CODE=$RC"
exit "$RC"
