#!/usr/bin/env bash
# ── 实验 1 拆分版：单档 dataloader-only 吞吐 job（workers × CPU 配比矩阵之一）─────
# 原 18C 大 job（58619536）排队过久，拆成独立小 job 并行排队；原 job 保留不取消。
# WORKERS / BENCH_SEED / TAG 必须经 --export 传入；--job-name / --cpus-per-task /
# --mem 由 submit_split_jobs.sh 在提交时用 sbatch CLI 覆盖（下面的值只是兜底）。
# 全部档位 1×A40 + ≤30 min，落在 greatlakes.md debug 包络内。
#
#SBATCH --job-name=v1-dlb-single
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"
NCPU="${SLURM_CPUS_PER_TASK:-6}"

for VAR in WORKERS BENCH_SEED TAG; do
    if [ -z "${!VAR:-}" ]; then
        echo "缺必传参数 $VAR（应由 submit_split_jobs.sh 经 --export 传入），拒绝起跑"
        exit 1
    fi
done

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RECORD_DIR="$STORE/bench/bottleneck/v1-dlb-${TAG}"
if [ -e "$RECORD_DIR" ]; then
    echo "错误: 记录目录已存在, 禁止覆盖: $RECORD_DIR"; exit 1
fi
mkdir -p "$RECORD_DIR"

"$PY" - "$RECORD_DIR" <<EOF
import json, platform, subprocess, sys, os
import jax
d = {
    "run_name": "v1-dlb-${TAG}", "batch_size": 64,
    "workers": int("${WORKERS}"), "seed": int("${BENCH_SEED}"),
    "warmup_batches": 5, "measure_batches": 40,
    "dataset_path": "$STORE/datasets/4task-gl",
    "git_head": subprocess.run(["git", "-C", "$REPO", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    "mem": os.environ.get("SLURM_MEM_PER_NODE"),
    "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
json.dump(d, open("$RECORD_DIR/env.json", "w"), indent=2, ensure_ascii=False)
EOF

echo "=== dataloader-only 单档基准 v1-dlb-${TAG} (workers=${WORKERS}, cpus=${NCPU}, node=$(hostname), job=$SLURM_JOB_ID) ==="
BENCH_RECORD_DIR="$RECORD_DIR" WORKERS_LIST="$WORKERS" BENCH_SEED="$BENCH_SEED" \
"$PY" "$REPO/scripts/bottleneck-bench/gl-dataloader/dataloader_bench.py" mme_vla_suite \
    --exp-name "v1-dlb-${TAG}" \
    --assets-base-dir "$STORE/train-assets" \
    --checkpoint-base-dir "$STORE/train-runs" \
    --batch-size 64 \
    --dataset-path "$STORE/datasets/4task-gl" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
RC=$?
echo "EXIT_CODE=$RC"
exit "$RC"
