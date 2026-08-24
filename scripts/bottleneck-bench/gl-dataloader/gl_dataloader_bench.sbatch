#!/usr/bin/env bash
# ── 实验 1：GL A40 节点 dataloader-only 吞吐基准（run_name v1-gl-dlbench）─────────
# batch 64、扫 workers 4/8/16，只迭代不训练；判据链见 scripts/bottleneck-bench/README.md。
# 资源在 debug 包络内（1 GPU、30 min；spgpu 强制 ≥1 GPU 故必须带一张卡）。
# 提交：gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-dataloader/gl_dataloader_bench.sbatch"
#
#SBATCH --job-name=v1-gl-dlbench
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"
NCPU="${SLURM_CPUS_PER_TASK:-18}"

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RECORD_DIR="$STORE/bench/bottleneck/v1-gl-dlbench"
if [ -e "$RECORD_DIR" ]; then
    echo "错误: 记录目录已存在, 禁止覆盖: $RECORD_DIR"; exit 1
fi
mkdir -p "$RECORD_DIR"

# 环境留档（含 SLURM 信息）
"$PY" - "$RECORD_DIR" <<EOF
import json, platform, subprocess, sys, os
import jax
d = {
    "run_name": "v1-gl-dlbench", "batch_size": 64, "workers_list": "4 8 16",
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

echo "=== dataloader-only 吞吐基准 (node=$(hostname), job=$SLURM_JOB_ID) ==="
BENCH_RECORD_DIR="$RECORD_DIR" WORKERS_LIST="4 8 16" \
"$PY" "$REPO/scripts/bottleneck-bench/gl-dataloader/dataloader_bench.py" mme_vla_suite \
    --exp-name v1-gl-dlbench \
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
