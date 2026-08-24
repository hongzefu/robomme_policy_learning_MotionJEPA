#!/usr/bin/env bash
# ── 实验 3：4×A40 compute-only 步时基准（run_name v1-computeonly-b64）────────────
# 喂同一缓存 batch 反复训练，测 batch 64 的纯计算 s/step → NFS 需求 = 1.20GB ÷ 步时。
# ⚠ 4 GPU 超出 greatlakes.md debug 包络（≤2 GPU），本 job 已由用户 2026-08-24 逐次特批
#   （--gpus-per-node=4 --cpus-per-task=8 --mem=64G --time=00:25:00）。
# 提交：gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-compute-only/gl_compute_only.sbatch"
#
#SBATCH --job-name=v1-computeonly-b64
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"
NCPU="${SLURM_CPUS_PER_TASK:-8}"
STEPS=100

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

RUN_NAME="v1-computeonly-b64"
RECORD_DIR="$STORE/bench/bottleneck/$RUN_NAME"
CKPT_DIR="$STORE/train-runs/mme_vla_suite/$RUN_NAME"
if [ -e "$RECORD_DIR" ]; then echo "错误: 记录目录已存在: $RECORD_DIR"; exit 1; fi
if [ -e "$CKPT_DIR" ]; then echo "错误: run 目录已存在: $CKPT_DIR"; exit 1; fi
mkdir -p "$RECORD_DIR"

"$PY" - "$RECORD_DIR" <<EOF
import json, platform, subprocess, sys, os
import jax
d = {
    "run_name": "$RUN_NAME", "batch_size": 64, "fsdp_devices": 4, "steps": $STEPS,
    "seed": 42, "history_config": "perceptual-framesamp-context.yaml",
    "dataset_path": "$STORE/datasets/4task-gl",
    "git_head": subprocess.run(["git", "-C", "$REPO", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
json.dump(d, open("$RECORD_DIR/env.json", "w"), indent=2, ensure_ascii=False)
EOF

# GPU 利用率采样（每 15 秒，4 卡逐卡一行）
GPULOG="$RECORD_DIR/gpu_util.csv"
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -v t="$(date +%s)" '{print t","$0}'
    sleep 15
  done ) >> "$GPULOG" 2>/dev/null &
SAMPLER=$!

echo "=== compute-only 步时基准 (node=$(hostname), job=$SLURM_JOB_ID, ${STEPS} steps) ==="
BENCH_RECORD_DIR="$RECORD_DIR" \
"$PY" "$REPO/scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py" mme_vla_suite \
    --exp-name "$RUN_NAME" \
    --assets-base-dir "$STORE/train-assets" \
    --checkpoint-base-dir "$STORE/train-runs" \
    --batch-size 64 \
    --num-workers 4 \
    --num-train-steps "$STEPS" \
    --log-interval 1 \
    --save-interval 1000 \
    --seed 42 \
    --fsdp-devices 4 \
    --dataset-path "$STORE/datasets/4task-gl" \
    --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
RC=$?
kill "$SAMPLER" 2>/dev/null || true

# 清理 run 目录（只剩 orbax 空壳与 history_config.txt）与节点侧 jax 缓存
if [ -e "$CKPT_DIR" ]; then
  case "$CKPT_DIR" in
    "$STORE/train-runs/mme_vla_suite/$RUN_NAME") rm -rf -- "$CKPT_DIR" ;;
  esac
fi
rm -rf -- "$HOME/.cache/jax_${RUN_NAME}"

if [ "$RC" -eq 0 ]; then
  "$PY" - "$RECORD_DIR" "$STEPS" <<'EOF'
import json, statistics, sys
d, steps = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if '"loss"' in l]
assert len(rows) == steps, f"metrics 行数 {len(rows)} != {steps}"
t = {r["step"]: r["wall_time"] for r in rows}
deltas = [t[s] - t[s - 1] for s in range(21, steps) if s in t and s - 1 in t]
steady = statistics.median(deltas)
need = 1.20e3 / steady   # b64 每步 ≈ 1.20 GB
print(f"RESULT compute_only={steady:.3f}s/step (n={len(deltas)}, "
      f"p10={sorted(deltas)[len(deltas)//10]:.3f}, p90={sorted(deltas)[len(deltas)*9//10]:.3f})")
print(f"RESULT NFS需求 = 1.20GB / {steady:.3f}s = {need:.0f} MB/s")
EOF
  RC=$?
fi
echo "EXIT_CODE=$RC"
exit "$RC"
