#!/usr/bin/env bash
# ── 实验 4：4×A40 官方口径 b64 端到端 300 步真实吞吐（run_name v1-e2e-b64）───────
# 真 dataloader 走 NFS + 真训练循环（零改动复用 scripts/smoke-local/bench_train_steps.py），
# 与官方 finetune_mme_vla_suite.sh 的差异仅：300 步截断、log-interval 1（逐步记录，
# 不进计算图）、save-interval 1000（参数校验和仅 train.py 强制的末步一次）、wandb 关、
# 路径显式化。与实验 1（供给）/实验 3（需求）互为验证：
#   实测稳态步时 ≈ max(compute-only 步时, 1.20GB ÷ 供给 MB/s)
# ⚠ 4 GPU + 1h 超出 greatlakes.md debug 包络，已由用户 2026-08-24 逐次特批
#   （--gpus-per-node=4 --cpus-per-task=8 --mem=64G --time=01:00:00）。
# 提交：gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-e2e/gl_e2e_b64.sbatch"
#
#SBATCH --job-name=v1-e2e-b64
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"
NCPU="${SLURM_CPUS_PER_TASK:-8}"
STEPS=300
MOUNT_POINT="/nfs/turbo/coe-chaijy-unreplicated"

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

RUN_NAME="v1-e2e-b64"
RECORD_DIR="$STORE/bench/bottleneck/$RUN_NAME"
CKPT_DIR="$STORE/train-runs/mme_vla_suite/$RUN_NAME"
if [ -e "$RECORD_DIR" ]; then echo "错误: 记录目录已存在: $RECORD_DIR"; exit 1; fi
if [ -e "$CKPT_DIR" ]; then echo "错误: run 目录已存在: $CKPT_DIR"; exit 1; fi
mkdir -p "$RECORD_DIR"

"$PY" - "$RECORD_DIR" <<EOF
import json, platform, subprocess, sys, os
import jax
d = {
    "run_name": "$RUN_NAME", "batch_size": 64, "fsdp_devices": 4, "num_workers": 4,
    "steps": $STEPS, "seed": 42, "history_config": "perceptual-framesamp-context.yaml",
    "dataset_path": "$STORE/datasets/4task-gl",
    "git_head": subprocess.run(["git", "-C", "$REPO", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
json.dump(d, open("$RECORD_DIR/env.json", "w"), indent=2, ensure_ascii=False)
EOF

# 遥测（每 15 秒）：GPU 利用率（4 卡逐卡）+ turbo 真实读字节（server_read；
# per-mount 全局计数，含本 job 的 spawn worker，也含节点其他进程——分析时注明）
GPULOG="$RECORD_DIR/gpu_util.csv"
NFSLOG="$RECORD_DIR/nfs_read.csv"
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -v t="$(date +%s)" '{print t","$0}'
    sleep 15
  done ) >> "$GPULOG" 2>/dev/null &
GPU_SAMPLER=$!
( while true; do
    "$PY" - "$MOUNT_POINT" <<'PYEOF'
import sys, time
mp = sys.argv[1]; in_m = False
for line in open("/proc/self/mountstats"):
    if line.startswith("device ") and f" mounted on {mp} " in line: in_m = True
    elif line.startswith("device "): in_m = False
    elif in_m and line.strip().startswith("bytes:"):
        f = line.split()[1:]
        print(f"{time.time()},{f[0]},{f[4]}")
        break
PYEOF
    sleep 15
  done ) >> "$NFSLOG" 2>/dev/null &
NFS_SAMPLER=$!

echo "=== e2e b64 真实吞吐基准 (node=$(hostname), job=$SLURM_JOB_ID, ${STEPS} steps) ==="
BENCH_RECORD_DIR="$RECORD_DIR" \
"$PY" "$REPO/scripts/smoke-local/bench_train_steps.py" mme_vla_suite \
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
kill "$GPU_SAMPLER" "$NFS_SAMPLER" 2>/dev/null || true

if [ -e "$CKPT_DIR" ]; then
  case "$CKPT_DIR" in
    "$STORE/train-runs/mme_vla_suite/$RUN_NAME") rm -rf -- "$CKPT_DIR" ;;
  esac
fi
rm -rf -- "$HOME/.cache/jax_${RUN_NAME}"

if [ "$RC" -eq 0 ]; then
  "$PY" - "$RECORD_DIR" "$STEPS" <<'EOF'
import json, math, statistics, sys
d, steps = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if '"loss"' in l]
assert len(rows) == steps, f"metrics 行数 {len(rows)} != {steps}"
losses = [r["loss"]["dec"] for r in rows]
bad = [v for v in losses if not math.isfinite(v)]
assert not bad, f"非有限 loss: {bad[:5]}"
t = {r["step"]: r["wall_time"] for r in rows}
warmup = 50
deltas = [t[s] - t[s-1] for s in range(warmup + 1, steps)]
steady = statistics.median(deltas)
t_lo, t_hi = t[warmup], t[steps - 1]
nfs = [l.strip().split(",") for l in open(f"{d}/nfs_read.csv") if l.strip()]
win = [(float(a), int(c)) for a, b, c in nfs if t_lo <= float(a) <= t_hi]
mbps = (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0]) / 1e6 if len(win) >= 2 else float("nan")
gpu = [l.strip().split(",") for l in open(f"{d}/gpu_util.csv") if l.strip()]
utils = [int(u) for ts, idx, u, m in gpu if t_lo <= float(ts) <= t_hi]
gpu_med = statistics.median(utils) if utils else float("nan")
print(f"OK loss n={len(losses)} min={min(losses):.4f} max={max(losses):.4f} 末值={losses[-1]:.4f}")
print(f"RESULT e2e={steady:.3f}s/step (n={len(deltas)}, "
      f"p10={sorted(deltas)[len(deltas)//10]:.3f}, p90={sorted(deltas)[len(deltas)*9//10]:.3f})")
print(f"RESULT 吞吐={64/steady:.2f} 样本/s  GPU利用率中位={gpu_med:.0f}%")
print(f"RESULT NFS实测={mbps:.0f} MB/s（公式口径 {1.20e3/steady:.0f} MB/s = 1.20GB/步时）")
print(f"RESULT epoch(6176步)={6176*steady:.0f}s ≈ {6176*steady/3600:.2f} 小时")
EOF
  RC=$?
fi
echo "EXIT_CODE=$RC"
exit "$RC"
