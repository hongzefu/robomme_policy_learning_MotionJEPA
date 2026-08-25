#!/usr/bin/env bash
# ── bottleneck-bench v2：CPU/worker/RAM 侧修复验证（e2e 600 步，只变 num_workers）──
# 目的：验证「16 CPU + 8-16 workers + 96G」能否把 v1-e2e-b64 暴露的 GPU 空转吃满
# （v1 基线：8C/4w/64G，util 均值 69-70%，慢步(>8s)占稳态墙钟 33%，6.933 s/step）。
# 与 v1 gl_e2e_b64.sbatch 的差异仅在启动参数与遥测，不改任何 dataloader 代码逻辑：
#   --cpus-per-task 8→16、--mem 64G→96G（v1 的 64G 在末步参数校验和 device_get
#   ~28GB 拉回 host 时被 OOM-kill）、步数 300→600、--time 1h→2h、
#   num_workers 由 --export 传入（8/12/16 三档）、GPU 采样改双通道（见下）。
# WORKERS / TAG / BENCH_SEED 必须经 --export 传入（TAG 即 run_name，如 v1-e2efix-w8c16；
# 各档 seed 互不相同且避开 42/200-205，防同节点复跑时 page cache 跨 job 污染）。
# ⚠ 4 GPU + 2h 超出 greatlakes.md debug 包络，已由用户 2026-08-24 计划批准逐次特批
#   （--gpus-per-node=4 --cpus-per-task=16 --mem=96G --time=02:00:00，三档）。
# 提交：见同目录 submit_fix_jobs.sh。
#
#SBATCH --job-name=v1-e2efix
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
PY="$REPO/.venv/bin/python"
STORE="$REPO/v1-store"
NCPU="${SLURM_CPUS_PER_TASK:-16}"
STEPS=600
MOUNT_POINT="/nfs/turbo/coe-chaijy-unreplicated"

for VAR in WORKERS TAG BENCH_SEED; do
    if [ -z "${!VAR:-}" ]; then
        echo "缺必传参数 $VAR（应由 submit_fix_jobs.sh 经 --export 传入），拒绝起跑"
        exit 1
    fi
done

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

RUN_NAME="$TAG"
RECORD_DIR="$STORE/bench/bottleneck/$RUN_NAME"
CKPT_DIR="$STORE/train-runs/mme_vla_suite/$RUN_NAME"
if [ -e "$RECORD_DIR" ]; then echo "错误: 记录目录已存在: $RECORD_DIR"; exit 1; fi
if [ -e "$CKPT_DIR" ]; then echo "错误: run 目录已存在: $CKPT_DIR"; exit 1; fi
mkdir -p "$RECORD_DIR"

"$PY" - "$RECORD_DIR" <<EOF
import json, platform, subprocess, sys, os
import jax
d = {
    "run_name": "$RUN_NAME", "batch_size": 64, "fsdp_devices": 4,
    "num_workers": int("$WORKERS"), "cpus": int("$NCPU"),
    "mem": os.environ.get("SLURM_MEM_PER_NODE"),
    "steps": $STEPS, "seed": int("$BENCH_SEED"),
    "history_config": "perceptual-framesamp-context.yaml",
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

# ── GPU 遥测双通道 ───────────────────────────────────────────────────────────────
# legacy 通道：与 v1 完全一致（15 秒循环，epoch秒,卡号,util%,显存MiB），
#   专供与 v1-e2e-b64 基线同口径对照，格式/间隔一律不动。
# dense 通道（主判读依据）：nvidia-smi 流式采样，单常驻进程，每 500ms 一轮，
#   自带毫秒时间戳（节点本地时区）。500ms 是有效密度上限——utilization.gpu 本身
#   是 NVML 约 1/6~1 秒内部周期的均值，更密只会重复读数。
GPULOG="$RECORD_DIR/gpu_util.csv"
GPUDENSE="$RECORD_DIR/gpu_util_dense.csv"
NFSLOG="$RECORD_DIR/nfs_read.csv"
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -v t="$(date +%s)" '{print t","$0}'
    sleep 15
  done ) >> "$GPULOG" 2>/dev/null &
GPU_SAMPLER=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 >> "$GPUDENSE" 2>/dev/null &
GPU_DENSE_SAMPLER=$!
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

echo "=== e2efix b64 修复验证 (tag=$TAG, workers=$WORKERS, cpus=$NCPU, node=$(hostname), job=$SLURM_JOB_ID, ${STEPS} steps) ==="
BENCH_RECORD_DIR="$RECORD_DIR" \
"$PY" "$REPO/scripts/smoke-local/bench_train_steps.py" mme_vla_suite \
    --exp-name "$RUN_NAME" \
    --assets-base-dir "$STORE/train-assets" \
    --checkpoint-base-dir "$STORE/train-runs" \
    --batch-size 64 \
    --num-workers "$WORKERS" \
    --num-train-steps "$STEPS" \
    --log-interval 1 \
    --save-interval 1000 \
    --seed "$BENCH_SEED" \
    --fsdp-devices 4 \
    --dataset-path "$STORE/datasets/4task-gl" \
    --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
RC=$?
kill "$GPU_SAMPLER" "$GPU_DENSE_SAMPLER" "$NFS_SAMPLER" 2>/dev/null || true

if [ -e "$CKPT_DIR" ]; then
  case "$CKPT_DIR" in
    "$STORE/train-runs/mme_vla_suite/$RUN_NAME") rm -rf -- "$CKPT_DIR" ;;
  esac
fi
rm -rf -- "$HOME/.cache/jax_${RUN_NAME}"

if [ "$RC" -eq 0 ]; then
  "$PY" "$REPO/scripts/bottleneck-bench-v2/analyze_gpu_util.py" \
      "$RECORD_DIR" --steps "$STEPS"
  RC=$?
fi
echo "EXIT_CODE=$RC"
exit "$RC"
