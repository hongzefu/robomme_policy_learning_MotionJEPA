#!/usr/bin/env bash
# ── 实验 2：本机冷缓存复测（run_name v1-coldcache-b8）────────────────────────────
# 目的：v1-2gpu-epoch-bench-b8 的 1.060 s/step 因第 2/3 轮同 seed 而受 page cache
# 污染（第 3 轮读的样本正是第 3 轮前两小时刚读过的）。本实验换 seed=123（冷样本）
# 重跑短基准，并同步采 GPU 利用率与 turbo NFS 真实读流量（mountstats server_read，
# page cache 命中不计入）——若步时明显变长且 NFS 读持续打满，说明本机口径下 IO
# 已是瓶颈。复用 scripts/smoke-local/bench_train_steps.py（零改动）。
# ⚠ 本机数字按 AGENTS.md 第 13 条只作旁证，权威判据是 GL 侧实验 1/3。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../data-preprocess-GL" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_venv
v1_require_models 1

RUN_NAME="v1-coldcache-b8"
STEPS="${STEPS:-150}"
SEED="${SEED:-123}"                       # 换 seed=冷样本；42 是热缓存基线的 seed
BATCH=8
WORKERS=4
WARMUP_STEPS="${WARMUP_STEPS:-50}"
RECORD_DIR="${V1_STORE}/bench/bottleneck/${RUN_NAME}"
CKPT_DIR="${TRAIN_RUNS}/mme_vla_suite/${RUN_NAME}"
LOG="${LOGS_DIR}/${RUN_NAME}.log"
MOUNT_POINT="/nfs/turbo/coe-chaijy-unreplicated"

[[ -e "${RECORD_DIR}" ]] && { echo "错误: 记录目录已存在: ${RECORD_DIR}" >&2; exit 1; }
[[ -e "${CKPT_DIR}" ]] && { echo "错误: run 目录已存在: ${CKPT_DIR}" >&2; exit 1; }
mkdir -p "${RECORD_DIR}"

# 环境留档
"${PY}" - "${RECORD_DIR}" <<EOF
import json, platform, subprocess, sys
import jax
d = {
    "run_name": "${RUN_NAME}", "batch_size": ${BATCH}, "workers": ${WORKERS},
    "steps": ${STEPS}, "seed": ${SEED}, "fsdp_devices": 2,
    "history_config": "perceptual-framesamp-context.yaml",
    "dataset_path": "${GL_DATASET}",
    "git_head": subprocess.run(["git", "-C", "${REPO_ROOT}", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95", "CUDA_VISIBLE_DEVICES": "0,1",
    "hostname": platform.node(), "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
json.dump(d, open("${RECORD_DIR}/env.json", "w"), indent=2, ensure_ascii=False)
EOF

# 遥测采样器（每 5 秒）：GPU 利用率 + turbo 挂载真实读字节（server_read）
GPULOG="${RECORD_DIR}/gpu_util.csv"
NFSLOG="${RECORD_DIR}/nfs_read.csv"
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -v t="$(date +%s.%N)" '{print t","$0}'
    sleep 5
  done ) >> "${GPULOG}" 2>/dev/null &
GPU_SAMPLER=$!
( while true; do
    python3 - "$MOUNT_POINT" <<'PYEOF'
import sys, time
mp = sys.argv[1]; in_m = False
for line in open("/proc/self/mountstats"):
    if line.startswith("device ") and f" mounted on {mp} " in line: in_m = True
    elif line.startswith("device "): in_m = False
    elif in_m and line.strip().startswith("bytes:"):
        f = line.split()[1:]
        print(f"{time.time()},{f[0]},{f[4]}")  # normal_read, server_read
        break
PYEOF
    sleep 5
  done ) >> "${NFSLOG}" 2>/dev/null &
NFS_SAMPLER=$!
trap 'kill ${GPU_SAMPLER} ${NFS_SAMPLER} 2>/dev/null || true' EXIT

echo "=== 冷缓存复测: ${RUN_NAME} (${STEPS} steps, batch ${BATCH}, seed ${SEED}) ==="
set +e
(
  set -e
  cd "${REPO_ROOT}"
  BENCH_RECORD_DIR="${RECORD_DIR}" \
  CUDA_VISIBLE_DEVICES=0,1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  "${PY}" "${REPO_ROOT}/scripts/smoke-local/bench_train_steps.py" mme_vla_suite \
    --exp-name "${RUN_NAME}" \
    --assets-base-dir "${TRAIN_ASSETS}" \
    --checkpoint-base-dir "${TRAIN_RUNS}" \
    --batch-size "${BATCH}" \
    --num-workers "${WORKERS}" \
    --num-train-steps "${STEPS}" \
    --log-interval 1 \
    --save-interval 1000 \
    --seed "${SEED}" \
    --fsdp-devices 2 \
    --dataset-path "${GL_DATASET}" \
    --weight-loader.params-path "${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
) 2>&1 | tee "${LOG}"
RC="${PIPESTATUS[0]}"
set -e
kill "${GPU_SAMPLER}" "${NFS_SAMPLER}" 2>/dev/null || true

if [[ -e "${CKPT_DIR}" ]]; then
  case "${CKPT_DIR}" in
    "${TRAIN_RUNS}/mme_vla_suite/${RUN_NAME}") rm -rf -- "${CKPT_DIR}" ;;
  esac
fi
rm -rf -- "${HOME}/.cache/jax_${RUN_NAME}"

if [[ "${RC}" -ne 0 ]]; then
  echo "错误: 冷缓存复测失败（退出码 ${RC}）: ${LOG}" >&2
  echo "EXIT_CODE=${RC}"; exit "${RC}"
fi

"${PY}" - "${RECORD_DIR}" "${STEPS}" "${WARMUP_STEPS}" <<'EOF'
import json, statistics, sys
d, steps, warmup = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if '"loss"' in l]
assert len(rows) == steps, f"metrics 行数 {len(rows)} != {steps}"
t = {r["step"]: r["wall_time"] for r in rows}
# save-interval=1000 → 中途无校验和；末步(steps-1)一次校验和只影响末步之后，不在 delta 集里
deltas = [t[s] - t[s-1] for s in range(warmup + 1, steps)]
steady = statistics.median(deltas)
t_lo, t_hi = t[warmup], t[steps - 1]      # 稳态窗口
# NFS 真实读速率（server_read，page cache 命中不计）
nfs = [l.strip().split(",") for l in open(f"{d}/nfs_read.csv") if l.strip()]
win = [(float(a), int(c)) for a, b, c in nfs if t_lo <= float(a) <= t_hi]
mbps = (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0]) / 1e6 if len(win) >= 2 else float("nan")
# GPU 利用率（稳态窗口内两卡合并中位）
gpu = [l.strip().split(",") for l in open(f"{d}/gpu_util.csv") if l.strip()]
utils = [int(u) for ts, idx, u, m in gpu if t_lo <= float(ts) <= t_hi]
gpu_med = statistics.median(utils) if utils else float("nan")
per_sample_mb = 18.7
demand = 8 * per_sample_mb / steady        # batch 8 的 NFS 需求
print(f"RESULT 冷缓存稳态={steady:.3f}s/step (n={len(deltas)}, "
      f"p10={sorted(deltas)[len(deltas)//10]:.3f}, p90={sorted(deltas)[len(deltas)*9//10]:.3f}) "
      f"vs 热缓存基线 1.060s/step ({steady/1.060:.2f}x)")
print(f"RESULT 稳态窗口 NFS真实读={mbps:.0f} MB/s（需求口径 {demand:.0f} MB/s），"
      f"GPU利用率中位={gpu_med:.0f}%")
if steady > 1.060 * 1.3:
    print("RESULT 判定: 冷缓存显著变慢 → 本机口径下 dataloader/NFS 已是瓶颈")
else:
    print("RESULT 判定: 冷缓存未显著变慢 → 本机口径下计算侧主导（或预取足够掩盖 IO）")
EOF

echo "记录文件保留在: ${RECORD_DIR}"
echo "EXIT_CODE=0"
echo "COLDCACHE_PASS 冷缓存复测完成"
