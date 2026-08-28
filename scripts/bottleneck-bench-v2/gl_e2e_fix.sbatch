#!/usr/bin/env bash
# ── bottleneck-bench v2：e2e 基准（600 步；S7.5 起参数化，默认值=现状）────────────
# 历史目的：验证「16 CPU + 8-16 workers + 96G」能否吃满 v1-e2e-b64 暴露的 GPU 空转
# （v1 基线：8C/4w/64G，util 均值 69-70%，慢步(>8s)占稳态墙钟 33%，6.933 s/step）。
# S7.5（v2 计划）参数化（--export 传入，默认值全部等于历史现状）：
#   DATASET_PATH           默认 $STORE/datasets/4task-gl（packed 库传 …/4task-gl-framesamp）
#   MMEVLA_DATA_BACKEND    默认 legacy（R16 显式三态；packed 由提交侧显式传入）
#   SAVE_INTERVAL          默认 1000（600 步内不触发 TrainState 摘要，性能口径不受污染）
#   STEPS                  默认 600
#   ANALYZE_ACCEPT         默认 0；=1 时 analyzer 跑 E2E95_ACCEPT 五项（v1-95util L0 起）
#   COLDHOT                默认 0；=1 时同 allocation 先 C1(${TAG}-coldlike) 后 H1(${TAG}-hot)
#                          各 COLDHOT_STEPS(默认300) 步、同 seed 冻结序列——需提交侧
#                          sbatch --time=04:00:00 覆盖（--time 无法用环境变量参数化）
# L0 量具参数（v1-95util，2026-08-28）：
#   LOG_INTERVAL           默认 1（现状）；≠1 时自动置 BENCH_PERF_MODE=1（生产口径，
#                          metrics 记录区间均值；bench 侧强制禁摘要）
#   GAUGE_DUAL             默认 0；=1 时同 allocation 先 ${TAG}-log1(seed=BENCH_SEED)
#                          后 ${TAG}-log100(seed=BENCH_SEED2，必传)——两档差异归因量具
#                          口径；需提交侧 sbatch --time=04:00:00 覆盖
#   EPOCH_STEPS            默认 6176（epoch 换算步数，写入 env.json 供 analyzer 取用）
#   PREFETCH_FACTOR        默认 2 = torch 现状默认值（L0 只记录实效值入 env.json，
#                          不改变 loader 行为；L1 起经 MMEVLA_PREFETCH_FACTOR 生效）
# WORKERS / TAG / BENCH_SEED 必须经 --export 传入（TAG 即 run_name；各档 seed 互不相同
# 且避开已用 42/200-205/210-212，防同节点复跑 page cache 跨 job 污染）。
# ⚠ 4 GPU + 2-4h 超出 greatlakes.md debug 包络，每个 job 提交前逐个经用户资源审批。
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
STORE="$REPO/v1-store"
# AGENTS 3：python 调用一律走 uv run（L0 修复存量 .venv/bin/python 直调，用户批准）；
# --no-sync = 只用现有 NFS venv、不做环境同步、不碰网络（用户 2026-08-28 拍板）
export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "错误: 计算节点找不到 uv"; exit 1; }
run_py() { uv run --no-sync --project "$REPO" python "$@"; }
NCPU="${SLURM_CPUS_PER_TASK:-16}"
MOUNT_POINT="/nfs/turbo/coe-chaijy-unreplicated"

STEPS="${STEPS:-600}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
DATASET_PATH="${DATASET_PATH:-$STORE/datasets/4task-gl}"
export MMEVLA_DATA_BACKEND="${MMEVLA_DATA_BACKEND:-legacy}"
COLDHOT="${COLDHOT:-0}"
COLDHOT_STEPS="${COLDHOT_STEPS:-300}"
ANALYZE_ACCEPT="${ANALYZE_ACCEPT:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
GAUGE_DUAL="${GAUGE_DUAL:-0}"
EPOCH_STEPS="${EPOCH_STEPS:-6176}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# write_env 的 python 子进程按 os.environ 读取——默认值路径也必须 export
# （2026-08-27 T1 job 58996024 实测 KeyError: 'SAVE_INTERVAL' 教训；WORKERS/TAG/
# BENCH_SEED 经 sbatch --export 已在环境，重复 export 无害）
export STEPS SAVE_INTERVAL DATASET_PATH COLDHOT COLDHOT_STEPS ANALYZE_ACCEPT
export LOG_INTERVAL EPOCH_STEPS PREFETCH_FACTOR
export WORKERS TAG BENCH_SEED

if [ "$GAUGE_DUAL" = "1" ] && [ -z "${BENCH_SEED2:-}" ]; then
    echo "GAUGE_DUAL=1 缺必传参数 BENCH_SEED2（log100 档 seed，各档 seed 互不相同），拒绝起跑"
    exit 1
fi

for VAR in WORKERS TAG BENCH_SEED; do
    if [ -z "${!VAR:-}" ]; then
        echo "缺必传参数 $VAR（应由提交脚本经 --export 传入），拒绝起跑"
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

write_env() {   # $1=RECORD_DIR $2=RUN_NAME $3=STEPS
    run_py - "$1" "$2" "$3" <<'EOF'
import json, os, platform, subprocess, sys
import jax
record_dir, run_name, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
repo = os.environ["REPO_ENV"]
store = os.environ["STORE_ENV"]
dataset_path = os.environ["DATASET_PATH"]
d = {
    "run_name": run_name, "batch_size": 64, "fsdp_devices": 4,
    "num_workers": int(os.environ["WORKERS"]), "cpus": int(os.environ["NCPU_ENV"]),
    "mem": os.environ.get("SLURM_MEM_PER_NODE"),
    "steps": steps, "seed": int(os.environ["BENCH_SEED"]),
    "save_interval": int(os.environ["SAVE_INTERVAL"]),
    "history_config": "perceptual-framesamp-context.yaml",
    "dataset_path": dataset_path,
    "coldhot": os.environ["COLDHOT"],
    "git_head": subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    # ── L0 量具口径（v1-95util）：analyzer 判读全部从这里取，不靠猜 ──
    "log_interval": int(os.environ["LOG_INTERVAL"]),
    "epoch_steps": int(os.environ["EPOCH_STEPS"]),
    "prefetch_factor": int(os.environ["PREFETCH_FACTOR"]),   # 实效值（L0=torch 默认 2）
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
    # ── S7.5 framesamp provenance（D 节清单；legacy 库下 store 字段为 None）──
    "MMEVLA_DATA_BACKEND": os.environ.get("MMEVLA_DATA_BACKEND", ""),
    "backend_source": "explicit",   # 本脚本恒显式设置（R16）
    "MMEVLA_FRAMESAMP_VERIFY": os.environ.get("MMEVLA_FRAMESAMP_VERIFY", ""),
    "MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED": os.environ.get("MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED", ""),
    "MMEVLA_FRAMESAMP_ALLOW_SUBSET": os.environ.get("MMEVLA_FRAMESAMP_ALLOW_SUBSET", ""),
    "MMEVLA_FRAMESAMP_SOURCE": os.environ.get("MMEVLA_FRAMESAMP_SOURCE", ""),
    "MMEVLA_FRAMESAMP_MANIFEST": os.environ.get("MMEVLA_FRAMESAMP_MANIFEST", ""),
    "MMEVLA_FRAMESAMP_LOCAL_CACHE": os.environ.get("MMEVLA_FRAMESAMP_LOCAL_CACHE", ""),
    "full_verify_in_allocation": False,   # 冷态自证：本 allocation 不跑 full 校验、不预热
    "local_copy_prewarm": False,
}
import hashlib, pathlib
sm = pathlib.Path(dataset_path) / "meta" / "store_meta.json"
if sm.exists():
    raw = sm.read_bytes()
    meta = json.loads(raw)
    d["store_meta_sha256"] = hashlib.sha256(raw).hexdigest()
    d["store_meta_status"] = meta.get("status")
    d["manifest_sha256"] = meta.get("manifest_sha256")
    d["source_dataset_root_resolved"] = (os.environ.get("MMEVLA_FRAMESAMP_SOURCE")
                                         or meta.get("source_dataset_root"))
    d["manifest_path_resolved"] = (os.environ.get("MMEVLA_FRAMESAMP_MANIFEST")
                                   or meta.get("manifest_path"))
else:
    d["store_meta_sha256"] = None
    d["store_meta_status"] = None
    d["manifest_sha256"] = None
    d["source_dataset_root_resolved"] = dataset_path
    d["manifest_path_resolved"] = None
json.dump(d, open(os.path.join(record_dir, "env.json"), "w"), indent=2, ensure_ascii=False)
EOF
}

SAMPLER_PIDS=()
start_samplers() {   # $1=RECORD_DIR
    local rd="$1"
    # legacy 通道：与 v1 同口径（15s，epoch秒,卡号,util%,显存MiB），对照用不动
    ( while true; do
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
          | awk -v t="$(date +%s)" '{print t","$0}'
        sleep 15
      done ) >> "$rd/gpu_util.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
    # dense 通道（主判读）：nvidia-smi -lms 500 流式（500ms 即 NVML 有效密度上限）
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -lms 500 >> "$rd/gpu_util_dense.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
    # NFS server_read 计数
    ( while true; do
        run_py - "$MOUNT_POINT" <<'PYEOF'
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
      done ) >> "$rd/nfs_read.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
    # S7.5：/proc/meminfo Cached + cgroup pgmajfault（冷/热证据链，D 节）
    ( while true; do
        CACHED=$(awk '/^Cached:/{print $2}' /proc/meminfo)
        CG="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)"
        MAJFLT=$(awk '$1=="pgmajfault"{print $2}' "$CG/memory.stat" 2>/dev/null || echo "")
        echo "$(date +%s),$CACHED,$MAJFLT"
        sleep 15
      done ) >> "$rd/meminfo.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
    # S7.5：worker CUDA context 存证（D 节）
    ( while true; do
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
          | awk -v t="$(date +%s)" '{print t","$0}'
        sleep 15
      done ) >> "$rd/compute_apps.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
}
stop_samplers() {
    kill "${SAMPLER_PIDS[@]}" 2>/dev/null || true
    SAMPLER_PIDS=()
}

run_bench() {   # $1=RUN_NAME $2=STEPS [$3=LOG_INTERVAL 覆盖，缺省用全局值] → 返回 bench 退出码
                # 副作用：写全局 BENCH_TRAIN_RC = 训练本身退出码（不含 analyzer accept 判定）
    local run_name="$1" steps="$2"
    export LOG_INTERVAL="${3:-$LOG_INTERVAL}"   # 逐 run 覆盖并入 env.json（write_env 读取）
    local perf_mode=0
    [ "$LOG_INTERVAL" != "1" ] && perf_mode=1   # ≠1 自动置性能模式（bench 侧强制禁摘要）
    BENCH_TRAIN_RC=1   # 提前 return 的分支也给确定值，不残留上一轮的值
    local record_dir="$STORE/bench/bottleneck/$run_name"
    local ckpt_dir="$STORE/train-runs/mme_vla_suite/$run_name"
    if [ -e "$record_dir" ]; then echo "错误: 记录目录已存在: $record_dir"; return 1; fi
    if [ -e "$ckpt_dir" ]; then echo "错误: run 目录已存在: $ckpt_dir"; return 1; fi
    mkdir -p "$record_dir"
    REPO_ENV="$REPO" STORE_ENV="$STORE" NCPU_ENV="$NCPU" write_env "$record_dir" "$run_name" "$steps"
    start_samplers "$record_dir"
    echo "=== e2e 基准 (run=$run_name, backend=$MMEVLA_DATA_BACKEND, dataset=$DATASET_PATH, workers=$WORKERS, cpus=$NCPU, node=$(hostname), job=${SLURM_JOB_ID:-?}, ${steps} steps) ==="
    # 性能 run 硬口径（B5）：禁全部摘要。BENCH_CHECKSUM=0 连步 0 的 TrainState 摘要
    # 一并关——步 0 摘要会把完整 TrainState device_get 拉回 host（实测 45.4 GiB），
    # 96G 节点直接 OOM（2026-08-27 T1 job 58996083 EXIT_CODE=137 实测教训；历史
    # e2efix 96G 能过是因当时 bench 尚无步 0 必记）。
    BENCH_RECORD_DIR="$record_dir" \
    BENCH_CHECKSUM=0 \
    BENCH_BATCH_DIGESTS=0 \
    BENCH_PERF_MODE="$perf_mode" \
    run_py "$REPO/scripts/smoke-local/bench_train_steps.py" mme_vla_suite \
        --exp-name "$run_name" \
        --assets-base-dir "$STORE/train-assets" \
        --checkpoint-base-dir "$STORE/train-runs" \
        --batch-size 64 \
        --num-workers "$WORKERS" \
        --num-train-steps "$steps" \
        --log-interval "$LOG_INTERVAL" \
        --save-interval "$SAVE_INTERVAL" \
        --seed "$BENCH_SEED" \
        --fsdp-devices 4 \
        --dataset-path "$DATASET_PATH" \
        --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
        --model.use-history \
        --model.history-config perceptual-framesamp-context.yaml \
        --no-wandb-enabled
    local rc=$?
    BENCH_TRAIN_RC="$rc"   # analyzer accept 判定会覆盖下面的 rc，先存住训练自身的退出码
    stop_samplers
    if [ -e "$ckpt_dir" ]; then
      case "$ckpt_dir" in
        "$STORE/train-runs/mme_vla_suite/$run_name") rm -rf -- "$ckpt_dir" ;;
      esac
    fi
    rm -rf -- "$HOME/.cache/jax_${run_name}"
    if [ "$rc" -eq 0 ]; then
      ACCEPT_FLAG=()
      [ "$ANALYZE_ACCEPT" = "1" ] && ACCEPT_FLAG=(--accept)
      run_py "$REPO/scripts/bottleneck-bench-v2/analyze_gpu_util.py" \
          "$record_dir" --steps "$steps" "${ACCEPT_FLAG[@]}"
      rc=$?
    fi
    return "$rc"
}

if [ "$GAUGE_DUAL" = "1" ]; then
    # L0 两档基准（v1-95util）：同 allocation 同节点先 S0c(log1 对照) 后 S0(log100 生产)，
    # 两档差异归因量具口径。S0c 判 E2E95 预期 FAIL（旧口径缺口所在），故对照档不带
    # --accept、只出报表；E2E95_ACCEPT 判定只落在 log100 档（S0 = 新 speed 锚）。
    # 各档 seed 互不相同（BENCH_SEED / BENCH_SEED2），防同节点复跑 page cache 污染归因。
    SAVED_ACCEPT="$ANALYZE_ACCEPT"
    ANALYZE_ACCEPT=0 run_bench "${TAG}-log1" "$STEPS" 1; RC1=$?
    G1_TRAIN_RC="$BENCH_TRAIN_RC"
    echo "GAUGE_DUAL: log1 对照档 rc=$RC1 (train_rc=$G1_TRAIN_RC)"
    if [ "$G1_TRAIN_RC" -eq 0 ]; then
        export BENCH_SEED="$BENCH_SEED2"
        ANALYZE_ACCEPT="$SAVED_ACCEPT" run_bench "${TAG}-log100" "$STEPS" 100; RC=$?
    else
        echo "log1 对照档训练本身失败（train_rc=$G1_TRAIN_RC），跳过 log100 档"
        RC=1
    fi
elif [ "$COLDHOT" = "1" ]; then
    # 同 allocation 串行先 C1 后 H1（排除节点差异）；同 seed 冻结 index 序列（D 节）
    run_bench "${TAG}-coldlike" "$COLDHOT_STEPS"; RC=$?
    C1_TRAIN_RC="$BENCH_TRAIN_RC"
    # 只看训练是否跑完，不看 RC：ANALYZE_ACCEPT=1 时 RC 已被 analyzer 的
    # E2E_ACCEPT 判定覆盖，而 C1 是冷态、判 FAIL 属预期内；COLDHOT 的判据是
    # (C1稳态−H1稳态)/H1 ≤ 15%，与 E2E_ACCEPT 五项无关。
    # 2026-08-28 job 59001191 实测教训：C1 E2E_ACCEPT=FAIL → rc=1 → H1 被跳过，
    # 4h allocation 只产出一半数据，对照组缺失、判据算不出来。
    if [ "$C1_TRAIN_RC" -eq 0 ]; then
        run_bench "${TAG}-hot" "$COLDHOT_STEPS"; RC=$?
    else
        echo "C1 训练本身失败（train_rc=$C1_TRAIN_RC），跳过 H1"
    fi
else
    run_bench "$TAG" "$STEPS"; RC=$?
fi
echo "EXIT_CODE=$RC"
exit "$RC"
