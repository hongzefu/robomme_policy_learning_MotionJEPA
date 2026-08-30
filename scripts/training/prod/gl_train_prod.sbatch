#!/usr/bin/env bash
# ── 正式训练（GL 4×A40）：10 小时利用率趋势观测 run ────────────────────────────
# 计划：~/.claude/plans/worker8-packed-run-name-smooth-flute.md（用户 2026-08-28 批准）
# v5.0（v5.0-train-entry-restructure-plan.md）：prod_train_once.py 删除，本脚本直调
# scripts/training/train.py（已改单跑、自带 TRAIN_RECORD_DIR 记录器）；同批做退出码
# 硬化——此前 write_env/analyzer/downsample 任一失败都不改变最终 EXIT_CODE，新记录器
# schema 出错会以「analyzer 崩溃」形式出现却仍报 EXIT_CODE=0，故改为四段 RC 各自留档、
# 最终取首个非零（训练码优先），并用 trap 保证 stop_samplers 与 EXIT_CODE= 行在任何
# 退出路径（含被 Slurm SIGTERM）都执行。
#
# 与历史 gl_e2e_fix.sbatch（bottleneck-bench-v2，已随目录退役，见 git 历史）的关系：
# 骨架、五路采样器、write_env、run_py 全部沿用；资源包络放大到 128G/12h、收尾多跑
# 降采样与趋势分析。
#
# ── 历史坑登记（自 prod_train_once.py 头注释迁入，详见 scripts/training/g0/README.md 第四节）──
# 1. train.py 双跑必崩：历史 __main__ 连跑两次 main()，第二次必 FileExistsError——
#    v5.0 已改单跑，此坑消失（历史记录）。
# 2. bench 不落权重：g0/bench_train_steps.py 把 save_state 换成摘要器，全程不落
#    checkpoint，拿它跑正式训练会白跑几小时。
# 3. 不能直接 patch wandb.log：wandb.init() 会重赋值模块级 log 盖掉 patch；train.py
#    内置 _MetricsProxy 用「替换模块全局名 wandb」的正确做法。
# 4. 记录目录只能走环境变量 TRAIN_RECORD_DIR（tyro 会吃掉全部 argv）；防覆盖判据是
#    metrics.jsonl 是否已存在，不是目录非空（本脚本先建目录写 env.json、起采样器）。
# ⚠ 续跑语义有损：checkpoint 只存 EMA，丢 AdamW 动量与 warmup 计数——train.py 已
#    fail-loud 禁用 --overwrite / --resume。
#
# ⚠ 本 job 远超 greatlakes.md 调试包络（≤2 GPU / ≤30 min），提交前须用户逐个放行。
#   放行记录见 docs/training-doc/<RUN_NAME>/launch.md。
#
# 必传（经 --export）：RUN_NAME
# 可覆盖：STEPS(7000) SEED(335) WORKERS(8) BATCH(64) FSDP(4) LOG_INTERVAL(100)
#         SAVE_INTERVAL(2000) EPOCH_STEPS(6176) DATASET_PATH TREND_STEP(1000)
#         DOWNSAMPLE_EVERY(10)
#
#SBATCH --job-name=v1-prod-trend
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail

# ── 退出码硬化（v5.0）：trap 保证采样器清理与 EXIT_CODE= 行在任何退出路径都执行 ──
SAMPLER_PIDS=()
stop_samplers() {
    [ "${#SAMPLER_PIDS[@]}" -eq 0 ] && return 0
    kill "${SAMPLER_PIDS[@]}" 2>/dev/null || true
    SAMPLER_PIDS=()
}
_finalize() {
    rc=$?
    stop_samplers
    echo "EXIT_CODE=$rc"
}
trap _finalize EXIT
trap 'exit 143' TERM INT

REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
STORE="$REPO/v1-store"
NCPU="${SLURM_CPUS_PER_TASK:-16}"
MOUNT_POINT="/nfs/turbo/coe-chaijy-unreplicated"

# AGENTS 3：python 一律走 uv run；--no-sync = 只用现有 NFS venv、不同步、不碰网络
export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "错误: 计算节点找不到 uv"; exit 1; }
run_py() { uv run --no-sync --project "$REPO" python "$@"; }

STEPS="${STEPS:-7000}"
SEED="${SEED:-335}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-64}"
FSDP="${FSDP:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
EPOCH_STEPS="${EPOCH_STEPS:-6176}"
TREND_STEP="${TREND_STEP:-1000}"
DOWNSAMPLE_EVERY="${DOWNSAMPLE_EVERY:-10}"
DATASET_PATH="${DATASET_PATH:-$STORE/datasets/4task-gl-framesamp}"
# write_env 的 python 子进程按 os.environ 读取，默认值路径也必须 export
export STEPS SEED WORKERS BATCH FSDP LOG_INTERVAL SAVE_INTERVAL EPOCH_STEPS DATASET_PATH
export RUN_NAME

if [ -z "${RUN_NAME:-}" ]; then
    echo "缺必传参数 RUN_NAME（应由提交脚本经 --export 传入），拒绝起跑"; exit 1
fi

export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled          # 用户拍板关 wandb；与所有已有基准同口径、无出网依赖

RECORD_DIR="$STORE/train-records/$RUN_NAME"
CKPT_DIR="$STORE/train-runs/mme_vla_suite/$RUN_NAME"
# run 目录 fail-loud（AGENTS 6：禁止复用 run_name / 覆盖已有 run）
if [ -e "$RECORD_DIR" ]; then echo "错误: 记录目录已存在: $RECORD_DIR"; exit 1; fi
if [ -e "$CKPT_DIR" ]; then echo "错误: run 目录已存在: $CKPT_DIR"; exit 1; fi
mkdir -p "$RECORD_DIR"

# jax 编译缓存：train.py 把它硬编码到 ~/.cache/jax_<exp_name>（AGENTS 14 已记为已知豁免）。
# 建软链指向 v1-store，正式 run 的缓存留存不删；禁止用覆盖 HOME 的方式绕过（会断 ssh）。
JAX_CACHE_REAL="$STORE/cache/jax/$RUN_NAME"
JAX_CACHE_LINK="$HOME/.cache/jax_$RUN_NAME"
mkdir -p "$JAX_CACHE_REAL" "$HOME/.cache"
[ -e "$JAX_CACHE_LINK" ] || ln -s "$JAX_CACHE_REAL" "$JAX_CACHE_LINK"

write_env() {   # $1=RECORD_DIR
    run_py - "$1" <<'EOF'
import json, os, platform, subprocess, sys
import jax
record_dir = sys.argv[1]
repo = os.environ["REPO_ENV"]
d = {
    "run_name": os.environ["RUN_NAME"],
    # ── analyzer 判读依赖的四键（缺任何一个都会告警或算错）──
    "log_interval": int(os.environ["LOG_INTERVAL"]),
    "epoch_steps": int(os.environ["EPOCH_STEPS"]),
    "batch_size": int(os.environ["BATCH"]),
    "num_workers": int(os.environ["WORKERS"]),
    # ── 留档 provenance（AGENTS 12）──
    "fsdp_devices": int(os.environ["FSDP"]),
    "steps": int(os.environ["STEPS"]),
    "seed": int(os.environ["SEED"]),
    "save_interval": int(os.environ["SAVE_INTERVAL"]),
    "cpus": int(os.environ["NCPU_ENV"]),
    "mem": os.environ.get("SLURM_MEM_PER_NODE"),
    "entry": "scripts/training/train.py",
    "tentative_run": False,
    "history_config": "perceptual-framesamp-context.yaml",
    "dataset_path": os.environ["DATASET_PATH"],
    "git_head": subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "WANDB_MODE": os.environ.get("WANDB_MODE"),
    "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
}
import hashlib, pathlib
sm = pathlib.Path(os.environ["DATASET_PATH"]) / "meta" / "store_meta.json"
if sm.exists():
    raw = sm.read_bytes()
    meta = json.loads(raw)
    d["store_meta_sha256"] = hashlib.sha256(raw).hexdigest()
    d["store_meta_status"] = meta.get("status")
    d["manifest_sha256"] = meta.get("manifest_sha256")
json.dump(d, open(os.path.join(record_dir, "env.json"), "w"), indent=2, ensure_ascii=False)
EOF
}

start_samplers() {   # $1=RECORD_DIR —— 五路，与 gl_e2e_fix.sbatch 逐字同口径
    local rd="$1"
    # legacy 通道（15s，epoch秒,卡号,util%,显存MiB），交叉核对用
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
    # NFS server_read 计数。⚠ 与 bench 的差别：这里用纯 awk 解析 /proc/self/mountstats，
    # 不再每 15 秒起一个 uv run python 子进程——10 小时会是 2400 次子进程启动，纯属浪费。
    # 输出列与 bench 完全一致（时间戳,normal_read累计,server_read累计）。
    ( while true; do
        awk -v mp="$MOUNT_POINT" -v t="$(date +%s)" '
          /^device / { inm = (index($0, " mounted on " mp " ") > 0); next }
          inm && /^[[:space:]]*bytes:/ { print t","$2","$6; exit }
        ' /proc/self/mountstats
        sleep 15
      done ) >> "$rd/nfs_read.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
    # /proc/meminfo Cached + cgroup pgmajfault（内存档位复核证据链）
    ( while true; do
        CACHED=$(awk '/^Cached:/{print $2}' /proc/meminfo)
        CG="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)"
        MAJFLT=$(awk '$1=="pgmajfault"{print $2}' "$CG/memory.stat" 2>/dev/null || echo "")
        ANON=$(awk '$1=="anon"{print $2}' "$CG/memory.stat" 2>/dev/null || echo "")
        echo "$(date +%s),$CACHED,$MAJFLT,$ANON"
        sleep 15
      done ) >> "$rd/meminfo.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
    # worker CUDA context 存证
    ( while true; do
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
          | awk -v t="$(date +%s)" '{print t","$0}'
        sleep 15
      done ) >> "$rd/compute_apps.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)
}
REPO_ENV="$REPO" NCPU_ENV="$NCPU" write_env "$RECORD_DIR"
ENV_RC=$?
echo "ENV_RC=$ENV_RC"
start_samplers "$RECORD_DIR"

echo "=== 正式训练 (run=$RUN_NAME, dataset=$DATASET_PATH, workers=$WORKERS, cpus=$NCPU, mem=${SLURM_MEM_PER_NODE:-?}, node=$(hostname), job=${SLURM_JOB_ID:-?}, ${STEPS} steps, log_interval=$LOG_INTERVAL, save_interval=$SAVE_INTERVAL) ==="

TRAIN_RECORD_DIR="$RECORD_DIR" \
run_py "$REPO/scripts/training/train.py" mme_vla_suite \
    --exp-name "$RUN_NAME" \
    --assets-base-dir "$STORE/train-assets" \
    --checkpoint-base-dir "$STORE/train-runs" \
    --batch-size "$BATCH" \
    --num-workers "$WORKERS" \
    --num-train-steps "$STEPS" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --seed "$SEED" \
    --fsdp-devices "$FSDP" \
    --dataset-path "$DATASET_PATH" \
    --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
TRAIN_RC=$?
stop_samplers
echo "TRAIN_RC=$TRAIN_RC"

ANALYZE_RC=0
DOWNSAMPLE_RC=0
if [ "$TRAIN_RC" -eq 0 ]; then
    # 判读永远跑全分辨率原件；--trend 出分时走势与 TREND_OK，长训不带 --accept
    # （E2E95_ACCEPT 的五项阈值是按 600 步、全程不落 checkpoint 标定的）
    run_py "$REPO/scripts/training/util/analyze_gpu_util.py" \
        "$RECORD_DIR" --steps "$STEPS" --trend --trend-step "$TREND_STEP"
    ANALYZE_RC=$?
    echo "ANALYZE_RC=$ANALYZE_RC"
    # 降采样只服务归档（AGENTS 12 留档 vs AGENTS 14 大产物不进 git 的化解方式）
    run_py "$REPO/scripts/training/util/downsample_util_csv.py" \
        "$RECORD_DIR/gpu_util_dense.csv" \
        "$RECORD_DIR/gpu_util_dense.ds${DOWNSAMPLE_EVERY}.csv" \
        --every "$DOWNSAMPLE_EVERY"
    DOWNSAMPLE_RC=$?
    echo "DOWNSAMPLE_RC=$DOWNSAMPLE_RC"
    echo "CKPT_LIST:"; ls -1 "$CKPT_DIR" 2>/dev/null
    echo "CKPT_SIZE: $(du -sh "$CKPT_DIR" 2>/dev/null | cut -f1)"
fi

# 最终 RC = 四段首个非零（训练码优先）；EXIT_CODE= 行由 trap 统一输出
RC=0
for r in "$TRAIN_RC" "$ENV_RC" "$ANALYZE_RC" "$DOWNSAMPLE_RC"; do
    if [ "$r" -ne 0 ]; then RC=$r; break; fi
done
exit "$RC"
