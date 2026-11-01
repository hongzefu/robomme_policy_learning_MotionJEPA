#!/usr/bin/env bash
# 1031 GL 复现专用占位 job 运行器。按正本 templates/run_in_hold.sh 实例化，
# 对应 greatlakes.md「调试 slurm 脚本」「算力使用规则」与 AGENTS.md 第 0、7、8、23 条。
#
# 用法：占位 job（同目录 hold_o16.sbatch）RUNNING 后，把训练经 srun 塞进去（预计超过 5 分钟的放进 detached tmux）：
#   srun --jobid=<占位 JobID> --overlap --exact --ntasks=1 --cpus-per-task=$((4*N)) --gpu_cmode=shared \
#     env GPUS=4 GLOBAL_BATCH=64 HISTORY=<yaml 文件名> RUN_NAME=<run 名> \
#         MMEVLA_STAGE2_EVIDENCE_ROOT=<turbo>/v1-store/validation/orig16-stage2 \
#         [TRAIN_HEAD=<40 位 sha>] [FINAL_RECORD_DIR=<绝对路径>/final-record] [MAX_STEPS=100] [TRAIN_TIMING_STEPS=100] \
#     bash <turbo>/scripts/training/gl/run_in_hold_o16.sh
# 本运行器不接受位置参数；只把上面 9 个 env 接口原样透传给 S1 入口 scripts/training/run_orig16_stage2_60k_gl.sh，
# 取值合法性由 S1 入口自己校验，这里不重复、不改写、不补默认值。
#
# 结构：计算节点判定 fail-fast（任一不符 exit 2，判定前不写任何文件）→ unset 诊断遗留开关 → cd 仓库
#       → 日志三件套 + trap → 前台运行 S1 入口 → cleanup 写 EXIT_CODE= 尾行。
# 与模板一致，为了 tee 日志不 exec 交棒，改为显式 trap TERM/INT 转成退出码（只挂 EXIT trap 时被 TERM 杀掉会误写 EXIT_CODE=0）。
# 集群侧只用仓库 .venv/bin/python 直调（由 S1 入口负责），不在计算节点安装依赖。
set -euo pipefail

REPO="/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA"
SHARED_ROOT="/nfs/turbo/coe-chaijy-unreplicated/hongzefu"
LOCAL_ROOT="/data/hongzefu"
EXPECTED_GPU="NVIDIA A40"
LOG_DIR="$REPO/v1-store/logs"
ENTRY="$REPO/scripts/training/run_orig16_stage2_60k_gl.sh"

[[ $# == 0 ]] || { echo "本运行器不接受位置参数，参数一律用 env 传入（见文件头用法）" >&2; exit 2; }

# ---- 计算节点判定（BEGIN）：本机或登录节点误跑一律 exit 2 ----
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "不在 Slurm 作业内：须经 srun --jobid=<占位 JobID> --overlap 塞入" >&2; exit 2; }
HOST="$(hostname)"
echo "HOST=$HOST"
[[ "$HOST" == gl* ]] || { echo "不是 GreatLakes 计算节点: $HOST" >&2; exit 2; }
[[ -d "$SHARED_ROOT" ]] || { echo "共享路径不可见: $SHARED_ROOT" >&2; exit 2; }
[[ ! -e "$LOCAL_ROOT" ]] || { echo "本机路径泄漏到计算节点: $LOCAL_ROOT" >&2; exit 2; }
! command -v micromamba >/dev/null || { echo "计算节点不应有 micromamba" >&2; exit 2; }
! command -v conda >/dev/null || { echo "计算节点不应有 conda" >&2; exit 2; }
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u || true)"
[[ "$GPU_NAMES" == "$EXPECTED_GPU" ]] || { echo "GPU 型号不符或 step 未拿到 GPU: ${GPU_NAMES:-无}" >&2; exit 2; }
[[ -f "$ENTRY" ]] || { echo "S1 入口不存在: $ENTRY" >&2; exit 2; }
# ---- 计算节点判定（END）----

# srun 默认把提交侧环境原样带进计算节点：显式 unset 诊断遗留开关。清单与原入口 run_orig16_stage2_60k.sh
# 的 unset 清单一致，但去掉 S1 入口的接口变量 TRAIN_TIMING_STEPS（由入口自己按接口处理），并一并清掉 *_ALLOW_UNVERIFIED。
DIAG_LEFTOVERS="PYTHONPATH PYTHONHOME XLA_FLAGS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET MMEVLA_FRAMESAMP_VERIFY WANDB_MODE JAX_COMPILATION_CACHE_DIR MMEVLA_MOTION_OVERFLOW"
# shellcheck disable=SC2086  # 有意按空格拆分成多个变量名
unset $DIAG_LEFTOVERS
for key in $(compgen -e); do
  case "$key" in *_ALLOW_UNVERIFIED) unset "$key" ;; esac
done

# S1 入口的 env 接口：只透传，已设的保持原值 export，未设的保持未设。
for key in GPUS GLOBAL_BATCH HISTORY RUN_NAME TRAIN_HEAD FINAL_RECORD_DIR MAX_STEPS TRAIN_TIMING_STEPS MMEVLA_STAGE2_EVIDENCE_ROOT; do
  if [[ -n "${!key+x}" ]]; then
    export "${key?}"
  fi
done

cd "$REPO" || { echo "仓库路径不可进入: $REPO" >&2; exit 2; }

# 运行器日志名以 o16gl- 开头（落在计划 R4 写入白名单 v1-store/logs/o16gl-* 内），带 run 名、占位 JobID 与 step 号，可回查。
RUNNER_TAG="${RUN_NAME:-noname}"
[[ "$RUNNER_TAG" =~ ^[a-zA-Z0-9_.-]+$ ]] || RUNNER_TAG=invalid-name
LOG="$LOG_DIR/o16gl-runner-${RUNNER_TAG}-${SLURM_JOB_ID}-s${SLURM_STEP_ID:-0}.log"
mkdir -p "$LOG_DIR"

cleanup() {
  local rc=$?
  echo "EXIT_CODE=$rc" >> "$LOG"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# 日志三件套：PYTHONUNBUFFERED=1、set -o pipefail（上方 set -euo pipefail 已开）、tee 落文件；cleanup 写 EXIT_CODE=。
echo "RUNNER=run_in_hold_o16 RUN_NAME=${RUN_NAME:-未设} JOB=$SLURM_JOB_ID STEP=${SLURM_STEP_ID:-0} HOST=$HOST GPU=$GPU_NAMES 起始=$(date -Is)" | tee -a "$LOG"
echo "ENV GPUS=${GPUS:-未设} GLOBAL_BATCH=${GLOBAL_BATCH:-未设} HISTORY=${HISTORY:-未设} MAX_STEPS=${MAX_STEPS:-未设} TRAIN_TIMING_STEPS=${TRAIN_TIMING_STEPS:-未设} TRAIN_HEAD=${TRAIN_HEAD:-未设} FINAL_RECORD_DIR=${FINAL_RECORD_DIR:-未设} MMEVLA_STAGE2_EVIDENCE_ROOT=${MMEVLA_STAGE2_EVIDENCE_ROOT:-未设}" | tee -a "$LOG"
echo "CMD=bash $ENTRY" | tee -a "$LOG"
PYTHONUNBUFFERED=1 bash "$ENTRY" 2>&1 | tee -a "$LOG"
