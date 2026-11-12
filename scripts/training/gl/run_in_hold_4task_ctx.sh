#!/usr/bin/env bash
# GL 四任务×400 context 512/1024 b128 50k 训练的占位 job 运行器（计划 1111-gl-4task400-ctx512-1024-b128-50k-plan.html，
# 第二部分逐文件清单 S3 行、runbook P4/P5、验收 V8）。
# 底稿（只读）：/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/scripts/training/gl/run_in_hold_o16.sh；
# train.py 调用方式、数据/assets 路径与缓存环境变量沿用 scripts/dataset/hf_export/v2_1600ep_runner/_train_common.sh。
#
# 用法：占位 job（同目录 hold_4task_ctx.sbatch）RUNNING 后，经 srun 塞进去（长任务放 detached tmux）：
#   srun --jobid=<占位 JobID> --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
#     /usr/bin/env RUN_NAME=<run 名> CONFIG=<config 名> HISTORY=<yaml 文件名> GRAD_ACCUM=<k> MAX_STEPS=<n> \
#         TRAIN_HEAD=<40 位 sha> EXPECT_LIB_PIN=<64 位 sha256> [FSDP=4] [WORKERS=8] [RESUME=0|1] [DRY_RUN=0|1] \
#     bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/scripts/training/gl/run_in_hold_4task_ctx.sh
#
# 必填：RUN_NAME CONFIG HISTORY GRAD_ACCUM MAX_STEPS TRAIN_HEAD EXPECT_LIB_PIN；可选：FSDP=4 WORKERS=8 RESUME=0 DRY_RUN=0。
#   HISTORY 是 src/mme_vla_suite/models/config/robomme/ 下的 yaml 文件名（train.py 按仓库根相对路径解析）。
#   RESUME=1：追加 --resume-full（完整状态续跑，CLI 由 S4 实现，按此名契约），要求 run 目录已存在且含 resume/manifest.json。
#   DRY_RUN=1：只做参数校验，跳过依赖集群文件的检查（计算节点判定、CLEAN_HEAD、库 pin 比对、数据与 run 目录检查），
#              不写任何文件，打印完整 train.py 命令行后 exit 0。
#
# 前置检查（任一失败 exit 非 0 并打印 FAIL 判定行，计划 V8）：
#   a. CLEAN_HEAD=PASS sha=<HEAD>：HEAD == TRAIN_HEAD 且 git status --porcelain 为空（v1-store 已被 gitignore）
#   b. DATA_LINEAGE=NA encoder=none motion_enabled=false vla_input_pin=<LIB_PIN>：本 run 无 motion encoder，
#      LIB_PIN = sha256(meta/input_manifest.json)，缺该文件时用 meta/episode_manifest.json；HISTORY 的 motion.enabled 为 true 即 FAIL
#   c. PAIRED_RUNS_SAME_LIB=PASS pin=<LIB_PIN>：LIB_PIN == EXPECT_LIB_PIN（两条并行 run 读同一库）
#   d. source/data、norm_stats.json 存在；RESUME=0 时 run 目录不得存在，RESUME=1 时必须存在且含 resume/manifest.json
#
# 日志：$V1_STORE/logs/$RUN_NAME.log；RESUME=1 时为 $RUN_NAME.attemptNN.log（NN 从 02 起取下一个未用号）；不覆盖已有日志。
# 三件套：set -o pipefail、PYTHONUNBUFFERED=1、tee；结束追加 EXIT_CODE=<rc>（显式 trap TERM/INT 转成退出码，
# 避免被 TERM 杀掉时误写 EXIT_CODE=0）。集群侧只用仓库 .venv/bin/python 直调，不在计算节点 uv sync；禁止覆盖 HOME。
set -euo pipefail

GL_REPO="/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA"
SHARED_ROOT="/nfs/turbo/coe-chaijy-unreplicated/hongzefu"
LOCAL_ROOT="/data/hongzefu"
EXPECTED_GPU="NVIDIA A40"
DS_NAME="4task-v2-1600ep-604f16da"
HISTORY_DIR_REL="src/mme_vla_suite/models/config/robomme"

fail() { echo "$1" >&2; exit "${2:-2}"; }

[[ $# == 0 ]] || fail "本运行器不接受位置参数，参数一律用 env 传入（见文件头用法）"

# 仓库根按脚本位置推出（scripts/training/gl/ 上三级）；正式运行时必须恰为 turbo 仓库。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[[ -f "$REPO/pyproject.toml" ]] || fail "仓库根解析失败: $REPO（缺 pyproject.toml）"
V1_STORE="$REPO/v1-store"
DS="$V1_STORE/datasets/$DS_NAME"
ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/$DS_NAME"

# ---- 参数校验（DRY_RUN 也做）----
FSDP="${FSDP:-4}"
WORKERS="${WORKERS:-8}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
for key in RUN_NAME CONFIG HISTORY GRAD_ACCUM MAX_STEPS TRAIN_HEAD EXPECT_LIB_PIN; do
  [[ -n "${!key:-}" ]] || fail "PARAMS=FAIL reason=missing key=$key"
done
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]] || fail "PARAMS=FAIL reason=bad_run_name value=$RUN_NAME"
[[ "$CONFIG" =~ ^[a-zA-Z0-9_]+$ ]] || fail "PARAMS=FAIL reason=bad_config value=$CONFIG"
[[ "$HISTORY" =~ ^[a-zA-Z0-9_.-]+\.yaml$ ]] || fail "PARAMS=FAIL reason=bad_history value=$HISTORY（只给文件名，不带目录）"
for key in GRAD_ACCUM MAX_STEPS FSDP WORKERS; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail "PARAMS=FAIL reason=not_positive_int key=$key value=${!key}"
done
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || fail "PARAMS=FAIL reason=bad_train_head value=$TRAIN_HEAD"
[[ "$EXPECT_LIB_PIN" =~ ^[0-9a-f]{64}$ ]] || fail "PARAMS=FAIL reason=bad_expect_lib_pin value=$EXPECT_LIB_PIN"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || fail "PARAMS=FAIL reason=bad_resume value=$RESUME"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "PARAMS=FAIL reason=bad_dry_run value=$DRY_RUN"
RUN_DIR="$V1_STORE/train-runs/$CONFIG/$RUN_NAME"
HISTORY_PATH="$REPO/$HISTORY_DIR_REL/$HISTORY"
echo "PARAMS=PASS run=$RUN_NAME config=$CONFIG history=$HISTORY grad_accum=$GRAD_ACCUM max_steps=$MAX_STEPS fsdp=$FSDP workers=$WORKERS resume=$RESUME dry_run=$DRY_RUN"

# 从 yaml 读顶层 motion: 段的 enabled 值（无该段或无该键输出空串）；只认两格及以上缩进的直接子键。
motion_enabled_of() {
  awk '
    /^[^ #][^:]*:/ { in_motion = ($0 ~ /^motion:[[:space:]]*(#.*)?$/); next }
    in_motion && /^[[:space:]]+enabled:/ {
      v = $0; sub(/^[[:space:]]+enabled:[[:space:]]*/, "", v); sub(/[[:space:]]*#.*$/, "", v); print tolower(v); exit
    }' "$1"
}

VERDICTS=()
if [[ "$DRY_RUN" == 0 ]]; then
  # ---- 计算节点判定（沿用底稿）：本机或登录节点误跑一律 exit 2 ----
  [[ "$REPO" == "$GL_REPO" ]] || fail "仓库不在 turbo 集群路径: $REPO"
  [[ -n "${SLURM_JOB_ID:-}" ]] || fail "不在 Slurm 作业内：须经 srun --jobid=<占位 JobID> --overlap 塞入"
  HOST="$(hostname)"
  [[ "$HOST" == gl* ]] || fail "不是 GreatLakes 计算节点: $HOST"
  [[ -d "$SHARED_ROOT" ]] || fail "共享路径不可见: $SHARED_ROOT"
  [[ ! -e "$LOCAL_ROOT" ]] || fail "本机路径泄漏到计算节点: $LOCAL_ROOT"
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u || true)"
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)"
  [[ "$GPU_NAMES" == "$EXPECTED_GPU" ]] || fail "GPU 型号不符或 step 未拿到 GPU: ${GPU_NAMES:-无}"
  [[ "$GPU_COUNT" -ge "$FSDP" ]] || fail "可见 GPU 数 $GPU_COUNT 少于 FSDP=$FSDP"
  MODULE=$(cd "$REPO" && env -u PYTHONPATH -u PYTHONHOME "$REPO/.venv/bin/python" -c 'import pathlib, mme_vla_suite; print(pathlib.Path(mme_vla_suite.__file__).resolve())') || exit 2
  [[ "$MODULE" == "$REPO/src/"* ]] || fail "mme_vla_suite 导入位置错误: $MODULE"

  # a. CLEAN_HEAD
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD)"
  [[ "$HEAD_SHA" == "$TRAIN_HEAD" ]] || fail "CLEAN_HEAD=FAIL reason=head_mismatch head=$HEAD_SHA expect=$TRAIN_HEAD"
  [[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail "CLEAN_HEAD=FAIL reason=dirty sha=$HEAD_SHA"
  VERDICTS+=("CLEAN_HEAD=PASS sha=$HEAD_SHA")
  echo "${VERDICTS[-1]}"
fi

# b. 库 pin 与 motion 关闭（DRY_RUN 下文件不存在时跳过）
LIB_PIN=""
if [[ -f "$DS/meta/input_manifest.json" ]]; then
  LIB_PIN="$(sha256sum "$DS/meta/input_manifest.json" | cut -d' ' -f1)"
elif [[ -f "$DS/meta/episode_manifest.json" ]]; then
  LIB_PIN="$(sha256sum "$DS/meta/episode_manifest.json" | cut -d' ' -f1)"
elif [[ "$DRY_RUN" == 0 ]]; then
  fail "DATA_LINEAGE=FAIL reason=no_manifest ds=$DS"
fi
if [[ -f "$HISTORY_PATH" ]]; then
  ME="$(motion_enabled_of "$HISTORY_PATH")"
  [[ "$ME" != true ]] || fail "DATA_LINEAGE=FAIL reason=motion_enabled history=$HISTORY（本 run 无 encoder，motion 必须关闭）"
elif [[ "$DRY_RUN" == 0 ]]; then
  fail "DATA_LINEAGE=FAIL reason=history_missing path=$HISTORY_PATH"
else
  echo "DRY_RUN: HISTORY 文件不在本副本（$HISTORY_PATH），跳过 motion.enabled 检查"
fi
if [[ -n "$LIB_PIN" ]]; then
  VERDICTS+=("DATA_LINEAGE=NA encoder=none motion_enabled=false vla_input_pin=$LIB_PIN")
  echo "${VERDICTS[-1]}"
fi

if [[ "$DRY_RUN" == 0 ]]; then
  # c. PAIRED_RUNS_SAME_LIB
  [[ "$LIB_PIN" == "$EXPECT_LIB_PIN" ]] || fail "PAIRED_RUNS_SAME_LIB=FAIL pin=$LIB_PIN expect=$EXPECT_LIB_PIN"
  VERDICTS+=("PAIRED_RUNS_SAME_LIB=PASS pin=$LIB_PIN")
  echo "${VERDICTS[-1]}"

  # d. 数据与 run 目录
  [[ -d "$DS/source/data" ]] || fail "PRECHECK=FAIL reason=no_source_data path=$DS/source/data"
  [[ -f "$ASSETS_DIR/robomme/norm_stats.json" ]] || fail "PRECHECK=FAIL reason=no_norm_stats path=$ASSETS_DIR/robomme/norm_stats.json"
  [[ -d "$DS/framesamp-8x8" ]] || fail "PRECHECK=FAIL reason=no_framesamp path=$DS/framesamp-8x8"
  if [[ "$RESUME" == 0 ]]; then
    [[ ! -e "$RUN_DIR" ]] || fail "PRECHECK=FAIL reason=run_dir_exists path=$RUN_DIR（禁止复用 run 名；续跑用 RESUME=1）"
  else
    [[ -f "$RUN_DIR/resume/manifest.json" ]] || fail "PRECHECK=FAIL reason=no_resume_manifest path=$RUN_DIR/resume/manifest.json"
  fi
  VERDICTS+=("PRECHECK=PASS resume=$RESUME run_dir=$RUN_DIR")
  echo "${VERDICTS[-1]}"
fi

# ---- 组装 train.py 命令行 ----
WANDB_ARGS=()
if [[ ! -f "$V1_STORE/secrets/wandb.env" ]]; then
  WANDB_ARGS=(--no-wandb-enabled)
fi
RESUME_ARGS=()
[[ "$RESUME" == 1 ]] && RESUME_ARGS=(--resume-full)
CMD=("$REPO/.venv/bin/python" scripts/training/train.py "$CONFIG"
  --exp-name "$RUN_NAME"
  --assets-base-dir "$V1_STORE/train-assets"
  --data.assets.assets-dir "$ASSETS_DIR"
  --data.assets.asset-id robomme
  --checkpoint-base-dir "$V1_STORE/train-runs"
  --dataset-path "$DS/framesamp-8x8"
  --model.history-config "$HISTORY"
  --fsdp-devices "$FSDP"
  --num-workers "$WORKERS"
  --grad-accum-steps "$GRAD_ACCUM"
  --num-train-steps "$MAX_STEPS"
  "${WANDB_ARGS[@]}" "${RESUME_ARGS[@]}")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY_RUN_CMD='; printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

# ---- 日志名：不覆盖已有日志 ----
LOG_DIR="$V1_STORE/logs"
if [[ "$RESUME" == 0 ]]; then
  LOG="$LOG_DIR/$RUN_NAME.log"
else
  n=2
  while :; do
    LOG="$(printf '%s/%s.attempt%02d.log' "$LOG_DIR" "$RUN_NAME" "$n")"
    [[ -e "$LOG" ]] || break
    n=$((n + 1))
    [[ $n -le 99 ]] || fail "续跑尝试号已用尽（02..99）"
  done
fi
[[ ! -e "$LOG" ]] || fail "拒绝覆盖已有日志: $LOG"
mkdir -p "$LOG_DIR"

# srun 默认把提交侧环境带进计算节点：显式清掉诊断遗留开关（沿用底稿与 _train_common.sh）。
unset PYTHONPATH PYTHONHOME XLA_FLAGS JAX_PLATFORMS MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET \
  MMEVLA_FRAMESAMP_VERIFY WANDB_MODE JAX_COMPILATION_CACHE_DIR MMEVLA_MOTION_OVERFLOW
for key in $(compgen -e); do
  case "$key" in *_ALLOW_UNVERIFIED) unset "$key" ;; esac
done

# 缓存类环境变量逐项指向 v1-store（禁止覆盖 HOME）。
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="$V1_STORE/cache/uv"
export XDG_CACHE_HOME="$V1_STORE/cache/xdg"
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN_NAME"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# store_meta.json 里写死的是建库机绝对路径；这两条让读侧改看 turbo 实际落点。别设 MMEVLA_FRAMESAMP_VERIFY=full。
export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"

cd "$REPO"   # history config 按相对 cwd 解析，必须从仓库根起跑

cleanup() {
  local rc=$?
  echo "EXIT_CODE=$rc" >> "$LOG"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

body() {
  echo "RUNNER=run_in_hold_4task_ctx RUN_NAME=$RUN_NAME JOB=$SLURM_JOB_ID STEP=${SLURM_STEP_ID:-0} HOST=$HOST GPU=$GPU_NAMES×$GPU_COUNT 起始UTC=$(date -u +%FT%TZ)"
  printf '%s\n' "${VERDICTS[@]}"
  if [[ ${#WANDB_ARGS[@]} -gt 0 ]]; then
    echo "提示: 没有 v1-store/secrets/wandb.env，本轮关闭 wandb"
  else
    set -a; . "$V1_STORE/secrets/wandb.env"; set +a
  fi
  printf 'CMD='; printf '%q ' "${CMD[@]}"; printf '\n'
  "${CMD[@]}"
}

set +e
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
exit "$rc"
