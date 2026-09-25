#!/usr/bin/env bash
# C4 上游 main 对拍驱动（v5.0，0830-train-entry-restructure-plan.md 第十节）。
#
# 子命令：
#   run-a   A 侧（上游 worktree @ ecf086c，官方 __main__ 双跑，加 --overwrite）
#   run-b   B 侧（本分支新 train.py，官方 __main__ 单跑，无 --overwrite）
#   judge   读两侧 record 比对，输出 4.5 判定行（ENTRY_EQ=PASS 一行定生死）
#
# 默认训练 argv 保留历史 b8/fsdp2/context；0925 计划必须显式提供对应覆盖项。
# 可覆盖：WORKTREE、RECORD_ROOT、RECORD_A/B、STEPS、EXP_A/B、HISTORY_YAML、BATCH、
# FSDP、GPUS、SAVE_INTERVAL、DATA_A/B、CHECKPOINT_A/B、ASSETS_DIR、ANCHOR_SHA256、
# HEAD_A/B、MODE、EXPECT_STATE_STEPS、EXPECT_TENTATIVE_A、EXECUTION_ROLE、
# FRAMESAMP_SOURCE/MANIFEST（仅 B）、JAX_CACHE_A、MMEVLA_JAX_CACHE_DIR（仅 B）、
# A_PYTHON（独立 uv 环境）、CONCURRENT_PEER_DIR（same-entry 的 S2 另一侧）。
# ANCHOR_SHA256 未设沿用历史值，显式空值表示跳过锚点。
# dry-run-a/b 逐行输出纯训练 argv；dry-judge 逐行输出判定 argv，均不启动 Python。
# P1 STEPS=2 只验 run 输出，不允许调用要求状态 step 无重复的 judge。
# 两侧使用相同确定性档，缓存落在本仓库 v1-store；不改变 HOME。
# B 侧不设 TRAIN_RECORD_DIR（内置记录器不装，观测全由 harness 代理承担）。
# 1000 步两侧各约 1 h + 冷编译，须放 detached tmux 各挂一个 Monitor（AGENTS 7）。

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STORE="$REPO/v1-store"
WORKTREE="${WORKTREE:-$STORE/entryeq/worktree-main}"
RECORD_ROOT="${RECORD_ROOT:-$STORE/entryeq/records}"
STEPS="${STEPS:-1000}"
EXP_A="${EXP_A:-entry-eq-a}"
EXP_B="${EXP_B:-entry-eq-b}"
HARNESS="$REPO/scripts/training/tests/entry_equiv.py"
ANCHOR_SHA256="${ANCHOR_SHA256-c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757}"
HISTORY_YAML="${HISTORY_YAML:-perceptual-framesamp-context.yaml}"
BATCH="${BATCH:-8}"
FSDP="${FSDP:-2}"
GPUS="${GPUS:-0,1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
DATA_A="${DATA_A:-$STORE/datasets/4task-gl}"
DATA_B="${DATA_B:-$STORE/datasets/4task-gl-framesamp}"
CHECKPOINT_A="${CHECKPOINT_A:-$STORE/entryeq/ckpt-a}"
CHECKPOINT_B="${CHECKPOINT_B:-$STORE/entryeq/ckpt-b}"
RECORD_A="${RECORD_A:-$RECORD_ROOT/a}"
RECORD_B="${RECORD_B:-$RECORD_ROOT/b}"
HEAD_A="${HEAD_A:-ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b}"
HEAD_B="${HEAD_B:-$(git -C "$REPO" rev-parse HEAD)}"
MODE="${MODE:-upstream}"
EXECUTION_ROLE="${EXECUTION_ROLE:-upstream}"
A_PYTHON="${A_PYTHON:-$WORKTREE/.venv/bin/python}"
JAX_CACHE_A="${JAX_CACHE_A:-$STORE/cache/jax/$EXP_A}"
MMEVLA_JAX_CACHE_DIR="${MMEVLA_JAX_CACHE_DIR:-$STORE/cache/jax/$EXP_B}"

for number in "$STEPS" "$BATCH" "$FSDP" "$SAVE_INTERVAL"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo "错误: 正整数参数无效: $number" >&2; exit 2; }
done
(( STEPS >= 2 )) || { echo '错误: 对拍至少两步' >&2; exit 2; }
[[ "$MODE" == upstream || "$MODE" == same-entry ]] || { echo '错误: MODE 无效' >&2; exit 2; }
if [[ -n "${ASSETS_DIR+x}" && "$ASSETS_DIR" != /* ]]; then
    echo '错误: ASSETS_DIR 必须为绝对路径' >&2; exit 2
fi
if [[ -z "${EXPECT_TENTATIVE_A+x}" ]]; then
    if [[ "$MODE" == same-entry ]]; then EXPECT_TENTATIVE_A=0
    elif (( STEPS < 12 )); then EXPECT_TENTATIVE_A="$STEPS"
    else EXPECT_TENTATIVE_A=12
    fi
fi
if [[ -z "${EXPECT_STATE_STEPS+x}" ]]; then
    state_steps=()
    for (( step=SAVE_INTERVAL; step<STEPS-1; step+=SAVE_INTERVAL )); do state_steps+=("$step"); done
    state_steps+=("$((STEPS-1))")
    EXPECT_STATE_STEPS="$(IFS=,; printf '%s' "${state_steps[*]}")"
fi

export UV_LINK_MODE=copy PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES="$GPUS"
export UV_CACHE_DIR="$STORE/cache/uv" OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache" HF_HOME="$STORE/cache/huggingface"

# argv 差异五处逐项登记（其余逐字符同）：入口文件、--exp-name、--dataset-path
# （链路差异载体，正是被测对象）、--checkpoint-base-dir、A 侧独有 --overwrite
common_args() {   # $1=exp_name $2=dataset_path $3=ckpt_base
    printf '%s\n' mme_vla_suite \
        --exp-name "$1" \
        --num-train-steps "$STEPS" --log-interval 1 --save-interval "$SAVE_INTERVAL" \
        --batch-size "$BATCH" --num-workers 4 --seed 42 --fsdp-devices "$FSDP" \
        --dataset-path "$2" \
        --assets-base-dir "$STORE/train-assets" \
        --checkpoint-base-dir "$3" \
        --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
        --model.use-history --model.history-config "$HISTORY_YAML" \
        --no-wandb-enabled
    if [[ -n "${ASSETS_DIR+x}" ]]; then
        printf '%s\n' --data.assets.assets-dir "$ASSETS_DIR" --data.assets.asset-id robomme
    fi
}

case "${1:-}" in
run-a|dry-run-a)
    mapfile -t ARGS < <(common_args "$EXP_A" "$DATA_A" "$CHECKPOINT_A")
    if [[ "$1" == dry-run-a ]]; then printf '%s\n' "${ARGS[@]}" --overwrite; exit 0; fi
    [ -d "$WORKTREE" ] || { echo "错误: worktree 不存在: $WORKTREE"; exit 1; }
    [[ -x "$A_PYTHON" ]] || { echo "错误: A 侧独立 uv 解释器不存在: $A_PYTHON" >&2; exit 1; }
    export CUDA_CACHE_PATH="$STORE/cache/cuda/$EXP_A"
    export WANDB_DIR="$STORE/entryeq/wandb/$EXP_A" WANDB_CACHE_DIR="$STORE/cache/wandb/$EXP_A"
    export WANDB_CONFIG_DIR="$STORE/cache/wandb-config/$EXP_A" WANDB_DATA_DIR="$STORE/cache/wandb-data/$EXP_A"
    cd "$WORKTREE"
    exec env -u PYTHONPATH -u PYTHONHOME -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR \
        -u TRAIN_TIMING_STEPS -u MMEVLA_FRAMESAMP_SOURCE -u MMEVLA_FRAMESAMP_MANIFEST \
        -u MMEVLA_JAX_CACHE_DIR "$A_PYTHON" "$HARNESS" run \
        --entry "$WORKTREE/scripts/train.py" \
        --record-dir "$RECORD_A" \
        --expect-root "$WORKTREE" \
        --expect-steps "$STEPS" \
        --expect-tentative "$EXPECT_TENTATIVE_A" --expect-head "$HEAD_A" \
        --execution-role "$EXECUTION_ROLE" --jax-cache-dir "$JAX_CACHE_A" \
        -- "${ARGS[@]}" --overwrite
    ;;
run-b|dry-run-b)
    mapfile -t ARGS < <(common_args "$EXP_B" "$DATA_B" "$CHECKPOINT_B")
    if [[ "$1" == dry-run-b ]]; then printf '%s\n' "${ARGS[@]}"; exit 0; fi
    export CUDA_CACHE_PATH="$STORE/cache/cuda/$EXP_B"
    export WANDB_DIR="$STORE/entryeq/wandb/$EXP_B" WANDB_CACHE_DIR="$STORE/cache/wandb/$EXP_B"
    export WANDB_CONFIG_DIR="$STORE/cache/wandb-config/$EXP_B" WANDB_DATA_DIR="$STORE/cache/wandb-data/$EXP_B"
    cd "$REPO"
    exec env -u PYTHONPATH -u PYTHONHOME -u TRAIN_RECORD_DIR -u TRAIN_FINAL_RECORD_DIR \
        -u TRAIN_TIMING_STEPS MMEVLA_FRAMESAMP_SOURCE="${FRAMESAMP_SOURCE:-}" \
        MMEVLA_FRAMESAMP_MANIFEST="${FRAMESAMP_MANIFEST:-}" MMEVLA_JAX_CACHE_DIR="$MMEVLA_JAX_CACHE_DIR" \
        uv run --no-sync python "$HARNESS" run \
        --entry "$REPO/scripts/training/train.py" \
        --record-dir "$RECORD_B" \
        --expect-root "$REPO" --forbid-root "$WORKTREE" \
        --expect-steps "$STEPS" \
        --expect-tentative 0 --expect-head "$HEAD_B" --execution-role "$EXECUTION_ROLE" \
        --jax-cache-dir "$MMEVLA_JAX_CACHE_DIR" \
        -- "${ARGS[@]}"
    ;;
judge|dry-judge)
    (( STEPS != 2 )) || { echo '错误: P1 只判 run 成功/分段/有限值/来源，不运行轨迹 judge' >&2; exit 2; }
    JUDGE_ARGS=(judge --a-dir "$RECORD_A" --b-dir "$RECORD_B" --mode "$MODE"
        --expect-steps "$STEPS" --expect-tentative-a "$EXPECT_TENTATIVE_A"
        --expect-state-steps "$EXPECT_STATE_STEPS" --expect-head-a "$HEAD_A" --expect-head-b "$HEAD_B")
    [[ -z "$ANCHOR_SHA256" ]] || JUDGE_ARGS+=(--expect-sha256 "$ANCHOR_SHA256")
    [[ -z "${CONCURRENT_PEER_DIR:-}" ]] || JUDGE_ARGS+=(--concurrent-peer-dir "$CONCURRENT_PEER_DIR")
    if [[ "$1" == dry-judge ]]; then printf '%s\n' "${JUDGE_ARGS[@]}"; exit 0; fi
    cd "$REPO"
    exec env -u PYTHONPATH -u PYTHONHOME uv run --no-sync python "$HARNESS" "${JUDGE_ARGS[@]}"
    ;;
*)
    echo "用法: $0 {run-a|run-b|judge|dry-run-a|dry-run-b|dry-judge}（详见文件头注释）"; exit 2
    ;;
esac
