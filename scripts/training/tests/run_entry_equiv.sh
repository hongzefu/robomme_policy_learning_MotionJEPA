#!/usr/bin/env bash
# C4 上游 main 对拍驱动（v5.0，v5.0-train-entry-restructure-plan.md 第十节）。
#
# 子命令：
#   run-a   A 侧（上游 worktree @ ecf086c，官方 __main__ 双跑，加 --overwrite）
#   run-b   B 侧（本分支新 train.py，官方 __main__ 单跑，无 --overwrite）
#   judge   读两侧 record 比对，输出 4.5 判定行（ENTRY_EQ=PASS 一行定生死）
#
# 可覆盖环境变量：
#   WORKTREE     A 侧 worktree 根（默认 $REPO/v1-store/entryeq/worktree-main）
#   RECORD_ROOT  记录根（默认 $REPO/v1-store/entryeq/records；P1 冒烟可指临时目录）
#   STEPS        步数（默认 1000；P1 冒烟用 2）
#   EXP_A/EXP_B  exp-name（默认 entry-eq-a / entry-eq-b）
#
# 两侧 env 同：确定性档 XLA_FLAGS、MEM_FRACTION=0.95、CUDA 0,1；编译缓存目录经
# ~/.cache/jax_{exp_name} 由 --exp-name 自动分离（跨编译 bitwise 由 D2-cold 背书）。
# B 侧不设 TRAIN_RECORD_DIR（内置记录器不装，观测全由 harness 代理承担）。
# 1000 步两侧各约 1 h + 冷编译，须放 detached tmux 各挂一个 Monitor（AGENTS 7）。

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STORE="$REPO/v1-store"
WORKTREE="${WORKTREE:-$STORE/entryeq/worktree-main}"
RECORD_ROOT="${RECORD_ROOT:-$STORE/entryeq/records}"
STEPS="${STEPS:-1000}"
EXP_A="${EXP_A:-entry-eq-a}"
EXP_B="${EXP_B:-entry-eq-b}"
HARNESS="$REPO/scripts/training/tests/entry_equiv.py"
ANCHOR_SHA256="c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757"

export UV_LINK_MODE=copy PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=0,1

# argv 差异五处逐项登记（其余逐字符同）：入口文件、--exp-name、--dataset-path
# （链路差异载体，正是被测对象）、--checkpoint-base-dir、A 侧独有 --overwrite
common_args() {   # $1=exp_name $2=dataset_path $3=ckpt_base
    printf '%s\n' mme_vla_suite \
        --exp-name "$1" \
        --num-train-steps "$STEPS" --log-interval 1 --save-interval 100 \
        --batch-size 8 --num-workers 4 --seed 42 --fsdp-devices 2 \
        --dataset-path "$2" \
        --assets-base-dir "$STORE/train-assets" \
        --checkpoint-base-dir "$3" \
        --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
        --model.use-history --model.history-config perceptual-framesamp-context.yaml \
        --no-wandb-enabled
}

case "${1:-}" in
run-a)
    [ -d "$WORKTREE" ] || { echo "错误: worktree 不存在: $WORKTREE"; exit 1; }
    mapfile -t ARGS < <(common_args "$EXP_A" "$STORE/datasets/4task-gl" "$STORE/entryeq/ckpt-a")
    cd "$WORKTREE"
    exec env -u PYTHONPATH -u PYTHONHOME uv run python "$HARNESS" run \
        --entry "$WORKTREE/scripts/train.py" \
        --record-dir "$RECORD_ROOT/a" \
        --expect-root "$WORKTREE" \
        --expect-steps "$STEPS" \
        -- "${ARGS[@]}" --overwrite
    ;;
run-b)
    mapfile -t ARGS < <(common_args "$EXP_B" "$STORE/datasets/4task-gl-framesamp" "$STORE/entryeq/ckpt-b")
    cd "$REPO"
    exec env -u PYTHONPATH -u PYTHONHOME -u TRAIN_RECORD_DIR uv run python "$HARNESS" run \
        --entry "$REPO/scripts/training/train.py" \
        --record-dir "$RECORD_ROOT/b" \
        --expect-root "$REPO" \
        --expect-steps "$STEPS" \
        -- "${ARGS[@]}"
    ;;
judge)
    cd "$REPO"
    exec env -u PYTHONPATH -u PYTHONHOME uv run python "$HARNESS" judge \
        --a-dir "$RECORD_ROOT/a" --b-dir "$RECORD_ROOT/b" \
        --expect-sha256 "$ANCHOR_SHA256"
    ;;
*)
    echo "用法: $0 {run-a|run-b|judge}（详见文件头注释）"; exit 2
    ;;
esac
