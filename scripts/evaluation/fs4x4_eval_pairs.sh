#!/usr/bin/env bash
# 1127 计划（1127-4task-modul-fs4x4-512-1024-60k-plan.md 第四节第 6 条、第五节 G5、第六节第 5 步）的评估调度：
# 在 v42 评估栈（分支 v20-evalstack-fs4x4，基线 c3876b9f）上评两个 4×4 run，照抄
# v1-store/jobs/t1s-1120/t1s-lib.sh 的 cfg()/run()/pair()（Table I fs 行同一口径）。
#
# 用法：
#   fs4x4_eval_pairs.sh --dry-run              只打印 3 对 6 次的 RUN_NAME、seed、shard_base、POLICY、POLICY_CONFIG、ckpt，不跑
#   fs4x4_eval_pairs.sh --g6b <ckpt> <budget>  G5：先 check_checkpoint.py <ckpt>，再 local_692_pool.sh --g6b <ckpt> 7；
#                                              判定行 FS4X4_G5=PASS|FAIL
#   fs4x4_eval_pairs.sh                        正式 3 对：(A s0, A s42)、(B s0, B s42)、(A s7, B s7)；
#                                              判定行 FS4X4_EVAL_PAIRS=PASS done=6 | FAIL done=<n> failed=<列表>
# 每对两次同卡并行：共用 EVAL_SLOT_DIR、EVAL_GPU_CAP=4、EVAL_SHARDS=16、EVAL_PER_GPU=2、EVAL_MEM_FRACTION=0.2，
# EVAL_SHARD_BASE 0 / 16，两次起跑间隔 20 秒。已有 EVAL692=PASS 的 RUN_NAME 跳过（幂等）。
# 评估目录 = <本 worktree>/v1-store/evaluation/<RUN_NAME>（local_692_pool.sh 按 EVAL_REPO 落盘，与 t1s 同口径：
# 评估 worktree 的 v1-store 必须是指向主检出 v1-store 的软链，非 dry-run 时核对，不符即拒绝起跑）。
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
TR=$MAIN/v1-store/train-runs/mme_vla_suite_b128_80k
LOGS=$MAIN/v1-store/logs
JOBS=$MAIN/v1-store/jobs/fs4x4-1127
EVROOT=$W/v1-store/evaluation
ENTRY=scripts/evaluation/local_692_pool.sh
POLICY_CONFIG=mme_vla_suite_b128_80k
FILT='LOCAL692_START|LOCAL692_DONE|EVAL692=|EVAL_LOAD_SMOKE=|SHARD_EXIT_CODE=[1-9]|SLOT_REFUSED|拒绝|停止起跑|未在占位文件登记|不属于已登记分片|Traceback|out of memory|RESOURCE_EXHAUSTED|E0_GATE=FAIL'
RUNS_ALL=(fs4x4-a-s0 fs4x4-a-s42 fs4x4-b-s0 fs4x4-b-s42 fs4x4-a-s7 fs4x4-b-s7)
pp(){ echo "$1/src:$1:$1/third_party/robomme_benchmark/src:$1/packages/openpi-client/src"; }

# cfg <a|b>：设 CKPT POLICY ENVV（数组）
cfg(){
  case $1 in
    a) CKPT=$TR/v2-1600ep-m32x4x4-modul-b128-50k/50000; POLICY=perceptual-framesamp-modul-32frame-4x4;;
    b) CKPT=$TR/v2-1600ep-m64x4x4-modul-b128-50k/50000; POLICY=perceptual-framesamp-modul-64frame-4x4;;
    *) echo "未知行键 $1"; return 2;;
  esac
  ENVV=(PYTHONPATH="$(pp "$W")" EXPECT_MOTION_STATS=0 MMEVLA_MOTION_OVERFLOW= POLICY=$POLICY POLICY_CONFIG=$POLICY_CONFIG)
}

# budget -> POLICY（G5 用）；冒烟 / 正式 run 根名 -> budget
policy_of_budget(){ case $1 in 512) echo perceptual-framesamp-modul-32frame-4x4;; 1024) echo perceptual-framesamp-modul-64frame-4x4;; *) return 2;; esac; }
budget_of_root(){
  case $1 in
    v2-1600ep-m32x4x4-modul-b128-50k|fs4a-smoke20) echo 512;;
    v2-1600ep-m64x4x4-modul-b128-50k|fs4b-smoke20) echo 1024;;
    *) return 2;;
  esac
}

check_v1store(){
  local got; got="$(readlink -f "$W/v1-store" 2>/dev/null)"
  if [[ "$got" != "$(readlink -f "$MAIN/v1-store")" ]]; then
    echo "拒绝：$W/v1-store 未指向主检出 v1-store（readlink=$got），评估产物会落到错误位置"; return 1
  fi
}

eval_first_line(){ local f=$EVROOT/$1/EVAL692.txt; [[ -f $f ]] && head -1 "$f" || echo missing; }
# 幂等判据：首行 EVAL692=PASS 且含 episodes=692
is_pass(){ local l; l="$(eval_first_line "$1")"; [[ "$l" == EVAL692=PASS* && " $l " == *" episodes=692 "* ]]; }

# run <a|b> <seed> <shard_base> <slot 目录>
run(){
  local k=$1 seed=$2 base=$3 D=$4 nm="fs4x4-$1-s$2"
  cfg "$k" || { echo "FS4X4_ITEM_FAIL run=$nm reason=cfg"; return; }
  if is_pass "$nm"; then echo "FS4X4_ITEM_SKIP run=$nm reason=already_pass $(eval_first_line "$nm")"; return; fi
  if [[ ! -d $CKPT/params ]]; then echo "FS4X4_ITEM_FAIL run=$nm reason=ckpt_missing ckpt=$CKPT"; return; fi
  echo "FS4X4_RUN_START run=$nm seed=$seed shard_base=$base policy=$POLICY ckpt=$CKPT $(date -u +%FT%TZ)"
  ( cd "$W" && env "${ENVV[@]}" EVAL_SLOT_DIR="$D" EVAL_GPU_CAP=4 EVAL_SHARDS=16 EVAL_PER_GPU=2 EVAL_MEM_FRACTION=0.2 \
      EVAL_SHARD_BASE="$base" RUN_NAME="$nm" PYTHONUNBUFFERED=1 bash "$ENTRY" "$CKPT" "$seed" 2>&1 \
      | tee "$LOGS/$nm.log" | grep --line-buffered -E "$FILT" | sed -u "s|^|[$nm] |" )
  if is_pass "$nm"; then echo "FS4X4_ITEM run=$nm $(eval_first_line "$nm")"
  else echo "FS4X4_ITEM_FAIL run=$nm eval692=$(eval_first_line "$nm")"; fi
}

# pair <对名> <键1> <seed1> <键2> <seed2>：第一次 shard_base 0，20 秒后第二次 shard_base 16，共用占位目录
pair(){
  local name=$1 n=1 D
  while [[ -e $JOBS/slots-$name-r$n ]]; do n=$((n + 1)); done
  D=$JOBS/slots-$name-r$n; mkdir -p "$D"
  echo "FS4X4_PAIR_START pair=$name slot_dir=$D $(date -u +%FT%TZ)"
  run "$2" "$3" 0 "$D" &
  sleep 20
  run "$4" "$5" 16 "$D" &
  wait
}

PAIRS=("a0a42 a 0 a 42" "b0b42 b 0 b 42" "a7b7 a 7 b 7")

case "${1:-}" in
  --dry-run)
    echo "FS4X4_DRY worktree=$W eval_root=$EVROOT logs=$LOGS v1store=$(readlink -f "$W/v1-store" 2>/dev/null || echo missing)"
    i=0
    for p in "${PAIRS[@]}"; do
      read -r name k1 s1 k2 s2 <<<"$p"; i=$((i + 1))
      for j in 1 2; do
        if [[ $j == 1 ]]; then k=$k1; s=$s1; base=0; else k=$k2; s=$s2; base=16; fi
        cfg "$k"
        echo "FS4X4_DRY pair=$i/$name run=fs4x4-$k-s$s seed=$s shard_base=$base POLICY=$POLICY POLICY_CONFIG=$POLICY_CONFIG ckpt=$CKPT ckpt_present=$([[ -d $CKPT/params ]] && echo 1 || echo 0) eval692=$(eval_first_line "fs4x4-$k-s$s")"
      done
    done
    echo "FS4X4_DRY env: PYTHONPATH=$(pp "$W") EXPECT_MOTION_STATS=0 MMEVLA_MOTION_OVERFLOW= EVAL_GPU_CAP=4 EVAL_SHARDS=16 EVAL_PER_GPU=2 EVAL_MEM_FRACTION=0.2 gap=20s"
    exit 0;;
  --g6b)
    CK="${2:?--g6b 需要 <ckpt> <budget>}"; BUDGET="${3:?--g6b 需要 <ckpt> <budget>}"
    [[ "$CK" == /* ]] || { echo "ckpt 必须是绝对路径：$CK"; echo "FS4X4_G5=FAIL reason=ckpt_not_abs"; exit 2; }
    POLICY="$(policy_of_budget "$BUDGET")" || { echo "FS4X4_G5=FAIL reason=bad_budget budget=$BUDGET"; exit 2; }
    RB="$(budget_of_root "$(basename "$(dirname "$CK")")")" || { echo "FS4X4_G5=FAIL reason=unknown_run_root ckpt=$CK"; exit 2; }
    [[ "$RB" == "$BUDGET" ]] || { echo "FS4X4_G5=FAIL reason=budget_mismatch run_budget=$RB arg=$BUDGET"; exit 2; }
    [[ -d $CK/params ]] || { echo "FS4X4_G5=FAIL reason=ckpt_missing ckpt=$CK"; exit 2; }
    check_v1store || { echo "FS4X4_G5=FAIL reason=v1store_link"; exit 2; }
    # 每次新目录：fs4x4-g6b-<budget>-<ckpt 父目录名>-r<n>，n 递增、不复用已存在目录，只读本次结果
    G6B_BASE="fs4x4-g6b-$BUDGET-$(basename "$(dirname "$CK")")"; n=1
    while [[ -e $EVROOT/$G6B_BASE-r$n || -e $LOGS/$G6B_BASE-r$n.log ]]; do n=$((n + 1)); done
    NM="$G6B_BASE-r$n"
    ENVV=(PYTHONPATH="$(pp "$W")" EXPECT_MOTION_STATS=0 MMEVLA_MOTION_OVERFLOW= POLICY=$POLICY POLICY_CONFIG=$POLICY_CONFIG)
    echo "FS4X4_G5_START run=$NM ckpt=$CK budget=$BUDGET policy=$POLICY $(date -u +%FT%TZ)"
    CKLOG=$LOGS/$NM-check_checkpoint.log
    ( cd "$W" && source scripts/evaluation/env.sh && env "${ENVV[@]}" JAX_PLATFORMS=cpu "${SERVER_UV[@]}" \
        scripts/evaluation/check_checkpoint.py "$CK" ) 2>&1 | tee "$CKLOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc != 0 ]] || ! grep -q '^PARAM_TREE_EXACT=PASS' "$CKLOG" || ! grep -q '^HISTORY_CONFIG=PASS' "$CKLOG"; then
      echo "FS4X4_G5=FAIL reason=check_checkpoint rc=$rc log=$CKLOG"; exit 1
    fi
    ( cd "$W" && env "${ENVV[@]}" RUN_NAME="$NM" PYTHONUNBUFFERED=1 bash "$ENTRY" --g6b "$CK" 7 2>&1 \
        | tee "$LOGS/$NM.log" | grep --line-buffered -E "$FILT" | sed -u "s|^|[$NM] |" )
    f=$EVROOT/$NM/EVAL_LOAD_SMOKE.txt
    line="$( [[ -f $f ]] && head -1 "$f" || echo missing)"
    if [[ "$line" == EVAL_LOAD_SMOKE=PASS* ]]; then echo "FS4X4_G5=PASS run=$NM budget=$BUDGET $line"; exit 0; fi
    echo "FS4X4_G5=FAIL run=$NM budget=$BUDGET eval_load_smoke=$line log=$LOGS/$NM.log"; exit 1;;
  "")
    check_v1store || { echo "FS4X4_EVAL_PAIRS=FAIL done=0 failed=v1store_link"; exit 2; }
    mkdir -p "$JOBS" "$LOGS"
    for p in "${PAIRS[@]}"; do
      read -r name k1 s1 k2 s2 <<<"$p"
      pair "$name" "$k1" "$s1" "$k2" "$s2"
    done
    done_n=0; failed=()
    for nm in "${RUNS_ALL[@]}"; do if is_pass "$nm"; then done_n=$((done_n + 1)); else failed+=("$nm"); fi; done
    if (( ${#failed[@]} == 0 )); then echo "FS4X4_EVAL_PAIRS=PASS done=$done_n"; exit 0; fi
    echo "FS4X4_EVAL_PAIRS=FAIL done=$done_n failed=$(IFS=,; echo "${failed[*]}")"; exit 1;;
  *) echo "用法：$0 [--dry-run | --g6b <ckpt> <budget>]"; exit 2;;
esac
