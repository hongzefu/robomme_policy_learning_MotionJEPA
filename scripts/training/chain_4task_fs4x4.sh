#!/usr/bin/env bash
# 1127 4task modulation 帧记忆每帧 4×4（预算 512 / 1024）触发链
# （1127-4task-modul-fs4x4-512-1024-60k-plan.md 第一部分六节、第二部分一节 S4 行与二节 runbook 第 6 步）。
# 在 tmux fs4-chain 内、于 `git checkout --detach <Beta>` 后的主检出上运行：
#   bash scripts/training/chain_4task_fs4x4.sh <BETA_SHA> <EVAL_WT> [--dry-run] [--from <step>]
# 步骤（严格串行，任一失败即停链；评估只在两个正式训练都完成之后）：
#   preflight     ① 主检出 HEAD == BETA 且工作区干净；EVAL_WT 存在且含两个评估入口
#   g4_512        ③ 8 卡空闲后跑 runner --budget 512 --smoke20（fs4a-smoke20），EXIT_CODE=0 且 ckpt 19 落盘
#   g4_1024       ③ 同上 --budget 1024（fs4b-smoke20）
#   g5            ④ 评估加载冒烟 fs4x4_eval_pairs.sh --g6b <fs4a-smoke20/19> 512 → FS4X4_G5=PASS
#   train_512     ⑤ 正式训练 A v2-1600ep-m32x4x4-modul-b128-50k（10 份 ckpt 5000…50000）
#   train_1024    ⑥ 正式训练 B v2-1600ep-m64x4x4-modul-b128-50k
#   eval_pairs    ⑦ 两个正式 run 的 EXIT_CODE=0 与 50000 都在后，fs4x4_eval_pairs.sh（3 对 6 次）→ FS4X4_EVAL_PAIRS=PASS done=6
#   summary       ⑧ fs4x4_3seed_summary.py → FS4X4_3SEED=PASS rows=2 seeds=3
#   ② 等 8 卡显存都 < 2000 MiB：在每个 GPU 步骤（g4_*、g5、train_*、eval_pairs）开始前做；每 60 s 查一次，
#     30 次（30 分钟）仍不空闲打印 FS4X4_CHAIN=FAIL step=gpu_wait 退出（有限循环，非 agent 轮询）。
# 每步打印 FS4X4_STEP=<名> START <UTC> 与 FS4X4_STEP=<名> PASS|FAIL rc=<n>；全部完成打印 FS4X4_CHAIN=DONE。
# 每步 PASS 后写状态标记 $JOBS/chain-state/<名>.pass；每步输出另 tee 一份到 $JOBS/chain-state/<名>.<UTC>.out。
# --from <step>：<step> 之前的步骤不再执行，逐个依据产物核实已 PASS（状态标记 + 日志/ckpt/判定行），核不过即
#   FS4X4_CHAIN=FAIL step=<名> 退出；preflight 永远执行。
# --dry-run：只打印将执行的命令（含 GPU 等待），不跑 runner/评估、不查 GPU、不写任何文件；
#   preflight 的三项只报告（FS4X4_PREFLIGHT_DRY …）不拦；--from 的产物核实照常只读执行。
#   测试钩子（只允许与 --dry-run 同用）：FS4X4_CHAIN_STORE=<目录> 替换 $MAIN/v1-store（夹具根）。
# 链里不起 HF 上传（fs4-incA / fs4-incB 由主会话另起 tmux 跑 scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh）。
# 不吞退出码：不用 || true / set +e。
set -uo pipefail

usage() { printf '用法: %s <BETA_SHA 40 位> <EVAL_WT> [--dry-run] [--from <step>]\n' "$0" >&2; exit 2; }

STEPS=(preflight g4_512 g4_1024 g5 train_512 train_1024 eval_pairs summary)

BETA="${1:-}"; EVAL_WT="${2:-}"
[ -n "$BETA" ] && [ -n "$EVAL_WT" ] || usage
shift 2
DRY=0; FROM=preflight
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --from) [ $# -ge 2 ] || usage; FROM="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$BETA" =~ ^[0-9a-f]{40}$ ]] || { echo "错误: BETA_SHA 必须是 40 位 sha: $BETA" >&2; exit 2; }
FROM_IDX=-1
for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$FROM" ] && FROM_IDX=$i; done
[ "$FROM_IDX" -ge 0 ] || { echo "错误: --from 只接受 ${STEPS[*]}，收到 $FROM" >&2; exit 2; }

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE_ROOT="$MAIN/v1-store"
if [ -n "${FS4X4_CHAIN_STORE:-}" ]; then
  [ "$DRY" = 1 ] || { echo "错误: FS4X4_CHAIN_STORE 只允许与 --dry-run 同用" >&2; exit 2; }
  STORE_ROOT="$FS4X4_CHAIN_STORE"
fi
RUNNER="$MAIN/scripts/training/run_4task_fs4x4_50k.sh"
TR="$STORE_ROOT/train-runs/mme_vla_suite_b128_80k"
LOGS="$STORE_ROOT/logs"
JOBS="$STORE_ROOT/jobs/fs4x4-1127"
STATE="$JOBS/chain-state"
SUMMARY_JSON="$JOBS/fs4x4_3seed.json"
EVAL_PAIRS="scripts/evaluation/fs4x4_eval_pairs.sh"
SUMMARY_PY="scripts/evaluation/fs4x4_3seed_summary.py"
GPU_LIMIT_MIB=2000
GPU_POLL_S=60
GPU_POLL_N=30
FORMAL_512=v2-1600ep-m32x4x4-modul-b128-50k
FORMAL_1024=v2-1600ep-m64x4x4-modul-b128-50k
SMOKE_512=fs4a-smoke20
SMOKE_1024=fs4b-smoke20
FORMAL_CKPTS=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)

utc() { date -u +%FT%TZ; }
q() { printf '%q' "$1"; }
chain_fail() { echo "FS4X4_CHAIN=FAIL step=$1 ${2:-}"; exit 1; }

# 日志末个 EXIT_CODE= 值（无则 missing）
log_exit() { local v; v="$(sed -n 's/^EXIT_CODE=\([0-9][0-9]*\)$/\1/p' "$1" 2>/dev/null | tail -n 1)"; echo "${v:-missing}"; }
# orbax ckpt 写完：<step>/_CHECKPOINT_METADATA 在、同号 tmp 目录不在（与 run_*_ckpt_export.sh 判据一致）
ckpt_ready() { [ -f "$1/$2/_CHECKPOINT_METADATA" ] && ! compgen -G "$1/$2.orbax-checkpoint-tmp*" > /dev/null; }

# ---- 产物核实（--from 跳过时与每步执行后共用；只读） ----
verify_run() {  # $1 run 名, $2 smoke|formal；返回 0 = 产物齐全，否则打印原因返回 1
  local run="$1" kind="$2" log="$LOGS/$1.log" rc s
  rc="$(log_exit "$log")"
  [ "$rc" = 0 ] || { echo "  产物核实失败: $log 末个 EXIT_CODE=$rc"; return 1; }
  if [ "$kind" = smoke ]; then
    ckpt_ready "$TR/$run" 19 || { echo "  产物核实失败: $TR/$run/19 未落盘完成"; return 1; }
  else
    for s in "${FORMAL_CKPTS[@]}"; do
      ckpt_ready "$TR/$run" "$s" || { echo "  产物核实失败: $TR/$run/$s 未落盘完成"; return 1; }
    done
  fi
  return 0
}
pass_marker_ok() { [ -f "$STATE/$1.pass" ] || { echo "  产物核实失败: 缺状态标记 $STATE/$1.pass"; return 1; }; }
grep_marker() {  # $1 步骤名, $2 正则：状态标记里记下的判定行须匹配
  pass_marker_ok "$1" || return 1
  grep -qE "$2" "$STATE/$1.pass" || { echo "  产物核实失败: $STATE/$1.pass 缺判定行 /$2/"; return 1; }
}
verify_step() {  # $1 步骤名
  case "$1" in
    g4_512)     pass_marker_ok g4_512 && verify_run "$SMOKE_512" smoke ;;
    g4_1024)    pass_marker_ok g4_1024 && verify_run "$SMOKE_1024" smoke ;;
    g5)         grep_marker g5 '^FS4X4_G5=PASS' ;;
    train_512)  pass_marker_ok train_512 && verify_run "$FORMAL_512" formal ;;
    train_1024) pass_marker_ok train_1024 && verify_run "$FORMAL_1024" formal ;;
    eval_pairs) grep_marker eval_pairs '^FS4X4_EVAL_PAIRS=PASS done=6' ;;
    summary)    grep_marker summary '^FS4X4_3SEED=PASS rows=2 seeds=3' \
                  && { [ -f "$SUMMARY_JSON" ] || { echo "  产物核实失败: 缺 $SUMMARY_JSON"; return 1; }; } ;;
    *) echo "  未知步骤 $1"; return 1 ;;
  esac
}

# ---- ② 8 卡空闲等待 ----
gpu_wait() {  # $1 = 将要开始的步骤名
  if [ "$DRY" = 1 ]; then
    echo "DRY gpu_wait before=$1: nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits（8 卡都 < ${GPU_LIMIT_MIB} MiB；每 ${GPU_POLL_S}s 一次，至多 ${GPU_POLL_N} 次）"
    return 0
  fi
  local n used ngpu busy
  for n in $(seq 1 "$GPU_POLL_N"); do
    if used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; then
      ngpu="$(printf '%s\n' "$used" | grep -cE '^[[:space:]]*[0-9]+[[:space:]]*$')"
      busy="$(printf '%s\n' "$used" | awk -v L="$GPU_LIMIT_MIB" '$1+0 >= L {c++} END {print c+0}')"
      if [ "$ngpu" = 8 ] && [ "$busy" = 0 ]; then
        echo "FS4X4_GPU_IDLE before=$1 used_mib=$(printf '%s' "$used" | tr -s '[:space:]' ',') $(utc)"
        return 0
      fi
      echo "  GPU 未空闲（before=$1 第 $n/$GPU_POLL_N 次 ngpu=$ngpu busy=$busy used_mib=$(printf '%s' "$used" | tr -s '[:space:]' ',')）"
    else
      echo "  nvidia-smi 失败（before=$1 第 $n/$GPU_POLL_N 次）"
    fi
    [ "$n" -lt "$GPU_POLL_N" ] && sleep "$GPU_POLL_S"
  done
  chain_fail gpu_wait "before=$1 8 卡 ${GPU_POLL_N} 分钟内未全部 < ${GPU_LIMIT_MIB} MiB"
}

# ---- 执行一条命令：dry-run 只打印；否则 tee 一份到步骤输出文件并返回命令自身退出码 ----
OUT=""
run_cmd() {  # $1 = 已转义好的 shell 命令串
  if [ "$DRY" = 1 ]; then echo "DRY cmd: $1"; return 0; fi
  echo "FS4X4_CMD $1"
  bash -c "$1" 2>&1 | tee -a "$OUT"
  return "${PIPESTATUS[0]}"
}

step_begin() {
  echo "FS4X4_STEP=$1 START $(utc)"
  if [ "$DRY" = 0 ]; then
    mkdir -p "$STATE"
    OUT="$STATE/$1.$(date -u +%Y%m%dT%H%M%SZ).out"
    : > "$OUT"
  fi
}
step_end() {  # $1 名, $2 rc, $3 写入状态标记的判定行（可空）
  if [ "$2" = 0 ]; then
    if [ "$DRY" = 1 ]; then echo "FS4X4_STEP=$1 PASS rc=0 dry"; else echo "FS4X4_STEP=$1 PASS rc=0"; fi
    if [ "$DRY" = 0 ]; then
      printf 'step=%s beta=%s utc=%s out=%s\n%s\n' "$1" "$BETA" "$(utc)" "$OUT" "${3:-}" > "$STATE/$1.pass"
    fi
  else
    echo "FS4X4_STEP=$1 FAIL rc=$2"
    chain_fail "$1" "rc=$2${OUT:+ out=$OUT}"
  fi
}

# runner 步骤（g4_* / train_*）
do_train() {  # $1 步骤名, $2 budget, $3 run 名, $4 smoke|formal
  local name="$1" budget="$2" run="$3" kind="$4" cmd rc pass_line
  gpu_wait "$name"
  step_begin "$name"
  cmd="bash $(q "$RUNNER") $(q "$BETA") --budget $budget"
  [ "$kind" = smoke ] && cmd="$cmd --smoke20"
  if run_cmd "$cmd"; then rc=0; else rc=$?; fi
  if [ "$DRY" = 0 ] && [ "$rc" = 0 ]; then
    pass_line="$(grep -E "^FS4X4_TRAIN=PASS run=$run( |$)" "$OUT" | tail -n 1)"
    [ -n "$pass_line" ] || { echo "  runner 输出缺 FS4X4_TRAIN=PASS run=$run"; rc=90; }
    [ "$rc" = 0 ] && { verify_run "$run" "$kind" || rc=91; }
  fi
  [ "$DRY" = 1 ] && echo "DRY verify: $LOGS/$run.log 末行 EXIT_CODE=0；ckpt $( [ "$kind" = smoke ] && echo "$TR/$run/19" || echo "$TR/$run/{5000…50000} 共 10 份")；输出含 FS4X4_TRAIN=PASS run=$run"
  step_end "$name" "$rc" "${pass_line:-}"
}

# 评估类步骤（g5 / eval_pairs / summary）：在 EVAL_WT 内执行，判定行取自输出
do_judged() {  # $1 步骤名, $2 命令, $3 判定正则, $4 gpu|cpu
  local name="$1" cmd="$2" re="$3" pass_line="" rc
  [ "$4" = gpu ] && gpu_wait "$name"
  step_begin "$name"
  if run_cmd "$cmd"; then rc=0; else rc=$?; fi
  if [ "$DRY" = 0 ] && [ "$rc" = 0 ]; then
    pass_line="$(grep -E "$re" "$OUT" | tail -n 1)"
    [ -n "$pass_line" ] || { echo "  输出缺判定行 /$re/"; rc=92; }
  fi
  [ "$DRY" = 1 ] && echo "DRY verify: 输出须含 /$re/"
  step_end "$name" "$rc" "$pass_line"
}

echo "FS4X4_CHAIN_START dry=$DRY beta=$BETA eval_wt=$EVAL_WT from=$FROM store=$STORE_ROOT $(utc)"
echo "FS4X4_CHAIN_PLAN ${STEPS[*]}"

# ① preflight（永远执行）
echo "FS4X4_STEP=preflight START $(utc)"
head="$(git -C "$MAIN" rev-parse HEAD 2>/dev/null)"; head="${head:-unknown}"
dirty="$(git -C "$MAIN" status --porcelain 2>/dev/null | head -n 20)"
clean=yes; [ -z "$dirty" ] || clean=no
wt_ok=yes
for f in "$EVAL_PAIRS" "$SUMMARY_PY"; do [ -f "$EVAL_WT/$f" ] || wt_ok=no; done
[ -d "$EVAL_WT" ] || wt_ok=no
if [ "$DRY" = 1 ]; then
  echo "FS4X4_PREFLIGHT_DRY head=$head head_match=$([ "$head" = "$BETA" ] && echo yes || echo no) clean=$clean eval_wt_ok=$wt_ok（dry-run 只报告不拦）"
  echo "FS4X4_STEP=preflight PASS rc=0 dry"
else
  [ "$head" = "$BETA" ] || { echo "  主检出 HEAD=$head != BETA=$BETA"; echo "FS4X4_STEP=preflight FAIL rc=1"; chain_fail preflight head_mismatch; }
  [ "$clean" = yes ] || { echo "  主检出工作区不干净:"; printf '%s\n' "$dirty"; echo "FS4X4_STEP=preflight FAIL rc=1"; chain_fail preflight dirty; }
  [ "$wt_ok" = yes ] || { echo "  EVAL_WT=$EVAL_WT 不存在或缺 $EVAL_PAIRS / $SUMMARY_PY"; echo "FS4X4_STEP=preflight FAIL rc=1"; chain_fail preflight eval_wt; }
  echo "FS4X4_STEP=preflight PASS rc=0"
fi

# --from：之前的步骤依据产物核实
for i in "${!STEPS[@]}"; do
  [ "$i" -ge 1 ] && [ "$i" -lt "$FROM_IDX" ] || continue
  s="${STEPS[$i]}"
  if verify_step "$s"; then
    echo "FS4X4_STEP=$s SKIP verified $(utc)"
  else
    echo "FS4X4_STEP=$s FAIL rc=1"
    chain_fail "$s" "from=$FROM 之前的步骤产物核实不过"
  fi
done

EVAL_CD="cd $(q "$EVAL_WT") &&"
for i in "${!STEPS[@]}"; do
  [ "$i" -ge 1 ] && [ "$i" -ge "$FROM_IDX" ] || continue
  case "${STEPS[$i]}" in
    g4_512)     do_train g4_512 512 "$SMOKE_512" smoke ;;
    g4_1024)    do_train g4_1024 1024 "$SMOKE_1024" smoke ;;
    g5)         do_judged g5 "$EVAL_CD bash $EVAL_PAIRS --g6b $(q "$TR/$SMOKE_512/19") 512" '^FS4X4_G5=PASS' gpu ;;
    train_512)  do_train train_512 512 "$FORMAL_512" formal ;;
    train_1024) do_train train_1024 1024 "$FORMAL_1024" formal ;;
    eval_pairs)
      # 评估前置：两个正式 run 都 EXIT_CODE=0 且 10 份 ckpt（含 50000）都在
      if [ "$DRY" = 1 ]; then
        echo "DRY precheck eval_pairs: $FORMAL_512 与 $FORMAL_1024 日志 EXIT_CODE=0、ckpt 50000 都在"
      else
        for r in "$FORMAL_512" "$FORMAL_1024"; do
          verify_run "$r" formal || { echo "FS4X4_STEP=eval_pairs FAIL rc=1"; chain_fail eval_pairs "precheck run=$r"; }
        done
      fi
      do_judged eval_pairs "$EVAL_CD bash $EVAL_PAIRS" '^FS4X4_EVAL_PAIRS=PASS done=6' gpu ;;
    summary)
      do_judged summary "$EVAL_CD uv run python $SUMMARY_PY --out $(q "$SUMMARY_JSON")" '^FS4X4_3SEED=PASS rows=2 seeds=3' cpu
      if [ "$DRY" = 0 ] && [ ! -f "$SUMMARY_JSON" ]; then chain_fail summary "缺 $SUMMARY_JSON"; fi ;;
  esac
done

if [ "$DRY" = 1 ]; then echo "FS4X4_CHAIN=DRY_DONE $(utc)"; else echo "FS4X4_CHAIN=DONE $(utc)"; fi
