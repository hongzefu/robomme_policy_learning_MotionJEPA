#!/usr/bin/env bash
# 每个占位（或 aspen 的一张卡）起一个长作业，作业内起 N 条 pool_runner.sh 通道（契约 C4、红线 R1，计划 1116-steer-speedup-fix-plan S-D）。
#
# GL（登录节点上运行）：
#   gl_hold_pool.sh gl --hold <JobID> --channels <N> --mem-frac <比例> --pool-root <O> --stage <池名> [--stage …]
#                      --env-file <env-gl.sh> --log-dir <目录> [--cpus 6] [--stagger 45] [--name <tmux 名>]
#                      [--run-cmd '<命令前缀>'] [--no-tmux] [--dry-run]
#   在 detached tmux 里只起一个作业步：
#     srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task=<cpus> --gpu_cmode=shared bash <本脚本> inner …
#   作业步结束会把 GPU 重置为独占模式（steer-4task-1115 launch.md 第 9 条），所以每个占位只起这一个长作业步，
#   作业步内（inner）起 N 条通道，全部通道退出后作业步才结束；inner 内绝不调用 srun（还把 srun 覆盖成报错函数并导出给通道）。
#
# aspen（本机运行，不经 Slurm）：
#   gl_hold_pool.sh aspen --gpu <卡号> --channels <N> --mem-frac <比例> --pool-root <O> --stage <池名> […]
#                         --env-file <env-aspen.sh> --log-dir <目录> [--stagger 45] [--name …] [--no-tmux] [--dry-run]
#   在 detached tmux 里直接跑 inner（--site aspen --gpu <卡号>）。
#
# inner（不直接手动调用）：依次起 N 条通道，第 i 条在第 i-1 条之后错开 --stagger 秒；
#   通道日志 <log-dir>/<名>-c<i>.log，每条通道一行 POOL_CHANNEL_END rc=…；末行 POOL_HOLD_DONE 与 EXIT_CODE=。
# tmux 会话日志 <log-dir>/<名>.log，末行 EXIT_CODE=（作业步 / inner 的退出码）。
set -o pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
HERE=$(dirname "$SELF")
MODE=${1:-}
[ $# -gt 0 ] && shift

HOLD=""; GPU=""; CHANNELS=""; MEM_FRAC=""; POOL_ROOT=""; ENV_FILE=""; LOG_DIR=""; CPUS=6; STAGGER=45
NAME=""; RUN_CMD=""; USE_TMUX=1; DRY_RUN=0; SITE=""
STAGES=()

die() { echo "GL_HOLD_POOL_ERROR $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --hold) HOLD=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --channels) CHANNELS=$2; shift 2;;
    --mem-frac) MEM_FRAC=$2; shift 2;;
    --pool-root) POOL_ROOT=$2; shift 2;;
    --stage) STAGES+=("$2"); shift 2;;
    --env-file) ENV_FILE=$2; shift 2;;
    --log-dir) LOG_DIR=$2; shift 2;;
    --cpus) CPUS=$2; shift 2;;
    --stagger) STAGGER=$2; shift 2;;
    --name) NAME=$2; shift 2;;
    --run-cmd) RUN_CMD=$2; shift 2;;
    --site) SITE=$2; shift 2;;
    --no-tmux) USE_TMUX=0; shift;;
    --dry-run) DRY_RUN=1; shift;;
    *) die "未知参数 $1";;
  esac
done

common_checks() {
  [[ "$CHANNELS" =~ ^[1-4]$ ]] || die "--channels 须为 1–4（计划 WORKER_FIT 上限 4），得到 '$CHANNELS'"
  [ -n "$MEM_FRAC" ] || die "缺 --mem-frac"
  [ -n "$POOL_ROOT" ] || die "缺 --pool-root"
  [ "${#STAGES[@]}" -gt 0 ] || die "缺 --stage"
  [ -n "$LOG_DIR" ] || die "缺 --log-dir"
  [[ "$STAGGER" =~ ^[0-9]+$ ]] || die "--stagger 须为非负整数秒"
  case "$POOL_ROOT" in /*) ;; *) POOL_ROOT=$PWD/$POOL_ROOT;; esac
  case "$LOG_DIR" in /*) ;; *) LOG_DIR=$PWD/$LOG_DIR;; esac
  if [ -n "$ENV_FILE" ]; then case "$ENV_FILE" in /*) ;; *) ENV_FILE=$PWD/$ENV_FILE;; esac; fi
}

# inner 的参数串（%q 转义），outer 拼进 srun / tmux 命令
inner_args() {
  local site=$1 gpu=$2
  local a=(inner --site "$site" --gpu "$gpu" --channels "$CHANNELS" --mem-frac "$MEM_FRAC"
    --pool-root "$POOL_ROOT" --log-dir "$LOG_DIR" --stagger "$STAGGER" --name "$NAME")
  local st
  for st in "${STAGES[@]}"; do a+=(--stage "$st"); done
  [ -n "$ENV_FILE" ] && a+=(--env-file "$ENV_FILE")
  [ -n "$RUN_CMD" ] && a+=(--run-cmd "$RUN_CMD")
  printf '%q ' bash "$SELF" "${a[@]}"
}

# 起 outer：tmux 里跑 $1（作业步命令或 inner），tee 会话日志并写 EXIT_CODE=
launch() {
  local cmd=$1 log=$LOG_DIR/$NAME.log
  local wrapped="set -o pipefail; $cmd 2>&1 | tee $(printf %q "$log"); echo \"EXIT_CODE=\$?\" >> $(printf %q "$log")"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN name=$NAME log=$log"
    echo "DRY_RUN cmd=$cmd"
    return 0
  fi
  mkdir -p "$LOG_DIR"
  if [ "$USE_TMUX" -eq 1 ]; then
    tmux has-session -t "=$NAME" 2>/dev/null && { echo "SKIP_EXISTS $NAME"; return 0; }
    [ -e "$log" ] && { echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; return 0; }
    tmux new-session -d -s "$NAME" "$wrapped" || die "tmux 启动失败 $NAME"
    echo "STARTED $NAME log=$log"
  else
    [ -e "$log" ] && die "日志已存在 $log（重跑前先改名留档）"
    bash -c "$wrapped"
    tail -n 1 "$log"
  fi
}

# BEGIN_INNER —— 作业步内运行的部分：只起通道、等待，绝不调用 srun（测试静态检查本段）
run_inner() {
  srun() { echo "GL_HOLD_POOL_ERROR 作业步内禁止再起作业步（R1）" >&2; return 3; }
  export -f srun
  local runner=$HERE/pool_runner.sh i pids=() rcs=() failed=0
  [ -f "$runner" ] || die "找不到 $runner"
  mkdir -p "$LOG_DIR"
  echo "POOL_HOLD_BEGIN name=$NAME site=$SITE gpu=$GPU channels=$CHANNELS mem_frac=$MEM_FRAC stages=${STAGES[*]} host=${HOSTNAME:-} slurm_job=${SLURM_JOB_ID:-none} slurm_step=${SLURM_STEP_ID:-none}"
  on_term() {
    echo "POOL_HOLD_INTERRUPTED name=$NAME，转发 TERM 给通道 ${pids[*]}"
    [ "${#pids[@]}" -gt 0 ] && kill -TERM "${pids[@]}" 2>/dev/null
    wait
    echo "EXIT_CODE=143"
    exit 143
  }
  trap on_term INT TERM
  for ((i = 0; i < CHANNELS; i++)); do
    if [ "$i" -gt 0 ] && [ "$STAGGER" -gt 0 ]; then sleep "$STAGGER"; fi
    local a=(--pool-root "$POOL_ROOT" --site "$SITE" --gpu "$GPU" --mem-frac "$MEM_FRAC" --channel "$NAME-c$i") st
    for st in "${STAGES[@]}"; do a+=(--stage "$st"); done
    [ -n "$ENV_FILE" ] && a+=(--env-file "$ENV_FILE")
    [ -n "$RUN_CMD" ] && a+=(--run-cmd "$RUN_CMD")
    bash "$runner" "${a[@]}" > "$LOG_DIR/$NAME-c$i.log" 2>&1 < /dev/null &
    pids+=($!)
    echo "POOL_CHANNEL_START name=$NAME channel=$i pid=$! log=$LOG_DIR/$NAME-c$i.log"
  done
  for ((i = 0; i < CHANNELS; i++)); do
    wait "${pids[$i]}"
    rcs[$i]=$?
    [ "${rcs[$i]}" -ne 0 ] && failed=$((failed+1))
    echo "POOL_CHANNEL_END name=$NAME channel=$i rc=${rcs[$i]}"
  done
  echo "POOL_HOLD_DONE name=$NAME channels=$CHANNELS failed=$failed"
  [ "$failed" -eq 0 ] && return 0 || return 1
}
# END_INNER

case "$MODE" in
  gl)
    common_checks
    [[ "$HOLD" =~ ^[0-9]+$ ]] || die "--hold 须为数字 JobID"
    [[ "$CPUS" =~ ^[0-9]+$ ]] || die "--cpus 须为正整数"
    NAME=${NAME:-s5t-pool-$HOLD}
    # 占位是单卡，作业步内卡号恒为 0
    step="srun --jobid=$HOLD --overlap --exact --ntasks=1 --cpus-per-task=$CPUS --gpu_cmode=shared $(inner_args gl 0)"
    launch "$step"
    ;;
  aspen)
    common_checks
    [[ "$GPU" =~ ^[0-9]+$ ]] || die "aspen 须给 --gpu <卡号>"
    NAME=${NAME:-s5t-pool-aspen-g$GPU}
    launch "$(inner_args aspen "$GPU")"
    ;;
  inner)
    common_checks
    case "$SITE" in gl|aspen) ;; *) die "inner 须给 --site gl|aspen";; esac
    [ -n "$GPU" ] || die "inner 须给 --gpu"
    [ -n "$NAME" ] || die "inner 须给 --name"
    run_inner
    exit $?
    ;;
  *)
    sed -n '2,22p' "$SELF"
    die "第一个参数须为 gl | aspen | inner"
    ;;
esac
