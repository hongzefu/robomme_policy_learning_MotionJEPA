#!/usr/bin/env bash
# NativeMEM 结束触发（1117-e2e-enc-train-plan.html 第二部分〇.1.1、RB-A4；S5a）。
#
# 用法（RB-A4，主检出根目录下、detached tmux e2e-trig 内）：
#   bash scripts/training/e2e_wait_nm.sh --nm-log /scratch/hongze/NativeMEM/logs/nm-pipe2.log --nm-tmux nm-pipe2
# 选项：
#   --nm-log <路径>       NativeMEM 流水线日志（只读）
#   --nm-tmux <会话名>    NativeMEM 流水线 tmux 会话（只读，精确匹配 has-session -t '=名'）
#   --dry-run             判一次 NM_DONE，列出闸门表里将起的会话与命令（TRIG_DRYRUN），不建锁、不起会话
#   --gates-file <文件>   闸门表（bash 文件，被 source，内部调用 gate_def）；缺省用本脚本内 default_gates
#   --lock-dir <目录>     启动锁与闸门启动脚本目录，缺省 <仓库根>/v1-store/logs（必须已存在，否则拒绝）
#   --interval <秒>       独立定时重查周期，缺省 60（环境变量 E2E_TRIG_INTERVAL 同义，测试用）
#   --max-rounds <n>      最多重查 n 轮后退出（0=不限，缺省；E2E_TRIG_MAX_ROUNDS 同义，测试用）
#
# NM_DONE=PASS 五条同时满足（〇.1.1）：① 日志有 ^EVAL692=PASS episodes=692（前缀）；② 日志有 ^NM_PIPELINE=PASS；
#   ③ 日志末行整行匹配 ^EXIT_CODE=[0-9]+$ 且为 0（SHARD_EXIT_CODE= / TRAIN_EXIT_CODE= 不算）；
#   ④ tmux has-session -t '=<nm-tmux>' 失败；⑤ 8 卡 memory.used 之和恰为 0（与 c0826be
#   run_4task_wanraw9s32_50k.sh:298 同一条命令）。满足打印 NM_DONE=PASS 与 A0 的 GPU_FREE=PASS gpus=8。
#   NM_PIPELINE=FAIL 或末行 EXIT_CODE 非 0 → NM_DONE=FAIL reason=…，不开跑，退出码 3。
# 之后按闸门表次序串行：每轮最多起一个闸门；前一闸门判定为 PASS/RECORDED/ACCEPTED_OVER_LIMIT（R10）且所有
#   闸门会话都已退出、GPU 之和为 0 才起下一个；闸门 FAIL/ASK → TRIG_STOP，退出码 4。
# 启动锁 <lock-dir>/e2e-trig.<gate>.launched（set -o noclobber 原子创建，含 attempt id）；锁在即 TRIG_SKIP，不重复启动；
#   触发器从不移动/删除锁（重做由主会话改名为 ….launched.superseded-aNN 并递增 attempt）。
# 对 NativeMEM 只读；本脚本只调用 tmux has-session / new-session，绝不调用任何 kill-* 子命令。
# 不起 G7/G8（RB-A4）：缺省闸门表止于 G5b，全部通过后打印 TRIG_DONE 并退出 0。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

NM_LOG=/scratch/hongze/NativeMEM/logs/nm-pipe2.log
NM_TMUX=nm-pipe2
DRY_RUN=0
GATES_FILE=""
LOCK_DIR="${E2E_TRIG_LOCK_DIR:-$REPO/v1-store/logs}"
INTERVAL="${E2E_TRIG_INTERVAL:-60}"
MAX_ROUNDS="${E2E_TRIG_MAX_ROUNDS:-0}"

die() { printf '拒绝：%s\n' "$*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --nm-log) NM_LOG="${2:?}"; shift 2 ;;
    --nm-tmux) NM_TMUX="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --gates-file) GATES_FILE="${2:?}"; shift 2 ;;
    --lock-dir) LOCK_DIR="${2:?}"; shift 2 ;;
    --interval) INTERVAL="${2:?}"; shift 2 ;;
    --max-rounds) MAX_ROUNDS="${2:?}"; shift 2 ;;
    *) die "未知参数 $1" ;;
  esac
done
[[ "$NM_TMUX" =~ ^[A-Za-z0-9._-]+$ ]] || die "nm-tmux 名非法：$NM_TMUX"
[[ "$INTERVAL" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "interval 非法：$INTERVAL"
[[ "$MAX_ROUNDS" =~ ^[0-9]+$ ]] || die "max-rounds 非法：$MAX_ROUNDS"
[ -d "$LOCK_DIR" ] || die "找不到锁目录 $LOCK_DIR（在 worktree 里运行时请显式 --lock-dir 绝对路径）"

ts_et() { TZ=America/New_York date +%H:%M:%S; }
ts_et_full() { TZ=America/New_York date '+%F %T %Z'; }

# ---------- 闸门表 ----------
GATE_NAMES=(); declare -A G_ATTEMPT G_SESSION G_LOG G_PREREQ G_LAUNCH G_VERDICT
# gate_def <gate> <attempt aNN> <tmux 会话> <日志绝对路径> <前置检查命令> <启动命令> <判定命令>
#   前置检查命令：退出 0 = 就绪；非 0 = 未就绪，stdout 首行作 missing 项（如 sub/S3 g0）。
#   启动命令：在 tmux 内经日志三件套包装运行（tee 到 <日志>，尾行 EXIT_CODE=）。
#   判定命令：stdout 末个词为 PASS|RECORDED|ACCEPTED_OVER_LIMIT|ASK|FAIL 之一，其余视为 PENDING。
#   三条命令都以 bash -c 执行，可用环境变量 GATE GATE_ATTEMPT GATE_SESSION GATE_LOG E2E_REPO E2E_LOCK_DIR
#   与下方导出的帮助函数 trig_verdict_all / trig_need_sub / trig_file_has。
gate_def() {
  [ $# -eq 7 ] || die "gate_def 需要 7 个参数：$*"
  local g="$1"
  [[ "$g" =~ ^[A-Za-z0-9._-]+$ ]] || die "gate 名非法：$g"
  [[ "$2" =~ ^a[0-9]{2}$ ]] || die "attempt 非法（应为 aNN）：$2"
  [[ "$3" =~ ^[A-Za-z0-9._-]+$ ]] || die "会话名非法：$3"
  [ -z "${G_SESSION[$g]+x}" ] || die "gate 重复定义：$g"
  GATE_NAMES+=("$g"); G_ATTEMPT[$g]="$2"; G_SESSION[$g]="$3"; G_LOG[$g]="$4"
  G_PREREQ[$g]="$5"; G_LAUNCH[$g]="$6"; G_VERDICT[$g]="$7"
}

# 缺省闸门表：推荐次序 G1/G1b/G1c → G3 → G4 → G6(bench→shmprobe→ckptprobe) → G5 → G5b（〇.1.1）。
# 启动命令与前置检查由主会话在 S1–S5 合入后填写（--gates-file 覆盖）；未填写的闸门前置恒为未就绪。
default_gates() {
  local L="$LOCK_DIR" g
  local nc='echo not_configured; exit 1'
  for g in G1 G1b G1c G3 G4; do
    local lc; lc="$(printf '%s' "$g" | tr 'A-Z' 'a-z')"
    gate_def "$g" a01 "e2e-$lc-a01" "$L/e2e-$lc-a01.log" "$nc" 'false' 'echo PENDING'
  done
  gate_def G6-bench     a01 e2e-bench-a01     "$L/e2e-bench-a01.log"     "$nc" 'false' 'echo PENDING'
  gate_def G6-shmprobe  a01 e2e-shmprobe-a01  "$L/e2e-shmprobe-a01.log"  "$nc" 'false' 'echo PENDING'
  gate_def G6-ckptprobe a01 e2e-ckptprobe-a01 "$L/e2e-ckptprobe-a01.log" "$nc" 'false' 'echo PENDING'
  gate_def G5  a01 e2e-g5-a01  "$L/e2e-g5-a01.log"  "$nc" 'false' 'echo PENDING'
  gate_def G5b a01 e2e-g5b-a01 "$L/e2e-g5b-a01.log" "$nc" 'false' 'echo PENDING'
}

# ---------- 供闸门命令使用的帮助函数（export -f） ----------
# trig_verdict_all <log> KEY...：每个 KEY 取日志中最后一行 ^KEY=<状态词>；任一 FAIL→FAIL，任一 ASK→ASK，
#   任一缺失→PENDING，否则 PASS（RECORDED / ACCEPTED_OVER_LIMIT 视同通过，R10）。
trig_verdict_all() {
  local log="$1"; shift
  [ -f "$log" ] || { echo PENDING; return 0; }
  local k st worst=PASS
  for k in "$@"; do
    st="$(grep -aE "^${k}=(PASS|RECORDED|ACCEPTED_OVER_LIMIT|ASK|FAIL)([[:space:]]|$)" "$log" | tail -n 1 \
          | sed -E "s/^${k}=([A-Z_]+).*/\\1/")"
    case "$st" in
      FAIL) echo FAIL; return 0 ;;
      ASK) worst=ASK ;;
      PASS|RECORDED|ACCEPTED_OVER_LIMIT) ;;
      *) [ "$worst" = ASK ] || worst=PENDING ;;
    esac
  done
  echo "$worst"
}
# trig_need_sub S3 [S5 ...]：工作分支 HEAD 可达历史里有 subject 以 "sub/S<n>:" 开头的提交才算合入；缺的打印出来并返回 1。
trig_need_sub() {
  local s miss=()
  for s in "$@"; do
    git -C "$E2E_REPO" log --format=%s HEAD 2>/dev/null | grep -qE "^sub/${s}[a-z]?: " || miss+=("sub/$s")
  done
  [ ${#miss[@]} -eq 0 ] || { echo "${miss[*]}"; return 1; }
}
# trig_file_has <文件> <ERE>：文件存在且有匹配行。
trig_file_has() { [ -f "$1" ] && grep -aqE "$2" "$1"; }
export -f trig_verdict_all trig_need_sub trig_file_has

if [ -n "$GATES_FILE" ]; then
  [ -f "$GATES_FILE" ] || die "找不到闸门表 $GATES_FILE"
  # shellcheck disable=SC1090
  source "$GATES_FILE"
else
  default_gates
fi
[ ${#GATE_NAMES[@]} -gt 0 ] || die "闸门表为空"
GATES_SHA="$(for g in "${GATE_NAMES[@]}"; do
  printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0' "$g" "${G_ATTEMPT[$g]}" "${G_SESSION[$g]}" "${G_LOG[$g]}" \
    "${G_PREREQ[$g]}" "${G_LAUNCH[$g]}" "${G_VERDICT[$g]}"; done | sha256sum | cut -c1-16)"
export E2E_REPO="$REPO" E2E_LOCK_DIR="$LOCK_DIR"

# ---------- 资源判据 ----------
# ⑤ 与 runner :298 同一条命令；输出恰为 0 才算空闲。查询失败返回字符串 err。
gpu_used_sum() {
  local s
  s="$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null)" || s=""
  [[ "$s" =~ ^[0-9]+$ ]] && echo "$s" || echo err
}
session_exists() { tmux has-session -t "=$1" >/dev/null 2>&1; }

gate_env_run() {  # gate_env_run <gate> <命令>：在闸门环境里 bash -c 执行
  local g="$1"
  GATE="$g" GATE_ATTEMPT="${G_ATTEMPT[$g]}" GATE_SESSION="${G_SESSION[$g]}" GATE_LOG="${G_LOG[$g]}" \
    bash -c "$2"
}
gate_verdict() {
  local w
  w="$(gate_env_run "$1" "${G_VERDICT[$1]}" 2>/dev/null | awk 'NF{last=$NF} END{print last}')"
  case "$w" in PASS|RECORDED|ACCEPTED_OVER_LIMIT|ASK|FAIL) echo "$w" ;; *) echo PENDING ;; esac
}
verdict_ok() { case "$1" in PASS|RECORDED|ACCEPTED_OVER_LIMIT) return 0 ;; *) return 1 ;; esac; }
lock_path() { echo "$LOCK_DIR/e2e-trig.$1.launched"; }

# ---------- NM_DONE 判定 ----------
# 输出：PASS / FAIL reason=… / WAIT missing=…
nm_status() {
  local miss=() last
  if [ ! -f "$NM_LOG" ]; then echo "WAIT missing=nm_log_absent"; return; fi
  if grep -aqE '^NM_PIPELINE=FAIL' "$NM_LOG"; then echo "FAIL reason=NM_PIPELINE=FAIL"; return; fi
  last="$(tail -n 1 "$NM_LOG" | tr -d '\r')"
  if [[ "$last" =~ ^EXIT_CODE=[0-9]+$ ]] && [ "$last" != "EXIT_CODE=0" ]; then
    echo "FAIL reason=last_line_${last}"; return
  fi
  grep -aqE '^EVAL692=PASS episodes=692([[:space:]]|$)' "$NM_LOG" || miss+=(eval692)
  grep -aqE '^NM_PIPELINE=PASS([[:space:]]|$)' "$NM_LOG" || miss+=(nm_pipeline)
  [ "$last" = "EXIT_CODE=0" ] || miss+=(exit_code_last_line)
  session_exists "$NM_TMUX" && miss+=("tmux_${NM_TMUX}_alive")
  local used; used="$(gpu_used_sum)"
  [ "$used" = 0 ] || miss+=("gpu_busy_${used}")
  if [ ${#miss[@]} -eq 0 ]; then echo PASS; else local IFS=,; echo "WAIT missing=${miss[*]}"; fi
}

# ---------- dry-run ----------
if [ "$DRY_RUN" = 1 ]; then
  st="$(nm_status)"
  case "$st" in
    PASS) echo "NM_DONE=PASS"; echo "GPU_FREE=PASS gpus=8" ;;
    FAIL*) echo "NM_DONE=FAIL ${st#FAIL }" ;;
    *) echo "TRIG_WAIT gate=A0 ${st#WAIT }" ;;
  esac
  for g in "${GATE_NAMES[@]}"; do
    pre="ready"; out="$(gate_env_run "$g" "${G_PREREQ[$g]}" 2>/dev/null)" || pre="missing=$(printf '%s' "$out" | head -n 1)"
    lk=none; [ -e "$(lock_path "$g")" ] && lk=locked
    printf 'TRIG_DRYRUN gate=%s tmux=%s attempt=%s log=%s prereq=%s lock=%s verdict=%s cmd=%q\n' \
      "$g" "${G_SESSION[$g]}" "${G_ATTEMPT[$g]}" "${G_LOG[$g]}" "$pre" "$lk" "$(gate_verdict "$g")" "${G_LAUNCH[$g]}"
  done
  echo "TRIG_DRYRUN_END gates=${#GATE_NAMES[@]} gates_sha=$GATES_SHA launched=0"
  exit 0
fi

# ---------- 启动 ----------
launch_gate() {  # 已通过资源重查后调用；返回 0=已启动，1=锁已存在
  local g="$1" lk wrap att="${G_ATTEMPT[$1]}" s="${G_SESSION[$1]}" log="${G_LOG[$1]}"
  lk="$(lock_path "$g")"
  if ! ( set -o noclobber; printf 'attempt=%s\ntmux=%s\nlog=%s\ngates_sha=%s\nlaunched_et=%s\n' \
           "$att" "$s" "$log" "$GATES_SHA" "$(ts_et_full)" > "$lk" ) 2>/dev/null; then
    return 1
  fi
  wrap="$LOCK_DIR/e2e-trig.$g.$att.launch.sh"
  {
    printf '#!/usr/bin/env bash\n# e2e_wait_nm.sh 生成：gate=%s attempt=%s\n' "$g" "$att"
    printf 'export GATE=%q GATE_ATTEMPT=%q GATE_SESSION=%q GATE_LOG=%q E2E_REPO=%q E2E_LOCK_DIR=%q\n' \
      "$g" "$att" "$s" "$log" "$REPO" "$LOCK_DIR"
    printf 'cd %q\n' "$REPO"
    printf 'set -o pipefail\n'
    printf 'PYTHONUNBUFFERED=1 bash -c %q 2>&1 | tee %q\n' "${G_LAUNCH[$g]}" "$log"
    printf 'echo "EXIT_CODE=$?" >> %q\n' "$log"
  } > "$wrap"
  if ! tmux new-session -d -s "$s" "bash $(printf '%q' "$wrap")"; then
    echo "TRIG_STOP gate=$g reason=tmux_new_session_failed attempt=$att 失败"
    exit 4
  fi
  echo "TRIG_LAUNCH gate=$g tmux=$s session=$s attempt=$att log=$log lock=$lk"
  return 0
}

declare -A SKIP_SHOWN GONE_ROUNDS
LAST_WAIT=""
say_wait() { [ "$1" = "$LAST_WAIT" ] || { echo "$1"; LAST_WAIT="$1"; }; }

NM_DONE=0
round=0
echo "TRIG_START ts=$(ts_et) nm_log=$NM_LOG nm_tmux=$NM_TMUX lock_dir=$LOCK_DIR interval=${INTERVAL}s gates=${GATE_NAMES[*]} gates_sha=$GATES_SHA"

# tail -n +1 -F 从头跟踪 NM 日志：新行只用于提前唤醒；判定一律在每轮重查里按整份日志做（不依赖日志增长）。
exec 3< <(exec tail -n +1 -F "$NM_LOG" 2>/dev/null)
TAIL_PID=$!
cleanup() { [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null; exec 3<&- 2>/dev/null; }
trap cleanup EXIT
wait_tick() {  # 等一个周期；NM 未结束时 NM 日志出新行可提前唤醒
  if [ "$NM_DONE" = 0 ] && [ -n "${TAIL_PID:-}" ]; then
    local line
    if IFS= read -r -t "$INTERVAL" -u 3 line; then
      while IFS= read -r -t 0.05 -u 3 line; do :; done   # 排空积压
      return
    fi
    # read 超时或 EOF；EOF 时退化为纯定时
    kill -0 "$TAIL_PID" 2>/dev/null && return
    TAIL_PID=""
  fi
  sleep "$INTERVAL"
}

while :; do
  round=$((round + 1))
  if [ "$NM_DONE" = 0 ]; then
    st="$(nm_status)"
    case "$st" in
      PASS)
        NM_DONE=1
        echo "NM_DONE=PASS"
        echo "GPU_FREE=PASS gpus=8"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/GPU_APPS /'
        kill "$TAIL_PID" 2>/dev/null; TAIL_PID="" ;;
      FAIL*)
        echo "NM_DONE=FAIL ${st#FAIL }"
        echo "TRIG_STOP gate=A0 reason=nm_failed 不自动开跑，交主会话报告用户"
        exit 3 ;;
      *) say_wait "TRIG_WAIT gate=A0 ${st#WAIT }" ;;
    esac
  fi

  pending=()
  if [ "$NM_DONE" = 1 ]; then
    all_ok=1
    for g in "${GATE_NAMES[@]}"; do
      v="$(gate_verdict "$g")"
      if verdict_ok "$v"; then continue; fi
      all_ok=0
      if [ "$v" = FAIL ] || [ "$v" = ASK ]; then
        echo "TRIG_STOP gate=$g verdict=$v attempt=${G_ATTEMPT[$g]} 停止后续启动，交主会话按 R4 处理"
        exit 4
      fi
      pending+=("$g")
      lk="$(lock_path "$g")"
      if [ -e "$lk" ]; then
        if [ -z "${SKIP_SHOWN[$g]+x}" ]; then
          echo "TRIG_SKIP gate=$g locked=$(sed -n 's/^attempt=//p' "$lk" | head -n 1) tmux=${G_SESSION[$g]}"
          SKIP_SHOWN[$g]=1
        fi
        if session_exists "${G_SESSION[$g]}"; then
          GONE_ROUNDS[$g]=0
        else
          GONE_ROUNDS[$g]=$(( ${GONE_ROUNDS[$g]:-0} + 1 ))
          if [ "${GONE_ROUNDS[$g]}" -ge 3 ]; then
            echo "TRIG_STOP gate=$g reason=session_gone_without_verdict attempt=${G_ATTEMPT[$g]} 失败"
            exit 4
          fi
        fi
        break
      fi
      if ! miss="$(gate_env_run "$g" "${G_PREREQ[$g]}" 2>/dev/null)"; then
        say_wait "TRIG_WAIT gate=$g missing=$(printf '%s' "${miss:-unknown}" | head -n 1 | tr ' ' ',')"
        break
      fi
      # 真正启动前重查资源：GPU 之和为 0，且闸门表内所有会话（含本闸门）都不存在
      busy=""
      for h in "${GATE_NAMES[@]}"; do
        session_exists "${G_SESSION[$h]}" && { busy="session_alive:${G_SESSION[$h]}"; break; }
      done
      used="$(gpu_used_sum)"
      [ -n "$busy" ] || [ "$used" = 0 ] || busy="gpu_busy"
      if [ -n "$busy" ]; then say_wait "TRIG_WAIT gate=$g missing=$busy gpu_used_mib=$used"; break; fi
      if [ -e "${G_LOG[$g]}" ]; then
        echo "TRIG_STOP gate=$g reason=log_exists_without_lock log=${G_LOG[$g]} 拒绝覆盖，主会话需递增 attempt"
        exit 4
      fi
      if launch_gate "$g"; then LAST_WAIT=""; else
        echo "TRIG_SKIP gate=$g locked=race"; SKIP_SHOWN[$g]=1
      fi
      break
    done
    if [ "$all_ok" = 1 ]; then
      echo "TRIG_DONE gates=${#GATE_NAMES[@]} all=PASS 触发器不起 G7/G8（RB-A4），交主会话"
      exit 0
    fi
  fi

  echo "TRIG_RECHECK ts=$(ts_et) gpu_used_mib=$(gpu_used_sum) pending=$( [ ${#pending[@]} -gt 0 ] && IFS=,; echo "${pending[*]:-A0}")"
  if [ "$MAX_ROUNDS" -gt 0 ] && [ "$round" -ge "$MAX_ROUNDS" ]; then
    echo "TRIG_MAX_ROUNDS rounds=$round"
    exit 0
  fi
  wait_tick
done
