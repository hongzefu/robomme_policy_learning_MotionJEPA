#!/usr/bin/env bash
# fs8 链式脚本（1109-ctx-framesamp1024-8gpu-b64-80k-plan.html §4、§8 D1–D6、第二部分 一/二/三）。
# 在 tmux fs8-chain 里运行，全部输出走 stdout，由外层 tee 到 v1-store/logs/fs8-chain.log：
#   a) 等 w768 评估日志出现 ^EVAL692=（PASS/FAIL 都继续，D1）
#   b) 等 tmux 会话 =w768-eval 退出
#   c) 等 8 卡空闲（memory.used<2000 且无计算进程，连续 3 次间隔 20 s）；
#      评估会话退出 60 分钟后仍不空，只对 cmdline 带 w768 评估特征的进程 TERM→KILL（D2），其余不碰
#   d) 等 /scratch 剩余 >= 150 GB（D6）
#   e) FS8_TRIGGER=PASS
#   f) 前台冒烟 fs8 smoke（失败 FS8_STOP reason=smoke_fail，退出 3；D4 由主会话处理）
#   g) tmux fs8-train 起正式 fs8 prod，tmux fs8-inc 起增量上传；盯到第一份 checkpoint 10000 出现退出 0；
#      10000 之前异常退出：RUN_SUFFIX 为空则停 fs8-inc、以 -r2 重起一次（D5），否则 FS8_STOP reason=prod_exit 退出 4
#
# 必需：CODE_ROOT（只读副本）、PY（训练解释器）
# 可覆盖（括号内为默认）：
#   EVAL_LOG（$PROJECT/v1-store/logs/w768-eval.log）  EVAL_SESSION（w768-eval）
#   STORE（$PROJECT/v1-store）  RUN_SCRIPT（$CODE_ROOT/scripts/training/run_ctx1024_b64_80k.sh）
#   INC_PY（$PY；上传脚本只用标准库，hf CLI 经 uvx）  RUN_SUFFIX（空）
#   NVIDIA_SMI（nvidia-smi）  TMUX_BIN（tmux）  DF_PATH（/scratch）  MIN_FREE_GB（150）
#   FS8_POLL（20，GPU/会话/日志轮询秒数）  FS8_IDLE_N（3）  FS8_ORPHAN_AFTER（3600）
#   FS8_REPORT_EVERY（600，FS8_GPU_WAIT/FS8_DISK_WAIT 间隔）  FS8_TERM_GRACE（30）
#   孤儿特征（只看 /proc/<pid>/cmdline，不看 cwd、不看父进程链）：
#     ORPHAN_A1 且 ORPHAN_A2（serve_policy.py + --policy.dir=<w768 50000 ckpt>）
#     ORPHAN_B1 且 ORPHAN_B2（motion_sidecar.py + wan_fixedpool768）
#     ORPHAN_C1 或 ORPHAN_C2（评估根目录 local692-wantp768-50k[-retry1] 或 examples/robomme/eval.py）
# tmux 只用 new-session -d -s <确切名> / has-session -t '=确切名' / kill-session -t '=fs8-inc'。
set -uo pipefail

: "${CODE_ROOT:?必须指定 CODE_ROOT}"
: "${PY:?必须指定 PY}"
PROJECT=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE=${STORE:-$PROJECT/v1-store}
EVAL_LOG=${EVAL_LOG:-$PROJECT/v1-store/logs/w768-eval.log}
EVAL_SESSION=${EVAL_SESSION:-w768-eval}
RUN_SCRIPT=${RUN_SCRIPT:-$CODE_ROOT/scripts/training/run_ctx1024_b64_80k.sh}
INC_PY=${INC_PY:-$PY}
RUN_SUFFIX=${RUN_SUFFIX:-}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
TMUX_BIN=${TMUX_BIN:-tmux}
DF_PATH=${DF_PATH:-/scratch}
MIN_FREE_GB=${MIN_FREE_GB:-150}
FS8_POLL=${FS8_POLL:-20}
FS8_IDLE_N=${FS8_IDLE_N:-3}
FS8_ORPHAN_AFTER=${FS8_ORPHAN_AFTER:-3600}
FS8_REPORT_EVERY=${FS8_REPORT_EVERY:-600}
FS8_TERM_GRACE=${FS8_TERM_GRACE:-30}
ORPHAN_A1=${ORPHAN_A1:-serve_policy.py}
ORPHAN_A2=${ORPHAN_A2:---policy.dir=$PROJECT/v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-wantp768-b128-50k/50000}
ORPHAN_B1=${ORPHAN_B1:-motion_sidecar.py}
ORPHAN_B2=${ORPHAN_B2:-wan_fixedpool768}
ORPHAN_C1=${ORPHAN_C1:-$PROJECT/v1-store/evaluation/local692-wantp768-50k}
ORPHAN_C2=${ORPHAN_C2:-examples/robomme/eval.py}

BASE0=orig100-ctx-framesamp1024-b64-80k-8gpu
BUCKET0=HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-8gpu-v1
FIRST_CKPT_STEP=10000

ts() { date -u +%FT%TZ; }
q() { printf '%q' "$1"; }
has_session() { "$TMUX_BIN" has-session -t "=$1" 2>/dev/null; }

# ── a) 等 EVAL692= ────────────────────────────────────────────────────────────
echo "FS8_WAIT_START eval_log=$EVAL_LOG session=$EVAL_SESSION code_root=$CODE_ROOT py=$PY $(ts)"
exec {EFD}< <(exec tail -n +1 -F "$EVAL_LOG" 2>/dev/null)
TAIL_PID=$!
EVAL_LINE=$(grep --line-buffered -m1 '^EVAL692=' <&"$EFD")
kill "$TAIL_PID" 2>/dev/null; wait "$TAIL_PID" 2>/dev/null; exec {EFD}<&-
if [[ $EVAL_LINE == EVAL692=PASS* ]]; then EVAL_RESULT=PASS; else EVAL_RESULT=FAIL; fi
echo "FS8_EVAL_RESULT=$EVAL_RESULT line=$EVAL_LINE $(ts)"

# ── b) 等评估会话退出 ─────────────────────────────────────────────────────────
while has_session "$EVAL_SESSION"; do sleep "$FS8_POLL"; done
EVAL_GONE=$(date +%s)
echo "FS8_EVAL_SESSION_GONE session=$EVAL_SESSION $(ts)"

# ── c) 等 8 卡空闲（D2 孤儿清理）──────────────────────────────────────────────
gpu_idle() {  # 输出状态串；空闲返回 0
  local mem apps n busy
  mem=$("$NVIDIA_SMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || { GPU_STATE="smi_fail"; APPS=""; return 1; }
  apps=$("$NVIDIA_SMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { GPU_STATE="smi_fail"; APPS=""; return 1; }
  n=$(grep -c '[0-9]' <<<"$mem")
  busy=$(awk '$1+0 >= 2000' <<<"$mem" | grep -c .)
  APPS=$(tr -d ' ' <<<"$apps" | grep -E '^[0-9]+$' | tr '\n' ' ')
  GPU_STATE="gpus=$n busy_mem=$busy apps=[${APPS% }]"
  [[ $n == 8 && $busy == 0 && -z ${APPS// /} ]]
}
cmdline_of() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null; }
is_orphan() {
  local c=$1
  [[ -n $c ]] || return 1
  { [[ $c == *"$ORPHAN_A1"* && $c == *"$ORPHAN_A2"* ]]; } && return 0
  { [[ $c == *"$ORPHAN_B1"* && $c == *"$ORPHAN_B2"* ]]; } && return 0
  [[ $c == *"$ORPHAN_C1"* || $c == *"$ORPHAN_C2"* ]]
}
orphan_sweep() {
  local pid c i
  for pid in $APPS; do
    c=$(cmdline_of "$pid")
    is_orphan "$c" || continue
    echo "FS8_ORPHAN_TERM pid=$pid cmd=$c $(ts)"
    kill -TERM "$pid" 2>/dev/null
    for ((i = 0; i < FS8_TERM_GRACE; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    # 发 KILL 前复核 cmdline，防 PID 复用
    if kill -0 "$pid" 2>/dev/null && is_orphan "$(cmdline_of "$pid")"; then
      echo "FS8_ORPHAN_KILL pid=$pid $(ts)"
      kill -KILL "$pid" 2>/dev/null
    fi
  done
}
APPS=""; ok=0; last_report=$(date +%s)
while ((ok < FS8_IDLE_N)); do
  if gpu_idle; then
    ok=$((ok + 1))
  else
    ok=0
    now=$(date +%s)
    if ((now - EVAL_GONE >= FS8_ORPHAN_AFTER)); then orphan_sweep; fi
    if ((now - last_report >= FS8_REPORT_EVERY)); then
      echo "FS8_GPU_WAIT $GPU_STATE since_eval_gone_s=$((now - EVAL_GONE)) $(ts)"; last_report=$now
    fi
  fi
  ((ok < FS8_IDLE_N)) && sleep "$FS8_POLL"
done
echo "FS8_GPU_IDLE $GPU_STATE checks=$FS8_IDLE_N $(ts)"

# ── d) 磁盘（D6）────────────────────────────────────────────────────────────
free_gb() { df -BG --output=avail "$DF_PATH" | tail -1 | tr -dc '0-9'; }
FREE=$(free_gb)
while [[ -z $FREE ]] || ((FREE < MIN_FREE_GB)); do
  echo "FS8_DISK_WAIT free_gb=${FREE:-?} need=$MIN_FREE_GB $(ts)"
  sleep "$FS8_REPORT_EVERY"; FREE=$(free_gb)
done

# ── e) 触发 ─────────────────────────────────────────────────────────────────
echo "FS8_TRIGGER=PASS eval=$EVAL_RESULT gpus_idle=8 free_gb=$FREE $(ts)"

# ── f) 冒烟（前台；完整输出在冒烟日志里）──────────────────────────────────────
SMOKE_LOG=$STORE/logs/$BASE0-smoke.log
echo "FS8_SMOKE_START log=$SMOKE_LOG $(ts)"
CODE_ROOT=$CODE_ROOT PY=$PY RUN_SUFFIX= bash "$RUN_SCRIPT" fs8 smoke >/dev/null
SMOKE_RC=$?
SMOKE_TAIL=$(grep -a '^EXIT_CODE=' "$SMOKE_LOG" 2>/dev/null | tail -1)
MESH=$(grep -a -o 'TRAIN_MESH shape=([0-9]*, [0-9]*)' "$SMOKE_LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*, [0-9]*' | tr -d ' ' | tr ',' 'x')
# 步速取最后一个未到 20 的 Progress 行（20/20 那行混入了 checkpoint 保存时间）
RATE=$(tr '\r' '\n' <"$SMOKE_LOG" 2>/dev/null | grep -a 'Progress on:' | grep -av ' 20.0it/20.0it' | tail -1 | grep -o 'rate:[^ ]*' | cut -d: -f2)
if [[ $SMOKE_RC != 0 || $SMOKE_TAIL != EXIT_CODE=0 || $MESH != 2x4 ]]; then
  echo "FS8_SMOKE=FAIL rc=$SMOKE_RC tail=${SMOKE_TAIL:-none} mesh=${MESH:-?} log=$SMOKE_LOG $(ts)"
  echo "FS8_STOP reason=smoke_fail $(ts)"
  exit 3
fi
echo "FS8_SMOKE=PASS steps=20 mesh=$MESH s_per_step=${RATE:-?} $(ts)"

# ── g) 正式 + 增量上传（D5）───────────────────────────────────────────────────
launch_prod() {
  RUN=$BASE0$RUN_SUFFIX
  BUCKET=$BUCKET0$RUN_SUFFIX
  PROD_LOG=$STORE/logs/$RUN.log
  INC_LOG=$STORE/logs/$RUN-inc.log
  CKPT_DIR=$STORE/train-runs/mme_vla_suite/$RUN/$FIRST_CKPT_STEP
  # 旧日志改名，避免读到上一次的 EXIT_CODE=（run 脚本自己也会改名，这里先做以免 tail 抢读）
  [[ -e $PROD_LOG ]] && mv "$PROD_LOG" "$PROD_LOG.$(date +%Y%m%d%H%M%S)"
  [[ -e $INC_LOG ]] && mv "$INC_LOG" "$INC_LOG.$(date +%Y%m%d%H%M%S)"
  # run 脚本自己 tee 日志并写 EXIT_CODE=；只在它没写（日志之前就失败）时补一行
  local train_cmd="set -o pipefail; CODE_ROOT=$(q "$CODE_ROOT") PY=$(q "$PY") RUN_SUFFIX=$(q "$RUN_SUFFIX") bash $(q "$RUN_SCRIPT") fs8 prod; rc=\$?; grep -q '^EXIT_CODE=' $(q "$PROD_LOG") 2>/dev/null || echo \"EXIT_CODE=\$rc\" >> $(q "$PROD_LOG")"
  local inc_cmd="set -o pipefail; cd $(q "$CODE_ROOT") && PATH=$(q "$PATH") PYTHONUNBUFFERED=1 $(q "$INC_PY") scripts/dataset/hf_export/ctx1024_ckpt_inc.py --run $(q "$RUN") --bucket $(q "$BUCKET") 2>&1 | tee $(q "$INC_LOG"); echo \"EXIT_CODE=\$?\" >> $(q "$INC_LOG")"
  "$TMUX_BIN" new-session -d -s fs8-train "$train_cmd" || { echo "FS8_STOP reason=tmux_fail session=fs8-train $(ts)"; exit 5; }
  "$TMUX_BIN" new-session -d -s fs8-inc "$inc_cmd" || echo "FS8_INC_LAUNCH_FAIL $(ts)"
  echo "FS8_PROD_LAUNCHED run=$RUN bucket=$BUCKET log=$PROD_LOG inc_log=$INC_LOG $(ts)"
}
watch_prod() {  # 0=第一份 checkpoint 出现；1=10000 之前退出（PROD_EXIT 存尾行）
  local started=0 line fd tpid
  mark_started() {
    if ((started == 0)) && [[ $1 == *"Step "* || $1 == *"Progress on:"* ]]; then
      started=1; echo "FS8_PROD_STARTED run=$RUN $(ts)"
    fi
  }
  exec {fd}< <(exec tail -n +1 -F "$PROD_LOG" 2>/dev/null)
  tpid=$!
  while true; do
    if IFS= read -r -t "$FS8_POLL" line <&"$fd"; then
      line=${line//$'\r'/}
      mark_started "$line"
      if [[ $line == EXIT_CODE=* ]]; then
        if [[ -d $CKPT_DIR ]]; then break; fi
        PROD_EXIT=$line; kill "$tpid" 2>/dev/null; wait "$tpid" 2>/dev/null; exec {fd}<&-; return 1
      fi
    fi
    if [[ -d $CKPT_DIR ]]; then
      # 先读完已到的行，保证 FS8_PROD_STARTED 在 FS8_FIRST_CKPT 之前
      local k; for ((k = 0; k < 200; k++)); do IFS= read -r -t 1 line <&"$fd" || break; mark_started "${line//$'\r'/}"; done
      # checkpoint 已出现即证明训练已跑；日志行若还没被 tail 读到，补一行 STARTED（标明来源）
      ((started == 0)) && echo "FS8_PROD_STARTED run=$RUN via=ckpt $(ts)"
      break
    fi
  done
  kill "$tpid" 2>/dev/null; wait "$tpid" 2>/dev/null; exec {fd}<&-
  echo "FS8_FIRST_CKPT run=$RUN step=$FIRST_CKPT_STEP dir=$CKPT_DIR $(ts)"
  return 0
}

launch_prod
if watch_prod; then exit 0; fi
echo "FS8_PROD_EXIT run=$RUN $PROD_EXIT before_step=$FIRST_CKPT_STEP $(ts)"
if [[ -n $RUN_SUFFIX ]]; then echo "FS8_STOP reason=prod_exit run=$RUN $(ts)"; exit 4; fi
# D5：只停本任务的上传会话（精确名），换 -r2 重起一次
"$TMUX_BIN" kill-session -t '=fs8-inc' 2>/dev/null
while has_session fs8-train || has_session fs8-inc; do sleep "$FS8_POLL"; done
RUN_SUFFIX=-r2
echo "FS8_RETRY run_suffix=$RUN_SUFFIX $(ts)"
launch_prod
if watch_prod; then exit 0; fi
echo "FS8_PROD_EXIT run=$RUN $PROD_EXIT before_step=$FIRST_CKPT_STEP $(ts)"
echo "FS8_STOP reason=prod_exit run=$RUN $(ts)"
exit 4
