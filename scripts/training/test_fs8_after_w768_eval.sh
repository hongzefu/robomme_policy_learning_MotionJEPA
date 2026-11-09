#!/usr/bin/env bash
# fs8_after_w768_eval.sh 与 run_ctx1024_b64_80k.sh（DRY）的离线测试：不碰真实 tmux、nvidia-smi、GPU。
# 用法：bash scripts/training/test_fs8_after_w768_eval.sh
# 判定行：FS8_DRY=PASS td=same fs=same fs8=ok；FS8_TEST=PASS cases=<n>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CHAIN=$HERE/fs8_after_w768_eval.sh
RUNSH=$HERE/run_ctx1024_b64_80k.sh
BASELINE_SHA=${BASELINE_SHA:-69536cf}
ROOT=$(mktemp -d /tmp/fs8-test.XXXXXX)
FAILS=0; CASES=0
BASE0=orig100-ctx-framesamp1024-b64-80k-8gpu
A2_TEST="--policy.dir=/fake/w768/50000"

ok() { echo "  ok: $*"; }
bad() { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }
expect() { grep -qF -- "$2" "$1" && ok "$2" || bad "缺少 [$2]"; }
expect_not() { grep -qF -- "$2" "$1" && bad "不应出现 [$2]" || ok "无 [$2]"; }
alive() { [[ -n $(tr -d '\0' </proc/$1/cmdline 2>/dev/null) ]]; }  # 僵尸进程 cmdline 为空；/proc 文件 st_size 恒 0，不能用 -s
spawn_bg() {  # 双 fork 让进程脱离本 shell（避免僵尸），打印 pid；参数为 bash -c 脚本与 cwd
  local dir=$2 pf; pf=$(mktemp "$ROOT/pid.XXXX")
  ( cd "$dir" || exit 1; bash -c "$1" </dev/null >/dev/null 2>&1 & echo $! >"$pf" ) >/dev/null 2>&1
  sleep 0.3; cat "$pf"
}
# 清理只针对本测试用 $! 亲手记录的 PID（ALL_PIDS 与假 tmux 的 pid 文件），且发信号前复核 cmdline 带本次唯一标记
# MARK 或本次临时目录 ROOT，防 PID 复用误伤；不用 ps/grep 找进程。
MARK=fs8test-$(basename "$ROOT" | tr -dc 'A-Za-z0-9')
ALL_PIDS=()
ours() { local c; c=$(tr '\0' ' ' </proc/$1/cmdline 2>/dev/null); [[ $c == *"$MARK"* || $c == *"$ROOT"* ]]; }
cleanup() {
  local p q
  for p in "${ALL_PIDS[@]}"; do ours "$p" && kill -KILL "$p" 2>/dev/null; done
  for p in "$ROOT"/*/tmux/*.pid; do
    [[ -f $p ]] || continue; q=$(cat "$p")
    ours "$q" && kill -KILL -- "-$q" 2>/dev/null
  done
}
trap cleanup EXIT

# ── 一、DRY：td/fs 与改前逐字相同，fs8 参数正确 ─────────────────────────────────
echo "== DRY 比对（改前 $BASELINE_SHA + 同样的 DRY 打印块）"
OLD=$ROOT/old_run.sh
git -C "$REPO" show "$BASELINE_SHA:scripts/training/run_ctx1024_b64_80k.sh" >"$OLD"
uv run --no-project python - "$OLD" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = "if [[ $2 == prod ]]; then\n  WANDB_API_KEY"
assert s.count(a) == 1; s = s.replace(a, "if [[ $2 == prod && ${DRY:-0} != 1 ]]; then\n  WANDB_API_KEY")
b = "LOG=$STORE/logs/$RUN.log\n"
assert s.count(b) == 1
s = s.replace(b, b + '''if [[ ${DRY:-0} == 1 ]]; then
  echo "DRY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "DRY MMEVLA_JAX_CACHE_DIR=$MMEVLA_JAX_CACHE_DIR"
  echo "DRY LOG=$LOG"
  echo "DRY PY=$PY"
  printf 'DRY ARG %s\\n' "${ARGS[@]}"
  exit 0
fi
''')
open(p, "w").write(s)
EOF
TD=same; FS=same; FS8=ok
for arm in td fs; do
  for mode in prod smoke perf; do
    o=$(CODE_ROOT=$REPO DRY=1 bash "$OLD" "$arm" "$mode"); rc1=$?
    n=$(CODE_ROOT=$REPO DRY=1 bash "$RUNSH" "$arm" "$mode"); rc2=$?
    if [[ $rc1 != 0 || $rc2 != 0 || "$o" != "$n" ]]; then
      echo "  FAIL: $arm $mode 不一致"; diff <(echo "$o") <(echo "$n"); [[ $arm == td ]] && TD=diff || FS=diff
    fi
  done
done
f8=$(CODE_ROOT=$REPO DRY=1 bash "$RUNSH" fs8 prod)
f8s=$(CODE_ROOT=$REPO DRY=1 RUN_SUFFIX=-r2 bash "$RUNSH" fs8 smoke)
fsp=$(CODE_ROOT=$REPO DRY=1 bash "$RUNSH" fs prod)
echo "$f8" | sed 's/^/  fs8 prod: /'
grep -qx 'DRY CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' <<<"$f8" || FS8=bad
grep -qx "DRY ARG $BASE0" <<<"$f8" || FS8=bad
grep -qx 'DRY MMEVLA_JAX_CACHE_DIR=/scratch/hongze/.cache/jax/'"$BASE0" <<<"$f8s" || FS8=bad
grep -qx "DRY ARG $BASE0-r2-smoke" <<<"$f8s" || FS8=bad
grep -qx "DRY LOG=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/$BASE0-r2-smoke.log" <<<"$f8s" || FS8=bad
# fs8 与 fs 的参数数组只差 run 名
[[ "$(grep '^DRY ARG' <<<"$f8" | sed "s/$BASE0/X/")" == "$(grep '^DRY ARG' <<<"$fsp" | sed 's/orig100-ctx-framesamp1024-b64-80k/X/')" ]] || FS8=bad
grep -qx 'DRY PY=/x/py' <<<"$(CODE_ROOT=$REPO DRY=1 PY=/x/py bash "$RUNSH" fs8 prod)" || FS8=bad
grep -qx 'DRY PY=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python' <<<"$f8" || FS8=bad
if [[ $TD == same && $FS == same && $FS8 == ok ]]; then DRY_V=PASS; else DRY_V=FAIL; FAILS=$((FAILS + 1)); fi
echo "FS8_DRY=$DRY_V td=$TD fs=$FS fs8=$FS8"

# ── 二、链式脚本 ────────────────────────────────────────────────────────────
setup_case() {  # $1=名字；导出 T 及全部桩
  T=$ROOT/$1; mkdir -p "$T/tmux" "$T/store/logs" "$T/store/train-runs/mme_vla_suite" "$T/main" "$T/state"
  : >"$T/gpu_pids"
  cat >"$T/nvidia-smi" <<EOF
#!/usr/bin/env bash
# 假 nvidia-smi：gpu_pids 里仍存活的进程算计算进程；有存活则卡 0 显存 30000
live=()
while read -r p; do [[ -n \$p && -n \$(tr -d '\\0' </proc/\$p/cmdline 2>/dev/null) ]] && live+=("\$p"); done <"$T/gpu_pids"
case "\$*" in
  *query-gpu=memory.used*) for i in 0 1 2 3 4 5 6 7; do if ((i == 0 && \${#live[@]} > 0)); then echo 30000; else echo 3; fi; done ;;
  *query-compute-apps*) for p in "\${live[@]}"; do echo "\$p"; done ;;
  *) exit 9 ;;
esac
EOF
  cat >"$T/tmux-fake" <<EOF
#!/usr/bin/env bash
# 假 tmux：new-session/has-session/kill-session 三个子命令；调用记入 calls
D=$T/tmux; echo "\$*" >>"$T/tmux.calls"
case "\$1" in
  new-session) shift; name=; while [[ \$# -gt 1 ]]; do case "\$1" in -d) shift ;; -s) name=\$2; shift 2 ;; *) break ;; esac; done
    [[ -f \$D/\$name.pid ]] && kill -0 "\$(cat \$D/\$name.pid)" 2>/dev/null && { echo "duplicate session: \$name" >&2; exit 1; }
    setsid bash -c "\$1" </dev/null >"\$D/\$name.out" 2>&1 & echo \$! >"\$D/\$name.pid" ;;
  has-session) n=\${3#=}; [[ -f \$D/\$n.pid ]] && [[ -n \$(tr -d '\\0' </proc/\$(cat \$D/\$n.pid)/cmdline 2>/dev/null) ]] ;;
  kill-session) n=\${3#=}; [[ -f \$D/\$n.pid ]] || exit 1; kill -TERM -- "-\$(cat \$D/\$n.pid)" 2>/dev/null; rm -f "\$D/\$n.pid" ;;
  *) echo "fake tmux: 不支持 \$1" >&2; exit 2 ;;
esac
EOF
  cat >"$T/run_stub.sh" <<'EOF'
#!/usr/bin/env bash
# 桩 run 脚本：按 STORE 写日志；smoke 由 SMOKE_RC 控制，prod 由 PROD_CRASH（first|all）控制
set -uo pipefail
RUN=orig100-ctx-framesamp1024-b64-80k-8gpu${RUN_SUFFIX:-}
echo "$1 $2 RUN_SUFFIX=${RUN_SUFFIX:-} CODE_ROOT=$CODE_ROOT PY=$PY" >>"$STORE/../run_stub.calls"
if [[ $2 == smoke ]]; then
  L=$STORE/logs/$RUN-smoke.log
  { echo "MME_VLA_SUITE_IMPORT=$CODE_ROOT/src/mme_vla_suite/__init__.py"
    echo "TRAIN_MESH shape=(2, 4) devices=8 fsdp_devices=4 batch_size=64 num_workers=4"
    echo "Step 0: loss=0.4"
    echo "13:48:47.338 [I] Progress on: 6.00it/20.0it rate:7.8s/it remaining:01:49"
    echo "13:49:19.558 [I] Progress on: 18.0it/20.0it rate:1.6s/it remaining:00:05"
    echo "13:49:36.415 [I] Progress on: 20.0it/20.0it rate:6.2s/it remaining:00:00"
    echo "EXIT_CODE=${SMOKE_RC:-0}"; } >"$L"
  echo "stub smoke stdout"
  exit "${SMOKE_RC:-0}"
fi
L=$STORE/logs/$RUN.log
echo "TRAIN_MESH shape=(2, 4)" >"$L"; sleep 1
echo "Step 0: loss=0.4" >>"$L"; sleep 1
if [[ ${PROD_CRASH:-} == all || ( ${PROD_CRASH:-} == first && -z ${RUN_SUFFIX:-} ) ]]; then
  echo "Traceback: boom" >>"$L"; echo "EXIT_CODE=1" >>"$L"; exit 1
fi
mkdir -p "$STORE/train-runs/mme_vla_suite/$RUN/10000"
sleep 1; echo "EXIT_CODE=0" >>"$L"; exit 0
EOF
  printf '#!/usr/bin/env bash\necho "INC $*"; sleep 37\n' >"$T/inc_stub.sh"
  chmod +x "$T/nvidia-smi" "$T/tmux-fake" "$T/run_stub.sh" "$T/inc_stub.sh"
}
run_chain() {  # $1=超时秒数；其余 env 由调用方给
  local to=$1; shift
  env CODE_ROOT="$REPO" PY=/fake/py STORE="$T/store" EVAL_LOG="$T/w768-eval.log" EVAL_SESSION=w768-eval \
    RUN_SCRIPT="$T/run_stub.sh" INC_PY="$T/inc_stub.sh" NVIDIA_SMI="$T/nvidia-smi" TMUX_BIN="$T/tmux-fake" \
    DF_PATH="$T" MIN_FREE_GB=1 FS8_POLL=1 FS8_IDLE_N=3 FS8_ORPHAN_AFTER=2 FS8_REPORT_EVERY=1 FS8_TERM_GRACE=2 \
    ORPHAN_A2="$A2_TEST" ORPHAN_C1="$T/eval-root/local692-wantp768-50k" "$@" \
    timeout "$to" bash "$CHAIN" >"$T/chain.log" 2>&1
  RC=$?
  sed 's/^/    | /' "$T/chain.log"
}
eval_line_later() {  # 文件先不存在，1 s 后写入
  ( sleep 1; echo "LOCAL692_START ..." >>"$T/w768-eval.log"; sleep 0.5; echo "$1" >>"$T/w768-eval.log" ) &
}
no_leftover_tail() {
  if ps -eo args | grep -F -- "tail -n +1 -F $T/" | grep -vq -- grep; then bad "残留 tail 进程"; else ok "无残留 tail"; fi
}

echo "== case 1：EVAL692=PASS，评估会话 2 s 后退出，正常起跑到第一份 checkpoint"
CASES=$((CASES + 1)); setup_case c1
"$T/tmux-fake" new-session -d -s w768-eval "sleep 2.3; : $ROOT"
eval_line_later "EVAL692=PASS episodes=692 success=300 rate=0.43"
run_chain 60
[[ $RC == 0 ]] && ok "rc=0" || bad "rc=$RC"
for s in "FS8_WAIT_START" "FS8_EVAL_RESULT=PASS line=EVAL692=PASS episodes=692" "FS8_EVAL_SESSION_GONE" "FS8_GPU_IDLE" \
         "FS8_TRIGGER=PASS eval=PASS gpus_idle=8 free_gb=" "FS8_SMOKE=PASS steps=20 mesh=2x4 s_per_step=1.6s/it" \
         "FS8_PROD_LAUNCHED run=$BASE0 bucket=HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-8gpu-v1 " \
         "FS8_PROD_STARTED run=$BASE0" "FS8_FIRST_CKPT run=$BASE0 step=10000"; do expect "$T/chain.log" "$s"; done
expect_not "$T/chain.log" "stub smoke stdout"
expect "$T/tmux.calls" "new-session -d -s fs8-inc"
expect "$T/tmux/fs8-inc.out" "INC scripts/dataset/hf_export/ctx1024_ckpt_inc.py --run $BASE0 --bucket HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-8gpu-v1"
expect "$T/run_stub.calls" "fs8 smoke RUN_SUFFIX= CODE_ROOT=$REPO PY=/fake/py"
sleep 2
n=$(grep -c '^EXIT_CODE=' "$T/store/logs/$BASE0.log"); [[ $n == 1 ]] && ok "prod 日志 EXIT_CODE= 恰 1 行" || bad "prod 日志 EXIT_CODE= 行数=$n"
no_leftover_tail

echo "== case 2：EVAL692=FAIL 仍起跑（D1）"
CASES=$((CASES + 1)); setup_case c2
eval_line_later "EVAL692=FAIL episodes=600"
run_chain 60
[[ $RC == 0 ]] && ok "rc=0" || bad "rc=$RC"
expect "$T/chain.log" "FS8_EVAL_RESULT=FAIL"
expect "$T/chain.log" "FS8_TRIGGER=PASS eval=FAIL gpus_idle=8"
expect "$T/chain.log" "FS8_FIRST_CKPT run=$BASE0"
no_leftover_tail

echo "== case 3：GPU 一直被非特征进程占用（cwd 在主检出），不杀、持续等"
CASES=$((CASES + 1)); setup_case c3
P=$(spawn_bg "exec -a '$MARK-innocent' sleep 3007" "$T/main"); ALL_PIDS+=("$P"); echo "$P" >"$T/gpu_pids"
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 8
[[ $RC == 124 ]] && ok "超时仍在等（rc=124）" || bad "rc=$RC"
expect "$T/chain.log" "FS8_GPU_WAIT gpus=8 busy_mem=1 apps=[$P]"
expect_not "$T/chain.log" "FS8_ORPHAN_TERM"
expect_not "$T/chain.log" "FS8_TRIGGER"
alive "$P" && ok "非特征进程仍存活" || bad "非特征进程被杀"

echo "== case 4：孤儿按 cmdline 特征 TERM（忽略 TERM 的升 KILL），非特征进程不碰"
CASES=$((CASES + 1)); setup_case c4
O1=$(spawn_bg "trap '' TERM; exec -a 'python scripts/training/serve_policy.py --seed=7 $A2_TEST $MARK' sleep 3013" "$T/main")
O2=$(spawn_bg "exec -a 'python /x/motion_sidecar.py --fd 3 --policy wan_fixedpool768 $MARK' sleep 3017" "$T/main")
O3=$(spawn_bg "exec -a 'python examples/robomme/eval.py --save $T/eval-root/local692-wantp768-50k-retry1/s3 $MARK' sleep 3019" "$T/main")
O4=$(spawn_bg "exec -a 'python scripts/training/serve_policy.py --policy.dir=/other/ckpt $MARK' sleep 3023" "$T/main")  # 只命中 A1，不算
I1=$(spawn_bg "exec -a '$MARK-innocent' sleep 3011" "$T/main")
ALL_PIDS+=("$O1" "$O2" "$O3" "$O4" "$I1"); printf '%s\n' "$O1" "$O2" "$O3" "$O4" "$I1" >"$T/gpu_pids"
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 14
[[ $RC == 124 ]] && ok "非特征进程仍占卡，持续等（rc=124）" || bad "rc=$RC"
for p in "$O1" "$O2" "$O3"; do expect "$T/chain.log" "FS8_ORPHAN_TERM pid=$p "; alive "$p" && bad "孤儿 $p 仍存活" || ok "孤儿 $p 已退出"; done
expect "$T/chain.log" "FS8_ORPHAN_KILL pid=$O1"
for p in "$O4" "$I1"; do expect_not "$T/chain.log" "pid=$p "; alive "$p" && ok "非特征 $p 存活" || bad "非特征 $p 被杀"; done

echo "== case 5：只有孤儿，清理后空闲并触发"
CASES=$((CASES + 1)); setup_case c5
O=$(spawn_bg "exec -a 'python serve_policy.py $A2_TEST $MARK' sleep 3013" "$T/main"); ALL_PIDS+=("$O"); echo "$O" >"$T/gpu_pids"
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 60
[[ $RC == 0 ]] && ok "rc=0" || bad "rc=$RC"
expect "$T/chain.log" "FS8_ORPHAN_TERM pid=$O "
expect "$T/chain.log" "FS8_TRIGGER=PASS eval=PASS"

echo "== case 6：冒烟失败退出 3，不起正式"
CASES=$((CASES + 1)); setup_case c6
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 60 SMOKE_RC=1
[[ $RC == 3 ]] && ok "rc=3" || bad "rc=$RC"
expect "$T/chain.log" "FS8_SMOKE=FAIL rc=1 tail=EXIT_CODE=1"
expect "$T/chain.log" "FS8_STOP reason=smoke_fail"
expect_not "$T/tmux.calls" "fs8-train"

echo "== case 7：正式在 10000 前崩溃 → 停 fs8-inc、以 -r2 重起一次（D5）"
CASES=$((CASES + 1)); setup_case c7
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 60 PROD_CRASH=first
[[ $RC == 0 ]] && ok "rc=0" || bad "rc=$RC"
expect "$T/chain.log" "FS8_PROD_EXIT run=$BASE0 EXIT_CODE=1"
expect "$T/tmux.calls" "kill-session -t =fs8-inc"
expect "$T/chain.log" "FS8_RETRY run_suffix=-r2"
expect "$T/chain.log" "FS8_PROD_LAUNCHED run=$BASE0-r2 bucket=HongzeFu/robomme-vla-orig100-ctx-framesamp1024-b64-80k-8gpu-v1-r2 "
expect "$T/chain.log" "FS8_FIRST_CKPT run=$BASE0-r2 step=10000"
expect "$T/run_stub.calls" "fs8 prod RUN_SUFFIX=-r2"
[[ $(grep -c 'fs8 smoke' "$T/run_stub.calls") == 1 ]] && ok "冒烟只跑一次" || bad "冒烟次数不对"
[[ $(grep -c 'kill-session' "$T/tmux.calls") == 1 ]] && ok "只 kill 一次 fs8-inc" || bad "kill-session 次数不对"

echo "== case 8：-r2 也崩溃 → FS8_STOP reason=prod_exit 退出 4"
CASES=$((CASES + 1)); setup_case c8
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 60 PROD_CRASH=all
[[ $RC == 4 ]] && ok "rc=4" || bad "rc=$RC"
expect "$T/chain.log" "FS8_STOP reason=prod_exit run=$BASE0-r2"

echo "== case 9：磁盘不足持续等（D6）"
CASES=$((CASES + 1)); setup_case c9
echo "EVAL692=PASS x" >"$T/w768-eval.log"
run_chain 8 MIN_FREE_GB=999999999
[[ $RC == 124 ]] && ok "rc=124" || bad "rc=$RC"
expect "$T/chain.log" "FS8_DISK_WAIT free_gb="
expect_not "$T/chain.log" "FS8_TRIGGER"

echo "== 红线：链式脚本不含禁用 tmux 写法与裸 pgrep"
CASES=$((CASES + 1))
if grep -nE 'kill-server|kill-session -a|pkill|killall|pgrep' "$CHAIN"; then bad "出现禁用写法"; else ok "无禁用写法"; fi
[[ $(grep -v '^[[:space:]]*#' "$CHAIN" | grep -c 'kill-session') == 1 ]] && grep -q "kill-session -t '=fs8-inc'" "$CHAIN" && ok "仅 kill-session =fs8-inc" || bad "kill-session 用法不符"

cleanup; trap - EXIT
if ((FAILS == 0)); then echo "FS8_TEST=PASS cases=$CASES"; rm -rf "$ROOT"; exit 0; fi
echo "FS8_TEST=FAIL cases=$CASES fails=$FAILS root=$ROOT"; exit 1
