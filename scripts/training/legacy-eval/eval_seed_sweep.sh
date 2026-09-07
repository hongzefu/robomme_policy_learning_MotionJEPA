#!/usr/bin/env bash
# 一个 (组, seed) 的完整评估驱动：把 4 个任务按批串行跑完，每批调 eval_all_shards.sh 起 N 个 worker 会话，
# 等本批全部会话结束、逐片核 EXIT_CODE=0 后再起下一批。
#
# 为什么必须按任务分批：单个 eval.py 进程建满 27 个仿真环境后，第 28 次 make_env 必抛
# vk::createInstanceUnique: ErrorIncompatibleDriver（docs/training-doc/eval-official-framesamp-context/result.md
# 「一个确定性上限」，w0/w1 两次精确复现）。一批一个任务时每 worker 只建 13 或 25 个环境，安全。
# 为什么每批换 RUN_NAME：eval.py 跑完集号区间即写 log.json，外层 while not exists(log.json) 见到它就直接退出；
# 若 4 个批次共用一个 RUN_NAME，第二批起来会空转退出（eval-official-framesamp-context/launch.md 踩过）。
#
# 用法：GROUP=official|motion SEED=42 [TASKS=…] [PORT_BASE=8041] [DRY_RUN=1] bash scripts/training/legacy-eval/eval_seed_sweep.sh
#   GROUP=official  官方 perceptual-framesamp-context ckpt79999，无 sidecar，WORKERS=4 GPU_LIST=0,0,1,1（每卡 2 worker）
#   GROUP=motion    awsprod40k-b128-motion ckpt39999，带 motion sidecar，WORKERS=2 GPU_LIST=0,1（每卡 1 worker，sidecar 独占）
# 日志：本 driver 由调用方 tee 落 v1-store/logs/sweep-<GROUP>-s<SEED>.log；分片日志照旧 v1-store/logs/<LOG_PREFIX>-w<k>.log
# 结束写 SWEEP_RESULT=PASS|FAIL 与 EXIT_CODE=。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"
: "${GROUP:?必须设置 GROUP=official|motion}"
: "${SEED:?必须设置 SEED}"
TASKS="${TASKS:-ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap}"
# 默认端口段 9200+：本机 8042/8044/8045/8047-8050 有用户的长期服务在监听，8040 段不可用
# （2026-09-06 实测：w1/w3 连上别人的服务后 eval.py 立刻 abort 但退出码仍是 0，静默跑了 0 集）。
PORT_BASE="${PORT_BASE:-9200}"
DRY_RUN="${DRY_RUN:-0}"
ROBOMME_PY="${ROBOMME_PY:-$HOME/micromamba/envs/robomme/bin/python}"
MODELS="${V1_STORE}/models"

case "${GROUP}" in
  official)
    WORKERS=4; GPU_LIST="0,0,1,1"; MEMF="0.40"; CKPT_ID=79999
    CKPT_DIR="${MODELS}/official-mme-vla/perceptual-framesamp-context/79999"
    RUN_PREFIX="official-ctx"; LOG_TAG="evctx" ;;
  motion)
    WORKERS=2; GPU_LIST="0,1"; MEMF="0.55"; CKPT_ID=39999
    CKPT_DIR="${MODELS}/awsprod40k-b128-motion/39999"
    RUN_PREFIX="awsprod40k-motion"; LOG_TAG="evmot" ;;
  *) echo "错误: GROUP 只能是 official 或 motion，实为 ${GROUP}" >&2; exit 2 ;;
esac

# 任务名 → 会话名里的短缩写（会话名要短且能一眼看出是哪批）
abbrev() {
  case "$1" in
    ButtonUnmask) echo bu ;; VideoUnmask) echo vu ;;
    ButtonUnmaskSwap) echo bus ;; VideoUnmaskSwap) echo vus ;;
    *) echo "$1" | tr '[:upper:]' '[:lower:]' ;;
  esac
}

echo "=== SWEEP_START group=${GROUP} seed=${SEED} workers=${WORKERS} gpu_list=${GPU_LIST} mem_fraction=${MEMF}"
echo "    ckpt=${CKPT_DIR} tasks=${TASKS} port_base=${PORT_BASE} start=$(date '+%F %T') ==="
[[ -d "${CKPT_DIR}/params" ]] || { echo "错误: checkpoint 缺 params: ${CKPT_DIR}" >&2; echo "EXIT_CODE=1"; exit 1; }

FAIL=0
i=0
IFS=',' read -r -a TASK_ARR <<<"${TASKS}"
for TASK in "${TASK_ARR[@]}"; do
  AB="$(abbrev "${TASK}")"
  LP="${LOG_TAG}-s${SEED}-${AB}"
  PB=$((PORT_BASE + i * 8)); i=$((i + 1))
  echo "--- BATCH task=${TASK} log_prefix=${LP} port_base=${PB} start=$(date '+%T') ---"
  # 本批要用 PB..PB+WORKERS-1，逐个确认无人监听（eval_shard.sh 里也有同样的守卫，这里提前拦下整批）
  BUSY=""
  for ((k = 0; k < WORKERS; k++)); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$((PB + k))") 2>/dev/null; then exec 3>&-; BUSY="${BUSY} $((PB + k))"; fi
  done
  if [[ -n "${BUSY}" ]]; then echo "错误: 端口被占用:${BUSY}，换 PORT_BASE 重跑" >&2; FAIL=1; break; fi
  if ! MODE=stride WORKERS="${WORKERS}" GPU_LIST="${GPU_LIST}" TASKS_ALL="${TASK}" \
       RUN_NAME="${RUN_PREFIX}-s${SEED}-${TASK}" CKPT_ID="${CKPT_ID}" CKPT_DIR="${CKPT_DIR}" \
       SEED="${SEED}" LOG_PREFIX="${LP}" PORT_BASE="${PB}" \
       POLICY_MEM_FRACTION="${MEMF}" ROBOMME_PY="${ROBOMME_PY}" DRY_RUN="${DRY_RUN}" \
       bash "$(dirname "${BASH_SOURCE[0]}")/eval_all_shards.remote.sh"; then
    echo "错误: 起批失败 task=${TASK}" >&2; FAIL=1; break
  fi
  [[ "${DRY_RUN}" == "1" ]] && continue
  # 等本批全部会话结束：会话名恒为 <LP>-w<k>，用确切名字逐个 has-session，不做前缀匹配
  while :; do
    ALIVE=0
    for ((k = 0; k < WORKERS; k++)); do
      tmux has-session -t "=${LP}-w${k}" 2>/dev/null && ALIVE=$((ALIVE + 1))
    done
    [[ "${ALIVE}" -eq 0 ]] && break
    sleep 20
  done
  sleep 10        # 留出 server 进程退出与显存释放的余量，再起下一批
  # 逐片核退出码（EXIT_CODE= 由 eval_shard.sh 写在自己 stdout 的末尾；强杀不会写，故缺失也算失败）
  DONE_EPS=0
  for ((k = 0; k < WORKERS; k++)); do
    L="${LOGS_DIR}/${LP}-w${k}.log"
    RC="$(grep -oE '^EXIT_CODE=[0-9]+' "${L}" 2>/dev/null | tail -1 | cut -d= -f2)"
    PROG="${V1_STORE}/evaluation/${RUN_PREFIX}-s${SEED}-${TASK}-w${k}/ckpt${CKPT_ID}/seed${SEED}/progress.json"
    NEP="$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1])).get(sys.argv[2],{})))" "${PROG}" "${TASK}" 2>/dev/null || echo 0)"
    DONE_EPS=$((DONE_EPS + NEP))
    if [[ "${RC}" == "0" && "${NEP}" -gt 0 ]]; then
      echo "  ${LP}-w${k} EXIT_CODE=0 episodes=${NEP}"
    else
      echo "  ${LP}-w${k} EXIT_CODE=${RC:-缺失} episodes=${NEP} ← 失败"; FAIL=1
    fi
  done
  # 退出码为 0 不等于评了集：端口被别的服务占用时 eval.py 会 abort 但仍返回 0（见上面的守卫注释），
  # 故这里再核本批实评集数必须等于该任务的 50 集。
  if [[ "${DONE_EPS}" -ne 50 ]]; then
    echo "  ${LP} 本批实评 ${DONE_EPS} 集 ≠ 50 ← 失败"; FAIL=1
  else
    echo "  ${LP} 本批实评 50/50 集"
  fi
  echo "--- BATCH done task=${TASK} end=$(date '+%T') ---"
done

if [[ "${FAIL}" -eq 0 ]]; then echo "SWEEP_RESULT=PASS group=${GROUP} seed=${SEED} end=$(date '+%F %T')"
else echo "SWEEP_RESULT=FAIL group=${GROUP} seed=${SEED} end=$(date '+%F %T')"; fi
echo "EXIT_CODE=${FAIL}"
exit "${FAIL}"
