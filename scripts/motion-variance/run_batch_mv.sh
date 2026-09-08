#!/usr/bin/env bash
# motion-variance 一批 = 一个 (COND, SPLIT, SEED)：起 WORKERS 个 stride 分片（一片一卡一 detached tmux 会话），等全部结束，逐片核对。
# 用法：COND=… SPLIT=… SEED=… [WORKERS=8] [GPU_LIST=0,1,…,7] [PORT_BASE=9300] [EP_COUNT=0] [TASKS_ALL=…] [DRY_RUN=0] [MV_ALLOW_DIRTY=0] [RUN_SUFFIX=]
#       bash scripts/motion-variance/run_batch_mv.sh
#   会话名 mv-<t|v><seed>-<cond><RUN_SUFFIX>-w<k>；日志 v1-store/logs/<同名>.log（tee）。
#   每批核对（阻断）：① 逐 worker EXIT_CODE=0；② progress.json 集数 == episode_plan 长度（stride 8：28/28/24×6）；③ MV_EP_DONE 数同值；
#   ④ 非 official：MV_EPISODE_BEGIN 数同值且 server.log 无 Traceback；⑤ swap：cross_seg 总数 0。结尾 MV_BATCH=PASS|FAIL + EXIT_CODE=。
#   清理只允许 tmux kill-session -t <确切会话名>（AGENTS 7）。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../training" && pwd)/paths.sh"
: "${COND:?}" ; : "${SPLIT:?}" ; : "${SEED:?}"
WORKERS="${WORKERS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-9300}"
EP_COUNT="${EP_COUNT:-0}"
TASKS_ALL="${TASKS_ALL:-ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap}"
DRY_RUN="${DRY_RUN:-0}"
MV_ALLOW_DIRTY="${MV_ALLOW_DIRTY:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
MV_DIR="${REPO_ROOT}/scripts/motion-variance"
case "${SPLIT}" in test) SP=t; SEED_SEG="seed${SEED}" ;; val) SP=v; SEED_SEG="val-seed${SEED}" ;; *) echo "错误: SPLIT=${SPLIT}" >&2; exit 2 ;; esac
case "${COND}" in official) CKPT_ID=79999 ;; normal|mask|swap) CKPT_ID=39999 ;; *) echo "错误: COND=${COND}" >&2; exit 2 ;; esac
LP="mv-${SP}${SEED}-${COND}${RUN_SUFFIX}"
IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -ge "${WORKERS}" ]] || { echo "错误: GPU_LIST 只有 ${#GPUS[@]} 张 < WORKERS=${WORKERS}" >&2; exit 2; }
N_TASKS=$(awk -F, '{print NF}' <<<"${TASKS_ALL}")
expected_of() {  # 与 mv_common.episode_plan 同一公式
  local k=$1 n_eff=50
  if [[ "${EP_COUNT}" -gt 0 ]]; then n_eff=$(( k + EP_COUNT < 50 ? k + EP_COUNT : 50 )); fi
  local per=$(( k < n_eff ? (n_eff - k + WORKERS - 1) / WORKERS : 0 ))
  echo $(( per * N_TASKS ))
}
echo "=== MV_BATCH_START cond=${COND} split=${SPLIT} seed=${SEED} workers=${WORKERS} gpu_list=${GPU_LIST} port_base=${PORT_BASE} ep_count=${EP_COUNT} tasks=${TASKS_ALL} log_prefix=${LP} start=$(date '+%F %T') ==="
BUSY=""
for ((k = 0; k < WORKERS; k++)); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$((PORT_BASE + k))") 2>/dev/null; then exec 3>&-; BUSY="${BUSY} $((PORT_BASE + k))"; fi
done
if [[ -n "${BUSY}" ]]; then echo "错误: 端口被占用:${BUSY}，换 PORT_BASE" >&2; echo "MV_BATCH=FAIL cond=${COND} split=${SPLIT} seed=${SEED} reason=port_busy"; echo "EXIT_CODE=1"; exit 1; fi
mkdir -p "${LOGS_DIR}"
T_START=$(date +%s)
for ((k = 0; k < WORKERS; k++)); do
  SESSION="${LP}-w${k}"
  if tmux has-session -t "=${SESSION}" 2>/dev/null; then echo "错误: 会话 ${SESSION} 已存在，拒绝重起（先按名清理）" >&2; echo "MV_BATCH=FAIL reason=session_exists"; echo "EXIT_CODE=1"; exit 1; fi
  LOG="${LOGS_DIR}/${SESSION}.log"
  ENVS="COND=${COND} SPLIT=${SPLIT} SEED=${SEED} K=${k} GPU=${GPUS[$k]} PORT=$((PORT_BASE + k)) WORKERS=${WORKERS} EP_COUNT=${EP_COUNT} TASKS_ALL=${TASKS_ALL} LOG_PREFIX=${LP} MV_ALLOW_DIRTY=${MV_ALLOW_DIRTY} RUN_SUFFIX=${RUN_SUFFIX}"
  CMD="set -o pipefail; ${ENVS} bash ${MV_DIR}/eval_shard_mv.sh 2>&1 | tee ${LOG}; sleep 2"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "DRY ${SESSION}: ${CMD}"; continue; fi
  tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "bash -c '${CMD}'"
  echo "起 ${SESSION}: gpu=${GPUS[$k]} port=$((PORT_BASE + k)) expected=$(expected_of "$k") log=${LOG}"
done
[[ "${DRY_RUN}" == "1" ]] && { echo "MV_BATCH=DRY cond=${COND} split=${SPLIT} seed=${SEED}"; exit 0; }
while :; do
  ALIVE=0
  for ((k = 0; k < WORKERS; k++)); do tmux has-session -t "=${LP}-w${k}" 2>/dev/null && ALIVE=$((ALIVE + 1)); done
  [[ "${ALIVE}" -eq 0 ]] && break
  sleep 20
done
sleep 10
FAIL=0; TOTAL=0; TOTAL_EXP=0; CROSS=0
for ((k = 0; k < WORKERS; k++)); do
  L="${LOGS_DIR}/${LP}-w${k}.log"; SL="${LOGS_DIR}/${LP}-w${k}.server.log"; EL="${LOGS_DIR}/${LP}-w${k}.eval.log"
  RC="$(grep -oE '^EXIT_CODE=[0-9]+' "${L}" 2>/dev/null | tail -1 | cut -d= -f2)"
  EXP=$(expected_of "$k")
  PROG="${V1_STORE}/evaluation/mv-${COND}-s${SEED}-${SPLIT}${RUN_SUFFIX}-w${k}/ckpt${CKPT_ID}/${SEED_SEG}/progress.json"
  NEP="$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(sum(len(v) for v in d.values()))" "${PROG}" 2>/dev/null || echo 0)"
  NDONE=$(grep -c '^MV_EP_DONE ' "${EL}" 2>/dev/null || true); NDONE=${NDONE:-0}
  NERR=$(grep -c '"error"' "${PROG}" 2>/dev/null || true); NERR=${NERR:-0}
  TB=0; NBEGIN="n/a"; CS=0
  if [[ "${COND}" != "official" ]]; then
    NBEGIN=$(grep -c '^MV_EPISODE_BEGIN ' "${SL}" 2>/dev/null || true); NBEGIN=${NBEGIN:-0}
    # 端口就绪探测（/dev/tcp 连上即断）会让 websockets 打 EOFError 栈，良性；只数其它异常
    TB=$(grep -E '^[A-Za-z_.]+(Error|Exception)' "${SL}" 2>/dev/null | grep -vcE 'EOFError: (stream ends|connection closed)' || true); TB=${TB:-0}
    if [[ "${COND}" == "swap" ]]; then CS=$(grep -oE 'cross_seg:[0-9]+' "${SL}" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}'); fi
  fi
  TOTAL=$((TOTAL + NEP)); TOTAL_EXP=$((TOTAL_EXP + EXP)); CROSS=$((CROSS + CS))
  OK=1
  [[ "${RC}" == "0" ]] || OK=0
  [[ "${NEP}" -eq "${EXP}" && "${NDONE}" -eq "${EXP}" && "${NERR}" -eq 0 ]] || OK=0
  if [[ "${COND}" != "official" ]]; then [[ "${NBEGIN}" -eq "${EXP}" && "${TB}" -eq 0 ]] || OK=0; fi
  echo "  ${LP}-w${k} EXIT_CODE=${RC:-缺失} expected=${EXP} progress=${NEP} ep_done=${NDONE} errors=${NERR} episode_begin=${NBEGIN} traceback=${TB} cross_seg=${CS} $([[ ${OK} -eq 1 ]] && echo ok || echo '← 失败')"
  [[ "${OK}" -eq 1 ]] || FAIL=1
done
[[ "${CROSS}" -eq 0 ]] || FAIL=1
WALL_MIN=$(( ($(date +%s) - T_START) / 60 ))
SIDECAR=off; [[ "${COND}" == "normal" ]] && SIDECAR=on
echo "MV_BATCH=$([[ ${FAIL} -eq 0 ]] && echo PASS || echo FAIL) cond=${COND} split=${SPLIT} seed=${SEED} episodes=${TOTAL}/${TOTAL_EXP} workers=${WORKERS}/${WORKERS} wall_min=${WALL_MIN} sidecar=${SIDECAR} cross_seg=${CROSS} suffix=${RUN_SUFFIX} end=$(date '+%F %T')"
echo "EXIT_CODE=${FAIL}"
exit "${FAIL}"
