#!/usr/bin/env bash
# motion-variance 单分片驱动（骨架抄 scripts/training/legacy-eval/eval_shard.remote.sh）：
#   起 policy server（official → 主线 scripts/training/serve_policy.py；normal/mask/swap → serve_policy_mv.py）
#   → 跑 scripts/motion-variance/robomme/eval.py（micromamba robomme 环境）→ 收 server → TIMING / MV_SHARD_SUMMARY / EXIT_CODE。
# 用法：COND=official|normal|mask|swap SPLIT=test|val SEED=42 K=<worker 号> GPU=<卡> PORT=<端口>
#       [WORKERS=8] [EP_STRIDE=$WORKERS] [EP_START=$K] [EP_COUNT=0] [TASKS_ALL=4 任务] [POLICY_MEM_FRACTION]
#       [SWAP_BANK=v1-store/reports/motion-variance/bank-lib] [ROBOMME_PY=…] [LOG_PREFIX=mv-<t|v><seed>-<cond>] [RUN_SUFFIX=]
#       [MV_ALLOW_DIRTY=0] [DUMP_OBS_EVERY=0]
#   结果   v1-store/evaluation/mv-<cond>-s<seed>-<split><RUN_SUFFIX>-w<K>/ckpt<id>/<seed<S>|<split>-seed<S>>/{progress.json,log.json,videos/}
#   日志   v1-store/logs/<LOG_PREFIX>-w<K>.{server.log,eval.log,probe.jsonl}；本脚本 stdout 由 run_batch_mv.sh tee 落 <LOG_PREFIX>-w<K>.log
#   MV_ALLOW_DIRTY=1 只供 smoke：跳过 clean-HEAD 闸并在头行打 DIRTY_TREE=1（run_matrix_mv.sh 一律拒绝）。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../training" && pwd)/paths.sh"
v1_require_models 1
: "${COND:?必须设置 COND=official|normal|mask|swap}"
: "${SPLIT:?必须设置 SPLIT=test|val}"
: "${SEED:?必须设置 SEED}"
: "${K:?必须设置 K}"
: "${GPU:?必须设置 GPU}"
: "${PORT:?必须设置 PORT}"
WORKERS="${WORKERS:-8}"
EP_STRIDE="${EP_STRIDE:-${WORKERS}}"
EP_START="${EP_START:-${K}}"
EP_COUNT="${EP_COUNT:-0}"
TASKS_ALL="${TASKS_ALL:-ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap}"
ROBOMME_PY="${ROBOMME_PY:-/scratch/hongze/micromamba/envs/robomme/bin/python}"
SWAP_BANK="${SWAP_BANK:-${V1_STORE}/reports/motion-variance/bank-lib}"
MV_ALLOW_DIRTY="${MV_ALLOW_DIRTY:-0}"
DUMP_OBS_EVERY="${DUMP_OBS_EVERY:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
MV_DIR="${REPO_ROOT}/scripts/motion-variance"
case "${SPLIT}" in test) SP=t; SEED_SEG="seed${SEED}" ;; val) SP=v; SEED_SEG="val-seed${SEED}" ;; *) echo "错误: SPLIT=${SPLIT}" >&2; exit 2 ;; esac
case "${COND}" in
  official) CKPT_DIR="${MODELS_DIR}/official-mme-vla/perceptual-framesamp-context/79999"; CKPT_ID=79999; MEMF_DEFAULT=0.40 ;;
  normal)   CKPT_DIR="${TRAIN_RUNS}/mme_vla_suite_b128/awsprod40k-b128-motion/39999"; CKPT_ID=39999; MEMF_DEFAULT=0.55 ;;
  mask|swap) CKPT_DIR="${TRAIN_RUNS}/mme_vla_suite_b128/awsprod40k-b128-motion/39999"; CKPT_ID=39999; MEMF_DEFAULT=0.40 ;;
  *) echo "错误: COND=${COND}" >&2; exit 2 ;;
esac
POLICY_MEM_FRACTION="${POLICY_MEM_FRACTION:-${MEMF_DEFAULT}}"
RUN_NAME="mv-${COND}-s${SEED}-${SPLIT}"
POLICY_NAME="${RUN_NAME}${RUN_SUFFIX}-w${K}"
LOG_PREFIX="${LOG_PREFIX:-mv-${SP}${SEED}-${COND}${RUN_SUFFIX}}"
SERVER_LOG="${LOGS_DIR}/${LOG_PREFIX}-w${K}.server.log"
EVAL_LOG="${LOGS_DIR}/${LOG_PREFIX}-w${K}.eval.log"
PROBE_OUT="${LOGS_DIR}/${LOG_PREFIX}-w${K}.probe.jsonl"
DUMP_DIR="${V1_STORE}/reports/motion-variance/obs-dump/${POLICY_NAME}"
[[ -d "${CKPT_DIR}/params" ]] || { echo "错误: checkpoint 缺 params: ${CKPT_DIR}" >&2; exit 1; }
[[ -x "${ROBOMME_PY}" ]] || { echo "错误: robomme 环境 python 不存在: ${ROBOMME_PY}" >&2; exit 1; }
if [[ "${COND}" == "normal" ]]; then
  [[ -x "${V1_STORE}/venvs/wan/bin/python" ]] || { echo "错误: sidecar venv 不存在: ${V1_STORE}/venvs/wan" >&2; exit 1; }
  [[ -f "${V1_STORE}/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt" ]] || { echo "错误: sidecar encoder checkpoint 不存在" >&2; exit 1; }
fi
if [[ "${COND}" == "swap" ]]; then
  [[ -f "${SWAP_BANK}/donor_bank.json" ]] || { echo "错误: donor bank 不存在: ${SWAP_BANK}" >&2; exit 1; }
fi
DIRTY=0
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain | head -c 1)" ]]; then
  if [[ "${MV_ALLOW_DIRTY}" == "1" ]]; then DIRTY=1; else
    echo "错误: 工作区不干净——正式评估必须从 clean HEAD 起（AGENTS 12）；smoke 可设 MV_ALLOW_DIRTY=1" >&2; exit 1; fi
fi
mkdir -p "${LOGS_DIR}" "${V1_STORE}/evaluation"
# 端口占用守卫（本机 8040 段有用户长期服务；退出码 0 不等于评了集）
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  exec 3>&-
  echo "错误: 端口 ${PORT} 起跑前已被占用，换 PORT_BASE 重试" >&2; exit 1
fi
# 每 worker 期望集数 = 与 mv_common.episode_plan 同一公式：每任务 ceil((N - start)/stride)，N = min(50, start+count) 当 count>0
N_EFF=50; if [[ "${EP_COUNT}" -gt 0 ]]; then N_EFF=$(( EP_START + EP_COUNT < 50 ? EP_START + EP_COUNT : 50 )); fi
PER_TASK=$(( EP_START < N_EFF ? (N_EFF - EP_START + EP_STRIDE - 1) / EP_STRIDE : 0 ))
N_TASKS=$(awk -F, '{print NF}' <<<"${TASKS_ALL}")
EXPECTED=$(( PER_TASK * N_TASKS ))
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
echo "=== MV_SHARD shard=w${K} cond=${COND} split=${SPLIT} seed=${SEED} HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD) DIRTY_TREE=${DIRTY} ckpt=${CKPT_DIR} gpu=${GPU} port=${PORT} tasks=${TASKS_ALL} ep_start=${EP_START} ep_stride=${EP_STRIDE} ep_count=${EP_COUNT} expected=${EXPECTED} policy_name=${POLICY_NAME} mem_fraction=${POLICY_MEM_FRACTION} xla_flags='${XLA_FLAGS}' start=$(date '+%F %T') ==="
cd "${REPO_ROOT}"
# ── policy server（后台；policy / sidecar / 仿真三者同一张卡）──
if [[ "${COND}" == "official" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION="${POLICY_MEM_FRACTION}" UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    uv run --no-sync python scripts/training/serve_policy.py --seed="${SEED}" --port="${PORT}" \
      policy:checkpoint --policy.dir="${CKPT_DIR}" --policy.config=mme_vla_suite > "${SERVER_LOG}" 2>&1 &
else
  EXTRA=()
  [[ "${COND}" == "normal" ]] && EXTRA+=(--motion-gpu="${GPU}")
  [[ "${COND}" == "swap" ]] && EXTRA+=(--swap-bank="${SWAP_BANK}")
  [[ "${DUMP_OBS_EVERY}" -gt 0 ]] && EXTRA+=(--dump-obs-dir="${DUMP_DIR}" --dump-obs-every="${DUMP_OBS_EVERY}")
  CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION="${POLICY_MEM_FRACTION}" UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    uv run --no-sync python "${MV_DIR}/serve_policy_mv.py" --cond="${COND}" --split="${SPLIT}" --seed="${SEED}" --port="${PORT}" \
      --ckpt="${CKPT_DIR}" --tasks="${TASKS_ALL}" --ep-start="${EP_START}" --ep-stride="${EP_STRIDE}" --ep-count="${EP_COUNT}" \
      --probe-out="${PROBE_OUT}" "${EXTRA[@]}" > "${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!
echo "server pid=${SERVER_PID} log=${SERVER_LOG}"
cleanup() { if kill -0 "${SERVER_PID}" 2>/dev/null; then kill "${SERVER_PID}"; sleep 2; kill -9 "${SERVER_PID}" 2>/dev/null || true; fi; }
trap cleanup EXIT
for _ in $(seq 1 600); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "错误: server 提前退出，见 ${SERVER_LOG}" >&2; tail -30 "${SERVER_LOG}" >&2; exit 1; fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then break; fi
  sleep 2
done
echo "server 端口就绪 $(date +%T)"
# ── eval（前台，micromamba robomme；GLIBC_TUNABLES 见 docs/training-doc/tic-vulkan-makeenv/result.md，单进程实测 30 集、外推 147）──
# eval stdout 同时落 EVAL_LOG（MV_SHARD_SUMMARY 要从里面数 MV_EP_DONE）；不能用 `wait`——它会等到 server 后台进程，永远不返回
set +e
( cd "${MV_DIR}/robomme" && CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=8192}" "${ROBOMME_PY}" eval.py --args.port="${PORT}" --args.model_seed="${SEED}" \
    --args.policy_name="${POLICY_NAME}" --args.model_ckpt_id="${CKPT_ID}" --args.only_tasks="${TASKS_ALL}" \
    --args.episode_start="${EP_START}" --args.max_episodes="${EP_COUNT}" --args.episode_stride="${EP_STRIDE}" \
    --args.dataset="${SPLIT}" --args.save_dir="${V1_STORE}/evaluation" ) 2>&1 | tee "${EVAL_LOG}"
RC=${PIPESTATUS[0]}
set -e
echo "EVAL_RC=${RC} end=$(date '+%F %T')"
UV_LINK_MODE=copy uv run --no-sync python - "${SERVER_LOG}" <<'PYEOF' 2>&1 | grep -v "dev-dependencies" || true
import re, sys, statistics as st
ab, inf = [], []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"TIMING add_buffer_ms=([\d.]+) frames=(\d+)", line)
    if m: ab.append((float(m.group(1)), int(m.group(2))))
    m = re.search(r"TIMING infer_ms=([\d.]+)", line)
    if m: inf.append(float(m.group(1)))
later = [a for a, f in ab if f <= 16]; first = [a for a, f in ab if f > 16]
def q(x): return f"n={len(x)} mean={st.mean(x):.0f} median={st.median(x):.0f} p90={sorted(x)[int(0.9*len(x))-1] if len(x)>=10 else float('nan'):.0f} max={max(x):.0f}" if x else "n=0"
print(f"TIMING_SUMMARY add_buffer(<=16帧) {q(later)} | add_buffer(首批>16帧) {q(first)} | infer {q(inf)}")
PYEOF
N_DONE=$(grep -c '^MV_EP_DONE ' "${EVAL_LOG}" 2>/dev/null || true); N_DONE=${N_DONE:-0}
N_INF=$(grep -c '^MV_INFER ' "${SERVER_LOG}" 2>/dev/null || true); N_INF=${N_INF:-0}
N_BEGIN=$(grep -c '^MV_EPISODE_BEGIN ' "${SERVER_LOG}" 2>/dev/null || true); N_BEGIN=${N_BEGIN:-0}
PROG="${V1_STORE}/evaluation/${POLICY_NAME}/ckpt${CKPT_ID}/${SEED_SEG}/progress.json"
N_PROG=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(sum(len(v) for v in d.values()))" "${PROG}" 2>/dev/null || echo 0)
echo "MV_SHARD_SUMMARY shard=w${K} cond=${COND} split=${SPLIT} seed=${SEED} expected=${EXPECTED} progress=${N_PROG} ep_done=${N_DONE} episode_begin=${N_BEGIN} infers=${N_INF} eval_rc=${RC} dirty=${DIRTY} progress_json=${PROG}"
echo "EXIT_CODE=${RC}"
exit "${RC}"
