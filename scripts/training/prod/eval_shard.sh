#!/usr/bin/env bash
# 正式评估单分片驱动（留档 docs/training-doc/eval-*/）：起 policy server（uv venv；带 motion 快照的 run 由
# create_trained_policy 自动拉起 motion sidecar）→ 跑 examples/robomme/eval.py（micromamba robomme 环境）→ 收 server。
# 骨架沿 scripts/training/tests/run_t3_eval_obs.sh；两点不同：
#   1. policy / sidecar / 仿真三者同一张卡（GPU）——链内三进程串行等待、不并发，同卡不损吞吐；sidecar 卡号经
#      MMEVLA_MOTION_ONLINE_GPU 覆盖 run 快照写死的 motion.online_gpu（否则多片 sidecar 全挤一张卡，历史实测 1.42 s/窗 → 3.5 s/窗）。
#   2. 按 episode 分片：eval.py 的 --args.episode_start / --args.max_episodes / --args.episode_stride。两种口径：
#      task 口径（旧）：TASK=<任务> K=<片号>，评 [K*EP_COUNT, (K+1)*EP_COUNT) 连续区间，分片名 <TASK>-<K>；
#      stride 口径（负载均衡）：TASKS=<全 4 任务> SHARD_ID=w<K> EP_STRIDE=8 EP_START=K EP_COUNT=0，worker K 评每任务 ep ∈ {K, K+8, …}，
#      任务级耗时差异（Video 类每集约 7 窗、Button 类 20–30 窗）被均摊到每个 worker（留档 eval-official-framesamp-context/）。
# run 根两种口径：本仓库 run 带 history_config.resolved.yaml（快照 + provenance，严格恢复）；官方 checkpoint 只有 history_config.txt
# （create_trained_policy 的旧兼容路径，非严格恢复），两者任一存在即可。
# 用法：TASK=ButtonUnmask K=0 GPU=0 PORT=8031 [EP_COUNT=25] [EP_START=K*EP_COUNT] [RUN_NAME=awsprod40k-b128-motion] [CKPT_ID=39999]
#       stride 口径改为 TASKS=<a,b,c,d> SHARD_ID=w<K> K=<K> EP_STRIDE=<worker 数> [EP_START=K] [EP_COUNT=0]（其余同）
#       [CKPT_DIR=$TRAIN_RUNS/mme_vla_suite_b128/$RUN_NAME/$CKPT_ID] [SEED=42] [LOG_PREFIX=ev40k] [POLICY_MEM_FRACTION=0.4] [ROBOMME_PY=…]
#       bash scripts/training/prod/eval_shard.sh
#   结果   v1-store/evaluation/<RUN_NAME>-<SHARD>/ckpt<CKPT_ID>/seed<SEED>/{progress.json,log.json,videos/}（SHARD = <TASK>-<K> 或 w<K>）
#   server v1-store/logs/<LOG_PREFIX>-<SHARD>.server.log（含 TIMING add_buffer_ms / infer_ms 行）
#   本脚本 stdout 由 eval_all_shards.sh 经 tee 落 v1-store/logs/<LOG_PREFIX>-<SHARD>.log，结束写 EXIT_CODE=。
#   断点续评：eval.py 按 progress.json 跳过已评集，同参数重跑即续。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"
v1_require_models 1
TASKS="${TASKS:-${TASK:-}}"                  # 逗号分隔任务列表；默认 = 单任务 TASK（task 口径）
: "${TASKS:?必须设置 TASK（task 口径）或 TASKS（stride 口径）}"
: "${K:?必须设置 K（分片号 / worker 号）}"
: "${GPU:?必须设置 GPU}"
: "${PORT:?必须设置 PORT}"
EP_STRIDE="${EP_STRIDE:-1}"
if [[ "${EP_STRIDE}" -gt 1 ]]; then
  EP_COUNT="${EP_COUNT:-0}"                  # stride 口径：0 = 不设集号上界（跑到 env.num_episodes）
  EP_START="${EP_START:-${K}}"               # worker K 取 ep ∈ {K, K+EP_STRIDE, …}
else
  EP_COUNT="${EP_COUNT:-25}"
  EP_START="${EP_START:-$((K * EP_COUNT))}"  # task 口径不变
fi
RUN_NAME="${RUN_NAME:-awsprod40k-b128-motion}"
CKPT_ID="${CKPT_ID:-39999}"
CKPT_DIR="${CKPT_DIR:-${TRAIN_RUNS}/mme_vla_suite_b128/${RUN_NAME}/${CKPT_ID}}"
SEED="${SEED:-42}"
LOG_PREFIX="${LOG_PREFIX:-ev40k}"
POLICY_MEM_FRACTION="${POLICY_MEM_FRACTION:-0.4}"
# 环境 B 的 robomme 仿真环境（run_t3_eval_obs.sh 默认的 $HOME/micromamba 在本机不存在）
ROBOMME_PY="${ROBOMME_PY:-/scratch/hongze/micromamba/envs/robomme/bin/python}"
SHARD="${SHARD_ID:-${TASK:+${TASK}-${K}}}"
: "${SHARD:?必须设置 SHARD_ID（stride 口径，如 w3）或 TASK（task 口径）}"
POLICY_NAME="${RUN_NAME}-${SHARD}"
SERVER_LOG="${LOGS_DIR}/${LOG_PREFIX}-${SHARD}.server.log"
RUN_ROOT="$(dirname "${CKPT_DIR}")"
[[ -d "${CKPT_DIR}/params" ]] || { echo "错误: checkpoint 缺 params: ${CKPT_DIR}" >&2; exit 1; }
[[ -x "${ROBOMME_PY}" ]] || { echo "错误: robomme 环境 python 不存在: ${ROBOMME_PY}" >&2; exit 1; }
if [[ -f "${RUN_ROOT}/history_config.resolved.yaml" ]]; then
  # 本仓库 run（快照口径）：motion 开启态会自动拉起 sidecar，先核 sidecar 依赖
  if grep -Eq '^[[:space:]]*enabled:[[:space:]]*true' "${RUN_ROOT}/history_config.resolved.yaml"; then
    [[ -x "${V1_STORE}/venvs/wan/bin/python" ]] || { echo "错误: sidecar venv 不存在: ${V1_STORE}/venvs/wan" >&2; exit 1; }
    [[ -f "${V1_STORE}/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt" ]] || { echo "错误: sidecar encoder checkpoint 不存在（v1-store/external/motionjepa/wan-v8-filter10-72ep-a）" >&2; exit 1; }
    RUN_KIND="snapshot+motion"
  else
    RUN_KIND="snapshot"
  fi
elif [[ -f "${RUN_ROOT}/history_config.txt" ]]; then
  RUN_KIND="legacy(history_config.txt=$(tr -d '\n' < "${RUN_ROOT}/history_config.txt"))"
else
  echo "错误: run 根 ${RUN_ROOT} 既无 history_config.resolved.yaml 也无 history_config.txt" >&2; exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain | head -c 1)" ]]; then
  echo "错误: 工作区不干净——正式评估必须从 clean HEAD 起（AGENTS 12）" >&2; exit 1
fi
mkdir -p "${LOGS_DIR}" "${V1_STORE}/evaluation"
echo "=== EVAL_SHARD shard=${SHARD} HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD) ckpt=${CKPT_DIR} run_kind=${RUN_KIND} gpu=${GPU} port=${PORT} seed=${SEED} tasks=${TASKS} ep_start=${EP_START} ep_stride=${EP_STRIDE} ep_count=${EP_COUNT} policy_name=${POLICY_NAME} mem_fraction=${POLICY_MEM_FRACTION} start=$(date '+%F %T') ==="
cd "${REPO_ROOT}"
# ── policy server（后台）：CUDA_VISIBLE_DEVICES 只作用于 policy 进程；sidecar 子进程的卡号由 MMEVLA_MOTION_ONLINE_GPU 给（绝对卡号）──
CUDA_VISIBLE_DEVICES="${GPU}" MMEVLA_MOTION_ONLINE_GPU="${GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION="${POLICY_MEM_FRACTION}" UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  uv run --no-sync python scripts/training/serve_policy.py --seed="${SEED}" --port="${PORT}" \
    policy:checkpoint --policy.dir="${CKPT_DIR}" --policy.config=mme_vla_suite >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "server pid=${SERVER_PID} log=${SERVER_LOG}"
cleanup() { if kill -0 "${SERVER_PID}" 2>/dev/null; then kill "${SERVER_PID}"; sleep 2; kill -9 "${SERVER_PID}" 2>/dev/null || true; fi; }
trap cleanup EXIT
# 等端口就绪（加载 checkpoint + jit + sidecar 握手，数分钟；上限 20 min）；server 先死则立即退出
for _ in $(seq 1 600); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "错误: server 提前退出，见 ${SERVER_LOG}" >&2; tail -30 "${SERVER_LOG}" >&2; exit 1; fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then break; fi
  sleep 2
done
echo "server 端口就绪 $(date +%T)"
# ── eval（前台，micromamba robomme 环境；ManiSkill 渲染卡 = sapien.Device("cuda") = 可见集第 0 张，即 GPU）──
RC=0
( cd examples/robomme && CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${ROBOMME_PY}" eval.py --args.port="${PORT}" --args.model_seed="${SEED}" \
    --args.policy_name="${POLICY_NAME}" --args.model_ckpt_id="${CKPT_ID}" --args.only_tasks="${TASKS}" \
    --args.episode_start="${EP_START}" --args.max_episodes="${EP_COUNT}" --args.episode_stride="${EP_STRIDE}" --args.save_dir="${V1_STORE}/evaluation" ) 2>&1 || RC=$?
echo "EVAL_RC=${RC} end=$(date '+%F %T')"
# 汇总 TIMING（同 run_t3_eval_obs.sh）
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
echo "EXIT_CODE=${RC}"
exit "${RC}"
