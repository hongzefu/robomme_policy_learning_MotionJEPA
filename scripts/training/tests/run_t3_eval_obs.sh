#!/usr/bin/env bash
# T3_EVAL_OBS 单侧驱动（motion-memory-plan.md 四节表二 / 七节 S3）：起 policy server（uv venv）→ 跑 examples/robomme/eval.py（micromamba robomme 环境）→ 收 server。
# 用法：SIDE=closed|open [TASKS=…] [MAX_EPISODES=10] [RUN_SUFFIX=-a] [OVERWRITE=1] [PORT=…] [POLICY_GPU=…] [SIM_GPU=…] bash scripts/training/tests/run_t3_eval_obs.sh
#   分片并行：同一侧按 TASKS 拆成多个进程时给不同 RUN_SUFFIX（结果目录 v1-store/evaluation/<RUN><RUN_SUFFIX>/…）与 PORT，事后合并。
#   closed：checkpoint v1-store/train-runs/motion-t3-closed-final/999（关闭态，不起 sidecar）
#   open  ：checkpoint v1-store/train-runs/motion-t3-open/mme_vla_suite/motion-t3-open/999（开启态，policy 构造时自动起 sidecar 于 motion.online_gpu=1）
# 两侧同任务（v1 四任务）/ 同 episode（eval.py 现行 50 集/任务）/ 同 seed；结果目录 v1-store/evaluation/<SIDE 的 run 名>/ckpt999/seed<SEED>/。
# server 日志（含 TIMING add_buffer_ms / infer_ms 行）落 v1-store/logs/motion-t3-<SIDE>-eval.server.log；eval 日志落 …-eval.log。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"
# 本脚本要跑 policy（pi05_base + tokenizer）与 open 侧的 motion sidecar（encoder ckpt），
# 此前 source 了 paths.sh 却没调任何前置；补上内容级资产校验（ASSETS_LOCK.json，cheap 档）
v1_require_models 1
: "${SIDE:?必须设置 SIDE=closed|open}"
SEED="${SEED:-42}"
PORT="${PORT:-8021}"
TASKS="${TASKS:-ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap}"
POLICY_GPU="${POLICY_GPU:-0}"
SIM_GPU="${SIM_GPU:-1}"
MAX_EPISODES="${MAX_EPISODES:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
OVERWRITE="${OVERWRITE:-0}"
ROBOMME_PY="${ROBOMME_PY:-${HOME}/micromamba/envs/robomme/bin/python}"
case "${SIDE}" in
  closed) RUN=motion-t3-closed; CKPT="${V1_STORE}/train-runs/motion-t3-closed-final/999" ;;
  open)   RUN=motion-t3-open;   CKPT="${V1_STORE}/train-runs/motion-t3-open/mme_vla_suite/motion-t3-open/999" ;;
  *) echo "SIDE 只能是 closed|open" >&2; exit 2 ;;
esac
[[ -d "${CKPT}/params" ]] || { echo "错误: checkpoint 缺 params: ${CKPT}" >&2; exit 1; }
[[ -f "$(dirname "${CKPT}")/history_config.resolved.yaml" ]] || { echo "错误: run 根缺 history_config.resolved.yaml" >&2; exit 1; }
[[ -x "${ROBOMME_PY}" ]] || { echo "错误: robomme 环境 python 不存在: ${ROBOMME_PY}" >&2; exit 1; }
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain | head -c 1)" ]]; then
  echo "错误: 工作区不干净——评估必须从 clean HEAD 起（AGENTS 12）" >&2; exit 1
fi
SERVER_LOG="${LOGS_DIR}/${RUN}${RUN_SUFFIX}-eval.server.log"
EVAL_LOG="${LOGS_DIR}/${RUN}${RUN_SUFFIX}-eval.log"
EXTRA_ARGS=()
[[ "${MAX_EPISODES}" -gt 0 ]] && EXTRA_ARGS+=("--args.max_episodes=${MAX_EPISODES}")
[[ "${OVERWRITE}" == "1" ]] && EXTRA_ARGS+=("--args.overwrite")
echo "=== T3_EVAL_OBS side=${SIDE} run=${RUN}${RUN_SUFFIX} HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD) ckpt=${CKPT} port=${PORT} seed=${SEED} tasks=${TASKS} max_episodes=${MAX_EPISODES} overwrite=${OVERWRITE} ==="
cd "${REPO_ROOT}"
# ── policy server（后台；开启态由 create_trained_policy 自动起 sidecar，CUDA_VISIBLE_DEVICES 只作用于 policy 进程，sidecar 用 motion.online_gpu）──
CUDA_VISIBLE_DEVICES="${POLICY_GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION="${POLICY_MEM_FRACTION:-0.6}" UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  uv run --no-sync python scripts/training/serve_policy.py --seed="${SEED}" --port="${PORT}" \
    policy:checkpoint --policy.dir="${CKPT}" --policy.config=mme_vla_suite >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "server pid=${SERVER_PID} log=${SERVER_LOG}"
cleanup() { if kill -0 "${SERVER_PID}" 2>/dev/null; then kill "${SERVER_PID}"; sleep 2; kill -9 "${SERVER_PID}" 2>/dev/null || true; fi; }
trap cleanup EXIT
# 等端口就绪（server 启动含加载 checkpoint + jit，数分钟）；server 若先死则立即退出
for _ in $(seq 1 600); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "错误: server 提前退出，见 ${SERVER_LOG}" >&2; tail -30 "${SERVER_LOG}" >&2; exit 1; fi
  if uv run --no-sync python -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', ${PORT}))==0 else 1)" 2>/dev/null; then break; fi
  sleep 2
done
echo "server 端口就绪 $(date +%T)"
# ── eval（前台，micromamba robomme 环境；仿真在 SIM_GPU）──
( cd examples/robomme && CUDA_VISIBLE_DEVICES="${SIM_GPU}" PYTHONUNBUFFERED=1 "${ROBOMME_PY}" eval.py --args.port="${PORT}" --args.model_seed="${SEED}" \
    --args.policy_name="${RUN}${RUN_SUFFIX}" --args.model_ckpt_id=999 --args.only_tasks="${TASKS}" --args.save_dir="${V1_STORE}/evaluation" "${EXTRA_ARGS[@]}" ) 2>&1 | tee -a "${EVAL_LOG}"
RC="${PIPESTATUS[0]}"
echo "EVAL_RC=${RC}"
# 汇总 TIMING
uv run --no-sync python - "${SERVER_LOG}" <<'EOF'
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
EOF
exit "${RC}"
