#!/usr/bin/env bash
# 从本仓库启动策略服务与仿真客户端，只清理本次服务进程。
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN_NAME="${1:?需要全新运行名}"
CKPT="${2:?需要 checkpoint 绝对路径}"
TASK="${3:?需要任务名}"
EPISODES="${4:?需要 episode 上限}"
SEED="${5:-7}"
POLICY=perceptual-framesamp-modul-8frame-8x8
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$TASK" =~ ^[a-zA-Z0-9]+$ && "$EPISODES" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || exit 2
[[ "$CKPT" == /* && -d "$CKPT/params" && -f "$CKPT/assets/robomme/norm_stats.json" ]] || { echo 'checkpoint 路径无效'; exit 2; }
CKPT_ID="$(basename "$CKPT")"
[[ "$CKPT_ID" == 59999 ]] || { echo '本轮只验收 59999'; exit 2; }
RUN_DIR="$EVAL_REPO/v1-store/evaluation/$RUN_NAME"
[[ ! -e "$RUN_DIR" ]] || { echo "运行目录已存在：$RUN_DIR"; exit 2; }
[[ -z "$(git -C "$EVAL_REPO" status --porcelain)" ]] || { echo '必须从 clean HEAD 启动'; exit 2; }
command -v uv
cd "$EVAL_REPO"
mkdir -p "$RUN_DIR"
SERVER_PID=''
cleanup() {
    RC=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    date -u +END_UTC=%Y-%m-%dT%H:%M:%SZ
    echo "EXIT_CODE=$RC"
    exit "$RC"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -u +START_UTC=%Y-%m-%dT%H:%M:%SZ
hostname
git rev-parse HEAD
git submodule status
uv --version
nvidia-smi --query-gpu=name,uuid,memory.used,compute_mode --format=csv
printf 'REPO=%s SLURM_JOB_ID=%s SLURM_STEP_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$EVAL_REPO" "${SLURM_JOB_ID:-local}" "${SLURM_STEP_ID:-local}" "${CUDA_VISIBLE_DEVICES:-}"
[[ "${ROBOMME_GPU_RASTER:-0}" == 0 && "${SAPIEN_DISABLE_RAY_TRACING:-0}" == 0 ]] || { echo '本轮仅允许原生渲染'; exit 2; }
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    [[ "$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u)" == Default ]] || { echo '需要 --gpu_cmode=shared'; exit 2; }
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo '分配 GPU 已有计算进程，停止'; exit 2; }
fi
"${CLIENT_UV[@]}" scripts/evaluation/verify_sources.py | tee "$RUN_DIR/sources.log"
timeout --kill-after=15s 180s "${CLIENT_UV[@]}" scripts/evaluation/render_probe.py "$RUN_DIR/render" "$TASK"
JAX_PLATFORMS=cpu "${SERVER_UV[@]}" scripts/evaluation/check_checkpoint.py "$CKPT" | tee "$RUN_DIR/checkpoint.log"
PORT=18011
while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
    PORT=$((PORT + 1))
    (( PORT < 18111 )) || exit 2
done
"${SERVER_UV[@]}" scripts/training/serve_policy.py --seed="$SEED" --port="$PORT" policy:checkpoint \
    --policy.dir="$CKPT" --policy.config=mme_vla_suite > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
READY=0
for ((ATTEMPT=0; ATTEMPT<180; ATTEMPT++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 80 "$RUN_DIR/server.log"; exit 1; }
    if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then READY=1; break; fi
    sleep 3
done
[[ "$READY" == 1 ]] || { tail -n 80 "$RUN_DIR/server.log"; exit 1; }
echo "SERVER_READY port=$PORT pid=$SERVER_PID"
timeout --signal=TERM --kill-after=20s 1100s "${CLIENT_UV[@]}" examples/robomme/eval.py \
    --args.host=127.0.0.1 --args.port="$PORT" --args.model_seed="$SEED" \
    --args.policy_name="$POLICY" --args.model_ckpt_id="$CKPT_ID" \
    --args.only_tasks="$TASK" --args.max_episodes="$EPISODES" --args.save_dir="$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/client.log"
"${CLIENT_UV[@]}" scripts/evaluation/check_result.py "$RUN_DIR" "$POLICY" "$CKPT_ID" "$TASK" "$EPISODES" "$SEED"
