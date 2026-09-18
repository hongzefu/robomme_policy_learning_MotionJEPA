#!/usr/bin/env bash
# 跑一个分片的 test/primary 候选：起一次 policy server，分块多次起客户端，收尾验收。
# 用法：run_shard.sh <运行名> <checkpoint绝对路径> <分片计划json> [策略seed=7] [分片标识]
# 可覆盖：EVAL_TIMEOUT（每块客户端超时秒，默认 7200）、CHUNK_EPISODES（每块最多新评几集，默认 20）
#        ALLOW_RESUME=1（运行目录已存在时继续跑，用于被 Slurm 砍掉后重提）
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN_NAME="${1:?需要运行名}"
CKPT="${2:?需要 checkpoint 绝对路径}"
PLAN="${3:?需要分片计划 json}"
SEED="${4:-7}"
SHARD_TAG="${5:-${SLURM_ARRAY_TASK_ID:-s0}}"
POLICY=perceptual-framesamp-modul-8frame-8x8
MAX_STEPS=2000
EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"
CHUNK_EPISODES="${CHUNK_EPISODES:-20}"
# Vulkan 静态 TLS：每次 make_env+close_env 净泄漏 64 字节，默认 512 档第 28 次必崩
# （docs/training-doc/tic-vulkan-makeenv/result.md，崩溃轮 = 19 + tls/64）。
# 分块已经把单进程压到 <=20 次，这里再抬一档做纵深防御；只给客户端，server 不建 Vulkan Context。
TLS_TUNABLE="glibc.rtld.optional_static_tls=65536"

[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$SEED" =~ ^[0-9]+$ && "$SHARD_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$EVAL_TIMEOUT" =~ ^[1-9][0-9]*$ && "$CHUNK_EPISODES" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$CKPT" == /* && -d "$CKPT/params" && -f "$CKPT/assets/robomme/norm_stats.json" ]] || { echo 'checkpoint 路径无效'; exit 2; }
[[ -f "$PLAN" ]] || { echo "分片计划不存在：$PLAN"; exit 2; }
CKPT_ID="$(basename "$CKPT")"
[[ "$CKPT_ID" == 59999 ]] || { echo '本轮只验收 59999'; exit 2; }
RUN_DIR="$EVAL_REPO/v1-store/evaluation/$RUN_NAME/$SHARD_TAG"
# 防覆盖判据是「已有评测结果」而不是「目录存在」：sbatch 的 start_samplers 会先把
# $RUN_DIR/records 建出来（采样必须覆盖起跑段），拿目录存在当判据会让每个 job 一起跑就自杀。
RESULT_ROOT="$RUN_DIR/$POLICY/ckpt$CKPT_ID/seed$SEED"
if [[ -e "$RESULT_ROOT/progress.json" && "${ALLOW_RESUME:-0}" != 1 ]]; then
    echo "运行结果已存在：$RESULT_ROOT/progress.json（要续跑请显式 ALLOW_RESUME=1）"; exit 2
fi
[[ -z "$(git -C "$EVAL_REPO" status --porcelain)" ]] || { echo '必须从 clean HEAD 启动'; exit 2; }
command -v uv
check_local_gpu() {
    [[ -z "${SLURM_JOB_ID:-}" ]] || return 0
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo '本机必须显式选择 GPU'; return 2; }
    local gpu_pids
    gpu_pids="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" || return 2
    if [[ -n "$gpu_pids" ]]; then
        echo "所选 GPU 已被占用，停止起跑：$gpu_pids"
        return 2
    fi
    echo "LOCAL_GPU_IDLE=PASS device=$CUDA_VISIBLE_DEVICES"
}
check_local_gpu
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
printf 'REPO=%s SLURM_JOB_ID=%s ARRAY_TASK=%s SHARD=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$EVAL_REPO" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-none}" "$SHARD_TAG" "${CUDA_VISIBLE_DEVICES:-}"
[[ "${ROBOMME_GPU_RASTER:-0}" == 0 && "${SAPIEN_DISABLE_RAY_TRACING:-0}" == 0 ]] || { echo '本轮仅允许原生渲染'; exit 2; }
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    [[ "$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u)" == Default ]] || { echo '需要 --gpu_cmode=shared'; exit 2; }
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo '分配 GPU 已有计算进程，停止'; exit 2; }
fi
PLAN_EPISODES="$("${CLIENT_UV[@]}" -c "
import json,sys
plan=json.load(open(sys.argv[1]))
print(sum(len(v) for v in plan['groups'].values()))
" "$PLAN")"
echo "PLAN=$PLAN episodes=$PLAN_EPISODES chunk=$CHUNK_EPISODES"
"${CLIENT_UV[@]}" scripts/evaluation/verify_sources.py | tee "$RUN_DIR/sources.log"
PROBE_TASK="$("${CLIENT_UV[@]}" -c "
import json,sys
plan=json.load(open(sys.argv[1]))
print(sorted(plan['groups'])[0].split('/')[0])
" "$PLAN")"
timeout --kill-after=15s 180s "${CLIENT_UV[@]}" scripts/evaluation/render_probe.py "$RUN_DIR/render" "$PROBE_TASK"
JAX_PLATFORMS=cpu "${SERVER_UV[@]}" scripts/evaluation/check_checkpoint.py "$CKPT" | tee "$RUN_DIR/checkpoint.log"

# 端口：按 array 下标错开基准，避开同节点多片同时探到同一个空闲端口的竞态窗口
PORT=$((18011 + (${SLURM_ARRAY_TASK_ID:-0} % 100) * 7))
while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
    PORT=$((PORT + 1))
    (( PORT < 18811 )) || exit 2
done
check_local_gpu
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

# healthz 通过还不够：得确认监听这个端口的就是自己的 server。
# 历史事故：同节点两片同时探到 18011 空闲，其中一片连上了别人的服务，12 秒 EXIT_CODE=0、跑了 0 集
# （docs/training-doc/eval-official-framesamp-context/result.md 的 w1/w3）。
# ⚠ 必须查整棵进程树：$! 拿到的是 wrapper，实际 bind 端口的是孙进程
#   （env -> uv -> python，本机实测 wrapper 1066086 / uv 1066093 / python 1066100），
#   socket fd 不在 wrapper 的 /proc/<pid>/fd 里。
collect_tree() {
    local pid="$1" child kids
    printf '%s\n' "$pid"
    kids="$(cat "/proc/$pid/task/$pid/children" 2>/dev/null)"
    [[ -n "$kids" ]] || kids="$(ps -o pid= --ppid "$pid" 2>/dev/null)"
    for child in $kids; do
        collect_tree "$child"
    done
}
HEXPORT=$(printf '%04X' "$PORT")
SERVER_TREE="$(collect_tree "$SERVER_PID" | tr '\n' ' ')"
OWN=0
for INODE in $(awk -v p=":$HEXPORT" '$2 ~ p"$" && $4=="0A" {print $10}' /proc/net/tcp /proc/net/tcp6 2>/dev/null); do
    for P in $SERVER_TREE; do
        if ls -l "/proc/$P/fd" 2>/dev/null | grep -q "socket:\[$INODE\]"; then OWN=1; break 2; fi
    done
done
[[ "$OWN" == 1 ]] || {
    echo "端口 $PORT 的监听者不在本次 server 进程树（根 PID=$SERVER_PID，树=$SERVER_TREE），停止"
    exit 2
}
echo "SERVER_READY port=$PORT pid=$SERVER_PID own=1 tree=$SERVER_TREE"

# 分块：每块客户端进程只新评 CHUNK_EPISODES 集，靠 progress.json 续跑衔接。
# 这样任何一个进程的 make_env 次数都远低于 Vulkan 静态 TLS 的 27 轮红线。
CHUNKS=$(( (PLAN_EPISODES + CHUNK_EPISODES - 1) / CHUNK_EPISODES ))
for ((CHUNK=1; CHUNK<=CHUNKS; CHUNK++)); do
    echo "CHUNK_START $CHUNK/$CHUNKS"
    timeout --signal=TERM --kill-after=20s "${EVAL_TIMEOUT}s" \
        env "GLIBC_TUNABLES=$TLS_TUNABLE" "${CLIENT_UV[@]}" examples/robomme/eval.py \
        --args.host=127.0.0.1 --args.port="$PORT" --args.model_seed="$SEED" \
        --args.policy_name="$POLICY" --args.model_ckpt_id="$CKPT_ID" \
        --args.max_steps="$MAX_STEPS" --args.save_dir="$RUN_DIR" \
        --args.episode_plan="$PLAN" --args.shard_tag="$SHARD_TAG" \
        --args.max_new_episodes="$CHUNK_EPISODES" \
        2>&1 | tee -a "$RUN_DIR/client.log"
    echo "CHUNK_DONE $CHUNK/$CHUNKS"
done

"${CLIENT_UV[@]}" scripts/evaluation/check_shard.py "$RUN_DIR" "$POLICY" "$CKPT_ID" "$SEED" "$PLAN"
echo "SHARD_DONE run=$RUN_NAME shard=$SHARD_TAG episodes=$PLAN_EPISODES"
