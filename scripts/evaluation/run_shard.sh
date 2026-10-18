#!/usr/bin/env bash
# 跑一个分片的 test/primary 候选：起一次 policy server，分块多次起客户端，收尾验收。
# 用法：run_shard.sh <运行名> <checkpoint绝对路径> <分片计划json> [策略seed=7] [分片标识]
# 可覆盖：EVAL_TIMEOUT（每块客户端超时秒，默认 7200）、CHUNK_EPISODES（每块最多新评几集，默认 20）
#        ALLOW_RESUME=1（运行目录已存在时继续跑，用于被 Slurm 砍掉后重提）
#        POLICY（结果目录层级名，默认 perceptual-framesamp-modul-8frame-8x8）、POLICY_CONFIG（serve_policy 的训练 config 名，默认 mme_vla_suite）
#        EXPECT_CKPT_ID（只接受这个 step 目录名，默认 59999）、EPISODE_WALL_S（单集墙钟秒，不设则用 eval.py 默认 900）
#        SHARD_INDEX（srun 进既有作业时代替 SLURM_ARRAY_TASK_ID 决定默认分片标识与端口基准）
#        MMEVLA_MOTION_PROV_RELAX（motion 轮：sidecar provenance 放行的硬件键，见 motion_client.py）
#        DEMO_PREFIX_STORE（BinFill demo 前缀库根目录绝对路径，给了才注入；见 0922-binfill-demo-prefix-plan.md）
#        MMEVLA_ENC_CHUNK（SigLIP 编码分批尺寸，0/不设 = 一次编完，与改动前逐字等价）
#        MMEVLA_MOTION_OVERFLOW（raise|resample，默认 raise；补 demo 后 motion 轮须设 resample）
#        EXPECT_MOTION_STATS=1（本轮 motion run 缺 motion_stats.json 即判 FAIL，不许走 SKIP）
#   motion 轮示例：POLICY=perceptual-framesamp-modul-8frame-8x8-motion POLICY_CONFIG=mme_vla_suite_b128_80k EXPECT_CKPT_ID=50000
#                 EPISODE_WALL_S=2400 MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN_NAME="${1:?需要运行名}"
CKPT="${2:?需要 checkpoint 绝对路径}"
PLAN="${3:?需要分片计划 json}"
SEED="${4:-7}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
SHARD_TAG="${5:-s$SHARD_INDEX}"
POLICY="${POLICY:-perceptual-framesamp-modul-8frame-8x8}"
POLICY_CONFIG="${POLICY_CONFIG:-mme_vla_suite}"
EXPECT_CKPT_ID="${EXPECT_CKPT_ID:-59999}"
EPISODE_WALL_S="${EPISODE_WALL_S:-}"
MAX_STEPS=2000
EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"
CHUNK_EPISODES="${CHUNK_EPISODES:-20}"
# Vulkan 静态 TLS：每次 make_env+close_env 净泄漏 64 字节，默认 512 档第 28 次必崩
# （docs/training-doc/tic-vulkan-makeenv/result.md，崩溃轮 = 19 + tls/64）。
# 分块已经把单进程压到 <=20 次，这里再抬一档做纵深防御；只给客户端，server 不建 Vulkan Context。
TLS_TUNABLE="glibc.rtld.optional_static_tls=65536"

[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$SEED" =~ ^[0-9]+$ && "$SHARD_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$EVAL_TIMEOUT" =~ ^[1-9][0-9]*$ && "$CHUNK_EPISODES" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$POLICY" =~ ^[a-zA-Z0-9_.-]+$ && "$POLICY_CONFIG" =~ ^[a-zA-Z0-9_]+$ && "$EXPECT_CKPT_ID" =~ ^[0-9]+$ && "$SHARD_INDEX" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ -z "$EPISODE_WALL_S" || "$EPISODE_WALL_S" =~ ^[1-9][0-9]*$ ]] || exit 2
DEMO_PREFIX_STORE="${DEMO_PREFIX_STORE:-}"
if [[ -n "$DEMO_PREFIX_STORE" ]]; then
    [[ "$DEMO_PREFIX_STORE" == /* ]] || { echo "DEMO_PREFIX_STORE 必须是绝对路径：$DEMO_PREFIX_STORE"; exit 2; }
    [[ -f "$DEMO_PREFIX_STORE/index.json" ]] || { echo "demo 前缀库缺 index.json：$DEMO_PREFIX_STORE"; exit 2; }
fi
[[ -z "${MMEVLA_ENC_CHUNK:-}" || "$MMEVLA_ENC_CHUNK" =~ ^[0-9]+$ ]] || { echo 'MMEVLA_ENC_CHUNK 必须是非负整数'; exit 2; }
[[ -z "${MMEVLA_MOTION_OVERFLOW:-}" || "$MMEVLA_MOTION_OVERFLOW" =~ ^(raise|resample)$ ]] || { echo 'MMEVLA_MOTION_OVERFLOW 只能是 raise / resample'; exit 2; }
[[ -z "${EXPECT_MOTION_STATS:-}" || "$EXPECT_MOTION_STATS" =~ ^[01]$ ]] || { echo 'EXPECT_MOTION_STATS 只能是 0 / 1'; exit 2; }
# 这三个要被子进程读到：前两个由 server 侧的 framesamp_memory.py / policy.py 读，
# EXPECT_MOTION_STATS 由客户端 uv 起的 check_shard.py 读。经 env 传进来时本已 exported，这里显式化以防漏。
[[ -z "${MMEVLA_ENC_CHUNK:-}" ]] || export MMEVLA_ENC_CHUNK
[[ -z "${MMEVLA_MOTION_OVERFLOW:-}" ]] || export MMEVLA_MOTION_OVERFLOW
[[ -z "${EXPECT_MOTION_STATS:-}" ]] || export EXPECT_MOTION_STATS
[[ "$CKPT" == /* && -d "$CKPT/params" && -f "$CKPT/assets/robomme/norm_stats.json" ]] || { echo 'checkpoint 路径无效'; exit 2; }
[[ -f "$PLAN" ]] || { echo "分片计划不存在：$PLAN"; exit 2; }
CKPT_ID="$(basename "$CKPT")"
[[ "$CKPT_ID" == "$EXPECT_CKPT_ID" ]] || { echo "本轮只验收 step $EXPECT_CKPT_ID，得到 $CKPT_ID"; exit 2; }
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
# 递归收集进程树（wrapper → uv → python → sidecar uv → sidecar python）；端口归属断言与收尾清理共用
collect_tree() {
    local pid="$1" child kids
    printf '%s\n' "$pid"
    kids="$(cat "/proc/$pid/task/$pid/children" 2>/dev/null)"
    [[ -n "$kids" ]] || kids="$(ps -o pid= --ppid "$pid" 2>/dev/null)"
    for child in $kids; do
        collect_tree "$child"
    done
}
cleanup() {
    RC=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        # motion 开启时 sidecar 是 server 的孙进程，只 kill wrapper 会留孤儿：先收树、再从叶到根逐个 TERM
        local P
        for P in $(collect_tree "$SERVER_PID" | tac); do
            kill "$P" 2>/dev/null || true
        done
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
PORT_SLOT="$SHARD_INDEX"; [[ "$PORT_SLOT" =~ ^[0-9]+$ ]] || PORT_SLOT=$(( $(printf '%s' "$PORT_SLOT" | cksum | cut -d' ' -f1) % 100 ))
PORT=$((18011 + (PORT_SLOT % 100) * 7))
while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
    PORT=$((PORT + 1))
    (( PORT < 18811 )) || exit 2
done
check_local_gpu
# motion sidecar 的卡号跟随本进程可见的第一张卡（Slurm 单卡 step 里是 0；本机冒烟是 CUDA_VISIBLE_DEVICES 所选卡）
export MMEVLA_MOTION_ONLINE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
[[ -n "$MMEVLA_MOTION_ONLINE_GPU" ]] || export MMEVLA_MOTION_ONLINE_GPU=0
echo "EVAL_PARAMS policy=$POLICY config=$POLICY_CONFIG ckpt=$CKPT_ID seed=$SEED max_steps=$MAX_STEPS wall=${EPISODE_WALL_S:-default} online_gpu=$MMEVLA_MOTION_ONLINE_GPU prov_relax=${MMEVLA_MOTION_PROV_RELAX:-none}"
echo "EVAL_PARAMS_DEMO demo_prefix_store=${DEMO_PREFIX_STORE:-none} enc_chunk=${MMEVLA_ENC_CHUNK:-0} motion_overflow=${MMEVLA_MOTION_OVERFLOW:-raise} expect_motion_stats=${EXPECT_MOTION_STATS:-0}"
"${SERVER_UV[@]}" scripts/training/serve_policy.py --seed="$SEED" --port="$PORT" policy:checkpoint \
    --policy.dir="$CKPT" --policy.config="$POLICY_CONFIG" > "$RUN_DIR/server.log" 2>&1 &
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
        ${EPISODE_WALL_S:+--args.episode_wall_s="$EPISODE_WALL_S"} \
        ${DEMO_PREFIX_STORE:+--args.demo_prefix_store="$DEMO_PREFIX_STORE"} \
        2>&1 | tee -a "$RUN_DIR/client.log"
    echo "CHUNK_DONE $CHUNK/$CHUNKS"
done

"${CLIENT_UV[@]}" scripts/evaluation/check_shard.py "$RUN_DIR" "$POLICY" "$CKPT_ID" "$SEED" "$PLAN"
echo "SHARD_DONE run=$RUN_NAME shard=$SHARD_TAG episodes=$PLAN_EPISODES"
