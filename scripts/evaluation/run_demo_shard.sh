#!/usr/bin/env bash
# 跑一个分片的 BinFill demo 前缀预生成：clean-HEAD 闸 + 来源校验 + 分块循环 + 验收。
# 与 run_shard.sh 是兄弟脚本，但不起 policy server——预生成只用 planner，不碰模型。
# 用法：run_demo_shard.sh <分片下标> [前缀库根目录]
#   STORE（前缀库根，默认 $EVAL_REPO/v1-store/demo-prefix/binfill-<identity12>）
#   SHARDS（分片数，默认 10）、MAX_NEW_ENVS（每块最多建几个 env，默认 20）、RETRIES（单集重试次数，默认 1）
# 判定行：DEMO_SHARD_DONE shard=<k> rc=<rc>
set -uo pipefail
source "$(dirname "$0")/env.sh"
SHARD="${1:?需要分片下标}"
SHARDS="${SHARDS:-10}"
MAX_NEW_ENVS="${MAX_NEW_ENVS:-20}"
RETRIES="${RETRIES:-1}"
[[ "$SHARD" =~ ^[0-9]+$ && "$SHARDS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$MAX_NEW_ENVS" =~ ^[1-9][0-9]*$ && "$RETRIES" =~ ^[0-9]+$ ]] || exit 2
(( SHARD < SHARDS )) || { echo "分片下标 $SHARD 越界（共 $SHARDS 片）"; exit 2; }
# Vulkan 静态 TLS：每次 make_env+close_env 净泄漏 64 字节，默认第 28 次必崩
# （docs/training-doc/tic-vulkan-makeenv/result.md）。分块已把单进程压到 <= MAX_NEW_ENVS 次，这里再抬一档做纵深防御。
TLS_TUNABLE="glibc.rtld.optional_static_tls=65536"

[[ -z "$(git -C "$EVAL_REPO" status --porcelain)" ]] || { echo '必须从 clean HEAD 启动'; exit 2; }
cd "$EVAL_REPO"

IDENTITY="$("${CLIENT_UV[@]}" -c "
import json,sys
print(json.loads(open(sys.argv[1], encoding='utf-8').readline())['identity_sha256'][:12])
" "$EVAL_REPO/third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl")"
[[ "$IDENTITY" =~ ^[0-9a-f]{12}$ ]] || { echo "取候选库身份失败：$IDENTITY"; exit 2; }
STORE="${2:-${STORE:-$EVAL_REPO/v1-store/demo-prefix/binfill-$IDENTITY}}"
[[ "$STORE" == /* ]] || { echo "前缀库根必须是绝对路径：$STORE"; exit 2; }
mkdir -p "$STORE/logs"
LOG_DIR="$STORE/logs"

cleanup() {
    RC=$?
    trap - EXIT
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
nvidia-smi --query-gpu=name,uuid,memory.used,compute_mode --format=csv
printf 'REPO=%s SLURM_JOB_ID=%s SHARD=%s/%s STORE=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$EVAL_REPO" "${SLURM_JOB_ID:-local}" "$SHARD" "$SHARDS" "$STORE" "${CUDA_VISIBLE_DEVICES:-}"
[[ "${ROBOMME_GPU_RASTER:-0}" == 0 && "${SAPIEN_DISABLE_RAY_TRACING:-0}" == 0 ]] || { echo '本轮仅允许原生渲染'; exit 2; }
"${CLIENT_UV[@]}" scripts/evaluation/verify_sources.py | tee "$LOG_DIR/sources-s$SHARD.log"

# 分块：每个客户端进程最多新建 MAX_NEW_ENVS 个 env 就退出（rc=3 表示本片还没跑完），外层 while 续跑。
# 这样任何一个进程的 make_env 次数都远低于 Vulkan 静态 TLS 的 27 轮红线。
RC=3
for ((BLOCK=1; BLOCK<=40; BLOCK++)); do
    echo "DEMO_BLOCK_START $BLOCK shard=$SHARD"
    env "GLIBC_TUNABLES=$TLS_TUNABLE" "${CLIENT_UV[@]}" scripts/evaluation/pregen_binfill_demo.py gen \
        --out "$STORE" --shard "$SHARD" --shards "$SHARDS" \
        --max-new-envs "$MAX_NEW_ENVS" --retries "$RETRIES" \
        2>&1 | tee -a "$LOG_DIR/shard$SHARD.log"
    RC=${PIPESTATUS[0]}
    echo "DEMO_BLOCK_DONE $BLOCK shard=$SHARD rc=$RC"
    [[ "$RC" == 3 ]] || break      # 0 = 本片全部落定；其它 = 真失败
done

echo "DEMO_SHARD_DONE shard=$SHARD rc=$RC store=$STORE"
exit "$RC"
