#!/usr/bin/env bash
# 共享队列 worker：在一个**既有** gpu-hold 作业里循环抢 demo 预生成分片并跑，直到队列扫完。
# 抢单原语逐字复制 gl_hold_pool.sh 的 `mkdir claims/<item>`（NFS 上服务端原子），只换队列目录、
# 清单（分片下标 0..N-1）与 body（srun 调 run_demo_shard.sh 而不是 gl_eval_shard.sbatch）。
# **不改 gl_hold_pool.sh / gl_hold_queue.sh**——那两个正跑着评测链，本轮不动它们的代码路径。
#
# 用法（登录节点）：gl_demo_pool.sh <jobid>
#   QUEUE_DIR（默认 $STORE/queue）、STORE（前缀库根；不给则由候选库身份推出）、SHARDS（默认 10）
#   MAX_NEW_ENVS（默认 20）、RETRIES（默认 1）、STEP_TIME（每个 step 的 --time，默认 04:00:00）
#   DRY_RUN=1（不 srun、不查 squeue，只打印会执行的命令并 sleep 1，用于本机并发抢单测试）
# 判定行：DEMO_WORKER_START job= node= / DEMO_CLAIM item= / DEMO_ITEM_DONE item= rc= / DEMO_WORKER_DONE job= ran= failed=
set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
JOB="${1:?需要既有作业 jobid}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
SHARDS="${SHARDS:-10}"
MAX_NEW_ENVS="${MAX_NEW_ENVS:-20}"
RETRIES="${RETRIES:-1}"
STEP_TIME="${STEP_TIME:-04:00:00}"
[[ "$SHARDS" =~ ^[1-9][0-9]*$ && "$MAX_NEW_ENVS" =~ ^[1-9][0-9]*$ && "$RETRIES" =~ ^[0-9]+$ ]] || exit 2
DRY="${DRY_RUN:-0}"
HEAD="$(git -C "$REPO" rev-parse HEAD)"
# clean-HEAD 闸：正式跑必须 clean；DRY_RUN 只测抢单逻辑、不跑任何预生成，允许带未提交改动
[[ "$DRY" == 1 || -z "$(git -C "$REPO" status --porcelain)" ]] || { echo '必须从 clean HEAD 启动'; exit 2; }

if [[ -z "${STORE:-}" ]]; then
    IDENTITY="$(python3 -c "
import json,sys
print(json.loads(open(sys.argv[1], encoding='utf-8').readline())['identity_sha256'][:12])
" "$REPO/third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl")"
    [[ "$IDENTITY" =~ ^[0-9a-f]{12}$ ]] || { echo "取候选库身份失败：$IDENTITY"; exit 2; }
    STORE="$REPO/v1-store/demo-prefix/binfill-$IDENTITY"
fi
[[ "$STORE" == /* ]] || { echo "STORE 必须是绝对路径：$STORE"; exit 2; }
QUEUE_DIR="${QUEUE_DIR:-$STORE/queue}"
[[ "$QUEUE_DIR" == /* ]] || { echo "QUEUE_DIR 必须是绝对路径：$QUEUE_DIR"; exit 2; }
LOG_DIR="$REPO/v1-store/logs/demo-pregen"
mkdir -p "$QUEUE_DIR/claims" "$QUEUE_DIR/done" "$LOG_DIR" "$STORE/logs"

# 等作业 RUNNING（起跑时还 PENDING 的作业也可先起 worker，它自己等）；DRY_RUN 跳过
until [[ "$DRY" == 1 ]] || squeue -j "$JOB" -h -o %T | grep -qx RUNNING; do sleep 60; done
if [[ "$DRY" == 1 ]]; then NODE="$(hostname)"; else NODE="$(squeue -j "$JOB" -h -o %N)"; fi
echo "DEMO_WORKER_START job=$JOB node=$NODE head=$HEAD store=$STORE queue=$QUEUE_DIR shards=$SHARDS $(date -u +%FT%TZ)"
RAN=0; FAILED=0
while true; do
    ITEM=""; K=""
    for ((I=0; I<SHARDS; I++)); do
        if mkdir "$QUEUE_DIR/claims/s$I" 2>/dev/null; then ITEM="s$I"; K="$I"; break; fi
    done
    [[ -n "$ITEM" ]] || break
    printf 'job=%s node=%s claimed=%s\n' "$JOB" "$NODE" "$(date -u +%FT%TZ)" > "$QUEUE_DIR/claims/$ITEM/owner"
    echo "DEMO_CLAIM item=$ITEM job=$JOB node=$NODE $(date -u +%FT%TZ)"
    LOG="$LOG_DIR/demo-$ITEM.log"
    T0="$(date -u +%FT%TZ)"
    if [[ "$DRY" == 1 ]]; then
        echo "DRY run_demo_shard.sh $K $STORE (SHARDS=$SHARDS MAX_NEW_ENVS=$MAX_NEW_ENVS RETRIES=$RETRIES)"
        sleep 1; RC=0
    else
        # -u MANIFEST：srun 默认把提交端环境整个带进 step，调用方环境里残留的 MANIFEST
        # 会污染下游（e9c46c6 踩过这个坑）。本脚本按分片把参数显式算好传入，step 里不需要也不能看见它。
        srun --jobid="$JOB" --account=chaijy2 --partition=spgpu --gpu_cmode=shared --overlap --exact \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-node=1 --time="$STEP_TIME" --chdir="$REPO" \
            /usr/bin/env -u ROBOMME_GPU_RASTER -u SAPIEN_DISABLE_RAY_TRACING -u MANIFEST PYTHONUNBUFFERED=1 \
            STORE="$STORE" SHARDS="$SHARDS" MAX_NEW_ENVS="$MAX_NEW_ENVS" RETRIES="$RETRIES" \
            /usr/bin/bash "$REPO/scripts/evaluation/run_demo_shard.sh" "$K" 2>&1 | tee -a "$LOG"
        RC=${PIPESTATUS[0]}
    fi
    printf 'rc=%s job=%s node=%s start=%s end=%s\n' "$RC" "$JOB" "$NODE" "$T0" "$(date -u +%FT%TZ)" \
        > "$QUEUE_DIR/done/$ITEM"
    echo "DEMO_ITEM_DONE item=$ITEM rc=$RC job=$JOB $(date -u +%FT%TZ)"
    RAN=$((RAN + 1)); [[ "$RC" == 0 ]] || FAILED=$((FAILED + 1))
done
echo "DEMO_WORKER_DONE job=$JOB ran=$RAN failed=$FAILED $(date -u +%FT%TZ)"
exit "$FAILED"
