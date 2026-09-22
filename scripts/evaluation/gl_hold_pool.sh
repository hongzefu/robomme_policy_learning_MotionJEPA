#!/usr/bin/env bash
# 共享队列 worker：在一个**既有** gpu-hold 作业里循环抢 (模型, 分片) 单元并跑，直到队列扫完（用户 2026-09-18：「使用动态的分配」）。
# 机制：队列目录含 items.txt（行格式 "<nomotion|motion> <shard_index>"，按「长任务先、短任务填尾」排好序）；抢单靠 NFS 上服务端原子的
# `mkdir claims/<model>-s<K>`，成功即归自己、失败即别人已抢；跑完写 done/<item>（rc、起止 UTC、作业、节点）。每轮重扫 items.txt，
# 后加入的 worker（如起跑时还在 PENDING 的作业）直接从当前未抢项接着抢。rc≠0 不自动重试，留给合并阶段按 retry_plan.json 处理。
# 「一致的调用」：两轮共用的一切写死在本脚本（EXPECT_CKPT_ID / EPISODE_WALL_S / EVAL_TIMEOUT / SEED / CHUNK_EPISODES / STEP_TIME /
# MMEVLA_MOTION_PROV_RELAX——无 motion 轮没有 sidecar、无人读取，为环境逐字相同照样传），两轮只差 RUN_NAME / CKPT / POLICY / POLICY_CONFIG。
# 每个单元仍走既有链：gl_hold_queue.sh → srun --jobid --overlap --exact → gl_eval_shard.sbatch → run_shard.sh。
# 用法（登录节点）：gl_hold_pool.sh <jobid>
#   QUEUE_DIR（默认 $REPO/v1-store/evaluation/primary700-ab50k-gl/queue）、RUN_SUFFIX（run_name 后缀，默认 -gl；冒烟用 -smoke-gl，两个 run_name 同时带上）、
#   CHUNK_EPISODES（默认 20）、DRY_RUN=1（不 srun、不查 squeue，只打印会执行的命令并 sleep 1，用于本机并发抢单测试）
#   MANIFEST：给出后改用 manifest 当待办清单（与 gl_eval_shard.sbatch 同一份文件、同一种六列 TAB 格式：
#             run_name / ckpt绝对路径 / policy / policy_config / expect_ckpt_id / shard_index），
#             单元名 = <run_name>-s<shard_index>，四个差异值与 step 逐行取，不再走下面写死的 nomotion / motion 两个 case。
#             这样同一套占卡 worker 能跑任意多个 run 的任意 step，加新 run 只改 manifest、不改脚本。
# 判定行：WORKER_START job= node= / CLAIM item= job= node= / （gl_hold_queue.sh 的 QUEUE_SHARD_DONE … rc=）/ ITEM_DONE item= rc= /
#         WORKER_DONE job= ran= failed=
set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
JOB="${1:?需要既有作业 jobid}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
QUEUE_DIR="${QUEUE_DIR:-$REPO/v1-store/evaluation/primary700-ab50k-gl/queue}"
# 待办清单：MANIFEST 模式取 manifest，否则取 QUEUE_DIR/items.txt；QUEUE_DIR 两种模式都用来放 claims/ 与 done/
[[ "$QUEUE_DIR" == /* ]] || { echo "QUEUE_DIR 必须是绝对路径：$QUEUE_DIR"; exit 2; }
if [[ -n "${MANIFEST:-}" ]]; then
    [[ -f "$MANIFEST" ]] || { echo "manifest 不存在：$MANIFEST"; exit 2; }
else
    [[ -f "$QUEUE_DIR/items.txt" ]] || { echo "队列不存在：$QUEUE_DIR/items.txt"; exit 2; }
fi
DRY="${DRY_RUN:-0}"
HEAD="$(git -C "$REPO" rev-parse HEAD)"
# clean-HEAD 闸：正式/冒烟必须 clean；DRY_RUN 只测抢单逻辑、不跑任何评测，允许带未提交改动（提交前本机自测用）
[[ "$DRY" == 1 || -z "$(git -C "$REPO" status --porcelain)" ]] || { echo '必须从 clean HEAD 启动'; exit 2; }
export STEP_TIME=12:00:00 EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 SEED=7 \
       MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count CHUNK_EPISODES="${CHUNK_EPISODES:-20}"
# 非 MANIFEST 模式沿用写死的 50000；MANIFEST 模式下逐行取
[[ -n "${MANIFEST:-}" ]] || export EXPECT_CKPT_ID=50000
# run_name 后缀：正式 -gl（primary700-nomotion50k-gl / primary700-motion50k-gl），冒烟传 -smoke-gl
SUF="${RUN_SUFFIX:--gl}"
[[ "$SUF" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p "$QUEUE_DIR/claims" "$QUEUE_DIR/done"
# 等作业 RUNNING（起跑时还 PENDING 的作业也可先起 worker，它自己等）；DRY_RUN 跳过
until [[ "$DRY" == 1 ]] || squeue -j "$JOB" -h -o %T | grep -qx RUNNING; do sleep 60; done
if [[ "$DRY" == 1 ]]; then NODE="$(hostname)"; else NODE="$(squeue -j "$JOB" -h -o %N)"; fi
echo "WORKER_START job=$JOB node=$NODE head=$HEAD queue=$QUEUE_DIR manifest=${MANIFEST:-none} suffix=${SUF:-none} chunk=$CHUNK_EPISODES $(date -u +%FT%TZ)"
RAN=0; FAILED=0
while true; do
    ITEM=""; MODEL=""; K=""; RUN=""; CKPT=""
    if [[ -n "${MANIFEST:-}" ]]; then
        while IFS=$'\t' read -r M_RUN M_CKPT M_POLICY M_CONFIG M_STEP M_SHARD; do
            [[ -n "$M_RUN" && "$M_SHARD" =~ ^[a-zA-Z0-9_-]+$ ]] || continue
            if mkdir "$QUEUE_DIR/claims/$M_RUN-s$M_SHARD" 2>/dev/null; then
                ITEM="$M_RUN-s$M_SHARD"; RUN="$M_RUN"; CKPT="$M_CKPT"; K="$M_SHARD"
                export POLICY="$M_POLICY" POLICY_CONFIG="$M_CONFIG" EXPECT_CKPT_ID="$M_STEP"
                break
            fi
        done < "$MANIFEST"
    else
        while read -r M KK; do
            [[ -n "$M" && "$KK" =~ ^[a-zA-Z0-9_-]+$ ]] || continue
            if mkdir "$QUEUE_DIR/claims/$M-s$KK" 2>/dev/null; then ITEM="$M-s$KK"; MODEL="$M"; K="$KK"; break; fi
        done < "$QUEUE_DIR/items.txt"
    fi
    [[ -n "$ITEM" ]] || break
    printf 'job=%s node=%s claimed=%s\n' "$JOB" "$NODE" "$(date -u +%FT%TZ)" > "$QUEUE_DIR/claims/$ITEM/owner"
    echo "CLAIM item=$ITEM job=$JOB node=$NODE $(date -u +%FT%TZ)"
    if [[ -z "${MANIFEST:-}" ]]; then
        case "$MODEL" in
            nomotion) RUN="primary700-nomotion50k$SUF"; CKPT="$REPO/v1-store/models/robomme-vla-modul-60k-v1/50000"
                      export POLICY=perceptual-framesamp-modul-8frame-8x8 POLICY_CONFIG=mme_vla_suite ;;
            motion)   RUN="primary700-motion50k$SUF";   CKPT="$REPO/v1-store/models/robomme-vla-modul-motion-80k-v1/50000"
                      export POLICY=perceptual-framesamp-modul-8frame-8x8-motion POLICY_CONFIG=mme_vla_suite_b128_80k ;;
            *) echo "未知模型 $MODEL"; exit 2 ;;
        esac
    fi
    T0="$(date -u +%FT%TZ)"
    if [[ "$DRY" == 1 ]]; then
        echo "DRY gl_hold_queue.sh $JOB $RUN $CKPT $REPO/v1-store/evaluation/$RUN/plans $HEAD $K POLICY=$POLICY POLICY_CONFIG=$POLICY_CONFIG STEP=$EXPECT_CKPT_ID"
        sleep 1; RC=0
    else
        bash "$REPO/scripts/evaluation/gl_hold_queue.sh" "$JOB" "$RUN" "$CKPT" "$REPO/v1-store/evaluation/$RUN/plans" "$HEAD" "$K"; RC=$?
    fi
    printf 'rc=%s job=%s node=%s start=%s end=%s\n' "$RC" "$JOB" "$NODE" "$T0" "$(date -u +%FT%TZ)" > "$QUEUE_DIR/done/$ITEM"
    echo "ITEM_DONE item=$ITEM rc=$RC job=$JOB $(date -u +%FT%TZ)"
    RAN=$((RAN + 1)); [[ "$RC" == 0 ]] || FAILED=$((FAILED + 1))
done
echo "WORKER_DONE job=$JOB ran=$RAN failed=$FAILED $(date -u +%FT%TZ)"
exit "$FAILED"
