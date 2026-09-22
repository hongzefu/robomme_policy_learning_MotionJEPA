#!/usr/bin/env bash
# 在**既有** GreatLakes 作业（如 gpu-hold-0K：1×A40 / 1 CPU / 24G）里串行跑若干分片，不提交新 job。
# 机制：从登录节点 `srun --jobid=<既有作业> --overlap --exact` 开 step，复用该作业已分配的卡；本脚本本身跑在登录节点
# 的 detached tmux 里（用户 ssh 断开不影响），每个分片一个 step，日志 tee 到 NFS 供本机 Monitor 跟踪。
# 用法（登录节点）：gl_hold_queue.sh <jobid> <run_name> <ckpt绝对路径> <plan_dir> <expected_git_head> <shard_index>...
#   ⚠ 必须 -u MANIFEST：srun 默认把提交端环境整个带进 step，调用方（gl_hold_pool.sh 的 manifest 模式）环境里的 MANIFEST
#   会让 gl_eval_shard.sbatch 误入 array 模式、因没有 SLURM_ARRAY_TASK_ID 而秒退（2026-09-22 实测 20 个单元全 rc=1）。
#   本脚本已按分片把六个值显式算好并经 PASS 传入，step 里不需要也不能再看见 MANIFEST。
#   透传给 gl_eval_shard.sbatch 的变量（有值才传）：POLICY POLICY_CONFIG EXPECT_CKPT_ID EPISODE_WALL_S EVAL_TIMEOUT
#   CHUNK_EPISODES SEED ALLOW_RESUME MMEVLA_MOTION_PROV_RELAX DEMO_PREFIX_STORE MMEVLA_ENC_CHUNK
#   MMEVLA_MOTION_OVERFLOW EXPECT_MOTION_STATS；STEP_TIME（每个 step 的 --time，默认 12:00:00）
# 判定行：每片 `QUEUE_SHARD_DONE job=<jobid> shard=<k> rc=<rc>`，结束 `QUEUE_DONE job=<jobid> failed=<n>`
set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
JOB="${1:?需要既有作业 jobid}"; RUN_NAME="${2:?需要 run_name}"; CKPT="${3:?需要 checkpoint 绝对路径}"
PLAN_DIR="${4:?需要分片计划目录}"; EXPECTED_GIT_HEAD="${5:?需要 expected_git_head}"; shift 5
[[ $# -ge 1 ]] || { echo '至少一个分片下标'; exit 2; }
[[ "$JOB" =~ ^[0-9]+$ && "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$CKPT" == /* && "$PLAN_DIR" == /* ]] || exit 2
STEP_TIME="${STEP_TIME:-12:00:00}"
LOG_DIR="$REPO/v1-store/logs/policy-eval"
mkdir -p "$LOG_DIR"
squeue -j "$JOB" -h -o '%T %N' | grep -q '^RUNNING' || { echo "作业 $JOB 不在 RUNNING"; exit 2; }
PASS=(RUN_NAME="$RUN_NAME" CKPT="$CKPT" PLAN_DIR="$PLAN_DIR" EXPECTED_GIT_HEAD="$EXPECTED_GIT_HEAD")
for V in POLICY POLICY_CONFIG EXPECT_CKPT_ID EPISODE_WALL_S EVAL_TIMEOUT CHUNK_EPISODES SEED ALLOW_RESUME MMEVLA_MOTION_PROV_RELAX \
         DEMO_PREFIX_STORE MMEVLA_ENC_CHUNK MMEVLA_MOTION_OVERFLOW EXPECT_MOTION_STATS; do
    [[ -z "${!V:-}" ]] || PASS+=("$V=${!V}")
done
FAILED=0
echo "QUEUE_START job=$JOB run=$RUN_NAME shards=$* head=$EXPECTED_GIT_HEAD node=$(squeue -j "$JOB" -h -o '%N') $(date -u +%FT%TZ)"
for K in "$@"; do
    [[ "$K" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "非法分片下标 $K"; exit 2; }
    LOG="$LOG_DIR/$RUN_NAME-s$K.log"
    echo "QUEUE_SHARD_START job=$JOB shard=$K log=$LOG $(date -u +%FT%TZ)"
    srun --jobid="$JOB" --account=chaijy2 --partition=spgpu --gpu_cmode=shared --overlap --exact \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-node=1 --time="$STEP_TIME" --chdir="$REPO" \
        /usr/bin/env -u ROBOMME_GPU_RASTER -u SAPIEN_DISABLE_RAY_TRACING -u MANIFEST PYTHONUNBUFFERED=1 "${PASS[@]}" SHARD_INDEX="$K" \
        /usr/bin/bash "$REPO/scripts/evaluation/gl_eval_shard.sbatch" 2>&1 | tee -a "$LOG"
    RC=${PIPESTATUS[0]}
    [[ "$RC" == 0 ]] || FAILED=$((FAILED + 1))
    echo "QUEUE_SHARD_DONE job=$JOB shard=$K rc=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
done
echo "QUEUE_DONE job=$JOB failed=$FAILED $(date -u +%FT%TZ)"
exit "$FAILED"
