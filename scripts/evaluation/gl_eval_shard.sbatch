#!/usr/bin/env bash
# ── test/primary 700 条评测：10 路单 GPU array，每片 70 集 ──────────────────────
# 计划：~/.claude/plans/https-github-com-hongzefu-robomme-bench-purring-zebra.md（用户 2026-09-18 批准）
#
# 资源档位由 primary700-scan 的四个探针实测给出，提交时用 CLI 覆盖下面的占位值：
#   sbatch --cpus-per-task=<扫描给出> --mem=<扫描给出> --gpu_cmode=shared ...
# CPU 硬上限 6（10 片 × N ≤ chaijy2 当时 66 核余量）。真正的排队瓶颈是 spgpu 物理 A40
# 全满（2026-09-18 实测 240 张已分配 238），档位越小越容易 backfill 进节点碎片。
#
# 必传（经 --export）：RUN_NAME、CKPT、PLAN_DIR
# 可选：SEED(7)、EXPECTED_GIT_HEAD（传入则起跑前断言 HEAD 未漂移 + 工作区 clean）、
#      ALLOW_RESUME(0)、EVAL_TIMEOUT(7200)、CHUNK_EPISODES(20)
#      SHARD_INDEX：用 srun --jobid=<既有作业> --overlap 直接跑时没有 SLURM_ARRAY_TASK_ID，由它指定分片下标
#                   （gl_hold_queue.sh 就是这么调的）；两者都没有则拒绝起跑
#      POLICY / POLICY_CONFIG / EXPECT_CKPT_ID / EPISODE_WALL_S / MMEVLA_MOTION_PROV_RELAX：原样透传给 run_shard.sh（motion 轮用）
#
#SBATCH --job-name=primary700-gl
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --signal=B:TERM@300
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%A_%a.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
source "$REPO/scripts/evaluation/samplers.sh"

_finalize() {
    rc=$?
    stop_samplers
    echo "EXIT_CODE=$rc"
}
trap _finalize EXIT
trap 'exit 143' TERM INT

RUN_NAME="${RUN_NAME:?需要 RUN_NAME}"
CKPT="${CKPT:?需要 CKPT 绝对路径}"
PLAN_DIR="${PLAN_DIR:?需要 PLAN_DIR}"
SEED="${SEED:-7}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:?需要 SHARD_INDEX 或 SLURM_ARRAY_TASK_ID}}"
export SHARD_INDEX
SHARD="s${SHARD_INDEX}"
PLAN="$PLAN_DIR/shard${SHARD_INDEX}.json"

# ── 起跑闸：array 可能排队数小时，期间仓库变动一律 fail-fast ──
if [ -n "${EXPECTED_GIT_HEAD:-}" ]; then
    ACTUAL_HEAD="$(git -C "$REPO" rev-parse HEAD)"
    if [ "$ACTUAL_HEAD" != "$EXPECTED_GIT_HEAD" ]; then
        echo "错误: HEAD 漂移（期望 $EXPECTED_GIT_HEAD，实际 $ACTUAL_HEAD），拒绝起跑"; exit 1
    fi
    if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
        echo "错误: 工作区不 clean，拒绝起跑"; git -C "$REPO" status --porcelain; exit 1
    fi
fi
[ -f "$PLAN" ] || { echo "错误: 分片计划不存在 $PLAN"; exit 1; }

export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "错误: 计算节点找不到 uv"; exit 1; }

RECORD_DIR="$REPO/v1-store/evaluation/$RUN_NAME/$SHARD/records"
start_samplers "$RECORD_DIR" 15

cd "$REPO"
unset ROBOMME_GPU_RASTER SAPIEN_DISABLE_RAY_TRACING
export PYTHONUNBUFFERED=1
echo "SHARD_START run=$RUN_NAME shard=$SHARD plan=$PLAN node=$(hostname) job=${SLURM_JOB_ID}"
bash scripts/evaluation/run_shard.sh "$RUN_NAME" "$CKPT" "$PLAN" "$SEED" "$SHARD"
