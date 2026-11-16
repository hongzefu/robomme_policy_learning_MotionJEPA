#!/usr/bin/env bash
# ── 资源探针：两维各自向下压，顺带做 TLS 实测与 JAX 缓存预热 ──────────────────
# 计划：~/.claude/plans/https-github-com-hongzefu-robomme-bench-purring-zebra.md（用户 2026-09-18 批准）
#
# 四个探针共用本脚本，档位与计划由提交行覆盖：
#   A 基准 2C/32G probe-A.json（32 集，故意跨过 Vulkan 静态 TLS 的第 28 次 make_env 红线）
#   B 压CPU 1C/32G probe-BCD.json（8 集，对照 A 的稳态步时判 1 核够不够）
#   C 压MEM 2C/16G probe-BCD.json
#   D 双压   1C/8G  probe-BCD.json（跑通就直接采用）
# OOM 本身就是有效结论，不重跑、不上调重试（playbook 先例见 docs/archive/v1-gl-resource-tier-bench.md）。
#
# 必传（经 --export）：RUN_NAME、CKPT、PLAN、PROBE
# 可选：SEED(7)、EXPECTED_GIT_HEAD、EVAL_TIMEOUT(7200)、CHUNK_EPISODES(20)
#
#SBATCH --job-name=primary700-scan
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
source "$REPO/scripts/evaluation/samplers.sh"

_finalize() {
    rc=$?
    stop_samplers
    echo "PROBE_EXIT probe=${PROBE:-?} cpu=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M rc=$rc"
    echo "EXIT_CODE=$rc"
}
trap _finalize EXIT
trap 'exit 143' TERM INT

RUN_NAME="${RUN_NAME:?需要 RUN_NAME}"
CKPT="${CKPT:?需要 CKPT 绝对路径}"
PLAN="${PLAN:?需要探针计划 json}"
PROBE="${PROBE:?需要探针编号 A/B/C/D}"
SEED="${SEED:-7}"
SHARD="probe-$PROBE"

if [ -n "${EXPECTED_GIT_HEAD:-}" ]; then
    ACTUAL_HEAD="$(git -C "$REPO" rev-parse HEAD)"
    if [ "$ACTUAL_HEAD" != "$EXPECTED_GIT_HEAD" ]; then
        echo "错误: HEAD 漂移（期望 $EXPECTED_GIT_HEAD，实际 $ACTUAL_HEAD），拒绝起跑"; exit 1
    fi
    if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
        echo "错误: 工作区不 clean，拒绝起跑"; git -C "$REPO" status --porcelain; exit 1
    fi
fi
[ -f "$PLAN" ] || { echo "错误: 探针计划不存在 $PLAN"; exit 1; }

export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "错误: 计算节点找不到 uv"; exit 1; }

RECORD_DIR="$REPO/v1-store/evaluation/$RUN_NAME/$SHARD/records"
start_samplers "$RECORD_DIR" 2      # 内存通道加密到 2s：ckpt 恢复峰值只持续约 26 秒

cd "$REPO"
unset ROBOMME_GPU_RASTER SAPIEN_DISABLE_RAY_TRACING
export PYTHONUNBUFFERED=1
echo "PROBE_START probe=$PROBE run=$RUN_NAME plan=$PLAN cpu=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M node=$(hostname) job=${SLURM_JOB_ID}"
bash scripts/evaluation/run_shard.sh "$RUN_NAME" "$CKPT" "$PLAN" "$SEED" "$SHARD"
