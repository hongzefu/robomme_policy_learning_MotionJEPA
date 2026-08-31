#!/usr/bin/env bash
# ── D2：集群侧上游对拍（v5.1-prod-60k-wandb-plan.md）────────────────────────────
# 单 job 串行跑 A（上游 main ecf086c，worktree + 独立 NFS venv，legacy 库，tentative
# 12 步 + 正式 1000 步）与 B（新入口 scripts/training/train.py，packed 库，1000 步），
# 生产口径 b64/fsdp4 + 确定性 XLA 档，judge 硬判 → ENTRY_EQ=PASS；随后同 job 用真
# wandb 代码跑出网探针（probe run 跑完即删）。
#
# 必传（经 --export）：EXPECTED_GIT_HEAD（仓库侧 HEAD 断言）、WANDB_PROJECT、
#         WANDB_ENTITY、WANDB_API_KEY（探针段）
# 判定行：A_SIDE_SEGMENTS / B_SIDE_SEGMENTS / ENTRY_SCALARS / ENTRY_STATE_DIGEST /
#         ENTRY_RESOLVED_CFG / ENTRY_PROVENANCE / WANDB_PROBE / ENTRY_EQ
# ⚠ 确定性 XLA 档在 A40 首测；ENTRY_SCALARS 失配先同侧重跑甄别硬件不确定性。
# ⚠ 本 job 4×A40×5h ≈ 20 GPU·h，超 greatlakes.md 调试包络，提交前须用户放行。
#
#SBATCH --job-name=v1-gl-entry-eq
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=FAIL
# 注：不要加 --qos=interactive —— chaijy2/spgpu 实测报 Invalid qos specification。

set -uo pipefail

_finalize() { rc=$?; echo "EXIT_CODE=$rc"; }
trap _finalize EXIT
trap 'exit 143' TERM INT

REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
STORE="$REPO/v1-store"
WORKTREE="$STORE/entryeq/worktree-main"
RECORDS="$STORE/entryeq/records"
UPSTREAM_HEAD=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
NCPU="${SLURM_CPUS_PER_TASK:-16}"
EQ_STEPS=1000

export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "错误: 计算节点找不到 uv"; exit 1; }
run_py() { uv run --no-sync --project "$REPO" python "$@"; }

# ── 起跑闸：EXPECTED_GIT_HEAD 必传；仓库/worktree 双侧 HEAD + porcelain + venv 复核 ──
if [ -z "${EXPECTED_GIT_HEAD:-}" ]; then
    echo "错误: 缺必传参数 EXPECTED_GIT_HEAD（应由提交命令经 --export 传入）"; exit 1
fi
ACTUAL_HEAD="$(git -C "$REPO" rev-parse HEAD)"
if [ "$ACTUAL_HEAD" != "$EXPECTED_GIT_HEAD" ]; then
    echo "错误: 仓库 HEAD 漂移（期望 $EXPECTED_GIT_HEAD，实际 $ACTUAL_HEAD）"; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
    echo "错误: 仓库工作区不 clean"; git -C "$REPO" status --porcelain; exit 1
fi
WT_HEAD="$(git -C "$WORKTREE" rev-parse HEAD)"
if [ "$WT_HEAD" != "$UPSTREAM_HEAD" ]; then
    echo "错误: worktree HEAD $WT_HEAD != 上游锚点 $UPSTREAM_HEAD"; exit 1
fi
if [ -n "$(git -C "$WORKTREE" status --porcelain)" ]; then
    echo "错误: worktree 不 clean"; git -C "$WORKTREE" status --porcelain; exit 1
fi
VENV_PY="$(readlink -f "$WORKTREE/.venv/bin/python")"
case "$VENV_PY" in
    /nfs/turbo/*) : ;;
    *) echo "错误: worktree venv 解释器 $VENV_PY 不在 /nfs/turbo/ 下（计算节点死链），见 greatlakes.md venv 硬规则"; exit 1 ;;
esac

# ── 共同 env：确定性 XLA 档（A40 首测）+ 缓存收敛 v1-store ──
export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export OPENPI_DATA_HOME="$STORE/models"
export XDG_CACHE_HOME="$STORE/cache/xdg" HF_HOME="$STORE/cache/hf"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled   # A/B 两侧均 --no-wandb-enabled，对拍纯净；探针段单独覆盖

# jax 编译缓存软链（照抄 gl_train_prod 惯例；train.py 硬编码 ~/.cache/jax_<exp_name>）
mkdir -p "$STORE/cache/jax" "$HOME/.cache" "$RECORDS"
for exp in gl-entry-eq-a gl-entry-eq-b; do
    mkdir -p "$STORE/cache/jax/$exp"
    [ -e "$HOME/.cache/jax_$exp" ] || ln -s "$STORE/cache/jax/$exp" "$HOME/.cache/jax_$exp"
done

# ── ENV 段：env.json 留档（provenance，AGENTS 12）──
run_py - "$RECORDS" <<'EOF'
import json, os, platform, subprocess, sys
import jax
d = {
    "run_name": "v1-gl-entry-eq", "eq_steps": 1000,
    "git_head": os.environ["EXPECTED_GIT_HEAD"],
    "upstream_head": "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": platform.node(),
    "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "python": sys.version, "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
json.dump(d, open(os.path.join(sys.argv[1], "env.json"), "w"), indent=2, ensure_ascii=False)
EOF
ENV_RC=$?
echo "ENV_RC=$ENV_RC"
if [ "$ENV_RC" -ne 0 ]; then exit "$ENV_RC"; fi

# ── A 段：上游 main（worktree 独立 venv，legacy 库，--overwrite，tentative 12 步）──
(
  cd "$WORKTREE" && env -u PYTHONPATH -u PYTHONHOME -u TRAIN_RECORD_DIR \
    uv run --no-sync --project "$WORKTREE" python \
    "$REPO/scripts/training/tests/entry_equiv.py" run \
    --entry "$WORKTREE/scripts/train.py" \
    --record-dir "$RECORDS/a" \
    --expect-root "$WORKTREE" \
    --expect-steps "$EQ_STEPS" -- mme_vla_suite \
    --exp-name gl-entry-eq-a --overwrite \
    --num-train-steps "$EQ_STEPS" --log-interval 1 --save-interval 100 \
    --batch-size 64 --num-workers 8 --seed 42 --fsdp-devices 4 \
    --dataset-path "$STORE/datasets/4task-gl" \
    --assets-base-dir "$STORE/train-assets" \
    --checkpoint-base-dir "$STORE/entryeq/ckpt-a" \
    --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
)
A_RC=$?
echo "A_RC=$A_RC"

# ── B 段：新入口（仓库 venv，packed 库，无 --overwrite，--forbid-root 拦 worktree）──
(
  cd "$REPO" && env -u PYTHONPATH -u PYTHONHOME -u TRAIN_RECORD_DIR \
    uv run --no-sync --project "$REPO" python \
    "$REPO/scripts/training/tests/entry_equiv.py" run \
    --entry "$REPO/scripts/training/train.py" \
    --record-dir "$RECORDS/b" \
    --expect-root "$REPO" \
    --forbid-root "$WORKTREE" \
    --expect-steps "$EQ_STEPS" -- mme_vla_suite \
    --exp-name gl-entry-eq-b \
    --num-train-steps "$EQ_STEPS" --log-interval 1 --save-interval 100 \
    --batch-size 64 --num-workers 8 --seed 42 --fsdp-devices 4 \
    --dataset-path "$STORE/datasets/4task-gl-framesamp" \
    --assets-base-dir "$STORE/train-assets" \
    --checkpoint-base-dir "$STORE/entryeq/ckpt-b" \
    --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context.yaml \
    --no-wandb-enabled
)
B_RC=$?
echo "B_RC=$B_RC"

# ── judge 段：全硬判（步集合/分段行数/状态摘要集合/HEAD+porcelain；A40 无锚点，
#     不传 --expect-sha256，anchor=SKIPPED）──
JUDGE_RC=1
if [ "$A_RC" -eq 0 ] && [ "$B_RC" -eq 0 ]; then
    run_py "$REPO/scripts/training/tests/entry_equiv.py" judge \
        --a-dir "$RECORDS/a" --b-dir "$RECORDS/b" \
        --expect-steps "$EQ_STEPS" \
        --expect-tentative-a 12 \
        --expect-state-steps 100,200,300,400,500,600,700,800,900,999 \
        --expect-head-a "$UPSTREAM_HEAD" \
        --expect-head-b "$EXPECTED_GIT_HEAD"
    JUDGE_RC=$?
else
    echo "跳过 judge：A_RC=$A_RC B_RC=$B_RC"
fi
echo "JUDGE_RC=$JUDGE_RC"

# ── 探针段：真 wandb 代码验证计算节点出网（probe run 跑完即删）──
PROBE_RC=1
if [ -n "${WANDB_PROJECT:-}" ] && [ -n "${WANDB_ENTITY:-}" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_MODE=online WANDB_DIR="$STORE/cache/wandb" \
    WANDB_CACHE_DIR="$STORE/cache/wandb" WANDB_CONFIG_DIR="$STORE/cache/wandb" \
    run_py - <<'EOF'
import os, wandb
project, entity = os.environ["WANDB_PROJECT"], os.environ["WANDB_ENTITY"]
run = wandb.init(project=project, entity=entity, name="gl-wandb-probe", mode="online")
wandb.log({"probe": 1.0}, step=0)
run_id = run.id
run.finish()
wandb.Api().run(f"{entity}/{project}/{run_id}").delete()
print("WANDB_PROBE=PASS", flush=True)
EOF
    PROBE_RC=$?
else
    echo "错误: 探针段缺 WANDB_PROJECT/WANDB_ENTITY/WANDB_API_KEY（应经 --export 传入）"
fi
[ "$PROBE_RC" -eq 0 ] || echo "WANDB_PROBE=FAIL"
echo "PROBE_RC=$PROBE_RC"

# 最终 RC = 五段首个非零（A/B/JUDGE/PROBE/ENV）；EXIT_CODE= 行由 trap 统一输出
RC=0
for r in "$A_RC" "$B_RC" "$JUDGE_RC" "$PROBE_RC" "$ENV_RC"; do
    if [ "$r" -ne 0 ]; then RC=$r; break; fi
done
exit "$RC"
