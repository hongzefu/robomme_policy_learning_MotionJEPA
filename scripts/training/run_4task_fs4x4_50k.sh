#!/usr/bin/env bash
# 4task 原训练库 + 帧记忆每帧 4×4（token_per_image 16）modulation 50k 正式训练 / 20 步冒烟 runner（motion 关闭）。
# 依据：根目录 1127-4task-modul-fs4x4-512-1024-60k-plan.md 第一节、第四节第 4 条、第五节 G3/G4、第二部分〇/一 S1 行。
# 以 v11 的 run_4task_wantp768_50k.sh 为底：TrainConfig mme_vla_suite_b128_80k 全部超参不变，只覆盖
# --num_train_steps=50001（最后一份 checkpoint 为 50000）；帧表换 framesamp-4x4；history YAML 按 budget 取
# perceptual-framesamp-modul-32frame-4x4.yaml（A 组 512）或 …-64frame-4x4.yaml（B 组 1024）。
# 与 v11 相比删除全部 motion 专属项（MOTION 路径、LATENTS_META_SHA、wan provenance 检查、MMEVLA_MOTION_*），
# 并 unset 全部 MMEVLA_MOTION_* 环境变量。
#
# 用法：
#   bash scripts/training/run_4task_fs4x4_50k.sh <TRAIN_HEAD 40 位 sha> --budget 512|1024             # 正式 50001 步
#   bash scripts/training/run_4task_fs4x4_50k.sh <TRAIN_HEAD> --budget 512|1024 --smoke20             # G4 冒烟 20 步
#   bash scripts/training/run_4task_fs4x4_50k.sh <任意> --budget 512|1024 [--smoke20] --dry-run       # 只打印，不写任何东西
#
# 冒烟与正式的 TRAIN_ARGS 只差 --exp-name 与 --num_train_steps 两处。
# 起跑前内置 G3 启动合同（check_4task_fs4x4_launch.py，含 CPU 实例化 FrameSampDataset）与 preflight，FAIL 即退出；
# 训练结束核 checkpoint 齐全并真实加载最后一份，末行前写 FS4X4_TRAIN=PASS|FAIL 判定行，末行 EXIT_CODE=<n>。
set -o pipefail                     # 外层记录正常返回和报错的退出码；外部强杀除外。

usage() { printf '用法: %s <TRAIN_HEAD> --budget 512|1024 [--smoke20] [--dry-run]\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] || usage; shift
SMOKE=0; DRY=0; BUDGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke20) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    --budget) [ $# -ge 2 ] || usage; BUDGET="$2"; shift ;;
    *) usage ;;
  esac
  shift
done

case "$BUDGET" in
  512)
    FORMAL_RUN=v2-1600ep-m32x4x4-modul-b128-50k; SMOKE_RUN=fs4a-smoke20
    HC=perceptual-framesamp-modul-32frame-4x4.yaml
    HC_SHA=14b54bae33d1a5ecb8240dff2e9f69727de6e43c6e5ad730b87c1e57b0e44acb ;;   # 定稿 YAML 文件字节 sha256
  1024)
    FORMAL_RUN=v2-1600ep-m64x4x4-modul-b128-50k; SMOKE_RUN=fs4b-smoke20
    HC=perceptual-framesamp-modul-64frame-4x4.yaml
    HC_SHA=a08f40d3b38bbe6fc94186375b9f00ac6ea130c7fc7fb1386ff362e309c0f520 ;;   # 定稿 YAML 文件字节 sha256
  *) printf -- '--budget 只能是 512 或 1024，得到 %s\n' "${BUDGET:-<缺>}" >&2; usage ;;
esac

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
if [ "$SMOKE" = 1 ]; then
  RUN="$SMOKE_RUN"; STEPS=20; EXPECT_CKPTS="19"
else
  RUN="$FORMAL_RUN"; STEPS=50001
  EXPECT_CKPTS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000"
fi
CONFIG=mme_vla_suite_b128_80k
NORM_SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173
MANIFEST_ID=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918   # load_manifest 重算的 canonical identity（不是文件字节摘要）
V1_STORE="$MAIN/v1-store"
DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
FS="$DS/framesamp-4x4"
ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
RUN_ROOT="$V1_STORE/train-runs/$CONFIG/$RUN"
LOG="$V1_STORE/logs/$RUN.log"
REC="$V1_STORE/bench/$RUN"
# 双根契约（dataloader._create_framesamp_dataset）：环境变量优先，不靠桶里 store_meta 记录的 9 月绝对路径兜底
FRAMESAMP_SOURCE="$DS/source"
FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"

TRAIN_ARGS=(
  "$CONFIG"
  --exp-name "$RUN"
  --assets-base-dir "$V1_STORE/train-assets"
  --data.assets.assets-dir "$ASSETS_DIR"
  --data.assets.asset-id robomme
  --checkpoint-base-dir "$V1_STORE/train-runs"
  --dataset-path "$FS"
  --model.history-config "$HC"
  "--num_train_steps=$STEPS"
)

if [ "$DRY" = 1 ]; then
  printf 'RUN=%s BUDGET=%s STEPS=%s SMOKE=%s EXPECT_CKPTS=%s\n' "$RUN" "$BUDGET" "$STEPS" "$SMOKE" "$EXPECT_CKPTS"
  printf 'HC=%s HC_SHA=%s NORM_SHA=%s MANIFEST_ID=%s\n' "$HC" "$HC_SHA" "$NORM_SHA" "$MANIFEST_ID"
  printf 'MMEVLA_FRAMESAMP_SOURCE=%s\nMMEVLA_FRAMESAMP_MANIFEST=%s\n' "$FRAMESAMP_SOURCE" "$FRAMESAMP_MANIFEST"
  printf 'LOG=%s\nRUN_ROOT=%s\n' "$LOG" "$RUN_ROOT"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  exit 0
fi

test ! -e "$LOG" || { printf '拒绝覆盖本轮日志 %s\n' "$LOG"; exit 2; }
mkdir -p "$(dirname "$LOG")" "$REC"

body() (
  set -euo pipefail
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志

  test "$(git rev-parse HEAD)" = "$TRAIN_HEAD"
  test -z "$(git status --porcelain)"
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export TRAIN_TIMING_STEPS=300

  unset PYTHONPATH
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  unset JAX_PLATFORMS WANDB_MODE
  # motion 关闭态：清掉全部 MMEVLA_MOTION_* 环境变量（含 MMEVLA_MOTION_STORE / MMEVLA_MOTION_OVERFLOW 等）
  for v in $(compgen -e | grep '^MMEVLA_MOTION_' || true); do unset "$v"; done
  test -z "$(compgen -e | grep '^MMEVLA_MOTION_' || true)"
  unset MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED MMEVLA_FRAMESAMP_ALLOW_SUBSET MMEVLA_FRAMESAMP_VERIFY   # 正式判据不得放行 unverified / subset
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export TRAIN_RECORD_DIR="$REC"
  export MMEVLA_FRAMESAMP_SOURCE="$FRAMESAMP_SOURCE"
  export MMEVLA_FRAMESAMP_MANIFEST="$FRAMESAMP_MANIFEST"

  test "$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  printf 'RUN=%s\nBUDGET=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTEPS=%s\nSMOKE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$BUDGET" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$STEPS" "$SMOKE" "$(date -u +%FT%TZ)"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'

  # ── 内置 G3 启动合同（yaml 差异 / sha 钉死 / 训练配置 / 帧表 meta / 两个 MMEVLA_FRAMESAMP_* / CPU 实例化取 1 样本）──
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/check_4task_fs4x4_launch.py \
    --budget "$BUDGET" --store "$FS"

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 "$NORM_SHA" \
    --dataset-path "$FS" \
    --run-root "$RUN_ROOT" \
    -- "${TRAIN_ARGS[@]}"

  # 全程 15 s GPU 采样：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上（否则 tee 不退出）
  nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
  kill "$sampler_pid" 2>/dev/null || true

  # ── 训练后：checkpoint 齐全 + 快照为 4×4 关闭态 + 真实加载最后一份 ──
  JAX_PLATFORMS=cpu uv run --no-sync python - "$RUN_ROOT" "$SMOKE" "$BUDGET" $EXPECT_CKPTS <<'PY'
import json, pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
import openpi.models.model as _model
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _params_have_motion

run_root, smoke, budget, expect = pathlib.Path(sys.argv[1]), sys.argv[2] == "1", int(sys.argv[3]), sys.argv[4:]
dirs = sorted((d.name for d in run_root.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
assert dirs == expect, f"checkpoint 集合 {dirs} != 期望 {expect}"
for d in expect:
    assert (run_root / d / "params").is_dir(), f"缺 {d}/params"
cfg, enabled = _load_resolved_snapshot(run_root)
assert not enabled, "快照 motion 应为关闭态"
assert int(cfg.budget) == budget and int(cfg.token_per_image) == 16, f"快照 budget/token_per_image={cfg.budget}/{cfg.token_per_image}"
prov = json.loads((run_root / "motion_provenance.json").read_text())
assert prov.get("motion_enabled") is False, "motion_provenance.motion_enabled 应为 false"
last = run_root / expect[-1]
params = _model.restore_params(last / "params", dtype=jnp.bfloat16)
assert not _params_have_motion(params), "关闭态 checkpoint 不应含 motion 参数"
leaves = jax.tree_util.tree_leaves(params)
assert leaves and all(bool(np.isfinite(np.asarray(x, dtype=np.float32)).all()) for x in leaves), "参数含非有限值"
print(f"CKPT_LOAD=PASS step={expect[-1]} leaves={len(leaves)} resolved_sha256={prov.get('resolved_sha256')}", flush=True)
if smoke:
    print("SMOKE20=PASS steps=20 save_load=PASS", flush=True)
else:
    print(f"TRAIN_FINAL_RECORD=PASS step={expect[-1]} ckpts={len(dirs)}", flush=True)
PY
)

body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
n_ckpt=0
for s in $EXPECT_CKPTS; do [ -d "$RUN_ROOT/$s/params" ] && n_ckpt=$((n_ckpt + 1)); done
n_expect=$(set -- $EXPECT_CKPTS; echo $#)
if [ "$rc" = 0 ] && [ "$n_ckpt" = "$n_expect" ]; then
  printf 'FS4X4_TRAIN=PASS run=%s steps=%s ckpts=%s\n' "$RUN" "$STEPS" "$n_ckpt" | tee -a "$LOG"
else
  [ "$rc" = 0 ] && rc=1
  printf 'FS4X4_TRAIN=FAIL run=%s steps=%s ckpts=%s/%s rc=%s\n' "$RUN" "$STEPS" "$n_ckpt" "$n_expect" "$rc" | tee -a "$LOG"
fi
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
