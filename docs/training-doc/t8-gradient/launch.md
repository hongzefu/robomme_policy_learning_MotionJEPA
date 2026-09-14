# 四种配置的单步全梯度验收启动记录

用户要求“一口气全做完”。本项为计划内第二块的补充验收，等待12条1000更新轨迹及其逐位gate完成后运行，不增加正式训练更新。每个profile在同一对物理GPU上按A→B依次运行；C32/C8使用4,5，M32/M8使用6,7，同时至多四卡。会话 `t8-grad-c32`、`t8-grad-m32` 为第一波，之后 `t8-grad-c8`、`t8-grad-m8`。日志 `v1-store/logs/<会话名>.log`，外层pipefail、tee、EXIT_CODE。

源码仍锚定REF `99faacb1319adfc63c0cf9a15187e24c34e38fd1` 与CAND `c08ec2060a544af1869c1e24f755e536150569ca`。A显式PYTHONPATH指REF/src，B清除此覆盖。实际起跑HEAD与干净状态检查随日志记录。所有输入为同一400ep源/packed库，norm_stats文件SHA为 `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`。每侧起跑先与同侧1000步轨迹指纹核对，现场init的完整TrainState再与该轨迹state_step0逐叶比对。

固定seed42、batch8、fsdp2，在allshort、allfull、mixed1三种真实batch上各计算一次完整梯度。每种比较索引、loss hex、完整梯度树叶集合与逐叶SHA，另检查全部loss和梯度统计有限。三种batch之间不更新参数。命令中的num-train-steps1000仅保持配置与对应基线相同，single_step_grad不执行训练更新循环。参数落点均为启动覆盖，未改全局默认。

逐profile要求 `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0` 和 `GRAD_FINITE=PASS`；数据、环境或初态不同即停止，不放宽逐位判据。存储为AWS本地NVMe RAID `/dev/md0`；记录在 `v1-store/bench/8x8/grad/`，batch数组在fixtures同名目录，不归档大数组或权重。

## C32

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
unset JAX_PLATFORMS DTYPE_GRAD_KINDS DTYPE_GRAD_ARRAYS_DIR DTYPE_DUMP_LIMIT
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
unset MMEVLA_MOTION_STORE
export DTYPE_MANIFEST="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json"
for side in a b; do
  if [ "$side" = a ]; then
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/ref-8x8"
    export PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv"
    export DTYPE_DUMP_IMPL=packed
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/framesamp"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-c32-a1"
  else
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
    unset PYTHONPATH
    export UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv" DTYPE_DUMP_IMPL=packed
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/framesamp"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-c32-b"
  fi
  test -z "$(git status --porcelain)"
  export DTYPE_GRAD_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c32-$side"
  export DTYPE_BATCH_FIXTURE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/fixtures/8x8/grad/c32-$side"
  export DTYPE_BASELINE_CHECKSUMS="$baseline/param_checksums.jsonl"
  export MMEVLA_JAX_CACHE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/t8-grad-c32-$side"
  test ! -e "$DTYPE_GRAD_DIR"
  test ! -e "$DTYPE_BATCH_FIXTURE_DIR"
  printf 'GRAD_START profile=c32 side=%s head=%s impl=%s\n' "$side" "$(git rev-parse HEAD)" "$DTYPE_DUMP_IMPL"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir "$DTYPE_GRAD_DIR" --v1-store "$V1_STORE" --source "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --manifest "$DTYPE_MANIFEST" --dataset "$input" --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "$baseline" --record-dir "$DTYPE_GRAD_DIR"
  uv run --no-sync python scripts/training/tests/single_step_grad.py mme_vla_suite --exp-name "t8-grad-c32-$side" \
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs/t8-grad-c32-$side" \
    --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
    --dataset-path "$input" --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$DTYPE_GRAD_DIR"
done
cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
unset PYTHONPATH
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c32-a" --record-dir "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c32-b"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_grad_summaries.py "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c32-a/grad_summary.json" "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c32-b/grad_summary.json"
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
import json, math, pathlib
for side in ("a","b"):
    p=pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c32-"+side+"/grad_summary.json")
    data=json.loads(p.read_text())
    assert set(data["results"])=={"mixed1","allshort","allfull"}
    assert data["same_origin"]["verdict"]=="PASS"
    for row in data["results"].values():
        assert math.isfinite(float.fromhex(row["loss_hex"]))
        assert row["n_leaves"]==len(row["per_leaf"])==len(row["stats"])
        assert all(math.isfinite(s[k]) for s in row["stats"].values() for k in ("max_abs","l2"))
print("GRAD_FINITE=PASS profile=c32 kinds=3")
PY

```

## M32

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
unset JAX_PLATFORMS DTYPE_GRAD_KINDS DTYPE_GRAD_ARRAYS_DIR DTYPE_DUMP_LIMIT
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
export MMEVLA_MOTION_STORE="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/motion"
export DTYPE_MANIFEST="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json"
for side in a b; do
  if [ "$side" = a ]; then
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/ref-8x8"
    export PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv"
    export DTYPE_DUMP_IMPL=packed
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/framesamp"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m32-a1"
  else
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
    unset PYTHONPATH
    export UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv" DTYPE_DUMP_IMPL=packed
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/framesamp"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m32-b"
  fi
  test -z "$(git status --porcelain)"
  export DTYPE_GRAD_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m32-$side"
  export DTYPE_BATCH_FIXTURE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/fixtures/8x8/grad/m32-$side"
  export DTYPE_BASELINE_CHECKSUMS="$baseline/param_checksums.jsonl"
  export MMEVLA_JAX_CACHE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/t8-grad-m32-$side"
  test ! -e "$DTYPE_GRAD_DIR"
  test ! -e "$DTYPE_BATCH_FIXTURE_DIR"
  printf 'GRAD_START profile=m32 side=%s head=%s impl=%s\n' "$side" "$(git rev-parse HEAD)" "$DTYPE_DUMP_IMPL"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir "$DTYPE_GRAD_DIR" --v1-store "$V1_STORE" --source "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --manifest "$DTYPE_MANIFEST" --dataset "$input" --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "$baseline" --record-dir "$DTYPE_GRAD_DIR"
  uv run --no-sync python scripts/training/tests/single_step_grad.py mme_vla_suite --exp-name "t8-grad-m32-$side" \
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs/t8-grad-m32-$side" \
    --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
    --dataset-path "$input" --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context-motion.yaml --no-wandb-enabled
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$DTYPE_GRAD_DIR"
done
cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
unset PYTHONPATH
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m32-a" --record-dir "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m32-b"
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_grad_summaries.py "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m32-a/grad_summary.json" "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m32-b/grad_summary.json"
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
import json, math, pathlib
for side in ("a","b"):
    p=pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m32-"+side+"/grad_summary.json")
    data=json.loads(p.read_text())
    assert set(data["results"])=={"mixed1","allshort","allfull"}
    assert data["same_origin"]["verdict"]=="PASS"
    for row in data["results"].values():
        assert math.isfinite(float.fromhex(row["loss_hex"]))
        assert row["n_leaves"]==len(row["per_leaf"])==len(row["stats"])
        assert all(math.isfinite(s[k]) for s in row["stats"].values() for k in ("max_abs","l2"))
print("GRAD_FINITE=PASS profile=m32 kinds=3")
PY

```

## C8

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
unset JAX_PLATFORMS DTYPE_GRAD_KINDS DTYPE_GRAD_ARRAYS_DIR DTYPE_DUMP_LIMIT
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
unset MMEVLA_MOTION_STORE
export DTYPE_MANIFEST="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json"
for side in a b; do
  if [ "$side" = a ]; then
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/ref-8x8"
    export PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv"
    export DTYPE_DUMP_IMPL=refnpy
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-c8-a1"
  else
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
    unset PYTHONPATH
    export UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv" DTYPE_DUMP_IMPL=packed
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/framesamp-8x8"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-c8-b"
  fi
  test -z "$(git status --porcelain)"
  export DTYPE_GRAD_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c8-$side"
  export DTYPE_BATCH_FIXTURE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/fixtures/8x8/grad/c8-$side"
  export DTYPE_BASELINE_CHECKSUMS="$baseline/param_checksums.jsonl"
  export MMEVLA_JAX_CACHE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/t8-grad-c8-$side"
  test ! -e "$DTYPE_GRAD_DIR"
  test ! -e "$DTYPE_BATCH_FIXTURE_DIR"
  printf 'GRAD_START profile=c8 side=%s head=%s impl=%s\n' "$side" "$(git rev-parse HEAD)" "$DTYPE_DUMP_IMPL"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir "$DTYPE_GRAD_DIR" --v1-store "$V1_STORE" --source "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --manifest "$DTYPE_MANIFEST" --dataset "$input" --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "$baseline" --record-dir "$DTYPE_GRAD_DIR"
  uv run --no-sync python scripts/training/tests/single_step_grad.py mme_vla_suite --exp-name "t8-grad-c8-$side" \
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs/t8-grad-c8-$side" \
    --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
    --dataset-path "$input" --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$DTYPE_GRAD_DIR"
done
cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
unset PYTHONPATH
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c8-a" --record-dir "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c8-b" --allow-difference dataset.store_meta_sha256
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_grad_summaries.py "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c8-a/grad_summary.json" "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c8-b/grad_summary.json"
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
import json, math, pathlib
for side in ("a","b"):
    p=pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/c8-"+side+"/grad_summary.json")
    data=json.loads(p.read_text())
    assert set(data["results"])=={"mixed1","allshort","allfull"}
    assert data["same_origin"]["verdict"]=="PASS"
    for row in data["results"].values():
        assert math.isfinite(float.fromhex(row["loss_hex"]))
        assert row["n_leaves"]==len(row["per_leaf"])==len(row["stats"])
        assert all(math.isfinite(s[k]) for s in row["stats"].values() for k in ("max_abs","l2"))
print("GRAD_FINITE=PASS profile=c8 kinds=3")
PY

```

## M8

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
unset JAX_PLATFORMS DTYPE_GRAD_KINDS DTYPE_GRAD_ARRAYS_DIR DTYPE_DUMP_LIMIT
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
export MMEVLA_MOTION_STORE="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/motion"
export DTYPE_MANIFEST="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json"
for side in a b; do
  if [ "$side" = a ]; then
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/ref-8x8"
    export PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv"
    export DTYPE_DUMP_IMPL=refnpy
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m8-a1"
  else
    cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
    unset PYTHONPATH
    export UV_PROJECT_ENVIRONMENT="/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv" DTYPE_DUMP_IMPL=packed
    input="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/framesamp-8x8"
    baseline="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/t8-m8-b"
  fi
  test -z "$(git status --porcelain)"
  export DTYPE_GRAD_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m8-$side"
  export DTYPE_BATCH_FIXTURE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/fixtures/8x8/grad/m8-$side"
  export DTYPE_BASELINE_CHECKSUMS="$baseline/param_checksums.jsonl"
  export MMEVLA_JAX_CACHE_DIR="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/jax/t8-grad-m8-$side"
  test ! -e "$DTYPE_GRAD_DIR"
  test ! -e "$DTYPE_BATCH_FIXTURE_DIR"
  printf 'GRAD_START profile=m8 side=%s head=%s impl=%s\n' "$side" "$(git rev-parse HEAD)" "$DTYPE_DUMP_IMPL"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir "$DTYPE_GRAD_DIR" --v1-store "$V1_STORE" --source "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source" --manifest "$DTYPE_MANIFEST" --dataset "$input" --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "$baseline" --record-dir "$DTYPE_GRAD_DIR"
  uv run --no-sync python scripts/training/tests/single_step_grad.py mme_vla_suite --exp-name "t8-grad-m8-$side" \
    --assets-base-dir "$V1_STORE/train-assets" --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs/t8-grad-m8-$side" \
    --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
    --dataset-path "$input" --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8-motion.yaml --no-wandb-enabled
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$DTYPE_GRAD_DIR"
done
cd "/scratch/hongze/robomme_policy_learning_MotionJEPA"
unset PYTHONPATH
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m8-a" --record-dir "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m8-b" --allow-difference dataset.store_meta_sha256
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_grad_summaries.py "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m8-a/grad_summary.json" "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m8-b/grad_summary.json"
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
import json, math, pathlib
for side in ("a","b"):
    p=pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/8x8/grad/m8-"+side+"/grad_summary.json")
    data=json.loads(p.read_text())
    assert set(data["results"])=={"mixed1","allshort","allfull"}
    assert data["same_origin"]["verdict"]=="PASS"
    for row in data["results"].values():
        assert math.isfinite(float.fromhex(row["loss_hex"]))
        assert row["n_leaves"]==len(row["per_leaf"])==len(row["stats"])
        assert all(math.isfinite(s[k]) for s in row["stats"].values() for k in ("max_abs","l2"))
print("GRAD_FINITE=PASS profile=m8 kinds=3")
PY

```
