# 400ep完整输入验收启动记录

40ep四组完整fixture、两种实现的worker矩阵和独立手算已全部通过。本阶段待400ep新库全量verify与独立xgrid检查通过后，从clean HEAD启动。候选源码为 `c08ec2060a544af1869c1e24f755e536150569ca`，参考源码为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`；实际HEAD逐份记入日志和DUMP_MANIFEST。

四个profile使用四个独立CPU会话 `p8-fx-400-c32`、`p8-fx-400-m32`、`p8-fx-400-c8`、`p8-fx-400-m8` 并行，每个内部先参考、后候选、再比较。日志为 `v1-store/logs/<会话名>.log`，外层pipefail、tee、EXIT_CODE，显式禁止GPU。没有更改任何训练超参或模型；线程环境只用于本次CPU诊断。

每档定点配额200、另随机1000；C32/M32每侧3000个样本，C8/M8每侧3200个样本，每侧完整200个batch。每份检查400集、101066执行样本身份与123044个合法帧索引。所有键的raw字节、dtype、shape、None以及transform后输出必须一致。norm_stats始终显式400ep文件，SHA `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`。

## C32

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
FX_REF="$V1_STORE/worktrees/ref-8x8"
FX_MAIN="$REPO_ROOT"
L="$V1_STORE/datasets/4task-motion-400ep"
export DTYPE_MANIFEST="$L/meta/episode_manifest.json" MMEVLA_MOTION_STORE="$L/motion"
for profile in c32; do
  case "$profile" in
    c32) yaml=perceptual-framesamp-context.yaml; subdir=framesamp; refimpl=packed ;;
    m32) yaml=perceptual-framesamp-context-motion.yaml; subdir=framesamp; refimpl=packed ;;
    c8) yaml=perceptual-framesamp-context-8frame-8x8.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
    m8) yaml=perceptual-framesamp-context-8frame-8x8-motion.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
  esac
  for side in ref cand; do
    if [ "$side" = ref ]; then
      cd "$FX_REF"
      export PYTHONPATH="$FX_REF/src" UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl="$refimpl"
    else
      cd "$FX_MAIN"
      unset PYTHONPATH
      export UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl=packed
    fi
    input="$L/$subdir"
    if [ "$impl" = refnpy ]; then input="$L/source"; fi
    printf 'FIXTURE_START profile=%s side=%s impl=%s head=%s\n' "$profile" "$side" "$impl" "$(git rev-parse HEAD)"
    DTYPE_DUMP_DIR="$V1_STORE/fixtures/8x8/400ep/$side-$profile" DTYPE_DUMP_IMPL="$impl" DTYPE_DUMP_GIT_HEAD="$(git rev-parse HEAD)" \
      uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite \
      --exp-name "fx400-$profile-$side" --assets-base-dir "$V1_STORE/train-assets" \
      --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
      --dataset-path "$input" --model.use-history --model.history-config "$yaml" --no-wandb-enabled
  done
  cd "$FX_MAIN"
  unset PYTHONPATH
  printf 'FIXTURE_COMPARE profile=%s\n' "$profile"
  uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py "$V1_STORE/fixtures/8x8/400ep/ref-$profile" "$V1_STORE/fixtures/8x8/400ep/cand-$profile"
done
printf 'FIXTURE_400=DONE profile=c32\n'
```

## M32

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
FX_REF="$V1_STORE/worktrees/ref-8x8"
FX_MAIN="$REPO_ROOT"
L="$V1_STORE/datasets/4task-motion-400ep"
export DTYPE_MANIFEST="$L/meta/episode_manifest.json" MMEVLA_MOTION_STORE="$L/motion"
for profile in m32; do
  case "$profile" in
    c32) yaml=perceptual-framesamp-context.yaml; subdir=framesamp; refimpl=packed ;;
    m32) yaml=perceptual-framesamp-context-motion.yaml; subdir=framesamp; refimpl=packed ;;
    c8) yaml=perceptual-framesamp-context-8frame-8x8.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
    m8) yaml=perceptual-framesamp-context-8frame-8x8-motion.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
  esac
  for side in ref cand; do
    if [ "$side" = ref ]; then
      cd "$FX_REF"
      export PYTHONPATH="$FX_REF/src" UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl="$refimpl"
    else
      cd "$FX_MAIN"
      unset PYTHONPATH
      export UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl=packed
    fi
    input="$L/$subdir"
    if [ "$impl" = refnpy ]; then input="$L/source"; fi
    printf 'FIXTURE_START profile=%s side=%s impl=%s head=%s\n' "$profile" "$side" "$impl" "$(git rev-parse HEAD)"
    DTYPE_DUMP_DIR="$V1_STORE/fixtures/8x8/400ep/$side-$profile" DTYPE_DUMP_IMPL="$impl" DTYPE_DUMP_GIT_HEAD="$(git rev-parse HEAD)" \
      uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite \
      --exp-name "fx400-$profile-$side" --assets-base-dir "$V1_STORE/train-assets" \
      --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
      --dataset-path "$input" --model.use-history --model.history-config "$yaml" --no-wandb-enabled
  done
  cd "$FX_MAIN"
  unset PYTHONPATH
  printf 'FIXTURE_COMPARE profile=%s\n' "$profile"
  uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py "$V1_STORE/fixtures/8x8/400ep/ref-$profile" "$V1_STORE/fixtures/8x8/400ep/cand-$profile"
done
printf 'FIXTURE_400=DONE profile=m32\n'
```

## C8

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
FX_REF="$V1_STORE/worktrees/ref-8x8"
FX_MAIN="$REPO_ROOT"
L="$V1_STORE/datasets/4task-motion-400ep"
export DTYPE_MANIFEST="$L/meta/episode_manifest.json" MMEVLA_MOTION_STORE="$L/motion"
for profile in c8; do
  case "$profile" in
    c32) yaml=perceptual-framesamp-context.yaml; subdir=framesamp; refimpl=packed ;;
    m32) yaml=perceptual-framesamp-context-motion.yaml; subdir=framesamp; refimpl=packed ;;
    c8) yaml=perceptual-framesamp-context-8frame-8x8.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
    m8) yaml=perceptual-framesamp-context-8frame-8x8-motion.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
  esac
  for side in ref cand; do
    if [ "$side" = ref ]; then
      cd "$FX_REF"
      export PYTHONPATH="$FX_REF/src" UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl="$refimpl"
    else
      cd "$FX_MAIN"
      unset PYTHONPATH
      export UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl=packed
    fi
    input="$L/$subdir"
    if [ "$impl" = refnpy ]; then input="$L/source"; fi
    printf 'FIXTURE_START profile=%s side=%s impl=%s head=%s\n' "$profile" "$side" "$impl" "$(git rev-parse HEAD)"
    DTYPE_DUMP_DIR="$V1_STORE/fixtures/8x8/400ep/$side-$profile" DTYPE_DUMP_IMPL="$impl" DTYPE_DUMP_GIT_HEAD="$(git rev-parse HEAD)" \
      uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite \
      --exp-name "fx400-$profile-$side" --assets-base-dir "$V1_STORE/train-assets" \
      --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
      --dataset-path "$input" --model.use-history --model.history-config "$yaml" --no-wandb-enabled
  done
  cd "$FX_MAIN"
  unset PYTHONPATH
  printf 'FIXTURE_COMPARE profile=%s\n' "$profile"
  uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py "$V1_STORE/fixtures/8x8/400ep/ref-$profile" "$V1_STORE/fixtures/8x8/400ep/cand-$profile"
done
printf 'FIXTURE_400=DONE profile=c8\n'
```

## M8

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
FX_REF="$V1_STORE/worktrees/ref-8x8"
FX_MAIN="$REPO_ROOT"
L="$V1_STORE/datasets/4task-motion-400ep"
export DTYPE_MANIFEST="$L/meta/episode_manifest.json" MMEVLA_MOTION_STORE="$L/motion"
for profile in m8; do
  case "$profile" in
    c32) yaml=perceptual-framesamp-context.yaml; subdir=framesamp; refimpl=packed ;;
    m32) yaml=perceptual-framesamp-context-motion.yaml; subdir=framesamp; refimpl=packed ;;
    c8) yaml=perceptual-framesamp-context-8frame-8x8.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
    m8) yaml=perceptual-framesamp-context-8frame-8x8-motion.yaml; subdir=framesamp-8x8; refimpl=refnpy ;;
  esac
  for side in ref cand; do
    if [ "$side" = ref ]; then
      cd "$FX_REF"
      export PYTHONPATH="$FX_REF/src" UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl="$refimpl"
    else
      cd "$FX_MAIN"
      unset PYTHONPATH
      export UV_PROJECT_ENVIRONMENT="$FX_MAIN/.venv"
      impl=packed
    fi
    input="$L/$subdir"
    if [ "$impl" = refnpy ]; then input="$L/source"; fi
    printf 'FIXTURE_START profile=%s side=%s impl=%s head=%s\n' "$profile" "$side" "$impl" "$(git rev-parse HEAD)"
    DTYPE_DUMP_DIR="$V1_STORE/fixtures/8x8/400ep/$side-$profile" DTYPE_DUMP_IMPL="$impl" DTYPE_DUMP_GIT_HEAD="$(git rev-parse HEAD)" \
      uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite \
      --exp-name "fx400-$profile-$side" --assets-base-dir "$V1_STORE/train-assets" \
      --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
      --dataset-path "$input" --model.use-history --model.history-config "$yaml" --no-wandb-enabled
  done
  cd "$FX_MAIN"
  unset PYTHONPATH
  printf 'FIXTURE_COMPARE profile=%s\n' "$profile"
  uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py "$V1_STORE/fixtures/8x8/400ep/ref-$profile" "$V1_STORE/fixtures/8x8/400ep/cand-$profile"
done
printf 'FIXTURE_400=DONE profile=m8\n'
```

四组完成后在主树再执行本库的独立手算，两种motion开关各要求HAND_CALC_8FRAME为PASS：

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
L="$V1_STORE/datasets/4task-motion-400ep"
uv run --no-sync python scripts/training/tests/hand_calc_8frame.py \
  --store "$L/framesamp-8x8" --manifest "$L/meta/episode_manifest.json" --motion "$L/motion"
```

本轮最终输入判据为四组SOURCE_IDENTITY、FRAME_INDEX_EXACT、SAMPLE_RAW_EXACT、BATCH_RAW_EXACT全PASS，加本库C8/M8独立手算全PASS。worker矩阵按原计划只在40ep跑，不重复扩展400ep矩阵。原始数组留在v1-store，档案仅收摘要、计划、身份和清洗日志。
