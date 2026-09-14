# 8×8 数据装配验收启动记录

用户要求“一口气全做完”，本阶段按完整计划先跑40ep开发库，再验400ep正式库。参考版本为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，候选版本为 `c08ec2060a544af1869c1e24f755e536150569ca`；候选启动HEAD允许仅增加档案提交，实际值记入日志。四个profile分别验C32、M32旧链回归以及C8、M8的新库对源npy搬运等价，profile间不比较数值。

本阶段纯CPU，AWS本地NVMe RAID，GPU不可见。40ep每档定点配额20、随机1000，全部合法帧索引与11530个样本身份全核验；raw和transformed所有键、形状、dtype及None状态逐项比较，每profile完整200个batch。两侧保留数组用于失配定位；不设DTYPE_DUMP_LIMIT。归一化显式使用400ep文件，SHA `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`。

40ep的worker矩阵两种实现各跑0/1/4/16 workers ×2个完整epoch，每档2882个batch，要求主进程fd回到基线。该生命周期专测与手算脚本使用各自既有的轻量统计量，数值等价由前述真实400ep norm_stats的fixture承担。手算独立核对两种motion开关、段边界和8帧公式。

会话 `p8-fx-40-dump` 与 `p8-fx-40-matrix` 分别运行下面代码；外层使用pipefail、tee、EXIT_CODE，日志为 `v1-store/logs/<会话名>.log`。从clean HEAD启动。REF侧显式PYTHONPATH指REF/src、共享主树uv环境；候选侧清除此覆盖。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
FX_REF="$V1_STORE/worktrees/ref-8x8"
FX_MAIN="$REPO_ROOT"
L="$V1_STORE/datasets/4task-motion-40ep"
export DTYPE_MANIFEST="$L/meta/episode_manifest.json" MMEVLA_MOTION_STORE="$L/motion"
for profile in c32 m32 c8 m8; do
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
    DTYPE_DUMP_DIR="$V1_STORE/fixtures/8x8/40ep/$side-$profile" DTYPE_DUMP_IMPL="$impl" DTYPE_DUMP_GIT_HEAD="$(git rev-parse HEAD)" \
      uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite \
      --exp-name "fx40-$profile-$side" --assets-base-dir "$V1_STORE/train-assets" \
      --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
      --dataset-path "$input" --model.use-history --model.history-config "$yaml" --no-wandb-enabled
  done
  cd "$FX_MAIN"
  unset PYTHONPATH
  printf 'FIXTURE_COMPARE profile=%s\n' "$profile"
  uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py "$V1_STORE/fixtures/8x8/40ep/ref-$profile" "$V1_STORE/fixtures/8x8/40ep/cand-$profile"
done
uv run --no-sync python scripts/training/tests/hand_calc_8frame.py --store "$L/framesamp-8x8" --manifest "$L/meta/episode_manifest.json" --motion "$L/motion"
printf 'FIXTURE_40=DONE\n'
```

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
L="$V1_STORE/datasets/4task-motion-40ep"
for impl in packed refnpy; do
  printf 'MATRIX_START impl=%s\n' "$impl"
  uv run --no-sync python scripts/training/tests/spawn_matrix.py --store "$L/framesamp-8x8" --source "$L/source" --manifest "$L/meta/episode_manifest.json" --yaml perceptual-framesamp-context-8frame-8x8.yaml --workers 0,1,4,16 --epochs 2 --impl "$impl"
done
```

判据逐profile为SOURCE_IDENTITY、FRAME_INDEX_EXACT、SAMPLE_RAW_EXACT及BATCH_RAW_EXACT全部PASS，另外两条MATRIX与C8/M8各自HAND_CALC_8FRAME必须PASS。400ep完整取证起跑前补充对应命令和来源，不能用40ep结果替代正式覆盖。阶段2的29个错误输入拒绝用例已经通过，最终档案统一引用。
