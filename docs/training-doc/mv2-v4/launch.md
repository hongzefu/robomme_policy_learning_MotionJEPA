# V4：新库数据交付逐位检查

本任务覆盖 motion 表到训练样本的读取、位置码、mask、右填充和 mem_order。用户要求按 0916 计划实施并保持严格逐位判据。新库建库已通过；本任务尚未执行。

## 版本与数据

从本页提交后的 clean CHECK_HEAD 启动，完整 SHA 作为外壳参数，并在起止与 git status --porcelain 一起检查。生产实现已由 f6915f2a09443d48c1bb57e9b5f83e95400704fd 锚定，此后当前提交只增加归档和启动文档。运行期间不改代码。

数据为 v1-store/datasets/4task-v2-1600ep-604f16da，framesamp-8x8 和 motion 均 verified。motion store_meta SHA 固定为 d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268，71316 行、demo 17/exec 33、repeat_last、budget 160。验证在 CPU 执行，存储为 AWS 本地 NVMe RAID /dev/md0 XFS。

## 覆盖与启动

独立手算所有 1600 个冷启动样本、1600 个首 exec 窗样本、全部 32 个 k≥140 样本及 16 种尾帧余数；实际去重后的样本数由报告记录。逐字节核对表行、时间码、零 padding、mask 和交错置换；V4_COVER 必须精确满足 cold=1600、first_exec=1600、kmax=141、kge140=32、pad_r=16/16。本检查的 state 统计是组件夹具，真实训练所用 norm_stats 由后续 V8/V9 另外核对。

使用 tmux 全名 mv2-v4，可与只占 GPU 4 的 mv2-m4 并行。命令为 `bash v1-store/logs/mv2-open-runner.sh v4 <CHECK_HEAD>`。日志为 v1-store/logs/mv2-v4.driver.log，报告为 v1-store/reports/motion/mv2-v4.json；EXIT_CODE 必须为 0。GPU 与长训练性能不属于本任务的结论。

## 完整共用外壳

V4、M4 和 rhythm 共用以下外壳，分别传 stage 参数；全部结果各有独立路径，拒绝覆盖。代码保存在 v1-store/logs/mv2-open-runner.sh。

```bash
#!/usr/bin/env bash
# 开启态验证入口；建库全部验收并提交留档后，传入新的 clean CHECK_HEAD 执行。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
STAGE="${1:?阶段 v4/m4/rhythm}"
CHECK_HEAD="${2:?完整提交}"
case "$STAGE" in v4|m4|rhythm) ;; *) exit 2 ;; esac
LOG="$PWD/v1-store/logs/mv2-$STAGE.driver.log"
test ! -e "$LOG" || exit 2
body() (
  set -euo pipefail
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  source scripts/dataset/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 UV_NO_SYNC=1
  export HF_HUB_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  unset MMEVLA_MOTION_STORE PYTHONPATH DTYPE_DUMP_DIR DTYPE_BASELINE_CHECKSUMS
  LIB="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  test "$(tail -n 1 "$LIB/logs/mv2-pack.driver.log")" = EXIT_CODE=0
  export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
  printf 'CHECK_HEAD=%s\nSTAGE=%s\nSTART_UTC=%s\n' "$CHECK_HEAD" "$STAGE" "$(date -u +%FT%TZ)"
  case "$STAGE" in
    v4)
      uv run --no-sync python scripts/training/tests/hand_calc_8frame.py \
        --store "$LIB/framesamp-8x8" --manifest "$LIB/meta/episode_manifest.json" --motion "$LIB/motion" \
        --yaml perceptual-framesamp-modul-8frame-8x8-motion.yaml --v2-coverage \
        --out "$V1_STORE/reports/motion/mv2-v4.json"
      ;;
    m4)
      test "$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
      export CUDA_VISIBLE_DEVICES=4 JAX_PLATFORMS=cuda
      export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
      uv run --no-sync python scripts/training/tests/motion_gates_model.py --gate m4 \
        --lib "$LIB" --store-subdir framesamp-8x8 --integration modulation --motion-budget 160 \
        --paligemma-variant gemma_150m --action-expert-variant gemma_300m --det-probes 3 \
        --out "$V1_STORE/reports/motion/mv2-m4.json"
      ;;
    rhythm)
      uv run --no-sync python scripts/training/tests/eval_rhythm_gates.py --gate all --lib "$LIB" \
        --yaml perceptual-framesamp-modul-8frame-8x8-motion.yaml --out "$V1_STORE/reports/motion/mv2-rhythm.json"
      ;;
  esac
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
```
