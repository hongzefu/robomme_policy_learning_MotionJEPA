# V-online：全量装配、stub 与分层真实编码三档验证

用户确认方案B，并补充「采用预生成并校验的 GPU 位置表，CPU 验证装配与次序」。三档已从同一clean CHECK_HEAD=`782696c231aace21c20200638ae302a5a1c7f277` 完整通过，详见 [结果](result.md)。期间未改代码或提交文档，聚合器已核各片及装配/真实编码档的源码身份。以下保留完整启动口径。

## 覆盖与边界

assembly在CPU逐窗复现FrameSampMemory装配，对全部71316窗的33帧原始字节SHA与离线元数据比较，覆盖全部1600补帧窗。stub在CPU覆盖全1600集，核起点集合、motion_pos/mask/mem_order及错误token处理。real以全部1600补帧窗加独立分层选择的整集走真实sidecar，其余窗注入已验证的离线行；必须包含k最大g367、至少三条es≥1000、16种尾帧余数及每个任务至少一集。只把实际调用encoder的行计入compared_real。

每档八片，分配规则为g % 8，各片episode两两不交、并集1600集，窗口并集71316。逐位比较使用原始字节，区分符号零；任一模式下的失配均阻断，stub不能短路失败。真实编码档会引用已通过的全量assembly报告。

## 位置表、代码与数据身份

CPU非装配档使用v1-store/reports/motion/mv2-posemb-v2-verified.json，其4096×64×768 float32位置表由原PosEmb3D在GPU生成，与离线2304行整表逐位核对。生成时为cleanf6915f2a09443d48c1bb57e9b5f83e95400704fd；加载时重新核对生产组件与导出器SHA、数据来源、shape、dtype和表字节SHA。该证明允许文档提交后复用同一未变组件输出；生产在线推理入口没有因此改为读取离线motion表。

本轮数据根v1-store/datasets/4task-v2-1600ep-604f16da，history YAML为perceptual-framesamp-modul-8frame-8x8-motion.yaml。比较前清空MMEVLA_MOTION_STORE并核YAML指向本库；encoder目录由source_run解析，握手核新checkpoint与数值provenance。所有原始H5和模型均为本机已核验副本。

## 启动与产物

严格顺序：mv2-vonl-a（assembly，CPU）→mv2-vonl-b（stub，CPU）→mv2-vonl-c（real，独占GPU0–7的八个sidecar）。主进程始终JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES为空，real子进程显式绑定--gpu 0..7。线程数1，uv --no-sync，不修改依赖。存储为AWS本地NVMe RAID /dev/md0 XFS。

命令依次为 `bash v1-store/logs/mv2-online-runner.sh assembly <CHECK_HEAD>`、stub、real。每片日志单独tee并记录SHARD_EXIT_CODE，全部成功后才结构化聚合。报告为v1-store/reports/motion/p5_online_v2_1600ep.<phase>.shard<i>of8.json和<phase>.aggregate.json；驱动日志mv2-vonl-{a,b,c}.driver.log。全部分片、聚合、总退出码必须为0。本任务不计算策略成功率。

## 完整外壳

```bash
#!/usr/bin/env bash
# 三档在线验证分片入口；每档八片均成功后才做结构化聚合。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
PHASE="${1:?assembly/stub/real}"
CHECK_HEAD="${2:?完整提交}"
case "$PHASE" in assembly) STAGE=vonl-a; PREV=aa ;; stub) STAGE=vonl-b; PREV=vonl-a ;; real) STAGE=vonl-c; PREV=vonl-b ;; *) exit 2 ;; esac
LOG="$PWD/v1-store/logs/mv2-$STAGE.driver.log"
test ! -e "$LOG" || exit 2
body() (
  set -euo pipefail
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  source scripts/dataset/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" UV_NO_SYNC=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
  export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  unset MMEVLA_MOTION_STORE PYTHONPATH NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE CUBLAS_WORKSPACE_CONFIG
  LIB="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  OUT="$V1_STORE/reports/motion"
  CACHE="$OUT/mv2-posemb-v2-verified.json"
  test "$(tail -n 1 "$V1_STORE/logs/mv2-$PREV.driver.log")" = EXIT_CODE=0
  if [[ "$PHASE" == real ]]; then
    test "$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = 0
  fi
  if [[ "$PHASE" != assembly ]]; then
    uv run --no-sync python - "$OUT/p5_online_v2_1600ep.assembly.aggregate.json" "$CHECK_HEAD" <<'PY'
import json,pathlib,sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d['passed'] and d['source_head']==sys.argv[2]
PY
  fi
  printf 'CHECK_HEAD=%s\nPHASE=%s\nSTART_UTC=%s\n' "$CHECK_HEAD" "$PHASE" "$(date -u +%FT%TZ)"
  options=()
  case "$PHASE" in
    assembly) options=(--assembly-hash "$LIB/wan-latents") ;;
    stub) options=(--stub --gpu-posemb-cache "$CACHE") ;;
    real) options=(--real-strata --gpu-posemb-cache "$CACHE") ;;
  esac
  pids=(); reports=()
  for i in 0 1 2 3 4 5 6 7; do
    report="$OUT/p5_online_v2_1600ep.$PHASE.shard${i}of8.json"
    shard_log="$V1_STORE/logs/mv2-$STAGE-shard$i.log"
    test ! -e "$report" && test ! -e "$shard_log"
    reports+=("$report")
    (
      set +e
      set -o pipefail
      uv run --no-sync python scripts/training/g0/compare_online_motion.py \
        --lib "$LIB" --yaml perceptual-framesamp-modul-8frame-8x8-motion.yaml --store-subdir framesamp-8x8 \
        --shard-idx "$i" --num-shards 8 --gpu "$i" --out "$report" "${options[@]}" 2>&1 | tee "$shard_log"
      code=${PIPESTATUS[0]}
      printf 'SHARD_EXIT_CODE=%s shard=%s phase=%s\n' "$code" "$i" "$PHASE" | tee -a "$shard_log"
      exit "$code"
    ) &
    pids+=("$!")
    printf 'ONLINE_SHARD_PID=%s shard=%s\n' "$!" "$i"
  done
  bad=0
  for pid in "${pids[@]}"; do if wait "$pid"; then :; else bad=1; fi; done
  test "$bad" = 0
  extra=()
  if [[ "$PHASE" == real ]]; then extra=(--assembly-report "$OUT/p5_online_v2_1600ep.assembly.aggregate.json"); fi
  uv run --no-sync python scripts/training/g0/compare_online_motion.py aggregate --reports "${reports[@]}" \
    --lib "$LIB" --num-shards 8 --expect-episodes 1600 --expect-rows 71316 \
    --out "$OUT/p5_online_v2_1600ep.$PHASE.aggregate.json" "${extra[@]}"
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
```
