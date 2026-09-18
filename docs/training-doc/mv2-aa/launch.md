# V8：开启态两轮 100 步 A/A

本任务按 0916 计划验证同一实现的可复现性和四个 motion 参数叶的更新。使用真新库、真实完整生产模型，GPU 4,5、batch8、fsdp2、workers4、seed42，各运行100步；两轮共用 EXP_NAME=mv2-aa 及同一 JAX 缓存，RUN_TAG 分别为 mv2-aa-a、mv2-aa-b。正式八卡 batch128 由后续 smoke 覆盖；本档带确定性和摘要记录，不用作吞吐结论。

## 版本与输入

从本页提交后的 clean CHECK_HEAD 启动，两轮及收尾均保持同一提交。入口记录完整 SHA 和空 porcelain。数据为 v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8，motion 表 metadata SHA 为 d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268，整表 SHA 为03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5；每轮开始前重新核对这两个指纹。

V8 显式沿用已锁定的新库 norm_stats：v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json，SHA 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。原双卡驱动的历史默认指向旧统计，所以只在本轮复制驱动的启动覆盖项中指定新路径，未改原驱动、全局默认或数据文件。训练入口的真实 loader 另记录实际统计数组与文件指纹，不能用另建配置冒充。

## 驱动还原

复制源为 scripts/training/g0/run_2gpu_epoch_bench.sh @ 81bd0217f890218a1134b135adcf3490cc68dd51。副本 v1-store/logs/mv2-aa-driver.sh 仅有以下四个改写点，其余逐字相同；实际 SHA 为4045850c4f445e29ac6696b607edf645d39535cf108b44adf55b2ffc50eaaa23。

| 稳定锚点 | 本轮准确覆盖 |
|---|---|
| 开头的 source paths.sh 命令 | source /scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/paths.sh |
| NORM_STATS 赋值 | ${TRAIN_ASSETS}/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json |
| ARGS 数组，紧随 --assets-base-dir | 增加 --data.assets.assets-dir "${TRAIN_ASSETS}/mme_vla_suite/4task-v2-1600ep-604f16da" 和 --data.assets.asset-id robomme |
| check_baseline_env.py dump 命令 | 增加 --norm-stats "${NORM_STATS}" |

## 判据、环境与产物

XLA_FLAGS=--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0，内存比例0.95，UV_NO_SYNC=1，缓存均在v1-store。存储AWS本地NVMe RAID /dev/md0 XFS，GPU A100-SXM4-80GB。两轮之间核对环境指纹及基线产物完整性；任一失败立即停止。

五项标量100步逐位同；完整状态摘要步0/25/50/75/99、原始与canonical batch摘要步0/1/2/25/50/75/99全部一致；前800个训练索引逐项一致；motion四键均非null；四个motion参数叶从初态到最后一步均发生更新。最终要求 AA_100、MOTION_PARAMS_UPDATED 与全部比较器判定通过，EXIT_CODE=0。

tmux全名mv2-aa，命令 `bash v1-store/logs/mv2-aa-runner.sh <CHECK_HEAD>`。外层日志mv2-aa.driver.log与两侧内部日志mv2-aa-a.log、mv2-aa-b.log分开。完整记录保留在v1-store/bench/2gpu-epoch-bench/mv2-aa-{a,b}，缓存保留供两轮复用；驱动只清理本轮无checkpoint的run壳。运行尚未开始。

## 完整外壳

```bash
#!/usr/bin/env bash
# 同一提交、同一缓存上的开启态 A/A；两轮只改变独立记录标签。
set -uo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA || exit 1
CHECK_HEAD="${1:?必须给出完整提交}"
LOG="$PWD/v1-store/logs/mv2-aa.driver.log"
test ! -e "$LOG" || exit 2
body() (
  set -euo pipefail
  test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
  test -z "$(git status --porcelain)"
  source scripts/training/paths.sh
  export UV_CACHE_DIR="$V1_STORE/cache/uv" UV_NO_SYNC=1 PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=1 WANDB_MODE=disabled CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  unset JAX_PLATFORMS PYTHONPATH MMEVLA_MOTION_STORE
  unset DTYPE_DUMP_LIMIT DTYPE_BASELINE_CHECKSUMS DTYPE_MANIFEST DTYPE_DUMP_IMPL
  unset BENCH_DATASET_IMPL BENCH_REF_SOURCE BENCH_REF_MANIFEST BENCH_REF_MOTION
  unset BENCH_REF_COMMIT BENCH_CAND_COMMIT BENCH_SAVE_FINAL_CKPT STATE_DUMP_STEPS BENCH_DUMP_IDX
  export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export CUDA_VISIBLE_DEVICES=4,5 BENCH_GPUS=4,5
  export STEPS=100 SAVE_INTERVAL=25 EXTRA_DIGEST_STEPS=99 WORKERS=4 WARMUP_STEPS=20
  export BATCH_DIGESTS=1 KEEP_JAX_CACHE=1 EXP_NAME=mv2-aa
  export HISTORY_CONFIG=perceptual-framesamp-modul-8frame-8x8-motion.yaml
  LIB="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
  export DATASET_PATH="$LIB/framesamp-8x8"
  export MMEVLA_FRAMESAMP_SOURCE="$LIB/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$LIB/meta/episode_manifest.json"
  NORM="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json"
  test "$(tail -n 1 "$V1_STORE/logs/mv2-m4.driver.log")" = EXIT_CODE=0
  test "$(tail -n 1 "$V1_STORE/logs/mv2-v4.driver.log")" = EXIT_CODE=0
  test "$(tail -n 1 "$V1_STORE/logs/mv2-rhythm-rep.driver.log")" = EXIT_CODE=0
  printf 'CHECK_HEAD=%s\nSTART_UTC=%s\n' "$CHECK_HEAD" "$(date -u +%FT%TZ)"
  for side in a b; do
    test "$(nvidia-smi --id=4,5 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = 0
    export RUN_TAG="mv2-aa-$side"
    test ! -e "$V1_STORE/train-runs/$RUN_TAG"
    test ! -e "$V1_STORE/bench/2gpu-epoch-bench/$RUN_TAG"
    uv run --no-sync python - "$LIB" "$NORM" <<'PY'
import hashlib,pathlib,sys
lib,norm=map(pathlib.Path,sys.argv[1:])
assert hashlib.sha256(norm.read_bytes()).hexdigest()=='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173'
assert hashlib.sha256((lib/'motion/meta/store_meta.json').read_bytes()).hexdigest()=='d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268'
assert hashlib.sha256((lib/'motion/motion_token.f32.bin').read_bytes()).hexdigest()=='03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5'
print('AA_INPUTS=PASS norm_stats=1 motion_meta=1 motion_table=1',flush=True)
PY
    if [[ "$side" == b ]]; then
      uv run --no-sync python scripts/training/g0/check_baseline_env.py check \
        --base "$V1_STORE/bench/2gpu-epoch-bench/mv2-aa-a" --dataset "$LIB/source" \
        --norm-stats "$NORM" --steps 100 --batch-size 8
    fi
    bash v1-store/logs/mv2-aa-driver.sh
    REC="$V1_STORE/bench/2gpu-epoch-bench/$RUN_TAG"
    uv run --no-sync python scripts/training/tests/project_scalars.py "$REC/metrics.jsonl" "$REC/scalars_hex.tsv"
    uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$REC"
    uv run --no-sync python - "$REC" "$CHECK_HEAD" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); meta=json.loads((root/'run_meta.json').read_text())
assert meta['start_head']==sys.argv[2] and meta['start_status']==''
assert meta['norm_stats_file_sha256']=='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173'
assert meta['norm_stats_actual']==meta['norm_stats_expected']
print('AA_RUNTIME_NORM=PASS',flush=True)
PY
    test "$(git rev-parse HEAD)" = "$CHECK_HEAD"
    test -z "$(git status --porcelain)"
  done
  uv run --no-sync python scripts/training/tests/finish_check.py aa \
    --base "$V1_STORE/bench/2gpu-epoch-bench/mv2-aa-a" --candidate "$V1_STORE/bench/2gpu-epoch-bench/mv2-aa-b" \
    --base-head "$CHECK_HEAD" --candidate-head "$CHECK_HEAD"
)
body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
```
