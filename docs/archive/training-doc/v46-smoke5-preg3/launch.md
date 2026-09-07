# v46-smoke5（预 G3 闸：SMOKE5 新布局重跑）launch

- **目的**：commitV4.6（scripts/ 目录统一）tip 上、以最终两域布局的新入口重跑
  SMOKE5，判据同 N2（含 STATE_DIGEST rows=1），作为 G3 起跑前的最后一道对拍闸。
- **commit**：`b30be80`（V4.6 tip，clean HEAD）。执行日期 2026-08-30。
- **命令**（新入口 scripts/training/bench/run_2gpu_epoch_bench.sh；参数为计划三节
  全局烟测口径）：

```bash
STEPS=5 SAVE_INTERVAL=5 BATCH_DIGESTS=1 WARMUP_STEPS=0 KEEP_JAX_CACHE=1 \
EXP_NAME=v1-restructure-smoke RUN_TAG=v46-smoke5 \
DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
bash scripts/training/bench/run_2gpu_epoch_bench.sh
```

- **A 侧**：G0b r1 固化件。判据同 N2；另核 n_keys=12 与 packed 库本体锚点
  （env.json 顶层 store_meta_sha256 / manifest_sha256 vs G2 records/env.json）。
