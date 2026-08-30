# v42-smoke5（闸门 N2 SMOKE5）launch

- **目的**：v3 破坏性重构计划的对拍闸门 N2——在 commitV4.2（transforms 瘦身）tip 上以确定性档跑 5 步，与 G0b 基线位级对拍，证明「删 recurrent/symbolic 键与分支」在交付面与训练标量上是恒等变换。
- **commit**：`c17a928`（commitV4.2 tip，clean HEAD 起跑）。
- **执行日期**：2026-08-29。
- **命令**（tmux detached，全局烟测口径，计划三节）：

```bash
STEPS=5 SAVE_INTERVAL=5 BATCH_DIGESTS=1 WARMUP_STEPS=0 KEEP_JAX_CACHE=1 \
EXP_NAME=v1-restructure-smoke RUN_TAG=v42-smoke5 \
DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
bash scripts/smoke-local/run_2gpu_epoch_bench.sh
```

- **环境**：本机 2×RTX 6000 Ada，b8，seed 42；编译缓存与 N1 共用（`EXP_NAME=v1-restructure-smoke` + `KEEP_JAX_CACHE=1`），训练段实测 7.1 s（热缓存；计划预估 30–40 min 为冷编译口径）。
- **A 侧**：`docs/training-doc/v1-grad-baseline-g0b/records/r1`（G 链固化基线，不重录）。
- **判据**（计划二节 N2 行，退出码与 DET_CHECK 总行不作判据，四分项逐行人工核对）：
  - `SCALARS steps=5 keys=5 hex_mismatch_steps=0`
  - raw `BATCH_DIGEST rows=3 mismatch=0`（raw 预期失配首现于步 100，5 步内必须全过）
  - `CANON_CHECK=PASS steps=3`
  - `STATE_DIGEST rows=1 mismatch=0`（步 0 初始 TrainState 逐叶摘要——rng 消耗序/构造顺序的直接证伪器）
  - `INDEX_SEQ=PASS`
  - 另核 `batch_digests.jsonl` 首行 `n_keys=12` 且键集无 `recur_*`/subgoal（比较器不产出该字段）
