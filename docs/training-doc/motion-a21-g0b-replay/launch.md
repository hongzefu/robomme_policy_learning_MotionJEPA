# motion-a21-g0b-replay（A21：S2 起工前 HEAD 原样复跑 G0b 黄金基线，1000 步）launch

- **目的**：`motion-memory-plan.md` 第二部分四节表二 A21 / 六节 runbook 第一步——S2 改任何模型代码之前，用当前 clean HEAD
  把 G0b 黄金基线原样复跑一遍，取得 `G0_EQ=PASS`；否则 T1 挂了分不清「基线腐烂 / 环境漂移 / 代码问题」。
  同时它兼作 S1 重构后的第四层检验（`AGENTS.md` 第 18 条第二块）：本 commit 相对 `442a7b9` 的 `src/` 改动只有新增
  `datastore/motion_store.py`（无人 import）与 `framesamp_store.py` 的一行注释，训练语义应逐位不变。
- **run_name**：`motion-a21-g0b-replay`（用户 2026-09-03 一次性批准的 8 个名字之一）；`EXP_NAME=RUN_TAG=motion-a21-g0b-replay`
  独立编译缓存（同 G3 口径，不复用任何烟测缓存）。
- **commit**：起跑时 `git rev-parse HEAD`（clean，porcelain 空）写入 `result.md`；S1 建库链在同一 HEAD 上并行收尾。
- **口径（与 G3 `v1-postclean-g3` 逐项相同）**：`scripts/training/g0/run_2gpu_epoch_bench.sh`，2×RTX 6000 Ada、b8、seed 42、`fsdp_devices=2`、
  `WORKERS=4`、`STEPS=1000`、`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`（TrainState 摘要 12 次、输入摘要 14 次）、
  `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`CUDA_VISIBLE_DEVICES=0,1`，
  `DATASET_PATH=v1-store/datasets/4task-gl-framesamp`（只读 symlink → turbo 归档；packed 库 `status=verified`，
  `store_meta_sha256=3990165c…`、`manifest_sha256=20da0dfe…`）。单 epoch 约束 1000×8 = 8,000 < 395,289。
- **与计划文字的一处偏差**：计划 A21 写「逐字复现 run_meta.json 的 argv（`--dataset-path v1-store/datasets/4task-gl`）」；
  但 commitV4.1 起 packed `FrameSampDataset` 是唯一数据路径、legacy 源根不再可读，G2/G3 已证 `4task-gl-framesamp` 上的
  scalars 与 G0b 逐位相同（`c799a0b2…` 第五份同值），故本 run 按 G3 口径用 framesamp 根，其余 argv 逐项相同。
- **存储介质**：数据集在 turbo NFS（symlink），与 G0b/G3 同介质；util / 步时只留档不作性能结论（红线 B7）。

## 前置门

```bash
cd /data/hongzefu/robomme_policy_learning_MotionJEPA
git status --porcelain | wc -l          # 须为 0
CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  UV_LINK_MODE=copy JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check \
  --baseline docs/training-doc/v1-grad-baseline-g0b/records/r1 --steps 1000 --batch-size 8 | tee v1-store/logs/motion-a21-g0b-replay.preflight.log
# 须 BASELINE_ENV=PASS；nvidia-smi 两卡空闲（S1 链 A/B 全部结束后再起）
```

## 起跑命令（tmux detached）

```bash
tmux new-session -d -s motion-a21-g0b-replay "set -o pipefail; cd /data/hongzefu/robomme_policy_learning_MotionJEPA; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=motion-a21-g0b-replay RUN_TAG=motion-a21-g0b-replay \
  DATASET_PATH=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  bash scripts/training/g0/run_2gpu_epoch_bench.sh 2>&1 | tee v1-store/logs/motion-a21-g0b-replay-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/motion-a21-g0b-replay-driver.log"
```

记录目录 `v1-store/bench/2gpu-epoch-bench/motion-a21-g0b-replay/`；预计 ≈1.5 h。

## 判读

```bash
R=v1-store/bench/2gpu-epoch-bench/motion-a21-g0b-replay
UV_LINK_MODE=copy uv run --no-sync python scripts/training/g0/compare_baseline.py docs/training-doc/v1-grad-baseline-g0b/records/r1 $R --tier a21-vs-g0b | tee $R/compare_vs_g0b_r1.txt
UV_LINK_MODE=copy uv run --no-sync python scripts/training/tests/g0_gate.py --compare-out $R/compare_vs_g0b_r1.txt --run-dir $R \
  --scalars $R/scalars_hex.tsv --expect-sha256 c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757 \
  --env-out v1-store/logs/motion-a21-g0b-replay.preflight.log
```

唯一成功行 `G0_EQ=PASS`（内含 `SCALARS 1000/5/0`、`STATE_DIGEST 12/0`、`BATCH_DIGEST_CANONICAL 14/0`、`CANON_CHECK=PASS/14`、
`INDEX_SEQ=PASS n≥8072`、`scalars_hex.tsv` sha256 命中 `c799a0b2…`、`n_keys=12`、`BASELINE_ENV=PASS`）。
