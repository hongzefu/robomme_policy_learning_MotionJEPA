# motion-t1-closed（T1：关闭态训练等价，旧库 1000 步，对 G0b r1 黄金基线）launch

- **目的**：`motion-memory-plan.md` 5.2 T1 / 四节表一——S2 接线完成后，`motion.enabled=false` 的新代码在旧库 `4task-gl-framesamp` 上
  1000 步 × batch 8，`scalars_hex.tsv` 必须逐位命中黄金锚点 `c799a0b2…`、`g0_gate.py --profile t1` 唯一成功行 `G0_EQ=PASS`。证明的是**代码等价**。
- **run_name**：`motion-t1-closed`（用户 2026-09-03 批准）；`EXP_NAME=RUN_TAG=motion-t1-closed`。
- **commit**：S2 合入后的 clean HEAD（sha 在 `result.md` 回填）。相对 A21（`c5925d9`）的差异 = 全部 S2 改动（5.1 一览表 21 项）。
- **口径**：与 A21 完全相同（`run_2gpu_epoch_bench.sh`，`HISTORY_CONFIG=perceptual-framesamp-context.yaml`，b8 / seed 42 / fsdp 2 / WORKERS 4 /
  STEPS 1000 / SAVE_INTERVAL 100 + EXTRA_DIGEST_STEPS 299 / 确定性档 XLA_FLAGS / MEM_FRACTION 0.95 / CUDA 0,1，`DATASET_PATH=v1-store/datasets/4task-gl-framesamp`）。
  新版驱动脚本从 `store_meta.json.num_exec_samples` 读 395,289（不再硬编码）；`HISTORY_CONFIG` 显式取 closed。
- **前置**：A21 `G0_EQ=PASS`（`c5925d9`）；`check_baseline_env.py check --baseline …/v1-grad-baseline-g0b/records/r1 --dataset v1-store/datasets/4task-gl` 再过一次 `BASELINE_ENV=PASS`；
  A13–A17 / M1–M5 / A18–A20 全过（`docs/training-doc/motion-t3-open/` 汇总）。

## 起跑命令

```bash
tmux new-session -d -s motion-t1-closed "set -o pipefail; cd /data/hongzefu/robomme_policy_learning_MotionJEPA; echo HEAD=\$(git rev-parse HEAD); \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 HISTORY_CONFIG=perceptual-framesamp-context.yaml \
  EXP_NAME=motion-t1-closed RUN_TAG=motion-t1-closed DATASET_PATH=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  bash scripts/training/g0/run_2gpu_epoch_bench.sh 2>&1 | tee v1-store/logs/motion-t1-closed-driver.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/motion-t1-closed-driver.log"
```

## 判读

同 A21：`compare_baseline.py <G0b r1> <records> --tier t1-vs-g0b` → `project_scalars.py` → `g0_gate.py --profile t1 … --expect-sha256 c799a0b2… --env-out <preflight>`；唯一成功行 `G0_EQ=PASS`。
另核 `batch_digests.jsonl` 首行 `n_keys=12`、`param_checksums.jsonl` `n_leaves=177`（关闭态两个新模块根本不创建，plan 2.9）。
