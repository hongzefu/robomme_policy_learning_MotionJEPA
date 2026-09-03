# motion-a22-grad — A22 单步定点梯度（launch）

- **目的**：motion-memory-plan.md 四节表二 **A22**——S2/S3 合入后的关闭态代码，在三个定点 batch（`mixed1` / `allshort` / `allfull`）上做单步前向反向，
  逐叶梯度 sha256 + 单步 loss `float.hex()` 与既有基线逐位比对。基线 = `docs/training-doc/v1-dtype-p5-grad/records/grad_summary.json`（dtype 统一修复后、
  commitV4.x 时代的同一模型 / 同一定点计划；其 32 个梯度叶子 = 可训练参数集合）。`allfull` 为阴性对照。
- **run 名**：`motion-a22-grad`（≤30 min 诊断 run，按 `AGENTS.md` 第 6 条短测自命名、第 17 条留档；不在 2026-09-03 批准的八个正式 run_name 之列，特此注明）。
- **入口**：`scripts/training/tests/run_dtype_grad.sh`（内部调 `tests/single_step_grad.py`；脚本自建 fixture、拒绝覆盖既有产物、要求 clean HEAD）：

```bash
tmux new-session -d -s motion-a22-grad "set -o pipefail; cd /data/hongzefu/robomme_policy_learning_MotionJEPA; \
  RUN_TAG=motion-a22-grad DATASET_PATH=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-gl-framesamp PYTHONUNBUFFERED=1 \
  bash scripts/training/tests/run_dtype_grad.sh 2>&1 | tee v1-store/logs/motion-a22-grad.wrapper.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/motion-a22-grad.wrapper.log"
# 对拍
JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run --no-sync python scripts/training/tests/compare_dtype_fix.py \
  --grad-a docs/training-doc/v1-dtype-p5-grad/records --grad-b v1-store/dtype-unify/motion-a22-grad-grad --report v1-store/reports/motion/a22_grad_report.json
```

- **口径**：2×RTX 6000 Ada、b8、seed 42、`fsdp_devices=2`、`XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`、`perceptual-framesamp-context.yaml`（关闭态）；
  初始 TrainState 现场 `init_train_state` 并与 G0b r1 `param_checksums.jsonl` step 0 逐叶核对同源（177 叶）。
- **判据**：`COMPARE_GRAD=PASS kinds=3 mismatches=0`（输入侧 batch canonical 摘要一致 + 32 叶梯度 sha256 逐叶相等 + loss hex 相等；`allfull` 失配单独标越界告警）。
