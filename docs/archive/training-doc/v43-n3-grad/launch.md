# v43-n3（闸门 N3 GRAD_FIXTURE）launch

- **目的**：v3 重构计划对拍闸门 N3——在 commitV4.3（模型侧单一化）tip 上跑单步定点
  梯度取证，与固化 A 侧对拍。`compute_loss` 去 stats、`nnx.value_and_grad` 去
  `has_aux`（本轮唯一可能动 jaxpr 的改动）的直接证伪器。
- **commit**：`07702f0`（commitV4.3 tip，clean HEAD 起跑，porcelain 空——V4.4 的
  三份未跟踪新文件起跑前暂移 scratchpad）。
- **执行日期**：2026-08-29。
- **命令**（tmux detached；驱动写死确定性档 XLA_FLAGS 与 2 卡口径）：

```bash
RUN_TAG=v43-n3 \
DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
bash scripts/dtype-unify/run_dtype_grad.sh
```

- **A 侧**：`docs/training-doc/v1-dtype-p5-grad/records/`（固化件，`same_origin=PASS`；
  三定点 batch × 32 梯度叶 sha + loss_hex），不现场采集。
- **对拍命令**：

```bash
JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run scripts/dtype-unify/compare_dtype_fix.py \
  --grad-a docs/training-doc/v1-dtype-p5-grad/records \
  --grad-b v1-store/dtype-unify/v43-n3-grad
```

- **判据**：`COMPARE_GRAD=PASS kinds=3 mismatches=0`（`allfull` 阴性对照必过）。
- **护栏**：`single_step_grad.py` 的 `_guard_train_step_source` 五 needle（含 V4.3
  新增的 `"return new_state, info"`）与 G0b 步 0 同源校验在驱动内强制执行。
