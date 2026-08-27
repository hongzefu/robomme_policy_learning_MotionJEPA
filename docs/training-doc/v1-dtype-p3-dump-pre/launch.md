# v1-dtype-p3-dump-pre（dtype 修复第一块 A 侧取证 + 单步梯度 A 侧）

- **目的**：在**修复尚未落地**的 clean HEAD 上取一次证——定点样本 / 定点 batch 的
  交付内容，以及三个定点 batch 的单步梯度。dtype 修复没有运行时开关（改动即行为），
  「修复前」的样子只能在修复代码写进文件之前取到，事后无法重建。本 run 因此是
  P4/P5 全部对拍的 A 侧唯一来源。
- **起跑 commit**：`f2e7348`（commitV2.4a，验证工具落地；**不含任何功能修复**，
  `src/` 下零改动）。起跑前 `git status --porcelain` 为空（驱动脚本内置硬闸）。
- **为什么单步梯度 A 侧并入本 run**：dtype 计划 T5 原把 P5 整体排在修复 commit
  之后，那时仓库里只剩修复后的代码、只能算出 B 侧。经用户 2026-08-27 裁定改为
  「并入 P3 提前算」，全程不必 checkout 回旧 commit，也没有脏工作区风险。
- **口径**：
  - dump：`JAX_PLATFORMS=cpu`，b8，`--dataset-path v1-store/datasets/4task-gl`，
    `perceptual-framesamp-context.yaml`；定点样本 ~2,600（step_idx ∈ {0,1,2,29,30}
    各 200 + {31,32,33} 各 200 + 固定 seed 随机 1,000）；定点 batch 200
    （mixed1 / allshort / allfull / random 各 50）。逐样本层走裸 `RoboMMEDataset`，
    batch 层走完整 `transform_dataset` + `_collate_fn`。
  - 单步梯度：2×RTX 6000 Ada、b8、seed 42、`fsdp_devices=2`、
    `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`（D2 档，
    正确性族口径）；三个定点 batch 各算一步；初始 state 现场 `init_train_state`
    并与 G0b r1 步 0 的 177 叶摘要逐条校验同源。
- **命令**：
  ```
  RUN_TAG=v1-dtype-p3-dump-pre bash scripts/dtype-unify/run_dtype_dump.sh
  RUN_TAG=v1-dtype-p3-dump-pre \
    FIXTURE_DIR=v1-store/dtype-unify/v1-dtype-p3-dump-pre-gradfix \
    GRAD_ARRAYS_DIR=/data/hongzefu/v1-baselines/dtype-p5-grad-pre \
    bash scripts/dtype-unify/run_dtype_grad.sh
  ```
  （tmux detached + `tee` + `EXIT_CODE=`，AGENTS 7；日志
  `v1-store/logs/v1-dtype-p3.log`）
- **产物**：
  - `v1-store/dtype-unify/v1-dtype-p3-dump-pre/`（samples/ + batches/ +
    fixture_plan.json + DUMP_MANIFEST.json）——不进 git，P4 对拍后清理；
  - `v1-store/dtype-unify/v1-dtype-p3-dump-pre-grad/grad_summary.json`——A 侧逐叶
    梯度摘要与统计；
  - `/data/hongzefu/v1-baselines/dtype-p5-grad-pre/`——A 侧梯度数组本体（约 11 GB ×
    3 个 batch），**本机盘、不留 NFS**（沿用 G0b state dump 先例）；按用户裁定
    **P6 验收通过后删除数组，只保留逐叶 sha256 清单与逐叶统计**。
- **数据事实声明**：短样本档（`step_idx ≤ 30`）只能取自 800 个 `exec_start_idx == 0`
  的 Button 系 episode。Video 系（VideoUnmask / VideoUnmaskSwap）的 `exec_start_idx`
  最小 66，其样本 `step_idx` 恒 ≥ 66、**永远走满长切片分支、根本产生不出短样本**。
  这是数据本身的性质，不是取样偏置；满长档与随机 1,000 自然覆盖两系。
