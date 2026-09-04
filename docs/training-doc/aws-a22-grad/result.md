# aws-a22-grad — 结果

- **起跑**：2026-09-04 07:15 → 07:27，GPU6,7；HEAD 侧树 `cbf24e9`（代码 == commitV6.12 `8093ebd` + `fix:` eval 驱动覆盖 `cbf24e9`；`git status` 有 2 个未提交**文档**改动，无代码改动）；
  旧码侧 worktree `c5925d96305f771058e2206ae89461269af9d97c`（clean）。fixture 400 ep 库（`records/grad_summary.*.json` 的 `results.*.indices`）。
- **判定行**（`records/aws-a22-grad-compare.txt`）：

```
GRAD_EQ=PASS kinds=3 leaves=32 mismatches=0
```

- **两侧原始数字（逐字相同）**：`mixed1 loss=0.626972 / allshort 0.250215 / allfull 0.735208`，每 kind 32 个梯度叶（`config.trainable_filter` 下 `_leaf_items(grads)` 的叶数）；
  `init_train_state` 14.3 s / 14.6 s；三 batch 梯度 92.5+64.0+64.0 s vs 92.9+64.2+64.5 s。
- **意义**：在 A100 + 确定性档下，motion 接线后的 HEAD 关闭态与接线前旧码，对同一 fixture（短样本 / 满长 / 混合三档 batch）的单步 loss 与全部 32 叶梯度**逐位相同**；
  与 `aws-t2-*`（100 步 TrainState 逐位）互为印证。不与环境 A `v1-dtype-p5-grad` 的 sha 比（跨架构不逐位）。
- **records/**：`grad_summary.head.json`、`grad_summary.base.json`（含逐叶 sha256、`stats` 的 max_abs / l2、`batch_keys` 摘要）、`aws-a22-grad-compare.txt`、`driver_summary.txt`。
- **盲区**：只覆盖 closed YAML；`DTYPE_BASELINE_CHECKSUMS` 置空即未做「初态与 G0b 同源」核（那份基线在环境 A）；HEAD 侧起跑时工作区有两个文档文件未提交（AGENTS 第 17 条口径下写明）。
