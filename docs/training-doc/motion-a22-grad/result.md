# motion-a22-grad — 结果

- **判定：`COMPARE_GRAD=PASS kinds=3 mismatches=0`**（`records/a22_grad_report.json`）；`GRAD_PASS run_tag=motion-a22-grad`、`EXIT_CODE=0`（`records/wrapper.log`）。
- **起跑**：2026-09-03 15:40–15:46，HEAD `e48433e`（clean）。初始 state 同源校验 `PASS（177 叶子）`。
- **三个定点 batch**（样本 indices 与基线相同，输入 canonical 摘要一致）：

| batch | loss hex（基线 / 本次） | 梯度叶 | 失配 | 用时 |
|---|---|---|---|---|
| `mixed1` | `0x1.0d48f00000000p-1` / 同 | 32 | 0 | 76.8 s |
| `allshort` | `0x1.0f06060000000p-2` / 同 | 32 | 0 | 55.5 s |
| `allfull`（阴性对照） | `0x1.37890e0000000p-1` / 同 | 32 | 0 | 55.9 s |

- **意外**：前两次起跑失败——第一次 `run_dtype_grad.sh` 默认 `GL_DATASET=4task-gl`（legacy 散 npy，packed 模式拒绝），改传 `DATASET_PATH=4task-gl-framesamp`；
  第二次因工作区有未提交的 P5 留档（未跟踪文件）被脚本的 clean-tree 检查拒绝，先提交 `e48433e` 后第三次起跑成功。两次残档（`v1-store/dtype-unify/motion-a22-grad-grad`）已清理后重建。
- **records/**：`grad_summary.json`（逐叶 sha256、统计、batch 键摘要）、`a22_grad_report.json`、`wrapper.log`。fixture 与 checkpoint 临时目录按脚本约定不归档。
