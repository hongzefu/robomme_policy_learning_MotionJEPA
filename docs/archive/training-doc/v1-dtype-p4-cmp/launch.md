# v1-dtype-p4-cmp（dtype 修复第一块 B 侧取证 + 三层对拍判定）

- **目的**：在**修复已落地**的 clean HEAD 上取第二次证，与 P3 的 A 侧产物离线对拍，
  给出第一块（非训练轻量化测试，AGENTS 18）的判定。
- **起跑 commit**：`a0f76f8`（commitV2.4b，三行 dtype 修复；`git diff` 三个 hunk、
  各 +2/−1，唯一文件 `src/mme_vla_suite/shared/data_utils.py`）。起跑前
  `git status --porcelain` 为空。
- **A 侧来源**：`v1-store/dtype-unify/v1-dtype-p3-dump-pre/`（P3 于 `f2e7348` 产出）。
  两侧 `fixture_plan.json` 的 seed / limit / groups / batches / manifest_sha256 必须
  逐项相同，对拍开头即断言——不同则定点集不可比，当场停下。
- **口径**：与 P3 逐字相同（驱动脚本把路径与参数固化，两侧只剩 `RUN_TAG` 一个变量）。
  `JAX_PLATFORMS=cpu`、b8、`--dataset-path v1-store/datasets/4task-gl`、
  `perceptual-framesamp-context.yaml`；2,600 定点样本 + 200 定点 batch。
- **命令**：
  ```
  RUN_TAG=v1-dtype-p4-cmp bash scripts/dtype-unify/run_dtype_dump.sh
  uv run scripts/dtype-unify/compare_dtype_fix.py \
    v1-store/dtype-unify/v1-dtype-p3-dump-pre \
    v1-store/dtype-unify/v1-dtype-p4-cmp \
    --report v1-store/dtype-unify/p4-compare-report.json
  ```
  （tmux detached + `tee` + `EXIT_CODE=`，AGENTS 7；日志 `v1-store/logs/v1-dtype-p4.log`）
- **判据四条**（全部零容差）：① 全键 shape 相同；② 数值 canonical 一致（等价于
  `astype(f32)` 后 `view(uint32)` 逐位相同）；③ dtype 变化逐键清单与预期完全一致，
  且 memory 之外全部键的 dtype 与 raw 摘要都必须完全相同；④ batch 级 memory 键 dtype
  恒定，不随 batch 组成摆动。外加归一化前纯函数位型测试，判定并入同一行。
