# v1-det-d2cold-r2（P2 确定性预备实验，档位 D2COLD）

- **目的**：独立重编译档：同 D2 flags，两轮各用全新空缓存目录（EXP_NAME 各异）强制独立编译两次——G0 跨期充当 bitwise 判据一侧的唯一授权闸。四档定义与判定见 `v1-gradient-baseline.md` P2 节；本轮为该档第 2 轮。
- **commit**：`9c49cf6`（clean HEAD，env.json `git_dirty=false`）。⚠ 本轮与 r1（`d9e509e`）之间插入了一个外部纯文档 commit（仅改 `v1-dtype-unify-plan.md`，G0_SCOPE 白名单内、训练语义零影响）；两轮对拍仍 bitwise PASS，附带证明白名单内文档 commit 不改变训练行为
- **入口**：`scripts/smoke-local/run_2gpu_epoch_bench.sh` → `bench_train_steps.py`（正确性族口径：TrainState 摘要 + 输入摘要开启）
- **命令**：`STEPS=100 SAVE_INTERVAL=50 WARMUP_STEPS=50 WORKERS=4 RUN_TAG=v1-det-d2cold-r2 EXP_NAME=v1-det-d2cold-b KEEP_JAX_CACHE=1 XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" bash scripts/smoke-local/run_2gpu_epoch_bench.sh`（由 P2 驱动脚本顺序调用，全程 tmux + tee）
- **口径**：本机 2×RTX 6000 Ada、batch 8、seed 42、fsdp_devices 2、数据集 `v1-store/datasets/4task-gl`；完整 argv 与环境指纹见 `records/env.json`
- ⚠ 正确性族 run：util/步时受摘要停顿与确定性档污染，仅留档参考，禁作性能结论（基线计划红线 B7）
