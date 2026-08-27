# v1-det-d2-r1（P2 确定性预备实验，档位 D2）

- **目的**：确定性 flags 档：D1 + deterministic_ops + autotune_level=0，两轮共用编译缓存。四档定义与判定见 `v1-gradient-baseline.md` P2 节；本轮为该档第 1 轮。
- **commit**：`d9e509e41a1665c15faff6ef62f2fef6ac813813`（V2.1，clean HEAD，env.json `git_dirty=false`）
- **入口**：`scripts/smoke-local/run_2gpu_epoch_bench.sh` → `bench_train_steps.py`（正确性族口径：TrainState 摘要 + 输入摘要开启）
- **命令**：`STEPS=100 SAVE_INTERVAL=50 WARMUP_STEPS=50 WORKERS=4 RUN_TAG=v1-det-d2-r1 EXP_NAME=v1-det-d2 KEEP_JAX_CACHE=1 XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" bash scripts/smoke-local/run_2gpu_epoch_bench.sh`（由 P2 驱动脚本顺序调用，全程 tmux + tee）
- **口径**：本机 2×RTX 6000 Ada、batch 8、seed 42、fsdp_devices 2、数据集 `v1-store/datasets/4task-gl`；完整 argv 与环境指纹见 `records/env.json`
- ⚠ 正确性族 run：util/步时受摘要停顿与确定性档污染，仅留档参考，禁作性能结论（基线计划红线 B7）
