# v1-g0-speed（speed 链锚点，PG0-speed）

- **目的**：性能族 speed 链的锚点 run（`v1-gradient-baseline.md` 符号总表）；G0 两轮之后同场次预跑。v4（IO 重构）与 roadmap 各项收官的 speed run 均与本 run 对比。
- **口径（生产 XLA 档）**：`bench_train_steps.py` 入口、2×RTX 6000 Ada、b8、300 步、seed 42、`num_workers=4`；**不注入 XLA_FLAGS**（autotune 默认开）；`SAVE_INTERVAL=0`（无 TrainState 摘要）、`BATCH_DIGESTS=0`（无输入摘要）；EXP_NAME=`v1-g0-speed` 独立、不与确定性档共享缓存（本轮冷编译）；`nvidia-smi -lms 500` + 15 s legacy 双通道采样。
- **commit**：`624d4177ab870d534756f7e90b767e0141c9763a`（与 G0 同 HEAD，clean）。
- **命令**：`STEPS=300 SAVE_INTERVAL=0 BATCH_DIGESTS=0 WORKERS=4 RUN_TAG=v1-g0-speed EXP_NAME=v1-g0-speed KEEP_JAX_CACHE=0 XLA_FLAGS="" bash scripts/smoke-local/run_2gpu_epoch_bench.sh`（完整 argv 见 `records/env.json`）
- ⚠ AGENTS 13：本机口径，不作最终吞吐结论（GL 侧验收另行）；AGENTS 16：判读以稳态均值 / 0% 采样占比 / 慢步分层为准，禁以中位数作标题结论。
