# v1-grad-baseline-g0（G0 黄金基线，PG0，commit V2.3）

> **已被取代（2026-08-27）**：本基线由 1000 步新版 [`v1-grad-baseline-g0b`](../v1-grad-baseline-g0b/launch.md) 取代（P1b 量具补遗后重跑；新 r1 前 300 步与本基线 raw 口径逐位前缀对拍 PASS 后，records 按用户裁定删除，本文件与 result.md 留作历史存证）。登记簿以 `v1-gradient-baseline.md` T8 为准。

- **目的**：三计划（dtype / IO 重构 / roadmap）均未实施的**原始训练语义黄金基线**，受控确定性档（P2 定档 D2）下两轮 300 步；round1 为正本、round2 为自证。产物固化进 git，之后所有改动离线对拍 G0，不再重跑对照侧。定义、判据、失效条件的权威载体：`v1-gradient-baseline.md`。
- **双重身份**：兼任 dtype 计划 P6 的 A 侧（修复前），该计划只跑 B 侧 `v1-dtype-ab-post`。
- **G0_SCOPE 断言（起跑前实测，PASS）**：锚点 `55e6e5bf8ef38b780902d0e63257ea859a432a2c` → `<G0-HEAD>=624d4177ab870d534756f7e90b767e0141c9763a`，`git diff --name-only` 共 76 个文件全部落在白名单（`docs/`、`scripts/smoke-local/`、`scripts/data-preprocess-GL/paths.sh`、根目录 `*.md`）——其中 `scripts/smoke-local/**` 的改动即 P1 仪器改造（V2.1），逐 hunk 均为观测/驱动层、训练语义零改动（见该 commit body）；`git status --porcelain` 空；submodule `third_party/robomme_benchmark` 指针 `856bc3a` 与锚点一致；两轮 env.json `git_dirty=false`。
- **口径**：`bench_train_steps.py` 入口、2×RTX 6000 Ada、b8、300 步、seed 42、`fsdp_devices=2`、`num_workers=4`、SAVE_INTERVAL=100（TrainState 摘要 @ 0/100/200/299，共 4 次）、输入摘要 @ 0/1/2/100/200/299、`XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`（D2 档）、EXP_NAME=`v1-grad-baseline-g0` 两轮共用（round2 编译缓存全命中）、`nvidia-smi -lms 500` + 15 s legacy 双通道采样。
- **命令**：`STEPS=300 SAVE_INTERVAL=100 WORKERS=4 RUN_TAG=v1-grad-baseline-g0-round{1,2} EXP_NAME=v1-grad-baseline-g0 KEEP_JAX_CACHE=1 XLA_FLAGS="…" bash scripts/smoke-local/run_2gpu_epoch_bench.sh`（完整 argv 见各轮 `records/round{1,2}/env.json`）
- ⚠ 红线 B7：本 run 带完整 TrainState 摘要 + 输入摘要 + 确定性档，util/步时**仅留档参考、禁作任何性能结论**；性能锚点见 `docs/training-doc/v1-g0-speed/`。
