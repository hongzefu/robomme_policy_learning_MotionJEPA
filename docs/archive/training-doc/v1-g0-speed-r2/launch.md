# v1-g0-speed-r2（speed 链锚点重测，1000 步，P1b 后新版）

- **目的**：P1b 量具改动（runner 收敛 `uv run`、`SAVE_INTERVAL=0` 联动 `BATCH_DIGESTS=0`）使 speed 链量具口径改变，且用户裁定 speed 基线升级 1000 步——重测速度锚点，取代 300 步旧版 `v1-g0-speed`（1.117 s/step）。自本 run 起 speed 链一律 1000 步与本锚点对比。
- **起跑 commit**：`570287f`（V2.3.1，clean HEAD，与 G0b 两轮同源）；run_name 经用户确认。
- **口径（符号总表权威）**：`bench_train_steps.py` 入口、2×RTX 6000 Ada、b8、**1000 步**、seed 42、`num_workers=4`、生产 XLA 档（不注入 `XLA_FLAGS`、autotune 默认开）、`SAVE_INTERVAL=0`（联动禁 TrainState 摘要与输入摘要，P1b 联动实测生效）、EXP_NAME=`v1-g0-speed-r2` 独立（不与确定性档共享编译缓存，跑完即清）、`nvidia-smi -lms 500` + 15 s legacy 双通道采样。
- **命令**：`STEPS=1000 SAVE_INTERVAL=0 WORKERS=4 EXP_NAME=v1-g0-speed-r2 RUN_TAG=v1-g0-speed-r2 bash scripts/smoke-local/run_2gpu_epoch_bench.sh`（完整 argv 见 `records/env.json`）
- ⚠ 本机口径，按 AGENTS 13 不作最终吞吐结论；GL 验收另计。
