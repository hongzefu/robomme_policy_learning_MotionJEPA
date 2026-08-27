# v1-g1-speed（G1-speed：dtype 修复后的性能 run，对比 speed 锚点）

- **目的**：产出 dtype 修复的**单独性能对比**，作为「是否实施 IO 重构计划（v4）」的
  两份用户决策输入之一（另一份是 G1 vs G0b 的正确性验收结论）。
- **起跑 commit**：`d227931`（P6 留档提交后的 clean HEAD；训练语义即 V2.4b `a0f76f8`
  的三行 dtype 修复）。`env.json` 记 `git_dirty=false`。
- **口径**（speed run 统一口径，`v1-gradient-baseline.md` 符号总表；2026-08-27 起步数
  统一为 1000）：`bench_train_steps.py` 入口、2×RTX 6000 Ada、b8、**1000 步**、seed 42、
  `num_workers=4`、`fsdp_devices=2`；**生产 XLA 档——不注入 `XLA_FLAGS`、autotune 默认
  开**；`SAVE_INTERVAL=0`（联动禁 TrainState 摘要与输入摘要，P1b 的联动实测生效：
  `save_interval_requested=0 / effective=1000000`、`batch_digests=0`）；EXP_NAME 独立
  （跑完即清缓存）；`nvidia-smi -lms 500` + 15 s legacy 双通道采样。
- **对比对象**：`v1-g0-speed-r2`（现行 speed 链锚点，1000 步口径）。两侧 `env.json`
  逐项核对一致：`XLA_FLAGS=''`、`XLA_PYTHON_CLIENT_MEM_FRACTION='0.95'`、steps=1000、
  workers=4、seed=42、fsdp=2。
- **命令**：
  ```
  STEPS=1000 SAVE_INTERVAL=0 WORKERS=4 EXP_NAME=v1-g1-speed RUN_TAG=v1-g1-speed \
  bash scripts/smoke-local/run_2gpu_epoch_bench.sh
  ```
  util 采样与统计：
  ```
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv,noheader -lms 500 > util-lms500.csv    # 另起 -l 15 的 legacy 通道
  uv run scripts/dtype-unify/analyze_util.py <records> --steps 1000 --warmup 50
  ```
  （tmux detached + `tee` + `EXIT_CODE=`，AGENTS 7）
- **统计口径**：`scripts/dtype-unify/analyze_util.py`，稳态窗步 50–999，慢步阈值
  1.5× 中位。**两侧同法重算**，不直接引用对方留档数字。该脚本对 `v1-g0-speed-r2`
  的复算结果与其 result.md 逐项吻合（中位 1.152 / 均值 1.1858 / p10 1.0967 /
  p90 1.2753 / 慢步 3 / util 均值 86.46% / 0% 占比 4.89%）。
- **⚠ 留档缺陷（如实记录）**：本轮 wrapper 的日志文件名与 `run_2gpu_epoch_bench.sh`
  内部的 `${LOGS_DIR}/${RUN_TAG}.log` 撞名，两个 `tee` 同时写同一文件，导致 wrapper
  在 bench 启动前打印的阶段行（`P7_HEAD=` 等）被覆盖。判定与口径不受影响——起跑
  commit、`git_dirty`、真实 argv、XLA 配置均以 `records/env.json` 与 `run_meta.json`
  为准（已逐项核对）。P6 同样撞名，同样以 env.json 为准。
