# v1-dtype-ab-post-r1（G1：dtype 修复后千步正确性 run，对拍 G0b）

- **目的**：第二块（本机真实训练、梯度一致）的收尾检验——在修复后的 clean HEAD 上跑
  1000 步真实训练，与黄金基线 G0b r1 的固化产物逐步对拍。
- **起跑 commit**：`a0f76f8`（commitV2.4b，三行 dtype 修复）。起跑前工作区 clean。
- **A 侧**：`docs/training-doc/v1-grad-baseline-g0b/records/r1`（不重跑）。
- **preflight（硬前置）**：在注入同口径 `XLA_FLAGS` / `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
  / `CUDA_VISIBLE_DEVICES=0,1` 的 shell 里跑（`collect_fingerprint` 直读 `os.environ`，
  不注入会因 `XLA_FLAGS` 空串误判 FAIL）：

  ```
  uv run scripts/smoke-local/check_baseline_env.py check \
    --baseline docs/training-doc/v1-grad-baseline-g0b/records/r1 --steps 1000 --batch-size 8
  ```
  实测 **`BASELINE_ENV=PASS`**（环境指纹逐项一致 + G0b 的 10 条产物 sha256 复验通过）。

- **口径**（逐项照抄 G0b launch.md，只换 EXP_NAME / RUN_TAG，且不落 TrainState 数组）：
  `bench_train_steps.py` 入口、2×RTX 6000 Ada、b8、**1000 步**、seed 42、
  `fsdp_devices=2`、`num_workers=4`、`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`
  （TrainState 摘要 12 次 @ 0/100/…/900/**299**/999，与 G0b 完全对齐）、输入摘要 14 次
  （raw + canonical 双口径 + 逐步样本 index）、index 全序列、
  `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`（D2 档）、
  `nvidia-smi -lms 500` + 15 s legacy 双通道采样。单 epoch 约束满足（1000×8 = 8,000 <
  395,289）。

- **EXP_NAME 独立**（`v1-dtype-ab-post-r1`，2026-08-27 裁定）：原「与 G0b 共用 EXP_NAME
  以共享 per-fusion autotune 缓存」的前提已不成立——G0b 的编译缓存已随 `c6830e0`
  清理（「G0b 缓存原为 G1 共享而保留，改为即时清理留证」），共用名只会拿到空目录 +
  冷编译；且确定性档 `--xla_gpu_autotune_level=0` 本就关闭 autotune。

- **命令**：
  ```
  STEPS=1000 SAVE_INTERVAL=100 WORKERS=4 EXTRA_DIGEST_STEPS=299 \
  RUN_TAG=v1-dtype-ab-post-r1 EXP_NAME=v1-dtype-ab-post-r1 KEEP_JAX_CACHE=1 \
  XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" \
  bash scripts/smoke-local/run_2gpu_epoch_bench.sh

  uv run scripts/smoke-local/compare_baseline.py \
    docs/training-doc/v1-grad-baseline-g0b/records/r1 \
    v1-store/bench/2gpu-epoch-bench/v1-dtype-ab-post-r1 --tier g1-vs-g0b
  ```
  （tmux detached + `tee` + `EXIT_CODE=`，AGENTS 7；日志
  `v1-store/logs/v1-dtype-ab-post-r1.log`）

- **判定口径（重要）**：输入侧**禁用 raw `batch_digests`**——dtype 变更使其必然失配、
  且无鉴别力（分不清「只是类型变了」与「数值真变了」），改用 canonical 数值口径
  + 全步 index 序列。该限定见 `v1-gradient-baseline.md` 三节口径限定与五节对拍矩阵
  G1 行。`compare_baseline.py` 的总判定行 `DET_CHECK=` 把 raw 也计入，故其 FAIL
  须按本条口径复核成因，不能直接当作判据不过。

- **轮数**：先跑 r1 一轮；仅当 bitwise 失败、需要启用量化兜底时才补 r2 自证
  （2026-08-27 用户裁定）。本轮 bitwise 全过，未补 r2。
