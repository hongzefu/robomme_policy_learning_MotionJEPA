# v2-1600ep-m32x4x4-modul-b128-50k 训练结果（1127 计划 A 组）

**一句话**：32 帧 × 4×4（预算 512）modulation 帧记忆在 4task 1600ep 库上训满 50k 步，正常结束，10 份 ckpt（5000…50000）齐全；评估（ckpt 50000，seed 7/0/42）排在 B 组训完之后。

- **起跑版本**：`TRAIN_HEAD=6184ff031e4c5c26a28a2bc86fcb5cb659d5cd62`（commitV20.6Beta），主检出 detach 在该提交、工作区干净；由触发链 `scripts/training/chain_4task_fs4x4.sh`（tmux `fs4-chain`）调起 `run_4task_fs4x4_50k.sh <Beta> --budget 512`。
- **起止**：`START_UTC=2026-11-27T19:59:36Z`（美东 11-27 14:59）→ 链记 `FS4X4_STEP=train_1024 START 2026-11-28T09:27:49Z`（美东 11-28 04:27）；tqdm 总计 `elapsed:13:25:14`。
- **步速**：metrics.jsonl 记录的 step 0→50000 墙钟 48,195 s，均 0.964 s/步（含 XLA 编译与首 300 步计时）；与 1122 那条 32 帧×4×4 modulation 的 0.971 s/步相当。
- **GPU**：`v1-store/bench/v2-1600ep-m32x4x4-modul-b128-50k/gpu_util_15s_full.csv` 15 s 采样 25,847 条，8 卡 utilization 均值 98.1%（含启动段，未切稳态窗口）。
- **loss**（metrics.jsonl，每 100 步一条，共 501 条）：step 0 = 0.0894，100 = 0.0563，1000 = 0.0147，10000 = 0.00447，25000 = 0.00261，49900 = 0.00172，50000 = 0.00164；grad_norm 由 1.278（step 0）降到 0.0216（step 50000）。
- **判定行**：`FS4X4_TRAIN=PASS run=v2-1600ep-m32x4x4-modul-b128-50k steps=50001 ckpts=10`，`EXIT_CODE=0`，`FS4X4_STEP=train_512 PASS rc=0`。
- **ckpt**：`v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m32x4x4-modul-b128-50k/{5000,…,50000}`，共 111 G；本地全部保留。
- **上传**：tmux `fs4-incA` 逐份上传公开桶 `HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1` 并读回校验，5000…45000 均 `STEP_VERIFY=PASS`（每份约 11.88 GB）；50000 与末行 `HF_VERIFY` 结果见 `v1-store/logs/fs4x4-incA.log`。
- **记录**：`v1-store/bench/v2-1600ep-m32x4x4-modul-b128-50k/`（metrics.jsonl、step_timing.jsonl、gpu_util_15s_full.csv、run_meta.json、runtime.json）；训练日志 `v1-store/logs/v2-1600ep-m32x4x4-modul-b128-50k.log`。
- **计划外事件**：无。
