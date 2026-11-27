# v2-1600ep-m32x4x4-modul-b128-50k 起跑记录（1127 计划 A 组）

- 计划：`1127-4task-modul-fs4x4-512-1024-60k-plan.md`（v20 分支根目录）；用户开工令「同意开工」（2026-11-27 美东）。
- 配方：4task 1600ep 库，modulation，帧记忆 32 帧 × 4×4（`token_per_image` 16，`budget` 512），motion 关闭；`mme_vla_suite_b128_80k`（8 卡 FSDP8，batch 128，预热 5k，lr 5e-5 恒定，EMA 0.999，seed 42），`--num_train_steps=50001`，ckpt 5000…50000 共 10 份，评估只用 `50000`（与 Table I 同位）。
- history yaml：`perceptual-framesamp-modul-32frame-4x4.yaml`，sha256 `14b54bae33d1a5ecb8240dff2e9f69727de6e43c6e5ad730b87c1e57b0e44acb`。
- 帧表：`v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-4x4`（数据集桶原生表拉回）；G1/G2 `FS4X4_CHECK=PASS failed=none`（`v1-store/logs/fs4x4-check-7ad7d742.log`：37 文件 sha 全对、ROWDIGEST 1192918 行 0 不符、XGRID min_cos 0.9999908、PROBE 8/8、META PASS）。
- G3：`LAUNCH_CONTRACT=PASS budget=512 frames=32`（主检出 7f62f1aa，CPU 实例化 `DATASET_SAMPLE=PASS frames=32 tokens_per_frame=16`）。
- 入口：`scripts/training/run_4task_fs4x4_50k.sh <Beta> --budget 512`，由触发链 `scripts/training/chain_4task_fs4x4.sh <Beta> /scratch/hongze/worktrees/fs4x4-ev` 调起（tmux `fs4-chain`）。
- 日志：`v1-store/logs/v2-1600ep-m32x4x4-modul-b128-50k.log`；ckpt：`v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m32x4x4-modul-b128-50k/`。
- 上传：tmux `fs4-incA`，`scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh v2-1600ep-m32x4x4-modul-b128-50k` → 公开桶 `HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1`。
- 起跑版本、实际起止时间与步速在训练结束后写入同目录 `result.md`。
