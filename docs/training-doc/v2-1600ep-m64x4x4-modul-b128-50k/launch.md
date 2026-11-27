# v2-1600ep-m64x4x4-modul-b128-50k 起跑记录（1127 计划 B 组）

- 计划：`1127-4task-modul-fs4x4-512-1024-60k-plan.md`（v20 分支根目录）；用户开工令「同意开工」（2026-11-27 美东）。
- 配方：与 A 组只差帧记忆预算——64 帧 × 4×4（`token_per_image` 16，`budget` 1024），motion 关闭；`mme_vla_suite_b128_80k`（8 卡 FSDP8，batch 128，预热 5k，lr 5e-5 恒定，EMA 0.999，seed 42），`--num_train_steps=50001`，ckpt 5000…50000 共 10 份，评估只用 `50000`（与 Table I 同位）。
- history yaml：`perceptual-framesamp-modul-64frame-4x4.yaml`，sha256 `a08f40d3b38bbe6fc94186375b9f00ac6ea130c7fc7fb1386ff362e309c0f520`。
- 数据集白名单：`framesamp_dataset.py:96` 已放行 (1024,16,1)（S1，合入 df5d1656）。
- 帧表与 G1/G2：同 A 组。
- G3：`LAUNCH_CONTRACT=PASS budget=1024 frames=64`（主检出 7f62f1aa，CPU 实例化 `DATASET_SAMPLE=PASS frames=64 tokens_per_frame=16`）。
- 入口：`scripts/training/run_4task_fs4x4_50k.sh <Beta> --budget 1024`，由触发链在 A 组训练完成后调起（tmux `fs4-chain`）。
- 日志：`v1-store/logs/v2-1600ep-m64x4x4-modul-b128-50k.log`；ckpt：`v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m64x4x4-modul-b128-50k/`。
- 上传：tmux `fs4-incB`，`scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh v2-1600ep-m64x4x4-modul-b128-50k` → 公开桶 `HongzeFu/robomme-vla-4task-modul-m64x4x4-50k-v1`。
- 起跑版本、实际起止时间与步速在训练结束后写入同目录 `result.md`。
