# 提交实录（2026-08-30）

- job id：**59345249**；提交后约 12 秒即 R，节点 **gl1524**，TIME_LIMIT 4-00:00:00（96h ✓）。
- EXPECTED_GIT_HEAD=`d88185ca037b2e2dbf49373ee0c52aadd1cf2b42`（launch 提交后 HEAD，起跑闸实测通过）。
- --export 全串（key 略）：RUN_NAME=v1-prod-60k,STEPS=60000,SAVE_INTERVAL=10000,LOG_INTERVAL=100,BATCH=64,FSDP=4,WORKERS=8,SEED=42,EPOCH_STEPS=6176,EXPECTED_GIT_HEAD=…,WANDB_PROJECT=robomme-framesamp,WANDB_ENTITY=hongzefu-university-of-michigan,WANDB_API_KEY=…
- 提交经 ControlMaster 零认证；提交顺序按用户改令先于 v1-gl-entry-eq（59345261）。
- 起跑首信号：`ENV_RC=0` 已确认。
