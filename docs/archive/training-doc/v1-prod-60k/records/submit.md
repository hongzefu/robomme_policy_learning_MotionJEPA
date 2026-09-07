# 提交实录（2026-08-30）

- job id：**59345249**；提交后约 12 秒即 R，节点 **gl1524**，TIME_LIMIT 4-00:00:00（96h ✓）。
- EXPECTED_GIT_HEAD=`d88185ca037b2e2dbf49373ee0c52aadd1cf2b42`（launch 提交后 HEAD，起跑闸实测通过）。
- --export 全串（key 略）：RUN_NAME=v1-prod-60k,STEPS=60000,SAVE_INTERVAL=10000,LOG_INTERVAL=100,BATCH=64,FSDP=4,WORKERS=8,SEED=42,EPOCH_STEPS=6176,EXPECTED_GIT_HEAD=…,WANDB_PROJECT=robomme-framesamp,WANDB_ENTITY=hongzefu-university-of-michigan,WANDB_API_KEY=…
- 提交经 ControlMaster 零认证；提交顺序按用户改令先于 v1-gl-entry-eq（59345261）。
- 起跑首信号：`ENV_RC=0` 已确认。

## 起跑首小时核验（三判据全过）

- wandb.init 成功：run https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/utktmnx4 在线上报，日志无 wandb ERROR。
- `ENV_RC=0`。
- 步时（metrics.jsonl 前 500 步逐 100 步区间）：4.860 / 4.873 / 4.869 / 4.867 / 4.860 s/步，
  均值 ≈ 4.87 s，与 v1-prod-trend-10h 底座 4.814 s 相差 +1.1%（节点方差内）；
  60k 外推 ≈ 81.1 h < 96 h walltime（1.18× 裕度）✓。
