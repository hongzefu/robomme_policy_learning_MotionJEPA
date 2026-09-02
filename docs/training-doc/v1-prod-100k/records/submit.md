# 提交实录

## 第一次提交（2026-09-01 16:30，job 59473806，被起跑闸拦下）

- job id：59473806；提交时刻 2026-09-01 16:30:02 EDT；`--time=168:00:00`；入队即 `PENDING (Priority)`。
- EXPECTED_GIT_HEAD=`05c1faabcc90dee1394671e783c3834890ace5cd`（launch 档提交后 HEAD）。
- --export 全串（key 略）：RUN_NAME=v1-prod-100k,STEPS=100000,SAVE_INTERVAL=10000,LOG_INTERVAL=100,BATCH=64,FSDP=4,WORKERS=8,SEED=42,EPOCH_STEPS=6176,EXPECTED_GIT_HEAD=…,WANDB_PROJECT=robomme-framesamp,WANDB_ENTITY=hongzefu-university-of-michigan,WANDB_API_KEY=…
- 提交经 ControlMaster 零认证；preflight 五项（master 存活、工作区 clean、HEAD=H1、无同名目录、队列无同名）全过后才提交。
- 提交时 `v1-prod-60k`（59345249）RUNNING @ gl1524 1-17:56，两 job 并行。
- **结果：排队约 5 小时后起跑，起跑闸拦下**——日志
  `错误: HEAD 漂移（期望 05c1faab…，实际 a9d13cab…），拒绝起跑`，`EXIT_CODE=1`。
  漂移来源：排队期间另一会话对 `v6-motion-memory-plan.md` 的四次 docs 提交
  （7867b62 → f96b390 → 08e272e → a9d13ca，最后一次 21:38 EDT）。起跑闸在 `mkdir` 之前，
  `train-records/` 与 `train-runs/` 均未创建，无残档。
- 顺带发现：起跑闸还断言 `git status --porcelain` 为空，**未跟踪的 `??` 文件同样会拦**——本文件
  若在排队期间以未跟踪状态留在工作区，即使 HEAD 未漂也会被判「工作区不 clean」。重提前必须先
  把本文件提交掉，排队期间不得再向仓库写任何未提交文件。

## 第二次提交（2026-09-02，不传 EXPECTED_GIT_HEAD）

- 用户拍板「去掉闸重提」（见 launch.md 口径变更一节）；--export 串去掉 `EXPECTED_GIT_HEAD=…`，其余逐字同第一次。
- 提交前先把本文件与 launch.md 更新提交入库（`docs:` commit），工作区 clean 后再提交 job。
- （起跑后补：jobid、提交时刻、排队时长、节点、`ENV_RC`、env.json 实录 git_head、wandb run URL、首 500 步步时与外推。）
