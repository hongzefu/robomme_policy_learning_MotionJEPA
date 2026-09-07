# 提交实录（2026-08-30）

## 首次提交 59345261：被起跑闸拦下（闸按设计工作）

- job id：**59345261**；提交后 PD（Priority 排队），TIME_LIMIT 5:00:00；
  EXPECTED_GIT_HEAD=`d88185ca037b2e2dbf49373ee0c52aadd1cf2b42`。
- 开跑即拦：`错误: 仓库 HEAD 漂移（期望 d88185c…，实际 221314b…）`，EXIT_CODE=1。
  根因：该 job 排队期间提交了两份 submit.md 实录（`221314b`），HEAD 相对提交时锚
  漂移——**闸的行为正确**，属提交流程失误（排队窗口内不应再动 HEAD）。
- v1-prod-60k（59345249）不受影响：它在 `221314b` 之前已 R，起跑闸当时通过。

## 重提（锚定本 commit 后 HEAD）

- 教训与规约：重提后至该 job 转 R 之前**冻结一切 commit**；job id 与实际起跑记录
  见 result.md（不再于排队窗口内回写本文件）。
- --export 全串同首次，仅 EXPECTED_GIT_HEAD 换为本 commit 后 HEAD。

## 重提实录

- 重提 job id：**59349729**，EXPECTED_GIT_HEAD=`5ddb19ca703ba5f7d13e6e15f7d4572b23617475`；
  已转 R 并过闸（`ENV_RC=0`），A 段（上游 1000 步）进行中。

## 第二次提交 59349729：双侧 OOM（v5.1 首个实质发现）

- 过闸后 A 段（legacy 库）跑至第 100 步、首次 save_state 摘要点被 SIGKILL(137)：步 100
  标量行已打出、摘要器「已记」行未出现。B 段（packed 库）**同构复现**——同在第 100 步
  摘要点、同 137。判定：摘要器（逐叶 device_get + tobytes 双拷贝）瞬时匿名尖峰叠加
  NFS 页缓存计入 cgroup（greatlakes.md 页缓存铁证），128G 包络被顶爆，**与数据库无关**。
- Slurm 铁证：`Detected 2 oom_kill events in StepId=59349729.batch`；State=OUT_OF_MEMORY；
  batch 步 MaxRSS=134217512K ≈ 128.0 GB 精确贴死申请上限。
- 额外收获：`WANDB_PROBE=PASS`（计算节点出网判定项已提前验证）；A 段实测 12-13 s/步
  （确定性 XLA 档），1000 步单侧 ≈ 3.5h → 原 5h walltime 亦不可行。
- 残档挪至 `v1-store/entryeq/records-oom-59349729/`（gitignored），ckpt-a/b 已删。
- 用户拍板（AskUserQuestion 两轮）：跑到 B 摘要点拿诊断数据再终止（实际 job 自然退出，
  未 scancel）；重提 `--mem=240G --time=12:00:00`（CLI 覆盖）；**摘要器不改代码**。

## 第三次提交前插曲 59357885：残档防覆盖闸拦下（12 秒退出）

- 重提秒排秒跑，撞上 59349729 残留 records/{a,b}，harness 防覆盖 fail-loud：
  `FileExistsError: 记录已存在，拒绝覆盖`，A_RC=1/B_RC=1，EXIT_CODE=1（清场晚了一步；
  该 job 探针亦 PASS）。教训：**重提前必须先清 `v1-store/entryeq/records/` 与 ckpt-a/b**。

## 第三次提交 59357942（当前有效）：240G/12h

- job id：**59357942**，EXPECTED_GIT_HEAD=`43680ee4180d8effaec178814b10e0e8859c2d12`
  （代码零改动，资源走 sbatch CLI 覆盖）；干净目录起跑，已过闸（`ENV_RC=0`）。
- 240G 对症验证点：A 段第 100 步摘要点（开跑后约 25 分钟）。
