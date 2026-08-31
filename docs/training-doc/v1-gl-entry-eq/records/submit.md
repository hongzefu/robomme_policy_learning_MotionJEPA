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
