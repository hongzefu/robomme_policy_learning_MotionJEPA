# launch.md — v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k（E2E-Enc，任务 id e2eenc-s16-50k）

计划：`1117-e2e-enc-train-plan.html`，用户批准版本 = `c618671`（v15 分支 commitV15.21）。

## 起点（RB-A1）
- 开工令：2026-11-17 23:4x 美东，用户原话「开工」（此前定稿审阅约定：「先定稿，定稿完了之后，不要立刻开工，让我再看一眼通过」；停止规则：「我同意开工之后就尽可能一路跑到底，除非有训练语义上的不同……」）。
- 实施分支 `v18-4task-modul-e2eenc-s16-50k` 自 `c618671` 创建（不从 c0826be 直接起）；代码基线 = `c0826be`（c618671 的祖先）。
- 执行位置：主检出 `/scratch/hongze/robomme_policy_learning_MotionJEPA`（runner 写死 `MAIN=` 该路径），由 v17 分支切到 v18；v17 工作区干净、已推送。
- 规则子模块随 repo_guard 更新到 b91c20d（`6d9e36a`）。

## 资源与 A0 现状（RB-A0/RB-A2）
- 2026-11-17 23:41 美东：8 卡 memory.used 全 0 MiB；NativeMEM 23:09:43 美东结束（NM_PIPELINE=PASS、EVAL692=PASS 290/692、EXIT_CODE=0）。
- `df -B1 /scratch`：可用 566,394,560,512 B（93% 已用）。

## 偏离/调整记录（R8 第 (i) 档等只记不报的项写这里）
（暂无）
