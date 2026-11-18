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

## RB-A3 核实：MotionMEM 80k 全局 grad_norm
- `v1-store/logs/v2-1600ep-m8x8-modul-motion-b128-80k.log`（1,233,164 B，2026-09-19）含 804 条 `grad_norm=`，每 100 步一条（如 `Step 0: grad_norm=1.1137, llm_grad_norm=1.1026, …`）。历史全局梯度范数轨迹可作参照；逐步裁剪系数仍只能由 e2e 自身逐步日志给出（log_interval 均值不等于逐步）。

## 实施合入记录
| 子任务 | TIP | 合并提交 | 合并前/后审查 |
|---|---|---|---|
| S5a 触发器 | b4895ab | 24bf94b | PASS/PASS findings=0 |
| S2 bf16 s16 表 | 5a62789 | 817551e | PASS/PASS findings=0 |
| S1 encoder 与对拍 | 436c28e | 22bc278 | PASS/PASS findings=0 |

## 主会话裁决（只记不报类，非训练语义）
- 标准化常量取 MotionJEPA ckpt buffer 的 fp32 位模式（与 VAE config 常数 max|Δmean|=4.17e-7、max|Δstd|=3.22e-5）；理由：depth=8 对拍逐位对齐，从头训练时差异可忽略。
- e2e-trig 触发器推迟到闸门表可填（S5 合入）后再起：开工时 A0 已满足（NativeMEM 已结束、8 卡 0 MiB），空表起触发器只会全部 not_configured。G1/G1b/G1c 由主会话直接在 1 卡串行跑（A0 现状已满足，见下）。
- 固定窗口清单 `scripts/training/g0/e2e_windows_v1.json` sha256 = 32e9aecc20d21bb68948a25f51657e1db2b3f50501dcbe2f0247ae28eae7b759（2048 窗，每任务 512）。
- 已知既有失败：`scripts/training/tests/test_s3_eval_local.py` c4/c5/c7 在不含 S4 的 v18 上同样失败（评估 slot 登记旧问题），非本轮引入。

## 闸门结果
- G2（tmux e2e-export，日志 v1-store/logs/e2e-export.log，HEAD 817551e，00:04–00:09 美东）：
  - `EXPORT_RAW16=PASS rows=71316 dtype=bf16 row_bytes=294912 demo=35913 exec=35403 budget_max=141 budget_limit=160 mismatch=0`，table_sha256=07f3766fdb28ab5f71bf30d4ff6eb5614747f350b978190cea658d5f8ccc5a2e
  - `INDEX_VS_MOTION768=PASS rows=71316 mismatches=0`
  - `S16_EVEN_VS_S32=PASS rows=36484 mismatches=0`
  - `K_DIST=43.906481,39,93,141 … over_budget=0 slot_util=27.44%`，`K_DIST_VS_AUDIT=PASS`（s16 index 文件 sha256=2b10c788…2151；审计的 d68696a0…7864 为 768 表 index，逐行相等已由 INDEX_VS_MOTION768 证明）
  - 表路径 `v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s16`
- G1/G1b/G1c（tmux e2e-g1，GPU0，日志 v1-store/logs/e2e-g1.log，HEAD 22bc278，真 raw_bf16）：
  - `E2E_PORT_PARITY=PASS windows=2048 … depth=1 init=seed params=20252928`
  - `E2E_BF16_FLOOR=RECORDED f=5.2745e-03`
  - `E2E_PORT_PARITY=PASS windows=2048 … depth=8 init=ckpt params=69808896`
  - `E2E_BF16_REF=PASS windows=2048 fwd_rel=5.1494e-03 fwd_limit=1.5420e-02 grad_rel_max=6.5487e-03 grad_limit_min=1.5180e-02 neg_x2=FAIL_AS_EXPECTED neg_stopgrad=FAIL_AS_EXPECTED`
