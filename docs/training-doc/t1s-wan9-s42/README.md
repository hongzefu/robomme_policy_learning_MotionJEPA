# t1s-wan9-s42 评估档案（Table I 三 seed / 三 ckpt 补评）

## ①一句话结论与指标速览
（评估完成后补写）

## ②版本与代码状态
- 计划：策略仓 v18 分支 `1120-table1-3seed-aws-local-eval-plan.html`（开工版 1431a26，开工令见 docs/handoff/current.json 任务 t1s-3seed-1120）。
- 评估栈：v18 主检出 local_692_pool.sh（EXPECT_MOTION_STRIDE=32 BUDGET=80）。v18 主检出含 E-A1（374d72f）、E-A2（9297dca）；v42 分支 t1s-1120-evalstack-v42@c3876b9（d6ece8b + E-B）；mm-ctx 用 8181918 detached 检出。

## ③启动与配置还原
- Table I 行：Wan-9 512+720 modulation；行键 `wan9`；seed 42；ckpt step 50000。
- ckpt：`v1-store/models/t1s-1120/v2-1600ep-m8x8-modul-wanraw9s32-b128-50k/50000`（check_checkpoint 全部 PASS，2026-11-20 美东）。
- 链脚本：`v1-store/jobs/t1s-1120/t1s-lib.sh` 的 `run`/`pair`/`pairck`（同行两次同卡静态配对，各 16 片、每卡 2 片，EVAL_GPU_CAP=4，第二次 EVAL_SHARD_BASE=16；mm-ctx、native 单跑）。
- 日志：`v1-store/logs/t1s-wan9-s42.log`；结果：`v1-store/evaluation/t1s-wan9-s42/`（native 在 NativeMEM/runs/eval/t1s-wan9-s42/）。
- 前缀库：本机 `v1-store/demo-prefix/binfill-3b4de03a0b46`（与 seed 7 的 GL 原库有 40 局 BinFill 前缀不同，用户 11-20 确认按 K2 不处理）。

## ④–⑫
（评估完成后补写：耗时、EVAL692 判定行、14 组计数、计划外事件、归档 records/）
