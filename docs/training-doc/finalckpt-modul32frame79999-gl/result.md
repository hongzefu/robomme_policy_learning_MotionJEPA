# finalckpt-modul32frame79999-gl — 结果（2048（32 帧）@79999，692 条）

**主对照表与全部分析在 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)**，
起跑口径见 [`../finalckpt-nomotion59999-gl/launch.md`](../finalckpt-nomotion59999-gl/launch.md)。本页只记本 run 独有的部分。

## 本 run 成绩

`MERGE_OK episodes=692 successes=316 errors=9`，**316/692 = 45.66%**（同模型 @50000 为 312/692 = 45.09%）。

逐组与逐难度数字见主结果页的「14 组逐格（最终 ckpt 档）」与「按难度」两节。

## 说明

本 run 不开 motion，十片 MOTION_WINDOWS=SKIP（EXPECT_MOTION_STATS=0），是预期行为。

BinFill 部分走补 demo 口径（142 条注入 demo 前缀），其余三任务 550 条不注入，
`DEMO_INJECT` 闸十片全 PASS、注入合计恰好 142。

## 产物

`records/merged/{log,progress}.json`。
