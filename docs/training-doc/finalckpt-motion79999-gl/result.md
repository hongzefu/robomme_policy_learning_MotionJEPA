# finalckpt-motion79999-gl — 结果（512+motion @79999，692 条）

**主对照表与全部分析在 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)**，
起跑口径见 [`../finalckpt-nomotion59999-gl/launch.md`](../finalckpt-nomotion59999-gl/launch.md)。本页只记本 run 独有的部分。

## 本 run 成绩

`MERGE_OK episodes=692 successes=390 errors=9`，**390/692 = 56.36%**（同模型 @50000 为 378/692 = 54.62%）。

逐组与逐难度数字见主结果页的「14 组逐格（最终 ckpt 档）」与「按难度」两节。

## 说明

本 run 是唯一涉及 motion 窗与降级的：十片 MOTION_WINDOWS 全 PASS，k_max 98–175，其中一片触发降级 15 次（BinFill/medium/195，闭式与实测精确相等）。详见主结果页。

BinFill 部分走补 demo 口径（142 条注入 demo 前缀），其余三任务 550 条不注入，
`DEMO_INJECT` 闸十片全 PASS、注入合计恰好 142。

## 产物

`records/merged/{log,progress}.json`、`records/merged/motion_stats.json`。
