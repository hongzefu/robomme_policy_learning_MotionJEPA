# finalckpt-nomotion59999-gl — 结果（512 / 8 帧 @59999，692 条）

**主对照表与全部分析在 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)**，
起跑口径见 [`launch.md`](launch.md)。本页只记本 run 独有的部分。

## 本 run 成绩

`MERGE_OK episodes=692 successes=228 errors=9`，**228/692 = 32.95%**（同模型 @50000 为 211/692 = 30.49%）。

**这是 60k run 的最终 ckpt——它训到 60000 步为止，没有 80k**。所以本 run 的 59999 与另两条的 79999
不是等步数，两档表格里的「最终 ckpt」列不能横向当作等步数比较。

## 说明

本 run 不开 motion，十片 `MOTION_WINDOWS=SKIP`（`EXPECT_MOTION_STATS=0`），是预期行为。
BinFill 部分走补 demo 口径，`DEMO_INJECT` 闸十片全 PASS、注入合计恰好 142。

## 产物

`records/merged/{log,progress}.json`；本目录另存三个 run 共用的 `manifest.tsv`、
`preseeded-errors.json`、`shard-checks.json`。
