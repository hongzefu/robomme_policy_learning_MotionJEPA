# binfilldemo-modul32frame-50k-gl — 结果（32 帧 / 感知预算 2048，BinFill 补 demo 前缀，142 条）

**主对照表与全部分析在 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)**，
起跑口径见 [`../binfilldemo-nomotion50k-gl/launch.md`](../binfilldemo-nomotion50k-gl/launch.md)。本页只记本 run 独有的部分。

## 本 run 成绩

`MERGE_OK shards=10 episodes=142 successes=84 errors=0`，macro 0.5792 / micro **0.5915**。

| 难度 | 无 demo（同 142 条） | 补 demo | 变化 |
|---|---|---|---|
| easy | 10/49 = 20.4% | 40/49 = 81.6% | +61.2 pp |
| medium | 6/49 = 12.2% | 34/49 = 69.4% | +57.1 pp |
| hard | 0/44 = 0.0% | 10/44 = 22.7% | +22.7 pp |
| **合计** | **16/142 = 11.3%** | **84/142 = 59.2%** | **+47.9 pp（5.25x）** |

回升倍数（5.25x）介于 8 帧基线的 1.70x 与 512+motion 的 6.17x 之间，与三者的记忆通道宽度单调对应。

本 run 的帧路变化最大：32 帧档此前 32 帧全部落在 exec 段，补 demo 后有 16–28 帧落进 demo 段。
**这不是同条件下的改进**，详见主结果页「这些数字不能怎么用」。

## motion

本 run 不开 motion，十片 `MOTION_WINDOWS=SKIP reason=no-motion-stats`（`EXPECT_MOTION_STATS=0`），
这是预期行为——闸只对 motion run 生效。

## 产物

`records/merged/{log,progress}.json`。
