# binfilldemo-motion50k-gl — 结果（512+motion，BinFill 补 demo 前缀，142 条）

**主对照表与全部分析在 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)**，
起跑口径见 [`../binfilldemo-nomotion50k-gl/launch.md`](../binfilldemo-nomotion50k-gl/launch.md)。本页只记本 run 独有的部分。

## 本 run 成绩

`MERGE_OK shards=10 episodes=142 successes=111 errors=0`，macro 0.7729 / micro **0.7817**。

| 难度 | 无 demo（同 142 条） | 补 demo | 变化 |
|---|---|---|---|
| easy | 13/49 = 26.5% | 44/49 = 89.8% | +63.3 pp |
| medium | 5/49 = 10.2% | 44/49 = 89.8% | +79.6 pp |
| hard | 0/44 = 0.0% | 23/44 = 52.3% | +52.3 pp |
| **合计** | **18/142 = 12.7%** | **111/142 = 78.2%** | **+65.5 pp（6.17x）** |

三个模型里回升最多的一条，与「记忆通道越宽、被 demo 缺失掏空越彻底」这条**写在跑之前**的预测一致。
hard 从一条都做不出来变成 52.3%。

**注意这不是同条件下的改进**——补 demo 同时改变了帧路与 motion 路的整个输入结构，
新旧两列是两种输入条件。详见主结果页「这些数字不能怎么用」。

## motion 窗与降级（本轮唯一涉及的 run）

十片全部 `MOTION_WINDOWS=PASS`。合并：`episodes=142 k_max=175 es_max=1107 downsampled_episodes=1`。

真实评测里降级只触发在 `BinFill/medium/195`（es=829、k_demo=51、跑满 2001 步 timeout），
客户端闭式与服务端实测在 k_max（175）、首次触发点（t_rel=1776）、降级次数（15）三项上**完全一致**，
`kept_demo=51` 全程一个没丢。

窗数最多的几条都够不到 160（`hard/179` k_max=140、`hard/177` 130、`hard/202` 128）——
成功的集都在 1200 步左右结束，根本活不到超预算。**补 demo 后 motion budget 160 在实际评测中够用，
降级是一道几乎用不到的保险。**

分层成功率（降级组 0/1 vs 未降级组 111/141）**只作描述、不作因果证据**：触发降级要求该集活到很晚，
早早成功的集天然进不了触发组，这是选择偏倚不是处理效应。配对实验记为待办。

## 端到端降级专项验证

`records/maxd-inject-hard202.json`——最大 D 的集（`BinFill/hard/202`，D=1107、k_demo=69）：
首批 1108 帧一次性编码 **179.9 秒、不 OOM**；人为喂满 95 批后降级 3 次，首次在 t_rel=1488，
与闭式「`69 + k_exec > 160 ⟹ k_exec ≥ 92 ⟹ t_rel ≥ 1488`」一步不差，三次都 `kept_demo=69` 全保。

该集在真实评测里只跑了 972 步（k_max=128），并未触发降级。

## 产物

`records/merged/{log,progress,motion_stats}.json`、`records/maxd-inject-hard202.json`。
