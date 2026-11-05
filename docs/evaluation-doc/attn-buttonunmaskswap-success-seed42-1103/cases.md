# 逐局结果与图索引

全部22条模型与回合组合均保留；以下结局来自独立终局字段，不把正常失败改成成功，也不重试。每局三图均使用原始全维记录计算，原始逐次JSON与NPZ留本机数据根。

头图斜线只表示该流质量低于1%，不解读其富集；没有据此少存任何头或token。单局图色条独立，跨局比较看汇总数值。阶段证据图为候选定位拼图，主会话已经结合闭合指令与CPU关节状态核实，具体锚点见分析清单。

## 带 motion token 的 VLA

| 回合 | 难度 | 交换次数 | 目标词顺序 | 本轮结局 | 推理次数 | 单局图 | 取物阶段证据 |
|---|---|---:|---|---|---:|---|---|
| 4 | 易 | 1 | `green` | 成功 | 22 | [头](figures/cases/with_motion_ep4/head_strength.png)／[开局](figures/cases/with_motion_ep4/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep4/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep4-pick1-phases.png) |
| 8 | 易 | 2 | `green` | 成功 | 20 | [头](figures/cases/with_motion_ep8/head_strength.png)／[开局](figures/cases/with_motion_ep8/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep8/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep8-pick1-phases.png) |
| 11 | 难 | 3 | `blue → green` | 成功 | 30 | [头](figures/cases/with_motion_ep11/head_strength.png)／[开局](figures/cases/with_motion_ep11/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep11/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep11-pick1-phases.png)／[第2次](figures/phases/with_motion-ep11-pick2-phases.png) |
| 12 | 易 | 1 | `red` | 成功 | 21 | [头](figures/cases/with_motion_ep12/head_strength.png)／[开局](figures/cases/with_motion_ep12/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep12/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep12-pick1-phases.png) |
| 17 | 易 | 1 | `green → red` | 成功 | 31 | [头](figures/cases/with_motion_ep17/head_strength.png)／[开局](figures/cases/with_motion_ep17/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep17/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep17-pick1-phases.png)／[第2次](figures/phases/with_motion-ep17-pick2-phases.png) |
| 18 | 中 | 2 | `red` | 成功 | 20 | [头](figures/cases/with_motion_ep18/head_strength.png)／[开局](figures/cases/with_motion_ep18/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep18/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep18-pick1-phases.png) |
| 20 | 易 | 2 | `red` | 成功 | 28 | [头](figures/cases/with_motion_ep20/head_strength.png)／[开局](figures/cases/with_motion_ep20/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep20/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep20-pick1-phases.png) |
| 26 | 中 | 1 | `green` | 失败 | 20 | [头](figures/cases/with_motion_ep26/head_strength.png)／[开局](figures/cases/with_motion_ep26/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep26/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep26-pick1-phases.png) |
| 34 | 中 | 1 | `blue` | 成功 | 21 | [头](figures/cases/with_motion_ep34/head_strength.png)／[开局](figures/cases/with_motion_ep34/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep34/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep34-pick1-phases.png) |
| 37 | 易 | 2 | `green` | 成功 | 25 | [头](figures/cases/with_motion_ep37/head_strength.png)／[开局](figures/cases/with_motion_ep37/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep37/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep37-pick1-phases.png) |
| 38 | 中 | 1 | `blue` | 成功 | 21 | [头](figures/cases/with_motion_ep38/head_strength.png)／[开局](figures/cases/with_motion_ep38/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep38/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep38-pick1-phases.png) |
| 40 | 易 | 2 | `blue` | 成功 | 20 | [头](figures/cases/with_motion_ep40/head_strength.png)／[开局](figures/cases/with_motion_ep40/opening_and_robust.png)／[motion与通路](figures/cases/with_motion_ep40/motion_and_paths.png) | [第1次](figures/phases/with_motion-ep40-pick1-phases.png) |

## 不带 motion token 的 VLA

| 回合 | 难度 | 交换次数 | 目标词顺序 | 本轮结局 | 推理次数 | 单局图 | 取物阶段证据 |
|---|---|---:|---|---|---:|---|---|
| 8 | 易 | 2 | `green` | 成功 | 20 | [头](figures/cases/without_motion_ep8/head_strength.png)／[开局](figures/cases/without_motion_ep8/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep8/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep8-pick1-phases.png) |
| 12 | 易 | 1 | `red` | 成功 | 21 | [头](figures/cases/without_motion_ep12/head_strength.png)／[开局](figures/cases/without_motion_ep12/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep12/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep12-pick1-phases.png) |
| 14 | 中 | 2 | `blue` | 失败 | 20 | [头](figures/cases/without_motion_ep14/head_strength.png)／[开局](figures/cases/without_motion_ep14/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep14/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep14-pick1-phases.png) |
| 16 | 易 | 1 | `green` | 失败 | 21 | [头](figures/cases/without_motion_ep16/head_strength.png)／[开局](figures/cases/without_motion_ep16/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep16/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep16-pick1-phases.png) |
| 18 | 中 | 2 | `red` | 成功 | 20 | [头](figures/cases/without_motion_ep18/head_strength.png)／[开局](figures/cases/without_motion_ep18/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep18/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep18-pick1-phases.png) |
| 20 | 易 | 2 | `red` | 失败 | 25 | [头](figures/cases/without_motion_ep20/head_strength.png)／[开局](figures/cases/without_motion_ep20/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep20/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep20-pick1-phases.png) |
| 26 | 中 | 1 | `green` | 失败 | 21 | [头](figures/cases/without_motion_ep26/head_strength.png)／[开局](figures/cases/without_motion_ep26/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep26/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep26-pick1-phases.png) |
| 34 | 中 | 1 | `blue` | 失败 | 24 | [头](figures/cases/without_motion_ep34/head_strength.png)／[开局](figures/cases/without_motion_ep34/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep34/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep34-pick1-phases.png) |
| 38 | 中 | 1 | `blue` | 成功 | 22 | [头](figures/cases/without_motion_ep38/head_strength.png)／[开局](figures/cases/without_motion_ep38/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep38/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep38-pick1-phases.png) |
| 47 | 难 | 2 | `green → red` | 成功 | 30 | [头](figures/cases/without_motion_ep47/head_strength.png)／[开局](figures/cases/without_motion_ep47/opening_and_robust.png)／[motion与通路](figures/cases/without_motion_ep47/motion_and_paths.png) | [第1次](figures/phases/without_motion-ep47-pick1-phases.png)／[第2次](figures/phases/without_motion-ep47-pick2-phases.png) |

## 本机完整数据

每局逐次完整JSON位于 `v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/analysis/<model>_ep<id>/episode_analysis.json`。两种模型的目录标识符分别是 `with_motion`、`without_motion`；该标识符不替代正文中的固定模型名称。

完整原始NPZ、视频和motion token生成记录仍在同一运行根，未将大体积原始数组提交进Git。
