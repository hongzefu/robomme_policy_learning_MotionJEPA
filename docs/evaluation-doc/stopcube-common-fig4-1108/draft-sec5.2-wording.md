# 5.2 与 Fig. 4 措辞草稿（待用户确认，未写入 tex）

这份文件只是草稿。按计划红线「稿件措辞先贴全文给用户确认再写入」，本轮没有改稿件仓库的 tex。数字全部来自本目录的 [result.md](result.md)。

篇幅约束：sources.md §54 记录正文已恰好排满 8 页、没有余量。所以两版草稿都按现有 5.2 正文段（约 150 词）和图注（约 55 词）的长度来写。

## 方案 A（推荐）：Fig. 4 换成「oracle 增量」图

图：`paper/figures/candidates/belief_beyond_oracle_common.pdf`，需移出 candidates 并改名为正式图。原 Fig. 4 下排的拟合线不再需要。

**图注**：

> **What hidden states add beyond the task specification on StopCube.** Both policies replay the same 98 official demonstrations (2,344 hover-phase points). Out-of-fold $R^2$ for the steps left to the correct window, from the true time step and target count $[t,N]$ plus one token group per layer. Dashed: $[t,N]$ alone; dotted: $[t,N]$ with the true cube speed $M$.

**正文段**（替换现有 5.2 第一段 “Task progress is central … uses this estimate.”）：

> Task progress is central to temporal memory, and StopCube tests it most directly: the robot hovers over a button and must press it when a cube reaches the target on its $N$-th visit, the only time-sensitive RoboMME task and the one humans find hardest (78\%)~\cite{dai2026robomme}. To compare representations rather than behaviors, we replay the same 98 official demonstrations through both policies and probe the steps left to the correct window (\cref{fig:belief}). The true time step and target count alone give $R^2=0.45$; adding the true cube speed gives 0.93. Adding MotionMem's text tokens raises $R^2$ to 0.78--0.84 at every layer, whereas FrameSamp's stay at 0.45--0.56 (layer 17: 16 steps lower error, 95\% CI [12, 20]). The added information is the cube's speed, which MotionMem's text tokens encode from layer 0 on (accuracy 0.80 vs.\ 0.57), the layer at which they read motion tokens (\cref{sec:attn}). Decoding alone does not show that the policy uses it.

要点：
- 「layer 0 from on」与 5.3 的 route B（language 只在第 0–1 层读 motion）互相印证。
- 0.80 对 0.57 是全数据第 0 层的读数；训练折内选层的值为 0.807 对 0.599。正文用哪个数需统一，建议用后者（0.81 对 0.60）。
- 「at every layer」：修正口径下 MotionMem Text 第 0–17 层为 0.78–0.84，成立。
- 「16 steps lower error」来自同点配对差 +16.0 [12.1, 19.7]。

## 方案 B：只把原 Fig. 4 换成共同轨迹版

图：`paper/figures/belief_probe_common.pdf`。

**图注**：

> **Linear probes of the steps left to the correct window on StopCube**, with both policies replaying the same 98 official demonstrations. *Top:* out-of-fold $R^2$ per layer for three token groups; dashed, an oracle probe given the true time step and target count; dotted, time alone. *Bottom:* linear fit of predictions against the truth at layer 17; dotted, $y=x$.

**正文段**：

> … we replay the same 98 official demonstrations through both policies and fit linear probes from their hidden states to the steps left until the correct window (\cref{fig:belief}). Layer by layer, the two policies are close and both slightly above an oracle given the true time step and target count ($R^2=0.45$); at layer 17, MotionMem's text tokens read 0.38 against 0.49. The difference lies elsewhere: added to the oracle inputs, MotionMem's text tokens raise $R^2$ to 0.78 (FrameSamp: 0.55), because they encode the cube's speed (accuracy 0.81 vs.\ 0.60). Decoding alone does not show that the policy uses it.

## 可选补句：三个模型的对应关系（result.md 第三节 J）

方案 A 或 B 的段末可加一句（约 35 词，篇幅紧时可放附录）：

> Across three policies, probe scores order as StopCube success does (FrameSamp 14\%, MotionMem with 96 motion tokens 43\%, with 384 tokens 66\%): text tokens decode the steps left with $R^2=0.51$, 0.65 and 0.77.

注意：
- FrameSamp 的 14% 来自论文 Table 3，另两个来自 official-stage2-1026 的三 seed 逐局结果，平台不同。
- split 和 concat 的差别同时涉及 motion 预算（384 对 96）和读出方式，所以句子只按 token 数描述，不归因。
- 候选图 `paper/figures/candidates/belief_dose_common.pdf`（若 F3 合入）。

## 两版都必须保留的限定

1. 回放是开环看专家轨迹；这些示教两个模型训练时都见过。
2. 优势很可能来自 motion 输入通道（近期相邻帧运动），没有做 motion 通道消融，不写成「记忆更强」。
3. 「距按下步数」上的优势依赖线性读法（kNN 下不成立）；速度 M 的优势在 kNN 下仍成立。
4. 旧版「over 50 rollouts」「22 vs 7」「0.77 vs 0.51」「0.34/0.40」「0.55/0.62」这些数字要全部删掉或改写。旧图的差距主要来自两个策略访问的状态不同，不能再引用。
