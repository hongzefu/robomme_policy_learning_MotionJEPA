# 5.2 与 Fig. 4 措辞草稿（待用户确认，未写入 tex）

这份文件只是草稿。按计划红线「稿件措辞先贴全文给用户确认再写入」，本轮没有改稿件仓库的 tex。数字全部来自本目录的 [result.md](result.md)。

篇幅约束：sources.md §54 记录正文已恰好排满 8 页、没有余量。所以三版草稿都按现有 5.2 正文段（约 150 词）和图注（约 55 词）的长度来写。

篇幅核对（2026-11-09 美东 07:33，按英文词数粗算，[图注, 正文段]）：方案 A [70, 186]；方案 B [60, 122]；方案 C [57, 143]。现有 5.2 正文段约 150 词、图注约 55 词；超出部分写入 tex 前需压缩（优先删限定从句，把限定移到附录）。

## 方案 A（推荐）：Fig. 4 换成「oracle 增量」图

图：`paper/figures/candidates/belief_beyond_oracle_common.pdf`，需移出 candidates 并改名为正式图。原 Fig. 4 下排的拟合线不再需要。

**图注**：

> **What hidden states add beyond the task specification on StopCube.** Both policies replay the same 98 official demonstrations (2,344 hover-phase points). Out-of-fold $R^2$ for the steps left to the correct window, from the true time step and target count $[t,N]$ plus the tokens of one group (text, current frame or action) at each layer. Dashed: $[t,N]$ alone; dotted: $[t,N]$ with the true cube speed $M$.

**正文段**（替换现有 5.2 第一段 “Task progress is central … uses this estimate.”）：

> Task progress is central to temporal memory, and StopCube tests it most directly: the robot hovers over a button and must press it when a cube reaches the target on its $N$-th visit, the only time-sensitive RoboMME task and the one humans find hardest (78\%)~\cite{dai2026robomme}. To compare representations rather than behaviors, we replay the same 98 official demonstrations through both policies and probe the steps left to the correct window (\cref{fig:belief}). The true time step and target count alone give $R^2=0.45$; adding the true cube speed gives 0.93. Adding MotionMem's text tokens raises $R^2$ to 0.78--0.84 at every layer, whereas FrameSamp's stay at 0.48--0.56 (at layer 17, MotionMem's MAE is 16 steps lower than FrameSamp's; paired 95\% CI [12, 20]). The added information appears to be mostly the cube's speed: MotionMem's text tokens decode it with accuracy 0.81 versus 0.60 for FrameSamp's (best layer chosen on training folds), mainly from layer 0, where text reads motion tokens (\cref{sec:attn}). Decoding alone does not show that the policy uses it, and we did not ablate the motion channel.

要点：
- 正文第一句的「the only time-sensitive RoboMME task」「humans find hardest (78%)」沿用稿件现有 5.2 原文，出处见稿件 sources.md §46（RoboMME 论文的 Human Performance 行与任务分类表，78% 为人类成功率，16 个任务中最低）；写入时可改为「human success 78\%」以免歧义。
- 「mainly from layer 0」与 5.3 的 route B（language 只在第 0–1 层读 motion）互相印证；写入前确认 tex 里有 `\label{sec:attn}`（2026-11-09 核对：`5_interpretability.tex` 第 39 行有）。
- 0.81 对 0.60 是训练折内选层的值（MotionMem 多选第 0 层，FrameSamp 选第 7–17 层）；全数据第 0 层为 0.80 对 0.57。方案 A、B 已统一用 0.81／0.60。
- 「at every layer」：修正口径下 MotionMem Text 第 0–17 层为 0.78–0.84，成立；FrameSamp Text 为 0.48–0.56（0.447 是 [t,N] 本身）。
- 时间上的限定：O 节显示早期（t<80）MotionMem Text 读速度只有 0.55，与 FrameSamp 无显著差，之后上升到 0.95；同局对照里 FrameSamp Text 也会上升（+0.149），只是幅度较小。正文篇幅不够时不写，附录写。
- 「16 steps lower error」来自同点配对差 +16.0 [12.1, 19.7]。


**方案 A 精简版**（按篇幅压缩；图注 54 词，正文段 142 词）：

图注：

> **What hidden states add beyond the task specification on StopCube.** Both policies replay 98 official demonstrations. Out-of-fold $R^2$ for the steps left to the correct window from the true time step and target count $[t,N]$ plus one token group per layer. Dashed: $[t,N]$; dotted: $[t,N]$ and the true cube speed.

正文段：

> StopCube tests task progress most directly: the robot must press a button when a cube reaches the target on its $N$-th visit, the only time-sensitive RoboMME task (human success 78\%)~\cite{dai2026robomme}. We replay the same 98 official demonstrations through both policies and probe the steps left to the correct window (\cref{fig:belief}). The true time step and target count give $R^2=0.45$; adding the true cube speed gives 0.93. Adding MotionMem's text tokens raises $R^2$ to 0.78--0.84 at every layer, FrameSamp's to 0.48--0.56 (layer 17: 16 steps lower MAE, paired 95\% CI [12, 20]). The gain appears to be mostly the cube's speed, which MotionMem's text tokens decode better (0.81 vs.\ 0.60), mainly from layer 0, where text reads motion tokens (\cref{sec:attn}). Decoding does not show use, and we did not ablate the motion channel.

## 方案 B：只把原 Fig. 4 换成共同轨迹版

图：`paper/figures/belief_probe_common.pdf`。

**图注**：

> **Linear probes of the steps left to the correct window on StopCube**, with both policies replaying the same 98 official demonstrations. *Top:* out-of-fold $R^2$ per layer for three token groups; dashed, an oracle probe given the true time step and target count; dotted, time alone. *Bottom:* linear fit of predictions against the truth at layer 17; dotted, $y=x$.

**正文段**：

> … we replay the same 98 official demonstrations through both policies and fit linear probes from their hidden states to the steps left until the correct window (\cref{fig:belief}). Over layers 8--16 the two policies are close and slightly above an oracle given the true time step and target count ($R^2=0.45$); MotionMem's text tokens are higher at layer 0 (0.65 vs.\ 0.46) and lower at layer 17 (0.38 vs.\ 0.49). The difference lies elsewhere: added to the oracle inputs, MotionMem's text tokens raise $R^2$ to 0.81--0.84 at layers 0--16 and 0.78 at layer 17 (FrameSamp: 0.48--0.56), apparently because they carry the cube's speed (accuracy 0.81 vs.\ 0.60, best layer). Decoding alone does not show that the policy uses it.

## 方案 C：Fig. 4 换成「按记忆需求分档」图

图：`paper/figures/candidates/belief_by_demand_common.pdf`（需移出 candidates 并改名为正式图）。

**图注**：

> **Probing the steps left to the correct window, split by memory demand.** Both policies replay the same 98 official StopCube demonstrations; out-of-fold $R^2$ of the text tokens per layer, evaluated separately on episodes needing $N\le3$ or $N\ge4$ target visits (probes trained on all). Success rates in the legend; dotted, the true $[t,N]$ reference.

**正文段**（替换现有 5.2 第一段）：

> Task progress is central to temporal memory, and StopCube tests it most directly: the robot must press a button when a cube reaches the target on its $N$-th visit~\cite{dai2026robomme}. We replay the same 98 official demonstrations through each policy and probe the steps left to the correct window from the text tokens (\cref{fig:belief}). When the target is reached late ($N\ge4$), text-token decodability is ordered FrameSamp $<$ MotionMem (96 motion tokens) $<$ MotionMem (384 tokens) at every layer with 64-d PCA probes (mean $R^2$ 0.35, 0.52, 0.64), in line with their StopCube success; when it is reached early ($N\le3$), FrameSamp is on par with or better than both (0.27 vs.\ 0.17 and $-0.59$). Part of this gap may reflect the cube's speed, which FrameSamp's tokens decode less well. Decoding alone does not show that the policy uses it.

要点：
- 数字来自 result.md 第三节 J 的分档表（第 0–17 层平均）。
- 「at every layer」：N≥4 时 18 层逐层都是 FrameSamp < concat < split；换 5 个分折种子共 90 格，90/90 成立。**但它依赖 PCA 维度**：64、128 维全部成立；16、32 维时 concat 常低于 FrameSamp（同序 41/72，result.md 第三节 J）。正文若写「at every layer」，须注明探针用 64 维 PCA，或改为较弱的说法，例如「on average」。
- 正文用的是 concat（96 tokens）；加入 split（384 tokens）需要在正文里交代它是同一方法的另一个预算配置。若不想引入 split，可删去 384 tokens 那条线和对应数字。

## 可选补句：三个模型的对应关系（result.md 第三节 J）

方案 A 或 B 的段末可加一句（约 35 词，篇幅紧时可放附录）：

> Across three policies, the best-layer $R^2$ for the steps left (0.51, 0.65, 0.77) increases with StopCube success (14\%, 43\%, 66\%; FrameSamp from the original paper); this correlation over three policies is not causal.

注意：
- FrameSamp 的 14% 来自论文 Table 3，另两个来自 official-stage2-1026 的三 seed 逐局结果，平台不同。
- split 和 concat 的差别同时涉及 motion 预算（384 对 96）和读出方式，所以句子只按 token 数描述，不归因。
- 不是所有读数都同序（反例见 result.md 第三节 J），所以用「several … rise」，不写「order as」。
- 0.51／0.65／0.77 是训练折内选层的值。固定层时，第 0、8 层在 5 个分折种子下都同序，但第 17 层不成立（concat 0.38–0.43，低于 FrameSamp 0.46–0.49）。正文若用这句，需写明是「best layer」，或改用第 8 层的数字（0.47–0.51／0.52–0.55／0.66–0.69）。
- 候选图 `paper/figures/candidates/belief_dose_common.pdf`（F3 已合入）。

## 三版都必须保留的限定

写入 tex 时：草稿里的 `**…**`、`*Top:*` 是 Markdown，要改成 `\textbf{}`、`\emph{}`；图注若用 `\caption[short]{…}`，方括号里的 `[t,N]` 要加花括号保护。


1. 回放是开环看专家轨迹；这些示教两个模型训练时都见过。
2. 优势很可能来自 motion 输入通道（近期相邻帧运动），没有做 motion 通道消融，不写成「记忆更强」。
3. 「距按下步数」上的优势依赖线性读法（concat 在 kNN 下不成立；split 只在 Text 第 8／12／17 层成立）；速度 M 的优势在 kNN 下对 Text 组成立，History 组只在第 4、8 层成立。
4. 旧版「over 50 rollouts」「22 vs 7」「0.77 vs 0.51」「0.34/0.40」「0.55/0.62」这些数字要全部删掉或改写。旧图的差距主要来自两个策略访问的状态不同，不能再引用。
