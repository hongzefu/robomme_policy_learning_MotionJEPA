# 主图逻辑说明：两条通路与「不按动作调整」

> 更新（美东 2026-11-06）：帧读 motion 与路径分解已在 PatternLock、RouteStick 上补测，三跳同样成立，经帧占比 58%～67%，经文字 13%～23%，见 `result.md`「三跳在 PatternLock、RouteStick 上的验证」和 `records/pathway-mot-summary.txt`。下文写「未测」的地方以这里为准；图本身还是 Swap 的数字。

写给要把 `fig_pathway.pdf` 放进论文的人。全部数字来自仓库 HEAD `1983b0b` 已归档的文件，记号 `D=docs/evaluation-doc/subgoal-rollout-attn-seed42-1106`、`S=docs/evaluation-doc/swap-first-bin-attn-seed42-1105`。图由 `plot_pathway_figure.py` 从数据文件生成，面板 (a) 示意图上的标签也由脚本从同一份数据格式化，不是手抄。数字写法「a|b」在第一部分指 Button | Video 两个 Swap 任务，在第二部分指 PatternLock、RouteStick。

## 1. 核心论点（一段话）

motion token 的内容确实能到达动作 token，而且走的是两条可以分开看的注意力通路：路线 A「motion → 帧记忆 → 动作」——帧记忆在第 0～1 层大量读 motion、到第 3 层前后累积到顶，动作在第 14～16 层直接读帧，这时帧里的 motion 含量已经到顶；路线 B「motion → 语言 → 动作」——语言在第 0～1 层直接读 motion，动作在各层平均约三成读语言。但「有通路」不等于「按需使用」：把 PatternLock、RouteStick 的全部 922 次推理按当前子目标分组后，动作和语言读示范帧、动作读示范 motion 都偏向示范开头，基线的语言读帧偏向示范末尾，没有一条按子目标走的对角线；唯一例外是 motion 模型的语言读 motion 流偏向示范后部、只看成功局时隐约有对角但未到门槛。两部分都是注意力读数，不能证明信息真的这样流动，也不能和成败挂钩。

## 2. 第一部分：两条通路怎样影响动作

数据：`S/records/action-flow-summary.json`，motion 模型，两个 Swap 任务各一格（`cells["motion|ButtonUnmaskSwap"]`，49 局；`cells["motion|VideoUnmaskSwap"]`，50 局），每局闭合前 4 次推理。轴含义按 `D/records/chain_hops.py`：`direct_layer_pct` 形状 (18 层, 12 读者组, 12 来源组)，组序 帧 4 段 (0–3)、motion 4 段 (4–7)、第三人称图像 8、手腕图像 9、文字 10、动作 11；`alpha_0.5/carry_by_layer` 形状 (19, 12, 12)，第 l 项是第 l 层输入时各 token 的来源构成；`alpha_*/route_via_source` 形状 (11 中转组, 12 来源组)。帧的量按各段 token 数加权（`token_counts`）。对应输出 `D/records/chain-hops.txt`。

### 跳 1：帧记忆吸收 motion（只在 Swap 上测过）

| 读数 | Button | Video | 来源 |
| --- | ---: | ---: | --- |
| 帧读 motion，L0 | 50.4% | 24.9% | `direct_layer_pct[0][0:4][:,4:8]` 加权和，`chain-hops.txt` 第 2/15 行 |
| 帧读 motion，L1 | 69.3% | 57.5% | 同上 `[1]` |
| 帧累积 motion，L1 输入 | 25.2% | 12.5% | `alpha_0.5/carry_by_layer[1][0:4][:,4:8]`，`chain-hops.txt` 第 3/16 行 |
| 帧累积 motion，L3 输入 | 54.0% | 40.7% | 同上 `[3]` |
| 帧累积 motion，末层（L18 输入 = L17 输出） | 60.1% | 47.4% | 同上 `[18]` |

图 (a) 帧记忆框里写的是第 L0 / L2 / L17 层之后的值（分别等于第 L1 / L3 / L18 层的输入），与 (b) 的蓝线是同一数组。L2 之后已到末值的 86%～90%，之后仍缓慢上升到约 L11，所以不写「到顶」。全部按 `token_counts` 加权。

### 路线 A：motion → 帧记忆 → 动作

| 读数 | Button | Video | 来源 |
| --- | ---: | ---: | --- |
| 动作读帧，L14 / L15 / L16 | 4.9 / 5.4 / 4.0% | 3.0 / 3.4 / 5.1% | `direct_layer_pct[l][11][0:4]` 求和，`chain-hops.txt` 第 9/22 行 |
| 动作读帧，L16，PatternLock / RouteStick，motion 模型 | 11.4% | 11.1% | `D/records/layer-flow.json["PatternLock/mot/action"]["by_layer_pct"][16][0]`、`["RouteStick/mot/action"]…` |
| 同上，基线 | 4.3% | 3.9% | `…/ctx/action…[16][0]` |
| 动作里来自 motion 的总量，α=0.5 | 7.70% | 4.67% | `alpha_0.5/route_via_source[:,4:8]` 全部求和，`chain-hops.txt` 第 12/25 行 |
| 其中经帧的占比，α 取 0.2 / 0.5 / 0.8 | 39.9 / 44.2 / 47.2% | 43.4 / 49.8 / 55.7% | `route_via_source[0:4, 4:8]` ÷ 总量（脚本内计算；`chain-hops.txt` 给的是绝对量 4.70 / 3.40 / 1.63 与 3.05 / 2.33 / 1.25） |

图上写「via frames 40–56%」是这 6 个占比的最小–最大；用户给的「40%～56%」与此一致。动作读帧的层正是 L14～16，此时帧的 motion 含量已到顶（60% | 47%），所以路线 A 在时序上是通的。

### 路线 B：motion → 语言（L0、L1）→ 动作

| 读数 | Button | Video | 来源 |
| --- | ---: | ---: | --- |
| 文字读 motion，L0 / L1 | 32.8 / 55.6% | 24.4 / 32.5% | `direct_layer_pct[l][10][4:8]` 求和，`chain-hops.txt` 第 5/18 行 |
| 动作读文字，L0～L17 范围 | 7.9–62.9% | 8.5–63.7% | `direct_layer_pct[:,11,10]`，`chain-hops.txt` 第 8/21 行；图上写合并范围 8–64% |
| 动作读文字，层平均 | 29.9% | 31.0% | 同上取均值（脚本内算）；图上写合并均值 30% |
| 经文字的占比，α 取 0.2 / 0.5 / 0.8 | 25.0 / 20.0 / 18.4% | 31.7 / 24.1 / 18.5% | `route_via_source[10, 4:8]` ÷ 总量 |

图上写「via text 18–32%」是这 6 个占比的最小–最大（用户给的「19%～32%」是四舍五入差异，以文件为准是 18.4%～31.7%）。本轮 PatternLock、RouteStick 的文字读取同样如此：L0 读 motion 24.0% / 32.0%、L1 17.6% / 34.8%（`layer-flow.json["PatternLock/mot/text"]["by_layer_pct"][0][1]`、`[1][1]`，RouteStick 同理），L2 起都很小。

### 为什么不是「motion → 帧 → 语言 → 动作」

| 读数 | Button | Video | 来源 |
| --- | ---: | ---: | --- |
| 文字读帧，L0 | 26.5% | 26.2% | `direct_layer_pct[0][10][0:4]` 求和，`chain-hops.txt` 第 4/17 行 |
| 文字读帧，L1 | 1.5% | 3.3% | 同上 `[1]` |
| 文字读帧时帧里的 motion，L0 | 0 | 0 | `carry_by_layer[0]` 的帧行对 motion 列为 0（第 0 层输入尚未混合），`chain-hops.txt` 第 3/16 行首项 |
| 文字经帧间接拿到 motion，L0～L15 每层 | ≤0.8% | ≤0.6% | `chain-hops.txt` 第 6/19 行（读帧 × 帧含 motion） |

语言读帧的尖峰在第 0 层（之后每层不超过约 5%），而帧要到第 0 层之后才开始含 motion、到第 2 层之后已有末值的大部分，两件事在层序上错开，所以这条串联很弱。图 (a) 用灰色虚线表示，(b) 里绿线（语言读帧）在 L0 的尖峰与蓝线（帧含 motion）的上升错开，就是这个意思。

### 当前图像的位置

掩码规定当前图像不能读记忆：`direct_layer_pct[:, 8:10, 0:8]` 全为 0（脚本里 `assert`；`S/result.md`「当前帧有没有融合历史」节的判定行 `CURRENT_FUSION=PASS … image_to_memory_max_pct=0.000e+00`）。图像只能经语言拿到记忆：图像读文字在 L14～16 为 61–83%（`direct_layer_pct[14:17, 8:10, 10]` 两路两任务的最小–最大；用户给的「60%～83%」在 `S/result.md` 原文为「约 60%～83%」，按文件精确值是 61.2%～82.7%）。动作读图像 L0～17 为 9–54%（`direct_layer_pct[:,11,8:10]` 求和）。这两条在图上是灰色实线，不属于两条 motion 通路，画出来是为了交代图像和语言的关系。

## 3. 第二部分：有通路，但读哪段记忆不随动作调整

数据：`D/records/summary.json["A2"][任务][模型][读者][流][权重]`，帧流权重 `single`、motion 流权重 `overlap`；每格有 `D`（对角富集，局级重采样 2000 次的均值与 95% 区间）、`slope`、`matrix.enrichment`（执行第 j 段时读示范第 j′ 段的 share ÷ uniform）、`matrix.n_episodes_per_cell`。拆解读数在 `D/records/decompose.json["cells"]["任务/模型/读者/流"]`，三个键 `first_later`（j≥2 时读第 1 段）、`diag_middle`（1<j<m 时读第 j 段，唯一能排除首尾偏好的「真对位」）、`last_earlier`（j<m 时读第 m 段）。判据在 `summary.json["thresholds"]`：D 下界 ≥1.2 或斜率下界 ≥0.2 算「明显」。

### 证据

- **12 组预定读数没有一组「明显」**（`D/result.md` A2 表；`summary.json` 各格 `tier` 为「弱」8 组、「未见」4 组）。
- **motion 模型的三条帧／motion 读数都偏向示范开头**（图 (c) 左上、右上，矩阵第 1 列整列偏高；图 (d) 蓝点）：

  | 任务 / 读者 · 流 | 读首段 | 真对位 | 读末段 | 来源（`decompose.json`） |
  | --- | ---: | ---: | ---: | --- |
  | PatternLock · 动作 · 帧 | 1.35 [1.28, 1.42] | 0.81 [0.78, 0.85] | 0.84 [0.81, 0.87] | `PatternLock/mot/action/frame` |
  | RouteStick · 动作 · 帧 | 1.41 [1.35, 1.47] | 0.85 [0.82, 0.88] | 0.82 [0.80, 0.84] | `RouteStick/mot/action/frame` |
  | PatternLock · 动作 · motion | 1.16 [1.05, 1.26] | 0.89 [0.75, 1.02] | 0.80 [0.71, 0.91] | `PatternLock/mot/action/motion` |
  | RouteStick · 动作 · motion | 1.35 [1.26, 1.46] | 0.95 [0.87, 1.03] | 0.81 [0.74, 0.89] | `RouteStick/mot/action/motion` |
  | PatternLock · 文字 · 帧 | 1.74 [1.56, 1.90] | 0.47 [0.38, 0.59] | 1.40 [1.27, 1.54] | `PatternLock/mot/text/frame` |
  | RouteStick · 文字 · 帧 | 2.36 [2.09, 2.62] | 0.35 [0.33, 0.38] | 1.10 [0.99, 1.23] | `RouteStick/mot/text/frame` |

  这六格读首段 1.16～2.36；帧流四格的真对位区间都低于 1，动作读 motion 两格的均值低于 1，但区间含 1；文字读帧的斜率为负（PatternLock −0.32 [−0.42, −0.25]、RouteStick −0.37 [−0.46, −0.30]，`summary.json["A2"][任务]["mot"]["text"]["frame"]["single"]["slope"]`），走得越后读得越靠前。
- **基线的文字读帧偏向示范末尾**（图 (c) 左下，右上角偏高；图 (d) 绿点）：读末段 1.82 [1.73, 1.91]、1.93 [1.85, 2.02]，读首段 0.30 [0.21, 0.41]、0.21 [0.19, 0.24]，真对位 0.80 [0.61, 0.96]、0.74 [0.50, 0.98]（`decompose.json["PatternLock/ctx/text/frame"]`、`["RouteStick/ctx/text/frame"]`）。基线动作读帧三项都在 0.76～0.91，没有偏向（`…/ctx/action/frame`）。
- **图 (c) 四个 RouteStick 面板的 D 与斜率**（`summary.json["A2"]["RouteStick"]…`）：motion·动作·帧 1.19 [1.12, 1.27]、−0.02；motion·文字·帧 1.19 [1.04, 1.36]、−0.37；基线·文字·帧 0.48 [0.36, 0.60]、+0.02；motion·文字·motion 0.89 [0.79, 0.99]、+0.01。每行局数 n=50、30、13、7、2（`matrix.n_episodes_per_cell[j][0]`），第 4、5 行样本很少。

### 范围与例外（必须如实写）

「基本都在看开头或末尾」只对**帧流**（动作读帧、文字读帧）和**动作读 motion** 成立。**motion 模型的文字读 motion 流是例外**：它不偏向开头而偏向后部——读首段 0.60 [0.40, 0.81]、0.57 [0.48, 0.66]，读末段 1.29 [1.18, 1.41]、1.23 [1.15, 1.31]，真对位 1.16 [0.97, 1.34]、1.30 [1.19, 1.39]（`decompose.json["PatternLock/mot/text/motion"]`、`["RouteStick/mot/text/motion"]`）。全部局的预定判定是「未见」（D 0.79 [0.67, 0.90]、0.89 [0.79, 0.99]）；只看成功局 D 变成 1.14 [1.06, 1.24]、1.14 [1.09, 1.19]，斜率 0.11 [0.05, 0.17]、0.07 [0.00, 0.15]，判「弱」，矩阵隐约有对角但没到门槛 1.2 / 0.2（`D/records/success-only.json["cells"]["PatternLock/mot/text/motion"]`、`["RouteStick/mot/text/motion"]` 的 `D`、`slope`、`tier`）。成功局的真对位 1.25 [0.90, 1.46]（5 局）、1.40 [1.34, 1.47]（7 局），但读末段也有 1.21、1.18，分不开是对位还是偏爱后半段（`success-only.json` 同格 `diag_middle`、`last_earlier`）。图 (c) 右下面板标题里写了这个成功局 D，图 (d) 单列一组「exception」。

所以论文里的总括应该写成「读示范哪一段没有随当前子目标调整」：帧流与动作读 motion 一直偏向开头、基线文字偏向末尾、文字读 motion 偏向后部，三种偏向都与进度无关；不要写成「所有读数都不变」或「所有读者都只看开头或末尾」。

## 4. 两部分怎么接成一个叙事

第一部分说明的是**能力**：motion 的内容有两条可用的通路进入动作，帧记忆是最大的中转（经帧 40–56%），语言是第二条（18–32%），而且两条路各自在层序上说得通。第二部分说明的是**没有看到按需使用**：如果模型真的靠示范来决定下一步，走到第 j 个子目标时应该多读示范第 j 段，矩阵该出现对角线；实际看到的是位置偏好——开头（motion 模型帧流、动作读 motion）、末尾（基线文字读帧）、后部（motion 模型文字读 motion）。基线文字读帧的 D 随 j 上升（RouteStick 0.21、0.91、1.57、2.92、3.65，后两项只有 3、1 局），已过去的段读得少，这和末段偏好、走得远的局的选择分不开，所以不写成「基线完全不随 j 变」。两部分合起来就是「有通路 ≠ 按需使用」：从注意力看，通路连到了动作 token，但读的始终偏向同几段，没有看到随当前动作切换。这是注意力读数，不等于信息真的这样传递。注意这两部分数据来源不同：第一部分来自 Swap（有 prefix 注意力，能测帧读 motion），第二部分来自 PatternLock、RouteStick（没有 prefix，只测了文字和动作的读取）；第一部分里「PL/RS L16 ≈11%」那条是本轮数据，已在图上单独标明。

## 5. 英文图注（可直接放进论文）

**Figure X. Motion-token content reaches the action tokens through two attention pathways, but which demonstration segment is read does not track the current sub-goal.** All numbers are attention read-outs (share of a reader's attention mass, head-averaged, episode-equal-weighted) for seed 42; they describe where the model looks, not causal information flow. **(a)** Pathway schematic on the two Swap tasks (ButtonUnmaskSwap | VideoUnmaskSwap, motion model, 49 | 50 episodes, the four inferences up to the first gripper closure). Route A (blue): frame memory reads motion tokens at L0–L1 (50 | 25%, 69 | 57%), its cumulative motion content (attention rollout, α = 0.5) reaches 54 | 41% after L2 and 60 | 47% after L17, and the action tokens read frame memory at L14–L16 (4.9/5.4/4.0 | 3.0/3.4/5.1%; 11% at L16 on PatternLock/RouteStick vs. 4% for the baseline). Route B (orange): the language tokens read motion tokens at L0–L1 (33 | 24%, 56 | 33%) and the action tokens read language at every layer (8–64%, mean 30%). Of the motion content that reaches the action tokens (7.7 | 4.7% at α = 0.5), 40–56% arrives via frame memory 18–32% via language and 13–18% via image tokens, which can reach memory only through language (last hop, α = 0.2–0.8). The dashed chain motion → frames → language is weak because language reads frame memory mostly at L0 (27 | 26%), before the frames carry any motion content. The current image cannot read memory (masked, 0%) and reaches it only through language (image reads language 61–83% at L14–L16). **(b)** The same four quantities per layer for each Swap task: frame-memory motion content rises to its plateau after language has already read memory at L0, and action reads frames late (L14–L16). **(c)** Sub-goal alignment matrices for RouteStick (50 episodes per model, all inferences): attention share on demonstration segment j′ while executing sub-goal j, divided by the token-uniform share (1 = uniform; n = episodes per row). The motion model's action→frames and language→frames read-outs mostly load on the first column at every j; the baseline's language→frames loads on the last segments; the motion model's language→motion stream is the one exception, loading on later segments (success-only, n = 16: D = 1.14 [1.09, 1.19], below the pre-specified threshold of 1.2). No panel shows a stable diagonal: the bright cells in rows 4–5 of the baseline panel come from very few episodes (n = 3, 1) and mostly coincide with the last segment. **(d)** Post-hoc decomposition of the diagonal enrichment D into reading the first segment (j ≥ 2), true alignment (1 < j < m reads segment j) and reading the last segment (j < m), mean and episode-level bootstrap 95% CI (2000 resamples), for RouteStick (RS) and PatternLock (PL). True alignment is below 1 for every frame-stream read-out of both models; n differs by series (episodes reaching j ≥ 2; reaching 1 < j < m, as few as 9; m ≥ 2). Together with (c), the motion model's first-segment bias shows no adjustment that follows j. The baseline reads past segments less at later j, which cannot be separated from its last-segment preference or from the selection of episodes that reach later sub-goals. Frame stream uses the "single" weighting, motion stream "overlap"; (c)–(d) are descriptive and not corrected for multiple comparisons.

## 6. 不能说的话与限制

- **注意力不等于因果，也不等于信息流。** 两部分都是注意力读数；模型可能经残差、MLP 或语言间接用到示范，这些读不出来。不要写「这导致失败」「motion 没用」「模型靠路线 A 决策」。第一部分的通路只说明「注意力这样连」，没有做干预验证（`D/result.md`「表征之间的三跳」节末句、「限制」节）。
- **累积与路径是 α 近似。** `R[l+1] = α R[l] + (1−α) A[l] R[l]` 忽略 MLP 和 value 范数，所以用 α = 0.2 / 0.5 / 0.8 复核；图上「60 | 47%」「7.7 | 4.7%」是 α = 0.5 的值，经帧／经文字的占比给的是三个 α 的范围（`S/result.md` 第二部分说明；`S/records/compute_action_flow.py` 顶部注释）。
- **只有 seed 42，单次。** 模型差异不能与 seed 波动分开；motion 模型与基线的训练来源也不同（`D/result.md`「限制」）。
- **两部分数据来源不同。** 第一部分的跳 1（帧读 motion）与路径分解只在 Swap 上测过（每局闭合前 4 次推理，不是全程）；PatternLock、RouteStick 没有存 prefix 注意力，帧读 motion 这一跳未测。文字、动作的逐层读取两轮都有。不要把 Swap 的百分比当成 PatternLock 的，图上已分别标注。
- **第二部分帧流全程只留 32 帧。** 长局每个子目标只分到几帧；motion 窗 32 帧按用户口径算，实际 33 帧；PatternLock 子目标中位 35 帧，相邻段会混（`D/result.md`「限制」）。
- **样本结构与成败样本少。** 多数局在第 1 个子目标就失败，大 j 的推理主要来自走得远的局，有选择偏差；成功局 PatternLock·motion 18、RouteStick·motion 16、基线各 9，且偏 easy，按结局筛选本身有选择偏差（`D/result.md`「成绩」「只看成功局」）。与成败的关系分不开（A3 节，只有 4 项每侧够 10 局）。
- **拆解与三跳都是事后补做、未校正多重比较。** 首段／真对位／末段三项与「只看成功局」都不在预定判据里（`decompose.json["note"]`、`success-only.json["note"]`）；三跳分析是用户追问后补做（`D/result.md`「表征之间的三跳」）。
- **例外要写足。** motion 模型文字读 motion 偏向后部、成功局隐约有对角，是「唯一值得再查的线索」，只作线索不作结论（`D/result.md`「唯一值得再查的线索」）。
- **图 (c) 第 4、5 行**只有 7、2 局（RouteStick），第 6、7 列只有 ≤5 局，这些格子的颜色不可解读为稳定结果。
- 未找到的项：无。用户给的「文字读帧 L1 起 1.5%～3.3%」在文件里对应的是 L1 这一层的两任务值（1.5、3.3），不是「L1 起所有层的范围」（L2～L15 为 0～1.4%，L16 为 3.9 | 4.6%，L17 为 1.7 | 1.5%，见 `chain-hops.txt` 第 4/17 行），图上写作「L1 1.5|3.3%」。
