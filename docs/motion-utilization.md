# motion memory 利用率评估

> 环境 B（AWS 单机 8×A100，2026-09-08）。工具在 `scripts/motion-variance/`（目录 README 有文件清单与完整跑法）；留档 `training-doc/mv-openloop-40k/`（开环与逐层分析）与 `training-doc/mv-matrix-40k/`（闭环成功率矩阵）；图在 `motion-utilization/figures/`。本页所有数字都取自两份留档的 `records/`。

<img src="motion-utilization/figures/fig1-design.png" alt="fig1" width="850">

## 一、结论

**模型确实在读 motion token，屏蔽它成功率掉 5 个百分点；但收益主要来自「这条通路存在」，换成别的集的 motion 内容也差不多——正确内容比错误内容好多少，本轮没分辨出来。自训 motion 模型与官方 baseline 的成功率没有检出差异。**

三个关键数字：

- **屏蔽 motion 通路，成功率 −5.00 个百分点**（test + val 合计 400 集，四个 policy seed 均值）。按四个 seed 算的 t 区间 [+3.44, +6.56]，按集重采样的 bootstrap 区间 [+2.31, +7.62]，两种区间都在 0 以上，判定有收益；四个 seed 逐一为 +5.25 / +6.25 / +3.25 / +5.25。屏蔽后还有 27% 的集跑满步数超时（正常只有 1.6%）。
- **换成同任务另一集的 motion 内容，比屏蔽略好 2.56 个百分点**，但 bootstrap 区间 [−5.56, +0.38] 含 0，未检出；正常内容比换入内容好 2.44 个百分点，bootstrap 区间 [−0.62, +5.62] 同样未检出。
- **开环看动作**：屏蔽通路后最终动作的变化是「只换采样噪声 seed」带来变化的 9.9 倍（相当于动作标准差的 73%），换内容是 6.4 倍（43%）。也就是说动作层面内容肯定被读了，只是没转化成可检出的成功率差。

自训模型 vs 官方 baseline：−0.81 个百分点，t 区间 [−2.63, +1.01]，未检出差异；下界跨出 ±2 个百分点，也不能说等价。

## 二、为什么做这件事

自训模型 `awsprod40k-b128-motion/39999` 比官方 baseline 每步多喂 96 个 motion token。此前两次评估结果打架：A100 单 seed 28% vs 官方 24%，Ada 三 seed 24.2% vs 24.5%（`training-doc/eval-3seed-context-vs-motion/result.md`）。`train-infer-consistency.md` 第十节把「模型各层到底用不用 motion」「推理时屏蔽 motion 成功率变多少」列为缺口，`motion-memory-audit-summary.md` 第 7 节第 2 条要求对真实 checkpoint 做内容干预。本轮就是补这两个缺口。

## 三、评的是什么

| 对象 | 路径 | 核对 |
|---|---|---|
| 自训 motion 模型 | `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999` | 加载后逐参数与 checkpoint 逐位一致（59/59 个参数树节点，`mv-matrix-40k/records/param_tree_motion.json`） |
| 官方 baseline | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | 同上，55/55 |
| 换入内容的来源库 | `v1-store/datasets/4task-motion-400ep/motion`，6832 段 × 768 维 | 没有任何一集会换到自己身上 |
| 评估集 | benchmark `env_metadata/{test,val}/`，四任务各 50 集 | test、val 的环境 seed 与训练 H5 的 100 集零重叠（val 本轮首次核） |

## 四、四种条件

| 条件 | 意思 | motion 内容从哪来 |
|---|---|---|
| `official` | 官方 baseline，模型里根本没有 motion 通路 | 无 |
| `normal` | 自训模型正常推理 | motion 编码器实时算出的向量 |
| `mask` | 自训模型，但在 attention 里把 motion token 对应的 key 列全部挡掉（读观测的 prefill 与生成动作的去噪两处都挡），其它 token 读不到它 | 喂零向量；自检证明挡掉后喂什么都不影响动作 |
| `swap` | 自训模型，motion 内容换成同任务另一训练集 episode 的 | 查上面那个库 |

`mask` 和 `swap` 都**不动**每个 motion token 的位置编码、有效位标记与排列顺序，只动内容或可见性。之所以不用「把 motion 标成无效」来屏蔽：token 的位置号是有效位累计数减一，一旦标无效，后面所有 token（包括动作）的 RoPE 位置会整体前移，那就不只是屏蔽 motion 了。

另外要注意 motion token 进模型前是「768 维内容 + 位置投影」拼起来再过一层编码，所以 `normal − mask` 衡量的是**整条 motion 通路**（内容加位置）的效应，不单是内容的效应；只有 `mask − swap` 和 `normal − swap` 才在说内容。

## 五、工具可靠性

评估用的推理函数是逐字复刻生产 `HistoryPi0.sample_actions`，只在两处 attention mask 加了 key 列开关；四种条件共用同一份编译产物。运行前自动核对源码没有漂移，漂移即拒跑。自检项目（全部通过，日志在 `mv-openloop-40k/records/adapter_selftest.txt`）：

- 复刻函数与生产推理逐位相同；开关全开时与不带开关逐位相同。
- 无效 motion 槽位随便灌什么（0、大随机数、±1e4），动作逐位不变。
- 挡掉 motion 后，有效槽位内容换 0 或随机数，动作逐位不变。
- 逐层分析用的 18 层手工展开版本：bf16 下与生产的 `nn.scan` 路径相对差约 3e-3、不逐位（两份 XLA 编译产物的固有差异，与仓库已知的 `VT_FULL_VS_CACHED` 同性质）；升到 f32、最高 matmul 精度后逐位相同，以此确认语义等价。
- 注意力均匀基准、梯度有限性、无效位梯度为 0 各有自检。

## 六、开环：屏蔽或换掉 motion 后，动作变多少

做法：在 6 个训练集 episode 上按真实评估节奏取 149 个决策点，同一观测、同一采样噪声下分别跑 normal / mask / swap，比较最终动作。标尺是「同一观测只换采样噪声 seed」带来的动作差（5 个 seed），把动作差除以它得到倍数。

事先约定的读法：倍数中位数低于 0.25 才算「动作不敏感」；成功率那边，t 区间与 bootstrap 区间都在 0 以上才写「有收益」，都落在 ±2 个百分点内才写「等价」，含 0 又跨出 ±2 只写「未检出」；若 swap 比 mask 差，只能写「错误内容有害」，不能写「正确内容有收益」。

结果（`mv-openloop-40k/result.md`）：

| 分层 | 点数 | 屏蔽 / 噪声 | 换内容 / 噪声 |
|---|---|---|---|
| 全部 | 149 | 9.88（中位 7.52） | 6.36（中位 4.34） |
| 还没有 motion 片段 | 4 | 0（逐位不变） | 0（逐位不变） |
| 已累计 1–4 段 | 11 | 2.4 | 2.3 |
| 5–12 段 | 39 | 9.5 | 6.3 |
| 超过 12 段 | 95 | 11.1 | 7.0 |

反归一化后屏蔽相当于动作标准差的 73%、换内容 43%。敏感性随累计片段数单调上升，四任务一致（屏蔽 8.5–11.0）；换内容在 ButtonUnmask 上明显小（2.7，其它三任务 6.5–7.9）。「动作不敏感」那一档被排除。

<img src="motion-utilization/figures/fig4-noise-facet.png" alt="fig4" width="850">
*fig4：横轴 = 当前观测里已累计的 motion 片段数，纵轴 = 动作差 / 噪声标尺；149 点 × 5 噪声 seed；虚线 1.0 = 噪声标尺，点线 0.25 = 「不敏感」阈值。数据 `mv-openloop-40k/records/open_loop.json`。*

## 七、模型在哪几层读 motion

在上面 6 集里每集取 4 个时刻（集开头、早、中、晚）共 24 点，从三个角度看 18 层：

- **注意力份额**：生成动作的 query 分给 motion token 的注意力，除以「对所有可见 token 均匀分配」的基准。第 0–14 层全部低于 1，**第 15 层 6.2 倍、第 16 层 4.1 倍**，第 17 层 0.18；第 15 层里单个 head 最高把 31% 注意力给了 motion。读观测那一侧，图像帧 token 在第 0/1 层对 motion 的份额是基准的 36/34 倍，文本 token 第 1 层 39 倍。
- **只干预一层看动作**：只在第 l 层挡掉 motion（其它层照常），看最终动作变化几倍噪声。第 1 层 2.9 倍、第 0 层 1.2 倍，其余层都不超过 0.26；只挡「动作 query 直接读 motion」这一项最大是第 16 层 0.25；把第 l 层的 motion 内容换成另一集的，最大是**第 10 层 0.32**。18 层全挡是 7.5 倍，与开环总效应一致。
- **梯度**：固定一组（观测、噪声动作、时间步、真值）算 loss，对第 l 层输入取梯度，比 motion 列与图像帧列的逐 token 大小。第 0 层 motion 是帧的 68–137 倍，第 16 层 20–23 倍，其余 2–6 倍。

三条线索合起来：motion 信息主要在**第 0–1 层就被图像与文本 token 吸收**，之后间接传播；动作 query 直接读 motion 集中在 15–16 层；对**内容**最敏感的读取点在第 10 层。没有任何单独一层能解释全部效应。

<img src="motion-utilization/figures/fig5-layers.png" alt="fig5" width="850">
*fig5：(a) 动作 query 给 motion 的注意力份额 / 均匀基准；(a') 逐 head；(b) 只干预第 l 层时的最终动作差 / 噪声；(c) 第 l 层输入的逐 token 梯度大小（log10）。24 点 = 6 集 × 4 个时刻。*

## 八、闭环成功率（主结果）

4 种条件 × 4 个 policy seed（42 / 7 / 2024 / 17）× (test 200 集 + val 200 集) = 6400 集，32 批，每批 8 个 worker 各占一张 GPU、按集编号轮流分片。6400 集零报错；四种条件在每一集的首次推理输入结构经 sha 核对完全一致（8 组各 200/200）。

统计方法（`summarize_mv.py`）：同一集在两种条件下配对比较；按四个 seed 的配对差算均值与 t 区间；按集重采样 10000 次算 bootstrap 区间（按 split × 任务分层，所有条件与 seed 共用同一套重采样索引）；两种区间都在 0 的同一侧才下「有收益 / 有损失」的结论。

| 条件 | test + val 400 集（4 seed 均值 ± sd） | test | val | 超时占比 |
|---|---|---|---|---|
| official | 25.25 ± 1.10 | 23.37 | 27.12 | 0.4% |
| normal | 24.44 ± 0.52 | 24.87 | 24.00 | 1.6% |
| mask | **19.44 ± 0.97** | 20.62 | 18.25 | **27.0%** |
| swap | 22.00 ± 1.24 | 22.38 | 21.62 | 10.1% |

| 配对 | 差（百分点） | t 区间（四 seed） | bootstrap 区间 | 判定 |
|---|---|---|---|---|
| normal − mask | **+5.00** | [+3.44, +6.56] | [+2.31, +7.62] | **有收益** |
| normal − swap | +2.44 | [+0.16, +4.72] | [−0.62, +5.62] | 未检出 |
| mask − swap | −2.56 | [−3.44, −1.68] | [−5.56, +0.38] | 未检出 |
| normal − official | −0.81 | [−2.63, +1.01] | [−4.31, +2.69] | 未检出 |

`normal − mask` 在 test 和 val 上各自也成立（+4.25 与 +5.75）。屏蔽后步数均值 560 vs 正常 232：没有 motion 通路时策略明显不收敛、不做终止动作。

分任务看屏蔽的损失：VideoUnmask +4.0、ButtonUnmaskSwap +9.75、VideoUnmaskSwap +6.75（都判定有收益），ButtonUnmask −0.5（未检出）。分难度：easy +6.0、hard +6.5（有收益），medium +1.3（未检出）。按集长度四分位：最长的四分之一 +9.95，集越长收益越大。换内容 vs 屏蔽：ButtonUnmaskSwap 上换入别的集反而比屏蔽好 6.5 个百分点（判定有损失，即屏蔽比错误内容更糟），其余三任务未检出。

<img src="motion-utilization/figures/fig2-rates.png" alt="fig2" width="850">
*fig2：成功率，柱 = 4 seed 均值，棒 = 最小–最大，点 = 各 seed；每格每 seed 50 集。*

<img src="motion-utilization/figures/fig3-forest.png" alt="fig3" width="850">
*fig3：配对差森林图，细线 = t 区间，粗线 = bootstrap 95% 区间，灰带 = ±2 个百分点等价界。*

## 九、swap 条件换入的内容从哪来

每一集换入的那一集由「split、任务、接收集编号」的哈希在同任务 100 集训练集里固定选出，四个 seed 共用。接收集跑得比换入集长、需要的 motion 段超出换入集拥有的段数时，按同任务里段数从多到少的顺序换下一集取对应段（fallback）；连整个库里最长的集都不够（上限 ButtonUnmask 27 / ButtonUnmaskSwap 33 / VideoUnmask 19 / VideoUnmaskSwap 22 段）时从头循环（cycle）。实际 756,823 次查表中，直接取到 55.8%，换下一集取到 15.9%，循环 28.3%；开环那 149 点直接取到 79%。

按这三档分层：`normal − swap` 在「换下一集」档 +12.3（有收益，但只有 59 集），直接取到档 +1.0、循环档 −0.4（未检出）；`mask − swap` 在循环档 −5.7（判定有损失，即屏蔽比循环回填的内容更糟）。

<img src="motion-utilization/figures/fig6-donor.png" alt="fig6" width="850">
*fig6：(a) swap 条件按实际推理次数统计的三档占比；(b) 库中各集 motion 段数的累计分布。图中 donor 指换入的那一集，exact / fallback / cycle 即上面三档。*

## 十、结论的边界

- 空干预确认过不改数：还没有 motion 片段的点，屏蔽或换内容动作逐位不变；无效槽位随便灌数动作不变；三种自训条件的首次推理输入结构逐集一致。
- 换入的是训练集专家演示的 motion，接收方是策略自己的 rollout，两者分布不同。`swap` 因此只能解读为「换成另一集专家 motion」，不是「随机噪声内容」。
- 跨卡：不同型号 GPU（A100 vs Ada）此前实测结果不一致（单任务差到 14 个百分点）。本轮在两张 A100 上复跑 4 集、125 次推理，动作 sha 全部一致；矩阵里同一集的四种条件固定跑在同一张卡上，跨卡差异进不了主比较。
- `mask − swap` 与 `normal − swap` 的 t 区间和 bootstrap 区间结论不一致（t 更窄），按事先约定以两者一致为准。
- 本轮只评 checkpoint 39999、固定 400 集、三种推理时干预，不能替代「带 / 不带 motion 训练」的因果对照。test 集前 10 集在 MotionJEPA 编码器训练集内。视频未人工抽看。

## 十一、留档索引

- `training-doc/mv-openloop-40k/{launch.md,result.md,records/}`：工具自检日志、开环与逐层分析的全部数据与判定行。
- `training-doc/mv-matrix-40k/{launch.md,result.md,records/}`：环境 seed 清单、参数核对、汇总判定行、逐集结果、配对统计、跨卡校验、批日志。
- 原始产物（不进 git）：`v1-store/reports/motion-variance/`、`v1-store/evaluation/mv-*/`、`v1-store/logs/mv-*`。
- 复现命令见 `scripts/motion-variance/README.md`「一次完整跑法」。计划稿相对 Codex 审计的修订对照见本文件 git 历史（commit `4534867` 版本第十一节）。
