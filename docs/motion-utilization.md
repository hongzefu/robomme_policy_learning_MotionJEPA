# motion memory 利用率评估（motion-variance）

> 环境 B（AWS 单机 8×A100，2026-09-08 起）。工具在 `scripts/motion-variance/`（目录 README 有文件清单与完整跑法）；留档 `training-doc/mv-openloop-40k/`（阶段 0+1）与 `training-doc/mv-matrix-40k/`（阶段 2）；图在 `motion-utilization/figures/`。
> 计划稿经 Codex 2026-09-08 审计修订（10 条），修订对照见第十一节。本页的数字一律现读留档 `records/`，未产出的段落标「待测」。

## 一、结论摘要

待测——阶段 2 矩阵跑完后按第六节预注册措辞填写：一句话结论 + 三个数（pooled 400 集 `normal−mask` 配对差与区间、`mask−swap`、阶段 0 `mask/noise`）。

阶段 0 smoke（12 点，2 noise seed，`open_loop.smoke.log`）已给出方向性信号：Video 类任务决策点上屏蔽 motion 通路的动作差是 noise 标尺的 3.5–23 倍，换异集内容为 1–7 倍；Button 类前两点 `k=0` 处严格为 0（`OL_ZEROK_NULL=PASS`）。正式数字以 `mv-openloop-40k/result.md` 为准。

## 二、问题与既有证据缺口

自训 `awsprod40k-b128-motion/39999` 每步多喂 96 个 motion token；两次评估结果打架（A100 单 seed 28% vs 官方 24%；Ada 三 seed 24.2% vs 24.5%，`training-doc/eval-3seed-context-vs-motion/result.md`）。`train-infer-consistency.md` 第十节把「未量化模型各层对 motion 的利用率」「未做推理时屏蔽 motion 的成功率」列为缺口；`motion-memory-audit-summary.md` 第 7 节第 2 条「真实 39999 内容干预」即本轮实验。

## 三、被评对象与 provenance

| 对象 | 路径 | 自证 |
|---|---|---|
| motion 模型 | `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999` | `PARAM_TREE_EXACT=PASS n_model=59 n_ckpt=59`（`mv-matrix-40k/records/param_tree_motion.json`） |
| 官方 baseline | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `PARAM_TREE_EXACT=PASS n_model=55 n_ckpt=55` |
| donor 库 | `v1-store/datasets/4task-motion-400ep/motion`（6832 行 × 768） | `DONOR_BANK=PASS rows=6832 self_loops=0`，bank sha `6e70604da518c156` |
| 评估 split | benchmark `env_metadata/{test,val}/`，每任务 50 集 | `TEST_SEED_DISJOINT=PASS` / `VAL_SEED_DISJOINT=PASS val=50x4 train_h5=100x4 overlap=0`（本轮首次核 val） |

## 四、四条件的定义与比较语义

| 条件 | 改了什么 | 没改什么 | sidecar |
|---|---|---|---|
| `official` | —（没有 motion 路） | — | 无 |
| `normal` | — | — | 真 sidecar |
| `mask` | prefill + 去噪两处 attention mask 的 motion key 列置 False，非 motion 的 query 读不到 motion token | `motion_emb/motion_pos/motion_mask/mem_order`、`positions=cumsum(input_mask)−1` 原样 | 无（零向量 stub，`MV_MASK_CONTENT_NULL` 证明内容无关） |
| `swap` | 有效槽 `motion_emb` 换成同任务另一训练集 episode 的内容 | `motion_pos/motion_mask/mem_order` 来自接收方（sha 断言） | 无（查表） |

motion token = `motion_encoder_static(concat(motion_emb, silu(motion_pos_proj(motion_pos))))`。因此 `normal−mask` 是**整个 motion token 通路**（内容 + 位置投影）的效应，不是「768 维内容的收益」；`mask−swap` 与 `normal−swap` 才涉及内容。为什么不能改 `motion_mask`：位置号是 `cumsum(input_mask)−1`，改 mask 会让其后所有 token（含动作）的 RoPE 位置整体前移。

## 五、实现要点与验收（`mv_model_adapter.py --selftest`）

- `f_sample`：逐字复刻 `HistoryPi0.sample_actions` context 分支，只在两处 mask 插 key 列门控；四臂共用 `apply_gate=True` 的同一份编译产物。
- 源码护栏：`inspect.getsource` 空白规范化后匹配预期片段，生产代码漂移即拒跑。
- 验收判定行（`v1-store/reports/motion-variance/adapter_selftest_*.log`，留档 `mv-openloop-40k/records/`）：

| 判定行 | 判据 | 结果 |
|---|---|---|
| `MV_NOOP_BITEXACT` | 无算子复刻 vs 生产 `policy._sample_actions` 逐位；带门控算子（gate=全 True）vs 无算子逐位 | PASS `ref_vs_prod=4/4 mv_vs_ref=4/4` |
| `MV_PAD_INVARIANT` | padding 槽灌 0 / N(0,10) / ±1e4，动作逐位不变 | PASS 12/12 |
| `MV_MASK_CONTENT_NULL` | `mask_all` 下有效槽内容换 0 / 随机，动作逐位不变 | PASS 12/12 |
| `UNROLL_VS_SCAN_BF16`（观察） | 18 层展开 vs `nn.scan`，bf16 | 不逐位：`rel_fro≈3.4e-3`，与仓库已知 `VT_FULL_VS_CACHED`（两份 XLA 编译产物）同性质 |
| `UNROLL_VS_SCAN_F32` / `LAYER_SELFCHECK_F32` | 参数与 `embed_dtype` 升 f32、`matmul_precision=highest`，`rel_fro ≤ 1e-5` | 待测（计划原定 bf16 逐位阻断，实测不成立后改为 f32 语义闸——对计划验收判据的一处偏离） |
| `ATTN_UNIFORM_SELFCHECK` | 合法 mask 均匀分布走同一归约，share == 逐 query 均匀基准 | PASS |
| `GRAD_SELFCHECK` | 18 层梯度有限、padding 位梯度严格 0 | PASS |

## 六、阶段 0：开环动作差与预注册解读

预注册措辞（Codex 审计 7）：

| 阶段 0 | 阶段 2（pooled 400 `normal−mask` 95% 区间） | 允许写的结论 |
|---|---|---|
| mask/noise 中位 < 0.25 | 区间完全落在 ±2pp 内（t 与 bootstrap 都在） | 「所测点动作敏感性较低；成功率在 ±2pp 实际等价界内」 |
| 任意 | 含 0 且跨出 ±2pp | 「尚未检出成功率收益（或损失）」，不写等价 |
| ≥ 0.25 | 全在 0 以上 | 「motion token 通路带来收益」；`mask−swap` 也为正 → 内容被读取；≈0 → 效应主要来自通路存在/位置信号 |
| 任意 | `normal ≈ mask`，`swap < mask` | 「异集内容有害」，不得写成「正确 motion 有收益」 |
| 任意 | 全在 0 以下 | 「motion token 通路造成损失」 |

结果：待测（`OL_ACT_DELTA` / `OL_ZEROK_NULL`，按 k 档、任务、cold/steady 分层；图 `figures/fig4-noise-facet.png`）。

## 七、阶段 1：18 层机制

每集 cold / early / mid / late 4 点、6 集（含 ButtonUnmask）。(i) `LAYER_ATTN`：去噪 action query 的 motion 份额 / 逐 query 合法-key 均匀基准（剔除 padding query）；(ii) `LAYER_ACT_DELTA`：`step_only(l)` / `both(l)` / `kv_vzero(l)` / `kv_donor(l)` 四种逐层干预的完整 10 步最终动作差，donor K/V 来自接收方同 obs 换内容后重新 prefill；(iii) `LAYER_GRAD`：固定 `(obs, x_t, t, u_t)` 的 loss 对第 l 层入口隐状态的梯度，t ∈ {0.1, 0.5, 0.9}。结果：待测（图 `figures/fig5-layers.png`）。

## 八、阶段 2：闭环矩阵与 pooled 400 配对统计

4 条件 × 4 seed（42/7/2024/17）× (test 200 + val 200) = 6400 集，32 批 seed-major，8 worker stride 分片（每片 28/28/24×6 集，worker k 固定 GPU k）。统计（`summarize_mv.py`）：逐 seed 配对差 `mean ± 3.182·sd/√4`；episode bootstrap 10000 次按 split×task 分层、所有条件与 seed 共享索引；McNemar b/c 与精确二项 p；等价判定 = t 与 bootstrap 区间都落在 ±2pp 内；分层 task / difficulty / 启动时空窗（首步 k=0）/ donor 覆盖档 / steps 四分位（取 normal seed 42 的 steps，描述性）。结果：待测（`mv-matrix-40k/records/{summary.txt,paired_stats.json}`，图 fig2 / fig3）。

## 九、donor bank 与覆盖率

主 donor = `hash(split|task|recv_ep)` 在同任务 100 集里固定选（四 seed 共用）；偏移超出主 donor → 沿同任务 exec 窗数降序的兜底链取精确偏移（fallback）；超全库上限（BU 27 / BUS 33 / VU 19 / VUS 22 窗）→ 循环（cycle）。最坏估计（接收方 t=es+1300）`DONOR_COVER_EST exact 20.9% / fallback 13.1% / cycle 66.1%`；实际比例按推理次数 `DONOR_COVER_ACTUAL` 重算（待测）。阻断只有 `self_loops=0`、`cross_seg=0`。

## 十、威胁到结论的因素与已做的否证

- k=0 的点屏蔽/替换逐位无影响（`OL_ZEROK_NULL`）；padding 扰动无影响（`MV_PAD_INVARIANT`）；三臂首步结构 sha 全等（`MV_STRUCT_IDENTICAL`，待测）。
- donor 是专家演示 motion、接收方是策略 rollout（分布差异）——`swap` 固定称「异集专家 motion」，与 `mask` 并列报。
- 跨卡：不同型号（A100 vs Ada）已实测不一致（单任务 ±14pp）；同型号 8 卡本轮做 8 集校验 `MV_XGPU`（待测）；矩阵固定映射使跨卡差异不进主比较。
- 本轮不能替代「带/不带 motion 训练」的因果对照；审计 D-A1 / B-A2 原样保留；test ep 0–9 在 encoder 训练集内。

## 十一、复现命令、判定行与计划修订对照

命令见 `scripts/motion-variance/README.md`「一次完整跑法」；判定行全文见两份 `result.md`。计划相对 Codex 审计的 10 条修订：donor 契约（主 donor + 兜底链，不设 wrap 闸）、`episode_plan` 三方唯一来源（28/28/24×6）、护栏空白规范化、展开路径含完整 10 步去噪与双门控自洽闸、逐 query 均匀基准、删除精度承诺 + pooled 400 主估计、预注册措辞收窄 + ±2pp 等价界、阶段 1 取点 cold/early/mid/late、固定-loss 梯度 + donor K 来源、`MV_ALLOW_DIRTY` 与不起 stub。实现期新增偏离：`UNROLL_VS_SCAN` 由 bf16 逐位改为 f32 语义闸（第五节）。

## 十二、留档索引

- `training-doc/mv-openloop-40k/{launch.md,result.md,records/}`：adapter 自检日志、`open_loop.{json,summary.txt}`。
- `training-doc/mv-matrix-40k/{launch.md,result.md,records/}`：`val_seeds.json`、`test_seeds.json`、`param_tree_{official,motion}.json`、`summary.txt`、`per_episode_by_cond.json`、`paired_stats.json`、`xgpu.txt`、批日志。
- 原始产物（不进 git）：`v1-store/reports/motion-variance/`（bank、open_loop、adapter 自检）、`v1-store/evaluation/mv-*/`、`v1-store/logs/mv-*`。
