# 梯度对拍黄金基线 G0 与基线链规约

> **实施状态（2026-08-26）**：P1（commit V2.1 `d9e509e`）与 P2（commit V2.2，四档八轮）已完成；**D2 与 D2-cold 双 PASS，三支处置走第一支**——G0 获准跨期充当 bitwise 判据一侧，正确性族固定确定性档 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`。核心一致性结论独立留档 [`docs/v1-determinism-conclusions.md`](docs/v1-determinism-conclusions.md)，逐轮产物 `docs/training-doc/v1-det-*/`。**PG0 已完成（commit V2.3）**：G0 两轮 300 步自证 PASS（逐位一致，`<G0-HEAD>=624d417`），产物固化 `docs/training-doc/v1-grad-baseline-g0/`；PG0-speed（`v1-g0-speed`）已预跑，speed 链锚点 1.117 s/step。**原计划工序（P1/P2/PG0/PG0-speed）全部执行完毕**，登记簿见 T8；2026-08-26 两份审计后增补量具补遗工序 **P1b——已完成（commitV2.3.1 `570287f`，2026-08-27）**：经用户裁定「允许全部重跑」，原第 5 项补录 replay 轮取消，改为量具落地后**重跑基线两轮（升级 1000 步，`v1-grad-baseline-g0b`）+ 重测速度锚点（`v1-g0-speed-r2`，1000 步）**；G0b 千步自证与新旧前 300 步前缀对拍全 PASS，旧 G0 records 按用户裁定删除，G0b 升任链头（T8）。**dtype 计划（V2.4 起）2026-08-27 早先一度中止，同日经用户「实现」指令恢复执行并已完成两块正确性验收**：工具 **V2.4a `f2e7348`** + 三行修复 **V2.4b `a0f76f8`**；第一块 `COMPARE_DTYPE=PASS`（2,600 样本 / 200 batch / 0 失配）、单步定点梯度 `COMPARE_GRAD=PASS`（3 档 32 叶逐位相同）、第二块 G1 vs G0b **千步 bitwise 全过**（`scalars_hex.tsv` sha256 与 G0b 逐字节相同）。T8 相关行已解除悬置并回填。**P7 `v1-g1-speed` 亦已完成**：步时均值 −7.21%、util 均值 +5.64pp、0% 采样 −3.69pp、慢步 3→0。dtype 计划 P3–P7 全部执行完毕，v4 决策的两份输入齐备。
>
> 本文件立项时为计划文档。2026-08-26 立项，用户指令：三份计划（[`v1-dtype-unify-plan.md`](v1-dtype-unify-plan.md)、[`v1-framesamp-restructure-plan.md`](v1-framesamp-restructure-plan.md)、[`v1-post-restructure-roadmap.md`](v1-post-restructure-roadmap.md)）的梯度对拍不仅要和自己的改动前比，还要和「三个都没动」的训练（当前仓库状态，git 锁定，必要时 sha256 校验）比；最好现在先跑一轮记下产物、产物进 git 后固化复用。方案经一轮 opus 对抗复核（12 条必须修 / 10 条建议修全部吸收）。
>
> **本文档是基线链的唯一权威载体**：G0 的定义、锁定断言、产物清单、失效条件、量化判据参数与登记簿都只在本文档维护一份；三份计划一律引用本文档章节，不复制其中数字（引用锚点用章节名，AGENTS 9）。

## Context（为什么做这件事）

- 三份计划各自的等价性验证都是「vs 自己改动前」的相邻对比。链条拉长后（dtype 修复 → IO 重构 → roadmap 项 1–3），每一环即使各自通过，也缺一个「相对原始训练累计漂移了多少」的直接锚点。G0 就是这个锚点：**一次跑定、产物固化进 git、之后所有改动都能拿它离线对拍，不必反复 checkout 旧 commit 重跑对照侧**。
- 「跑一轮就固化」成立有前提，本计划把前提做成可证伪的闸门：记录仪器补完整（P1）、确定性确立且必须包含**独立重编译可复现**（P2 新增 D2-cold 档）、环境指纹做成机器判定的 preflight（引用产物前强制跑）。
- G0 的语义必须澄清：它是「**受控确定性档位下的当前训练语义**」，不是字面现状——XLA 确定性 flags 本身就会改变位级结果（`scripts/smoke-local/README.md` 现文亦声明「加了 deterministic_ops 就偏离官方口径」）。「字面现状」的样貌与噪声底由 P2 的 D0 档两轮产物一并固化留档，标注「非判据基线」。

---

# 第一部分（给人看）

## 〇、符号总表（两族 run 的权威定义，其余文档只引用不复制；2026-08-26 增订）

全部 run 分两族，**正确性族证明「改动没改变训练结果」，性能族回答「优化值不值得做」，一个 run 不得身兼两职**：

### 正确性族（带 TrainState 摘要 + batch 输入摘要 + 确定性 XLA 档；util/步时仅留档参考，禁作性能结论）

| 符号 | run_name | 是什么 |
|---|---|---|
| D0 / D1 / D2 / D2-cold | `v1-det-*` | 确定性预备实验（P2），各两轮 100 步，证明「同配置重跑两次结果逐位一样」在什么条件下成立 |
| **G0**（round1 正本 + round2 自证） | `v1-grad-baseline-g0` | 三计划均未实施的原始训练语义黄金基线，300 步，产物固化进 git，之后所有改动拿它对拍 |
| G1 | `v1-dtype-ab-post-r1`（B 侧） | dtype 修复后节点，对拍 **G0b**（1000 步；2026-08-27 起 A 侧换代、run_name 按 T7 体例带 `-r1`） |
| G2 | IO 重构计划 C.3 的 packed 侧 | packed IO 后节点，对拍 G1（bitwise）+ G0（对账） |
| G3+ | 届时命名 | roadmap 各项节点，对拍上一节点 + G0 |

### 性能族——speed 链（无 TrainState 摘要、无 batch_digests、**生产 XLA 档**：不注入确定性 flags、autotune 默认开）

| 符号 | run_name | 什么时候跑 | 对比对象 |
|---|---|---|---|
| **G0-speed**（步骤名 PG0-speed） | `v1-g0-speed` →（2026-08-27 起）`v1-g0-speed-r2` | 300 步版随本计划在 G0 之后预跑；P1b 后按用户裁定以 1000 步 `v1-g0-speed-r2` 重测换锚 | speed 链的锚点（现行 = `v1-g0-speed-r2`，1000 步口径；后续 speed run 一律 1000 步与之对比） |
| G1-speed | `v1-g1-speed` | dtype 计划 P7：两块正确性（G1 vs G0b）通过后 | vs **`v1-g0-speed-r2`**（1000 步现行锚点；dtype 修复单独性能效果；v4 决策输入之一） |
| G2-speed | `v1-g2-speed` | IO 重构计划（v4）收官后 | vs `v1-g1-speed` + `v1-g0-speed-r2`（dtype 效果已由 G1-speed 单独体现，本级归因干净） |
| G3+-speed | `v1-g<n>-speed` | roadmap 每项收官后 | vs 上一 speed 节点 + `v1-g0-speed-r2` |

- **G1-speed（2026-08-26 同日反转裁定）**：原裁定「G1 无 speed run（dtype 修复预期影响仅 ~80 ms/step，不值专门一轮；性能效果并入 G2-speed 观察）」，同日经用户再裁定反转——新增 `v1-g1-speed`（dtype 计划 P7，两块正确性通过后跑，口径同下条 speed run 统一口径），vs `v1-g0-speed` 产出 dtype 修复的单独性能对比，作为「是否实施 IO 重构计划（v4）」的用户决策两份输入之一（另一份是 G1 vs G0 正确性验收结论）；副产收益是 G2-speed 不再合并承载 dtype 效果、IO 重构性能归因变干净。（**本段是 2026-08-26 当时的裁定原文，保留作历史记录**；其中的 `v1-g0-speed` 与「G1 vs G0」已于 2026-08-27 分别换代为 `v1-g0-speed-r2` 与「G1 vs G0b」，现行口径以上表两行为准。）
- **speed run 统一口径（权威定义；2026-08-27 订正步数）**：`bench_train_steps.py` 入口、本机 2×RTX 6000 Ada、b8、**1000 步**、seed 42、num_workers=4；（原文写 300 步，与同表锚点行「后续 speed run 一律 1000 步与之对比」自相矛盾——换锚 `v1-g0-speed-r2` 后统一为 1000 步）不注入 `XLA_FLAGS`；`SAVE_INTERVAL=0`（禁 TrainState 摘要）、`BATCH_DIGESTS=0`（禁输入摘要），逐步 loss 标量记录保留（毫秒级）；`nvidia-smi -lms 500` 密集采样 + 15 s legacy 通道；报 util 稳态均值 / 0% 采样占比 / 慢步分层均值 / 步时中位与均值（AGENTS 16，禁中位数标题结论），标注「本机口径，不作最终吞吐结论」（AGENTS 13）。GL 吞吐验收（v4 计划 D 节）照旧是 GL 侧主判据，speed 链是本机口径的链式对账，两者并存。
- 每个 speed run >5 min，按 AGENTS 17 留档 `docs/training-doc/<run_name>/`，留档以 `docs:` commit 提交、不占 V2.x 编号；run_name 起跑前仍按 AGENTS 6 逐个向用户确认。

## 一、G0 是什么，为什么能「跑一次就不再重跑」

- **G0**：三个计划都未实施的训练语义，在受控确定性环境（P2 选定档）下的一轮真实训练。口径：本机 2×RTX 6000 Ada、b8、300 步、seed 42、SAVE_INTERVAL=100（步 0 与末步必记，共 4 次摘要——2026-08-26 由 25 稀疏化：一次完整 TrainState 摘要约 95–140 s，每 25 步记 12 次纯停顿 19–28 min、比训练本体还贵；分歧定位主靠逐步标量 hex，密摘要买不到额外灵敏度），走 `bench_train_steps.py` 入口，并行 `nvidia-smi -lms 500` 采样。
- **跑两轮**（G0 主轮 + G0' 自证轮，同配置重跑）：第二轮提供 300 步尺度的可复现自证（P2 只证到 100 步）、产物交叉校验、以及量化判据的同 HLO null 对；第二轮走编译缓存命中，耗时更短。round1 为正本，round2 为自证。
- **双重身份（仅限正确性对拍）**：G0（2026-08-27 起为 **G0b r1**）兼任 dtype 计划 P6 的 A 侧（修复前），P6 随之只跑 B 侧。**G0 的 util/步时数据不作任何性能结论**——正确性 run 的摘要停顿（一次完整 TrainState 摘要实测 47.3 s/次、扩完整 TrainState 后 2–3×）与确定性 XLA 档都污染性能口径；性能基线由同场次预跑的 **G0-speed** 承担（当时为 `v1-g0-speed`；2026-08-27 起现行锚点为 `v1-g0-speed-r2`，符号总表）。
- **「不再重跑」的三个前提**（缺一即不成立，全部做成硬判定）：
  1. **仪器完整**（P1）：现有 checksum recorder 只哈希 params/EMA，缺 Adam 动量（opt_state）与 step——动量是「两条轨迹是否同一条」最灵敏的累积量，基线跑在扩展之前这一列就永久缺失且补不回来；
  2. **确定性成立**（P2）：尤其 D2-cold 档（独立重编译两次仍 bitwise 一致）——未来与 G0 对拍的 run 计算图都变了、必然现场重编译，无缓存可继承，**D2-cold PASS 是 G0 跨期充当 bitwise 判据一侧的唯一授权闸**；
  3. **环境指纹不变**（preflight）：每次引用 G0 产物前必须先过 `check_baseline_env.py`（`BASELINE_ENV=PASS|FAIL`），任一失效条件触发即基线作废、必须重跑并在登记簿记版本。

## 二、锁定方式（git + 指纹）

- **锚点 commit**：`55e6e5bf8ef38b780902d0e63257ea859a432a2c`（立项时 HEAD，工作区 clean）。G0 起跑 commit（记 `<G0-HEAD>`，实施时回填登记簿）允许晚于锚点——P1/P2 会先落两个 commit——但必须通过下述断言证明训练语义与锚点逐字节相同。
- **G0_SCOPE 断言（反向白名单，判定行 `G0_SCOPE=PASS`）**：`git diff --name-only 55e6e5b <G0-HEAD>` 的**全部**输出必须落在白名单内——`docs/`、`scripts/smoke-local/`、`scripts/data-preprocess-GL/paths.sh`、仓库根目录 `*.md`；任何一条越界即 FAIL、G0 不得起跑。白名单内的脚本改动（P1 所致）须在 launch.md 里逐 hunk 说明为何不改变训练语义。附加断言：起跑时 `git status --porcelain` 为空、`git rev-parse HEAD == <G0-HEAD>`、submodule 指针与锚点一致。（正向枚举 `git diff -- src/` 不完备：`uv.lock`、`paths.sh` 等都在 `src/` 之外却决定训练语义/环境，反向白名单在构造上完备。）
- **git 外指纹（sha256 单列，git sha 覆盖不到）**：`v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`、`pi05_base` 初始权重、paligemma tokenizer 模型、`episode_manifest.json` 顶层 `sha256`、数据集抽样指纹（沿用 IO 重构计划 `source_spot_sha256` 机制，不另发明）。git 跟踪内容由 commit sha 内容寻址锁定，不再额外 sha256。

## 三、产物清单（全文本，进 git `docs/training-doc/v1-grad-baseline-g0/records/round{1,2}/`）

1. `metrics.jsonl`：逐步五标量（loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm）十进制 + hex；
2. `param_checksums.jsonl`：每 100 步（步 0 与末步必记）**完整 TrainState** 摘要（params/opt_state/EMA/step 全部叶子逐个 sha256 + `state_digest`）；
3. `batch_digests.jsonl`：步 0/1/2 + 每 100 步，交付 batch 逐键 `sha256(dtype‖shape‖bytes)`。**性质与输出摘要完全不同**：与 XLA/缓存/驱动无关、跨计算图（HLO）永远逐位可比——roadmap 项 2/3（改输入签名）场景对拍 G0 的**主判据**。**口径限定（2026-08-26 审计修正）**：上式是 **raw 物理口径**（dtype 参与哈希），只适用于「输入应逐字节不变」的场景；**跨 dtype 场景（G1 vs G0）它必然全线失配且无鉴别力**（分不清「只是类型变了」与「数值真变了」），须改用 P1b 增设的 **canonical 数值口径**（逐键升到 f32 后按位视图哈希）。G0 固化产物只含 raw 摘要，sha256 事后不可换算——基线侧 canonical 摘要由 G0b 重跑原生产出（2026-08-27 起链头为 `v1-grad-baseline-g0b`，补录 replica 轮已取消，见七节 P1b）；
4. `scalars_hex.tsv`：`metrics.jsonl` 的规范化投影（`step<TAB>loss.hex<TAB>…`，剔除 wall_time 等易变字段）+ 其 sha256——「两轮是否一致」退化为一次 sha256 比较，人和机器都不会搞错（`metrics.jsonl` 含 wall_time，不可直接 diff）；
5. `env.json`：环境指纹（真实 argv、库版本、GPU/驱动、XLA_FLAGS、编译缓存命中/编译计数——见 T2）；
6. `BASELINE_MANIFEST.json`：逐产物 sha256 / 行数 / schema 版本——防产物腐烂与工具漂移；
7. util 采样原始数据与统计（AGENTS 16 口径：稳态均值、0% 采样占比、慢步分层均值）、launch.md、result.md。**util/步时数据受摘要停顿与确定性档影响，仅作留档参考，禁止作性能结论**（性能口径以符号总表的 speed 链为准）。

## 四、失效条件与 preflight

失效条件实现为 `scripts/smoke-local/check_baseline_env.py` 的机器断言（清单见 T5），**每次引用 G0（或任何登记簿基线）产物前强制跑**；散文清单必被遗忘，机器判定不会。触发任一 → 该基线失效，必须重跑并在登记簿记新版本。要点：

- 库版本（uv.lock 哈希 + 单列 torch/jax/jaxlib/numpy/ml_dtypes——torch 版本决定 `randperm` 的 index 排列，变了则样本序列变、一切失效）；
- GPU 型号 + 驱动 + 拓扑（jax 编译缓存 key 含加速器拓扑序列化；换卡/换驱动位级行为可能变）；
- 二节的全部 git 外指纹；
- XLA 确定性档位（XLA_FLAGS 原文逐字比对）与 JAX 配置快照（`jax_enable_x64`、matmul precision、`XLA_PYTHON_CLIENT_MEM_FRACTION`、device_count/fsdp_devices）；
- **单 epoch 约束**：凡与 G0 对拍的 run 必须 `steps × batch_size < 395,289`（单 epoch 内）。IO 重构计划 1.6 节已证：跨 epoch 边界后 index 序列与 num_workers 相关（torch 既有语义），超出单 epoch 的对拍失去意义。当前口径 300×8=2,400 远在界内，写死此条防将来加步数踩雷。

## 五、三方对拍矩阵与基线链

基线链：**G0（原始；2026-08-27 起链头为 G0b）→ G1（dtype 修复后）→ G2（packed IO）→ G3…（roadmap 各项）**。每个新节点：vs 上一节点（主判据，尽可能 bitwise）+ vs 链头（锚定）。**与之平行的 speed 链**：`v1-g0-speed-r2` → `v1-g1-speed` → `v1-g2-speed` → `v1-g<n>-speed`（符号总表；原「G1 跳过」已于 2026-08-26 同日反转）——每个基线链节点收官后跑对应 speed run，与上一 speed 节点及现行锚点 `v1-g0-speed-r2` 对比，作为该级优化的性能结论与下一级立项输入。

| 对拍 | 判据 | 说明 |
|---|---|---|
| G1 vs G0b | bitwise 主判据 + 量化兜底（六节） | dtype 计划 P6：A 侧=**G0b r1** 固化产物（1000 步，不重跑），只跑 B 侧；B 侧应尽快接续 G0（理想同场次），起跑前必过 preflight。HLO 因输入 dtype 改变而不同，bitwise 存在虚假失败可能，兜底见六节。**输入侧对拍禁用 raw `batch_digests`**（dtype 变更使其必然失配）——用 P1b canonical 口径 + 全步 index 序列一致 |
| G2 vs G1 | 同 clean HEAD、同 HLO、共用编译缓存的 bitwise | IO 重构计划 C.3 原判据不变——链条中最强的一节 |
| G2 vs G0 | **对账，非独立判据** | 若 G2 vs G1 bitwise PASS，则 G2 轨迹与 G1 逐位相同，G2 vs G0 数学上恒等于 G1 vs G0——不是独立证据。判定标准：用 git 里 G0/G1 固化产物重算的报告须与 dtype 计划验收留档的 G1 vs G0 报告**逐字节相同**；它检出的是产物腐烂/对比工具漂移/留档记错，不是链路问题。若 G2 vs G1 未达 bitwise（按 C.3 属必须修复的失败），此项自动失去意义，**不得用它「曲线救国」放行** |
| G3+（roadmap 各项）vs 上一基线 + vs G0 | **主判据 = 输入侧逐位对拍**（口径按场景选 raw / canonical，跨 HLO 有效）+ 量化复核（六节）+ 单步 fixture 回归 | 项 2/3 改输入签名，输出侧 bitwise 天然不可得；例：项 2 把 pos 挪到 GPU 侧生成后，把设备端 gather 出的 pos 张量与 G0 的 `static_pos_emb` 摘要对拍即是逐位判据，比 300 步量化统计硬得多。**前置缺口（2026-08-26 审计修正）**：现量具只在 collate 后、device_put 前记录 host batch，设备端张量不在任何记录点，且项 2 落地后 `static_pos_emb` 不再出现在 host batch——该对拍成立的前提是先落地 roadmap 项 2 前置的「设备端观测点」（gather/清零/展开后、进投影层前 `device_get` 回 host 哈希，schema 版本化），见 roadmap 该节 |

- **粗差/细差分工（不得拿「跑完了」当全覆盖）**：1000 步 × b8 = 8,000 样本（G0b/G1 口径；旧 300 步为 2,400）只覆盖粗差（行号错位、选帧错误、dtype 交付错误会在头几十步撞穿任何阈值）；「万分之一错帧」类细差的覆盖责任在各计划第 1 层定点样本对拍（dtype 计划 T3 约 2,600 / IO 重构计划 C.2 约 8,200）与全量 verify。
- **单步 fixture 常规回归闸**：dtype 计划 T4 的单步定点梯度对拍升格为可复用 fixture——固定初始 state + 固定 batch（按 dtype 计划 T3 的位型容器格式存 `v1-store/fixtures/`——npy/npz 会丢 bf16 类型、禁用，逐文件 sha256 摘要进 git）。后续任何 commit 花约 2 分钟即可重锚 G0，替代 1.5 h 的轨迹重跑，适合当常规回归闸。
- **revert 链形态**：若 dtype 修复被 revert（其失败处置），G1 不存在、IO 重构计划退回 v3 形态，链条变为 G0 → G2'（v3 形态）；G0 保持链头不变，登记簿如实记录。

## 六、量化判据（等价性检验形态，权威版本）

> 本节替换 dtype 计划 T4 原「rel 三档先验阈值 + OLS 趋势」判据（原 OLS「β>0 且 p≤0.05 即 FAIL」已删除：rel 序列强自相关使 p 值失标定，且混沌轨迹下任何扰动都 β>0，无鉴别力）。适用场景：跨 HLO 对拍的兜底评估；同 HLO A/B（如 C.3）不适用——必须 bitwise。

- `rel(a,b) = |a−b| / max(|a|,|b|,1e-8)` 逐步计算，对五个标量分别统计 median / p95 / max。
- **null 对（噪声底）选取，与被比场景同构优先**：跨 HLO 场景 → D2-cold 两轮（编译期噪声）；同 HLO 场景 → G0 vs G0'（D2-cold PASS 时应逐位为零，null 退化）；均不可得时 → D0 两轮（现状 autotune 噪声，注明是上界）。
- **判据**：A/B 的 rel 各统计档 ≤ null 对相应档 × 2（余量）；**下限守卫**：null 档位低于绝对下限（loss 1e-6 / 三个梯度范数 1e-5 / 末步 param_norm 1e-5）时以绝对下限为准——比天然噪声还小的差异不必吹毛求疵。（阈值低于噪声底必误报、远高于噪声底无鉴别力，故必须经验标定而非先验拍数。）
- **趋势判据**：主用包络——A/B 的 rel(t) 逐步不超过 null 对 rel(t) 上包络 × 2；可选辅助诊断 `log(rel)` 斜率与 null 斜率对比（差异增长是乘性的），仅作定位参考、不单独作 FAIL 依据。
- 量化判据在 IO 重构计划 C.3 场景仍**不作为放行依据**（该计划 3.4 既有裁定不变：无 dtype 差异的 A/B 出现任何残差都指向 bug 或非确定性，必须修到 bitwise）。
- **TrainState 数值裁决（2026-08-26 审计修正）**：`state_digest` 失配时，仅凭五个标量的量化统计**不足以判 PASS**——参数、Adam 动量或 EMA 明显不同但范数接近时会漏判。失配后必须给出**逐叶数值统计**（每叶 max-abs、max-rel、L2 相对差、cosine，覆盖 params / opt_state / EMA 全部叶子），由 `compare_baseline.py` 计算（P1b 扩展）。固化产物只含 sha256、事后补算不出数值——与固化基线对拍时，数值参照取 G0b r1 落盘的摘要步 TrainState 数组（步 0/299/999，存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`，sha 清单进 git；失配步不在所存步时按 D2-cold 授权重放补落）；拿不出逐叶统计时判定只能写 `INCONCLUSIVE`，不得写 PASS。

## 七、执行序列（本文档实施时执行；预计 4.5–5.5 h 墙钟——摘要稀疏化省约 2–3 h、新增 PG0-speed 约 +40 min；全程 tmux + Monitor）

总逻辑：G0 要「一次跑定、以后所有改动拿它当参照」，前提有二——**P1 把记录仪器补完整**（仪器缺项，基线就有永久盲区；仪器说谎，指纹就没法比）、**P2 证明同配置重跑两次结果完全一样**并找出成立条件（否则以后任何对拍差异都分不清是「改动引起」还是「重跑噪声」）。之后才轮到 G0 本体。

### P1（commit V2.1，bench 驱动改造）——G0 由 `run_2gpu_epoch_bench.sh` + `bench_train_steps.py` 驱动，它们就是量具

含 dtype 计划 T2/S0 原有四项（EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE + 缓存软链进 `v1-store/cache/jax/`、XLA_FLAGS 外部注入、checksum recorder 扩展完整 TrainState），本计划扩展六项（逐项作用）：

1. **`batch_digests` 输入摘要记录**：其余产物全是输出（loss/梯度/参数），输出把「输入变了」与「计算变了」混在一起，且计算图一变就整体失比；输入侧逐键 sha256 与 XLA/缓存/驱动无关、跨 HLO 永远逐位可比。记录点：bench 入口 monkeypatch 交付层（与既有 `_WandbProxy`/`checksum_state` 同一 patch 风格），对 collate 后、device_put 前的 host 侧 batch 逐键 `sha256(dtype‖shape‖bytes)`，步 0/1/2 + 每 SAVE_INTERVAL。
2. **env.json 改记真实 argv**：现在 heredoc 里 `"seed": 42`、`"fsdp_devices": 2` 是字面量而非取自实际命令；参数化后它会开始说谎，而它正是 preflight 比对的指纹。改为 dump 真实 argv 数组（单一真值源）。
3. **编译缓存命中/编译计数进 env.json**：注册 jax `/jax/compilation_cache/cache_hits`、`/jax/compilation_cache/compile_requests_use_cache` monitoring listener 计数（可辅以 `JAX_EXPLAIN_CACHE_MISSES=1`）——「这轮是热缓存还是冷编译」从口头猜测变成留档事实，缓存冷热直接影响位级可复现性。
4. **`--checkpoint-base-dir` 按 RUN_TAG 分目录**：A/B 共用 EXP_NAME（为共享 per-fusion autotune 缓存，见 dtype 计划 T4 修订）后 run 目录撞名，`initialize_checkpoint_dir` 遇已存在目录直接 `FileExistsError` 拒跑；按轮次分目录，不靠「跑完删目录」避让（崩溃残留会卡死第二轮）。
5. **两个新脚本**：`check_baseline_env.py`（preflight，断言项见 T5）；`compare_baseline.py`（对拍工具与基线同 commit 固化，防一年后工具漂移、口径变了；功能见 T2）。
6. **`scripts/smoke-local/README.md` 同步**：现文「本基准未额外加 `--xla_gpu_deterministic_ops`」在 P1/P2 后变假，不改即错误文档；P2 收官把 D0–D2-cold 结论一并写入。
7. **性能口径开关**：`SAVE_INTERVAL=0` 禁 TrainState 摘要、`BATCH_DIGESTS=0` 禁输入摘要（默认值保持现行为）——供 speed 链全部 run（`v1-g0-speed` 起）使用（正确性与性能分跑，口径以符号总表为权威）。

P1 验收：STEPS=3 连跑两次不拒跑、缓存落 `v1-store/cache/jax/` 且未被删、argv 如实进 env.json、`batch_digests.jsonl` 落盘；另加一条 autotune 共享实证——A 轮后记 `xla_gpu_per_fusion_autotune_cache_dir` 条目数、B 轮后复查复用（把「共用 EXP_NAME 即共享 autotune」从机制推断变实测）。

### P2（commit V2.2，确定性实验，四档各两轮 100 步、SAVE_INTERVAL=50——步 0 与末步必记，每轮 3 次摘要）

每档判定行相同：两轮逐步标量 hex + 全部 `state_digest` diff 为空。

| 档 | 设置 | 证明什么 |
|---|---|---|
| D0 | 现状：删缓存、无 flags | 预期 FAIL——「字面现状」存档：既记录三个都没动的训练长什么样，也测出其自身重跑噪声（六节的 null 上界）。**两轮产物固化留档**（`docs/training-doc/v1-det-d0-r{1,2}/`），标注「非判据基线，只作噪声底与口径对照」 |
| D1 | 两轮共用编译缓存 | 「给定同一可执行体，运行期可复现」（排除 atomics/dataloader/RNG 漂移） |
| D2 | D1 + `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` | flags 带来的确定性（仍共用缓存——**注意它没测独立编译**） |
| D2-cold | 同 D2 flags，两轮**各用全新空缓存目录**（EXP_NAME 各异） | 强制独立编译两次，证「**编译两次得到同样行为**」——未来 G1/G2/G3 计算图都变、必然现场重编译无缓存可继承，这是「G0 固化后跨期复用」唯一依赖、而 D1/D2 测不到的性质。成本 +2 轮 ≈ 40 min，一次性 |

**三支处置（按结果写死，不留「看情况」）**：

| P2 结果 | 处置 |
|---|---|
| D2-cold PASS | G0 可跨期充当 bitwise 判据一侧；G0 缓存目录允许清理（留 sha256 清单作证据）；G1/G2/G3 vs G0 主判据 bitwise |
| D2-cold FAIL、共享缓存档（D1 或 D2）PASS | G0 仍固化，但**跨 HLO 对拍降级**：主判据改为 `batch_digests` 输入摘要 bitwise + 量化判据（null 以 D0 两轮标定）；同场次同 HLO A/B（如 C.3 legacy vs packed）仍走 bitwise。**不得以「保留缓存」为由声称跨期 bitwise 可得**——jax 编译缓存 key 含 HLO 与拓扑，跨 commit 模块级缓存必 miss，保留无用 |
| 全档 FAIL | 停：按 dtype 计划 T2 既有路径排查（加 exclude flag、降 50 步二分），G0 暂不固化 |

**P2 实测结果（2026-08-26，commit V2.1 起跑，权威结论见 [`docs/v1-determinism-conclusions.md`](docs/v1-determinism-conclusions.md)）**：D0 FAIL（步 0 起全分歧，loss rel 噪声底 median 2.7e-3 / max 4.6e-2，作六节 null 上界）；D1 FAIL 但仅 2 步 `llm_grad_norm` 差 ULP、分歧定位 embedder scatter-add atomics；**D2 PASS、D2-cold PASS**（独立冷编译两次逐位一致，交叉对拍亦 PASS）——**三支处置走第一支**。八轮输入摘要全一致：同 seed 下 dataloader 交付逐位确定。

### PG0（commit V2.3，G0 两轮 + 产物固化）与 PG0-speed（速度基线预跑）

1. 从 clean HEAD 起跑前先过 `G0_SCOPE` 断言（二节）；run_name `v1-grad-baseline-g0`（用户已确认；实施起跑前仍按 AGENTS 6 再次确认）。
2. round1（正本）→ round2（自证轮，同配置重跑）；round1 vs round2 按 P2 结果判定：D2-cold PASS 环境下应逐位一致，否则记录残差并入六节 null 对。
3. 产物按三节清单固化，`BASELINE_MANIFEST.json` 校验通过后逐文件 `git add` 提交；登记簿（T8）回填 `<G0-HEAD>`、指纹 sha、判定结论。
4. util/步时统计写入 result.md，标注「本机口径，不作最终吞吐结论」（AGENTS 13/16）且**仅作留档参考**——受摘要停顿与确定性档污染，性能结论一律以 speed 链为准。
5. **PG0-speed（速度基线预跑，随本计划收官完成）**：G0 两轮之后同场次跑 `v1-g0-speed` 一轮（口径以符号总表为权威：生产 XLA 档、`SAVE_INTERVAL=0`、`BATCH_DIGESTS=0`、300 步 b8），留档 `docs/training-doc/v1-g0-speed/`——它是 speed 链锚点，v4 与 roadmap 各项收官的 speed run 都与它对比。约 40 min。

### P1b（量具补遗，2026-08-26 审计修正增补；G1——dtype 计划 P6——起跑前必须完成）

P1 固化的量具经审计发现四处缺口，在 G0 已固化的前提下按下列顺序补齐（P1b 属量具改动，独立功能 commit 落地，完成后重跑 P1 验收口径的 STEPS=3 双烟测）：

1. **canonical 输入摘要通道**：`batch_digests` 每键在 raw 字段外增记 canonical 字段（升到 f32 后按位视图哈希，schema 版本号随之升级；raw 字段保留），并记录全步 index 序列摘要——跨 dtype 对拍（G1 vs G0）的输入侧判据由此可得（三节口径限定）。
2. **逐叶数值统计**：`compare_baseline.py` 在 digest 失配时输出逐叶 max-abs / max-rel / L2 / cosine（params/opt_state/EMA 全覆盖）；无数组可算时输出 `INCONCLUSIVE`（六节口径）。
3. **runner 收敛 `uv run`**：bench 驱动的全部 Python 调用改为 `UV_LINK_MODE=copy uv run`（同一 `.venv` 解释器，计算行为预期不变；八节差异表 runner 行随之收敛）。此为 AGENTS 3 合规修正，经用户 2026-08-26 拍板；发生在 G0 固化之后，故补录轮（下条）同时充当 runner 改造的等价性实证。
4. **speed 口径联动**：驱动脚本在 `SAVE_INTERVAL=0` 时默认联动 `BATCH_DIGESTS=0`（保留显式覆盖）——speed 链口径要求两者都关，此前须调用方手动同时设置，易漏。
4b. **README 同步**：`scripts/smoke-local/README.md` 可调变量清单补 `EXP_NAME/RUN_TAG/KEEP_JAX_CACHE/BATCH_DIGESTS` 四项，步数上限「≤500」订正为与 `bench_train_steps.py` 护栏一致的「≤600」（两处，含差异表）——P1 落地时漏同步。
5. ~~**G0 摘要补录 replica 轮**~~（**取消，2026-08-26 用户裁定「允许全部重跑」后由 G0b 重跑取代**）：改为 P1b 代码落地后直接重跑基线两轮 `v1-grad-baseline-g0b-r{1,2}`（**升级 1000 步**，同用户裁定；SAVE_INTERVAL=100 + 附加摘要步 299 对齐旧末步；r1 另落 TrainState 数组 @0/299/999）+ 重测速度锚点 `v1-g0-speed-r2`（1000 步）。判定三条：① 新 r1 vs r2 千步逐位自证；② 新 r1 前 300 步 vs 旧固化 G0 raw 口径逐位前缀对拍（= uv runner 与 canonical 通道不改变计算的等价性实证；FAIL 即 revert P1b 排查）；③ manifest 校验。**已执行，全 PASS（2026-08-27，T8）**：canonical 摘要与 index 序列由 G0b 原生产出；TrainState 数组经逐文件 sha256 核对迁存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（用户裁定不留 NFS，sha 清单进 git）；旧 G0 records 对拍通过后删除。

P1b 实际落地记录（2026-08-27）：功能 commit **V2.3.1 `570287f`**（上列 1–4、4b，另含配合 1000 步的护栏放宽 600→1200、`EXTRA_DIGEST_STEPS` 附加摘要步与 `STATE_DUMP_STEPS` TrainState 数组落盘机制）；STEPS=6 三连烟测 + `compare_baseline.py` 四场景构造样例自检全过；G0b 两轮与 `v1-g0-speed-r2` 留档 `docs/training-doc/v1-grad-baseline-g0b/`、`docs/training-doc/v1-g0-speed-r2/`。

后续（不在本计划）：dtype 修复本体自 commit V2.4 起（其 P3–P7，见该计划 T5；P7 即 `v1-g1-speed`——原「dtype 不跑 speed run」裁定已于 2026-08-26 同日反转，见符号总表）。

## 八、G0 与正式训练入口的入口层差异（有效域声明，2026-08-26 增补）

正式训练入口是 `scripts/finetune_mme_vla_suite.sh` → `scripts/train.py`，G0 走 `scripts/smoke-local/run_2gpu_epoch_bench.sh` → `scripts/smoke-local/bench_train_steps.py`。本节只隔离比较**入口包装本身**：双方均使用同一个 `perceptual-framesamp-context`，dataset、初始权重、batch、硬件、seed、训练超参和 XLA 配置完全相同；这些共同条件不在本节重复比较。

### 8.1 比较前提——共享同一条核心训练路径

两种入口最终都调用同一个 `train.main(config)`，并共享 `init_train_state()`、dataloader 创建与迭代、`train_step()`、loss、optimizer 和 EMA 更新。因而在解析后的 config、初始 TrainState、输入 batch 与 XLA executable 完全相同时，入口包装本身不应改变 loss、梯度或参数更新；下面只比较顶层控制流、观测行为、保存行为与运行时副作用。

### 8.2 入口差异表

| 维度 | 正式入口 | G0 bench 入口 | 影响 |
|---|---|---|---|
| 顶层调用 | `scripts/train.py` 的 `__main__` | `bench_train_steps.py::main()` | 只改变入口包装，核心训练函数同源 |
| `main()` 次数 | tentative + 正式，共两次 | 单次 | 两边实际训练都调用同一个 `train.main(config)` |
| tentative 更新 | tentative 循环临时执行 step 0–11，共 12 个 optimizer step | 无 tentative 轮 | tentative 的 TrainState 随调用结束被丢弃，不进入随后重新初始化的正式参数轨迹 |
| checkpoint 目录 | tentative 已创建目录；第二次 `main()` 在默认 `overwrite=False`、`resume=False` 下触发 `FileExistsError` | 单次初始化目录，能够进入并完成受控短轨迹 | 这是正式入口的可运行性阻断，不是模型计算或训练公式差异 |
| 配置解析 | 每次调用 `_config.cli()` | 调用同一个 `_config.cli()` | 同 argv 时解析结果应一致；入口差异不改变 config 含义 |
| 初始化与训练循环 | `train.main(config)` | 同一个 `train.main(config)` | 同初始状态、batch 和 XLA executable 时，loss、梯度和参数更新预期一致 |
| 编译预热 | tentative 提前完成训练图编译与部分 autotune | 正式轨迹内首次编译 | 会改变缓存命中、启动时间；默认 autotune 下还可能影响 kernel 选择，从而使结果不再 bitwise |
| wandb | 正常初始化并上报 | disabled，并由 `_WandbProxy` 记录指标 | 不进入更新公式；影响网络、日志、墙钟与 wandb 失败面 |
| 日志频率 | `log_interval=100`，写区间聚合均值 | `log_interval=1`，写逐步值 | 不改变 TrainState；bench 的 host 同步更频繁，且两份日志不能直接逐行比较 |
| checkpoint 行为 | 调用真实 `_checkpoints.save_state()` | monkeypatch 为 checksum recorder | checksum 不修改当前 TrainState，但 G0 不验证真实 checkpoint 序列化、异步完成或 resume |
| checksum 同步 | 无高频完整状态拉回 | G0 计划每 100 步（步 0 与末步必记）执行大规模 `device_get` | 显著增加墙钟并扰动吞吐、预取与 GPU 利用率，不改变训练更新公式 |
| checksum 实测开销 | 无对应高频开销 | 既有 300 步 bench 前身共 12 次，单次中位约 47.3 s，合计约 9.4 min | 占该轮 15.3 min 总墙钟约六成；这是验证观测开销，不是训练数值差距 |
| 步数限制 | 无 bench 上限，按正式配置运行 | 入口护栏最多 600 步，G0 计划跑 300 步 | 前 300 步的训练定义不变，但 G0 不覆盖后续轨迹、正式保存点或长程行为 |
| runner / cwd | `uv run scripts/train.py`，依赖从仓库根启动 | P1b 前：固定项目解释器；P1b 起：同为 `UV_LINK_MODE=copy uv run`（同一 `.venv` 解释器），并在调用前显式切到仓库根 | 环境版本相同时不产生数值差异（G0b 重跑 vs 旧 G0 前 300 步 raw 口径逐位前缀对拍实证）；正式入口对调用环境和 cwd 更敏感 |
| 最终训练数值差距 | 当前默认入口因目录冲突不能形成完整正式轨迹 | 能够完成受控短轨迹 | 修复正式入口后，同条件下入口本身的**预期数值差距为 0**；仍须用入口-only A/B 实测确认 |

> “预期数值差距为 0”是由两种入口共享同一核心源码路径得出的结论；当前尚未完成“只切换入口、其余条件完全相同”的直接 A/B，因此不得写成已经实测 bitwise 一致。

### 8.3 边界结论

1. 入口包装本身不改变模型计算、loss、optimizer 或 EMA 更新公式；相同条件下的核心训练轨迹预期一致。
2. 当前正式入口的双 `main()` checkpoint 目录冲突是可运行性阻断，G0 的单次入口绕开了该问题，但没有修复或验证正式 wrapper。
3. tentative 预热、编译缓存和 autotune 可能改变 kernel 选择与位级结果，但不改变训练目标；未做入口-only A/B 前不能宣称 bitwise 一致。
4. wandb、逐步日志和 checksum 主要改变同步频率、墙钟、吞吐与失败面，不应被解释为 loss/梯度公式差异。
5. G0 不验证真实 checkpoint/save-resume 链路；证明入口完全等价仍需在相同 config、固定 batch 和相同 XLA/cache 条件下，仅切换入口，对拍逐步指标与完整 TrainState。

---

# 第二部分（技术细节，供 agent 追踪）

## T1 G0_SCOPE 断言实现

- 白名单 regex：`^(docs/|scripts/smoke-local/|scripts/dtype-unify/|scripts/data-preprocess-GL/paths\.sh$|[^/]+\.md$)`。
- `scripts/dtype-unify/` 于 2026-08-27 加入白名单（commitV2.4a `f2e7348`）：该目录是纯离线验证资产，不被训练进程 import、不参与训练语义——与已在白名单内的 `scripts/smoke-local/`（它本身就是 bench 训练入口）相比更无影响面。
- 判定命令（进 launch.md，含原始输出）：`git diff --name-only 55e6e5bf8ef38b780902d0e63257ea859a432a2c HEAD | grep -Ev '<白名单>'` 输出为空即 PASS。
- 附加：`git status --porcelain` 为空；`git submodule status` 指针与锚点一致；env.json `git_dirty == false`。

## T2 新脚本职责

- `scripts/smoke-local/check_baseline_env.py`：读取目标基线的 `env.json` 与 `BASELINE_MANIFEST.json`，对 T5 清单逐项断言当前环境一致；输出单行 `BASELINE_ENV=PASS|FAIL`（FAIL 非零退出 + 逐项差异清单）。
- `scripts/smoke-local/compare_baseline.py`：输入两份 records 目录（固化产物或在跑 run 的 records），产出：逐步标量 hex 列 diff、`state_digest` diff、`batch_digests` 逐键 diff（raw 与 canonical 双口径，P1b 起）、rel 分布（median/p95/max）与包络对比报告（六节口径）；digest 失配时输出逐叶数值统计（max-abs/max-rel/L2/cosine，P1b 扩展；无数组可算即 `INCONCLUSIVE`）；先校验双方 `BASELINE_MANIFEST.json`（产物 sha256 不符即 fail-loud）。

## T3 P2 四档参数

- 全档：本机 2 卡、b8、100 步、SAVE_INTERVAL=50（步 0 与末步必记）、seed 42；run_name `v1-det-d{0,1,2}-r{1,2}` + `v1-det-d2cold-r{1,2}`；>5 min 一律 `docs/training-doc/<run_name>/` 留档（AGENTS 17）。
- D2-cold 的两轮 EXP_NAME 各异（如 `det-d2cold-a` / `det-d2cold-b`），各自空缓存目录，flags 与 D2 逐字相同。
- 判定行示例：`DET_CHECK=PASS tier=d2cold steps=100 scalar_hex_diff=0 state_digest_diff=0`。

## T4 G0 run 参数

- 入口与口径：`bench_train_steps.py`、2×RTX 6000 Ada、b8、300 步、seed 42、SAVE_INTERVAL=100（步 0 与末步必记）、P2 选定确定性档的 XLA_FLAGS、`nvidia-smi -lms 500` 并行采样（另保留 15 s legacy 通道对照，AGENTS 16）。
- PG0-speed（`v1-g0-speed`）：同机同 b8 同 seed，300 步；生产 XLA 档（不注入 `XLA_FLAGS`）、`SAVE_INTERVAL=0`、`BATCH_DIGESTS=0`；EXP_NAME 独立（不与确定性档共享编译缓存）；留档 `docs/training-doc/v1-g0-speed/`。
- ~~EXP_NAME 与 dtype 计划 B 侧共用（per-fusion autotune 共享）~~——**2026-08-27 撤销**：G0b 的编译缓存已随 `c6830e0` 清理（「G0b 缓存原为 G1 共享而保留，改为即时清理留证」），共用名只会拿到空目录 + 冷编译；且确定性档 `--xla_gpu_autotune_level=0` 本就关闭 autotune，共享 autotune 结论的动机不成立。G1 改用独立 EXP_NAME。RUN_TAG 仍区分各轮。
- 留档：`docs/training-doc/v1-grad-baseline-g0/{launch.md,result.md,records/round{1,2}/}`。

## T5 preflight 断言项清单（`check_baseline_env.py`）

1. `uv.lock` 文件 sha256；单列版本断言：torch、jax、jaxlib、numpy、ml_dtypes（`importlib.metadata` 现场取）；
2. GPU：`nvidia-smi --query-gpu=name,driver_version` 与 `jax.devices()` 数量/型号；CUDA_VISIBLE_DEVICES；
3. git 外指纹：`norm_stats.json`、`pi05_base/params`（目录树逐文件或抽样 sha256，方式在实施时定一次写进 env.json schema）、tokenizer 模型文件、`episode_manifest.json` 顶层 `sha256` 字段、数据集 `source_spot_sha256`（16 抽样，复用 IO 重构计划机制）；
4. XLA_FLAGS 原文逐字比对；JAX 配置：`jax_enable_x64`、`jax_default_matmul_precision`、`XLA_PYTHON_CLIENT_MEM_FRACTION`、fsdp_devices；
5. 对拍 run 的 `steps × batch_size < 395,289` 单 epoch 约束；
6. 引用产物完整性：目标基线 `BASELINE_MANIFEST.json` 全部条目 sha256 复验。

## T6 量化判据参数（六节的机读版）

- `rel(a,b)=|a−b|/max(|a|,|b|,1e-8)`；统计档 median/p95/max；余量系数 2×。
- 绝对下限（下限守卫）：loss 1e-6；grad_norm/llm_grad_norm/mem_enc_norm 1e-5；末步 param_norm 1e-5。
- null 对优先级：D2-cold 两轮 →（同 HLO 场景）G0/G0' → D0 两轮（标注上界）。
- 包络：`rel_AB(t) ≤ 2 × max_null_envelope(t)` 逐步；null 逐位为零时退化为下限守卫。
- 输出判定行：`QUANT_EQUIV=PASS|FAIL scalars=5 null=<pair> margin=2.0`。

## T7 commit 切分与 run_name 汇总

- V2.1 = P1（bench 驱动改造 + 两个新脚本 + README 同步）；V2.2 = P2（四档八轮 + D0 固化留档 + 三支处置结论）；V2.3 = PG0（G0 两轮留档 + 产物固化 + 登记簿回填）；**P1b（审计修正增补）单独占一个功能 commit——已落地为 V2.3.1 `570287f`**（量具补遗；G0b/speed-r2 重跑留档随后以 `docs:` 提交），排在 V2.4（dtype 修复）之前。dtype 修复顺延 **V2.4**（原 V2.3），IO 重构计划自 **V2.5** 顺延（原 V2.4–V2.8 → V2.5–V2.9）。
- run_name 全表（起跑前逐个交用户确认，AGENTS 6）：`v1-det-d{0,1,2}-r{1,2}`、`v1-det-d2cold-r{1,2}`、`v1-grad-baseline-g0`（已被 G0b 取代）、`v1-g0-speed`（原 300 步锚点，已被 r2 取代）、`v1-grad-baseline-g0b-r{1,2}` 与 `v1-g0-speed-r2`（P1b 重跑，1000 步，取代已取消的补录轮 `v1-grad-baseline-g0-replay`）；dtype B 侧沿用其计划的 `v1-dtype-ab-post`（`v1-dtype-ab-pre` 已由 G0 兼任、不再单独跑）；`v1-g1-speed`（dtype 计划 P7，2026-08-26 反转裁定新增）。后续 speed 链 run_name 按 `v1-g<n>-speed` 命名（符号总表）。
- **run_name 轮次规约（2026-08-26 审计修正）**：G0 两轮共用 `v1-grad-baseline-g0`、以 RUN_TAG 分轮，与 AGENTS 6「每次正式 run 全新 run_name」存在语义歧义——该 run 已固化留档，属既成事实，**不追溯改名**；自本条起，同一符号需多轮正式 run 的场景，run_name 一律带 `-r<N>`（或语义后缀如 `-replay`）逐轮确认，P2 的 `v1-det-*-r{1,2}` 即该体例。

## T8 基线链登记簿（唯一权威；实施时回填，三份计划只引用本表）

| 链节 | run_name | commit | env 指纹 sha | 判据 | 结论 | 产物路径 |
|---|---|---|---|---|---|---|
| D0（字面现状，非判据） | `v1-det-d0-r{1,2}` | `d9e509e`（V2.1） | 各轮 env.json `fingerprint` 键（uv.lock `02cbc3ba…`） | 两轮重跑噪声底 | FAIL（预期）：loss rel median 2.7e-3 / max 4.6e-2，全表见 `docs/v1-determinism-conclusions.md` 三节 | `docs/training-doc/v1-det-d0-r{1,2}/` |
| D1/D2/D2-cold（定档实验） | `v1-det-d{1,2}-r{1,2}`、`v1-det-d2cold-r{1,2}` | `d9e509e`（V2.1） | 同上 | 两轮逐步 hex + state_digest + batch_digest diff 为空 | D1 FAIL（ULP 级，atomics）；D2 PASS；**D2-cold PASS（授权闸开）** | `docs/training-doc/v1-det-*/` |
| G0（300 步旧版，已被 G0b 取代） | `v1-grad-baseline-g0` | `624d417` | scalars_hex sha256 `5da1a1c6…`（两轮相同） | 同下（历史） | PASS（300 步两轮逐位一致）；**records 已于 2026-08-27 删除**（新旧前缀对拍通过后，用户裁定），launch/result.md 留存证 | `docs/training-doc/v1-grad-baseline-g0/`（仅 md） |
| **G0b（现行链头）** | `v1-grad-baseline-g0b-r{1,2}` | `570287f`（`<G0b-HEAD>`=V2.3.1，锚点差异全过白名单） | r1/r2 env.json `fingerprint` 键；scalars_hex sha256 `c799a0b2…`（两轮相同） | G0_SCOPE=PASS + preflight（vs 旧 G0）PASS + r1/r2 千步自证 + r1 前 300 步 vs 旧 G0 前缀对拍 | **PASS**：1000 步标量 hex / 12×state_digest / 14×batch_digest（raw+canonical）/ index 8072 全逐位一致；前缀对拍逐位一致（P1b 量具等价性实证）；TrainState 数组 @0/299/999 迁存 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（sha 清单进 git）；缓存已清理留 sha256 清单 | `docs/training-doc/v1-grad-baseline-g0b/` |
| **G1（dtype 修复后）** | `v1-dtype-ab-post-r1` | `a0f76f8`（V2.4b） | records/env.json `fingerprint` 键；起跑前 `BASELINE_ENV=PASS`（vs G0b r1） | vs G0b r1：五标量 hex + 12×state_digest bitwise；输入侧 canonical + index 序列（**raw batch_digests 按三节口径不计入**） | **PASS**：1000 步五标量 hex 零失配（rel median/p95/max 全为 0）、12×`state_digest` 零失配、`CANON_CHECK=PASS steps=14`、`INDEX_SEQ=PASS n=8072`；**`scalars_hex.tsv` sha256 `c799a0b2…` 与 G0b r1/r2 逐字节相同**。`compare_baseline.py` 总判定行为 `DET_CHECK=FAIL`，唯一成因是 raw `batch_digest` 4 处失配（步 100/299/400/999，每步仅 `static_image_emb` 与 `static_pos_emb` 两键，同步 canonical 均一致）——即工具总判定未区分 raw/canonical 口径，非判据不过；量化兜底未启用（前置是 bitwise 失败，本轮 rel 恰为 0） | `docs/training-doc/v1-dtype-ab-post-r1/` |
| G2（packed） | IO 重构计划 C.3 的 packed 侧 | 待回填 | 待回填 | vs G1 bitwise；vs G0 对账 | 待回填 | IO 重构计划留档 |
| G0-speed（300 步旧锚，已被 r2 取代） | `v1-g0-speed` | `624d417` | records/env.json `fingerprint` 键 | 历史锚点 | 稳态 1.117 s/step（中位，n=249）、util 均值 86.3%、0% 采样 5.3%、epoch 外推 15.33 h（留档保存，不再作对比对象） | `docs/training-doc/v1-g0-speed/` |
| **G0-speed-r2（现行锚点）** | `v1-g0-speed-r2` | `570287f` | records/env.json `fingerprint` 键 | speed 链锚点（AGENTS 16 稳态统计，1000 步口径） | 稳态中位 **1.152 s/step**（n=949，p10 1.097/p90 1.276）、均值 1.186、util 均值 86.5%、0% 采样 4.9%、慢步 3（分层 1.959 vs 1.184 s）、epoch 外推 15.82 h（本机口径，非最终吞吐结论；vs 旧锚 +3.1%，主因 1000 步窗稀释 page cache 乐观偏差） | `docs/training-doc/v1-g0-speed-r2/` |
| **G1-speed** | `v1-g1-speed` | `d227931`（训练语义即 V2.4b `a0f76f8`） | records/env.json `fingerprint` 键 | vs `v1-g0-speed-r2`（1000 步口径，两侧同法重算） | **PASS**：步时均值 **1.1003 s（−7.21%）**、util 均值 **92.10%（+5.64pp）**、0% 采样占比 **1.20%（−3.69pp）**、慢步 0（锚点 3）、epoch 外推 15.10 h（均值口径，本机口径非最终吞吐结论）；绝对收益 85.5 ms/step，与计划按 collate+device_put 分解预估的约 80 ms/step 吻合 | `docs/training-doc/v1-g1-speed/` |
| G2-speed | `v1-g2-speed` | 待回填 | 待回填 | vs `v1-g1-speed` + `v1-g0-speed-r2` | 待回填 | IO 重构计划留档 |

## T9 红线（实施期逐条自检）

| # | 红线 |
|---|---|
| B1 | 本计划全部代码改动限于验证资产（`scripts/smoke-local/**`），训练语义零改动；G0 起跑前 `G0_SCOPE=PASS` 是硬闸 |
| B2 | 引用任何基线产物前必跑 `check_baseline_env.py`，`BASELINE_ENV=FAIL` 即停（AGENTS 18 末句） |
| B3 | D2-cold 未 PASS 不得宣称 G0 可跨期作 bitwise 一侧；不得以保留缓存为由绕过 |
| B4 | G2 vs G0 只作对账，不作独立判据，不得用于放行 |
| B5 | run_name 起跑前用户确认；>5 min run 全部留档；收官清理临时 run 与缓存软链（保留 G0 固化产物） |
| B6 | 登记簿数字只在本文档维护一份，三份计划只引用不复制 |
| B7 | 性能结论只取 speed 链 run（符号总表口径）；带 TrainState 摘要 / batch_digests / 确定性 XLA 档的正确性 run，其 util/步时禁作任何性能结论 |

## T10 审计修正记录（2026-08-26，两份对抗审计逐条核对后落实）

本轮修订（对应上文各处「2026-08-26 审计修正」标注）：

1. `batch_digests` 明确 raw / canonical 双口径，raw 禁用于跨 dtype 对拍（三节、五节 G1 行）；
2. TrainState 等价新增逐叶数值裁决，拿不出即 `INCONCLUSIVE`（六节、T2）；
3. 新增 P1b 量具补遗工序（canonical 通道、逐叶统计、uv runner 收敛、speed 口径联动、G0 补录 replica 轮），置于 G1 之前（七节、T7）；
4. run_name 轮次规约：G0 共用名属既成事实不追溯，后续多轮 run 一律 `-r<N>`/语义后缀（T7）；
5. G3+ 设备端对拍的前置缺口如实标注（五节）。

**2026-08-27 P1b 落地与 G0b 重跑记录（用户裁定变更，附记于此）**：

6. 用户裁定「基线可全部重跑」+「G0 两轮与 G0-speed 升级 1000 步」——P1b 第 5 项补录 replica 轮取消，改为 G0b 重跑取代（七节 P1b 落地记录）；量具功能 commit V2.3.1 `570287f`（含护栏 600→1200、`EXTRA_DIGEST_STEPS`、`STATE_DUMP_STEPS`）。
7. G0b 千步自证与新旧前 300 步前缀对拍全 PASS（T8），旧 G0 records 删除、G0b 升任链头；速度锚点换 `v1-g0-speed-r2`（1000 步口径）。TrainState 数组参照经用户裁定迁存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（不留 NFS）。
8. ~~dtype 统一计划（V2.4 起）执行由用户 2026-08-27 指令中止，T8 相关行悬置。~~

**2026-08-27 同日恢复执行（用户「实现」指令）**：

9. dtype 计划恢复执行，T8 的 G1 / G1-speed 两行解除悬置；工具 commit **V2.4a `f2e7348`**（`scripts/dtype-unify/` 七个文件 + pyproject 的 ruff 豁免，`src/` 零改动）已落地，同时把该目录加入 T1 的 G0_SCOPE 白名单。
10. **A 侧换代**：旧 300 步 G0 的 records 已删除，dtype 计划 P6 的 A 侧改为 **G0b r1**（1000 步），G1 随之升级 1000 步、摘要步集对齐（`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`）、run_name 按 T7 体例定为 `v1-dtype-ab-post-r1`（先跑一轮，仅在需要量化兜底时补 r2）。
11. **符号总表两处订正**：speed run 统一口径的「300 步」订正为 1000 步（原与同表锚点行的「后续 speed run 一律 1000 步」自相矛盾）；G1-speed 的对比对象由退役的 `v1-g0-speed` 改为现行锚点 `v1-g0-speed-r2`，T8 的 G2-speed 行同步。
12. **T4 的「EXP_NAME 与 dtype B 侧共用」撤销**（G0b 缓存已清理，共用无收益且确定性档本就关闭 autotune）。
13. **G0b 作为 A 侧的可用性已获实证**：dtype 计划 P3 的单步梯度取证中，同 seed / 同 config 现场 `init_train_state` 产出的完整 TrainState 与 G0b r1 步 0 的 **177 个叶子摘要逐个相同**（`single_step_grad.py` 的同源校验），跨 commit 引用 G0b 初始状态因此机器可证，无需加载那份 45.4 GiB 的 `state_step_0.bin`。

审计指出、经核对**已在此前提交解决**的问题（存证，不再列为待办）：

- 「`SAVE_INTERVAL=0` 会使 `step % save_interval` 除零」：V2.1（`d9e509e`）已在驱动层以哨兵值规避（0 → 大于步数上限的间隔 + 关摘要开关，`env.json` 记 requested/effective 双值），`train.py` 不吃 0；`v1-g0-speed` 实跑验证。
- 「步 0 TrainState 摘要按 train.py 条件永远不会产生」：V2.1 已包装 `init_train_state` 在初始化后立即记步 0 摘要（fail-loud），驱动判定断言含步 0。
- 「计划宣称不新增环境变量却新增四个」：变量归属已由计划瘦身（`9c49cf6`）理清——四个 bench 变量归本计划 P1（已固化），dtype 计划红线限定为不新增影响生产训练语义的开关。
