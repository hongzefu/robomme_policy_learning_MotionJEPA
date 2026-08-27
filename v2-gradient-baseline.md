# 基线链规约 v2（梯度对拍黄金基线与三块验证结构）

> 本文件是 [`v1-gradient-baseline.md`](v1-gradient-baseline.md) 的**干净重写版**（2026-08-27，用户拍板：只写现行口径，历史裁定沉淀到文末附录）。自本文件定稿起，**基线链的唯一权威载体由 v1 移交本文件**：符号定义、登记簿、量化判据、preflight 规约都只在本文件维护一份，三份计划（[`v1-dtype-unify-plan.md`](v1-dtype-unify-plan.md)、[`v1-framesamp-restructure-plan.md`](v1-framesamp-restructure-plan.md)、[`v1-post-restructure-roadmap.md`](v1-post-restructure-roadmap.md)）一律引用本文件章节、不复制数字（AGENTS 9/B6）。v1 原文保留作历史存档，其顶部已加指针。
>
> **当前状态（2026-08-27）**：基线计划工序（P1/P2/PG0/P1b）与 dtype 计划工序（P3–P7）**全部执行完毕**——链头 G0b、速度锚 `v1-g0-speed-r2`、G1 千步 bitwise 全过、`v1-g1-speed` 步时 −7.21%。**下一环节是 IO 重构（framesamp 计划 v4，G2 级），尚未开工，等用户拍板**；其验证结构按本文件「三、下一环节 G2 的三块验证」执行。

## Context（为什么有基线链）

- 三份计划各自的等价性验证都是「vs 自己改动前」的相邻对比。链条拉长后（dtype 修复 → IO 重构 → roadmap 项 1–3），每一环即使各自通过，也缺一个「相对原始训练累计漂移了多少」的直接锚点。链头基线（现为 **G0b**）就是这个锚点：**一次跑定、产物固化进 git、之后所有改动都能拿它离线对拍，不必反复 checkout 旧 commit 重跑对照侧**。
- 「跑一次就固化」成立有三个前提，全部已做成可证伪的闸门并全部通过：记录仪器完整（P1+P1b 量具）、确定性确立且包含独立重编译可复现（P2 的 D2-cold PASS——这是跨期充当 bitwise 判据一侧的唯一授权闸）、环境指纹机器判定（`check_baseline_env.py` 的 `BASELINE_ENV` preflight，每次引用基线产物前强制跑）。
- 基线的语义：它是「**受控确定性档位下的当前训练语义**」（`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`），不是生产口径的字面现状；生产口径的性能问题由平行的 speed 链回答。核心确定性结论独立留档 [`docs/v1-determinism-conclusions.md`](docs/v1-determinism-conclusions.md)。

---

# 第一部分（给人看）

## 〇、符号总表（现行口径；两族 run 各司其职，一个 run 不得身兼两职）

**正确性族**证明「改动没改变训练结果」：带 TrainState 摘要 + batch 输入摘要 + 确定性 XLA 档；其 util/步时仅留档参考，禁作性能结论。**性能族（speed 链）**回答「优化值不值得做」：生产 XLA 档（不注入确定性 flags、autotune 默认开）、`SAVE_INTERVAL=0`、`BATCH_DIGESTS=0`。

### 正确性族（基线链）

| 符号 | run_name | 是什么 | 状态 |
|---|---|---|---|
| **G0b**（链头） | `v1-grad-baseline-g0b-r{1,2}` | 三计划均未实施的原始训练语义黄金基线，1000 步，两轮逐位自证，产物固化进 git | **已固化**（登记簿） |
| G1 | `v1-dtype-ab-post-r1` | dtype 修复后节点 | **PASS**：vs G0b 千步 bitwise 全过 |
| G2 | framesamp 计划 C.3 的 packed 侧 | packed IO 后节点 | 待跑（下一环节，见三节） |
| G3+ | 届时命名 | roadmap 各项节点 | 待立项 |

### 性能族（speed 链）

| 符号 | run_name | 结果 / 对比对象 | 状态 |
|---|---|---|---|
| **G0-speed（锚点）** | `v1-g0-speed-r2` | 稳态中位 1.152 s/step、util 均值 86.5%、0% 采样 4.9%（1000 步口径） | **已固化**（锚点） |
| G1-speed | `v1-g1-speed` | vs 锚点：步时均值 −7.21%、util +5.64pp、0% 采样 −3.69pp、慢步 3→0 | **PASS** |
| G2-speed | `v1-g2-speed` | vs `v1-g1-speed` + vs 锚点（dtype 效果已由 G1-speed 单独体现，本级归因干净） | 待跑（下一环节） |
| G3+-speed | `v1-g<n>-speed` | vs 上一 speed 节点 + vs 锚点 | 待立项 |

- **speed run 统一口径（权威定义）**：`bench_train_steps.py` 入口、本机 2×RTX 6000 Ada、b8、**1000 步**、seed 42、num_workers=4；不注入 `XLA_FLAGS`；`SAVE_INTERVAL=0`（驱动层自动联动 `BATCH_DIGESTS=0`）；`nvidia-smi -lms 500` 密集采样 + 15 s legacy 通道；报 util 稳态均值 / 0% 采样占比 / 慢步分层均值 / 步时中位与均值（AGENTS 16，禁中位数标题结论），标注「本机口径，不作最终吞吐结论」（AGENTS 13）。
- 每个 speed run >5 min，按 AGENTS 17 留档 `docs/training-doc/<run_name>/`，留档以 `docs:` commit 提交；run_name 起跑前按 AGENTS 6 逐个向用户确认，多轮场景一律 `-r<N>` 后缀（run_name 轮次规约，见 T7）。

## 一、G0b 是什么，为什么能「跑一次就不再重跑」

- **G0b**：三个计划都未实施的训练语义，在确定性档（D2 flags）下的一轮真实训练。口径：本机 2×RTX 6000 Ada、b8、1000 步、seed 42、`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`（步 0 与末步必记，共 12 次完整 TrainState 摘要），`bench_train_steps.py` 入口。两轮（r1 正本 + r2 自证）千步逐位一致；r1 前 300 步与退役的旧 G0 逐位前缀对拍通过（量具改造的等价性实证）。
- **双重身份（仅限正确性对拍）**：G0b r1 固化产物兼任下游 A/B 的 A 侧（dtype 计划 P6 已用），后续链节只跑 B 侧。**其 util/步时数据不作任何性能结论**——摘要停顿（一次完整 TrainState 摘要约 47–140 s）与确定性 XLA 档都污染性能口径；性能基线由 `v1-g0-speed-r2` 承担。
- **「不再重跑」的三个前提（均已达成，引用时仍逐次核验第 3 条）**：
  1. **仪器完整**（P1+P1b）：逐步五标量 hex、完整 TrainState 摘要（params/opt_state/EMA/step 全叶子）、batch 输入摘要 raw+canonical 双口径、全步 index 序列、真实 argv 与编译缓存计数进 env.json；
  2. **确定性成立**（P2）：D2 与 D2-cold 双 PASS——独立冷编译两次逐位一致，未来对拍 run 计算图变了、必然现场重编译，D2-cold PASS 是跨期 bitwise 判据的唯一授权闸；
  3. **环境指纹不变**（preflight）：每次引用基线产物前必过 `check_baseline_env.py`（`BASELINE_ENV=PASS|FAIL`），任一失效条件触发即基线作废、必须重跑并在登记簿记版本。

## 二、锁定方式（git + 指纹）

- **锚点 commit**：`55e6e5bf8ef38b780902d0e63257ea859a432a2c`（立项时 HEAD）。`<G0b-HEAD>` = `570287f`（V2.3.1），锚点到起跑 commit 的全部 diff 已过 G0_SCOPE 反向白名单断言（`docs/`、`scripts/smoke-local/`、`scripts/dtype-unify/`、`scripts/data-preprocess-GL/paths.sh`、根目录 `*.md`——白名单 regex 见 T1）。
- **git 外指纹（sha256 单列）**：`norm_stats.json`、`pi05_base` 初始权重、paligemma tokenizer、`episode_manifest.json` 顶层 `sha256`、数据集抽样指纹（`source_spot_sha256` 机制）。git 跟踪内容由 commit sha 锁定，不再额外 sha256。
- **TrainState 数组参照**（量化裁决用，固化产物只有 sha256、补算不出数值）：G0b r1 摘要步 @0/299/999 的完整 TrainState 数组存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（用户裁定不留 NFS，sha 清单进 git）；失配步不在所存步时按 D2-cold 授权重放补落。
- 跨 commit 引用 G0b 初始状态机器可证：同 seed / 同 config 现场 `init_train_state` 的 177 个叶子摘要与 G0b r1 步 0 逐个相同（dtype 计划 P3 已实证），无需加载 45.4 GiB 的 `state_step_0.bin`。

## 三、下一环节 G2（IO 重构）的三块验证（memory token 一致性的实施步骤）

本节是 IO 重构（framesamp 计划 v4）验收的**结构总纲**——回答「memory token 近乎一致（目标：受控环境逐位一致）怎么保证、GPU 有没有吃满」。三块递进、各有硬判据、各司其职；具体工具与参数以 framesamp 计划 C/D/E 节为实施细则，判据归属与放行规则以本节为准。

### 第一块：非训练轻量对拍（不启动训练，证明新旧链路交付内容逐位一致）

对应 framesamp 计划 C.1/C.2（其 S5 步）。两侧交付 dtype 天然相同（dtype 已由 G1 环节统一），判据是**不折算的直接逐位零容差**：

1. **index 序列对拍**（C.1）：新旧 loader 同 seed dump 序列，w0/w4/w8 三档 diff 为空（dump 步数 < 1 个 epoch，防 torch 跨 epoch 既有分叉制造假阳性）；另有 `BENCH_DUMP_IDX` batch_sampler 层的真实链路旁证。
2. **样本/batch 内容对拍**（C.2）：约 8,200 个定点样本（step 边界全覆盖 + 每 episode 首样本 + 随机）在 **transform 之后**逐样本对拍全键，另加 200 个真实 batch 过 `_collate_fn` 对拍；判据全键 shape/dtype/`view(uintN)` 逐位零容差，判定行 `COMPARE_BATCH=PASS`。
3. 守卫测试（C.5 的 Store 组 + Dataset 组）与全量打包 verify（483,291 帧写×读对拍零遗漏）是本块的前置资产。

**本块不过，不开第二块。**

### 第二块：本机真实训练对拍——G2（正确性）+ G2-speed（本机性能对账）

- **G2（framesamp 计划 C.3，其 S6 步，本计划终局等价性检验）**：同一 clean HEAD 下 `MMEVLA_DATA_BACKEND=legacy` vs `packed` 的 A/B，本机 2 卡 b8，确定性档；两侧交付 dtype 相同 → HLO 相同 → 共用编译缓存，**bitwise 是唯一放行判据**：逐步五标量 hex + 摘要步 `state_digest` 全空 diff，且两侧 `batch_digests`（raw 口径即可比）与 index 序列逐位一致。量化判据只用于定位、**不作为放行依据**——本 A/B 无 dtype 差异，任何残差都指向 bug 或非确定性，必须修到 bitwise。步数口径：framesamp 计划现文为 300 步；G0b 换代后是否升 1000 步与 A 侧是否复用 G1 固化产物，**G2 开工时交用户拍板**（待拍板项，见 T7）。
- **G2 vs G0b 对账（非独立判据）**：G2 bitwise PASS 时「packed vs G0b」数学上恒等于已留档的「G1 vs G0b」；收官用 git 固化产物离线重算一次，报告须与 dtype 计划验收留档逐字节相同——它检出产物腐烂/工具漂移/留档记错，不检链路问题；G2 bitwise 未过时对账自动失效，**不得用它「曲线救国」放行**。
- **G2-speed（framesamp 计划 S9-speed）**：`v1-g2-speed` 一轮，speed run 统一口径（〇节），vs `v1-g1-speed` + vs 锚点 `v1-g0-speed-r2`——本机口径的链式对账，**不设阈值、只报数**，作为 IO 重构单独性能归因与下一级立项输入。

**覆盖范围声明**：本块的 bitwise 结论成立于本机受控确定性环境；它证明「重构没改数」，不证明「正式平台吞吐达标」——后者归第三块。

### 第三块：GL 吞吐验收——north star「GPU 吃满」判据（framesamp 计划 D 节，S8a/S8b）

正式训练平台是 GL 4×A40，v1 瓶颈基线（6.933 s / util 69.7%）与全部验收阈值都是 GL 实测；AGENTS 13 明文本机吞吐不作最终结论——**吞吐过/不过在本块判定**：

- **S8a**：GL dataloader-only 四档（w2/w4/w8/w16），fast 校验档 + 冷态自证 provenance。
- **S8b（全链收官测试）**：GL e2e 600 步 T1–T3（+条件档）+ cold-like/hot 双跑；主判据表 5 项机器判定 `E2E_ACCEPT=PASS|FAIL`（必达：步时中位 ≤5.00 s、util 稳态均值 ≥90%、0% 采样 ≤5%、慢步墙钟 ≤5%、epoch ≤8.6 h——数值定义与附加判据以 framesamp 计划 D 节为准），并附「距 100% 的残差分解」作为下一轮优化（roadmap 项 1）的立项输入。**方向性目标：GPU 占用 100%（north star，用户 2026-08-26 拍板）**；阈值不按字面 100% 定。
- 依赖与秩序：S8a 可在全量打包（S4）后先行；**S8b 必须在第二块 G2 bitwise 通过后才跑**；每个超 GL 硬限的 job（4×A40 / 2–4 h）提交前逐个向用户做资源审批并在 `greatlakes.md` 留记录，run_name 确认不能替代资源审批。
- GL 侧不重复 bitwise 证明（本机与 GL 是两种硬件、GL 无稳定确定性基线）；GL 侧异常按量化判据思想兜底、量级由本块吞吐验收覆盖。

### 三块的放行规则

1. 第一块 + 第二块 G2 bitwise 全过 → 允许宣称「IO 重构不改变训练语义」，回填登记簿；
2. G2-speed 与第三块产出性能结论（本机对账 + GL 主判据）；
3. 任何一块失败按 framesamp 计划 3.4 / F 节处置（定位 → 修复 → 重跑至过，**无「差不多就行」路径**）；功能回滚 = `MMEVLA_DATA_BACKEND` 切回 `legacy` + `--dataset-path` 指回源库，打包库保留作证据。

## 四、对拍矩阵与基线链（含已完成链节）

基线链：**G0b（链头）→ G1（已过）→ G2（待跑）→ G3+**；平行 speed 链：**`v1-g0-speed-r2` → `v1-g1-speed`（已过）→ `v1-g2-speed` → `v1-g<n>-speed`**。每个新节点：vs 上一节点（主判据，尽可能 bitwise）+ vs 链头（锚定/对账）。

| 对拍 | 判据 | 状态与说明 |
|---|---|---|
| G1 vs G0b | bitwise 主判据 + 量化兜底 | **已 PASS**（登记簿）；输入侧禁用 raw `batch_digests`（跨 dtype 必失配），用 canonical 口径 + index 序列 |
| G2 vs G1 | 同 clean HEAD、同 HLO、共用编译缓存的 bitwise | 链条中最强的一节；三节第二块 |
| G2 vs G0b | 对账，非独立判据 | 三节第二块；不得用于放行 |
| G3+ vs 上一节点 + vs G0b | 主判据 = 输入侧逐位对拍（按场景选 raw/canonical，跨 HLO 有效）+ 量化复核 + 单步 fixture 回归 | roadmap 项 2/3 改输入签名，输出侧 bitwise 天然不可得；**前置缺口**：需先落地「设备端观测点」（现量具只记 collate 后、device_put 前的 host batch），见 roadmap 该节 |

- **粗差/细差分工**：1000 步 × b8 = 8,000 样本只覆盖粗差；「万分之一错帧」类细差的覆盖责任在各计划第一块的定点样本对拍与全量 verify。
- **单步 fixture 常规回归闸**：dtype 计划 T4 的单步定点梯度对拍已升格为可复用 fixture（位型容器存 `v1-store/fixtures/`，sha 进 git）——后续任何 commit 约 2 分钟即可重锚基线，替代 1.5 h 轨迹重跑。
- **单 epoch 约束**：凡与基线对拍的 run 必须 `steps × batch_size < 395,289`（跨 epoch 后 index 序列与 num_workers 相关，对拍失去意义；preflight 硬断言）。

## 五、量化判据（等价性检验形态，权威版本）

适用场景：跨 HLO 对拍的兜底评估；同 HLO A/B（如 G2 vs G1）不适用——必须 bitwise。

- `rel(a,b) = |a−b| / max(|a|,|b|,1e-8)` 逐步计算，五个标量各统计 median / p95 / max。
- **null 对（噪声底）与被比场景同构优先**：跨 HLO → D2-cold 两轮（编译期噪声）；同 HLO → G0b r1 vs r2（逐位为零，null 退化）；均不可得 → D0 两轮（现状 autotune 噪声，注明是上界：loss rel median 2.7e-3 / max 4.6e-2）。
- **判据**：A/B 的 rel 各统计档 ≤ null 对相应档 × 2；**下限守卫**：null 档位低于绝对下限（loss 1e-6 / 三个梯度范数 1e-5 / 末步 param_norm 1e-5）时以绝对下限为准。**趋势判据**主用包络（rel(t) 逐步 ≤ null 上包络 × 2）；`log(rel)` 斜率仅作定位参考。
- **TrainState 数值裁决**：`state_digest` 失配时仅凭五标量统计不足以判 PASS——必须由 `compare_baseline.py` 输出逐叶数值统计（max-abs / max-rel / L2 / cosine，params/opt_state/EMA 全叶子；数值参照取二节的本机 TrainState 数组）；拿不出逐叶统计时判定只能写 `INCONCLUSIVE`，不得写 PASS。
- 输出判定行：`QUANT_EQUIV=PASS|FAIL scalars=5 null=<pair> margin=2.0`。

---

# 第二部分（技术细节，供 agent 追踪）

## T1 G0_SCOPE 断言

- 白名单 regex：`^(docs/|scripts/smoke-local/|scripts/dtype-unify/|scripts/data-preprocess-GL/paths\.sh$|[^/]+\.md$)`。
- 判定命令（进 launch.md，含原始输出）：`git diff --name-only 55e6e5bf8ef38b780902d0e63257ea859a432a2c HEAD | grep -Ev '<白名单>'` 输出为空即 PASS。附加：`git status --porcelain` 为空；submodule 指针与锚点一致；env.json `git_dirty == false`。
- 后续链节（G2 起）若需把新增验证资产目录（如 `scripts/data-pack-framesamp/`）加入白名单，随其工具 commit 一并修订本节并在 commit body 说明「纯离线验证资产、不被训练进程 import」。

## T2 量具脚本职责（均在 `scripts/smoke-local/`）

- `check_baseline_env.py`（preflight）：读取目标基线 `env.json` 与 `BASELINE_MANIFEST.json`，按 T5 清单逐项断言，输出单行 `BASELINE_ENV=PASS|FAIL`（FAIL 非零退出 + 逐项差异清单）。
- `compare_baseline.py`（对拍）：输入两份 records 目录，产出逐步标量 hex diff、`state_digest` diff、`batch_digests` 逐键 diff（raw 与 canonical 双口径）、rel 分布与包络对比（五节口径）；digest 失配时输出逐叶数值统计（无数组可算即 `INCONCLUSIVE`）；先校验双方 `BASELINE_MANIFEST.json`。**已知缺口**：总判定行 `DET_CHECK` 未区分 raw/canonical 口径——G1 验收时 raw `batch_digest` 的预期失配（dtype 变更所致，4 处、canonical 均一致）拖累总行为 FAIL，判定按分项人工复核记 PASS；G2 场景两侧 dtype 相同、raw 可比，不受影响；跨 dtype 场景复用前应修此缺口或沿用分项判读。
- bench 驱动（`run_2gpu_epoch_bench.sh` + `bench_train_steps.py`）现行能力：EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE 与缓存软链进 `v1-store/cache/jax/`、XLA_FLAGS 外部注入、逐步五标量 hex、完整 TrainState 摘要（步 0 与末步必记）、batch 输入摘要 raw+canonical、全步 index 序列、真实 argv 与编译缓存命中计数进 env.json、`SAVE_INTERVAL=0` 联动 `BATCH_DIGESTS=0`、步数护栏 ≤1200、`EXTRA_DIGEST_STEPS`、`STATE_DUMP_STEPS`。

## T3 确定性档与 null 资产（P2 结论，权威留档 `docs/v1-determinism-conclusions.md`）

- 正确性族固定档：`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，两轮共用编译缓存（D2）；独立冷编译亦逐位一致（D2-cold）。
- D0 两轮（现状档，FAIL 预期）产物固化 `docs/training-doc/v1-det-d0-r{1,2}/`，作量化判据 null 上界；D1 FAIL（ULP 级，embedder scatter-add atomics）。

## T4 基线 run 参数（G0b 口径，后续正确性 run 对齐）

- `bench_train_steps.py`、2×RTX 6000 Ada、b8、1000 步、seed 42、num_workers=4、`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`、确定性档 XLA_FLAGS、`nvidia-smi -lms 500` + 15 s legacy 通道。
- EXP_NAME 各链节独立（不共用——G0b 缓存已清理留 sha 清单，且确定性档本就关 autotune，共用无收益）；RUN_TAG 分轮。
- speed run 参数见〇节统一口径。

## T5 preflight 断言项（`check_baseline_env.py`）

1. `uv.lock` sha256；单列版本：torch、jax、jaxlib、numpy、ml_dtypes（torch 版本决定 `randperm` 排列，变了则样本序列变、一切失效）；
2. GPU 型号 + 驱动 + `jax.devices()` 数量/型号；CUDA_VISIBLE_DEVICES；
3. 二节全部 git 外指纹；
4. XLA_FLAGS 原文逐字比对；JAX 配置（`jax_enable_x64`、matmul precision、`XLA_PYTHON_CLIENT_MEM_FRACTION`、fsdp_devices）；
5. 对拍 run 的 `steps × batch_size < 395,289` 单 epoch 约束；
6. 目标基线 `BASELINE_MANIFEST.json` 全部条目 sha256 复验。

## T6 量化判据参数（五节的机读版）

- `rel(a,b)=|a−b|/max(|a|,|b|,1e-8)`；统计档 median/p95/max；余量系数 2×。
- 绝对下限：loss 1e-6；grad_norm/llm_grad_norm/mem_enc_norm 1e-5；末步 param_norm 1e-5。
- null 对优先级：D2-cold 两轮 →（同 HLO）G0b r1/r2 → D0 两轮（上界）。
- 包络：`rel_AB(t) ≤ 2 × max_null_envelope(t)` 逐步；null 逐位为零时退化为下限守卫。
- 判定行：`QUANT_EQUIV=PASS|FAIL scalars=5 null=<pair> margin=2.0`。

## T7 commit 编号、run_name 与待拍板项

- commit 编号现状：V2.1（P1）、V2.2（P2）、V2.3（PG0）、V2.3.1（P1b）、V2.4a/V2.4b（dtype 工具/修复）已落；**IO 重构计划自 V2.5 起**（其内部切分见 framesamp 计划 E 节）。
- run_name 规约：起跑前逐个交用户确认（AGENTS 6）；同一符号多轮正式 run 一律 `-r<N>`（或语义后缀）。已用名录见登记簿；G2 级建议名（`v1-framesamp-cmp`、`v1-framesamp-ab-{legacy,packed}`、`v1-g2-speed` 等）以 framesamp 计划 E 节为准。
- **G2 开工前待拍板项**：① IO 重构是否开工（v4 决策，两份输入已齐备：G1 正确性 + `v1-g1-speed` 性能）；② G2 步数（framesamp 计划现文 300 步 vs 链头已升 1000 步）与 A 侧形态（同场次双跑 legacy/packed vs legacy 侧复用 G1 固化产物——后者须先论证「G2 的 legacy 侧与 G1 同 HLO 同语义」并过 preflight）。

## T8 基线链登记簿（唯一权威；实施时回填，各计划只引用本表）

| 链节 | run_name | commit | 判据 | 结论 | 产物路径 |
|---|---|---|---|---|---|
| D0（字面现状，非判据） | `v1-det-d0-r{1,2}` | `d9e509e`（V2.1） | 两轮重跑噪声底 | FAIL（预期）：loss rel median 2.7e-3 / max 4.6e-2 | `docs/training-doc/v1-det-d0-r{1,2}/` |
| D1/D2/D2-cold | `v1-det-d{1,2}-r{1,2}`、`v1-det-d2cold-r{1,2}` | `d9e509e`（V2.1） | 两轮逐步 hex + state_digest + batch_digest diff 为空 | D1 FAIL（ULP，atomics）；D2 PASS；**D2-cold PASS（授权闸开）** | `docs/training-doc/v1-det-*/` |
| G0（300 步旧版，退役） | `v1-grad-baseline-g0` | `624d417` | （历史） | PASS 后被 G0b 取代；records 已删（前缀对拍通过后，用户裁定），launch/result.md 留存证 | `docs/training-doc/v1-grad-baseline-g0/`（仅 md） |
| **G0b（现行链头）** | `v1-grad-baseline-g0b-r{1,2}` | `570287f`（V2.3.1） | G0_SCOPE + preflight + r1/r2 千步自证 + r1 前 300 步 vs 旧 G0 前缀对拍 | **PASS**：1000 步标量 hex / 12×state_digest / 14×batch_digest（raw+canonical）/ index 8072 全逐位一致（scalars_hex sha256 `c799a0b2…`）；TrainState 数组 @0/299/999 存 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（sha 清单进 git）；缓存已清理留 sha 清单 | `docs/training-doc/v1-grad-baseline-g0b/` |
| **G1** | `v1-dtype-ab-post-r1` | `a0f76f8`（V2.4b） | vs G0b r1：五标量 hex + 12×state_digest bitwise；输入侧 canonical + index（raw 不计入） | **PASS**：千步全零失配，`scalars_hex.tsv` sha256 与 G0b 逐字节相同；`CANON_CHECK=PASS steps=14`、`INDEX_SEQ=PASS n=8072`（`DET_CHECK` 总行 FAIL 仅因 raw 口径预期失配，见 T2 缺口） | `docs/training-doc/v1-dtype-ab-post-r1/` |
| G2（packed） | framesamp 计划 C.3 packed 侧 | 待回填 | vs G1 bitwise；vs G0b 对账 | 待回填 | framesamp 计划留档 |
| G0-speed（300 步旧锚，退役） | `v1-g0-speed` | `624d417` | （历史） | 稳态中位 1.117 s/step（留档保存，不再作对比对象） | `docs/training-doc/v1-g0-speed/` |
| **G0-speed-r2（现行锚点）** | `v1-g0-speed-r2` | `570287f` | speed 链锚点（AGENTS 16 稳态统计，1000 步口径） | 稳态中位 **1.152 s/step**（n=949）、均值 1.186、util 均值 86.5%、0% 采样 4.9%、慢步 3、epoch 外推 15.82 h（本机口径） | `docs/training-doc/v1-g0-speed-r2/` |
| **G1-speed** | `v1-g1-speed` | `d227931`（训练语义 = V2.4b） | vs `v1-g0-speed-r2`（两侧同法重算） | **PASS**：步时均值 **1.1003 s（−7.21%）**、util 均值 **92.10%（+5.64pp）**、0% 采样 **1.20%（−3.69pp）**、慢步 0、epoch 外推 15.10 h；绝对收益 85.5 ms/step，与预估 ~80 ms/step 吻合 | `docs/training-doc/v1-g1-speed/` |
| G2-speed | `v1-g2-speed` | 待回填 | vs `v1-g1-speed` + `v1-g0-speed-r2` | 待回填 | framesamp 计划留档 |

## T9 红线（实施期逐条自检）

| # | 红线 |
|---|---|
| B1 | 基线链自身的代码改动限于验证资产（`scripts/smoke-local/**` 等白名单目录），训练语义零改动；正式 run 起跑前 `G0_SCOPE=PASS` 是硬闸 |
| B2 | 引用任何基线产物前必跑 `check_baseline_env.py`，`BASELINE_ENV=FAIL` 即停 |
| B3 | 跨期 bitwise 判据的唯一授权是 D2-cold PASS；不得以保留缓存为由绕过 |
| B4 | G2 vs G0b 只作对账，不作独立判据，不得用于放行 |
| B5 | run_name 起跑前用户确认；>5 min run 全部留档；收官清理临时 run 与缓存软链（保留固化产物） |
| B6 | 登记簿数字只在本文档维护一份，各计划只引用不复制 |
| B7 | 性能结论只取 speed 链 run（〇节口径）；带 TrainState 摘要 / batch_digests / 确定性 XLA 档的正确性 run，其 util/步时禁作任何性能结论 |
| B8 | 三块验证的放行规则以三节为准：第一块不过不开第二块；S8b 在 G2 bitwise 通过后才跑；GL 超限 job 逐个资源审批 |

## 附录：历史沿革（压缩版；全过程见 v1 文件与 git 历史）

- **2026-08-26 立项**（锚点 `55e6e5b`）：P1 量具（V2.1 `d9e509e`）→ P2 四档八轮确定性实验（V2.2，D2/D2-cold 双 PASS）→ PG0 固化 300 步 G0 与 `v1-g0-speed`（V2.3 `624d417`）。同日两份对抗审计催生 P1b（canonical 通道、逐叶统计、uv runner 收敛、speed 口径联动）；同日用户反转「dtype 不跑 speed run」裁定，新增 G1-speed。
- **2026-08-27 G0b 换代**：用户裁定「允许全部重跑 + 升 1000 步」——P1b 落地（V2.3.1 `570287f`）后重跑 `v1-grad-baseline-g0b-r{1,2}` 与 `v1-g0-speed-r2`，千步自证 + 前 300 步前缀对拍全 PASS，旧 G0 records 删除、G0b 升任链头；TrainState 数组参照迁存本机；「EXP_NAME 与下游 B 侧共用」撤销。
- **2026-08-27 dtype 环节收官**：一度中止旋即经用户「实现」指令恢复；V2.4a `f2e7348`（工具）+ V2.4b `a0f76f8`（三行修复）；第一块 `COMPARE_DTYPE=PASS`（2,600 样本/200 batch/0 失配）、单步梯度三档 PASS、G1 vs G0b 千步 bitwise 全过、`v1-g1-speed` −7.21%。v4 决策两份输入齐备。
- **判据沿革**：原 OLS-p 趋势判据废除（rel 序列强自相关致 p 值失标定），改等价性检验形态（null 对标定 + 包络，五节）；`batch_digests` 拆 raw/canonical 双口径，raw 禁用于跨 dtype 对拍。
