# 第一阶段报告：训练确定性定档与梯度对拍黄金基线

> **范围**：v1 dataloader 重构链条的第一阶段——把「同配置重跑两次结果完全一样」做成可证伪的前提，
> 并跑出一条**一次跑定、产物固化进 git**的黄金基线，供后续所有改动离线对拍。
> 本报告只保留人类审阅需要的内容与实测结论；实现级细节（断言实现、脚本职责、参数表、commit 切分、
> 红线自检、审计修正逐条记录）留在源计划文件 [`v1-gradient-baseline.md`](../v1-gradient-baseline.md) 第二部分。
> 第二阶段见 [`v1-phase2-dtype-unify-report.md`](v1-phase2-dtype-unify-report.md)。
>
> **状态：全部执行完毕**（2026-08-26 立项 → 2026-08-27 收官）。确定性定档 D2/D2-cold 双 PASS，
> 黄金基线链头为 **G0b**（1000 步，两轮逐位一致），速度链锚点为 **`v1-g0-speed-r2`**。

---

## 一、结论先行

1. **确定性成立，且成立条件已定死**：注入 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`
   后，本机同配置重跑逐位一致；**并且两次完全独立的冷编译也逐位一致**（D2-cold PASS）。
   后者是黄金基线能跨期复用的唯一授权闸——将来的改动都会改变计算图、必然现场重编译。
2. **黄金基线已固化**：`v1-grad-baseline-g0b` 两轮 1000 步，逐步标量、TrainState 摘要、输入摘要、
   样本 index 序列全部逐位一致，产物进 git。后续任何改动拿它离线对拍，**不必再 checkout 旧 commit 重跑对照侧**。
3. **性能与正确性彻底分跑**：正确性 run 带状态摘要与确定性 XLA 档，其步时/GPU 利用率一律作废；
   性能结论只取「speed 链」的生产档 run。现行速度锚点 `v1-g0-speed-r2` = 稳态中位 1.152 s/step、
   GPU util 均值 86.5%。
4. **同 seed 下 dataloader 交付是逐位确定的**（八轮实验 40 条输入摘要零分歧）。这条把「数据侧」
   从后续一切分歧的嫌疑名单里摘了出去：日后再见到输入摘要分歧，直接指向数据侧改动而非计算噪声。
5. **有效域声明**：基线走的是 bench 入口而非正式训练入口。两者共享同一条核心训练路径，入口包装本身
   预期不改变数值，但**尚未做「只切入口、其余全同」的直接 A/B**，因此不得宣称两入口已实测 bitwise 一致。

---

## 二、Context（为什么做这件事）

- 后续三份计划（dtype 修复 → IO 重构 → roadmap 各项）各自的等价性验证都是「vs 自己改动前」的相邻对比。
  链条拉长后，每一环即使各自通过，也缺一个「相对原始训练累计漂移了多少」的直接锚点。
- 「跑一轮就固化」这件事本身有前提，本阶段把前提做成了可证伪的闸门：**记录仪器补完整**、
  **确定性确立且必须包含独立重编译可复现**、**环境指纹做成机器判定的 preflight**（引用产物前强制跑）。
- 基线的语义必须澄清：它是「**受控确定性档位下的当前训练语义**」，不是字面现状——XLA 确定性 flags
  本身就会改变位级结果。「字面现状」的样貌与它自身的重跑噪声由 D0 档两轮产物一并固化留档，
  标注「非判据基线」。

---

## 三、符号总表（两族 run 的权威定义）

全部 run 分两族，**正确性族证明「改动没改变训练结果」，性能族回答「优化值不值得做」，一个 run 不得身兼两职**。

### 正确性族

带完整 TrainState 摘要 + batch 输入摘要 + 确定性 XLA 档；其 util/步时**仅留档参考，禁作性能结论**。

| 符号 | run_name | 是什么 |
|---|---|---|
| D0 / D1 / D2 / D2-cold | `v1-det-*` | 确定性预备实验，各两轮 100 步，证明「同配置重跑两次结果逐位一样」在什么条件下成立 |
| G0（已退役） | `v1-grad-baseline-g0` | 300 步初版黄金基线，已被 G0b 取代 |
| **G0b（现行链头）** | `v1-grad-baseline-g0b-r{1,2}` | 三计划均未实施的原始训练语义，1000 步，产物固化进 git |
| G1 | `v1-dtype-ab-post-r1` | dtype 修复后节点，对拍 G0b（第二阶段） |
| G2 | IO 重构后的 packed 侧 | 对拍 G1（bitwise）+ 对 G0 做账 |
| G3+ | 届时命名 | roadmap 各项节点，对拍上一节点 + 链头 |

### 性能族（speed 链）

无 TrainState 摘要、无输入摘要、**生产 XLA 档**（不注入确定性 flags、autotune 默认开）。
统一口径：`bench_train_steps.py` 入口、本机 2×RTX 6000 Ada、batch 8、**1000 步**、seed 42、
num_workers=4；`nvidia-smi -lms 500` 密集采样 + 15 s legacy 对照通道；按 AGENTS 16 报
util 稳态均值 / 0% 采样占比 / 慢步分层均值 / 步时中位与均值（禁以中位数作标题结论），
并标注「本机口径，不作最终吞吐结论」。

| 符号 | run_name | 对比对象 |
|---|---|---|
| G0-speed（已退役） | `v1-g0-speed` | 300 步旧锚点 |
| **G0-speed-r2（现行锚点）** | `v1-g0-speed-r2` | speed 链锚点本身 |
| G1-speed | `v1-g1-speed` | vs `v1-g0-speed-r2`（第二阶段） |
| G2-speed / G3+-speed | `v1-g<n>-speed` | vs 上一 speed 节点 + 现行锚点 |

---

## 四、黄金基线是什么，为什么能「跑一次就不再重跑」

**定义**：三个计划都未实施的训练语义，在受控确定性环境下的一轮真实训练。口径为本机 2×RTX 6000 Ada、
batch 8、seed 42，走 bench 入口，并行 GPU 密集采样。**跑两轮**（正本 + 自证轮，同配置重跑）：
第二轮提供该步数尺度的可复现自证、产物交叉校验，以及量化判据所需的同计算图 null 对。

**「不再重跑」的三个前提**（缺一即不成立，全部做成硬判定）：

1. **仪器完整**：原有 checksum recorder 只哈希 params 与 EMA，缺 Adam 动量与 step——动量是「两条轨迹
   是否同一条」最灵敏的累积量，基线若跑在仪器扩展之前，这一列就永久缺失且补不回来。
2. **确定性成立**：尤其是**独立重编译两次仍逐位一致**这一条。未来与基线对拍的 run 计算图都变了、
   必然现场重编译、无缓存可继承，这是基线跨期充当 bitwise 判据一侧的唯一授权闸。
3. **环境指纹不变**：每次引用基线产物前必须先跑 preflight（`scripts/smoke-local/check_baseline_env.py`，
   输出 `BASELINE_ENV=PASS|FAIL`）。任一失效条件触发即基线作废、必须重跑并在登记簿记新版本。

**双重身份（仅限正确性对拍）**：链头同时兼任第二阶段 dtype 对拍的「修复前」一侧，因此第二阶段只跑修复后一侧。
但**链头的 util/步时不作任何性能结论**——摘要停顿（一次完整 TrainState 摘要实测 47.3 s，扩完整后 2–3×）
与确定性 XLA 档都污染性能口径。

---

## 五、锁定方式与失效条件

**git 侧**：以立项时 HEAD `55e6e5bf8ef38b780902d0e63257ea859a432a2c` 为锚点。基线起跑 commit 允许晚于锚点，
但两者之间的**全部** diff 必须落在反向白名单内（`docs/`、`scripts/smoke-local/`、
`scripts/data-preprocess-GL/paths.sh`、仓库根 `*.md`），任何一条越界即拒跑；起跑时工作区必须干净。
用反向白名单而非正向枚举 `src/`，是因为 `uv.lock`、`paths.sh` 这些都在 `src/` 之外却决定训练语义与环境。

**git 覆盖不到的**（单列 sha256）：norm stats、`pi05_base` 初始权重、tokenizer 模型、
episode 清单顶层 sha256、数据集抽样指纹。

**失效条件**（实现为 preflight 的机器断言，散文清单必被遗忘）：

- 库版本（`uv.lock` 哈希 + 单列 torch / jax / jaxlib / numpy / ml_dtypes）——**torch 版本决定 `randperm`
  的 index 排列，变了则样本序列变、一切失效**；
- GPU 型号 + 驱动 + 拓扑（jax 编译缓存 key 含加速器拓扑，换卡/换驱动位级行为可能变）；
- 上述全部 git 外指纹；
- XLA 确定性档位原文逐字比对，以及 JAX 配置快照；
- **单 epoch 约束**：凡与基线对拍的 run 必须满足 `步数 × batch < 395,289`。跨 epoch 边界后 index 序列
  与 num_workers 相关（torch 既有语义），超出单 epoch 的对拍失去意义。当前 1000×8 = 8,000 远在界内。

---

## 六、固化产物清单

全部为文本，进 git `docs/training-doc/<run_name>/records/round{1,2}/`：

1. `metrics.jsonl`——逐步五标量（loss / grad_norm / llm_grad_norm / mem_enc_norm / param_norm）十进制 + hex；
2. `param_checksums.jsonl`——**完整 TrainState** 摘要（params / opt_state / EMA / step 全部叶子逐个 sha256 + 总 digest）；
3. `batch_digests.jsonl`——交付 batch 逐键摘要。**性质与输出摘要完全不同**：与 XLA / 缓存 / 驱动无关、
   跨计算图永远逐位可比，是「改了输入签名」类改动对拍基线的**主判据**。有两个口径：
   **raw**（dtype 参与哈希，只适用于「输入应逐字节不变」的场景）与 **canonical**（逐键升到 f32 后按位视图哈希，
   跨 dtype 场景唯一有鉴别力的口径）；
4. `scalars_hex.tsv`——逐步标量的规范化投影 + 其 sha256。「两轮是否一致」由此退化为一次 sha256 比较，
   人和机器都不会搞错（`metrics.jsonl` 含 wall_time，不可直接 diff）；
5. `env.json`——环境指纹（真实 argv、库版本、GPU/驱动、XLA flags、编译缓存命中与编译计数）；
6. `BASELINE_MANIFEST.json`——逐产物 sha256 / 行数 / schema 版本，防产物腐烂与工具漂移；
7. GPU 采样原始数据与统计、launch.md、result.md。

---

## 七、对拍矩阵与基线链

基线链：**G0b（原始）→ G1（dtype 修复后）→ G2（packed IO）→ G3…**。每个新节点：vs 上一节点（主判据，
尽可能 bitwise）+ vs 链头（锚定）。与之平行的 speed 链：`v1-g0-speed-r2` → `v1-g1-speed` → `v1-g2-speed` → …

| 对拍 | 判据 | 说明 |
|---|---|---|
| G1 vs G0b | bitwise 主判据 + 量化兜底 | A 侧用固化产物、不重跑。因 dtype 改变导致计算图不同，bitwise 存在虚假失败可能。**输入侧禁用 raw 摘要**（dtype 变更使其必然失配、无鉴别力），改用 canonical 口径 + 全步 index 序列 |
| G2 vs G1 | 同 clean HEAD、同计算图、共用编译缓存的 bitwise | 链条中最强的一节 |
| G2 vs G0 | **对账，非独立判据** | 若 G2 vs G1 bitwise 通过，则 G2 vs G0 数学上恒等于 G1 vs G0，不是独立证据。它检出的是产物腐烂 / 对比工具漂移 / 留档记错。**不得用它「曲线救国」放行** |
| G3+ vs 上一节点 + vs 链头 | **主判据 = 输入侧逐位对拍**（按场景选 raw / canonical）+ 量化复核 + 单步 fixture 回归 | 改输入签名的改动，输出侧 bitwise 天然不可得。**前置缺口**：现量具只记 collate 后、device_put 前的 host batch，设备端张量不在任何记录点——该对拍成立需先落地「设备端观测点」 |

**粗差 / 细差分工（不得拿「跑完了」当全覆盖）**：1000 步 × batch 8 = 8,000 样本只覆盖粗差（行号错位、
选帧错误、dtype 交付错误会在头几十步撞穿任何阈值）；「万分之一错帧」类细差的覆盖责任在各阶段第一块的
定点样本对拍与全量 verify。

**单步 fixture 常规回归闸**：单步定点梯度对拍升格为可复用 fixture（固定初始 state + 固定 batch，
按位型容器格式存盘、sha256 进 git）。后续任何 commit 花约 2 分钟即可重锚基线，替代 1.5 h 的轨迹重跑。

---

## 八、量化判据（跨计算图对拍的兜底形态）

只用于跨计算图对拍的兜底评估；同计算图 A/B 不适用——那种场景必须 bitwise。

- 逐步计算相对差 `rel(a,b) = |a−b| / max(|a|,|b|,1e-8)`，对五个标量分别统计 median / p95 / max。
- **null 对（噪声底）与被比场景同构优先**：跨计算图场景取 D2-cold 两轮（编译期噪声）；同计算图场景取
  基线两轮（确定性成立时应逐位为零，null 退化）；均不可得时取 D0 两轮（现状 autotune 噪声，注明是上界）。
- **判据**：A/B 的 rel 各统计档 ≤ null 对相应档 × 2。**下限守卫**：null 档位低于绝对下限
  （loss 1e-6 / 三个梯度范数 1e-5 / 末步 param_norm 1e-5）时以绝对下限为准——比天然噪声还小的差异不必吹毛求疵。
  阈值低于噪声底必误报、远高于噪声底无鉴别力，所以必须经验标定而不是先验拍数。
- **趋势判据**主用包络（逐步不超过 null 上包络 × 2）；`log(rel)` 斜率仅作定位参考，不单独作失败依据。
- **TrainState 数值裁决**：状态摘要失配时，仅凭五个标量的量化统计**不足以判通过**——参数、Adam 动量或 EMA
  明显不同但范数接近时会漏判。失配后必须给出逐叶数值统计（每叶 max-abs / max-rel / L2 相对差 / cosine，
  覆盖 params / opt_state / EMA 全部叶子）。固化产物只含 sha256、事后补算不出数值，因此数值参照取链头落盘的
  摘要步 TrainState 数组（存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`，sha 清单进 git）；
  拿不出逐叶统计时判定只能写 `INCONCLUSIVE`，不得写 PASS。

---

## 九、确定性实验：四档结果与机制定位

### 9.1 实验口径

commit `d9e509e`（V2.1）clean HEAD 起跑（唯一例外：`v1-det-d2cold-r2` 起跑于 `9c49cf6`——期间插入的外部纯文档
commit，在白名单内、训练语义零影响，该轮与 r1 对拍仍 bitwise 通过）；本机 2×RTX 6000 Ada、batch 8、seed 42、
`fsdp_devices=2`、`num_workers=4`、100 步；TrainState 摘要 @ 步 0/50/99（177 叶子，含 params / opt_state / EMA / step），
输入摘要 @ 步 0/1/2/50/99；数据集 `v1-store/datasets/4task-gl`。判定工具 `scripts/smoke-local/compare_baseline.py`，
每档判据：两轮逐步五标量 hex diff + 全部状态摘要 diff + 全部输入摘要 diff 均为空。

### 9.2 四档结果总表

| 档 | 设置 | 判定 | 分歧概貌 |
|---|---|---|---|
| D0 | 每轮删缓存、无 XLA flags（字面现状） | **FAIL**（预期内） | 步 0 起 100/100 步标量全分歧；步 50 起 TrainState 124 叶子分歧 |
| D1 | 两轮共用编译缓存、无 flags | **FAIL**（仅差 ULP） | 100 步中 2 步（step 7/43）`llm_grad_norm` 差一个最低位；TrainState 分歧只有 embedder `input_embedding` 一族 4 叶子（params/mu/nu/ema） |
| D2 | D1 + `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` | **PASS** | 逐位一致，零分歧 |
| D2-cold | 同 D2 flags，两轮各用全新空缓存（强制独立编译） | **PASS** | 各自冷编译（各 2 次 miss、零命中）仍逐位一致；与 D2 轮交叉对拍亦通过 |

全部八轮的输入摘要 **5/5 逐位一致**：同 seed 下 dataloader 交付内容完全确定，一切分歧均来自计算侧。

### 9.3 机制定位

1. **D0 的分歧源是「独立重编译 + 默认 autotune 的 kernel 选择」**：删缓存重跑时 XLA per-fusion autotune
   可选中不同 kernel / 归约实现，从步 0 第一次前向就逐位不同。其重跑噪声底（rel，100 步）：

   | 标量 | median | p95 | max |
   |---|---|---|---|
   | loss | 2.721e-03 | 1.149e-02 | 4.624e-02 |
   | grad_norm | 2.439e-02 | 1.146e-01 | 5.399e-01 |
   | llm_grad_norm | 2.292e-02 | 1.107e-01 | 5.358e-01 |
   | mem_enc_norm | 2.746e-02 | 1.224e-01 | 5.538e-01 |
   | param_norm | 0 | 0 | 6.770e-08 |

   此表即量化判据的 **D0 null 上界**：任何跨计算图对拍若 rel 与它同量级，等价性不可判。也说明 100 步内
   混沌放大已把 kernel 级微差放大到梯度范数 54%。
2. **D1 的残余分歧源是 embedding 反向 scatter-add 的 atomics 非确定累加**：同一可执行体（第二轮编译缓存
   全命中、零 miss）运行期仍在 `input_embedding` 梯度处产生 ULP 级差异，并经 Adam 动量累积进
   mu / nu / params / ema 四叶子。loss 与其余标量全程逐位一致——非确定点狭窄且定位精确。
   附带实证：**完整 TrainState 摘要比标量灵敏**——仪器验收的 3 步 run 中标量全一致而 Adam mu 已分歧。
3. **D2 的两个 flag 同时消除以上两源**：`deterministic_ops` 治 atomics，`autotune_level=0` 治 kernel 选择漂移；
   D2-cold 进一步证明在此档位下**编译本身是确定的**——两次独立冷编译产出行为逐位相同的可执行体。

### 9.4 结论与授权

1. **D2-cold 通过 ⇒ 黄金基线可「一次跑定、跨期充当 bitwise 判据一侧」**：未来节点计算图改变、必然现场
   重编译，也不损失 bitwise 可比性（前提：过 preflight，环境指纹逐项一致）。
2. **正确性族 run 的固定确定性档**：`XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`——
   基线及后续一切正确性 A/B 一律注入；基线的编译缓存目录允许清理（留 sha256 清单作证据）。
3. **性能族不注入上述 flags**：确定性档的 kernel 选择不代表生产性能口径。
4. **D0 两轮产物固化留档**（`docs/training-doc/v1-det-d0-r{1,2}/`），标注「非判据基线，只作噪声底与口径对照」。
5. 同 seed 下 dataloader 输入交付逐位确定，后续对拍中若见输入摘要分歧即直接指向数据侧改动而非计算噪声。

---

## 十、执行过程与实测结果

### 10.1 工序

| 工序 | commit | 内容 | 结果 |
|---|---|---|---|
| P1 | V2.1 `d9e509e` | bench 驱动改造：输入摘要记录、env.json 改记真实 argv、编译缓存命中计数、按轮次分 checkpoint 目录、preflight 与对拍两个新脚本、README 同步、性能口径开关 | 通过（3 步双烟测；另实测确认共用 EXP_NAME 时 autotune 缓存确被复用） |
| P2 | V2.2 | 确定性四档各两轮 100 步 | D2 / D2-cold 双 PASS，走「三支处置」的第一支（见九节） |
| PG0 | V2.3 `624d417` | 初版基线 `v1-grad-baseline-g0` 两轮 300 步 + 产物固化；同场次预跑速度锚点 `v1-g0-speed` | 两轮逐位一致（`scalars_hex.tsv` sha256 `5da1a1c6…`） |
| P1b | V2.3.1 `570287f` | 量具补遗（审计修正）：canonical 输入摘要通道 + 全步 index 序列摘要、失配时逐叶数值统计、运行器统一收敛到 `uv run`、speed 口径联动、README 同步 | 通过；随后重跑基线两轮升级为 1000 步（G0b）+ 重测速度锚点（`v1-g0-speed-r2`） |

**P1 的必要性**：现有 checksum recorder 只哈希 params 与 EMA。缺 Adam 动量意味着基线永久缺一列最灵敏的
累积量；`env.json` 里 seed 与 fsdp_devices 原本是 heredoc 字面量而非取自实际命令，参数化后它会开始说谎，
而它正是 preflight 比对的指纹。这两处不修，基线跑出来就是带盲区的。

**摘要密度的取舍**：TrainState 摘要由每 25 步稀疏化到每 100 步（步 0 与末步必记）。一次完整摘要约 95–140 s，
每 25 步记 12 次就是 19–28 min 纯停顿、比训练本体还贵；而分歧定位主靠逐步标量 hex，密摘要买不到额外灵敏度。

**P1b 的四处缺口**（两份对抗审计发现）：① raw 输入摘要在跨 dtype 场景必然全线失配、分不清「只是类型变了」
与「数值真变了」，需要 canonical 口径；② 摘要失配时拿不出逐叶数值统计就无法裁决；③ 运行器未走 `uv run`；
④ speed 口径需要手动同时关两个开关、易漏。原计划的「补录 replica 轮」经用户裁定「允许全部重跑」后取消，
改为量具落地后直接重跑基线（升级 1000 步）与速度锚点。

### 10.2 基线链登记簿

| 链节 | run_name | commit | 判据 | 结论 | 产物 |
|---|---|---|---|---|---|
| D0（非判据） | `v1-det-d0-r{1,2}` | `d9e509e` | 两轮重跑噪声底 | FAIL（预期）：loss rel median 2.7e-3 / max 4.6e-2，全表见九节 | `docs/training-doc/v1-det-d0-r{1,2}/` |
| D1 / D2 / D2-cold | `v1-det-d{1,2}-r{1,2}`、`v1-det-d2cold-r{1,2}` | `d9e509e` | 两轮逐步 hex + 状态摘要 + 输入摘要 diff 为空 | D1 FAIL（ULP 级，atomics）；D2 PASS；**D2-cold PASS（授权闸开）** | `docs/training-doc/v1-det-*/` |
| G0（300 步旧版，已退役） | `v1-grad-baseline-g0` | `624d417` | 两轮逐位自证 | PASS；**records 已于 2026-08-27 删除**（新旧前缀对拍通过后按用户裁定），launch/result.md 留存作证 | `docs/training-doc/v1-grad-baseline-g0/`（仅 md） |
| **G0b（现行链头）** | `v1-grad-baseline-g0b-r{1,2}` | `570287f` | 白名单断言 + preflight + 千步两轮自证 + 前 300 步 vs 旧版前缀对拍 | **PASS**：1000 步标量 hex / 12×状态摘要 / 14×输入摘要（raw + canonical）/ index 序列 8072 项全逐位一致；`scalars_hex.tsv` sha256 `c799a0b2…`；前缀对拍逐位一致（即 P1b 量具改造的等价性实证）；摘要步 TrainState 数组迁存本机、sha 清单进 git | `docs/training-doc/v1-grad-baseline-g0b/` |
| G0-speed（旧锚，已退役） | `v1-g0-speed` | `624d417` | 300 步历史锚点 | 稳态中位 1.117 s/step（n=249）、util 均值 86.3%、0% 采样 5.3%、epoch 外推 15.33 h | `docs/training-doc/v1-g0-speed/` |
| **G0-speed-r2（现行锚点）** | `v1-g0-speed-r2` | `570287f` | 1000 步 speed 链锚点 | 稳态中位 **1.152 s/step**（n=949，p10 1.097 / p90 1.276）、均值 1.186、util 均值 **86.5%**、0% 采样 4.9%、慢步 3（分层 1.959 vs 1.184 s）、epoch 外推 15.82 h。vs 旧锚 +3.1%，主因 1000 步窗稀释了 page cache 的乐观偏差 | `docs/training-doc/v1-g0-speed-r2/` |

> 第二阶段的 G1 与 G1-speed 两行见 [`v1-phase2-dtype-unify-report.md`](v1-phase2-dtype-unify-report.md)；
> 登记簿的现行权威版本在 [`v2-framesamp-restructure-plan.md`](../v2-framesamp-restructure-plan.md)。

### 10.3 run_name 轮次规约

初版基线两轮共用 `v1-grad-baseline-g0`、以轮次标签分轮，与 AGENTS 6「每次正式 run 全新 run_name」存在语义歧义。
该 run 已固化留档，属既成事实，**不追溯改名**；自 2026-08-26 起，同一符号需多轮正式 run 的场景，
run_name 一律带 `-r<N>`（或语义后缀），确定性实验的 `v1-det-*-r{1,2}` 即该体例。

---

## 十一、与正式训练入口的差异（有效域声明）

正式训练入口是 `scripts/finetune_mme_vla_suite.sh` → `scripts/train.py`，基线走
`scripts/smoke-local/run_2gpu_epoch_bench.sh` → `scripts/smoke-local/bench_train_steps.py`。

**比较前提**：两种入口最终都调用同一个 `train.main(config)`，共享初始化、dataloader 创建与迭代、
训练步、loss、optimizer 与 EMA 更新。因而在解析后的 config、初始状态、输入 batch 与 XLA 可执行体完全相同时，
入口包装本身不应改变数值。dataset、初始权重、batch、硬件、seed、训练超参和 XLA 配置两边完全相同，不在此比较。

| 维度 | 正式入口 | bench 入口 | 影响 |
|---|---|---|---|
| `main()` 次数 | tentative + 正式共两次 | 单次 | 两边实际训练都调同一个 `train.main(config)` |
| tentative 更新 | tentative 循环临时执行 12 个 optimizer step | 无 | tentative 的状态随调用结束被丢弃，不进入随后重新初始化的正式轨迹 |
| checkpoint 目录 | tentative 已创建目录，第二次 `main()` 触发 `FileExistsError` | 单次初始化，可完成受控短轨迹 | 这是正式入口的**可运行性阻断**，不是计算差异 |
| 编译预热 | tentative 提前完成编译与部分 autotune | 正式轨迹内首次编译 | 改变缓存命中与启动时间；默认 autotune 下还可能影响 kernel 选择，使结果不再 bitwise |
| wandb | 正常初始化并上报 | 关闭，由代理记录指标 | 不进更新公式；影响网络、日志、墙钟与失败面 |
| 日志频率 | 每 100 步写区间聚合均值 | 每步写逐步值 | 不改变状态；bench 的 host 同步更频繁，两份日志不能直接逐行比较 |
| checkpoint 行为 | 真实保存 | 替换为 checksum recorder | checksum 不修改状态，但基线**不验证真实 checkpoint 序列化、异步完成或 resume** |
| checksum 同步 | 无 | 每 100 步大规模 `device_get` | 显著增加墙钟、扰动吞吐与 GPU 利用率，不改变更新公式。实测：300 步版共 12 次、单次中位 47.3 s、合计约 9.4 min，占该轮 15.3 min 总墙钟约六成 |
| 步数限制 | 无上限 | 入口护栏封顶 | 前若干步的训练定义不变，但基线不覆盖后续轨迹、正式保存点或长程行为 |
| runner / cwd | `uv run scripts/train.py` | P1b 起同为 `uv run`，并显式切到仓库根 | 环境版本相同时不产生数值差异（已由 G0b 重跑 vs 旧版前 300 步逐位前缀对拍实证） |

**边界结论**：

1. 入口包装本身不改变模型计算、loss、optimizer 或 EMA 更新公式；相同条件下核心训练轨迹预期一致。
2. 正式入口的双 `main()` 目录冲突是可运行性阻断，bench 入口的单次调用绕开了它，但**没有修复或验证正式 wrapper**。
3. tentative 预热、编译缓存与 autotune 可能改变 kernel 选择与位级结果，但不改变训练目标；
   **未做「只切入口、其余全同」的 A/B 前，不能宣称两者 bitwise 一致**。
4. wandb、逐步日志与 checksum 主要改变同步频率、墙钟与失败面，不应被解释为 loss/梯度公式差异。
5. 基线不验证真实 checkpoint / save-resume 链路。

---

## 十二、溯源

- 源计划与实现级细节（断言实现、脚本职责、参数表、preflight 断言清单、commit 切分、红线、审计修正记录）：
  [`v1-gradient-baseline.md`](../v1-gradient-baseline.md) 第二部分
- 逐轮留档：`docs/training-doc/v1-det-*/`、`docs/training-doc/v1-grad-baseline-g0{,b}/`、
  `docs/training-doc/v1-g0-speed{,-r2}/`
- 量具与判据说明：`scripts/smoke-local/README.md`；工具 `bench_train_steps.py`、`check_baseline_env.py`、
  `compare_baseline.py`
- 登记簿的现行权威版本：[`v2-framesamp-restructure-plan.md`](../v2-framesamp-restructure-plan.md)
