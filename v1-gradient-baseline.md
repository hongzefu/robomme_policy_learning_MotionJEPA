# 梯度对拍黄金基线 G0 与基线链规约

> 本文件是计划文档，**尚未实施**。2026-08-26 立项，用户指令：三份计划（[`v1-dtype-unify-plan.md`](v1-dtype-unify-plan.md)、[`v1-framesamp-restructure-plan.md`](v1-framesamp-restructure-plan.md)、[`v1-post-restructure-roadmap.md`](v1-post-restructure-roadmap.md)）的梯度对拍不仅要和自己的改动前比，还要和「三个都没动」的训练（当前仓库状态，git 锁定，必要时 sha256 校验）比；最好现在先跑一轮记下产物、产物进 git 后固化复用。方案经一轮 opus 对抗复核（12 条必须修 / 10 条建议修全部吸收）。
>
> **本文档是基线链的唯一权威载体**：G0 的定义、锁定断言、产物清单、失效条件、量化判据参数与登记簿都只在本文档维护一份；三份计划一律引用本文档章节，不复制其中数字（引用锚点用章节名，AGENTS 9）。

## Context（为什么做这件事）

- 三份计划各自的等价性验证都是「vs 自己改动前」的相邻对比。链条拉长后（dtype 修复 → IO 重构 → roadmap 项 1–3），每一环即使各自通过，也缺一个「相对原始训练累计漂移了多少」的直接锚点。G0 就是这个锚点：**一次跑定、产物固化进 git、之后所有改动都能拿它离线对拍，不必反复 checkout 旧 commit 重跑对照侧**。
- 「跑一轮就固化」成立有前提，本计划把前提做成可证伪的闸门：记录仪器补完整（P1）、确定性确立且必须包含**独立重编译可复现**（P2 新增 D2-cold 档）、环境指纹做成机器判定的 preflight（引用产物前强制跑）。
- G0 的语义必须澄清：它是「**受控确定性档位下的当前训练语义**」，不是字面现状——XLA 确定性 flags 本身就会改变位级结果（`scripts/smoke-local/README.md` 现文亦声明「加了 deterministic_ops 就偏离官方口径」）。「字面现状」的样貌与噪声底由 P2 的 D0 档两轮产物一并固化留档，标注「非判据基线」。

---

# 第一部分（给人看）

## 一、G0 是什么，为什么能「跑一次就不再重跑」

- **G0**：三个计划都未实施的训练语义，在受控确定性环境（P2 选定档）下的一轮真实训练。口径：本机 2×RTX 6000 Ada、b8、300 步、seed 42、SAVE_INTERVAL=25，走 `bench_train_steps.py` 入口，并行 `nvidia-smi -lms 500` 采样。
- **跑两轮**（G0 主轮 + G0' 自证轮，同配置重跑）：第二轮提供 300 步尺度的可复现自证（P2 只证到 100 步）、产物交叉校验、以及量化判据的同 HLO null 对；第二轮走编译缓存命中，耗时更短。round1 为正本，round2 为自证。
- **双重身份**：G0 兼任 dtype 计划 P6 的 A 侧（修复前），P6 随之只跑 B 侧；G0 的 util/步时采样同时充当其决策门的「修复前」侧数据。
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
2. `param_checksums.jsonl`：每 25 步**完整 TrainState** 摘要（params/opt_state/EMA/step 全部叶子逐个 sha256 + `state_digest`）；
3. `batch_digests.jsonl`：步 0/1/2 + 每 25 步，交付 batch 逐键 `sha256(dtype‖shape‖bytes)`。**性质与输出摘要完全不同**：与 XLA/缓存/驱动无关、跨计算图（HLO）永远逐位可比——roadmap 项 2/3（改输入签名）场景对拍 G0 的**主判据**；
4. `scalars_hex.tsv`：`metrics.jsonl` 的规范化投影（`step<TAB>loss.hex<TAB>…`，剔除 wall_time 等易变字段）+ 其 sha256——「两轮是否一致」退化为一次 sha256 比较，人和机器都不会搞错（`metrics.jsonl` 含 wall_time，不可直接 diff）；
5. `env.json`：环境指纹（真实 argv、库版本、GPU/驱动、XLA_FLAGS、编译缓存命中/编译计数——见 T2）；
6. `BASELINE_MANIFEST.json`：逐产物 sha256 / 行数 / schema 版本——防产物腐烂与工具漂移；
7. util 采样原始数据与统计（AGENTS 16 口径：稳态均值、0% 采样占比、慢步分层均值）、launch.md、result.md。

## 四、失效条件与 preflight

失效条件实现为 `scripts/smoke-local/check_baseline_env.py` 的机器断言（清单见 T5），**每次引用 G0（或任何登记簿基线）产物前强制跑**；散文清单必被遗忘，机器判定不会。触发任一 → 该基线失效，必须重跑并在登记簿记新版本。要点：

- 库版本（uv.lock 哈希 + 单列 torch/jax/jaxlib/numpy/ml_dtypes——torch 版本决定 `randperm` 的 index 排列，变了则样本序列变、一切失效）；
- GPU 型号 + 驱动 + 拓扑（jax 编译缓存 key 含加速器拓扑序列化；换卡/换驱动位级行为可能变）；
- 二节的全部 git 外指纹；
- XLA 确定性档位（XLA_FLAGS 原文逐字比对）与 JAX 配置快照（`jax_enable_x64`、matmul precision、`XLA_PYTHON_CLIENT_MEM_FRACTION`、device_count/fsdp_devices）；
- **单 epoch 约束**：凡与 G0 对拍的 run 必须 `steps × batch_size < 395,289`（单 epoch 内）。IO 重构计划 1.6 节已证：跨 epoch 边界后 index 序列与 num_workers 相关（torch 既有语义），超出单 epoch 的对拍失去意义。当前口径 300×8=2,400 远在界内，写死此条防将来加步数踩雷。

## 五、三方对拍矩阵与基线链

基线链：**G0（原始）→ G1（dtype 修复后）→ G2（packed IO）→ G3…（roadmap 各项）**。每个新节点：vs 上一节点（主判据，尽可能 bitwise）+ vs G0（锚定）。

| 对拍 | 判据 | 说明 |
|---|---|---|
| G1 vs G0 | bitwise 主判据 + 量化兜底（六节） | dtype 计划 P6：A 侧=G0 固化产物（不重跑），只跑 B 侧；B 侧应尽快接续 G0（理想同场次），起跑前必过 preflight。HLO 因输入 dtype 改变而不同，bitwise 存在虚假失败可能，兜底见六节 |
| G2 vs G1 | 同 clean HEAD、同 HLO、共用编译缓存的 bitwise | IO 重构计划 C.3 原判据不变——链条中最强的一节 |
| G2 vs G0 | **对账，非独立判据** | 若 G2 vs G1 bitwise PASS，则 G2 轨迹与 G1 逐位相同，G2 vs G0 数学上恒等于 G1 vs G0——不是独立证据。判定标准：用 git 里 G0/G1 固化产物重算的报告须与 dtype 计划验收留档的 G1 vs G0 报告**逐字节相同**；它检出的是产物腐烂/对比工具漂移/留档记错，不是链路问题。若 G2 vs G1 未达 bitwise（按 C.3 属必须修复的失败），此项自动失去意义，**不得用它「曲线救国」放行** |
| G3+（roadmap 各项）vs 上一基线 + vs G0 | **主判据 = `batch_digests` 输入侧逐位对拍**（跨 HLO 有效）+ 量化复核（六节）+ 单步 fixture 回归 | 项 2/3 改输入签名，输出侧 bitwise 天然不可得；例：项 2 把 pos 挪到 GPU 侧生成后，把设备端 gather 出的 pos 张量与 G0 的 `static_pos_emb` 摘要对拍即是逐位判据，比 300 步量化统计硬得多 |

- **粗差/细差分工（不得拿「300 步过了」当全覆盖）**：300 步 × b8 = 2,400 样本只覆盖粗差（行号错位、选帧错误、dtype 交付错误会在头几十步撞穿任何阈值）；「万分之一错帧」类细差的覆盖责任在各计划第 1 层定点样本对拍（dtype 计划 T3 约 2,600 / IO 重构计划 C.2 约 8,200）与全量 verify。
- **单步 fixture 常规回归闸**：dtype 计划 T4 的单步定点梯度对拍升格为可复用 fixture——固定初始 state + 固定 batch（npz 存 `v1-store/fixtures/`，逐文件 sha256 摘要进 git）。后续任何 commit 花约 2 分钟即可重锚 G0，替代 1.5 h 的轨迹重跑，适合当常规回归闸。
- **revert 链形态**：若 dtype 修复被 revert（其失败处置），G1 不存在、IO 重构计划退回 v3 形态，链条变为 G0 → G2'（v3 形态）；G0 保持链头不变，登记簿如实记录。

## 六、量化判据（等价性检验形态，权威版本）

> 本节替换 dtype 计划 T4 原「rel 三档先验阈值 + OLS 趋势」判据（原 OLS「β>0 且 p≤0.05 即 FAIL」已删除：rel 序列强自相关使 p 值失标定，且混沌轨迹下任何扰动都 β>0，无鉴别力）。适用场景：跨 HLO 对拍的兜底评估；同 HLO A/B（如 C.3）不适用——必须 bitwise。

- `rel(a,b) = |a−b| / max(|a|,|b|,1e-8)` 逐步计算，对五个标量分别统计 median / p95 / max。
- **null 对（噪声底）选取，与被比场景同构优先**：跨 HLO 场景 → D2-cold 两轮（编译期噪声）；同 HLO 场景 → G0 vs G0'（D2-cold PASS 时应逐位为零，null 退化）；均不可得时 → D0 两轮（现状 autotune 噪声，注明是上界）。
- **判据**：A/B 的 rel 各统计档 ≤ null 对相应档 × 2（余量）；**下限守卫**：null 档位低于绝对下限（loss 1e-6 / 三个梯度范数 1e-5 / 末步 param_norm 1e-5）时以绝对下限为准——比天然噪声还小的差异不必吹毛求疵。（阈值低于噪声底必误报、远高于噪声底无鉴别力，故必须经验标定而非先验拍数。）
- **趋势判据**：主用包络——A/B 的 rel(t) 逐步不超过 null 对 rel(t) 上包络 × 2；可选辅助诊断 `log(rel)` 斜率与 null 斜率对比（差异增长是乘性的），仅作定位参考、不单独作 FAIL 依据。
- 量化判据在 IO 重构计划 C.3 场景仍**不作为放行依据**（该计划 3.4 既有裁定不变：无 dtype 差异的 A/B 出现任何残差都指向 bug 或非确定性，必须修到 bitwise）。

## 七、执行序列（本文档实施时执行；预计 5.5–6.5 h 墙钟，全程 tmux + Monitor）

总逻辑：G0 要「一次跑定、以后所有改动拿它当参照」，前提有二——**P1 把记录仪器补完整**（仪器缺项，基线就有永久盲区；仪器说谎，指纹就没法比）、**P2 证明同配置重跑两次结果完全一样**并找出成立条件（否则以后任何对拍差异都分不清是「改动引起」还是「重跑噪声」）。之后才轮到 G0 本体。

### P1（commit V2.1，bench 驱动改造）——G0 由 `run_2gpu_epoch_bench.sh` + `bench_train_steps.py` 驱动，它们就是量具

含 dtype 计划 T2/S0 原有四项（EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE + 缓存软链进 `v1-store/cache/jax/`、XLA_FLAGS 外部注入、checksum recorder 扩展完整 TrainState），本计划扩展六项（逐项作用）：

1. **`batch_digests` 输入摘要记录**：其余产物全是输出（loss/梯度/参数），输出把「输入变了」与「计算变了」混在一起，且计算图一变就整体失比；输入侧逐键 sha256 与 XLA/缓存/驱动无关、跨 HLO 永远逐位可比。记录点：bench 入口 monkeypatch 交付层（与既有 `_WandbProxy`/`checksum_state` 同一 patch 风格），对 collate 后、device_put 前的 host 侧 batch 逐键 `sha256(dtype‖shape‖bytes)`，步 0/1/2 + 每 SAVE_INTERVAL。
2. **env.json 改记真实 argv**：现在 heredoc 里 `"seed": 42`、`"fsdp_devices": 2` 是字面量而非取自实际命令；参数化后它会开始说谎，而它正是 preflight 比对的指纹。改为 dump 真实 argv 数组（单一真值源）。
3. **编译缓存命中/编译计数进 env.json**：注册 jax `/jax/compilation_cache/cache_hits`、`/jax/compilation_cache/compile_requests_use_cache` monitoring listener 计数（可辅以 `JAX_EXPLAIN_CACHE_MISSES=1`）——「这轮是热缓存还是冷编译」从口头猜测变成留档事实，缓存冷热直接影响位级可复现性。
4. **`--checkpoint-base-dir` 按 RUN_TAG 分目录**：A/B 共用 EXP_NAME（为共享 per-fusion autotune 缓存，见 dtype 计划 T4 修订）后 run 目录撞名，`initialize_checkpoint_dir` 遇已存在目录直接 `FileExistsError` 拒跑；按轮次分目录，不靠「跑完删目录」避让（崩溃残留会卡死第二轮）。
5. **两个新脚本**：`check_baseline_env.py`（preflight，断言项见 T5）；`compare_baseline.py`（对拍工具与基线同 commit 固化，防一年后工具漂移、口径变了；功能见 T2）。
6. **`scripts/smoke-local/README.md` 同步**：现文「本基准未额外加 `--xla_gpu_deterministic_ops`」在 P1/P2 后变假，不改即错误文档；P2 收官把 D0–D2-cold 结论一并写入。

P1 验收：STEPS=3 连跑两次不拒跑、缓存落 `v1-store/cache/jax/` 且未被删、argv 如实进 env.json、`batch_digests.jsonl` 落盘；另加一条 autotune 共享实证——A 轮后记 `xla_gpu_per_fusion_autotune_cache_dir` 条目数、B 轮后复查复用（把「共用 EXP_NAME 即共享 autotune」从机制推断变实测）。

### P2（commit V2.2，确定性实验，四档各两轮 100 步、SAVE_INTERVAL=10）

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

### PG0（commit V2.3，G0 两轮 + 产物固化）

1. 从 clean HEAD 起跑前先过 `G0_SCOPE` 断言（二节）；run_name `v1-grad-baseline-g0`（用户已确认；实施起跑前仍按 AGENTS 6 再次确认）。
2. round1（正本）→ round2（自证轮，同配置重跑）；round1 vs round2 按 P2 结果判定：D2-cold PASS 环境下应逐位一致，否则记录残差并入六节 null 对。
3. 产物按三节清单固化，`BASELINE_MANIFEST.json` 校验通过后逐文件 `git add` 提交；登记簿（T8）回填 `<G0-HEAD>`、指纹 sha、判定结论。
4. util/步时统计写入 result.md，标注「本机口径，不作最终吞吐结论」（AGENTS 13/16），充当 dtype 计划决策门 A 侧数据。

后续（不在本计划）：dtype 修复本体自 commit V2.4 起（其 P3–P6，见该计划 T5）。

## 八、G0 与正式训练入口默认配置的差距（有效域声明，2026-08-26 增补）

正式训练入口是 `scripts/finetune_mme_vla_suite.sh` → `scripts/train.py`（配置真值源 `src/mme_vla_suite/training/config.py` 的 `TrainConfig(name="mme_vla_suite")`）。G0 走 bench 入口，两者差距逐项如下（脚本与配置实读核对，2026-08-26）：

### 8.1 训练语义核心——完全相同（这是 G0 能当基线的根据）

模型结构与 history 机制（同一 `HistoryPi0Config`：pi05、action_horizon=20、use_history）、lr schedule（CosineDecay warmup 10k / peak 5e-5 / decay 100k）、optimizer（AdamW + clip_gradient_norm=1.0）、EMA（0.999）、freeze_filter、norm stats、数据链路代码（`RoboMMEDataset` → transforms → collate）、初始权重（pi05_base，bench 显式传路径、finetune 走 `OPENPI_DATA_HOME` 默认，同一份文件）、seed（双方均 42：bench 显式传、finetune 走 TrainConfig 默认）、num_workers（均 4）、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`——**逐项同值**。

### 8.2 差异项（逐项列死，防止将来拿 G0 数字直接当 finetune 默认配置的参照）

| 维度 | finetune 默认 | G0 | 影响 |
|---|---|---|---|
| 入口 | `scripts/train.py`（尾部双次 `main()`：先 tentative 后正式） | `bench_train_steps.py`（单次 main，`--num-train-steps` 截断） | 计算图定义同源；bench 以 monkeypatch 挂记录器，不改训练语义 |
| history 变体 | 脚本文件默认 `MME_VLA_TYPE="perceptual-framesamp-modul"`（**非 context**，跑 v1 范围须手改） | `perceptual-framesamp-context.yaml` | **模型不同**——G0 只锚定 context 变体（v1 范围唯一支持） |
| 硬件/并行 | 4 GPU（`CUDA_VISIBLE_DEVICES=0,1,2,3`、fsdp_devices=4；实践为 GL 4×A40） | 本机 2×RTX 6000 Ada、fsdp_devices=2 | 位级不可比（三条不可比之二） |
| batch | 64 | 8（本机 2 卡 64/32/16 全 OOM 实测，b8 唯一可跑档） | 位级不可比；样本序列也不同（同 seed 下序列由 batch_size 参与决定） |
| 步数 | 80,000（完整训练） | 300 | G0 只覆盖前 300 步轨迹 |
| checkpoint | 真落 ckpt（save_interval 10,000；仅 assets/params，train_state handler 被注释——IO 重构计划 4.2 既有问题） | 不落 ckpt（`save_state` 被替换为校验和记录器；校验和步有 device_get 开销，正式训练没有） | 记录行为差异，不进计算图 |
| wandb | 启用（须填 `WANDB_API_KEY`） | `WANDB_MODE=disabled` + `--no-wandb-enabled` | 不进计算图 |
| log_interval | 100 | 1 | 不进计算图 |
| dataset-path | `data/robomme_preprocessed_data`（示例占位路径） | `v1-store/datasets/4task-gl` | G0 锚定 4task-gl 库（含 manifest 指纹） |
| XLA_FLAGS | 无（autotune 默认开） | P2 选定确定性档（deterministic_ops + autotune 0 等） | **核心差异**：G0 是受控确定性口径，finetune 默认属「生产 autotune」不可比（三条不可比之三） |
| 编译缓存 | `train.py` 硬编码 `~/.cache/jax_<exp_name>`，自然增长 | P1 后 KEEP_JAX_CACHE + 软链收敛 `v1-store/cache/jax/` | 机制同源（同一 `jax_compilation_cache_dir`），管理方式不同 |

### 8.3 边界结论

1. **G0 锚定的是训练语义**（数据链路交付内容 + 模型前向/反向计算的定义，8.1 那一列），**不是 finetune 启动面的位级轨迹**——batch/卡数/XLA 档不同，位级结果本就不同，此差距已被二节「三条不可比」中 GL 4×A40 与生产 autotune 两条覆盖，本节把逐项差距写死防误用。
2. 基线链的对拍语义之所以成立：链上所有 A/B 都在 G0 同口径（bench 入口、本机 2 卡、b8、确定性档）下进行，唯一变量是被测改动本身；改动对 finetune 默认配置的等价性由「改动经 G0 口径证明等价 + 改动不含任何 batch/卡数/入口相关分支」间接保证（各计划红线已禁止此类分支）。
3. finetune 脚本自身的三个既有问题**不在基线链 scope**（如实登记，处置须用户单独拍板）：`MME_VLA_TYPE` 默认是 modul 非 context；`train.py` 双次 `main()` 在全新 run_name 下必然 `FileExistsError`（IO 重构计划 4.2 已记录）；`dataset-path` 是示例占位路径。

---

# 第二部分（技术细节，供 agent 追踪）

## T1 G0_SCOPE 断言实现

- 白名单 regex：`^(docs/|scripts/smoke-local/|scripts/data-preprocess-GL/paths\.sh$|[^/]+\.md$)`。
- 判定命令（进 launch.md，含原始输出）：`git diff --name-only 55e6e5bf8ef38b780902d0e63257ea859a432a2c HEAD | grep -Ev '<白名单>'` 输出为空即 PASS。
- 附加：`git status --porcelain` 为空；`git submodule status` 指针与锚点一致；env.json `git_dirty == false`。

## T2 新脚本职责

- `scripts/smoke-local/check_baseline_env.py`：读取目标基线的 `env.json` 与 `BASELINE_MANIFEST.json`，对 T5 清单逐项断言当前环境一致；输出单行 `BASELINE_ENV=PASS|FAIL`（FAIL 非零退出 + 逐项差异清单）。
- `scripts/smoke-local/compare_baseline.py`：输入两份 records 目录（固化产物或在跑 run 的 records），产出：逐步标量 hex 列 diff、`state_digest` diff、`batch_digests` 逐键 diff、rel 分布（median/p95/max）与包络对比报告（六节口径）；先校验双方 `BASELINE_MANIFEST.json`（产物 sha256 不符即 fail-loud）。

## T3 P2 四档参数

- 全档：本机 2 卡、b8、100 步、SAVE_INTERVAL=10、seed 42；run_name `v1-det-d{0,1,2}-r{1,2}` + `v1-det-d2cold-r{1,2}`；>5 min 一律 `docs/training-doc/<run_name>/` 留档（AGENTS 17）。
- D2-cold 的两轮 EXP_NAME 各异（如 `det-d2cold-a` / `det-d2cold-b`），各自空缓存目录，flags 与 D2 逐字相同。
- 判定行示例：`DET_CHECK=PASS tier=d2cold steps=100 scalar_hex_diff=0 state_digest_diff=0`。

## T4 G0 run 参数

- 入口与口径：`bench_train_steps.py`、2×RTX 6000 Ada、b8、300 步、seed 42、SAVE_INTERVAL=25、P2 选定确定性档的 XLA_FLAGS、`nvidia-smi -lms 500` 并行采样（另保留 15 s legacy 通道对照，AGENTS 16）。
- EXP_NAME 与 dtype 计划 B 侧共用（per-fusion autotune 共享，见该计划 T4 修订）；RUN_TAG 区分 round1/round2 与未来 B 侧。
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

- V2.1 = P1（bench 驱动改造 + 两个新脚本 + README 同步）；V2.2 = P2（四档八轮 + D0 固化留档 + 三支处置结论）；V2.3 = PG0（G0 两轮留档 + 产物固化 + 登记簿回填）。dtype 修复顺延 **V2.4**（原 V2.3），IO 重构计划自 **V2.5** 顺延（原 V2.4–V2.8 → V2.5–V2.9）。
- run_name 全表（起跑前逐个交用户确认，AGENTS 6）：`v1-det-d{0,1,2}-r{1,2}`、`v1-det-d2cold-r{1,2}`、`v1-grad-baseline-g0`；dtype B 侧沿用其计划的 `v1-dtype-ab-post`（`v1-dtype-ab-pre` 已由 G0 兼任、不再单独跑）。

## T8 基线链登记簿（唯一权威；实施时回填，三份计划只引用本表）

| 链节 | run_name | commit | env 指纹 sha | 判据 | 结论 | 产物路径 |
|---|---|---|---|---|---|---|
| D0（字面现状，非判据） | `v1-det-d0-r{1,2}` | 待回填 | 待回填 | 两轮重跑噪声底 | 待回填 | `docs/training-doc/v1-det-d0-r{1,2}/` |
| G0 | `v1-grad-baseline-g0` | 待回填（`<G0-HEAD>`） | 待回填 | G0_SCOPE + round1/2 自证 | 待回填 | `docs/training-doc/v1-grad-baseline-g0/` |
| G1（dtype 修复后） | `v1-dtype-ab-post` | 待回填 | 待回填 | vs G0：bitwise + 量化兜底 | 待回填 | dtype 计划留档 |
| G2（packed） | IO 重构计划 C.3 的 packed 侧 | 待回填 | 待回填 | vs G1 bitwise；vs G0 对账 | 待回填 | IO 重构计划留档 |

## T9 红线（实施期逐条自检）

| # | 红线 |
|---|---|
| B1 | 本计划全部代码改动限于验证资产（`scripts/smoke-local/**`），训练语义零改动；G0 起跑前 `G0_SCOPE=PASS` 是硬闸 |
| B2 | 引用任何基线产物前必跑 `check_baseline_env.py`，`BASELINE_ENV=FAIL` 即停（AGENTS 18 末句） |
| B3 | D2-cold 未 PASS 不得宣称 G0 可跨期作 bitwise 一侧；不得以保留缓存为由绕过 |
| B4 | G2 vs G0 只作对账，不作独立判据，不得用于放行 |
| B5 | run_name 起跑前用户确认；>5 min run 全部留档；收官清理临时 run 与缓存软链（保留 G0 固化产物） |
| B6 | 登记簿数字只在本文档维护一份，三份计划只引用不复制 |
