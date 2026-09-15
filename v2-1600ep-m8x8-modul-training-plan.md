# 1600 集新库 · 8 帧 8×8 · modulation · 无 motion · b128 60k · lr 5e-5 正式训练计划

> **实施过程档案**（2026-09-15 第六稿，用户批准）。结果以 [`docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/`](docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md) 为准；本文按 `AGENTS.md` 第 2 条分两部分，另含 F（守卫放宽与增补验证）、G（worktree 隔离）、H（审计结论）、I（本文固化）四节。
>
> **修订史**：第一稿 b64 80k（官方口径）；第二稿 b128 40k lr 1e-4；第三稿 b128 60k lr 5e-5（用户「lr 等效比官方低一倍、40k 延长到 60k」）；第四稿加 F 节（smoke 被 dataset G13 守卫挡下）；第五稿加 H（motion 接入不影响无 motion 的 modulation 的审计结论）、G（不在主工作副本起训练，改从 detached worktree 快照起跑）、I（本文固化到根目录）；**第六稿**按 Codex 审计（`AUDIT_BASE=b625f48`）八条意见 + 逐条对抗验证的结论重修，改动集中在四处：B/C 两节启动命令重写为 runner 内外分层、F 节 commit 顺序倒置并给 F3 独立留档、第 7 点两处论证措辞纠正、G 节隔离补 `openpi_client` 与 norm_stats 绝对路径。
>
> **进度**：步骤 1 已完成（`commitV9.5` = `e0bcb45`，后续又有 `b625f48`）。**注意：第五稿写的「守卫白名单已改在工作区」已失真** —— `b625f48` 工作区 `git status --porcelain` 为空，`framesamp_dataset.py` 的 `_req(hc.integration_type == "context", ...)` 与 `_req(int(hc.memory_token_dim) == 2048, ...)` 仍是原守卫（现位于 `_req` 断言组内，紧接 `perceptual_memory.type` 那条之后）。**执行步骤 2 前必须重新写入 F1 的白名单**，不得以为已改而直接跑 F2。smoke 临时产物已清理；其余步骤待用户指令后执行。

环境判定：**环境 B（AWS 单机）**。仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`（HEAD `b625f48`，clean），8 × A100-SXM4-80GB。无 turbo、无 GreatLakes。

**起跑前实况（2026-09-15 核对，第五稿的「当前全空 / 余 1.4 T」已失真）**：GPU 4–7 空闲（`memory.used` 均为 0 MiB），但 **GPU 0 正在被另一个任务占用**（util 100%、3867 MiB），对应 tmux 会话 `eval-vla0914-full1600-best-a`（2026-09-15 02:50 起）。本轮只用 4–7，不受影响，但 **E 节的 tmux 清单必须把这个会话计入「不得触碰」**。`/scratch` 余 **1.3 T**（`/dev/md0` 6.9 T，已用 5.6 T）。

用户已拍板（2026-09-15）：GPU **4,5,6,7**；wandb **开启**；batch **128**；步数 **60k**；**lr peak = decay = 5e-5**（b128 下等效为官方 5e-5@b64 的一半，不做线性缩放）；改动落在**新配置条目 `mme_vla_suite_b128_60k`**。run_name 随步数改为 **`v2-1600ep-m8x8-modul-b128-60k`**（原拍板 `…-40k` 已不适用，起跑前再确认一次）。

---

## 第一部分（给人看）

### Context：为什么做这件事

上一条生产 run `awsprod40k-b128-motion`（2026-09-04→06）用旧 400 ep 四任务库、32 帧 × 4×4、context + motion。2026-09-14 新库 `4task-v2-1600ep-604f16da`（BinFill / RouteStick / VideoRepick / VideoUnmaskSwap 各 400 集，605,611 执行样本）建成并通过 20 步可读性验收；同日 8 帧 × 8×8 布局（`framesamp-8x8` 库，292 G，`VERIFY_PACK=PASS`）完成逐位验收。本轮在新库上训练第一条 **8 帧 × 8×8 + modulation + 无 motion** 的正式模型，作为后续对照基线。lr 取比线性缩放保守一档、步数从 40k 延长到 60k，以补偿更小步长。

### 结论先行：两处配置改动、零代码改动；先 20 步 smoke 再起正式 run

**1. 改动只有两个配置文件。**
- 新 YAML `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml`：以 `perceptual-framesamp-modul.yaml` 为底只把 `token_per_image` 16 → 64。现有 modul.yaml 是 32 帧 × 4×4，context-8frame-8x8.yaml 是 8×8 但 context，各差一两行。
- 新条目 `mme_vla_suite_b128_60k`（`src/mme_vla_suite/training/config.py`）：复制 `mme_vla_suite_b128`，改 `num_train_steps 60_000`、`peak_lr = decay_lr = 5e-5`、`decay_steps 60_000`（peak == decay 时余弦段为常数，改它只为自洽）、`data.assets.assets_dir` 指到新库 norm_stats 目录。`mme_vla_suite`（官方参照）与 `mme_vla_suite_b128`（上次生产口径）一字不动。

代码侧已就位：dataset `_req` 接受 `(512, 64, 1)` 档并核 `store_meta.spec.tokens_per_frame == 64`，`_max_frames = 8`；`PerceptualMemory` 只断言 `budget == 512`；`MemoryAttention` 对 token 数无假设；motion 节整节缺省即关闭态。

**2. 学习率口径（与官方对照）。** 官方 b64：warmup 10k、peak = decay 5e-5、80k 步。本 run b128：warmup 5k（= 640k 样本，与官方 warmup 样本数相同）、peak = decay 5e-5、60k 步。曲线形状相同（线性升、然后恒定不衰减）。batch 翻倍而 lr 不变，等效步长是线性缩放（1e-4）的一半；总样本 60k × 128 = 7.68M，是官方 5.12M 的 1.5 倍，对新库约 **12.7 epoch**。更新次数 60k，官方 80k。

**3. 为什么必须先 smoke。** modulation 分支在本仓库从未训练过（`v1-postclean-g3` 登记 UNVERIFIED，只有环境 A 用官方 modul 权重评估过）；t8 那轮 12 条轨迹只覆盖 context（C8）与 context+motion（M8）。smoke 用真实 b128、真实 4 卡 fsdp4 跑 20 步，回答两件事：modulation × 8×8 能否编译并出有限数；per-device 32 在 modulation 下是否 OOM（旁证：`bench-b128-util` 在 context+motion、608 位进 prefix 的更重口径下 4 卡 b128 300 步无 OOM；modulation 记忆不进 prefix，激活只会更小）。≤ 5 分钟，临时 run 跑完即删。

**4. 预期。** 4 卡 b128 稳态 3.831 s/step（`bench-b128-util`，util 均值 94.75%）；60k ≈ 63.9 h，加编译与 12 次存盘约 **64.5 h（2.7 天）**。checkpoint **12 × 12 G ≈ 144 G**（按 modulation run `repro-4a100-fsdp4-80g/19` 实测单个 12 G 计；context+motion 的 `awsprod40k-b128-motion/5000` 才是 11 G，第五稿按 11 G 估的 132 G 偏低），磁盘余 1.3 T，仍充裕；但同期 GPU 0 上的评估也在产出，起跑前再 `df -h /scratch` 复核一次。起跑后以前 300 步稳态复核 ETA。

**5. 与上次生产 run 的差异表**（引用结果须带此声明，两条不逐项可比）

| 项 | awsprod40k-b128-motion | 本 run |
|---|---|---|
| 数据 | 400 ep 旧四任务，101,066 样本 | 1600 ep 新四任务，605,611 样本 |
| 记忆布局 | 32 帧 × 4×4，`framesamp` | 8 帧 × 8×8，`framesamp-8x8` |
| 接入 | context，608 位进 prefix | modulation，cross-attention，`memory_token_dim` 1024 |
| motion | 开 | 关 |
| 步数 / lr | 40k / 1e-4 | 60k / 5e-5 |
| 卡 / worker | 8 卡 w16，mesh (2,4) | 4 卡 w8，mesh (1,4)，per-device 32 |
| norm_stats | `robomme-400ep` | `4task-v2-1600ep-604f16da`（`856c75ea…`） |
| 新增可训练参数 | motion 两层 3.35 M | `MemoryAttention`（q/kv/out einsum）+ 每层 `MemoryRMSNorm` modulation Dense；`encoder_static` 2816→1024 |

**6. 第四稿新增：一处 dataloader 守卫要放宽，因此多跑三项验证。** 第三稿的 smoke 起跑后在 `FrameSampDataset.__init__`（`src/mme_vla_suite/training/framesamp_dataset.py`）被一条有意加的守卫挡下：`integration_type='modulation' != 'context'`。这条和它下一行 `memory_token_dim == 2048` 是 commitV3.1（`6ee7494`，2026-08-27）按 Codex 审计 G13 加的，目的是拒绝「同形的 modul 配置」；它们只在构造时拒绝配置，后面的取样、填充、store 读取一行都不读这两个键。

**训练主链路（`train.py`）上没有别的 context 专属守卫**——`training/`、`train.py`、`compute_norm_stats.py`、`policies/` 全部 grep 过，只此两行；模型侧 `history_pi0.py` 本来就接受 modulation。但第五稿把 `g0/` 一并写进「grep 过、只此两行」是**错的**：`scripts/training/g0/bench_train_steps.py` 的 `_EXPECTED_HISTORY_CONFIGS` 是一份只含四个 **context** YAML 的白名单，命中不上即 `raise ValueError`；同一份白名单在 `tests/dump_fixture_samples.py`、`tests/single_step_grad.py`、`tests/motion_gates_model.py` 各有一份，`tests/closed_equiv.py` 的 `CLOSED_YAML` 与 `tests/ref_npy_dataset.py`（非 context 直接 raise）同理。本轮 smoke 与正式 run 走 `train.py`（无此白名单），F3 走 `bench_train_steps.py` 但用的正是 context 8×8 YAML，**所以本计划不受阻**；然而任何 modulation 侧的吞吐 bench 或固定 batch 取证都会被挡，需要时另立任务逐个放宽。

处理方式是把两条守卫合成一条**成对白名单**：只允许 `(context, 2048)` 或 `(modulation, 1024)`，expert 与错配对（例如 modulation 配 2048）照样拒。这是对 dataloader 文件的改动，按 AGENTS 第 18 条要证明「不改数」，所以比第三稿多跑三项，总共约 20–25 分钟：

- **G13 测试改写**（`scripts/training/tests/test_pack_guards.py`）：原测试断言 modul.yaml 必拒，改成「modul.yaml 通过、expert.yaml 拒、错配对拒」三条。该 pytest 依赖 `v1-store/datasets/ref-shard` 迷你库，本环境没有，跑不了；三条断言改在 1600ep 库上用一次性脚本执行，留档写明「本环境未执行 pytest」。
- **第一块（轻量对拍，CPU 约 3–5 分钟）**：用 `perceptual-framesamp-modul-8frame-8x8.yaml` 和 `perceptual-framesamp-context-8frame-8x8.yaml` 在同一 `framesamp-8x8` 库上各构造一个 Dataset，对同一组 256 个索引逐样本逐键比 dtype、shape、raw sha256，判定行 `DS_EQUIV=PASS samples=<n> keys=<k> mismatches=0`。**注意这两份 YAML 的差异不止 `integration_type` / `memory_token_dim` 两个键**：context 版还多出整节 `motion:`（`enabled: false` 等），modul 版整节缺省，而 `framesamp_dataset.py` 与 `dataloader.py::_motion_gates` 都用 `getattr(hc, "motion", None)` 读这一节。所以 `DS_EQUIV=PASS` 证明的是这两组差异的**合集**不影响交付——对第 7 点反而更有利（顺带实测了「整节缺省 ≡ `enabled: false`」的数据侧等价），但不能把结论单独归因给 `integration_type`。
- **第二块（真实训练梯度一致，GPU 4,5 约 30–35 分钟）**：复用 09-14 已固化的 `t8-c8-b` 轨迹（400ep 库、context 8×8、batch 8、fsdp 2、seed 42、确定性 XLA），在**已提交** `commitV9.6` 的 clean HEAD 上跑前 100 步（**不是照抄 t8-c8-b 命令**，六处差异见 F3），先过 `check_baseline_env.py check` 指纹 preflight（`BASELINE_ENV=PASS`），再用 `compare_baseline.py` 一次性产出全部判定行。主判据是 100/100 步全覆盖的五标量 hex 与 `index_sequence.json` 的 800 个样本索引；`BENCH_CHECKSUM=1`（用户 2026-09-15 拍板）另拿六份完整 TrainState 逐叶摘要对照，代价是六步 checksum 实测合计约 1182 s。这证明 context 链在守卫改动前后逐位相同。
- **commit 顺序（用户 2026-09-15 拍板，与第五稿相反）**：F2 通过后**先**提交 `commitV9.6`（守卫 + 测试改写）并 push，F3 再从 clean HEAD 起跑；F3 FAIL 则 `git revert` 该 commit（AGENTS 11 自带 `revert:` 通道，不改写历史、不用 force），把 FAIL 判定行写进 `result.md` 后停下交你处置，不放宽判据。理由：AGENTS 12/17 要求 clean HEAD 起跑，而 `bench_train_steps.py` 会把 `git status --porcelain` 原文写进 `run_meta.json` 的 `start_status`——脏树跑出的产物自带脏树记录，拿它当正式等价证据属自证不合规。

**7. 审计结论：motion 接入不影响「无 motion 的 modulation」——源码级成立，运行时证据只覆盖 context。** 按 motion 接入 commit `06220c4`（commitV6.5）的 diff 原文逐层核：
- **模型侧参数树不变**：`PerceptualMemory.__init__` 的两层新参数 `motion_pos_proj`、`motion_encoder_static` 只在 `motion.enabled` 为真时创建，且建在 `feature_encoder` 之后（nnx 默认 RNG 流按调用顺序 fold_in，帧路初始化值不变）。modul-8x8 YAML 没有 `motion` 节，`_motion_enabled` 为 False。
- **`embed_memory` 关闭态是编译期早返回，执行路径与接入前逐位等价**：函数签名逐字未变（两版都是 `def embed_memory(self, obs: HistAugObservation)`），唯一改动是 `self.mem_encoder(...)` 调用点多传三个关键字实参 `motion_emb / motion_pos / motion_mask`——这三个值在关闭态恒为 `None`（`FrameSampDataset` 的 `_NONE_KEYS` 补键 → `from_dict` 的 `data.get(key, None)`），且 `PerceptualMemory.__call__` 在 `if not self.motion_enabled: return hidden_states, None, None` 之前一条语句都不读它们，传给 `feature_encoder.encode_perceptual_memory` 的三个实参与接入前逐字相同。随后 `if not self.mem_encoder.motion_enabled` 是 Python 编译期分支，早返回分支内的四行（`input_mask = obs.static_mask`、两个全 False 列表、`return`）与接入前的后四行逐字相同。合起来：关闭态执行的运算与返回的四个值与 `06220c4~1` 逐位一致。（第五稿写的「函数体与接入前逐字相同」字面不成立——`mem_encoder` 调用在早返回之前，且函数体多了 `if` 与整段开启态代码。）modulation 分支在 `compute_loss` / `sample_actions` 里只取前两个返回值 `mem_seq, mem_mask` 喂 `PaliGemma.llm(..., mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])`。
- **modulation 的 prefix 不含记忆区**：`embed_prefix` 只在 `integration_type == "context"` 时才把记忆 token 拼进 prefix；modulation 的 prefix 是两路视角各 256 位共 512 图像 token + prompt（**这是训练侧口径**；推理侧 `robomme_policy.py` 只喂 `base_0_rgb` 一路 = 256，引用到评估语境时须注明），RoPE 的 `positions = jnp.cumsum(input_mask, axis=1) - 1` 因此不含记忆位次。**注意力掩码上两条路径口径不同但结果相同**：推理 `sample_actions` 的 modulation 分支把 `na_mask` 丢进 `_`、走 `make_attn_mask` 两参数版；训练 `compute_loss` 在 `use_history=True` 时统一走三参数版（modulation 也不例外），但 modulation 的 `na_mask` 首位就是图像 token 的 `True`（`embed_prefix` 里 `na_mask += [True] * image_tokens.shape[1]`），于是 `make_attn_mask` 内 `jnp.cumsum(mask_na, axis=1) <= 0` 恒为全 False、`mask_not_attend` 恒空，三参数版与两参数版逐位相同——记忆屏蔽项在 modulation 下恒不生效。（第五稿只写了推理那半句，独立读会误以为训练也走两参数版。）
- **608 位交错 / `take_along_axis` 为何对 modulation 不可达**（第五稿把它挂在上一条名下，**推理不成立**，此处更正）：modulation 恰恰是通过 `embed_memory` 的返回值 `mem_seq` 拿到记忆并走 cross-attention 的（`compute_loss` 与 `sample_actions` 的 modulation 分支），开启态产出的 608 位重排序列**会**直达 modulation。真正的不可达性来自两处：关闭态的编译期早返回（上上条），以及下一条的两道显式闸。三项里只有「RoPE 位次变化不可达」是真由 prefix 组成推出的。
- **cross-attention 本体零改动**：接入前后 `history_gemma.py`（`MemoryAttention` / `MemoryRMSNorm`）、`integration/utils.py`、`openpi/models/gemma.py`、`representation/mem_encoder.py` 四文件 `git diff` 为空。
- **两道显式闸**：`HistoryPi0.__init__` 里 `motion_enabled and integration_type != "context"` 即 raise；`inputs_spec` 与 `PerceptualMemory` 的 `motion_enabled` 不一致即 raise。开启态混进 modulation 会在建模型时报错。
- **数据侧交付不读 `integration_type`**：`FrameSampDataset.__getitem__` 的 motion 代码全在 `if self._motion_enabled` 内，关闭态只在样本末尾按 `_NONE_KEYS` 追加四个 None（与旧路径「尾部补空键」逐字一致），`HistAugObservation.from_dict` 用 `data.get(key, None)` 接住，None 在 jax pytree 里是空节点、不进数值图。context 与 modulation 拿到同一份 batch。
- **数据侧交付不读 `integration_type`——但只对 `__getitem__` 成立**：`FrameSampDataset.__getitem__` 的 motion 代码全在 `if self._motion_enabled` 内，关闭态只在样本末尾按 `_NONE_KEYS` 追加四个 None（与旧路径「尾部补空键」逐字一致），`HistAugObservation.from_dict` 用 `data.get(key, None)` 接住，None 在 jax pytree 里是空节点、不进数值图。context 与 modulation 拿到同一份 batch。**`__init__` 则会直接拒掉 modul-8x8**（就是 F 节要放宽的那两条守卫），这句摘抄进 `launch.md` 时必须补半句「`__init__` 的形制守卫另由 F 节放宽，见 `commitV9.6`」，否则会让人误以为数据侧零改动。
- **证据边界**：运行时逐位证据只在 context 关闭态取过（环境 A `motion-t1-closed` 对 G0b 逐位同、`motion-t2-ref/cand`；环境 B `aws-t3-closed-s100`、t8 C8/C32）。modulation 关闭态没有单独跑过对拍，`v1-postclean-g3` 登记的 UNVERIFIED 状态未变。F 节第一块把「两份配置在**当前源码**下数据侧交付一致」从论证变成实测——**它不比较接入前源码**，接入前/接入后的数据侧逐位证据仍只有 context 关闭态那几次。
- **模型侧仍是源码级；若要实测，对照只能是「接入前代码 vs 接入后代码的两版模型」，但照官方 modul.yaml 直接起训练起不来。** `06220c4~1`（= `c5925d9`，commitV6.4）的 `FrameSampDataset.__init__` 里那两条 G13 守卫（commitV3.1 `6ee7494` 于 2026-08-27 加入，**比 motion 接入 `06220c4`（2026-09-03）早七天**）会在构造期直接 raise；换 400ep 4×4 库也一样被拒——4×4 只过得了 `(budget, token_per_image, num_views) == (512,16,1)` 那条，过不了 `integration_type == "context"` 与 `memory_token_dim == 2048` 这两条。另立任务时有两条可行路径，二选一：**(a)** 把 F1 的同一条成对白名单原样搬到 `06220c4~1` 与 `06220c4` 两个 detached worktree 上再各起训练——该守卫是纯构造期 `raise`-or-noop、交付路径一行不读，改它不可能改数，两侧同补即可；**(b)** 完全绕开 Dataset：用 context YAML（过守卫）取一份固定 batch 落盘，再把同一份 batch 分别喂给两版模型的 modulation 分支比 loss / 梯度逐叶 hex。注意 modulation 的记忆只经 cross-attention 进 LLM，**没有 `embed_prefix` 那种便宜的中间产物**，(b) 必须跑完整前向，宜复用 `single_step_grad.py` 的固定 state + 固定 batch 口径；且 `closed_equiv.py`、`ref_npy_dataset.py`、`motion_gates_model.py`、`dump_fixture_samples.py`、`single_step_grad.py`、`g0/bench_train_steps.py` 全部把 context 写死，覆盖 modulation 需逐个放宽（估约 150–250 行新代码），**(b) 省的是 GPU 时间不是工作量**。另：「接入前 HEAD 不认 8×8」只是 Dataset 侧的 `(512,16,1)` 断言，模型侧 `PerceptualMemory.__call__` 仅断言 `static_image_emb.shape[1] == budget`（512），(b) 路径下可直接用 8×8。**若目的是把差异归因到 motion 接入本身，对照应取 `06220c4~1` vs `06220c4` 这对相邻 commit，而不是 vs 当前 HEAD**——后者之间还隔着 V9.0–V9.5 的 8×8 支持、新 config 条目与 `train.py` 改动。两条路径都属另立任务，不在本计划内。

**8. 隔离：训练不从主工作副本起跑，而从 detached worktree 快照起跑。** 主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA` 在 65 小时训练期间随时可能有新 commit（留档、评估、其他任务）；Python 在 dataloader worker 重建、checkpoint 保存等时刻会重新 import 源文件，主树被改就可能把新代码带进正在跑的训练。做法沿用 t8 那轮 REF worktree 的既有口径：起跑 commit 记为 `TRAIN_HEAD`，`git worktree add --detach v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k $TRAIN_HEAD` 建只读快照；训练命令 `cd` 到快照根、`PYTHONPATH=<快照>/src:<快照>/packages/openpi-client/src`（压过主树 `.venv` 里 editable 安装的绝对路径 `.pth`；**必须含 openpi-client**，见下）、`UV_PROJECT_ENVIRONMENT=<主树>/.venv`（快照里没有 venv，复用主树的）、`uv run --no-sync`；`source` 的是**主树**的 `scripts/training/paths.sh`（它按自身位置解析 `REPO_ROOT`，所以 `V1_STORE` / `OPENPI_DATA_HOME` 仍指主树 `v1-store`），数据、权重、checkpoint、日志全部走主树 `v1-store` 绝对路径；`get_history_config` 按 cwd 相对路径读 YAML，cwd 是快照，读到的是快照里的 YAML。起跑前一行 preflight 打印并断言 **`mme_vla_suite`、`openpi`、`openpi_client` 三个包**的 `find_spec().origin` 与 `train.py` 路径都落在快照内，写进日志与 `launch.md`。smoke 也从快照跑，同时验证隔离机制本身。训练结束、`result.md` 提交后再 `git worktree remove` 快照。

**第五稿此处有两个错，第六稿更正：**

1. **「`openpi-client` 训练不 import」是错的。** `src/openpi/transforms.py` 顶层 `from openpi_client import image_tools` 无条件执行，而 `train.py` → `mme_vla_suite/training/config.py` 顶层 `import openpi.transforms`，一跳不落；且自建 dataloader 只替换了 Dataset 本体，`dataloader.py` 仍调 openpi 的 `transform_dataset`，transform 链里 `ResizeImages(224,224)` **每个样本调两次** `resize_with_pad`（两路视角）——历史 run 启动日志 `docs/training-doc/awsprod40k-b128-motion/records/run.txt` 的 Data config 行逐字印证。所以 `PYTHONPATH` 必须同时含快照的 `packages/openpi-client/src`（它是仓库跟踪的普通目录，不是 submodule，快照里必然存在；`.gitmodules` 只有 `third_party/robomme_benchmark` 一条）。严重度实为 P2 而非 P1：该包自建库以来只有 1 个 commit，训练期被改的概率极低——但修复成本只是一个环境变量。
2. **norm_stats 必须传绝对路径 flag。** 新条目里的 `data.assets.assets_dir` 是**仓库相对路径**，而 `config.py` 的取值是 `self.assets.assets_dir or assets_dirs` ——短路，条目非 None 就永远走条目；`maybe_download` 对无 scheme 的路径直接 `pathlib.Path(url)`，纯 **cwd 相对**，全链路无 repo-root 锚定。`cd $WT` 后会去找 `$WT/v1-store/train-assets/...`，而快照里没有 `v1-store`，加载返回 None，随即在 `framesamp_dataset.py` 取 `data_config.norm_stats["state"]` 时抛 `TypeError`。**所以 B/C 两节都必须显式传 `--data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"`**（该 flag 真实存在，`v2b-read20-20260914T174147Z` 等十余份 launch.md 都这么写）。顺带：`--assets-base-dir "$V1_STORE/train-assets"` 对本条目其实是**死参数**——回退目标是 `train-assets/<config.name>` = `.../mme_vla_suite_b128_60k`，而该目录下只有 `mme_vla_suite/`；它在历史 run 里"看着有用"只因那些 run 的 config 名恰好是 `mme_vla_suite`。保留无害，但别把它当保险。

**注意不能用「cwd 留主树」来绕开第 2 点**：`get_history_config` 同样是 cwd 相对读 YAML，改 cwd 会读到主树 YAML、破坏隔离本身。传绝对 flag 是唯一正确解。

**隔离机制本身已实测有效**：主树 `.venv` 里是两个**纯路径 `.pth`**（`_editable_impl_openpi.pth` / `_editable_impl_openpi_client.pth`），不是 `__editable___*_finder.py` 那种 MetaPathFinder，且三个包都有 `__init__.py`（regular package）。`.pth` 的路径由 `site.addsitedir` **append** 到 `sys.path` 尾部，而 `PYTHONPATH` 在解释器初始化时就在前段——所以 `PYTHONPATH` 确实压得过。这两点是 G 节成立的前提，`launch.md` 里要把 `cat` 两个 `.pth` 的输出与「site-packages 内无 `__editable___*_finder.py`」一并记下；将来若 `uv sync` 重装换成 finder 风格的 editable，整个机制会静默失效（届时 `IMPORT_ORIGIN` 断言会抓住它）。

**已知残余**：`third_party/` 子模块快照里未初始化，训练链路不 import（已 grep 确认）；`scripts/training/paths.sh` 用主树那份（按 `BASH_SOURCE` 解析 `REPO_ROOT`，只导出路径变量、不 mkdir、不含训练逻辑）。两者写进 launch.md。

**当前状态**：`commitV9.5` 已 push（其后又有 `b625f48`）；**守卫白名单不在工作区**（工作区 clean，第五稿此处失真），步骤 2 要重新写入；smoke 临时产物与 tmux `m8-smoke` 已清理；GPU 4–7 空闲，GPU 0 有他轮评估在跑（tmux `eval-vla0914-full1600-best-a`，不得触碰）。

### 执行顺序（第六稿，八步；第五稿的六步编号在 F 节末尾还残留着第四稿的旧编号，此处统一，全文以本表为准）

1. 新 YAML + 新条目 → 验证 → `commitV9.5` → push。**（已完成，`e0bcb45`）**
2. **F1/F1b**：重新写入守卫成对白名单 + G13 测试改写（都只改工作区，不提交）。
3. **F2**：CPU 轻量对拍 + 三条守卫断言（3–5 分钟，≤5 分钟不触发 AGENTS 17）→ `DS_EQUIV=PASS`。
4. **F2.5 → F3**：先 `commitV9.6`（守卫 + 测试改写）+ push，工作区回到 clean；再从该 clean HEAD 起跑 `t8-c8-guard-s100`（tmux `m8-guard`，30–35 分钟，`BENCH_CHECKSUM=1`）→ `GUARD_GRAD_100=PASS` → 建 `docs/training-doc/t8-c8-guard-s100/` 三件套 + README 加行 → `docs:` commit → push。**FAIL 则 `git revert` commitV9.6 + push，停下交用户处置。**
5. 本计划固化为根目录 `v2-1600ep-m8x8-modul-training-plan.md` + `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md` 初稿 + `docs/training-doc/README.md` 加行 → `docs:` commit → push → **记 `TRAIN_HEAD`（40 位 sha，抄进 launch.md，后续命令一律粘字面量）** → 建 worktree 快照。（G、I 节）
6. 从快照起 smoke（tmux `m8-smoke`）→ 核判据（含 `IMPORT_ORIGIN=PASS`）→ 删临时产物。（B 节）
7. `launch.md` 补 smoke 与步骤 3/4 的判定行摘录 → `docs:` commit → push（主树；快照仍停在 `TRAIN_HEAD`，训练代码不受影响）→ 从快照起正式 run（tmux `m8-prod`）→ 挂 Monitor。（C 节）
8. 跑完（约 65 h 后）：`result.md` + `records/` → `docs:` commit → push → `git worktree remove` 快照。评估另立任务。

> **步骤 7 正是 `TRAIN_HEAD` 不能重算的原因**：它要求起正式 run 之前先给 `launch.md` 补 smoke 判定行并 commit + push，所以 C 节起跑时主树 HEAD **一定已不等于** `TRAIN_HEAD`。命令里若写 `$(git rev-parse HEAD)`，那条断言会 100% 失败。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 配置改动（两文件，一次 commit）

**A1. 新 YAML** `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml`：

```yaml
# 8 帧 × 8×8（token_per_image 64 × 8 帧 = 512 位）+ modulation 接入，无 motion 节（关闭态）。
# 以 perceptual-framesamp-modul.yaml 为底，唯一差异 token_per_image: 16 → 64；对应库布局 framesamp-8x8-v1。
budget: 512
num_views: 1
token_per_image: 64
streaming_obs_horizon: 16
pool_type: mean
use_pos_emb: true
use_state_emb: false
memory_feature:
  img:
    net: identity
    input_dim: 2048
  pos:
    input_dim: 768
    hidden_dim: 768
  state:
    input_dim: 8
    hidden_dim: 512
integration_type: modulation
memory_token_dim: 1024
representation_type: perceptual
perceptual_memory:
  type: frame_sampling
```

**A2. 新条目** 追加在 `_CONFIGS` 列表 `mme_vla_suite_b128` 之后（`src/mme_vla_suite/training/config.py`），带中文注释说明与 b128 条目的差异：

```python
    # 环境 B 新库 1600ep 档（2026-09-15，用户拍板）：与 mme_vla_suite_b128 的差异只有四项——
    #   num_train_steps 40_000 → 60_000；peak_lr / decay_lr 1e-4 → 5e-5（b128 下不做线性缩放，等效为官方
    #   5e-5@b64 的一半，用延长步数补偿）；decay_steps 50_000 → 60_000（peak==decay 时余弦段为常数，只为自洽）；
    #   data.assets 指到 1600ep 新库的 norm_stats。warmup 5_000 不变（= 640k 样本，与官方 10k×64 同）。
    TrainConfig(
        name="mme_vla_suite_b128_60k",
        model=history_pi0.HistoryPi0Config(pi05=True, action_horizon=20, use_history=True,
                                           history_config=None, discrete_state_input=False),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            assets=AssetsConfig(assets_dir="v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da",
                                asset_id="robomme"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=60_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params")),
        num_train_steps=60_000,
        save_interval=5_000,
        keep_period=5_000,
        project_name="robomme-framesamp",
        num_workers=8,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
```

**验证（≤ 2 分钟，CPU）**：
```bash
diff src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml \
     src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml   # 期望：仅注释 + token_per_image 一行
JAX_PLATFORMS=cpu uv run --no-sync python - <<'PY'
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.config import get_config
c=get_history_config('perceptual-framesamp-modul-8frame-8x8.yaml')
assert (c.budget,c.token_per_image,c.num_views,c.integration_type,c.memory_token_dim)==(512,64,1,'modulation',1024)
assert getattr(c,'motion',None) is None
t=get_config('mme_vla_suite_b128_60k'); s=t.lr_schedule
assert (t.batch_size,t.num_train_steps,s.warmup_steps,s.peak_lr,s.decay_lr,t.save_interval,t.keep_period,t.num_workers,t.fsdp_devices,t.ema_decay)==(128,60000,5000,5e-5,5e-5,5000,5000,8,4,0.999)
b=get_config('mme_vla_suite_b128'); assert (b.num_train_steps,b.lr_schedule.peak_lr)==(40000,1e-4)   # 旧条目未动
sched=s.create(); import jax.numpy as jnp
assert abs(float(sched(5000))-5e-5)<1e-12 and abs(float(sched(59999))-5e-5)<1e-12   # warmup 后恒定
print('CONFIG_OK')
PY
git diff --check
```
commit：`git add` 仅这两个文件；subject `commitV9.5: 新增 modulation 8帧8×8 配置与 b128 60k lr5e-5 训练条目`；`git push`。

### B. smoke（20 步、真实 b128 / 4 卡、临时 run，跑完删）

run_name `smoke-m8x8-modul-$(date -u +%Y%m%dT%H%M%SZ)`，tmux `m8-smoke`，日志 `v1-store/logs/m8-smoke.log`。**从 worktree 快照起跑（G 节），不在主树跑。**

**第六稿为什么整段重写命令。** 第五稿的命令块照原样粘贴**一步都跑不起来**，且有一种死法会写出假成功，四条实测：

1. 块内从未给 `RUN` / `LOG` / `TRAIN_HEAD` 赋值，而 `paths.sh` 开头是 `set -euo pipefail`——`set -u` 下第一次展开 `$RUN` 就 `unbound variable` 退出。
2. `set -e` 会持续作用于调用方 shell，于是 `uv run ... | tee -a "$LOG"` 失败后 shell **当场退出**，后面那行 `echo "EXIT_CODE=${PIPESTATUS[0]}"` 永远执行不到（AGENTS 7 要求的结束标记丢失）。
3. 最"自然"的修法 `cmd | tee log || true` **绝不可用**：实测 `|| true` 会把 `PIPESTATUS` 冲成 `(0)`，`EXIT_CODE=` 稳定打印 0，把失败 run 记成成功。
4. `nvidia-smi ... > csv &` 只重定向了 stdout，stderr 仍挂在 body 的管道上；训练结束后若采样器未被杀，`tee` 因写端未全关而不退出，`EXIT_CODE=` 同样写不进日志。

解法照抄 `docs/training-doc/t8-c8-b/launch.md` 已跑通的**内外分层**：`set -e` 活在 `body()` 里（保住「不满足就不起跑」的断言硬闸），`tee` + `EXIT_CODE` 活在 body 外（保住必落盘），`trap ... EXIT` 收采样器。命令落成 runner 脚本再交给 tmux，而不是内联进 `tmux new-session "..."` 字符串——块内有 `<<'PY'` heredoc 与 `${PIPESTATUS[0]}`，内联要做三层转义。顺带解决另一个坑：`paths.sh` 有一整片 `readonly`，把 `source` 关进 `body()`（管道 → 子 shell）后，重跑 runner 不会被「`REPO_ROOT: readonly variable`」打死。

> 注：`docs/training-doc/awsprod40k-b128-motion/launch.md` 那份历史留档**带同款缺陷**（`$LOG` 未赋值 + `set -e` 下 `EC=${PIPESTATUS[0]}` 拿不到），它留下 `EXIT_CODE=0` 纯粹因为那次训练成功了、没走失败路径。**不要拿它当模板。**

**前置**：主树 `git status --porcelain` 空且 HEAD 含 `commitV9.6`；快照存在且 `git -C <快照> rev-parse HEAD == TRAIN_HEAD`；`nvidia-smi --id=4,5,6,7` 显存 0；run 根不存在。这些都写成 `body()` 里的 `test`，由 `set -e` 兜底。

**B-0. 生成 runner（在主树用普通 Bash 执行，不进 tmux）**

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
mkdir -p "$MAIN/v1-store/logs"
cat > "$MAIN/v1-store/logs/m8-smoke-runner.sh" <<'RUNNER'
#!/usr/bin/env bash
# 外层：故意不开 -e，保证 EXIT_CODE 在任何死法下都落盘（t8-c8-b「外层 tee + EXIT_CODE」口径）
set -o pipefail

RUN="${1:?用法: m8-smoke-runner.sh <RUN> <TRAIN_HEAD>}"
TRAIN_HEAD="${2:?用法: m8-smoke-runner.sh <RUN> <TRAIN_HEAD>}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
WT="$MAIN/v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k"
LOG="$MAIN/v1-store/logs/m8-smoke.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
mkdir -p "$(dirname "$LOG")"

body() {
  # 内层：-euo pipefail 就位，下面每条 test 都是「不满足就不起跑」的硬闸
  source "$MAIN/scripts/training/paths.sh"   # 只导出路径变量；不 mkdir；按 BASH_SOURCE 解析 REPO_ROOT=$MAIN

  printf 'RUN=%s\nTRAIN_HEAD=%s\nWORKTREE=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$WT" "$V1_STORE" "$(date -u +%FT%TZ)"

  cd "$WT"                                   # cwd = 快照：get_history_config 按 cwd 相对路径读快照 YAML
  export PYTHONPATH="$WT/src:$WT/packages/openpi-client/src"
  export UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/m8-smoke"   # 固定目录：避免每轮 smoke 都从零编译
  export TRAIN_RECORD_DIR="$V1_STORE/bench/$RUN"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  export WANDB_MODE=disabled

  test -z "$(git -C "$MAIN" status --porcelain)"
  test "$(git -C "$WT" rev-parse HEAD)" = "$TRAIN_HEAD"
  test -z "$(git -C "$WT" status --porcelain)"
  test ! -e "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN"
  test ! -e "$TRAIN_RECORD_DIR"

  # 导入来源断言（四项：三个包 + train.py，全部须在 $WT 下）；输出经外层 tee 落日志
  JAX_PLATFORMS=cpu uv run --no-sync python - "$WT" <<'PY'
import importlib.util, pathlib, sys
wt = pathlib.Path(sys.argv[1]).resolve()
o = {n: pathlib.Path(importlib.util.find_spec(n).origin).resolve()
     for n in ("mme_vla_suite", "openpi", "openpi_client")}
tp = pathlib.Path("scripts/training/train.py").resolve()
assert tp.is_file(), f"train.py 不存在: {tp}"
o["train.py"] = tp
bad = {k: str(v) for k, v in o.items() if wt not in v.parents}
print("IMPORT_ORIGIN=" + ("PASS" if not bad else "FAIL") + " "
      + " ".join(f"{k}={v}" for k, v in o.items()), flush=True)
sys.exit(1 if bad else 0)
PY

  uv run --no-sync python scripts/training/train.py mme_vla_suite_b128_60k \
    --exp-name "$RUN" --num-train-steps 20 --log-interval 1 \
    --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
    --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs" \
    --dataset-path "$DS/framesamp-8x8" \
    --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml \
    --no-wandb-enabled
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
RUNNER
chmod +x "$MAIN/v1-store/logs/m8-smoke-runner.sh"
```

**B-1. 起跑（`RUN` 与 `TRAIN_HEAD` 在这里显式赋值并作为参数传进 tmux）**

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN="smoke-m8x8-modul-$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_HEAD=<把步骤 5 打印的 40 位 sha 原样粘在这里，不得写 $(git rev-parse HEAD)>
echo "RUN=$RUN"; echo "TRAIN_HEAD=$TRAIN_HEAD"      # 两行抄进 launch.md
tmux ls                                             # 起前快照，确认没有同名 m8-smoke
tmux new-session -d -s m8-smoke \
  "bash $MAIN/v1-store/logs/m8-smoke-runner.sh '$RUN' '$TRAIN_HEAD'"
tmux has-session -t m8-smoke && echo "SESSION_UP=m8-smoke"
```

（`--num-train-steps 20 --log-interval 1 --no-wandb-enabled` 为 smoke 专用覆盖，正式 run 不带。用 `bash <runner>` 而非 `bash -lc`：`-l` 会重载登录 profile，可能覆盖 `PATH` / 代理 / conda 初始化，让快照隔离失去确定性。）

Monitor（**绝对路径**；Monitor 与后续 Bash 工具跑在主工作目录，与 tmux pane 里的 `cd $WT` 无关，但写绝对路径免歧义）：
```bash
tail -n +1 -F /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8-smoke.log \
  | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "IMPORT_ORIGIN=|Loaded norm stats|Norm stats not found|Integration Type|Step 19:|EXIT_CODE=|Error|Traceback|RESOURCE_EXHAUSTED|out of memory|unbound variable"
```
（比第五稿多三条：`IMPORT_ORIGIN=` 把 preflight 判定行主动送到监听端；`Norm stats not found` 抓 norm_stats 静默降级——`_load_norm_stats` 失败只打 `logging.info`，不告警；`unbound variable` 让变量漏赋值这类死法立刻可见。）

判据（全部满足才进入 C）：
- `IMPORT_ORIGIN=PASS`，且四个路径逐个目视确认都在 `$WT` 下。
- 日志含 `Loaded norm stats from $V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme`，且**不含** `Norm stats not found in`。
- 日志含 `Integration Type: modulation`；`EXIT_CODE=0`；`Step 0`…`Step 19` 的 loss / grad_norm / mem_enc_norm 全部有限，无 `RESOURCE_EXHAUSTED`。
- run 根 `history_config.resolved.yaml` 含 `token_per_image: 64`、`integration_type: modulation`；`motion_provenance.json` 的 `motion_enabled=false`、`framesamp_manifest_sha256` = `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`（该值取自 `framesamp-8x8/meta/store_meta.json` 的 `manifest_sha256` 字段，**不是** `episode_manifest.json` 文件的裸 sha256，后者为 `df0ec8ed…`，两个口径别混）。
- checkpoint 19 的 `assets/robomme/norm_stats.json` sha256 == `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`（证明上面那个绝对路径 flag 生效并随 checkpoint 落盘；**注意 `save_assets` 在 `norm_stats is None` 时静默什么都不写**，所以这条空了就说明 flag 没生效）。
- **参数树（两步；第五稿那条判据自相矛盾，第六稿改写）**：
  1. `JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/legacy-eval/check_ckpt_param_tree.py --config mme_vla_suite_b128_60k --ckpt-dir <run根>/19 --out <scratchpad>/param_tree.json`（在 `$WT` 快照那份脚本上跑：它 `os.chdir(_REPO_ROOT)`，而 `get_history_config` 按 cwd 读 YAML），判定行须逐字为
     `PARAM_TREE_EXACT=PASS config=mme_vla_suite_b128_60k history_config=perceptual-framesamp-modul-8frame-8x8.yaml yaml_sha256=5b5ac2f85302d4d8 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0`。
     其中 **`n_model=61` 就是 modulation 的正面签名**（context 为 55、context+motion 为 59，见 `docs/training-doc/eval-official-framesamp-context/records/param_tree.json` 与 `eval-hard-patternlock-routestick/records/modul/param_tree.json`）；`missing=extra=0` 是双向精确比对，故 PASS 已**蕴含**六个 modulation 专属叶子存在于 checkpoint。
     **不得再要求 json 中出现参数路径**——`missing` / `extra` / `shape_mismatch` 只在 FAIL 时才列路径，PASS 时三者恒为空列表，两个要求互斥（第五稿正是这么写的）。
  2. 正面列出六条 modulation 专属参数路径（只读 orbax `_METADATA`，纯 grep，不进 JAX）：
     ```bash
     M=<run根>/19/params/_METADATA
     N=$(grep -o "q_einsum_mem', 'w'\|kv_einsum_mem', 'w'\|out_einsum_mem', 'w'\|mem_rms_norm', 'scale'\|mem_rms_norm_ffn', 'Dense_0', 'kernel'\|mem_rms_norm_ffn', 'Dense_0', 'bias'" "$M" | sort -u | wc -l)
     echo "MEM_PARAMS=$([ "$N" = 6 ] && echo PASS || echo FAIL) n=$N"
     ```
     期望 `MEM_PARAMS=PASS n=6`。**第五稿只列了四个名字、漏了 `mem_rms_norm_ffn`（2 叶）**——而它才是 modulation 真正的作用点（`history_gemma.py` 里 `MemoryRMSNorm(name="mem_rms_norm_ffn")(x, mem_mod_vec)` 的 `Dense_0` 才产出 scale/shift）；且 `mem_rms_norm` 是 `mem_rms_norm_ffn` 的**子串**，所以上面每条都带 `', '<字段名>'` 定界。负对照已实测：同一 grep 在 `awsprod40k-b128-motion/5000`（context+motion）上命中数为 0。
- 显存以「20 步跑完、退出 0」为 OOM 判据；`nvidia-smi` 数字只记录不判读。

清理（**全绝对路径 + `${RUN:?}` 防空变量**；`MMEVLA_JAX_CACHE_DIR` 已改成固定的 `m8-smoke` 目录，故不随 run 删）：
```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=<粘贴 B-1 打印的 smoke RUN>
ls -ld "$MAIN/v1-store/train-runs/mme_vla_suite_b128_60k/${RUN:?}" "$MAIN/v1-store/bench/${RUN:?}"   # 删前目视
rm -rf "$MAIN/v1-store/train-runs/mme_vla_suite_b128_60k/${RUN:?}"
rm -rf "$MAIN/v1-store/bench/${RUN:?}"
tmux ls                                   # 删前：确认 m8-smoke 在自己的清单里
tmux kill-session -t m8-smoke             # 只按确切名；禁通配、禁前缀模糊、禁 xargs 批量
tmux ls                                   # 删后：差集必须恰好只少 m8-smoke
```
判定行与 20 条 Step 行摘录进正式 run 的 `launch.md`「起跑前 smoke」节，不单独建目录（smoke ≤5 分钟，不触发 AGENTS 17）。

### C. 正式 run：`v2-1600ep-m8x8-modul-b128-60k`

**起跑前**：写 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md`（起跑 commit、`TRAIN_HEAD`、`WORKTREE`、四条 `IMPORT_ORIGIN` 原文、两个 `.pth` 的 `cat` 输出、差异表、lr 口径说明、完整命令、数据三件套 sha、输出路径、判据、smoke 摘录、F2/F3 判定行摘录、tmux 清单 `m8-prod` / `m8-prod-dense`）；`docs/training-doc/README.md` 表格加一行；`git add` 两文件 → `docs: v2-1600ep-m8x8-modul-b128-60k 起跑留档` → push。起跑 HEAD clean 且含该 commit。

**C-0. 生成 runner**（同 B 节分层口径；差别只在 wandb 开、不带 smoke 覆盖、多一个 GPU 采样器）

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
mkdir -p "$MAIN/v1-store/logs"
cat > "$MAIN/v1-store/logs/m8-prod-runner.sh" <<'RUNNER'
#!/usr/bin/env bash
set -o pipefail                     # 外层不开 -e：EXIT_CODE 必落盘

TRAIN_HEAD="${1:?用法: m8-prod-runner.sh <TRAIN_HEAD>}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=v2-1600ep-m8x8-modul-b128-60k
WT="$MAIN/v1-store/worktrees/train-$RUN"
LOG="$MAIN/v1-store/logs/$RUN.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
REC="$MAIN/v1-store/bench/$RUN"
mkdir -p "$(dirname "$LOG")" "$REC"

body() {
  source "$MAIN/scripts/training/paths.sh"
  set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志

  cd "$WT"
  export PYTHONPATH="$WT/src:$WT/packages/openpi-client/src"
  export UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"   # 正式 run 保持按 $RUN 独立
  export TRAIN_RECORD_DIR="$REC"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"

  test "$(git -C "$WT" rev-parse HEAD)" = "$TRAIN_HEAD"
  test -z "$(git -C "$WT" status --porcelain)"
  test ! -e "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN"

  printf 'RUN=%s\nTRAIN_HEAD=%s\nWORKTREE=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$WT" "$V1_STORE" "$(date -u +%FT%TZ)"

  JAX_PLATFORMS=cpu uv run --no-sync python - "$WT" <<'PY'
import importlib.util, pathlib, sys
wt = pathlib.Path(sys.argv[1]).resolve()
o = {n: pathlib.Path(importlib.util.find_spec(n).origin).resolve()
     for n in ("mme_vla_suite", "openpi", "openpi_client")}
tp = pathlib.Path("scripts/training/train.py").resolve()
assert tp.is_file(), f"train.py 不存在: {tp}"
o["train.py"] = tp
bad = {k: str(v) for k, v in o.items() if wt not in v.parents}
print("IMPORT_ORIGIN=" + ("PASS" if not bad else "FAIL") + " "
      + " ".join(f"{k}={v}" for k, v in o.items()), flush=True)
sys.exit(1 if bad else 0)
PY

  # 全程 15 s GPU 采样：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上（否则 tee 不退出，EXIT_CODE 写不进）
  nvidia-smi --id=4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py mme_vla_suite_b128_60k \
    --exp-name "$RUN" \
    --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" \
    --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs" \
    --dataset-path "$DS/framesamp-8x8" \
    --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
RUNNER
chmod +x "$MAIN/v1-store/logs/m8-prod-runner.sh"
```

**C-1. 起跑**

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
TRAIN_HEAD=<粘贴步骤 5 记录的 40 位 sha；此刻主树 HEAD 已不等于它，绝不可重算>
tmux ls
tmux new-session -d -s m8-prod "bash $MAIN/v1-store/logs/m8-prod-runner.sh '$TRAIN_HEAD'"
tmux has-session -t m8-prod && echo "SESSION_UP=m8-prod"
```

**C-2. 前 30 min 密采**（AGENTS 16；另起会话，不与训练日志共壳）

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
tmux new-session -d -s m8-prod-dense \
  "timeout 1800 nvidia-smi --id=4,5,6,7 --query-gpu=timestamp,index,utilization.gpu,memory.used \
   --format=csv,noheader,nounits -lms 500 \
   > $MAIN/v1-store/bench/v2-1600ep-m8x8-modul-b128-60k/gpu_util_lms500_first30min.csv 2>&1"
```

未覆盖项全走条目默认：`num_train_steps 60000`、`batch_size 128`、lr 5e-5 / warmup 5k、`num_workers 8`、`fsdp_devices 4`、`seed 42`、`log_interval 100`、`save_interval/keep_period 5000`、wandb 开（project `robomme-framesamp`，run 名同 exp-name）。

Monitor（绝对路径，一份日志一个 Monitor）：
```bash
tail -n +1 -F /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/v2-1600ep-m8x8-modul-b128-60k.log \
  | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "IMPORT_ORIGIN=|Norm stats not found|Step [0-9]*00:|Finished asynchronous save|EXIT_CODE=|Error|Traceback|RESOURCE_EXHAUSTED|nan|unbound variable"
```
存活判断用 `tmux has-session -t m8-prod && echo ALIVE || echo GONE`（AGENTS 7 禁裸 `pgrep -f`）。起跑后 300 步复核稳态 s/step（tqdm `Progress on` 行时间戳差分，同 `bench-b128-util` 口径），在 launch.md 补 ETA。

**输出**：checkpoint `v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-modul-b128-60k/{5000,…,55000,59999}`（12 个；存盘条件是 `step % save_interval == 0 and step > start_step` 或 `step == num_train_steps - 1`，故最后一个是 59999 而非 60000）；单个实测约 12 G，合计约 144 G；日志与 GPU csv 落 `records/`；wandb run `robomme-framesamp/v2-1600ep-m8x8-modul-b128-60k`。

**跑完**：`result.md`（起止、EXIT_CODE、12 ckpt 齐全、loss 里程碑、稳态 s/step、util 均值 / 0% 占比 / 慢步分层、显存口径声明、下一步）；`records/` 收清洗日志、两份 GPU csv、两份 runner 脚本；不归档权重。`docs:` commit + push。

### D. 关键文件与复用

- 配置：`src/mme_vla_suite/training/config.py`（新增条目；`mme_vla_suite` 与 `mme_vla_suite_b128` 不动）；`src/openpi/training/optimizer.py::CosineDecaySchedule`（不动）。
- 训练入口：`scripts/training/train.py`（`init_history_config` 写 provenance）；环境变量解析 `src/mme_vla_suite/training/dataloader.py::_create_framesamp_dataset`。
- 路径源：`scripts/training/paths.sh`。
- 参数树核对：`scripts/training/legacy-eval/check_ckpt_param_tree.py`（只输出计数与 missing/extra/shape_mismatch，PASS 时后三者恒为空列表；modulation 的正面签名是 `n_model=61`）。
- F3 对拍：`scripts/training/g0/check_baseline_env.py`（dump / check）、`scripts/training/g0/compare_baseline.py`（只比交集并打印 `rows=`）、`scripts/training/tests/project_scalars.py`（`_HEADER` 强制写表头，故 TSV 行数 = 1 + 步数）、`scripts/training/g0/bench_train_steps.py::_make_digest_gate`（`BENCH_EXTRA_DIGEST_STEPS` 越界即 raise）。
- 留档样板：`docs/training-doc/awsprod40k-b128-motion/{launch,result}.md`、`docs/training-doc/v2b-read20-20260914T174147Z/`。

### F. 守卫放宽与增补验证（第四稿新增，第六稿重排顺序；对应执行顺序步骤 2–4）

**为什么要多跑。** `framesamp_dataset.py::FrameSampDataset.__init__` 的 `_req` 里有两条 commitV3.1（`6ee7494`，2026-08-27）按 Codex 审计 G13 加的守卫：`integration_type == "context"`、`memory_token_dim == 2048`，目的是挡住「同形的 modul 配置」。它们只在 `__init__` 拒绝配置，后续交付路径（`__getitem__`、`_pad`、store 读取）不读这两个键。放宽它们是对 dataloader 文件的改动，按 AGENTS 第 18 条要证明不改数，所以多跑 F2、F3 两块验证。`train.py` 主链路上没有其他 context 专属守卫；`g0/bench_train_steps.py` 与 `tests/` 下四个取证工具各有一份 context-only 白名单，本轮不经过（见第一部分第 6 点）。

**F1. 守卫改成成对白名单**（**第六稿注意：`b625f48` 的工作区里没有这段改动，必须重新写入**）：

```python
        _req((str(hc.integration_type), int(hc.memory_token_dim)) in {("context", 2048), ("modulation", 1024)},
             f"(integration_type, memory_token_dim)=({hc.integration_type!r}, {hc.memory_token_dim}) "
             f"不在支持的 (context,2048)/(modulation,1024) 档位中")
```
替换原来那两条相邻的 `_req`（`integration_type` 一条、`memory_token_dim` 一条），位置在 `perceptual_memory.type` 那条断言之后、`(budget, token_per_image, num_views)` 那条之前。context 对 `budget/token_per_image/num_views/feature dims/use_state_emb` 的其余断言原样保留。expert（`memory_token_dim 1024` 但 `integration_type expert`）与错配对（如 `modulation,2048`）仍拒。

**F1b. G13 测试改写** `scripts/training/tests/test_pack_guards.py::test_g13_modul_config_rejected` → 改为三条断言：`perceptual-framesamp-modul.yaml` 现在**通过**构造（4×4 迷你库、`(modulation,1024)`）；`perceptual-framesamp-expert.yaml` 拒；用 `OmegaConf` 把 modul.yaml 的 `memory_token_dim` 改成 2048 的错配对拒。该测试依赖 `v1-store/datasets/ref-shard` 迷你库，**本环境不存在**，pytest 跑不了；同样三条断言改在 1600ep 库上用一次性脚本执行（F2 里一并做），测试文件的改写只保证逻辑正确、在留档里写明「本环境未执行 pytest」。

**F2. 第一块：轻量对拍 + 三条守卫断言（CPU，约 3–5 分钟，≤5 分钟不触发 AGENTS 17）。** 一次性脚本放 scratchpad、`JAX_PLATFORMS=cpu uv run --no-sync`，cwd 仓库根：
- 守卫：`get_history_config` 分别加载 modul-8frame-8x8（期望通过）、expert（期望 `ValueError` 含「形制断言失败」）、modul-8frame-8x8 且 `memory_token_dim` 改 2048（期望拒）；另在 4×4 库 `framesamp/` 上加载 `perceptual-framesamp-modul.yaml`（期望通过）。
- 对拍：`_create_framesamp_dataset`（`src/mme_vla_suite/training/dataloader.py`）分别用 `perceptual-framesamp-modul-8frame-8x8.yaml` 与 `perceptual-framesamp-context-8frame-8x8.yaml` 在 `framesamp-8x8` 库上构造 Dataset（`MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 同正式 run），取 256 个索引（`np.random.default_rng(20260915).choice(605611, 256)` 并加 0、605610 与每个 episode 边界附近各 1 个），逐样本逐键比 `dtype / shape / raw sha256`。判定行 `DS_EQUIV=PASS samples=<n> keys=<k> mismatches=0`；任一不等即停。
- **判据声称范围（第六稿收窄）**：这证明的是「**在当前源码下**，两份配置在所选 256 个索引上交付的原始 Dataset 输出逐字节相同」。差异是 `integration_type` / `memory_token_dim` **外加 context 版多出的整节 `motion:`** 的合集（见第一部分第 6 点）。Dataset 之后不另做对拍，理由是源码级封闭：`transform_dataset` 只收 `data_config`，`RoboMMEDataConfig.create` 里 `RepackTransform` 的键表写死、`ModelTransformFactory` 只读 `model_config.model_type / max_token_len / action_dim / discrete_state_input`，`TorchDataLoader` 的 collate 与 `HistAugObservation.from_dict` 都不接收 `history_config`；全仓库这两个键除 `framesamp_dataset.py` 那两条守卫外只出现在 `src/mme_vla_suite/models/`，**`src/openpi/` 零出现**。两侧 `data_config` 与 `model_config` 其余字段完全相同，因此 Dataset 输出相同即最终进模型的 batch 相同。它**不比较接入前源码**。

**F2.5. 先 commit 再验（用户 2026-09-15 拍板，与第五稿顺序相反）。** F2 全 PASS 后立即：
```bash
git add src/mme_vla_suite/training/framesamp_dataset.py scripts/training/tests/test_pack_guards.py
git commit    # subject: commitV9.6: dataset 形制守卫放宽为 (integration_type, memory_token_dim) 成对白名单
              # body: F2 判定行原文 + 「F3 梯度等价验证待跑，FAIL 则 revert」
git push
git status --porcelain     # 必须为空，F3 要从这个 clean HEAD 起跑
```
理由见第一部分第 6 点第三条。**这个 commit 必须早于步骤 5 建 worktree 快照**，否则快照里还是旧守卫，正式 run 会被 `_req` 拒掉 modul 配置。

**F3. 第二块：复用 t8-c8-b 固化轨迹前 100 步（GPU 4,5，约 30–35 分钟，tmux `m8-guard`）。**

**第六稿不再说「照抄 t8-c8-b 命令只改四项」** —— 照抄会在启动阶段直接抛 `ValueError: BENCH_EXTRA_DIGEST_STEPS 越界（须在 0..99）: [199, 399, 599, 799]`（`bench_train_steps.py::_make_digest_gate` 按 `config.num_train_steps` 做范围校验，且该 gate 在 `checksum_on` 之后**无条件**构造，`BENCH_CHECKSUM` 碰不到它）。更糟的是 t8-c8-b 的命令体只有 `set -o pipefail`、**没有 `set -e`**：bench 退出 1 之后 shell 会继续跑完后处理并无条件 `printf 'TRAJECTORY_DONE …'`，`EXIT_CODE=${PIPESTATUS[0]}` 取到 printf 的 0 —— **日志尾行写 `EXIT_CODE=0` 而训练一步没跑**。所以 F3 写成独立完整命令，相对 t8-c8-b 共**六处**差异（第五稿只列了四处）。

摘要步集必须正好是 `1,2,24,49`：gate 自动含 step 0 与 `last_step=99`，得 `{0,1,2,24,49,99}`，**恰是基线在 0..99 区间的完整子集**；改成 `BENCH_DIGEST_INTERVAL=1` 写满 100 份没有意义（基线侧 2026-09-14 已冻结，多出的 94 份没有对照物）。`BENCH_DIGEST_INTERVAL=1000` 必须保留——`extra` 非空而 interval 未设是另一条 raise。

```bash
# —— 生成 runner（主树，普通 Bash）——
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
cat > "$MAIN/v1-store/logs/m8-guard-runner.sh" <<'RUNNER'
#!/usr/bin/env bash
set -o pipefail                 # 外层不开 -e：EXIT_CODE 必落盘
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=t8-c8-guard-s100
LOG="$MAIN/v1-store/logs/$RUN.log"
D8="$MAIN/v1-store/datasets/4task-motion-400ep"

body() {
  cd "$MAIN"                    # F3 验的是主树 HEAD 的守卫改动，不走 worktree
  source "$MAIN/scripts/training/paths.sh"
  export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1
  export UV_PROJECT_ENVIRONMENT="$MAIN/.venv"
  export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled
  unset JAX_PLATFORMS PYTHONPATH MMEVLA_MOTION_STORE
  unset BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR BENCH_SAVE_FINAL_CKPT BENCH_FINAL_STEP
  unset BENCH_PERF_MODE BENCH_DUMP_IDX BENCH_DATASET_IMPL BENCH_REF_SOURCE BENCH_REF_MANIFEST BENCH_REF_MOTION
  export BENCH_RECORD_DIR="$V1_STORE/bench/8x8/$RUN"
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000
  export BENCH_EXTRA_DIGEST_STEPS=1,2,24,49          # ★差异1：删 99,199,399,599,799（99 自动入集，>99 越界即 raise）
                                                      # ★差异2：不设 BENCH_SAVE_FINAL_CKPT / BENCH_FINAL_STEP

  test -z "$(git -C "$MAIN" status --porcelain)"      # ★差异3：F2.5 已 commit，这条现在为真（第五稿顺序下它恒假）
  test ! -e "$BENCH_RECORD_DIR"
  test ! -e "$V1_STORE/train-runs/$RUN"
  printf 'RUN=%s\nHEAD=%s\nSTART_UTC=%s\n' "$RUN" "$(git -C "$MAIN" rev-parse HEAD)" "$(date -u +%FT%TZ)"

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump \
    --record-dir "$BENCH_RECORD_DIR" --v1-store "$V1_STORE" \
    --source "$D8/source" --manifest "$D8/meta/episode_manifest.json" --dataset "$D8/framesamp-8x8" \
    --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check \
    --base "$V1_STORE/bench/8x8/t8-c8-b" --record-dir "$BENCH_RECORD_DIR" \
    --steps 100 --batch-size 8 --dataset "$D8/framesamp-8x8"   # ★差异4：去掉 --allow-difference（同一个库，指纹应严格全等）

  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name "$RUN" \
    --assets-base-dir "$V1_STORE/train-assets" \
    --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep" --data.assets.asset-id robomme \
    --checkpoint-base-dir "$V1_STORE/train-runs/$RUN" --batch-size 8 --num-workers 4 \
    --num-train-steps 100 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
    --dataset-path "$D8/framesamp-8x8" \
    --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
    --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/project_scalars.py \
    "$BENCH_RECORD_DIR/metrics.jsonl" "$BENCH_RECORD_DIR/scalars_hex.tsv"
  # ★差异5：删掉 t8-c8-b 末尾读 param_checksums.jsonl 的性能 heredoc 与 nvidia-smi 密采（性能不是本轮判据）
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/compare_baseline.py \
    "$V1_STORE/bench/8x8/t8-c8-b" "$BENCH_RECORD_DIR"
  # ★差异6：判据改走 compare_baseline.py（天然只比交集并打印 rows），不手工 head 比行数
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
RUNNER
chmod +x "$MAIN/v1-store/logs/m8-guard-runner.sh"
tmux ls
tmux new-session -d -s m8-guard "bash $MAIN/v1-store/logs/m8-guard-runner.sh"
tmux has-session -t m8-guard && echo "SESSION_UP=m8-guard"
```

Monitor：
```bash
tail -n +1 -F /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/t8-c8-guard-s100.log \
  | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "BASELINE_ENV=|SCALARS |INDEX_SEQ|BATCH_DIGEST|STATE_DIGEST|DET_CHECK|EXIT_CODE=|Error|Traceback|越界|unbound variable"
```

**判据（第六稿：全部带量词，不再写无条件的 `mismatches=0`）**：
- `BASELINE_ENV=PASS` —— 指纹 preflight（jax 0.5.3、A100 驱动 595.71.05、norm_stats `750a8e9b…`、pi05_base 与库摘要逐项同）。**注意 `env.json.fingerprint` 不含任何 commit/代码 sha**，所以 F1 改了 `framesamp_dataset.py` 不会让它 FAIL——它保的是环境不是代码。基线 `v1-store/bench/8x8/t8-c8-b/` 的 `BASELINE_MANIFEST.json` 所列 10 个产物已逐个核过 sha，10/10 未腐烂。
- `SCALARS steps=100 keys=5 hex_mismatch_steps=0` —— **主判据**，100/100 步全覆盖（五标量位于 batch 下游，batch 变了 loss 必变）。
- `INDEX_SEQ=PASS n=800` —— 100 步 × 8 个样本索引逐个一致。
- `BATCH_DIGEST rows=6 mismatch=0` —— 基线在 step 0..99 只固化了 `{0,1,2,24,49,99}` 六份，**这是基线的既有取证密度，不是本轮覆盖不足**；`rows` 必须写进判定行，不得省略。
- `STATE_DIGEST rows=6 mismatch=0` —— `BENCH_CHECKSUM=1`（用户拍板）下六份完整 TrainState 逐叶摘要对照。若将来改回 `0`，`compare_baseline.py` 对不存在的 `param_checksums.jsonl` 返回空 dict、会打印 `rows=0 mismatch=0` 并汇入 `DET_CHECK=PASS`，那是**结构性空判**，必须显式标注。
- `CANON_CHECK=PASS steps=6`。
- 汇总判定行：`GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=6 state_digest_rows=6`。
- 旁证（可选）：`cmp <(head -n 101 docs/training-doc/t8-c8-b/records/scalars_hex.tsv) "$BENCH_RECORD_DIR/scalars_hex.tsv"` —— **101 = 1 行表头 + step 0–99**，候选文件恰好 101 行，整文件对比。第五稿写的「前 100 行」会静默漏掉 step 99（该 TSV 实测 1001 行 = 表头 + 1000 步，表头由 `project_scalars.py` 的 `_HEADER` 强制写入）。

**F3 留档（用户 2026-09-15 拍板；第五稿的「≤15 分钟不建独立留档」作废）**：F3 实际 30–35 分钟 > AGENTS 17 的 5 分钟阈值，且它的定位正是 AGENTS 18 第二块的**正式等价证据**，不能借用 09-14 那条明写「不作正式等价证据」的排错例外。仓库里四个同类先例（`aws-t2-cand-s100`、`aws-t2-ref-s100`、`aws-t3-closed-s100`、`aws-t3-open-s100`）都是三件套齐全——留档才是惯例。建 `docs/training-doc/t8-c8-guard-s100/`：
- `launch.md`：起跑 commit（= `commitV9.6`）、完整命令、与 t8-c8-b 的六处差异、`BASELINE_ENV` 输出原文；
- `result.md`：`compare_baseline.py` 全部判定行原文、墙钟、结论；
- `records/`：`metrics.jsonl`、`scalars_hex.tsv`、`batch_digests.jsonl`、`param_checksums.jsonl`、`index_sequence.json`、`env.json`、`run_meta.json`（约 0.5 MB）；
- `docs/training-doc/README.md` 加一行。体例照抄 `docs/training-doc/aws-t2-cand-s100/`。

**清理（顺序写死：先拷后删，否则删完就没了）**：
```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
mkdir -p "$MAIN/docs/training-doc/t8-c8-guard-s100/records"
cp "$MAIN"/v1-store/bench/8x8/t8-c8-guard-s100/{metrics.jsonl,scalars_hex.tsv,batch_digests.jsonl,param_checksums.jsonl,index_sequence.json,env.json,run_meta.json} \
   "$MAIN/docs/training-doc/t8-c8-guard-s100/records/"
ls -ld "$MAIN"/v1-store/bench/8x8/t8-c8-guard-s100 "$MAIN"/v1-store/train-runs/t8-c8-guard-s100 "$MAIN"/v1-store/cache/jax/t8-c8-guard-s100
rm -rf "$MAIN"/v1-store/bench/8x8/t8-c8-guard-s100 "$MAIN"/v1-store/train-runs/t8-c8-guard-s100 "$MAIN"/v1-store/cache/jax/t8-c8-guard-s100
tmux ls; tmux kill-session -t m8-guard; tmux ls
```

**F4. 留档 commit**：`git add` 留档三件套 + README 加行 → `docs: t8-c8-guard-s100 守卫等价验证留档` → push。
**F3 FAIL 的处置**：立即 `git revert` commitV9.6（subject `revert: 撤销 commitV9.6 守卫放宽（GUARD_GRAD_100 FAIL）`）+ push，把 FAIL 判定行写进 `t8-c8-guard-s100/result.md`，停下交用户处置，**不放宽判据**。

### G. 隔离机制：detached worktree 快照（步骤 5 建、步骤 6/7 用、步骤 8 删）

**建**（步骤 5，`docs:` commit push 之后、主树 clean；`TRAIN_HEAD` 必须已包含 `commitV9.6`）：
```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
test -z "$(git status --porcelain)"
TRAIN_HEAD=$(git rev-parse HEAD); echo "TRAIN_HEAD=$TRAIN_HEAD"       # 写进 launch.md
git worktree add --detach v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k "$TRAIN_HEAD"
git worktree list                                                      # 确认新行且为 detached
```
`v1-store/worktrees/` 已有先例（`official-89efeaab`），整体不进 git。快照里没有 `v1-store/`、`.venv`、`third_party/` 子模块（数据 / 权重 / checkpoint 全走主树 `V1_STORE` 绝对路径，venv 由 `UV_PROJECT_ENVIRONMENT` 指主树，`third_party` 训练链路不 import）。**但快照里有 `packages/openpi-client/src`**（仓库跟踪的普通目录，不是 submodule），它必须进 `PYTHONPATH`——见下。

**注意 `git worktree list` 当前实有五条**（主树、`.claude/worktrees/b128cfg`、`.claude/worktrees/sgab`、`v1-store/reports/tic-adversarial-20260908/source`、`v1-store/worktrees/official-89efeaab`），不是第五稿以为的一条。删除时只 `remove` 本轮这一个，`prune` 不得波及其余。

**用**（B、C 节命令已改）：**四件事缺一不可**（第五稿只列了三件，漏了第 3 件，第 2 件也不完整）——
1. `cd $WT`（`get_history_config` 按 cwd 相对路径读快照 YAML）；
2. `PYTHONPATH=$WT/src:$WT/packages/openpi-client/src`（`mme_vla_suite`、`openpi`、**`openpi_client`** 三个包都从快照 import，压过主树 `.venv` 里 `_editable_impl_openpi.pth` 与 `_editable_impl_openpi_client.pth` 的绝对路径）；
3. `--data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"`（条目里是仓库相对路径，按 cwd 解析，`cd $WT` 后会落空并在 `norm_stats["state"]` 抛 `TypeError`）；
4. `UV_PROJECT_ENVIRONMENT=$MAIN/.venv` + `uv run --no-sync`（不让 uv 在快照里另建 venv）。

起跑前 `IMPORT_ORIGIN=PASS` 断言**四个路径**（三个包 origin + `train.py`）都在 `$WT` 下——第五稿那段脚本只查两个包，且 `train.py` 那项因 cwd 已是 `$WT` 而恒真、提供零信息，第六稿加了 `assert tp.is_file()` 让它至少验证存在。`launch.md` 记 `TRAIN_HEAD`、`WORKTREE`、四个 origin 原文、两个 `.pth` 的 `cat` 输出。主树期间可以照常 commit / push 文档，快照始终停在 `TRAIN_HEAD`——**也正因如此，B/C 命令里的 `TRAIN_HEAD` 只能粘 40 位 sha 字面量，写 `$(git rev-parse HEAD)` 会让断言 100% 失败**。

**禁**：训练期间不 `git worktree remove` / `prune`、不在快照里改文件、不 `git checkout` 快照；`git clean -x` 全仓禁止（AGENTS 19 附）。

**删**（步骤 8，`result.md` 提交之后）：`git worktree remove v1-store/worktrees/train-v2-1600ep-m8x8-modul-b128-60k && git worktree prune && git worktree list`。删前 `tmux has-session -t m8-prod` 必须已不存在。

**已知残余**：`scripts/training/paths.sh` 用的是主树那份（按 `BASH_SOURCE` 解析 `REPO_ROOT`，只导出路径变量、不 mkdir、不含训练逻辑）；`third_party/` 子模块快照里未初始化，训练链路不 import（已 grep 确认）。两者写进 launch.md。（第五稿把 `openpi-client` 列在这里并称「训练不 import」，**是错的**，已改为进 `PYTHONPATH`，见第一部分第 8 点。）

**两条 shell 陷阱**（`paths.sh` 第 21 行是 `set -euo pipefail`，`source` 后持续作用于调用方 shell）：一是 `set -e` 会让训练失败时 shell 当场退出、`EXIT_CODE=` 写不进日志，故 B/C/F3 一律采用 runner 的内外分层；二是 `paths.sh` 有一整片 `readonly`（`V1_STORE` 等），同一个 shell 里二次 `source` 会以「`REPO_ROOT: readonly variable`」报错退出、且错误信息与真实问题毫无关系，把 `source` 关进 `body()`（管道 → 子 shell）即可免疫。另：`V1_STORE` 等是 `readonly` 但**未 export**，只在当前 shell 展开有效；将来把训练命令拆到子脚本里再引用会拿到空串 + `set -u` 报错，届时一律用 `$MAIN/v1-store/...` 绝对路径或重新 source。

### H. 审计结论固化（第一部分第 7 点的落点）

结论正文见第一部分第 7 点。固化位置：根目录计划文件（I 节）第一部分同款一节；`launch.md`「与官方 / 上次 run 的关系」一节引用该节并写明「modulation 关闭态无运行时对拍，本 run 不据此宣称与接入前逐位等价」。不改 `docs/motion-memory.md` 等正本（评审性结论，非链路事实；正本改动另立）。

### I. 计划固化到仓库根目录（步骤 5）

新建 `v2-1600ep-m8x8-modul-training-plan.md`（与 `8frame-8x8-training-plan.md` 同级、同体例）：内容 = 本计划文件全文（去掉 harness 版本注，保留第一 / 第二部分与 F、G、H 节），文首加一行状态说明「实施过程档案；结果以 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/` 为准」。与 `launch.md` 初稿、`docs/training-doc/README.md` 加行一起 `git add` 三个文件，subject `docs: v2-1600ep-m8x8-modul-b128-60k 计划固化与起跑留档初稿`，push。之后再记 `TRAIN_HEAD`（保证快照里含这份计划）。验证：`git diff --check`；Markdown 链接 `docs/training-doc/README.md` → 新 run 目录可解析。

### E. 红线与不做的事

- 不改 dataloader / 模型代码（AGENTS 18 不触发）；不动既有两个条目。
- 训练从 worktree 快照起跑（G 节），主树期间的 commit 不影响在跑训练；但仍不在快照里改任何文件，不建库、不起第二个 GPU 大任务。
- tmux 只按确切名 kill，本轮起过的会话清单是 **`m8-guard`（F3）、`m8-smoke`（B）、`m8-prod` / `m8-prod-dense`（C）** 四个，清理时以这份清单为唯一依据，删前删后各一次 `tmux ls`。**不在清单内的一律不动**，含用户会话 `0`、`1`、`claude-private`、`codex`、`codex-repo`，以及他轮评估会话 **`eval-vla0914-full1600-best-a`**（2026-09-15 02:50 起，正占用 GPU 0，第五稿的清单漏了它）。禁 `tmux kill-server` 及一切全局杀法（AGENTS 7 红线）。
- 评估（legacy-eval / motion-variance 对 modulation + 8×8 的 prefix 长度、VideoRepick 驱动）不在本轮范围。
