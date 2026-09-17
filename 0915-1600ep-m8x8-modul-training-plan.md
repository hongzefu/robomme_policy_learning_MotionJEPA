# 1600 集新库 · 8 帧 8×8 · modulation · 无 motion · b128 60k · lr 5e-5 正式训练计划

> **实施过程档案**（2026-09-15，用户批准）。结果以 [`docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/`](docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md) 为准；本文按 `AGENTS.md` 第 2 条分两部分，另含 F（守卫放宽与增补验证）、G（训练锁主副本 + `-temp` 开发副本）、H（审计结论）、I（本文固化）、J（motion 接入前后的 modulation 梯度对拍）五节。
>
> **进度（2026-09-17 收尾完成）**：步骤 1–11 均已完成。正式训练从 clean HEAD `dd07f18fc385b01eb52db7563fe5f202997b9706` 于 `2026-09-15T05:48:27Z` 启动，已于 `2026-09-16T20:53:31Z` 正常结束，`EXIT_CODE=0`，60k 步耗时 **39h05m04s**。12 个 checkpoint 全部落盘，最终 `59999` 参数树 61/61 精确匹配；V10 复查主副本干净且三份源码 SHA 未变。主副本已解锁并快进同步开发副本的三个文档提交，最终训练留档 `9f7f423` 已推送；开发副本已按用户批准删除，主数据及全部 checkpoint 验收未变。最终结果见 [`result.md`](docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/result.md)，下文的预计耗时、起跑命令和训练期间规定保留为历史计划。

环境判定：**环境 B（AWS 单机）**。仓库主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，8 × A100-SXM4-80GB。无 turbo、无 GreatLakes。起跑 commit 见 `TRAIN_HEAD`（执行顺序步骤 8 记下）。

**起跑前实况（2026-09-15 核对）**：GPU 4–7 空闲（`memory.used` 均为 0 MiB），但 **GPU 0 正在被另一个任务占用**（util 100%、3867 MiB），对应 tmux 会话 `eval-vla0914-full1600-best-a`（2026-09-15 02:50 起）。本轮只用 4–7，不受影响，但 **E 节的 tmux 清单必须把这个会话计入「不得触碰」**。`/scratch` 余 **1.3 T**（`/dev/md0` 6.9 T，已用 5.6 T）。

用户已拍板（2026-09-15）：GPU **4,5,6,7**；wandb **开启**；batch **128**；步数 **60k**；**lr peak = decay = 5e-5**（b128 下等效为官方 5e-5@b64 的一半，不做线性缩放）；改动落在**新配置条目 `mme_vla_suite_b128_60k`**。run_name **`v2-1600ep-m8x8-modul-b128-60k`**（用户在本轮已再次确认，起跑前核实目录仍不存在）。

---

## 第一部分（给人看）

### 这轮做什么

在 2026-09-14 建成的 1600 集新库上，训练第一条 **8 帧 × 8×8 + modulation + 无 motion** 的模型。它是后续所有对照的基线 —— 上一条生产 run `awsprod40k-b128-motion` 用的是旧 400 集库、32 帧 × 4×4、context + motion，两条**不逐项可比**。

### 拍板的参数

| 项 | 值 | 备注 |
|---|---|---|
| GPU / batch | 4,5,6,7 四卡 / 128 | per-device 32，fsdp4 |
| 步数 / lr | 60k / peak = decay 5e-5 | warmup 5k。b128 下不做线性缩放，等效步长是 1e-4 的一半，用延长步数补偿 |
| 与官方对照 | 官方是 b64 / 80k / 5e-5 / warmup 10k | 曲线形状相同（线性升后恒定）。本 run 总样本 7.68M ≈ 新库 12.7 epoch，官方 5.12M |
| 配置条目 | 新增 `mme_vla_suite_b128_60k` | 既有两个条目一字不动 |
| 预计 | **约 64.5 h（2.7 天）** | 稳态 3.831 s/step（`bench-b128-util` 实测，util 均值 94.75%） |
| 产出 | 12 个 checkpoint ≈ **144 G** | 单个 12 G 实测；磁盘余 1.3 T |
| wandb | 开 | project `robomme-framesamp` |

run_name **`v2-1600ep-m8x8-modul-b128-60k`**（用户在本轮已再次确认，起跑前核实目录仍不存在）。

### 要改的东西：两个配置文件 + 两处代码

- **配置**：新 YAML `perceptual-framesamp-modul-8frame-8x8.yaml`（以 modul.yaml 为底，只把 `token_per_image` 16 → 64）、新条目 `mme_vla_suite_b128_60k`。（A 节）
- **代码一**：dataloader 的形制守卫要放宽 —— 现在它硬性只收 `context` / `2048`，会把 modul 配置挡在构造期。改成成对白名单 `(context,2048)` / `(modulation,1024)`。（F 节）
- **代码二**：新增起跑前自检脚本 `preflight_train_launch.py`。（G 节）

### 三件需要你知道的事

**① 守卫放宽要配两块验证。** 那两条守卫只在构造时拒配置，后面的取样、填充、store 读取一行都不读这两个键 —— 但它毕竟是 dataloader 改动，按 AGENTS 第 18 条要证明「不改数」，所以配了 V3（轻量对拍）和 V4（真实训练梯度一致）。**顺序是先 commit 再验**，因为 AGENTS 12/17 要求 clean HEAD 起跑；V4 不过就 `git revert`。细节见 F 节。

**② motion 接入不影响这条不用 motion 的 run。** 逐层核过源码：motion 的新参数在关闭态**根本不创建**，`embed_memory` 是编译期早返回，cross-attention 本体零改动，还有两道显式闸挡住误配。**源码级成立**，完整论证见 H 节。但这只是「读代码读出来的」，所以还要真跑一遍 —— 见下一节。

**③ 训练期间主副本锁死只读，开发搬到 `-temp` 副本。** 训练跑 65 小时，而 Python 在 dataloader worker 重建、存 checkpoint 时会**重新 import 源文件** —— 主副本被改就可能把新代码带进正在跑的训练。所以起跑后 `chmod -R a-w src scripts packages`，开发全在 `git clone` 出来的 `-temp` 里做（它有自己的 `.venv`，`v1-store` 是指向主副本的 symlink）。细节与三条红线见 G 节。

### 关于那个「追溯对拍」（执行顺序第 5 步）

**要回答的问题**：今天跑出来的这个模型，和「motion 功能还没被加进代码之前」跑出来的，是不是同一个东西？

**为什么非问不可**：这条 run 是后续所有对比的基线。基线本身如果混进了不该有的变化，后面拿它比 motion、比不同布局、比不同接入方式，差异就说不清是哪来的。

**难在哪**：最直觉的做法是把两个版本的代码各跑一次训练，比结果。但老代码**读不了** 8×8 格式的库 —— 它有四道硬性检查（库的格式名、表名、文件名、形状），全部对不上。要让它能读，就得把新的读取代码搬过去，而那部分代码恰恰动了八处真正读数据的地方（读多少字节、从哪个偏移开始读、怎么重排）。**为了做实验，先把对照组污染了**，这就本末倒置。

**怎么绕开**：模型其实**不关心**是「8 帧 × 每帧 64 个 token」还是「32 帧 × 每帧 16 个」—— 它只看到一个 512 × 2048 的张量，两种布局算出来的形状完全一样。所以不走数据库那条路：把一份**事先存好的** batch 直接喂给两个版本的模型，各跑一次前向和反向，比 loss 和每个参数的梯度。老代码根本不需要认识新格式的库，只要能吃下这个张量就行。

好消息是这份 batch **现成**（2026-09-14 存的，两档各 92 M，一直在盘上），不用重新生成。

**那为什么 2026-08-29 的老代码也能做 8×8 那一档？**（8×8 支持是 2026-09-14 才加的，老代码比它早半个月）—— 因为它**根本不需要「支持 8×8」**。两档 fixture 的张量实测如下：

| 键 | 32 帧 4×4 | 8 帧 8×8 |
|---|---|---|
| `static_image_emb` | `[8,512,2048]` bfloat16 | **完全相同** |
| `static_pos_emb` | `[8,512,768]` float32 | **完全相同** |
| `static_mask` | `[8,512]` bool，sha `906b1856…` | **连 sha256 都逐位相同** |

形状、dtype 全同，掩码甚至一模一样（两档都是 512 位全有效）；**只有 `static_image_emb` / `static_pos_emb` 的内容不同**。老代码那条断言从 2026-02-20 起一直是「总数必须等于 512」，**从不检查这 512 个 token 是怎么分帧的** —— 分帧只是取数据时的事，而那一步被我们跳过了。

**由此有一个要诚实说明的推论**：两档在模型侧走的是**同一段代码**，不是两条不同路径。所以做两档的价值是「用两组不同的真实数值各验一遍」（万一某处对特定数值敏感 —— 溢出、NaN、极端量级 —— 多一层保险），而**不是**「覆盖了两条代码路径」。严格说，**一档就足以证明代码等价**。时间紧时可以只做 32 帧 4×4 那档：它是历史原生布局，与老代码同源，风险最低。

#### 锚点 `07702f0` 是什么，它卡在哪两个改动之间

本来只打算比 motion 接入前后那两个版本（`c5925d9` vs `06220c4`）。核实之后发现可以把对照侧一直往前推到 **`07702f0`（2026-08-29，commitV4.3「模型侧单一化」）**。

**这个 commit 本身干了什么**：把模型侧的旧分支一次性删干净 —— recurrent（RMT / TTT）、symbolic、tokendrop 三套配置与代码分支全删（十几个 YAML，共 400 多行），同时把 `compute_loss` 的返回从 `(loss, stats)` 二元组改成**单返回**。

**为什么卡在这个点**：因为 HEAD 版的对拍工具是按「单返回」写的。再往前一个 commit（`a879763`）`compute_loss` 还返回二元组，工具跑上去直接报错。所以 `07702f0` 是一条**天然的零适配分界** —— 从它起，工具不用改就能在老代码上跑；比它更早，就得额外适配。

**锚点之后到今天，模型代码究竟动过几次**：区间内有 18 个 commit 碰过 `src/`，但碰过 `src/mme_vla_suite/models/` 的只有 4 个，其中**三个是纯加 YAML 文件**：

| commit | 日期 | 对 `models/` 的改动 | 性质 |
|---|---|---|---|
| `c5925d9` V6.4 | 09-03 | +1 YAML（39 行） | 纯新增文件 |
| **`06220c4` V6.5** | **09-03** | **改 `.py`** | **motion 接入，区间内唯一一次** |
| `99faacb` V9.0 | 09-14 | +2 YAML（74 行） | 纯新增文件 |
| `e0bcb45` V9.5 | 09-15 | +1 YAML（24 行） | 纯新增文件 |

也就是说，**整个区间里真正动过模型代码的只有 motion 接入这一次**，其余全是往目录里放新配置文件。所以一次对拍覆盖整条链，归因也是唯一的 —— 万一比出差异，责任 commit 不用二分就能确定。

**`06220c4` 具体改了什么，为什么关闭态不受影响**（完整论证在 H 节，这里只列三类）：

1. **新增两个参数层**（`motion_pos_proj`、`motion_encoder_static`）—— 但它们只在 motion 开启时才创建。关闭态下这两行根本不执行，所以参数树和随机数的消耗顺序都不变（后者很关键：flax 的随机数流按调用顺序递进，中间多消耗一次就会让后面所有权重的初始值全变）。
2. **`embed_memory` 多传三个参数** —— 但关闭态下它们恒为 `None`，而且被调用方在提前返回之前一个字都不读它们。
3. **新增两道检查** —— 关闭态下一道条件为假、一道被短路，都不触发。

而 modulation 真正用的那部分（cross-attention 本体，`history_gemma.py` 里的 `MemoryAttention` / `MemoryRMSNorm`）**四个文件 `git diff` 为空** —— motion 接入根本没碰它。

**再往前推为什么没意义**：`07702f0` 往前到 2026-02-20 那半年，`src/mme_vla_suite/models/` 只有三个文件各一行的差异，而且都跟 framesamp modulation 无关、后来还都在 `07702f0` 被删掉了。`history_gemma.py` 这套 modulation 本体更是从 2026-02-06 起**逐字未变**。再往前（早于 `bc3ab59`，2026-02-20）配置形制就不同源了 —— 那时 YAML 用的是 `perceptual_memory.budget` 而不是顶层 `budget`，两边压根不是同一份配置写法，没法比。

**必须先做的一件事**：以前所有这类逐位比对**都只做过 context 模式**，modulation 这条路从来没验证过「同样的输入跑两次，结果是不是完全一样」。如果不先确认这点，万一比出差异，就分不清是「motion 接入真改了数」还是「这条路本身每次跑就有微小抖动」。所以先在**同一个版本上跑两次**；这一步不过，就停下来问你，再决定判据要不要从「逐位相同」放宽成「数值阈值内」。

**代价**：约 **20–35 分钟、两张卡**；要写 200–250 行工具代码，但**不碰任何核心代码**（两个版本都跑各自原封不动的历史代码，工具只负责「把存好的 batch 读回来」和「多记一份初始权重的指纹」）。

两档都做：32 帧 4×4 与 8 帧 8×8。细节见 J 节。

### 与上次生产 run 的差异

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

### 起跑前要过十道验证

判据行全部摘进 `launch.md`；任一 FAIL 即停、把原文交用户处置，不放宽判据。

| # | 验证 | 步骤 | 证明什么 | 判据行 | 耗时 / 资源 |
|---|---|---|---|---|---|
| V1 | 配置自检 | 1（已完成） | 新 YAML 与新条目的键值正确、lr 曲线形状对、既有两个条目未被碰 | `CONFIG_OK` | 2 min / CPU |
| V2 | 守卫三条断言 | 3 | 放宽后的成对白名单确实只放行 `(context,2048)` 与 `(modulation,1024)`，expert 与错配对仍拒 | 三条断言各自 PASS | 含在 V3 内 |
| V3 | Dataset 轻量对拍 | 3 | **守卫改动不改交付内容**：modul-8x8 与 context-8x8 两份配置在同一库、同一组 256 个索引上，逐样本逐键的 dtype / shape / raw sha256 全等 | `DS_EQUIV=PASS samples=<n> keys=<k> mismatches=0` | 3–5 min / CPU |
| V4 | 守卫梯度一致 | 4 | **守卫改动不改训练数值**：在 `t8-c8-b` 固化轨迹上重跑前 100 步，逐步五标量 hex、样本索引序列、输入摘要、TrainState 逐叶摘要与基线全等 | `GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=6 state_digest_rows=6` | 30–35 min / GPU 4,5 |
| V5 | modulation A/A 自复现 | 5（V6 前置） | **modulation 路径本身是否 bit 级可复现** —— 同一棵树同一命令连跑两次比逐叶梯度。仓库全部逐位对拍清一色是 context，这条路径从未在确定性档下验过 | `GRAD_EQ=PASS`（A1 vs A2） | 20–35 min / 2 卡 |
| V6 | 追溯梯度对拍 | 5 | **从 `07702f0`（2026-08-29）到 HEAD，modulation 关闭态的数值一字未变** —— 同一份固定 batch、同一初态，两版代码的 loss 与全部可训练叶梯度逐位相同。一次覆盖整条演进链（motion 接入 `06220c4` 只是其中一个 commit）。8×8 与 4×4 两档各做一次 | 两侧初态逐叶 sha 相同 + `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0` | 20–35 min / 2 卡 |
| V7 | smoke 20 步 | 7 | **这个配置能不能真跑起来**：modulation × 8×8 能编译、出有限数、per-device 32 不 OOM；norm_stats 正确加载并随 checkpoint 落盘 | `EXIT_CODE=0` + `Integration Type: modulation` + 20 步有限 + norm_stats sha `856c75ea…` | ≤5 min / GPU 4–7 |
| V8 | 参数树核对 | 7 | **存下来的确实是 modulation 模型**：checkpoint 与 modulation 配置构造出的模型参数树双向精确匹配，六个 modulation 专属叶子都在 | `PARAM_TREE_EXACT=PASS … n_model=61 n_ckpt=61 missing=0 extra=0` + `MEM_PARAMS=PASS n=6` | 1 min / CPU |
| V9 | 起跑前自检 | 7、9 | **训练从对的地方、用对的环境、拿对的数据起跑**：25 项，含五条专防「在 `-temp` 开发副本里误起训练」 | `PREFLIGHT=PASS n=25` | 5 s / CPU |
| V10 | 只读复查 | 11 | **训练期间主副本代码一个字节都没变** | `git status --porcelain` 为空 + 三个关键文件 `sha256sum -c` 全对 | 1 min / CPU |

**分属四组目的，别混为一谈**：V2–V4 管「守卫改动等价」（AGENTS 18 的两块），V5–V6 管「motion 接入等价」，V7–V8 管「这条 run 本身跑得对」（功能性确认，不是等价性证明），V9–V10 管「训练读的代码确实是起跑那一刻的」。覆盖边界与每项细节见第二部分对应节。

### 执行顺序（十一步，全文以本表为准）

1. 新 YAML + 新条目 → 验证 → `commitV9.5` → push。**（已完成，`e0bcb45`）**
2. **F1/F1b**：重新写入守卫成对白名单 + G13 测试改写（只改工作区，不提交）。
3. **F2**：CPU 轻量对拍 + 三条守卫断言（3–5 分钟，≤5 分钟不触发 AGENTS 17）→ `DS_EQUIV=PASS`。
4. **F2.5 → F3**：先 `commitV9.6` + push，工作区回到 clean；再从该 clean HEAD 起跑 `t8-c8-guard-s100`（tmux `m8-guard`，30–35 分钟，`BENCH_CHECKSUM=1`）→ `GUARD_GRAD_100=PASS` → 建 `docs/training-doc/t8-c8-guard-s100/` 三件套 + README 加行 → `docs:` commit → push。**FAIL 则 `git revert` commitV9.6 + push，停下交用户处置。**
5. **modulation 关闭态的追溯梯度对拍**（用户 2026-09-15 拍板：放在正式 run 起跑前做）。锚点 `07702f0`（2026-08-29）对 HEAD，**一次覆盖整条演进链**（motion 接入 `06220c4` 只是其中一个 commit）；两档都做：32 帧 4×4 与 8 帧 8×8。走「固定 batch 喂两版模型」，**不走**「两棵源码树各起训练」。**先跑 A/A 自复现**（V5）确认 modulation 路径 bit 级可复现，不过即停、请示后再决定判据是否降级。详见 J 节。
6. 本计划固化为根目录 `0915-1600ep-m8x8-modul-training-plan.md` + `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md` 初稿 + `docs/training-doc/README.md` 加行 → `docs:` commit → push。（I 节）
7. 从**主副本**起 smoke（tmux `m8-smoke`）→ 核判据（含 `PREFLIGHT=PASS`）→ 删临时产物。（B 节）
8. `launch.md` 补 smoke 与步骤 3/4/5 的判定行摘录 → `docs:` commit → push → **此刻记 `TRAIN_HEAD` = 主副本 HEAD**（抄进 launch.md）→ `git clone` 建开发副本 `-temp` + symlink v1-store + `uv sync`。（G 节）
9. 从**主副本**起正式 run（tmux `m8-prod`）→ `PREFLIGHT=PASS` 且训练确认进入稳态后，立即 `chmod -R a-w src scripts packages` 锁死只读 → 挂 Monitor。（C 节）
10. 训练期间（约 65 h）：**一切开发、验证与留档编辑都在 `-temp` 里做**，主副本只读不写；禁止在主副本跑 `uv sync` / `uv add` / `uv pip`。
11. 跑完：`chmod -R u+w src scripts packages` 解除只读 → 复查 `git status --porcelain` 仍为空且关键文件 sha 与起跑时一致（写进 `result.md`）→ `result.md` + `records/` → `docs:` commit → push → 视情况删除 `-temp`。评估另立任务。

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

run_name `smoke-m8x8-modul-$(date -u +%Y%m%dT%H%M%SZ)`，tmux `m8-smoke`，日志 `v1-store/logs/m8-smoke.log`。**从主副本起跑**（G 节），此时尚未锁只读。

**为什么命令是这个形状。** 四条实测教训，写法直接粘贴到别处时容易踩中，其中一条会写出假成功：

1. 块内从未给 `RUN` / `LOG` 赋值，而 `paths.sh` 开头是 `set -euo pipefail` —— `set -u` 下第一次展开 `$RUN` 就 `unbound variable` 退出。
2. `set -e` 会持续作用于调用方 shell，于是 `uv run ... | tee -a "$LOG"` 失败后 shell **当场退出**，后面那行 `echo "EXIT_CODE=${PIPESTATUS[0]}"` 永远执行不到。
3. 最"自然"的修法 `cmd | tee log || true` **绝不可用**：实测 `|| true` 会把 `PIPESTATUS` 冲成 `(0)`，`EXIT_CODE=` 稳定打印 0，把失败 run 记成成功。
4. `nvidia-smi ... > csv &` 只重定向了 stdout，stderr 仍挂在 body 的管道上；采样器未被杀时 `tee` 因写端未全关而不退出，`EXIT_CODE=` 同样写不进日志。

所以采用 `docs/training-doc/t8-c8-b/launch.md` 已跑通的**内外分层**：`set -e` 活在 `body()` 里（保住断言硬闸），`tee` + `EXIT_CODE` 活在 body 外（保住必落盘），`trap ... EXIT` 收采样器。命令落成 runner 脚本再交给 tmux，而不是内联进 `tmux new-session "..."` 字符串。把 `source` 关进 `body()`（管道 → 子 shell）还顺带免疫了 `paths.sh` 那片 `readonly` 导致的「重跑时 `REPO_ROOT: readonly variable` 报错退出、错误信息与真实问题毫无关系」。

> 注：`docs/training-doc/awsprod40k-b128-motion/launch.md` 那份历史留档**带同款缺陷**（`$LOG` 未赋值 + `set -e` 下 `EC=${PIPESTATUS[0]}` 拿不到），它留下 `EXIT_CODE=0` 纯粹因为那次训练成功了。**不要拿它当模板。**

**前置**：主副本 clean 且 HEAD 含 `commitV9.6`；GPU 4–7 显存 0；run 根不存在。除 GPU 外全部由 preflight 断言，GPU 那条写成 `body()` 里的 `test`。

**B-0. 生成 runner（在主副本用普通 Bash 执行，不进 tmux）**

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
mkdir -p "$MAIN/v1-store/logs"
cat > "$MAIN/v1-store/logs/m8-smoke-runner.sh" <<'RUNNER'
#!/usr/bin/env bash
# 外层：故意不开 -e，保证 EXIT_CODE 在任何死法下都落盘（t8-c8-b「外层 tee + EXIT_CODE」口径）
set -o pipefail

RUN="${1:?用法: m8-smoke-runner.sh <RUN> <TRAIN_HEAD> <HC_SHA256>}"
TRAIN_HEAD="${2:?}"
HC_SHA="${3:?}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
LOG="$MAIN/v1-store/logs/m8-smoke.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
mkdir -p "$(dirname "$LOG")"

body() {
  # 内层：-euo pipefail 就位，下面每条 test 都是「不满足就不起跑」的硬闸
  cd "$MAIN"                                 # 训练就跑在主副本上，不再有 worktree
  source "$MAIN/scripts/training/paths.sh"   # 只导出路径变量；不 mkdir；按 BASH_SOURCE 解析 REPO_ROOT

  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$(date -u +%FT%TZ)"

  unset PYTHONPATH                           # 靠主副本 .venv 的 editable 安装；设了会被 preflight 拦
  export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/m8-smoke"   # 固定目录：避免每轮 smoke 从零编译
  export TRAIN_RECORD_DIR="$V1_STORE/bench/$RUN"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  export WANDB_MODE=disabled

  # GPU 空闲（preflight 不查硬件）
  test "$(nvidia-smi --id=4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-8frame-8x8.yaml

  # 唯一真源：preflight 校验的 argv 与 train.py 真收到的 argv 是同一个数组
  TRAIN_ARGS=(
    mme_vla_suite_b128_60k
    --exp-name "$RUN"
    --num-train-steps 20 --log-interval 1 --no-wandb-enabled
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR"
    --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8"
    --model.history-config "$HC"
  )

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" \
    --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN" \
    -- "${TRAIN_ARGS[@]}"

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
}

body 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
RUNNER
chmod +x "$MAIN/v1-store/logs/m8-smoke-runner.sh"
```

**B-1. 起跑**

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$MAIN"
RUN="smoke-m8x8-modul-$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_HEAD=$(git rev-parse HEAD)     # smoke 阶段主副本就是训练树，可直接取；正式 run 见 C 节
HC_SHA=$(sha256sum src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml | cut -d' ' -f1)
echo "RUN=$RUN"; echo "TRAIN_HEAD=$TRAIN_HEAD"; echo "HC_SHA=$HC_SHA"   # 三行抄进 launch.md
tmux ls                                             # 起前快照，确认没有同名 m8-smoke
tmux new-session -d -s m8-smoke \
  "bash $MAIN/v1-store/logs/m8-smoke-runner.sh '$RUN' '$TRAIN_HEAD' '$HC_SHA'"
tmux has-session -t m8-smoke && echo "SESSION_UP=m8-smoke"
```

（`--num-train-steps 20 --log-interval 1 --no-wandb-enabled` 为 smoke 专用覆盖，正式 run 不带。用 `bash <runner>` 而非 `bash -lc`：`-l` 会重载登录 profile，可能覆盖 `PATH` / 代理 / conda 初始化。）

Monitor（**绝对路径**；Monitor 与后续 Bash 工具跑在主工作目录，与 tmux pane 里的 `cd $WT` 无关，但写绝对路径免歧义）：
```bash
tail -n +1 -F /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8-smoke.log \
  | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "PREFLIGHT=|CHECK_.*FAIL|Loaded norm stats|Norm stats not found|Integration Type|Step 19:|EXIT_CODE=|Error|Traceback|RESOURCE_EXHAUSTED|out of memory|unbound variable"
```
（`PREFLIGHT=` 与 `CHECK_.*FAIL` 把自检结果主动送到监听端；`Norm stats not found` 抓 norm_stats 静默降级——`_load_norm_stats` 失败只打 `logging.info`，不告警；`unbound variable` 让变量漏赋值这类死法立刻可见。）

判据（全部满足才进入 C）：
- `PREFLIGHT=PASS n=25`，且 25 行 `CHECK_*` 逐行目视确认（尤其 `CHECK_CWD` / `CHECK_V1_STORE_REAL` / `CHECK_NOT_DEV_COPY` 三条，它们防的是「在 `-temp` 开发副本里误起训练」）。
- 日志含 `Loaded norm stats from $V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme`，且**不含** `Norm stats not found in`。
- 日志含 `Integration Type: modulation`；`EXIT_CODE=0`；`Step 0`…`Step 19` 的 loss / grad_norm / mem_enc_norm 全部有限，无 `RESOURCE_EXHAUSTED`。
- run 根 `history_config.resolved.yaml` 含 `token_per_image: 64`、`integration_type: modulation`；`motion_provenance.json` 的 `motion_enabled=false`、`framesamp_manifest_sha256` = `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`（该值取自 `framesamp-8x8/meta/store_meta.json` 的 `manifest_sha256` 字段，**不是** `episode_manifest.json` 文件的裸 sha256，后者为 `df0ec8ed…`，两个口径别混）。
- checkpoint 19 的 `assets/robomme/norm_stats.json` sha256 == `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`（证明上面那个绝对路径 flag 生效并随 checkpoint 落盘；**注意 `save_assets` 在 `norm_stats is None` 时静默什么都不写**，所以这条空了就说明 flag 没生效）。
- **参数树（两步）**：
  1. `JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/legacy-eval/check_ckpt_param_tree.py --config mme_vla_suite_b128_60k --ckpt-dir <run根>/19 --out <scratchpad>/param_tree.json`（在 `$WT` 快照那份脚本上跑：它 `os.chdir(_REPO_ROOT)`，而 `get_history_config` 按 cwd 读 YAML），判定行须逐字为
     `PARAM_TREE_EXACT=PASS config=mme_vla_suite_b128_60k history_config=perceptual-framesamp-modul-8frame-8x8.yaml yaml_sha256=5b5ac2f85302d4d8 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0`。
     其中 **`n_model=61` 就是 modulation 的正面签名**（context 为 55、context+motion 为 59，见 `docs/training-doc/eval-official-framesamp-context/records/param_tree.json` 与 `eval-hard-patternlock-routestick/records/modul/param_tree.json`）；`missing=extra=0` 是双向精确比对，故 PASS 已**蕴含**六个 modulation 专属叶子存在于 checkpoint。
     **不得要求 json 中出现参数路径**——`missing` / `extra` / `shape_mismatch` 只在 FAIL 时才列路径，PASS 时三者恒为空列表，「PASS 且 json 里有路径」是一对互斥条件，写进判据就是一条永远不可能通过的验收。
  2. 正面列出六条 modulation 专属参数路径（只读 orbax `_METADATA`，纯 grep，不进 JAX）：
     ```bash
     M=<run根>/19/params/_METADATA
     N=$(grep -o "q_einsum_mem', 'w'\|kv_einsum_mem', 'w'\|out_einsum_mem', 'w'\|mem_rms_norm', 'scale'\|mem_rms_norm_ffn', 'Dense_0', 'kernel'\|mem_rms_norm_ffn', 'Dense_0', 'bias'" "$M" | sort -u | wc -l)
     echo "MEM_PARAMS=$([ "$N" = 6 ] && echo PASS || echo FAIL) n=$N"
     ```
     期望 `MEM_PARAMS=PASS n=6`。**六条里 `mem_rms_norm_ffn`（2 叶）不能漏** —— 它才是 modulation 真正的作用点（`history_gemma.py` 里 `MemoryRMSNorm(name="mem_rms_norm_ffn")(x, mem_mod_vec)` 的 `Dense_0` 才产出 scale/shift），而 `mem_attn` 内的 `mem_rms_norm` 只是一个无条件 norm；且 `mem_rms_norm` 是 `mem_rms_norm_ffn` 的**子串**，所以上面每条都带 `', '<字段名>'` 定界。负对照已实测：同一 grep 在 `awsprod40k-b128-motion/5000`（context+motion）上命中数为 0。
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

**起跑前**（步骤 8）：写 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md`（起跑 commit = `TRAIN_HEAD`、25 行 `CHECK_*` 原文、差异表、lr 口径说明、完整命令、数据三件套 sha、输出路径、判据、smoke 摘录、F2/F3/梯度对拍的判定行摘录、开发副本 `-temp` 的建立命令与三条红线、tmux 清单 `m8-prod` / `m8-prod-dense`、「仍然防不住的」四条、失败重试要先清哪些残骸）；`docs/training-doc/README.md` 表格加一行；`git add` 两文件 → `docs: v2-1600ep-m8x8-modul-b128-60k 起跑留档` → push。**此刻记 `TRAIN_HEAD` = 主副本 HEAD**，然后建开发副本（G 节）。

**C-0. 生成 runner**（同 B 节分层口径；差别只在 wandb 开、不带 smoke 覆盖、多一个 GPU 采样器）

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
mkdir -p "$MAIN/v1-store/logs"
cat > "$MAIN/v1-store/logs/m8-prod-runner.sh" <<'RUNNER'
#!/usr/bin/env bash
set -o pipefail                     # 外层不开 -e：EXIT_CODE 必落盘

TRAIN_HEAD="${1:?用法: m8-prod-runner.sh <TRAIN_HEAD> <HC_SHA256>}"
HC_SHA="${2:?}"
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=v2-1600ep-m8x8-modul-b128-60k
LOG="$MAIN/v1-store/logs/$RUN.log"
DS="$MAIN/v1-store/datasets/4task-v2-1600ep-604f16da"
REC="$MAIN/v1-store/bench/$RUN"
mkdir -p "$(dirname "$LOG")" "$REC"

body() {
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志

  unset PYTHONPATH
  export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export TRAIN_RECORD_DIR="$REC"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"

  test "$(nvidia-smi --id=4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$(date -u +%FT%TZ)"

  ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  HC=perceptual-framesamp-modul-8frame-8x8.yaml

  TRAIN_ARGS=(
    mme_vla_suite_b128_60k
    --exp-name "$RUN"
    --assets-base-dir "$V1_STORE/train-assets"
    --data.assets.assets-dir "$ASSETS_DIR"
    --data.assets.asset-id robomme
    --checkpoint-base-dir "$V1_STORE/train-runs"
    --dataset-path "$DS/framesamp-8x8"
    --model.history-config "$HC"
  )

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173 \
    --dataset-path "$DS/framesamp-8x8" \
    --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_60k/$RUN" \
    -- "${TRAIN_ARGS[@]}"

  # 全程 15 s GPU 采样：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上（否则 tee 不退出）
  nvidia-smi --id=4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
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
cd "$MAIN"
TRAIN_HEAD=$(git rev-parse HEAD)     # 此刻 HEAD 即起跑 commit；与 launch.md 里记的那个必须一致
HC_SHA=$(sha256sum src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml | cut -d' ' -f1)
echo "TRAIN_HEAD=$TRAIN_HEAD"; echo "HC_SHA=$HC_SHA"
tmux ls
tmux new-session -d -s m8-prod "bash $MAIN/v1-store/logs/m8-prod-runner.sh '$TRAIN_HEAD' '$HC_SHA'"
tmux has-session -t m8-prod && echo "SESSION_UP=m8-prod"
```

**C-2. 确认进入稳态后立即锁只读**（G 节；等日志出现若干条 `Step` 行、确认不是起跑即挂之后）

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=v2-1600ep-m8x8-modul-b128-60k
sha256sum src/mme_vla_suite/training/framesamp_dataset.py scripts/training/train.py \
          src/mme_vla_suite/models/integration/history_pi0.py > "v1-store/bench/$RUN/lock_sha256.txt"
chmod -R a-w src scripts packages
ls -ld src scripts packages          # 确认 w 位已去掉
```

**C-3. 前 30 min 密采**（AGENTS 16；另起会话，不与训练日志共壳）

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
  | grep --line-buffered -E "PREFLIGHT=|CHECK_.*FAIL|Norm stats not found|Step [0-9]*00:|Finished asynchronous save|EXIT_CODE=|Error|Traceback|RESOURCE_EXHAUSTED|nan|unbound variable"
```
存活判断用 `tmux has-session -t m8-prod && echo ALIVE || echo GONE`（AGENTS 7 禁裸 `pgrep -f`）。起跑后 300 步复核稳态 s/step（tqdm `Progress on` 行时间戳差分，同 `bench-b128-util` 口径），在 launch.md 补 ETA。

**输出**：checkpoint `v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-modul-b128-60k/{5000,…,55000,59999}`（12 个；存盘条件是 `step % save_interval == 0 and step > start_step` 或 `step == num_train_steps - 1`，故最后一个是 59999 而非 60000）；单个实测约 12 G，合计约 144 G；日志与 GPU csv 落 `records/`；wandb run `robomme-framesamp/v2-1600ep-m8x8-modul-b128-60k`。

**跑完**：`result.md`（起止、EXIT_CODE、12 ckpt 齐全、loss 里程碑、稳态 s/step、util 均值 / 0% 占比 / 慢步分层、显存口径声明、下一步）；`records/` 收清洗日志、两份 GPU csv、两份 runner 脚本；不归档权重。`docs:` commit + push。

### D. 关键文件与复用

- 配置：`src/mme_vla_suite/training/config.py`（新增条目；`mme_vla_suite` 与 `mme_vla_suite_b128` 不动）；`src/openpi/training/optimizer.py::CosineDecaySchedule`（不动）。
- 训练入口：`scripts/training/train.py`（`init_history_config` 写 provenance）；环境变量解析 `src/mme_vla_suite/training/dataloader.py::_create_framesamp_dataset`。
- 路径源：`scripts/training/paths.sh`。
- 参数树核对：`scripts/training/legacy-eval/check_ckpt_param_tree.py`（只输出计数与 missing/extra/shape_mismatch，PASS 时后三者恒为空列表；modulation 的正面签名是 `n_model=61`）。
- 起跑前自检：`scripts/training/preflight_train_launch.py`（纯 stdlib、零副作用、25 项检查；`--repo` 传主副本根，trailing `--` 之后传与 `train.py` 逐字相同的 argv 数组）。
- 追溯梯度对拍（J 节）：以 `scripts/training/tests/single_step_grad.py` 为底改出 `single_step_grad_fixed.py`（删 `ref_npy_dataset` 顶层 import —— 它 import 了 `06220c4` 才新增的 `motion_store` 与 `sampling.memory_order`，老树没有会 `ImportError`；`_EXPECTED_HISTORY_CONFIGS` 加两个 modul YAML；`_build_batches` 换成磁盘 loader）；`scripts/training/tests/_common.py` 的 `load_array`（bfloat16 走 `.bin` + 旁置 JSON，逐位无损）；对拍器参考 `compare_fixture_dumps.py` / `compare_grad_summaries.py`；定点 batch `v1-store/fixtures/8x8/grad/{c8-b,c32-b}`（现成，不重新生成）。
- F3 对拍：`scripts/training/g0/check_baseline_env.py`（dump / check）、`scripts/training/g0/compare_baseline.py`（只比交集并打印 `rows=`）、`scripts/training/tests/project_scalars.py`（`_HEADER` 强制写表头，故 TSV 行数 = 1 + 步数）、`scripts/training/g0/bench_train_steps.py::_make_digest_gate`（`BENCH_EXTRA_DIGEST_STEPS` 越界即 raise）。
- 留档样板：`docs/training-doc/awsprod40k-b128-motion/{launch,result}.md`、`docs/training-doc/v2b-read20-20260914T174147Z/`。

### F. 守卫放宽与增补验证（对应执行顺序步骤 2–4）

**为什么要多跑。** `framesamp_dataset.py::FrameSampDataset.__init__` 的 `_req` 里有两条 commitV3.1（`6ee7494`，2026-08-27）按 Codex 审计 G13 加的守卫：`integration_type == "context"`、`memory_token_dim == 2048`，目的是挡住「同形的 modul 配置」。它们只在 `__init__` 拒绝配置，后续交付路径（`__getitem__`、`_pad`、store 读取）不读这两个键。放宽它们是对 dataloader 文件的改动，按 AGENTS 第 18 条要证明不改数，所以多跑 F2、F3 两块验证。`train.py` 主链路上没有其他 context 专属守卫；`g0/bench_train_steps.py` 与 `tests/` 下四个取证工具各有一份 context-only 白名单，本轮不经过（见第一部分第 6 点）。

**F1. 守卫改成成对白名单**（已由 `81ba002` 实施）：

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
- **判据声称范围**：这证明的是「**在当前源码下**，两份配置在所选 256 个索引上交付的原始 Dataset 输出逐字节相同」。差异是 `integration_type` / `memory_token_dim` **外加 context 版多出的整节 `motion:`** 的合集（见第一部分第 6 点）。Dataset 之后不另做对拍，理由是源码级封闭：`transform_dataset` 只收 `data_config`，`RoboMMEDataConfig.create` 里 `RepackTransform` 的键表写死、`ModelTransformFactory` 只读 `model_config.model_type / max_token_len / action_dim / discrete_state_input`，`TorchDataLoader` 的 collate 与 `HistAugObservation.from_dict` 都不接收 `history_config`；全仓库这两个键除 `framesamp_dataset.py` 那两条守卫外只出现在 `src/mme_vla_suite/models/`，**`src/openpi/` 零出现**。两侧 `data_config` 与 `model_config` 其余字段完全相同，因此 Dataset 输出相同即最终进模型的 batch 相同。它**不比较接入前源码**。

**F2.5. 先 commit 再验（用户 2026-09-15 拍板）。** F2 全 PASS 后立即：
```bash
git add src/mme_vla_suite/training/framesamp_dataset.py scripts/training/tests/test_pack_guards.py
git commit    # subject: commitV9.6: dataset 形制守卫放宽为 (integration_type, memory_token_dim) 成对白名单
              # body: F2 判定行原文 + 「F3 梯度等价验证待跑，FAIL 则 revert」
git push
git status --porcelain     # 必须为空，F3 要从这个 clean HEAD 起跑
```
理由见第一部分第 6 点第三条。**这个 commit 必须早于步骤 7 的 smoke**，否则 smoke 会被 `_req` 拒掉 modul 配置。

**F3. 第二块：复用 t8-c8-b 固化轨迹前 100 步（GPU 4,5，约 30–35 分钟，tmux `m8-guard`）。**

**不要照抄 t8-c8-b 的命令。** 照抄会在启动阶段直接抛 `ValueError: BENCH_EXTRA_DIGEST_STEPS 越界（须在 0..99）: [199, 399, 599, 799]`（`bench_train_steps.py::_make_digest_gate` 按 `config.num_train_steps` 做范围校验，且该 gate 在 `checksum_on` 之后**无条件**构造，`BENCH_CHECKSUM` 碰不到它）。更糟的是 t8-c8-b 的命令体只有 `set -o pipefail`、**没有 `set -e`**：bench 退出 1 之后 shell 会继续跑完后处理并无条件 `printf 'TRAJECTORY_DONE …'`，`EXIT_CODE=${PIPESTATUS[0]}` 取到 printf 的 0 —— **日志尾行写 `EXIT_CODE=0` 而训练一步没跑**。所以 F3 写成独立完整命令，相对 t8-c8-b 共**六处**差异。

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
  cd "$MAIN"                    # F3 验的是主副本 HEAD 的守卫改动
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

  test -z "$(git -C "$MAIN" status --porcelain)"      # ★差异3：F2.5 已 commit，故这条为真（若把 commit 排在 F3 之后，它恒假）
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

**判据（全部带量词，不写无条件的 `mismatches=0`）**：
- `BASELINE_ENV=PASS` —— 指纹 preflight（jax 0.5.3、A100 驱动 595.71.05、norm_stats `750a8e9b…`、pi05_base 与库摘要逐项同）。**注意 `env.json.fingerprint` 不含任何 commit/代码 sha**，所以 F1 改了 `framesamp_dataset.py` 不会让它 FAIL——它保的是环境不是代码。基线 `v1-store/bench/8x8/t8-c8-b/` 的 `BASELINE_MANIFEST.json` 所列 10 个产物已逐个核过 sha，10/10 未腐烂。
- `SCALARS steps=100 keys=5 hex_mismatch_steps=0` —— **主判据**，100/100 步全覆盖（五标量位于 batch 下游，batch 变了 loss 必变）。
- `INDEX_TRAIN=PASS n=800` —— 由收尾脚本显式核对 100 步 × 8 个训练索引。实测 `compare_baseline.py::compare_index_seq` 比较整个共同前缀，输出 `INDEX_SEQ=PASS n=872`（含 72 个预取索引）；两种数量分别留档。
- `BATCH_DIGEST rows=6 mismatch=0` —— 基线在 step 0..99 只固化了 `{0,1,2,24,49,99}` 六份，**这是基线的既有取证密度，不是本轮覆盖不足**；`rows` 必须写进判定行，不得省略。
- `STATE_DIGEST rows=6 mismatch=0` —— `BENCH_CHECKSUM=1`（用户拍板）下六份完整 TrainState 逐叶摘要对照。若将来改回 `0`，`compare_baseline.py` 对不存在的 `param_checksums.jsonl` 返回空 dict、会打印 `rows=0 mismatch=0` 并汇入 `DET_CHECK=PASS`，那是**结构性空判**，必须显式标注。
- `CANON_CHECK=PASS steps=6`。
- 汇总判定行：`GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=6 state_digest_rows=6`。
- 旁证（可选）：`cmp <(head -n 101 docs/training-doc/t8-c8-b/records/scalars_hex.tsv) "$BENCH_RECORD_DIR/scalars_hex.tsv"` —— **101 = 1 行表头 + step 0–99**，候选文件恰好 101 行，整文件对比。**不能写成「前 100 行」** —— 该 TSV 实测 1001 行 = 表头 + 1000 步（表头由 `project_scalars.py` 的 `_HEADER` 强制写入），按「前 100 行」比会静默漏掉 step 99。

**F3 留档（用户 2026-09-15 拍板）**：F3 实际 30–35 分钟 > AGENTS 17 的 5 分钟阈值，且它的定位正是 AGENTS 18 第二块的**正式等价证据**，不能借用 09-14 那条明写「不作正式等价证据」的排错例外。仓库里四个同类先例（`aws-t2-cand-s100`、`aws-t2-ref-s100`、`aws-t3-closed-s100`、`aws-t3-open-s100`）都是三件套齐全——留档才是惯例。建 `docs/training-doc/t8-c8-guard-s100/`：
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

### G. 隔离机制：训练锁主副本 + `-temp` 开发副本（步骤 8 建、步骤 9–10 用、步骤 11 收）

第一部分第 3 条给了结论，本节给完整机制与操作。

**训练侧（主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`）**：cwd = 主副本；不设 `PYTHONPATH`（靠主副本 `.venv` 的 editable 安装）；`UV_PROJECT_ENVIRONMENT` 无需设置；`source` 主副本的 `paths.sh`。起跑后立即 `chmod -R a-w src scripts packages` 锁死只读，训练结束后 `chmod -R u+w` 恢复 —— 把「不改主副本」从纪律变成技术闸。

**开发侧（`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`）**：`git clone` 主副本建立（不用 `git worktree`，clone 出来完全独立，主副本 `.git` 一个字节都不会被写）；`ln -s <主副本>/v1-store v1-store`；`uv sync` 建**自己的** `.venv`。

**三条红线（`AGENTS.md` 第 14 条已同步修订，加了开发副本例外条款）**：
1. `-temp/v1-store` 是**可写**的 symlink（与环境 A 的只读 turbo 链性质不同）—— 在开发副本里写 `v1-store/` **等同于直接写主副本数据**。
2. **开发副本里禁止执行任何带 `--force` 或输出根参数的破坏性命令** —— `build_dataset.py --force` 会 `rmtree` 整个输出根，穿透 symlink 即删主副本数据。确需执行时回到主副本。
3. 开发副本**必须有自己的 `.venv`**，不得共用主副本的 —— 共用时 `uv sync` / `uv add` 会换掉正被训练进程使用的包文件。

**起跑前自检 `scripts/training/preflight_train_launch.py`**（新增，纯 stdlib、零副作用，非零退出即中止）。最要紧的一类误操作是**在开发副本里误起正式训练** —— 两边源码高度相似，跑起来不会有任何症状，但训练读的是随时在改的开发代码。五条从不同角度堵它：

| 检查 | 判据 |
|---|---|
| `CHECK_CWD` | cwd 必须等于主副本路径（路径直比，与内容无关） |
| `CHECK_PKG_*` | `mme_vla_suite` / `openpi` / `openpi_client` 三个包的 origin 都在主副本下 |
| `CHECK_SYS_PREFIX` | 解释器就是主副本的 `.venv` |
| `CHECK_V1_STORE_REAL` | `<repo>/v1-store` 是**实体目录** —— 开发副本那份是 symlink，天然可分 |
| `CHECK_NOT_DEV_COPY` | 仓库根目录名不以 `-temp` 结尾 |

其余检查覆盖：YAML 路径与 sha256、`norm_stats.json` 的存在与 sha256（**在训练启动前就查**，不等 dataloader 抛 `TypeError`）、数据集与两个 `MMEVLA_FRAMESAMP_*` 环境变量、主副本 HEAD == `TRAIN_HEAD` 且 clean、run 根不存在，以及**把交给 `train.py` 的参数数组同时喂给 preflight 逐字校验**四个关键 flag。共 25 项，2026-09-15 本轮在 clean HEAD 上实测 `PREFLIGHT=PASS n=25`。旧草稿的 23 项计数不完整；当前保留全部检查。旧负向测试中漏传 `--data.assets.assets-dir` 被准确抓出并归因到「数据参数」。

**两条容易误判的事实**：
1. **`openpi-client` 确实在训练链路上**（`openpi/transforms.py` 顶层 `from openpi_client import image_tools`，`ResizeImages` 每样本调两次）—— 但新方案不需要手工设 `PYTHONPATH`，它靠主副本 `.venv` 的 editable 安装天然从主副本加载。preflight 仍查它的 origin，用来抓「`PYTHONPATH` 被污染指向 `-temp`」。
2. **norm_stats 的 `assets_dir` 是 cwd 相对路径** —— cwd = 主副本时本来就解析得对，所以传不传绝对路径都跑得起来。但命令仍显式传：消除对 cwd 的隐式依赖，且让 preflight 的 `CLI_ASSETS_DIR` 有东西可校验。顺带记住 `--assets-base-dir` 对本条目是**死参数**（回退目标 `train-assets/mme_vla_suite_b128_60k` 不存在），别把它当保险。

**仍然防不住的**（写进 `launch.md`）：
- **【高】两份 argv 不同源** —— preflight 的保证建立在「它校验的 argv 与 `train.py` 真收到的是同一个 bash 数组」上。分别手写两份，这层保护就回到零。无技术兜底，靠 runner 写法 + 纪律。
- **【中】训练期间在主副本跑 `uv sync` / `uv add` / `uv pip`** —— 会换掉正被训练使用的包文件。写进禁令表；开发副本有自己的 `.venv` 正是为此。
- **【中】`chmod` 之后仍可能被 root 或 `chmod u+w` 绕过** —— 跑完复查 `git status --porcelain` 仍为空 + 关键文件 sha 与起跑时一致，写进 `result.md`。
- **【低】失败重试要先清残骸** —— 任何在 `initialize_checkpoint_dir` 之后的失败都会留下半截 run 根与一个 wandb run，而正式 run 名字固定（不像 smoke 带时间戳），重试前必须手工删掉 run 根、`TRAIN_RECORD_DIR` 与那个 wandb run。

**建开发副本**（步骤 8，`docs:` commit push 之后、主副本 clean、记下 `TRAIN_HEAD` 之后）：
```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
DEV=${MAIN}-temp
test -z "$(git -C "$MAIN" status --porcelain)"
TRAIN_HEAD=$(git -C "$MAIN" rev-parse HEAD); echo "TRAIN_HEAD=$TRAIN_HEAD"   # 抄进 launch.md
test ! -e "$DEV"                                   # 已存在则先问用户，不覆盖
git clone "$MAIN" "$DEV"                           # 不用 git worktree：clone 完全独立，主副本 .git 不被写
ln -s "$MAIN/v1-store" "$DEV/v1-store"             # 可写 symlink，见下红线
ls -ld "$DEV/v1-store"                             # 确认是 symlink 且指向主副本
cd "$DEV" && UV_CACHE_DIR=/scratch/hongze/.cache/uv uv sync   # 自己的 .venv，约 7.1 G
uv run --no-sync python -c "import jax,sys;print('DEV_VENV=',sys.prefix)"   # 必须是 $DEV/.venv
```

**用**（步骤 9–10）：训练只在主副本跑，命令是历史跑通形态 —— cwd = 主副本、不设 `PYTHONPATH`、不设 `UV_PROJECT_ENVIRONMENT`、`source` 主副本 `paths.sh`。起跑并确认进入稳态后立即锁只读：
```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
sha256sum src/mme_vla_suite/training/framesamp_dataset.py scripts/training/train.py \
          src/mme_vla_suite/models/integration/history_pi0.py > v1-store/bench/$RUN/lock_sha256.txt
chmod -R a-w src scripts packages
ls -ld src scripts packages                       # 确认 w 位已去掉
```
训练期间一切开发、验证、留档编辑都在 `$DEV` 里做。

**红线**（`AGENTS.md` 第 14 条已同步修订）：
- `$DEV/v1-store` 是**可写**的 symlink，在开发副本里写 `v1-store/` **等同于直接写主副本数据**；
- **开发副本里禁止任何带 `--force` 或输出根参数的破坏性命令**（`build_dataset.py --force` 会 `rmtree` 整个输出根，穿透 symlink 即删主副本数据）；确需执行时回到主副本并先 `ls -ld <输出根>`；
- 开发副本**必须用自己的 `.venv`**；**训练期间禁止在主副本跑 `uv sync` / `uv add` / `uv pip`**（会换掉正被训练进程使用的包文件，worker 重建时读到新文件）；
- 不在主副本改任何文件、不 `git pull`；`git clean -x` / `-X` 全仓禁止（AGENTS 19 附）。

**收**（步骤 11，训练结束、`tmux has-session -t m8-prod` 已不存在之后）：
```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
chmod -R u+w src scripts packages
git status --porcelain                            # 必须仍为空
sha256sum -c v1-store/bench/$RUN/lock_sha256.txt  # 三个文件的 sha 必须与起跑时一致
```
两条结果写进 `result.md`。`$DEV` 视情况保留或删除（它是临时工作区，不跨长训练周期保留；删前确认其中的 commit 已 push）。

**注意 `git worktree list` 当前实有五条**（主副本、`.claude/worktrees/b128cfg`、`.claude/worktrees/sgab`、`v1-store/reports/tic-adversarial-20260908/source`、`v1-store/worktrees/official-89efeaab`）。本方案**不新增 worktree**，也不动这五条。

### H. 审计结论：motion 接入不影响「无 motion 的 modulation」

**结论：源码级成立**（下面九条逐层核过 motion 接入 commit `06220c4` 的 diff 原文）；**运行时实测由 J 节的追溯对拍（V6）补齐**。

- **模型侧参数树不变**：`PerceptualMemory.__init__` 的两层新参数 `motion_pos_proj`、`motion_encoder_static` 只在 `motion.enabled` 为真时创建，且建在 `feature_encoder` 之后（nnx 默认 RNG 流按调用顺序 fold_in，帧路初始化值不变）。modul-8x8 YAML 没有 `motion` 节，`_motion_enabled` 为 False。
- **`embed_memory` 关闭态是编译期早返回，执行路径与接入前逐位等价**：函数签名逐字未变（两版都是 `def embed_memory(self, obs: HistAugObservation)`），唯一改动是 `self.mem_encoder(...)` 调用点多传三个关键字实参 `motion_emb / motion_pos / motion_mask`——这三个值在关闭态恒为 `None`（`FrameSampDataset` 的 `_NONE_KEYS` 补键 → `from_dict` 的 `data.get(key, None)`），且 `PerceptualMemory.__call__` 在 `if not self.motion_enabled: return hidden_states, None, None` 之前一条语句都不读它们，传给 `feature_encoder.encode_perceptual_memory` 的三个实参与接入前逐字相同。随后 `if not self.mem_encoder.motion_enabled` 是 Python 编译期分支，早返回分支内的四行（`input_mask = obs.static_mask`、两个全 False 列表、`return`）与接入前的后四行逐字相同。合起来：关闭态执行的运算与返回的四个值与 `06220c4~1` 逐位一致。**注意不能笼统说「函数体与接入前逐字相同」** —— `mem_encoder` 调用在早返回之前，函数体也多了 `if` 与整段开启态代码；成立的是「关闭态执行路径逐位等价」这个更精确的表述。modulation 分支在 `compute_loss` / `sample_actions` 里只取前两个返回值 `mem_seq, mem_mask` 喂 `PaliGemma.llm(..., mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])`。
- **modulation 的 prefix 不含记忆区**：`embed_prefix` 只在 `integration_type == "context"` 时才把记忆 token 拼进 prefix；modulation 的 prefix 是两路视角各 256 位共 512 图像 token + prompt（**这是训练侧口径**；推理侧 `robomme_policy.py` 只喂 `base_0_rgb` 一路 = 256，引用到评估语境时须注明），RoPE 的 `positions = jnp.cumsum(input_mask, axis=1) - 1` 因此不含记忆位次。**注意力掩码上两条路径口径不同但结果相同**：推理 `sample_actions` 的 modulation 分支把 `na_mask` 丢进 `_`、走 `make_attn_mask` 两参数版；训练 `compute_loss` 在 `use_history=True` 时统一走三参数版（modulation 也不例外），但 modulation 的 `na_mask` 首位就是图像 token 的 `True`（`embed_prefix` 里 `na_mask += [True] * image_tokens.shape[1]`），于是 `make_attn_mask` 内 `jnp.cumsum(mask_na, axis=1) <= 0` 恒为全 False、`mask_not_attend` 恒空，三参数版与两参数版逐位相同——记忆屏蔽项在 modulation 下恒不生效。**只写推理那半句会让人误以为训练也走两参数版，引用时须带上训练侧口径。**
- **608 位交错 / `take_along_axis` 为何对 modulation 不可达**（**注意不能把它挂在上一条「prefix 不含记忆区」名下，那个推理不成立**）：modulation 恰恰是通过 `embed_memory` 的返回值 `mem_seq` 拿到记忆并走 cross-attention 的（`compute_loss` 与 `sample_actions` 的 modulation 分支），开启态产出的 608 位重排序列**会**直达 modulation。真正的不可达性来自两处：关闭态的编译期早返回（上上条），以及下一条的两道显式闸。三项里只有「RoPE 位次变化不可达」是真由 prefix 组成推出的。
- **cross-attention 本体零改动**：接入前后 `history_gemma.py`（`MemoryAttention` / `MemoryRMSNorm`）、`integration/utils.py`、`openpi/models/gemma.py`、`representation/mem_encoder.py` 四文件 `git diff` 为空。
- **两道显式闸**：`HistoryPi0.__init__` 里 `motion_enabled and integration_type != "context"` 即 raise；`inputs_spec` 与 `PerceptualMemory` 的 `motion_enabled` 不一致即 raise。开启态混进 modulation 会在建模型时报错。
- **数据侧交付不读 `integration_type`——但只对 `__getitem__` 成立**：`FrameSampDataset.__getitem__` 的 motion 代码全在 `if self._motion_enabled` 内，关闭态只在样本末尾按 `_NONE_KEYS` 追加四个 None（与旧路径「尾部补空键」逐字一致），`HistAugObservation.from_dict` 用 `data.get(key, None)` 接住，None 在 jax pytree 里是空节点、不进数值图。context 与 modulation 拿到同一份 batch。**`__init__` 则会直接拒掉 modul-8x8**（就是 F 节要放宽的那两条守卫），这句摘抄进 `launch.md` 时必须补半句「`__init__` 的形制守卫另由 F 节放宽，见 `commitV9.6`」，否则会让人误以为数据侧零改动。
- **证据边界**：运行时逐位证据只在 context 关闭态取过（环境 A `motion-t1-closed` 对 G0b 逐位同、`motion-t2-ref/cand`；环境 B `aws-t3-closed-s100`、t8 C8/C32）。modulation 关闭态没有单独跑过对拍，`v1-postclean-g3` 登记的 UNVERIFIED 状态未变。F 节第一块把「两份配置在**当前源码**下数据侧交付一致」从论证变成实测——**它不比较接入前源码**，接入前/接入后的数据侧逐位证据仍只有 context 关闭态那几次。
- **模型侧的运行时证据由 J 节的追溯对拍补齐**：锚点取 `07702f0`（2026-08-29，commitV4.3，零适配分界 —— 它之前 `compute_loss` 返回二元组），一次对拍覆盖 `07702f0..HEAD` 整条链。该区间内 `src/mme_vla_suite/models/**/*.py` 只被 `06220c4` 一个 commit 触碰，**动了数值路径的改动为零**。不再往前推的理由：`git diff bc3ab59 07702f0^ --stat -- src/mme_vla_suite/models/` 只有 3 个文件各 1 行，且都与 framesamp modulation 无关、都在 `07702f0` 被删除 —— models 侧 2026-02-20 到 2026-08-29 半年冻结，往前推换不来新信息（`history_gemma.py` 这套 modulation 本体更是自 `602d0d5`、2026-02-06 起逐字未变）。做法见 J 节：绕开 Dataset、把现成的定点 batch 直接喂两版模型比 loss 与逐叶梯度 hex。

**固化位置**：`launch.md`「与官方 / 上次 run 的关系」一节引用本节。V6 未通过之前，不得据本节宣称「与接入前逐位等价」—— 源码级论证与运行时证据是两回事。不改 `docs/motion-memory.md` 等正本（评审性结论，非链路事实；正本改动另立）。

### I. 计划固化到仓库根目录（步骤 6）

本文件已在仓库根目录（与 `0908-8frame-8x8-training-plan.md` 同级、同体例），文首带状态说明。步骤 6 只需与 `launch.md` 初稿、`docs/training-doc/README.md` 加行一起 `git add`，subject `docs: v2-1600ep-m8x8-modul-b128-60k 计划固化与起跑留档初稿`，push。之后再记 `TRAIN_HEAD`（保证起跑 commit 含这份计划）。验证：`git diff --check`；Markdown 链接 `docs/training-doc/README.md` → 新 run 目录可解析。

### J. modulation 关闭态的追溯梯度对拍（`07702f0` → HEAD，执行顺序步骤 5）

**命题**：在**完全相同的输入张量**与**完全相同的初态权重**上，`07702f0`（2026-08-29，commitV4.3）与当前 HEAD 两份代码，对 **modulation 关闭态**跑一次前向+反向，得到的 loss 与全部可训练叶梯度**逐位相同**。两档都做：**32 帧 4×4** 与 **8 帧 8×8**。

这比「只比 motion 接入前后（`c5925d9` vs `06220c4`）」强一档 —— **一次覆盖整条演进链**，motion 接入只是其中一个 commit。

#### J1. 为什么锚点取 `07702f0`，而不是推到更早

| 层级 | commit | 日期 | 代价 |
|---|---|---|---|
| **采用** | **`07702f0`** | 2026-08-29 | 三处工具侧适配，**`src/` 零改动** |
| 可选 | `6ee7494` | 2026-08-27 | 还要适配 `compute_loss` 二返回 |
| 理论极限 | `bc3ab59` | 2026-02-20 | 再往前 YAML 用 `perceptual_memory.budget` 口径，与 HEAD 不同源，**不可推** |

**往前推换不来新信息**：`git diff bc3ab59 07702f0^ --stat -- src/mme_vla_suite/models/` 只有 3 个文件各 1 行（`config/base.yaml`、`representation/recur_mem.py`、`representation/rmt.py`），三者都与 framesamp modulation 无关，且都在 `07702f0` 被删除 —— **models 侧从 2026-02-20 到 2026-08-29 半年冻结**。旁证：`history_gemma.py`（`MemoryAttention` / `MemoryRMSNorm` 全套，modulation 本体）自 `602d0d5`（2026-02-06）起**逐字未变**；`openpi/models/gemma.py` 与 `pi0.py` 的 `git log` 都只有 first commit 一条。

`07702f0` 是**零适配分界**：它之前 `compute_loss` 返回 `(loss, stats)` 二元组，HEAD 版工具按单返回写，会硬失败。

#### J2. 改动分类：C 类为零，因此一次对拍即可，不分段

`07702f0..HEAD` 区间内，`src/mme_vla_suite/models/**/*.py` 只被 **`06220c4` 一个 commit** 触碰：

- **A 类（纯新增、关闭态不可达）**：`percep_mem.py` 的 `motion_pos_proj` / `motion_encoder_static` 两个 `nnx.Linear` —— modul.yaml 无 `motion:` 节 → `motion_enabled=False` → **两个 Linear 根本不创建**，参数树与 nnx RNG 消耗序不变；`__call__` 的 `if not self.motion_enabled: return hidden_states, None, None` 早返回与改前逐字相同；`embed_memory` 早返回四行与改前逐字相同；`history_observation.py` 新增四字段默认 `None`；`_motion_specs()` 关闭态返回 `{}`。
- **B 类（放宽断言 / 新增守卫，数值路径不变）**：两条新 `raise` 在关闭态一条为假、一条被 `and` 短路，都不触发；`framesamp_dataset.py` 的形制断言放宽（Dataset 侧，本方案绕开）。
- **C 类（动了数值路径）**：**无。** `src/openpi/` 在区间内唯一改动是 `shared/download.py`（+6/−2），与数值无关。

责任 commit 唯一，没有可二分的对象，**一次对拍即可**。若真 FAIL 再补分段也来得及（分段点取 `07702f0` → `c5925d9` → `06220c4` → HEAD，四锚点约 45–70 分钟）。

#### J3. 为什么走「固定 batch 喂两版模型」而不是「两棵源码树各起训练」

后者对 8×8 走不通：`07702f0` 的 `framesamp_store.py` 有四道硬拒读不了 8×8 库（`LAYOUT = "framesamp-4x4-v1"` 常量校验、表名 `image_emb_4x4` 与 `row_shape [16,2048]`、文件名 `pos_emb_4x4.f32.bin`、Dataset 的 `_req(... == (512,16,1))`）。要让它认，必须把 commitV9.1 `236765f` 的 `StoreSpec` 参数化改动整块 backport，而那次改动动了 **8 处热读路径**（pread offset、输出缓冲字节数、`posix_fadvise` 区间、memoryview 切片、`.reshape()`、pos 小表 `np.fromfile` 路径与 shape、两处 `run_*_checks`）—— 属「可能改数」，搬进对照侧就让基础塌掉。4×4 档虽只需放宽一条断言，但要在两棵树里各打掉 `test_g13_modul_config_rejected` 这条生产安全闸测试。

**固定 batch 对两档都成立**：模型侧除 `budget`（都是 512）外**不依赖帧数或每帧 token 数** —— `token_per_image` / `num_views` / `max_frames` / `tokens_per_frame` 在整个 `src/mme_vla_suite/models/` 里零命中；`PerceptualMemory.__call__` 只有一条 `assert static_image_emb.shape[1] == budget`；`FeatureEncoder` 是逐 token 的 pointwise 运算，`pos_emb` 不是查找表、没有行数概念；`MemoryAttention` 只读 `mem_seq.shape[1]`。两档喂进模型的四个张量**逐维逐 dtype 完全相同**（实测自 `docs/training-doc/t8-gradient/records/{c8,c32}/b/allfull.batch_meta.json`）：`static_image_emb (8,512,2048) bfloat16`、`static_pos_emb (8,512,768) float32`、`static_state_emb (8,512,8) float64`、`static_mask (8,512) bool`。**所以对照侧根本不需要认识 8×8 库，只要能吃下 `[8,512,2048]`。** 实测两档 fixture 的 `static_mask` 连 sha256 都逐位相同（`906b1856…`，两档都是 512 位全有效），`static_image_emb` / `static_pos_emb` 形状 dtype 全同、只有内容不同。

**推论（判据解读时必须带上）**：两档在模型侧走**同一段代码**，不是两条路径 —— 做两档等于「用两组不同的真实数值各验一遍」，多的是数值覆盖面（溢出 / NaN / 极端量级）不是代码覆盖面。**一档即足以证明代码等价**；若要压缩墙钟，优先保 32 帧 4×4（历史原生布局，与锚点代码同源）。

#### J4. 定点 batch：现成，且键集天然兼容

`v1-store/fixtures/8x8/grad/c32-b/`（32 帧 4×4）与 `c8-b/`（8 帧 8×8），各 92 MB，三档 `allfull` / `allshort` / `mixed1`，2026-09-14 由 t8-gradient 产出，数据侧代码自那以后零改动。

`batch_meta.json` 共 16 个键，其中 `motion_emb` / `motion_pos` / `motion_mask` / `mem_order` 四个是 `kind:"none"`，**连 `.bin` 都没落盘**（`ls c32-b/allfull/` 只有 12 对 `.bin`/`.json`）。重建 batch 时按 `kind=="array"` 取那 12 个键即可，四个 motion 键自然不存在，**不必 pop、不必改输入，两侧完全同源**。（即便留着也无妨：老树 `HistAugObservation.from_dict` 是 `data.get(key, None)` 逐键显式取，多余键静默忽略；父类 `Observation.from_dict` 唯一的严格检查是 `tokenized_prompt` 与 `tokenized_prompt_mask` 必须成对，定点 batch 两者都在。）

**1600 集新库不能用作 fixture 来源**：`exec_start_idx` 最小值是 100（0 出现 0 次），而 `_common.fixture_per_step` 要求存在 `exec_start_idx=0` 的 episode，否则直接 raise；且所有样本 `step ≥ 100 > max_frames`，`allshort` / `mixed1` 两档结构性不存在。必须用 400 集库 —— 现成 fixture 正是从它产出的。代价是**本对拍不覆盖 1600 集库的读数路径**（那属于 `236765f` 的 8×8 支持，已由 t8-gradient C8 单独验过）。

#### J5. 两个必须先做的前置

**① A/A 自复现（最高优先级）。** 仓库全部逐位梯度对拍（`t8-gradient` 四档、`v1-grad-baseline-g0b`）**清一色是 context**，`MemoryAttention` 里 `kv_einsum("BSD,2KDH->2BSKH", mem_seq)` 这条路径的 XLA 归约顺序**从未在确定性档下验证过可复现**。必须先在 `07702f0` 上**同一棵树、同一命令跑两次**（A1/A2）确认 `GRAD_EQ=PASS`。A/A 不过即说明 modulation 在当前 XLA 档下不可逐位复现，判据须从「逐位」降级为「数值阈值」，**先停下请示用户**。必带 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'` 与独立的 `MMEVLA_JAX_CACHE_DIR`。这条纪律沿用 `docs/training-doc/t8-c32-b/launch.md` 的「A1→A2→B，A1/A2 完整重复性先通过再起 B」。

**② 初态同源性自证。** 仓库**没有 modulation 的黄金基线**（`DTYPE_BASELINE_CHECKSUMS` 那套全是 context / motion 侧），该变量必须省略。改为在对拍脚本里自己落一份 `init_train_state` 之后 `state.params` 的逐叶 sha256，**先断言两侧初态逐位相同，再比梯度** —— 不做这步，FAIL 时无法区分「初值不同」与「数值路径不同」。注意 modulation 的参数叶数比 context 多（多出 `MemoryAttention` 的 kernel），**不会是 context 档记录的 32**，别照抄那个数字当判据。

#### J6. 三处工具侧适配（`src/` 零改动）

**先澄清一个前提**：HEAD 版 `scripts/training/tests/single_step_grad.py` **不从磁盘读 batch，它是现场造 batch 并落盘** —— `c32-b` / `c8-b` 正是它的输出。所以必须新写 loader 替换 `_build_batches`，这是主要代码量来源。

1. **删掉 `ref_npy_dataset` 的顶层 import**（`single_step_grad.py` 第 60 行）。`ref_npy_dataset.py` 顶层 import 了 `motion_store`（`06220c4` 新增的 528 行新文件）与 `sampling.memory_order`（同 commit 新增），**老树都没有，会当场 `ImportError`**。本方案绕开 Dataset，删掉即可。
2. **`_EXPECTED_HISTORY_CONFIGS` 加两个 modul YAML**（第 66–68 行现只列四个 context 系文件名）。
3. **`_build_batches` 换成磁盘 loader**：按 `batch_meta.json["keys"]` 里 `kind=="array"` 的 12 个 keystr，用 `_common.load_array` 读回并按 keystr 重建嵌套 dict（实测只有 `['image'][*]` / `['image_mask'][*]` 两层）。`_common.py` 写盘时有 round-trip 字节守卫、读回有 `nbytes` 校验，bfloat16 走 `.bin` + 旁置 JSON（躲开 `np.save` 把 `ml_dtypes.bfloat16` 写成 `V2` 的坑），逐位无损。

**「不改数」的证明在这里是平凡的**：没有任何改动落在 `src/`，两棵树跑各自原封不动的历史代码；工具侧新增的只是「从磁盘读回已落盘的 batch」与「多落一份初态摘要」。

另注：必须用条目 `mme_vla_suite`（`07702f0` 版 `config.py` 只有这一条，`_b128` / `_b128_60k` 是后加的）；`_guard_train_step_source()` 检查的是**主副本 HEAD 版** `train.py`，与锚点无关，两侧同时成立。

#### J7. 执行形态与成本

```
A 侧（锚点）：git worktree add --detach v1-store/worktrees/s2-base 07702f0
              cd <该树>；PYTHONPATH=$PWD/src；UV_PROJECT_ENVIRONMENT=<主副本>/.venv
B 侧（HEAD）：cd 主副本，unset PYTHONPATH
两侧共用：主副本 HEAD 版的 scripts/ 工具（cd 老树后显式跑 /主副本/scripts/training/tests/<新工具>.py）、
         同一份定点 batch、seed 42、fsdp 2、同一对物理 GPU、
         XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
档 1（4×4）：--model.history-config perceptual-framesamp-modul.yaml            batch=v1-store/fixtures/8x8/grad/c32-b
档 2（8×8）：--model.history-config perceptual-framesamp-modul-8frame-8x8.yaml batch=v1-store/fixtures/8x8/grad/c8-b
判定：GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0，外加两侧初态逐叶 sha 相同
```

`07702f0` 树里没有 modul-8x8 的 YAML（它是后加的）—— **把该 YAML 复制进 worktree**（未跟踪文件、不进 git；这是配置数据不是代码，两侧读同一份字节，`launch.md` 记两份 sha256 并在两侧各自打印核对）。`docs/training-doc/aws-a22-grad/launch.md` 用过同类手法。`perceptual-framesamp-modul.yaml` 本身两侧都有且 `git diff bc3ab59 HEAD` 为空，不必复制。

| 项 | 量 |
|---|---|
| 新代码 | **200–250 行**，全部落 `scripts/training/tests/`，`src/` 零改动（loader ~50、`single_step_grad_fixed.py` ~150、对拍器 ~30） |
| 定点 batch 生成 | **0**（复用现成） |
| GPU | 一次对拍两侧 **20–35 分钟 / 2 卡**（实测取自 `t8-gradient` 的 `grad_summary.json` 的 `seconds`：热缓存 c32 三 kind 共 ~230 s，冷缓存 ~815 s；modulation 的 HLO 与 context 不同，两侧缓存都是冷的，取上界）。加 A/A 自复现再算一轮 |
| 留档 | 按 AGENTS 12/17 建 `docs/training-doc/<run_name>/`（launch.md + result.md + `records/grad_summary.*.json` + 判定行） |

**环境两条**：必须 `source scripts/training/paths.sh` 或显式 `export OPENPI_DATA_HOME=<主副本>/v1-store/models`（默认 `~/.cache/openpi` 下只有 `big_vision/`，没有 `pi05_base`），或用 `--weight-loader.params-path` 覆盖；`uv.lock` **自 2026-03-20 至今一字未变**，覆盖全部候选锚点，jax / flax / ml_dtypes 版本两侧一致，老树包版本无风险（仍须 `UV_PROJECT_ENVIRONMENT` + `--no-sync`，防 uv 在老树另建 venv）。

**worktree 与 tmux 纪律**：老树放 `v1-store/worktrees/`（`ref-8x8` 同级，既有约定）；**本节的 worktree 是本对拍专用的临时快照，与 G 节的训练隔离无关**（训练本身不用 worktree）；对拍结束即 `git worktree remove` + `prune`。tmux 会话名带 `tr-` 前缀，清理只能 `tmux kill-session -t <确切会话名>`。

#### J8. 这个对拍不覆盖什么

**① 分帧语义与整套装配，全部不在覆盖内。** 这是最容易被误读的一条。8 帧 / 32 帧之分**只存在于 dataloader 侧**，源头是 YAML 里的一个参数：

```
4×4:  token_per_image: 16  ->  _max_frames = 512 // (16×1) = 32 帧
8×8:  token_per_image: 64  ->  _max_frames = 512 // (64×1) =  8 帧
```

`FrameSampDataset.__getitem__` 拿它做三件事：`even_sampling_indices(step, self._max_frames)` 采帧号 → `store.read_image_rows(rows)` / `store.pos_rows(frames_arr)` 按帧号查表 → `reshape(-1, ...)` 压成 `[512, ...]`（`static_state_emb` 与 `static_mask` 则是 `np.repeat(..., tokens_per_frame)`，把每帧一个的值摊到该帧每个 token）。另有一道守卫 `_req(self._meta.spec.tokens_per_frame == token_per_image * num_views, ...)` 防止拿 8×8 配置去读 4×4 库。

**模型侧一个「帧」字都没有** —— 时序与位置信息全部经 `static_pos_emb` 注入（pos 表按帧号索引，同一帧的 64 或 16 个 token 拿同一行内容），模型只做 `silu(Linear(pos))` 再 concat，逐 token。

**所以：若帧采样或查表本身错了（采错帧号、pos 对错行），V6 照样 PASS** —— 两侧吃的是同一份错输入。这一段的正确性由三层兜：库表本身靠 store_meta 的 sha256（8×8 的 `pos_emb_8x8.f32.bin` 是 `709a52a9…`，2304 行 × `[64,768]`；4×4 的是 `3176ac09…`，586 行 × `[16,768]`）加建库 `VERIFY_PACK=PASS`；查表与装配靠 **V3**（逐样本逐键比 raw sha）与 **V4**（走真实 Dataset 跑 100 步比梯度）—— **V4 是唯一同时覆盖「装配 + 模型计算」的验证**，代价是只在 context 8×8 上跑。

**② 1600 集库的读数路径**（见 J4，用的是 400 集库的定点 batch；那属于 8×8 支持那次改动，已由 `t8-gradient` C8 单独验过）。

**③ `07702f0` 之前的历史**（见 J1，那半年 models 侧冻结，无信息可得）。

**反过来，V6 内部「两侧输入同源」不是假设而是有证据链**：两侧从同一份文件读同一份字节，`batch_meta.json` 记了每个键的 `raw` sha256（`c32-b` 的 `static_pos_emb` 是 `6aedccac…`，`c8-b` 是 `639dc1a3…`），`load_array` 读回有 `nbytes` 校验，对拍器另比两侧 `batch_keys` 摘要。

### E. 红线与不做的事

- 不改模型代码；不动既有两个训练条目（`mme_vla_suite` / `mme_vla_suite_b128`）。dataloader 的守卫白名单是本轮唯一的 `src/` 改动，按 AGENTS 18 走 F2/F3 两块验证。
- 训练在**主副本**跑并 `chmod -R a-w` 锁只读（G 节）；训练期间一切开发与留档编辑在 `-temp` 开发副本里做，主副本不改文件、不 `git pull`、不跑 `uv sync` / `uv add` / `uv pip`，不建库、不起第二个 GPU 大任务。
- **开发副本红线**：`-temp/v1-store` 是可写 symlink，写它等同于写主副本数据；`-temp` 里禁止任何带 `--force` 或输出根参数的破坏性命令；`-temp` 必须用自己的 `.venv`。（`AGENTS.md` 第 14 条环境 B 段的例外条款。）
- tmux 只按确切名 kill，本轮起过的会话清单是 **`m8-guard`（F3）、`m8-smoke`（B）、`m8-prod` / `m8-prod-dense`（C）** 四个，清理时以这份清单为唯一依据，删前删后各一次 `tmux ls`。**不在清单内的一律不动**，含用户会话 `0`、`1`、`claude-private`、`codex`、`codex-repo`，以及他轮评估会话 **`eval-vla0914-full1600-best-a`**（2026-09-15 02:50 起，正占用 GPU 0）。禁 `tmux kill-server` 及一切全局杀法（AGENTS 7 红线）。
- 评估（legacy-eval / motion-variance 对 modulation + 8×8 的 prefix 长度、VideoRepick 驱动）不在本轮范围。
