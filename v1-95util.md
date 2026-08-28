# v1：GPU 利用率冲刺 ≥95% —— 主循环取数/计算重叠（L1）先行，L2/L3 按决策门后置

> **本文件是本轮（2026-08-28 起）的单一权威计划文档。** 它接续 v2 计划 S8b 的收官结论，
> 直接对应 [`docs/training-doc/v1-framesamp-e2e/result.md`](docs/training-doc/v1-framesamp-e2e/result.md)
> 第五节待办 3「真实 epoch 9.26 h 距 8.6 h 差 7.7%，须另行立项」。
>
> **与既有文档的关系（只引用、不复制）**：
> - IO 重构本体、G0/G2 定义、红线表（G 节 R1–R17）、白名单（T1）、验收判据（D 节）、基线链登记簿（T8）
>   的权威源是 [`v2-framesamp-restructure-plan.md`](v2-framesamp-restructure-plan.md)；本文件只引用其条目编号。
> - L2/L3 的 scope 权威源是 [`v1-post-restructure-roadmap.md`](v1-post-restructure-roadmap.md) 的项 2/项 3；
>   本文件只补该文件未覆盖的两条实施坑，不重述其规格。
> - S8b 各档实测数字与 epoch 口径缺陷的权威源是上述 `result.md`；登记簿数字只在 v2 计划 T8 维护一份（T9-B4）。
>
> **目标**：GL 4×A40 e2e 的 dense 稳态 **util 均值 ≥95%**（用户 2026-08-28 指定），
> 且全程可证明与 G0 训练语义逐位等价。

---

# 第一部分（给人看）

## Context：为什么做这件事

v2 计划的 packed IO 重构已经收官——正确性侧 G2 对拍 bitwise 全过，性能侧 S8a/S8b 全部 run 跑完，
真实墙钟口径下 packed 比 legacy 快约 30%（`result.md` 四节 4.2）。但绝对目标没达到：

- **GPU 利用率停在 89.2%（w8）/ 89.5%（w12）**，`E2E_ACCEPT` 五项判据中 util 与 0% 采样两项 FAIL；
- **加 worker 已经完全失效**（w8→w12 只 +0.3pp），说明纯参数调整的空间已经耗尽；
- 真实 epoch 9.26–9.58 h，距 8.6 h 阈值差约 7.7%（`result.md` 四节 4.1）。

本轮重新分析全部已完成实验后确认：**剩下的缺口与 dataloader 的读取速度已经无关**。它来自训练主循环
把「等 GPU 算完」和「取下一个 batch」串成了一条直线，使取数时间完整暴露成 GPU 空转。本计划先用一个
**不改训练语义、可逐位证明与 G0 等价**的调度改动（L1）把这段空转藏进 GPU 计算，再按 roadmap 既有的
决策门判断是否还需要削载荷（L2）或换 IPC 通道（L3）。

## 〇、本轮先订正的两个口径

### 0.1 真正过 IPC 的 batch 是 256.7 MB

一度出现过「313.2 MB」的说法，那是**上卡后 Observation** 口径，不是 worker→主进程的 IPC 载荷。
两张 224² 原图在 IPC 这一段是 **uint8 共 19.3 MB**；f32（共 77 MB）是 `device_put` 之后
`openpi/models/model.py::Observation.from_dict` 里 `astype(f32)/255*2-1` 才产生的。
**v2 计划 1.4 节字节帐的「~257 MB」是对的。** 连带结论：原图在 IPC 上已是最窄的 uint8，无窄化空间。

真实 IPC 载荷逐键：

| 键 | host 侧 dtype | 字节 |
|---|---|---|
| `static_image_emb` | bf16 | 134.2 MB |
| `static_pos_emb` | f32 | 100.7 MB |
| `static_state_emb` | **f64** | 2.10 MB |
| 两张原图 | **uint8** | 19.3 MB |
| `static_mask` / `actions` / `state` / `tokenized_prompt` 等 | bool/f64/i32 | 约 0.4 MB |
| **合计** | | **≈ 256.7 MB** |

（`static_state_emb` 在 host 侧是 f64，上卡后显示的 f32 是 jax 按 x64-disabled 规范化的结果。）

### 0.2 COLDHOT 与「节点漂移」的说法作废

分析初期曾用**步时中位数**比较 C1（冷）与 H1（热），得出「C1 比 H1 快 11–13%、冷态惩罚为负、
疑 A40 热降频或跨节点漂移」的结论。该结论**不成立**：两侧分布形状不同（C1 有 2.9% 慢步、p90=6.681；
H1 零慢步、中位≈均值），中位数比较正是 AGENTS 16 禁止的做法。按稳态墙钟均步时，
**C1 5.224 s vs H1 5.180 s = +0.7%，COLDHOT 判据通过、冷态惩罚可忽略**（`result.md` 二节）。

**连带作废**：「C1 的 4.576 s 击穿 compute-only 的 4.778 s，故跨节点绝对阈值不可判」这条论据同属
中位数假象（C1 均值 5.224 s，并未击穿）。因此本轮不把「步时/epoch 改相对判据」当作必需修正，
只保留为可选加强（见 3.4）。

**未受影响**：epoch 换算的中位数缺陷（`result.md` 三节已独立证实并量化），以及下面第一节的全部诊断
——它们建立在快慢步分层与步内相位分析上，与 epoch 口径无关。

## 一、诊断：缺口到底在哪里

以下全部为本轮实测，数据源是 `v1-store/bench/bottleneck/` 下各 run 的 `metrics.jsonl` 与
`gpu_util_dense.csv`，复现命令见第二部分 D 节。

### 1.1 天花板不是 90%

`v1-computeonly-b64` 走的是**同一套训练循环**（同样每步 `jax.device_get`），唯一差别是
`create_data_loader` 被换成「首 batch 缓存后无限重复」的 wrapper
（`scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py::_RepeatFirstBatchLoader`），
于是 `next(data_iter)` 变成零成本。结果：**util 均值 99.9%、0% 采样 0.0%**。

→ 计算图、FSDP、每步的 `device_get` 同步屏障**都不是**利用率缺口的原因。缺口只可能来自 `next(data_iter)`。

### 1.2 快步 / 慢步分层：缺口的 84% 集中在 25% 的步上

把稳态窗口按「步时 ≤ p10×1.15」分成快步与慢步，再分别做步内相位分析（把每个 dense 500 ms 采样点
按它在所属步中的相对位置分桶），`v1-framesamp-e2e-w8c16`：

| 分层 | 步数 | 墙钟占比 | 步时均值 | util 均值 | 0% 采样 | 空窗落点 |
|---|---|---|---|---|---|---|
| 快步 | 409（75%） | 68.1% | 4.948 s | **97.4%** | 1.5% | 相位 0.0–0.2 轻微（81%→93%→100%） |
| 慢步 | 140（25%） | 31.9% | 6.767 s | **71.6%** | 27.4% | 相位 0.0–0.3 深度停摆（27%→1%→8%） |

- 慢步贡献的缺口 = 28.4pp × 31.9% 墙钟 = **9.1pp**，占总缺口 10.8pp 的 **84%**。
- **快步已经 97.4%，本身就在目标线以上**——这是「L1 单独很可能就够」的直接依据。
- 两个分层的空窗**都落在步的开头**，步的后半段（快步相位 ≥0.2、慢步相位 ≥0.4）是**恒定 100.0%**、
  零个 0% 采样。GPU 计算段本身干净利落。

w12c16 同构（快步 97.9% / 慢步 71.2%），w4c16 因 worker 更少而快步也被拖到 92.0%。

### 1.3 慢步严格周期性，周期恰等于 num_workers

| run | 慢步间隔模式 | 周期和 | 慢步占比 | 慢步超快步基线的总延迟 |
|---|---|---|---|---|
| w4c16 | 4,4,4,4,… | 4 = W | 29.9% | 293 s / 3067 s |
| w8c16 | 5,3,5,3,… | 8 = W | 25.5% | 269 s / 2971 s |
| w12c16 | 5,5,2,5,5,2,… | 12 = W | 25.0% | 269 s / 2964 s |

**周期恒等于 worker 数、比例恒约 25%、总延迟恒约 270 s**——这是主进程与 worker 之间数据交接的节奏特征
（torch DataLoader 的 round-robin 取数 + `prefetch_factor=2`），既不是 NFS 抖动（S8a 实测 w8 的
dataloader 吞吐是训练需求的 3.4×），也不是 worker 产能不足。

### 1.4 交接为什么慢：IPC 通道只有约 520 MB/s

本机微基准（fake dataset，worker 只造零数组，零 IO、零反序列化；batch 形状/dtype 逐键照抄 GL 日志的
`Initialized data loader` 段）：

| 环节 | 实测 | 等效带宽 |
|---|---|---|
| worker→主进程 IPC（num_workers=4 / 8） | 591 / 601 ms | **约 520 MB/s** |
| 折算真实 IPC 载荷 256.7 MB | **约 490 ms** | 同上 |
| `jax.make_array_from_process_local_data`（H2D） | **9.5 ms** | 33 GB/s |

根因：`openpi/training/data_loader.py::_collate_fn` 返回的是 **numpy** 数组，而 torch 的
shared-memory 零拷贝 reduction 只对 `torch.Tensor` 生效，于是整批走标准 pickle + 管道，
并且在**主进程单线程**完成反序列化。这解释了两件事：为什么 w4 与 w8 的 IPC 耗时完全一致；
为什么加 worker 对 util 毫无帮助。**H2D 只占 2%，不是问题。**

### 1.5 根因：主循环里两行语句的先后

`scripts/train.py::main` 的 `for step in pbar:` 循环体，关键三步的现有顺序是：

1. `ptrain_step(...)` —— 异步 dispatch，立刻返回；
2. `jax.device_get(reduced_info)` —— `log_interval=1` 时每步执行，**阻塞直到 GPU 算完本步**；
3. `batch = next(data_iter)` —— **屏障之后**才去取下一个 batch，此时 GPU 完全空转。

取数（含慢步那 ~1.9 s 的交接阻塞）100% 暴露在 GPU 空窗里，与计算零重叠。

### 1.6 顺带查出的一件事：roadmap 项 1 的前提不成立

v2 计划 1.5 节第 5 条与 roadmap 项 1 都以「每个 spawn worker 在 GPU0 建约 442 MiB CUDA context」
为前提。本轮核对 GL packed run 的 `compute_apps.csv`（`nvidia-smi --query-compute-apps` 采样存证）：
**全程只有主进程一个 pid**，worker 侧 CUDA context **没有出现**。

→ roadmap 项 1 应从「性能补救首选项」上撤下，其立项前提需重新采集后再议。

## 二、四层优化方案详解（L0 / L1 / L2 / L3）

四层按「改动量」从小到大排列。**先给结论**：L0 够不到目标，L1 是本轮首选且很可能单独达标，
L2/L3 是余量加固，按 roadmap 决策门在 L1 收官后再定。

| 层 | 一句话 | 改哪些文件 | 收益 | 能否达标 | 与 G0 的对齐强度 |
|---|---|---|---|---|---|
| **L0** | 选对档位与线程数 | 提交参数 + 1 个 sbatch 的 2 个环境变量 | util +1–2pp | ❌ **够不到** | 不涉及（不改任何逻辑） |
| **L1** | 让取数与 GPU 计算重叠 | 1 个文件（落法不同则不同） | **util +7–9pp** | ✅ **很可能单独达标** | ✅ **逐位 bitwise** |
| **L2** | 不传 pos/state，GPU 侧查表 | **9 个 `src/` 文件**（5 个在红线内）+ 4 个验证资产 | IPC −40% | 余量加固 | ⚠ 输出侧 bitwise **不可得** |
| **L3** | IPC 换 shared memory 通道 | 1 个自建 loader + **改动量具本体** | IPC −98% | 余量加固 | ⚠ 理论可得，**但尺子要换** |

---

### 2.0 L0：纯配置、零代码 —— 必须说在前面：够不到目标

**改哪些（两项，都不碰任何 Python 代码）**

1. **提交参数 `WORKERS` 固定为 8**。依据是 S8b 实测：w8 慢步墙钟 0.0%，而 w12 反而升到 10.7%
   （`result.md` 一节表），util 只从 89.2% 到 89.5%——**加 worker 已无增益且引入慢步**，
   超订拐点早于原先预期的 w16。
2. **`scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` 里的 `OMP_NUM_THREADS` / `MKL_NUM_THREADS`
   从 `$NCPU`（=16）降到 1–2**。现状是每个 spawn worker 都继承 `OMP_NUM_THREADS=16`，
   w12 档即 12 worker × 16 线程压 16 核，与 w12 慢步 util 59.5%（「变慢但仍在算」，
   区别于 w4 慢步的 7.7%「完全停摆」）的特征吻合。

**为什么只有这点收益**：L0 能压掉的是**线程超订**引起的那部分慢步，而 1.2/1.3 节已经证明，
缺口的主体是**每 `num_workers` 步一次、总量约 270 s 的数据交接阻塞**——它是主循环
「等 GPU → 才取数」这个结构造成的，与线程数无关。w8 档慢步墙钟本来就已经是 0.0%，
L0 在这一档几乎无事可做。

**预期（估算，非实测）**：util 89.2% → 90–91%，0% 采样仍在 9% 上下，**过不了 ≤5%**。

**结论**：**纯配置改动够不到 D 节阈值，也够不到用户要的 95%。** 但 L0 成本为零、与 L1 不冲突，
建议在本轮 V4 的 GL run 里一并带上（见五节）。

---

### 2.1 L1：让取数与 GPU 计算重叠 —— 本轮首选

**改的是什么（人话）**：现在主循环是「发计算任务 → 等它算完 → 再去拿下一批数据」，
GPU 在「去拿数据」这段时间里完全闲着。L1 就是把「去拿数据」挪到「等它算完」之前（或交给
一个后台线程），让搬数据和算数据同时发生。**不改任何一个数、不改任何一个张量、不改喂给
模型的顺序**，只改「什么时候去搬」。

#### 2.1.1 三种落法，各自改哪些文件

| | **L1a** | **L1b（本轮采用）** | **L1c** |
|---|---|---|---|
| **改哪个文件** | `scripts/train.py` | `scripts/smoke-local/bench_train_steps.py` | `src/mme_vla_suite/training/dataloader.py` |
| **改成什么样** | 把 `main()` 的 `for step in pbar:` 里那一行 `batch = next(data_iter)` 从 `device_get` **之后**移到**之前**（合法插入点：`infos.append(info)` 之后、`if step % log_interval` 块之前） | 新增一个 monkeypatch：包住 `_train._data_loader.create_data_loader`，在它返回的 `DataLoaderImpl` 外面再套一层后台预取层（有界队列 + 单线程）。由 `BENCH_PREFETCH` 环境变量控制，**默认关闭** | 在 `DataLoaderImpl.__iter__` 里加同样的后台预取层（同样默认关闭） |
| **改动规模** | **一行位置调换** | 约一个类 + 一处安装函数 | 约一个类 + 改一个方法 |
| **谁受益** | 三条链全部（`train.py` 自身、bench、compute-only 都跑同一个 `train.main`） | **只有 bench / GL 验收链**（正式训练入口不受益） | 三条链全部（都经 `create_data_loader` → `DataLoaderImpl`） |
| **红线** | ❌ **撞 R1 + R2 双硬红线**（R2 首位点名「`scripts/train.py` 不动」），且会让 `G0_SCOPE` 断言输出非空，需显式解禁 | ✅ **T1 白名单内整目录**，且被 R2 的「验证验收资产参数化」除外条款覆盖 | ✅ 不在 R2 不动清单（v2 的 B.0 表把它列在「修改·接线」行），走三块验证即可 |
| **index 序列** | **逐位不变（n=8072）** | 尾部超前量 `pf×w` → `pf×w + k`，**前缀逐位不变** | 同 L1b |
| **实现复杂度** | 最低 | 线程 + 有界队列 + 与 `finalize()` 的 join 时序 | 同 L1b，但落在生产代码里 |

**本轮采用 L1b**（用户 2026-08-28 拍板）：先用零红线风险的落点拿到 GL 上的实据，再拿实据决定
是否解禁 R2 走 L1a、或走 L1c。GL 的 e2e 入口 `gl_e2e_fix.sbatch` 调的正是 `bench_train_steps.py`，
所以 L1b 足以拿到完整 GL 验收数据。

#### 2.1.2 本机实测：重叠确实能把空转吃掉

微基准把 GPU 计算段用 `jax.lax.fori_loop` 调到 4.258 s（与 GL 同量级），配一个与真实 batch
逐键同形同 dtype 的 fake dataset（worker 只造零数组，零 IO 零反序列化），本机 2×RTX 6000 Ada：

| 调度 | 步时均值 | max | 相对纯计算的开销 |
|---|---|---|---|
| 纯 GPU 计算（参照） | 4.258 s | — | — |
| **串行（复刻现状）** | **5.272 s** | **9.31 s** | +23.8% |
| **同线程重叠**（取数移到屏障前） | **4.629 s** | **4.74 s** | +8.7% |
| **后台线程预取** | **4.691 s** | **4.81 s** | +10.2% |

三点读数：① 省 0.64 s（−12.2%）；② **长尾从 9.31 s 塌到 4.74 s，周期性慢步完全消失**——
这正是 1.3 节那个「每 W 步一次」的交接阻塞被藏进计算的直接证据；③ 两条路径效果相当，
说明 **GIL 不构成障碍**（主线程等 GPU 时会释放 GIL，预取线程能拿满）。

> 口径注记：该微基准的 batch 按 313 MB 构造（原图用了 f32），**大于真实 IPC 载荷 256.7 MB**。
> 也就是说真实 IPC 比微基准更短、更容易藏住，**上表的重叠收益是保守估计**。

#### 2.1.3 折算到 GL（w8c16 档）

| 指标 | 现状（实测） | L1 后（预估） | 目标 |
|---|---|---|---|
| util 稳态均值 | 89.2% | **96–98%** | ≥95% |
| 0% 采样占比 | 9.7% | 1–2% | ≤5% |
| 步时**均值** | 5.412 s | 4.85–5.00 s | — |
| 步时中位 | 4.865 s | 4.85–4.90 s | ≤5.00 s |
| epoch（**均值**口径） | **9.28 h** | **8.3–8.6 h** | ≤8.6 h（余量小） |

预估的依据是 1.2 节的分层实测：**快步（占 68% 墙钟）已经 97.4%**，L1 把慢步那 ~1.9 s 的交接
阻塞藏进 4.8 s 的 GPU 计算之后，全部步趋向快步形态。

> ⚠ 此处订正一个早期口径：曾按**步时中位**折算出「4.865 → 4.35–4.5 s、epoch → 7.5–7.7 h」，
> 那是中位数口径且低估（`result.md` 三节已证实 epoch 必须按均值换算）。按均值口径，
> **epoch 落在 8.3–8.6 h，对 8.6 h 阈值余量很小**——util ≥95% 能达成，epoch 仍需单独判读。

#### 2.1.4 为什么 L1 能保住逐位 bitwise

memory token 由四个因素完全决定（v2 计划三节），L1 一个都没碰：① 每 step 取哪些样本
——同一个 `data_iter`、抽取顺序不变；② 每样本选哪 32 帧——`even_sampling_indices` 不动；
③ 每帧特征的字节——packed 库不动；④ 交付 dtype——不动。

另外 `train_rng` 是循环外常量、在 `train_step` 内按 `jax.random.fold_in(rng, state.step)` 折入，
**host 侧的取数时机与记录频率对数值零影响**。因此 L1 可以直接要求最强判据：G2 那套四分项
原样复用，一行收官是 `scalars_hex.tsv` 的 sha256 仍等于 `c799a0b2…`。

---

### 2.2 L2：不传 pos/state，GPU 侧查表 —— 把 IPC 载荷砍掉 40%

即 roadmap 的**项 2（pos_emb 移出 batch）+ 项 3（`static_state_emb` 清除）**。roadmap 已把 scope
规格化，本轮核对确认其规格仍准确，并补出两条它没写的坑。

**改的是什么（人话）**：`static_pos_emb` 是 100.7 MB 的大张量，但它其实是「第几帧」的纯函数
——同一个帧号在任何 episode 里都是同一串字节，packed 库里已经存成一张 586 行的小表。
所以完全不必每步把它从 worker 搬到主进程再搬上卡，只要搬 32 个帧号（8 KB），
让 GPU 从常驻的小表里查出来就行。`static_state_emb` 更简单：`use_state_emb=false`，
GPU 根本不用它，纯属白算白传。

**收益**：IPC 载荷 256.7 MB → **153.9 MB（−40%）**，IPC 耗时约 490 ms → 约 294 ms。
与 roadmap 项 2 原文的「约 40%」一致。

#### 2.2.1 要改哪些文件（9 个 `src/` + 4 个验证资产）

| 文件 | 改什么 | 红线 |
|---|---|---|
| `training/framesamp_dataset.py` | `__getitem__` 不再交付 pos 张量，改交付定长 int32 帧号（`-1` 哨兵补位）；`_pad` 同步调整 | ✅ 可改（走三块验证） |
| `datastore/framesamp_store.py` | pos 小表不再逐样本 `pos_rows`；表本身改为供设备端加载 | ✅ 可改 |
| `training/config.py` | **两处** `RepackTransform` 结构表增删键 | ⚠ **不在 v2 的 B.0 授权范围** |
| `models/integration/history_observation.py` | `HistAugObservation` 字段增删、`from_dict`/`to_dict`/`from_base_obs` 同步 | ❌ **R2 硬红线** |
| `models/integration/history_pi0.py` | `HistoryPi0Config.inputs_spec` 的 perceptual 分支（决定 init 与 HLO）、`embed_memory` 调用点 | ❌ **R2 硬红线** |
| `models/representation/percep_mem.py` | `__call__` 形参与断言 | ❌ **R2 硬红线** |
| `models/representation/mem_encoder.py` | `encode_perceptual_memory` 形参；新增设备端 gather + mask 清零 | ❌ **R2 硬红线** |
| `policies/robomme_policy.py` | `RoboMMEInputs.__call__` 的键透传 | ⚠ 灰区（R2 未点名，也不在 B.0 内） |
| `policies/policy.py` | `MME_VLA_Policy._prepare_history` 的在线侧生产 | ⚠ 灰区，且其 pos 来自 R2 不动的 `MemoryBuffer` |
| 验证资产 ×4 | `scripts/dtype-unify/_common.py` 的 `MEMORY_KEYS`、`compare_dtype_fix.py`、`test_padding_dtype.py`、`scripts/data-pack-framesamp/test_pack_guards.py` 的 dtype 断言 | ✅ 白名单内 |

**5 个在 R2 硬红线内**（`src/mme_vla_suite/models/**` 整目录），**v2 期间绝对禁改**
——L2 必须等 v2 收官后按 roadmap 项 2 独立立项。

#### 2.2.2 三个必须先解决的前置

1. **判据得先自己造**（roadmap 原文）：G0 的量具只记录 collate 后、上卡前的 host batch；
   L2 落地后 pos 不再出现在 host batch，「与 G0 的 pos 摘要对拍」**目前无从采集**。
   第一步必须先落地**设备端观测点**（在 gather、mask 清零、展开之后、进投影层之前把有效 pos
   取回 host 做逐位摘要），这部分工作量计入 L2 本身。
2. **`-1` 哨兵与 mask 粒度**（后者是本轮新补的坑）：短样本现语义是 pos 补**零向量**；改交付索引后
   若用 `0` 补位，gather 会取回真实 `pos[0]` 而**不是零**，**不等价**——必须定长 int32 + `-1` 哨兵
   + gather 后按 mask 显式清零 + 越界测试。而且现有 `static_mask` 是 **token 级 `(b,512)`**
   （帧级 mask 经 `np.repeat` 展开而来），gather 索引却是**帧级 `(b,32)`**，
   清零时必须显式处理这个粒度差——**roadmap 未涉及此点**。
3. **legacy 回滚链与在线推理链的裁决**（本轮新补的坑）：`MMEVLA_DATA_BACKEND=legacy` 是
   `_resolve_backend` 的**默认值**且仍交付 pos 张量，模型侧一改签名 legacy 立刻崩；而
   `training/dataset.py` 与 `shared/**` 都在 R2 不动清单、R7 又要求旧链路原地保留。
   在线侧同理——`policies/policy.py::_prepare_history` 显式写这两个键，其 pos 来自
   R2 不动的 `shared/mem_buffer.py::MemoryBuffer`，且在线表按 4096 步现算、训练表 586 行，
   **长度口径不同，step 超界即越界**。→ L2 计划必须显式裁决：同步迁移（破 R2/R7），
   还是宣布 legacy backend 与在线旧路径报废（破 v2 的回滚保障）。

#### 2.2.3 对齐强度：输出侧 bitwise 天然不可得

L2 改的是**模型输入签名**，`ptrain_step` 是 `jax.jit` 的 → **HLO 必变、编译缓存必失效**，
输出侧的逐位等价从原理上就拿不到。主判据只能退到「输入侧逐位 + 输出侧量化复核」，
而输入侧的 pos 摘要还得先造观测点才采得到。**这是 L2 排在最后的根本原因。**

---

### 2.3 L3：把 IPC 通道换成 shared memory —— 根治，但要换尺子

**改的是什么（人话）**：现在 worker 把整批数据 pickle 一遍、经管道发给主进程，主进程再单线程
反序列化，实测只有 520 MB/s。torch 对 `torch.Tensor` 有共享内存的零拷贝通道，走通了这段就从
约 490 ms 降到约 10 ms。

#### 2.3.1 三个技术阻断点（本轮已用仓库自带 venv 实测，jax 0.5.3 / torch 2.7.1）

| 实测项 | 结果 |
|---|---|
| `torch.as_tensor(ml_dtypes.bfloat16 数组)` | ❌ `can't convert np.ndarray of type bfloat16` |
| `np.asarray(torch bf16 tensor)` | ❌ `Got unsupported ScalarType BFloat16` |
| `jax.make_array_from_process_local_data(sharding, torch.Tensor)` | ❌ `Cannot interpret 'torch.bfloat16' as a data type` |
| `uint16` 位视图双向桥 | ✅ 可用 |
| `jax.dlpack.from_dlpack(cpu torch bf16)` → `jax.device_put` | ✅ 可用 |

即：collate 一旦改出 torch tensor，① bf16 在 collate 内就转不过去（必须走 uint16 位视图）、
② 主进程的接收端会直接 raise（不是慢，是不能用）、③ bench 的摘要链会当场炸。
唯一自洽的接线是「collate 出 torch（bf16 走 uint16 位视图）→ 主进程 dlpack 还原 → `device_put`」。

#### 2.3.2 要改哪些文件

- **不能改 `src/openpi/training/data_loader.py`**：它是 R2 硬红线，且「给 `TorchDataLoader` 加
  `prefetch_factor` 形参」已被三份文档（v2 计划 5.1、B.0 表、roadmap「已裁定不做」）裁定不实施，
  理由正是「与不动 `src/openpi/**` 红线矛盾」——该理由对 `pin_memory`、对改 `_collate_fn` 同样适用。
- **唯一合规落点**：在 `src/mme_vla_suite/training/dataloader.py` 里自建一个 `TorchDataLoader`
  等价类，自带 collate（torch + uint16 桥）与 dlpack 交付。但必须**逐字复刻 index 序列的全部恒等要素**
  才能满足 R3：`shuffle`/`sampler` 互斥、`persistent_workers=num_workers>0`、`drop_last=True`、
  `generator.manual_seed(seed)`、spawn context、`worker_init_fn`、以及「`StopIteration` 即重建
  `iter()`」的跨 epoch 外层循环。
- **连带必须改量具**：`scripts/smoke-local/bench_train_steps.py` 的三处安装点
  （patch `TorchDataLoader.__iter__`、其源码守卫、`_install_idx_probe` 取 `torch_loader`）全部要跟着改。

#### 2.3.3 三个代价

1. **要动 G0/G2 的量具本体**：`_install_batch_digest_recorder` 的 `record()` 依赖
   `np.asarray(leaf)` / `arr.tobytes()` / `dtype.kind in "fV"`，三处对 torch tensor 全部失效；
   `main()` 对 `TorchDataLoader.__iter__` 的源码守卫也会先炸。`bench_train_steps.py` **是**
   G0/G2 对拍的尺子，改它等于改判据口径——**这是 L3 与 L1 的本质差别**：L1 用原尺子量，
   L3 得先换尺子再量。
2. **GL 内存帐要重算**：torch shm 计入 cgroup 的 shmem，而 `greatlakes.md` 已实证 anon+shmem
   **不可回收**。256.7 MB × `prefetch_factor` × workers 是新增的常驻不可回收内存，
   w16 档约 8–10 GB，现申请的 `--mem=96G` 需按档位重新核算。
3. **既有验收判据失效**：`scripts/data-pack-framesamp/spawn_matrix.py` 的 PASS 判据是
   「主进程 fd 计数回到基线」，torch 默认 `file_descriptor` sharing 会新增数十~上百常驻 fd，
   该判据需重定基线。

---

### 2.4 分水岭：四层与 G0 的对齐强度

`ptrain_step` 是 `jax.jit` 的：**模型输入签名一旦变化，HLO 必变、编译缓存必失效，输出侧的
逐位等价就天然不可得**。这条把四层劈成两类：

| 层 | 输入签名 | HLO | 量具是否要改 | **与 G0 的最强可得判据** |
|---|---|---|---|---|
| **L0** | 不变 | 不变 | 否 | 不适用（不改任何逻辑，无需对拍） |
| **L1** | **不变** | **不变** | **否** | ✅ **逐位 bitwise**：G2 全套四分项 + `scalars_hex.tsv` sha256 相等 |
| **L2** | **必变** | **必变** | 是（要造设备端观测点） | ⚠ 输入侧逐位 + 输出侧量化，**输出侧 bitwise 不可得** |
| **L3** | 不变 | 不变 | **是（尺子本身）** | ⚠ 理论可得，但判据口径需重新定义后才能主张 |

**L1 是唯一既可能达标、又能用原来那把尺子证明逐位等价的方案**——这是它排第一的根本原因，
也是本轮先做 L1 的全部理由。

## 三、本轮要做的事：L1b

用户 2026-08-28 拍板：**先走 L1b（bench 侧、白名单内）拿 GL 实据，再决定最终落点。**

### 3.1 落点与形态

改 **`scripts/smoke-local/bench_train_steps.py`**（T1 白名单内整目录；R2 的「验证验收资产参数化」
除外条款覆盖）：新增一个 monkeypatch，包装 `_train._data_loader.create_data_loader`，在其返回的
`DataLoaderImpl` 外面再套一层**后台预取层**（有界队列 + 单预取线程）。主线程 `next()` 从队列取，
队列非空时瞬时返回；预取线程在主线程被 `device_get` 阻塞（GPU 计算中、GIL 已释放）的窗口里把 IPC 做完。

结构模板照抄仓库已有先例
`scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py::_RepeatFirstBatchLoader`
——它演示了「包一层 loader、透传 `data_config()`、自定义迭代、经 `create_data_loader` monkeypatch
装配」的最小写法，且已在真实 bench 链路验证过。

**为什么 L1b 覆盖得住 GL 验收**：GL 的 e2e 入口 `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch`
调的就是 `bench_train_steps.py`，所以 L1b 足以拿到完整的 GL 实据。它的局限是正式训练入口
（`scripts/train.py` 自身）不受益——那是阶段二的事。

### 3.2 四条硬要求

1. **默认关闭、显式开启**（R2 除外条款要求「默认值必须等价现状」）：`BENCH_PREFETCH` 未设时走现行为、
   逐字节不变，显式设置才启用。全部历史 run 与 G0/G2 基线的可比性零受损，回滚只需去掉环境变量。
   这一形态与 R16 的 `MMEVLA_DATA_BACKEND` 显式三态风格一致。
2. **不做成 generator**：用显式迭代器对象（`__iter__` / `__next__` / `close()`）自管线程与队列生命周期；
   generator 被 GC 时的 `GeneratorExit` 时序不可靠。队列深度 k 从 **1** 起。
3. **异常必须透传**：预取线程捕获的异常要在主线程重新 raise，不能吞成静默挂起。
4. **`finalize()` 之前必须 join 预取线程**：否则 `_LoggingSampler` 的 `index_log` 会被并发 append，
   `index_sequence.json` 的 `n` 非确定、甚至截到半个 batch。

### 3.3 已核实的三个前提

- **改代码不会让 G0 基线失效**：`scripts/smoke-local/check_baseline_env.py` 的 T5 指纹清单只含
  `uv.lock`、库版本、GPU/驱动、git 外资产（norm_stats / pi05_base / tokenizer / manifest sha /
  数据集抽样）、XLA_FLAGS、产物 sha，**不含仓库代码文件的 sha**。`BASELINE_ENV=PASS` 仍成立，
  G0 固化产物可直接引用做 bitwise 对拍。这是整个验证方案成立的前提。
- **显存**：batch 按 B 轴 sharded 到 4 卡，多驻留一个 batch 只增加 256.7/4 ≈ **64 MB/卡**，
  46 GB 卡上可忽略；`ptrain_step` 的 `donate_argnums=(1,)` 只捐 `train_state`，batch 不在捐赠列表，
  无 donation-after-use 风险。
- **index 序列**：预取只让尾部超前量从 `prefetch_factor × workers` 变成 `prefetch_factor × workers + k`，
  **前缀逐位不变、不重排**（满足 R3）。`compare_baseline.py::compare_index_seq` 取
  `min(n_a, n_b)` 共同前缀，判据天然容忍；`run_2gpu_epoch_bench.sh` 的 `n ≥ steps × batch`
  只增不减，仍成立。

### 3.4 判据口径

**必改（`result.md` 三节已证实，用户 2026-08-28 拍板）**：`analyze_gpu_util.py` 的 epoch 换算从
`EPOCH_STEPS × 步时中位` 改为 `EPOCH_STEPS × 步时均值`，中位口径保留为附加信息以便与历史 run 对照。
这同时回应了 `result.md` 五节待办 1。

**主判据（用户指定）**：dense 稳态 **util 均值 ≥95%**；0% 采样 ≤5%；慢步(>8s)墙钟 ≤5%。

**可选加强（用户已拍板，但论据已按 0.2 节降级）**：给 `gl_e2e_fix.sbatch` 加一个**默认关闭**的
`COMPUTEONLY_REF` 分支——同一 allocation 内先跑 100 步 compute-only 拿同节点下界，再跑 600 步 e2e，
把步时判据表述成 `e2e 步时均值 / 同节点 compute-only 步时均值 ≤ 1.05`。成本 +8 分钟/job。
它的价值是消除跨节点比较的隐患并给出干净的残差分解，**不是**因为绝对阈值不可判。

## 四、验证阶梯：如何证明与 G0 对齐

沿用 v2 计划三节的三块结构，判据全部现成、不造新工具。**V2 不过不开 V4**（沿用 T9-B6 三块秩序）。

| 步 | 内容 | 判据 | 成本 |
|---|---|---|---|
| **V1 第一块·不训练** | 预取开/关两侧各 dump 一次 index 序列 | 共同前缀逐位一致；尾部超前量 = `pf×w + k`（只增不重排） | 分钟级 |
| **V2 第二块·本机 bitwise（核心闸）** | 预取开启，本机 2 卡 b8 **1000 步** seed 42 确定性档，摘要步集与 G0 对齐，`compare_baseline.py` 离线对拍 G0 r1 固化产物 | `SCALARS hex_mismatch_steps=0` + `STATE_DIGEST mismatch=0` + `CANON_CHECK=PASS` + `INDEX_SEQ=PASS`；一行收官：`scalars_hex.tsv` 的 sha256 == `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757` | ~33 min + 对拍 |
| **V3 本机 speed 先导** | speed 统一口径 1000 步（无摘要、生产 XLA 档），vs 锚点 `v1-g0-speed-r2`（1.152 s/step、util 86.5%、0% 采样 4.9%） | 不设阈值只报数；**最早的「有没有效果」信号** | ~20 min |
| **V4 第三块·GL 验收** | GL e2e w8c16 600 步（4×A40/16C/96G，2 h），`ANALYZE_ACCEPT=1` | **util 稳态均值 ≥95%** + 0% 采样 ≤5% + 慢步墙钟 ≤5% + epoch（均值口径） | 1 h + 排队 |

**L1 为什么能保住 bitwise**：memory token 由四个因素完全决定（v2 计划三节）——① 每 step 取哪些样本、
② 每样本选哪 32 帧、③ 每帧特征的字节、④ 交付 dtype。L1 一个都没碰：同一个 `data_iter`、
抽取顺序不变；`even_sampling_indices` 不动；packed 库不动；交付 dtype 不动。另外 `train_rng` 是
循环外常量、在 `train_step` 内按 `jax.random.fold_in(rng, state.step)` 折入，**host 侧的取数时机
与记录频率对数值零影响**。

其余纪律：起跑前 `BASELINE_ENV=PASS` preflight 必过；每个 >5 min 的 run 按 AGENTS 17 留档
`docs/training-doc/<run_name>/`；run_name 起跑前逐个交用户确认（AGENTS 6）；GL 的 4×A40 / 2 h
超出 `greatlakes.md` 硬限，提交前逐个走资源审批并留放行记录。

## 五、预期数字与后续分叉

| 指标 | 现状 w8c16 | L1 后预估 | 目标 |
|---|---|---|---|
| util 稳态均值 | 89.2% | **96–98%** | **≥95%** |
| 0% 采样占比 | 9.7% | 1–2% | ≤5% |
| 步时均值 | 5.412 s | 4.85–5.00 s | — |
| epoch（均值口径） | **9.28 h** | **8.3–8.6 h** | ≤8.6 h（余量小） |

依据是 1.2 节的分层实测：快步（68% 墙钟）已经 97.4%，L1 把慢步那 ~1.9 s 的交接阻塞藏进 4.8 s 的
GPU 计算之后，全部步趋向快步形态。本机微基准（把 GPU 计算段调到 4.258 s、配 313 MB batch）实测：
串行均值 5.272 s / max 9.31 s → 重叠后均值 4.629 s / **max 4.74 s，长尾完全消失**，
是同一机制的直接证据。

**注意**：epoch 一项余量很小（预估 8.3–8.6 h 对阈值 8.6 h）。util ≥95% 是用户指定的主目标，
epoch 若仍卡线需单独判读，不因其失守而否定 util 结论。

**L0 一并带上**：2.0 节的两项纯配置改动成本为零、与 L1 不冲突，本轮 V4 的 GL run 直接采用
`WORKERS=8` 并把 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 降到 1–2。注意这会让 V4 同时含两个变量
（L1 的预取 + L0 的线程数），若需要单独归因，可在 V4 之后补一档「L1 开 + L0 不开」的对照；
但由于 w8 档慢步墙钟本来就是 0.0%、L0 在该档几乎无事可做，**默认不安排这一档对照**。

收官后按 roadmap 决策门分叉（用户 2026-08-28 拍板）：

- **util ≥95%** → L2（roadmap 项 2/3）与 L3 **缓做或不做**（roadmap 决策门原文），本轮结束，
  回填 v2 计划 T8 登记簿与 roadmap；
- **93–95%** → 先上 **L3**（仍属 bitwise 可对齐一类），但须先解决 GL shmem 内存帐与 bench 量具口径；
- **<93%** → 再评估 L2，且必须先解决三件事：设备端 pos 观测点、`-1` 哨兵 padding 与帧级/token 级
  mask 粒度、legacy 回滚链与在线推理链的裁决。

---

# 第二部分（技术细节，供 agent 追踪）

## A. 改动清单（本轮授权范围，超出即越界）

| 文件 | 改什么 | 红线归属 |
|---|---|---|
| `scripts/smoke-local/bench_train_steps.py` | 新增预取层安装函数：包 `_train._data_loader.create_data_loader`，返回值再套预取层（有界队列 + 单线程）。`BENCH_PREFETCH` 默认 `0`。`main()` 的 `finally` 里在 `finalize_digests()` **之前** join/close。新增一条**恒定生效**的 `create_data_loader(` 源码断言（现有那条只在 `BENCH_DUMP_IDX=1` 时检查） | T1 白名单内；R2 除外条款（默认值等价现状） |
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | `epoch_h` 改用稳态窗口**均值**步时（中位口径保留为附加信息同时打印）；`ACCEPT_THRESHOLDS` 的 `util_mean_pct` 改 95.0 | 不在 T1 白名单但在 v2 的 B.0 表内（S7.5 已参数化过），`G0_SCOPE` 输出需逐 hunk 说明 |
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | 透传 `BENCH_PREFETCH`；**L0 第 2 项**：`OMP_NUM_THREADS` / `MKL_NUM_THREADS` 由 `$NCPU` 改为 1–2（建议同样做成可覆盖的默认值）；（可选加强）新增默认关闭的 `COMPUTEONLY_REF` 分支 | 同上 |
| 提交参数（不改文件） | **L0 第 1 项**：V4 起 `WORKERS` 固定为 8 | 不涉及 |

**明确不动**：`scripts/train.py`、`src/openpi/**`、`src/mme_vla_suite/models/**`、
`src/mme_vla_suite/training/dataset.py`、`src/mme_vla_suite/shared/**`（均属 v2 计划 G 节 R2 硬红线）；
`src/mme_vla_suite/training/dataloader.py`（本轮不动，留给阶段二的 L1c）。

## B. 关键实现点

- **预取层结构**：`__init__(inner, depth)` 存 `inner`；`data_config()` 透传 `inner.data_config()`；
  迭代对象内部起 `threading.Thread(daemon=True)` 跑「从 `inner` 拉取并入队」的循环，异常存入实例字段；
  `__next__` 先检查异常字段并 re-raise，再从队列取；`close()` 置停止标志 + 排空队列 + join。
- **与现有 monkeypatch 的关系**：`_install_batch_digest_recorder` 替换的是
  `_openpi_dl.TorchDataLoader.__iter__`（内层），预取层套在 `DataLoaderImpl`（外层）之外，两者不冲突。
  副作用：batch 摘要的 `record()` 会在预取线程里执行——性能 run 里 `BENCH_BATCH_DIGESTS=0` 无影响；
  正确性 run 里是单线程串行写文件、无竞态，但 `batch_digests.jsonl` 的 `digest_seconds` 语义从
  「主线程停顿」变成「后台线程耗时」，须在留档写明。
- **`openpi/training/sharding.py::_MeshState.active_mesh` 是类级全局、非 thread-local**，
  但预取线程不调 `set_mesh`，且 `make_array_from_process_local_data` 不读 `active_mesh`
  （只有 `activation_sharding_constraint` 读），当前不冲突——这是隐含前提，实现里需注释记一笔。
- **绝不能把取数挪到 `ptrain_step` 之前**（阶段二若走 L1a/L1c 时同样适用）：那样 step *s* 用的 batch
  会从 idx *s* 变成 idx *s+1*，是「看起来只是重排」的静默错帧，G0/G2 对拍会全线 FAIL。
  L1a 的唯一合法插入点是 `infos.append(info)` 之后、`if step % log_interval` 块之前。
- **现有 fail-loud 断言探测不到语句顺序变化**：`bench_train_steps.py` 与
  `compute_only_train_steps.py` 的 `inspect.getsource` 守卫全是子串存在性检查。这是隐患而非好消息
  ——阶段二若改 train.py，应同时加一条顺序敏感断言。

## C. run 命名与留档（起跑前逐个交用户确认）

| 用途 | 建议 run_name | 族 |
|---|---|---|
| V2 本机 bitwise | `v1-prefetch-g3` | 正确性族（确定性 XLA 档 + 摘要） |
| V3 本机 speed | `v1-prefetch-g3-speed` | 性能族（无摘要 + 生产 XLA 档） |
| V4 GL 验收 | `v1-prefetch-e2e-w8c16` | 性能族 |

commit 切分沿用 v2 体例：本文件落档走 `docs:` → `commitV3.6`（L1b 工具 + epoch 口径修正）→
launch.md 预提交 → clean HEAD 起跑 → 结果留档提交。收官后回填 v2 计划 T8 登记簿。

## D. 本轮诊断的可复现命令

全部为只读分析，数据源是既有 run 的记录目录，无需重跑训练。

```bash
# 各档 util / 步时 / accept 判定（--steps 按 run 实际步数：600 或 300）
uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py \
  v1-store/bench/bottleneck/v1-framesamp-e2e-w8c16 --steps 600 --accept
```

- **步内相位分析**与**快步/慢步分层**：把 `metrics.jsonl` 的逐步 `wall_time` 差分作为步边界，
  将 `gpu_util_dense.csv` 的每个采样点按其在所属步中的相对位置分桶统计 util 均值与 0% 占比；
  分层阈值取「步时 ≤ 稳态窗口 p10 × 1.15」为快步、其余为慢步。
- **慢步周期性**：对同一组步时差分取「> p10×1.15」的步号序列，看相邻步号间隔——三档分别呈现
  周期 4 / 8 / 12，恰等于各自的 `num_workers`。
- **IPC / H2D 微基准**：构造与 GL 日志逐键同形同 dtype 的 fake dataset（worker 只造零数组），
  用 `torch.utils.data.DataLoader`（spawn、`persistent_workers`、自定义 collate 做 `np.stack`）
  测稳态 `next()` 耗时；H2D 单独用 `jax.make_array_from_process_local_data` + `block_until_ready` 测。
- **重叠可行性**：把一段 GPU 计算（`jax.lax.fori_loop` 包住的矩阵乘链，调到与 GL 同量级的约 4.3 s）
  与上述 DataLoader 组合，分别测「串行（复刻现状）」「同线程重叠」「后台线程预取」三种调度的步时分布。
