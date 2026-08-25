# v1-framesamp-restructure-plan 对抗验证报告

> 被审对象：`v1-framesamp-restructure-plan.md`（定稿 2026-08-24 23:19，尚未实施）
> 方法：10 个维度独立证伪式核查 → 每条质疑经两名独立复核员（攻证据链视角 / 查误读与计划自答视角）判定 → 本报告汇总并对分裂意见亲自裁决。
> 汇总人复核方式：所有 high 条目与全部分裂条目均由本人重新读码/实测复算，不采信单方转述。

---

## 一、总评

**计划整体可信度高，可以在补齐下列条目后进入实施。** 核心事实基座（生产侧格式、消费侧取数链、字节账、算术账、数据实测）经十个维度逐条核查基本准确：npy 内部偏移（262,595 / 541,352 / 602,906）、单文件 602,951 B、`pos_emb_4x4` 跨 episode 逐字节恒等、6.27% 短样本占比、31.7 GB 新库体积、v1-e2e-b64 的 69.7%/27.8%/32.9% 三项 util 判读——本人独立复算全部精确复现。计划的方法论（判据梯子 + 可证伪硬判据 + 降级路径 + 守卫测试）本身健壮，多数问题即使不修也会被 C 节梯子在实施期抓住。

**但有 4 处 high 问题必须在定稿前修：** 其中 1 处是**论证方向性倒置**（把「有 dtype 差异的场景」认反，导致 3.2/C.4 关于 b8/b64 严格度的核心论证方向整体反了，且被两个独立维度分别命中）；1 处是**新设计的关键实现空白**（`FrameSampStore` 的 32 个 fd 与 2 个 mmap 未规定构造进程/时机，与 spawn 语义正面冲突，直接威胁 B.2「跨 worker 共享 page cache 零副本」的收益承诺）；1 处是**防线宣称夸大**（写侧 100% memcmp 实际只钉死 t、钉不死 g，F 节头号风险的第一层防线名不副实）；1 处是**计划内部自相矛盾**（B.1「不动 openpi」与 B.4「给 TorchDataLoader 加形参」不可能同时成立）。

**统计**（本人裁决后的最终口径）：

| severity | 数量 | 说明 |
|---|---|---|
| high | 4 | 双复核员一致确认，本人复核复现 |
| medium | 6 | 双复核员一致确认 |
| low | 10 | 其中 6 条双确认、4 条由分裂意见裁决后降级确认 |
| 驳回 | 3 | 2 条分裂后由本人裁定驳回、1 条双方均判不成立 |

合计提出 23 条质疑，**成立 20 条**（high 4 / medium 6 / low 10），**驳回 3 条**。没有一条动摇「重构可行、收益量级成立、等价性可验证」这一总体结论。

---

## 二、确认的问题

> 排序：severity 降序。除特别标注外均为**已确认**（两名复核员均判 CONFIRMED）。

### A1【high · 已确认 · 双维度独立命中】3.2 与 C.4：「有 dtype 差异的场景」认反，b8/b64 严格度论证方向整体倒置

- **位置**：3.2 节判据梯子第 3 层行 + 表下说明第二点（第 269、274 行）；C.4 节（第 424 行）。
- **原断言**：「本机 b8 是比集群 b64 严 38 倍的检验：dtype 差异只出现在「整批满长」的 batch，其占比 b8 下 59.6%、b64 下仅 1.6%——本机通过则集群更稳」；C.4「取「整批满长」batch（**唯一有 dtype 差异的场景**）……满长 batch 逐位相同即基本结案」。
- **问题**：真实关系恰好相反。`right_padding_token_emb` 只有短样本分支（`shape[0] < max_size`）才用未指定 dtype 的 `np.zeros` 做 `np.concatenate` 提升为 f64；满长分支是纯切片，dtype 原样保持 bf16。因此**「整批满长」batch 在 replica 与 native 下都是 bf16，本就没有 dtype 差异**，比对它必然逐位相同、零证伪力；真正存在 f64(replica) vs bf16(native) 差异、需要验证「精确升位不改数」的，是**「含短样本」的 batch**。按正确场景重算：含短样本占比 b64 = 98.4%、b8 = 40.4%，比值约 2.4 ——**b64 对真正的差异场景暴露度比 b8 高约 2.4 倍**，而不是「b8 严 38 倍」。计划由此得出的「本机 b8 通过即可推断集群 b64 更稳、无需在集群补做等价性校验」失去论证基础；C.4「满长 batch 逐位相同即基本结案」把无诊断力的对照当成了收官判据。
- **关键证据**（本人复核）：`src/mme_vla_suite/shared/data_utils.py::right_padding_token_emb` 第 23 行 `if sampled_img_emb.shape[0] < max_size:` 分支内四处 `np.concatenate([..., np.zeros(...)])`（img/pos/state 三处未指定 dtype → f64，仅 mask 显式 `np.bool_`），`else` 分支 `sampled_img_emb[:max_size]` 等纯切片零转换。计划**自身在 1.3 节⑤⑥、3.1 节点 4、C.2 节定点集（`step_idx∈{0,1,2,29,30}（触发 f64）`）、C.5 节 G2（`step=30→replica f64 / step=31→bf16`）四处的表述都是正确的**——只有 3.2 与 C.4 这两处（共三句）方向反了，属文档内部自相矛盾而非孤立笔误。`q=0.9373`：`q**8=59.57%`、`q**64=1.586%`、比值 37.57≈38（数字算对了，标签贴错）；`1-q**64=98.41%`、`1-q**8=40.43%`、比值 2.43。
- **影响缓解**（不改变必须修的结论）：C.4 的实际动作里两种 batch「各一」都测了，且 300 步 replica vs native 主判据在 b8 下每步有约 40% 概率命中含短样本 batch，累计几乎必然覆盖真实差异场景——**执行动作不会漏测，错的是判据叙述与由此推出的策略结论**。
- **建议修正**：① 3.2 说明第二点与判据梯子第 3 层行改为「dtype 差异只出现在**含短样本**的 batch，其占比 b64 98.4% / b8 40.4%」，并**删除或改写「b8 比 b64 严 38 倍」**——b8 在这个维度上并不更严，若要保留 b8 作为本机验证档，改用「b8 迭代快、300 步内命中差异场景约 120 次，足以证伪」这类正确理由；② C.4 首句改为「取「含短样本」batch（唯一有 dtype 差异的场景）作主判据，「整批满长」batch 作阴性对照（应逐位相同，否则说明 gather/pad 另有 bug）」；③ 相应地重新评估「是否需要在集群 b64 规模补一次等价性抽查」——原文用倒置的论证支撑了「不补」这一决定。

### A2【high · 已确认】B.2/B.3：`FrameSampStore`（32 个 fd + 2 个 mmap）的构造进程/时机全文未规定，与 spawn 语义冲突

- **位置**：B.1（第 351 行）、B.2（第 355–356 行）、B.3 `__init__` 注释（第 361–364 行）。
- **原断言**：「32 个 part fd `os.open` 常驻……pos/state 表 `np.memmap(mode='r')`——**跨 worker 由 page cache 共享，零副本**」；B.3 `__init__` 只列形制断言、清单查表与 `dtype_mode`；`__getitem__` 伪代码里 `store.read_image_rows(rows)` 的 `store` 从何而来未定义。
- **问题**：计划全文（B 节四小节 + C/F/G 全部章节）**没有任何一处**提到 `worker_init_fn` / lazy / per-worker 构造（唯一带「worker 初始化」字样的第 1.5 节第 6 条讲的是现状里 JAX 的 per-worker 初始化，属旧链路既有现象）。而本仓库训练链路 `num_workers>0` 时用 `multiprocessing.get_context("spawn")`，torch 把**整个 Dataset 对象作为 `Process` 的 args 通过 pickle 送进子进程**（早于 `worker_init_fn` 执行）。若照 `create_data_loader`（主进程构造 Dataset）的既有惯性把 store 建在 `__init__` 里：裸 `int` fd 在子进程中的同一数值不指向同一内核对象；`np.memmap` 被 pickle 后不会按路径重新映射，而是把整块数据序列化进 pickle 流，**直接推翻「跨 worker 共享 page cache、零副本」这一设计目标**。B.4 又明确「`TorchDataLoader` 一行不动」，堵死了改 `worker_init_fn` 这条常规修法，只剩「在 `__getitem__`/首次访问时懒加载」一条路，而计划同样没写。
- **关键证据**（本人复核）：`src/openpi/training/data_loader.py::TorchDataLoader.__init__` 确认 `if num_workers > 0: mp_context = multiprocessing.get_context("spawn")`，随后 `persistent_workers=num_workers>0`、`worker_init_fn=_worker_init_fn`（后者只设两个 XLA 环境变量，与 dataset 状态无关）；torch `dataloader.py:1145-1163` `multiprocessing_context.Process(target=_worker_loop, args=(..., self._dataset, ...))`。复核员实测：spawn 子进程对父进程 `os.open` 得到的 fd 数值做 `os.pread` 报 `OSError(29, 'Illegal seek')`；`pickle.dumps(np.memmap(...))` 反序列化后 `filename` 属性变 `None`、字节数与源数据同量级。现状 `RoboMMEDataset` 全程「现开现关」，本仓库没有 spawn + 持久资源组合的既有先例可隐式沿用。
- **严重度说明**：一名复核员建议降为 medium，理由是 image fd 被朴素 pickle 后首次读取几乎必然**响亮崩溃**（而非静默错读），会在首次多 worker 冒烟测试立刻暴露；两张 pos/state 小表即便被拷贝也只是每 worker 多背约 44 MB，数值仍正确。我**维持 high**：不是因为它会静默错数，而是因为①它是新设计中唯一一处「核心收益承诺（0 open / 常驻 fd / 零副本）缺少实现路径」的空白；②B.4 的「TorchDataLoader 一行不动」红线与常规修法直接冲突，属于必须在计划层面拍板、不能留到编码期即兴决定的架构点。
- **建议修正**：在 B.2/B.3 显式写死构造契约，二选一并说明理由：**(a) 懒加载**——`FrameSampStore` 在 `FrameSampDataset.__getitem__` 首次调用时按 `os.getpid()` 惰性构造并缓存于实例（`__getstate__` 剔除 store 句柄，保证 pickle 干净），全部 fail-loud 校验在每个 worker 内各跑一次；或 **(b)** 放宽 B.4 红线，允许在 `worker_init_fn` 里重建（须同时修正 B.1 的「不动 openpi」，见 A4）。同时给 C.5 补一条守卫（如 G10：在 spawn 子进程内断言 `store` 的 fd 属于本进程、`memmap.filename is not None`）。

### A3【high · 已确认】A.2 与 F 节：写侧 100% pos/state memcmp 只能钉死 t，钉不死 g

- **位置**：A.2 逐帧校验①（第 331 行）；F 节风险 1 第一层防线（第 471 行）。
- **原断言**：A.2「① 该帧 `pos_emb_4x4` ≟ `pos_table[t]`（memcmp，**同时钉死「文件确实是 (g,t)」**与「pos 只依赖 t」两条不变量）」；F 节「四层防线：**写侧 100% pos/state memcmp 钉死 (g,t)**」。
- **问题**：计划 1.2 节自己已证实 `pos_emb_4x4` 是 step_idx 的**纯函数**（跨 episode 逐字节相同）——一个只依赖 t 的量，其 memcmp 在数学上不可能证明含 g 维度的命题。把两个 episode 在同一 t 处的 image 数据互换（写侧 g 级下标算错），该 memcmp 会 100% 通过。校验②「`state_emb` ≟ state 表同一行」也补不上：state 表是打包器自己在同一趟按同一个 `row_of(g,t)` 写出的，若行号本身算错，image 与 state 会共享同一个错误行号「自洽地」通过，属写后自洽核验而非独立第三方比对（校验③写后 `pread` 读回同理）。于是**四层防线里真正能捕捉 g 级错位的只剩第三层 `verify` 的 5 万帧抽样（10.35%）**：对「整段 episode 换位」这类覆盖数百行的系统性错误命中率趋近 100%，但对「part/episode 边界处 fencepost 式单行错位」漏检率约 89.6%。C.5 的 G1–G9 也没有一条构造「pos 相同、来源 episode 不同」的跨 episode 调换场景。
- **关键证据**（本人复核）：`episode_0/token_emb_5.npy` 与 `episode_900/token_emb_5.npy` 实测 `pos_emb_4x4` 逐字节相同（`np.array_equal → True`），而 `state_emb`、`image_emb_4x4` 均不同。计划第 79 行自述「`pos_emb_4x4` 实测是 step_idx 的纯函数（跨 episode 逐字节相同）」与第 331 行「同时钉死「文件确实是 (g,t)」」直接矛盾。抽样算术：50,000/483,291 = 10.35%。
- **建议修正**：① 把 A.2 校验①的括注改为「钉死 t 与「pos 只依赖 t」两条不变量；**不钉 g**」，F 节第一层防线相应改为「写侧 100% pos memcmp 钉死 **t**」，不再声称 (g,t)；② 补一条真正能钉 g 的写侧校验——最直接的是**逐帧比对 `state_emb` 与「由独立来源（另起一趟、按 `episode_manifest` 路径重新读源 npy）构造的参照」**，或在 slab 写入前对该 episode 抽若干帧回读源 npy 路径并断言内容一致；③ 给 C.5 补一条守卫（G10/G11）：构造两个 episode 在同一 t 的帧互换的迷你库，断言校验链能亮红灯；④ 若不补写侧防线，则须在 F 节明写「g 级单行错位的唯一防线是 verify 10% 抽样，漏检率约 90%」，不要用「四层防线」的措辞给出超出实际的信心。

### A4【high · 已确认】B.1 与 B.4 自相矛盾：「不动 `src/openpi/**`」与「给 `TorchDataLoader` 加 `prefetch_factor` 形参」不可能同时成立

- **位置**：B.1（第 351 行）、B.4（第 390 行与第 393 行）。
- **原断言**：B.1「改动既有文件**仅一处**：`src/mme_vla_suite/training/dataloader.py`……**不动**：`scripts/train.py`、`src/openpi/**`、……」；B.4 第 390 行「其余（`transform_dataset` + `TorchDataLoader` + `DataLoaderImpl`）**一行不动**」；B.4 第 393 行「可选加性改动：`TorchDataLoader` 加 `prefetch_factor: int | None = None` 形参」。
- **问题**：`TorchDataLoader` 定义在 `src/openpi/training/data_loader.py`，`mme_vla_suite/training/dataloader.py` 只 import 它；其 `__init__` 是显式命名参数列表，**没有 `**kwargs` 透传通道**，也没有子类化接口——给它加形参只能编辑 `src/openpi/training/data_loader.py`。矛盾有两层：跨节的 B.1「不动 openpi / 仅改一处」，以及 B.4 小节内部相隔三行的「一行不动」与「加形参」。全文 grep「openpi」只命中第 351 行一处，无任何限定或例外说明；G 节红线清单也**没有**把「不动 `src/openpi/**`」列为正式红线，即实施期逐条自检不会拦下这个越界。
- **关键证据**（本人复核）：`src/openpi/training/data_loader.py` 第 384 行 `class TorchDataLoader:`，`__init__(self, dataset, local_batch_size, *, sharding=None, shuffle=False, sampler=None, num_batches=None, num_workers=0, seed=0, framework="jax")`，构造函数体直接传 `torch.utils.data.DataLoader(...)`，无 `prefetch_factor`、无 kwargs；`src/mme_vla_suite/training/dataloader.py` 第 10 行 `from openpi.training.data_loader import DataLoader, TorchDataLoader,transform_dataset`。
- **严重度说明**：一名复核员建议 medium（矛盾出现在「可选」条目上、修复成本极低）。我**维持 high**：它与 A2 是同一个红线的两面——A2 的常规修法（`worker_init_fn` 重建 store）同样撞这条红线。两处叠加说明「不动 openpi」这条边界在计划里没有被真正想清楚，属于必须在实施前拍板的范围问题，而非笔误。
- **建议修正**：三选一并写进 B.1 与 G 节：**(a)** 删除 B.4 的 `prefetch_factor` 可选项，「不动 openpi」保持为硬红线（同时要求 A2 走懒加载方案）；**(b)** 把 B.1 改为「改动既有文件两处：`mme_vla_suite/training/dataloader.py`（分派）+ `src/openpi/training/data_loader.py`（**纯加性、默认值保持现状行为逐字节不变**的 `prefetch_factor` 形参）」，并在 G 节把红线改为「`src/openpi/**` 仅允许默认值等价的加性改动，禁止任何改变现有默认行为的修改」；**(c)** 明确「B.4 该条为未来备选、本轮不实施」。无论选哪条，都要把「不动 `src/openpi/**`」这条边界补进 G 节红线清单——目前它只在 B.1 出现一次，逐条自检覆盖不到。

### A5【medium · 已确认】2.3 节「训练期每 step 对比表」重构后列：「2.4 MB/样本」与「159.5 MB/step」口径混用，同表内不能互相导出

- **位置**：2.3 节对比表（第 235 行「每样本读盘」、第 240 行「每 step 读盘 / 打开」）。
- **原断言**：「每样本读盘 … 2.4 MB（几乎全用到）」与「每 step 读盘 / 打开 … 159.5 MB / 64 次」。
- **问题**：2.4 MB × 64 = 153.6 MB ≠ 159.5 MB；反过来 159.5 ÷ 64 = 2.492 MB，同精度应写「2.5 MB」。用 `episode_manifest.json` 逐样本精算发现二者用了两套基准：**2.4 MB 是「真实平均帧数」口径**（`sum min(t+1,32)` / 395,289 = 30.996 帧 → 30.996×65,536 + 395,440 = 2.4268 MB），**159.5 MB 是「固定 32 帧上界」口径**（64×(32×65,536+395,440) = 159,525,888 B，精确对应）。对照 1.4 节现状侧同一行写的是「均值 18.7 MB（上界 19.7 MB）」——显式分列标注；重构后列却把均值与上界混进同一逻辑而未做同样区分，且现状列自身自洽（18.7×64 = 1,196.8 MB ≈ 1.20 GB），左右两列处理不对等。
- **关键证据**：复核员用 manifest 独立复算 `total_frames_read = 12,252,448`、`avg = 30.996`；均值口径 per-step = 155.3 MB ≠ 159.5 MB；B.3 伪代码 `frames = even_sampling_indices(step, 32)` → `read_image_rows(rows)` 按 `len(frames)` 变长读取，实现走的是均值路径，159.5 MB 是人为构造的保守上界。
- **建议修正**：统一口径并显式标注，例如「每样本读盘：均值 2.43 MB（上界 2.49 MB）」+「每 step 读盘：均值 155 MB（上界 159.5 MB）/ 64 次」，与 1.4 节写法对齐。

### A6【medium · 已确认】1.6 / 3.1(1) / C.1：「index 序列与 num_workers 无关」只在第 1 个 epoch 内成立

- **位置**：1.3 节链路图（第 86–87 行）、1.6 节（第 157 行）、3.1 恒等链第 1 条（第 251 行）、C.1 判据（第 407 行「w0/w4/w8 三档 diff 为空」）。
- **原断言**：「torch 的 index 序列只由 `(len(dataset), seed, batch_size, drop_last, shuffle)` 决定，**与 num_workers 无关**（已读 torch 源码确认：worker base_seed 的抽取时机恒定，不额外消耗 generator）」；1.6「num_workers 只影响交付时机不影响内容」。
- **问题**：该断言只在同一个 DataLoader 迭代器不被重建的窗口内（实质是第 1 个 epoch）成立。`_BaseDataLoaderIter.__init__` **每次构造迭代器**都从 `loader.generator` 抽一次 `_base_seed`，而 `_reset()` **不重抽**；`DataLoader.__iter__` 在 `persistent_workers and num_workers>0` 时只在首次构造迭代器、此后只 `_reset`，在 `num_workers=0` 时每个 epoch 都全新构造。`_base_seed` 与 `RandomSampler` 的 `torch.randperm` **共用同一个 generator 对象**，两条路径消耗节奏不同 → 同 seed 下 w0 与 w>0 **从第 2 个 epoch 起排列逐位不同**。本仓库 `TorchDataLoader.__init__` 正是用 `persistent_workers=num_workers>0`，其 `__iter__` 采用「StopIteration 即重建 `iter()`」的外层循环——精确命中该行为差异；生产训练 `shuffle=True`、`num_train_steps` 20k–100k 而 b64 下一个 epoch 仅 6,176 步，必然跨越多个 epoch 边界；仓库确实做过 w8c16/w12c16/w16c16 三档 num_workers 对比，「换 workers 重跑」是真实操作模式。
- **关键证据**（本人复核）：torch 2.7.1 `dataloader.py` 第 486–492 行（persistent 分支只 `_reset`）、第 698 行（`_base_seed` 在 `_BaseDataLoaderIter.__init__` 内抽取）、`_reset()` 函数体只重建 `_sampler_iter` 不重抽 seed、第 385 行 `RandomSampler(dataset, generator=generator)` 共用同一 generator。两名复核员各自独立写脚本复现，输出一致：3 epoch × 5 步下 epoch 0 逐位相同、epoch 1 起全部不同。
- **影响**：不推翻恒等链本身（新旧链路对比时 num_workers 相同即序列相同），但：① 1.6/3.1 的表述过于绝对；② 给 C.1 新增的 `dump_index_seq.py`「w0/w4/w8 三档 diff 为空」埋了假阳性——只要探针数据集偏小、dump 步数跨过 1 个 epoch，就必然得到非空 diff，而 C.1 给出的失败定位（「序列不同 → 查 sampler/drop_last」）不指向真实根因，容易被误判为新 Dataset/新 Store 出错。
- **建议修正**：① 三处表述加限定：「在同一迭代器生命周期内（即单个 epoch 内）与 num_workers 无关；跨 epoch 边界因 `persistent_workers` 是否复用迭代器而消耗 generator 节奏不同，w0 与 w>0 从第 2 个 epoch 起序列会分叉——这是 torch 既有语义，与本重构无关」；② C.1 明确约束「dump 步数必须 < 一个 epoch 的 batch 数」，并在失败定位里加一条「若差异恰从 epoch 边界开始 → 检查是否跨 epoch，非 Dataset 问题」。

### A7【medium · 已确认】C.1 第三处 monkeypatch 可行性论证不完整；且 1.3 图「主进程 collate」是事实错误

- **位置**：C.1（第 407 行）；连带 1.3 节链路图（第 116 行）。
- **原断言**：「端到端旁证：`bench_train_steps.py` 加第三处 monkeypatch（`BENCH_DUMP_IDX=1` 时 patch `_collate_fn`，记录 `_probe_idx` 后删键再交原 collate，交付内容不受影响）」。
- **问题**：两点。**(1) 执行位置**：`num_workers>0` 时 `_collate_fn` 由 **spawn 出的 worker 子进程**调用（torch 把 `collate_fn` 作为 `Process` 的 args 传给 `_worker_loop`，`_MapDatasetFetcher.fetch()` 内 `return self.collate_fn(data)`），不是主进程。现有两处 monkeypatch（`_install_metrics_recorder` / `_install_checksum_recorder`）都是主进程内替换 `_train.*` 调用点，**不构成先例**；要在子进程内记录 `_probe_idx` 并汇总回主进程做 diff，需要额外的跨进程持久化/IPC 机制，计划未说明。本仓库 framesamp 配置 `num_workers` 默认为 4（非 0），该路径必然触发，不是边缘情形。**(2) idx 来源**：`_collate_fn(items)` 签名只接收 `__getitem__` 的返回值列表，**不携带原始 dataset idx**（`possibly_batched_index` 留在 `fetch()` 作用域内），计划未交代 `_probe_idx` 这个键从何注入。
- **连带事实错误**：1.3 节链路图把 collate 画在「主进程 collate + 交付」框内，与 torch 真实语义（worker 内 collate 完成后，已合并的 batch 才经 IPC 回主进程）不符。这一点也影响 1.3/1.4 对「IPC 757 MB/batch」的归因叙述（体积结论仍对，发生位置的描述需修）。
- **关键证据**：`.venv/.../torch/utils/data/dataloader.py:1145-1163`、`_utils/worker.py:349`、`_utils/fetch.py::_MapDatasetFetcher.fetch`；`src/openpi/training/data_loader.py:432-449`、`_collate_fn(items)` 签名；`scripts/smoke-local/bench_train_steps.py::_install_metrics_recorder / _install_checksum_recorder` 均只 patch `_train.wandb` / `_train._checkpoints.save_state`。
- **影响缓解**：C.1 的**主判据**是 `dump_index_seq.py`（不依赖这个补丁），该 monkeypatch 只是「端到端旁证」，且「worker 内落盘 + 主进程读回汇总」是常规手段、可实现——问题是可行性论证缺失，不是不可行。
- **建议修正**：① C.1 补上机制说明：worker 内把 `_probe_idx` 追加到 `$BENCH_RECORD_DIR/idx_seq_w{worker_id}.jsonl`（带 `worker_id` 与全局 step），主进程按序合并后再与第 0 层 dump 对拍；并交代 `_probe_idx` 由探针 Dataset 的 `__getitem__` 注入（这意味着旁证只能配探针数据集用，不能直接跑真实 Dataset）；② 修正 1.3 图，把 collate 画进 worker 框、IPC 箭头改在 collate 之后。

### A8【medium · 已确认】Context/D 节：定稿时 w8c16 结果已落地，仍与另两档一并写成「在跑、结果落地后补入」

- **位置**：Context 第 10 行、D 节标题（第 430 行）、D 节对照组（第 434 行）。
- **原断言**：「GL e2e 验收尽快并行提交（**接受与在跑三档互相排队**）」；「对照组：v1-e2e-b64（6.933 s / 69.7%）、**三档 v1-e2efix（结果落地后补入对照**，作「只调参上限」）」。
- **问题**：`v1-e2efix-w8c16-58638708.log` 于 **23:09:01** 完成（`EXIT_CODE=0`，RESULT 完整：步时中位 5.301 s、GPU util 均值 71.2%、epoch 32,739 s ≈ 9.09 h），比计划文件 mtime **23:19:12 早 10 分钟**；另两档（w16c16/w12c16）确实到 23:38 之后才收尾，写「在跑」没问题。计划把三档笼统归为一类，既未把已有数字纳入 D 节对照表（全文 grep「5.301 / 71.2%」零命中），也没有反映 w8c16 已是既成对照。
- **关键证据**（本人复核）：`stat` 显示计划 mtime 2026-08-24 23:19:12、w8c16 日志 mtime 23:09:01；日志尾部 `EXIT_CODE=0`。
- **勘误一处**：finder 附带指出「`docs/training-doc/v1-e2efix-w8c16/` 只有 launch.md、未按 AGENTS 12/17 留档」——本人复核时该目录已含 `launch.md`、`records`、`result.md`，该留档缺口**已被补齐**，这半条不再成立。
- **建议修正**：D 节对照组直接写入 w8c16 已落地数字（5.301 s / 71.2% / 9.09 h），把「结果落地后补入」限定为 w16c16/w12c16 两档；顺带在 D 节判据表里把「只调参上限」这一行的锚点更新为实测的 5.301 s（当前必达 ≤5.00 s 的余量因此变得更紧，值得在定稿时复看一眼）。

### A9【medium · 已确认】C.0/S0 的 `KEEP_JAX_CACHE=1` 让 `$HOME/.cache/jax_*` 变成跨轮持久目录，与 AGENTS 14 冲突且计划未处理

- **位置**：C.0（第 400–402 行）、E 节 S0（第 455 行）；连带 B.1「不动 `scripts/train.py`」（第 351 行）。
- **原断言**：「S0 脚本改动……加 `KEEP_JAX_CACHE=1`」，让 A/B 两轮「共用同一份 jax 编译缓存」。
- **问题**：该修法把 jax 编译缓存从「每轮结束即删的临时产物」（现状 `run_2gpu_epoch_bench.sh` 结尾 `rm -rf -- "${HOME}/.cache/jax_${RUN_NAME}"`）变成**跨轮持久保留、不断增长的本机目录**，而物理路径仍是 `scripts/train.py` 里硬编码的 `~/.cache/jax_<exp_name>`。AGENTS 14 要求「缓存……一律收敛到单一根 `v1-store/`……逐项显式设置缓存类环境变量指向 `v1-store/cache/`」，且**禁止覆盖 `HOME`**。计划同时把 `scripts/train.py` 列为「不动」，形成一个未处理的张力：既没重定向，也没说明如何合规，更没安排验证结束后的清理步骤（E 节 S0–S8、G 节红线清单均未覆盖）。`scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` 已 `export JAX_COMPILATION_CACHE_DIR="$STORE/cache/jax"`——说明仓库其他脚本本就认为该缓存应落 `v1-store`，但被 `train.py` 的 `jax.config.update` 直接覆盖、从未生效。
- **关键证据**：`scripts/train.py::main()` `jax.config.update("jax_compilation_cache_dir", str(epath.Path(f"~/.cache/jax_{config.exp_name}").expanduser()))`（无条件、硬编码，不读环境变量）；`run_2gpu_epoch_bench.sh` 第 109–117 行清理逻辑；全文 grep 无 `v1-store/cache` 与该目录的任何关联。
- **建议修正**（不必改 `train.py`，也不必覆盖 `HOME`）：在 S0 里把缓存目录做成 `v1-store/cache/jax/<exp_name>` 的**符号链接目标**——脚本启动前 `mkdir -p v1-store/cache/jax/$EXP_NAME && ln -sfn` 到 `~/.cache/jax_$EXP_NAME`；或直接放宽「不动 `train.py`」，改成读 `JAX_COMPILATION_CACHE_DIR`（若已设则不覆盖）——这同时修好 `gl_e2e_fix.sbatch` 里那条从未生效的 export。无论哪种，都要在 E 节补一步「验证收官后清理 `KEEP_JAX_CACHE` 保留的目录」，并把这条写进 G 节红线。

### A10【medium · 已确认】E 节 S5（30–60 min）未分配 run_name、未安排 `docs/training-doc/` 留档，AGENTS 17 覆盖缺口

- **位置**：E 节 S 表 S5 行（第 460 行）与 run_name 建议清单（第 466 行）。
- **原断言**：run_name 清单只列 `v1-framesamp-det-d{0,1,2}-r{1,2}`（S1）、`v1-framesamp-ab-{old,replica,native}`（S6/S7）、`v1-framesamp-dl-w*` / `e2e-w*c16` / `…-cold/-hot`（S8）。
- **问题**：S5（定点 8,200 样本 + 200 真实 batch 对拍，计划自标 **30–60 min**）远超 AGENTS 17 的 5 分钟阈值，属「诊断 run」，应「同等适用第 12 条：从 clean HEAD 启动、在 `docs/training-doc/<run_name>/` 留档」。E 节把同量级甚至更短的 S1（~1 h）、S6（~30 min）、S7（~35 min）都认真分配了具名 run_name，唯独漏了 S5。D 节末句「每个 >5 min run 留档」上下文限定在 GL 吞吐验收（records 字段全是 GPU bench 专属产物），读不出覆盖 S5；G 节红线只有笼统一句「正式 run clean HEAD 起跑+留档+run_name 用户确认」。S3（全量打包）走的是 `docs/dataset-build-doc/` 分支（A.2 已明确），不适用于 S5。
- **建议修正**：run_name 清单补 `v1-framesamp-cmp-{replica,native,f32}`（或类似），并在 S5 行的判定列加「+ `docs/training-doc/<run_name>/` 留档」。

### A11【low · 已确认，双方均建议降级】1.2 节：`data/*.pkl` 的「395,440 B」不是常量

- **位置**：1.2 节产物结构（第 70 行）。
- **原断言**：「`data/{0..395288}.pkl` 每执行样本一个，**395,440 B**」——与上一行 npy 的「每帧一个，602,951 B」采用完全相同的句式，无「约/均值」限定词。
- **问题**：pkl 内嵌 `prompt` 与四个 `*subgoal*` 变长字符串字段，实测大小在 **395,440–395,636 B** 浮动（跨 4 任务 32+ 样本抽样，仅少数恰为 395,440），395,440 是分布下界而非常量。作为对照，npy 的 602,951 B 抽样 30+ 个文件全部精确相同，是真常量（且被 A.2 用作 `st_size==602,951` 硬守卫）——同一句式对应一真一假，构成文档内部不一致。
- **影响**：极小。A.3 的体积估算用均值重算为 156.36 GB vs 156.31 GB，偏差 0.03%，「156 GB」结论不变；计划任何守卫/判据都未对 pkl 字节数做精确断言。
- **建议修正**：改为「每执行样本一个，**约 395.4–395.6 KB**（内嵌变长 prompt/subgoal 字符串，非定长）」。

### A12【low · 已确认】1.2 节：`episode_manifest.json` 顶层字段名是 `sha256`，不是 `manifest_sha256`

- **位置**：1.2 节 ASCII 图（第 51 行）。
- **原断言**：「整体带 **`manifest_sha256`**（被改动即 fail-loud）」——处在逐项罗列清单自身字段（`h5_file`/`raw_ep_idx`/`num_timesteps`/三个偏移）的并列句式中。
- **问题**：实测 `episode_manifest.json` 顶层键为 `['version','raw_dir','canonical_order','num_shards','totals','shard_load_timesteps','episodes','sha256']`，**不含 `manifest_sha256`**。`manifest_sha256` 是 `scan_manifest.py::manifest_sha256()` 的函数名，以及其他产物（`build_shard.py` 写出的 `meta/_shard{i}of8.json` sidecar、sample json）引用该值时自取的字段名。同文档 A.1 第 322 行写 store_meta 的「`manifest_sha256`（须等于当前 `episode_manifest.json` 的 **sha256**）」是准确的，A.2 也用准确的 `sha256`——只有 1.2 节这一处把引用名当成了清单自身字段名。
- **建议修正**：第 51 行改为「整体带完整性字段 `sha256`（被改动即 fail-loud；下游 store_meta 引用时命名为 `manifest_sha256`）」。

### A13【low · 已确认】1.4 / 1.5 节：「5.4×」应为 5.3×

- **位置**：1.4 节表格「每样本读盘」行（第 138 行）、1.5 节浪费排序第 2 条（第 148 行），同一错误值出现两次。
- **原断言**：「单看 npy 是 589 KiB 只用 112 KiB（**5.4×**）」；「整包 pickle 反序列化：7 键绑死，读 **5.4×** 于所需字节」。
- **问题**：602,951 / 114,720 = 5.2558 → 5.3×；用文中已取整的 589/112 = 5.2589 也是 5.3×。同表另一处「18.7/4.1 ≈ 4.6×」四舍五入规范，排除「统一向上取整」的特殊约定。全仓库检索 `5.4×` 只命中该文件这两处，无独立实测来源，是同一次心算误差的传播。
- **建议修正**：两处均改为 5.3×。

### A14【low · 已确认】B.4：「f64 降精度……jaxlib `Squash64BitTypes`」机制误标

- **位置**：B.4 dtype 三模式帐（第 386 行）。
- **原断言**：「device_put 73→23/38 ms（f64 降精度已核实发生在 host 侧，jaxlib **`Squash64BitTypes`**）」。
- **问题**：效果观测（host 侧降精度、耗时下降）正确，但机制名张冠李戴。本仓库 `.venv` 的 jax 全部 Python 源码检索不到任何 `Squash64BitTypes`/`squash_64` 引用；`xla_extension.so` 里确有 mangled 符号 `_ZN3xla16Squash64BitTypesENS_13PrimitiveTypeE`，但无任何绑定或文档证据表明它是这条 host 侧路径的实现载体（从命名与 XLA 代码组织看更像编译/HLO 阶段的 64 位类型降级工具）。**实际执行者是纯 Python/numpy 的** `jax/_src/interpreters/xla.py::_canonicalize_ndarray_dtype`（`np.asarray(x, dtypes.canonicalize_dtype(x.dtype))`，经 `canonicalize_dtype_handlers[np.ndarray]` 注册，在 `pxla.shard_args` 里被调用，先于数据交给 PJRT）。复核员用 monkeypatch 实测：`jax.device_put(np.float64 数组)` 时该函数被调用一次、入参为 host 侧 f64、输出设备张量为 f32。
- **建议修正**：改为「……已核实发生在 host 侧（`jax/_src/interpreters/xla.py::_canonicalize_ndarray_dtype`，在 `pxla.shard_args` 内对 np.ndarray 做 `np.asarray(x, canonicalize_dtype(...))`）」。

### A15【low · 已确认】3.3 节：「同 commit 重跑两轮」实为两个不同 commit

- **位置**：3.3 节（第 278 行）。
- **原断言**：「2026-08-24 实测：**同 commit** 同配置同 seed 重跑两轮，参数校验和逐步全不相同」。
- **问题**：该结论唯一对应的留档 `docs/training-doc/v1-2gpu-epoch-bench-b8/result.md` 白纸黑字写的是「第 2 轮（commit `891d6e3`）与第 3 轮（commit `22baa1c`）……每个记录步的参数校验和均不同」——两个不同 commit，且留档自己用的措辞是「**同配置**重跑」而非「同 commit」。`git diff --stat 891d6e3 22baa1c` 只改了 `scripts/smoke-local/bench_train_steps.py`（把 `wandb.log` 的直接 monkeypatch 换成 `_WandbProxy` 代理类，修 `wandb.init` 覆盖 log patch 的 bug）与一份 launch.md，训练循环/模型/dataloader 逐字未变，所以「非 bitwise 确定」这个定性结论**大概率仍成立**；但「同 commit」的字面表述与留档不符，会让读者误以为已经做过真正意义上的「同一份代码原地重跑两次」对照，而留档里从未有过这样一次对照。
- **影响**：有限——S1 的 D0/D1/D2 是全新设计的「两轮相同 run」实验，真正的把关在 S1，3.3 只是动机描述。
- **建议修正**：改为「同配置同 seed 重跑两轮（两轮间仅有 bench 记录层的 `_WandbProxy` 改动，不进计算图，见 `docs/training-doc/v1-2gpu-epoch-bench-b8/result.md`）」。

### A16【low · 已确认】E 节：「红线 7」编号错位，应指向第 6 条

- **位置**：E 节 commit 切分（第 465 行）。
- **原断言**：「旧链路原地保留（**红线 7**），不安排删除 legacy 的 commit」。
- **问题**：G 节红线清单是按中文分号排列的无编号列表，按出现顺序数：①训练循环/模型/超参/seed 零改动 ②index 序列构造性不变 ③`even_sampling_indices` 复用不重写 ④4task-gl 只读+旁路新增 ⑤身份只从 `episode_manifest.json` ⑥**旧分支代码不惊动** ⑦**禁复活 d951aef** ⑧uv 纪律……。「旧链路原地保留」对应第⑥条；第⑦条是内容完全不同的「禁复活 d951aef」。复核时按「红线 7」去核对会查到不相关条目。
- **建议修正**：改为「红线 6」；更稳妥的做法是**给 G 节清单加显式编号**（顺带把 A4 建议新增的「`src/openpi/**` 边界」一并编号纳入），避免后续引用继续漂移。

---

## 三、存疑条目（复核意见分裂，本人裁决）

### B1【裁定：不成立 → 驳回】d1「`bench_train_steps.py` 六道护栏」计数不准

- **计划原文**：B.4「不新增 CLI 参数（`bench_train_steps.py` **六道护栏**与全部 sbatch 零改动）」。
- **CONFIRMED 方（查自答视角）**：`main()` 里字面上有 8 处 raise（步数上限、wandb、overwrite/resume、use_history、history_config、`batch_size%fsdp_devices`、`log_interval`、`inspect.getsource` 源码校验），脚本自身 docstring 只说「几道」不给确数，计划却给了一个与实测不符的确数。
- **REFUTED 方（攻证据链视角）**：`scripts/smoke-local/README.md` C4 行早有仓库自己的护栏枚举——「≤500 步、强制关 wandb、禁 overwrite/resume、锁 history_config、强制 `log_interval=1`」，与 `bench_train_steps.py` 同一 commit（`fb4f03a`）写入；按这 5 个命名类目映射到代码（history 类含 `use_history` + `history_config` 两处 raise）恰为 6，与「六道」吻合；另 2 处（`batch_size%fsdp_devices` 算术合法性、`getsource` 的 monkeypatch 前提校验，后者代码里有独立注释标明是另一类保护）从创建之日起就不在仓库的「护栏」叙事内。
- **本人核实**：`grep -n "raise" scripts/smoke-local/bench_train_steps.py` 命中第 58（在 `_resolve_record_dir` 一类的辅助逻辑里，不在 `main()`）、147、152、154、156、158、160、164、169 共 9 处，`main()` 内 8 处——CONFIRMED 方的计数无误。`scripts/smoke-local/README.md:51` 的 C4 行原文确如 REFUTED 方所述，5 个类目映射到代码恰为 6 处 raise。
- **裁定**：**驳回。** 「护栏」在本仓库没有唯一 canonical 计数口径（代码 docstring 自己只说「几道」），而 README C4 提供了一个先在的、与「六」自洽的枚举。更关键的是，B.4 这句话的**实质断言是「零改动」**，本人已核实计划确实未提出修改该文件——「六道护栏」只是支撑该结论的描述性括注，不是独立技术断言。把一个本无客观基准的描述性数字判为「事实性错误」属过度苛责。
- **可选打磨**（非必修）：把「六道护栏」改为「多道 fail-loud 护栏」以消除歧义。

### B2【裁定：部分成立 → 降为 low】逐位一致结论向生产默认 autotune 环境的迁移性未标注（d6-high 与 d8-medium 同一主题，合并裁决）

- **计划原文**：第三节标题「**结论：可以做到逐位一致**」（无条件陈述）；C.0/S1「取首个 PASS 档固定为**全部 A/B 环境**」（即 C.1–C.4 全部跑在 D2 = `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` 受控环境）；D 节吞吐验收全文无任何 XLA_FLAGS/autotune/deterministic 设置，S0 改动被明标为「验证资产，非训练代码」。
- **CONFIRMED 方**：C 节证明的有效性域被限定在关闭 autotune 的 D2 环境；D 节 GL 验收与正式训练跑在默认 autotune 下，native（dtype 恒定 → 单份 HLO）与 replica/旧链路（dtype 随 batch 摆动 → 两份 HLO）是两个独立编译产物，各自的 autotune 可能选中不同 kernel，引入 C 节未覆盖的浮点路径差异；且验证硬件是本机 2× RTX 6000 Ada、部署是 GL 4×A40，双重差异；D 节判据表只有吞吐指标、无任何数值判据；计划未显式承认或缓解这段迁移性 gap。
- **REFUTED 方**：GEMM 的 autotune 缓存 key 落在算子自身的 dtype/shape/kernel-config 上，两条路径经 `promote_dtype` 后喂给 `dot_general` 的是同 dtype/shape 的物化 bf16 缓冲区（计划 3.1(4) 已 memcmp 证明逐位相同），签名相同则搜索空间相同；OpenXLA determinism 文档把非确定性框定为「同一张 HLO 编译两次」的 profiling 抖动，正是计划 3.3 已发现并命名的问题；且计划 3.4 已备有比 bf16 的 1 ULP 保守两个数量级的量化兜底；此外该检验本身不良定义——按 3.3 实测，旧链路在默认环境下自身重跑就不 bitwise，没有固定基线可比。另有复核员从「A/B 是 C 节专有术语、E 节 S8 依赖表显式把 D 节与确定性链路解耦、D 节从设计起就只承担吞吐验收」论证这是刻意的职责分离而非遗漏。
- **本人裁定**：**核心风险不成立，表述缺陷成立，降为 low。** 理由三点：① REFUTED 方对机制的分析更站得住——`promote_dtype` 在任何算术前把输入统一转 bf16，到达 `dot_general` 的张量已被计划实测证明逐位相同，两条路径的 GEMM 算子签名一致，没有机制支撑「上游有无 cast 会改变该 GEMM 选中的 kernel」；② 即便退一步承认存在核选择差异，其量级与**计划自己已经识别、命名并主动构造 C.0 去隔离的既有 XLA 非确定性完全同类**——它同样作用于「旧链路 vs 旧链路」的重跑，**不是本重构引入或放大的新风险**，也无法通过任何计划能新增的测试消除（没有可比基线）；③ 但 CONFIRMED 方指出的**有效性域未标注**是实打实的：第三节以「结论：可以做到逐位一致」作无条件标题，而证明成立于 C.0 固定的受控 XLA 环境，读者（含未来的实施者）容易把它误读为「生产环境下 native 与旧链路逐位相同」。
- **建议修正**（low，改一句话即可）：在第三节标题或 3.1 结尾加一句有效性域限定——「本节的『逐位一致』成立于 C.0 固定的受控 XLA 环境（`deterministic_ops=true`/`autotune_level=0` 或共用编译缓存）；生产默认 autotune 下的残差属 3.3 所述的既有 XLA 非确定性（旧链路自身重跑同样不 bitwise），其量级由 3.4 的量化判据兜底，不因本重构而变化」。**不建议**为此在 GL 上补一轮 bitwise 校验——按 3.3 的实测，那样测到的只会是已知的 autotune 抖动噪声，无法归因。

### B3【裁定：部分成立 → 降为 low】B.2 `os.preadv`「返回字节数不符即 raise」未处理合法短读

- **计划原文**：B.2「按连续行游程合并 `os.preadv` 直读进预分配 bf16 数组（短样本 32 行天然连续 → 1 次调用），**返回字节数不符即 raise**」。
- **CONFIRMED 方**：POSIX 允许 read/pread/preadv 返回小于请求量且不算错误（信号打断等），Unix 惯例是「循环读满」而非把任何未满都当损坏；本仓库 turbo NFS4.2 的 `rsize=1047672`（约 1 MB），而 32 行 = 32×65,536 = 2,097,152 B（2 MB）已是单次 RPC 上限的 2 倍，跨 RPC 边界的大读更容易暴露合法部分完成；把它当数据损坏 raise 会中断训练。
- **REFUTED 方**：对**常规文件**（本地盘或 NFS），内核读路径在内核态内部循环，透明发起所需次数的底层 I/O（NFS 就是若干次 ≤rsize 的 READ RPC）直到凑满或 EOF，`rsize` 只是网络 RPC 载荷上限而非用户态单次调用的返回上限（`man 5 nfs` 原文即此义）；`man 2 pread` 举的短读诱因只有「接近文件末尾/从 pipe 读/从终端读/被信号打断」，不含「请求量超过内部传输单元」；该挂载是 `hard`（无 `soft`），超时会重试整个 RPC 而非返回部分。并做了 220 次同挂载、同请求量（含 `POSIX_FADV_DONTNEED` 逼近冷读）的实测，零短读。
- **本人核实**：独立实测——在同一 turbo NFS4.2 挂载的 69.8 GB 常规文件上，随机 offset 发起 **100 次单调用 2,097,152 B**（精确等于 rsize 的 2.00 倍）的 `os.pread`/`os.preadv`，每次读前对该区段 `posix_fadvise(POSIX_FADV_DONTNEED)`，**短读 0 次**。`mount` 确认 `vers=4.2,rsize=1047672,wsize=1047532,hard`。
- **裁定**：**「合法短读非罕见、high 严重度」不成立**——跨 rsize 边界不构成短读诱因，这一点两轮独立实测（220 + 100 = 320 次调用）零短读，可以定论。但「未读满即 raise」把一个**理论上仍可能出现**（致命信号打断已部分拷贝的场景）的情形当作数据损坏，确实不是稳健写法，且修复成本仅几行、失败后果是崩掉一个跑了数小时的训练。**保留为 low 的稳健性建议。**
- **建议修正**（low）：B.2 改为「返回字节数不足则从已读偏移继续 `preadv` 补齐，连续多次（如 3 次）仍无进展才 raise；**读到 0 字节（EOF）或总量超出 part 边界立即 raise**」——后半句才是真正的完整性判据，与前半句的部分完成情形分开处理。

### B4【裁定：成立 → 降为 low】A.1 把 32 个 part 统一描述为「≈990 MB」，末个 part 实际 620.7 MB

- **计划原文**：2.1 节「压成 32 个约 990 MB 的连续大文件」；A.1「`part_000.bf16.bin … part_031.bf16.bin` # 各 **≈990 MB**，(rows,16,2048) bf16 裸字节」。
- **CONFIRMED 方**：按计划自己给出的切法（按 `global_episode_idx` 升序累积 `num_timesteps`，累计 ≥ `ceil(483291/32)=15103` 即切、切点在 episode 边界）套真实 manifest 逐条模拟，确得 32 个 part，但 `part_031` 只有 9,471 行 ≈620.7 MB，是其余 31 个（990–1020 MB）的约 62%，偏差 38%——是「贪心累加、最后剩多少算多少」的固有尾部效应，不是四舍五入误差。
- **REFUTED 方**：真实、精确的每 part 边界写进 `store_meta.json.parts[i]`，`FrameSampStore.__init__` 按 `st_size==meta.bytes` 逐 part 精确校验，读侧完全不依赖这个描述性数字；计划里所有量化环节（A.2 打包耗时 20–40 min、B.4 本地盘 `df ≥40 GB`/cp 65 s/sha256 32 s）都基于总量 31.7 GB，无一处以「32×990 MB」为输入；「≈」本身已是近似标记。
- **本人核实**：独立复算确认——threshold=15,103，切出 32 个 part，前 31 个落在 **990.7–1020.1 MB**，`part_031` 覆盖 episodes[1573..1599]、9,471 行、**620.7 MB**，为前 31 个均值的 **0.620**。
- **裁定**：**成立，但降为 low。** REFUTED 方关于「功能上无影响」的论证完全正确，我核实无误；但 38% 的偏差已经超出「≈」的合理覆盖范围，而这是一份要指导实施与人工核对的文档——实施者拿到 32 个文件后目视核对「是否齐整」时，一个 620 MB 的文件会引发不必要的排查。这属于应当打磨的描述精度问题，不是设计缺陷。
- **建议修正**：2.1/A.1 两处改为「31 个 ≈990–1020 MB + 末 1 个 ≈621 MB（尾部效应：最后一刀只能拿剩余量）」。

### B5【裁定：不成立 → 驳回】B.4「cp 31.7 GB 到 `/tmp/$SLURM_JOB_ID/`（~65 s）」隐含吞吐口径存疑

- **计划原文**：B.4 节点本地盘拷贝开关（默认关）：「df 守卫（≥40 GB）→ cp 31.7 GB 到 `/tmp/$SLURM_JOB_ID/`（**~65 s**）→ 逐 part sha256 校验（~32 s）」。
- **CONFIRMED 方**：31.7 GB/65 s ≈ 488 MB/s 逼近甚至可能超过本项目实测的「NFS 供给 398–628 MB/s」；而该区间的方法论是**多 worker 并发**的聚合吞吐（`docs/v1-nfs-bottleneck-analysis.md`：「dataloader-only 实测，1×A40×9 个数据点，对 worker 数不敏感」），与单进程 `cp` 顺序拷贝口径不同；计划未交代 65 s 的实测来源，也没说会不会用并行拷贝。
- **REFUTED 方**：488 MB/s 明确落在 398–628 区间**内部**（比中点 513 还低，距上界 628 有 22% 余量），谈不上「逼近上界」更谈不上「可能超过」——质疑给出的数字与其自身结论方向相反。更关键的是访问模式方向搞反了：既有区间是在**更差**的模式下测出的（大量小文件随机 open，计划自述「32 次 open 本身 74.3 ms」），而 `cp` 读的是 32 个约 990 MB 大文件、只需 32 次 open、走内核 readahead 的顺序读——在 NFS 上通常**比小文件随机访问更快**，因此 488 MB/s 是偏保守而非乐观的估计。且该开关默认关、有 `df` 前置守卫、有逐 part sha256 兜底、数字带「~」，与文档内其余估算（「预计 20–40 min」）风格一致。
- **本人裁定**：**驳回。** REFUTED 方的三点我都核实认同：数字在区间内、访问模式方向对该估算有利而非不利、该开关有多重前置与事后校验兜底。CONFIRMED 方唯一站得住的是「未标注估算依据」，但这在一份充满量级预估的计划里不构成缺陷。
- **可选打磨**（非必修）：把「~65 s」改为「~65 s（按 GL 侧实测 NFS 供给中位 ≈513 MB/s 折算；顺序大文件读通常优于该聚合值，此为保守估计）」。

### B6【裁定：部分成立 → 降为 low】多处标「已实测」的精确数字在仓库中找不到产出脚本或原始输出

- **计划原文**：1.4 节「每样本耗时（热/冷）25.4 / 132.4 ms」「gather 热 17.7 ms / 冷 ~110 ms」「32 次 open 本身 74.3 ms」「collate / IPC / device_put 52 ms / 757 MB / 73 ms」；A.2「已实测 npy 内部偏移恒定：`image_emb_4x4`@262,595、`pos_emb_4x4`@541,352、`state_emb`@602,906，120/120 文件大小一致、60/60 memcmp 通过」；3.1「已用真实形状实测：三种输入的 memory token 输出全等（max 差 0.0）」；A.2「已实测 CPU 后端生成与库中值不逐位一致（差 1e-7~1e-5），GPU 后端一致」。
- **CONFIRMED 方**：这些精确到个位毫秒/字节的数字，在 `docs/`、`v1-store/reports/`、`v1-store/probe-attrib/` 及全仓库 `.py`/`.md` 里都找不到出处——没有产出脚本，也没有记录运行方式与原始输出的报告；计划引为权威汇总的 `docs/v1-nfs-bottleneck-analysis.md` 全文 33 行只有「69-70%」这类粗粒度范围。而 3.1 的「dtype 不改数」是「逐位一致」论证链的核心一环，读者无法独立复查。
- **REFUTED 方**：逐条独立复现，几乎全部精确吻合——9/9 个随机 episode 的 npy 三个偏移全部**逐字节等于**计划所写数字；120 个文件 `st_size` 全部 602,951；`pickle.load` 热态 2.49–2.81 ms（计划「~2.7 ms」）；32 次串行 open 热态 79.85 ms（计划 74.3 ms，同量级、偏差 <10%）；`JAX_PLATFORMS=cpu` 下用仓库真实 `PosEmb3D(dim=768)` 重新生成 pos 与库中值比对，`max|diff| = 7.15e-07`，**恰落在计划所述 1e-7~1e-5 区间**且逐位不同，与 `docs/v1-gl-dataset-consistency-report.md` 记录的「GPU 上生成逐位相同」互补印证；`nnx.Linear(dtype=bf16)` 下 bf16/f32 输入输出逐位相同、`max|diff| = 0.0`。结论是「未留痕」而非「捏造」。
- **本人核实**：抽查 3 个跨任务 episode（`episode_0/t=0`、`episode_801/t=5`、`episode_1300/t=10`）的 npy 内部偏移，**三者全部为 `{image_emb_4x4: 262595, pos_emb_4x4: 541352, state_emb: 602906}`、文件大小全部 602,951 B**，与计划所写逐字节一致。
- **裁定**：**「数字不可信」不成立，「可追溯性缺失」成立，降为 low。** 独立复算已把最关键、最容易出错的数字（A.2 的 slice reader 偏移量、A.2 的 CPU/GPU pos 后端差异、3.1 的 promote_dtype 逐位一致）精确复现，说明它们来自真实测量。另需指出：AGENTS 17 的留档义务只约束 >5 分钟的调试/基准/诊断 run，这些是秒级探针，**不构成规则违反**。但作为一份要指导多轮实施与验收的计划，把无出处的精确数字写成「已实测」确实降低了可核查性。
- **建议修正**（low）：把这批探针脚本（哪怕是几十行）落进 `scripts/data-pack-framesamp/probe_*.py` 或在 `docs/` 加一个附录记录命令与原始输出；至少在计划里给这些数字加一句来源标注（「2026-08-24 一次性交互式探针，未留档；A.2 偏移量已由本次对抗验证独立复现 9/9」）。

---

## 四、被驳回的质疑

| # | 质疑（一句话） | 驳回理由（一句话） |
|---|---|---|
| C1 | C.2 定点集里「每 episode 首样本 1,600 个」对 VideoUnmask（前缀恒 66）与 VideoUnmaskSwap（前缀 114–216）不会命中 `t<31` 的 f64 边界，与「边界全覆盖」的设计意图不符 | 两名复核员均判不成立：计划用「+」把四组样本并列列出，「边界全覆盖」这个标签在语法上只挂在 `step∈{0,1,2,29,30,31,32,33}` 这一组上，从未声称首样本组承担边界覆盖；且 f64 触发逻辑（`even_sampling_indices` / `right_padding_token_emb`）只依赖 step_idx 数值、不读任务身份，边界组「由清单精确构造」已从 Button* 任务取到全部边界值，覆盖完整。 |
| B1 | `bench_train_steps.py`「六道护栏」与 `main()` 里实际 8 处 raise 不符 | 本人裁定驳回：`scripts/smoke-local/README.md` C4 行早有仓库自己的 5 类护栏枚举，映射到代码恰为 6 处 raise，另 2 处（算术合法性、monkeypatch 前提）本就不在护栏叙事内；且该句实质断言「零改动」正确，「六道」只是描述性括注（详见 §三 B1）。 |
| B5 | B.4「cp 31.7 GB（~65 s）」隐含 488 MB/s，逼近甚至可能超过实测 NFS 供给上限，且口径（单流 vs 多 worker 聚合）不同 | 本人裁定驳回：488 MB/s 落在 398–628 区间内部（低于中点），且既有区间是在小文件随机访问这一**更差**模式下测得的，顺序大文件读通常更快，该估算偏保守；开关默认关、有 df 守卫与逐 part sha256 兜底（详见 §三 B5）。 |

---

## 五、覆盖面

### 5.1 已核过且确认无误的断言（按维度综述）

- **d1 消费侧代码事实**：1.3 节取数链①–⑦全部核实与代码一致——`RoboMMEDataset.__getitem__` 的单 pkl + 逐帧 `np.load(allow_pickle).item()`、每次新建 `ThreadPoolExecutor(≤32)` 用完即弃；`even_sampling_indices` 的 `t<32` 分支与 `max_size = 512//(16×1) = 32`；`right_padding_token_emb` 四处 `np.zeros` 中三处未指定 dtype、触发阈值精确为 `t<31`；短样本占比用真实 manifest 实算 **6.2739%**（与计划 6.27% 精确吻合，且验证仅 Button 两任务贡献短样本）；`_collate_fn` 的 `np.stack` 混合 dtype 提升；`use_state_emb=false`（GPU 侧确不消费）；transforms 顺序（Repack→RoboMMEInputs→DeltaActions→Normalize→InjectDefaultPrompt→ResizeImages→PaligemmaTokenizer→PadStatesAndActions）；`resize_with_pad` 带 `jax.jit` 装饰器（worker 内触发 JAX 初始化的机制成立）；`HistoryPi0Config.max_token_len=64`、prefix 顺序 `[512 mem|256 img|256 wrist|64 txt]`；`Pi0Config.dtype` 默认 `"bfloat16"` 且 `_CONFIGS` 未覆盖；`TorchDataLoader` 无 `prefetch_factor` 形参、torch 内部默认 2；`drop_last=True`、`RandomSampler` 在主进程采样；`assets_dirs = assets_base_dir/self.name`（「不新增 `_CONFIGS`」的理由前提成立）；`dataset_path` 已是 `TrainConfig` 字段（tyro 自动生成 `--dataset-path`）；`_normalize_state` 因 norm stats q01/q99 为 f64 而恒输出 f64；独立复算 v1-e2e-b64 的 util 均值 **69.66%**、0% 采样 **27.8%**、慢步墙钟 **32.8%**、步时中位 **6.926 s**，与计划引用数字一致。
- **d2 生产侧代码事实**：`scan_manifest.py` 规范序 `sorted(*.h5) × sorted(episode)`、sha256 fail-loud（`raise SystemExit`）；`episode_manifest.json` 字段齐全、totals 与 `meta/stats.json`/`provenance.json` 三方一致（1600 ep / 483,291 帧 / 395,289 执行样本）；每帧 npy 7 键的 shape/dtype/字节数与 1.2 节列表逐一吻合；`pos_emb_4x4` 跨 episode 逐位相同（本人复核确认）；`build_shard.py` 从 `scan_manifest.py` import 而非复制（「格式常量绝不复制」有先例）；`kept_indices.json` 只被 `prepare_token_drop` 读、framesample 路径不涉及；features 按全部 timestep 存 / data 按执行样本存；`row(g,t)=total_sample_offset[g]+t` 前提成立（offset 序列 1600 条连续无空洞，抽查 6 个 episode 磁盘文件数与 manifest 一致）；exec_start_idx 规律（Button* 恒 0、VideoUnmask 恒 66、VideoUnmaskSwap ∈{114,168,216}）；原始 H5 `chunks=None, compression=None`；A.2 的三个 npy 内部偏移（本人复核 3 个跨任务样本全部精确吻合）；`num_timesteps` 范围 163–586（与 pos 表 586 行设计吻合）；678 GB 来自既有留档 `du -sh` 实测（与理论 448 GB 的差距由 NFS 小文件块分配开销解释，反而佐证计划自身论点）。
- **d3 数据实测**：602,951 B 跨 4 任务 16 个文件全部命中；framesample 三键 112 KiB = 19.0%；三个偏移在 16 个跨任务文件上按偏移直读与 `np.load` 结果 memcmp 全通过；`pos_emb_4x4` 纯函数性（9 个 t × 最多 8 个 episode 共 71 组比较全等）；manifest sha256 自洽（用仓库自带 `load_manifest()` 重算通过）；新库 31.7 GB 体积推算与「文件数×单文件大小」公式精确吻合（31.673 + 0.0288 + 0.0155 = 31.717 GB），单 part ≈990 MB、「特征侧 1/9」均吻合。
- **d4 算术自洽**：存储账、每帧字节账、读放大上界（19.69 MB）与「真正用到 4.1 MB / 4.6×」、概率链（98.4%/59.6%/1.6%）、内存账（537 MB → 134 MB）、时间账（6,176 步 / 11.9 h / 8.5 h / 8.2 h，且 6,176×5.00 s = 8.578 h 与「必达 ≤8.6 h」严格对应）、291 GB 总读与 30 GB verify 读、2,112 次/step 与 1.20 GB/step、供给余量 ≈15×、D 节判据表四行单调有序——全部逐位重算无误。
- **d5 torch 语义**：`spawn` context、`persistent_workers`、`worker_init_fn` 只设 XLA 环境变量、`prefetch_factor` 内部默认 2、`drop_last`/`generator` 语义、`even_sampling_indices`/`right_padding_token_emb` 为纯函数无随机源、**perceptual 路径零 Python RNG 调用**（`random.random`/`random.gauss` 只在 symbolic 分支）、worker 启动时 torch 无条件 `random.seed(base_seed+worker_id)` 但不影响 perceptual 路径。
- **d6 jax/flax/XLA 语义**：`.venv` 实际版本 jax/jaxlib 0.5.3、flax 0.10.2、ml_dtypes 0.4.1；`nnx.Linear.__call__` 在 `dot_general` 前调 `promote_dtype`、`dtype` 显式给定时直接 `jnp.asarray(x, dtype)`；对**全部 65,536 个 bf16 位型**做 bf16→f32→f64→f32→bf16 往返，位模式 100% 复原（含负零/subnormal）；f64→bf16 单步与 f64→f32→bf16 两级结果 100% 一致；`jax_persistent_cache_enable_xla_caches` 默认值确为 `xla_gpu_per_fusion_autotune_cache_dir`；六个 XLA flag 在 `xla_cuda_plugin.so` 中逐个 strings 命中且 jax 0.5.3 无对应 Python config（只能走 `XLA_FLAGS`）；`np.save` 对 bf16 写出 `V2` descr 而 `np.memmap`/`frombuffer().view()` 可用；x64 处于默认关闭（仓库无覆盖）。
- **d7 存储与打包**：消费侧确为 spawn；按 A.1 给出的切法套真实 manifest 恰得 32 个 part（本人复核）；`pos_emb_4x4.f32.bin` 的 586 行与 manifest 最大 `num_timesteps` 吻合；`posix_fadvise(WILLNEED)` 在本仓库 NFS4.2 挂载上可用；`np.memmap` 可 pickle（不崩溃，只是语义退化）；打包 20–40 min 的预算与 291 GB 读 + 31.7 GB 写 + 回读校验 + 16 进程反序列化的粗算自洽。
- **d8 等价性梯子**：C.2 的 f64/bf16 边界划分（`{0,1,2,29,30}` vs `{31,32,33}`）与代码分支精确对应；定点集数字自洽（5×200 + 3×200 + 1,600 + 5,000 = 8,200）；`run_2gpu_epoch_bench.sh` 的 RUN_NAME 写死 + 记录目录存在即拒跑 + 结尾删 jax 缓存三个现状问题全部属实；`bench_train_steps.py` 现有两处 monkeypatch 属实；pos/state 小表字节数复算吻合；verify 抽样比例 50,000/483,291 ≈ 10.35%。
- **d9 文档一致性**：v1-e2e-b64 核心数字（6.933 s / compute-only 4.778 s / 11.9 h / 6,176 步 / NFS 398–628 vs 251 MB/s）与既有留档一致；69.7%/27.8%/32.9% 虽未见于文档原文（文档只写「69-70%」区间），但用仓库自带 `analyze_gpu_util.py` 对该 run 的 records 重新分析可**精确复现**；`greatlakes.md` 确未记载 spgpu `/tmp` 规格（计划的前提成立）；D 节「2.389 MB/样本」新公式核算正确（395,440 + 30.4×65,536），与旧公式同一构造方式。
- **d10 规则合规**：A.2「本机 tmux 打包不违反 AGENTS 13」的论证成立（从 NFS 唯一副本执行、纯 CPU、产物是可 memcmp 复核的确定性字节）；`.gitignore` 已整体忽略 `/v1-store/`（AGENTS 14「不进 git」合规）；打包只读 4task-gl、不触原始 H5（AGENTS 15）；npy 内部字节偏移是数据格式常量非代码行号（不违反 AGENTS 9），S0/C.3 引用的 `scripts/smoke-local/README.md`「第二节」「第 3 级」是真实存在的语义锚点；C 节的 b8/300 步/save-interval 25 与 `run_2gpu_epoch_bench.sh` 现有默认值一致（复用既有覆盖参数，不触发 AGENTS 10）；E 节 commit 切分体例与依赖链（S1←S0、S6←S1+S5、S7←S6、S4←S3、S8←S4）自洽无环。

### 5.2 本次验证未覆盖的地方

1. **一切尚不存在的代码**：`scripts/data-pack-framesamp/*`、`src/mme_vla_suite/datastore/*`、`framesamp_dataset.py`、`compare_batches.py`、`dump_index_seq.py`、`test_pack_guards.py` 均未创建，本次只能审设计描述，**无法验证实现正确性**。A2/A3 两条 high 正是这类「设计描述留白」的产物，实施期仍需按 C 节梯子逐层验收。
2. **未实际跑任何 GPU 训练**：C.0 的 D0/D1/D2 确定性前提、C.3/C.4 的 300 步 bitwise、D 节的吞吐判据表，全部是**未来才能验证**的预期值。计划声称的「≈4.9 s 步时 / ≈8.5 h epoch / util ≥90%」本次一律未验证。
3. **未验证的性能预估**：B.2 的「32×64 KiB 常开 fd pread 0.33 ms」「np.memmap 切片 2.4–2.5 ms」、2.3 表的「单样本 ≈7 ms 热 / 15–40 ms 冷」、A.2 的「打包 20–40 min」、B.4 的「cp ~65 s / sha256 ~32 s」——只做了量级自洽性核查，未做端到端复测。
4. **未审 GL 集群侧环境**：spgpu 节点 `/tmp` 容量与规格（计划自己也标注「greatlakes.md 未记载、需先跑一个 job 确认」）、A40 上的 XLA flag 实际行为、GL 侧 NFS 供给在新访问模式（32 个大文件顺序读）下的实测值。
5. **未审 Phase C 预案**（A.3 pkl 打包）与 `docs/v1-framesamp-dataflow.md`（尚未撰写）。
6. **未做完整的正确性攻击面扫描**：如打包期多进程崩溃/断电后的 `--resume` 正确性、`store_meta.json` 与 `pack_progress.jsonl` 的写序竞态、`os.replace` 在 NFS 上的原子性保证——这些只在设计描述层面读过一遍，未做对抗性推演。

---

## 六、定稿前必须修的条目清单

### 必修（阻断级，建议逐条改完再进 S0）

| # | 条目 | 修改动作 |
|---|---|---|
| **A1** | 3.2/C.4 dtype 差异场景方向倒置 | 三句话改方向；删除或重写「b8 比 b64 严 38 倍」；C.4 主判据换成「含短样本」batch，满长 batch 降为阴性对照；重新评估「是否需在 b64 规模补抽查」这一被倒置论证支撑的决定 |
| **A2** | `FrameSampStore` 构造进程/时机未定义 | 在 B.2/B.3 写死构造契约（推荐懒加载 + `__getstate__` 剔句柄），并给 C.5 补一条 spawn 子进程内的守卫（G10） |
| **A3** | 写侧 memcmp 只钉 t 不钉 g | A.2/F 节改口径为「钉 t 不钉 g」；补一条真正独立的 g 级校验，或明写「g 级单行错位仅靠 verify 10% 抽样、漏检约 90%」；C.5 补跨 episode 换帧的守卫 |
| **A4** | B.1「不动 openpi」与 B.4 加 `prefetch_factor` 矛盾 | 三选一拍板（删该可选项 / 改 B.1 为「两处、纯加性」/ 明标本轮不实施），并把「`src/openpi/**` 边界」补进 G 节红线 |

### 应修（medium，影响判据可执行性或文档准确性）

| # | 条目 | 修改动作 |
|---|---|---|
| **A5** | 2.3 表 2.4 MB 与 159.5 MB 口径混用 | 按 1.4 节写法显式标注「均值（上界）」 |
| **A6** | index 序列跨 epoch 与 num_workers 相关 | 1.6/3.1(1) 加有效性域限定；C.1 约束 dump 步数 < 一个 epoch，并把该根因写进失败定位 |
| **A7** | C.1 第三处 monkeypatch 可行性论证缺失 | 补 worker 内落盘 + 主进程汇总的机制说明，交代 `_probe_idx` 由探针 Dataset 注入；修正 1.3 图「主进程 collate」 |
| **A8** | w8c16 结果已落地却写成「在跑」 | D 节对照组写入 5.301 s / 71.2% / 9.09 h；复看必达 ≤5.00 s 的余量 |
| **A9** | `KEEP_JAX_CACHE` 与 AGENTS 14 冲突 | 用符号链接或 `JAX_COMPILATION_CACHE_DIR` 把缓存收敛进 `v1-store/cache/jax/`；E 节补收官清理步骤；写进 G 节红线 |
| **A10** | S5 无 run_name / 无留档 | run_name 清单补 `v1-framesamp-cmp-*`；S5 行判定列加留档要求 |

### 建议修（low，文档精度与稳健性）

| # | 条目 | 修改动作 |
|---|---|---|
| **A11** | pkl「395,440 B」非常量 | 改为「约 395.4–395.6 KB（内嵌变长字符串）」 |
| **A12** | 清单字段名 `manifest_sha256` | 改为 `sha256`，并注明下游引用时的命名 |
| **A13** | 「5.4×」 | 两处均改为 5.3× |
| **A14** | `Squash64BitTypes` 误标 | 改为 `xla.py::_canonicalize_ndarray_dtype`（经 `pxla.shard_args`） |
| **A15** | 3.3「同 commit」 | 改为「同配置（两轮间仅 bench 记录层改动，不进计算图）」 |
| **A16** | 「红线 7」编号错位 | 改为红线 6；建议给 G 节清单加显式编号 |
| **B2** | 「逐位一致」有效性域未标注 | 第三节标题/3.1 结尾加一句受控环境限定 + 指向 3.3/3.4 |
| **B3** | preadv 未读满即 raise | 改为「续读补齐、多次无进展才 raise；EOF/越界立即 raise」 |
| **B4** | 32 个 part 统一写 ≈990 MB | 改为「31 个 ≈990–1020 MB + 末 1 个 ≈621 MB」 |
| **B6** | 「已实测」数字无出处 | 落一份探针脚本或附录，至少加来源标注 |

### 无需改动

- **B1**（六道护栏）、**B5**（cp ~65 s）、**C1**（C.2 首样本组边界覆盖）：经裁定不成立，保持原文即可（B1/B5 可做一句话措辞打磨，非必需）。

---

*报告完。审查对象为计划文档本身，不涉及任何代码改动；本次验证全程只读，未修改仓库任何既有文件。*
