# IO 重构（v4）之后的立项路线（high-level）

> 本文件只记录**立项顺序结论与各项的确定 scope**，不含实施细节。前提链条：
> [`v1-dtype-unify-plan.md`](v1-dtype-unify-plan.md)（前置）→ [`v1-framesamp-restructure-plan.md`](v1-framesamp-restructure-plan.md)（IO 重构 v4）→ 本文件所列各项。
> **本文件所列各项一律在 v4 落地并通过 GL 验收之后才允许立项**，且每项立项前须用户单独拍板；此处不构成实施授权（AGENTS 2）。
> 结论来源：2026-08-26 会话讨论（问题一"dtype 之外的机制修复"与问题二"memory token 取数机制可读性重构"）。

## 为什么全部后置

v4 的等价性证明依赖"新旧两侧只有『字节从哪读』一个变量"。下列任何一项混入当前计划都会破坏该单变量前提：或改 transforms 计算实现、或改模型输入签名（HLO 必变、编译缓存失效）、或动 legacy 基线（违反 v4 红线"`training/dataset.py`、`shared/**` 不动"）。因此一律后置、逐项独立验证。

## 决策门：不是固定队列

三个加速项的优先级**由 v4 的 GL 验收数据决定**，v4 收官后先看 util 数字再逐个拍板：

- util 均值已达 95%+ → 项 2、3（省 IPC/传输）收益明显缩水，缓做或不做；
- util 仍卡在 80–90% → 项 1（worker 去 JAX 化）是第一嫌疑，优先立项。

## 候选项清单（按当前预估优先级排列）

### 项 1：worker 去 JAX 化

- **动机**：transforms 中 `ResizeImages` 在每个 spawn worker 内独立初始化 JAX（实测约 8 s）并在 GPU0 建约 442 MiB CUDA context——16 workers 约 7 GB 显存加上下文抢占。v4 只解决读盘供给面，这一项解决 worker 计算面与 GPU 抢占，是 v4 落地后 util 仍不达标的头号嫌疑（v4 计划 4.1 节已留立项依据与 bench 采样存证）。
- **确定 scope**：把 dataloader worker 内的 resize 从 jax.jit 换为 CPU 实现，消除 worker 侧 JAX 初始化与 CUDA context。
- **确定前置**：需先证明 CPU resize 与现 jax resize 数值一致（逐位或量化判据，届时商定），是一条独立的完整验证梯子；适用 AGENTS 18 两张链路图 + 两块一致性讨论。

### 项 2：pos_emb 移出 batch、GPU 侧生成

- **动机**：`pos_emb_4x4` 已实证是 step_idx 的纯函数（586 行小表）。现每 batch 的 `static_pos_emb` 约 100 MB，占 memory 三键载荷四成多，却完全可由 GPU 侧从常驻小表按 step 编号 gather 得到——IPC / collate / device_put 全线再砍约 40%。
- **确定 scope**：dataloader 只交付每样本的选帧 step 编号，`embed_memory` 输入签名相应改变，pos 表常驻 device。
- **确定约束**：输入签名改变 → HLO 必变、编译缓存必失效，等价性须重新自证，与 v4 的恒等链完全独立。

### 项 3：`static_state_emb` 白算白传清除（并入项 2 一轮）

- **动机**：`use_state_emb=false`，GPU 不消费该键，worker 却照算 `_normalize_state` 并随 batch 传输（且该键恒为 f64）。体量小（约 2 MB/batch），单独立项不值一轮验证成本。
- **确定 scope**：从 framesamp 交付键中移除该键及其 worker 侧计算。**不单独立项，并入项 2 同一轮**。

### 项 4（可选，排最后）：`MemoryBuffer` 可读性拆分

- **背景**：训练侧 framesamp 取数链的可读性问题（三层回调倒挂、dict-of-dicts 中间格式）**随 v4 的 `FrameSampDataset` 交付自动消失**，无需立项。剩余的"复杂"只在旧类里：`MemoryBuffer` 一类身兼在线评估有状态 buffer 与离线训练无状态工具两职，另含 token_drop / recurrent 分支。
- **确定 scope**：把 `MemoryBuffer` 拆为"在线评估 buffer"与"纯函数集"两块。这些代码在线评估仍在用、不可删；v1 范围外，属锦上添花，优先级排在项 1–3 之后，仅在确有需要时立项。

## 已裁定不做 / 无独立价值（避免重复讨论）

- 每样本新建线程池、每样本 32 次 NFS open：v4 本体消灭，无"提前单独修"的空间。
- `TorchDataLoader` 加 `prefetch_factor` 形参：v4 红线裁定本轮不实施，列为未来备选（见 v4 计划 4.1 节）。
- framesamp 训练取数链的单独可读性重构（无论 v4 之前还是与 v4 合并）：已裁定不做，理由见"为什么全部后置"。
