# IO 重构（v4）之后的立项路线（high-level）

> 本文件只记录**立项顺序结论与各项的确定 scope**，不含实施细节。前提链条：
> [`v1-dtype-unify-plan.md`](v1-dtype-unify-plan.md)（前置）→ [`v1-framesamp-restructure-plan.md`](v1-framesamp-restructure-plan.md)（IO 重构 v4）→ 本文件所列各项。
> 2026-08-26 增补：全链梯度对拍锚定黄金基线 G0，权威载体 [`v1-gradient-baseline.md`](v1-gradient-baseline.md)，本文件各项适用文末「梯度对拍规约」。
> **立项门（2026-08-26 审计修正：拆双状态，消除「性能失败无补救路径」死锁）**：v4 的 GL 验收拆为两个独立结论——`CORRECTNESS_PASS`（等价性梯子全过）与 `PERFORMANCE_PASS|FAIL`（GL 性能判据表机器判定）。本文件所列各项一律以 **`CORRECTNESS_PASS` 为硬前提**；`PERFORMANCE_PASS` 时按下方决策门排优先级，**`PERFORMANCE_FAIL`（如 util 落在 80–90%）时允许用户单独批准针对性的「v4 性能补救项」（首选项 1）**——但此时不得称 v4 已通过 GL 验收，补救项收官后须重跑 GL 验收。每项立项前均须用户单独拍板；此处不构成实施授权（AGENTS 2）。（原规格「必须先通过 GL 验收才允许立项」与项 1 的 80–90% 启动条件互斥：性能必达 util ≥90%，恰是失败场景里补救分支不可达。）
> 结论来源：2026-08-26 会话讨论（问题一"dtype 之外的机制修复"与问题二"memory token 取数机制可读性重构"）。

## 为什么全部后置

v4 的等价性证明依赖"新旧两侧只有『字节从哪读』一个变量"。下列任何一项混入当前计划都会破坏该单变量前提：或改 transforms 计算实现、或改模型输入签名（HLO 必变、编译缓存失效）、或动 legacy 基线（违反 v4 红线"`training/dataset.py`、`shared/**` 不动"）。因此一律后置、逐项独立验证。

## 决策门：不是固定队列

三个加速项的优先级**由 v4 的 GL 验收数据决定**（辅以本机 speed 链对比 `v1-g2-speed` vs `v1-g0-speed`），v4 收官后先看 util 数字再逐个拍板：

- util 均值已达 95%+ → 项 2、3（省 IPC/传输）收益明显缩水，缓做或不做；
- util 仍卡在 80–90% → v4 性能验收为 FAIL，走立项门的补救分支：项 1（worker JAX 导入链处置）是第一嫌疑，可单独批准为「v4 性能补救项」。

## 候选项清单（按当前预估优先级排列）

### 项 1：worker JAX 导入链的根因处置（2026-08-26 审计修正：原「去 JAX 化＝替换 ResizeImages」定性作废）

- **现象（真实）**：每个 spawn worker 独立初始化 JAX（实测约 8 s）并在 GPU0 建约 442 MiB CUDA context——16 workers 约 7 GB 显存加上下文抢占（v4 计划 4.1 节 bench 采样存证）。
- **归因修正（审计确认）**：原文「transforms 中 `ResizeImages` 是 jax.jit、换成 CPU 实现」不成立——`transforms.ResizeImages` 调用的是 `openpi_client.image_tools`（NumPy/PIL 的 CPU 实现），**本来就没有 JAX**；同名 JAX 版在 `openpi/shared/image_tools.py`，只被 mem_buffer / 模型侧引用。worker 里的 JAX 实际来自**模块级导入链**：`training/dataset.py` 顶部 import `MemoryBuffer` → `mem_buffer.py` 顶部 import `openpi.shared.image_tools`（该模块 import 即 `import jax`）。照原 scope 替换 resize 不会消除 worker 侧 JAX；误改 JAX 版 resize 反而击中 MemoryBuffer/模型侧，范围越界。
- **确定 scope（改写）**：第一步是**根因定位**——在 worker 内实测首次触发 JAX/CUDA 初始化的真实调用点（导入期还是调用期、是否仅上述导入链一条）；第二步按定位结果处置（候选：延迟导入 / 把 worker 所需纯函数从 mem_buffer 导入链剥离 / 限制 worker 可见设备），scope 届时另行拍板。
- **确定前置**：任何处置不得改变交付数值——适用 AGENTS 18 两张链路图 + 两块一致性讨论；若处置涉及 resize 实现变更（目前无此必要），才需要数值一致性证明。

### 项 2：pos_emb 移出 batch、GPU 侧生成

- **动机**：`pos_emb_4x4` 已实证是 step_idx 的纯函数（586 行小表）。现每 batch 的 `static_pos_emb` 约 100 MB，占 memory 三键载荷四成多，却完全可由 GPU 侧从常驻小表按 step 编号 gather 得到——IPC / collate / device_put 全线再砍约 40%。
- **确定 scope（2026-08-26 审计修正补全——原文只写两处，实际是一次训练/在线双侧的签名迁移）**：
  - **迁移矩阵（train/serve 全部同步点，缺一即 shape 冲突或签名不一致）**：dataloader 交付键（step 编号替代 pos 张量）；`HistAugObservation` 数据类（字段、shape 注解、from_dict/to_dict/from_base_obs 及模块内预处理包装）；`HistoryPi0Config.inputs_spec` 的 perceptual 分支（abstract spec 决定 init 与 HLO）；`training/config.py` 两处 `RepackTransform` 结构表；在线侧 `RoboMMEInputs` 输入拼装与 `Policy` 的 history 准备路径；`embed_memory` 及其下游两层（`percep_mem` / `mem_encoder` 的形参）。
  - **索引 padding 语义（必须显式规定）**：现语义是短样本 pos 补**零向量**；改为交付索引后若用 `0` 补位，gather 会取回真实 `pos[0]` 而非零——**不等价**。规定：索引用定长 int32、缺位以 `-1` 哨兵填充、gather 后按 mask 显式清零，并配越界（OOB）测试。
  - **设备端表的双口径（必须显式规定）**：训练侧 packed pos 表 586 行，在线 `MemoryBuffer` 默认按 4096 步生成——常驻 device 的表须定义来源、长度、内容 hash、checkpoint/重启语义；若统一用 586 行表，在线路径 step 超界即越界，须一并裁决。
- **确定前置（判据可得性，审计确认的缺口）**：G0 量具只记录 collate 后、上卡前的 host batch——设备端 gather 出的 pos 不在任何记录点，且本项落地后 pos 张量不再出现在 host batch，「与 G0 的 pos 摘要对拍」目前**无从采集**。立项后第一步是落地**设备端观测点**：在 gather、mask 清零、展开之后、进投影层之前把有效 pos 取回 host 做逐位摘要（schema 版本化），与 G0（P1b 补录轮）的 pos 摘要对拍；该观测点工作量计入本项，不得假设量具现成。
- **确定约束**：输入签名改变 → HLO 必变、编译缓存必失效，等价性须重新自证，与 v4 的恒等链完全独立。

### 项 3：`static_state_emb` 白算白传清除（2026-08-26 审计修正：改为独立一轮，不再并入项 2）

- **动机**：`use_state_emb=false`，GPU 不消费该键，worker 却照算 `_normalize_state` 并随 batch 传输（且该键恒为 f64）。
- **确定 scope（审计修正补全——按原文字面「从交付键移除」第一步就 KeyError）**：交付键移除必须同步改两处**硬阻断**——`RepackTransform`（对结构表中的键做无默认值的硬索引，缺键即 KeyError）与 `HistoryPi0Config.inputs_spec`（无条件声明该键的 abstract spec）；下游 `HistAugObservation.from_dict` 与 `RoboMMEInputs` 对缺键容错、无需改。配套验证：transform → 模型 init → 短训 + 在线路径各一次冒烟。
- **单变量裁定（2026-08-26 用户拍板：拆开两轮）**：先项 2 单独一轮验证收官，再叠加项 3 一轮——两项是可分离变量（pos 交付格式 vs state 键移除），合并后对拍出差异无法归因，且项 2 恰是判据链最重的一项。项 3 体量小，其单独一轮按缩减档执行（单步 fixture + 短训对拍即可，届时商定）。

### 项 4（可选，排最后）：`MemoryBuffer` 可读性拆分

- **背景**：训练侧 framesamp 取数链的可读性问题（三层回调倒挂、dict-of-dicts 中间格式）**随 v4 的 `FrameSampDataset` 交付自动消失**，无需立项。剩余的"复杂"只在旧类里：`MemoryBuffer` 一类身兼在线评估有状态 buffer 与离线训练无状态工具两职，另含 token_drop / recurrent 分支。
- **确定 scope（2026-08-26 审计修正：调用面如实登记，原「在线评估仍在用」严重低估）**：`MemoryBuffer` 共 **7 个调用面**——①类本体及子类 `MemoryBufferRecurrent`（继承耦合）；②在线评估（policy 侧有状态 buffer，原文唯一提到的一个）；③legacy 离线训练（`training/dataset.py`，同时是 v4 的 A 侧基线与 worker JAX 导入源）；④GL 建库 `build_shard.py`（还隐式依赖「构造即初始化 JAX」的副作用）；⑤建库收尾校验 `finalize_checks.py`；⑥新旧数据集对拍 `compare_datasets.py`（**v4 自身的验收资产**）；⑦旧建库脚本 `build_robomme_dataset.py`。拆分前须先建「调用方 × 方法 × 数据表示」矩阵；实施采用保留门面类或逐调用方原子迁移，**不得破坏 v4 明确保留的 legacy 回滚链与验收资产**，测试覆盖 frame_sampling / token_drop / recurrent / 建库 / finalize / legacy 全部分支。在线评估代码不可删；v1 范围外，属锦上添花，优先级排在项 1–3 之后，仅在确有需要时立项。

## 梯度对拍规约（2026-08-26 增补，适用于上列每一项）

每项立项后的等价性验证除各自的「vs 自己改动前」（即基线链上一节点）外，**必须同时锚定 G0**（三个计划都未实施的原始训练，定义、产物与失效条件以 [`v1-gradient-baseline.md`](v1-gradient-baseline.md) 为权威）：

- **vs 上一基线**：主判据届时商定，能 bitwise 则 bitwise。
- **vs G0**：项 2/3（改输入签名，HLO 必变）的输出侧 bitwise 天然不可得（项 1 若不改计算实现则输出侧仍可争取 bitwise），**主判据 = 输入侧逐位对拍**（与 XLA/缓存/驱动无关、跨 HLO 有效），辅以量化复核（等价性检验形态，基线计划「量化判据」节）与单步 fixture 回归闸（约 2 分钟重锚 G0）。**判据可得性限定（2026-08-26 审计修正）**：G0 固化的 `batch_digests` 是 host 侧 raw 口径——跨 dtype 场景须用基线计划 P1b 的 canonical 口径；项 2 的「设备端 gather 的 pos vs G0 pos 摘要」须先落地项 2 前置的设备端观测点（见项 2），现量具采不到，不得当作现成判据引用。
- **引用 G0 产物前必须 `BASELINE_ENV=PASS` preflight**（AGENTS 18 末句）；结论回填基线计划登记簿。
- **revert 链形态**：若 dtype 修复被 revert，G1 不存在、v4 退回 v3 形态，基线链变为 G0 → G2'（v3 形态）；G0 保持链头不变。
- **正确性 run 与性能 run 必须分跑——speed 链通则（2026-08-26 用户裁定，适用上列每一项）**：带 TrainState 摘要 / batch_digests / 确定性 XLA 档的对拍 run，其 util/步时一律不作性能结论；**每项收官后必须跑对应的 speed run**（`v1-g<n>-speed`，同 `v1-g0-speed` 口径：无摘要、生产 XLA 档，权威定义见基线计划「符号总表」），与上一 speed 节点及 `v1-g0-speed` 对比，作为该项的性能结论与下一项立项输入。

## 已裁定不做 / 无独立价值（避免重复讨论）

- 每样本新建线程池、每样本 32 次 NFS open：v4 本体消灭，无"提前单独修"的空间。
- `TorchDataLoader` 加 `prefetch_factor` 形参：v4 红线裁定本轮不实施，列为未来备选（见 v4 计划 4.1 节）。
- framesamp 训练取数链的单独可读性重构（无论 v4 之前还是与 v4 合并）：已裁定不做，理由见"为什么全部后置"。

## 审计修正记录（2026-08-26，两份对抗审计逐条核对后落实）

1. 立项门拆 `CORRECTNESS_PASS` / `PERFORMANCE_PASS|FAIL` 双状态，性能失败开出合法补救分支（文件头）——原规格下 util 80–90% 恰好落入「须先通过验收才能立补救项」的死锁；
2. 项 1 定性由「替换 JAX 版 ResizeImages」改为「worker JAX 导入链根因处置」——`transforms.ResizeImages` 实为 NumPy/PIL 实现，JAX 来自 dataset→mem_buffer 的模块级导入链；同一错误在 v4 计划链路图与烘焙一节的两处副本已同步改正；
3. 项 2 scope 补全为 train/serve 迁移矩阵，显式规定索引 padding 哨兵语义与设备端表双口径，并把「设备端观测点」列为前置工作量（原文把采不到的判据当现成引用）；
4. 项 3 改为独立一轮（用户拍板），scope 补入 Repack 与 inputs_spec 两处硬阻断——按原文字面删键第一步即 KeyError；
5. 项 4 登记 7 个调用面与迁移纪律——原「在线评估仍在用」低估牵连面，且其中两处是 v4 自身的验收/建库资产；
6. 梯度对拍规约补判据可得性限定（raw/canonical 口径、设备端观测点）。
