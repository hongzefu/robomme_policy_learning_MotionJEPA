# 第二阶段报告：dtype 统一修复（framesample+context 双 dtype 路径消除）

> **范围**：v1 dataloader 重构链条的第二阶段——把旧训练链路里「dtype 随 batch 组成摆动」的双路径原地修掉，
> 并用两块验证证明修复前后训练完全等价。本阶段是 IO 重构（第三阶段）的前置。
> 本报告只保留人类审阅需要的内容与实测结论；实现级细节（代码改动逐行、工具参数、落盘格式定义、
> commit 拓扑、红线自检、审计修正记录）留在源计划文件 [`v1-dtype-unify-plan.md`](../v1-dtype-unify-plan.md) 第二部分。
> 前置的确定性定档与黄金基线见 [`v1-phase1-gradient-baseline-report.md`](v1-phase1-gradient-baseline-report.md)。
>
> **状态：全部执行完毕**（2026-08-26 立项 → 2026-08-27 收官）。两块正确性验收全过、性能对比已产出。

---

## 一、结论先行

1. **修复等价，且是 bitwise 等价**：修复后的 1000 步训练与黄金基线**逐字节相同**——
   `scalars_hex.tsv` 的 sha256 与基线两轮完全一致。这比原本预期的「跨计算图只能靠量化兜底」强得多。
2. **交付内容逐位不变**：2,600 个定点样本 + 200 个真实 batch 全部对拍通过、零失配；单步定点梯度
   三档场景 32 个叶子逐位相同。
3. **性能是净收益**：步时均值 **−7.21%**（85.5 ms/step）、GPU util 均值 **+5.64pp**（86.5% → 92.10%）、
   0% 采样占比 **−3.69pp**、慢步从 3 降到 0。绝对收益与按 collate + device_put 分解预估的约 80 ms/step 吻合。
4. **改动面极小**：一个函数、三行（三个 `np.zeros` 各加一个 dtype 参数）。这不是引入新行为，
   而是**把现状 1.6% batch 的行为推广到 100%**。
5. **验收范围有边界**：两块验证只覆盖 `perceptual-framesamp-context` 训练路径。同一函数也被在线评估路径与
   modulation/expert 变体调用，那些路径**不在验收范围内**——风险同质（同为补 dtype 参数、填充值恒 0 的精确升位消除），
   但不得据此宣称已验证。

---

## 二、Context（为什么做这件事）

**现状**：`right_padding_token_emb` 的三个 `np.zeros` 未指定 dtype，默认 float64。
`step_idx ≤ 30` 的短样本（占 6.27%；`max_size = 512 // (16 × 1) = 32`）整体被提升到 f64；
batch 内只要含任一短样本（batch 64 时概率 98.4%、batch 8 时 40.4%），collate 的 `np.stack` 就把 memory 三键
整批提升 f64——batch 载荷约 757 MB（仅 `static_image_emb` 一键就 537 MB）在
worker → collate → IPC → device_put 全程白搬运，host 侧再由 jax 降回 f32 上卡。
剩余 1.6% 的满长 batch 以 bf16 上卡——**dtype 随 batch 摆动，XLA 因此编译两份产物**。

**模型侧已实测钉死**：第一层 `nnx.Linear` 显式 `dtype=bfloat16`，flax 的 `promote_dtype` 在任何算术前把输入
统一转 bf16；bf16→f32→f64 均为精确升位（全部 65,536 个 bf16 位型验证过往返无损）——**三种交付
（bf16 / f32 / f64）进投影层与编码器的实际张量逐位相同**（真实形状实测 memory token 输出全等，max 差 0.0）。

**为什么单独拆成一个阶段**：先在旧链路上原地修掉这个双路径，IO 重构的 A/B 两侧 dtype 就天然相同，
其 replica 复刻模式、f32 回退、dtype 三态开关、第 3 层验证与集群侧 dtype 抽查全部可删——
每份计划变成单变量、归因干净。

**预期收益（实测数字）**：batch 载荷 757 → 约 257 MB、collate 52 → 19 ms、device_put 73 → 23 ms、
XLA 编译产物 2 份合 1 份、worker 在途内存约降 2/3。

---

## 三、改动内容（唯一一处，三行）

[`src/mme_vla_suite/shared/data_utils.py`](../src/mme_vla_suite/shared/data_utils.py) 的
`right_padding_token_emb`：三个 `np.zeros`（img / pos / state 的 padding 段）各自加 `dtype=对应输入.dtype`。
mask 的 padding 已显式 `dtype=np.bool_`，不动；满长分支（纯切片）不动。

- **红线例外授权**：该文件属 IO 重构计划红线「`shared/**` 不动」的范围；用户于 2026-08-26 授权**精确到函数**的例外——
  仅限该函数的三个 `np.zeros` 加 dtype 参数，其余 `shared/**` 仍不动。
- **实际影响面**：全仓库只有 `mem_buffer._prepare_frame_sampling` 调用该函数，但它**不区分
  context / modulation / expert 变体**，且同时被训练侧（`training/dataset.py`）与在线评估侧
  （`policies/policy.py` 的 history 准备路径）调用——修复会同时改变全部 frame_sampling 变体与在线评估的
  padding dtype。recurrent 路径用 `left_padding_token_emb`（不动），symbolic / token_drop 零波及。
- **state 修复的可观测性缺口**：`static_state_emb` 交付键经 `_normalize_state`（norm stats 为 f64）恒为 f64、
  修复前后不变——第三处 `np.zeros` 的修复在交付键与梯度上**均不可观测**，其唯一有效证据是归一化前的
  纯函数位型测试（见第五节）。
- **无运行时开关**：改动即行为。A/B 对比是跨 commit 的（修复前 clean HEAD vs 修复后 clean HEAD），
  diff 面唯一，归因无歧义。

---

## 四、重构前后链路图（dtype 视角）

### 4.1 修复前

```
 npy 存储值：image bf16 / pos f32 / state f32
   │ ③④ 选帧 + gather                          （dtype 不变：bf16 / f32 / f32）
   ▼
 ⑤ right_padding_token_emb
   ├─ 短样本 step_idx≤30（6.27%）：np.zeros 无 dtype → concatenate 整体提升
   │     image bf16→f64（2.1 MB→8.4 MB）、pos f32→f64、state f32→f64
   └─ 满长样本 step_idx≥31：纯切片分支，保持 bf16 / f32 / f32
   ▼
 ⑥ state 支路：`_normalize_state`（norm stats q01/q99 为 f64）
   → state 交付键**恒 f64**（与 ⑤ 的 padding dtype 无关，短/满长样本同）
   ▼
 ⑧ worker 内 collate（np.stack）：batch 含任一短样本（b64 98.4% / b8 40.4%）
   → image/pos 整批提升 f64（batch 载荷 ~757 MB，image 单键 537 MB）；state 本就恒 f64
   ▼
 IPC 回主进程（757 MB）→ jax host 侧把 f64 降回 f32（白搬运）→ H2D（73 ms）
   ⚠ 1.6% 满长 batch 以 bf16 交付 → dtype 随 batch 摆动，XLA 编译两份
   ▼
 GPU promote_dtype → bf16 进 pos_proj / encoder_static（三种交付逐位相同，已实测）
```

### 4.2 修复后

```
 npy 存储值：image bf16 / pos f32 / state f32
   │ ③④ 选帧 + gather                          （不变）
   ▼
 ⑤ right_padding_token_emb：np.zeros(dtype=输入.dtype)
   短样本 padding 后每键 dtype 与满长样本完全一致：image bf16 / pos f32 / state f32
   （`max_size = 32`，边界在 step_idx 的 30 / 31 之间）
   ▼
 ⑥ state 支路：`_normalize_state`（norm stats q01/q99 为 f64）
   → state 交付键**恒 f64**（此跳修复前后不变——state 的 f64 消不掉）
   ▼
 ⑧ collate：image/pos 键 batch 内 dtype 一致，np.stack 零提升；state 键恒 f64（前后同）
   （batch 载荷 ~257 MB，image 单键 134 MB）
   ▼
 IPC（257 MB）→ device_put：image bf16 / pos f32 直付；state f64 由 jax host 侧
   canonicalize 回 f32（此跳修复前后不变，仅剩 state 一键、字节量极小）（23 ms）
   100% batch 同一 dtype 组合 → XLA 编译产物 1 份
   ▼
 GPU promote_dtype → bf16 进 pos_proj / encoder_static（张量与修复前逐位相同）
```

**每一跳有没有改数**：③④⑦ 不动；⑥ 代码不动、输出恒 f64（修复前后同）；
⑤ 只改 padding 填充区的 dtype（填充值恒为 0，`astype(f32)` 视角下数值逐位不变）；
⑧ 与交付只随 ⑤ 的 dtype 变化，数值不变；GPU 侧输入张量逐位不变。

---

## 五、第一块验证：非训练轻量化对拍

不启动训练，直接对数据集逐样本 / 逐 batch 对拍修复前后的交付内容。工具在 `scripts/dtype-unify/`。

**落盘格式：位型容器，禁用 npy/npz。** `np.save` / `np.savez` 会把 `ml_dtypes.bfloat16` 写成 `V2` void 类型、
读回即丢类型（本仓已实测），用 npz 存 fixture 会让对拍读到错误对象。改为每键一个 `.bin`（原始字节，C-order）
+ 旁置 JSON 记录 shape、逻辑 dtype、字节序、键名；读回按 JSON 重建。dump 工具自带 **round-trip 守卫**：
每键写盘后立即读回，断言与内存对象逐位相同且逻辑 dtype 还原，失败即 fail-loud，不产出半套 fixture。

**定点样本集（约 2,600 个）由 episode 清单精确构造，不依赖 shuffle 撞边界**，覆盖 padding 全部边界：
`step_idx ∈ {0,1,2,29,30}`（触发 padding 的短样本档）各 200 + `{31,32,33}`（满长边界档）各 200 +
固定 seed 均匀随机 1,000。另取 **200 个真实 batch 过 collate** 做 batch 级对拍，专验「batch 内含短样本时
`np.stack` 整批提升 f64」这一行为的消失。

**流程**：先提交全部工具（否则「修复前 clean HEAD」上工具不存在，跑 dump 必破坏 clean HEAD）→
在该 clean HEAD 跑修复前 dump → 提交三行修复 → 修复后 clean HEAD 重跑同 dump → 离线对拍。

**判据四条**：① 全键 shape 相同；② 数值 `astype(f32)` 后按位视图逐位相同（零容差）；
③ dtype 变化逐键清单与预期完全一致（`static_image_emb` 短样本 f64→bf16、满长不变；
`static_pos_emb` 短 f64→f32、长不变；`static_state_emb` 恒 f64 不变；`static_mask` 恒 bool 位相同；
memory 之外全部键 dtype 与位均须完全相同）；④ collate 后 memory 三键 dtype 恒定、不随 batch 组成摆动。

**归一化前纯函数位型测试**：绕开 `_normalize_state`，直接对 `right_padding_token_emb` 做函数级对拍——
构造三键输入（短样本 / 满长 / 边界各档），断言修复后输出三键 dtype 与各自输入一致、填充区全零、
非填充区与输入逐位相同。对 state 键这是**唯一有效验证**（交付键恒 f64 掩盖了修复效果），
对 modulation/expert 变体与在线路径充当函数级证据。

**实测结果**：`COMPARE_DTYPE=PASS`——**2,600 样本 / 200 batch / 0 失配**。
单步定点梯度对拍 `COMPARE_GRAD=PASS`——三档场景（含短样本 batch 为主判据、全短样本 batch 差异密度最大化、
整批满长 batch 作阴性对照）× 32 个叶子全部逐位相同。

---

## 六、第二块验证：本机训练前 1000 步梯度一致（最后检验）

即基线链的 **G1 vs G0b** 对拍：A 侧 = 黄金基线 `v1-grad-baseline-g0b` r1 的固化产物（不重跑），
B 侧 = 修复后的 `v1-dtype-ab-post-r1`。本机 2×RTX 6000 Ada、batch 8、**1000 步**——
batch 8 下含短样本的 batch 占 40.4%，1000 步期望命中约 404 次有差异的场景。

起跑前必过环境指纹 preflight（`BASELINE_ENV=PASS`）。逐步比五个标量的 hex + 12 次完整 TrainState 摘要
（摘要步集与基线完全对齐）。主判据 bitwise；因两侧输入 dtype 不同、XLA 编译产物不同，bitwise 存在
虚假失败的可能，故预设量化判据兜底（形态见第一阶段报告「量化判据」节）。

**实测结果：bitwise 全过。**

| 项 | 结果 |
|---|---|
| 1000 步五标量 hex | 零失配（rel 的 median / p95 / max 全为 0） |
| 12 × 完整 TrainState 摘要 | 零失配 |
| canonical 输入摘要 | `CANON_CHECK=PASS steps=14` |
| 样本 index 序列 | `INDEX_SEQ=PASS n=8072` |
| `scalars_hex.tsv` sha256 | `c799a0b2…`——**与基线两轮逐字节相同** |
| 量化兜底 | 未启用（其前置是 bitwise 失败，本轮 rel 恰为 0） |

**一处需要解释的判定行**：对拍工具的总判定行输出 `DET_CHECK=FAIL`，唯一成因是 **raw 输入摘要 4 处失配**
（步 100/299/400/999，每步仅 `static_image_emb` 与 `static_pos_emb` 两键，同步的 canonical 摘要均一致）。
这正是本次修复要改的东西——dtype 变了，raw 口径（dtype 参与哈希）必然失配。按第一阶段报告确立的口径，
**跨 dtype 场景 raw 摘要不计入判据**，输入侧只看 canonical + index 序列。即工具的总判定行未区分两种口径，
不是判据不过。

---

## 七、性能口径与结果

**正确性 run 与性能 run 必须分跑**，原「A 侧取基线留档、B 侧顺带产出决策数据」的设计作废。三个污染源：

1. **摘要停顿**：一次完整 TrainState 摘要实测 47.3 s（110 叶子），扩到完整 TrainState 后约 95–140 s——
   而本修复的预期收益只有约 80 ms/step，**信号比噪声小两个数量级**；
2. **确定性档非生产口径**：正确性 run 跑在 `deterministic_ops` + `autotune_level=0` 下，kernel 选择不代表正常训练性能；
3. **污染不对称**：修复前后张量 dtype 与字节量不同，摘要与输入摘要的 device_get / sha256 耗时两侧不同，
   污染不会相互抵消。

因此两块正确性通过后，另跑生产档的 `v1-g1-speed`（1000 步、batch 8、seed 42、num_workers=4、
不注入 XLA flags、关闭一切摘要），对比现行速度锚点 `v1-g0-speed-r2`。

| 指标 | `v1-g0-speed-r2`（锚点） | `v1-g1-speed`（修复后） | 变化 |
|---|---|---|---|
| 步时均值 | 1.186 s | **1.1003 s** | **−7.21%**（−85.5 ms/step） |
| GPU util 均值 | 86.5% | **92.10%** | **+5.64pp** |
| 0% 采样占比 | 4.9% | **1.20%** | −3.69pp |
| 慢步数 | 3 | **0** | — |
| epoch 外推 | 15.82 h | 15.10 h | 均值口径 |

绝对收益 85.5 ms/step 与按 collate + device_put 分解预估的约 80 ms/step 吻合。
⚠ 本机口径，按 AGENTS 13 不作最终吞吐结论；集群侧吞吐验收由第三阶段（IO 重构）承担。

---

## 八、执行记录

| 步 | run_name | commit | 内容 | 判定 |
|---|---|---|---|---|
| 工具提交 | — | **V2.4a `f2e7348`** | `scripts/dtype-unify/` 全部验证工具 + README（先行提交，保证修复前侧从 clean HEAD 起跑） | — |
| P3 | `v1-dtype-p3-dump-pre` | `f2e7348` | 修复前 dump + 单步梯度 A 侧 | 位型容器完整、round-trip 守卫过、初始 state 与基线步 0 同源校验 PASS |
| 修复提交 | — | **V2.4b `a0f76f8`** | 三行功能修复（**唯一的 revert 对象**） | — |
| P4 | `v1-dtype-p4-cmp` | `a0f76f8` | 修复后 dump + 第一块对拍 | `COMPARE_DTYPE=PASS` |
| P5 | `v1-dtype-p5-grad` | `a0f76f8` | 单步定点梯度 B 侧 + 与 A 侧逐元素对拍 | `COMPARE_GRAD=PASS`（含阴性对照逐位相同） |
| P6 | `v1-dtype-ab-post-r1` | `a0f76f8` | 1000 步 G1 + 与基线固化产物对拍 | **bitwise PASS**（第六节） |
| P7 | `v1-g1-speed` | `d227931` | 生产档性能 run | **PASS**：−7.21% / +5.64pp（第七节） |

**commit 切分的设计**：工具与修复解绑成两个 commit，是为了消除拓扑矛盾——修复前的 dump 必须在
「修复前 clean HEAD」上跑，若工具此时尚未提交，就只能带着未跟踪文件起跑、破坏 clean HEAD 前提。
失败处置只 revert 修复 commit，验证工具与失败证据（dump、对拍报告、留档）全部保留、不随修复回滚。

**失败处置预案（未触发）**：任何一块不过且量化判据也不过 → revert 修复 commit，IO 重构计划退回
replica/native 机器恢复的形态。

---

## 九、对第三阶段的交接

本阶段产出了 IO 重构（v4 决策）所需的两份输入，均已齐备：

1. **正确性**：G1 vs G0b 千步 bitwise 全过——修复不改变训练结果；
2. **性能**：`v1-g1-speed` vs `v1-g0-speed-r2` 步时 −7.21%、util +5.64pp。

因此后续 IO 重构的 A/B 两侧 dtype 天然相同、变为单变量对比；其 speed 链节点也不再合并承载 dtype 效果，
性能归因干净。IO 重构的现行权威计划见
[`v2-framesamp-restructure-plan.md`](../v2-framesamp-restructure-plan.md)。

---

## 十、溯源

- 源计划与实现级细节（代码改动逐行、工具参数、位型容器格式定义、commit 拓扑、红线、审计修正记录）：
  [`v1-dtype-unify-plan.md`](../v1-dtype-unify-plan.md) 第二部分
- 逐步留档：`docs/training-doc/v1-dtype-p3-dump-pre/`、`v1-dtype-p4-cmp/`、`v1-dtype-p5-grad/`、
  `v1-dtype-ab-post-r1/`、`v1-g1-speed/`
- 工具与判据说明：`scripts/dtype-unify/README.md`
- 前置的确定性定档与黄金基线：[`v1-phase1-gradient-baseline-report.md`](v1-phase1-gradient-baseline-report.md)
