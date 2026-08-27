# dtype 统一修复计划（framesample+context 双 dtype 路径消除）

> **实施状态（2026-08-27）：P3–P7 全部执行完毕，两块正确性与性能对比均已产出。** 工具 commit **V2.4a `f2e7348`** + 三行修复 **V2.4b `a0f76f8`**；第一块 `COMPARE_DTYPE=PASS`（2,600 样本 / 200 batch / 0 失配）、单步定点梯度 `COMPARE_GRAD=PASS`（3 档 × 32 叶逐位相同）、第二块 G1 vs G0b 千步 bitwise 全过（`scalars_hex.tsv` sha256 与 G0b r1/r2 逐字节相同）、P7 `v1-g1-speed` vs `v1-g0-speed-r2` 步时均值 **−7.21%** / util 均值 **+5.64pp**。留档见 `docs/training-doc/v1-dtype-p{3,4,5}-*`、`v1-dtype-ab-post-r1`、`v1-g1-speed`；登记簿以 [`v1-gradient-baseline.md`](v1-gradient-baseline.md) T8 为权威。
>
> 本文件原为计划文档。2026-08-26 自 `v1-framesamp-restructure-plan.md`（v3）拆分而来：**本计划先行、独立验收，是该 IO 重构计划（v4）的前置**。范围只兼容 `perceptual-framesamp-context` 一种 run。
>
> 拆分依据（用户拍板）：旧训练自身存在两条 dtype 路径并存——98.4% 的 batch 因 padding 未指定 dtype 被提升 float64、回主进程降回 f32 上卡，1.6% 的「整批满长」batch 不触发 padding、以 bf16 上卡。先用本计划把这个问题在旧链路上原地修掉，IO 重构计划的 A/B 两侧 dtype 天然相同，其 replica 复刻模式、f32 回退、dtype 三态开关、第 3 层验证（原 C.4）、GL b64 dtype 抽查全部可删——每份计划变成单变量、归因干净。
>
> **2026-08-26 增补（G0 黄金基线，权威载体 [`v1-gradient-baseline.md`](v1-gradient-baseline.md)，本计划只引用不复制）**：① S1 确定性实验增 **D2-cold** 档（独立重编译可复现，G0 跨期复用的唯一授权闸）；② P6 的 A 侧（修复前）由 G0 基线 run 兼任（**2026-08-27 换代为 G0b `v1-grad-baseline-g0b` r1，1000 步**；旧 300 步 G0 的 records 已删除、无产物可对）（**仅限正确性对拍**，见⑥），本计划只跑 B 侧，`v1-dtype-ab-pre` 不再单独跑；③ ~~300 步 A/B 改为**共用 `EXP_NAME`**（共享 per-fusion autotune 缓存，原「独立 EXP_NAME」裁定反转）~~——**2026-08-27 再反转回独立 `EXP_NAME`**：G0b 的编译缓存已随 `c6830e0` 清理（原文「G0b 缓存原为 G1 共享而保留，改为即时清理留证」），共用名只会拿到一个空目录 + 冷编译；且确定性档 `--xla_gpu_autotune_level=0` 本就关闭 autotune，「共享 autotune 结论」这一动机不成立；④ 量化兜底判据改等价性检验形态（null 标定，删 OLS-p 判据）；⑤ commit 编号顺延——V2.1（bench 驱动改造）、V2.2（确定性四档）、V2.3（G0 基线固化）**均归基线计划**，本计划只占 V2.4，IO 重构计划自 V2.5 顺延；⑥ **正确性与性能分跑，且本计划不跑性能 run（2026-08-26 用户裁定）**——正确性对拍 run（带 TrainState 摘要 / batch_digests / 确定性 XLA 档）的 util/步时数据一律不作性能结论（原「B 侧顺带产出决策门数据」设计作废）；dtype 修复预期影响仅 ~80 ms/step，不值专门一轮性能对比，其性能效果并入 IO 重构收官的 `v1-g2-speed` vs `v1-g0-speed` 一并观察（speed 链符号与口径以基线计划「符号总表」为权威）——**⑥ 的后半（不跑性能 run）已于同日反转，见⑦**；⑦ **G1-speed 增设（2026-08-26 用户再裁定，反转⑥后半；分跑规约本身不变）**——两块正确性全部通过后新增 P7 跑 `v1-g1-speed`（生产 XLA 档，口径以基线计划「符号总表」speed run 统一口径为权威），vs `v1-g0-speed` 产出 dtype 修复的单独性能对比；是否实施 IO 重构计划（v4）由「两块正确性验收结论（G1 vs G0）+ `v1-g1-speed` vs `v1-g0-speed` 性能对比」两份结论齐备后交用户拍板；G2-speed 对比对象顺延为 vs `v1-g1-speed` + `v1-g0-speed`。

## Context（为什么做这件事）

- 现状（实测，数字来源见 IO 重构计划 1.3–1.5 节）：`right_padding_token_emb` 的三个 `np.zeros` 未指定 dtype（默认 float64），`step_idx ≤ 30` 的短样本（占 6.27%；`max_size = budget 512 // (token_per_image 16 × num_views 1) = 32`）整体提升 f64；batch 内含任一短样本（b64 概率 98.4%、b8 概率 40.4%）时 collate 的 `np.stack` 把 memory 三键整批提升 f64——batch 载荷 ~757 MB（仅 `static_image_emb` 一键 537 MB）在 worker→collate→IPC→device_put 全程白搬运，host 侧再由 jax 降回 f32 上卡。剩余 1.6% 满长 batch 以 bf16 上卡——**dtype 随 batch 摆动，XLA 编译两份产物**。
- 模型侧已实测钉死：第一层 `nnx.Linear` 显式 `dtype=bfloat16`，flax `promote_dtype` 在任何算术前把输入统一转 bf16；bf16→f32→f64 均为精确升位（全部 65,536 个 bf16 位型验证过往返无损）——**三种交付（bf16/f32/f64）进 `pos_proj`/`encoder_static` 的实际张量逐位相同**（真实形状实测 memory token 输出全等，max 差 0.0）。因此本修复**不是引入新行为，而是把现状 1.6% batch 的行为推广到 100%**。
- 直接收益（实测数字）：batch 载荷 757→~257 MB、collate 52→19 ms、device_put 73→23 ms、XLA 编译产物 2 份合 1 份、worker 在途内存约降 2/3。本机 b8 步时预期变化很小（步时大头在 IO，collate+device_put 合计仅省 ~80 ms/step 量级）——**本计划的本职是消除双路径、给 IO 重构铺路，吞吐硬判据由 IO 重构计划的 GL 验收承担**。
- 最终目标口径（用户 2026-08-26 拍板）：整个两计划链条的方向性目标是 **GPU 占用 100%**（north star；验收阈值仍按 IO 重构计划 D 节的 util 均值必达 ≥90%/期望 ≥95% 执行，字面 100% 物理不可达）。**本计划的性能对比只走 speed 链**（2026-08-26 早先裁定「不值专门一轮 run、并入 G2-speed 一并体现」，同日经用户再裁定反转，见顶部⑦）：两块正确性通过后跑 P7 `v1-g1-speed` vs `v1-g0-speed`（第四节）；是否实施 IO 重构计划（v4）由「两块正确性验收结论（G1 vs G0）+ 该性能对比」两份结论齐备后交用户拍板。

---

# 第一部分（给人看）

## 一、改动内容（唯一一处，三行）

[`src/mme_vla_suite/shared/data_utils.py`](src/mme_vla_suite/shared/data_utils.py) 的 `right_padding_token_emb`：三个 `np.zeros`（img / pos / state 的 padding 段）各自加 `dtype=对应输入.dtype`。mask 的 padding 已显式 `dtype=np.bool_`，不动；满长分支（纯切片）不动。

- **红线例外授权记录**：该文件属 IO 重构计划红线 R2 的「`shared/**` 不动」范围；用户已于 2026-08-26 授权**精确到函数**的例外——仅限该函数的三个 `np.zeros` 加 dtype 参数，其余 `shared/**` 仍不动。
- **实际影响面与验收范围分开声明（2026-08-26 审计修正）**：
  - **实际影响面**：全仓库只有 `mem_buffer._prepare_frame_sampling` 调用 `right_padding_token_emb`，但该函数**不区分 context/modulation/expert 变体**，且同时被训练侧（`training/dataset.py`）与在线评估侧（`policies/policy.py` 的 history 准备路径）调用——修复会同时改变全部 frame_sampling 变体与在线评估的 padding dtype。recurrent 路径用 `left_padding_token_emb`（不动），symbolic/token_drop 零波及。
  - **验收范围**：本计划两块验证只覆盖 `perceptual-framesamp-context` 训练路径；其余受影响路径（在线评估、modulation/expert 变体）**不在验收范围**——修复同为「补 dtype 参数、填充值恒 0」的精确升位消除，风险同质，但不得据此宣称已验证，函数级证据由 T3 新增的纯函数位型测试提供。
  - **state 修复的可观测性缺口**：`static_state_emb` 交付键经 `_normalize_state`（norm stats 为 f64）恒为 f64、修复前后不变——第三处 `np.zeros` 的修复在交付键与梯度上均不可观测，其唯一有效证据是 T3 的归一化前纯函数位型测试。
- 修复无运行时开关：改动即行为。A/B 对比是**跨 commit** 的（修复前 clean HEAD vs 修复后 clean HEAD），diff 面唯一（一个函数三处 dtype 参数），归因无歧义。

## 二、重构前后链路图（dtype 视角，AGENTS 18）

### 2.1 修复前（现状）

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

### 2.2 修复后

```
 npy 存储值：image bf16 / pos f32 / state f32
   │ ③④ 选帧 + gather                          （不变）
   ▼
 ⑤ right_padding_token_emb：np.zeros(dtype=输入.dtype)
   短样本 padding 后每键 dtype 与满长样本完全一致：image bf16 / pos f32 / state f32
   （`max_size = 32`，边界在 step_idx 的 30 / 31 之间）
   ▼
 ⑥ state 支路：`_normalize_state`（norm stats q01/q99 为 f64）
   → state 交付键**恒 f64**（此跳修复前后不变——state 的 f64 消不掉，见 T1 注）
   ▼
 ⑧ collate：image/pos 键 batch 内 dtype 一致，np.stack 零提升；state 键恒 f64（前后同）
   （batch 载荷 ~257 MB，image 单键 134 MB）
   ▼
 IPC（257 MB）→ device_put：image bf16 / pos f32 直付；state f64 由 jax host 侧
   canonicalize 回 f32（此跳修复前后不变，仅剩 state 一键、字节量极小）（23 ms）
   100% batch 同一 dtype 组合（bf16/f32/f64-state 恒定）→ XLA 编译产物 1 份
   ▼
 GPU promote_dtype → bf16 进 pos_proj / encoder_static（张量与修复前逐位相同）
```

每一跳「有没有改数」：③④⑦ 不动；⑥（`_normalize_state`）代码不动、输出恒 f64（修复前后同，2026-08-26 审计修正补画此跳）；⑤ 只改 padding 填充区的 dtype（填充值恒为 0，astype(f32) 视角下数值逐位不变）；⑧ 与交付只随 ⑤ 的 dtype 变化，数值不变；GPU 侧输入张量逐位不变（恒等链见第三节）。

## 三、一致性讨论（两块，AGENTS 18）

### 第一块：非训练轻量化测试下的一致性

不启动训练，直接对 `RoboMMEDataset` 逐样本 / 逐 batch 对拍修复前后交付内容。工具是新目录 `scripts/dtype-unify/` 下的三件：`dump_fixture_samples.py`（定点样本/batch dump，位型容器落盘）、`compare_dtype_fix.py`（离线对拍）、`README.md`（机读/实现级细节见 T3）。

**落盘格式：位型容器，禁用 npy/npz（2026-08-26 审计修正）。** `np.save`/`np.savez` 会把 `ml_dtypes.bfloat16` 写成 `V2` void 类型、`np.load` 读回即丢类型（本仓在 IO 重构计划探针中已实测）——用 npz 存 fixture 会让对拍读到错误对象。改为每键一个 `.bin`（原始字节，C-order）+ 旁置 JSON 记录 shape、逻辑 dtype（含 `bfloat16`）、字节序、键名；读回按 JSON 以 `np.frombuffer` + `view(逻辑 dtype)` 重建（格式定义锚定 T3，基线计划五节亦引用于彼）。dump 工具自带 **round-trip 守卫**：每键写盘后立即读回，断言与内存对象逐位相同且逻辑 dtype 还原（覆盖 bf16/f32/f64/bool 全部出现的类型），失败即 fail-loud，不产出半套 fixture。

**定点样本集（~2,600 个）由 `episode_manifest.json` 精确构造，不依赖 shuffle 撞边界**，覆盖 padding 全部边界：step_idx∈{0,1,2,29,30}（触发 padding 的短样本档）各 200 + {31,32,33}（满长边界档）各 200 + 固定 seed 均匀随机 1,000。较 IO 重构计划 C.2 的 8,200 缩减：本计划改动不涉身份/行号，无需每 episode 首样本组。另取 **200 个真实 batch 过 `_collate_fn`** 做 batch 级对拍，专验「batch 内含短样本时 `np.stack` 整批提升 f64」这一 collate 提升行为的消失。

**流程（跨 commit dump-and-diff，commit 拓扑见 T5）**：先以 V2.4a 提交全部工具（否则「修复前 clean HEAD」上工具不存在，跑 dump 必破坏 clean HEAD）→ 在 V2.4a 这个 clean HEAD 跑修复前 dump（逐样本全键位型容器 + 200 个真实 batch 的 collate 结果）→ 以 V2.4b 提交三行修复 → 修复后 clean HEAD 重跑同 dump → `compare_dtype_fix.py` 离线对拍。运行 `JAX_PLATFORMS=cpu`（transforms 里的 jit 走 CPU 即可，dump 两侧同后端自洽）。

**判据四条（判定行 `COMPARE_DTYPE=PASS samples=2600 batches=200 mismatches=0`；失败输出首个失配 idx/键/元素 hex）**：

1. 全键 shape 相同；
2. 数值：`astype(f32)` 后 `view(uint32)` 逐位相同（零容差）；
3. dtype 变化逐键清单与预期完全一致：`static_image_emb` 短样本 f64→bf16 / 满长 bf16 不变；`static_pos_emb` 短 f64→f32 / 长 f32 不变；`static_state_emb` 交付键恒 f64（`_normalize_state` 所致）不变；`static_mask` 恒 bool 位相同；memory 之外全部键（原图/actions/prompt/state 等）dtype 与位均须完全相同；
4. collate 后 batch：修复后 memory 三键 dtype 恒定（bf16/f32/f64-state），不随 batch 组成摆动。

**归一化前纯函数位型测试（2026-08-26 审计修正新增）**：绕开 `_normalize_state`，直接对 `right_padding_token_emb` 做函数级对拍——构造 bf16/f32/f32 三键输入（短样本/满长/边界各档），断言修复后输出三键 dtype 与各自输入一致、填充区全零、非填充区与输入逐位相同。对 state 键这是**唯一有效验证**（交付键恒 f64 掩盖修复效果，见一节可观测性缺口）；对 modulation/expert 变体与在线路径充当函数级证据。判定并入 `COMPARE_DTYPE` 输出行。

### 第二块：本机启动训练，前 1000 步梯度一致（最后检验）——G1 vs G0b

本块即基线链的 **G1 vs G0b** 对拍（符号以基线计划「符号总表」为权威）：**G0b** = `v1-grad-baseline-g0b` r1 黄金基线固化产物（三计划均未实施的原始训练语义，1000 步，兼任本计划修复前一侧，不重跑）；**G1** = `v1-dtype-ab-post-r1`（修复后 B 侧，本计划现跑）。本机 2×RTX 6000 Ada、b8、**1000 步**（2026-08-27 用户拍板，与 G0b 逐项对齐；b8 下含短样本 batch 占 40.4%，1000 步期望命中约 404 次差异场景）：

1. 前置（**不由本计划执行**，见 T2）：确定性环境与 G0 均由 [`v1-gradient-baseline.md`](v1-gradient-baseline.md) 提供——其 P2 选出「同配置重跑两轮逐步校验和完全一致」的确定性档（D2/D2-cold 双 PASS），其 PG0 跑出并固化 G0 产物，其 P1b 量具补遗（canonical 输入摘要、逐叶数值统计）须在 G1 起跑前完成。本计划不重跑这些，只在 G1 起跑前过 `BASELINE_ENV` preflight。
2. 单步定点梯度对拍（最便宜先跑）：含短样本 batch（主判据，唯一有 dtype 差异的场景）/ 全短样本 batch（差异密度最大化）/ 整批满长 batch（阴性对照，两侧本就同为 bf16，必须逐位相同）。
3. 1000 步 G1 vs G0b：G1 起跑前必过 `BASELINE_ENV` preflight，并应尽快接续 G0（理想同场次——跨期复用能力是给无法重跑 A 侧的后续链节用的，不是推迟 G1 的理由）。逐步比 loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm 五个标量 hex + 12 次完整 TrainState 摘要（`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`，与 G0b 的摘要步集完全对齐）。主判据 bitwise；因 G0b/G1 输入 dtype 不同、XLA 编译产物不同，bitwise 存在虚假失败的可能——预设量化判据兜底（等价性检验形态，权威版本见基线计划「量化判据」节），启用前须先证 G1 自身重跑稳定。
4. **两块全部通过才允许宣称修复等价；任何一块不过且量化判据也不过 → revert 修复 commit，IO 重构计划退回 v3 形态（replica/native 机器恢复）。** 两块全过 → 接 P7 跑 `v1-g1-speed`（第四节），性能对比与两块正确性结论一并交用户做 v4 决策。

## 四、性能口径（speed 链与 G1-speed；2026-08-26 反转裁定）

**正确性 run 与性能 run 必须分跑（2026-08-26 用户裁定，维持不变）**，原「A 侧取 G0 留档、B 侧顺带产出决策门数据」的设计作废，依据是正确性 run 对性能口径的三个污染源（留作分跑规约的论证记录）：

1. **摘要停顿**：一次完整 TrainState 摘要实测 47.3 s/次（110 叶子），扩完整 TrainState 后按 2–3× 估（约 95–140 s/次）——本修复的预期收益只有约 80 ms/step（collate+device_put 合计），信号比噪声小两个数量级，util 均值 / 0% 采样占比 / 步时全部失真；
2. **确定性档非生产口径**：G0/B 侧跑在 deterministic_ops + autotune 0 下，kernel 选择不代表正常训练性能（基线计划八节 8.2 已把此列为「三条不可比」之一）；
3. **污染不对称**：修复前后张量 dtype/字节量不同，摘要与 batch_digests 的 device_get / sha256 耗时两侧不同，污染不会相互抵消。

**G1-speed 增设（2026-08-26 用户再裁定，反转同日早先裁定；见顶部⑦）**：早先同批裁定「dtype 修复本身不值专门一轮性能对比（预期影响 ~80 ms/step 量级），其效果并入 IO 重构收官的 `v1-g2-speed` vs `v1-g0-speed` 一并体现、v4 决策不依赖性能对比」——本轮用户拍板反转：**v4 决策需要同时看到「vs G0 的正确性」与「vs G0-speed 的性能」两份结论**。新口径：

- 两块正确性全部通过后，新增 **P7 跑 `v1-g1-speed`**（G1-speed；生产 XLA 档、`SAVE_INTERVAL=0`、`BATCH_DIGESTS=0`、**1000 步** b8 seed 42 num_workers=4——口径以基线计划「符号总表」的 speed run 统一口径为权威），vs 速度基线 **`v1-g0-speed-r2`**（现行 speed 链锚点：稳态中位 1.152 s/step、util 均值 86.5%、0% 采样 4.9%，1000 步口径；旧 300 步锚 `v1-g0-speed` 已退役）产出 dtype 修复的单独性能对比。
- **是否实施 IO 重构计划（v4）由两份结论齐备后交用户拍板**：① 两块正确性验收结论（G1 vs G0）；② `v1-g1-speed` vs `v1-g0-speed` 性能对比。
- G2-speed 对比对象顺延为 vs `v1-g1-speed` + vs `v1-g0-speed`（基线计划符号总表已同步修订）；副产收益：G2-speed 不再合并承载 dtype 效果，IO 重构的性能归因变干净。
- G0 与 G1（`v1-dtype-ab-post`）等一切带摘要/确定性档的 run，其 util/步时数据一律仅作留档参考，禁作性能结论。

---

# 第二部分（技术细节，供 agent 追踪）

## T1 代码改动

`right_padding_token_emb`（`src/mme_vla_suite/shared/data_utils.py`）短样本分支三处：

```python
np.zeros((max_size - sampled_img_emb.shape[0], *sampled_img_emb.shape[1:]), dtype=sampled_img_emb.dtype)
np.zeros((max_size - sampled_pos_emb.shape[0], *sampled_pos_emb.shape[1:]), dtype=sampled_pos_emb.dtype)
np.zeros((max_size - sampled_state_emb.shape[0], *sampled_state_emb.shape[1:]), dtype=sampled_state_emb.dtype)
```

其余（mask 分支、满长切片分支、`left_padding_token_emb`、其他一切文件）零改动。注意 `state_emb` 的下游 `_normalize_state`（norm stats q01/q99 为 f64）输出恒 f64，修复前后同——第一块对拍的 dtype 预期清单以 padding 输出为准、以最终交付键复核。

## T2 前置：确定性前提与 G0 基线（已移交 `v1-gradient-baseline.md`，本计划不执行）

本节曾持有 S0 bench 驱动改造与 S1 四档确定性实验的细节，2026-08-26 G0 基线计划立项后整体移交，权威版本在 [`v1-gradient-baseline.md`](v1-gradient-baseline.md)，此处只留指针、不复制内容：

- **S0 bench 驱动改造**（`EXP_NAME`/`RUN_TAG` 拆分、`KEEP_JAX_CACHE` 与缓存软链、`XLA_FLAGS` 外部注入、checksum recorder 扩展完整 TrainState，加基线计划自身的六项扩展）→ 基线计划执行序列 **P1** 节。**已实施**，commit `d9e509e`（V2.1）。
- **S1 四档确定性实验**（D0 / D1 / D2 / D2-cold，各两轮 100 步，判定行与三支处置）→ 基线计划执行序列 **P2** 节与 T3；本计划第二块所用的确定性环境即 P2 选定的首个 PASS 档。排查兜底路径不变：D2 仍 FAIL 则加 exclude flag 并降 50 步二分。
- **G0 黄金基线两轮 + 产物固化 + 速度基线 `v1-g0-speed`** → 基线计划 **PG0** 节与「符号总表」。G0 固化产物兼任本计划 P6 的 A 侧（仅正确性对拍）。

（原 S0 的 preflight packed 兼容与 `BENCH_DUMP_IDX` 两项既不在本计划、也不在基线计划，留给 IO 重构计划 v4 自补。）

## T3 第一块工具与判据（机读细节；权威全文见第一部分「三、第一块」）

工具选型论证、定点集构造理由、流程叙述与判据展开已全文上移第一部分「三、第一块」（2026-08-26 用户指令重构），本节只留机读/实现级细节，不重复散文：

- 脚本：`scripts/dtype-unify/dump_fixture_samples.py`（定点样本/batch dump）+ `scripts/dtype-unify/compare_dtype_fix.py`（离线对拍）+ `README.md`。
- **位型容器格式定义（外部引用锚点，基线计划五节「dtype 计划 T3 的位型容器格式」引用于此）**：每键一个 `.bin`（原始字节，C-order）+ 旁置 JSON 记录 `shape`、逻辑 dtype（含 `bfloat16`）、字节序、键名；读回按 JSON 以 `np.frombuffer` + `view(逻辑 dtype)` 重建。禁用 npy/npz（丢 bf16 类型，论证见第一部分）；round-trip 守卫覆盖 bf16/f32/f64/bool 全部出现的类型。
- 定点集规模：step_idx∈{0,1,2,29,30} 各 200 + {31,32,33} 各 200 + 固定 seed 随机 1,000 = ~2,600 样本；batch 级对拍 200 个真实 batch 过 `_collate_fn`。
- 判定行精确格式：`COMPARE_DTYPE=PASS samples=2600 batches=200 mismatches=0`（归一化前纯函数位型测试的判定并入本行；四条判据与 dtype 预期清单见第一部分）。
- 失败输出：首个失配 idx/键/元素 hex。
- 运行口径：`JAX_PLATFORMS=cpu`（transforms 里的 jit 走 CPU 即可，dump 两侧同后端自洽）。

## T4 第二块流程与判据

1. **单步定点梯度对拍**（~5 min）：同一初始 state 各算一步逐元素比梯度；三种 batch 用 T3 落盘的位型容器直接构造（含短样本 batch 主判据 / 全短样本补充 / 整批满长阴性对照）。判据：主判据与补充 bitwise；阴性对照必须逐位相同（不同则说明改动越界，与 dtype 无关，立即停下排查）。**A 侧（修复前）必须在 V2.4a 的 clean HEAD 上与 P3 同场次算出**（2026-08-27 用户裁定：修复无开关，修复落地后就再也取不到 A 侧；原表把 P5 整体排在 V2.4b 之后，只能算出 B 侧）。初始 state 不加载 G0b 那份 45.4 GiB 的 `state_step_0.bin`，而是同 seed 现场 `init_train_state` 后与 G0b r1 步 0 的 177 叶摘要逐条校验同源。**修复后一侧的定点 fixture（固定 batch，T3 位型容器格式）存 `v1-store/fixtures/`、逐文件 sha256 摘要进 git，升格为基线链的常规回归闸**；A 侧梯度数组（32 个可训练叶子、约 11 GB/batch × 3）落本机 `/data/hongzefu/v1-baselines/dtype-p5-grad-pre/`，**P6 验收通过后删除数组本体，只保留逐叶 sha256 清单与逐叶统计**（2026-08-27 用户裁定）——后续任何 commit 花约 2 分钟即可重锚 G0（见基线计划「三方对拍矩阵」节）。
2. **1000 步 A/B**（本机 2 卡 b8，基线计划 P2 选定的确定性档 D2）：**A=G0b r1 固化产物（不重跑），B=修复后 commit 现跑** `v1-dtype-ab-post-r1`；同 seed 42、同 num_workers、同 `XLA_FLAGS`；**独立 `EXP_NAME`**（2026-08-27 裁定再反转：G0b 编译缓存已随 `c6830e0` 清理，共用名只剩空目录 + 冷编译；且确定性档 `autotune_level=0` 本就关闭 autotune，共享 autotune 结论的动机不成立。`--checkpoint-base-dir` 仍按 RUN_TAG 分 run 目录，避免 `initialize_checkpoint_dir` 的 `FileExistsError`）；`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`（摘要步 0/100/…/900/299/999 共 12 次，与 G0b 完全对齐；步集不对齐会被 `compare_baseline.py` 打 WARN 并削掉可比步）；B 起跑前 `BASELINE_ENV=PASS` 是硬前置。判据：五个标量 hex 列 + 12 次 `state_digest` 逐步 diff 全空；**输入侧一致性以 canonical `batch_digests` + index 序列为判据——raw `batch_digests` 因 dtype 变更必然失配，不计入判定**（基线计划三节口径限定；G0b 已原生产出 canonical 与 index 全序列）。
3. **量化兜底**（仅当 bitwise 失败且先证 B 重跑两轮自身 bitwise 稳定后启用）：等价性检验形态——rel 各统计档以 null 对（D2-cold 两轮，或 D0 两轮作上界）实测分布标定上界（×2 余量），原绝对阈值（loss 1e-6 / 梯度范数 1e-5 / 末步 param_norm 1e-5 等）降级为下限守卫；趋势判据用逐步包络，不用 OLS-p（原「β>0 且 p≤0.05 即 FAIL」已删除：rel 序列强自相关使 p 值失标定、混沌轨迹下任何扰动都 β>0，无鉴别力）。参数化定义以基线计划「量化判据」节为权威。
4. **性能对比由 P7 专门产出**（2026-08-26 反转裁定，见第四节）：P6 的 A/B 两侧均为正确性口径（摘要停顿 + 确定性档），其 util/步时仅落 records 作留档参考；dtype 修复的性能效果由 P7 的 `v1-g1-speed` vs `v1-g0-speed` 对比产出（口径以基线计划符号总表为权威），作为 v4 决策的两份输入之一。
5. 失败处置：任何判据最终不过 → `revert` 修复 commit（**仅 V2.4b**——验证工具 V2.4a 与失败证据留档不回滚），本计划关闭，IO 重构计划退回 v3 形态；留档记录失败证据（基线链形态变化见基线计划「revert 链形态」）。

## T5 实施顺序、commit 切分、run_name、留档

**前置（不由本计划执行，流程与判定见基线计划七节，本计划只消费其结论与产物）**：

| 前置步 | 归属 | 本计划要的是什么 | 状态 |
|---|---|---|---|
| P1 bench 驱动改造 | 基线计划 P1 | 能记逐步标量 hex、完整 `state_digest`、`batch_digests` 的量具 | 已完成（commit `d9e509e` / V2.1） |
| P2 确定性四档 | 基线计划 P2 + T3 | 选定的确定性档（`XLA_FLAGS` 与缓存口径），P5/P6 照此配置 | 已完成（V2.2，D2/D2-cold 双 PASS） |
| PG0 G0 基线 + `v1-g0-speed` | 基线计划 PG0 + 符号总表 | （已被 G0b 取代；旧 300 步 records 于 2026-08-27 删除） | 已完成后退役（V2.3） |
| **G0b 重跑 + `v1-g0-speed-r2`** | 基线计划 P1b 落地记录 + T8 | **G0b r1 固化产物作 P6 的 A 侧**（1000 步，原生含 canonical 摘要与 index 全序列）；`BASELINE_ENV=PASS` preflight 通过 | 已完成（`570287f` 起跑，千步两轮逐位一致，`c6830e0` 留档） |
| P1b 量具补遗（审计修正增补） | 基线计划 P1b | canonical 输入摘要口径、逐叶数值统计、uv runner、speed 口径联动——P6 的 G1 vs G0b 对拍依赖前两项 | **已完成**（commitV2.3.1 `570287f`；原「G0 补录 replica」取消，由 G0b 重跑取代） |

**本计划自己要跑的五步**：

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| P3 | 修复前 dump（T3 修复前侧）**+ 单步梯度 A 侧**（2026-08-27 裁定并入） | V2.4a 已提交（工具进 clean HEAD） | 位型容器落盘完整 + round-trip 守卫过 + 初始 state 与 G0b 步 0 同源校验 PASS | ~35 min |
| P4 | 提交 V2.4b 修复 + 修复后 dump + 第一块对拍 | P3 | `COMPARE_DTYPE=PASS` | ~30 min |
| P5 | 单步定点梯度 **B 侧** + 与 P3 的 A 侧逐元素对拍（fixture 落 `v1-store/fixtures/`） | 基线计划 P2 选定档 + P3 的 A 侧 + P4 | `COMPARE_GRAD=PASS`（三种 batch 判据全过；阴性对照 allfull 必须逐位相同） | ~20 min |
| P6 | **1000 步** G1（B 侧，`v1-dtype-ab-post-r1`）+ 与 **G0b r1** 固化产物对拍（纯正确性，util 数据仅留档参考） | G0b r1 固化产物 + P5，起跑前 `BASELINE_ENV=PASS` | bitwise（或量化兜底）PASS | ~55 min |
| P7 | G1-speed 性能 run `v1-g1-speed`（生产 XLA 档、**1000 步**，符号总表口径；2026-08-26 反转裁定新增，见第四节） | P6 PASS（两块正确性全过） | run 完成 + AGENTS 16 稳态统计 + vs **`v1-g0-speed-r2`** 对比报告 | ~30 min |

- **commit 切分（2026-08-26 审计修正：工具与修复解绑，消除拓扑矛盾）**：本计划占 **V2.4a + V2.4b** 两个功能 commit——**V2.4a = `scripts/dtype-unify/` 全部验证工具 + README**（先行提交：否则 P3 的「修复前 clean HEAD」上工具不存在，以未跟踪工具起跑必破坏 clean HEAD）；**V2.4b = T1 三行功能修复**（**唯一的 revert 对象**）。P4–P6 的验证结论与留档随后以 `docs:` commit 提交（分文件逐个 add）。失败处置只 revert V2.4b——验证工具与失败证据（dump、对拍报告、留档）全部保留、不随修复回滚。V2.1（bench 驱动改造，已落地 `d9e509e`）、V2.2（确定性四档）、V2.3（G0 基线固化 + `v1-g0-speed`）均归 [`v1-gradient-baseline.md`](v1-gradient-baseline.md)，编号与内容以该计划 T7 为准（其审计修正增补的 **P1b 量具补遗须在 P6 之前完成**）；IO 重构计划从 V2.5 顺延。
- **run_name 与留档（2026-08-26 审计修正：P3–P5 不再按短程豁免）**：P3/P4/P5 预计 15–30 min，均超 5 min，按 AGENTS 17 视作完整运行——各自从 clean HEAD 起跑（P3 于 V2.4a、P4/P5 于 V2.4b）、唯一 run 名（`v1-dtype-p3-dump-pre` / `v1-dtype-p4-cmp` / `v1-dtype-p5-grad`，起跑前确认）、留档 `docs/training-doc/<run 名>/`（launch.md、result.md、records/）。1000 步 B 侧改名 **`v1-dtype-ab-post-r1`**（基线计划 T7 的「同一符号需多轮正式 run 时一律带 `-r<N>`」体例；`v1-dtype-ab-pre` 不单独跑，A 侧由 G0b 兼任正确性一侧）；P7 的 `v1-g1-speed` 按 speed run 规约留档 `docs/training-doc/v1-g1-speed/`、以 `docs:` commit 提交不占 V2.x（基线计划符号总表），起跑前仍按 AGENTS 6 确认 run_name；前置的 `v1-det-*`、`v1-grad-baseline-g0`、`v1-g0-speed` 由基线计划确认与留档，本计划不重复。
- **运行纪律**：所有训练/dump 运行走 `UV_LINK_MODE=copy uv run`（bench 驱动自身的 uv 收敛由基线计划 P1b 完成）；P3–P7 全部按 AGENTS 7 tmux + tee + `EXIT_CODE=` + 逐级行缓冲 Monitor，不因「短」豁免。

## T6 红线（实施期逐条自检）

| # | 红线 |
|---|---|
| D1 | 代码改动仅限 `right_padding_token_emb` 三个 `np.zeros` 加 dtype（授权例外）；其余 `shared/**` 与一切训练代码零改动 |
| D2 | 训练循环/模型/超参/seed 零改动；不新增任何运行时开关或环境变量 |
| D3 | 本计划新增的验证资产只有 `scripts/dtype-unify/`（离线 dump 与对拍），不得改动 `scripts/smoke-local/` 既有 bench 行为——量具归基线计划 P1，已固化于 `d9e509e` |
| D4 | **性能数字禁止取自带 TrainState 摘要 / batch_digests / 确定性 XLA 档的 run**（正确性 run 的 util/步时仅留档参考）；本计划的性能结论**仅**出自 P7 `v1-g1-speed` vs `v1-g0-speed-r2`（speed 链现行锚点，基线计划符号总表口径），本机数字标注与 util 判读按 AGENTS 13/16 |
| D5 | 任何一致性判据最终不过即 revert（对象仅 V2.4b 修复 commit，工具与证据不回滚），禁止「差不多就行」 |
| D6 | run_name 起跑前用户确认；>5 min run 留档（含 P3–P5）；收官清理临时 run 与 jax 缓存软链 |

## T7 审计修正记录（2026-08-26，两份对抗审计逐条核对后落实）

1. fixture 落盘弃 npz、改位型容器 + round-trip 守卫（T3）——npz 会丢 bf16 类型，按现规格做出的 fixture 读回即错；
2. commit 拓扑拆 V2.4a（工具）/ V2.4b（三行修复），revert 只回滚 V2.4b（T3 流程、T4、T5、D5）——原规格下 P3 所需工具在修复前 clean HEAD 不存在，revert 又会连工具与证据一起删；
3. P3–P5 取消短程豁免，按 AGENTS 17 完整运行处理（T5）；
4. 「波及面已查证封闭」改写为「实际影响面 / 验收范围」两分（一节），新增归一化前纯函数位型测试（T3）——state 修复在交付键上不可观测，原判据等于没验证第三处改动；
5. 两张链路图补画 `_normalize_state` 的 state-f64 一跳（二节），消除与 T3 判据 4「f64-state」的自相矛盾；
6. P6 输入侧判据改用基线计划 P1b 的 canonical 口径（T4），raw 摘要跨 dtype 必然失配、不计入判定；前置表补 P1b 依赖，P2/PG0 状态回填为已完成。

**2026-08-26 用户裁定变更（非审计修正，附记于此）**：

7. 第一部分「三、第一块」详版化——工具与判据全文自 T3 上移，T3 瘦身为机读细节 + 指针（位型容器格式定义仍锚定 T3，保外部引用不破）；「三、第二块」改用 G0/G1 符号口径叙述。
**2026-08-27 用户裁定变更（恢复执行 + 基线链换代对齐，附记于此）**：

9. **本计划恢复执行**——2026-08-27 早先的「放弃执行」指令由同日「实现」指令撤销，基线计划 T8 的 G1 / G1-speed 两行随之解除悬置。
10. **A 侧换代 G0b**：旧 300 步 `v1-grad-baseline-g0` 的 records 已删除，唯一可用 A 侧是 1000 步的 `v1-grad-baseline-g0b` r1；G1 随之升级 1000 步、摘要步集与 G0b 对齐（`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`）。
11. **G1 先跑一轮**（`v1-dtype-ab-post-r1`）；仅当 bitwise 失败、需要启用量化兜底时才补 r2 自证。
12. **EXP_NAME 改回独立**（G0b 缓存已清理，共用无收益）；**speed 锚点换 `v1-g0-speed-r2`**（1000 步，1.152 s/step）。
13. **P5 的 A 侧并入 P3**（修复无开关，A 侧只能在 V2.4a 的 clean HEAD 上取到）；A 侧梯度数组存本机 `/data`，**验收后删除、只留 sha 清单与逐叶统计**。
14. **`left_padding_token_emb` 的同型缺陷不动、也不加注释**（用户裁定「只关注 framesample+context」）——该函数有完全同型的三处无 dtype `np.zeros`，本轮不在范围内，事实记于留档与 commit body。
15. **第一块 batch 级对拍走完整 `transform_dataset` 链**（逐样本层仍走裸 dataset），两层证据互相印证，且 batch 层与 P6 `batch_digests` 的记录点逐字对齐。
16. **`max_size` 订正为 32**（`budget 512 // (token_per_image 16 × num_views 1)`），短样本边界是 `step_idx ≤ 30`；原正文口语的「t < 31」表述已改。实测复算：`step_idx ≤ 30` 占 24800/395289 = 6.2739%，与原文 6.27% 吻合。

---

8. 「G1 无 speed run」裁定同日反转——新增 P7 `v1-g1-speed`（第四节、T5），v4 决策改为「两块正确性验收结论（G1 vs G0）+ `v1-g1-speed` vs `v1-g0-speed` 性能对比」双输入；G2-speed 对比对象顺延 vs `v1-g1-speed` + `v1-g0-speed`；基线计划符号总表/五节/七节/T7/T8 已同步修订。
