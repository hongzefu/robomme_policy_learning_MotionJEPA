# dtype 统一修复计划（framesample+context 双 dtype 路径消除）

> 本文件是计划文档，尚未实施。2026-08-26 自 `v1-framesamp-restructure-plan.md`（v3）拆分而来：**本计划先行、独立验收，是该 IO 重构计划（v4）的前置**。范围只兼容 `perceptual-framesamp-context` 一种 run。
>
> 拆分依据（用户拍板）：旧训练自身存在两条 dtype 路径并存——98.4% 的 batch 因 padding 未指定 dtype 被提升 float64、回主进程降回 f32 上卡，1.6% 的「整批满长」batch 不触发 padding、以 bf16 上卡。先用本计划把这个问题在旧链路上原地修掉，IO 重构计划的 A/B 两侧 dtype 天然相同，其 replica 复刻模式、f32 回退、dtype 三态开关、第 3 层验证（原 C.4）、GL b64 dtype 抽查全部可删——每份计划变成单变量、归因干净。
>
> **2026-08-26 增补（G0 黄金基线，权威载体 [`v1-gradient-baseline.md`](v1-gradient-baseline.md)，本计划只引用不复制）**：① S1 确定性实验增 **D2-cold** 档（独立重编译可复现，G0 跨期复用的唯一授权闸）；② P6 的 A 侧（修复前）由 G0 基线 run `v1-grad-baseline-g0` 兼任（**仅限正确性对拍**，见⑥），本计划只跑 B 侧，`v1-dtype-ab-pre` 不再单独跑；③ 300 步 A/B 改为**共用 `EXP_NAME`**（共享 per-fusion autotune 缓存，原「独立 EXP_NAME」裁定反转）；④ 量化兜底判据改等价性检验形态（null 标定，删 OLS-p 判据）；⑤ commit 编号顺延——本计划占 V2.1、V2.2、V2.4，V2.3 归 G0 基线固化，IO 重构计划自 V2.5 顺延；⑥ **正确性与性能分跑，且本计划不跑性能 run（2026-08-26 用户裁定）**——正确性对拍 run（带 TrainState 摘要 / batch_digests / 确定性 XLA 档）的 util/步时数据一律不作性能结论（原「B 侧顺带产出决策门数据」设计作废）；dtype 修复预期影响仅 ~80 ms/step，不值专门一轮性能对比，其性能效果并入 IO 重构收官的 `v1-g2-speed` vs `v1-g0-speed` 一并观察（speed 链符号与口径以基线计划「符号总表」为权威）。

## Context（为什么做这件事）

- 现状（实测，数字来源见 IO 重构计划 1.3–1.5 节）：`right_padding_token_emb` 的三个 `np.zeros` 未指定 dtype（默认 float64），`t < 31` 的短样本（占 6.27%）整体提升 f64；batch 内含任一短样本（b64 概率 98.4%、b8 概率 40.4%）时 collate 的 `np.stack` 把 memory 三键整批提升 f64——batch 载荷 ~757 MB（仅 `static_image_emb` 一键 537 MB）在 worker→collate→IPC→device_put 全程白搬运，host 侧再由 jax 降回 f32 上卡。剩余 1.6% 满长 batch 以 bf16 上卡——**dtype 随 batch 摆动，XLA 编译两份产物**。
- 模型侧已实测钉死：第一层 `nnx.Linear` 显式 `dtype=bfloat16`，flax `promote_dtype` 在任何算术前把输入统一转 bf16；bf16→f32→f64 均为精确升位（全部 65,536 个 bf16 位型验证过往返无损）——**三种交付（bf16/f32/f64）进 `pos_proj`/`encoder_static` 的实际张量逐位相同**（真实形状实测 memory token 输出全等，max 差 0.0）。因此本修复**不是引入新行为，而是把现状 1.6% batch 的行为推广到 100%**。
- 直接收益（实测数字）：batch 载荷 757→~257 MB、collate 52→19 ms、device_put 73→23 ms、XLA 编译产物 2 份合 1 份、worker 在途内存约降 2/3。本机 b8 步时预期变化很小（步时大头在 IO，collate+device_put 合计仅省 ~80 ms/step 量级）——**本计划的本职是消除双路径、给 IO 重构铺路，吞吐硬判据由 IO 重构计划的 GL 验收承担**。
- 最终目标口径（用户 2026-08-26 拍板）：整个两计划链条的方向性目标是 **GPU 占用 100%**（north star；验收阈值仍按 IO 重构计划 D 节的 util 均值必达 ≥90%/期望 ≥95% 执行，字面 100% 物理不可达）。**本计划不做性能对比**（2026-08-26 裁定：预期影响 ~80 ms/step，不值专门一轮 run）——是否实施 IO 重构计划由本计划两块正确性验收结论交用户拍板；dtype 修复的性能效果留到 IO 重构收官的 speed 链对比（`v1-g2-speed` vs `v1-g0-speed`）一并体现。

---

# 第一部分（给人看）

## 一、改动内容（唯一一处，三行）

[`src/mme_vla_suite/shared/data_utils.py`](src/mme_vla_suite/shared/data_utils.py) 的 `right_padding_token_emb`：三个 `np.zeros`（img / pos / state 的 padding 段）各自加 `dtype=对应输入.dtype`。mask 的 padding 已显式 `dtype=np.bool_`，不动；满长分支（纯切片）不动。

- **红线例外授权记录**：该文件属 IO 重构计划红线 R2 的「`shared/**` 不动」范围；用户已于 2026-08-26 授权**精确到函数**的例外——仅限该函数的三个 `np.zeros` 加 dtype 参数，其余 `shared/**` 仍不动。
- **波及面已查证封闭**：全仓库只有 `mem_buffer._prepare_frame_sampling`（frame_sampling 训练与在线评估共用）调用 `right_padding_token_emb`；recurrent 路径用的是 `left_padding_token_emb`（不动）。其他 memory 类型（symbolic/recurrent/token_drop）零波及。
- 修复无运行时开关：改动即行为。A/B 对比是**跨 commit** 的（修复前 clean HEAD vs 修复后 clean HEAD），diff 面唯一（一个函数三处 dtype 参数），归因无歧义。

## 二、重构前后链路图（dtype 视角，AGENTS 18）

### 2.1 修复前（现状）

```
 npy 存储值：image bf16 / pos f32 / state f32
   │ ③④ 选帧 + gather                          （dtype 不变：bf16 / f32 / f32）
   ▼
 ⑤ right_padding_token_emb
   ├─ 短样本 t<31（6.27%）：np.zeros 无 dtype → concatenate 整体提升
   │     image bf16→f64（2.1 MB→8.4 MB）、pos f32→f64、state f32→f64
   └─ 满长样本 t≥31：纯切片分支，保持 bf16 / f32 / f32
   ▼
 ⑧ worker 内 collate（np.stack）：batch 含任一短样本（b64 98.4% / b8 40.4%）
   → memory 三键整批提升 f64（batch 载荷 ~757 MB，image 单键 537 MB）
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
   ▼
 ⑧ collate：batch 内 dtype 一致，np.stack 零提升（batch 载荷 ~257 MB，image 单键 134 MB）
   ▼
 IPC（257 MB）→ device_put bf16/f32 直付，host 侧无降精度搬运（23 ms）
   100% batch 同一 dtype 组合 → XLA 编译产物 1 份
   ▼
 GPU promote_dtype → bf16 进 pos_proj / encoder_static（张量与修复前逐位相同）
```

每一跳「有没有改数」：③④⑥⑦ 不动；⑤ 只改 padding 填充区的 dtype（填充值恒为 0，astype(f32) 视角下数值逐位不变）；⑧ 与交付只随 ⑤ 的 dtype 变化，数值不变；GPU 侧输入张量逐位不变（恒等链见第三节）。

## 三、一致性讨论（两块，AGENTS 18）

### 第一块：非训练轻量化测试下的一致性

不启动训练，直接对 `RoboMMEDataset` 逐样本 / 逐 batch 对拍修复前后交付内容（工具与判据见 T3）：

- 定点样本集覆盖 padding 全部边界（step∈{0,1,2,29,30,31,32,33} + 随机），逐键判据：`astype(f32)` 后逐位相同 + dtype 逐键变化清单与预期完全一致（image 短样本 f64→bf16、pos 短样本 f64→f32、state padding 段 f64→f32、满长样本全键位相同、mask 恒 bool 位相同）。
- 200 个真实 batch 过 `_collate_fn` 对拍，专验 collate 提升行为的消失。

### 第二块：本机启动训练，前 300 步梯度一致（最后检验）

本机 2×RTX 6000 Ada、b8、300 步（用户拍板；b8 下含短样本 batch 占 40.4%，300 步期望命中约 121 次差异场景）：

1. 前置：确定性前提确立（原 IO 重构计划 S0/S1 移入本计划，见 T2）——同配置重跑两轮逐步校验和完全一致后，才开任何 A/B。
2. 单步定点梯度对拍（最便宜先跑）：含短样本 batch（主判据，唯一有 dtype 差异的场景）/ 全短样本 batch（差异密度最大化）/ 整批满长 batch（阴性对照，两侧本就同为 bf16，必须逐位相同）。
3. 300 步 A/B：**A 侧＝G0 黄金基线固化产物**（`v1-grad-baseline-g0`，兼任本计划修复前一侧，定义与产物见 `v1-gradient-baseline.md`），本计划只跑 B（修复后，`v1-dtype-ab-post`）；B 起跑前必过 `BASELINE_ENV` preflight，并应尽快接续 G0（理想同场次——跨期复用能力是给无法重跑 A 侧的后续链节用的，不是推迟 B 侧的理由）。逐步比 loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm 五个标量 hex + 每 100 步（步 0 与末步必记）完整 TrainState 摘要。主判据 bitwise；因 A/B 输入 dtype 不同、XLA 编译产物不同，bitwise 存在虚假失败的可能——预设量化判据兜底（等价性检验形态，权威版本见基线计划「量化判据」节），启用前须先证 B 自身重跑稳定。
4. **两块全部通过才允许宣称修复等价；任何一块不过且量化判据也不过 → revert 修复 commit，IO 重构计划退回 v3 形态（replica/native 机器恢复）。**

## 四、性能口径说明（本计划不跑性能 run）

**正确性 run 与性能 run 必须分跑（2026-08-26 用户裁定）**，原「A 侧取 G0 留档、B 侧顺带产出决策门数据」的设计作废，依据是正确性 run 对性能口径的三个污染源（留作分跑规约的论证记录）：

1. **摘要停顿**：一次完整 TrainState 摘要实测 47.3 s/次（110 叶子），扩完整 TrainState 后按 2–3× 估（约 95–140 s/次）——本修复的预期收益只有约 80 ms/step（collate+device_put 合计），信号比噪声小两个数量级，util 均值 / 0% 采样占比 / 步时全部失真；
2. **确定性档非生产口径**：G0/B 侧跑在 deterministic_ops + autotune 0 下，kernel 选择不代表正常训练性能（基线计划八节 8.2 已把此列为「三条不可比」之一）；
3. **污染不对称**：修复前后张量 dtype/字节量不同，摘要与 batch_digests 的 device_get / sha256 耗时两侧不同，污染不会相互抵消。

同批裁定 **dtype 修复本身不值专门一轮性能对比**（预期影响 ~80 ms/step 量级）：

- 性能对比统一走 **speed 链**（符号、命名与口径以基线计划「符号总表」为权威）：速度基线 `v1-g0-speed` 已随基线计划 PG0-speed 步骤在修复前预跑；dtype 修复的性能效果并入 IO 重构收官的 `v1-g2-speed` vs `v1-g0-speed` 一并体现，本计划不产出性能数字。
- 是否实施 IO 重构计划（v4）由本计划**两块正确性验收结论**交用户拍板，不再依赖性能对比。
- G0 与 B 侧（`v1-dtype-ab-post`）等一切带摘要/确定性档的 run，其 util/步时数据一律仅作留档参考。

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

## T2 确定性前提（原 IO 重构计划 C.0/S1 移入，内容不变，此处为权威版本）

已核实根因链与杠杆（详细论证见 IO 重构计划 3.3，不重复）：共用 jax 编译缓存即复用编译产物与 XLA per-fusion autotune 结果；六个确定性相关 `XLA_FLAGS` 已确认存在于 venv 的 XLA 插件二进制。

- **S0 脚本改动**（验证资产，非训练代码；只做本计划需要的子集）：
  1. `scripts/smoke-local/run_2gpu_epoch_bench.sh` 拆 `EXP_NAME`（决定编译缓存目录，A/B 共用）与 `RUN_TAG`（决定记录目录，各异）；加 `KEEP_JAX_CACHE=1`；`XLA_FLAGS` 外部注入并写进 env.json。
  2. 缓存目录收敛（AGENTS 14，不动 `train.py`、不覆盖 `HOME`）：启动前 `mkdir -p v1-store/cache/jax/$EXP_NAME && ln -sfn <该目录> ~/.cache/jax_$EXP_NAME`；收官统一清理软链与缓存目录。
  3. `scripts/smoke-local/bench_train_steps.py` checksum recorder 扩展：逐步标量 hex（loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm）外，每 `SAVE_INTERVAL` 记一次完整 TrainState 摘要（params/opt_state/EMA/step 全部叶子逐个 sha256 汇成 `state_digest`）。
  4. 同步更新 `scripts/smoke-local/README.md`。
  5. （原 S0 的 preflight packed 兼容与 `BENCH_DUMP_IDX` 两项**不在本计划**，留给 IO 重构计划 v4 自补。）
  6. **G0 基线计划的 P1 扩展项一并在本步交付**（清单与逐项作用以 `v1-gradient-baseline.md` 执行序列 P1 节为权威）：`batch_digests` 输入摘要记录、env.json 改记真实 argv（现 heredoc 硬编码 seed/fsdp_devices，参数化后会说谎）、编译缓存命中/编译计数进 env.json、`--checkpoint-base-dir` 按 RUN_TAG 分目录、新脚本 `check_baseline_env.py` 与 `compare_baseline.py`、README 同步扩展至确定性口径；另加**性能口径开关**——`SAVE_INTERVAL=0` 禁 TrainState 摘要、`BATCH_DIGESTS=0` 禁输入摘要（默认值保持现行为，符合红线 D3），供 speed 链全部 run 使用（口径以基线计划「符号总表」为权威）。
- **S1 四档实验**（各两轮相同 run，100 步，SAVE_INTERVAL=50——步 0 与末步必记，每轮 3 次摘要；2026-08-26 由 10 稀疏化，分歧定位主靠逐步标量 hex）：D0=现状删缓存（预期 FAIL，复现对照；**两轮产物固化留档**，标注「字面现状口径，非判据基线，只作噪声底与口径对照」）；D1=共用缓存；D2=D1+`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`（期望 PASS）；**D2-cold**=同 D2 flags、两轮各用全新空缓存目录（`EXP_NAME` 各异）——强制独立编译两次，证「编译两次得到同样行为」，这是 D1/D2 共用缓存测不到、而「G0 固化后跨期复用」唯一依赖的性质，**D2-cold PASS 是 G0 跨期充当 bitwise 判据一侧的唯一授权闸**。判定行：两轮逐步标量 hex 与全部 `state_digest` diff 为空（逐档）。取首个 PASS 档固定为本计划第二块与 IO 重构计划全部 A/B 的环境；按 D2-cold 结果落三支处置（PASS / FAIL 但共享缓存档 PASS / 全 FAIL，权威表述见基线计划「P2 三支处置」）；D2 仍 FAIL 则加 exclude flag 并降 50 步二分。
- 摘要开销按实测 47.3 s/次（110 叶子）为基准，扩完整 TrainState 后按 2–3× 估；摘要频率已稀疏化（100 步 run 3 次、300 步 run 4 次），每轮摘要总停顿控制在 10 min 以内。

## T3 第一块工具与判据

新目录 `scripts/dtype-unify/`：`dump_fixture_samples.py`（定点样本/batch dump 落 npz）+ `compare_dtype_fix.py`（离线对拍）+ `README.md`。

- 定点集（由 `episode_manifest.json` 精确构造，不依赖 shuffle 撞边界）：step_idx∈{0,1,2,29,30}（触发 padding）各 200 + {31,32,33}（满长边界）各 200 + 固定 seed 均匀随机 1,000——共 ~2,600 样本（较 IO 重构计划 C.2 的 8,200 缩减：本计划改动不涉身份/行号，无需每 episode 首样本组）。
- 流程（跨 commit dump-and-diff）：修复前 clean HEAD 跑 `dump_fixture_samples.py`（逐样本全键 npz + 200 个真实 batch 过 `_collate_fn` 的结果）→ 提交修复 → 修复后 clean HEAD 重跑同 dump → `compare_dtype_fix.py` 对拍。
- 判据（判定行 `COMPARE_DTYPE=PASS samples=2600 batches=200 mismatches=0`）：
  1. 全键 shape 相同；
  2. 数值：`astype(f32)` 后 `view(uint32)` 逐位相同（零容差）；
  3. dtype 变化逐键清单与预期完全一致：`static_image_emb` 短样本 f64→bf16 / 满长 bf16 不变；`static_pos_emb` 短 f64→f32 / 长 f32 不变；`static_state_emb` 交付键恒 f64（`_normalize_state` 所致）不变；`static_mask` 恒 bool 位相同；memory 之外全部键（原图/actions/prompt/state 等）dtype 与位均须完全相同；
  4. collate 后 batch：修复后 memory 三键 dtype 恒定（bf16/f32/f64-state），不随 batch 组成摆动。
- 失败输出首个失配 idx/键/元素 hex。运行 `JAX_PLATFORMS=cpu`（transforms 里的 jit 走 CPU 即可，dump 两侧同后端自洽）。

## T4 第二块流程与判据

1. **单步定点梯度对拍**（~5 min）：同一初始 state 各算一步逐元素比梯度；三种 batch 用 T3 落盘的 npz 直接构造（含短样本 batch 主判据 / 全短样本补充 / 整批满长阴性对照）。判据：主判据与补充 bitwise；阴性对照必须逐位相同（不同则说明改动越界，与 dtype 无关，立即停下排查）。**该定点 fixture（固定初始 state + 固定 batch npz）存 `v1-store/fixtures/`、逐文件 sha256 摘要进 git，升格为基线链的常规回归闸**——后续任何 commit 花约 2 分钟即可重锚 G0（见基线计划「三方对拍矩阵」节）。
2. **300 步 A/B**（本机 2 卡 b8，S1 选定的确定性环境）：**A=G0 基线固化产物（不重跑），B=修复后 commit 现跑** `v1-dtype-ab-post`；同 seed 42、同 num_workers、同 `XLA_FLAGS`；**共用 `EXP_NAME`**（2026-08-26 裁定反转：模块级缓存 key 含 HLO，A/B 天然不互相命中、无污染风险，共用目录才能让 `xla_gpu_per_fusion_autotune_cache_dir` 这一目录级按-fusion 缓存把两侧未变化 fusion——整个 LLM 主干——的 autotune 结论复用起来，消除「两次独立 autotune」噪声源；`RUN_TAG` 区分记录目录、`--checkpoint-base-dir` 按 RUN_TAG 分 run 目录避免 `initialize_checkpoint_dir` 的 `FileExistsError`）；SAVE_INTERVAL=100（步 0 与末步必记）；B 起跑前 `BASELINE_ENV=PASS` 是硬前置。判据：五个标量 hex 列 + 每 100 步 `state_digest` 逐步 diff 全空。
3. **量化兜底**（仅当 bitwise 失败且先证 B 重跑两轮自身 bitwise 稳定后启用）：等价性检验形态——rel 各统计档以 null 对（D2-cold 两轮，或 D0 两轮作上界）实测分布标定上界（×2 余量），原绝对阈值（loss 1e-6 / 梯度范数 1e-5 / 末步 param_norm 1e-5 等）降级为下限守卫；趋势判据用逐步包络，不用 OLS-p（原「β>0 且 p≤0.05 即 FAIL」已删除：rel 序列强自相关使 p 值失标定、混沌轨迹下任何扰动都 β>0，无鉴别力）。参数化定义以基线计划「量化判据」节为权威。
4. **本计划不产出性能数据**（2026-08-26 裁定）：P6 的 A/B 两侧均为正确性口径（摘要停顿 + 确定性档），其 util/步时仅落 records 作留档参考；dtype 修复的性能效果并入 IO 重构收官的 `v1-g2-speed` vs `v1-g0-speed` 对比（第四节、基线计划符号总表）。
5. 失败处置：任何判据最终不过 → `revert` 修复 commit（V2.4），本计划关闭，IO 重构计划退回 v3 形态；留档记录失败证据（基线链形态变化见基线计划「revert 链形态」）。

## T5 实施顺序、commit 切分、run_name、留档

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| P1 | S0 bench 驱动改造（T2 子集 + G0 基线计划 P1 扩展项） | — | STEPS=3 连跑两次不拒跑、缓存未删且落 `v1-store/cache/jax/`、argv 如实进 env.json、batch_digests 落盘、autotune 共享实证 | ~1 h |
| P2 | S1 确定性 D0/D1/D2/D2-cold（各两轮 100 步、SAVE_INTERVAL=50）+ 留档（D0 产物固化） | P1 | 两轮标量 hex + state_digest diff 空（逐档）+ 三支处置结论落档 | ~2 h |
| PG0 | G0 基线两轮 + 产物固化 + PG0-speed 速度基线 `v1-g0-speed`（流程与判定以基线计划 PG0 节与符号总表为权威；G0 兼任本计划 P6 的 A 侧，仅正确性对拍） | P2 | `G0_SCOPE=PASS` + `BASELINE_MANIFEST.json` 校验 + 固化 commit + speed 基线落档 | ~3 h |
| P3 | 修复前 dump（T3 修复前侧） | — | npz 落盘完整 | ~15 min |
| P4 | 应用 T1 修复 + 修复后 dump + 第一块对拍 | P3 | `COMPARE_DTYPE=PASS` | ~30 min |
| P5 | 单步定点梯度对拍（fixture 落 `v1-store/fixtures/`） | P2+P4 | 三种 batch 判据全过 | ~15 min |
| P6 | 300 步 B 侧 + 与 G0 固化产物对拍（纯正确性，util 数据仅留档参考；本计划到此收官，不跑性能 run） | PG0+P5，起跑前 `BASELINE_ENV=PASS` | bitwise（或量化兜底）PASS | ~1 h |

- **commit 切分**（2026-08-26 G0 增补后顺延：本计划占 V2.1、V2.2、V2.4，**V2.3 归 G0 基线固化**，IO 重构计划从 V2.5 顺延）：V2.1 bench 驱动改造（P1，含扩展项）→ V2.2 确定性前提确立（P2 四档，含留档）→ V2.3 G0 基线固化（PG0 + PG0-speed，归 `v1-gradient-baseline.md`；`v1-g0-speed` 留档以 `docs:` commit 提交、不占 V2.x 编号）→ V2.4 dtype 修复 + 两块验证（P4–P6，代码改动与工具、留档分文件逐个 add）。
- **run_name（起跑前逐个交用户确认，AGENTS 6）**：确定性实验 `v1-det-d{0,1,2}-r{1,2}` + `v1-det-d2cold-r{1,2}`（同时服务后续 IO 重构计划）；G0 基线 `v1-grad-baseline-g0`（兼任修复前一侧，归 V2.3）；300 步 B 侧 `v1-dtype-ab-post`（`v1-dtype-ab-pre` 不再单独跑，由 G0 兼任正确性一侧）。本计划无性能 run（speed 链 run_name 见基线计划符号总表）。>5 min 的 run 一律 `docs/training-doc/<run_name>/` 留档（AGENTS 17）；P3–P5 短程不强制留档，但对拍判定行与单步梯度结论并入 V2.4 commit body 与 `v1-dtype-ab-post` 的 result.md。
- 所有训练/dump 运行走 `uv run`；>5 min 的 P2/P6 按 AGENTS 7 tmux + tee + `EXIT_CODE=` + Monitor。

## T6 红线（实施期逐条自检）

| # | 红线 |
|---|---|
| D1 | 代码改动仅限 `right_padding_token_emb` 三个 `np.zeros` 加 dtype（授权例外）；其余 `shared/**` 与一切训练代码零改动 |
| D2 | 训练循环/模型/超参/seed 零改动；不新增任何运行时开关或环境变量 |
| D3 | 验证资产改动（S0）默认行为等价现状（新变量不设即旧行为） |
| D4 | 本计划不产出任何性能结论（第四节）；**性能数字禁止取自带 TrainState 摘要 / batch_digests / 确定性 XLA 档的 run**——性能对比统一走基线计划符号总表的 speed 链，本机数字标注与 util 判读按 AGENTS 13/16 |
| D5 | 任何一致性判据最终不过即 revert，禁止「差不多就行」 |
| D6 | run_name 起跑前用户确认；>5 min run 留档；收官清理临时 run 与 jax 缓存软链 |
