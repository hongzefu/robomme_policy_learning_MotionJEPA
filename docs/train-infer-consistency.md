# 训练 / 推理一致性验证（TIC）

本文件是 TIC 的现行正本；`~/.claude-personal/plans/hashed-petting-garden.md`（Codex 审计修订版计划）保留为过程档案，冲突以本文为准。

> **适用环境**：本文正文默认**环境 B**（AWS 单机 8 × A100-SXM4-80GB，仓库工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，存储介质 AWS 本地 NVMe RAID `/dev/md0`；判定口径见 [`../AGENTS.md`](../AGENTS.md) 「运行环境判定」）。凡引用环境 A（GreatLakes / turbo + 2 × RTX 6000 Ada）的数字，句子里一律显式标注「环境 A 历史」，且**不与环境 B 数字放同一张表**。
> **代码锚点写法**：全文引用代码只写 `文件::类/函数/配置键`，不写行号（`AGENTS.md` 第 9 条）。
> **占位符**：`【待填：…】` 是正式实跑尚未收工的位置，收工后逐处回填。

## 目录

1. [结论先行](#一结论先行)
2. [为什么做这件事](#二为什么做这件事)
3. [两侧对照图：训练链路 vs 推理链路](#三两侧对照图训练链路-vs-推理链路)
4. [六关方案与分批卡位](#四六关方案与分批卡位)
5. [判据总表](#五判据总表)
6. [帧特征编码器：训练 f32 离线表 vs 推理 bf16 checkpoint](#六帧特征编码器训练-f32-离线表-vs-推理-bf16-checkpoint)
7. [整段前向 vs 缓存分步：`VT_FULL_VS_CACHED` 超阈值的归因](#七整段前向-vs-缓存分步vt_full_vs_cached-超阈值的归因)
8. [单进程第 28 次 `make_env` 必崩：Vulkan 根因](#八单进程第-28-次-make_env-必崩vulkan-根因)
9. [两项 FAIL 的闭合口径与结果](#九两项-fail-的闭合口径与结果)
10. [其他实测发现](#十其他实测发现)
11. [明确不做与以后可立项](#十一明确不做与以后可立项)
12. [溯源](#十二溯源)

---

## 一、结论先行

**结论措辞（已按 Codex 审计收窄，不得扩写）**：

> 在指定 checkpoint（`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`）、指定 5 条训练集 episode 与 24 集仿真、明确列出的隔离项（帧特征编码器的**权重精度**与**编码批形状**；整段前向 vs 缓存分步两路的 **bf16 kernel 选择与归约序**差）下，**所覆盖路径未发现额外的训练 / 推理不一致**。

**不宣称什么**：不宣称「已排除全部训练 / 推理不一致」。三个理由，缺一不可地限制了结论范围——样本有限（5 条训练集 episode / 24 集仿真，非全库全任务）；在线那一关（第 6 关）没有训练侧真值可比，只验不变量；SigLIP 帧特征差异与 bf16 kernel 差异是被**隔离掉**而非被消除的，它们的量化影响在第六、七章单列。

**正式实跑判定行**（各组唯一成功行；`tic-l0-rhythm-40k` 已收工回填，其余待回填）：

```
TIC_L0=PASS
TIC_RHYTHM=PASS
A19_VALID_DIST=PASS  /  MOTION_DELIVERY=PASS      （400 ep 库，samples=101066）
组 A（`tic-l0-rhythm-40k`，PASS）：`TIC_L0=PASS`、`TIC_RHYTHM=PASS`、`A19_VALID_DIST=PASS source=manifest samples=101066 median=9.0 mean=10.31 max=34 p25/p75/p90/p95/p99=5.0/15.0/20.0/23.0/27.0 zero_frac=0.0633 fill_rate=0.107 expect_mismatches=0 measured_from=dataset_m…`、`MOTION_DELIVERY=PASS samples=101066 mismatches=0 helper_checked=1863`
组 B（`tic-obs-model-40k`）：`TIC_OBS_MODEL=FAIL episodes=5 points=120 motion=store blocking=12/13 observe=8`——13 条阻断 12 条 PASS，唯一 FAIL 为 `VT_FULL_VS_CACHED`（第七章）
组 C（`tic-sidecar-40k`）：`TIC_SIDECAR=FAIL episodes=5 points=120 motion=sidecar windows=140 blocking=13/14 observe=7`——14 条阻断 13 条 PASS，`MOTION_S_VS_SIDECAR=PASS windows=140 mismatches=0`，唯一 FAIL 同上
组 D（`tic-t3-causal-40k`，PASS）：`T3_MOTION_CAUSAL=PASS pad_bitexact=1 loss_bitexact=1 emb_effect=1 pos_effect=1 det_probes=3 nondeterministic_leaves=[] excluded=0 covered=36/36 excluded_diag=[]`、`T3_MECHANISM=PASS step=0 input_grad_ok=1 group_norms_ok=1`
组 F（`tic-vulkan-makeenv`，PASS）：基线 `CRASH at round=28`；`tic-pin rounds_done=35 crash_round=-1`；`tic-tls8192 rounds_done=35 crash_round=-1`
组 E（`tic-eval-probe-40k`）：【待填：TIC_EVAL 实跑判定行（运行中）】
```

**已知的两处不逐位，均已定位到数值来源、非语义错误**：

| 处 | 量级 | 归因 | 本文位置 |
|---|---|---|---|
| 帧特征编码器 | 记忆 token `rel_fro` 0.33%、动作 RMS 3.1e-4 ≈ 采样噪声的 2.3% | 训练建表用离线 **f32** 权重逐帧 `batch=1`，推理用 checkpoint 内 **bf16** 权重整批 16 帧 | [第六章](#六帧特征编码器训练-f32-离线表-vs-推理-bf16-checkpoint) |
| 整段前向 vs 缓存分步 | `v_t` `rel_fro` 1.9e-3 – 3.0e-3、`ulp_p99` 12–31 | 同权重同输入同 mask 同位置号下，`llm([prefix, suffix], …)` 与 `llm([prefix, None], …)` 的 **bf16 kernel 选择与归约序**不同；f32 + `matmul_precision=highest` 下前缀 KV **逐位相同** | [第七章](#七整段前向-vs-缓存分步vt_full_vs_cached-超阈值的归因) |

**一条与评估口径直接相关的工程结论**：单进程第 28 次 `make_env` 必崩的根因已坐实为 **glibc surplus static TLS 被 NVIDIA Vulkan ICD 的反复 `dlopen`/`dlclose` 耗尽**，两种修法各 35 轮不崩，详见[第八章](#八单进程第-28-次-make_env-必崩vulkan-根因)。

---

## 二、为什么做这件事

三 seed 闭环评估里 motion 组 24.2% ± 1.3 与官方 `framesamp+context` 组 24.5% ± 0.5 没有可辨别差异（`training-doc/eval-3seed-context-vs-motion/result.md`；**这组数字出自环境 A 历史的 2 × RTX 6000 Ada，不与环境 B 数字同表**）。可能的解释有三类：

- **(a)** motion token 本身没用；
- **(b)** 训练确实学到了，但**推理侧喂进模型的记忆与训练时不是一回事**；
- **(c)** 评估口径 / 环境噪声。

**本轮只针对 (b)**，不碰 (a)(c)——(a) 要做预算消融与更长训练，(c) 要重跑多 seed，都不在本轮范围。

**此前的闸门止于「键级」**。motion 接入时建立的闸门体系（M1–M5、P1–P5、T2、A22 式 `GRAD_EQ`、T3 五项；环境 B 结果见 [`motion-memory.md`](motion-memory.md) 第八章）证明的是「在线装配出的 8 个记忆键与训练表里那份逐位相同」，`scripts/training/g0/compare_siglip_replay.py` 又用真 39999 checkpoint 在 1 集 11 个 infer 点上把 `_prepare_history` 八键 vs `FrameSampDataset[idx]` 做到逐位（PASS）。**再往下一步都没有对照**：

- 预处理变换之后（两侧 transforms 链结构不同）没比过；
- 模型内部（前缀 token、attention mask、位置号、每层 KV）没比过；
- 最终动作没比过；
- 训练的「整段一次前向」与推理的「前缀缓存 + 分步去噪」两条路没比过；
- 真实评估节奏（哪些时刻做推理、最多多少次、运动 token 会不会撞预算）只有推算没有实测；
- 39999 checkpoint 自身没有离线对拍；训练 run 快照里 motion 库路径记的是 40 ep、实际用的是 400 ep，无闸门；在线位置编码表若落到 CPU 上算会静默偏离，无闸门。

TIC 就是把这几段全部补上，并把结论的适用边界写死。

---

## 三、两侧对照图：训练链路 vs 推理链路

两侧逐跳。「改数」= 这一跳输出的数值与输入不同（编码 / 归一化 / 投影 / 增广算改数；搬运、拼装、丢键、加 batch 维不算）。链路细节的权威描述在 [`motion-memory.md`](motion-memory.md) 第五、六章与 [`dataloader-restructure.md`](dataloader-restructure.md)，此处只列对拍关心的跳。

### 3.1 训练侧（离线表 → `compute_loss`）

| # | 跳 | 代码锚点 | 输出形状 / dtype | 改数？ |
|---|---|---|---|---|
| T0 | 原始 h5 → 建库 pkl | `scripts/dataset/…` 建库链 | `front_rgb`/`wrist_rgb` (256,256,3) uint8；`prompt` = `setup/task_goal`**转小写** | **是**（lower；四任务原文本身即全小写，实际是恒等映射，见第十章） |
| T1 | 离线 SigLIP 编帧 | `dataset_builder/siglip_tokenizer.py::SigLipTokenizer` | 每帧 (16,1152) f32，**权重 f32、逐帧 batch=1** | **是** |
| T2 | 离线 Wan → motion 表 | `scripts/dataset/wan/…` → `scripts/dataset/pack_motion_store.py` | 每 33 帧窗 (768,) f32 | **是** |
| T3 | 读表 + 右填充 + 排序 | `training/framesamp_dataset.py::FrameSampDataset.__getitem__` | 八键：`static_image_emb` (32,16,1152)、`static_pos_emb`、`static_state_emb`、`static_mask` (512)、`motion_emb` (96,768)、`motion_pos`、`motion_mask` (96)、`mem_order` (608) int32 | 否（搬运 / 拼装 / 补零） |
| T4 | `RepackTransform` | `training/config.py::RoboMMEDataConfig.create` 链内 | 丢 8 个非模型键（`epis_idx`、`exec_start_idx`、`grounded_subgoal`、`grounded_subgoal_online`、`is_demo`、`simple_subgoal`、`simple_subgoal_online`、`step_idx`） | 否（`repack_dropped_reach_model=0` 实测） |
| T5 | 归一化 + tokenize | `RoboMMEInputs` / `TokenizePromptWithState` / `PaligemmaTokenizer` | `state`/`actions` 归一化；`tokenized_prompt` (64) int32 + mask | **是** |
| T6 | `preprocess_observation(train=True)` | `openpi/models/model.py` | 图像增广；**记忆 token 不动** | **是**（`AUG_EFFECT_OBS` 实测 `mem_tokens_unchanged=1`） |
| T7 | `embed_prefix` | `history_pi0.py::HistoryPi0.embed_prefix` | (b,**1184**,2048) = 记忆 608 + 图像 512 + 文本 ≤64 | **是**（两路投影 + 一次 gather） |
| T8 | `embed_suffix` + mask/positions + 整段前向 | `history_pi0.py::HistoryPi0.compute_loss` | 拼 **1204** 位；`llm([prefix, suffix], mask, positions)` 一次算完 | **是** |
| T9 | `action_out_proj` | 同上 | `v_t_full` (b,20,32) | **是** |

### 3.2 推理侧（在线现算 → `sample_actions`）

| # | 跳 | 代码锚点 | 输出形状 / dtype | 改数？ |
|---|---|---|---|---|
| I0 | 仿真观测 | `examples/robomme/env_runner.py::EnvRunner` | 同形制 uint8 帧；`task_goal` **原文**（不 lower） | 否 |
| I1 | 在线编帧 | `history_pi0.py::HistoryPi0.vision_encode`（checkpoint 内 `PaliGemma.img`） | 每帧 (16,1152)，**权重 bf16、整批 16 帧（首批 es+1 帧）** | **是** ← 与 T1 的唯一算法性差异 |
| I2 | sidecar 编运动 | `policies/motion_protocol.py::MotionEncoderClient` → `scripts/dataset/wan/wan_motion_infer.py` 复制件 | 每窗 (768,) f32 | **是**（与 T2 同一份代码同一口径；P5 实测在线现编与离线表 772 窗逐位相同） |
| I3 | 在线装配八键 | `policies/framesamp_memory.py::FrameSampMemory` → `policies/policy.py::MME_VLA_Policy._prepare_history` | 与 T3 同八键同形状 | 否 |
| I4 | `InjectDefaultPrompt` | 推理 transforms 链 | obs 已带 prompt ⇒ **no-op** | 否（`inject_default_prompt_noop=1` 实测） |
| I5 | 归一化 + tokenize | 与 T5 **同一批 transform 对象** | 同 T5 | **是** |
| I6 | `preprocess_observation(train=False)` | 同 T6 的函数 | **不增广、不加噪、不采时间** | 否 |
| I7 | `embed_prefix` | 与 T7 **同一函数** | (1,1184,2048) | **是** |
| I8 | 前缀 pass + kv_cache | `history_pi0.py::HistoryPi0.sample_actions` | `llm([prefix, None], mask, positions)` → kv_cache (18,1,1184,1,256)×2 | **是** |
| I9 | 去噪 ×10 | 同上 | `embed_suffix` → `llm([None, suffix], mask, positions, kv_cache)` → `v_t_cached`；`x_t ← x_t − 0.1·v_t` | **是** |
| I10 | 反归一化 | `policies/policy.py::MME_VLA_Policy.infer` | 动作 (20,32) → 关节弧度 | **是** |

### 3.3 逐跳对应关系与对拍关口

```
   训练侧                            对拍关口                    推理侧
T0 h5 → pkl        ──── L1 RAW_OBS_VS_PKL ────────────────  I0 仿真观测（本轮用 h5 回放）
T1 离线 f32 SigLIP ──── 【隔离项】S 臂灌真值 / B 臂只观察 ──  I1 在线 bf16 vision_encode
T2 离线 Wan 表     ──── L1 MOTION_S_VS_SIDECAR（组 C）────  I2 sidecar 现算
T3 读表装配八键    ──── L1 MEM_S_VS_TRAINSET ─────────────  I3 _prepare_history
T4 RepackTransform ──── L2 TRANSFORMS_CHAIN（两侧多出的步骤都是 no-op）── I4 InjectDefaultPrompt
T5 norm + tokenize ──── L2 OBS_T_VS_I / OBS_PROMPT ───────  I5 同一批 transform
T6 增广（train）   ──── L5 AUG_EFFECT_OBS（观察）─────────  I6 不增广
T7 embed_prefix    ──── L3 PREFIX_SHAPE / PREFIX_T_VS_I /   I7 embed_prefix
                            PREFIX_POSITIONS / MEM_INVPERM /
                            KV_T_VS_I
T8 整段 1204 前向  ──── L4 FULL_VS_CACHED_STRUCT /          I8+I9 前缀缓存 + 单步
                            VT_FULL_VS_CACHED
T9 v_t_full        ──── L5 ACT_T_VS_I（10 步去噪逐位）─────  I10 v_t → x_0 → 动作
```

**两侧链条结构上的差异只有两处**，L2 的 `TRANSFORMS_CHAIN` 把它们都证成了不改数：训练侧多一个 `RepackTransform`（只丢 8 个不进模型的键），推理侧多一个 `InjectDefaultPrompt`（obs 已带 prompt 时为空操作）。

---

## 四、六关方案与分批卡位

### 4.1 六关

**背景一句话**：模型每走一步看的输入分两部分——当前两张图 + 一句任务描述；以及「记忆」：过去 32 张历史帧的特征（512 token）、过去若干段运动的特征（≤96 token）、一张告诉模型这 608 个 token 按时间怎么排的次序表。训练时记忆提前算好存在硬盘表里按行读，推理时没有表、机器人边跑边现算。**要验证的就一句话**：同一时刻两边是不是逐字节相同；再往下，同一份输入进模型后两条前向路径算出来的是不是一样。

**样本**：训练集 5 条 episode，覆盖全部 5 种 demo 段长度 `es ∈ {0, 66, 114, 168, 216}`，每种取最长一条——`ButtonUnmaskSwap:3`（g103 / es 0 / T 555）、`VideoUnmask:3`（g203 / 66 / 399）、`VideoUnmaskSwap:5`（g305 / 114 / 446）、`VideoUnmaskSwap:31`（g331 / 168 / 461）、`VideoUnmaskSwap:3`（g303 / 216 / 586）。原始帧按真实评估节奏（首批整段 demo `[0, es]`，之后每批 16 帧）喂给生产口径 policy；**决策时刻清单由脚本独立生成并与训练表逐点核对，不写死**。

| 关 | 比什么 | 白话 |
|---|---|---|
| **第 0 关** 配置与库同源 | 归一化统计量、checkpoint 参数树与 dtype 画像、训练 run 记录的库指纹 vs 当前库、motion 库路径 | 两边用的「尺子」和「权重」是不是同一份；本轮对拍用的库是不是训练时那张（不只比清单，比表内容指纹）；快照里记错的库路径会不会静默用错库 |
| **第 1 关** 输入键 | 记忆的 8 个数组 + 当前图、状态、任务描述原文 | 推理端拼出来的记忆和训练表里那份是不是逐字节一样；运动特征另比「sidecar 现算 vs 表」 |
| **第 2 关** 预处理后 | 两边各过完自己那串 transforms 后的全部键，含任务描述的 token | 证明两侧多出来的步骤都不改数；任务描述按实测判，不预设结论 |
| **第 3 关** 模型内部 | 1184 个前缀 token、attention mask、位置号、每层 key/value | 同一份权重下两边进模型后到注意力前后是不是逐字节一样；顺带补测前缀长度确实 1184、次序表能原样还原 |
| **第 4 关** 整段前向 vs 缓存分步 | 固定同权重、同 obs、同 `x_t`、同 `time`：训练路（前缀+动作一次算完）vs 推理路（前缀存缓存、动作单独算） | 此前方案没有的一关——之前两臂都走推理函数，动作段位置号哪怕错一位也查不出来。结构（mask 前缀子块 / suffix 行 / suffix positions）必须全等；数值差看事先定死的阈值 |
| **第 5 关** 最终动作 | 同一噪声下两边各去噪 10 步的 20 步动作 | 前面都一样，动作就必须逐字节一样 |
| **第 6 关** 真仿真在线 | 不变量（评估用 test split，训练表里没有这些 episode，没有真值可比） | 在线运动 token 个数是否等于按公式该有的个数；次序表是否合法排列（探针记完整数组、汇总器独立重算）；任务描述 token 是否在训练集见过的集合里；位置编码表是否与库表逐位同；24 集与探针记录一一对应；有无任何一集因抛错被记成 error |

**第 6 关的两条纪律**：用 4 任务各 6 集 = 24 集单进程跑（24 < 27，见第八章），且**一次跑完不续评**——续评会把 `error` 条目删掉重评，把问题掩盖过去。

**「唯一一处两边永远不会逐字节一样的地方」**是帧特征编码器（第六章）。本轮所有「必须逐字节」的判据都把它隔离掉：**S 臂**把推理端帧特征直接灌训练表里的真值，全部阻断判据都在 S 臂；真现算的 **B 臂**与离线 f32 成批的 **A 臂**只出观察行。

**另外两件用现成日志能定、本轮变实测的事**：评估最多 1300 步，在线最晚决策时刻是 `es + 1296`（含首批共 82 次推理），此时运动 token 最多 92 个，预算 96 只余 4；第一个撞上限的 demo 段长度是 `es = 289`（当前四任务最长 216）。撞上限的后果链是**静默失败**，见第十章。

### 4.2 分批卡位

| 波 | 组 | 干什么 | 卡 | 留档 |
|---|---|---|---|---|
| 1 | **A** | 第 0 关 + 评估真节奏 CPU 复刻（用 `examples/robomme/utils.py::EpisodeState` 驱动）+ 长时刻与上限边界扫描 + A19 参数化后在 400 ep 库重跑 M1 | 不占卡 | [`training-doc/tic-l0-rhythm-40k/`](training-doc/tic-l0-rhythm-40k/) |
| 1 | **B** | 第 1–5 关，运动特征从库里查表（`--motion store`） | GPU 0 | [`training-doc/tic-obs-model-40k/`](training-doc/tic-obs-model-40k/) |
| 1 | **D** | `T3_MOTION_CAUSAL` 按闭合口径在原库（40 ep）原记录原 batch 重跑 | GPU 2+3 | [`training-doc/tic-t3-causal-40k/`](training-doc/tic-t3-causal-40k/) |
| 1 | **F** | 「单进程第 28 次 `make_env` 必崩」根因排查与修法验证 | GPU 7 | [`training-doc/tic-vulkan-makeenv/`](training-doc/tic-vulkan-makeenv/) |
| 2 | **C** | 第 1–5 关再跑一遍，运动特征换真 sidecar 现算（多比一条「现算 vs 表」） | GPU 0 + 1 | `docs/training-doc/tic-sidecar-40k/`（第二波产出，尚未落盘） |
| 3 | **E** | 第 6 关：探针版 policy server + 主线 `eval.py` 跑 24 集仿真 | GPU 0 + 1 | `docs/training-doc/tic-eval-probe-40k/`（第三波产出，尚未落盘） |

第一波四组同时起互不抢卡；tmux 会话一律 `tic-` 前缀（`tic-l0rhythm`、`tic-obsmodel`、`tic-t3`、`tic-vulkan`），清单写进各 `launch.md`，清理只按确切名 `tmux kill-session -t`。

---

## 五、判据总表

40 行来自计划第二部分五节，另加两类本轮新增：`VT_FULL_VS_CACHED_F32`（第七章的裁决依据，观察）与 `VULKAN_*`（组 F 的复现与修法判据）。

**列的含义**：「开发跑结果」是 commitV7.1 起跑前在 `v1-store/reports/tic-dev/` 与 `v1-store/reports/tic-vulkan/` 做的工具验证跑（样本更小，只用来证明工具本身可用），**不是正式结论**；「正式实跑结果」才是结论来源。组 A 五行已按 [`training-doc/tic-l0-rhythm-40k/result.md`](training-doc/tic-l0-rhythm-40k/result.md)（commit `f302f34`）回填，组 B/C/D/E/F 待各自收工后回填。

| # | 判定行 | 组 | 性质 | 开发跑结果（原文摘要） | 正式实跑结果 |
|---|---|---|---|---|---|
| 1 | `NORM_STATS_SAME` | A | 阻断 | 无（组 A 直接正式跑） | `PASS sha=41ff90e4bfacc158 keys=[actions,state] arrays=8` |
| 2 | `LIB_PROVENANCE_MATCH` | A | 阻断 | 无 | `PASS … all_equal=1`（六指纹逐个相等） |
| 3 | `MOTION_STORE_PATH` | A | 阻断 | 无 | `PASS snapshot=…/4task-motion-40ep/motion effective=…/400ep/motion guard=raise infer_reads_store=0 same_manifest=1 snapshot_default=raise` |
| 4 | `CKPT_PARAM_TREE` | A | 阻断 | 无 | `PASS missing=0 extra=0 leaves=59 model_leaves=59 ckpt_has_motion=1` |
| 5 | `CKPT_DTYPE_PROFILE` | A | 观察 | 无 | `f32_leaves=36 bf16_leaves=23 img_all_bf16=1 trainable_all_f32=1` |
| 6 | `A19_VALID_DIST` | A | 阻断 | 无 | `PASS source=manifest samples=101066 median=9.0 mean=10.31 max=34 zero_frac=0.0633 fill_rate=0.107 expect_mismatches=0 measured_from=dataset_motion_mask` |
| 7 | `MOTION_DELIVERY` | A | 阻断 | 无 | `PASS samples=101066 mismatches=0 helper_checked=1863` |
| 8 | `RHYTHM_EQ` | A | 阻断 | 无 | `PASS episodes=5 points=120 tau_mismatches=0 es_mismatches=0 k_mismatches=0 order_sha_mismatches=0` |
| 9 | `EVAL_TERMINATION` | A | 阻断 | 无 | `PASS max_steps=1300 infer_calls=82 tau_max=es+1296 last_partial_batch_dropped=1 residual_frames=4 termination=count_guard env_done_cases=5` |
| 10 | `TAU_LONG` | A | 阻断 | 无 | `PASS cases=2 es=0 tau=1296 k=80 \| es=216 tau=1512 k=92 budget=96 headroom_min=4` |
| 11 | `ES_BOUNDARY` | A | 阻断 | 无 | `PASS scanned=[200,320) first_raise_es=289 predicted=289 k_at_288=96 k_at_289=97 real_es_values=[0,66,114,168,216] real_max_k=92 margin_windows=4` |
| 12 | `T3_MOTION_CAUSAL`（收窄语义） | D | 阻断 | `PASS pad_bitexact=1 loss_bitexact=1 emb_effect=1 pos_effect=1 det_probes=3 nondeterministic_leaves=[] excluded=0 covered=36/36`（`t3-smoke.log`） | `pad_bitexact=1 loss_bitexact=1 emb_effect=1 pos_effect=1 det_probes=3 nondeterministic_leaves=[] excluded=0 covered=36/36 excluded_diag=[]`（base loss 0.706138；`tic-t3-causal-40k`） |
| 13 | `T3_MECHANISM` | D | 阻断 | `PASS step=0 input_grad_ok=1 group_norms_ok=1` | `step=0 input_grad_ok=1 group_norms_ok=1` |
| 14 | `RAW_OBS_VS_PKL`（图像/状态） | B/C | 阻断 | `PASS points=10 keys=[image,wrist_image,state,prompt_text] mismatches=none prompt_text_mismatch=0` | B/C 同：`points=120 keys=[image,wrist_image,state,prompt_text] mismatches=none prompt_text_mismatch=0` |
| 15 | `MEM_S_VS_TRAINSET` | B/C | 阻断 | `PASS points=10 keys=8 key_mismatches=none` | B/C 同：`points=120 keys=8 key_mismatches=none` |
| 16 | `MEM_S_VS_B` / `MEM_S_VS_A` / `MEM_A_VS_B` | B/C | 观察 | `rel_fro=0.003253 / 0.002874 / 0.003227`，`cos_min=0.999985`，`ulp_p50=2`，frames=649 | B：`MEM_S_VS_B frames=2409 max_abs=1 mean_abs=0.007449 rel_fro=0.003328 cos_min=0.999976 cos_mean=0.999995 ulp_p50=2 ulp_p99=85.2 ulp_m…`；`MEM_S_VS_A rel_fro=0.002957`；`MEM_A_VS_B rel_fro=0.003309`（2409 帧） |
| 17 | `MOTION_S_VS_SIDECAR` | C | 阻断 | `PASS windows=30 mismatches=0 provenance_keys_equal=1 lib=4task-motion-400ep`（`obsmodel-sidecar-dev.log`） | `windows=140 mismatches=0 provenance_keys_equal=1 lib=4task-motion-400ep policy=create_trained_policy motion_client=MotionEncoderClient(online_gpu=1)`（`tic-sidecar-40k`） |
| 18 | `OBS_T_VS_I` | B/C | 阻断 | `PASS points=10 keys=all_equal mismatches=none train_only=['actions'] infer_only=[]` | B/C 同：`points=120 keys=all_equal mismatches=none train_only=['actions'] infer_only=[]` |
| 19 | `OBS_PROMPT`（按实测，可能 FAIL） | B/C | 阻断 | `PASS points=10 text_equal=10 tok_mismatches=0 tokenizer=…(strip,underscore,newline;lower=no)` | B/C 同：`PASS points=120 text_equal=120 tok_mismatches=0`（5 集 h5 原文全小写） |
| 20 | `TRANSFORMS_CHAIN` | B/C | 阻断 | `PASS train_only=['RepackTransform'] infer_only=['InjectDefaultPrompt'] noop_proofs=2 inject_default_prompt_noop=1 repack_dropped_reach_model=0` | B/C 同：`train_only=['RepackTransform'] infer_only=['InjectDefaultPrompt'] noop_proofs=2 inject_default_prompt_noop=1 repack_dropped_keys=['epis_idx', 'exec_start_idx', 'grounded_…` |
| 21 | `PREFIX_SHAPE` | B/C | 阻断 | `PASS len=1184 mem=608 img=512 txt=64 img_per_view=256 views=2` | B/C 同：`len=1184 mem=608 img=512 txt=64 img_per_view=256 views=2 want_mem=608` |
| 22 | `PREFIX_T_VS_I` | B/C | 阻断 | `PASS points=10 leaves=4 mismatches=none` | B/C 同：`points=120 leaves=4 mismatches=none` |
| 23 | `PREFIX_POSITIONS` | B/C | 阻断 | `PASS positions_mismatches=0 attn_mask_mismatches=0 attn_mask_shape=(1,1184,1184) na_branch=1` | B/C 同：`positions_mismatches=0 attn_mask_mismatches=0 attn_mask_shape=(1, 1184, 1184) na_branch=1` |
| 24 | `MEM_INVPERM` | B/C | 阻断 | `PASS token_mismatches=0 mask_mismatches=0 perm_valid=10/10` | B/C 同：`token_mismatches=0 mask_mismatches=0 perm_valid=120/120` |
| 25 | `MEM_SCALE_OBS` | B/C | 观察 | `points=8 ratio_mean=0.08274 min=0.07838 max=0.09423 a20_band=[0.3,3.0] in_band=0/8`（越界，见第十章） | B/C 同：`points=118 ratio_mean=0.08106 min=0.07184 max=0.09954 a20_band=[0.3,3.0] in_band=0/118` |
| 26 | `KV_T_VS_I` | B/C | 阻断 | `PASS layers=18 k_mismatches=0 v_mismatches=0` | B/C 同：`layers=18 k_mismatches=0 v_mismatches=0` |
| 27 | `FULL_VS_CACHED_STRUCT` | B/C | 阻断 | `PASS points=10 prefix_mask_block_equal=10/10 suffix_rows_equal=10/10 suffix_positions_equal=10/10 prefix_kv_bitexact=0/10` | B/C 同：`points=120 prefix_mask_block_equal=120/120 suffix_rows_equal=120/120 suffix_positions_equal=120/120 prefix_kv_bitexact_valid=0/120 prefix_kv_pad_positions_excluded=14716` |
| 28 | `VT_FULL_VS_CACHED` | B/C | 阻断（阈值 `rel_fro ≤ 1e-3`、`ulp_p99 ≤ 4`） | **FAIL** `points=10 rel_fro=0.00304 max_abs=0.01562 vt_rms=1.004 ulp_p99=22 ulp_max=398`；sidecar 档 `rel_fro=0.003017 ulp_p99=31` | **FAIL**（保持）B：`points=120 rel_fro=0.001855 max_abs=0.01953 vt_rms=1.001 ulp_p99=10 ulp_max=492 thr_rel_fro=0.001 thr_ulp_p99=4.0`；C：`points=120 rel_fro=0.003051 max_abs=0.03125 vt_rms=1.001 ulp_p99=22.5 ulp_max=384 thr_rel_fro=0.001 thr_ulp_p99=4.0` |
| 28b | `VT_FULL_VS_CACHED_F32`（本轮新增） | B/C | 观察 | f32+`matmul_precision=highest`：`rel_fro=1.352e-07 prefix_kv_rel_fro=0 prefix_kv_max_abs=0`；f32+TF32：`rel_fro=4.871e-04` | B/C 同：`points=5 rel_fro=1.352e-07 max_abs=7.153e-07 vt_rms=0.994 max_rel_p99=9.369e-06 max_rel_max=0.002407 prefix_kv_rel_fro=0 prefix_kv_max_abs=0 dtype=f32(params=restore_params(dtype=float32),embed_dtype=…` |
| 29 | `AUG_EFFECT_OBS` | B/C | 观察（子判据硬） | `points=10 img_changed=10/10 mem_tokens_unchanged=1 act_rms_norm=0.003949` | B：`points=120 img_changed=120/120 mem_tokens_unchanged=1 act_rms_norm=0.005437`；C：`points=120 img_changed=120/120 mem_tokens_unchanged=1 act_rms_norm=0.005417` |
| 30 | `ACT_T_VS_I` | B/C | 阻断 | `PASS points=10 seeds=1 mismatches=0 determinism_rerun=PASS` | B/C 同：`points=120 seeds=3 mismatches=0 determinism_rerun=PASS` |
| 31 | `ACT_S_VS_B` / `NOISE_S` | B/C | 观察 | `rms_norm=0.0004529 max_abs_norm=0.001953 rms_unnorm=0.0003123 act_std_mean=0.2047`（dev 跑单 seed，`NOISE_S=n/a`） | B：`points=120 rms_norm=0.0004596 max_abs_norm=0.004395 rms_unnorm=0.0003057 \| NOISE_S seeds=3 rms_norm=0.008318 rms_unnorm=0.009596 \| ratio_norm=0.05525 \| act_std_mean=0.2047 rms_unnorm/act_std=0.0014…` |
| 32 | `ACT_CKPT_DTYPE` | B | 观察 | `points=1 ckpt_dtypes=['bfloat16','float32'] rms_norm=0.0008164 max_abs_norm=0.004301 act_std_mean=0.2047` | `points=1 ckpt_dtypes=['bfloat16', 'float32'] rms_norm=0.0006769 max_abs_norm=0.004085 noise_rms_norm=0.008318 ratio=0.08138 act_std_mean=0.2047 wall_s=25.4` |
| 33 | `EVAL_EPISODE_MAP` | E | 阻断 | 2 集 smoke：`FAIL … resets=2 … one_to_one=1 es_match=2/2 goal_match=2/2 expect_episodes=24`（只因样本 2≠24 而 FAIL，映射本身全对） | 【待填】 |
| 34 | `EVAL_TAU_K` | E | 阻断 | `PASS episodes=2 infer_points=8 tau_max=114 k_max=5 budget=96 headroom=91 tau_mismatches=0 k_over_budget=0` | 【待填】 |
| 35 | `EVAL_K_FORMULA` | E | 阻断 | `PASS points=8 mismatches=0 frames_sampled_mismatches=0 window=33 stride=16 max_frames=32` | 【待填】 |
| 36 | `EVAL_ORDER_LEGAL` | E | 阻断 | `PASS points=8 nonperm=0 dtype_int32=1 len608=1 expected_order_mismatches=0 static_mask_bad=0 motion_mask_bad=0` | 【待填】 |
| 37 | `EVAL_BACKEND` | E | 阻断 | `PASS backend=gpu pos_rows=586 pos_table_sha=74ced98d… store_pos_sha=74ced98d… equal=1` | 【待填】 |
| 38 | `EVAL_PROMPT` | E | 阻断 | `PASS tasks=4 episodes=2 train_distinct_prompt=26 tok_in_trainset=8/8 text_in_trainset=8/8` | 【待填】 |
| 39 | `EVAL_NO_RAISE` | E | 阻断 | `PASS episodes=2 errors=0 timeouts=0 unknown=0 log_error_lines=0 server_tracebacks=0` | 【待填】 |
| 40 | `EVAL_DIST_OBS` | E | 观察 | `online_k_median=2.5 mean=2.25 max=5 \| train_k_median=9.0 mean=10.31 max=34 train_samples=101066 \| online_tau_max=114 train_tau_max=585` | 【待填】 |
| F1 | `VULKAN_REPRO`（基线 35 轮） | F | 复现判据 | `CRASH at round=28 err=RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`（`baseline-noreset.log`） | `CRASH at round=28 err=RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`；`PROBE_RESULT tag=tic-baseline rounds_done=28 crash_round=28` |
| F2 | `VULKAN_FIX_PIN`（钉住 RenderSystem） | F | 修法判据 | `rounds_done=35 crash_round=-1`（`pin-renderer.log`） | `PROBE_RESULT tag=tic-pin rounds_done=35 crash_round=-1`（`make_env` 均耗时 6.83 s → 1.19 s） |
| F3 | `VULKAN_FIX_TLS8192`（`GLIBC_TUNABLES`） | F | 修法判据 | `rounds_done=35 crash_round=-1`（`tls8192.log`） | `PROBE_RESULT tag=tic-tls8192 rounds_done=35 crash_round=-1` |
| F4 | `VULKAN_ROOTCAUSE` | F | 结论行 | `glibc_surplus_static_TLS_exhausted_by_svulkan2_Context_recreate … repro_crash_at=28 fix=pin_render_system_in_env_runner fixed_runs=35`（`rootcause.log`） | 同排查阶段结论行（正式 run 三段全部按预期，`EXIT_CODE=0`） |

按上表标注计：**阻断 33 行、观察 7 行**（计划口径），另加 1 条观察（`VT_FULL_VS_CACHED_F32`）与 4 条组 F 判据。各组唯一成功行：`TIC_L0` / `TIC_RHYTHM` / `TIC_OBS_MODEL` / `TIC_SIDECAR` / `TIC_EVAL`。

**唯一一条「FAIL 即停后续层」的项**是 `OBS_PROMPT`——两侧 prompt token 不等时 L3–L5 没有比较意义。其余阻断项 FAIL 时脚本仍把已算出的全部判定行打印完再非零退出。

---

## 六、帧特征编码器：训练 f32 离线表 vs 推理 bf16 checkpoint

这是整条链路上**唯一一处两边永远不会逐字节相同**的地方，2026-09-06 用户拍板不改。本章六段说清它是什么、有多大、为什么不改、本轮怎么隔离、以后想闭合要付什么代价。

### 6.1 问题是什么

同一份 SigLIP 视觉编码器，两侧的用法有**两处**不同：

| | 训练建表 | 在线推理 |
|---|---|---|
| 代码 | `dataset_builder/siglip_tokenizer.py::SigLipTokenizer` | `history_pi0.py::HistoryPi0.vision_encode`（checkpoint 内 `PaliGemma.img`） |
| 权重精度 | 离线 **f32**（`siglip_params.pkl`） | checkpoint 内 **bf16** |
| 编码批形状 | **逐帧 batch = 1** | **整批 16 帧**（首批 `es+1` 帧） |

两处差异各自贡献一份数值噪声，且**互不叠加地同量级**。

### 6.2 权重同源证明

两份权重是同一份参数，不是两次训练的产物——`training-doc/siglip-ab-replay-40k/result.md`：

```
PARAM_SAME_ENC=PASS n_leaves=23 max_abs=0.000e+00
CKPT_IMG_FROZEN=PASS mismatched_leaves=0
```

即：23 个 `img` 子树叶按 f32 比对**逐值相同**（`max_abs=0`），且 39999 checkpoint 里的 `img` 子树与初始权重**一模一样**（训练全程冻结）。所以两侧差异**只能**来自 dtype 与批形状，不可能来自权重本身。第 0 关的 `CKPT_DTYPE_PROFILE` 从另一侧印证：checkpoint 盘上 `img_all_bf16=1`，可训练叶全 f32（`trainable_all_f32=1`）。

### 6.3 实测差多少

**上一轮定量（`siglip-ab-replay-40k`，1 集 11 个 infer 点 / 227 帧）**：

- 记忆 token：`rel_fro = 0.34%`、逐 token `cos_min = 0.999988`、`ulp_p50 = 2`；
- **两个来源同量级**：权重 dtype（A vs B）**0.32%**、批形状（S vs A）**0.30%**；
- **没有第三个来源**：诊断臂 C（A 权重 cast 到 bf16 后成批）与 B **227/227 逐位相等**；
- 传到动作上：固定噪声下 S vs B 的动作 RMS = **0.0003 弧度**，是同一记忆换采样噪声 seed 的 RMS（0.0133）的 **2.3%**，是 `norm_stats.actions.std`（0.2047）的 **0.15%**。

**本轮开发跑（5 集 / 649 帧，`obsmodel-dev.log`）**——量级完全一致：

```
MEM_S_VS_B  frames=649 rel_fro=0.003253 cos_min=0.999985 ulp_p50=2 ulp_p99=86  frac_nonzero=0.781
MEM_S_VS_A  frames=649 rel_fro=0.002874 cos_min=0.999984 ulp_p50=1 ulp_p99=73  frac_nonzero=0.745
MEM_A_VS_B  frames=649 rel_fro=0.003227 cos_min=0.999984 ulp_p50=2 ulp_p99=85  frac_nonzero=0.779
ACT_S_VS_B  points=10 rms_norm=0.0004529 rms_unnorm=0.0003123 act_std_mean=0.2047 rms_unnorm/act_std=0.001526
```

sidecar 档（5 点 / 569 帧）：`MEM_S_VS_B rel_fro=0.003258`、`ACT_S_VS_B rms_unnorm=0.0002845`。**正式实跑数字**：store 档 120 点 / 2409 帧：`MEM_S_VS_B frames=2409 max_abs=1 mean_abs=0.007449 rel_fro=0.003328 cos_min=0.999976 cos_mean=0.999995 ulp_p50=2 ulp_p99=85.2 ulp_max=509 frac_nonzero=0.783`；`ACT_S_VS_B points=120 rms_norm=0.0004596 max_abs_norm=0.004395 rms_unnorm=0.0003057 \| NOISE_S seeds=3 rms_norm=0.008318 rms_unnorm=0.009596 \| ratio_norm=0.05525 \| act_std_mean=0.2047 rms_unnorm/act_std=0.001494`——帧特征差导致的动作差为采样噪声的 5.5%、动作 std 的 0.15%。

### 6.4 为什么不改

1. **用户 2026-09-06 拍板不改**；
2. **在线改 f32 不能闭合**：即便把在线 `vision_encode` 升到 f32，批形状差（成批 vs 逐帧）仍在，rel_fro 只会从 0.33% 降到约 0.30%——要真闭合必须在线也逐帧 `batch=1`，推理延迟随帧数线性上涨；
3. **离线表改 bf16 代价更大**：要重建整个 400 ep 库，并与官方 `framesamp` 表脱钩（对照组就不再是同一份输入）；
4. **影响量级不构成决策依据**：动作差是采样噪声的 2.3%、动作 std 的 0.15%，解释不了三 seed 上 24.2% vs 24.5% 的量级。

### 6.5 本轮怎么隔离

三臂设计（沿用 `compare_siglip_replay.py` 口径，见 `scripts/training/g0/compare_train_infer_obs.py` 模块 docstring）：

- **S 臂**：推理端帧特征**直接灌训练库真值行**，运动路按档位取表（组 B）或真 sidecar（组 C）。**全部阻断判据都在 S 臂**。
- **A 臂**：离线 f32 `SigLipTokenizer` 成批现算，只出帧级观察行。
- **B 臂**：在线 bf16 `policy._vision_encode` 现算（生产推理真身），只出帧级与动作级观察行。

因此本文的结论措辞固定为「**除帧特征编码器精度 / 批形状外**……未发现额外的训练 / 推理不一致」——这一项不是被证明不存在，是被显式排除在结论覆盖范围之外。

### 6.6 想闭合的两条路及代价

| 路 | 做什么 | 代价 | 能达到的等价级别 |
|---|---|---|---|
| 路 1：在线对齐离线 | 在线 `vision_encode` 升 f32 **且**逐帧 `batch=1` | 每次 infer 的帧编码从 1 次 16 帧前向变成 16 次 1 帧前向，延迟线性涨；显存与 f32 权重另占一份 | 逐位相同（两侧同权重同精度同批形状） |
| 路 2：离线对齐在线 | 离线建表改用 checkpoint 内 bf16 权重、成批 16 帧 | 重建 400 ep 库全部帧特征（数百 GB 级重算）；与官方 `framesamp` 表脱钩，官方对照组失效 | 逐位相同，但对照基线要一并重建 |

两条都**不在本轮范围**，作为以后可立项项记录在[第十一章](#十一明确不做与以后可立项)。

---

## 七、整段前向 vs 缓存分步：`VT_FULL_VS_CACHED` 超阈值的归因

这是本轮唯一一条**实测超出事先定死阈值**的阻断判据。本章说清阈值怎么来的、实测是多少、根因是什么、以及为什么本轮**保持 FAIL 记录不放宽**。

### 7.1 阈值怎么定的

第 4 关固定同一权重、同一 obs（S 臂）、同一 `x_t`（固定 noise 与固定 `time = 0.5` 构造 `x_t = t·noise + (1−t)·actions`）、同一 `time`，两路各算一次：

- **训练路径**（照抄 `history_pi0.py::HistoryPi0.compute_loss` 主体）：`embed_prefix` + `embed_suffix` 拼 **1204** 位 → `make_attn_mask(input_mask, ar_mask, na_mask)` → `positions = cumsum(input_mask) − 1` → `llm([prefix, suffix], mask, positions, adarms_cond)` → `action_out_proj(suffix_out[:, -20:])` = `v_t_full`；
- **推理路径**（照抄 `history_pi0.py::HistoryPi0.sample_actions` 的前缀 pass + 一次 step）：`llm([prefix, None], …)` 拿前缀 kv → `embed_suffix` 同 `x_t`/`time` → `suffix_attn_mask = [repeat(prefix_mask) | make_attn_mask(suffix_mask, suffix_ar_mask)]` → `positions = sum(prefix_mask) + cumsum(suffix_mask) − 1` → `llm([None, suffix], mask, positions, kv_cache)` = `v_t_cached`。

阈值 **`rel_fro ≤ 1e-3` 且 bf16 `ulp_p99 ≤ 4`** 由计划事先定死，依据三条：两路的输入、权重、位置号、mask 完全相同，差异只可能来自 1204 行 vs 20 行 query 的 kernel 选择与 bf16 归约序；假设「18 层每层累积 ≤ 1 ULP 量级」则总量应在个位 ULP；`1e-3` 的相对 Frobenius 约为已接受的 SigLIP 差异对记忆影响（0.33%）的 1/3。**事后看，第一条依据成立、第二条（每层 ≤1 ULP）不成立**。

### 7.2 实测

**结构三项 5/5 或 10/10 全等**——这是判断「不是语义错误」的关键：

```
FULL_VS_CACHED_STRUCT=PASS points=10 prefix_mask_block_equal=10/10 suffix_rows_equal=10/10
                            suffix_positions_equal=10/10 prefix_kv_bitexact=0/10
```

即训练大表 `[:1184,:1184]` 子块 == 推理前缀表；训练大表最后 20 行 == 推理 20 行 mask；训练 `positions` 后 20 位 == 推理 suffix `positions`。**mask、位置号、可见性结构完全一致，动作段位置号没有错位。**

**bf16 生产档四次独立跑**：

| 跑 | 点数 | `v_t` `rel_fro` | `max_abs` | `ulp_p99` | 前缀 KV `rel_fro`（只统计 `prefix_mask=True` 的位） |
|---|---|---|---|---|---|
| `obsmodel-dev.log`（store） | 10 | 3.04e-3 | 0.01562 | 22 | 5.963e-3 |
| `obsmodel-sidecar-dev.log` | 5 | 3.017e-3 | 0.01562 | 31 | 3.618e-3 |
| `obsmodel-f32-dev.log`（bf16 主臂） | 5 | 1.915e-3 | 0.01562 | 12 | 4.596e-3 |
| `obsmodel-f32hi-dev.log`（bf16 主臂） | 5 | 1.853e-3 | 0.01562 | 12 | 4.418e-3 |

即 `v_t` `rel_fro` 落在 **1.9e-3 – 3.0e-3**（阈值 1e-3 的 2–3 倍）、`ulp_p99` **12–31**（阈值 4 的 3–8 倍）、前缀 KV valid 位 **0.36% – 0.60%**，`vt_rms ≈ 1.0`。四次跑的数字互不相同，说明其中还含 XLA autotune 的非确定性。

**最关键的一条**：`prefix_kv_bitexact = 0/N` —— **差异在前缀 pass 就已经产生**，与 suffix 段无关。`llm([prefix, suffix], …)` 与 `llm([prefix, None], …)` 在前缀那 1184 行上跑的是**不同的 kernel**（前者的 query 有 1204 行、后者 1184 行，XLA 选的分块与归约顺序不同），于是同一份前缀 token 算出的 k、v 就已经差了 0.36–0.60%；这个差异再经 18 层放大到 `v_t` 的 1.9e-3 – 3.0e-3。

### 7.3 f32 诊断：把差异降到零

`VT_FULL_VS_CACHED_F32` 是为这条裁决专门加的**观察行**：把参数树与 `embed_dtype` 一起升到 f32 后，在同一 obs / `x_t` / `time` 上重跑两路。

| 档 | `v_t` `rel_fro` | `v_t` `max_abs` | 前缀 KV `rel_fro` | 前缀 KV `max_abs` |
|---|---|---|---|---|
| f32（默认 TF32 matmul） | **4.871e-4** | 2.213e-3 | 5.507e-4 | 0.107 |
| f32 + `matmul_precision=highest` | **1.352e-7** | 7.153e-7 | **0** | **0** |

判读非常干净：

1. 只把权重升到 f32、matmul 仍走 TF32 时，差异从 1.9e-3 掉到 4.9e-4（约 1/4）——**说明差异随数值精度走**；
2. 再把 matmul 精度拉满（`matmul_precision=highest`，即真 f32 乘加），**前缀 KV 逐位相同（`rel_fro = 0`、`max_abs = 0`）**，`v_t` 只剩 1.35e-7（f32 机器精度量级，来自 18 层累积）。

**归因结论**：`VT_FULL_VS_CACHED` 的超阈值是 **bf16 归约序与 kernel 选择的纯数值来源**，不是语义错误。语义（mask 结构、位置号、可见性、缓存内容的定义）在 `FULL_VS_CACHED_STRUCT` 里已 5/5 全等，且在 f32+highest 下两路逐位重合——如果两路在语义上有任何不同，提高数值精度不会让差异归零。

### 7.4 本轮处置

**阈值是否按实测重定，属于「放宽判据」，必须交用户裁决**（`AGENTS.md` 第 2 条：范围与实现方式存在歧义先问用户）。脚本不自行改阈值、不自行改阻断性质，判定行里直接带上待裁决标记：

```
VT_FULL_VS_CACHED=FAIL … thr_rel_fro=0.001 thr_ulp_p99=4.0 …
  | note=threshold_pending_user_review(阈值为计划事先定死，实测超出属待裁决项，本脚本不自行放宽；
    结构三项全等，差异自前缀 pass 起，见 VT_FULL_VS_CACHED_F32)
```

**本轮 `VT_FULL_VS_CACHED` 保持 FAIL 记录**，正式实跑结果：store 档 `VT_FULL_VS_CACHED=FAIL points=120 rel_fro=0.001855 max_abs=0.01953 vt_rms=1.001 ulp_p99=10 ulp_max=492 thr_rel_fro=0.001 thr_ulp_p99=4.0`；sidecar 档 `VT_FULL_VS_CACHED=FAIL points=120 rel_fro=0.003051 max_abs=0.03125 vt_rms=1.001 ulp_p99=22.5 ulp_max=384 thr_rel_fro=0.001 thr_ulp_p99=4.0`；两档 `VT_FULL_VS_CACHED_F32 points=5 rel_fro=1.352e-07 max_abs=7.153e-07 vt_rms=0.994 max_rel_p99=9.369e-06 max_rel_max=0.002407 prefix_kv_rel_fro=0 prefix_kv_max_abs=0 dtype=f32(params=restore_params(dtype=float32),embed_dtype=float32,…`。裁决所需的全部依据已在 7.2 / 7.3 就位；若用户裁定按实测重定阈值，建议的口径与依据一并留档，不在本文预判。

**这条 FAIL 不影响第一章的结论措辞**：结论已把「整段 vs 缓存两路的 bf16 kernel 差」列为明确隔离项。

---

## 八、单进程第 28 次 `make_env` 必崩：Vulkan 根因

### 8.1 现象与此前的处置

首次记录在 `training-doc/eval-official-framesamp-context/result.md`：8 worker × 200 集的分片评估里，分到 28 集的 w0、w1 **都在第 28 次 `make_env` 抛** `RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`（`svulkan2` 先报 `Your GPU driver does not support Vulkan`），前 27 次正常；分到 24 集的 6 个 worker 全部正常。当时只判定为「单进程内 Vulkan 资源泄漏累积到上限」，处置是划一条经验红线：**每个 `eval.py` 进程分到的集数 ≤ 27**。

本轮组 F 把这条经验红线做成根因。**环境**：驱动 `595.71.05`、glibc `2.34`、SAPIEN `3.0.3`、Vulkan Loader `1.3.224`（`vkloader-*.log` 的 `Vulkan Loader Version` 行）、ICD `libGLX_nvidia.so.0` / `nvidia_icd.json`。

### 8.2 复现数据：逐轮资源全平坦

`scripts/training/legacy-eval/probe_vulkan_makeenv.py` 循环 `make_env → close_env` 35 轮，每轮记 fd 数、`/proc/self/maps` 里 libvulkan/libnvidia 映射数、host RSS、GPU 显存、sapien 对象存活数（`baseline-noreset.log`）：

| 轮 | fd | fd_nvidia | vk_maps | maps_total | rss_mb | gpu_mem_mb | sapien_objs | dt |
|---|---|---|---|---|---|---|---|---|
| 1 | 155 | 130 | 85 | 2613 | 1588.7 | 774.0 | 12 | 21.57 s |
| 2 | 155 | 130 | 85 | 2621 | 1818.6 | 774.0 | 12 | 9.74 s |
| … | 155 | 130 | 85 | 2612–2628 | 1827–2116 | 774.0 | 12 | 8.2–13.1 s |
| 27 | 155 | 130 | 85 | 2612 | 2037.7 | 774.0 | 12 | 10.00 s |
| **28** | **69** | **51** | **5** | 2282 | 1719.0 | **492.0** | 11 | 1.94 s → `CRASH` |

**判读**：前 27 轮 fd、Vulkan 映射数、显存、sapien 对象数**一个都不涨**，RSS 在 1.6–2.1 GB 之间平坦波动。**第 28 轮那些数字的骤降是崩溃的结果，不是原因**——Vulkan Context 建不出来，本来该被映射的库和该分配的显存自然就少了。轮 2–27 的每轮墙钟均值 **9.87 s**。

### 8.3 五条假设逐条证伪

| # | 假设 | 证伪证据 |
|---|---|---|
| 1 | Vulkan instance 没销毁、累积到上限 | C 层最小复现 `vk_instance_limit.c`（**只建不销毁**）连建 **64 个 instance 全部 `result=0`**（`vk_limit_nodestroy.log`：`VKDONE rounds=64 destroy=0`）。「不销毁」反而不崩 ⇒ 不是 instance 数量上限 |
| 2 | NVIDIA 驱动有每进程逻辑 device 上限 | `vk_device_limit.c` **不销毁**连建 48 个 instance+device 全过（`vk_device_nodestroy.log`：`VKDONE rounds=48 destroy=0`）；同一程序加 `--destroy`（每轮建完立刻销毁）在**第 27 轮** `VKCRASH stage=instance result=-9`（`vk_device_destroy.log`）。**关键对照：销毁才崩** |
| 3 | fd 用尽 | 前 27 轮 fd 恒定 155（其中 nvidia 130），远低于任何上限 |
| 4 | Python / SAPIEN 侧对象没释放，`gc` 能救 | `--gc --renderer-release`（显式 `gc.collect()` + 丢弃 RenderSystem 引用）仍**第 28 轮 CRASH**（`gc-rendrelease.log`），与基线完全一致 |
| 5 | 显存泄漏 | 前 27 轮 GPU 显存恒 774 MB，零增长 |

### 8.4 因果链

```
每轮 make_env
  → ManiSkill BaseEnv 新建 scene → SAPIEN 重建 svulkan2 全局渲染 Context
      → vkCreateInstance
        → Vulkan loader dlopen NVIDIA ICD（libGLX_nvidia.so.0 → libnvidia-*.so）
每轮 close_env
  → BaseEnv._clear() 释放 scene 与 RenderSystem
      → svulkan2 全局 Context 引用计数归零 → 析构 → vkDestroyInstance
        → loader dlclose NVIDIA ICD
每经历一轮 create → destroy：
  glibc 的 surplus static TLS 净泄漏 64 字节（dlopen 的 static TLS 块在 dlclose 后不完全归还）
默认 glibc.rtld.optional_static_tls = 512 字节
  → 第 27~28 次 dlopen ICD 失败 → vkCreateInstance 返回 VK_ERROR_INCOMPATIBLE_DRIVER (-9)
    → svulkan2 抛 `vk::createInstanceUnique: ErrorIncompatibleDriver`
```

`dlopen_icd_limit.c` 直接测这一环：反复 `dlopen`/`dlclose` `libGLX_nvidia.so.0` 本身能做多少次（`--keep` 时不 dlclose 作对照）。

### 8.5 定量验证：崩溃轮 = 19 + tls/64

调 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=<N>` 扫六档（`tls_scan.log`，C 层 `vk_device_limit --destroy`）：

| `optional_static_tls`（字节） | 崩溃轮（实测） | 公式 `19 + N/64` |
|---|---|---|
| 0 | 19 | 19 |
| 128 | 21 | 21 |
| 256 | 23 | 23 |
| 512（默认） | **27** | 27 |
| 1024 | 35 | 35 |
| 2048 | 51 | 51 |

**六档全部精确落在 `19 + N/64` 上**——每轮净泄漏恰好 64 字节，误差为零。这条线性关系是根因成立的决定性证据：泄漏量与 TLS 余量严格成正比，且斜率就是每轮 64 B。

> **口径说明**：`tls_scan.log` 只落了上表六档。8192 档没有进 C 层扫描表；它的落盘证据是 Python 层 `tls8192.log` 的 35 轮不崩（下节）。按公式 `19 + 8192/64 = 147` 外推可支撑 ≥128 轮，`fix-a-glibc-tunables.patch` 的注释按这个外推写。

### 8.6 两种修法与实测

| 修法 | 做法 | 实测 | 副作用 |
|---|---|---|---|
| **B：钉住 RenderSystem**（推荐） | 进程启动时建一个 `sapien.render.RenderSystem(sapien.Device("cuda"))` 挂在模块级变量上长期持有，让 svulkan2 全局 Context 引用计数**永不归零**——整个进程只建一个 `VkInstance` | `pin-renderer.log`：`rounds_done=35 crash_round=-1`，`vk_create=2 vk_destroy=0` 全程不变；**`make_env` 每轮墙钟由 9.87 s 降到 0.90 s**（省掉每集重建 Context + 重编译 shader） | 需改 `env_runner.py`；钉不住时打警告退回旧行为，不让评估直接失败 |
| **A：抬 TLS 余量** | 启动 eval 时加 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192` | `tls8192.log`：`rounds_done=35 crash_round=-1`；每轮 `vk_create` 从 1 递增到 36（Context 照旧每轮重建） | 零代码改动，但**只是把上限推高**，不根治；每轮仍付 6.4 s 重建成本 |

修法 B 的额外收益值得单说：它同时解决了崩溃与性能——`make_env` 从约 10 s 降到约 1 s，24 集评估里光这一项就省下约 3.5 分钟。

### 8.7 落点判断与两份 patch

**本轮不改主线**。两份 patch 只作文本产出、**未应用**：

- `v1-store/reports/tic-vulkan/fix-a-glibc-tunables.patch` → 改 `scripts/training/legacy-eval/eval_shard.local.sh` 与 `eval_shard.remote.sh`，给 `eval.py` 加 `GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=8192}"`；
- `v1-store/reports/tic-vulkan/fix-b-pin-renderer.patch` → 改 `scripts/training/legacy-eval/robomme-local/env_runner.py`，加模块级 `_PINNED_RENDER_SYSTEM` 与 `_pin_render_context()`，在 `EnvRunner.__init__` 开头调用。

**落点建议 legacy-eval，主线零改动**：评估链路的现行维护位置在 `scripts/training/legacy-eval/`（且有 `.local` / `.remote` 两套 shard 脚本），主线 `examples/robomme/` 与 `src/mme_vla_suite/policies/` 本轮承诺零改动。是否应用、应用到哪一侧，交用户裁决。

### 8.8 对评估口径的影响

**「每个 `eval.py` 进程分到的集数 ≤ 27」这条红线可以解除，但有前提**：必须先应用上述任一修法并在目标集数上实测通过。在两份 patch 未应用之前，红线继续有效——本轮第 6 关（组 E）正是按它设计的：4 任务 × 6 集 = **24 集 < 27**，单进程一次跑完。

解除后的收益：8 worker × 200 集不必再为「w0/w1 恰好 28 集」而拆批或改成 10 worker；修法 B 还顺带把每集建环境的 10 s 压到 1 s。

正式实跑结果：基线 `CRASH at round=28 err=RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`（`rounds_done=28 crash_round=28`）；`tic-pin rounds_done=35 crash_round=-1`；`tic-tls8192 rounds_done=35 crash_round=-1`；`EXIT_CODE=0`（`docs/training-doc/tic-vulkan-makeenv/result.md`）。

---

## 九、两项 FAIL 的闭合口径与结果

环境 B 复刻中有两项闸门判 FAIL（原文与证据链见 [`motion-memory.md`](motion-memory.md) 第 8.3 节）。用户 2026-09-06 拍板：**两项都按候选修法闭合**。

### 9.1 `A19_VALID_DIST`：判据参数化

**问题**：`scripts/training/tests/motion_gates_model.py::cmd_m1` 把 **40 ep 库的四个分布量写死**（`k_mean 11.46 ± 0.05`、`k_median == 11`、`k_max == 34`、`zero_frac 0.0555 ± 0.001`），400 ep 库分布不同就必然 FAIL——`MOTION_DELIVERY=FAIL` 是连带失败，不是交付出错（那次逐样本 `mismatches=0` 本就成立）。

**闭合口径**（含审计第 7 条收窄）：

- **期望侧**：新增 `--expect-source {manifest,args}`（默认 `manifest`），从当前库 `meta/episode_manifest.json` 按 oracle 同一公式对**每个 exec 样本独立重算**分布，不再写死；
- **实测侧**：改为统计 `FrameSampDataset[idx]["motion_mask"].sum()`——**真实交付**，而不是此前的 oracle 的 `k`（审计指出旧口径实际是拿 oracle 比 oracle）；
- 两侧逐值相等；逐样本 M1 比较保留。

**正式实跑结果（已收工，`tic-l0-rhythm-40k`）**：

```
[m1 real] samples=101066 mismatches=0 有效数分布 {k_median 9.0, k_mean 10.314339…, k_max 34,
          p25 5 / p75 15 / p90 20 / p95 23 / p99 27, zero_frac 0.06332…, fill_rate 0.10744…}
A19_VALID_DIST=PASS source=manifest samples=101066 median=9.0 mean=10.31 max=34
               p25/p75/p90/p95/p99=5.0/15.0/20.0/23.0/27.0 zero_frac=0.0633 fill_rate=0.107
               expect_mismatches=0 measured_from=dataset_motion_mask
MOTION_DELIVERY=PASS samples=101066 mismatches=0 helper_checked=1863
```

**闭合。** 400 ep 库 101,066 个 exec 样本逐样本与按清单独立重算的期望全等。**回归验证**：40 ep 库上按清单重算出的期望与旧写死数字一致（中位 11 / 均值 11.46 / 最大 34 / 零起点 5.55%），证明改动没有把判据改松。

### 9.2 `T3_MOTION_CAUSAL`：单列不确定叶 + 语义收窄

**问题**（2026-09-04，HEAD `8093ebd`，`training-doc/aws-t3-open-s100/records/t3_mechanism.txt`）：判据是「padding 垫料 → loss 与 36 个 trainable 叶的梯度摘要逐位不变」。实测 loss 三档垫料（1e3 / 1 / 1e-3）**逐位相同**、36 叶里 **35 叶逐位相同**，唯一变化的是 LLM 词表 embedding 叶 `['PaliGemma']['llm']['embedder']['input_embedding']`；而该叶**在同一 obs 不改任何输入连算两次时也变** ⇒ 它在 A100 + jax 0.5.3 + 确定性档下于该诊断脚本的 `jax.jit(value_and_grad)` 里本身不确定，与 motion 垫料无关。

**闭合口径**（含审计第 4 条收窄）：

- 先**无条件**跑「同一 obs 连算 R=3 次」确定性探针（`--det-probes`，默认 3），叶级摘要不全同者进 `nondeterministic_leaves`；
- 这些叶从梯度摘要中**排除并显式打印**；`loss_bitexact`、`emb_effect`、`pos_effect` **不豁免**；
- **不确定叶里出现任何 `mem_encoder` / motion 叶立即 FAIL**（防止把真问题当噪声排掉）；
- **PASS 的含义收窄为「排除叶之外的全部叶逐位一致」**，判定行报覆盖比例 `covered=<n>/<N>`；
- 被排除叶仍记录 base–base 与 base–padding 的数值差（`excluded_diag`），**不得称「全梯度一致」**。

**重跑口径**（审计第 5 条）：与 `aws-t3-open-s100` **同库（40 ep）、同记录、原 batch**，先过初态命中校验（`t3_common_init_reference.json`），不命中即停；`JAX_PLATFORMS=cuda`（脚本默认 `cpu`，必须显式覆盖）。

**开发 smoke（2026-09-07，GPU 2,3，`t3-smoke.log`）**：

```
T3_MOTION_CAUSAL=PASS pad_bitexact=1 loss_bitexact=1 emb_effect=1 pos_effect=1 det_probes=3
                 nondeterministic_leaves=[] excluded=0 covered=36/36 excluded_diag=[]
T3_MECHANISM=PASS step=0 input_grad_ok=1 group_norms_ok=1
```

**即 2026-09-04 记录到的不确定性在本轮开发 smoke 里未复现**（`excluded=0`，36 叶全覆盖且逐位一致）。两次的 step-0 base loss 并列如下，**如实记录、不解释、不取其一**：

| 场次 | HEAD | base loss | `pad_bitexact` | 不确定叶 |
|---|---|---|---|---|
| 2026-09-04 `aws-t3-open-s100` | `8093ebd` | **0.704831** | 0（FAIL） | `['PaliGemma']['llm']['embedder']['input_embedding']` |
| 2026-09-07 开发 smoke | commitV7.1 `a8cfa17` | **0.706138** | 1（PASS） | 无（`excluded=0 covered=36/36`） |

两次 batch 索引相同（`[6556, 671, 8452, 3987, 10070, 3804, 8928, 2595]`），loss 相差 1.3e-3。**正式实跑结果**：`T3_MOTION_CAUSAL=PASS pad_bitexact=1 loss_bitexact=1 emb_effect=1 pos_effect=1 det_probes=3 nondeterministic_leaves=[] excluded=0 covered=36/36 excluded_diag=[]`、`T3_MECHANISM=PASS step=0 input_grad_ok=1 group_norms_ok=1`，base loss 0.706138（与 09-04 的 0.704831 并列，见 `docs/training-doc/tic-t3-causal-40k/result.md` 第三节）。

---

## 十、其他实测发现

本章收本轮顺带查实、但不属于任何一条阻断判据的事实。

### 10.1 训练 run 快照的 `store_path` 记的是 40 ep 库

生产 run `awsprod40k-b128-motion` 的 `history_config.resolved.yaml` 把 `motion.store_path` 记成了 **40 ep 库**路径，训练时靠环境变量 `MMEVLA_MOTION_STORE` 覆盖到 400 ep。**这是记录瑕疵，不是数据风险**——`MOTION_STORE_PATH=PASS` 用三条证据兜住：

- 不设覆盖时，`training/dataloader.py::_motion_gates` 会被 `motion_store.check_same_source` 直接 **raise**（判定行里的 `snapshot_default=raise`），不会静默用错库；
- provenance 里 motion 库与 framesamp 库绑的清单摘要相同（`same_manifest=1`）；
- `src/mme_vla_suite/policies/` 下**没有任何** `MotionStore` / `MMEVLA_MOTION_STORE` 引用（`infer_reads_store=0`）——推理侧根本不读离线表，记错的快照路径影响不到推理。

本轮**不修**这个记录瑕疵（计划第八节第 3 项）：它有 sha256 硬校验兜底，改它要重写已完成 run 的快照。

### 10.2 在线位置编码表与库表逐位相同

`FrameSampMemory.__init__` 在线用 `PosEmb3D(dim)(arange(4096), 4)` 现算一张 4096 行的位置编码表；训练侧用的是建库时落盘的 `FrameSampStore.pos_rows`。**两者在 GPU 上算出的前 586 行 sha256 相同**（586 = 400 ep 库 `store_meta.num_pos_rows` = 最长 episode 帧数）：

```
PROBE_ENV backend=gpu devices=cuda:0 pos_rows=586
  pos_table_sha=74ced98dfb55728177abeafcc5ba80c27fd668b1febec9599095a5310eec2972
  store_pos_sha=74ced98dfb55728177abeafcc5ba80c27fd668b1febec9599095a5310eec2972
  pos_table_full_sha=f8a302ff… pos_table_full_rows=4096
```

**注意 `backend=gpu` 是判据的一部分**：这张表在 CPU 上算与在 GPU 上算**不逐位**（P5 留档记过）。所以组 E 的 `EVAL_BACKEND` 把「后端是 gpu」与「两表逐位同」放在同一行里判——policy server 如果意外落到 CPU，位置编码就会静默偏离而没有任何报错。

### 10.3 `OBS_PROMPT` 在训练集 5 集 PASS：lower 恰好是恒等映射

计划把 `OBS_PROMPT` 列为**最可能真 FAIL**的一条：训练建表时把 prompt 全部转成了小写，而 `PaligemmaTokenizer` 本身**不做**小写转换（`tokenizer=…(strip,underscore,newline;lower=no)`），在线送的是 h5 `setup/task_goal` 原文。

**实测 PASS**，原因是这 5 条 episode 的 h5 原文**本身就是全小写**，建表时的 lower 是恒等映射：

```
RAW_OBS_VS_PKL=PASS … prompt_text_mismatch=0
OBS_PROMPT=PASS points=10 text_equal=10 tok_mismatches=0
  train_text='first press both buttons on the table, then pick up the container hiding the green cube,
              finally pick up another container hiding the blue cube'
  infer_text=（同上，逐字符相同）
```

**这个结论只覆盖训练集这 5 条 episode**。评估用的是 benchmark **test split**，那批 episode 的 `task_goal` 原文是否也全小写，由组 E 的 `EVAL_PROMPT` 判——它把训练侧 4 任务 pkl 的 `prompt` 过同一 tokenizer 得到 token 集合，在线 `tok_ids` 不在其中即 FAIL。两条判定必须结论一致。开发 smoke（2 集）`tok_in_trainset=8/8 text_in_trainset=8/8`，`train_distinct_prompt=26`。

### 10.4 `_drive` 与 `eval.py` 在 `C ≡ 0 (mod 16)` 时差一个决策点

既有闸门用的简化驱动 `motion_gates_online.py::_drive` 的循环条件是 `while t + 16 < T`，给出 `⌊C/16⌋+1` 个点；真实 `examples/robomme/eval.py::EpisodeEvaluator.eval_each_episode` 的推理发生在循环体开头 `count = 16j` 处，循环体开头出现过的 `count` 是 `0 .. min(C, max_steps+1)−1`（`count = C` 那一次在体尾 `break`，不再进入下一轮体首），其中 `C = T − 1 − es` 是环境步数。

**两者在 `C` 恰为 16 的倍数时差一个点**——`_drive` 会多出一个训练表里不存在的时刻。本轮五集 `C % 16 = 10 / 12 / 11 / 4 / 1`，**不触发**；`RHYTHM_EQ` 因此 `episodes=5 points=120` 三方（真 `eval.py` 控制流 / `_drive` / 脚本独立清单）全等。处置：`compare_train_infer_obs.py::expected_taus` 与 `eval_rhythm_gates.py::expected_taus` 都改用与 `eval.py` 同一公式**各自独立实现**（不互相 import，因为后者顶层会把 jax 打到 CPU），`_drive` 只留作对照臂。

### 10.5 eval 日志与 `progress.json` 的真实格式：三条与计划不符

读 `examples/robomme/eval.py` 与 `training-doc/eval-awsprod40k-b128-motion/records/` 实测，与计划假设不符的三条及处置：

| # | 计划假设 | 实际 | 处置 |
|---|---|---|---|
| 1 | eval stdout 有逐集成功 / 失败 / timeout 结果行 | **没有**。每集只有三行：`[robomme] env for task <task> episode <n> setup finished` / `task_goal: <goal>` / `exec_start_idx: <es>`；异常集另有 `Error evaluating episode <n> for task <task>: <e>` | `EVAL_EPISODE_MAP` 改用这三行 + 探针 `episode_seq` 做一一对应，并三重核对 `(task, ep)` 顺序、`exec_start_idx`、`task_goal` |
| 2 | `progress.json` 能区分 timeout 与失败 | **不能**。格式是 `{task: {"<ep>": true｜false｜"error"}}`，**timeout 与普通失败都写 `false`** | timeout **主判据取探针 infer 次数**（`== ⌈(max_steps+1)/16⌉ = 82`），交叉核取视频文件名 |
| 3 | 三分结果可从日志得到 | 唯一来源是 `save_dir/videos/<task>_ep<n>_<flag>_<goal>_<difficulty>.mp4` **文件名**（`eval.py::EpisodeEvaluator.eval_each_episode` 写、`legacy-eval/merge_eval_shards.py` 也这么读） | `EVAL_NO_RAISE` 的 error 从 eval 日志与 server 日志的 Traceback 判，timeout 交叉核视频文件名 |

**还有一条与「不续评」纪律直接相关**：续评时 `setup_log_dict` 会从 `progress.json` 读回并把 `"error"` 条目**重评覆盖**掉——所以组 E 硬性要求全新 `save_dir`、单次运行、禁止续评。

### 10.6 撞 motion 预算的后果是静默失败

`ES_BOUNDARY=PASS … first_raise_es=289 predicted=289 k_at_288=96 k_at_289=97`：demo 段长度 `es ≥ 289` 时，1300 步评估的最晚决策时刻上合法运动起点数 k = 97 > budget 96，`FrameSampMemory._prepare_motion` 直接 raise（**不做最近 N 裁剪**）。后果链：

```
_prepare_motion raise
  → policy server 侧异常经 websocket 传回
    → examples/robomme/eval.py::evaluate 的 except Exception 把该集记成 "error"
      → 续评时 setup_log_dict 从 progress.json 读回，该集被重评、error 条目被覆盖
        → 汇总 sum(log_dict[task].values()) 遇到字符串 "error" 抛 TypeError，又被外层 except 吞掉
```

**即：撞上限不会报警，只会让该集悄悄消失在成功率分母之外。** 当前四任务最长 `es = 216`（`real_max_k=92`，余量 4 窗），未触发；但这条边界离得不远，扩任务时必须先查 `es`。本轮**不给 `visible_motion_frames` 加上界**（计划第八节第 3 项），只把边界与后果链写清。

### 10.7 `MEM_SCALE_OBS` 比值 0.083，低于 A20 观察带

A20 是「运动 token 与帧 token 的数值尺度比」观察项，band `[0.3, 3.0]`。本轮实测（取数点在 `take_along_axis` 之前的并列序上）：

```
MEM_SCALE_OBS points=8 ratio_mean=0.08274 min=0.07838 max=0.09423 a20_band=[0.3,3.0] in_band=0/8
```

sidecar 档 `points=4 ratio_mean=0.08261 in_band=0/4`。**8/8 全部越界。** 对照：随机初始化时（`training-doc/motion-t3-open/launch.md` 与 `motion-t3-closed/launch.md`）该比值是 **0.166**，用户当时已拍板把 A20 降为**观察项、不改模型**。

**即：40k 步训练之后，运动 token 的相对尺度从 0.166 进一步降到 0.083（约再降一半）。** 这是本轮新出的观察事实，**本文不作效果解读**——它可以是「模型学会了少用运动路」，也可以只是两层线性投影的权重衰减，两种解释本轮都没有证据分辨。若以后要立项查 (a)「motion token 本身有没有用」，这个数字是起点之一。

### 10.8 `ACT_CKPT_DTYPE`：盘上原精度 vs 生产 bf16 加载

`CKPT_DTYPE_PROFILE` 显示 checkpoint 盘上是 **f32 36 叶 / bf16 23 叶**（`img` 子树全 bf16、可训练叶全 f32），而生产 `create_trained_policy` 以 `dtype=bf16` 加载**全部**叶。这一步 cast 对动作的影响：

```
ACT_CKPT_DTYPE points=1 ckpt_dtypes=['bfloat16','float32'] rms_norm=0.0008164
                max_abs_norm=0.004301 act_std_mean=0.2047
```

即归一化动作 RMS **8.2e-4**，是 `norm_stats.actions.std`（0.2047）的 0.40%，与第六章的 SigLIP 动作差（`rms_norm` 4.5e-4）同量级。**观察项，不阻断**；它量化的是「用盘上原精度推理」与「生产 bf16 推理」的差，不是训练 / 推理不一致。

> 本项按审计第 8 条从原计划的「EMA vs 训练 params 对比」改来——后者做不了：`checkpoints.py::_split_params` 在有 EMA 时只把 EMA 存进 `params`，实测 39999 / 35000 / 10000 / 5000 目录都只有 `_CHECKPOINT_METADATA` / `assets` / `params`。

---

## 十一、明确不做与以后可立项

### 11.1 本轮明确不做（计划第八节）

1. **不做 EMA vs 训练 params 对比**——做不了，原因见 10.8 末尾的引用块；改做 `CKPT_DTYPE_PROFILE` + `ACT_CKPT_DTYPE`。
2. **不合并 legacy-eval 回主线**、不给主线 `src/mme_vla_suite/policies/` 与 `examples/robomme/` 加埋点——闭环组探针全部靠**实例属性遮蔽**（`serve_policy_probe.py` 遮蔽 `policy.reset` / `policy._prepare_history` / `policy._input_transform`）。8 卡 200 集扩展需拷回 5 个文件并单独 commit，本轮不做。
3. **不修 SigLIP bf16 差异**（第六章）、**不修快照 `store_path` 记录瑕疵**（10.1，有 sha256 硬校验）、**不给 `visible_motion_frames` 加上界**（10.6）、不做预算消融、不重跑三 seed、不在 400 ep 库重跑 D2/D3、不改 `compare_siglip_replay.py`。
4. **不删任何 docs 目录、不改 `AGENTS.md`、不改冻结的根计划文件正文。**
5. **不自行放宽 `VT_FULL_VS_CACHED` 阈值**（第七章）——属放宽判据，交用户裁决。
6. **不应用两份 Vulkan patch**（8.7）——落点交用户裁决。

### 11.2 以后可立项

| 项 | 内容 | 代价 / 前提 |
|---|---|---|
| 闭合帧特征编码器差异 | 第 6.6 节两条路择一 | 路 1：在线延迟线性涨；路 2：重建 400 ep 库并与官方 framesamp 表脱钩 |
| `VT_FULL_VS_CACHED` 阈值重定 | 按 7.2 / 7.3 的实测与 f32 诊断重新定标，或改为「f32 档逐位 + bf16 档观察」双档判据 | 需用户裁决；重定后要说明依据，不能只是把数字抬到实测之上 |
| 应用 Vulkan 修法 | 8.7 的 patch B（钉住 RenderSystem）落 legacy-eval，解除「每进程 ≤27 集」红线 | 需在目标集数上实测通过；顺带把 `make_env` 从 9.87 s 压到 0.90 s |
| 扩大 TIC 样本 | 从 5 集 / 24 集扩到多任务多集；8 卡 200 集闭环探针 | 需拷回 5 个文件到主线并正式 commit |
| 查 (a)「motion token 有没有用」 | 预算消融、更长训练、多 seed | 不属 TIC 范围；10.7 的尺度比是起点之一 |

---

## 十二、溯源

### 12.1 run 留档

| run | 组 | 内容 | 状态 |
|---|---|---|---|
| [`training-doc/tic-l0-rhythm-40k/`](training-doc/tic-l0-rhythm-40k/) | A | 第 0 关 + 评估真节奏 + 400 ep A19 | **已收工，三段全 PASS，`EXIT_CODE=0`** |
| [`training-doc/tic-obs-model-40k/`](training-doc/tic-obs-model-40k/) | B | 第 1–5 关（`--motion store`） | 12/13 阻断 PASS；`VT_FULL_VS_CACHED` 待裁决 |
| [`training-doc/tic-t3-causal-40k/`](training-doc/tic-t3-causal-40k/) | D | `T3_MOTION_CAUSAL` 闭合重跑 | PASS（`covered=36/36 excluded=0`） |
| [`training-doc/tic-vulkan-makeenv/`](training-doc/tic-vulkan-makeenv/) | F | Vulkan 复现与修法验证 | PASS（复现 28；两修法 35 轮不崩） |
| `docs/training-doc/tic-sidecar-40k/` | C | 第 1–5 关（`--motion sidecar`） | 第二波，尚未落盘 |
| `docs/training-doc/tic-eval-probe-40k/` | E | 第 6 关：24 集闭环探针 | 第三波，尚未落盘 |

**依据的既有留档**：[`training-doc/siglip-ab-replay-40k/`](training-doc/siglip-ab-replay-40k/)（第六章全部数字）、[`training-doc/aws-t3-open-s100/`](training-doc/aws-t3-open-s100/)（9.2 的 09-04 记录）、[`training-doc/eval-official-framesamp-context/`](training-doc/eval-official-framesamp-context/)（8.1 的首次崩溃记录）、[`training-doc/awsprod40k-b128-motion/`](training-doc/awsprod40k-b128-motion/)（生产 run 与 `motion_provenance.json`）、[`training-doc/aws-p5-online/`](training-doc/aws-p5-online/)（在线现编与离线表 772 窗逐位）、[`training-doc/eval-3seed-context-vs-motion/`](training-doc/eval-3seed-context-vs-motion/)（第二章的三 seed 数字，**环境 A 历史**）。

### 12.2 commit

| commit | 内容 |
|---|---|
| `892f73e` | commitV7.0：`docs/` 环境 A 历史留档迁入 `docs/archive/`，同步全部路径引用与搬迁文件内链接 |
| `a8cfa17` | commitV7.1：训练/推理一致性对拍工具（配置与库指纹同源、eval 真节奏闸门、obs/模型内层/整段 vs 缓存/动作对拍、闭环探针、Vulkan 复现；A19 期望按库重算且实测取真实交付、`T3_MOTION_CAUSAL` 单列不确定叶并收窄语义） |
| `9b3b95f` | tic 第一波四个 run 的 `launch.md` 预提交（正式 run 从此 clean HEAD 起跑） |

### 12.3 脚本

| 脚本 | 关 | 说明 |
|---|---|---|
| `scripts/training/g0/check_config_provenance.py` | 第 0 关 | CPU；5 条判定行 + `TIC_L0`。骨架仿 `check_baseline_env.py` |
| `scripts/training/tests/eval_rhythm_gates.py` | 节奏层 | CPU + stub；挂 `examples/robomme/utils.py::{EpisodeState, pack_buffer}` 按 `eval.py` 逐语句照抄控制流；4 条判定行 + `TIC_RHYTHM` |
| `scripts/training/g0/compare_train_infer_obs.py` | 第 1–5 关 | GPU；三臂 S/A/B；`TIC_OBS_MODEL`（`--motion store`）/ `TIC_SIDECAR`（`--motion sidecar`）。派生自 `compare_siglip_replay.py` |
| `scripts/training/g0/serve_policy_probe.py` | 第 6 关 | 探针版 policy server，实例属性遮蔽三处，逐次 infer 落 jsonl（记**完整数组**而非只记 sha）；启动打 `PROBE_ENV` |
| `scripts/training/g0/summarize_eval_probe.py` | 第 6 关 | 读 jsonl + `progress.json` + eval 日志 + server 日志；7 条阻断 + 1 条观察 + `TIC_EVAL`；次序表与 k 公式在本文件内**独立重写**，不 import `shared/sampling` |
| `scripts/training/tests/motion_gates_model.py` | A19 / T3 | 本轮改 `cmd_m1`（`--expect-source`、实测取 `motion_mask.sum()`）与 `cmd_t3mechanism`（`--det-probes`、`nondeterministic_leaves`、`covered=<n>/<N>`） |
| `scripts/training/legacy-eval/probe_vulkan_makeenv.py` | 组 F | 纯排查工具；循环 `make_env`/`close_env`，逐轮记 fd / vk_maps / RSS / 显存 / sapien 对象；`--pin-renderer`、`--gc --renderer-release`、`--vk-loader-debug` 等开关 |

### 12.4 原始证据落点

- 开发跑判定行：`v1-store/reports/tic-dev/`（`obsmodel-dev.log`、`obsmodel-sidecar-dev.log`、`obsmodel-f32-dev.log`、`obsmodel-f32hi-dev.log`、`t3-smoke.log`、`probe-server.log`、`eval_probe.jsonl`、`eval_probe_summary.json`）；
- Vulkan 全部证据：`v1-store/reports/tic-vulkan/`（`rootcause.log`、`tls_scan.log`、`baseline-noreset.log`、`pin-renderer.log`、`tls8192.log`、`gc-rendrelease.log`、`vk_limit_nodestroy.log`、`vk_device_{destroy,nodestroy}.log`、`probe-*.json`、`vk_instance_limit.c` / `vk_device_limit.c` / `dlopen_icd_limit.c`、`fix-a-glibc-tunables.patch` / `fix-b-pin-renderer.patch`）；
- 正式跑判定行：各 run 的 `records/` 与 `result.md`；
- 生产口径输入：checkpoint `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`、库 `v1-store/datasets/4task-motion-400ep`、原始 h5 `/scratch/hongze/robomme_data_h5/`（介质：AWS 本地 NVMe RAID `/dev/md0`）。

### 12.5 过程档案（冲突以本文为准）

- 计划文件：`~/.claude-personal/plans/hashed-petting-garden.md`（TIC + `docs/` 重组，Codex 审计修订版，`AUDIT_BASE=9d6c846`）；
- 兄弟正本：[`motion-memory.md`](motion-memory.md)（motion 接入现行正本）、[`dataloader-restructure.md`](dataloader-restructure.md)（dataloader 四阶段现行正本）。
