# v6 计划：motion memory 接入——framesample 记忆双路化（帧路 + 运动路）

> **本文件是 v6 工作的权威计划**（2026-09-01 第一版，方案已与用户三轮澄清对齐，待批准实施）。
> **锚点**：分支 `v1-dataloader-Restructure`，HEAD = `95fc8e3`（工作区 clean）。
> **commit 编号**：代码切片 **commitV6.x**；本文件本身按 `docs:` 提交。
> **外部依赖仓库**：MotionJEPA 单副本 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA`
> （HEAD 与 checkpoint 选型见第二部分〇节，**起工前须锚定并写死**）。
> **本计划只规划、不实施**：所有 S0–S4 步骤须逐步获批后动手。

---

# 第一部分（给人看）

## 一、Context（为什么做这件事）

`AGENTS.md` 的项目 scope 写明「仓库总体目标：修改 MME-VLA 的 `perceptual-framesamp-context`，
并在后续阶段接入 MotionJEPA motion token」。v1–v5 已把 dataloader 与训练入口收敛到 packed
framesamp 单一路径（v5.2 收官，60k 全量 run 在跑），现在做的正是那个「后续阶段」：把
MotionJEPA 的 Wan latent 运动编码器产出的 motion token 接进 VLA 的感知记忆。

当前记忆只有**一路**：`even_sampling_indices(step, 32)` 在 `[0, t]` 上均匀选 32 个历史帧，
每帧给 16 个 4×4 池化的 SigLIP token，共 512 个 memory token。这一路是**静态外观**的——
每个 token 描述「那一帧长什么样」，不描述「那一刻在发生什么运动」。v6 补上第二路。

## 二、用户拍板汇总（2026-09-01 会话，三轮澄清）

| 议题 | 用户原话 / 拍板 |
|---|---|
| 特征来源 | 「从 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA` 来的」——Wan VAE（离线冻结）→ `WanLatentMotionEncoder` 两级链路 |
| 编码粒度 | 「对每个 action chunk 的起点都编码一次，注意长度不是 chunk 而是 33 frame」 |
| **接入形态** | 「**我的意思是作为 memory 的一部分**，33 个连续 raw 帧作为一个窗口，每个 chunk 作为起点」——**不是插单个 token，而是记忆序列的第二路** |
| **训练/推理分工** | 「**训练的时候预先抽取，测试的时候同步推理**」——训练读离线表；在线评估每步现编 |
| 本轮范围 | 「暂时不管 dataloader，只看 model 怎么走」——本计划以 model 侧为主体，数据侧只定契约不展开实现 |
| 窗口方向 / 编码器承载 / 注入点 | 三问均答「无偏好」，授权按推荐定稿；本计划据实测数据定死（见五节），用户可随时推翻 |

## 三、方案总览

**一句话**：memory 从「32 帧 × 16 image token」变成「32 帧 × 16 image token」**并列**
「32 帧 × 1 motion token」，prefix 记忆区由 512 增至 544；motion token 是该帧**后视 33 帧窗口**
（`[f−32, f]`）的运动概括，训练读离线表、在线每步增量现编。

四个已定死的口径，各自的依据在第五节：

1. **窗口挂后视端**：`motion[f]` = 窗口 `[f−32, f]`，而不是前视的 `[f, f+32]`。
2. **并列拼接**：运动路走独立投影，与帧路 concat 成 544，而不是交错或特征维融合。
3. **缺失用 missing embedding 兜**，不用 `input_mask` 屏蔽。
4. **encoder 冻结、离线抽表**（承载方案 A），JAX 移植微调留到 S4 消融。

## 四、两库对齐的实测底座（方案能成立的地基）

整个方案建立在「MotionJEPA 已抽好的 latent 能被 MME-VLA 的 `(g, step)` 直接查表命中」之上。
**这一条已实测确认，不需要重抽 261 GB latent。**

### 4.1 同源性核对（2026-09-01 全量跑过）

| 项 | 实测结果 |
|---|---|
| MME-VLA 侧 | `v1-store/episode_manifest.json`：1600 episode（4 任务 × 400 ep），483,291 timesteps，**395,289 exec 样本**；`raw_dir = /nfs/turbo/.../robomme_data_h5_v2_4env400ep` |
| MotionJEPA 侧 | `dataset-4env-v8/dataset-token/wan_chunk_latents/metadata.json`：2400 条目 = ButtonUnmask/ButtonUnmaskSwap 各 400 个 `_exec`（无 demo）+ VideoUnmask/VideoUnmaskSwap 各 400 个 `_exec` + 400 个 `_demo`；`source_h5 = /nfs/turbo/.../motionjepa-v8-gl/data-raw/...` |
| **demo 段帧数** | **1600 个 episode 全部差 0** —— 即 `MME.exec_start_idx == MJ.demo.frames`，帧号 1:1 同源 |
| **exec 段帧数** | MotionJEPA 恒短 4~12 帧（分布 {4:231, 5:537, 6:237, 7:127, 8:142, 9:180, 10:99, 11:41, 12:6}），成因是 `build_data_raw_from_h5.py` 的 `EXEC_TRUNCATE_TAIL = 2`（首个 `is_completed=True` 后保留一帧即截断） |
| chunk 密度 | `num_chunks = frames − 32`，**stride = 1**——每个段内帧都是一个合法窗口起点 |
| 体量 | latent 261 GB（同一 turbo 文件系统，`df` 余 6.0 TB）；exec chunk 333,900 + demo chunk 62,402 = **396,302** |

**结论**：两库是同一批原始数据的两个派生副本，demo 前缀长度逐 episode 完全一致，exec 起点一致，
只是 MotionJEPA 在 exec 尾部多截了 4~12 帧。帧号可以直接换算，**零重抽**。

### 4.2 索引映射（后视口径）

`build_exec_lookup` 给出的 `_step_of[idx] = exec_start_idx + k` 是**全 timestep 域帧号**（含 demo
前缀），`even_sampling_indices` 选出的 32 个帧号 `f` 也在同一域。查表：

```
采样帧 f ──┬─ f <  exec_start_idx[g] → demo 段：段内帧号 u = f       → <Task>_ep<j>_demo.bin
           └─ f ≥  exec_start_idx[g] → exec 段：段内帧号 u = f − es  → <Task>_ep<j>_exec.bin
chunk 索引 c = u − 32          合法条件：0 ≤ c < num_chunks[该段]
```

⚠ **窗口不跨 demo/exec 边界**——latent 是分段抽的，exec 段开头 32 帧的窗口会跨界，一律判缺失。

## 五、三个关键口径的原理与实测依据

### 5.1 为什么窗口挂后视端（`[f−32, f]`）而不是前视端（`[f, f+32]`）

MotionJEPA 的 chunk 定义是「锚点帧 t + 未来 32 帧」（`extract_wan_chunk_latents_all.py` 的
`WINDOW = 4*K+1 = 33`）。把 motion token 挂在窗口的哪一端，是查表时 `c = u` 还是 `c = u − 32`
的一个常数差——**物理表完全相同，抽一次即可**。但两种挂法的可用性天差地别。

严格因果（窗口必须全部落在已观测历史内）+ 段内约束下，抽样 20,000 个样本 / 619,797 个采样帧实测：

| 挂载口径 | 采样帧合法占比 | 逐样本覆盖均值 | 逐样本覆盖中位 | **最新采样帧（=当前时刻）有 motion** |
|---|---|---|---|---|
| 前视 `motion[f]` = `[f, f+32]` | 69.33% | 0.671 | 0.750 | **0.00%** |
| **后视 `motion[f]` = `[f−32, f]`** | 68.59% | 0.664 | 0.750 | **84.59%** |

总覆盖率几乎相同（差 0.74 个百分点），但前视口径下「当前时刻在发生什么运动」**恒为空**——因为
`[t, t+32]` 永远在未来。对一个要输出下 20 步动作的策略来说，那恰恰是最该有的一个 token。
**后视用 0.74 个百分点换回了它，定后视。**

后视还让在线实现退化成一行：每来一帧 `t`，编码窗口 `[t−32, t]`，存 `motion[t]`，**零滞后、
每步只增量一次**；查表时 `even_sampling` 选出的 `f_i` 直接查 `motion[f_i]`，训练侧离线表与在线侧
共用同一索引语义。前视口径则在线永远滞后 32 帧。

### 5.2 为什么并列拼接而不是交错或特征维融合

三种把运动路并进 memory 的方式：

| 方式 | 序列形态 | 代价 |
|---|---|---|
| **A. 并列拼接（选定）** | `[512 image mem] + [32 motion mem]` = 544 | 运动路完全独立：独立投影、独立 mask、`motion.enabled=false` 一键退回逐位等价的旧链路 |
| B. 交错 | 每帧 `16 image + 1 motion` = 17，共 544 | 时间局部性更好，但要动 `PerceptualMemory.__call__` 的 `assert static_image_emb.shape[1] == self.config.budget` 和一串 reshape，且开关不能干净退出 |
| C. 特征维融合 | 每个 image token concat 同帧 motion 向量 | `encoder_static` 从 `Linear(2816→2048)` 拓成 `Linear(3584→2048)`，同一 motion 信号冗余 16 份；注入强度最大但最不可控 |

选 A 的决定性理由是**可退性**：`AGENTS.md` 第 18 条要求「新旧链路一致性」有显式判据，A 能给出
最强的那一条——关闭态下 `embed_prefix` 的输出张量与当前 HEAD **逐位相同**。B 和 C 都做不到
（它们改动了既有 token 的构成或投影形状）。

32 : 512 的信号比也够——不像「插单个 token」那样有被 512 个 memory token 淹没的风险。

### 5.3 缺失 31.4% 怎么兜——为什么用 missing embedding 而不是屏蔽

缺失集中在两处：**每段开头 32 帧**（历史不足一个窗口）、**exec 段尾部 4~12 帧**
（`EXEC_TRUNCATE_TAIL` 截断）。实测 **6.49% 的样本 32 个采样帧全缺**（`step < 32` 的 episode 开头）。

**不能用 `input_mask=False` 屏蔽**：`embed_prefix` 之后 `positions = jnp.cumsum(input_mask, axis=1) - 1`，
一旦某个样本屏蔽掉 k 个 motion 位置，它后面所有 token 的 RoPE 位置会整体前移 k——同一 batch 内
不同样本的 image / prompt / action 落在不同位置上。现有 `static_mask` 已经有这个行为（padding 的
memory token），那是既有语义；但 motion 的缺失率高得多（31.4% vs framesamp 只在 `step < 32` 时
padding），会把位置抖动放大成系统性噪声。

**定：motion 段恒占 32 个位置、`input_mask` 恒 True**，缺失位用一个可学习的
`missing_motion_emb: nnx.Param(768)` 顶替，由 `motion_valid` 经 `jnp.where` 选择。位置口径恒定，
且模型拿到「此处无运动信息」的明确信号——而缺失恰好集中在 episode 开头与任务完成阶段，
这个信号本身携带语义。

替代方案「剔除缺失样本」已否决：会系统性删掉 6.49% 的 episode 开头样本，训练分布偏移。

## 六、链路图（`AGENTS.md` 第 18 条）

```
【改动前】memory 单路
even_sampling_indices(step, 32) → 32 个全 timestep 域帧号 f_i
  ├─ image_emb_4x4  (32,16,2048) bf16 ─┐
  ├─ pos_emb_4x4    (32,16,768)  f32  ─┼→ reshape (512,·) → FeatureEncoder.encode_perceptual_memory
  └─ state_emb      (32,8) repeat×16  ─┘    concat[img 2048 ⊕ silu(pos_proj: 768→768)]  (use_state_emb=False)
                                            → encoder_static  Linear(2816→2048)
                                            → mem tokens (b, 512, 2048)   ar=F  na=F
prefix: ┌── mem 512 ──┬── img 2×256 ──┬── prompt ≤64 ──┐
        │  ar=F na=F  │  ar=T/F na=T  │   ar=F na=F    │
suffix: 20 个 action token（pi05=True，无 state token；状态/时间走 adarms_cond）


【改动后】memory 双路并列（运动路为新增，帧路一字不动）
even_sampling_indices(step, 32) → 32 个帧号 f_i
  ├─ 路1 帧路（逐位不变）───────────────────────────────→ mem tokens    (b, 512, 2048)
  │
  └─ 路2 运动路 ★新增
       [离线 · 训练]  motion_token.f32.bin  seek((row_base[g,seg] + u − 32) × 3072)
       [在线 · 评估]  raw frames [t−32, t] → Wan VAE(冻结) → (9,16,32,32) → encoder(冻结)
         │
         ├─ motion_emb   (32, 768) f32       每帧 1 个，后视窗口 [f−32, f] 的运动概括
         └─ motion_valid (32,) bool
              → jnp.where(valid[:,None], motion_emb, missing_motion_emb)      ← 兜 31.4% 缺失
              → concat[ motion 768 ⊕ silu(motion_pos_proj(pos_f: 768→768)) ]  pos_f = 该帧 16 个 4×4 pos 的均值
              → motion_encoder_static  nnx.Linear(1536→2048, kernel_init=normal(0.02))
              → motion tokens (b, 32, 2048)   ar=F  na=F

prefix: ┌── mem 512 ──┬── motion 32 ★──┬── img 2×256 ──┬── prompt ≤64 ──┐
        │  ar=F na=F  │   ar=F na=F    │  ar=T/F na=T  │   ar=F na=F    │
        └──────── cumsum(na)==0 的「记忆区」：image 看不到，prompt / action 看得到 ────┘
suffix: 不变
```

**motion 段为什么放在 img 之前**：`make_attn_mask` 的
`mask_not_attend = (na[k] | na[q]) & (cumsum(na) <= 0)`，第一个 `na=True` 的位置是 image token。
放在 img 之前 ⇒ motion 落入记忆区，沿用原设计「image token 不 attend memory」对预训练 VLM
视觉-语言对齐的保护；放在 img 之后则 motion 与所有 token 双向可见——那是另一种语义，
列为 S4 消融项，不做默认。

**两级边界**：Wan VAE 与 `WanLatentMotionEncoder` 都在训练环外、无梯度回传；训练环内只有
`motion_pos_proj` 与 `motion_encoder_static` 两个新投影 + `missing_motion_emb` 一个新参数。

## 七、最大风险：在线同步推理的延迟账

用户已定「测试的时候同步推理」，这条的可行性取决于一个硬数字。

MotionJEPA v8 全量抽取的**实测吞吐是 0.635 chunk/s**（单 A40，fp32、**关 TF32**、窗口 batch 恒 1；
`docs/dataset-build-doc/v8-400ep-full/README.md` 记 396,302 chunk ÷ 8 分片，sacct 实测 Elapsed
22:21:51–22:56:48，全部 `COMPLETED 0:0`）。折算 **≈1.57 秒/窗口**。

照抽取口径原样搬到在线，**每个推理 step 会多出约 1.57 秒**。`MME_VLA_Policy.infer` 现有
`infer_time_ms` 是几十到几百毫秒量级——这是 5~20 倍减速。motion encoder 本身可忽略
（69.8M 参数、序列长 9）；瓶颈全在 Wan VAE 编 33×256×256。

三个缓解手段，按推荐顺序：

1. **在线放开数值约束**。抽取时关 TF32、batch=1 是为了 finalize 语义 oracle 能逐位复现
   （`extract_wan_chunk_latents_all.py` 头部注释写明这不是可选优化）；**在线推理不需要这个保证**。
   开 TF32 + bf16 预计快 3~5 倍（→ 0.3~0.5 s/step）。代价是在线特征与训练用的 fp32 离线表存在
   数值漂移，**上线前必须实测漂移量**（同一窗口两种口径各编一次，比余弦相似度与 L2 相对误差）。
2. **降低更新频率**。motion 是 33 帧窗口的概括，相邻步高度冗余；每 4 或 8 步编一次、查表最近邻，
   线性省 4~8 倍。代价是最新 motion 最多滞后 k−1 步。
3. 两者叠加。

**这条是 S0，必须在 S1 抽表之前先做**——如果在线达不到可接受的延迟，「测试时同步推理」的前提
不成立，整个方案要回退到 motion predictor 路线（训练时用真值 motion、推理时用一个从当前观测
预测 motion 的小模型），那是完全不同的形态。

## 八、五步走

| 阶段 | 内容 | 判据 |
|---|---|---|
| **S0 延迟先验** | 单窗口在线编码基准：fp32/关TF32 vs TF32+bf16，测 ms/窗口与两口径数值漂移 | 拿到可接受的 ms/step；漂移余弦 ≥ 阈值（阈值起工前拍板） |
| **S1 特征就位** | 冻结 encoder 离线抽 396,302 × 768 f32 表 → `v1-store/datasets/4task-gl-motion/`，沿用 framesamp 的 packed→verified 两阶段 + 逐行 digest | 200 条在线跑 encoder vs 表逐位相等；500 样本索引映射对拍；覆盖率账目与 396,302 对上 |
| **S2 model 接线** | 双路 memory + `motion.enabled` 开关 + missing embedding（本计划主体） | 关闭态与当前 HEAD **逐位等价**；开启态 smoke 跑通；`‖motion_tok‖ / ‖mem_tok‖` 同量级 |
| **S3 在线接线** | `FrameSampMemory` 增量编码支路 + Wan VAE 常驻 | 在线/离线同一 `(g, f)` 特征一致（阈值同 S0） |
| **S4 消融** | ① 并列 vs 特征维融合 ② 叠加 adaRMS 调制 ③ motion 段放 img 之后 ④ 更新频率 k ⑤ 冻结 vs JAX 移植微调 | 训练曲线 + 在线成功率 |

S1 与 S3 都属「预计超过 5 分钟的全量数据构建 / 评估」，按 `AGENTS.md` 第 12、17 条从 clean HEAD
起跑并留档（`docs/dataset-build-doc/4task-gl-motion/` 与 `docs/training-doc/<run_name>/`）。

## 九、影响面结论

- **训练语义**：`motion.enabled=false` 时零影响（逐位等价，S2 判据）；`true` 时 prefix 记忆区
  512 → 544，其后所有 token 的 RoPE 位置整体右移 32。
- **参数量**：新增 `motion_pos_proj`(768×768=589,824) + `motion_encoder_static`(1536×2048=3,145,728)
  + `missing_motion_emb`(768) = **3,736,320 ≈ 3.74 M**，相对 pi05 主干可忽略。
- **冻结**：三个新参数挂在 `HistoryPi0.mem_encoder`（`PerceptualMemory`）下，路径形如
  `mem_encoder.motion_encoder_static`。当前 `HistoryPi0Config.get_freeze_filter` 返回
  `PathRegex(".*img.*")`（`paligemma_variant="gemma_2b"`，无 lora），该路径不匹配 → **默认可训练**。
  若日后启用 lora，filters 变成 `Any(All(".*llm.*", Not(".*lora.*"), Not(".*mem.*")), ".*img.*")`，
  路径含 `mem` 恰好被 `Not(".*mem.*")` 排除出冻结集 → **仍可训练**。两种情形都安全，无需为
  freeze 改名。
- **数据**：新增 1.13 GiB 离线表（`v1-store/` 内，不进 git，符合第 14 条）；不动 261 GB latent。
- **在线评估**：多背一个 Wan VAE（PyTorch）常驻，延迟见七节。
- **不影响**：正在跑的 `v1-prod-60k` 全量 run（本计划一行代码都还没动）。

---

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

1. **本计划只规划不实施**。S0–S4 每一步动手前须单独获批（`AGENTS.md` 第 2 条）。
2. **外部仓库锚定（起工第一件事）**：记录并写死
   - MotionJEPA 仓库 HEAD：`git -C /nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA rev-parse HEAD`
   - **checkpoint 选型待用户拍板**：候选 `runs/wan-v8-filter10-72ep-a` / `runs/wan-v8-armw01-72ep-b` /
     `runs/wan-v8-filter2-72ep-b`，各含 `checkpoint_epoch_*.pt`（单个 ~1.1 GB）。
     取 `ckpt["encoder"]`（EMA 权重）还是 `ckpt["encoder_live"]`（live 权重）**一并拍板**；
     `scripts/train.py` 的保存逻辑为 `has_live_weights` 时两者都存。
   - 选定后把 run 名 + epoch + `sha256` 写进 `store_meta.json` 的 `provenance` 块。
3. **归一化常数不得二次读取**。`WanLatentMotionEncoder` 继承 `LatentAffineMixin`，
   `latents_mean` / `latents_std` 是 **persistent buffer、随 checkpoint 存档**；抽取脚本必须
   `load_state_dict(..., strict=True)` 让 buffer 从 ckpt 填充，**禁止**再调
   `load_wan_latent_stats(vae_id)`（MotionJEPA 仓库规定它是全仓库唯一读取点，二次读取会绕过
   「strict load 失败即在第一次前向炸」的保护——`normalize()` 首行的 finite 断言就是为此存在）。
4. **抽取口径必须与 MotionJEPA 一致**：fp32、`torch.backends.cuda.matmul.allow_tf32 = False`、
   窗口 batch 恒 1。这三条是 finalize 语义 oracle 逐位可复现的前提，抽表阶段不得放开
   （在线阶段可放开，见 S0）。
5. **新参数必须在所有现有模块之后创建**。`HistoryPi0.__init__` 里 `rngs` 的消耗序决定既有模块的
   初始化值（`datastore/README.md` 明记 `use_pos_emb`/`use_state_emb` 影响
   「`FeatureEncoder` 的参数树与 RNG 消耗序，禁改」）。运动路的三个新参数一律在
   `PerceptualMemory.__init__` 现有 `feature_encoder` **之后**建，否则 `motion.enabled=true` 会
   连带改变帧路的初始化值，S2 的等价判据失去意义。
6. **禁止 `git clean -x` / `-X`**（`AGENTS.md` 第 19 条附则），会删掉 `v1-store/` 全部产物。

## 一、离线 motion 表格式契约（S1）

新建独立 store，**不混进** `v1-store/datasets/4task-gl-framesamp/`（帧路的行号公式 `row_of()` 与
motion 的段内公式不同，混放会让两套索引互相污染）：

```
v1-store/datasets/4task-gl-motion/
├── meta/store_meta.json          唯一契约，两阶段写：pack→"packed"、verify→"verified"
├── meta/motion_index.json        段基址表（唯一身份来源）
├── meta/row_digests.blake2b.bin  逐行 blake2b-128（verify 产出）
├── meta/pack_progress.jsonl      断点续跑记录
└── motion_token.f32.bin          (396302, 768) f32 裸字节 = 1.13 GiB
```

布局常量（照 `datastore/framesamp_store.py` 的 `LAYOUT` 体例，新增到同包内新模块
`motion_store.py`，**不改 `framesamp_store.py`**）：

```python
LAYOUT = "motion-768-v1"
META_SCHEMA = 1
MOTION_KEY = "motion_token"
MOTION_ROW_SHAPE = (768,)
MOTION_DTYPE = np.float32
MOTION_ROW_BYTES = 768 * 4            # 3,072
MOTION_TABLE_RELPATH = "motion_token.f32.bin"
WINDOW_FRAMES = 33                    # 与 MotionJEPA 的 WINDOW 同值，写死并在 verify 时核对
WINDOW_ANCHOR = "trailing"            # 后视：motion[f] = 窗口 [f-32, f]
```

⚠ 沿用 framesamp 的**禁 `.npy` 容器**定论（`np.save` 对 ml_dtypes bf16 写 `V2` descr），一律裸
`.bin` + meta 声明 dtype。motion 是 f32、不涉 bf16，但保持同一体例。

**行序（定序规则，写进 `store_meta.json`）**：按 `episode_manifest.json` 的
`canonical_order` 遍历 1600 个 episode，每个 episode 内先 `demo` 段后 `exec` 段，段内按 chunk
索引 `c` 升序。`motion_index.json` 记录：

```json
{"schema": 1,
 "entries": [{"g": 0, "task": "ButtonUnmask", "raw_ep_idx": 0,
              "demo": {"row_base": null, "num_chunks": 0},
              "exec": {"row_base": 0, "num_chunks": 227}}, ...],
 "totals": {"rows": 396302, "exec_chunks": 333900, "demo_chunks": 62402},
 "mj_metadata_sha256": "<MotionJEPA metadata.json 的 sha256>"}
```

**查表**（`c = u − 32`，后视）：
```
row = entries[g][seg].row_base + (u - 32)      seg = "demo" if f < exec_start_idx[g] else "exec"
                                               u   = f      if seg == "demo" else f - exec_start_idx[g]
valid = 0 <= (u - 32) < entries[g][seg].num_chunks
```

**读取实现**：照抄 `FrameSampStore` 的模式——构造即开 fd（`O_RDONLY|O_CLOEXEC`）、
`posix_fadvise(WILLNEED)` + 连续行游程合并 `os.preadv` 直读预分配数组、短读循环补齐、
`__reduce__` 直接 raise 禁 pickle、记 `owner_pid`。motion 表只有 1.13 GiB，**也可整表
`np.fromfile` 读入进程内**（每 worker 常驻 1.13 GiB × workers 数——8 workers 即 9 GB，
需按当次 `num_workers` 权衡；默认走 pread，整表读入作为可选加速项）。

## 二、model 侧逐文件改动清单（S2）

按 `AGENTS.md` 第 9 条，以下全部用函数/类/配置键作锚点，不写行号。

### 2.1 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml`

新增 `motion` 节（**只加节，不动任何既有键**）：

```yaml
motion:
  enabled: false            # 总开关；false 时链路逐位等价于当前 HEAD
  dim: 768                  # = MotionJEPA config 的 motion.dim
  num_tokens_per_frame: 1   # = MotionJEPA config 的 motion.num_tokens（M=1）
  window_frames: 33
  window_anchor: trailing   # 后视；改成 leading 即前视（S4 消融用）
  store_path: v1-store/datasets/4task-gl-motion
  source_run: ???           # MotionJEPA run 名 + epoch，S1 锚定后填
  pos_source: frame_mean    # motion token 的 pos：该帧 16 个 4×4 pos 的均值
```

⚠ 核对过：`scripts/training/g0/bench_train_steps.py` 与
`scripts/training/tests/dump_fixture_samples.py` 的 `_EXPECTED_HISTORY_CONFIG` 只断言**文件名**
（`"perceptual-framesamp-context.yaml"`），不逐字校验内容——加节不会触发它们。
真正逐键校验的是 `FrameSampDataset.__init__` 的 `_req(...)` 形制断言序列，那些只查既有键的值，
同样不会被加节打断；但**必须新增对 `motion.*` 的同款 `_req` 断言**（显式 `raise`，禁 `assert`
——`PYTHONOPTIMIZE=1` 会剥离 `assert`，见该文件头部注释的 R6）。

### 2.2 `src/mme_vla_suite/models/integration/history_observation.py`

`HistAugObservation` 新增两字段（`@at.typecheck` + `@struct.dataclass` 下必须同步四处）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `motion_emb` | `at.Float[at.Array, "b l4 d4"] \| None` | `l4 = 32`，`d4 = 768` |
| `motion_valid` | `at.Bool[at.Array, "b l4"] \| None` | 逐帧有效位 |

同步改动：`from_dict`（`data.get(..., None)`）、`to_dict`、`from_base_obs` 形参与传递、
模块级 `preprocess_observation` 的透传（它调完基类 `_preprocess_observation` 后重建
`HistAugObservation`，漏传即静默丢特征）。

### 2.3 `src/mme_vla_suite/models/integration/history_pi0.py`

- `HistoryPi0Config.inputs_spec`：在 `with at.disable_typechecking():` 块内补两个
  `jax.ShapeDtypeStruct`——`[batch_size, 32, 768] float32` 与 `[batch_size, 32] bool_`。
  ⚠ 该方法目前从 `self.history_config.budget` 等字段推形状；motion 的 32 应从
  `budget // (token_per_image * num_views)` 推导（= `FrameSampDataset._max_frames` 同式），
  不得写死字面量。
- `HistoryPi0.embed_memory`：现签名返回 `(tokens, input_mask, ar_mask, na_mask)`，内部调
  `self.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb)`。
  改为把 `obs.motion_emb` / `obs.motion_valid` 一并传入 `PerceptualMemory.__call__`；
  `input_mask` 由 `obs.static_mask` 扩成 `concat([static_mask, ones(b,32)], axis=1)`
  （motion 段恒 True，见 5.3）；`ar_mask` / `na_mask` 各追加 32 个 `False`。
  **`motion.enabled=false` 时这三处一个元素都不追加**，返回值与当前 HEAD 逐位相同。
- `HistoryPi0.embed_prefix`：无需改动——它只是把 `embed_memory` 的四元组 append 进列表，
  长度变化自动透传。这是选并列拼接的直接收益。
- `HistoryPi0.compute_loss` / `sample_actions`：`integration_type == "context"` 分支不碰
  `embed_memory` 之外的东西，无需改动；`expert` / `modulation` 两分支本轮**不接 motion**
  （训练链固定 `context`，`FrameSampDataset` 形制断言已挡住其余两种）。

### 2.4 `src/mme_vla_suite/models/representation/percep_mem.py` / `mem_encoder.py`

`PerceptualMemory.__init__` 在现有 `self.feature_encoder` **之后**（红线 5）新建三件：

```python
self.motion_pos_proj       = nnx.Linear(pos.input_dim, pos.hidden_dim, rngs=rngs, dtype=dtype,
                                        kernel_init=kernel_init)          # 768 → 768
self.motion_encoder_static = nnx.Linear(motion.dim + pos.hidden_dim, memory_token_dim,
                                        rngs=rngs, dtype=dtype,
                                        kernel_init=kernel_init)          # 1536 → 2048
self.missing_motion_emb    = nnx.Param(kernel_init(rngs.params(), (motion.dim,), dtype))
```

`PerceptualMemory.__call__` 现有 `assert static_image_emb.shape[1] == self.config.budget` 保留不动
（帧路仍是 512）；新增运动路分支，形制断言同款显式 `raise`：

```
motion_emb (b,32,768) ─ jnp.where(motion_valid[...,None], motion_emb, missing_motion_emb) ─┐
pos_f      (b,32,768) ─ silu(motion_pos_proj(pos_f)) ──────────────────────────────────────┼→ concat(-1)
                                                                                            └→ motion_encoder_static → (b,32,2048)
```

`pos_f` 的来源（`pos_source: frame_mean`）：`obs.static_pos_emb` 是 `(b, 512, 768)`，
reshape 成 `(b, 32, 16, 768)` 后沿 `axis=2` 取均值。**不改 `PosEmb3D`**——`PosEmb3D(ranges, 1)`
拿 1×1 档理论上更干净，但其 spatial index 公式在 `grid=1` 时的合法性未验证，列为 S4 可选项。

返回值仍为 `(hidden_states, None, None)` 三元组以保持 `embed_memory` 的解包不变；
motion 段作为 `hidden_states` 的后 32 个位置拼接返回。

`mem_encoder.py` 的 `FeatureEncoder` **一字不动**——运动路不复用它（复用会共享
`use_pos_emb` 分支与参数树，破坏可退性）。

### 2.5 `src/mme_vla_suite/policies/robomme_policy.py`

`RoboMMEInputs.__call__` 的 `inputs` 字典补两键，写法与既有四个 `static_*` 键完全一致：
```python
"motion_emb":   data.get("motion_emb", None),    # (32, 768)
"motion_valid": data.get("motion_valid", None),  # (32,)
```

### 2.6 数据侧（本轮只定契约，实现归 S1/S3）

- `src/mme_vla_suite/training/framesamp_dataset.py`：`FrameSampDataset.__getitem__` 已有
  `frames = even_sampling_indices(step, self._max_frames)`，motion 查表直接复用同一 `frames`
  与同一 `g`；`_NONE_KEYS` 尾部补空键列表加 `motion_emb` / `motion_valid` 两项；
  `_pad` 的目标长度 `_max_frames`(32) 对 motion 同样适用（缺失位写 0 + `valid=False`）。
- `src/mme_vla_suite/training/dataloader.py`：`_create_framesamp_dataset` 的三闸
  （`require_no_pack_lock` / `StoreMeta.load` / `require_verified`）对 motion store 照做一遍。

## 三、在线侧改动（S3）

`src/mme_vla_suite/policies/framesamp_memory.py` 的 `FrameSampMemory`：

- `__init__` 注入 `motion_enc_fn`（同 `vision_enc_fn` 的注入范式），内部持 Wan VAE + encoder。
- `add_buffer`：现按 `step_idx_list` 逐步存 `image_emb_4x4` / `pos_emb_4x4` / `state_emb`；
  新增——当 `step_idx >= 32` 时，取缓冲区里 `[step_idx-32, step_idx]` 共 33 帧
  **256×256 原始 uint8**（⚠ 不是 `resize_with_pad` 到 224 的那份，Wan VAE 要 256 域），
  按 `uint8 → /255 → ×2−1 → permute (1,3,33,256,256)` 编码，存 `_history_feats[step_idx]["motion"]`。
  **必须新缓一份 256 域原始帧**——现有 `add_buffer` 把 `images` 直接 resize 成 224 后就丢了原图。
- `_prepare_frame_sampling`：`_load_emb(history_feats, indices_to_load, "motion")` 追加一路，
  缺失帧（`idx < 32` 或未编）给零 + `valid=False`；`right_padding_token_emb` 的复用照旧
  （该函数是 `mem_buffer` 时代的数值路径，注释明记「只换模块、不换数值路径」，motion 走
  独立的 pad，不塞进它）。
- `MME_VLA_Policy._prepare_history`：补 `inputs["motion_emb"]` / `inputs["motion_valid"]`。
- ⚠ 注释里那条红线仍然有效：**禁把 encode 与 pool 包进新的 `jax.jit`**（融合边界变了，
  bf16 累加序可能变位）。motion 编码走 PyTorch、在 jit 之外，天然不违反。

## 四、对拍闸门总表

| 闸 | 阶段 | 判据 | 失败处置 |
|---|---|---|---|
| **M0** 环境指纹 | S0 前 | 引用既有基线 run 时先过指纹 preflight（`AGENTS.md` 第 18 条末款） | 指纹不符即基线失效，重跑基线 |
| **M1** 延迟 | S0 | ms/窗口实测（fp32/关TF32 与 TF32+bf16 两档）；两档输出余弦 ≥ 阈值 | 达不到 → 方案回退 motion predictor 路线 |
| **M2** 抽表逐位 | S1 | 随机 200 个 `(段, c)`，在线跑 encoder vs 表逐位相等（`np.array_equal` 于 f32 位型） | 任一不等即停，查 dtype/TF32/batch 口径 |
| **M3** 索引映射 | S1 | 随机 500 个 `(g, f)`，`row_base + (u−32)` 读出的行 == 按 episode 名直读 `.bin` 第 `c` 个 chunk 经 encoder 的输出 | 不等即查 `motion_index.json` 定序 |
| **M4** 覆盖率账 | S1 | 表行数 == 396,302；exec 333,900 + demo 62,402；逐段 `num_chunks == frames − 32` | 不符即 MotionJEPA metadata 与实际 `.bin` 不配套 |
| **M5** 关闭态等价 | S2 | `motion.enabled=false`，`embed_prefix` 的四个返回张量与当前 HEAD **逐位相同**（同 rng、同输入 fixture） | 不等即红线 5 被违反（RNG 消耗序变了） |
| **M6** 开启态形制 | S2 | prefix 序列长 512+32+512+L_p；`ar_mask`/`na_mask` 在 motion 段全 False；`positions` 逐样本一致（无抖动） | — |
| **M7** 尺度 | S2 | `‖motion_tok‖₂ / ‖mem_tok‖₂` 的 batch 均值落在 [0.3, 3.0] | 越界则在 `motion_encoder_static` 后补 RMSNorm |
| **M8** 梯度一致 | S2 收尾 | `motion.enabled=false` 本机跑前 N 步，逐步 loss / grad-norm / 参数摘要与既有基线一致（N 起工前商定） | 不一致即 S2 不得宣称等价 |
| **M9** 在线/离线一致 | S3 | 同一 `(g, f)` 的在线编码 vs 离线表，余弦 ≥ M1 阈值 | — |

## 五、第一块：非训练轻量对拍明细（`AGENTS.md` 第 18 条第一块）

不启动训练，四项：

1. **M5 关闭态逐位**：用 `scripts/training/tests/dump_fixture_samples.py` 现成的 fixture 机制
   dump 同一批样本，改动前后各跑一次 `embed_prefix`，比 `tokens` / `input_mask` / `ar_mask` /
   `na_mask` 四个张量的原始字节。
2. **M2/M3 表与索引**：独立小脚本，不进训练环。
3. **逐样本内容对拍**：`FrameSampDataset.__getitem__` 改动前后，对同一批 `idx` 比全部键的
   dtype / shape / 字节——`motion.enabled=false` 时新增的两个键应为 `None`（走 `_NONE_KEYS`），
   其余键逐位不变。
4. **index 序列对拍**：`scripts/training/tests/dump_index_seq.py` 同款，确认 shuffle 序不受影响。

## 六、第二块：本机训练梯度一致 runbook（`AGENTS.md` 第 18 条第二块）

- 本机可跑档位启动真实训练，`motion.enabled=false` 跑前 N 步，与既有基线逐步比对
  loss / grad-norm / 参数摘要。
- **若复用既有基线 run 的固化产物**（而非同场次重跑对照侧），必须先过环境指纹 preflight，
  并在留档写明所引用基线的 `run_name`、commit 与指纹比对结论；**指纹不符即该基线失效，
  必须重跑基线后再对拍**。
- 第二块不通过不得宣称改动等价（M8）。
- `motion.enabled=true` 的梯度**不做等价对拍**——那是新语义，只做 M6/M7 形制与尺度检查。

## 七、风险登记

| # | 风险 | 概率 | 影响 | 处置 |
|---|---|---|---|---|
| R1 | 在线 1.57 s/step 降不下来 | 中 | 高——「同步推理」前提不成立 | S0 前置实测；降不下来即整体回退 motion predictor 路线 |
| R2 | TF32+bf16 在线口径与 fp32 离线表漂移过大 | 中 | 中——训练/推理特征分布不一致 | M1 定量；超阈值则离线表也改用同口径重抽（1.13 GiB，重抽成本可接受） |
| R3 | 新参数插入位置错误改变 RNG 消耗序 | 低 | 高——M5 直接失败且难定位 | 红线 5 明写；M5 是它的探测器 |
| R4 | 31.4% 缺失让 `missing_motion_emb` 主导表征 | 中 | 中 | M7 之外增记 valid 比例与 motion 段激活统计；S4 消融「只在 valid 率 > 阈值的样本上启用」 |
| R5 | MotionJEPA encoder 的已知缺陷（§3 R3：对 32×32 网格零权重共享、两种编码模式共用一个投影） | 已知 | 未知 | 该缺陷在 MotionJEPA 侧由用户拍板不动，检测靠 motion std / 余弦仪表；本计划沿用其 checkpoint，不修 |
| R6 | 在线侧多背一个 Wan VAE 的显存 | 中 | 中 | S3 实测；Wan2.1-T2V-1.3B 的 VAE 部分显存可控，但与 pi05 主干共卡需实测 |
| R7 | exec 段开头 32 帧窗口跨 demo/exec 边界被误判为合法 | 低 | 高——喂错数据静默训坏 | M3 专项覆盖跨界样本；`motion_index.json` 按段独立记 `row_base` 使跨界在结构上不可表达 |

## 八、盲区诚实清单（写入 S1/S2 的 `result.md`）

1. **motion token 的语义未经独立验证**。它是 MotionJEPA 为「从 z0 预测未来 8 段 latent」训练出来的，
   在 VLA 里当历史运动特征用属于跨任务迁移，本计划不含对该迁移有效性的先验证据。
2. **后视口径下 encoder 的输入分布与其训练分布一致**（都是「某个 33 帧窗口的 9 个 latent」），
   但 **motion token 表达的是「相对窗口锚点 z0 的运动」**——后视口径下锚点是 33 帧前那一帧。
   这是有意的（描述「从 33 帧前到现在发生了什么」），但与 MotionJEPA 训练时「锚点即当前」的
   使用语境不同。
3. **`pos_source: frame_mean` 是权宜**。16 个 4×4 空间 pos 取均值得到的向量，其在
   `PosEmb3D` 的 sin/cos 结构里没有明确语义（均值不是某个网格点的编码）。列为 S4 可选项。
4. **消融覆盖不全**：S4 五项互相有交互，本计划不承诺跑满全矩阵。
5. **未覆盖 `expert` / `modulation` 两种 integration_type**。

## 九、留档与 commit 纪律

- S1 属正式全量数据集构建 → `docs/dataset-build-doc/4task-gl-motion/`（`AGENTS.md` 第 12 条）：
  记 commit、命令、配置、数据来源、输出路径、M2–M4 判据结果；不归档 encoder 权重。
- S2 的等价对拍 run 若超 5 分钟 → 视作完整运行，`docs/training-doc/<run_name>/`
  （launch.md / result.md / records/，第 17 条）。
- 正式 run 起跑前按第 6 条向用户确认全新 `run_name`；从 clean HEAD 起跑（第 12 条）。
- 代码切片按 `commitV6.<小版本>` 编号，文档/修补用 `docs:` / `fix:`；逐文件 `git add`，
  禁 `git add .` / `-A` / `commit -a`；每次 commit 后立即 `git push` 同步 origin（第 11 条）。
- 集群作业一律先读仓库根 `greatlakes.md`（第 8 条）；ssh 前必须问用户 Okta 验证方式。
