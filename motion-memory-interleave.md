# motion memory 交错拼接说明——运动路按时间插进帧路

> **本文性质**：说明文档，只负责把「交错拼接」方案从训练链路第一步讲到 gemma 内部，再补在线推理；不是实施授权，不写实现细节。
> **对应主计划**：`motion-memory-plan.md`（并列拼接方案）。主计划一至三节处于冻结状态，本文不改它，冲突处只点名。
> **定稿日期**：2026-09-02。**代码锚点**：HEAD `4503ea2` 不变，本文写作期间 `src/` `scripts/` 零改动。

**一句话结论**：与并列方案相比，交错方案送进模型的 token 内容、权重、mask、计算量全部相同；数学上唯一的区别是 memory 段 592 个 token 各自拿到的 RoPE 位置号；实现上多一步排序、一步重排、一个交付键。

---

## 一、三条已定口径

本轮用户拍板的三条，原话保留：

1. 「帧路不动！只是交错拼接运动路！」——`even_sampling_indices(t, 32)` 一字不动，仍是 32 帧 × 16 token = 512 位；运动路仍是最多 80 个 motion token。
2. 「按起点 f 插入」——一个 motion token 描述窗口 [f, f+32] 的运动，插入位置由起点 f 决定，不按尾端、不按中点。
3. 「运动起点和采样帧号相同时，采样帧在前」——同一时刻既有采样帧又有 motion 起点时，先放该帧的 16 个 token，再放这个 motion token。

附带一条不需要单独拍板的事实：padding 全部落在尾部。这是排序的自然结果（见三节 3.0 的 e 步与四节 4.2 的 k 步），不是额外要求，也不是正确性所需。

---

## 二、数轴：并列 vs 交错

取当前帧 t = 200、段起点 0 的样本。帧路 32 帧全满，运动路合法起点 0, 20, …, 160 共 9 个。两种方案的 token 一个不多一个不少，只是顺序换了。

```
时间轴（帧号）
0    6   12  19  25  32  38  45  51  58  64  70  77  83  90  96  103 109 116 122 129 135 141 148 154 161 167 174 180 187 193 200
▮    ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ← 帧路，每 ▮ = 16 token
●         ●          ●           ●           ●            ●           ●            ●            ●                                ← 运动路，每 ● = 1 token
0         20         40          60          80           100         120          140          160

━━━━━━━━━━ 并列（主计划现行方案）━━━━━━━━━━
┌── 帧路 512 位 ──────────────────────────────────┬── 运动路 80 位 ─────────┐
│ f0 f6 f12 … f193 f200（32 帧 × 16）             │ m0 m20 … m160 │ pad×71 │
└─────────────────────────────────────────────────┴───────────────┴────────┘
  位 0–511                                          位 512–520     位 521–591

━━━━━━━━━━ 交错（本文方案）━━━━━━━━━━
f0×16 m0 f6×16 f12×16 f19×16 m20 f25×16 f32×16 f38×16 m40 f45×16 f51×16 f58×16 m60 f64×16 f70×16 f77×16 m80
f83×16 f90×16 f96×16 m100 f103×16 f109×16 f116×16 m120 f122×16 f129×16 f135×16 m140 f141×16 f148×16 f154×16 m160
f161×16 f167×16 f174×16 f180×16 f187×16 f193×16 f200×16 │ pad×71
└──────────────────── 521 个 True，同样的 521 个 token ────────────────────┘ 位 521–591 False
```

两条规律从图上就能看出来：motion 落在起点之后最近的那个采样帧之前（m160 在 f154 之后、f161 之前）；帧 0 与起点 0 同刻，按口径 3 帧在前，所以 m0 排在 f0 的 16 个 token 之后。

再看一个帧路没填满的样本，t = 5：帧路只有 6 帧 96 位，运动路一个合法起点都没有。

```
并列：[f0 … f5 共 96 位 │ 帧路 pad×416 │ 运动路 pad×80]     True 在位 0–95，False 在 96–511 ∪ 512–591
交错：[f0 … f5 共 96 位 │ pad×496]                          True 在位 0–95，False 在 96–591
```

两种布局 True 的集合完全相同。并列方案里帧路自己的 padding 卡在段中间，交错方案里所有 padding 一并排到尾部。

---

## 三、训练：一步 `train_step` 里的全部数值计算

本节把训练时一个 batch 从进入模型到算出梯度的每一次数值计算按顺序列出，每一步写明是哪个函数在算、输入输出形状是什么。代码分三层叠上去：**openpi 的 `Pi0`**（`src/openpi/models/pi0.py`）是底；**`HistoryPi0`**（`src/mme_vla_suite/models/integration/history_pi0.py`）把 `Pi0` 的 `embed_prefix` / `embed_suffix` / `compute_loss` / `sample_actions` 四个函数整体复制一份，在前面加了记忆段，并给 `make_attn_mask` 多加一条 `na_mask`；**motion memory** 只改 `HistoryPi0.embed_memory` 里面那一小块。每一步末尾用〔Pi0〕〔HistoryPi0〕〔motion〕标出这一步是哪一层加的，用 ★ 标出交错方案与并列方案有差别的地方。

batch 大小记 b，PaliGemma 主干宽度 2048，action expert 宽度 1024，18 层，8 个 query 头、1 个 key/value 头、每头 256 维。下文默认 t = 200 的样本，memory 段 521 个真 token。

### 3.0 数据侧：`FrameSampDataset.__getitem__` 交付的键〔dataloader，motion 新增运动路〕

模型看到的第一份数据来自 dataloader worker 进程里的 `FrameSampDataset.__getitem__`（`src/mme_vla_suite/training/framesamp_dataset.py`），全部是 numpy 计算：

| 步 | 函数 | 计算 | 输出 |
|---|---|---|---|
| a | `even_sampling_indices(step, 32)` | t ≥ 31 时 `linspace(0, t, 32)` 取整，否则 `range(t+1)` | n 个帧号 |
| b | `FrameSampStore.read_image_rows` / `pos_rows` / `state_rows` | 按帧号读离线表 | (n,16,2048) bf16、(n,16,768) f32、(n,8) f32 |
| c | `_pad(img, pos, stt, n)` | 预分配 32 行，前 n 行放真数据，后 32−n 行写 0，帧级 mask (32,) 前 n 位 True | (32,16,2048) 等 |
| d | reshape + `np.repeat(mask, 16)` | 摊成 512 位，`static_state_emb` 经 `_normalize_state` 归一化 | `static_image_emb` (512,2048)、`static_pos_emb` (512,768)、`static_mask` (512,) |
| d′〔motion〕 | 运动路（主计划第二部分一节） | 网格起点 0,20,… 中 f+32 ≤ t 者，逐个读 `motion_token.f32.bin` 一行、切 `pos_rows(f)[0,0,:256]`，另写的填充函数补到 80 行 | `motion_emb` (80,768)、`motion_pos` (80,256)、`motion_mask` (80,) |
| e ★ | 排序（新增） | 592 个候选位各配键 (时刻, 类型)：帧 i 的 16 位 (帧号, 0)，motion k (起点 f, 1)，两路 padding (无穷大, 各自类型)；`np.argsort(kind="stable")` | `mem_order` (592,) int32 |

e 步的键定义把三条口径全兑现了：按时刻排就是「按起点 f 插入」，类型 0 < 1 就是「同刻帧在前」，padding 时刻无穷大就是「mask 全在尾部」；稳定排序保证同一帧的 16 个 token 保持原有网格顺序。t = 200 的排序结果就是二节那条交错序列；t = 5 是 96 个帧 token 后跟 496 个 padding。新键每样本 592 × 4 B = 2.3 KB，相对现有 3.52 MiB 可忽略。

collate 把 64 个样本逐键摞成 batch，`RepackTransform` 与 `HistAugObservation.from_dict` 把键装进观测对象。motion 关闭时 d′ 与 e 两步不存在，三键与 `mem_order` 为 None。

### 3.1 入口：`train_step` → `loss_fn` → `compute_loss`〔Pi0，train.py 包一层〕

`scripts/training/train.py` 的 `train_step` 被 `jax.jit` 编译。它把参数合进模型，定义 `loss_fn` 为 `model.compute_loss(rng, observation, actions, train=True)` 的均值，用 `nnx.value_and_grad` 一次算出 loss 与所有可训练参数的梯度，再由 optax 更新参数、更新 EMA。它记的四个标量是 `loss`、`grad_norm`、`param_norm`（所有二维以上 kernel 的全局范数）、`llm_grad_norm`，开了记忆再加 `mem_enc_norm`。交错方案不动这一层。

### 3.2 观测预处理与加噪〔Pi0〕

`HistoryPi0.compute_loss` 先把 rng 一分为三，然后：

- `preprocess_observation`（`history_observation.py`）拆出基类观测交给 openpi 的同名函数：图像若不是 224×224 先 `resize_with_pad`；训练时对非腕部图像做随机裁剪 95%、放回原尺寸、旋转 ±5°，再做亮度 / 对比度 / 饱和度抖动。做完把 `static_*` 四键原样放回。★交错方案下 `motion_*` 三键与 `mem_order` 也在这里原样透传，不参与任何运算。
- 加噪：`noise ~ N(0,1)`，形状与 actions 同 (b, 20, action_dim)；`time ~ Beta(1.5, 1) × 0.999 + 0.001`，形状 (b,)；`x_t = time · noise + (1 − time) · actions`；目标速度 `u_t = noise − actions`。

### 3.3 `embed_prefix`：把记忆、图像、文本编成一条前缀〔HistoryPi0 改写自 Pi0〕

`Pi0.embed_prefix` 只做图像与文本两段。`HistoryPi0.embed_prefix` 在 `integration_type == "context"` 时先调 `embed_memory` 得到记忆段，把它排在最前面，然后图像、文本照 `Pi0` 的写法。三段各自产出 token、`input_mask`、`ar_mask`，`HistoryPi0` 再多产一条 `na_mask`。

**3.3.1 记忆段：`embed_memory`〔HistoryPi0〕→ `PerceptualMemory.__call__` → `FeatureEncoder.encode_perceptual_memory`**

帧路（并列与交错完全相同）：

| 步 | 函数 | 计算 | 输出 |
|---|---|---|---|
| a | `FeatureEncoder._add_pos_emb` | `static_pos_emb` (b,512,768) 过 `pos_proj`（Linear 768→768），过 `nnx.silu`，与 `static_image_emb` (b,512,2048) 在最后一维拼接 | (b,512,2816) |
| b | `FeatureEncoder._encode_memory` | 过 `encoder_static`（Linear 2816→2048） | (b,512,2048) |

`use_state_emb=False`，`static_state_emb` 不参与。`PerceptualMemory.__call__` 开头断言 `static_image_emb.shape[1] == budget`（512），返回 `(hidden_states, None, None)`。padding 行是零向量，过两层 Linear 后是由 bias 决定的非零向量，这里不做任何分支。

运动路〔motion〕（并列与交错完全相同）：

| 步 | 函数 | 计算 | 输出 |
|---|---|---|---|
| c | `PerceptualMemory.__call__` 运动分支 | `motion_pos` (b,80,256) 过 `motion_pos_proj`（Linear 256→768），过 `nnx.silu`，与 `motion_emb` (b,80,768) 拼接 | (b,80,1536) |
| d | 同上 | 过 `motion_encoder_static`（Linear 1536→2048） | (b,80,2048) |
| e | 同上 | 与帧路在长度轴拼接，得并列顺序的 memory | (b,592,2048) |

`embed_memory` 再把 `input_mask = [static_mask ⊕ motion_mask]` (b,592) 接好，`ar_mask` 与 `na_mask` 各 592 个 False。

★交错方案在 e 之后多一步 f：

| 步 | 函数 | 计算 | 输出 |
|---|---|---|---|
| f ★ | `embed_memory` 新增 | 按 `mem_order` (b,592) 用 `jnp.take_along_axis` 把 memory 的 592 行与 `input_mask` 的 592 位一起换到时间序位置 | (b,592,2048)、(b,592) |

`ar_mask` / `na_mask` 全 False，不需要换。motion 关闭时 c–f 四步都不存在，`embed_memory` 与今天逐字相同。

**3.3.2 图像段〔Pi0〕**

每个视角的图像 (b,224,224,3) 过 `self.PaliGemma.img`，即 SigLIP So400m/14，patch 14 像素，16×16 = 256 个 patch，输出 (b,256,2048)。两个视角共 512 个 token。`input_mask` 由 `image_masks` 广播得到；`HistoryPi0` 把第一个视角第一个 token 的 `ar_mask` 记 True（这是记忆块与图像块的分界），其余 False；`na_mask` 图像段全 True。SigLIP 参数被 `get_freeze_filter` 的 `.*img.*` 冻结，不参与梯度。

**3.3.3 文本段〔Pi0〕**

`tokenized_prompt` (b,≤64) 过 `self.PaliGemma.llm(..., method="embed")`，即 gemma `Embedder.encode`：查 (257152, 2048) 的词表再乘 √2048，输出 (b,≤64,2048)。`input_mask` 取 `tokenized_prompt_mask`；`ar_mask` 与 `na_mask` 全 False。

三段在长度轴拼接：prefix tokens (b,1168,2048)，`prefix_mask` (b,1168)，`prefix_ar_mask` (1168,)，`prefix_na_mask` (1168,)。

### 3.4 `embed_suffix`：把带噪动作编成后缀〔Pi0，pi05 分支〕

- `action_in_proj`（Linear action_dim→1024）作用于 `x_t` (b,20,action_dim)，得 (b,20,1024)。
- `posemb_sincos(time, 1024, 4e-3, 4.0)` 把标量时间编成 (b,1024)，过 `time_mlp_in`（Linear 1024→1024）、`nnx.swish`、`time_mlp_out`、`nnx.swish`，得 `adarms_cond` (b,1024)。pi05 下没有 state token，状态只通过 `adarms_cond` 进模型。
- `input_mask` 全 True (b,20)；`ar_mask = [True] + [False]×19`；`na_mask` 全 False。

### 3.5 三条 mask 拼接、`make_attn_mask`、位置号〔HistoryPi0 改写自 Pi0〕

`compute_loss` 把 prefix 与 suffix 首尾相接：`input_mask` (b,1188)、`ar_mask` (1188,)、`na_mask` (1188,)。

`make_attn_mask(input_mask, ar_mask, na_mask)`（`history_pi0.py` 版本，比 openpi 版多 `na_mask` 一参）：

1. `cumsum(ar_mask)` 给每个 token 一个块号：记忆段 592 位块号 0，图像 + 文本块号 1，动作块号 2。`attn_mask[q,k] = 块号[k] ≤ 块号[q]`，即只能看同块或更早的块。
2. `valid_mask = input_mask[:,None,:] ∧ input_mask[:,:,None]`，padding 位整行整列 False。
3. `mask_not_attend`：`na_mask` 为 True 的 token（图像）不能看 `cumsum(na_mask) ≤ 0` 的 token（图像段之前的所有位置，即记忆段）。
4. 三者合成 `attn_mask` (b,1188,1188)。

效果是：记忆只看记忆；图像看图像与文本、看不到记忆；文本看记忆、图像、文本；动作看一切；padding 谁都看不到。★交错方案下这张表的 True/False 个数不变，只是记忆段内 padding 列的编号从两段合成尾部一段。

`positions = jnp.cumsum(input_mask, axis=1) − 1` (b,1188) int。★交错方案下 memory 段 592 个整数的取值变了（帧 token 后移、motion 前移，二节数字），第 593 位起（图像、文本、动作）逐位不变。

### 3.6 主干：`PaliGemma.llm([prefix_tokens, suffix_tokens], mask, positions, adarms_cond)`〔Pi0〕

`history_gemma.Module.__call__` 先把两段 token 转成 bf16，mask 加一维成 (b,1,1188,1188)，然后 `nn.scan` 依次跑 18 个 `HistoryBlock`。`HistoryBlock` 在 `integration_type == "context"` 下与 openpi 的 `Block` 逐步相同（`MemoryAttention` 只在 modulation 模式出现，本链路不走），其内部的 `Attention`、`RMSNorm`、`FeedForward`、`_apply_rope`、`_gated_residual` 全部直接 import 自 `src/openpi/models/gemma.py`。每层按顺序：

| 步 | 函数 | 计算 |
|---|---|---|
| g | `RMSNorm`（`pre_attention_norm`） | prefix 走普通 RMSNorm：x / √(mean(x²)+1e−6) × (1+scale)；suffix 走自适应版：用 `adarms_cond` 过一个 Dense 得 scale / shift / gate，归一后 × (1+scale) + shift |
| h | `Attention.__call__` 投影 | prefix 过 `q_einsum` (2048→8×256) 与 `kv_einsum` (2048→1×256 两份)；suffix 过 `q_einsum_1` / `kv_einsum_1`（1024 宽）；两段在长度轴拼成 q (b,1188,8,256)、k (b,1188,1,256)、v 同 k |
| i ★ | `_apply_rope(q, positions)`、`_apply_rope(k, positions)` | 对每个 token 的 256 维按 `positions / 10000^(2j/256)` 算旋转角，前后两半做二维旋转；q 再乘 256^−0.5。**这是整条训练链里唯一直接消费位置号的计算**，交错方案改的数值只从这里进入 |
| j | `Attention.__call__` 打分 | `logits = einsum(q, k)` 以 f32 累加，得 (b,1,8,1188,1188)；`where(mask, logits, −2.3819763e38)`；`softmax` 沿最后一维；`probs @ v` 得 (b,1188,8,256) |
| k | `attn_vec_einsum` / `attn_vec_einsum_1` | 8×256 拼回各自宽度，prefix 段回 2048、suffix 段回 1024 |
| l | `_gated_residual` | prefix：x + attn_out；suffix：x + attn_out × gate |
| m | `RMSNorm`（`pre_ffw_norm`）+ `FeedForward` | 归一后过 gating_einsum 两路（gelu 门 × 线性）再过 linear，prefix 宽 2048→16384→2048，suffix 宽 1024→4096→1024；再一次 `_gated_residual` |

18 层跑完各段过自己的 `final_norm`。输出 `prefix_out` (b,1168,2048) 与 `suffix_out` (b,20,1024)。

j 步里被 mask 为 False 的格子经 `where` 换成 f32 最小值，`exp` 后精确为 0，所以 padding 列在 `probs @ v` 里乘 0，对任何输出零贡献。这与位置号无关，交错方案不改这一机制。

### 3.7 输出头与 loss〔Pi0〕

`action_out_proj`（Linear 1024→action_dim）作用于 `suffix_out[:, −20:]` 得 `v_t` (b,20,action_dim)。`compute_loss` 返回 `mean((v_t − u_t)², axis=−1)` (b,20)，`loss_fn` 再对 b 与 20 取均值得标量。

### 3.8 反向与更新〔train.py〕

`nnx.value_and_grad` 沿 3.2–3.7 反传，SigLIP 被冻结不拿梯度，其余（gemma 两个 expert、`mem_encoder` 下的四个 Linear、action 头与 time MLP）全部更新。★交错方案的 `take_along_axis` 是可微的索引操作，梯度按同一张 `mem_order` 表原路搬回并列顺序，再分别流向帧路与运动路的投影。

**三节小结**：交错方案在训练链上只出现在四处——3.0 的 e 步（dataloader 排序）、3.3.1 的 f 步（模型侧 gather）、3.5 的 memory 段位置号取值、3.6 的 i 步旋转角。其余每一步的函数与计算逐字不变。

---

## 四、推理：一次 `infer` 里的全部数值计算

推理入口是 `src/mme_vla_suite/policies/policy.py` 的 `MME_VLA_Policy`。与训练的两点结构差异：记忆特征不是从离线表读，是每步现编后存进内存字典；主干不是一次算完 1188 个 token，而是先算 1168 个前缀存 kv_cache，再在去噪循环里每步只算 20 个动作 query。

### 4.1 每步入库：`add_buffer`〔HistoryPi0 在线侧，motion 新增运动路〕

`MME_VLA_Policy.add_buffer` 把当前帧交给 `FrameSampMemory.add_buffer`：

| 步 | 函数 | 计算 |
|---|---|---|
| a | `add_buffer` | uint8 图像 → float32 / 255 × 2 − 1，`resize_with_pad` 到 224 |
| b | `vision_enc` = `module_jit(HistoryPi0.vision_encode)` → `PaliGemma.img` | 同一个 SigLIP，输出 (t,v,256,2048) |
| c | `pool_tokens_to_size(…, 16)` | 16×16 网格平均池化成 4×4，得 (t,v,16,2048) |
| d | `PosEmb3D` 预计算表 `pos_emb_4x4[step]` | 该帧 (v,16,768) 的位置编码，与训练侧离线表同一算法同一输入 |
| e | 存字典 | `_history_feats[step] = {image_emb_4x4, pos_emb_4x4, state_emb}` |

★运动路〔motion，主计划六节〕：另存一份 256 域原图滚动缓冲；当「下一个网格起点 + 32」这 33 帧到齐，喂 sidecar 里的 Wan VAE 与 `WanLatentMotionEncoder` 得一个 768 维 token，存 `_history_feats_motion[起点]`。每 20 帧触发一次。交错方案不改这一步。

### 4.2 组装记忆：`infer` → `_prepare_history`〔HistoryPi0 在线侧〕

| 步 | 函数 | 计算 |
|---|---|---|
| f | `FrameSampMemory.get_frame_sampling_indices` → `even_sampling_indices(step_idx, 32)` | 与训练同一个选帧函数 |
| g | `_prepare_frame_sampling` → `_load_emb` | 从字典按帧号堆出 (n,16,2048) / (n,16,768) / (n,8) |
| h | `right_padding_token_emb` | 用 `np.concatenate` 补零块到 32 帧，帧级 mask (32,)；与训练侧 `_pad` 数值相同 |
| i | reshape + `np.repeat(mask, 16)` | 摊成 (512,2048) / (512,768) / (512,) |
| j ★ | `_prepare_motion`〔motion〕 | 按同一网格公式取合法起点、取最近 80 个、右填充；`motion_pos` 从 `pos_emb_4x4[起点, 0, :256]` 切 |
| k ★ | 排序（与训练共用同一个函数） | 按 (时刻, 类型) 稳定排序得 `mem_order` (592,) |

`_prepare_history` 把这些键塞进 `inputs`，再过 `_input_transform`（状态归一化等）、`HistAugObservation.from_dict`、加 batch 维 b = 1。

### 4.3 前缀 pass：`sample_actions` 前半〔HistoryPi0 改写自 Pi0〕

`_sample_actions` 是 `module_jit(HistoryPi0.sample_actions)`。

| 步 | 函数 | 计算 |
|---|---|---|
| l | `preprocess_observation(None, obs, train=False)` | 只做尺寸检查，不做增广；记忆键原样透传 |
| m | `embed_prefix` | 与训练 3.3 逐字同一函数：`embed_memory`（含 ★ f 步 gather）、SigLIP、词表；得 (1,1168,2048) |
| n | `make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)` | (1,1168,1168)，规则同 3.5 |
| o ★ | `positions = cumsum(prefix_mask) − 1` | (1,1168)；memory 段 592 个取值随交错而变 |
| p | `PaliGemma.llm([prefix_tokens, None], mask, positions)` | 只跑 expert 0。每层 `Attention` 里 k 经 `_apply_rope(k, positions)` 旋转后与 v 一起作为该层的 `(k, v)` 返回；18 层堆成 kv_cache，形状 18 × (1,1168,1,256) × 2 |

★交错方案下 memory token 的旋转角在 p 步就烙进 kv_cache 的 k 里，后面去噪循环直接消费这份缓存。

### 4.4 去噪循环：`step` × `num_steps`〔Pi0〕

`jax.lax.while_loop` 从 `x_t = noise`、`time = 1.0` 起，每步 `dt = −1/num_steps`（默认 10 步）：

| 步 | 函数 | 计算 |
|---|---|---|
| q | `embed_suffix(obs, x_t, time)` | 同训练 3.4，得 (1,20,1024) 与 `adarms_cond` |
| r | `make_attn_mask(suffix_mask, suffix_ar_mask)` | 20×20 因果表 |
| s | `einops.repeat(prefix_mask, "b p -> b s p", s=20)` | 动作对前缀的可见性直接复制 `prefix_mask` 20 行，padding 列第二次被封；与 r 拼成 (1,20,1188) |
| t | `positions = sum(prefix_mask) + cumsum(suffix_mask) − 1` | 动作的位置号接在前缀真 token 数之后；交错不改 |
| u | `PaliGemma.llm([None, suffix_tokens], mask, positions, kv_cache)` | 只跑 expert 1 的 q / k / v 投影；`Attention` 把缓存的 1168 个 k、v 与新的 20 个拼接；20 个 q 经 `_apply_rope` 旋转；logits (1,1,8,20,1188)；`where` / `softmax` / `@v`；`attn_vec_einsum_1`；残差、FFN 同 3.6 |
| v | `action_out_proj` | `v_t` (1,20,action_dim)；Euler 一步 `x_t ← x_t + dt · v_t`，`time ← time + dt` |

循环结束的 `x_0` 经 `_output_transform` 反归一化后作为动作输出。

**四节小结**：推理链上交错方案改的地方与训练完全对应——k 步排序（对应训练 dataloader 的 ④′）、m 步 gather（同一个 `embed_memory`）、o 步位置号、p 步 k 的旋转角。去噪循环 q–v 六步代码与数值机制不变，只是消费的 kv_cache 已经不同。训练与在线必须共用同一个排序函数，规则只在一处定义；两侧若各写一份，排序稍有出入就会让在线模型看到与训练不同的位置号分配，不报错，只静默降效果。

---

## 五、数值层面的结论

**数学上**：transformer 的每一层对 token 顺序是置换等变的，把 token 行、它的 mask 行列、它的位置号一起换顺序，每个 token 算出来的输出向量不变，只是行的摆放顺序换了。loss 只读 action 那 20 行，那 20 行不动。所以物理交错这件事本身没有效果，效果全部来自「换了位置号」，进入计算的位置只有 3.6 的 i 步与 4.3 的 p 步那两行 `_apply_rope`。

**数值上**：交错方案与并列方案不逐位，有两个原因。一是位置号不同，这是语义差异，本来就该不同。二是行序不同，矩阵乘与 softmax 的浮点累加顺序跟着变，会有末位差异。因此交错方案与并列方案之间不做等价对拍，等价对拍只做一条：motion 关闭态对 HEAD 逐位相同。

**计算量**：attention 仍是 1188 × 1188，padding 个数相同，jit 编译形状相同，kv_cache 形状相同。多出的两步里，排序是 numpy 里 592 个元素的一次排序，重排是 GPU 上每样本约 2.4 MB 的一次搬运，相对 1188² 的 attention 都可忽略。「mask 全在尾部」不省任何计算。

---

## 六、与主计划的关系

主计划一至三节冻结，本文不改它。交错方案与主计划冲突或需补充的位置如下，供日后决定是否回写：

- 一节「五个已定死的口径」第 4 条「并列拼接」：交错方案下改为「按起点时刻交错，同刻帧在前」。
- 3.1.2 改动后链路图的最后一跳「长度轴 concat：512 + 80 = 592」：之后多一步重排。
- 3.3 的 mask 数轴图（`docs/motion-memory-mask-axis.svg`）：图里写死的「记忆 512 位 + 运动 80 位」两段边界在交错后不再固定。
- 3.4 全节里「运动路第 521 到 591 列」「第 9 到 79 行」这类固定编号：交错后随样本变化。
- 5.1 一览表：交付键从 7 个变 8 个，`n_keys` 开启态从 15 变 16；`HistAugObservation` 多一个字段。

主计划口径 1（段内绝对网格）、2（前视窗口尾端不越当前帧）、3（预算 80 零截断）、5（缺失走 `motion_mask`）全部不变。

---

## 七、已知的未知

- 交错对效果有没有帮助，没有先验证据。token 内容里已带时间码，交错只是把「时间相邻」额外写进 RoPE 距离。
- RoPE 位置密度不均：一个帧占 16 个连续位置号，一个 motion 只占 1 个。
- 同刻冲突只在帧 0 与起点 0 之间必然发生，其余靠巧合；规则固定后没有随机性。
- 建议把「并列 vs 交错」列为消融的一项与并列方案对照，而不是直接替换默认。
