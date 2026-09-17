# motion memory 注入方式说明——context 与 modulation 从 dataloader 到 gemma 内部的分叉

> **本文性质**：说明文档，只负责把「记忆怎么送进模型」这件事从 dataloader 交付的键讲到 gemma 每一层的张量形状，再补推理侧；不是实施授权，不写实现细节。
> **对应主计划**：`0916-motion-modul-8x8-plan.md` 第四节「context 改 modulation」与 I.1–I.4；两文冲突以主计划为准。**姊妹文档**：`motion-memory-interleave.md` 讲的是 context 口径下的交错拼接（32 帧 × 16 token、budget 96），本文的共用前半段直接引用它，不重写。
> **定稿日期**：2026-09-17。**代码锚点**：HEAD `2e48362`，本文写作期间 `src/` 零改动。注意 HEAD 上 `HistoryPi0.__init__` 仍有一行 `raise ValueError("motion memory 只接 integration_type=context …")`，即「modulation + motion 开启」在 HEAD 上跑不起来；本文按主计划目标口径（8 帧 × 8×8、modulation、`motion.budget` 160）写，凡是计划值而非 HEAD 实测值的数字标 ★。

**一句话结论**：从 dataloader 到 `HistoryPi0.embed_memory` 的输出为止，context 与 modulation 吃的是同一份记忆序列（只差宽度 2048 / 1024）；分叉只在「这条序列去哪」——context 把它拼进主干前缀、与图像文字动作一起过 18 层 self-attention，modulation 不让它进主干、每层只让 20 个动作 token 对它做一次 cross-attention 并用读出的向量调制 FFN 输入。由此派生的最重要后果是：modulation 的 cross-attention 给 padding 行也编 RoPE 位置号，`motion.budget` 因此从「缓冲区大小」变成决定前向计算的模型语义参数，训练 / 评估 / 在线三侧必须逐值相同。

---

## 一、先把常量定下来

后文每一步的形状都由这张表推出，记号 b = batch。

| 常量 | 值 | 出处 |
|---|---|---|
| 主干（expert 0，PaliGemma） | `gemma_2b`：宽 2048、18 层、8 个 query 头、1 个 kv 头、head_dim 256、mlp 16384 | `src/openpi/models/gemma.py::get_config` |
| action expert（expert 1） | `gemma_300m`：宽 1024、18 层、8 / 1 头、head_dim 256、mlp 4096 | 同上 |
| 图像 | 两视角，每视角 SigLIP 出 256 个 2048 宽 token | `embed_prefix` 里 `PaliGemma.img` |
| 文本 | 64 个 token | `HistoryPi0Config.max_token_len = 64` |
| 动作 | 20 步 × 32 维；进 expert 1 前经 `action_in_proj` 32→1024 | `embed_suffix` |
| 帧路记忆 | 8 帧 × 64 token = 512 位；每位 2048 维外观 + 768 维位置码 | YAML `budget: 512`、`token_per_image: 64` |
| 帧路投影 | `pos_proj` 768→768，`encoder_static` 2816→D | `mem_encoder.py::FeatureEncoder` |
| 运动路记忆 | 每窗 33 帧 → 768 维 motion token；时间码 256 维；网格步长 16 | YAML `motion.dim / pos_dim / stride / window_frames` |
| 运动路投影 | `motion_pos_proj` 256→768，`motion_encoder_static` 1536→D | `percep_mem.py::PerceptualMemory.__init__` |
| `motion.budget` | context 口径 96；本轮 ★160 | 两份 YAML |
| D = `memory_token_dim` | context 2048；modulation 1024 | 两份 YAML |
| 记忆序列长度 L_mem = 512 + budget | context 32 帧档 608；本轮 ★672 | `embed_memory` |
| 主干前缀 / 全序列 | context ★1248 / ★1268（= 672 + 512 + 64，再 + 20）；modulation 576 / 596（与基线同） | `embed_prefix`、`compute_loss` |

`motion-memory-interleave.md` 里的 608 / 1184 / 1204 是 context 32 帧 × 16 token + budget 96 的数字；换到本轮 8 帧 × 64 token + budget 160 后，帧路仍是 512 位，记忆区变 672，context 前缀变 1248。

---

## 二、总图：五站流水线，第四站分叉

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ ① dataloader  FrameSampDataset.__getitem__                                              │
 │    帧路 static_* 四键 (512,…)  +  运动路 motion_emb/pos/mask (160,…)  +  mem_order (672,)   │
 └───────────────────────────────────────┬─────────────────────────────────────────────────┘
                                         │  collate → 每键前加 b 维
 ┌───────────────────────────────────────▼─────────────────────────────────────────────────┐
 │ ② PerceptualMemory.__call__   帧路 (b,512,D) ‖ 运动路 (b,160,D) → 并列序 (b,672,D)          │
 └───────────────────────────────────────┬─────────────────────────────────────────────────┘
 ┌───────────────────────────────────────▼─────────────────────────────────────────────────┐
 │ ③ HistoryPi0.embed_memory   take_along_axis(mem_order) → 时间序 mem_seq (b,672,D)、mem_mask │
 └───────────────────────────────────────┬─────────────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┴──────────────────────────────┐
              │ context（D = 2048）                                      │ modulation（D = 1024）
 ┌────────────▼────────────────────────┐              ┌─────────────────▼──────────────────────────┐
 │ ④c embed_prefix 把 672 位排在最前     │              │ ④m 主干序列 = 基线 596 位，记忆不进主干           │
 │    前缀 1248 = 记忆 672+图像 512+文本 64│              │    mem_seq 经 llm(mem_seq=[None, mem_seq])       │
 │    全序列 1268，一张 1268×1268 mask    │              │    只交给 expert 1，nn.broadcast 给 18 层同一份    │
 │    18 层 self-attention，记忆逐层被改写 │              │    每层：20 个动作 token → MemoryAttention 读记忆  │
 │    positions = cumsum(input_mask)−1   │              │    → MemoryRMSNorm 出 scale/shift 调制 FFN 输入   │
 │    padding 不占位置号                  │              │    k 位置 = arange(672)，padding 占号            │
 └────────────┬────────────────────────┘              └─────────────────┬──────────────────────────┘
              └──────────────────────────┬──────────────────────────────┘
 ┌───────────────────────────────────────▼─────────────────────────────────────────────────┐
 │ ⑤ action_out_proj(suffix_out[:, −20:]) → v_t (b,20,32)；loss = mean((v_t − u_t)²)        │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

①②③⑤ 两边逐字同一份代码；`src/mme_vla_suite/policies/` 目录下 grep `integration_type` 零命中，所以在线装配层也不分叉（主计划 I.1 第 3 条）。

---

## 三、共用前半段：dataloader 交付 → 记忆序列 (b,672,D)

### 3.1 dataloader `FrameSampDataset.__getitem__`：离线表 → 帧路 512 位 + 运动路 160 位 + 次序表 (672,)

| 步 | 函数 | 计算 | 输出 |
|---|---|---|---|
| a | `even_sampling_indices(t, 8)` | t < 8 时 `range(t+1)`，否则 `linspace(0, t, 8)` 取整；本库 exec 样本 t ≥ es ≥ 100，恒取满 8 帧 | 8 个帧号 |
| b | 帧库按帧号读行 | 每帧 64 个 patch | (8,64,2048)、(8,64,768) |
| c | reshape + `np.repeat(mask, 64)` | 8 帧 × 64 摊成 512 位 | `static_image_emb` (512,2048)、`static_pos_emb` (512,768)、`static_mask` (512,) 全 True |
| d | `visible_motion_rows(entry, t)` | demo 段起点 s = 16m 且 s + 32 ≤ es − 1；exec 段 u = 16m 且 u + 32 ≤ t − es，全域起点 f = es + u；合并按 f 升序 | k 个表行号与 k 个全域起点（本库 k ∈ [6, 141]★） |
| e | 读 motion 表 k 行、切 `pos_rows(f)[:, 0, :256]`、右填充到 160 | padding 行全零，`pad_times` 把 padding 行的时刻记为 `MEM_ORDER_SENTINEL` | `motion_emb` (160,768)、`motion_pos` (160,256)、`motion_mask` (160,) 前 k 位 True |
| f | `memory_order(帧时刻, 64, motion 时刻)` | 672 个候选位各配键「时刻 × 2 + 类型」（帧 0、motion 1），padding 记哨兵；`argsort(kind="stable")` | `mem_order` (672,) int32，0..671 的置换 |

f 步的效果：motion token 插在其起点 f 之后最近的采样帧之前，同刻帧在前，两路 padding 一并落到尾部。排序规则与图示见 `motion-memory-interleave.md` 二节、4.0 e 步。

### 3.2 `PerceptualMemory.__call__`：两路各自投影后首尾拼成并列序 (b,672,D)

| 步 | 权重 | 输入 | 输出 |
|---|---|---|---|
| 帧路 | `pos_proj` 768→768 + `silu`；与外观拼 2816 维；`encoder_static` 2816→D | (b,512,2048) + (b,512,768) | (b,512,D) |
| 运动路 | `motion_pos_proj` 256→768 + `silu`；与 motion token 拼 1536 维；`motion_encoder_static` 1536→D | (b,160,768) + (b,160,256) | (b,160,D) |
| 拼接 | 沿长度轴 concat（并列序：帧路 512 在前、运动路 160 在后） | 上两行 | (b,672,D) |

进入运动路前有三道形状闸（`motion_emb` 必须是 (b, budget, 768) 等），错一位直接 raise。关闭态（`motion.enabled=false`）两个运动路 Linear 根本不建，`__call__` 在帧路之后早返回 (b,512,D)。

### 3.3 `HistoryPi0.embed_memory`：并列序 → 时间序 mem_seq (b,672,D) 与 mem_mask (b,672)

| 步 | 计算 | 输出 |
|---|---|---|
| mask 拼接 | `input_mask = [static_mask ⊕ motion_mask]` | (b,672)，前 512 + 后 k 位 True |
| 形状闸 | `mem_order.shape[1] == tokens.shape[1] == input_mask.shape[1]` 且 dtype int32，否则 raise | — |
| 重排 | `jnp.take_along_axis(tokens, mem_order[:, :, None], axis=1)`、`jnp.take_along_axis(input_mask, mem_order, axis=1)` | (b,672,D)、(b,672)；重排后前 512 + k 位 True、尾部 160 − k 位 False |
| 常量 mask | `ar_mask`、`na_mask` 各 672 个 False（(L,) 常量，不重排） | — |

返回四元组 `(tokens, input_mask, ar_mask, na_mask)`。**两种注入方式对这四个返回值的用法从这里开始不同**：context 四个都用，modulation 只用前两个（`mem_seq, mem_mask, _, _ = self.embed_memory(observation)`），后两个丢弃。

### 3.4 `embed_prefix` 的图像与文本、`embed_suffix` 的动作：与 `Pi0` 相同

| 段 | 函数 | 输出 | `input_mask` | `ar_mask` | `na_mask` |
|---|---|---|---|---|---|
| 图像视角 1 | `PaliGemma.img` | (b,256,2048) | `image_masks` 广播，恒 True | `[True] + [False]×255` | 全 True |
| 图像视角 2 | 同上 | (b,256,2048) | 恒 True | 全 False | 全 True |
| 文本 | `PaliGemma.llm(method="embed")` | (b,64,2048) | `tokenized_prompt_mask` | 全 False | 全 False |
| 动作（suffix） | `action_in_proj` 32→1024；时间经 `posemb_sincos` + 两层 MLP 得 `adarms_cond` (b,1024) | (b,20,1024) | 全 True | `[True] + [False]×19` | 全 False |

---

## 四、context：记忆拼进主干前缀，随 18 层一起被改写

`embed_prefix` 开头的 `if self.integration_type == "context":` 分支先调 `embed_memory`，把它的四个返回值排在图像之前。下面每步标形状，与 `motion-memory-interleave.md` 四节逐步对应，那里的 608 / 1184 / 1204 换成 672 / 1248 / 1268。

### 4.1 拼接：前缀 (b,1248,2048) + 三条 mask

| 量 | 形状 | 取值 |
|---|---|---|
| `prefix_tokens` | (b,1248,2048) | [记忆 672 ‖ 图像 512 ‖ 文本 64]；记忆段 D 必须等于 2048 才能拼，这就是 context YAML 写 `memory_token_dim: 2048` 的原因 |
| `input_mask` | (b,1268) | 记忆段前 512 + k 位 True、尾部 False；图像 512 位 True；文本按实际长度；动作 20 位 True |
| `ar_mask` | (1268,) | 只有第 672 位（第一张图第一个 token）与第 1248 位（第一个动作 token）True |
| `na_mask` | (1268,) | 第 672 到 1183 位（图像）True，其余 False |

`make_attn_mask(input_mask, ar_mask, na_mask)` 出 (b,1268,1268)：

1. `cumsum(ar_mask)`：记忆段块号 0、图像与文本 1、动作 2；`块号[k] ≤ 块号[q]` 才可见。
2. `valid_mask`：padding 位整行整列 False。
3. `na` 规则：`na_mask` 为 True 的图像 token 与 `cumsum(na_mask) ≤ 0` 的记忆段互相屏蔽。

合起来：**记忆只看记忆；图像看图像与文本、看不到记忆；文本看记忆、图像、文本；动作看一切。**

```
            可见方向（行 = query 段，列 = key 段），○ 可见 × 屏蔽
                      记忆 672   图像 512   文本 64   动作 20
            记忆         ○          ×          ×         ×
            图像         ×          ○          ○         ×
            文本         ○          ○          ○         ×
            动作         ○          ○          ○         ○
```

### 4.2 位置号：`positions = cumsum(input_mask) − 1` (b,1268)，padding 不占号

记忆段真 token 拿到 0 … 511 + k，尾部 padding 位不推进计数，图像第一个 token 拿到 512 + k。k 随样本变，所以图像、文本、动作的位置号也随 k 整体平移，但它们相互之间的差不变，RoPE 只看差。

### 4.3 主干 `PaliGemma.llm([prefix, suffix])`：每层七步，记忆与其他 token 同一张 attention

| 步 | expert 0（前缀，含记忆） | expert 1（动作） | 合并 |
|---|---|---|---|
| g 归一化 | `RMSNorm` (b,1248,2048) | 自适应 `RMSNorm` 吃 `adarms_cond` (b,20,1024)，另出 gate | — |
| h 投影 | `q_einsum` (8,2048,256)、`kv_einsum` (2,1,2048,256) → q (b,1248,8,256)、k/v (b,1248,1,256) | `q_einsum_1`、`kv_einsum_1` → q (b,20,8,256)、k/v (b,20,1,256) | 沿长度轴 concat → q (b,1268,8,256)、k/v (b,1268,1,256) |
| i RoPE | `_apply_rope(q, positions)`、`_apply_rope(k, positions)`，q 再乘 1/16 | 同 | 形状不变 |
| j 打分 | — | — | logits (b,1,8,1268,1268)；`where(mask, logits, −2.38e38)`；softmax；乘 v → (b,1268,8,256) |
| k 输出投影 | `attn_vec_einsum` (8,256,2048) → (b,1248,2048) | `attn_vec_einsum_1` → (b,20,1024) | — |
| l 残差 | `x + y` | `x + y × gate` | — |
| m 前馈 | `RMSNorm` → `FeedForward` 2048→16384→2048 → 残差 | 自适应 `RMSNorm` → 1024→4096→1024 → 门控残差 | — |

记忆段的 672 行在 expert 0 里，每层的 h–m 都作用在它身上：第 2 层看到的记忆已经是第 1 层改写过的。这就是「记忆走 18 层残差流、逐层被加工」的意思。

### 4.4 推理：kv_cache (18,b,1248,1,256) × 2，记忆烙进缓存

`sample_actions` 的 `else` 分支先跑 `llm([prefix_tokens, None])` 存缓存，缓存长度随 budget 变；去噪 10 步每步只算 20 个动作 token 的 q、k、v，与缓存里的 1248 个 k、v 打分，logits (b,1,8,20,1268)。记忆在 10 步里不再被触碰。

---

## 五、modulation：记忆走旁路，只被 20 个动作 token 读

### 5.1 主干序列与基线逐位相同：前缀 576、全序列 596

`embed_prefix` 里的 context 分支不执行，`prefix_tokens` (b,576,2048)、`input_mask` (b,596)、`ar_mask` 只有第 0 位与第 576 位 True、`na_mask` 前 512 位 True。`make_attn_mask` 出 (b,596,596)，`positions = cumsum − 1` (b,596)，全部与 motion 关闭态、也与基线 `perceptual-framesamp-modul-8frame-8x8.yaml` run 逐位相同。

`na_mask` 在这里是恒等空操作：`cumsum(na_mask) ≤ 0` 的位置集合为空（第 0 位就是图像、就是 True），`mask_not_attend` 全 False。`compute_loss` 传三参、`sample_actions` 的 modulation 分支传两参，两者等价，不是训推不一致（主计划 I.1 表 n 行）。

### 5.2 记忆的入口：`llm(…, mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])`

`compute_loss` 的 modulation 分支单独调 `embed_memory` 取前两个返回值，以列表形式传给 `PaliGemma.llm`：第 0 元 `None` 给 expert 0，第 1 元给 expert 1。`history_gemma.py::Module.setup` 的 `nn.scan` 把 `mem_seq`、`mem_mask` 声明为 `nn.broadcast`（`in_axes` 第 5、6 位），18 层拿到的是**同一份** (b,672,1024)，不像 `xs` 那样逐层传递更新。

### 5.3 每层 `HistoryBlock.__call__`：expert 1 那条流多两步

只看 expert 1（x 是 (b,20,1024)），expert 0 那条流与基线逐步相同、没有任何一行碰 `mem_seq`（`if i == len(xs) − 1 and self.integration_type == "modulation"` 只对最后一条流成立）：

| 步 | 计算 | 输出 |
|---|---|---|
| g | 自适应 `RMSNorm(pre_attention_norm_1)` 吃 `adarms_cond`，另出 gate | (b,20,1024) |
| h–k | 与四节相同的共享 self-attention，但序列只有 596：q (b,20,8,256) 与 k/v (b,596,1,256) 打分 | (b,20,1024) |
| l | 门控残差 `x + y × gate` | x₁ (b,20,1024) |
| **★ m1** | `mem_attn(x₁, mem_seq, mem_mask)`（`MemoryAttention`，5.4） | `mem_mod_vec` (b,20,1024) |
| **★ m2** | `MemoryRMSNorm("mem_rms_norm_ffn")(x₁, cond=mem_mod_vec)`（5.5） | x₂ (b,20,1024) |
| n | 自适应 `RMSNorm(pre_ffw_norm_1)(x₂)` → `FeedForward` 1024→4096→1024 | y (b,20,1024) |
| o | 门控残差 **`x₁ + y × gate`** | (b,20,1024) |

o 步残差的另一端是 x₁ 而不是 x₂：`xs` 在 m1 之前已经定下，m2 的输出只送进 FFN 分支。所以调制改变的是「这一层 FFN 看到什么」，不直接覆写残差流。

### 5.4 `MemoryAttention.__call__`：20 个动作 token 对 672 位记忆做 4 头 cross-attention

写死 `num_heads=4, num_kv_heads=1, head_dim=256, width=1024`，并 `assert mem_width == x_width == width`——这就是 modulation YAML 必须写 `memory_token_dim: 1024` 的原因，也是 CPU dummy（宽 64）撞断言、V5 只能用 `gemma_150m`（宽 1024）替身的原因。

| 行 | 权重 / 计算 | 输入 | 输出 |
|---|---|---|---|
| 1 | `mem_rms_norm(x)`：无 cond 分支，`x/√(mean x²+1e−6) × (1+scale)`，`scale` (1024,) | x₁ (b,20,1024) | (b,20,1024) |
| 2 | `q_einsum_mem` (4,1024,256) | 上一行 | q (b,20,4,256) |
| 3 | `mem_rms_norm(mem_seq)`：**同一个 `scale` 参数**再作用于记忆序列 | (b,672,1024) | (b,672,1024) |
| 4 | `kv_einsum_mem` (2,1,1024,256) | 上一行 | k、v 各 (b,672,1,256) |
| 5 | `q_positions = arange(672, 692)`、`k_positions = arange(0, 672)`，各 `einops.repeat` 到 (b,·) | — | (b,20)、(b,672) |
| 6 | `_apply_rope(q, q_positions)`，再 `q *= 1/16`；`_apply_rope(k, k_positions)` | — | 形状不变 |
| 7 | `einsum("BTKGH,BSKH->BKGTS")` | q (b,20,1,4,256)、k | logits (b,1,4,20,672) |
| 8 | `attn_mask = mem_mask[:, None, None, None, :]`；`where(attn_mask, logits, −2.3819763e38)` | (b,1,1,1,672) | (b,1,4,20,672) |
| 9 | softmax 沿最后一维；`einsum(probs, v)`；合并头 | — | (b,20,4,256) → (b,20,1024) |
| 10 | `out_einsum_mem` (4,256,1024) | 上一行 | `mem_mod_vec` (b,20,1024) |

第 5、6 行是本文全部结论的核心：**k 的位置号是 `arange(672)`，padding 行照样占号；q 的位置号从 `mem_len` 起跳**。第 8 行保证被 mask 的 672 − 512 − k 个 padding 列 softmax 权重严格为 0，所以 padding 行里塞什么值都不影响输出（主计划 V5 的 `PAD_CONTENT_INVARIANCE`）。记忆序列只出 k、v，不出 q：记忆 token 之间没有 attention，第 3 行看到的 `mem_seq` 在 18 层里都是 `embed_memory` 输出的原值。

### 5.5 `MemoryRMSNorm(x₁, cond)`：读出向量 → scale、shift → 调制

| 行 | 计算 | 输出 |
|---|---|---|
| 1 | `normed = x₁ / √(mean(x₁²) + 1e−6)`（无学习参数） | (b,20,1024) |
| 2 | `Dense_0` 1024→2048 作用于 `cond = mem_mod_vec`，`kernel_init = normal(stddev=0.002)`（`representation/utils.py::kernel_init_out_proj`），bias 默认零 | (b,20,2048) |
| 3 | `split` 成 scale、shift 各 (b,20,1024) | — |
| 4 | `normed × (1 + scale) + shift` | x₂ (b,20,1024) |

初始化时 kernel 元素量级 0.002，`cond` 经 RMSNorm 与 lecun 初始化的投影后量级 O(1)，所以 scale、shift 初始约 O(0.06)、远小于 1：x₂ ≈ `normed(x₁)`，记忆对主干几乎零贡献，「门」要靠训练打开。这一层与 `mem_attn` 共 6 个参数叶，每层一套：

| 叶 | 形状 | 参数量 |
|---|---|---|
| `mem_attn/q_einsum_mem/w` | (4,1024,256) | 1,048,576 |
| `mem_attn/kv_einsum_mem/w` | (2,1,1024,256) | 524,288 |
| `mem_attn/out_einsum_mem/w` | (4,256,1024) | 1,048,576 |
| `mem_attn/mem_rms_norm/scale` | (1024,) | 1,024 |
| `mem_rms_norm_ffn/Dense_0/kernel` | (1024,2048) | 2,097,152 |
| `mem_rms_norm_ffn/Dense_0/bias` | (2048,) | 2,048 |

每层 4,721,664，18 层 84,989,952 ≈ 85 M。**这六个叶由 `integration_type: modulation` 决定，与 motion 开不开无关**——基线 run（motion 关闭）已经有它们，只是那时 `mem_seq` 只有 512 个帧 token。

### 5.6 推理：kv_cache 与基线同长，记忆每步每层重算

`sample_actions` 的 modulation 分支：先 `llm([prefix_tokens, None])` 存 kv_cache (18,b,576,1,256) × 2（与基线同长，不含记忆），再调一次 `embed_memory` 得 `mem_seq`、`mem_mask`；去噪 `step` 里每步 `llm([None, suffix_tokens], kv_cache=…, mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])`。`MemoryAttention` 没有缓存：5.4 的第 3、4 行（`mem_rms_norm(mem_seq)` 与 `kv_einsum_mem`）在 10 步 × 18 层 = 180 次前向里每次重算。算量按主计划 I.1 表 o 行：S = 672 时每次 infer 约 75.9 GMAC，比 S = 512 多 16.3 GMAC（+27%），折合每集约 +9 ms（推算未实测）；评估耗时的大头仍是 sidecar 逐窗编码。

### 5.7 梯度流向

loss 只读 `suffix_out[:, −20:]`。记忆到 loss 的唯一通路是 5.3 的 m1 → m2 → FFN → o 步残差，所以四个 motion 叶与帧路投影的梯度**只来自 20 个动作 query**，且要穿过 `Dense_0` 这道初始近零的门。context 下记忆段在 expert 0 里，文本与动作的全部 query 都给它回传梯度。「motion 能不能被学到」因此不能靠接线正确推断，主计划用 V8 的 `MOTION_PARAMS_UPDATED` 与训完后的全遮消融回答。

---

## 六、两种注入方式逐项对照

| # | 维度 | context | modulation |
|---|---|---|---|
| a | 记忆去哪 | 拼进 expert 0 前缀，(b,1248,2048) | 不进主干；(b,672,1024) 经 `mem_seq=[None, ·]` 只给 expert 1 |
| b | 谁看得到 | 文本、动作（图像被 `na_mask` 挡） | 只有 20 个动作 token，单向 cross-attention |
| c | 记忆是否被逐层改写 | 是，18 层残差流 | 否，`nn.broadcast` 同一份；记忆 token 之间无 attention |
| d | attention 形状 | logits (b,1,8,1268,1268) | 主干 (b,1,8,596,596) + 旁路 (b,1,4,20,672) |
| e | 位置号 | `cumsum(input_mask) − 1`，padding 不占号 | 主干同左；旁路 `k = arange(672)`、`q = arange(672, 692)`，padding 占号 |
| f | mask | 三条 (`input / ar / na`) 合成 | 主干两条等价三条；旁路只有 K 侧一维 `mem_mask` |
| g | 记忆宽度 D | 2048 | 1024 |
| h | 运动路新增参数 | 3,345,152（`motion_encoder_static` 1536→2048） | 1,771,264（1536→1024） |
| i | 初始贡献 | 记忆一进主干就与其他 token 等权参与 attention，无门 | scale/shift 由 stddev 0.002 的 Dense 出，初始近零 |
| j | 梯度来源 | 文本 + 动作全部 query | 20 个动作 query，经门 |
| k | kv_cache | (18,b,1248,1,256)，随 budget 变长 | (18,b,576,1,256)，与基线同 |
| l | 去噪 10 步里记忆的算量 | 0（已在缓存里） | 每步每层重算 k、v |

带代码锚点的完整版见主计划 I.1「4.2 九项不等价」与「4.3 推理侧差异」。

---

## 七、加了 motion 之后，模型里多了什么

分「数据侧 / 参数侧 / 序列侧」三层说，每层都区分「谁决定它存在」。

### 7.1 数据侧：四个交付键，由 `motion.enabled` 决定

关闭态 `framesamp_dataset.py` 的 `_NONE_KEYS` 把 `motion_emb / motion_pos / motion_mask / mem_order` 四键恒交付 None；开启态按 3.1 的 d–f 步构造。这一层对 context 与 modulation 完全相同，在线侧 `_prepare_history` 用同一份 `memory_order` 函数。

### 7.2 参数侧：两组叶，两个开关

| 组 | 由谁决定存在 | 叶 | 参数量 | 基线 run 有没有 | 新 run 有没有 |
|---|---|---|---|---|---|
| modulation 六叶（5.5 表） | `integration_type: modulation` | `mem_attn/*` 四叶 + `mem_rms_norm_ffn/Dense_0/*` 两叶，18 层堆叠 | ≈ 85 M | **有** | 有 |
| motion 四叶 | `motion.enabled: true` | `mem_encoder/motion_pos_proj/{kernel (256,768), bias (768,)}`、`mem_encoder/motion_encoder_static/{kernel (1536,1024), bias (1024,)}` | 1,771,264 | **没有** | 有 |

参数树顶层叶数 61 → 65（主计划 V9 的 `PARAM_TREE_EXACT … n_model=65`），多出来的正是 motion 四叶。`percep_mem.py::PerceptualMemory.__init__` 的 `if self.motion_enabled:` 是编译期 Python 分支，关闭态这四个数组不存在，模型与基线的参数树逐位相同——这是主计划八节 A 组「关闭态一个比特都不能变」能成立的前提。

### 7.3 序列侧：记忆长度 512 → 672，两边后果不同

| | context | modulation |
|---|---|---|
| 变长的是什么 | 主干前缀 1088 → 1248，全序列 1108 → 1268，kv_cache 同步变长 | 主干不变（596）；旁路 `mem_len` 512 → 672 |
| 位置号后果 | 记忆段真 token 0 … 511 + k，后面各段整体平移 k，各段内部差不变 | 旁路 q 起跳位从 512 变 672（八节） |
| 算量后果 | attention 1268² 对 1108²，约 +31% | 主干不变；旁路 +27%，绝对量小 |

### 7.4 关闭态：两边都退化成各自的基线

`embed_memory` 关闭态早返回 `(tokens (b,512,D), static_mask, [False]×512, [False]×512)`，不追加、不重排。context 下记忆段 512 位进前缀，即 `HistoryPi0` 原样；modulation 下 `mem_seq` (b,512,1024) 走旁路，即基线 `perceptual-framesamp-modul-8frame-8x8` run 原样。主计划 V1 / V6 / V7 证的就是这一句。

---

## 八、为什么 `motion.budget` 在 modulation 下是模型语义参数

### 8.1 budget 不出现在任何权重形状里

七节两张表里十个叶的形状由 `pos_dim=256`、`dim=768`、`pos.hidden_dim=768`、`memory_token_dim=1024`、`width=1024`、`head_dim=256` 决定，没有一个含 160。budget 只决定**输入张量**的行数：`motion_emb` 是 (b,160,768) 还是 (b,96,768)，`mem_seq` 是 (b,672,·) 还是 (b,608,·)。所以拿 budget=96 的 YAML 建模型去加载 budget=160 训出的 checkpoint，`PARAM_TREE_EXACT` 逐叶名字形状全对、照样 PASS。

### 8.2 context 下 budget 只是容量：padding 不占号

四节 4.2：`positions = cumsum(input_mask) − 1`。budget 从 96 改 160，只是尾部多 64 个 False 位，它们不推进计数，真 token 的位置号一个不变，图像文本动作的位置号也不变。多留的坑对前向计算零影响，改 budget 只改显存与算量。

```
 context，同一样本 k = 6，位置号 = cumsum − 1
 budget 96 : [帧+motion 真 518 位: 0…517][pad 90 位: 不占号][图像: 518…1029][文本: 1030…][动作: …]
 budget 160: [帧+motion 真 518 位: 0…517][pad 154 位: 不占号][图像: 518…1029][文本: 1030…][动作: …]
             ──────────────── 两行的每个真 token 位置号完全相同 ────────────────
```

### 8.3 modulation 下 budget 决定 RoPE 起跳位：padding 占号、query 整体平移

5.4 第 5 行：`k_positions = arange(mem_len)`、`q_positions = arange(mem_len, mem_len + 20)`，`mem_len = 512 + budget`。padding 排在尾部（3.1 f 步的哨兵保证），真 token 的 k 位置号 0 … 511 + k 不随 budget 变；**变的是 q 的起点**。`_apply_rope` 是纯相对位置旋转，q_i · k_j 只取决于 `q_pos − k_pos`，所以：

```
 modulation，同一样本 k = 6，旁路位置号 = arange
 budget 96 : k 位置 [真 518 位: 0…517][pad 90 位: 518…607]        q 位置 [608…627]
              q₀ 到最后一个真 k 的差 = 608 − 517 = 91
 budget 160: k 位置 [真 518 位: 0…517][pad 154 位: 518…671]       q 位置 [672…691]
              q₀ 到最后一个真 k 的差 = 672 − 517 = 155
             ──────── 同一份记忆内容、同一套权重，20 × 518 个 q·k 相对距离全部 +64 ────────
```

训练在 672 起跳下学出的 `q_einsum_mem` / `kv_einsum_mem` 权重，评估时若 YAML 写 96、q 从 608 起跳，每个 q·k 的旋转角都不一样，attention 打分全变，输出随之变。这和改 depth、改 width 是同一性质的「换模型」，只是参数树看不出来。

### 8.4 顺带的旁路信号：q 到最后一个真 token 的距离 = budget − k + 1

真 token 占 0 … 511 + k，q₀ 在 512 + budget，两者之差 = budget − k + 1；padding 行数 = budget − k。k 是当前样本能看到的 motion 窗数，逐样本不同（本库 6 … 141★，对应距离 20 … 155）。这个距离与 motion 内容无关、只随 k 跳动，模型可以从中读出「现在看到了几个窗」——一个 context 下不存在的信号（那里 padding 不占号，q 到最后一个真 token 永远是固定差）。改 budget 就改了这个信号的取值范围。主计划第四节把它记为 `budget − k`，本文按 `arange` 端点写成 budget − k + 1，差 1 只是计数约定。

### 8.5 后果与守卫

- **三侧逐值相同**：训练、评估、在线任何一侧改 budget 都等同换模型；`train.py` 硬禁 `--resume`，不存在 resume 侧。
- **参数树守不住**（8.1），主计划 V9 加 `BUDGET_CONSISTENT`：run 内 `history_config.resolved.yaml` vs 仓库新 YAML vs 期望值 160 三方跨源比对；在线侧 budget 归 V-online 的 CPU 前置门（断言 `_motion_cfg["budget"] == 160`）。
- **生产评估有结构性保护**：`create_trained_policy` 直接丢弃外部传入的 history_config、整体换成 run 目录内的快照，正常评估路径用不错 budget；真实暴露面是按仓库 YAML 文件名建模型的 `check_ckpt_param_tree.py` 这类 `--yaml` 驱动工具。
- **同 budget 下 padding 内容无关**：5.4 第 8 行的 `where(mem_mask, logits, −2.38e38)` 让 padding 列 softmax 权重恰为 0，主计划 V5 的 `PAD_CONTENT_INVARIANCE` 证这一条；`ROPE_LEN_EFFECT` 探针量化 608 → 672 的输出变化幅度。

---

## 九、为什么训完必须做全遮消融

新 run（modulation + motion，`mem_len` 672）与基线（modulation、无 motion，`mem_len` 512）的差分里混了两件事：一是 motion 内容有没有用，二是 8.3 的 q 起跳位 512 → 672 改了位置编码。两者在同一次训练里不可分。主计划拍板的消融臂：训完在**同一 ckpt** 上把 `motion_mask` 整列置 False 再评一次——同权重、同 `mem_len` 672、同 q 位置，只是 160 个 motion 列全部被 5.4 第 8 行屏蔽、softmax 权重为 0（帧路 `static_mask` 恒有 True，不存在全 mask 退化）。这一臂与正常评估的差才是 motion 内容的净贡献。`docs/motion-utilization.md` 的 +5.00pp 是 context 口径、`scripts/motion-variance` 硬编码 `integration_type="context"` 与 `PREFIX_LEN 1184`，结构性不适用于本轮，不得引用。

---

## 十、已知的未知与主计划落点

- **能不能学到**：5.7 的两条动力学后果（只有 20 个 query 回传、初始门近零）没有先验证据说一定能学出来；由 V8 `MOTION_PARAMS_UPDATED`（4 叶 sha 随步变化）与九节消融回答。
- **RoPE 平移的幅度**：主计划第三轮引用的五个数字（0.33 / 0.76 / 0.87 / ≈0.4 / 1e−17）无留档来源，V0 阶段用 `ROPE_LEN_EFFECT` 探针重取后才写入正本。
- **`mem_rms_norm` 共用 scale**：5.4 第 1、3 行同一个 (1024,) `scale` 同时作用于动作隐状态与记忆序列，两者分布不同，是否合适没有验证。
- **主计划落点**：本文对应 `0916-motion-modul-8x8-plan.md` 第四节（结论与六层等价）、I.1（差异矩阵与代码锚点）、I.2（`INIT_COMMON` / `ROPE_LEN_EFFECT` / 消融臂判据）、I.3（措辞纠错表）、I.4（budget 语义参数的技术表述）、I.5（推理侧算量）。正本 `docs/motion-memory.md` 3.1 / 3.4 / 3.5 / 4.2 / 5.3 / 5.4 仍是 context 口径，待实施后按 I.3 清单另立 `docs:` 改写。
