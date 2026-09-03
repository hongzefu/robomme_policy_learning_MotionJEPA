# motion memory 交错拼接说明——运动路按时间插进帧路

> **本文性质**：说明文档，只负责把「交错拼接」方案从训练链路第一步讲到 gemma 内部，再补在线推理；不是实施授权，不写实现细节。
> **对应主计划**：`motion-memory-plan.md`。主计划已于 2026-09-02 按本文改为交错方案（同时 `motion.stride` 20 → 16、`motion.budget` 80 → 96），本文随之换档；两文冲突以主计划为准。
> **定稿日期**：2026-09-02（同日随主计划换档）。**代码锚点**：HEAD `4503ea2` 不变，本文写作期间 `src/` `scripts/` 零改动。

**一句话结论**：与并列方案相比，交错方案送进模型的 token 内容、权重、mask、计算量全部相同；数学上唯一的区别是 memory 段 608 个 token 各自拿到的 RoPE 位置号；实现上多一步排序、一步重排、一个交付键。

---

## 一、三条已定口径

本轮用户拍板的三条，原话保留：

1. 「帧路不动！只是交错拼接运动路！」——`even_sampling_indices(t, 32)` 一字不动，仍是 32 帧 × 16 token = 512 位；运动路仍是最多 96 个 motion token（主计划 2.3，预算按 16 任务全集零截断定 96）。
2. 「按起点 f 插入」——一个 motion token 描述窗口 [f, f+32] 的运动，插入位置由起点 f 决定，不按尾端、不按中点。
3. 「运动起点和采样帧号相同时，采样帧在前」——同一时刻既有采样帧又有 motion 起点时，先放该帧的 16 个 token，再放这个 motion token。

附带两条不需要单独拍板的事实：起点网格是段内绝对位置 0, 16, 32, …（stride 16 = 推理阶段一个 action chunk 的执行长度，主计划 2.1）；padding 全部落在尾部——这是排序的自然结果（见四节 4.0 的 e 步与五节 5.2 的 k 步），不是额外要求，也不是正确性所需。

---

## 二、数轴：并列 vs 交错——同样 523 个真 token 的两种排法

取当前帧 t = 200、段起点 0 的样本。帧路 32 帧全满，运动路合法起点 0, 16, …, 160 共 11 个（160+32 = 192 ≤ 200，176+32 = 208 > 200）。两种方案的 token 一个不多一个不少，只是顺序换了。

```
时间轴（帧号）
0    6   12  19  25  32  38  45  51  58  64  70  77  83  90  96  103 109 116 122 129 135 141 148 154 161 167 174 180 187 193 200
▮    ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ▮   ← 帧路，每 ▮ = 16 token
●        ●        ●        ●          ●          ●          ●          ●          ●          ●          ●                     ← 运动路，每 ● = 1 token
0        16       32       48         64         80         96         112        128        144        160

━━━━━━━━━━ 并列（旧方案，主计划已不采用）━━━━━━━━━━
┌── 帧路 512 位 ──────────────────────────────────┬── 运动路 96 位 ─────────┐
│ f0 f6 f12 … f193 f200（32 帧 × 16）             │ m0 m16 … m160 │ pad×85 │
└─────────────────────────────────────────────────┴───────────────┴────────┘
  位 0–511                                          位 512–522     位 523–607

━━━━━━━━━━ 交错（本文方案，主计划现行）━━━━━━━━━━
f0×16 m0 f6×16 f12×16 m16 f19×16 f25×16 f32×16 m32 f38×16 f45×16 m48 f51×16 f58×16 f64×16 m64 f70×16 f77×16 m80
f83×16 f90×16 f96×16 m96 f103×16 f109×16 m112 f116×16 f122×16 m128 f129×16 f135×16 f141×16 m144 f148×16 f154×16 m160
f161×16 f167×16 f174×16 f180×16 f187×16 f193×16 f200×16 │ pad×85
└──────────────────── 523 个 True，同样的 523 个 token ────────────────────┘ 位 523–607 False
```

两条规律从图上就能看出来：motion 落在起点之后最近的那个采样帧之前（m160 在 f154 之后、f161 之前；m128 在 f122 之后、f129 之前）；同刻的位置在这个样本里有四处——起点 0、32、64、96 恰好也是采样帧（stride 16 整除 32，且 linspace 在 t = 200 时落在这些帧上），按口径 3 帧在前，所以 m0 排在 f0 的 16 个 token 之后，m32、m64、m96 同理。

再看一个帧路没填满的样本，t = 5：帧路只有 6 帧 96 位，运动路一个合法起点都没有。

```
并列：[f0 … f5 共 96 位 │ 帧路 pad×416 │ 运动路 pad×96]     True 在位 0–95，False 在 96–511 ∪ 512–607
交错：[f0 … f5 共 96 位 │ pad×512]                          True 在位 0–95，False 在 96–607
```

两种布局 True 的集合完全相同。并列方案里帧路自己的 padding 卡在段中间，交错方案里所有 padding 一并排到尾部。

---

## 三、三层代码：openpi `Pi0` → `HistoryPi0` → 加 motion memory——前缀从 576 位到 1088 位再到 1184 位，各层改了哪些函数

后面两节讲数值计算时，每一步都标注它是哪一层加的。本节先把三层的关系讲清：每一层在上一层的哪些函数上改了什么、加了什么，没提到的函数就是一字不改。

### 3.1 底层 `Pi0`：图像 2×(b,256,2048) + 文本 (b,64,2048) + 带噪动作 (b,20,1024) → 速度 (b,20,32)（`src/openpi/models/pi0.py`）

模块四件：`PaliGemma.img`（SigLIP 图像编码器）、`PaliGemma.llm`（gemma `Module`，装两个 expert）、动作侧两个 Linear（`action_in_proj` 32→1024、`action_out_proj` 1024→32）与时间 MLP 两个（`time_mlp_in`、`time_mlp_out`，pi05 才有；本 run pi05=True，`else` 分支的 state 投影等三件不建）。

函数五个：

| 函数 | 做什么 | 返回 |
|---|---|---|
| `embed_prefix` | 图像过 SigLIP、文本查词表，拼成前缀 | tokens、`input_mask`、`ar_mask` 三元组 |
| `embed_suffix` | 带噪动作过 `action_in_proj`；时间过 `posemb_sincos` 与时间 MLP 得 `adarms_cond` | tokens、`input_mask`、`ar_mask`、`adarms_cond` |
| `make_attn_mask(input_mask, ar_mask)` | 块号规则 ∧ padding 外积 | (b,L,L) 布尔表 |
| `compute_loss` | prefix 与 suffix 拼成一条序列一次前向，`positions = cumsum(input_mask) − 1`，`action_out_proj` 得速度，与 `noise − actions` 算均方误差 | (b,20) |
| `sample_actions` | 先跑 prefix 存 kv_cache，再 `while_loop` 去噪，每步只算 suffix | (b,20,32) |

gemma 侧（`src/openpi/models/gemma.py`）：`Module` 用 `nn.scan` 叠 18 个 `Block`；每个 `Block` = `RMSNorm`（expert 0 普通版，expert 1 自适应版吃 `adarms_cond`）→ `Attention`（含 `_apply_rope`）→ `_gated_residual` → `RMSNorm` → `FeedForward` → `_gated_residual`。expert 0 是 PaliGemma（宽 2048，作用于 prefix），expert 1 是 action expert（宽 1024，作用于 suffix），各有一套权重，只在 `Attention` 里把两段的 q、k、v 沿长度轴拼起来打分时交汇。位置信息只有 `_apply_rope` 这一种一维 RoPE，作用在 q 与 k 上，不往 token 内容里加位置向量。

### 3.2 中层 `HistoryPi0`：前缀最前加 512 位记忆 token (b,512,2048)，prefix 576 → 1088（`src/mme_vla_suite/models/integration/history_pi0.py`）

`HistoryPi0` 不继承 `Pi0`，而是把 `Pi0` 的 `__init__` 与 `embed_prefix` / `embed_suffix` / `compute_loss` / `sample_actions` 四个方法复制一份再改，外加模块级 `make_attn_mask` 也复制一份改成三参，共六处；`posemb_sincos` 直接从 `openpi.models.pi0` import、不复制；另有 `Pi0` 没有的 `vision_encode`（在线侧现编帧特征用）。逐项对照：

| 项 | `Pi0` | `HistoryPi0` |
|---|---|---|
| 观测类型 | `Observation` | `HistAugObservation`，多四个字段 `static_image_emb` / `static_pos_emb` / `static_state_emb` / `static_mask`；`preprocess_observation` 包一层，把这四键原样透传 |
| 新模块 | 无 | `mem_encoder = PerceptualMemory`（`percep_mem.py`），内含 `FeatureEncoder`（`mem_encoder.py`）的两个 Linear：`pos_proj` 768→768、`encoder_static` 2816→2048；`use_state_emb=False` 时 `state_proj` 根本不建 |
| `embed_memory` | 无 | 新增：`mem_encoder` 把 512 位记忆编成 (b,512,2048)；`input_mask = static_mask`；`ar_mask`、`na_mask` 各 512 个 False |
| `embed_prefix` | 图像 + 文本；图像 `ar_mask` 全 False | context 模式下先把 `embed_memory` 的记忆段排在最前；第一个图像 token 的 `ar_mask` 改为 True（记忆块与图像块的分界）；新增 `na_mask`（图像段 True，其余 False）；返回四元组 |
| `embed_suffix` | 三元组 + `adarms_cond` | 多返回一条全 False 的 `na_mask`，其余逐字同 `Pi0` |
| `make_attn_mask` | 两参 | 三参，多 `mask_na`：`na_mask` 为 True 的 token（图像）不能看 `cumsum(na_mask) ≤ 0` 的位置（记忆段） |
| `compute_loss` / `sample_actions` | 两条 mask | 三条 mask；`compute_loss` 按 `use_history`、`sample_actions` 按 `integration_type == "context"` 调三参 `make_attn_mask`（本 run 两者同真，行为一致）；另有 `expert` / `modulation` 两个分支，本链路走 `context` 不涉及 |
| gemma | `gemma.Module` / `Block` | `history_gemma.py` 复制一份 `Module`，`Block` 改名 `HistoryBlock`，多 `mem_seq` / `mem_mask` 两参与 `MemoryAttention`（只在 modulation 模式生效）；context 模式下 `HistoryBlock` 与 openpi `Block` 逐步相同，`Attention`、`RMSNorm`、`FeedForward`、`_apply_rope`、`_gated_residual` 都直接 import 自 openpi |
| 序列长度 | prefix 576 = 图像 512 + 文本 64；全序列 596 | prefix 1088 = 记忆 512 + 图像 512 + 文本 64；全序列 1108 |
| 训练数据 | 图像、文本、动作 | 多四个 `static_*` 键，由 `FrameSampDataset.__getitem__` 从离线表读出 |
| 在线推理 | 直接喂图像 | `MME_VLA_Policy` 持 `FrameSampMemory`，每批现编当前批帧特征、按同一选帧函数组装四个 `static_*` 键 |

块号规则合起来的效果：记忆只看记忆；图像看图像与文本、看不到记忆；文本看记忆、图像、文本；动作看一切。

### 3.3 上层 motion memory：记忆 512 → 608 位（并 96 个运动 token），★ 交错再加一张 (b,608) 次序表

并列方案与交错方案共有的部分：

| 项 | 改动 |
|---|---|
| 观测 | `HistAugObservation` 多三字段 `motion_emb` (b,96,768)、`motion_pos` (b,96,256)、`motion_mask` (b,96)；`preprocess_observation` 透传 |
| `PerceptualMemory.__init__` | `motion.enabled` 时在 `feature_encoder` 之后条件新建两个 Linear：`motion_pos_proj` 256→768、`motion_encoder_static` 1536→2048；关闭时根本不建 |
| `PerceptualMemory.__call__` | 多运动分支：`motion_pos` 过 `motion_pos_proj` + `silu`，与 `motion_emb` 拼 1536 维，过 `motion_encoder_static` 得 (b,96,2048)；与帧路 (b,512,2048) 沿长度轴拼成 (b,608,2048)（并列序） |
| `embed_memory` | `input_mask` 拼成 `[static_mask ⊕ motion_mask]` (b,608)；`ar_mask`、`na_mask` 各追加 96 个 False |
| 其余函数 | `embed_prefix`、`embed_suffix`、`make_attn_mask`、`compute_loss`、`sample_actions`、gemma 全部一字不改，记忆段长度 608 自动透传：prefix 1184、全序列 1204 |
| 训练数据 | `FrameSampDataset.__getitem__` 多运动路三键（查 motion 表、切时间码、右填充到 96） |
| 在线推理 | `FrameSampMemory` 多 `_prepare_motion` 与每批（16 帧 = 推理阶段一个 action chunk）一次的增量编码 |
| 关闭态 | `motion.enabled=false` 时两个 Linear 不建、三键为 None，与 `HistoryPi0` 逐位相同 |

★ 交错方案独有的部分：

| 项 | 改动 |
|---|---|
| 观测 | 再多一个字段 `mem_order` (b,608) int32（`at.Int`、新维名，取值为 0..607 的置换） |
| `embed_memory` | 拼成 608 之后，用 `jnp.take_along_axis(…, mem_order)` 把 token 与 `input_mask` 一起按时间序重排；`ar_mask` / `na_mask` 是 (L,) 常量、记忆段恒 False，不重排 |
| 训练数据 | `__getitem__` 末尾对 608 个候选位按 (全域时刻, 类型) 稳定排序得 `mem_order`，并显式 raise 校验它是置换 |
| 在线推理 | `_prepare_history` 用同一个排序函数得同一张表 |

除此之外交错方案与并列方案没有任何代码差异。

---

## 四、训练：一个 batch 从 dataloader 交付的键到标量 loss 与梯度

本节按顺序列出训练一步的每次数值计算，每步给函数与输入输出形状；〔〕标出三节的哪一层，★ 标出交错方案与并列方案有差别的地方。常量：batch 记 b，主干宽 2048，action expert 宽 1024，18 层，8 个 query 头、1 个 key/value 头、每头 256 维；示例样本 t = 200、es = 0，memory 段 523 个真 token。

### 4.0 数据侧 `FrameSampDataset.__getitem__`：离线表 → 每样本帧路 512 位 + 运动路 96 位 + 次序表 (608,)〔dataloader，motion 新增运动路〕

| 步 | 函数 | 计算 | 输出 |
|---|---|---|---|
| a | `even_sampling_indices(step, 32)` | t < 32 时 `range(t+1)`，t ≥ 32 时 `linspace(0, t, 32)` 取整（t = 31 两分支等值） | n 个帧号 |
| b | `FrameSampStore.read_image_rows` / `pos_rows` / `state_rows` | 按帧号读离线表 | (n,16,2048)、(n,16,768)、(n,8) |
| c | `_pad(img, pos, stt, n)` | 前 n 行放真数据，后 32−n 行写 0；帧级 mask 前 n 位 True | (32,16,2048)、(32,16,768)、(32,8)、(32,) |
| d | reshape + `np.repeat(mask, 16)` | 32 帧 × 16 token 摊成 512 位 | `static_image_emb` (512,2048)、`static_pos_emb` (512,768)、`static_state_emb` (512,8)、`static_mask` (512,) |
| d′〔motion〕 | 运动路 | 网格起点中 f+32 ≤ t 者（demo 段 s = 16m、全域 f = s；exec 段 u = 16m、全域 f = es+u），逐个读 motion 表一行、切 `pos_rows(np.asarray([f]))[0,0,:256]`，右填充到 96 行，并记下每行的全域时刻（padding 行记哨兵） | `motion_emb` (96,768)、`motion_pos` (96,256)、`motion_mask` (96,) |
| e ★ | 排序 | 608 个候选位各配一维键 `key = 2·时刻 + 类型`（float64）：帧 i 的 16 位共享 (帧号, 0)，motion k 记 (全域起点 f, 1)，两路 padding 记 +∞（`np.inf`；「∞」不是 int32 可表示值，用 int 键就得换 `np.iinfo(np.int32).max` 哨兵）；`np.argsort(key, kind="stable").astype(np.int32)`；产出后显式 raise 校验 `np.array_equal(np.sort(mem_order), np.arange(608))` | `mem_order` (608,) int32 |

e 步的键定义兑现了三条口径：按时刻排即「按起点 f 插入」，类型 0 < 1 即「同刻帧在前」（若不用组合键而写二元组，`np.argsort` 收不了，只能 `np.lexsort((类型, 时刻))`；本文用一维组合键），padding 时刻 +∞ 即「mask 全在尾部」。同刻帧在前还由 `[帧路 512 | 运动路 96]` 的拼接顺序加稳定排序共同保证。collate 后每键前面多一个 b 维。

### 4.1 入口 `train_step` → `loss_fn` → `compute_loss`：一个 batch → 标量 loss 与全部梯度〔Pi0〕

`loss_fn = mean(model.compute_loss(rng, observation, actions))` 得标量；`nnx.value_and_grad` 得与参数树同形的梯度。

### 4.2 观测预处理与加噪：图像增广形状不变；`actions` (b,20,32) → `x_t`、`u_t` 各 (b,20,32)〔Pi0〕

`compute_loss` 收到的输入：

| 键 | 形状 |
|---|---|
| `images["base_0_rgb"]`、`images["left_wrist_0_rgb"]` | (b,224,224,3) 各一，值域 [−1,1] |
| `image_masks` 两键 | (b,) 恒 True |
| `tokenized_prompt` / `tokenized_prompt_mask` | (b,64) int32 / (b,64) bool；本 run `discrete_state_input=False`，state 不写进文本 |
| `state` | (b,32) f32；pi05 无 state token、`use_state_emb=False`，数值全程不进模型，只被取 shape 当 batch 载体 |
| `static_image_emb` / `static_pos_emb` / `static_state_emb` / `static_mask` | (b,512,2048) / (b,512,768) / (b,512,8) / (b,512) |
| `motion_emb` / `motion_pos` / `motion_mask`〔motion〕 | (b,96,768) / (b,96,256) / (b,96) |
| `mem_order` ★ | (b,608) int32 |
| `actions` | (b,20,32) |

两步计算：

- `preprocess_observation`：训练时 `base_0_rgb` 做 RandomCrop(212,212) + Resize(224,224) + Rotate(±5°) + ColorJitter，`left_wrist_0_rgb` 只做 ColorJitter（代码里 `"wrist" not in key` 才加几何增广），形状不变；记忆相关各键原样透传。
- 加噪：`noise ~ N(0,1)` (b,20,32)；`time ~ Beta(1.5,1) × 0.999 + 0.001` (b,)；`x_t = time · noise + (1 − time) · actions` (b,20,32)；`u_t = noise − actions` (b,20,32)。

### 4.3 `embed_prefix`：记忆 608 + 图像 512 + 文本 64 → 前缀 token (b,1184,2048) 与三条 mask〔HistoryPi0〕

| 步 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 记忆·帧路 | `pos_proj`（768→768）+ `silu`，与图像特征拼 2816 维，过 `encoder_static`（2816→2048） | (b,512,2048) + (b,512,768) | (b,512,2048) |
| 记忆·运动路〔motion〕 | `motion_pos_proj`（256→768）+ `silu`，与 motion token 拼 1536 维，过 `motion_encoder_static`（1536→2048） | (b,96,768) + (b,96,256) | (b,96,2048) |
| 记忆·拼接 | 长度轴 concat（并列序）；`input_mask = [static_mask ⊕ motion_mask]` | 上两行 | (b,608,2048)、(b,608) |
| 记忆·重排 ★ | `jnp.take_along_axis(tokens, mem_order[:, :, None], axis=1)` 与 `jnp.take_along_axis(input_mask, mem_order, axis=1)` 各一次；gather 前显式 raise 校验 `mem_order` 长度 == 608 且 dtype int32 | (b,608,2048)、(b,608)、(b,608) int | 形状不变，行序按时间 |
| 图像 | `PaliGemma.img` | (b,224,224,3) × 2 | (b,256,2048) × 2 |
| 文本 | `PaliGemma.llm(method="embed")` | (b,64) int | (b,64,2048) |

三段拼成 `prefix_tokens` (b,1184,2048)；`prefix_mask` (b,1184)；`prefix_ar_mask` (1184,) 只有第 608 位 True；`prefix_na_mask` (1184,) 第 608 到 1119 位 True。

### 4.4 `embed_suffix`：`x_t` (b,20,32) → 动作 token (b,20,1024)；`time` (b,) → `adarms_cond` (b,1024)〔Pi0〕

| 步 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 动作投影 | `action_in_proj`（32→1024） | `x_t` (b,20,32) | `suffix_tokens` (b,20,1024) |
| 时间编码 | `posemb_sincos(time, 1024, 4e-3, 4.0)` | `time` (b,) | (b,1024) |
| 时间 MLP | `time_mlp_in` → `swish` → `time_mlp_out` → `swish` | (b,1024) | `adarms_cond` (b,1024) |

`suffix_mask` (b,20) 全 True；`suffix_ar_mask` (20,) = [True, False × 19]；`suffix_na_mask` (20,) 全 False。

### 4.5 mask 与位置号：三条 (b,1204) 布尔向量 → 可见表 (b,1204,1204) + 序号 (b,1204)〔HistoryPi0〕

| 量 | 形状 | 取值 |
|---|---|---|
| `input_mask` | (b,1204) | 记忆段真 token True（t=200 为 523 位）、图像 512 位 True、文本按实际长度、动作 20 位 True |
| `ar_mask` | (1204,) | 只有第 608 位与第 1184 位 True |
| `na_mask` | (1204,) | 第 608 到 1119 位 True |

`make_attn_mask(input_mask, ar_mask, na_mask)` 四步：

1. `cumsum(ar_mask)` (b,1204)：记忆段块号 0，图像与文本块号 1，动作块号 2；`attn_mask[q,k] = 块号[k] ≤ 块号[q]` → (b,1204,1204)。
2. `valid_mask = input_mask[:,None,:] ∧ input_mask[:,:,None]` → (b,1204,1204)，padding 位整行整列 False。
3. `mask_not_attend`：图像 token 不能看记忆段 → (b,1204,1204)。
4. 合成 `attn_mask` (b,1204,1204)。★ 交错方案下 True 的个数不变，记忆段内 padding 列从两段合成尾部一段。

`positions = cumsum(input_mask, axis=1) − 1` → (b,1204) int32，每个 token 的一维序号，padding 位不推进。它不是 RoPE 本身：RoPE 在 4.6 的 i 步把这个序号变成角度作用到 q、k 上；gemma 不往 token 内容里加位置向量，图像的二维位置在 SigLIP 内部、记忆的帧号时间码在 PosEmb3D 内容里，与 RoPE 叠加。★ 交错方案改的是 memory 段 608 个整数的取值（二节表），第 609 位起不变。

### 4.6 主干 `PaliGemma.llm`：(b,1184,2048) + (b,20,1024) → 18 层 → (b,1184,2048) + (b,20,1024)〔Pi0〕

输入：`prefix` (b,1184,2048)、`suffix` (b,20,1024)、`positions` (b,1204)、`mask` (b,1204,1204)、`adarms_cond` (b,1024)。18 层结构相同，每层七步：

**g. 注意力前归一化 `RMSNorm`：逐 token 归一化，(b,1184,2048) / (b,20,1024) → 同形，expert 1 另出 gate (b,1,1024)**

| expert | 计算 | 输入 | 输出 |
|---|---|---|---|
| 0 | `x / √(mean(x²) + 1e−6) × (1 + scale)`，`scale` (2048,) | (b,1184,2048) | (b,1184,2048) |
| 1 | `Dense(1024→3072)(adarms_cond)` 切成 scale / shift / gate 各 (b,1,1024)；`x / √(mean(x²) + 1e−6) × (1 + scale) + shift` | (b,20,1024) + (b,1024) | (b,20,1024)，gate (b,1,1024) |

**h. q / k / v 投影：两段 token → q (b,1204,8,256)、k 与 v 各 (b,1204,1,256)**

| expert | 权重 | 输入 | 输出 |
|---|---|---|---|
| 0 | `q_einsum` (8,2048,256)；`kv_einsum` (2,1,2048,256) | (b,1184,2048) | q (b,1184,8,256)；k、v 各 (b,1184,1,256) |
| 1 | `q_einsum_1` (8,1024,256)；`kv_einsum_1` (2,1,1024,256) | (b,20,1024) | q (b,20,8,256)；k、v 各 (b,20,1,256) |
| 合并 | 沿长度轴 concat | | q (b,1204,8,256)；k、v 各 (b,1204,1,256) |

**i. RoPE `_apply_rope`：按序号旋转 q 与 k，形状不变 ★**

| 子步 | 计算 | 形状 |
|---|---|---|
| 频率 | `timescale_j = 10000^(2j/256)`，j = 0…127 | (128,) |
| 角度 | `radians = positions[…,None] / timescale` | (b,1204,1,128) |
| 三角 | `sin(radians)`、`cos(radians)` | 各 (b,1204,1,128) |
| 旋转 q | q 切前后两半 x1、x2 各 (b,1204,8,128)；`[x1·cos − x2·sin, x2·cos + x1·sin]` 拼回 | (b,1204,8,256) |
| 缩放（不属于 RoPE） | `q *= head_dim ** −0.5`（= 1/16），在 q 的旋转之后、**只作用于 q** | (b,1204,8,256) |
| 旋转 k | 同 positions、同 sin/cos 旋转，旋转后不缩放 | (b,1204,1,256) |

旋转后 q_i · k_j 只取决于序号之差 `positions[i] − positions[j]`。所以交错只改三类序号之差：memory 内部两两之差、文本到各 memory token 之差、action 到各 memory token 之差（4.5 的块号规则里文本与 action 都能看记忆段，图像被 `na_mask` 挡住，看不到）；图像、文本、action 相互之差不变。

**j. 打分、屏蔽、softmax、加权：q·k → 权重 (b,1,8,1204,1204) → 加权 v 得 (b,1204,8,256)**

| 子步 | 计算 | 形状 |
|---|---|---|
| logits | `einsum(q, k)`，8 个 q 头共用 1 个 k 头 | (b,1,8,1204,1204) |
| 屏蔽 | `where(mask, logits, −2.3819763e38)` | (b,1,8,1204,1204) |
| softmax | 沿最后一维 | (b,1,8,1204,1204) |
| 加权 | `einsum(probs, v)` | (b,1204,8,256) |

False 格子 softmax 后权重严格为 0，padding 列对任何输出零贡献。

**k. 输出投影：(b,1204,8,256) 拆回 (b,1184,2048) 与 (b,20,1024)**

| expert | 权重 | 输入 | 输出 |
|---|---|---|---|
| 0 | `attn_vec_einsum` (8,256,2048) | 前 1184 行 (b,1184,8,256) | (b,1184,2048) |
| 1 | `attn_vec_einsum_1` (8,256,1024) | 后 20 行 (b,20,8,256) | (b,20,1024) |

**l. 残差：注意力输出加回输入，形状不变**——expert 0 `x + y` → (b,1184,2048)；expert 1 `x + y × gate` → (b,20,1024)。

**m. 前馈 `FeedForward` 与残差：2048 → 16384 → 2048 / 1024 → 4096 → 1024，形状不变**

| expert | 计算 | 形状 |
|---|---|---|
| 0 | `RMSNorm` 同 g；`gelu(x·W0) × (x·W1)`，W0、W1 各 (2048,16384)；再过 (16384,2048)；残差 `x + y` | (b,1184,2048) → (b,1184,16384) → (b,1184,2048) |
| 1 | 自适应 `RMSNorm` 同 g；W0、W1 各 (1024,4096)；再过 (4096,1024)；残差 `x + y × gate` | (b,20,1024) → (b,20,4096) → (b,20,1024) |

每层的 k、v 各 (b,1204,1,256) 作为 kv_cache 返回，18 层堆成 (18,b,1204,1,256) × 2；训练时丢弃，推理时即 5.3 的缓存。18 层跑完各过 `final_norm`：`prefix_out` (b,1184,2048)、`suffix_out` (b,20,1024)，只用后者。

### 4.7 输出头与 loss：(b,20,1024) → `v_t` (b,20,32) → loss (b,20) → 标量〔Pi0〕

`action_out_proj`（1024→32）作用于 `suffix_out[:, −20:]` (b,20,1024) 得 `v_t` (b,20,32)；`mean((v_t − u_t)², axis=−1)` → (b,20)；再对 b 与 20 取均值得标量。

### 4.8 反向与更新：标量 loss → 与参数树同形的梯度 → 参数更新

SigLIP 不更新，其余参数全部更新。★ `take_along_axis` 可微，梯度按同一张 `mem_order` 表搬回并列顺序，再分别流向帧路与运动路的投影。

**四节小结**：交错方案在训练链上只出现在四处——4.0 的 e 步（dataloader 排序）、4.3 的重排步（模型侧 gather）、4.5 的 memory 段位置号取值、4.6 的 i 步旋转角。其余每一步的函数与计算逐字不变。

---

## 五、推理：一批观测 → 20 步动作 (20,32)

推理入口是 `src/mme_vla_suite/policies/policy.py` 的 `MME_VLA_Policy`。与训练的三点结构差异：帧成批到货（首批整段 pre_traj，之后每批 16 帧 = 推理阶段一个 action chunk），infer 只在每批之后发生一次；记忆特征不是从离线表读，是每批现编后存进内存字典；主干不是一次算完 1204 个 token，而是先算 1184 个前缀存 kv_cache，再在去噪循环里每步只算 20 个动作 query。

### 5.1 每批入库 `add_buffer`：本批帧 (n,224,224,3) → 每帧外观 token (16,2048) + 位置码 (16,768) 存字典；再把所有已凑齐 33 帧的网格窗口各编一个 motion token (768,)〔HistoryPi0 在线侧，motion 新增运动路〕
`MME_VLA_Policy.add_buffer` 把本批帧交给 `FrameSampMemory.add_buffer`。训练的记忆特征是离线抽好存在表里的，推理没有表，每批帧到货现编一次存进内存字典；编码链与抽表脚本同一套。

| 步 | 函数 | 与训练的区别 |
|---|---|---|
| a | `add_buffer` 归一化与缩放 | 同抽表脚本，整批一起做 |
| b | `vision_enc`，即 `module_jit(HistoryPi0.vision_encode)` 里的 `PaliGemma.img` | 同一个 SigLIP，同一份权重，整批一次前向 |
| c | `pool_tokens_to_size(…, 16)` | 同一档 4×4 池化 |
| d | `PosEmb3D` 预计算表取各帧一行 | 与离线 `pos_emb_4x4` 表同一算法、同一输入 |
| e | 存字典 `_history_feats[step]` | 训练存离线表，推理存内存；一帧一条 |
| ★ 运动路〔motion〕 | sidecar 里的 Wan VAE + `WanLatentMotionEncoder` | 训练从离线 motion 表读；推理每批入库后用 while 循环把所有「33 帧已凑齐」的网格起点编掉（demo 段的起点在首批一次编完，exec 段稳态每批恰新增 1 窗，每窗约 1.57 s、同步等待，主计划 2.6），存 `_history_feats_motion[f]`（键 = 全域起点帧号）；编码口径与离线表同源 |
### 5.2 组装记忆 `_prepare_history`：字典 → 帧路 512 位 + 运动路 96 位 + 次序表 (608,)，加 batch 维 b = 1〔HistoryPi0 在线侧〕
`infer` 调 `_prepare_history`。组装规则与训练的 `__getitem__` 一模一样，区别只是数据来自字典、一次只装一个样本。f–i 由 `FrameSampMemory.prepare_frame_sampling` / `_prepare_frame_sampling` 执行，j 是并列的新方法 `_prepare_motion`，`_prepare_history` 自己只做调用、`_normalize_state`、写键与 k 步排序。

| 步 | 函数 | 与训练的区别 |
|---|---|---|
| f | `even_sampling_indices(step_idx, 32)` | 同一个函数 |
| g | `_load_emb` 从字典堆出各帧特征 | 训练从离线表按行号读 |
| h | `right_padding_token_emb` | 与训练 `_pad` 数值相同，只是拼零块而非原地填 |
| i | reshape + `np.repeat(mask, 16)` | 同 4.0 的 d 步 |
| i′ | `_normalize_state` 对 `static_state_emb` 归一化 | 训练侧在 transform 里做；该键随行交付、不进模型 |
| j ★ | `_prepare_motion`〔motion〕 | 同一套网格、同一取法（取最近 96 个）、同一右填充；数据从 `_history_feats_motion` 取而非离线表；段边界 `exec_start_idx` 须由 `MME_VLA_Policy` 下传 |
| k ★ | 排序 | `_prepare_history` 再调一次 `get_frame_sampling_indices` 拿到帧路 32 个帧号（纯函数、与 f 步同值），与运动路 96 个全域起点一起送进**与训练共用的同一个排序函数**得同一张次序表；两侧若各写一份，排序稍有出入就会让在线静默看到与训练不同的次序 |

`_input_transform`、`HistAugObservation.from_dict` 同训练，只是加的 batch 维为 1。
### 5.3 前缀 pass：前缀 (1,1184,2048) → 18 层 → kv_cache (18,1,1184,1,256) × 2〔HistoryPi0 改写自 Pi0〕
`_sample_actions` 是 `module_jit(HistoryPi0.sample_actions)`。训练把前缀与动作拼成一条序列一次算完；推理先只算前缀这一段，把每层的 k、v 存下来。

| 步 | 函数 | 与训练的区别 |
|---|---|---|
| l | `preprocess_observation(None, obs, train=False)` | 不增广、不加噪、不采时间，整段没有随机数 |
| m | `embed_prefix` | 同 4.3，含 ★ 重排 |
| n | `make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)` | 只对前缀建表，没有动作块；等于训练那张大表的左上角 |
| o ★ | `positions = cumsum(prefix_mask) − 1` | 只算前缀；memory 段取值与训练相同，★交错改的就是这些 |
| p | `PaliGemma.llm([prefix_tokens, None], mask, positions)` | 只跑 expert 0，序列里没有动作 token；每层旋转后的 k 与 v 收集成 kv_cache，前缀输出丢弃 |

能只算一次的原因：前缀 token 本来就看不到动作段，每个 token 的 k、v 只由它自己的输入与序号决定，所以缓存里的 k、v 与训练时前缀部分算出的相同。★交错方案改的 memory 段旋转角在这一步烙进缓存。
### 5.4 去噪循环 `step` × 10：噪声 (1,20,32) 逐步 Euler 积分 → 动作 (1,20,32)〔Pi0〕
`jax.lax.while_loop` 从纯噪声起，时间从 1 到 0 固定走十档，每档只算 20 个动作 token，前缀的 k、v 从缓存读。

| 步 | 函数 | 与训练的区别 |
|---|---|---|
| q | `embed_suffix(obs, x_t, time)` | 同 4.4；但 `x_t` 是上一步积分结果而非真实动作加噪，`time` 是循环变量而非 Beta 采样 |
| r | `make_attn_mask(suffix_mask, suffix_ar_mask)` | 两参版，只对 20 个动作 token 建表，互相全可见 |
| s | `einops.repeat(prefix_mask, …, s=20)` 与 r 拼接 | 动作对前缀的可见性直接复制 `prefix_mask`，不再算外积与块号；padding 列第二次被封 |
| t | `positions = sum(prefix_mask) + cumsum(suffix_mask) − 1` | 用前缀真 token 数作偏移，不再对整条序列 `cumsum`；取值与训练动作段相同，交错不改 |
| u | `PaliGemma.llm([None, suffix_tokens], mask, positions, kv_cache)` | 只跑 expert 1；前缀的 k、v 从缓存读，只现算自己 20 个；logits 是 20 行而非整张方表 |
| v | `action_out_proj` | 同 4.7 得速度，但不与 `u_t` 比较、没有 loss，改做一步 Euler 积分 |

**u 步展开：`PaliGemma.llm([None, suffix_tokens], mask, positions, kv_cache)` 内部每层七步，与训练 4.6 的区别**

expert 0 的输入是 `None`，18 层里只有 expert 1 那一列在算；kv_cache 逐层取出本层缓存。

| 步 | 与训练 4.6 的区别 |
|---|---|
| g 归一化 | 只有自适应 `RMSNorm` 那一路，吃 q 步的 `adarms_cond`；expert 0 的普通版不跑 |
| h 投影 | 只有 `q_einsum_1` / `kv_einsum_1`，只对 20 个动作 token 算 q、k、v；训练里前缀 1184 个 token 的投影这里不算 |
| i RoPE | 只旋转这 20 个 q 与 20 个 k，用的是 t 步那 20 个序号；前缀的 k 不再旋转，因为缓存里存的已经是旋转后（未缩放）的 k |
| 缓存拼接（训练没有这一步） | 本层缓存的前缀 k、v 接在新算的 20 个前面，拼成完整的 1204 个 k、v；训练里这 1204 个是同一次前向现算的 |
| j 打分 | query 只有 20 行，logits 是 20 × 1204 而不是 1204 × 1204；mask 用 s 步的 20 行表；softmax、加权同训练 |
| k 输出投影 | 只有 `attn_vec_einsum_1` |
| l 残差 | 只有门控残差那一路 |
| m 前馈 | 只有 action expert 那一路 `FeedForward`；结束后每层也会返回拼好的 k、v，但被丢弃，缓存不更新 |

18 层跑完只过 `final_norm_1`，得到 20 个动作 token 的输出交给 v 步。与训练的本质区别只有一条：训练里动作 token 打分用的 1184 个前缀 k、v 是同一次前向现算的，推理里从缓存读，数值上是同一个东西（5.3 表后那句的推论）。

这 20 行 mask 与 20 个序号与训练那张大表的最后 20 行相同：训练里动作 query 能看所有块，只受 `valid_mask` 限制，而它的一行就是 `input_mask` 本身。十步跑完得到动作，去掉 batch 维、反归一化后输出。

**五节小结**：推理链上交错方案改的地方与训练完全对应——k 步排序（对应训练 4.0 的 e 步）、m 步 gather（同一个 `embed_memory`）、o 步位置号、p 步 k 的旋转角。去噪循环 q–v 六步代码与数值机制不变，只是消费的 kv_cache 已经不同。训练与在线必须共用同一个排序函数，规则只在一处定义；两侧若各写一份，排序稍有出入就会让在线模型看到与训练不同的位置号分配，不报错，只静默降效果。

### 5.5 训练与推理数值链对比图

左列训练一次前向，右列推理前缀一次加去噪十次。同一行是同一站，★ 标交错改动处，形状随行给出。

```
      训练 compute_loss（一次前向，b 个样本）                 推理 sample_actions（前缀一次 + 去噪 10 次，b = 1）
 ═══════════════════════════════════════════════════   ═══════════════════════════════════════════════════════
 4.0 dataloader 键                                      5.1 add_buffer 每批现编 → 字典（含已凑齐的运动窗口）
     帧路 512 位 / 运动路 96 位 / ★mem_order (608,)          5.2 _prepare_history：帧路 512 / 运动路 96 / ★mem_order
              │                                                        │
 4.2 preprocess(train=True) 图像增广                    5.3.1 preprocess(train=False) 不增广
     noise (b,20,32)、time~Beta (b,)                        （无 noise、无 time）
     x_t = time·noise + (1−time)·actions (b,20,32)
     u_t = noise − actions (b,20,32)
              │                                                        │
 4.3 embed_prefix → (b,1184,2048)                       5.3.2 embed_prefix → (1,1184,2048)
     ★ embed_memory 里 take_along_axis 重排                 ★ 同一函数，同样重排
 4.4 embed_suffix → (b,20,1024), cond (b,1024)               （此时没有 suffix）
              │                                                        │
 4.5 三条 mask 拼成 (b,1204)                            5.3.3 三条 mask 只有前缀 (1,1184)
     make_attn_mask → (b,1204,1204)                          make_attn_mask → (1,1184,1184)
     positions = cumsum − 1 → (b,1204)  ★memory 段            positions = cumsum − 1 → (1,1184)  ★memory 段
              │                                                        │
 4.6 llm([prefix, suffix]) 18 层                        5.3.4 llm([prefix, None]) 18 层，只跑 expert 0
     q (b,1204,8,256)  k,v (b,1204,1,256)                    q (1,1184,8,256)  k,v (1,1184,1,256)
     ★ _apply_rope 用 positions 旋转 q、k                    ★ _apply_rope 旋转后的 k、v 存下 →
     logits (b,1,8,1204,1204)                                kv_cache (18,1,1184,1,256) × 2
     → prefix_out (b,1184,2048) 丢弃                          prefix_out 丢弃
     → suffix_out (b,20,1024)                                          │
              │                                         ┌──── 循环 10 次，time = 1.0 → 0.1 ────────────────┐
              │                                         │ 5.4.1 embed_suffix(x_t, time)                    │
              │                                         │       → (1,20,1024), cond (1,1024)               │
              │                                         │ 5.4.2 mask = [prefix_mask×20 行 | 20×20 全 True]  │
              │                                         │       → (1,20,1204)                              │
              │                                         │       positions = Σprefix_mask + 0…19 → (1,20)   │
              │                                         │ 5.4.3 llm([None, suffix], kv_cache) 只跑 expert 1│
              │                                         │       q (1,20,8,256)                             │
              │                                         │       k,v = [缓存 1184 | 新 20] → (1,1204,1,256)  │
              │                                         │       logits (1,1,8,20,1204)                     │
              │                                         │       → suffix_out (1,20,1024)                   │
 4.7 action_out_proj → v_t (b,20,32)                    │ 5.4.4 action_out_proj → v_t (1,20,32)            │
     loss = mean((v_t − u_t)²) → 标量                    │       x_t ← x_t − 0.1·v_t ；time ← time − 0.1     │
              │                                         └──────────────────────────────────────────────────┘
 4.8 grad → 参数更新                                     x_0 (1,20,32) → (20,32) → 反归一化 → 动作
```

两列的对应关系：训练 4.5 的 (b,1204,1204) 表等于推理 5.3.3 的 1184 × 1184 表放在左上角、5.4.2 的 20 × 1204 表放在最后 20 行；训练 4.6 里前缀 token 的 k、v 等于推理 5.3.4 缓存的 k、v；训练 4.6 里动作 token 的 q 与 logits 等于推理 5.4.3 每一步算的 q 与 logits。★ 两列各有三处，位置一一对应：memory 次序表、memory 段位置号、memory 段 k 的旋转角。

---

## 六、数值层面的结论：交错只改 memory 段 608 个序号，计算量与 padding 数不变

**数学上**：transformer 的每一层对 token 顺序是置换等变的，把 token 行、它的 mask 行列、它的位置号一起换顺序，每个 token 算出来的输出向量不变，只是行的摆放顺序换了。loss 只读 action 那 20 行，那 20 行不动。所以物理交错这件事本身没有效果，效果全部来自「换了位置号」，进入计算的位置只有 4.6 的 i 步与 5.3 的 p 步那两行 `_apply_rope`。

**数值上**：交错方案与并列方案不逐位，有两个原因。一是位置号不同，这是语义差异，本来就该不同。二是行序不同，矩阵乘与 softmax 的浮点累加顺序跟着变，会有末位差异。因此交错方案与并列方案之间不做等价对拍（用户已拍板只保留交错、不做并列对照），等价对拍只做一条：motion 关闭态对 HEAD 逐位相同。

**计算量**：attention 仍是 1204 × 1204，padding 个数相同，jit 编译形状相同，kv_cache 形状相同。多出的两步里，排序是 numpy 里 608 个元素的一次排序，重排是 GPU 上每样本约 2.5 MB 的一次搬运（608 × 2048 × 2 B），相对 1204² 的 attention 都可忽略。「mask 全在尾部」不省任何计算。

---

## 七、与主计划的关系：已回写，落点清单

主计划 `motion-memory-plan.md` 已于 2026-09-02 按本文改为交错方案（与 `motion.stride` 16、`motion.budget` 96 一并换档），两文冲突以主计划为准。本文所述在主计划里的落点：

- 一节「五个已定死的口径」第 4 条：「交错拼接」，附用户三句原话。
- 2.6 新增「在线采样节奏与固定延迟」：每 16 步一批到货、每次 infer 前同步编 1 个窗口、接受固定 +1.57 s，附平行数轴图 `docs/motion-memory-online-timeline.svg`。
- 3.1.2 改动后链路图：concat 之后多一跳 `take_along_axis`，两列汇合处新增交付键 `mem_order (b,608) int32`。
- 3.3 与 `docs/motion-memory-mask-axis.svg`：记忆区改为一段 608 位，`input_mask` 改为「重排后前 16k+m 位 True」，SVG 重绘。
- 3.4：调用链 A 加 ④′ 排序步、⑤ 加重排；调用链 B 改成批到货与两段式；走查样本改为 t=200（11 个起点、523 个真 token、padding 523–607）与 t=5（padding 96–607）。
- 5.1 一览表：四个新键、`n_keys` 开启态 16、`MEMORY_KEYS` 8、新增 `shared/sampling.py`（共用排序函数）与 `compute_norm_stats.py`（镜像 `_NONE_KEYS`）。
- 第二部分 2.2（`mem_order` 字段：`at.Int`、维名 `l5`）、2.3（`embed_memory` 四步 + 关闭态守卫 + 形状闸）、2.4（重排归属 `embed_memory`）、2.5–2.9（四键、置换 raise、`n_keys` 16）、三节（在线 while 补齐、段边界下传、k 步排序）、四节 A18（置换与逆置换判据）/ A23（`mem_order` 逐位）、七节 R19 / R20、十节字节账（`mem_order` 2,432 B）。

---

## 八、已知的未知：效果无先验、位置密度不均

- 交错对效果有没有帮助，没有先验证据。token 内容里已带时间码，交错只是把「时间相邻」额外写进 RoPE 距离。用户已拍板只保留交错一种布局、不做并列对照（主计划 S4 无此档），这一差异在本轮不再验证。
- RoPE 位置密度不均：一个帧占 16 个连续位置号，一个 motion 只占 1 个。
- 同刻冲突：es ≥ 33（demo 网格非空）或 es = 0 且 t ≥ 32 时，全域帧 0 既是采样帧又是起点、必然同刻；0 < es < 33 时最早的起点是全域 es，同刻纯属巧合（v1 数据里 es ∈ {0, 66, 114, 168, 216}，该类不存在）；stride 16 整除 32，其余同刻（如 t=200 时的 32、64、96）取决于 linspace 是否恰好落在 16 的倍数上，规则固定后没有随机性。
