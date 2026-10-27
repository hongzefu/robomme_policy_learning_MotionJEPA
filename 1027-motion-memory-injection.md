# 1027-motion-memory-injection.md — motion memory 是怎么注入 π0.5 的（memory-as-modulation）

**一句话结论**：每个 33 帧窗经冻结的 stage2 encoder（EMA）变成 1 个 1024 维 motion token，与 512 个帧路 token 一起投影成 608 个 1024 维 memory token、按时刻排序；它们**不经任何自注意力编码**，直接在 action expert 的 18 层里各被一个 `MemoryAttention`（Q 来自 action token，4 头×256，1 KV 头）读取，读出的 `mem_mod_vec` 经插入式 `MemoryRMSNorm` 对 FFN 输入做 scale/shift 调制；paligemma 一侧完全不见 memory。

![motion memory 注入全链路](docs/figures/1027-motion-memory-injection.svg)

**阅读指引**：本文件只讲机制（数据怎么流、在哪里注入、算了什么），不讲训练计划、闸门与 run 安排；那些见 `1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md` 与 `1026-vla-stage2-memory-1024-train-plan.md`。补位（motion 不满 96 时）的细节见既有图 `docs/figures/1026-motion-pad-modulation.svg`。

**口径**：代码锚点均以 policy 仓 HEAD `4029b1f`（`commitV11.57`）为准，路径省略前缀 `src/`；MotionJEPA 侧事实照抄 1026 计划 §2.2–§2.4。参数量是按权重形状推算的，未实际运行统计（见第 10 节）。行号与旧计划/旧注释不一致处一律以代码为准（见第 9 节）。

---

## 1. 全链路总览

| # | 输入 | 模块 | 输出（形状 / dtype / 字节） | 可训？ | 代码锚点 |
|---|---|---|---|---|---|
| 1 | 33 帧 256×256 RGB 窗（窗 33、步长 32、segment_start、forward） | Wan2.1 VAE | latent (9,16,32,32) fp32 = 589,824 B/窗 | 冻结（离线缓存） | 1026 计划 §2.2；policy `scripts/dataset/wan/extract_wan.py` |
| 2 | latent | `Stage2WanLatentMotionEncoder.forward`（576 patch + 4 readout，8 层） | (B,4,256) fp32 → reshape (B,1024) | 冻结、`eval()`、`no_grad`、EMA 权重 | MJ `wan_latent_encoder.py` |
| 3 | (1024,) 向量 | 离线 store | 每行 (1,1024) f32 = 4,096 B；layout `motion-stage2-1x1024-grid32-demopad17-v1` | — | `v1-store/datasets/<库>/motion-stage2-1024/motion_token.f32.bin` |
| 4 | 帧特征 + store 行 | `FramesampDataset` | 帧路 512 token；motion 路 96 位；`mem_order` (608,) int32 | — | `mme_vla_suite/training/framesamp_dataset.py:373-413` |
| 5 | 帧路 `static_image_emb` (512,2048) ‖ pos | `pos_proj` + `encoder_static` | (B,512,1024) | 可训，从零 | `mme_vla_suite/models/representation/mem_encoder.py:35-51,60,86` |
| 6 | motion 路 `motion_emb` (96,1024) ‖ pos | `motion_pos_proj` + `motion_encoder_static` | (B,96,1024) | 可训，从零 | `mme_vla_suite/models/representation/percep_mem.py:28-44,85-87` |
| 7 | 两路拼接 | `PerceptualMemory` concat | (B,608,1024) | — | `percep_mem.py:87` |
| 8 | (B,608,1024) + mask | `HistoryPi0.embed_memory` 按 `mem_order` 重排 | `mem_seq` (B,608,1024)、`mem_mask` (B,608) bool | — | `mme_vla_suite/models/integration/history_pi0.py:326-360` |
| 9 | action expert 残差流 x (B,T,1024) + `mem_seq` | 每层 `MemoryAttention` | `mem_mod_vec` (B,T,1024) | 可训，每层独立 | `mme_vla_suite/models/integration/history_gemma.py:45-108` |
| 10 | x、`mem_mod_vec` | 每层 `MemoryRMSNorm("mem_rms_norm_ffn")` → 原版 `pre_ffw_norm_1` → FFN → gated residual | 调制后的 FFN 输出 | 可训 | `history_gemma.py:177-192` |
| 11 | 18 层后 suffix 输出 | action expert 末端（openpi 原样） | 速度场 / 去噪动作 | — | `history_pi0.py:547-558`（训练）、`609-682`（推理） |

T 为 action expert 的 suffix token 数（π0.5 下即 action 序列长度；本文不展开其取值）。

---

## 2. 上游：motion token 怎么来的

- **窗**：每窗 33 帧、256×256 RGB，步长 32，从段起点（segment_start）正向（forward）枚举；经 Wan2.1 VAE（`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`）得 latent (9,16,32,32) fp32 = 589,824 B。
- **encoder**：`Stage2WanLatentMotionEncoder.forward(latents, geometry_valid=全 True)`：先用 ckpt 内 `latents_mean/std` 做 affine 归一化 → 576 个 patch token（9×8×8）+ 4 个 readout token（源时间片 [0,3,5,8]，32×32 → 4×4 掩码均值）= 580 token，过 8 层 → 取 4 个 readout 得 (B,4,256) fp32 → reshape (B,1024)，即槽 0‖3‖5‖8 首尾拼接，不改数。
- **权重与模式**：encoder 取 `stage2-full144-1025-a/checkpoint_epoch_105.pt` 的 `model` 键（EMA，不是 `model_live`）；冻结、`eval()`（encoder 内 dropout 0.05 在 eval 下恒等）、`no_grad`、fp32。
- **离线与在线同一 encoder**：训练库离线抽取与推理时在线 sidecar 编码用同一份 encoder、同一式。
- **store 行格式**：每窗 1 行 (1,1024) f32 = 4,096 B，文件 `motion_token.f32.bin`，layout `motion-stage2-1x1024-grid32-demopad17-v1`。
- **窗口枚举与 demo 补帧**（1026 计划 §2.3）：`real = min(33, seg_len − off)`；demo 段末窗真实帧 ≥17 保留，并把段末帧 `frames[seg_len−1]` 在像素层重复补到 33 帧（repeat_last，整窗再过 VAE），≤16 丢弃；exec 窗必须 33 帧全真实（exec 需补帧直接 raise）。例：demo 段 len=100 时 stride32 起点为 0/32/64（64 窗 33 帧全真实，96 窗只剩 4 帧 ≤16 不存在）；exec 段 len=100 起点同为 0/32/64，三窗全真实。
- **可见窗条件**：demo 窗起点 s 满足 `s + (demo_min_real − 1) ≤ es − 1`（即 `s+16 ≤ es−1`）；exec 窗起点 u 满足 `u + (W − 1) ≤ step − es`（即 `u+32 ≤ t−es`），es 为 demo 段长、W=33。
- **上限**：`examples/robomme/eval.py` 执行步数上限 1300 下，最坏 demo（VideoPlaceOrder 1247 帧）39 窗 + exec 40 窗 = **79**，motion 预算 96，余量 17。

**代码锚点**：MJ `wan_latent_encoder.py`（encoder 与 dropout）；policy `scripts/dataset/wan/extract_wan.py`（离线补帧）；`mme_vla_suite/policies/framesamp_memory.py:189-212`（在线 `_encode_window`，尾部补帧复用 `es-1`）、`:235-248`（`visible_motion_frames`）；`examples/robomme/eval.py`（1300 步）。

---

## 3. Dataset 侧两路输入

**帧路（512 token）**：`max_frames = budget // (token_per_image × num_views) = 512 // 64 = 8`（`framesamp_dataset.py:199`），`tokens_per_frame = 64`（`:200`）；「8×8」指每帧 64 个 token 的 8×8 空间网格（在线键 `image_emb_8x8`）。

| 字段 | 形状 / dtype | 每样本字节（按形状推算） | 说明 |
|---|---|---|---|
| `static_image_emb` | (512,2048) bf16 | 2,097,152 | 8 帧 × 64 token 的图像特征 |
| `static_pos_emb` | (512,768) f32 | 1,572,864 | 位置编码 |
| `static_state_emb` | (512,8) | — | 每帧 state 重复 64 次、已归一化；本配置 `use_state_emb false`，不进投影 |
| `static_mask` | (512,) bool | 512 | 不足 8 帧时 `_pad`（`:305-326`）右补零、mask False |

**motion 路（96 位，仅 `motion.enabled`）**：

| 字段 | 形状 / dtype | 每样本字节（按形状推算） | 说明 |
|---|---|---|---|
| `motion_emb` | (96,1024) f32 | 393,216 | `_pad_motion`（`:328-349`）；stage2 时 `:402` reshape 成 (k,1024) |
| `motion_pos` | (96,256) f32 | 98,304 | `pos_rows(f)[:,0,:256]`（`:405-406`），取窗时刻的位置编码前 256 维 |
| `motion_mask` | (96,) bool | 96 | 前 k 位 True |
| `mem_order` | (608,) int32 | 2,432 | `:413`，帧路与 motion 路合并后的重排索引 |

- **零截断**：k > 96 直接 raise（`:334-336`），不静默截断。
- **补位（简述）**：k 个真实窗之后的位 emb/pos 填零、mask False、时刻记哨兵 int32.max；补位也会过两层 Linear（零输入经 bias 得非零向量），但在注意力里被 mask 屏蔽，权重为 0、也不进 softmax 分母。序列长恒为 608，算力不随 k 变。详图见 `docs/figures/1026-motion-pad-modulation.svg`，展开见 1026 计划第二部分 §七。
- **排序键**（`mme_vla_suite/shared/sampling.py`）：`pad_times`（`:26-35`）给补位记哨兵（`:23`）；`memory_order`（`:38-58`）中帧路键 = 帧时刻×2+0（每帧重复 64 次），motion 键 = 窗时刻×2+1，拼接后 `argsort(kind="stable")`。语义：按全域时刻升序；**同一时刻帧在 motion 之前**；哨兵时刻的补位落在最尾。
- **512 run 关闭态**：四个 motion 键（`motion_emb`/`motion_pos`/`motion_mask`/`mem_order`）为 None（`_NONE_KEYS`，`:54-65`），memory 只有 512 个帧路 token，不重排。

**代码锚点**：`mme_vla_suite/training/framesamp_dataset.py:54-65,199-200,305-349,373-413`；`mme_vla_suite/shared/sampling.py:23,26-35,38-58`。

---

## 4. PerceptualMemory 投影（两路分别投到 1024）

| 路 | 模块 | 计算 | 参数量 |
|---|---|---|---|
| 帧路 | `pos_proj` | Linear(768→768, bias) + SiLU | 590,592 |
| 帧路 | image | identity（2048 不投影） | 0 |
| 帧路 | `encoder_static` | concat[img 2048 ‖ pos 768] = 2816 → Linear(2816→1024, bias)，无激活 | 2,884,608 |
| 帧路 | state 分支 | `use_state_emb false`，不建 | 0 |
| motion 路 | `motion_pos_proj` | Linear(256→768, bias) + SiLU | 197,376 |
| motion 路 | `motion_encoder_static` | concat[motion 1024 ‖ pos 768] = 1792 → Linear(1792→1024, bias) | 1,836,032 |
| **合计** | | 帧路 3,475,200 + motion 路 2,033,408 | **5,508,608** |

- 输出：帧路 (B,512,1024)，motion 路 (B,96,1024)，`concat(axis=1)` → (B,608,1024)（`percep_mem.py:87`）。全部从零训。
- motion 模块只在 `motion.enabled` 时于 `__init__` 中创建（不是在 `__call__` 里跳过），且建在帧路 `feature_encoder` 之后，以免改变帧路初始化的 RNG 序（`percep_mem.py:28-44` 注释）。
- **勘误**：1026 计划 §2.2 写的「新参数 1,836,032」只是 `motion_encoder_static` 一层；motion 路实际合计 2,033,408（再加 `motion_pos_proj` 197,376）。

**代码锚点**：`mme_vla_suite/models/representation/mem_encoder.py:35-41,44-51,60,86`；`mme_vla_suite/models/representation/percep_mem.py:28-44,85-87`。

---

## 5. 进入 action expert

- **`embed_memory`**（`history_pi0.py:326-360`）：开启态先做非 None 与形状闸，`input_mask = concat[static_mask, motion_mask]`（`:349`），再把 tokens 与 mask 一起按 `mem_order` 用 `take_along_axis` 重排（`:355-356`），得 `mem_seq` (B,608,1024)、`mem_mask` (B,608) bool。关闭态早返回，tokens 与 `static_mask` 原样。
- **LLM 组成**：modulation 下 LLM 只含 `[paligemma, action_expert]`，`mem_mods=[False, True]`（`:246-263`）；`gemma_150m`（memory expert）只在 `integration_type=="expert"` 分支构建（`:226-243`），modulation 下未使用。
- **不进 prefix**：context 分支把记忆放进 VLM prefix（`:376-387`）；modulation 不进 prefix，而是以 `mem_seq=[None, mem_seq]`、`mem_mask=[None, mem_mask]` 传给 LLM，只有 action expert 一侧收到。
- **不经自注意力**：memory token 在进入 `MemoryAttention` 前不经过任何 Transformer 层，投影输出直接作为每层 K/V 的来源；`nn.scan` 对 `mem_seq`/`mem_mask` 用 `nn.broadcast`，18 层看到的是同一份 memory。

**代码锚点**：`history_pi0.py:226-263,326-360,376-387`；`history_gemma.py:241-260`（scan 的 `in_axes`）。

---

## 6. 注入点逐层剖析（核心）

### 6.1 一层 HistoryBlock 的顺序

action expert 为 `gemma_300m`：width 1024、depth 18、mlp 4096、8 头、1 KV 头、head_dim 256（`openpi/models/gemma.py:69-77`）。openpi 原版 Block：`pre_attention_norm → Attention → gated residual → pre_ffw_norm → FFN → gated residual`（`gemma.py:284-331`）。

`HistoryBlock`（`history_gemma.py:111-205`）在 modulation 下对 action expert（`i == len(xs)-1`）做：

1. `pre_attention_norm_1`（adaRMS，cond = time）→ 与 paligemma 联合的 `Attention` → gated residual（与原版相同）；
2. **`MemoryAttention`**：以本层残差流 x 为 Q、608 个 memory token 为 K/V，得 `mem_mod_vec` (B,T,1024)（`:177-178`）；
3. **`MemoryRMSNorm("mem_rms_norm_ffn")(x, mem_mod_vec)`**：对 x 做 RMSNorm 后按 `mem_mod_vec` scale/shift（`:179`）；
4. 原版 `pre_ffw_norm_1`（adaRMS，cond = time），产出 FFN 输出门 gate（`:181-184`）；
5. `mlp_1` FFN（`:186-192`）；
6. gated residual：`xs[i] + ffn_out * gate`（`:198-202`）。

与原版对照：**`MemoryRMSNorm` 是插入在 `pre_ffw_norm` 之前的一步，不替换原 norm**；`pre_ffw_norm_1` 与其 time 条件、gate 全部保留。`mem_attn` 只在 `integration_type=="modulation"` 时创建（`:135-136`）；paligemma 一侧（`i==0`）走原版路径，没有任何 memory 计算。

从代码可直接读出的两点（不额外推断效果）：
- 第 6 步的残差加法用的是 `xs[i]`（注意力残差后的流），不是调制后的 x；即 memory 只通过「改变 FFN 的输入」影响残差流，`mem_mod_vec` 本身不直接加到残差流上。
- 第 3 步调制后的 x 紧接着又被第 4 步 `pre_ffw_norm_1` 重新做 RMS 归一化，所以调制作用到 FFN 的是方向/各维相对大小，整体幅值由第 4 步重定。

### 6.2 MemoryAttention（`history_gemma.py:45-108`）

| 项 | 值 |
|---|---|
| 头配置 | `num_heads=4`、`num_kv_heads=1`、`head_dim=256`、`width=1024`（`:56-61`，硬编码）；与 action expert 本身的 8 头不同，宽度相同（4×256=1024） |
| 预归一化 | 内部 `MemoryRMSNorm(name="mem_rms_norm")`，cond=None，即普通 RMSNorm、scale 零初始化；**同一个实例同时作用于 x 与 `mem_seq`**（共享 1,024 个 scale） |
| Q | `q_einsum_mem` (4,1024,256)，来自 action expert token x (B,T,1024)；1,048,576 参数 |
| K/V | `kv_einsum_mem` (2,1,1024,256)，来自 608 个 memory token；524,288 参数 |
| RoPE | Q 位置 `arange(608, 608+T)`，K 位置 `arange(608)`（`mem_len` 取实际长度，512 run 为 512）；即 memory 在前、action token 接在其后编号 |
| 缩放 | RoPE 后 `q *= head_dim**-0.5`（=1/16） |
| logits | `einsum(..., preferred_element_type=float32)`，f32 |
| mask | `jnp.where(mask, logits, -2.3819763e38)`（`:97-98`），mask 形状 (B,1,1,1,S)，只屏蔽 key；无因果掩码，每个 action token 可看全部有效 memory |
| softmax | 在 f32 上做，再 `astype(x.dtype)` |
| 输出 | `out_einsum_mem` (4,256,1024)，1,048,576 参数 → `mem_mod_vec` (B,T,1024) |
| 单层参数 | 1,048,576 + 524,288 + 1,048,576 + 1,024 = **2,622,464** |

4 个 Q 头共用 1 组 K/V（GQA，`K=1, G=4`）。

### 6.3 MemoryRMSNorm（`history_gemma.py:27-43`）

- `cond` 给定时：`nn.Dense(2048, kernel_init=kernel_init_out_proj, bias)(cond)`（`:40`），`kernel_init_out_proj = normal(stddev=0.002)`（`mme_vla_suite/models/representation/utils.py:5`）；split 成 scale、shift 各 1024（`:41`）；`normed*(1+scale)+shift`（`:42`）。
- 注释 `:39` 保留了原零初始化写法，现改为小随机初始化：**初始不是严格恒等**（kernel 非零，bias 默认零初始化）。对比之下 openpi 原版 adaRMS 的 Dense 是 `zeros` 初始化（`gemma.py:128`）。
- 参数：Dense(1024→2048, bias) = 2,099,200 / 层。
- RMS 方差在 f32 计算，eps 1e-6，结果转回输入 dtype。

### 6.4 一层的公式（伪代码，action expert 一侧）

```
# x: (B,T,1024) action expert 残差流；M: (B,608,1024) mem_seq；m: (B,608) mem_mask；c: time 条件
h, g1 = adaRMS_attn(x, c)                    # pre_attention_norm_1
x = x + g1 * Attn([prefix..., h])            # 与 paligemma 联合注意力（原版）

xq, Mk = RMS_s(x), RMS_s(M)                  # mem_rms_norm，共享 scale
q = rope(xq @ Wq, 608..608+T-1) / 16         # (B,T,4,256)
k, v = rope(Mk @ Wk, 0..607), Mk @ Wv        # (B,608,1,256)
p = softmax_f32(where(m, q·kᵀ, -2.38e38))    # 只屏蔽补位 key
u = (p @ v) @ Wo                             # mem_mod_vec (B,T,1024)

s, b = split(Dense_{1024→2048}(u))           # mem_rms_norm_ffn
z = RMS(x) * (1 + s) + b                     # 插入的调制
z, g2 = adaRMS_ffw(z, c)                     # pre_ffw_norm_1（原版）
x = x + g2 * FFN(z)                          # 残差用调制前的 x
```

### 6.5 每层独立、每步重算、参数量

- `nn.scan(variable_axes={"params":0}, length=depth=18)`（`history_gemma.py:241-260`），每层各有一套 `mem_attn` 与 `mem_rms_norm_ffn`，参数不共享。
- Q/K/V 每层、每个去噪步都重算（memory 本身在推理中只算一次，见第 7 节；但它在每层投出的 K/V 不缓存）。块用 `jax.checkpoint` 且 `policy=nothing_saveable`（`:236-240`），训练反传时整块重算。
- 每层新增 2,622,464 + 2,099,200 = **4,721,664**；×18 = **84,989,952**（按形状推算）。全部只在 action expert 一侧（参数名带 `_1` 一侧），paligemma 一侧无 memory 参数。

**代码锚点**：`history_gemma.py:27-43,45-108,111-205,236-260`；`mme_vla_suite/models/representation/utils.py:5`；`openpi/models/gemma.py:69-77,113-131,284-331,453-459`。

---

## 7. 训练与推理同路

- **训练** `compute_loss`（`history_pi0.py:547-558`）：modulation 分支调 `embed_memory`，把 `[prefix_tokens, suffix_tokens]` 与 `mem_seq=[None, mem_seq]`、`mem_mask=[None, mem_mask]` 一次送入 LLM。
- **推理** `sample_actions`（`:609-628`）：先用 prefix 建 KV cache（不含 memory），再在**去噪循环外**算一次 `embed_memory`（`:616`）；循环 `step` 内每个去噪步把同一份 `mem_seq`/`mem_mask` 传入（`:669-682`），由各层 `MemoryAttention` 重新投 K/V。
- **在线构造**：`FrameSampMemory._encode_window`（`framesamp_memory.py:189-212`）尾部补帧复用 `es-1`（repeat_last），编码得 (1024,)；`visible_motion_frames`（`:235-248`）按第 2 节同式判定可见窗；`_prepare_motion`（`:250-269`）给出 emb (B,1024) f32、pos = `pos_emb[f,0,:256]`、mask、times；`policy.py::_prepare_history`（`mme_vla_suite/policies/policy.py:144-168`）组装帧路 + motion 路，并在 `:166` 调用与离线相同的 `memory_order`。
- **增强与 dropout 不作用于 memory**：图像增强只在 `preprocess_observation(train=True)` 对当前帧做（`openpi/models/model.py`），推理传 `train=False`；memory 一直从干净帧离线（或在线同式）抽取，不增强；encoder `eval()` 下 dropout 恒等；`HistoryBlock` 的 dropout 默认 0.0（1026 计划 §2.4）。

**代码锚点**：`history_pi0.py:547-558,609-628,669-682`；`mme_vla_suite/policies/framesamp_memory.py:189-212,235-269`；`mme_vla_suite/policies/policy.py:144-168`；`openpi/models/model.py`。

---

## 8. 「512」与「512+motion」两条 run 的差别

| 项 | 512 | 512+motion（stage2） |
|---|---|---|
| history yaml | 无 `motion` 节 | `motion` 节：enabled、source_shape [1,1024]、readout `stage2_pool4_concat_v1`、num_tokens 1、dim 1024、budget 96、pos_dim 256、demo_min_real_frames 17、demo_tail_pad repeat_last、stride 32、window_frames 33、forward、segment_start、frame_size 256、store_path、source_run、checkpoint_sha256、encoder_kind、online_gpu |
| memory token 数 | 512（帧路） | 512 + 96 = 608 |
| Dataset 键 | 四个 motion 键为 None | `motion_emb`/`motion_pos`/`motion_mask`/`mem_order` 全有 |
| PerceptualMemory 参数 | 3,475,200 | 5,508,608（多 `motion_pos_proj` + `motion_encoder_static` 共 2,033,408） |
| MemoryAttention K/V 长度 | 512 | 608 |
| MemoryAttention / MemoryRMSNorm / action expert | 结构相同（每层 4,721,664，共 84,989,952） | 同左 |
| 共同配置 | budget 512、num_views 1、token_per_image 64、use_pos_emb true、use_state_emb false、integration_type modulation、memory_token_dim 1024、representation perceptual / frame_sampling | 同左 |

**速度**：只有 10-17 同批 4+4 并行实测可引用——同口径（5000→75000 平均）512 为 1.7183 s/步、512+motion 为 1.7346 s/步，慢约 0.95%（`docs/training-doc/mixed8-200-m8x8-modul-motion-b128-80k/result.md:66`）。注意那条 motion run 是旧的 160×768 motion，不是本文的 96×1024 stage2 motion；stage2 版的步时本文不推断。

**代码锚点**：`mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml`；`framesamp_dataset.py:54-65`；`percep_mem.py:28-44`。

---

## 9. 已知口径差异与旧注释勘误（以代码为准）

1. 帧路投影输入是 2816（image 2048 + pos 768）→ 1024，不是 2048 → 1024。
2. motion 路参数 2,033,408；1026 计划写的 1,836,032 只是 `motion_encoder_static`。
3. `MemoryRMSNorm` 是插入在 `pre_ffw_norm` 之前，不是替换 `pre_ffw_norm`。
4. `MemoryAttention` 是 4 头（×256，1 KV 头），不是 action expert 的 8 头。
5. `gemma_150m` 在 modulation 下未构建、未使用。
6. 多处旧注释仍写 768 维，实际 1024：`history_observation.py:26-29`、`robomme_policy.py:68`、`training/config.py:314`、`framesamp_memory.py:252`。
7. （本文核对补充）`percep_mem.py:42` 注释写「1536 → memory_token_dim」，stage2 配置下实际输入为 1024+768 = 1792。
8. yaml 里 `streaming_obs_horizon` 与 `pool_type` 两个键未被传入使用：`pool_type` 作为构造参数出现在 `mme_vla_suite/dataset_builder/mem_buffer.py:37,45`，但 yaml 的值没传进去；`streaming_obs_horizon` 只出现在 yaml。
9. `MemoryRMSNorm` 注释 `:39` 说明已由零初始化改为 `normal(0.002)`，初始非严格恒等；不要按「零初始化恒等」理解。

---

## 10. 未核实项

- 训练侧 `visible_motion_rows`（`motion_store.py`）与在线 `visible_motion_frames` 同式，目前**仅依据注释与 1026 计划描述**，本文未逐行对拍两侧实现（1026 计划 §2.3 曾指出训练侧步长写死在 `GRID_STRIDE`、需按 layout 参数化）。
- 补位位置经 bias 得到的「right padding token」向量在离线与在线两侧数值是否逐位一致，未核实。
- 第 4、6、8 节全部参数量（5,508,608、4,721,664、84,989,952 等）均按权重形状推算，未实际加载模型统计参数树。
- 第 3 节各字段的每样本字节为按形状 × dtype 推算，未在 dataloader 上实测。
- suffix 长度 T 的具体取值本文未核对。
- 第 6.1 节关于「残差用调制前的 x」「pre_ffw_norm 重新归一化」两点是对代码的直接读取，未做数值实验验证其对训练的影响。
- stage2 96×1024 motion run 的步时未在本文引用任何实测（第 8 节只引用旧 160×768 版本）。
