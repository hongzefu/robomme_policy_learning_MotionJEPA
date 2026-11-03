# 说明：pi0.5 上的 context 512 + motion 1×1024（concat）是怎么接的、encoder 怎么推理、预存与重建脚本在哪

> 对象：AWS 8×A100 上训练的 `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k`（官方 16 任务×100 集，60k 步，global batch 128，seed 42），history 配置 `perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml`（在训练分支 `origin/v4-256concat1`；评估副本资产目录里的 `history_config.resolved.yaml` 就是它的解析快照）。
> 代码锚点一律写 `文件::类/函数/配置键`，不写行号。本文只描述现状，不是实施授权。
> 评估结果：官方 test 800 局、三 seed（7/42/0）均值 **37.33 ± 0.59%**（`docs/evaluation-doc/o16-3seed-1031/result.md`）。

## 目录

1. [一句话总览](#一一句话总览)
2. [在 pi0.5 上怎么加 context：两路记忆、一个前缀](#二在-pi05-上怎么加-context两路记忆一个前缀)
3. [encoder 怎么推理：帧路 SigLIP、运动路 Wan VAE + MotionJEPA stage2](#三encoder-怎么推理)
4. [预存了什么、H5 在哪一步用、推理时怎么重建](#四预存了什么h5-在哪一步用推理时怎么重建)
5. [训练与推理一致性是怎么保证的](#五训练与推理一致性是怎么保证的)
6. [数字速查](#六数字速查)

---

## 一、一句话总览

基座是 openpi 的 **pi0.5**（PaliGemma 2B 主干 + Gemma 300M 动作专家，flow matching 出 20 步动作）。我们没有改 pi0.5 的任何层，而是用 `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0` 把它包一层：在 PaliGemma 的 prefix **最前面**多拼一段「记忆区」，记忆区由两路 token 按时间交错组成：

| 路 | 来源 | 每个 token 代表什么 | 位置数 | 原始维度 | 投到 2048 维的层 |
|---|---|---|---|---|---|
| 帧路（context 512） | 历史帧经 pi0.5 自带的 SigLIP，再 4×4 池化 | 某一历史帧「长什么样」 | 32 帧 × 16 token = **512** | 2048（bf16）+ 768 维时间码 | `FeatureEncoder.encoder_static`：Linear(2816→2048) |
| 运动路（motion 1×1024） | 33 帧窗口 → Wan2.1 VAE → MotionJEPA 阶段二 encoder（EMA） | 某一段 33 帧「发生了什么运动」 | 最多 **96** 个窗口，每窗 1 个 token | 1024（f32）+ 256 维时间码 | `PerceptualMemory.motion_encoder_static`：Linear(1792→2048) |

两路合计 608 个记忆位，加上当前帧图像 512 + 文本 64，prefix 为 1184；再加 20 个动作 token，全序列 1204。相对不带 motion 的纯 context 512 模型，每样本训练 FLOPs 只多 **8.9%**（`results-now.md` §2.4）。

「concat 1×1024」的含义：阶段二 encoder 每窗输出 (4, 256)，**concat** 读法是把它展平成**一个** 1024 维 token（`readout: stage2_pool4_concat_v1`）；对照的 **split** 读法（`stage2_pool4_split_v1`）是把同一份输出拆成 4 个 256 维 token、占 384 位。两者训练库同一张、encoder 同一个，只差 YAML 的 `readout / num_tokens / dim / budget` 四键。

---

## 二、在 pi0.5 上怎么加 context：两路记忆、一个前缀

### 2.1 总开关与配置键

history YAML 的 `integration_type: context` 决定接入方式（另两种 `modulation`、`expert` 不在本文范围）；`memory_token_dim: 2048` 是记忆 token 投到主干宽度；`budget: 512`、`token_per_image: 16` 定帧路；`motion.*` 节定运动路。本模型的 motion 节原文：

```yaml
motion:
  enabled: true
  source_shape: [1, 1024]          # 落盘每行形状
  readout: stage2_pool4_concat_v1  # 一窗一个 1024 维 token
  num_tokens: 1
  dim: 1024
  budget: 96                        # 最多 96 个窗口位置，零截断
  pos_dim: 256                      # 时间码取 pos_emb 前 256 维
  demo_min_real_frames: 17          # demo 段尾窗最少 17 个真实帧，不足 33 用末帧补
  demo_tail_pad: repeat_last
  stride: 32                        # 段内绝对网格 0,32,64,…（阶段二口径；旧 768 维 encoder 是 16）
  window_frames: 33
  window_direction: forward
  grid_origin: segment_start
  frame_size: 256
  store_path: v1-store/datasets/16task-orig100-1600ep-9e109419/motion-stage2-1024
  source_run: stage2-full144-1025-a/checkpoint_epoch_105.pt#model
  checkpoint_sha256: 27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
  encoder_kind: stage2_pool4_concat
  online_gpu: 1
```

`src/mme_vla_suite/datastore/motion_adapter.py::motion_policy_spec` 把这一节解析成 `MotionPolicySpec`：readout 为 stage2 时强制 `source_shape=[1,1024]、num_tokens=1、dim=1024`，布局名必须是 `motion-stage2-1x1024-grid32-demopad17-v1`。

### 2.2 帧路：32 帧 × 16 token

- **选帧**：`src/mme_vla_suite/shared/sampling.py::even_sampling_indices(step, 32)`——当前步 `t` 不足 32 帧时全取，否则在 `[0, t]` 上 `linspace` 等距取 32 帧。训练与在线共用同一函数。
- **每帧特征**：SigLIP So400m（pi0.5 自带的视觉塔）输出 256 个 2048 维 token，`pool_tokens_to_size` 均值池化到 4×4 = 16 个 token。所以 32 帧 × 16 = 512 个位置。
- **时间码**：`pos_emb` 是帧号 `t` 的纯函数（3D 位置编码，768 维），训练库只存一份；每个 token 带所在帧的时间码，state 路本模型关闭（`use_state_emb: false`）。
- **投影**：`src/mme_vla_suite/models/representation/mem_encoder.py::FeatureEncoder`——`concat(image_emb[2048], silu(pos_proj(pos)[768→768]))` → `encoder_static` Linear(2816→2048)。`img.net: identity`，图像特征本身不再过网络。

### 2.3 运动路：最多 96 个窗口，每窗 1×1024

- **窗口**：从起点 `f` 往后连续 33 帧 `[f, f+32]`（前视，与 encoder 训练语义一致），尾端不得超过当前帧。
- **起点网格**：钉在**段内绝对位置** `0, 32, 64, …`，demo 段与 exec 段各自起算、窗口不跨段。当前帧只决定哪些起点「已可见」，网格本身不随时间平移，因此离线表小、在线每 32 帧只新编 1 个窗。demo 段尾窗允许只有 17 个真实帧、其余用 demo 最后一帧补齐；exec 段始终要求完整 33 帧。公式在 `src/mme_vla_suite/datastore/motion_store.py::visible_motion_rows`，训练侧与在线侧 `import` 同一份。
- **预算 96、零截断**：合法起点数超过 96 直接 raise，不抽稀（官方评估口径）。没凑够的位置右填充零向量、`motion_mask=False`。
- **投影**：`src/mme_vla_suite/models/representation/percep_mem.py::PerceptualMemory`——`pos_h = silu(motion_pos_proj(motion_pos[256]))`（256→768），`motion_encoder_static(concat(motion_emb[1024], pos_h[768]))`（1792→2048）。本模型新增可训练参数：`motion_pos_proj` 197,376 + `motion_encoder_static` 1792×2048+2048 = 3,672,064，合计约 **3.87 M**；`motion.enabled=false` 时这两层根本不创建。

### 2.4 两路怎么合成一段记忆区

`PerceptualMemory.__call__` 先返回并列序 `(b, 512 + 96, 2048)`；`HistoryPi0.embed_memory` 再用 `mem_order` 重排：

- `src/mme_vla_suite/shared/sampling.py::memory_order(frame_times, 16, motion_times)`：键 = `时刻×2 + 类型`（帧 0、motion 1），稳定排序；同一时刻帧 token 在 motion 前，同帧 16 位保持内部次序，两路 padding 一并落到尾部。产出后显式校验是合法置换。
- `jnp.take_along_axis(tokens, mem_order)` 与 `take_along_axis(mask, mem_order)` 同步重排。结果是一条**按时间交错**的 608 位记忆区，时间上相邻的「样子」和「运动」token 靠在一起。

### 2.5 拼进 prefix 与 attention 规则

`HistoryPi0.embed_prefix`：`integration_type == "context"` 时先 `embed_memory`，记忆区 append 在最前，然后才是当前帧图像（SigLIP 512 token）与文本（64）。prefix 顺序：`[记忆 608 | 图像 512 | 文本 64]`。

attention 由 `history_pi0.py::make_attn_mask(input_mask, ar_mask, na_mask)` 决定：

- 记忆区 `ar_mask` 全 False，位于最前（cumsum 为 0）；第一个图像 token `ar=True`。因此记忆 token 之间**互看**（双向），记忆看不到图像与文本。
- 图像 token `na_mask=True`，`mask_not_attend` 屏蔽「query 或 key 是 na 且 key 落在记忆区」的格子，所以**图像 token 看不到记忆**（保持 SigLIP→PaliGemma 的原始视觉流不被历史干扰）；文本 token 和动作 token 可以看到记忆。
- 动作专家的 20 个 suffix token 按 pi0.5 原规则看到全部 prefix（含记忆），再走 10 步去噪。
- 位置编码：`positions = cumsum(input_mask) - 1`，记忆 token 占 RoPE 位置 0…607，padding 位不占号。

`HistoryPi0.sample_actions` 推理时先对 prefix 跑一遍 PaliGemma 得 `kv_cache`，去噪 10 步只重算 suffix——记忆区的开销只付一次。

### 2.6 训练时数据从哪来

`src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__getitem__` 产出 8 个键：

| 键 | 形状 | 来源 |
|---|---|---|
| `static_image_emb` | (512, 2048) bf16 | packed 帧表 `framesamp-4x4/image_emb_4x4/part_*.bf16.bin`，按 `row_of(g, t)` 直读 |
| `static_pos_emb` | (512, 768) f32 | `pos_emb_4x4.f32.bin` |
| `static_state_emb` | (512, 8) f32 | `state_emb.f32.bin`（本模型 `use_state_emb=false`，模型侧不用） |
| `static_mask` | (512,) bool | 真帧 True、padding False |
| `motion_emb` | (96, 1024) f32 | motion 表 `motion-stage2-1024/motion_token.f32.bin`，按 `visible_motion_rows` 查 `meta/motion_index.json` 取行，`reshape(k, 1024)` |
| `motion_pos` | (96, 256) f32 | 起点帧 `pos_emb[f, 0, :256]` |
| `motion_mask` | (96,) bool | 前 k 个 True |
| `mem_order` | (608,) int32 | `memory_order(...)` |

`check_motion_contract` 在构造时核：表布局 = `motion-stage2-1x1024-grid32-demopad17-v1`、`provenance.encoder` 为阶段二 `model/ema`、`checkpoint_sha256` 与 YAML 一致、`stride` 与表的 `grid_stride` 一致。openpi 的 `_collate_fn` 对每键 `np.stack` 加 batch 维，`HistAugObservation.from_dict` 接收。

---

## 三、encoder 怎么推理

两路 encoder 都是**冻结**的，训练时不反传到它们；只训 pi0.5 本体 + 上面两组投影层。

### 3.1 帧路：SigLIP

- **离线建库**（训练用）：`src/mme_vla_suite/dataset_builder/build_robomme_dataset.py` 逐 episode 读原始 H5 的 `episode_{i}/timestep_{t}/obs/front_rgb`（256×256 uint8），resize 到 224，用 `siglip_tokenizer.py::SigLipTokenizer` 加载 f32 的 `siglip_params.pkl`（资产锁 `siglip_params`，HF `Yinpei/pi05_vision_encoder`）逐帧编码，写 `source/features/.../token_emb_{t}.npy` 与每步 `data/{idx}.pkl`；`scripts/dataset/pack_framesamp_store.py pack` 再做 4×4 池化打成三张裸 `.bin` 表。
- **在线**（评估用）：`src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory.add_buffer` 把新到的帧归一化、`resize_with_pad` 到 224，调 `HistoryPi0.vision_encode`（checkpoint 里的 bf16 SigLIP）整批编码，`pool_tokens_to_size` 池化到 16 token，存进 `_history_feats[step]`。

两侧差别：离线 f32 逐帧、在线 bf16 整批，相对差约 0.33%，用户早已拍板接受（`docs/train-infer-consistency.md`）。

### 3.2 运动路：Wan2.1 VAE → MotionJEPA 阶段二 encoder

**输入**：`(33, 256, 256, 3)` uint8 原始帧（不是 224 的 SigLIP 输入，所以在线侧要单独缓存 256 域原图 `_raw_frames`）。

**第一跳 Wan VAE**：`scripts/dataset/wan/wan_motion_infer.py::encode_chunk`——`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 `AutoencoderKLWan`，fp32、关 TF32、关 tiling/slicing、batch 恒 1；像素 `/127.5 - 1` 到 [-1,1]，`vae.encode(x).latent_dist.mode()`（取均值不采样），输出 `(1,16,9,32,32)` 转成组优先 `(9, 16, 32, 32)` f32 latent（33 帧压成 9 个时间组，每组 16 通道 32×32）。一窗 VAE 约 0.85 s（RTX 6000 Ada 实测，`docs/dataset-build-doc/4task-motion-40ep/result.md`），是运动路耗时的主体。

**第二跳 阶段二 encoder**：MotionJEPA 仓库 `src/motion_jepa/models/wan_latent_encoder.py::Stage2WanLatentMotionEncoder`（锚点 commit `b5d23a5`，经 `scripts/dataset/wan_stage2/pyproject.toml` 以 git 依赖钉死）：

- 9 组 latent 先按 checkpoint 内的 `latents_mean/std` 做 fp32 affine 归一化；每组 32×32 按 4×4 patch 切成 64 个 patch token（共 576），`input_proj` Linear(256→256)，加可学习的 t/h/w 绝对位置。
- 另从归一化 latent 上做 8×8→4×4 掩码均值池化，取**来源时刻槽 [0, 3, 5, 8]**（9 组中的 4 组）得 4 个内容池化 token，`pool_proj` Linear(256→256)，加对应时间位置。
- 580 行一起过 8 层 Transformer（宽 256、4 头、SwiGLU、3D RoPE），attention 是**块因果**（`block_ids` 小于等于才可见，4 个读出 token 的块号为 9，能看到全部 patch）。
- 取末尾 4 个读出位，`final_norm` + `output_proj` Linear(256→256)，得 `(4, 256)` f32。

**导出接口**（离线与在线唯一入口）：`src/motion_jepa/export/stage2_memory.py::encode_windows`——要求 eval 模式、fp32、关 TF32、`autocast(enabled=False)`，输出 `(B, 4, 256)` 后 `reshape(B, 1024)`。`load_ema_encoder` 加载 `stage2-full144-1025-a/checkpoint_epoch_105.pt` 的 `model` 键（`weight_type=ema`，arch `wan-latent-stage2-blockwise-pool4-three-decoder`），逐张量核 EMA 状态精确相等，并产出 65 个叶的 `encoder_state_sha256` 身份。

所以每个窗口最终是一个 **1024 维 f32 向量**（4 个时刻槽 × 256），这就是 `motion_emb` 的一行。

### 3.3 在线推理时运动路怎么跑（评估 server 内）

```
benchmark client ──websocket──▶ serve_policy.py（JAX 主进程，GPU 0）
                                   │ MME_VLA_Policy.add_buffer / infer
                                   │ FrameSampMemory._encode_ready_windows
                                   │     33 帧 uint8（6,488,064 B）
                                   ▼  Unix socketpair，一次一窗，Lock 串行
                              stage2_sidecar.py（torch 子进程，GPU=motion.online_gpu）
                                   │ wan_motion_infer.encode_chunk → stage2_memory.encode_windows
                                   ▼
                              (1024,) f32，4096 B 回包
```

- **进程**：`src/mme_vla_suite/policies/policy_config.py::create_trained_policy` 读 run 根的 `history_config.resolved.yaml`（核 sha256）与 `motion_provenance.json`，`encoder_kind == "stage2_pool4_concat"` 时构造 `src/mme_vla_suite/policies/motion_client.py::MotionEncoderClient`，由它 `Popen` 起 `scripts/dataset/wan_stage2/stage2_sidecar.py`：独立 uv 子环境 `v1-store/venvs/wan-stage2`（torch 2.9.0 + diffusers 0.39.0），`CUDA_VISIBLE_DEVICES=<online_gpu>`，socketpair 经 `pass_fds` 传入。一个 policy 配一个 sidecar，跨 episode 常驻。
- **握手**：sidecar 先核 `SOURCE_PIN.json`（`wan_motion_infer.py` 复制件 sha、`uv.lock` sha、实际导入的 MotionJEPA 包 `source_pin()` 三字段），加载 VAE（核权重指纹）与 EMA encoder（核 checkpoint sha），把 `encoder / encoder_state_sha256 / source_pin / numerics / vae` 五组 provenance 发回；客户端与 run 内 `motion_provenance.json`（= 离线建表时的 provenance）逐键比对，任一不等即 raise。协议见 `src/mme_vla_suite/policies/motion_protocol.py`。
- **增量编码**：`FrameSampMemory.add_buffer` 每收一批帧（首批是整段 demo + exec 第 0 帧共 `es+1` 帧，之后每批 16 帧）就调 `_encode_ready_windows`：demo 段的所有起点首批一次编完；exec 段用 `while` 把「本批末帧已凑齐 33 帧」的起点全部编掉，每个起点编一次、结果存 `_history_feats_motion[f]`；编完就丢掉不再需要的原始帧。stride 32 意味着稳态每 2 个 action chunk 才新编 1 窗。
- **组包**：`FrameSampMemory._prepare_motion(step)` 取当前可见起点（与训练同一公式）、填 (96,1024)/(96,256)/(96,) 并右填充，`MME_VLA_Policy._prepare_history` 再用同一个 `memory_order` 算 `mem_order`，交给 `sample_actions`。

---

## 四、预存了什么、H5 在哪一步用、推理时怎么重建

### 4.1 原始 H5 只在建库时读一次

RoboMME 原始 H5（每任务一个 `record_dataset_<Task>.h5`，`episode_{i}/timestep_{t}/obs/front_rgb` 256×256 uint8，共 16 任务×100 集）**只在建库阶段被读**：

- SigLIP 建库读它出帧特征（§3.1）；
- Wan 抽取 `scripts/dataset/wan/extract_wan.py`（经 `wan_common.py::read_segment_frames`）读它切 33 帧窗口→latent；
- 打包 `pack_motion_store.py pack --raw-dir …` 对 H5 做 sha256 写进 `input_manifest`，绑定血缘。

训练与推理**都不读 H5**。训练读下面两张预存表；推理读环境实时帧。

### 4.2 预存产物（训练库 `16task-orig100-1600ep-9e109419`，AWS 建、公开 bucket 发布）

| 目录 | 内容 | 读者 |
|---|---|---|
| `meta/episode_manifest.json`、`meta/input_manifest.json` | 1600 集清单（canonical 顺序）、16 个 H5 的 sha256 | 全部 |
| `source/data/{idx}.pkl`、`source/features/` | 每步样本（动作、prompt 等）与逐帧 SigLIP 原始特征 | `FrameSampDataset` 必读 pkl；features 只做快速校验 |
| `framesamp-4x4/` | 帧表三张裸 `.bin`（`image_emb_4x4` bf16、`pos_emb_4x4` f32、`state_emb` f32），`meta/store_meta.json` 状态 `verified` | 训练帧路 |
| `wan-latents/` | 逐段 `.bin`，每窗 589,824 B = `(9,16,32,32)` f32，2496 段，26.8 GB | 只用于编 motion 表、对拍；训练不读 |
| `motion-tokens-stage2/` | 导出脚本产出的逐段 `.f32.bin`，每窗 4096 B，共 95 MB | 打包输入 |
| `motion-stage2-1024/` | **motion 表**：`motion_token.f32.bin`（23,253 行 × (1,1024) f32）、`meta/motion_index.json`（段基址，唯一身份）、`meta/store_meta.json`（含 `provenance.encoder/vae/numerics/source_pin/…`）、`meta/row_digests.blake2b.bin` | 训练运动路；在线握手核对的 provenance 真源 |
| `evidence/` | `G0.json`（64 个固定样本窗）、`G2.json`、`G3.json`、`SOURCE_PIN.json`、`wan_cache_import.json` | 训练起跑与封口时核；在线恢复复算 |
| `assets/robomme/norm_stats.json` | 动作/状态归一化统计 | 训练与推理 |

表格式层唯一实现是 `src/mme_vla_suite/datastore/motion_store.py` 与 `framesamp_store.py`，打包器与 dataset 都从这里 import，不复制。

### 4.3 「重新构造推理 encoder」的脚本一览

有，而且分离线重建与在线复算两套：

| 用途 | 脚本 | 做什么 |
|---|---|---|
| 离线：帧 → latent | `scripts/dataset/wan/extract_wan.py`（经 `run_local.py --stage wan` 多卡调度） | 33 帧窗 → `encode_chunk` → `wan-latents/<段>.bin` + `.sha256` + `.metadata.json` |
| 离线：latent → 1024 维 token | MotionJEPA `scripts/stage2_memory_export.py`（GL 分支 `v4-gl-orig100-1031` 以单文件 vendoring 放在 `scripts/dataset/wan_stage2/export_stage2_memory.py`） | `load_ema_encoder` + `encode_windows`，参数 `--grid-subsample 2`（stride 16 的 latent 网格只取偶数位 = stride 32）、`--batch 1`、`--check-dropout 64`，写 `motion-tokens-stage2/` 与 export_record |
| 离线：打包成表 | `scripts/dataset/pack_motion_store.py pack --layout stage2 … --stage2-validation <evidence 目录> --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json`，再 `verify --resume --check-padding` | 按行序契约拼表、逐行 digest、写 provenance，`packed → verified` |
| 离线：数值闸门 | `scripts/dataset/wan_stage2/validate_stage2.py --gate G2 / G3 / G4-reference` | G2：缓存 latent 与现算 latent 对拍；G3：从原像素重算 encoder 输出、两条调用路径逐位相等；G4-reference：对 G0 的 64 窗生成参考 token 供在线对拍 |
| 在线：sidecar 本体 | `scripts/dataset/wan_stage2/stage2_sidecar.py` | §3.3 的子进程；`--stub` 模式只回帧号用于状态机测试 |
| 在线：复算对拍 | `scripts/training/check_stage2_online.py` | 真实客户端 + 真实 sidecar 对 G0 的 64 窗现算，与 `validate_stage2.py --gate G4-reference` 的参考值比对，输出 `ONLINE_OFFLINE_PARITY=PASS` |
| 交付恢复 | `scripts/training/check_stage2_checkpoint.py --mode delivery …` | 下载某步 checkpoint 后核清单、root 快照（`history_config.resolved.yaml`、`motion_provenance.json`、`encoder_dependencies/`、`online_restore_inputs.npz`）并做一次真实在线推理 |
| 启动合同 | `scripts/training/check_mixed8_stage2_1024_launch.py`（mixed8 版）、GL 分支的 `check_orig16_stage2_launch.py` | 起跑前核 argv、配置、数据身份、14 个记忆叶冷初始化、`motion_encoder_static` 形状 |

旧 768 维 encoder 那套（`scripts/dataset/wan/encode_motion.py`、`motion_sidecar.py`，输出 (8,256)、stride 16）仍在仓库，但**本模型不用**；两套靠 `encoder_kind` 分派（`policy_config.py` 按 `stage2_pool4_concat` 走 `MotionEncoderClient._init_stage2`）。

### 4.4 评估副本里随 checkpoint 一起发的东西

公开 bucket `HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1` 的根快照（本机评估副本 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/assets/vla-o16cc-59999-bd35a857…/`）除 Orbax 参数外还有：`history_config.resolved.yaml`、`motion_provenance.json`、`encoder_dependencies/stage2-full144-1025-a/`（encoder 权重与 config 实体，477 MB）、`motion_evidence/{G0,G2,G3,SOURCE_PIN,wan_cache_import}.json`、`online_restore_inputs.{json,npz}`（PatternLock ep36 的 161 帧真实在线输入，供异地做一次真实推理核身份）。Wan VAE、tokenizer、pi05_base 不随包发，按 `scripts/assets/ASSETS_LOCK.json` 的 sha 用本机缓存。

---

## 五、训练与推理一致性是怎么保证的

缺任何一条都「不报错、只静默降效果」：

1. **同一套起点公式**：`motion_store.py::visible_motion_rows`（训练）与 `framesamp_memory.py::visible_motion_frames`（在线）同式；`memory_order` 两侧 `import` 同一个函数对象。
2. **同一个编码入口**：离线导出与在线 sidecar 都走 `wan_motion_infer.py::encode_chunk`（VAE）+ `stage2_memory.py::encode_windows`（encoder），同样 fp32、关 TF32、batch 1、eval；sidecar 起手核 `SOURCE_PIN.json`。
3. **身份逐键比对**：在线握手 provenance 与离线表 `store_meta.provenance` 五组逐键相等（含 `encoder_state_sha256` 65 叶、VAE 权重 sha、torch/cuda/cudnn 版本、`numerics`）；训练 dataset 构造时核 YAML 的 `checkpoint_sha256` 与表一致。
4. **真实复算闸门**：`check_stage2_online.py` 的 `ONLINE_OFFLINE_PARITY`、交付恢复的真实在线推理。阶段二跨卡型（A100 建表、A40/A6000 评估）的数值差按用户裁决记 `NUMERIC_EQUIVALENCE=NOT_REQUIRED`，只要求身份与来源一致，不要求逐位。

---

## 六、数字速查

| 项 | 值 | 出处 |
|---|---|---|
| 记忆位 | 512 帧路 + 96 运动路 = 608 | YAML |
| prefix / 全序列 | 1184 / 1204 | `results-now.md` §2.1 口径 |
| 每样本训练 FLOPs（相对纯 512） | 15.17 TFLOP，×1.089 | `results-now.md` §2.4 |
| 新增可训练参数（运动路投影） | 3,672,064 | `percep_mem.py` 形状手算 |
| motion 表 | 23,253 行 × 4096 B；wan-latents 2496 段 26.8 GB | `1101-gl-ablation-encoders-ctx-b64-80k-plan.md` |
| encoder | `stage2-full144-1025-a` ep105 EMA，sha `27daa832…`，8 层宽 256，输出 (4,256)→1024 | `motion_provenance.json` |
| VAE | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，fp32，33 帧→(9,16,32,32) | `wan_motion_infer.py` |
| 在线单窗耗时 | 约 1.4 s（A100，VAE 占绝大部分；阶段二 encoder 本身毫秒级） | `docs/training-doc/aws-p5-online/result.md`（旧 encoder 实测，VAE 同一份） |
| 三 seed 成功率 | 37.33 ± 0.59%（split 38.42 ± 1.25，modulation split 43.42 ± 0.52，论文 FrameSamp+Context 30.68 ± 1.17） | `docs/evaluation-doc/o16-3seed-1031/result.md` |
| 训练 | AWS 8×A100-80GB，b128，60k，seed 42，`mme_vla_suite_orig16_b128_60k`，初始化 `pi05_base` | bucket README |
