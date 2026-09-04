---
license: other
license_name: internal-research-only
license_link: https://huggingface.co/HongzeFu/MotionJEPA/blob/main/README.md
library_name: pytorch
inference: false
tags:
  - motion-jepa
  - jepa
  - world-model
  - latent-dynamics
  - motion-token
  - robotics
  - manipulation
  - wan2.1
  - video-latent
  - robomme
language:
  - zh
  - en
---

<!-- ⚠ 私有仓库专用。本卡与 wan-v8-filter10-72ep-a/SHA256SUMS.src-vs-copy.txt 含内部集群绝对路径、
     私有 GitHub repo 名与未公开数据集信息。转 public 前必须逐项清洗。 -->

# MotionJEPA · `wan-v8-filter10-72ep-a` · epoch 72

**TL;DR (EN).** Self-supervised motion encoder + latent DiT decoder trained on Wan2.1-VAE latents of
robot manipulation video (4 RoboMME tasks × 400 episodes, private recording). The encoder maps a
9-step latent block `(9, 16, 32, 32)` to a single 768-d motion token; the decoder is the training-time
latent predictor. Single checkpoint: **epoch 72, the 36th and final save of the run**, EMA + live
weights, **no optimizer state**. Architecture tag `wan-latent-v7`. For inference, load the EMA key
`encoder` with `strict=True` and do **not** hand-fill the affine buffers.

---

## 1. 四问四答（本卡最重要的一节）

| 问题 | 答案 |
|---|---|
| **在哪个 repo 训练的？** | `https://github.com/hongzefu/MotionJEPA`（**private**，匿名访问返回 404） |
| **在哪个 commit 训练的？** | **`7388a42`**（`commitV8.4Beta`，2026-08-20 00:10:28 -0400，分支 `v6.1.1-slurmWanExtract`） |
| **在哪个数据集上训练的？** | `dataset-4env-v8`（见第 5 节），源自私有录制 `robomme_data_h5_v2_4env400ep`（4 任务 × 400 episode） |
| **这是第几个 ckpt？** | **第 36 个，也是最后一个**。`checkpoint.save_every: 2` → 该 run 共存 epoch 2,4,…,72 计 36 个；本 repo 只上传 epoch 72 |

**关于 commit 的一条必须知道的补充**：下游消费方（`robomme_policy_learning_MotionJEPA`）的推理脚本钉的是
MotionJEPA `2a484ad960ed6155321dc34def9011eb119f857f`（2026-09-02），**与训练 commit `7388a42` 不是同一个**。
两者之间全树改了 163 个文件，但已实测 `git diff --stat 7388a42 2a484ad -- src/motion_jepa scripts/train.py`
**零改动**——模型定义与训练脚本逐字节相同。所以用 `2a484ad` 重建模型加载本权重是安全的；用更晚的 commit
则必须重跑一次这条 diff 才能作同样的断言。因为 MotionJEPA 是私有 repo、外部无法自行核验，这条判据写在卡里。

---

## 2. 仓库内容

```
README.md                                        本卡
SHA256SUMS.txt                                   2 行清单，路径相对本目录
wan-v8-filter10-72ep-a/
  ├── checkpoint_epoch_72.pt      954,853,147 B  sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
  ├── config.yaml                       1,927 B  sha256 99548a6ca23522c235281e45819ae6d5e96a916709cb4b9c0b47142832c90946
  └── SHA256SUMS.src-vs-copy.txt          704 B  训练集群源盘 vs 本地复制件同 sha 的原始记录
```

下载并自验：

```bash
hf download HongzeFu/MotionJEPA --revision wan-v8-filter10-72ep-a-e72 --local-dir ./MotionJEPA
cd ./MotionJEPA && sha256sum -c SHA256SUMS.txt      # 两行必须 OK
# 此后 ./MotionJEPA/wan-v8-filter10-72ep-a 就是 load_encoder 要的 run_dir
```

`wan-v8-filter10-72ep-a/` 这个目录名不是装饰：它对应 MotionJEPA 的 `runs/<run_name>/` 布局，
加载器按 `run_dir/config.yaml` + `run_dir/<checkpoint_name>` 取文件，改名会打断该约定。

---

## 3. 模型结构与参数量

两个模块，同一个 checkpoint：

| 模块 | 类 | 参数量 | 作用 |
|---|---|---:|---|
| **encoder** | `motion_jepa.models.WanLatentMotionEncoder` | **69,808,896** | `(B,9,16,32,32)` 原始 Wan latent → `(B,1,768)` motion token |
| **decoder** | `motion_jepa.models.WanLatentDiT` | **49,270,272** | 训练期的 latent 残差预测器：`(z₀, motion, k) → ẑ_k`（归一化域） |

四份 state_dict 全为 fp32；按 (69,808,896 + 49,270,272) × 2 套 × 4 B ≈ 952.6 MB 估算，与文件
954,853,147 B 相符（余量是冻结 config 与 pickle 结构开销）。

**encoder 前向**：入口 affine 归一化 → 断言序列长 `== seq_len=9` 与形状 `(16,32,32)` →
`input_proj: Linear(16*32*32=16384 → 768, bias=False)` → 8 层 `EncoderBlock`（12 头 × 64 dim_head，
mlp_ratio 4，RMSNorm，1D RoPE `max_seq_len=9`）→ `final_norm: RMSNorm(768)` →
`output_proj: Linear(768 → 768, bias=False)` → GPT 式末位读出 `num_tokens=1` 个 token。

**decoder 前向**（`WanLatentDiT.forward(current_latent, motion_tokens, timestep)`）：`z₀` 归一化 →
2×2 patchify 得 `(B,256,64)` → `input_proj + pos_embed` → 正弦 timestep 嵌入 → 4 层
`LocalMotionDiTBlock`（12 头，局部注意力 `window_size=7`，grid 16×16，motion token 交叉注入）→
零初始化 `output_proj` → **返回 `ẑ₀ + residual`，在归一化域**。

**arch 标签**：`ckpt["arch"] == "wan-latent-v7"`。加载器会硬校验这个字符串，不等即 raise。

---

## 4. checkpoint 内部结构：四个 state_dict，该用哪个

顶层键：

| 键 | 值 | 说明 |
|---|---|---|
| `epoch` | `72` | |
| `arch` | `'wan-latent-v7'` | 被硬校验 |
| `config` | omegaconf `DictConfig` | run 冻结配置（内容同 `config.yaml`） |
| `val_loss` / `val_sigreg_loss` / `val_latent_loss` | 全为 `None` | 因为 `training.validate: false`，本 run 未在线验证 |
| `ema_only` | `False` | |
| `has_live_weights` | `True` | |
| **`encoder`** | **77 键，EMA** | ✅ **推理用这个** |
| **`wan_decoder`** | **50 键，EMA** | ✅ 需要 decoder 时用这个 |
| `encoder_live` | 77 键，非 EMA 原始权重 | 续训用 |
| `wan_decoder_live` | 50 键，非 EMA 原始权重 | 续训用 |

**没有 `optimizer` 键**（`checkpoint.save_optimizer: false`）。

**取键规则：**

- **推理 / 特征抽取 → 一律取 EMA 主键 `encoder`（与 `wan_decoder`）。** 下游
  `wan_motion_infer.py::load_encoder` 就是这么写的：遍历 `ckpt["encoder"]`，不读 `encoder_live`。
  想复现下游已产出的 motion token 就必须走这条。
- **续训 / 恢复训练 → 按 MotionJEPA `scripts/train.py` 的口径**：
  `enc_sd = ckpt["encoder_live"] if ckpt["has_live_weights"] else ckpt["encoder"]`（本 ckpt
  `has_live_weights=True`，故取 `_live`）。但因为没有 optimizer 状态，**只能「重新起 Adam」式续训，
  不能逐位恢复动量与二阶矩**，见第 9 节。
- state_dict 的键可能带 `module.` / `_orig_mod.` 前缀（DDP / `torch.compile` 残留），加载前循环剥掉。
- **必须 `strict=True` 整份加载。** 77 个键里含 persistent buffer `latents_mean` / `latents_std`
  与 RoPE cache。这些常数 buffer 也随 EMA 一起存档，与直接从 Wan VAE config 读出的真值差约 1e-5。
  所以**禁止**「构造模型 → 调 `load_wan_latent_stats()` 手填 affine 常数 → `strict=False` 加载权重」这条路，
  那样得到的 token 与已有产物不逐位一致。正确做法是构造时 `latents_mean=latents_std=None`
  （代码会填 NaN 占位），由 strict 加载从 ckpt buffer 带入真值，加载后断言
  `torch.isfinite(encoder.latents_std).all()`。

**安全提示**：这是 `torch.save` 的 pickle 文件，且顶层含 omegaconf `DictConfig`，
必须 `torch.load(..., weights_only=False)` 才能读。只在信任来源时加载。

---

## 5. 训练数据

- **数据集名**：`dataset-4env-v8`。训练时路径
  `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/dataset-4env-v8`。
- **原始来源**：`/data/hongzefu/robomme_data_h5_v2_4env400ep` —— RoboMME 4 个任务
  （`ButtonUnmask` / `ButtonUnmaskSwap` / `VideoUnmask` / `VideoUnmaskSwap`）× 400 episode 的**私有录制版**。
- ⚠ **这不是 Hub 上的 `Yinpei/robomme_data_h5`。** 后者是 16 任务 × 100 episode 的公开版，
  与本 run 用的 4 任务 × 400 episode 私有版**不是同一份数据**，任务集合与每任务 episode 数都不同。
  想复现训练必须拿到私有录制版。
- **切段与编码**：2,400 个视频变体（1,600 exec + 800 demo）；frame_size 256；窗口 33 raw 帧；
  K = 8 个 latent 步；stride 1。用**冻结的 Wan2.1-T2V-1.3B VAE** 编码为 `(9, 16, 32, 32)` fp32 latent
  （组优先，单 chunk 589,824 B）。全库 **396,302 chunk / 218 GiB**。
- **运动过滤 `filter10`**：`data.motion_filter.threshold = 0.045993`（v7 全量 `rgb_mag` 分布的第 10 百分位，
  v8 沿用这个冻结阈值，不重算）。keep **355,492** / drop **40,810**（**10.30%**）。
- **划分**：holdout `episodes 90-99` → train **2,340 变体 / 346,190 chunk**；
  val **60 变体 / 9,302 chunk**。`training.validate: false`，val 集在训练中未使用，仅保留作离线评估
  （所以 ckpt 里三个 `val_*` 都是 `None`）。
- `loss.arm_mask_weight: null` → 机械臂格加权路径未构造，`arm_chunk_masks` 未参与本 run。

---

## 6. 训练配置与指标

### 6.1 超参（全部出自 run 冻结 `config.yaml`）

| 组 | 值 |
|---|---|
| motion | `dim 768`，`num_tokens 1` |
| encoder | `hidden_dim 768`，`depth 8`，`heads 12`，`dim_head 64`，`mlp_ratio 4`，`dropout 0.05` |
| dit | `depth 4`，`hidden_dim 768`，`heads 12`，`dim_head 64`，`mlp_ratio 4`，`window_size 7`，`dropout 0.05` |
| wan | `latent_channels 16`，`latent_size 32`，`patchify 2`，`vae_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| data | `max_horizon 8`（→ `seq_len 9`），`stride 1`，`num_patches 256` |
| training | `max_epochs 72`，`batch_size 88`/卡，`grad_accum_steps 2`，2 GPU → **global 176 / 有效 352**；`lr 3e-4` cosine，`warmup_epochs 1`，`weight_decay 1e-4`，`gradient_clip 0.1`，`precision bf16`，`num_workers 4`，`compile false` |
| ema | `enabled true`，`decay 0.999` |
| loss | 归一化 latent MSE + SIGReg；`sigreg_weight 1e-4`，`sigreg_global true`，`sigreg_warmup_epochs 1.0`，`sigreg_warmup_floor_ratio 0.01`，`sigreg.knots 17`，`sigreg.num_proj 1024` |
| checkpoint | `save_every 2`，`save_optimizer false`，`save_live true` |
| 其他 | `seed 42`，`wandb.enabled false`（无 wandb run 可查，指标只在日志里） |

### 6.2 训练环境

- 集群 UMich **GreatLakes** `spgpu` 分区，Slurm job **`58355024`**，节点 **gl1511**。
- **2 × NVIDIA A40（46,068 MiB）**，`torch 2.9.0+cu128`。
- 窗口 2026-08-20 00:24 → 2026-08-21 13:48（约 37 h 24 min），**≈1,864 s/epoch**。
- **983 optimizer step / epoch × 72 = 70,776 step**（346,190 chunk ÷ 有效批 352 ≈ 983.5）。
- peak memory **35.74 GiB**。

### 6.3 指标

| epoch | `loss_total` | latent MSE | SIGReg |
|---|---|---|---|
| 1 | 0.03371 | 0.033089 | 27.5998 |
| **72（本 ckpt）** | **0.0013245** | **0.00116063** | **1.63899** |

epoch 72 附带表征健康指标：`motion/std_batch 0.99331`（token 各维批内标准差，接近 1 说明未塌缩），
`motion/cos_cross_chunk -1.105e-06`（跨 chunk 余弦，接近 0 说明不同片段的 token 未退化成同一向量）。

注：epoch 72 满足 `total = latent + 1e-4 × sigreg`（0.00116063 + 1.63899e-4 = 0.00132453）；
epoch 1 不满足，因为 SIGReg 权重仍在 1 epoch 线性 warmup 中（floor ratio 0.01）。

**本 ckpt 没有验证集指标**（`validate: false`）。上表都是训练集上的运行均值，
**不能当作泛化能力的证据**。

---

## 7. 最小加载示例

前置：`pip install "git+https://github.com/hongzefu/MotionJEPA@7388a42"`（需私有 repo 访问权；
`2a484ad` 在 `src/motion_jepa` 与 `scripts/train.py` 上与 `7388a42` 逐字节相同，见第 1 节）。
版本必须钉成第 8 节的那一组。

### 7.1 encoder：latent → motion token

```python
import os, yaml, torch
from motion_jepa.models import WanLatentMotionEncoder

RUN_DIR  = "./MotionJEPA/wan-v8-filter10-72ep-a"
CKPT     = "checkpoint_epoch_72.pt"
ARCH_TAG = "wan-latent-v7"
DEVICE   = "cuda"

pin_numerics(); check_env()          # 第 8 节，必须在建模型/前向之前

cfg     = yaml.safe_load(open(os.path.join(RUN_DIR, "config.yaml")))
enc_cfg = cfg["motion"]["encoder"]
assert int(cfg["data"]["max_horizon"]) == 8

encoder = WanLatentMotionEncoder(
    latent_channels=int(cfg["wan"]["latent_channels"]),   # 16
    latent_size=int(cfg["wan"]["latent_size"]),           # 32
    hidden_dim=int(enc_cfg["hidden_dim"]),                # 768
    motion_dim=int(cfg["motion"]["dim"]),                 # 768
    num_tokens=int(cfg["motion"]["num_tokens"]),          # 1
    seq_len=int(cfg["data"]["max_horizon"]) + 1,          # 9
    depth=int(enc_cfg["depth"]), heads=int(enc_cfg["heads"]),
    dim_head=int(enc_cfg["dim_head"]), mlp_ratio=int(enc_cfg["mlp_ratio"]),
    dropout=0.0,                                          # 推理关；eval() 下与 0.05 等价，显式写出防漂移
    # latents_mean / latents_std 保持默认 None → 构造成 NaN 占位。
    # ⛔ 不要在这里手填，也不要事后调 load_wan_latent_stats()——真值必须由下面的 strict 加载带入。
).to(DEVICE)

ckpt = torch.load(os.path.join(RUN_DIR, CKPT), map_location=DEVICE, weights_only=False)
assert ckpt["arch"] == ARCH_TAG, f'arch={ckpt["arch"]!r} ≠ {ARCH_TAG!r}'

state = {}
for k, v in ckpt["encoder"].items():          # 🔒 EMA 主键 encoder，不是 encoder_live
    while k.startswith("module.") or k.startswith("_orig_mod."):
        k = k.split(".", 1)[1]
    state[k] = v
encoder.load_state_dict(state, strict=True)   # 🔒 整份 strict，含 affine buffer 与 RoPE cache
assert torch.isfinite(encoder.latents_std).all(), "affine buffer 仍是 NaN 占位"
encoder.eval()

use_amp = cfg["training"]["precision"] == "bf16" and DEVICE == "cuda"   # True

# latent_block: (9,16,32,32) fp32，**原始未归一化** Wan latent，组优先，contiguous
@torch.no_grad()
def motion_token(latent_block):
    assert latent_block.shape == (9, 16, 32, 32) and latent_block.dtype == torch.float32
    x = latent_block.contiguous().unsqueeze(0).to(DEVICE)      # 🔒 B=1 是硬约束，见第 8 节
    torch.clear_autocast_cache()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        m = encoder(x)                                         # (1, 1, 768)
    return m.reshape(-1).float().cpu().numpy()                 # (768,)
```

### 7.2 latent 从哪来（Wan VAE 段）

```python
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                                       subfolder="vae", torch_dtype=torch.float32)
vae.eval().to(DEVICE)
for p in vae.parameters(): p.requires_grad_(False)
assert not vae.use_tiling and not vae.use_slicing        # 🔒 开了会改空间拼接，破坏对拍前提

@torch.no_grad()
def encode_chunk(frames_window):        # (33,256,256,3) uint8 → (9,16,32,32) fp32
    import numpy as np
    frames_window = np.ascontiguousarray(frames_window)   # 🛡 stride 不同 cudnn 会换 kernel（差 2e-6）
    with torch.autocast(device_type="cuda", enabled=False):        # 🛡 VAE 恒 fp32
        f = torch.from_numpy(frames_window).permute(0,3,1,2).float().to(DEVICE)
        x = (f / 127.5 - 1.0).permute(1,0,2,3).unsqueeze(0)         # (1,3,33,256,256) ∈ [-1,1]
        z = vae.encode(x).latent_dist.mode()                        # 🔒 mode()，不是 sample()
    assert z.shape == (1,16,9,32,32)
    return z.permute(0,2,1,3,4).contiguous()[0].float()             # 通道优先 → 组优先
```

### 7.3 decoder（如需）

```python
from motion_jepa.models import WanLatentDiT
d = cfg["dit"]
decoder = WanLatentDiT(
    hidden_dim=int(d["hidden_dim"]), motion_dim=int(cfg["motion"]["dim"]),
    depth=int(d["depth"]), heads=int(d["heads"]), dim_head=int(d["dim_head"]),
    mlp_ratio=int(d["mlp_ratio"]), num_motion_tokens=int(cfg["motion"]["num_tokens"]),
    window_size=int(d["window_size"]), dropout=0.0,
    latent_channels=int(cfg["wan"]["latent_channels"]),
    latent_size=int(cfg["wan"]["latent_size"]), patchify=int(cfg["wan"]["patchify"]),
).to(DEVICE)
decoder.load_state_dict({k: v for k, v in ckpt["wan_decoder"].items()}, strict=True)  # EMA 主键
decoder.eval()
# forward(current_latent (B,16,32,32) 原始域, motion_tokens (B·K,1,768), timestep (B·K,) k∈1..K)
#   → (B·K,16,32,32)，**归一化域**；要回到原始 latent 域需自行反 affine。
```

decoder 的语义是「给定当前 latent 与 motion token，预测第 k 步未来 latent」，是训练目标的一部分。
本项目的下游只用 encoder；decoder 一并上传是为了让这份 run 自洽可复现。

---

## 8. 数值复现合同（要逐位一致就必须全部满足）

**版本三件套**（下游 `wan_motion_infer.py::PINNED` 硬断言）：

```
torch      2.9.0+cu128
cudnn      91002
diffusers  0.39.0
```

**pin_numerics() 的 12 个开关**（全部显式钉住，读回值必须逐项相等）：

| 开关 | 值 |
|---|---|
| `torch.backends.cuda.matmul.allow_tf32` | `False` 🔒 |
| `torch.backends.cudnn.allow_tf32` | `False` 🔒（VAE 是 conv3d 网络，实质项） |
| `torch.backends.cudnn.benchmark` | `False` 🔒（True 会换 conv 算法） |
| `torch.backends.cudnn.deterministic` | `False` |
| `torch.are_deterministic_algorithms_enabled()` | `False` |
| `torch.get_float32_matmul_precision()` | `"highest"` |
| `matmul.allow_bf16_reduced_precision_reduction` | `True`（管 encoder 段 bf16 GEMM 累加） |
| `matmul.allow_fp16_reduced_precision_reduction` | `True` |
| `flash_sdp_enabled()` | `True` |
| `mem_efficient_sdp_enabled()` | `True` |
| `math_sdp_enabled()` | `True` |
| `cudnn_sdp_enabled()` | `True` |

SDPA 四后端**保持默认全开、不强制单一后端**——强制某一后端会与实测时的启发式分叉。

**禁止的环境变量**（会绕过上面的代码开关）：`NVIDIA_TF32_OVERRIDE`、
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`、`CUBLAS_WORKSPACE_CONFIG`。任一非空即视为合同不成立。

**Wan VAE 必须是同一份权重**：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，
snapshot **`0fad780a534b6463e45facd96134c9f345acfa5b`**，
state_dict 指纹（sorted keys + tensor bytes 的 sha256）
**`9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0`**。
`vae.use_tiling` / `use_slicing` 必须关闭，encode 用 `.mode()` 不是 `.sample()`。

**B = 1 是硬约束**：实测 batch 1 vs 8 会改 motion token 的最后一位（cuBLAS 按 GEMM 形状换 kernel，
首处差异出现在 `input_proj`）。批量抽取必须逐窗 B=1 前向。

**输入必须 contiguous**：张量 stride 不同会让 cudnn 换 kernel，实测差 2e-6。

**已验证范围**：同机双卡（RTX 6000 Ada）、64 个窗口的 latent 与 motion token **逐位相同**（max|Δ| = 0）。
**跨 GPU 架构未做逐位验证**，见第 9 节。

---

## 9. 已知限制

1. **无 optimizer 状态，不能完整 resume。** `checkpoint.save_optimizer: false`，ckpt 里没有
   Adam 的一阶/二阶矩与 step 计数。可以拿 `encoder_live` / `wan_decoder_live` 起一个新 optimizer
   继续训，但那不是「从 epoch 72 无缝续跑」——前若干步的更新方向会与原轨迹分叉。
2. **跨 GPU 架构未做逐位验证。** 逐位一致只在 RTX 6000 Ada 同机双卡上实测过。已知的跨架构现象：
   encoder 训练时吃的是 **A40** 抽的 latent，下游喂的是 **Ada** 抽的 latent，两者差约 **1.24e-5**
   （集中在 VAE `conv_out`、沿 group 累积），传到 motion token 只落在最后一位（cos 0.999995），
   经 encoder 入口 affine 归一化后可忽略——但这是「可忽略」，不是「逐位相同」。换到 H100 / L40S 等
   其他架构上，差异量级需要重测。
3. **训练数据是私有的 4 任务 × 400 episode 录制版，不是 Hub 上的 `Yinpei/robomme_data_h5`（16 任务 × 100 episode）。**
   没有私有录制版就无法复现训练；也不要假设本模型在公开版的另外 12 个任务上有对应能力。
4. **没有验证集指标。** `training.validate: false`，ckpt 里 `val_loss` / `val_sigreg_loss` /
   `val_latent_loss` 全是 `None`。第 6.3 节的数字全部来自训练集。9,302 chunk 的 val 集已划出但未使用，
   任何泛化结论都需要另做离线评估。
5. **没有 best / latest 软链，也没有早停。** 36 个存档里没有「最优」的概念，epoch 72 是**最后一个**、
   不是「验证集上最好的一个」。
6. **只覆盖 4 个 RoboMME 任务的 front_rgb 视角、256×256、33 帧窗口。** 换分辨率、换视角、
   换窗口长度都超出训练分布；`seq_len` 与 `max_horizon` 有显式断言，喂错长度会直接 raise 而非静默降级。
7. **checkpoint 是 pickle**，需 `weights_only=False` 才能读（顶层含 omegaconf `DictConfig`）。
8. **推理开销不低**：单个 33 帧窗口过 Wan VAE + encoder，A40 实测 ≈ 1.57 s/窗，
   RTX 6000 Ada ≈ 0.85 s/窗。在线闭环使用时必须为这笔延迟做预算。
9. **模型代码不在本 repo**，在私有 GitHub `hongzefu/MotionJEPA`。没有该 repo 访问权就无法实例化模型。

---

## 10. 许可与引用

**许可**：`internal-research-only`。本权重训练于**未公开**的机器人录制数据，仅供项目内部研究使用；
不授权再分发权重、不授权商业使用。上游依赖各自的许可独立适用——MotionJEPA 训练代码为
Apache-2.0（同 `robomme_policy_learning_MotionJEPA` 仓库），Wan2.1-T2V-1.3B VAE 遵循其自身发布许可。
本 repo 转为 public 之前，必须先取得数据录制方的授权，并清洗本卡与
`SHA256SUMS.src-vs-copy.txt` 中的内部路径。

**引用**：

```bibtex
@misc{motionjepa_wan_v8_filter10_72ep_a,
  title  = {MotionJEPA: Wan-latent motion encoder (run wan-v8-filter10-72ep-a, epoch 72)},
  author = {Fu, Hongze},
  year   = {2026},
  note   = {Trained from github.com/hongzefu/MotionJEPA at commit 7388a42 on dataset-4env-v8.
            Checkpoint sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a.},
  howpublished = {\url{https://huggingface.co/HongzeFu/MotionJEPA}}
}
```

论文引用信息：**待补**（尚无对应论文/预印本）。

---

## 11. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-09-04 | 首次上传：`wan-v8-filter10-72ep-a` epoch 72（encoder + wan_decoder，EMA + live），tag `wan-v8-filter10-72ep-a-e72` |
