# 我们是怎么在 pi0.5 上加 motion memory 的：memory 怎么拼、motion token 怎么拿、窗口怎么补帧

> 对象：`context 512 + motion 1×1024（concat）` 模型（AWS 8×A100 训练，官方 800 局三 seed 37.33 ± 0.59%）。只写我们做了什么。

## 0. 资源一览（2026-11-03 11:26 美东，不带任何 token 实测匿名访问）

| 资源 | 地址 | 匿名可达 |
|---|---|---|
| encoder checkpoint | HF bucket `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1`，文件 `stage2-full144-1025-a/checkpoint_epoch_105.pt`（314,265,763 B，sha256 `27daa832…6035`）与 `config.yaml`；下载 `https://huggingface.co/buckets/HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1/resolve/<路径>` | ✅ `resolve` 匿名 200（bucket 的网页 `tree` 视图匿名 404，不影响下载） |
| 预处理好的训练数据集（含 motion 表、帧表、Wan latent） | HF bucket `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`，先取 `delivery_manifest.json` 看全部文件与 sha256；motion 表在 `motion-stage2-1024/` | ✅ `resolve` 匿名 200 |
| 原始 H5（16 任务 × 100 集） | HF dataset `Yinpei/robomme_data_h5`（预处理 pkl 版 `Yinpei/robomme_preprocessed_data`） | ✅ 公开 |
| 训好的 VLA checkpoint（12 步） | HF bucket `HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1` | ✅ `resolve` 匿名 200 |
| VLA 仓库（本文全部路径相对其仓库根） | `github.com/hongzefu/robomme_policy_learning_MotionJEPA`，训练分支 `v4-256concat1` | ✅ 公开 |
| encoder 仓库 | `github.com/hongzefu/MotionJEPA`，commit `b5d23a5`（encoder 定义与 `encode_windows` 导出接口） | ❌ **私有**，匿名 404；`scripts/dataset/wan_stage2/pyproject.toml` 以 git 依赖钉它，无权限的人装不了这个环境，需要先开权限或公开 |
| Wan VAE | HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 子目录 `vae` | ✅ 公开 |

## 1. memory 怎么拼接

pi0.5 原本的输入是 `[当前帧图像 | 文本]`，动作专家读它的 K/V。我们在最前面多放一段 **记忆区**，由两路 token 组成：

| 路 | 一个 token 代表什么 | 怎么得到 | 位置数 |
|---|---|---|---|
| 帧路 | 某一历史帧「长什么样」 | 在 `[0, 当前步]` 上等距取 32 帧，每帧过 SigLIP 后池化到 4×4 = 16 个 token | 32 × 16 = 512 |
| 运动路 | 某段连续 33 帧「发生了什么运动」 | 窗口过 Wan2.1 VAE 再过 MotionJEPA 阶段二 encoder，输出 (4,256) 展平成一个 1024 维向量 | 最多 96 个窗口 × 1 |

拼接分四步：

1. **投影**。每个 token 拼上它所在帧（或窗口起点帧）的时间码，过一层 Linear 投到主干宽度 2048。帧路：`[2048 维视觉特征 | 时间码 768]` → 2048；运动路：`[1024 维 motion | 时间码 256→768]` → 2048。两个 encoder 冻结，只训这两个 Linear 和 pi0.5 本体。
2. **交错**。两路 token 合在一起按时刻稳定排序，同一时刻帧 token 在 motion token 前。结果是时间上相邻的「样子」和「运动」靠在一起，而不是 512 个帧 token 后跟 96 个 motion token。排序用的置换 `mem_order` 在数据侧算好、随样本进模型。
3. **固定形状 + mask**。帧路填满 512 位、运动路填满 96 位，不够的填零向量、mask 置 False；排序时 padding 的时刻记最大值，所以全部落在记忆区尾部。位置编号按有效 token 累加，padding 不占号。运动路预算 96 按全部数据集的最长 episode 定，合法窗口数超过 96 直接报错，不抽稀。
4. **放进前缀**。序列变成 `[记忆区 608 | 当前帧图像 512 | 文本 64]`，再接 20 个动作 token。attention：记忆 token 之间双向互看、看不到当前图像和文本；当前图像 token 看不到记忆；文本 token 和动作 token 看到全部记忆。记忆只在前缀里算一次进 KV cache，去噪多步只重算动作段。**这一步是 pi0.5（PaliGemma 前缀 + 动作专家）的实现；换到 Qwen 时放哪里、谁能看谁要单独讨论，不在本文范围。**

**记忆区的位置编码**（pi0.5 一共有四处与「位置／时间」有关的编码，记忆区碰到前三处）：

| 处 | 是什么 | 对记忆区怎么作用 |
|---|---|---|
| ① SigLIP 内部 patch 位置编码 | ViT 对 224 输入的 16×16 个 patch 加一张**可学习**位置表（`siglip.py::get_posemb`，`posemb="learn"`） | 历史帧和当前帧一样先过 SigLIP，所以每个 patch 都带了它；之后 4×4 均值池化把 16 个 patch 揉成 1 个，细粒度空间位置被抹平 |
| ② 记忆 token 自带的 3D 时间码 | `PosEmb3D`：768 维固定 sin/cos，无参数。按 6 等份分：2 份时间（帧号 `t` 的 sin/cos，base 10000，共 256 维）+ 4 份空间（池化后格点的行、列各 sin/cos，base 1000，共 512 维；4×4 池化的格点坐标取 2/6/10/14） | 帧 token 用完整 768 维：`silu(Linear(768→768))` 后与 2048 维视觉特征 concat 再投影。motion token 只取**前 256 维时间部分**（窗口起点帧的 `t`），`silu(Linear(256→768))` 后与 1024 维 motion concat 再投影。`t` 是含 demo 段的全域帧号，所以记忆 token 带的是**绝对时间**；训练时这张表按 `t` 预存（`pos_emb_4x4.f32.bin`），在线用同一个 `PosEmb3D` 现算 |
| ③ Gemma 每层的 RoPE | PaliGemma 主干 18 层、动作专家 18 层，每层 attention 对 q、k 施加 RoPE（`gemma.py::_apply_rope`，波长 10000），输入端没有绝对位置向量 | 位置号 `positions = cumsum(input_mask) − 1`：记忆区有效 token 占 0 … 607 中的前若干号，padding 不占号；当前图像、文本接着排；动作 token 位置 = prefix 有效数 + 自身序号。记忆区按 `mem_order` 时间排序后，RoPE 位置号随时间单调，但它只是**稠密的排名**（第几个 token），不是帧号；帧间隔多大只能从 ② 的时间码里读 |
| ④ flow-matching 时间步 | `posemb_sincos(timestep)` → MLP → adaRMS 调制动作专家 | 与序列位置无关，记忆区不参与 |

一句话：记忆 token「是什么时候的」靠 ② 的绝对帧号时间码；「在序列里排第几」靠 ③ 的 RoPE 稠密位置号；两者都不训练位置表本身，只训 ② 后面的两个 Linear。

```
时间轴 →   f0 ─ m0 ─ f1 ─ f2 ─ m32 ─ f3 ─ … ─ f31 ─ [帧 padding] ─ [motion padding] | 当前图像 512 | 文本 64 | 动作 20
           16位  1位  16位 16位  1位                       mask=False
```

**代码范围**（`robomme_policy_learning_MotionJEPA`）：

| 步 | 位置 |
|---|---|
| 投影 | `src/mme_vla_suite/models/representation/percep_mem.py::PerceptualMemory`（`motion_pos_proj`、`motion_encoder_static`）；帧路 `mem_encoder.py::FeatureEncoder` |
| 交错排序 | `src/mme_vla_suite/shared/sampling.py::memory_order`；帧采样 `even_sampling_indices` |
| 填充与数据侧 8 个键 | `src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__getitem__ / _pad / _pad_motion`（训练）；`src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory._prepare_motion`（在线） |
| 重排、放进前缀、attention mask | `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.embed_memory / embed_prefix / make_attn_mask` |
| 配置 | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml`（分支 `v4-256concat1`；`integration_type: context`、`budget: 512`、`motion.budget: 96`、`motion.readout: stage2_pool4_concat_v1`） |

## 2. motion token 怎么拿

同一个 encoder、同一个入口函数。先跑通一个窗口，剩下的只是「训练时提前算好存表，推理时现算」。

### 2.1 资产

| 东西 | 从哪拿 |
|---|---|
| encoder `stage2-full144-1025-a/checkpoint_epoch_105.pt`（EMA，sha256 `27daa832…6035`，314 MB）+ `config.yaml` | 公开 bucket `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1` |
| Wan VAE | HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 子目录 `vae`，diffusers `AutoencoderKLWan`，fp32 |
| torch 环境 | `scripts/dataset/wan_stage2/pyproject.toml`（torch 2.9.0、diffusers 0.39.0、`motion-jepa` 以 git 依赖钉 `b5d23a5`）：`UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2 uv sync --project scripts/dataset/wan_stage2` |
| VAE 适配函数 | `scripts/dataset/wan/wan_motion_infer.py::load_vae / encode_chunk` |

### 2.2 跑通一个窗口

```python
import torch
from motion_jepa.export.stage2_memory import load_ema_encoder, encode_windows   # MotionJEPA b5d23a5
import wan_motion_infer as W                                                   # scripts/dataset/wan/

W.check_env(); W.pin_numerics()                # fp32、关 TF32
dev = torch.device("cuda:0")
vae, _ = W.load_vae("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", dev)                  # fp32、eval、关 tiling
encoder, einfo = load_ema_encoder("<…>/checkpoint_epoch_105.pt", device="cuda:0",
                                  expected_sha256="27daa832…6035")
frames = ...                                   # (33, 256, 256, 3) uint8 原图，按第 3 节取窗
latent = W.encode_chunk(vae, frames, dev)      # (9, 16, 32, 32) f32
token  = encode_windows(encoder, latent.unsqueeze(0))[0]   # (1024,) f32 ← 一个 motion token
```

一窗约 1 s，几乎全在 VAE。`einfo` 里的 `encoder_state_sha256` 与 `numerics` 存下来，训练与推理两侧比对。

### 2.3 训练：提前建库

太慢，不能在线算。对训练集所有窗口跑一遍 2.2，按「清单顺序逐 episode、每 episode 先 demo 后 exec、段内起点升序」存成一张 `(rows, 1024)` f32 表，训练时按 (episode, 当前步) 查可见窗口（第 3 节公式）。脚本：`scripts/dataset/wan/extract_wan.py`（帧→latent）→ MotionJEPA `scripts/stage2_memory_export.py`（latent→token）→ `scripts/dataset/pack_motion_store.py`（打包、校验）；表格式见 `src/mme_vla_suite/datastore/motion_store.py`。1600 集训练库的表 23,253 行、约 95 MB，建好的在公开 bucket `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1` 的 `motion-stage2-1024/`。

### 2.4 推理：在线现算

帧是新的，只能现算。起一个 torch 子进程常驻一张卡，每凑齐一个新窗口就送 33 帧过去、收回 (1024,)；一个起点只编一次，缓存到 episode 结束。实现：子进程 `scripts/dataset/wan_stage2/stage2_sidecar.py`，socket 协议与客户端 `src/mme_vla_suite/policies/{motion_protocol,motion_client}.py`，窗口状态机 `src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory`（首批送整段 demo 一次编完，之后每 16 帧一批）。子进程启动时回传 encoder / VAE / 数值环境身份，与离线表记录的逐键比对，不等即拒绝启动。

## 3. 窗口怎么取、怎么补帧

窗口 = 从起点往后连续 33 帧。起点钉在**段内绝对网格** 0, 32, 64, …，demo 段与 exec 段各自从 0 起算、窗口不跨段；当前时刻只决定哪些起点「已经可见」。每过 32 帧才多一个窗口。

```
demo 段（es 帧）           exec 段
|0 ─32─ 64 ─ … ─ (es−1)|es ─ es+32 ─ es+64 ─ … ─ s(当前)
 ▲   ▲   ▲ 尾窗可补帧     ▲      ▲  必须整 33 帧真实才可见
```

补帧规则，训练建库与在线现算一字不差：

- **demo 段**：尾部窗口不足 33 帧时，用 **demo 最后一帧（es−1）重复补到 33 帧**；要求至少 17 个真实帧，不足 17 的起点不算窗口；**绝不借用 exec 的帧**。
- **exec 段**：**永不补帧**，必须凑齐 33 个真实帧才算可见；没凑齐就等后面的帧。

```python
real = min(33, es - f) if f < es else 33            # es = exec 起点帧号，f = 窗口起点
assert real >= (17 if f < es else 33)
frame_ids = [f + j for j in range(real)] + [es - 1] * (33 - real)
```

可见窗口：demo 段起点 `s` 合法条件 `s + 16 ≤ es − 1`（与当前步无关，首批一次编完）；exec 段起点 `u` 合法条件 `u + 32 ≤ 当前步 − es`。代码：`src/mme_vla_suite/datastore/motion_store.py::visible_motion_rows / LAYOUT_SPECS`（契约）、`scripts/dataset/wan/extract_wan.py`（离线）、`src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory._encode_window`（在线）。
