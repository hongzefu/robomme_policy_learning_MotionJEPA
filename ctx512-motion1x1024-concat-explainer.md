# 把 motion memory 接进 Qwen3-VL：memory 怎么拼、motion token 怎么拿、窗口怎么补帧

> 续接《Wiring Qwen3-VL into RoboMME》的第三步。前两步已经让 Qwen3-VL 吃到 FrameSamp 的 32 帧历史（context 512）；本文讲在它旁边再并联一路 **motion memory**（最多 96 个窗口、每窗一个 1024 维 token），对照我们在 pi0.5 上的实现 `context 512 + motion 1×1024（concat）`（AWS 8×A100，官方 800 局三 seed 37.33 ± 0.59%）。每一项先写「我们（pi0.5 侧）怎么做」，再写「Qwen 侧怎么做」。

## 0. 三个仓库与要做什么

| | robomme_policy_learning_MotionJEPA | MotionJEPA | StarVLA（QwenPI_v3_Mem） |
|---|---|---|---|
| 是什么 | 我们在 pi0.5 上加记忆的 policy 仓库（JAX / openpi） | motion encoder 的训练与导出 | 目标框架，Qwen3-VL + DiT 动作头 |
| 在哪 | `github.com/hongzefu/robomme_policy_learning_MotionJEPA`，训练分支 `v4-256concat1`，下文路径都相对它的仓库根 | `github.com/hongzefu/MotionJEPA`，encoder 与导出接口锁 commit `b5d23a5`，离线导出脚本取 commit `4a16ab4` | 第二步已接好 FrameSamp+Context 的那份代码 |
| 拿来做什么 | **参考**：拼接方式、窗口规则、建库脚本、在线编码子进程全部沿用 | **直接用**：encoder 权重与 `encode_windows` 入口 | **修改**：序列组装处多插一路 motion token |

要做的事三件：① 用第 3 节的脚本给训练集建一张 motion 表（或直接下我们建好的）；② 按第 2 节把 motion token 投影后按时间插进 Qwen 的历史段；③ 评估时按第 3.3 节起在线编码子进程。窗口与补帧规则（第 4 节）两侧必须一字不差。

## 1. motion token 是什么

| | 帧路 token（第二步已有） | motion token（本文新增） |
|---|---|---|
| 代表什么 | 某一历史帧「长什么样」 | 某段连续 33 帧「发生了什么运动」 |
| 怎么得到 | 历史帧过 ViT，池化到 16 token | 33 帧 → Wan2.1 VAE（fp32）→ MotionJEPA 阶段二 encoder（EMA）→ (4,256) 展平为 **一个 1024 维 f32 向量** |
| 每样本多少 | 32 帧 × 16 = 512 | 最多 96 个窗口 × 1 |
| 窗口起点 | `even_sampling_indices(s, 32)` 等距取帧 | 段内绝对网格 0, 32, 64, …（第 4 节） |
| encoder 是否训练 | 冻结 | 冻结；只训一个投影 |

encoder 内部（`MotionJEPA/src/motion_jepa/models/wan_latent_encoder.py::Stage2WanLatentMotionEncoder`）：9 组 latent 切 576 个 patch，加 4 个来源时刻槽 [0,3,5,8] 的池化读出 token，8 层宽 256 块因果 Transformer，取 4 个读出位 → (4,256)。「concat 1×1024」就是把这 4×256 展平成一个 token；对照的 split 读法是拆成 4 个 256 维 token 占 384 位（三 seed 38.42 ± 1.25，与 concat 同量级）。Qwen 侧先做 concat，一窗一 token 最省位置。

## 2. memory 怎么拼接

拆成四件：投影、交错、放置、谁能看谁。

### 2.1 投影：1024 维 → 主干宽度

**我们**（`src/mme_vla_suite/models/representation/percep_mem.py::PerceptualMemory`）：每个 motion token 拼上它起点帧的时间码，过一层 Linear 到主干宽 2048。帧 token 同构（2048 + 768 → 2048）。

```python
self.motion_pos_proj       = nnx.Linear(256, 768)          # 时间码：起点帧位置编码的前 256 维
self.motion_encoder_static = nnx.Linear(1024 + 768, 2048)  # 投到 PaliGemma 宽度
pos_h         = silu(self.motion_pos_proj(motion_pos))                       # (b, 96, 768)
motion_tokens = self.motion_encoder_static(concat([motion_emb, pos_h], -1))  # (b, 96, 2048)
```

`motion.enabled=false` 时两层根本不创建，关闭态与原模型逐位相同。padding 行（零向量）照常过投影，靠 mask 屏蔽。

**Qwen 侧**：一层 `Linear(1024 → Qwen 隐宽)` + 可选 LayerNorm，输出直接作为 `inputs_embeds` 里的一段，和 ViT 出来的图像 embedding 同等地位。时间码按第二步的做法走文本 `t=s`（mrope 只编码顺序，数值进不去），所以 motion 侧不需要 `motion_pos_proj`。新增参数只有这一层。

### 2.2 交错：按时刻排，不是先 512 后 96

**我们**（`src/mme_vla_suite/shared/sampling.py::memory_order`）：

```python
frame_keys  = np.repeat(frame_times * 2 + 0, 16)   # 每帧 16 位连续
motion_keys = motion_times * 2 + 1                 # 同一时刻帧在 motion 前
order = np.argsort(np.concatenate([frame_keys, motion_keys]), kind="stable")
# padding 的时刻记 int32 最大值 → 全排到尾部；最后校验是 0..607 的置换
```

训练 dataloader 与在线 policy import 同一个函数，产出 `mem_order` 随样本交给模型，模型里 `take_along_axis` 重排。

```
时间轴 →   f0 ─ m0 ─ f1 ─ f2 ─ m32 ─ f3 ─ … ─ f31 ─ [帧 padding] ─ [motion padding]
           16位  1位  16位 16位  1位                      mask=False
```

**Qwen 侧**：组装 `content` 列表时按同一规则排：`[t=0, F₁] [t=0, M₀] [t=16, F₂] … [t=32, M₁] …`，motion 单元 = `[t=起点帧号 文本, 1 个 motion embedding 占位]`，与帧单元 `[t=s 文本, 16 个图像 token]` 并列、同一时刻帧在前。不需要 `mem_order`，顺序本身就是位置。

### 2.3 放置：全部在历史段最前

**我们**（`src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.embed_prefix`）：

```
prefix = [ 记忆区 608 位（交错后） | 当前帧 SigLIP 512 | 文本 64 ]  →  PaliGemma  →  动作专家 20 步
positions = cumsum(input_mask) − 1     # padding 不占位置号
```

记忆只在 prefix 里算一次进 KV cache，去噪多步只重算动作段。

**Qwen 侧**：一条 user message，内容按时间：`[demo 段的帧单元与 motion 单元] → [exec 段的帧单元与 motion 单元] → front → wrist → 指令 + state bins`。只改 `build_qwenvl_inputs` 的组装：在历史段插入 motion 占位 token，前向时把占位位置的 embedding 换成 2.1 的投影输出（与图像 embedding 的 scatter 同一机制）。96 个窗口位实际只有 k 个真窗口，Qwen 侧不必填满 96，**只放真窗口**，变长由 batch padding 处理；但训练时 k > 96 仍要报错（第 2.5 节零截断）。

### 2.4 谁能看谁

**我们**（`history_pi0.py::make_attn_mask`）：prefix 双向加两道切口。`ar_mask` 让记忆只看记忆；`na_mask` 让当前图像看不到记忆；文本与动作看全部。

```python
attn_mask  = cumsum(ar)[:, None, :] <= cumsum(ar)[:, :, None]       # 记忆块看不到图像文本
valid_mask = input_mask[:, None, :] * input_mask[:, :, None]         # padding 既不被看也不发问
mask_not_attend = (na[:, None, :] | na[:, :, None]) & (cumsum(na) <= 0)[:, None, :]   # 图像不看记忆
```

**Qwen 侧**：因果 LM，顺序即可见性。历史段在前：motion token 只看更早的历史（第一道切口自然满足）；当前图像能看到历史（第二道不满足，但「后图看前图」在 Qwen 预训练分布内，与第二步帧路的处理一致）。DiT 经 cross-attention 读全部 hidden。不另写 4D mask。

### 2.5 数据侧产出什么

**我们**（`src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__getitem__`）每样本 8 个键：

| 键 | 形状 | 说明 |
|---|---|---|
| `static_image_emb` / `static_pos_emb` / `static_state_emb` / `static_mask` | (512, 2048) bf16 / (512, 768) / (512, 8) / (512,) bool | 帧路，不足 32 帧右补零 |
| `motion_emb` / `motion_pos` / `motion_mask` | (96, 1024) f32 / (96, 256) / (96,) bool | 查表取可见窗口 k ≤ 96，其余零；**k > 96 直接 raise，不抽稀** |
| `mem_order` | (608,) int32 | 2.2 的置换 |

**Qwen 侧**：dataset 多输出 `motion_tokens (k, 1024)` 与 `motion_step_ids (k,)`，k 由第 4 节的可见窗口公式算；组装时按 2.2 插入。

## 3. motion token 的两种获取方式

同一个 encoder、同一个入口函数；训练用离线预存表，推理用在线现算。两侧一致性只靠三条：同一权重、同一窗口定义、同一数值口径（fp32、关 TF32、batch 1）。

### 3.0 资产与环境

| 东西 | 从哪拿 |
|---|---|
| encoder `stage2-full144-1025-a/checkpoint_epoch_105.pt`（EMA，sha256 `27daa832…6035`，314 MB）+ `config.yaml` | 公开 bucket `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1`；VLA 交付包 `HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1` 的 `encoder_dependencies/stage2-full144-1025-a/` 也有 |
| Wan VAE | HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 子目录 `vae`，diffusers `AutoencoderKLWan`，fp32 |
| 我们建好的训练库 | 公开 bucket `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`：`motion-stage2-1024/` 是最终表（23,253 行 × 4096 B ≈ 95 MB），`wan-latents/` 是中间 latent（26.8 GB，可复用免跑 VAE），`framesamp-4x4/` 是帧表 |
| torch 环境 | `scripts/dataset/wan_stage2/pyproject.toml`（torch 2.9.0、diffusers 0.39.0、`motion-jepa` 以 git 依赖钉 `b5d23a5`）：`UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2 uv sync --project scripts/dataset/wan_stage2` |
| VAE 适配函数 | `scripts/dataset/wan/wan_motion_infer.py::load_vae / encode_chunk` |

### 3.1 先跑通一个窗口

离线与在线都是这十行，Qwen 侧第一件事就是把它跑通：

```python
import torch
from motion_jepa.export.stage2_memory import load_ema_encoder, encode_windows   # MotionJEPA b5d23a5
import wan_motion_infer as W                                                   # scripts/dataset/wan/

W.check_env(); W.pin_numerics()
dev = torch.device("cuda:0")
vae, _     = W.load_vae("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", dev)             # fp32、eval、关 tiling
encoder, einfo = load_ema_encoder("<…>/checkpoint_epoch_105.pt", device="cuda:0",
                                  expected_sha256="27daa832…6035")
frames = ...                                   # (33, 256, 256, 3) uint8，按第 4 节取窗
latent = W.encode_chunk(vae, frames, dev)      # (9, 16, 32, 32) f32
token  = encode_windows(encoder, latent.unsqueeze(0))[0]   # (1024,) f32 ← 一个 motion token
```

`einfo["encoder_state_sha256"]`（65 个叶）与 `einfo["numerics"]` 随表存下，在线启动时比对。

### 3.2 方式一：离线建库预存（训练用）

对整个训练集提前算好所有窗口存成一张表，dataloader 按 (episode, 当前步) 查可见窗口。三步，各自可断点续跑：

```bash
LIB=<数据集根>   # 需有 meta/episode_manifest.json（episode 清单，含 exec_start_idx）与 meta/input_manifest.json（H5 sha256）
R=<robomme_policy_learning_MotionJEPA 根>

# 1) 帧 → Wan latent：读原始 H5，每 GPU 一个 worker 动态领段（已有 wan-latents/ 可跳过）
uv run --no-sync python $R/scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1 --raw-dir <H5 目录>
#    → $LIB/wan-latents/<段>.bin（每窗 589,824 B，stride 16 网格）+ .sha256 + .metadata.json（记 real_frames / pad_frames）

# 2) latent → 1024 维 token：MotionJEPA 4a16ab4 的 scripts/stage2_memory_export.py；--grid-subsample 2 取偶数位 = stride 32
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 $R/v1-store/venvs/wan-stage2/bin/python <MotionJEPA>/scripts/stage2_memory_export.py \
  --ckpt <…>/checkpoint_epoch_105.pt --expected-sha256 27daa832…6035 \
  --latents $LIB/wan-latents --manifest $LIB/meta/episode_manifest.json --input-manifest $LIB/meta/input_manifest.json \
  --out $LIB/motion-tokens-stage2 --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts <段数>,<行数>
#    → $LIB/motion-tokens-stage2/<段>.f32.bin（每窗 4096 B）；末行 STAGE2_EXPORT=PASS

# 3) 打包成表并校验
uv run --no-sync python $R/scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 \
  --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens-stage2 --latents $LIB/wan-latents \
  --out $LIB/motion-stage2-1024 --encoder-run-dir <…>/stage2-full144-1025-a --raw-dir <H5 目录> \
  --expected-ckpt-sha256 27daa832…6035 --stage2-validation $LIB/evidence --stage2-source-pin $R/scripts/dataset/wan_stage2/SOURCE_PIN.json
uv run --no-sync python $R/scripts/dataset/pack_motion_store.py verify --store $LIB/motion-stage2-1024 --resume --check-padding
```

表的格式（`src/mme_vla_suite/datastore/motion_store.py`），Qwen 侧照着读就行：

```
motion-stage2-1024/
├── motion_token.f32.bin     (rows, 1, 1024) f32 裸字节；行序 = 清单顺序逐 episode，每 episode 先 demo 段后 exec 段，段内按起点 0,32,64,… 升序
├── meta/motion_index.json   每 episode 两段的行基址与起点数（查表唯一依据）
├── meta/store_meta.json     布局名 motion-stage2-1x1024-grid32-demopad17-v1 + encoder / VAE / 数值环境 provenance
└── meta/row_digests.blake2b.bin
```

读表：`MotionStore.rows(行号)`；当前步可见的行号由 `motion_store.py::visible_motion_rows(entry, t)` 算（第 4 节规则）。

### 3.3 方式二：在线现算（推理用）

评估时在 policy 旁边起一个 encoder 子进程，环境每送来新帧就把凑齐的窗口现场编码，不读表。

| 部件 | 文件 | 说明 |
|---|---|---|
| 子进程 | `scripts/dataset/wan_stage2/stage2_sidecar.py` | 在 3.0 环境里跑；加载 VAE 与 encoder，循环收请求。启动：`UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2 CUDA_VISIBLE_DEVICES=<卡> uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/stage2_sidecar.py --fd <socketpair 子端> --ckpt <…> --expected-sha256 <…> --expected-vae-sha256 <…>` |
| 协议 | `src/mme_vla_suite/policies/motion_protocol.py` | Unix socketpair；请求 = 起点帧号 + 33×256×256×3 uint8（6,488,064 B）；响应 = (1024,) f32 4096 B；一次一窗、同步、加锁串行。握手回传 encoder / VAE / 数值 provenance，客户端与离线表 `store_meta.json` 逐键比对，不等即拒绝启动 |
| 客户端 | `src/mme_vla_suite/policies/motion_client.py::MotionEncoderClient` | `Popen` 起子进程、握手、`__call__(frames) -> (1024,)` |
| 窗口状态机 | `src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory` | `add_buffer` 收帧（首批 = 整段 demo + exec 第 0 帧，之后每批 16 帧），缓存 256² 原图；`_encode_ready_windows` 用 while 把已凑齐的起点全部编掉，一个起点只编一次；`_prepare_motion(step)` 取当前可见窗口 |

**Qwen 侧**：这四件都不依赖 pi0.5，整体拿走。第二步里记忆在 client 侧、server 无状态，所以 `FrameSampMemory` 的运动路部分放进 client：reset 时 demo 帧进 buffer 并一次编完 demo 窗口，之后每 16 步 `add_buffer` 一批、编新凑齐的 exec 窗口，replan 时把 k 个 (1024,) 向量连同起点帧号一起发给 server。sidecar 单独占一张卡（或与 server 共卡但显存约 4 GB）。

训练不在线算是因为一窗 VAE 约 1 s；推理不读表是因为帧是新的。两侧对拍工具 `scripts/training/check_stage2_online.py`：对 64 个固定窗口在线现算 vs 离线表，判定 `ONLINE_OFFLINE_PARITY=PASS`。

## 4. 窗口怎么取、怎么补帧（两侧必须一字不差）

窗口 = 从起点往后连续 33 帧。起点钉在**段内绝对网格** 0, 32, 64, …，demo 段与 exec 段各自从 0 起算、窗口不跨段；当前时刻只决定哪些起点「已经可见」。每过 32 帧才多一个窗口，离线表小，在线每两个 action chunk 才编一次。

```
demo 段（es 帧）           exec 段
|0 ─32─ 64 ─ … ─ (es−1)|es ─ es+32 ─ es+64 ─ … ─ s(当前)
 ▲   ▲   ▲ 尾窗可补帧     ▲      ▲  必须整 33 帧真实才可见
```

补帧规则：

- **demo 段**：尾部窗口不足 33 帧时，用 **demo 最后一帧（es−1）重复补到 33 帧**；要求至少 17 个真实帧，不足 17 的起点不算窗口；**绝不借用 exec 的帧**。
- **exec 段**：**永不补帧**，必须凑齐 33 个真实帧才算可见；没凑齐就等后面的帧。

```python
# 在线侧 framesamp_memory.py::_encode_window；离线侧 extract_wan.py 同式
real = min(33, es - f) if f < es else 33            # es = exec 起点帧号，f = 窗口起点
assert real >= (17 if f < es else 33)
frame_ids = [f + j for j in range(real)] + [es - 1] * (33 - real)
```

可见窗口数（`motion_store.py::visible_motion_rows`）：demo 段起点 `s` 合法条件 `s + 16 ≤ es − 1`（与当前步无关，首批一次编完）；exec 段起点 `u` 合法条件 `u + 32 ≤ s − es`。预算 96 按 16 任务全集最长 episode 定标，零截断；超过直接报错。

## 5. 评估怎么跑

记忆在 client，server 无状态（沿用第二步）。reset：demo 帧进 buffer，帧路照旧，运动路把 demo 段所有窗口一次编完。每步：追加 256² front 帧与 state。每 16 步 replan：帧路取 ≤32 帧，运动路取当前可见的 k 个 motion 向量，按 2.2 交错组装发 server；server 前向 + 反归一化返回 20 步动作，client 执行 16 步。sidecar 跨 episode 常驻，reset 不重启。

## 6. 开工前要定的事

- motion token 的时间码走文本 `t=s`（与帧路一致，零参数）还是像我们一样投影进 embedding。建议先文本。
- 是否按我们那样填满 96 位。建议 Qwen 侧只放真窗口，省位置；训练仍保留 k > 96 报错。
- 当前图像看不到记忆这道切口不做（第二步已放弃）；是否补一个消融。
- 是否同时跑 split（4×256）对照。我们两种读法三 seed 相差 1.1 个百分点，与 seed 波动同量级，建议先只做 concat。

## 7. 已知坑

- VAE 输入必须是 256×256 原图，不是喂 ViT 的缩放图；喂 224 或 128 照样能算，结果与训练静默不一致。所以在线 buffer 要单独存 256² 原图。
- 33 = 4k+1 是 VAE 时间压缩的对齐条件，diffusers 对 T≠4k+1 会静默截断；`encode_chunk` 有形状断言，别绕过。
- stride、窗口长度、补帧规则三者改任何一个，都要同时改离线导出与在线编码器；不一致不报错。
- encoder 必须 eval、fp32、关 TF32、batch 1，`encode_windows` 会检查；跨卡型（A100 建表、A40/A6000 推理）数值不逐位，只要求身份一致。
- 首批 `add_buffer` 必须恰好是整段 demo + exec 第 0 帧并传 `exec_start_idx`，否则 demo 窗口无法成形。

## 附：我们代码的位置索引

| 内容 | 位置（robomme_policy_learning_MotionJEPA） |
|---|---|
| 记忆拼接、attention mask | `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.embed_prefix / embed_memory / make_attn_mask` |
| 两路投影 | `src/mme_vla_suite/models/representation/percep_mem.py::PerceptualMemory` |
| 交错排序 | `src/mme_vla_suite/shared/sampling.py::memory_order` |
| 训练数据侧 8 键与 padding | `src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__getitem__ / _pad / _pad_motion` |
| 窗口网格、补帧契约、表读取 | `src/mme_vla_suite/datastore/motion_store.py::visible_motion_rows / LAYOUT_SPECS / MotionStore` |
| 离线建库 | `scripts/dataset/wan/extract_wan.py`、`scripts/dataset/run_local.py`、`scripts/dataset/pack_motion_store.py`；MotionJEPA `scripts/stage2_memory_export.py` |
| 在线现算 | `scripts/dataset/wan_stage2/stage2_sidecar.py`、`src/mme_vla_suite/policies/{motion_protocol,motion_client,framesamp_memory}.py`、装配 `policies/policy_config.py::create_trained_policy` |
| encoder 本体与导出接口 | MotionJEPA `src/motion_jepa/models/wan_latent_encoder.py::Stage2WanLatentMotionEncoder`、`src/motion_jepa/export/stage2_memory.py::load_ema_encoder / encode_windows` |
| 训练配置 YAML（分支 `v4-256concat1`） | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml` |
