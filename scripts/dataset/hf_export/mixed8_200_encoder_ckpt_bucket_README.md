# motionjepa-wan-mixed8-200-72ep-v1 — motion encoder run `wan-mixed8-200-filter2-stop075-b88x8-72ep-a`

本 bucket 公开存放 motion encoder 训练 run **`wan-mixed8-200-filter2-stop075-b88x8-72ep-a`** 的
**全部 @@NUM_CKPT@@ 个 epoch checkpoint**（epoch 2…72，每 2 epoch 一存）、训练配置、逐 epoch 指标、
清洗后的训练日志，以及用 epoch 72 对 encoder 留出集全部 @@NUM_MP4@@ 集做的重建可视化（`eval-mp4/`）。
由 `scripts/dataset/hf_export/run_mixed8_200_encoder_ckpt_export.sh` 上传，上传前 / 上传后两遍独立
sha256 逐行对照验收。

> 本文件里的数字与 sha256 由上传驱动从 stage 里的文件现读、现算后填进来，不经人手。

## 这条 run 是什么

在 Wan2.1 VAE latent 空间上训练的 motion encoder：每个 chunk 压成 **1 个 768 维 motion token**，配一个
轻量 DiT decoder 做重建监督，另加 sigreg 正则。产出的 motion 特征离线算好存进 VLA 训练库的 motion 表。

**`checkpoint_epoch_72.pt` 就是 16 任务 mixed8-200 VLA 训练库（`HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1`）
motion 表所用的 encoder**，sha256 `@@E72_SHA@@`（上传驱动断言它与 VLA 侧拷入件逐字节相同）。

| 项 | 值 |
|---|---|
| seed | 42 |
| motion token | `dim 768`，`num_tokens 1` |
| encoder | hidden 768，depth 8，heads 12，dim_head 64，mlp_ratio 4，dropout 0.05 |
| DiT decoder | depth 4，hidden 768，heads 12，dim_head 64，window_size 7，dropout 0.05 |
| latent 空间 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，16 通道，32×32，patchify 2 |
| 数据 | 16 任务 mixed8-200 的 latent 库（`HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1`），`max_horizon 8`，`num_patches 256`，`stride 1` |
| motion filter | 开，阈值 `0.016552884131669998`（StopCube 单独 `0.012414663098752499`） |
| 留出集 | `training.holdout_episodes`（16 任务共 100 格 × 2 集 = 200 集），只被排除出训练集 |
| epochs / batch | 72 / 88 × `grad_accum_steps 2` |
| lr / schedule | `6e-4`，cosine，warmup 1 epoch，`weight_decay 1e-4`，`gradient_clip 0.1` |
| 精度 / EMA | bf16 / `decay 0.999` |
| loss | latent 重建 + sigreg（`weight 1e-4`，global，`knots 17`，`num_proj 1024`，warmup 1 epoch） |
| checkpoint | `save_every 2`，**`save_optimizer: false`，`save_live: false`** |
| 硬件 | 8 × NVIDIA A100-SXM4-80GB |

完整配置全文见 bucket 根下的 `config.yaml`（其中 `data_root` 等几行是建库机上的绝对路径，原样保留）。

## 训练结果

`train.log` 末行 `EXIT_CODE=0`，72 个 epoch 跑满（`train_metrics_epoch.jsonl` 共 @@NUM_EPOCH_ROWS@@ 行）。

| epoch | `loss_total` | `loss_raw/latent` | `loss_raw/sigreg` | `motion/std_batch` | `motion/cos_cross_chunk` |
|---|---:|---:|---:|---:|---:|
| 1 | @@E1_TOTAL@@ | @@E1_LATENT@@ | @@E1_SIGREG@@ | @@E1_STD@@ | @@E1_COS@@ |
| 72 | @@E72_TOTAL@@ | @@E72_LATENT@@ | @@E72_SIGREG@@ | @@E72_STD@@ | @@E72_COS@@ |

**这些都是训练集上的目标值。** 配置里 `training.validate: false`，全程没有跑验证，本表不含任何泛化指标；
留出集上的重建效果看 `eval-mp4/`，下游策略成功率另测。

## EMA-only：只能推理或做初始化，不能续训

训练配置 `save_optimizer: false`、`save_live: false`：每份 checkpoint 只存 EMA 后的 `encoder` 与
`wan_decoder` 权重，**没有 live（非 EMA）权重，也没有优化器状态**。上传驱动对全部 @@NUM_CKPT@@ 份逐个
`torch.load` 核实元数据 `ema_only is True`、`has_live_weights is False` 且无优化器键（判定行 `CKPT_EMA_ONLY=PASS`）。

checkpoint 字节数：@@CKPT_SIZES@@（合计 @@CKPT_BYTES_TOTAL@@ 字节）。

```python
import torch
ck = torch.load("checkpoint_epoch_72.pt", map_location="cpu", weights_only=False)
enc_state = ck["encoder"]          # motion encoder 的 EMA 权重
```

## eval-mp4/：留出集重建可视化

用 `checkpoint_epoch_72.pt` 对 encoder 留出集全部 @@NUM_MP4@@ 集（100 格 × 2 集，不混训练集）逐集出的重建对照视频：

- `eval-mp4/videos/<任务>_ep<编号>.mp4`：@@NUM_MP4@@ 个，h264 / yuv420p / 768×448 / 30 fps；交付侧验收判定行
  `DELIVERY=PASS episodes=200 cells=100 per_cell=2 ckpt=epoch_72`（逐集完整解码、完成凭据与报告复算）。
- `eval-mp4/reports/`：@@NUM_REPORTS@@ 个文本报告（逐集 / 逐格指标与 checkpoint 选择记录）。

## 布局

```
checkpoint_epoch_2.pt … checkpoint_epoch_72.pt      # 每 2 epoch 一存，EMA-only
config.yaml                                         # 该 run 的完整训练配置
train.log                                           # 训练 stdout，已去掉 tqdm 中间态（Epoch 汇总行与 EXIT_CODE= 尾行零丢失）
train_metrics_epoch.jsonl                           # 逐 epoch 全量指标，每行一个 epoch
sig_loss_epoch1.csv                                 # epoch 1 的 sigreg 逐步曲线
gpu_samples.csv                                     # GPU util / 显存采样
eval-mp4/videos/*.mp4  eval-mp4/reports/*           # 留出集重建可视化
SHA256SUMS.pre.txt                                  # 上传前逐文件 sha256（路径相对 bucket 根）
```

wandb 本地 run 目录**刻意没有上传**：其元数据含私有源码仓库的 remote 地址与建库机解释器路径，
逐 epoch 指标已在 `train_metrics_epoch.jsonl`。

## 恢复与自验

```bash
hf sync hf://buckets/HongzeFu/motionjepa-wan-mixed8-200-72ep-v1 ./enc
cd enc && sha256sum -c --strict SHA256SUMS.pre.txt

# 只取 VLA 训练库实际用的那一个
hf sync hf://buckets/HongzeFu/motionjepa-wan-mixed8-200-72ep-v1 ./enc --include 'checkpoint_epoch_72.pt'
```

## 已知事项

1. `config.yaml`、`train.log` 与 `eval-mp4/reports/` 里出现训练 / 评测机上的绝对路径，原样保留、不改字节（公开前体检逐处裁决，
   见 `scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_encoder_ckpt.json`）。
2. 训练 VLA **不需要**本 bucket：motion token 已离线烘焙进 VLA 训练库的 `motion/motion_token.f32.bin`。
   本 bucket 用于对新数据重算 motion 特征，或复核 VLA 训练库 motion 表的来源。
