# motionjepa-wan-full1600-72ep-v1 — MotionJEPA encoder run `wan-full1600-filter2-b176x4-72ep-a` 全量备份

本 bucket 存放 MotionJEPA encoder 训练 run **`wan-full1600-filter2-b176x4-72ep-a`** 的
**全部 36 个 epoch checkpoint**（epoch 2…72，每 2 epoch 一存，各 ~456 MiB，合计 17.19 GB）
以及训练日志、逐 epoch 指标、wandb 本地 run 目录，由
`scripts/dataset/hf_export/run_motionjepa_encoder_full1600_export.sh` 上传，
上传前 / 上传后两遍独立 sha256 逐行对照验收。

## 这条 run 是什么

在 Wan2.1 VAE latent 空间上训练的 motion encoder：每个 chunk 压成 **1 个 768 维 motion token**，
配一个轻量 DiT decoder 做重建监督，另加 sigreg 正则。产出的 motion 特征离线算好存进 motion store，
供下游 VLA 训练读取。

**`checkpoint_epoch_72.pt` 就是 VLA run `v2-1600ep-m8x8-modul-motion-b128-80k`
（bucket `HongzeFu/robomme-vla-modul-motion-80k-v1`）实际使用的那一个 encoder**，
sha256 `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca`，
与该 run 的 `motion_provenance.json` 里 `encoder.checkpoint_sha256` 一致
（导出脚本阶段 5 对这一条做了硬断言）。

⚠️ 别和私有模型库 `HongzeFu/MotionJEPA` 里的 `wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt`
搞混 —— 那是**上一代** run（911 MB，不同架构配置），不是这一版。

| 项 | 值 |
|---|---|
| 仓库 / commit | `MotionJEPA` @ `f43b38fe829f6fdc2df33fa24e4d2a4cc085685e` |
| seed | 42 |
| motion token | `dim 768`，`num_tokens 1` |
| encoder | hidden 768，depth 8，heads 12，dim_head 64，mlp_ratio 4，dropout 0.05 |
| DiT decoder | depth 4，hidden 768，heads 12，dim_head 64，window_size 7，dropout 0.05 |
| latent 空间 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，16 通道，32×32，patchify 2 |
| 数据 | `dataset-local-a100-4task-full1600/dataset-token`，`max_horizon 8`，`num_patches 256`，`stride 1` |
| motion filter | 开，阈值 `0.02711051143705845` |
| epochs / batch | 72 / 176 × `grad_accum_steps 2`（等效 352） |
| lr / schedule | `6e-4`，cosine，warmup 1 epoch，`weight_decay 1e-4`，`gradient_clip 0.1` |
| 精度 / EMA | bf16 / `decay 0.999` |
| loss | latent 重建 + sigreg（`weight 1e-4`，global，`knots 17`，`num_proj 1024`，warmup 1 epoch） |
| checkpoint | `save_every 2`，**`save_optimizer: false`** |
| 硬件 | 4 × NVIDIA A100-SXM4-80GB，峰值显存 68.53 GiB/卡，util 均值 98.5% |
| wandb | project `robomme-encoder-4taskV2`，run `sjddeth0` |

完整配置全文见 bucket 根下的 `config.yaml`。

## 训练结果

`train.log` 末尾 `TRAIN_DONE` / `EXIT_CODE=0`，72 个 epoch 跑满。

| epoch | `loss_total` | `loss_raw/latent` | `loss_raw/sigreg` | `motion/std_batch` | `motion/cos_cross_chunk` |
|---|---:|---:|---:|---:|---:|
| 1 | 0.04013439 | 0.03729954 | 140.19130 | 0.85745 | 0.09646 |
| 72 | 0.00156758 | 0.00131148 | 2.56074 | 0.99541 | −0.0000060 |

逐 epoch 全量指标在 `train_metrics_epoch.jsonl`（每行一个 epoch，含 `latent_by_k`、
`latent_identity_by_k`、grad_clip 分位数、吞吐、峰值显存）。

吞吐：约 **874–883 秒/epoch、872 samples/s**（4×A100，batch 176，worker 4）。

**这些都是训练集上的目标值。** 配置里 `training.validate: false`，
`holdout_episodes` 只是被排除出训练集，**全程没有跑过验证**，所以本表**不包含任何泛化指标**；
下游策略成功率更是要另测，不能从这里推断。

## 完整性锚点

- 36 个 checkpoint 只有**两种**字节数：epoch 2/4/6/8 各 `477,432,314` 字节，其余 32 个各
  `477,432,433` 字节，合计 `17,187,567,112` 字节。差的 119 字节在存档元数据里，不是权重形状变化
  ——`save_optimizer: false`，每份只存 EMA `encoder` 与 `wan_decoder` 的 state_dict。
- `checkpoint_epoch_72.pt` 的 sha256 与下游 VLA run 的 `motion_provenance.json` 交叉一致（见上）。
- `SHA256SUMS.pre.txt` 覆盖 bucket 内除清单自身外的全部文件。

## 布局

```
checkpoint_epoch_2.pt … checkpoint_epoch_72.pt      # 36 个，每 2 epoch 一存，各 ~477.4 MB
config.yaml                                         # 该 run 的完整训练配置
train.log                                           # 训练进程完整 stdout（尾部含 TRAIN_DONE / EXIT_CODE=0）
train_metrics_epoch.jsonl                           # 逐 epoch 全量指标，每行一个 epoch
sig_loss_epoch1.csv                                 # epoch 1 的 sigreg 逐步曲线
gpu_samples.csv                                     # GPU util / 显存采样
wandb/run-20260914_041732-sjddeth0/                 # wandb 本地 run 目录（含 .wandb 事件流）
SHA256SUMS.pre.txt                                  # 上传前逐文件 sha256（路径相对 bucket 根）
```

## 恢复与自验

```bash
hf sync hf://buckets/HongzeFu/motionjepa-wan-full1600-72ep-v1 ./motionjepa-wan-full1600-72ep-v1
cd motionjepa-wan-full1600-72ep-v1
sha256sum -c --strict SHA256SUMS.pre.txt

# 只取下游 VLA 实际用的那一个（456 MiB）
hf sync hf://buckets/HongzeFu/motionjepa-wan-full1600-72ep-v1 ./enc --include 'checkpoint_epoch_72.pt'
```

checkpoint 是普通的 torch 存档，`encoder` 键即 motion encoder 的 EMA 权重：

```python
import torch
ck = torch.load("checkpoint_epoch_72.pt", map_location="cpu")
print(ck.keys())              # encoder / wan_decoder / epoch 等
enc_state = ck["encoder"]
```

**没有 optimizer 状态**（`save_optimizer: false`），因此**不能从这些 checkpoint 续训**，
只能用于推理（算 motion 特征）或作为初始化。
