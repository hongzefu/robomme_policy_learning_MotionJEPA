# robomme-vla-modul-motion-80k-v1 — v2-1600ep-m8x8-modul-motion-b128-80k checkpoint 备份

本 bucket 存放训练 run **`v2-1600ep-m8x8-modul-motion-b128-80k`** 的全部 16 个 orbax checkpoint
（约 190 GB），由 `scripts/dataset/hf_export/run_motion80k_ckpt_export.sh` 上传，上传前 / 上传后
两遍独立 sha256 逐行对照验收（`SHA256SUMS.pre.txt` 与回读侧重算的 post 清单 diff 为空）。

## 这条 run 是什么

本仓库第一条 **modulation 集成 + motion 通道同时开启**的生产训练：`integration_type: modulation`，
perceptual memory 走 8 帧 × 8×8 frame sampling，**`motion.enabled: true`**，
motion 特征来自 MotionJEPA encoder（见下「motion 链路」）。80,000 step 跑满。

它与同账号另外两个 bucket 构成**三点不同**的一组，**任意两者都不是单变量对照，不可混作一组读**：

| bucket | run | 集成方式 | motion | steps × batch |
|---|---|---|---|---|
| `HongzeFu/robomme-vla-modul-60k-v1` | v2-1600ep-m8x8-modul-b128-60k | modulation | **off** | 60k × 128 |
| `HongzeFu/robomme-motionjepa-vla-v1` | awsprod40k-b128-motion | （上一代） | on | 40k × 128 |
| **本 bucket** | v2-1600ep-m8x8-modul-motion-b128-80k | modulation | **on** | 80k × 128 |

与 60k 那条的差异是「**motion 开关 + 训练步数**」两项，**不是**单变量消融。
`lr_schedule` 取 `peak_lr == decay_lr == 5e-5`，因此**任意 step 的学习率与 60k 条目逐步相同**，
两条 run 在前 60k step 上学习率不构成混淆因素；`fsdp_devices`（4→8）与 `num_workers`（8→16）
也不同，属并行/吞吐配置，不改变数学更新量。

| 项 | 值 |
|---|---|
| config 条目 | `mme_vla_suite_b128_80k`（`src/mme_vla_suite/training/config.py`） |
| history config | `perceptual-framesamp-modul-8frame-8x8-motion.yaml` |
| resolved history config sha256 | `9650f225fe42b802262f93b71139cebbda934dc5ccfaa849fa4dadaf30d73110` |
| integration_type | `modulation` |
| representation_type / perceptual_memory | `perceptual` / `frame_sampling` |
| streaming_obs_horizon / budget / token_per_image | 16 / 512 / 64 |
| memory_token_dim / pool_type | 1024 / `mean` |
| global batch / steps | 128 / 80,000 |
| save_interval / keep_period | 5,000 / 5,000 |
| fsdp_devices / num_workers / dtype | 8 / 16 / bfloat16 |
| optimizer | AdamW，`clip_gradient_norm=1.0`，`ema_decay=0.999` |
| lr schedule | CosineDecay，warmup 5,000，`peak_lr = decay_lr = 5e-5`，`decay_steps=80,000` |
| 初始权重 | `openpi-assets/checkpoints/pi05_base/params` |
| 任务 | BinFill、RouteStick、VideoRepick、VideoUnmaskSwap |
| framesamp 库 | `4task-v2-1600ep-604f16da/framesamp-8x8` |
| framesamp manifest sha256 | `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918` |
| framesamp store meta sha256 | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| 起跑 commit | `2f10473161b760f16d9240d3c2959ff326cde66b` |
| wandb run id | `uzv8avpq`（project `robomme-framesamp`） |
| 硬件 | 8 × NVIDIA A100-SXM4-80GB，driver 595.71.05 |

完整 history config 全文见 bucket 根下的 `history_config.resolved.yaml`，
motion / framesamp 全链路指纹见 `motion_provenance.json`。

## motion 链路

motion 特征是**离线预计算**好落在 `motion` store 里的，训练时按索引读取，不在训练进程内跑 encoder。

| 项 | 值 |
|---|---|
| motion store | `4task-v2-1600ep-604f16da/motion` |
| motion manifest sha256 | `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918` |
| motion index sha256 | `d68696a0c260d689e05f600b135d8372b7bf953a491be41cf364d703068b7864` |
| motion store meta sha256 | `d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268` |
| motion table sha256 | `03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5` |
| encoder run | MotionJEPA `wan-full1600-filter2-b176x4-72ep-a` |
| encoder checkpoint | `checkpoint_epoch_72.pt`（epoch 72，state key `encoder`） |
| encoder checkpoint sha256 | `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca` |
| encoder arch / 精度 | `wan-latent-v7` / bf16（amp bf16，tf32 off） |
| motion_dims | `[1, 768]` |
| VAE | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，fp32，latent_mode `mode` |
| VAE state sha256 | `9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0` |

motion 采样几何（`history_config.resolved.yaml` 的 `motion` 段）：`dim 768`、`budget 160`、
`stride 16`、`window_frames 33`、`window_direction forward`、`grid_origin segment_start`、
`pos_dim 256`、`frame_size 256`、`demo_min_real_frames 17`、`demo_tail_pad repeat_last`。

**encoder 本体就在本 bucket 的 `_motion-encoder/` 下**（`checkpoint_epoch_72.pt` 456 MB +
`config.yaml`），上传前按 `motion_provenance.json` 的 `encoder.checkpoint_sha256` 校验过。
拿到本 bucket 即可对新数据重算 motion 特征，不需要另找 encoder。

⚠️ 注意别拿错：私有模型库 `HongzeFu/MotionJEPA` 里那份 `checkpoint_epoch_72.pt` 属于
**上一代 run `wan-v8-filter10-72ep-a`**（911 MB），**不是**本 run 用的这一个。
这条 encoder 训练 run 的全部 36 个 epoch checkpoint 另见 bucket
`HongzeFu/motionjepa-wan-full1600-72ep-v1`。

<!-- TODO-PASS-B: 训练结束后回填本节，未回填前脚本会拒绝上传本 README -->

## 训练结果

<!-- TODO-PASS-B-RESULT -->

## 完整性锚点

- 16 份 `assets/robomme/norm_stats.json` 的 sha256 **全部相同**，为
  `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`
  ——与无 motion 的 `robomme-vla-modul-60k-v1` **同值**，两条 run 用的是同一份归一化统计。
- 每份 `_CHECKPOINT_METADATA` 均带完成提交时间。
- `SHA256SUMS.pre.txt` 覆盖 bucket 内除清单自身外的全部文件（含 `_run-meta/`）。

## 重要声明

1. **不能续训。** checkpoint 只含 EMA `params` 与 `assets`，**没有 optimizer / train_state**
   ——`src/openpi/training/checkpoints.py` 里 `"train_state"` 的 handler 在注册表和 `save_state`
   两处都被注释掉了。这些 checkpoint 只能用于推理或作为 finetune 初始化。
2. **不与上游官方 run 逐项可比。** 官方 `mme_vla_suite` 是 80k × batch 64，本条是 80k × batch 128，
   optimizer 更新次数与样本吞吐都不同，引用时须带此声明。
3. **训练 loss 不代表策略评估成功率。** 本 README 里的 loss 只是训练目标值。

## 布局

```
5000/ 10000/ 15000/ 20000/ 25000/ 30000/ 35000/ 40000/
45000/ 50000/ 55000/ 60000/ 65000/ 70000/ 75000/ 79999/   # 16 个 orbax checkpoint，每个约 11 GiB
  _CHECKPOINT_METADATA
  assets/robomme/norm_stats.json                          # 该 step 的归一化统计
  params/
    _METADATA  _sharding  manifest.ocdbt  array_metadatas/
    ocdbt.process_0/d/<hash>                              # OCDBT 数据块
history_config.resolved.yaml                              # 冻结的 history config 全文
history_config.resolved.sha256                            # 上一文件的 sha256
history_config.txt                                        # 人类可读摘要
motion_provenance.json                                    # motion / framesamp / VAE / encoder 全链路指纹
wandb_id.txt                                              # wandb run id
_motion-encoder/
  checkpoint_epoch_72.pt                                  # 本 run 实际用的 MotionJEPA encoder 本体，456 MB
  config.yaml                                             # 该 encoder 的训练配置
_run-meta/
  train.log                                               # 训练进程完整 stdout 日志
  wandb/run-<ts>-uzv8avpq/                                # wandb 本地 run 目录（含 .wandb 事件流）
  git-commit.txt                                          # 起跑 commit + 工作区状态
  train-cmd.txt                                           # 完整训练命令行
SHA256SUMS.pre.txt                                        # 上传前逐文件 sha256（路径相对 bucket 根）
```

step 号即训练步数，checkpoint 每 5,000 步一存，`79999` 是最后一个（80k 的末步）。
本 run 不写 tensorboard，训练曲线在 `_run-meta/wandb/` 里（也可在 wandb 上按 run id `uzv8avpq` 查）。

## 恢复与自验

```bash
# 下载整个 bucket（huggingface_hub >= 1.x）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-motion-80k-v1 ./robomme-vla-modul-motion-80k-v1
cd robomme-vla-modul-motion-80k-v1

# 校验下载完整性（覆盖除清单自身外的全部字节）
sha256sum -c --strict SHA256SUMS.pre.txt

# 只取最后一个 checkpoint（约 11 GiB）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-motion-80k-v1 ./ckpt-79999 --include '79999/*'
```

各 step 目录保持 orbax 原结构，下载下来即可直接 restore，无需解包：

```python
import orbax.checkpoint as ocp
params = ocp.PyTreeCheckpointer().restore("robomme-vla-modul-motion-80k-v1/79999/params")
```

推理所需的归一化统计在同一 step 下的 `assets/robomme/norm_stats.json`。

```bash
# 只取 motion encoder（456 MB）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-motion-80k-v1 ./enc --include '_motion-encoder/*'
```

复现训练还需要 framesamp store 与 motion store（不在本 bucket 内），指纹见
`motion_provenance.json`，**数据本体在 `HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1`**
（公开 bucket，含 `framesamp-8x8/`、`motion/`、`source/data_tars/`、`meta/` 与 norm_stats，
下完即可开训；`runner/` 下有现成的恢复与起跑脚本）。

> 此处旧文写的是「数据本体见 `HongzeFu/robomme-4task-motion-full1600-20260914-v1`」，**那是错的**：
> 那个 bucket 装的是阶段一 MotionJEPA encoder 自己的训练数据（Wan chunk latent + chunk motion），
> 既没有 framesamp store 也没有 motion store，照它走复现不了训练。
