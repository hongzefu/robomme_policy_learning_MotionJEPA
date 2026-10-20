# v9-st mixed8-200 编码器与重建评估

本公开 bucket 保存运行 `wan-mixed8-200-st512q16-b88x8-60ep-a` 的全部 **@@NUM_CKPT@@ 个 EMA 检查点**（e2…e60，每两轮一份）、训练档案及训后重建评估。上传驱动对全部文件做上传前 SHA、全量读回 SHA 和源侧第三遍校验；完成以驱动末尾 `RESULT=PASS` 和「全部完成」为准。

## 训练口径

| 项目 | 值 |
|---|---|
| 架构 | `wan-latent-v9-st`，空间编码器 576 输入 token、宽 512、8 层、8 heads、4×4 patch、2 层 Perceiver 读出 |
| motion | 16 个 512 维 token；编码器 activation checkpoint 开启 |
| latent decoder | 内联 DiT，宽 512、4 层、8 heads、window 7 |
| 数据 | mixed8-200，16 任务 × 200 个源 episode；200 条 holdout 从训练集剔除 |
| 过滤 | 阈值 `0.016552884131669998`，StopCube × 0.75 |
| 批量与硬件 | 8 × A100-SXM4-80GB，每卡 88，累积 2，有效批量 1408 |
| 优化 | 60 epoch，lr `6e-4`，cosine，warmup 1 epoch，bf16，EMA 0.999 |
| 目标 | latent 重建与 SIGReg；完整参数以 `config.yaml` 为准 |

只训练编码器与训练内联的 latent decoder。本 bucket 不包含策略训练产物。

`train.log` 末行 `EXIT_CODE=0`；60 轮汇总与 @@NUM_EPOCH_ROWS@@ 行指标均已核对。清洗仅删除 tqdm 进度中间态，粘连的 `MOTION_` 诊断先剥出后保留。

| epoch | loss_total | latent | sigreg | motion/std_batch | motion/cos_cross_chunk |
|---|---:|---:|---:|---:|---:|
| 1 | @@E1_TOTAL@@ | @@E1_LATENT@@ | @@E1_SIGREG@@ | @@E1_STD@@ | @@E1_COS@@ |
| 60 | @@E60_TOTAL@@ | @@E60_LATENT@@ | @@E60_SIGREG@@ | @@E60_STD@@ | @@E60_COS@@ |

以上为训练指标，不能替代留出集评估。

## 四臂与 RGB 重建

new = 本次 e60；old = 历史 v7 e60；ref = 历史 v7 e72。old 为数据暴露量相同的历史基线，其学习率调度位置不同，不能当作纯架构对照。16 条 ep199 × 3 个模型 × normal/zero/cross/neighbor，合计 **@@NUM_MP4@@ 段视频**。

指标同时提供全部窗口与剔除 108 个已知重复窗口后的结果，PatternLock 单列。原严格四臂行只记录；稳健行以剔除后为准，在 fp32、TF32 关闭下比较 B1/B16，容差为四臂均值最大漂移的 10 倍。稳健判据是数值筛查，不是统计检验。四臂 `FAIL` 如实归档，不等于评估执行失败；RGB 两条 `PASS` 是数值有限与视频齐全的验收。

下列行从评估日志原样摘录：

```text
@@FOURARM@@
@@FOURARM_ROBUST@@
@@RGB@@
@@RGB_VIDEOS@@
```

## 中途诊断

只枚举 e10/e20/e30/e40/e50。本次纳入 **@@NUM_MIDPROBE@@ 份成功计算报告**；`robust=FAIL` 仍是成功完成计算，报告照常纳入。失败、跳过、未完成、缺失或无效状态只记入下列清单，不计作成功，也不伪造成功报告。报告来自 EMA 的 16 个中窗与 16 个邻窗，仅评 latent，不代表全部 holdout。

成功行从 probe 写入的 `midprobe_line` 原样摘录；其后的 `MIDPROBE_EXPORT` 为导出器根据看守终态生成的收录清单：

```text
@@MIDPROBE@@
```

## 文件布局与复核

```text
checkpoint_epoch_2.pt … checkpoint_epoch_60.pt
config.yaml
train_metrics_epoch.jsonl
sig_loss_epoch1.csv
gpu_samples.csv
train.log
README.md
eval-fourarm/videos/<任务>_ep<编号>_<new|old|ref>_<消融>.mp4
eval-fourarm/reports/summary.json
eval-fourarm/reports/windows.json
eval-fourarm/reports/result-<new|old|ref>.json
eval-fourarm/reports/midprobe-e<N>.json
SHA256SUMS.pre.txt
```

报告共 @@NUM_REPORTS@@ 份，总文件数为 `234+k`，其中 `k` 为成功收录的中途诊断数。检查点大小为 @@CKPT_SIZES@@ 字节，合计 @@CKPT_BYTES_TOTAL@@ 字节。每份均核对 epoch、架构、`ema_only=True`、`has_live_weights=False` 与无优化器键。检查点仅有 EMA 权重，可做推理或新训练段初始化，不能无损续训。

```bash
hf sync hf://buckets/HongzeFu/motionjepa-wan-mixed8-200-st512q16-60ep-v1 ./enc
cd enc
sha256sum -c --strict SHA256SUMS.pre.txt
```

`config.yaml`、`train.log` 与报告中的机器路径原样保留，用于溯源；这些路径在外部不可访问。公开前扫描按专用白名单裁决，疑似凭据一律拒绝。`wandb/` 本地元数据不上传；逐轮指标已单独保留。
