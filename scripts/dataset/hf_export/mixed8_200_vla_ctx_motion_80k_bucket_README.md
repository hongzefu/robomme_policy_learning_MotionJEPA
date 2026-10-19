# robomme-vla-mixed8-200-context-motion-80k-v1 — mixed8-200-m8x8-context-motion-b128-80k checkpoint

本 bucket 公开存放训练 run **`mixed8-200-m8x8-context-motion-b128-80k`** 的全部 @@NUM_STEPS@@ 份 orbax checkpoint
（step @@STEPS@@），以及 run 根下的 history config 快照、provenance 与清洗后的训练日志。由
`scripts/dataset/hf_export/run_mixed8_200_vla_ctx_motion_80k_ckpt_export.sh` 上传，上传前 / 上传后两遍独立
sha256 逐行对照验收（`SHA256SUMS.pre.txt` 与回读侧重算的 post 清单 diff 为空）。

> 本文件里的 sha256、时间与字节数由上传驱动从 run 目录与训练日志现读、现算后填进来，不经人手。

## 这条 run 是什么

16 任务 mixed8-200 新库上的 **512+motion context 同路、motion 开启**那条：`integration_type: context`，帧 512 +
motion 160 都拼进 PaliGemma 主干前缀（`memory_token_dim: 2048`，参数树 59 叶，从 `pi05_base` 冷启动），perceptual
memory 走 8 帧 × 8×8 frame sampling，history yaml 为 `perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml`，
80,000 step × global batch 128（config 条目 `mme_vla_suite_b128_80k`）。
**motion 通道开启**（`motion_provenance.json` 的 `motion_enabled: true`）：每个 chunk 的 768 维 motion token
从训练库的 `motion/motion_token.f32.bin` 读，训练时不跑 encoder。同一库、同一 config 条目、同一 512+motion 记忆、
只换接入方式（`integration_type: modulation`，`memory_token_dim: 1024`）的那条是 commitV11.25 的 512+motion
modulation run，两者构成一组对照；motion 关闭、modulation 接入的 512 基线在
`HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1`。

| 项 | 值 |
|---|---|
| config 条目 | `mme_vla_suite_b128_80k` |
| history config | `@@HC@@` |
| resolved history config sha256 | `@@RESOLVED_SHA@@` |
| global batch / steps | 128 / 80,000（`save_interval` 5000，保留 5000…75000 与收尾 79999） |
| fsdp_devices / num_workers（起跑 env） | @@FSDP_DEVICES@@ / @@NUM_WORKERS@@ |
| 训练库 | `HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1`（`16task-mixed8-200-3200ep-e3c7564f`） |
| framesamp-8x8 store meta sha256 | `@@FS8_META_SHA@@`（上传驱动断言与训练库现场重算值、与 `motion_provenance.json` 三者一致） |
| motion store meta sha256 | `@@MOTION_META_SHA@@`（同上三者一致） |
| motion encoder | `wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt`，sha256 `@@ENC_SHA@@`（= `motion_provenance.json` 的 `encoder.checkpoint_sha256` = VLA 侧拷入件；本体在 `HongzeFu/motionjepa-wan-mixed8-200-72ep-v1`） |
| 起跑 commit | `@@GIT_HEAD@@` |
| 起止（UTC，训练日志原样） | `@@START_UTC@@` → `@@END_UTC@@` |

完整 history config 全文见 bucket 根下的 `history_config.resolved.yaml`，framesamp / motion 全链路指纹见
`motion_provenance.json`；loss 与吞吐的逐区间记录在 `_run-meta/train.log`。

与官方上游 `mme_vla_suite`（80k × batch 64）相比 optimizer 更新次数不同，**结果不与官方 run 逐项可比**，
引用时须带此声明。**训练 loss 不代表策略评估成功率。**

## 布局

```
5000/ 10000/ … 75000/ 79999/      # 16 份 orbax checkpoint 目录（各含 _CHECKPOINT_METADATA 提交标记）
history_config.resolved.yaml      # 训练实际使用的完整 history config（字节即 sha 锚点）
history_config.resolved.sha256
history_config.txt                # 源 yaml 文件名标签
motion_provenance.json            # motion 开关与 framesamp / motion 指纹
wandb_id.txt                      # wandb run id（起跑时开了 wandb 才有）
_run-meta/train.log               # 起跑脚本 tee 的训练日志（已去 tqdm 中间态，EXIT_CODE= 尾行零丢失）
README.md
SHA256SUMS.pre.txt                # 上传前逐文件 sha256（路径相对 bucket 根）
```

checkpoint 合计 @@CKPT_BYTES_TOTAL@@ 字节，每份 @@STEP_BYTES_RANGE@@ 字节。不打 tar：下载下来直接 `orbax.restore`。
wandb 本地 run 目录刻意没有上传（元数据含建库机主机名、GPU UUID 与本机路径），loss 曲线以 `_run-meta/train.log` 为准。

## 恢复与自验

```bash
hf sync hf://buckets/HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1 ./ckpt
cd ckpt && sha256sum -c --strict SHA256SUMS.pre.txt

# 只取最后一份
hf sync hf://buckets/HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1 ./ckpt --include '79999/*'
```

评估侧按 run 根读 `history_config.resolved.yaml`（先核 `.sha256` 与 `motion_provenance.json`），因此下载时
保留 run 根下这几个文件，不要只拷 step 目录。

## 已知事项

1. `motion_provenance.json` 与 `_run-meta/train.log` 里出现训练机上的绝对路径（framesamp / motion 库根、
   起跑命令行、解释器路径），原样保留、不改字节（公开前体检逐处裁决，见
   `scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_ctx_motion_80k.json`）。
2. 推理时若要对新数据在线算 motion token，需要 encoder 本体（`HongzeFu/motionjepa-wan-mixed8-200-72ep-v1` 的
   `checkpoint_epoch_72.pt`）与 Wan2.1 VAE；只复现训练则都不需要。
