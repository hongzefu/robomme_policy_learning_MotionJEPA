# robomme-motionjepa-vla-v1 — awsprod40k-b128-motion checkpoint 备份

本 bucket 存放训练 run **`awsprod40k-b128-motion`** 的全部 8 个 orbax checkpoint，
由 `scripts/dataset/hf_export/run_ckpt_export.sh` 上传，上传前 / 上传后两遍独立 sha256
逐行对照验收（`SHA256SUMS.pre.txt` 与回读侧重算的 post 清单 diff 为空）。

## 这条 run 是什么

带 motion memory 的正式训练，40k step，是本仓库在 AWS 8×A100 单机（环境 B）上的**第一条
生产口径训练**。起跑 commit `934ccea`，训练全程 `EXIT_CODE=0`。

| 项 | 值 |
|---|---|
| config 条目 | `mme_vla_suite_b128` |
| history config | `perceptual-framesamp-context-motion.yaml`（motion 开启） |
| global batch / steps | 128 / 40,000（总样本 5.12 M） |
| lr | warmup 5,000 → peak = decay = 1e-4 |
| optimizer | AdamW(b1=0.9, b2=0.95, eps=1e-8, wd=1e-10, clip_grad_norm=1.0) |
| EMA | 0.999 |
| 初始权重 | `pi05_base/params` |
| seed / fsdp_devices / dtype | 42 / 4 / bfloat16 |
| 任务 | ButtonUnmask、VideoUnmask、ButtonUnmaskSwap、VideoUnmaskSwap |

与官方上游 `mme_vla_suite`（80k × batch 64）相比总样本量相同、optimizer 更新次数减半，
lr 按 batch 翻倍线性缩放。**因此结果不与官方 run 逐项可比**，引用时须带此声明。

motion 侧来源（完整指纹见 `motion_provenance.json`）：MotionJEPA encoder
`wan-v8-filter10-72ep-a` / `checkpoint_epoch_72.pt`（sha256 `bae96037…`），
Wan2.1-T2V-1.3B VAE，motion_dims `[1, 768]`。

## 布局

```
5000/ 10000/ 15000/ 20000/ 25000/ 30000/ 35000/ 39999/   # 8 个 orbax checkpoint，每个 11 GB
  _CHECKPOINT_METADATA
  assets/robomme/norm_stats.json                          # 该 step 的归一化统计
  params/
    _METADATA  _sharding  manifest.ocdbt  array_metadatas/
    ocdbt.process_0/d/<hash>                              # OCDBT 数据块，1–2.2 GB 若干
history_config.resolved.yaml                              # 冻结的 history config 全文
history_config.resolved.sha256                            # 上一文件的 sha256
history_config.txt                                        # 人类可读摘要
motion_provenance.json                                    # motion / VAE / encoder 全链路指纹
wandb_id.txt                                              # wandb run id
SHA256SUMS.pre.txt                                        # 上传前逐文件 sha256（路径相对 bucket 根）
```

step 号即训练步数，`39999` 是最后一个（40k 的末步）。checkpoint 每 5,000 步一存。

## 恢复与自验

```bash
# 下载整个 bucket（huggingface_hub >= 1.x；旧版是 `hf buckets sync`）
hf sync hf://buckets/HongzeFu/robomme-motionjepa-vla-v1 ./robomme-motionjepa-vla-v1
cd robomme-motionjepa-vla-v1

# 校验下载完整性（覆盖除清单自身外的全部字节）
sha256sum -c --strict SHA256SUMS.pre.txt

# 只取最后一个 checkpoint（约 11 GB）
hf sync hf://buckets/HongzeFu/robomme-motionjepa-vla-v1 ./ckpt-39999 --include '39999/*'
```

各 step 目录保持 orbax 原结构，下载下来即可直接 restore，无需解包：

```python
import orbax.checkpoint as ocp
params = ocp.PyTreeCheckpointer().restore("robomme-motionjepa-vla-v1/39999/params")
```

推理所需的归一化统计在同一 step 下的 `assets/robomme/norm_stats.json`。
