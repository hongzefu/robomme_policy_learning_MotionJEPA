# robomme-vla-modul-60k-v1 — v2-1600ep-m8x8-modul-b128-60k checkpoint 备份

本 bucket 存放训练 run **`v2-1600ep-m8x8-modul-b128-60k`** 的全部 12 个 orbax checkpoint，
由 `scripts/dataset/hf_export/run_modul60k_ckpt_export.sh` 上传，上传前 / 上传后两遍独立
sha256 逐行对照验收（`SHA256SUMS.pre.txt` 与回读侧重算的 post 清单 diff 为空）。

## 这条 run 是什么

本仓库第一条 **modulation 集成方式**的生产训练：`integration_type: modulation`，
perceptual memory 走 8 帧 × 8×8 frame sampling。60,000 step 跑满，`EXIT_CODE=0`。

**motion 通道在这条 run 里是关闭的**（`motion_provenance.json` 的 `motion_enabled: false`，
`motion_root` / `vae` / `encoder` 均为 `null`）。这与同账号 bucket
`HongzeFu/robomme-motionjepa-vla-v1` 存放的 `awsprod40k-b128-motion`（motion 开启）
是**不同的实验条件**，两者不可混作一组对照读。

| 项 | 值 |
|---|---|
| config 条目 | `mme_vla_suite_b128_60k` |
| history config | `perceptual-framesamp-modul-8frame-8x8.yaml` |
| resolved history config sha256 | `804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92` |
| integration_type | `modulation` |
| representation_type / perceptual_memory | `perceptual` / `frame_sampling` |
| streaming_obs_horizon / budget / token_per_image | 16 / 512 / 64 |
| memory_token_dim | 1024 |
| global batch / steps | 128 / 60,000 |
| fsdp_devices / dtype | 4 / bfloat16 |
| 任务 | ButtonUnmask、VideoUnmask、ButtonUnmaskSwap、VideoUnmaskSwap |
| framesamp 库 | `4task-v2-1600ep-604f16da/framesamp-8x8` |
| framesamp manifest sha256 | `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918` |
| framesamp store meta sha256 | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| 起跑 commit | `dd07f18fc385b01eb52db7563fe5f202997b9706` |
| 复核 commit | `9a58dc8`（与起跑版本的 `src`/`scripts`/`packages`/`pyproject.toml`/`uv.lock` 无差异） |
| wandb run id | `6ubtaf9l` |

完整 history config 全文见 bucket 根下的 `history_config.resolved.yaml`，
motion / framesamp 全链路指纹见 `motion_provenance.json`。

与官方上游 `mme_vla_suite`（80k × batch 64）相比 optimizer 更新次数不同，
**结果不与官方 run 逐项可比**，引用时须带此声明。

## 训练结果

起跑 `2026-09-15T05:48:27Z`，退出 `2026-09-16T20:53:31Z`，耗时 **39 小时 5 分 4 秒**
（140,704 秒，含初始化、checkpoint 保存与退出）。

| 日志步 | loss（区间统计，step 0 除外） |
|---|---:|
| 0 | 0.08936774 |
| 5000 | 0.00827814 |
| 10000 | 0.00547655 |
| 30000 | 0.00277228 |
| 50000 | 0.00201081 |
| 59900 | 0.00177253 |

`log_interval=100`，除 step 0 外均为日志区间统计。最后一条 loss **0.0017725301 属于
step 59900**，覆盖训练 step 59801–59900，**不能称作 step 59999 的单步 loss**；
step 59901–59999 共 99 步没有独立标量日志。**训练 loss 不代表策略评估成功率。**

性能（存储为 AWS 本地 NVMe RAID `/dev/md0`，batch 128、worker 8、fsdp 4）：step 100→59900
的 59,800 次更新耗时 139,951.786 秒，均值 **2.34033 秒/步、54.69 samples/s**；四卡 util
均值 **72.39%**、0% 采样占比 **25.39%**（15 秒采样 37,520 条，采样间隔大于步时，
属离散估计，不能据此逐步定位等待）。

## 完整性锚点

- 12 份 `assets/robomme/norm_stats.json` 的 sha256 **全部相同**，为
  `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。
- 末步 `59999` 在 CPU 上完整读取参数并与 modulation 配置精确匹配：
  `PARAM_TREE_EXACT=PASS n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0`
  （`yaml_sha256=5b5ac2f85302d4d8`，六条 modulation 专属参数路径齐全）。
- 12 份 `_CHECKPOINT_METADATA` 均带完成提交时间。

## 布局

```
5000/ 10000/ 15000/ 20000/ 25000/ 30000/ 35000/
40000/ 45000/ 50000/ 55000/ 59999/                        # 12 个 orbax checkpoint，每个 11.06 GiB
  _CHECKPOINT_METADATA
  assets/robomme/norm_stats.json                          # 该 step 的归一化统计
  params/
    _METADATA  _sharding  manifest.ocdbt  array_metadatas/
    ocdbt.process_0/d/<hash>                              # OCDBT 数据块，最大 2.66 GB
history_config.resolved.yaml                              # 冻结的 history config 全文
history_config.resolved.sha256                            # 上一文件的 sha256
history_config.txt                                        # 人类可读摘要
motion_provenance.json                                    # motion / framesamp 全链路指纹
wandb_id.txt                                              # wandb run id
SHA256SUMS.pre.txt                                        # 上传前逐文件 sha256（路径相对 bucket 根）
```

step 号即训练步数，checkpoint 每 5,000 步一存，`59999` 是最后一个（60k 的末步）。

## 恢复与自验

```bash
# 下载整个 bucket（huggingface_hub >= 1.x；旧版是 `hf buckets sync`）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-60k-v1 ./robomme-vla-modul-60k-v1
cd robomme-vla-modul-60k-v1

# 校验下载完整性（覆盖除清单自身外的全部字节）
sha256sum -c --strict SHA256SUMS.pre.txt

# 只取最后一个 checkpoint（约 11 GiB）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-60k-v1 ./ckpt-59999 --include '59999/*'
```

各 step 目录保持 orbax 原结构，下载下来即可直接 restore，无需解包：

```python
import orbax.checkpoint as ocp
params = ocp.PyTreeCheckpointer().restore("robomme-vla-modul-60k-v1/59999/params")
```

推理所需的归一化统计在同一 step 下的 `assets/robomme/norm_stats.json`。
