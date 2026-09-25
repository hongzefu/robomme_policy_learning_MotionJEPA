# robomme-vla-modul-1024-80k-v1 — v2-1600ep-m16x8x8-modul-b128-80k checkpoint 备份

本 bucket 存放训练 run **`v2-1600ep-m16x8x8-modul-b128-80k`** 的全部 16 个 orbax checkpoint
（327 文件 / 190,050,687,155 B，约 177 GiB），由 `scripts/dataset/hf_export/run_m1024_80k_ckpt_export.sh` 上传，上传前 / 上传后
两遍独立 sha256 逐行对照验收（`SHA256SUMS.pre.txt` 与回读侧重算的 post 清单 diff 为空）。

## 这条 run 是什么

**把 perceptual memory 的 token 预算降到 1024** 的生产训练：`integration_type: modulation`，
perceptual memory 走 **16 帧 × 8×8 frame sampling**（`budget: 1024` = 16 帧 × `token_per_image: 64`），
**`motion` 通道关闭**（`motion_provenance.json` 的 `motion_enabled` 为 `false`、`encoder` 为 `null`）。
80,000 step 跑满，`EXIT_CODE=0` 正常退出，完成器判定
`RUN_COMPLETED=PASS run=v2-1600ep-m16x8x8-modul-b128-80k budget=1024 steps=80000 checkpoints=16 final=79999`。
除 `budget` 外与 2048 那条（`robomme-vla-modul-2048-80k-v1`）、4096 那条（`robomme-vla-modul-4096-80k-v1`）
配置逐字段一致，是 0922 预算扫描「4096 → 1024」顺序训练的第二档；4096 完成验收、八卡释放后才起跑。

它与同账号其他 checkpoint bucket 的关系如下。**与 2048、4096 两条互为单变量对照**（三条只差 `budget`），
与其余各条都不是，不可混作一组读：

| bucket | run | 集成方式 | 帧数 / budget | motion | steps × batch |
|---|---|---|---|---|---|
| `HongzeFu/robomme-vla-modul-60k-v1` | v2-1600ep-m8x8-modul-b128-60k | modulation | 8 / 512 | off | 60k × 128 |
| `HongzeFu/robomme-vla-modul-motion-80k-v1` | v2-1600ep-m8x8-modul-motion-b128-80k | modulation | 8 / 512 | **on** | 80k × 128 |
| `HongzeFu/robomme-motionjepa-vla-v1` | awsprod40k-b128-motion | （上一代） | — | on | 40k × 128 |
| `HongzeFu/robomme-vla-modul-2048-80k-v1` | v2-1600ep-m32x8x8-modul-b128-80k | modulation | 32 / 2048 | off | 80k × 128 |
| `HongzeFu/robomme-vla-modul-4096-80k-v1` | v2-1600ep-m64x8x8-modul-b128-80k | modulation | 64 / 4096 | off | 80k × 128 |
| **本 bucket** | v2-1600ep-m16x8x8-modul-b128-80k | modulation | **16 / 1024** | off | 80k × 128 |

与 2048、4096 两条的**唯一配置差异是 history YAML 的 `budget`（2048 / 4096 → 1024）**：YAML 解析后逐键比较
只差这一个键；`steps × batch`、`lr_schedule`、`fsdp_devices`、`num_workers`、seed、初始权重、
数据集与 `norm_stats` 全部相同。注意预算变化会同时改变「被选帧集合」和动作 query 的 RoPE 位置
（本条为 `1024..1043`，2048 为 `2048..2067`，4096 为 `4096..4115`），所以三条的 loss 不要求相等。
本库每个执行样本的全域帧号都 ≥ 100，1024 档每样本恰好选满 16 帧，补零分支在真实数据上碰不到。

与 motion 开启的 80k 那条的差异是「帧数/token 预算 + motion 开关」两项，与 60k 那条的差异是
「帧数/token 预算 + 训练步数」两项，都**不是**单变量消融。`lr_schedule` 取 `peak_lr == decay_lr == 5e-5`，
因此任意 step 的学习率在各条之间逐步相同，学习率不是混淆因素。

| 项 | 值 |
|---|---|
| config 条目 | `mme_vla_suite_b128_80k`（`src/mme_vla_suite/training/config.py`） |
| history config | `perceptual-framesamp-modul-16frame-8x8.yaml` |
| history config raw sha256 | `ec1308fe96e1eca2028980258a361d863b22e80de6115ebd6e9cb8f4cb6520ed` |
| resolved history config sha256 | `9d7a18fe2cdb720dce21dc7d1c03b7a905fc6f65d98758a582268e5fa9e5390b` |
| integration_type | `modulation` |
| representation_type / perceptual_memory | `perceptual` / `frame_sampling` |
| streaming_obs_horizon / budget / token_per_image / num_views | 16 / **1024** / 64 / 1 |
| memory_feature.img | `net: identity`，`input_dim: 2048` |
| memory_token_dim / pool_type | 1024 / `mean` |
| use_pos_emb / use_state_emb | `true` / `false` |
| motion | **关闭**（`motion_enabled: false`，无 encoder、无 motion store） |
| global batch / steps | 128 / 80,000 |
| save_interval / keep_period | 5,000 / 5,000 |
| fsdp_devices / num_workers / dtype | 8 / 16 / bfloat16 |
| optimizer | AdamW，`clip_gradient_norm=1.0`，`ema_decay=0.999` |
| lr schedule | CosineDecay，warmup 5,000，`peak_lr = decay_lr = 5e-5`，`decay_steps=80,000` |
| seed | 42 |
| 初始权重 | `openpi-assets/checkpoints/pi05_base/params`（独立初始化，不接续任何验证/测速 checkpoint） |
| 任务 | BinFill、RouteStick、VideoRepick、VideoUnmaskSwap |
| framesamp 库 | `4task-v2-1600ep-604f16da/framesamp-8x8`（1,600 集、605,611 执行样本、1,192,918 帧） |
| framesamp manifest sha256 | `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918` |
| framesamp store meta sha256 | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| norm_stats sha256 | `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` |
| 起跑 commit | `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb`（commitV11.6Beta） |
| 起止（UTC） | `2026-09-24T12:44:27Z` → `2026-09-25T10:45:41Z` |
| wandb run id | `0ayufnrr`（project `robomme-framesamp`） |
| 硬件 | 8 × NVIDIA A100-SXM4-80GB，AWS 本地 NVMe RAID `/dev/md0` |

完整 history config 全文见 bucket 根下的 `history_config.resolved.yaml`，
framesamp 链路指纹见 `motion_provenance.json`（该文件的 motion / VAE / encoder 各字段在本 run 全为 `null`）。

## 起跑与完成验收

本 run 按仓库根目录 `0922-4096-1024-8gpu-training-plan.md` 执行，用户原话
「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做
有问题立刻问用户 起泡后给出预计的时间」。由顺序调度器 `scripts/training/prod/run_modul4096_then1024.sh`
经 `run_modul_budget.sh` 起跑。起跑前先要求 4096 完成验收（`RUN_COMPLETED=PASS`）且八卡释放，再通过本档的
输入/模型/20 步对拍/100 步保存加载验证、八卡 20 步容量检查与 1000 步测速（报告 `READY`，稳态步时 0.986 s、
八卡 GPU 利用率均值 98.65%、0% 采样占比 0.07%）。

训练结束后由完成器 `scripts/training/tests/check_modul_completion.py` 验收：唯一终态 `EXIT_CODE=0`、
16 份 checkpoint、800 条 log100 全部有限、末 99 步五项标量全部有限、最终 `79999` 原样读回的数组与
保存现场的 EMA 逐叶摘要一致、全叶有限、固定输入/noise 的 10 步动作输出有限。完成 JSON 与保存现场记录
一并放在 `_run-meta/completed.json` 与 `_run-meta/train-record/` 下。

<!-- TODO-PASS-B: 训练结束后回填本节，未回填前脚本会拒绝上传本 README -->

## 训练结果

<!-- TODO-PASS-B-RESULT -->

## 完整性锚点

- 16 份 `assets/robomme/norm_stats.json` 的 sha256 **全部相同**，为
  `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`
  ——与 `robomme-vla-modul-4096-80k-v1`、`robomme-vla-modul-2048-80k-v1`、`robomme-vla-modul-60k-v1`、`robomme-vla-modul-motion-80k-v1`
  **同值**，这几条 run 用的是同一份归一化统计。
- 每份 `_CHECKPOINT_METADATA` 均带完成提交时间。
- `SHA256SUMS.pre.txt` 覆盖 bucket 内除清单自身外的全部文件（含 `_run-meta/`）。

## 重要声明

1. **不能续训。** checkpoint 只含 EMA `params` 与 `assets`，**没有 optimizer / train_state**
   ——`src/openpi/training/checkpoints.py` 里 `"train_state"` 的 handler 在注册表和 `save_state`
   两处都被注释掉了。这些 checkpoint 只能用于推理或作为 finetune 初始化。
2. **不与上游官方 run 逐项可比。** 官方 `mme_vla_suite` 是 80k × batch 64，本条是 80k × batch 128，
   optimizer 更新次数与样本吞吐都不同，引用时须带此声明。
3. **训练 loss 不代表策略评估成功率。** 本 README 里的 loss 只是训练目标值。
   本 run 的完整策略评估另行开展，不在本 bucket 内。

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
motion_provenance.json                                    # framesamp 链路指纹（motion 字段全为 null）
wandb_id.txt                                              # wandb run id
_run-meta/
  train.log                                               # 训练进程完整 stdout 日志
  wandb/run-<ts>-0ayufnrr/                                # wandb 本地 run 目录（含 .wandb 事件流）
  git-commit.txt                                          # 起跑 commit + 工作区状态
  train-cmd.txt                                           # 完整训练命令行
  completed.json                                          # 完成器判定（status=PASS、run_uuid、训练 HEAD、各 checkpoint 摘要）
  train-record/                                           # 保存现场记录：final/（末步 EMA 逐叶摘要、尾窗标量）、run_meta.json、runtime.json、metrics.jsonl
SHA256SUMS.pre.txt                                        # 上传前逐文件 sha256（路径相对 bucket 根）
```

本 bucket **没有 `_motion-encoder/`**：这条 run 的 motion 通道是关闭的，训练全程不涉及
MotionJEPA encoder。需要 encoder 的是 motion 开启组，见 `HongzeFu/robomme-vla-modul-motion-80k-v1`。

step 号即训练步数，checkpoint 每 5,000 步一存，`79999` 是最后一个（80k 的末步）。
本 run 不写 tensorboard，训练曲线在 `_run-meta/wandb/` 里（也可在 wandb 上按 run id `0ayufnrr` 查）。

## 恢复与自验

```bash
# 下载整个 bucket（huggingface_hub >= 1.x）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-1024-80k-v1 ./robomme-vla-modul-1024-80k-v1
cd robomme-vla-modul-1024-80k-v1

# 校验下载完整性（覆盖除清单自身外的全部字节）
sha256sum -c --strict SHA256SUMS.pre.txt

# 只取最后一个 checkpoint（约 11 GiB）
hf sync hf://buckets/HongzeFu/robomme-vla-modul-1024-80k-v1 ./ckpt-79999 --include '79999/*'
```

各 step 目录保持 orbax 原结构，下载下来即可直接 restore，无需解包：

```python
import orbax.checkpoint as ocp
params = ocp.PyTreeCheckpointer().restore("robomme-vla-modul-1024-80k-v1/79999/params")
```

推理所需的归一化统计在同一 step 下的 `assets/robomme/norm_stats.json`。

复现训练还需要 framesamp store（不在本 bucket 内），指纹见 `motion_provenance.json`，
**数据本体在 `HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1`**
（公开 bucket，含 `framesamp-8x8/`、`source/data_tars/`、`meta/` 与 norm_stats，下完即可开训；
`runner/` 下有现成的恢复与起跑脚本）。本 run 只用其中的 `framesamp-8x8/`，
该 bucket 里的 `motion/` store 对本 run 不需要。
