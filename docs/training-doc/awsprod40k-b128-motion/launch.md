# awsprod40k-b128-motion — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，96 vCPU，1121 GB RAM，`/dev/md0` 6.9 T 本地 NVMe RAID）**

起跑时间：2026-09-04 20:13:14
起跑 commit：`934ccea`（clean HEAD，与 origin/v2-motionmem 同步）
tmux 会话：`prod-motion`（本轮起过的会话仅此一个；用户自有会话 `0`、`1`、`claude-private` 不动）

## 这条 run 是什么

带 motion memory 的正式训练，40k step。本仓库在环境 B 上的**第一条生产口径训练**——
此前 A100 上只跑过 100 步 × batch 8 × 2 卡的链路验证 run（`aws-t3-{closed,open}-s100`）。

**本轮不含 nomotion 基线对照**（用户明确「正式先只启动带 motion 的」）。
**后果**：单侧结果无法判断 motion 是否有效。后续起对照时用同一 config 条目、同 seed 42、
仅把 `--model.history_config` 换成 `perceptual-framesamp-context.yaml` 即可与本轮直接比较
（global batch 与数据顺序相同）。

## 配置：`mme_vla_suite_b128`

与官方上游 `RoboMME/robomme_policy_learning`（commit `ecf086c`）的 `mme_vla_suite` 条目相比：

**影响训练语义的 5 项（有意偏离，用户逐项拍板）**

| 项 | 官方 | 本 run |
|---|---|---|
| `batch_size`（global） | 64 | **128** |
| `num_train_steps` | 80_000 | **40_000** |
| `lr_schedule.warmup_steps` | 10_000 | **5_000** |
| `lr_schedule.peak_lr` | 5e-5 | **1e-4** |
| `lr_schedule.decay_lr` | 5e-5 | **1e-4** |

总样本量 40k×128 = 5.12M，与官方 80k×64 **相同**；但 optimizer 更新次数只有官方一半。
lr 按 batch 翻倍线性缩放；warmup 按样本量等价缩放（5k×128 == 10k×64）。
`decay_lr` 必须与 `peak_lr` 同步抬——官方 `peak==decay` 使 warmup 后 lr 恒定不衰减，
只抬 peak 会把曲线变成真实余弦衰减（形状改变而非线性缩放）。

**与官方逐项相同**：`optimizer=AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10,
clip_gradient_norm=1.0)`、`ema_decay=0.999`、`fsdp_devices=4`、`seed=42`、`action_horizon=20`、
`ResizeImages(224,224)`、`freeze_filter=PathRegex('.*img.*')`、初始权重 `pi05_base/params`、
`dtype=bfloat16`。`src/openpi/training/optimizer.py` 与上游零 diff。

**不影响训练语义的项**：`decay_steps` 50_000（因 peak==decay，cosine 段为常数，无数值影响）、
`save_interval`/`keep_period` 5_000、`project_name=robomme-framesamp`、
`data.assets=AssetsConfig(assets_dir='v1-store/train-assets/mme_vla_suite/robomme-400ep', asset_id='robomme')`。

**结果不与官方 run 逐项可比**，引用时须带此声明。

## 档位：8 卡 / w16（依据 bench-b128-util 实测）

| 档 | 卡 | worker | s/step | util 均值 | 40k 外推 |
|---|---|---|---|---|---|
| 4 卡 w8 独占 | 4 | 8 | 3.831 | 94.75% | 42.6 h |
| **8 卡 w16 独占** | **8** | **16** | **2.656** | 70.25% | **29.5 h** ← 本 run |
| 8 卡 w32 独占 | 8 | 32 | 2.827 | 66.36% | 31.4 h |

选 8 卡 w16 的理由与完整五档扫描见 `docs/training-doc/bench-b128-util/result.md`。
util 70.25% 低于 4 卡档，但 util 是效率指标而非目标函数——8 卡即便空转 27.5% 仍比 4 卡满载快 13 h。

per-device batch = 128/8 = **16**，与官方 4 卡×batch 64 的 per-device 相同。
mesh = `(jax.device_count()//fsdp_devices, fsdp_devices)` = **(2, 4)**，即 2 路数据并行 × 4 路 FSDP。
`num_workers=16` 经命令行传入，**config 条目默认值 8 不变**（AGENTS.md 第 10 条：属启动脚本覆盖参数）。

## 数据来源

- 数据集：`v1-store/datasets/4task-motion-400ep/framesamp`
  （400 ep / 123,044 timestep / **101,066 exec 样本**，`status=verified`，manifest sha256 `92fa17e9…`）
- motion store：`v1-store/datasets/4task-motion-400ep/motion`（6832 行），经 `MMEVLA_MOTION_STORE`
  覆盖两份 YAML 中硬写的 40ep 路径。覆盖点 `training/dataloader.py::_motion_gates`（env 优先）。
  漏设不会静默串数据：`datastore/motion_store.py::check_same_source` 比对 manifest sha256 后 fail-loud。
- norm_stats：`v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`
- 初始权重：`v1-store/models/openpi-assets/checkpoints/pi05_base/params`（12 G）

**epoch 数**：40k × 128 = 5.12M 样本 ÷ 101,066 = **约 50.6 个 epoch**。
若官方那份 `robomme_preprocessed_data` 是全 16 任务，官方 80k×64 只相当于约 12.7 epoch——
我们只有 4 个任务，等于把同一批数据多看约 4 倍，**过拟合风险显著更高**。本轮接受此风险。

## 启动命令

```bash
. scripts/dataset/paths.sh              # OPENPI_DATA_HOME=v1-store/models 等（不覆盖 HOME）
set -a; . v1-store/secrets/wandb.env; set +a    # WANDB_API_KEY / WANDB_ENTITY，不落日志
export WANDB_DIR=v1-store/cache/wandb
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-400ep/motion
uv run scripts/training/train.py mme_vla_suite_b128 \
  --exp-name=awsprod40k-b128-motion \
  --num-workers=16 \
  --checkpoint-base-dir=v1-store/train-runs \
  --dataset-path=v1-store/datasets/4task-motion-400ep/framesamp \
  --model.use_history \
  --model.history_config=perceptual-framesamp-context-motion.yaml 2>&1 | tee -a "$LOG"
EC=${PIPESTATUS[0]}; echo "EXIT_CODE=$EC" | tee -a "$LOG"
```

未覆盖 `--num-train-steps`（用条目默认 40_000）、未传 `--no-wandb-enabled`（wandb 开启）、
未覆盖 `--log-interval`（用条目默认 100；基准用 10 是为取步时，生产用 100 少约 14% device_get 开销）。

## 输出路径

- 日志：`docs/training-doc/awsprod40k-b128-motion/records/run.txt`
- GPU 采样：`records/gpu_util_lms500_first30min.csv`（前 30 min 密采，PID 266362）、
  `records/gpu_util_15s_full.csv`（全程 15 s，PID 266363）
- checkpoint：`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/`
  （`save_interval=5000` → 8 次，`keep_period=5000`）
- wandb：`https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp`，run 名 `awsprod40k-b128-motion`

## 凭据

`v1-store/secrets/wandb.env`（权限 600，`v1-store/` 整体不进 git）。
**API key 不出现在本留档、commit、日志与 records 任何文件中。**
注：所用为 wandb 新格式 key（`wandb_v1_` 前缀，86 字符）；锁定的 wandb 0.19.11 中
`wandb.login()` 硬校验 40 字符会 ValueError，但 train.py 链路不调用 `login()`、仅读
`WANDB_API_KEY` 环境变量，该路径对新格式 key 正常（`WANDB_PROBE=PASS`），故未升级 wandb。

## 预期

稳态 2.656 s/step → 40k ≈ 29.5 h，叠加起始 XLA 编译与 8 次 checkpoint 约 **30 h**，
预计 2026-09-06 02:00 前后完成。
