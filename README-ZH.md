# RoboMME Policy Learning + MotionJEPA

## 当前状态

当前工作分支为 `v1-dataloader-Restructure`。

仓库 `/init` 已完成。当前任务只增加与原版行为一致的可复现外围脚本，尚未处理四任务数据、运行 smoke run 或接入 MotionJEPA motion token；原版 builder、dataset、dataloader、frame sampling 和训练逻辑均保持不变。

## 项目总体目标

本仓库以 MME-VLA 的 `perceptual-framesamp-context` 为起点，后续将修改该链路并接入 [MotionJEPA](https://github.com/hongzefu/MotionJEPA) 生成的 motion token。

MotionJEPA 的参考分支为 `v6.1.1-slurmWanExtract`。本仓库当前只完成项目范围和协作约束的初始化，不复制其 Wan/v7 训练实现。

## v1 计划范围

`v1-dataloader-Restructure` 未来只负责重构 dataloader，并在不改变模型训练语义的前提下尽可能优化训练吞吐。v1 只关注以下四个任务：

- `ButtonUnmask`
- `VideoUnmask`
- `ButtonUnmaskSwap`
- `VideoUnmaskSwap`

以下事项属于 v1 后续工作，不属于本次 `/init` 已完成内容：

- 处理四个任务的原始数据。
- 运行本地功能 smoke run。
- 重构和验证 dataloader。
- 在 NFS 环境进行正式吞吐测试。

## 数据、模型与运行边界

### 全局原始 H5

最初的全局原始 H5 可以保留在项目目录外。当前本地四任务数据位于：

```text
/data/hongzefu/robomme_data_h5_v2_4env400ep
```

该目录只是未来数据处理的输入，本次初始化不会读取、转换或复制这些 H5。

### 项目内联要求

除全局原始 H5 外，后续产生的预处理数据、索引、缓存、模型、tokenizer、checkpoint、日志和 smoke 产物都必须放在以下项目目录内：

```text
/data/hongzefu/robomme_policy_learning_MotionJEPA
```

本轮已固定以下项目内路径：

- `.openpi-data/`：SigLIP、PaliGemma tokenizer 和 pi05_base。
- `data/robomme_preprocessed_4task_original_smoke/`：每任务 `episode_0` 的原版 smoke 数据。
- `data/robomme_preprocessed_4task_original/`：每任务 400 episodes 的原版全量数据。
- `runs/assets/mme_vla_suite/robomme/norm_stats.json`：当前数据对应的 norm stats。
- `.cache/` 与 `.runtime-home/`：uv、JAX、wandb 等运行缓存。
- `artifacts/v1_dataloader_restructure/`：构建日志、smoke 日志和不可由 Git 还原的验收结果。

## 固定可复现入口

所有实际命令均写入 Bash，不依赖聊天记录：

```bash
bash scripts/v1_dataloader_restructure/stage_project_models.sh
bash scripts/v1_dataloader_restructure/launch_4task_smoke_dataset_tmux.sh
bash scripts/v1_dataloader_restructure/launch_4task_training_smoke_tmux.sh
bash scripts/v1_dataloader_restructure/launch_4task_full_dataset_tmux.sh
```

训练 smoke 由 Bash 调用 `scripts/smoke_train_once.py`，该入口只调用一次现有 `scripts.train.main()`，并强制原版 tentative 口径的 12 steps。它不复制或修改 dataloader、loss、优化器与模型逻辑。

### 本地与 NFS

- `/data/hongzefu` 只用于未来的数据处理、正确性检查和 smoke run。
- 本地 NVMe 上得到的 batch latency 或 samples/s 不得作为最终吞吐指标。
- 正式吞吐结论必须在 NFS 数据副本上重新测试，并记录数据位置、batch size、worker 数、warmup 和稳定态统计。

## 环境与协作约定

- Python 环境和命令统一使用 uv。
- 所有计划、文档、进度和总结使用简体中文。
- 文档中的未来 scope 不代表已经实现，也不代表可以在未获用户指令时直接开始实施。
- 详细工作规则见 [`AGENTS.md`](AGENTS.md)。

## 后续阶段

当前已获用户确认的执行顺序是：复制项目内模型，构建四任务 smoke 数据，运行原版 12-step training smoke，再启动四任务各 400 episodes 的原版全量预处理。MotionJEPA motion token 接入仍属于更后的独立阶段。
