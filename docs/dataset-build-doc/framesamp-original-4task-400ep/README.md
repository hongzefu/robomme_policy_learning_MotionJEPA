# framesamp-original-4task-400ep

## 当前状态

状态: 起跑前准备. 正式全量构建尚未启动.

## 用户决策

- 要求与原版实现一致, 不做 dataloader 读取性能优化.
- 处理 `ButtonUnmask`、`VideoUnmask`、`ButtonUnmaskSwap`、`VideoUnmaskSwap` 四个任务, 每个任务 400 episodes.
- 另建每任务 `episode_0` 的 smoke 数据集.
- smoke 使用 `perceptual-framesamp-context` 跑原版完整训练链路 12 steps.
- dataset 生成、模型准备和 smoke 启动命令全部固化为 Bash 脚本.
- 除全局原始 H5 外, 模型、派生数据、缓存、日志和运行产物全部放在当前项目内.

## 原始输入

```text
/data/hongzefu/robomme_data_h5_v2_4env400ep
```

该目录恰好包含四个目标 H5, metadata sidecar 均记录 400 episodes.

## 原版输出契约

保持现有 `scripts/build_dataset.py --dataset_type robomme_pkl` 行为, 不修改 builder、dataset 或 dataloader:

```text
data/<execution_sample_id>.pkl
features/episode_<global_episode_id>/token_emb_<step>.npy
features/episode_<global_episode_id>/kept_indices.json
meta/stats.json
```

## 项目内路径

- 模型与 tokenizer: `.openpi-data/`
- smoke 数据: `data/robomme_preprocessed_4task_original_smoke/`
- 全量数据: `data/robomme_preprocessed_4task_original/`
- norm stats: `runs/assets/mme_vla_suite/robomme/norm_stats.json`
- 日志与不可由 Git 还原的结果: `artifacts/v1_dataloader_restructure/`

## 固定入口

- `scripts/v1_dataloader_restructure/stage_project_models.sh`
- `scripts/v1_dataloader_restructure/launch_4task_smoke_dataset_tmux.sh`
- `scripts/v1_dataloader_restructure/launch_4task_training_smoke_tmux.sh`
- `scripts/v1_dataloader_restructure/launch_4task_full_dataset_tmux.sh`

## 起跑与验收

正式全量构建只允许从 clean HEAD 通过固定 tmux 启动器运行. 启动器会把 HEAD 写入项目内 artifact, 完成后必须满足:

- `features/episode_*` 数量为 1600.
- `meta/stats.json` 的 `execution_samples` 和 `total_samples` 均大于 0.
- 全量 norm stats 成功生成.
- 日志最后出现 `EXIT_CODE=0`.
- 本地运行仅用于构建与正确性验证, 不把本地耗时作为 NFS 吞吐结论.
