# RoboMME Policy Learning + MotionJEPA

## 当前状态

当前工作分支为 `v1-dataloader-Restructure`。

本轮只完成仓库 `/init`：创建中文项目说明和 agent 工作规则。当前尚未实现 dataloader 重构，尚未处理四任务数据，尚未运行 smoke run，也尚未接入 MotionJEPA motion token。

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

具体子目录结构将在对应实现任务开始前由用户确认，本次初始化不预设新的数据格式或模型目录。

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

完成本次 `/init` 后，后续任务将由用户逐项明确启动。预期顺序是：先处理 v1 dataloader 和四任务 smoke，再在 NFS 上验证吞吐；MotionJEPA motion token 接入属于更后的独立阶段。
