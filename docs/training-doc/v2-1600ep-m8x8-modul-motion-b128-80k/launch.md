# modulation 8×8 motion 80k 起跑档案（准备中）

用户指令原话：「开始实施 有问题尽早问用户」；名称确认原话：「采用 v2-1600ep-m8x8-modul-motion-b128-80k」。本档案随名称确认建立，正式训练尚未启动，无训练结果。

实施依据为根目录 [0916-motion-modul-8x8-plan.md](../../../0916-motion-modul-8x8-plan.md)。先固定关闭态改前基线，完成生产改动及前后逐位验证，再完成新 motion 表及开启态验证；全部闸门通过后起跑。用户已确认建库顺序为 Wan → encode → oracle 重算/汇总 → pack/verify → compare，保留三方 `raw_dir` 强校验。

## 版本与代码状态

实施起始提交 `f9f668920923c58e5d7b721c2827da17b6afa865`，`git status --porcelain` 为空。此提交只是实施起点，**不是正式训练的启动版本**。实际启动提交、命令、进程与退出码待起跑时记录。

## 启动与配置还原

已确认配置落点：新具名配置 `mme_vla_suite_b128_80k`，80,000 步、global batch 128、`fsdp_devices=8`、`num_workers=16`；新 history YAML 为 `perceptual-framesamp-modul-8frame-8x8-motion.yaml`，`motion.budget=160`，demo 最少真实帧 17，尾部重复最后帧。其余训练参数沿用计划锁定的 60k 基线。启动命令尚未生成。

目标库为 `v1-store/datasets/4task-v2-1600ep-604f16da`；原始输入为 `v1-store/raw-h5/4task-20260912-v2`。正式产物将位于 `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-motion-b128-80k`，不复用既有 run。

## 环境与实测状态

2026-09-17 本机检查：环境 B，8 × NVIDIA A100-SXM4-80GB，检查时各卡显存占用 0 MiB；仓库位于 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，底层为 AWS 本地 NVMe RAID `/dev/md0`（XFS）。`df -h` 显示 scratch 剩余约 1.1 TB，`/dev/shm` 剩余 561 GiB。尚无本轮性能数据，不将计划估算写作实测结果。
