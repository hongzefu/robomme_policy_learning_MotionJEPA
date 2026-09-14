# 4×4 在线回归启动记录

本任务对应 `8frame-8x8-training-plan.md` 第三块第一步，使用已有真模型验证在线记忆改为网格驱动后，4×4 行为没有改变。参考代码 `REF=99faacb1319adfc63c0cf9a15187e24c34e38fd1`；候选代码为本文件随附的 `commitV9.2`，启动前要求工作区干净，实际完整 HEAD 在日志首行 `START_HEAD=` 固化，完成后写入结果。

环境 B：AWS 本地 NVMe RAID `/dev/md0`，本轮仅使用物理 GPU 7。数据、模型与缓存均来自当前仓库 `v1-store/` 的已核实副本；不连接集群、不下载或重抽特征。原始 H5 位于 `/scratch/hongze/robomme_data_h5`。

用户指令：“开始实现 有问题越早问用户越好 一口气全做完！”；“注意你只能用4个gpu”。用户确认：“采用文件 SHA＋解析后数组摘要双重检查”；“同意 让评估脚本读取 checkpoint 保存的配置，自动算出正确长度”。本轮服务端按已保存配置计算前缀长度，M8 为 1184、C8 为 1088，采样实现保留。

## 启动命令与产物

四条命令在同一个 detached tmux 会话 `t8-infer-regress-4x4` 中顺序执行，任一失败即停。整体使用 `set -o pipefail`、`PYTHONUNBUFFERED=1` 和 `tee`，日志为 `v1-store/logs/t8-infer-regress-4x4.log`，结束记录 `EXIT_CODE=`。每条命令起跑均不修改仓库源码，输出仅写 `v1-store/reports/t8-infer-regress-4x4/`。

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export JAX_COMPILATION_CACHE_DIR="$V1_STORE/cache/jax/t8-infer-regress-4x4"
uv run --no-sync python scripts/training/tests/online_mem_regress.py \
  --ref-commit 99faacb1319adfc63c0cf9a15187e24c34e38fd1 \
  --frames-from "$V1_STORE/datasets/4task-motion-40ep/source" --n-frames 256
uv run --no-sync python scripts/training/g0/compare_online_memory.py \
  --token-per-image 16 \
  --h5 /scratch/hongze/robomme_data_h5/record_dataset_ButtonUnmask.h5 \
  --out "$V1_STORE/reports/t8-infer-regress-4x4/online_memory"
MMEVLA_MOTION_STORE="$V1_STORE/datasets/4task-motion-400ep/motion" \
uv run --no-sync python scripts/training/g0/compare_train_infer_obs.py \
  --lib "$V1_STORE/datasets/4task-motion-400ep" \
  --ckpt "$V1_STORE/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999" \
  --train-config mme_vla_suite_b128 \
  --norm-stats "$V1_STORE/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json" \
  --motion sidecar --motion-gpu 7 --f32-diag-points-per-episode 3 \
  --out "$V1_STORE/reports/t8-infer-regress-4x4/tic.json"
uv run --no-sync python scripts/training/g0/compare_online_motion.py \
  --gpu 7 --yaml perceptual-framesamp-context-motion.yaml \
  --lib "$V1_STORE/datasets/4task-motion-40ep" \
  --out "$V1_STORE/reports/t8-infer-regress-4x4/online_motion.json"
```

## 判据与已经完成的短测

正式判据为 `ONLINE_MEM_REGRESS=PASS frames=256 keys=4 mismatches=0`、三方池化与装配全部通过、真模型 S 臂各项绝对比较通过，以及 40ep 的 772 个 motion 窗全部逐位相同。TIC 的 bf16 `VT_FULL_VS_CACHED` 按用户裁决只作观察；f32 项必须有有效前缀 KV 零差异且 `rel_fro≤1e-6`。本次显式要求每个 episode 三个 f32 检查点，不能用缺失结果或跳过代替通过。

提交前短测已通过：实际 `FrameSampMemory.add_buffer` 的 4×4、8×8 池化与独立算术公式逐字节一致，交付形状与补零正确；四种非法 token 数在构造时报错；`motion_gates_online.py --gate p3 --yaml perceptual-framesamp-context-8frame-8x8-motion.yaml` 的 32 个推理时刻排列与训练侧公式逐位相同。闭环驱动 DRY_RUN 显示四个 worker 分别绑定物理 GPU 4、5、6、7，并携带指定 checkpoint、UV 缓存与 motion 关闭态参数。这些短测不替代本档案的真实模型回归，也不替代后续 C8/M8 的推理与闭环验收。
