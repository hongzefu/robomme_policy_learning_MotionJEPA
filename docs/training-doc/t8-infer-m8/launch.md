# M8 推理验收启动记录

用户要求“一口气全做完”，并限定只使用四张GPU；本轮固定物理GPU 4–7。参考版本为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，候选代码为 `c08ec2060a544af1869c1e24f755e536150569ca`。M8完整验收使用 `v1-store/train-runs/t8-m8-b/mme_vla_suite/t8-m8-b/999` 的1000更新EMA参数；该训练已完成且三条轨迹逐位通过。七关与48集闭环的完整命令、资源和判据见[共同启动口径](../t8-infer/launch.md)，下方保留更早的独立池化检查启动记录。

## 无需新checkpoint的真实8×8池化检查

在40ep完整fixture运行期间，提前执行计划内 `ENC_LAYER_8X8`。这项检查不依赖新训练checkpoint：使用既有真实SigLIP编码器和H5无损帧，在同一进程中对旧实现、建库域冻结副本与候选在线记忆进行三方比较。CPU fixture与该GPU检查并行，模型代码未变。实际clean启动HEAD记入日志。

输入H5为 `/scratch/hongze/robomme_data_h5/record_dataset_ButtonUnmask.h5`，编码器参数为 `v1-store/models/pi05_vision_encoder/siglip_params.pkl`。AWS本机A100-80GB物理GPU7，存储为本地NVMe RAID `/dev/md0`。会话为 `t8-online-pool-8x8`，日志为 `v1-store/logs/t8-online-pool-8x8.log`，外层使用pipefail、tee和EXIT_CODE。

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
uv run --no-sync python scripts/training/g0/compare_online_memory.py \
  --token-per-image 64 \
  --h5 /scratch/hongze/robomme_data_h5/record_dataset_ButtonUnmask.h5 \
  --out "$V1_STORE/reports/t8-infer-m8/online_memory"
```

要求POS_TABLE、ENC_LAYER_8X8、ASSEMBLY、OOB_PROBE及ONLINE_MEM全部PASS，保留逐步摘要。这个结果覆盖两种motion开关共用的帧编码机制，不能替代后续各自checkpoint上的训练/推理装配、动作与闭环检查。
