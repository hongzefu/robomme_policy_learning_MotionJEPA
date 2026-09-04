# aws-p5-online（P5：在线 sidecar encoder 逐窗 vs 离线 motion 表）launch

> 环境 B（AWS 单机 8×A100-SXM4-80GB，仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，介质 AWS 本地 NVMe RAID `/dev/md0`），2026-09-04；库 `v1-store/datasets/4task-motion-40ep`（环境 B 复刻，见 `docs/dataset-build-doc/4task-motion-40ep-aws/`）。
> 用户要求测试类训练 ≤100 步；本轮全部 run 统一 **100 步 × b8 × 2 卡 / fsdp 2 / seed 42 / WORKERS 4 / 确定性档 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` / `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`**，
> 摘要步 {0,25,50,75,99}（`SAVE_INTERVAL=25 + EXTRA_DIGEST_STEPS=99`），输入摘要步 {0,1,2,25,50,75,99}（7 × 8 = 56 样本），800 样本 < 11,530 单 epoch。
> 8 卡当 4 组并行：T2 ref GPU0,1 / T2 cand GPU2,3 / T3 closed GPU4,5 / T3 open GPU6,7。本机数字与环境 A（Ada / turbo）不得混比，且确定性档 run 不作性能结论。

- **目的**：`scripts/training/g0/compare_online_motion.py`——按 eval.py 节奏驱动 `FrameSampMemory` + 真 sidecar（`motion_sidecar.py`，wan 子 venv、fp32 / 关 TF32 / B=1 / 33 帧一次喂，独占 `--gpu` 那张卡），
  40 条 episode 全跑，每窗 token 与离线 `motion/` 表对应行逐位比，同时核起点集合 / motion_pos / 交错次序 / provenance。
- **命令**（HEAD `8093ebda23ec566533067e319bab506baaf80de5`）：
  `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run --no-sync python scripts/training/g0/compare_online_motion.py --lib $V1/datasets/4task-motion-40ep --gpu 1 --out $V1/reports/motion/p5_online.json`
- **GPU**：主进程 GPU0，sidecar GPU1；与 t3common（GPU2,3）、两条 T3 run（GPU4–7）并行。
- **判据**：`ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0`、`ONLINE_START_SET / ONLINE_POS / ONLINE_ORDER / PROVENANCE=PASS`、`P5_ONLINE=PASS episodes=40 stub=False`。
