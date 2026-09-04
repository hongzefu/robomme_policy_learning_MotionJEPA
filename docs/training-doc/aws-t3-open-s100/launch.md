# aws-t3-open-s100（T3 open 态 100 步）launch

> 环境 B（AWS 单机 8×A100-SXM4-80GB，仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，介质 AWS 本地 NVMe RAID `/dev/md0`），2026-09-04；库 `v1-store/datasets/4task-motion-40ep`（环境 B 复刻，见 `docs/dataset-build-doc/4task-motion-40ep-aws/`）。
> 用户要求测试类训练 ≤100 步；本轮全部 run 统一 **100 步 × b8 × 2 卡 / fsdp 2 / seed 42 / WORKERS 4 / 确定性档 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` / `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`**，
> 摘要步 {0,25,50,75,99}（`SAVE_INTERVAL=25 + EXTRA_DIGEST_STEPS=99`），输入摘要步 {0,1,2,25,50,75,99}（7 × 8 = 56 样本），800 样本 < 11,530 单 epoch。
> 8 卡当 4 组并行：T2 ref GPU0,1 / T2 cand GPU2,3 / T3 closed GPU4,5 / T3 open GPU6,7。本机数字与环境 A（Ada / turbo）不得混比，且确定性档 run 不作性能结论。

- **目的**：T3 开启态 run：motion memory 接线后 100 步真实训练；提供 t3verifyinit（193 叶）、T3_SMOKE（4 个 motion params 叶初末态变）、t3trace（open 四键由 M1 oracle 重建）、t3mechanism（step 0 batch）、t3phase 的 open 最终 ckpt。跨侧判定与描述性观察都写在本目录 result.md。
- **HEAD**：`8093ebda23ec566533067e319bab506baaf80de5`（commitV6.12，clean）。history config **固定** `perceptual-framesamp-context-motion.yaml`（红线 16，不手改开关）。
- **驱动**：`scripts/training/g0/run_2gpu_epoch_bench.sh`（本 commit 起支持 `BENCH_GPUS`，只换卡号、`BATCH=8` / `--fsdp-devices 2` 不动）；
  `BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=99`（末步经原版 save_state 写目录 **999**——外层编号固定，`final_checkpoint.json` 记 `state_step=100`，EMA）。

```bash
STEPS=100 SAVE_INTERVAL=25 EXTRA_DIGEST_STEPS=99 WORKERS=4 WARMUP_STEPS=20 HISTORY_CONFIG=perceptual-framesamp-context-motion.yaml \
BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=99 EXP_NAME=aws-t3-open-s100 RUN_TAG=aws-t3-open-s100 BENCH_GPUS=6,7 \
DATASET_PATH=$V1/datasets/4task-motion-40ep/framesamp XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
bash scripts/training/g0/run_2gpu_epoch_bench.sh 2>&1 | tee v1-store/logs/aws-t3-open-s100-driver.log
```

- **GPU**：CUDA 6,7。**起跑前**：`t3common`（GPU2,3，`--fsdp 2`）已冻结 reference `v1-store/reports/motion/t3_common_init_reference.json`（`T3_COMMON_INIT=PASS`）。
- **判据**：`BENCH_PASS` / `EXIT_CODE=0`；`metrics.jsonl` 100 行；`param_checksums.jsonl` 步集 {0,25,50,75,99}；`batch_digests.jsonl` 步集 {0,1,2,25,50,75,99}；`index_sequence.json` n ≥ 800；run 目录含 `999`。
