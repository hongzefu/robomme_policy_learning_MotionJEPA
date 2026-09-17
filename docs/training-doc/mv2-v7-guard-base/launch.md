# 关闭态真实 100 步基线

环境 B：AWS 本地 NVMe RAID（/dev/md0，XFS），8 × A100-SXM4-80GB。用户原话「开始实施 有问题尽早问用户」。依据 [0916 计划](../../../0916-motion-modul-8x8-plan.md) 步骤 2a，只采集生产改动前证据。

BASE 为包含本档案与步骤 1b 修补的 clean HEAD，由 runner 在启动前断言，并写入 `v1-store/bench/mv2/<阶段>-launch.json`；启动版本不是较早的计划提交。运行期间主副本不改码、不改依赖。所有 Python 命令使用现有 uv 环境，`UV_NO_SYNC=1`、`UV_CACHE_DIR=v1-store/cache/uv`，其余缓存经 `scripts/training/paths.sh` 收敛到 v1-store；额外设置 CUDA / WandB / XDG data 缓存。使用 detached tmux，管道启用 pipefail，日志 tee 并记录 EXIT_CODE。

## 运行口径与判据

GPU 4,5，batch 8、fsdp 2、seed 42，WORKERS=4、WARMUP_STEPS=20、STEPS=100、SAVE_INTERVAL=25、EXTRA_DIGEST_STEPS=99。EXP_NAME=mv2-v7、RUN_TAG=mv2-v7-base、KEEP_JAX_CACHE=1，前后基线共用 v1-store/cache/jax/mv2-v7。输出 v1-store/bench/2gpu-epoch-bench/mv2-v7-base。预期 metrics 100 行、状态摘要步 {0,25,50,75,99}、输入摘要步 {0,1,2,25,50,75,99}、至少 800 个采样索引。驱动本身采集环境指纹，随后 project_scalars 与 check_baseline_env manifest 固化结果。驱动只清理本轮独立 run 空壳，原始指标与摘要始终保留在 bench 目录；先归档这些文件再清理其他本轮临时产物。此档开启确定性与摘要，步时不作性能结论。

## 命令

先 source 训练 paths.sh；BASE 由提交完成后读取，runner 核对工作区为空。

```bash
STEPS=100 SAVE_INTERVAL=25 EXTRA_DIGEST_STEPS=99 WORKERS=4 WARMUP_STEPS=20 \
HISTORY_CONFIG=perceptual-framesamp-modul-8frame-8x8.yaml EXP_NAME=mv2-v7 RUN_TAG=mv2-v7-base \
BENCH_GPUS=4,5 KEEP_JAX_CACHE=1 DATASET_PATH="$V1_STORE/datasets/4task-motion-400ep/framesamp-8x8" \
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
bash scripts/training/g0/run_2gpu_epoch_bench.sh
uv run --no-sync python scripts/training/tests/project_scalars.py "$REC/metrics.jsonl" "$REC/scalars_hex.tsv"
uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest "$REC"
```

本轮对应 tmux 全名 `mv2-bench-base`。统一外壳为 `bash v1-store/logs/mv2-baseline-runner.sh v7 "$BASE"`；日志 `v1-store/logs/mv2-v7-base.log`。V6 与 V7 串行占用同两张 GPU；不触碰既有 tmux 会话。

## 结果状态

尚未起跑。步骤 1b 已完成真实 400ep 的 24 样本、3 batch 参考链与生产链逐位短测（含 transforms/collate），白名单正负例与 scratch 缓存传递检查均通过。完整本项结论待启动后写入 result.md 与 records/。
