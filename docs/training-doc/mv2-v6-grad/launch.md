# 关闭态固定 batch 梯度取证

环境 B：AWS 本地 NVMe RAID（/dev/md0，XFS），8 × A100-SXM4-80GB。用户原话「开始实施 有问题尽早问用户」。依据 [0916 计划](../../../0916-motion-modul-8x8-plan.md) 步骤 2a，只采集生产改动前证据。

BASE 为包含本档案与步骤 1b 修补的 clean HEAD，由 runner 在启动前断言，并写入 `v1-store/bench/mv2/<阶段>-launch.json`；启动版本不是较早的计划提交。运行期间主副本不改码、不改依赖。所有 Python 命令使用现有 uv 环境，`UV_NO_SYNC=1`、`UV_CACHE_DIR=v1-store/cache/uv`，其余缓存经 `scripts/training/paths.sh` 收敛到 v1-store；额外设置 CUDA / WandB / XDG data 缓存。使用 detached tmux，管道启用 pipefail，日志 tee 并记录 EXIT_CODE。

## 运行口径与判据

GPU 4,5，batch 8、fsdp 2、seed 42；固定 c8-b 的 mixed1/allshort/allfull 三组，DTYPE_BASELINE_CHECKSUMS 不设置。XLA_FLAGS 为 --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0。DTYPE_SOURCE_COMMIT=$BASE，DTYPE_BATCH_FIXTURE_DIR=v1-store/fixtures/8x8/grad/c8-b，DTYPE_GRAD_DIR=v1-store/bench/mv2/v6-base；MMEVLA_JAX_CACHE_DIR=v1-store/cache/jax/mv2-v6。预期三组 FIXTURE/GRAD_KIND 与 GRAD_DONE kinds=3；不执行参数更新，初态与梯度用于改后逐位比较。

## 命令

先 source 训练 paths.sh；BASE 由提交完成后读取，runner 核对工作区为空。

```bash
uv run --no-sync python scripts/training/tests/single_step_grad_fixed.py mme_vla_suite \
  --exp-name mv2-v6 --batch-size 8 --fsdp-devices 2 --seed 42 \
  --assets-base-dir "$V1_STORE/train-assets" --dataset-path "$V1_STORE/datasets/4task-motion-400ep/framesamp-8x8" \
  --weight-loader.params-path "$V1_STORE/models/openpi-assets/checkpoints/pi05_base/params" \
  --model.use-history --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml --no-wandb-enabled
```

本轮对应 tmux 全名 `mv2-v6`。统一外壳为 `bash v1-store/logs/mv2-baseline-runner.sh v6 "$BASE"`；日志 `v1-store/logs/mv2-v6-base.log`。V6 与 V7 串行占用同两张 GPU；不触碰既有 tmux 会话。

## 结果状态

尚未起跑。步骤 1b 已完成真实 400ep 的 24 样本、3 batch 参考链与生产链逐位短测（含 transforms/collate），白名单正负例与 scratch 缓存传递检查均通过。完整本项结论待启动后写入 result.md 与 records/。
