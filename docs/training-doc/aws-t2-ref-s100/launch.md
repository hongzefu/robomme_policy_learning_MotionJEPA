# aws-t2-ref-s100（T2 reference：S2_BASE 旧码在环境 B 40 ep 库上的 100 步参照）launch

> 环境 B（AWS 单机 8×A100-SXM4-80GB，仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，介质 AWS 本地 NVMe RAID `/dev/md0`），2026-09-04；库 `v1-store/datasets/4task-motion-40ep`（环境 B 复刻，见 `docs/dataset-build-doc/4task-motion-40ep-aws/`）。
> 用户要求测试类训练 ≤100 步；本轮全部 run 统一 **100 步 × b8 × 2 卡 / fsdp 2 / seed 42 / WORKERS 4 / 确定性档 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` / `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`**，
> 摘要步 {0,25,50,75,99}（`SAVE_INTERVAL=25 + EXTRA_DIGEST_STEPS=99`），输入摘要步 {0,1,2,25,50,75,99}（7 × 8 = 56 样本），800 样本 < 11,530 单 epoch。
> 8 卡当 4 组并行：T2 ref GPU0,1 / T2 cand GPU2,3 / T3 closed GPU4,5 / T3 open GPU6,7。本机数字与环境 A（Ada / turbo）不得混比，且确定性档 run 不作性能结论。

- **目的**：环境 B 没有环境 A 的任何固化基线（T1 / A21 锚点绑死 4task-gl 私有库、生产 norm_stats 与 Ada 卡），关闭态等价改为 **T2 式同机对拍**：
  以 `S2_BASE = c5925d96305f771058e2206ae89461269af9d97c`（motion 接线前最后一个 commit）的旧码作 reference，HEAD 关闭态作 candidate，
  `g0_gate.py --profile t2` 要求 `scalars_hex.tsv` sha256 相等、TrainState 摘要逐位、index 前缀相同。
- **旧码来源**：`git worktree add --detach v1-store/worktrees/s2-base c5925d9`（`uv.lock` 与 HEAD 字节相同，共用主 `.venv`）。**直跑** `bench_train_steps.py`，
  `cd <worktree>` 并 `PYTHONPATH=<worktree>/src UV_PROJECT_ENVIRONMENT=<主树>/.venv uv run --no-sync`——PYTHONPATH 盖过 editable 安装指向的主树 `src`，
  保证 import 的是 c5925d9 的 `mme_vla_suite`。不 source 旧 `paths.sh`（其前缀白名单没有 `/scratch/hongze/`）。
- **norm_stats**：仓库内 `assets/norm_stats.json`（sha `f332bbd3…`）复制到 `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`（用户拍板；生产 `709f22ff…` 在本机不可得）。
- **GPU**：CUDA 0,1。

## 生命周期（脚本 `t2-run.sh ref`，与 `motion-t2-ref/launch.md` 同序）

```bash
REC=$V1/bench/2gpu-epoch-bench/aws-t2-ref-s100; SRC=$(jq -r .source_dataset_root $LIB/framesamp/meta/store_meta.json)
jq -n '{}' > $REC/env.json                                            # 空壳
CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS=<确定性档> XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir $REC --dataset $SRC
ln -sfn $V1/cache/jax/aws-t2-ref-s100 ~/.cache/jax_aws-t2-ref-s100      # 编译缓存收敛（同驱动脚本做法，收尾删除软链）
cd v1-store/worktrees/s2-base && PYTHONPATH=$PWD/src UV_PROJECT_ENVIRONMENT=<主树>/.venv \
  BENCH_RECORD_DIR=$REC BENCH_DIGEST_INTERVAL=25 BENCH_EXTRA_DIGEST_STEPS=99 BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS=<确定性档> XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=disabled \
  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name aws-t2-ref-s100 --assets-base-dir $V1/train-assets --checkpoint-base-dir $V1/train-runs/aws-t2-ref-s100 \
    --batch-size 8 --num-workers 4 --num-train-steps 100 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 --dataset-path $LIB/framesamp \
    --weight-loader.params-path $V1/models/openpi-assets/checkpoints/pi05_base/params --model.use-history --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled
uv run --no-sync python scripts/training/tests/project_scalars.py $REC/metrics.jsonl $REC/scalars_hex.tsv
<python> 写 $REC/t2_reference_manifest.json（S2_BASE、YAML sha、num_exec_samples=11530、steps 100、record_steps/digest_steps、scalars_sha256、argv、medium）
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest $REC
CUDA_VISIBLE_DEVICES=0,1 … JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py check --baseline $REC --dataset $SRC     # BASELINE_ENV=PASS
```

## 判据

tmux 日志尾行 `EXIT_CODE=0`；`metrics.jsonl` 100 行；`param_checksums.jsonl` 步集 {0,25,50,75,99}（`n_leaves=177`）；`batch_digests.jsonl` 步集 {0,1,2,25,50,75,99}（`n_keys=12`）；
`index_sequence.json` n ≥ 800；`check` → `BASELINE_ENV=PASS`。
