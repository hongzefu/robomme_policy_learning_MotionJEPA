# motion-t2-ref（T2 reference：S2 改码前在 40 ep 新库上冻结 300 步参照）launch

- **目的**：`motion-memory-plan.md` 第一部分 5.2 / 第二部分六节——S1 完成后的 clean HEAD 记为 `S2_BASE`，在任何 S2 模型代码改动前，
  以关闭态 YAML 在新库 `v1-store/datasets/4task-motion-40ep/framesamp` 跑 300 步 × batch 8 的 reference；S2 完成后 candidate
  只与它比（`g0_gate.py --profile t2` 唯一成功行 `T2_EQ=PASS`）。
- **run_name**：`motion-t2-ref`（用户 2026-09-03 批准）；`EXP_NAME=RUN_TAG=motion-t2-ref`。
- **S2_BASE / commit**：起跑时 `git rev-parse HEAD`（porcelain 空）写入 `result.md` 与 `t2_reference_manifest.json`。
- **口径**：`scripts/training/g0/bench_train_steps.py` 直跑（不经仍写死 `EPOCH_SAMPLES=395289` 的 `run_2gpu_epoch_bench.sh`），
  argv 与 A21 逐项相同，只换 `--exp-name motion-t2-ref`、`--checkpoint-base-dir …/train-runs/motion-t2-ref`、`--num-train-steps 300`、
  `--dataset-path …/4task-motion-40ep/framesamp`、`--save-interval 1`（记录器按 `BENCH_DIGEST_INTERVAL=100` + `BENCH_EXTRA_DIGEST_STEPS=299` 自选）；
  `--model.history-config perceptual-framesamp-context.yaml`（closed）；2×RTX 6000 Ada、b8、seed 42、`fsdp_devices=2`、`WORKERS=4`、
  确定性档 `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`CUDA_VISIBLE_DEVICES=0,1`。
  样本数从 `framesamp/meta/store_meta.json.num_exec_samples` 现场读得 **11,530**，300 × 8 = 2,400 单 epoch内；`index_sequence` 须覆盖 ≥ 2,400 条。
- **存储介质**：新库在本机 NVMe（工作副本内 `v1-store/`），与 G0b（turbo NFS）不同介质，本 run 只作等价对拍参照、不作吞吐结论。
- **framesamp 库锚点**：`status=verified`、`num_rows=13756`、`num_exec_samples=11530`、`num_pos_rows=586`、`manifest_sha256=fee2777f…`；
  `store_meta.json` sha256 `022e3ba2…`（完整值见 manifest）。

## 生命周期（严格顺序，plan 5.2）

```bash
cd /data/hongzefu/robomme_policy_learning_MotionJEPA; V1=$PWD/v1-store; LIB=$V1/datasets/4task-motion-40ep
REC=$V1/bench/2gpu-epoch-bench/motion-t2-ref; mkdir -p $REC
SRC=$(jq -r '.source_dataset_root' $LIB/framesamp/meta/store_meta.json)            # 源 pkl 根（禁止把 packed 根传给旧 checker）
jq -n '{}' > $REC/env.json.tmp && mv $REC/env.json.tmp $REC/env.json                 # 原子空壳（旧 checker 的 check 不读 standalone fingerprint.json）
CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  UV_LINK_MODE=copy JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir $REC --dataset $SRC
# jax 编译缓存收敛进 v1-store（同 run_2gpu_epoch_bench.sh 的软链法）
mkdir -p $V1/cache/jax/motion-t2-ref ~/.cache && ln -sfn $V1/cache/jax/motion-t2-ref ~/.cache/jax_motion-t2-ref
tmux new-session -d -s motion-t2-ref "set -o pipefail; cd $PWD; \
  BENCH_RECORD_DIR=$REC BENCH_DIGEST_INTERVAL=100 BENCH_EXTRA_DIGEST_STEPS=299 BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 \
  CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name motion-t2-ref \
    --assets-base-dir $V1/train-assets --checkpoint-base-dir $V1/train-runs/motion-t2-ref --batch-size 8 --num-workers 4 \
    --num-train-steps 300 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 --dataset-path $LIB/framesamp \
    --weight-loader.params-path $V1/models/openpi-assets/checkpoints/pi05_base/params --model.use-history \
    --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled 2>&1 | tee $V1/logs/motion-t2-ref.log; \
  echo \"EXIT_CODE=\$?\" >> $V1/logs/motion-t2-ref.log"
# 训练成功后
UV_LINK_MODE=copy uv run --no-sync python scripts/training/tests/project_scalars.py $REC/metrics.jsonl $REC/scalars_hex.tsv
jq -n --arg s2base "$(git rev-parse HEAD)" ... > $REC/t2_reference_manifest.json.tmp && mv $REC/t2_reference_manifest.json.tmp $REC/t2_reference_manifest.json
UV_LINK_MODE=copy JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest $REC
CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS=... XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 UV_LINK_MODE=copy JAX_PLATFORMS=cpu \
  uv run --no-sync python scripts/training/g0/check_baseline_env.py check --baseline $REC --dataset $SRC      # BASELINE_ENV=PASS
```

`t2_reference_manifest.json` 记：`S2_BASE`、源 YAML 路径与原始字节 sha256、`num_exec_samples=11530`、训练语义 argv、环境指纹路径、
日志路径、`store_meta_sha256`、`manifest_sha256`、`scalars_hex.tsv` sha256、记录步集 `{0,1,2,100,200,299}`。

## 判据

- tmux 日志尾行唯一 `EXIT_CODE=0`；`metrics.jsonl` 300 行；`param_checksums.jsonl` 记录步集 = {0,100,200,299}（`n_leaves=177`）；
  `batch_digests.jsonl` 步集 = {0,1,2,100,200,299}（`n_keys=12`）；`index_sequence.json` n ≥ 2,400；
  `check` 三次（起跑前 dump 自校、S2 candidate 起跑前、T2 gate 前）均 `BASELINE_ENV=PASS`。
