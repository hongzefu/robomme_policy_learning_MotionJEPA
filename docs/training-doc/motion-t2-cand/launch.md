# motion-t2-cand（T2 candidate：新库 300 步，对冻结的 motion-t2-ref 严格 A/B）launch

- **目的**：`motion-memory-plan.md` 5.2 T2——S2 后以相同训练语义参数在 `4task-motion-40ep/framesamp` 跑 300 步 × batch 8，
  用 `g0_gate.py --profile t2` 对 reference（`motion-t2-ref`，`S2_BASE=c5925d9`）逐位：日志唯一 `EXIT_CODE=0`、环境指纹相同、规范化 argv 只差
  run / output 路径、配置只新增规范 `motion.enabled:false` 节、step 集 / scalar 键全集相同且 hex 逐位、TrainState 摘要步集 {0,100,200,299} `state_digest` 逐位且 `n_leaves=177`、
  输入摘要步集 {0,1,2,100,200,299} raw 逐位且 `n_keys=12`、index 序列前 2,400 逐项相同。唯一成功行 `T2_EQ=PASS`。
- **run_name**：`motion-t2-cand`；`EXP_NAME=RUN_TAG=motion-t2-cand`。
- **commit**：S2 合入后的 clean HEAD（与 T1 同一 HEAD；sha 在 `result.md` 回填）。
- **口径**：与 reference 完全相同的 `bench_train_steps.py` 直跑 argv（只换 `--exp-name` / `--checkpoint-base-dir`），`BENCH_DIGEST_INTERVAL=100`、
  `BENCH_EXTRA_DIGEST_STEPS=299`、确定性档、CUDA 0,1；数据根 `<lib>/framesamp`（本机 NVMe）。
- **前置**：candidate 起跑前对 reference 再跑一次 `check_baseline_env.py check --baseline <ref records> --dataset <lib>/source` → `BASELINE_ENV=PASS`；
  T2 gate 前第三次 `check`。任何一次无 PASS 都使 reference 失效。

## 起跑命令

```bash
V1=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store; LIB=$V1/datasets/4task-motion-40ep; REC=$V1/bench/2gpu-epoch-bench/motion-t2-cand; mkdir -p $REC
jq -n '{}' > $REC/env.json.tmp && mv $REC/env.json.tmp $REC/env.json
CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 UV_LINK_MODE=copy JAX_PLATFORMS=cpu \
  uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir $REC --dataset $LIB/source
mkdir -p $V1/cache/jax/motion-t2-cand && ln -sfn $V1/cache/jax/motion-t2-cand ~/.cache/jax_motion-t2-cand
tmux new-session -d -s motion-t2-cand "set -o pipefail; cd /data/hongzefu/robomme_policy_learning_MotionJEPA; echo HEAD=\$(git rev-parse HEAD); \
  BENCH_RECORD_DIR=$REC BENCH_DIGEST_INTERVAL=100 BENCH_EXTRA_DIGEST_STEPS=299 BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 \
  CUDA_VISIBLE_DEVICES=0,1 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name motion-t2-cand --assets-base-dir $V1/train-assets \
    --checkpoint-base-dir $V1/train-runs/motion-t2-cand --batch-size 8 --num-workers 4 --num-train-steps 300 --log-interval 1 --save-interval 1 --seed 42 \
    --fsdp-devices 2 --dataset-path $LIB/framesamp --weight-loader.params-path $V1/models/openpi-assets/checkpoints/pi05_base/params \
    --model.use-history --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled 2>&1 | tee $V1/logs/motion-t2-cand.log; \
  echo \"EXIT_CODE=\$?\" >> $V1/logs/motion-t2-cand.log"
# 收官：project_scalars → g0_gate --profile t2
uv run --no-sync python scripts/training/tests/project_scalars.py $REC/metrics.jsonl $REC/scalars_hex.tsv
uv run --no-sync python scripts/training/tests/g0_gate.py --profile t2 --reference-manifest $V1/bench/2gpu-epoch-bench/motion-t2-ref/t2_reference_manifest.json \
  --run-dir-b $REC --log-b $V1/logs/motion-t2-cand.log --steps 300 --batch-size 8 --env-out <第三次 check 输出>
```
