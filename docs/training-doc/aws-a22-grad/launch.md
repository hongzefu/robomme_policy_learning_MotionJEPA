# aws-a22-grad（A22 式：S2_BASE 旧码 vs HEAD 单步定点梯度两侧互核）launch

> 环境 B（AWS 8×A100），2026-09-04。原 A22 是把 HEAD 的 `single_step_grad.py` 产物与环境 A 固化基线 `v1-dtype-p5-grad`（Ada 卡产出）逐叶比 sha；
> 本机没有该基线且 A100 与 Ada 不逐位，改为**同机两侧互核**：`S2_BASE = c5925d9` 旧码（worktree `v1-store/worktrees/s2-base`，`PYTHONPATH=<worktree>/src`）与 HEAD 各跑一次
> `single_step_grad.py`，`compare_grad_summaries.py` 逐 kind 比 batch 索引、loss `float.hex()`、逐叶梯度 sha256（全部零容差）。

- **fixture 数据**：`_common.build_fixture_indices` 的 `PER_STEP=200` 要求每个 step_idx ≥ 200 个候选（短样本档只能来自 `exec_start_idx==0` 的 Button 系 episode），
  40 ep 库只有 20 个 Button episode → 结构性不足；改用 **400 ep 完整库**（`v1-store/datasets/4task-motion-400ep/framesamp`，200 个 Button episode 恰好够）。
  清单经 `DTYPE_MANIFEST=$LIB4/meta/episode_manifest.json`（commitV6.12 加的覆盖；旧码侧没有该变量，把同一份清单复制到 `<worktree>/v1-store/episode_manifest.json`——
  旧码按 `REPO_ROOT/v1-store/episode_manifest.json` 读，worktree 内 `v1-store/` 是未跟踪目录）。
- **口径**：closed YAML `perceptual-framesamp-context.yaml`、b8、seed 42、fsdp 2、`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`，
  **两侧同两张卡 GPU6,7**（先 HEAD 后旧码，串行），`DTYPE_BASELINE_CHECKSUMS=` 置空（跳过对 G0b step-0 的同源核——那份基线在环境 A），不落梯度数组（只摘要）。
- **为什么直调 `single_step_grad.py` 而不经 `run_dtype_grad.sh`**：驱动脚本写死 `CUDA_VISIBLE_DEVICES=0,1`（当时 0–5 被 400 ep Wan 抽取占用）、要求 `git status --porcelain` 为空
  （当时有两个未提交的**文档**改动 `external-assets-lock.md` / `scripts/dataset/README.md`，代码零改动）、且 source 的旧 `paths.sh` 在 worktree 内前缀断言不成立；
  按计划「改为直调并在留档写明」。直调复刻了驱动脚本的全部 env（`DTYPE_GRAD_DIR / DTYPE_BATCH_FIXTURE_DIR / DTYPE_GRAD_ARRAYS_DIR= / DTYPE_GRAD_KINDS= / DTYPE_BASELINE_CHECKSUMS=`、
  jax 编译缓存软链、`--exp-name/--assets-base-dir/--checkpoint-base-dir/…` argv）。

```bash
# side=head：cd 主树；side=base：cd worktree 且 PYTHONPATH=<worktree>/src UV_PROJECT_ENVIRONMENT=<主树>/.venv
DTYPE_GRAD_DIR=$V1/dtype-unify/aws-a22-grad-<side> DTYPE_BATCH_FIXTURE_DIR=$V1/fixtures/aws-a22-grad-<side> DTYPE_GRAD_ARRAYS_DIR= DTYPE_GRAD_KINDS= DTYPE_BASELINE_CHECKSUMS= \
DTYPE_MANIFEST=$LIB4/meta/episode_manifest.json CUDA_VISIBLE_DEVICES=6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
uv run --no-sync python scripts/training/tests/single_step_grad.py mme_vla_suite --exp-name aws-a22-grad-<side> --assets-base-dir $V1/train-assets --checkpoint-base-dir $V1/train-runs/aws-a22-grad-<side> \
  --batch-size 8 --num-workers 4 --fsdp-devices 2 --seed 42 --dataset-path $LIB4/framesamp --weight-loader.params-path $V1/models/openpi-assets/checkpoints/pi05_base/params \
  --model.use-history --model.history-config perceptual-framesamp-context.yaml --no-wandb-enabled
uv run --no-sync python scripts/training/tests/compare_grad_summaries.py $V1/dtype-unify/aws-a22-grad-base/grad_summary.json $V1/dtype-unify/aws-a22-grad-head/grad_summary.json
```

判据：两侧 `GRAD_DONE kinds=3`；`GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0`。
