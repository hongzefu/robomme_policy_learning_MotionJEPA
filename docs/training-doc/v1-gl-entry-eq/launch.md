# v1-gl-entry-eq 起跑档（D2：集群 A40 上游对拍 + wandb 出网探针）

> 计划：`v5.1-prod-60k-wandb-plan.md`（第二版）D2 节。用户 2026-08-30 授权直接提交；
> 按用户改令，本 job 在 v1-prod-60k **之后**提交（抢时间保证 60k 先入队），其「放行闸」
> 语义改为事后核验：judge FAIL 时处置回用户拍板，不自行动 60k。

## 可复现锚

- 代码切片：commitV5.1 `baeaee7`（entry_equiv.py judge 硬化 + 新增
  `scripts/training/tests/gl_entry_eq.sbatch`）；EXPECTED_GIT_HEAD 为本 launch 档提交
  后的 HEAD（与 v1-prod-60k 同值），sbatch 起跑闸断言仓库 HEAD + porcelain。
- A 侧：上游 main `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，worktree
  `v1-store/entryeq/worktree-main`（独立 NFS venv：`uv venv --python
  /nfs/turbo/.../uv-python/cpython-3.11.14.../python3.11` + `uv sync --frozen`，
  `readlink -f .venv/bin/python` 落 `/nfs/turbo/` 实测 ✓，构建 EXIT_CODE=0），legacy 库
  `4task-gl`，`--overwrite`（tentative 12 步 + 正式 1000 步）。
- B 侧：新入口 `scripts/training/train.py`，packed 库 `4task-gl-framesamp`，无
  `--overwrite`，`--forbid-root <worktree>`（堵 B 误载 A 模块）。
- 共同：b64 / fsdp4 / workers 8 / seed 42 / log-interval 1 / save-interval 100 /
  1000 步；确定性档 `XLA_FLAGS='--xla_gpu_deterministic_ops=true
  --xla_gpu_autotune_level=0'`（⚠ A40 首测）；两侧均 `--no-wandb-enabled` 对拍纯净。

## 提交命令（钉死口径，实际执行实录落 records/）

```bash
uv run --no-project --with pexpect python scripts/dataset/gl/gl_submit.py \
  "cd $REPO && sbatch --job-name=v1-gl-entry-eq \
     --export=ALL,EXPECTED_GIT_HEAD=<launch 后 HEAD>,WANDB_PROJECT=robomme-framesamp,WANDB_ENTITY=hongzefu-university-of-michigan,WANDB_API_KEY=<key 不落档> \
     scripts/training/tests/gl_entry_eq.sbatch"
```

资源：spgpu 4×A40 / 16C / 128G / 5h ≈ 20 GPU·h（用户已放行）。

## 判定行（judge 硬判，全部进退出码；A40 无本机锚点 → anchor=SKIPPED）

```
A_SIDE_SEGMENTS tentative_rows=12 main_rows=1000
B_SIDE_SEGMENTS tentative_rows=0 main_rows=1000
ENTRY_SCALARS steps=1000 keys=5 hex_mismatch=0        # 步集合恰 0..999
ENTRY_STATE_DIGEST rows=10 mismatch=0                 # 步集合恰 {100..900,999}，无重复
ENTRY_RESOLVED_CFG mismatch=0 whitelist=4
ENTRY_PROVENANCE=PASS                                 # HEAD 断言 + porcelain 空 + forbid-root
WANDB_PROBE=PASS                                      # 真代码探针，probe run 跑完即删
ENTRY_EQ=PASS
```

FAIL 处置：`ENTRY_SCALARS` 失配 → 先同侧（B）重跑一次甄别 A40 确定性档硬件不确定性，
仍失配再查入口；不得直接判「入口改坏」。任何 FAIL 均回用户拍板 60k 处置。

## 收官清理（PASS 后）

worktree（`git worktree remove` + prune）、`v1-store/entryeq/`、集群侧
`~/.cache/jax_gl-entry-eq-*` 软链。
