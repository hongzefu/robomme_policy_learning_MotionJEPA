# motion-t1-closed — 结果（T1：S2 合入后的关闭态 1000 步命中 G0b 黄金锚点）

- **判定：`G0_EQ=PASS`**（`records/g0_gate_t1.txt`，`g0_gate.py --profile t1`）。`scalars_hex.tsv` sha256 = `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`，
  命中 G0b 黄金锚点（继 G0b r1 / r2、v1 e2e、A21 等之后第七份同值）；`records/scalars_hex.sha256`。
- **起跑**：2026-09-03 14:05:57 → 14:56:03（50 min），HEAD `3b02f18`（commitV6.5 `06220c4` 之后、纯文档提交之前，clean），2×RTX 6000 Ada，b8，1000 步，
  `--dataset-path v1-store/datasets/4task-gl-framesamp`（symlink → turbo，NFS 介质，与 A21 同口径）、`perceptual-framesamp-context.yaml`（关闭态，resolved sha `0c28bcb2…`），
  `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`；`records/preflight.log` `BASELINE_ENV=PASS`。
- **compare_baseline（`records/compare_vs_g0b_r1.txt`，tier `t1-vs-g0b`）**：

```
SCALARS steps=1000 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
REL 五键 median/p95/max 全 0
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST rows=14 mismatch=4 first_bad_step=100 bad_keys=2 首个: ["['static_image_emb']"]   ← raw 口径，已知（commitV4.1 起规范化口径为准）
BATCH_DIGEST_CANONICAL rows=14 mismatch=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致, steps≈1000）
```

- **关闭态形制**：`batch_digests.jsonl` 首行 `n_keys=12`、`param_checksums.jsonl` `n_leaves=177`（两个 motion 模块根本不创建，plan 2.9；开启态为 16 / 193）。
- **中途抽查**：第 7 步、第 501 步两次对 G0b r1 五标量 hex 逐位相同。
- **吞吐（仅记录，NFS 介质、不作指标）**：`RESULT batch=8 稳态=1.942s/step (n=929, p10=1.846, p90=2.079)`；每 100 步的 TrainState 摘要各耗时约 89 s。
- **意外**：无。run 期间主树只提交了纯文档 commit（`e927e51`、`7ff0a17`），代码未动；S3 改码在 worktree `s2-dev` 内进行。
- **records/**：`compare_vs_g0b_r1.txt`、`g0_gate_t1.txt`、`scalars_hex.sha256`、`preflight.log`、`run_meta.json`、`env.json`、`param_checksums.jsonl`、`batch_digests.jsonl`
  （`metrics.jsonl` / `scalars_hex.tsv` / `index_sequence.json` 留在 `v1-store/bench/2gpu-epoch-bench/motion-t1-closed/`，可由 git 记录的命令复现，不归档）。
