# motion-a21-g0b-replay result — G0_EQ=PASS，scalars sha256 命中黄金锚点（第六份同值）

- **起跑 HEAD**：`c5925d96305f771058e2206ae89461269af9d97c`（commitV6.4；porcelain 空，`env.json.git_dirty=false`）。
  前两次起跑失败留档：`v1-store/bench/2gpu-epoch-bench/motion-a21-g0b-replay.failed-1`（HEAD `1dd42d2`，
  `download.maybe_download` 在 symlink 资产上 `relative_to` 崩，`fix:` `7ec7e49` 修）；第二次在 `7ec7e49` 上起跑时工作区含未跟踪的
  open YAML（`git_dirty` 会记 true），30 s 内主动 kill、清残档、提交 commitV6.4 后从 clean HEAD 第三次起跑（本 result 即此次）。
- **preflight**：`BASELINE_ENV=PASS`（对 `v1-grad-baseline-g0b/records/r1`，带 `CUDA_VISIBLE_DEVICES=0,1` / `XLA_FLAGS` 确定性档 /
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`，`--steps 1000 --batch-size 8`）；`records/preflight.log`。
- **执行**：2026-09-03 13:0x–14:2x，2×RTX 6000 Ada，b8，1000 步，`DATASET_PATH=v1-store/datasets/4task-gl-framesamp`（symlink → turbo，NFS 介质）；
  jax 编译缓存事件 `compile_requests_use_cache=14`（冷编译）；`BENCH_PASS`；驱动日志 `v1-store/logs/motion-a21-g0b-replay-driver.log`。

## 判定行（`records/compare_vs_g0b_r1.txt`、`records/g0_gate.txt`）

```
SCALARS steps=1000 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST rows=14 mismatch=4 first_bad_step=100 bad_keys=2 首个: ["['static_image_emb']"]   ← 已知预期失配（dtype 统一口径差），不作判据
BATCH_DIGEST_CANONICAL rows=14 mismatch=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致, steps≈1000）
PROJECTED rows=1000 sha256=c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757
G0_EQ=PASS
```

`sha256(records/scalars_hex.tsv) = c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`，与 G0b r1 / r2 / G1 / G2 / G3 同值——第六份。
中途抽查：前 70 步五标量 hex 与 G0b r1 逐位相同（起跑 3 min 时人工核）。

## 结论

- 基线未腐烂、环境未漂移；S1 的 `scripts/dataset/` 重构与新增 `datastore/motion_store.py`（无人 import）、`download.py` 路径修补
  对训练语义零影响（`AGENTS.md` 第 18 条第二块对 S1 的收尾检验）。
- 本 commit 即 T2 reference 与 S2 改码前的 `S2_BASE`：`c5925d9`。
- 性能数字（稳态 1.938 s/step、epoch 估算 26.6 h）仅留档：确定性档 + 摘要 + NFS 介质，红线 B7 禁作性能结论。
