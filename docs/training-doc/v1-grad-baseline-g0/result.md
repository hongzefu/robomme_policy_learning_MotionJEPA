# v1-grad-baseline-g0 结果

> **已被取代（2026-08-27）**：records 已删除（新旧逐位对拍 PASS 后，用户裁定），本文数字留作历史存证；现行基线见 [`v1-grad-baseline-g0b`](../v1-grad-baseline-g0b/result.md)。

## 自证判定（round1 vs round2，`compare_baseline.py`）

```
DET_CHECK=PASS tier=g0 steps=300 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
```

- 300 步五标量 hex、4 次完整 TrainState 摘要（177 叶子）、6 次输入摘要**全部逐位一致**。
- 两轮 `scalars_hex.tsv` sha256 相同：`5da1a1c6186056598a09d6ceef13e79f751602254b802e95e9b7b40b379f1293`——「两轮一致」闭合为一次哈希比较。
- 编译缓存：round1 冷编译（cache_misses=2）→ round2 全命中（cache_hits=2、零 miss），与 P2 的 D2/D2-cold 结论自洽。

## 运行统计（仅留档参考，禁作性能结论——红线 B7）

| 轮 | 稳态步时中位 | util 稳态均值（非摘要段） | 0% 采样占比 | 摘要段 util 均值 | 摘要单次耗时 |
|---|---|---|---|---|---|
| round1 | 1.964 s/step (n=245, p10 1.872 / p90 2.010) | 78.0% | 13.3% | 78.0% | ~90 s ×4 |
| round2 | 1.978 s/step (n=245) | 67.9% | 21.7% | 83.3% | ~90 s ×4 |

- 采样：`nvidia-smi -lms 500`（原始 csv 在 `records/round{1,2}/util-lms500.csv`，15 s legacy 通道并存）；慢步分层：两轮稳态窗口内均无 >1.5× 中位的慢步。
- epoch 外推（同口径 26.95 h / 27.15 h）为确定性档数字，被摘要停顿与 deterministic flags 双重污染，禁止引用；生产口径见 `v1-g0-speed`（1.117 s/step、15.33 h）。

## 产物与缓存处置

- 固化产物：`records/round{1,2}/`（metrics / param_checksums / batch_digests / scalars_hex.tsv / env / run_meta / util×2 + `BASELINE_MANIFEST.json`）。引用前必过 `check_baseline_env.py check`（`BASELINE_ENV=PASS` 硬前置）。
- 编译缓存目录已按三支处置第一支清理，sha256 清单留证：`jax-cache-sha256-manifest.txt`（73 条）。
