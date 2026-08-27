# v1-g0-speed 结果

> **已被取代（2026-08-27）**：speed 链锚点换为 1000 步口径的 [`v1-g0-speed-r2`](../v1-g0-speed-r2/result.md)（P1b 后重测，用户裁定升级步数）；本 records 保留作历史参照，数字不再作对比对象。（speed 链锚点）

## 稳态统计（本机口径，不作最终吞吐结论）

- 稳态步时：中位 **1.117 s/step**（n=249，p10 1.074 / p90 1.246）；均值口径见 util 窗口墙钟 287 s / 250 步 ≈ 1.148 s/step。
- util（`nvidia-smi -lms 500`，步 50→299 窗口，n=1056 采样）：**稳态均值 86.3%、0% 采样占比 5.3%**；无 >1.5× 中位慢步，慢步分层不适用。
- epoch 外推：49,411 步/epoch ≈ 55,188 s ≈ **15.33 h**（b8、NFS turbo 数据、本机 2 卡）。
- 对照：确定性档 G0 两轮 1.964/1.978 s/step——deterministic_ops + autotune_level=0 相对生产档慢约 76%，佐证正确性/性能分跑的必要性（红线 B7）。

## 判定

- 300 步全部完成、loss 有限；speed 口径断言通过（无 param_checksums.jsonl / batch_digests.jsonl 落盘）；缓存事件：冷编译 cache_misses=2。
- 产物：`records/`（metrics / env / run_meta / util×2 + `BASELINE_MANIFEST.json`）。逐步 loss 标量保留（毫秒级墙钟），供 speed 链后续 run 对账步时分布。
