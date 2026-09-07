# v1-coldcache-b8 结果留档（实验 2：本机冷缓存复测）

起跑 commit `16bd8b8`（clean HEAD），2026-08-24 完成，`EXIT_CODE=0`，150 步全过、loss 全有限。

## 核心结果

| 项 | 值 |
|---|---|
| 冷缓存稳态（seed 123，丢前 50 步，n=99 中位） | **1.148 s/step**（p10 1.026 / p90 1.486） |
| vs 热缓存基线（v1-2gpu-epoch-bench-b8，seed 42） | **1.08×**——未显著变慢 |
| 稳态窗口 GPU 利用率（两卡合并中位） | **99%** |
| 稳态窗口 NFS 真实网络读（mountstats `server_read`） | **86 MB/s** |
| 公式需求口径（8 × 18.7 MB ÷ 1.148 s） | 130 MB/s |

## 判定与解释

- **本机口径（2 卡、batch 8、workers 4）下计算侧主导**：GPU 打满 99%，冷缓存只慢 8%，4 worker 预取足以掩盖 IO。此前对 1.060 s/step「可能严重受 page cache 污染」的担忧被证伪到只有 ~8% 的量级。
- **实测 86 MB/s < 公式 130 MB/s 的原因**：同一 episode 相邻样本的 frame-sampling 索引高度重叠（`even_sampling_indices` 前 32 步逐步累积、之后 linspace 缓变），重复的 token_emb 文件被客户端缓存吃掉——**18.7 MB/样本的公式是「无重叠上界」，真实网络需求略低**。这对 4 卡 b64 的需求估计同样成立（偏保守方向，利好）。
- p90 1.486 的尾部慢步与 NFS 延迟抖动一致，但未影响中位。
- ⚠ 本机数字按 AGENTS.md 第 13 条只作旁证；对目标问题（4×A40 b64）的最终判定以实验 1（供给）与实验 3（需求）为准。

## 产物位置

- 记录（git 归档）：本目录 `records/`（`metrics.jsonl` 150 行、`gpu_util.csv`、`nfs_read.csv`、`env.json`、`param_checksums.jsonl` 仅末步 1 行）；`v1-store/bench/bottleneck/v1-coldcache-b8/` 为工作副本
- 日志：`v1-store/logs/v1-coldcache-b8.log`、`v1-coldcache-b8-driver.log`
- run 目录与 `~/.cache/jax_v1-coldcache-b8`：已自动清理，无权重产物
