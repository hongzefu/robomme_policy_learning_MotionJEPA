# v1-computeonly-b64 结果留档（实验 3：4×A40 纯计算步时）

起跑 commit `16bd8b8`，job 58619547（gl1505），2026-08-24 完成，`EXIT_CODE=0`，100 步全过。

| 项 | 值 |
|---|---|
| compute-only 稳态（丢前 20 步，n=79 中位） | **4.778 s/step**（p10 4.770 / p90 4.785，分布极紧） |
| 换算吞吐 | 13.4 样本/s |
| **NFS 供给需求** | **1.20 GB ÷ 4.778 s ≈ 251 MB/s** |
| 纯计算下限 epoch（6,176 步） | **8.20 小时** |
| 首 batch NFS 冷读（4 worker） | 1.2 GB / 86.3 s ≈ 14 MB/s（含 worker spawn 起步，仅作参考） |

机制：`create_data_loader` 被换成「首 batch 缓存后无限重复」wrapper，训练循环零改动；`save_state` no-op、无权重落盘。步时分布极紧（p90-p10 = 0.015 s）证明纯计算无抖动，是干净的需求侧基准。

产物：本目录 `records/`（metrics.jsonl 100 行 / gpu_util.csv / save_state_calls.jsonl / env.json，git 归档）；工作副本 `v1-store/bench/bottleneck/v1-computeonly-b64/`；日志 `v1-store/logs/v1-computeonly-b64-58619547.log`。
