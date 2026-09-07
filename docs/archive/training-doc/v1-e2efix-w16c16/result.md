# v1-e2efix-w16c16 结果（16C/16w/96G，e2e 600 步）

首次提交 job 58638543 在起跑前被 scancel（与 w8 档同因：bench 入口 500 步守卫拒绝 600 步，守卫修复后统一重提）。重跑 job **58638709** @ gl1522：训练本体完整跑完 600 步、metrics 全量落盘，但**末步参数校验和阶段被 host OOM-kill**（`sacct` 终态 `OUT_OF_MEMORY`，MaxRSS 100,663,492K ≈ 96G 顶格，`EXIT_CODE=137`）——与 v1 的 64G 末步 OOM 同模式：16 个 spawn worker 常驻内存比 8w 档高出数 GB，96G 容不下末步 device_get ~28GB。吞吐结论无损（下表为本地复算）。

| 项 | 值（稳态：丢前 50 步，dense 500ms 通道，n采样=31,928） |
|---|---|
| 步时中位 | **5.327 s/step**（n=549，p10 5.198 / p90 9.065） |
| 吞吐 | **12.01 样本/s** |
| epoch(6,176 步) | **32,901 s ≈ 9.14 小时** |
| GPU util 均值 | **67.1%**（0% 采样占比 29.5%，<100% 占比 35.3%；四卡一致） |
| 慢步(>8s) | 墙钟占比 **35.0%**；慢步时段 util 均值 **24.2%**，非慢步时段 **90.2%** |
| 显存峰值 | 43,757 MiB |
| NFS 稳态实测 | 123 MB/s（公式口径 225 MB/s） |
| legacy 15s 对照 | 67.7%，dense−legacy = −0.6pp |

**判读**：与 w8 档几乎持平（5.327 vs 5.301 s/step；util 均值 67.1% vs 71.2%，差值在两 job 停顿相位的噪声量级）——**workers 8→16 无增益**，残余的整段长停顿与 worker 数无关；非慢步时段 util 同样到 90%，佐证 16 CPU 已解决供给面。RAM 教训：16 workers 档 96G 不够末步校验和，若后续复跑此档需 ≥112G 或免跑末步校验和。综合归因见 `docs/v1-nfs-bottleneck-analysis.md`。

产物：`records/`（env.json / metrics.jsonl / gpu_util.csv / gpu_util_dense.csv / nfs_read.csv；param_checksums.jsonl 因 OOM 未产出）；日志 `v1-store/logs/v1-e2efix-w16c16-58638709.log`；run 目录与节点 jax 缓存已按脚本自动清理。复算命令：`uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py docs/training-doc/v1-e2efix-w16c16/records --steps 600`。
