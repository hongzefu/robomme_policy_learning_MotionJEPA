# v1-e2efix-w8c16 结果（16C/8w/96G，e2e 600 步）

job 58638542 起跑即被 bench 入口 500 步守卫拒绝（守卫已在 fix commit 提至 600）；重跑 job **58638708** @ gl1521，`EXIT_CODE=0` 完整跑完，末步参数校验和已落盘。首次提交与重跑之间无代码差异（仅守卫上限一行）。

| 项 | 值（稳态：丢前 50 步，dense 500ms 通道，n采样=31,674） |
|---|---|
| 步时中位 | **5.301 s/step**（n=549，p10 5.196 / p90 16.566） |
| 吞吐 | **12.07 样本/s**（v1 基线 9.23） |
| epoch(6,176 步) | **32,739 s ≈ 9.09 小时**（v1 基线 11.89 h） |
| GPU util 均值 | **71.2%**（0% 采样占比 25.3%，<100% 占比 30.8%；四卡一致） |
| 慢步(>8s) | 墙钟占比 **32.0%**；慢步时段 util 均值 **32.6%**，非慢步时段 **89.4%** |
| 显存峰值 | 43,757 MiB（贴 0.95 预算，正常） |
| NFS 稳态实测 | 108 MB/s（公式口径 226 MB/s） |
| legacy 15s 对照 | 71.9%，dense−legacy = −0.7pp（v1 的 15s 采样无系统偏差，误判源纯为中位数统计） |

**对 v1-e2e-b64（8C/4w/64G）的判读**：CPU 8→16 + workers 4→8 把「普遍性小等待」修掉了——非慢步时段 util 从 79% 升到 89.4%、步时中位从 6.933 压到 5.301 s（逼近 compute-only 下界 4.778 s）；但 **GPU 仍未吃满**：util 均值仅 71.2%（基线 69.7%），慢步墙钟占比仍 32%，且尾部更极端（p90 8.54→16.57 s）——残余瓶颈是**少量但更长的整段停顿**，停顿时段 NFS 供数掉底（实测 108 MB/s ≪ 公式 226 MB/s）。与 w16 档（同结果）互证：停顿与 worker 数无关，不是 CPU 配比问题，详见 `docs/v1-nfs-bottleneck-analysis.md`。

产物：`records/`（env.json / metrics.jsonl / gpu_util.csv / gpu_util_dense.csv / nfs_read.csv / param_checksums.jsonl）；日志 `v1-store/logs/v1-e2efix-w8c16-58638708.log`；run 目录与节点 jax 缓存已按脚本自动清理。复算命令：`uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py docs/training-doc/v1-e2efix-w8c16/records --steps 600`。
