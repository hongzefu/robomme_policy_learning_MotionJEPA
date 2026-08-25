# v1-e2efix-w12c16 结果（16C/12w/96G，e2e 600 步）

job **58638741** @ gl1521（w8 档刚跑完的同一节点，seed 212 与 w8 的 210 互异、防 page cache 跨 job 污染），`EXIT_CODE=0` 完整跑完，末步参数校验和落盘（96G 撑住了 12 workers，对照 16 workers 档 96G OOM）。

| 项 | 值（稳态：丢前 50 步，dense 500ms 通道） |
|---|---|
| 步时中位 | **5.319 s/step**（n=549，p10 5.195 / p90 10.825） |
| 吞吐 | **12.03 样本/s** |
| epoch(6,176 步) | **32,852 s ≈ 9.13 小时** |
| GPU util 均值 | **70.6%**（0% 采样占比 25.7%，<100% 占比 31.7%；四卡一致） |
| 慢步(>8s) | 墙钟占比 **36.4%**；慢步时段 util 均值 **32.0%**，非慢步时段 **92.8%** |
| 显存峰值 | 43,757 MiB |
| NFS 稳态实测 | 106 MB/s（公式口径 226 MB/s） |
| legacy 15s 对照 | 71.1%，dense−legacy = −0.5pp |

**判读**：与 w8（5.301 s / 71.2%）、w16（5.327 s / 67.1%）三档打平——**workers 8/12/16 曲线完全平坦**，16 CPU 下步时中位稳定在 ~5.3 s、util 均值 ~67-71%、慢步墙钟 ~32-36%。非慢步时段 util 92.8% 为三档最高，进一步佐证供给面已被 16 CPU 解决；残余整段长停顿与 worker 数无关。综合归因与正式训练参数建议见 `docs/v1-nfs-bottleneck-analysis.md`。

产物：`records/` 全量（含 param_checksums.jsonl）；日志 `v1-store/logs/v1-e2efix-w12c16-58638741.log`；run 目录与节点 jax 缓存已按脚本自动清理。复算命令：`uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py docs/training-doc/v1-e2efix-w12c16/records --steps 600`。
