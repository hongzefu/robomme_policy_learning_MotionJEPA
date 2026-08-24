# v1-gl-dlbench 起跑留档（实验 1：GL dataloader-only 吞吐）

按 AGENTS.md 第 12 条。目的与判据链见 `scripts/bottleneck-bench/README.md`（供给侧权威测量，第 13 条正式口径）。

- **commit**：本文件所在提交即起跑 commit（clean HEAD），见 git log。
- **命令**：`uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-dataloader/gl_dataloader_bench.sbatch"`（jobid 见 result.md）
- **配置**：batch 64、workers 扫 4/8/16（各档 seed=42+w）、warmup 5 批 + 计时 40 批/档、framesamp-context、不训练不落权重；SLURM：spgpu/chaijy2、1×A40、18 CPU、48G、30 min（debug 包络内）。
- **数据来源**：`v1-store/datasets/4task-gl`（GL 构建全量库）；norm stats `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`。
- **输出路径**：记录 `v1-store/bench/bottleneck/v1-gl-dlbench/`（batches.jsonl / summary.jsonl / env.json）；日志 `v1-store/logs/v1-gl-dlbench-<jobid>.log`（NFS 双端可见）。
- **结果**：见同目录 `result.md`。

## 拆分轮（2026-08-24 追加）

原 18C 单 job（58619536）排队过久，按用户指示**保留排队不取消**，另拆 6 个独立单档 job 并行排（矩阵 workers × CPU 配比，含 w16c10 超订档；模板 `gl_dlbench_single.sbatch`，驱动 `submit_split_jobs.sh`；全部 1×A40/30min，debug 包络内）：

| job-name | workers | cpus | mem | seed |
|---|---|---|---|---|
| v1-dlb-w4c6 | 4 | 6 | 24G | 200 |
| v1-dlb-w4c10 | 4 | 10 | 24G | 201 |
| v1-dlb-w8c10 | 8 | 10 | 32G | 202 |
| v1-dlb-w8c18 | 8 | 18 | 32G | 203 |
| v1-dlb-w16c10（超订） | 16 | 10 | 48G | 204 |
| v1-dlb-w16c18 | 16 | 18 | 48G | 205 |

各 job 记录目录 `v1-store/bench/bottleneck/v1-dlb-<tag>/`；jobid 见 result.md。
