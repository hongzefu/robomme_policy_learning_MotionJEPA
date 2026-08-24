# v1-gl-dlbench 起跑留档（实验 1：GL dataloader-only 吞吐）

按 AGENTS.md 第 12 条。目的与判据链见 `scripts/bottleneck-bench/README.md`（供给侧权威测量，第 13 条正式口径）。

- **commit**：本文件所在提交即起跑 commit（clean HEAD），见 git log。
- **命令**：`uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-dataloader/gl_dataloader_bench.sbatch"`（jobid 见 result.md）
- **配置**：batch 64、workers 扫 4/8/16（各档 seed=42+w）、warmup 5 批 + 计时 40 批/档、framesamp-context、不训练不落权重；SLURM：spgpu/chaijy2、1×A40、18 CPU、48G、30 min（debug 包络内）。
- **数据来源**：`v1-store/datasets/4task-gl`（GL 构建全量库）；norm stats `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`。
- **输出路径**：记录 `v1-store/bench/bottleneck/v1-gl-dlbench/`（batches.jsonl / summary.jsonl / env.json）；日志 `v1-store/logs/v1-gl-dlbench-<jobid>.log`（NFS 双端可见）。
- **结果**：见同目录 `result.md`。
