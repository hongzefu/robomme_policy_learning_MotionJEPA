# v1-computeonly-b64 起跑留档（实验 3：4×A40 compute-only 步时）

按 AGENTS.md 第 12 条。目的：测官方口径（4 卡、batch 64）的纯计算 s/step → NFS 需求 = 1.20 GB/步 ÷ 步时（需求侧，判据链见 `scripts/bottleneck-bench/README.md`）。

- **特批**：4 GPU 超出 greatlakes.md debug 包络（≤2 GPU），已由用户 2026-08-24 逐次批准（--gpus-per-node=4 --cpus-per-task=8 --mem=64G --time=00:25:00）。
- **commit**：本文件所在提交即起跑 commit（clean HEAD），见 git log。
- **命令**：`uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-compute-only/gl_compute_only.sbatch"`（jobid 见 result.md）
- **配置**：batch 64、fsdp_devices 4、100 步、log-interval 1、seed 42、framesamp-context；`create_data_loader` 被换成「首 batch 缓存后无限重复」wrapper（首 batch 真读一次 NFS ≈1.2 GB，稳态丢前 20 步剔除）；`save_state` no-op，不落权重不做校验和。
- **数据来源**：同上（首 batch 来自 `v1-store/datasets/4task-gl`，pi05_base 初始化权重）。
- **输出路径**：记录 `v1-store/bench/bottleneck/v1-computeonly-b64/`（metrics.jsonl / gpu_util.csv / save_state_calls.jsonl / env.json）；日志 `v1-store/logs/v1-computeonly-b64-<jobid>.log`。
- **结果**：见同目录 `result.md`。
