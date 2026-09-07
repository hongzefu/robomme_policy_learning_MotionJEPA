# v1-e2e-b64 起跑留档（实验 4：4×A40 官方口径 b64 端到端真实吞吐）

按 AGENTS.md 第 12 条。目的：官方口径（4 卡、全局 batch 64、workers 4）端到端 300 步实测吞吐，作为判据链的实测校验项——实测稳态步时应 ≈ max(实验 3 compute-only 步时, 1.20 GB ÷ 实验 1 供给 MB/s)，并给出 epoch(6,176 步) 实测时长。

- **特批**：4 GPU + 1h 超出 greatlakes.md debug 包络，已由用户 2026-08-24 逐次批准（--gpus-per-node=4 --cpus-per-task=8 --mem=64G --time=01:00:00）。
- **commit**：本文件所在提交即起跑 commit（clean HEAD），见 git log。
- **命令**：`uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py "sbatch --parsable scripts/bottleneck-bench/gl-e2e/gl_e2e_b64.sbatch"`（jobid 见 result.md）
- **配置**：与官方 `finetune_mme_vla_suite.sh` 一致的部分——batch 64、fsdp_devices 4、num-workers 4、seed 42、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、framesamp-context、真 dataloader（GL 全量库）+ 真训练循环；必要差异仅——300 步截断、log-interval 1（逐步记录不进计算图）、save-interval 1000（参数校验和仅末步一次，不落权重）、wandb 关、路径显式化。入口零改动复用 `scripts/smoke-local/bench_train_steps.py`。
- **数据来源**：`v1-store/datasets/4task-gl` + 既有 norm stats；pi05_base 初始化权重。
- **输出路径**：记录 `v1-store/bench/bottleneck/v1-e2e-b64/`（metrics.jsonl / param_checksums.jsonl / gpu_util.csv / nfs_read.csv / env.json）；日志 `v1-store/logs/v1-e2e-b64-<jobid>.log`；run 目录与节点 jax 缓存跑完即删。
- **结果**：见同目录 `result.md`。
