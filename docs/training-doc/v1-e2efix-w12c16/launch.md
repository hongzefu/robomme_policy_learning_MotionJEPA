# v1-e2efix-w12c16 起跑留档（bottleneck-bench v2：16C/12w/96G 修复验证，第二批）

按 AGENTS.md 第 12/17 条。目的：三档矩阵的中间档（12 workers），与第一批 `v1-e2efix-w8c16` / `v1-e2efix-w16c16` 对比补全 workers 曲线，确定正式训练的最终 `num_workers`；判据与三档矩阵见 `scripts/bottleneck-bench-v2/README.md`。按用户要求晚于第一批提交、排在其后。

- **特批**：4 GPU + 2h 超出 greatlakes.md debug 包络，已由用户 2026-08-24 经计划批准逐次特批（--gpus-per-node=4 --cpus-per-task=16 --mem=96G --time=02:00:00，三档各一）。
- **commit**：本文件所在提交即起跑 commit（clean HEAD），见 git log。
- **命令**：`BATCH=2 bash scripts/bottleneck-bench-v2/submit_fix_jobs.sh`（经 `gl_submit.py`；jobid 见 result.md）
- **配置**：与 v1-e2e-b64 的差异仅启动参数与遥测——cpus 16、num-workers 12、mem 96G、600 步截断、seed 212（防 page cache 跨 job 污染）、GPU 双通道采样（dense `nvidia-smi -lms 500` 主判读 + legacy 15s 对照）；其余同 v1：batch 64、fsdp_devices 4、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、framesamp-context、log-interval 1、save-interval 1000（参数校验和仅末步一次）、wandb 关，入口零改动复用 `scripts/smoke-local/bench_train_steps.py`（步数守卫已在 fix commit 提至 600）。
- **数据来源**：`v1-store/datasets/4task-gl` + 既有 norm stats；pi05_base 初始化权重。
- **输出路径**：记录 `v1-store/bench/bottleneck/v1-e2efix-w12c16/`（metrics.jsonl / param_checksums.jsonl / gpu_util_dense.csv / gpu_util.csv / nfs_read.csv / env.json）；日志 `v1-store/logs/v1-e2efix-w12c16-<jobid>.log`；run 目录与节点 jax 缓存跑完即删。
- **结果**：见同目录 `result.md`。
