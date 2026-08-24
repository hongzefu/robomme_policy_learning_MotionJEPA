# v1-coldcache-b8 起跑留档（实验 2：本机冷缓存复测）

按 AGENTS.md 第 12 条。目的：v1-2gpu-epoch-bench-b8 的 1.060 s/step 受 page cache 污染（同 seed 重跑 + 本机 377 GB 内存），换 seed=123 取冷样本复测并采遥测，作本机口径旁证（第 13 条：不作正式吞吐结论）。

- **commit**：本文件所在提交即起跑 commit（clean HEAD），见 git log。
- **命令**：`tmux new-session -d -s coldcache "… bash scripts/bottleneck-bench/local-coldcache/run_local_coldcache.sh …"`（完整模板见 `scripts/bottleneck-bench/README.md`）。
- **配置**：batch 8、workers 4、2×RTX 6000 Ada、150 步、seed 123、save-interval 1000（无中途校验和）、framesamp-context、不落权重；遥测每 5 s：nvidia-smi 利用率 + turbo mountstats `server_read`。
- **数据来源**：同 v1-gl-dlbench（`v1-store/datasets/4task-gl` + 既有 norm stats）。
- **输出路径**：记录 `v1-store/bench/bottleneck/v1-coldcache-b8/`（metrics.jsonl / gpu_util.csv / nfs_read.csv / env.json）；日志 `v1-store/logs/v1-coldcache-b8.log` 与 `-driver.log`。
- **结果**：见同目录 `result.md`。
