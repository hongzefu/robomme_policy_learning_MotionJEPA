# bottleneck-bench v2：CPU/worker/RAM 侧修复 GPU 空转（v1 目录只读不动）

**要回答的问题**：v1 判定的瓶颈是 CPU 配比（8 CPU 喂 4 张 A40 不够，GPU util 均值仅 69-70%、慢步(>8s)占稳态墙钟 33%）。只动启动参数——CPU 数、`num_workers`、内存——**完全不改 dataloader 代码逻辑**，能否把 GPU 吃满、步时从 6.933 s/step 压回 ~4.8-5.2 s、epoch 从 11.9 h 压回 ~8.2-8.9 h？

**与 v1（`scripts/bottleneck-bench/gl-e2e/`）的全部差异**（其余零改动，入口仍是 `scripts/smoke-local/bench_train_steps.py`）：

| 项 | v1（v1-e2e-b64） | v2 |
|---|---|---|
| `--cpus-per-task` | 8 | **16** |
| `num_workers` | 4 | **8 / 12 / 16 三档**（`--export` 传入） |
| `--mem` | 64G（末步校验和 device_get ~28GB 被 OOM-kill） | **96G** |
| 步数 / `--time` | 300 步 / 1h | **600 步 / 2h** |
| GPU 采样 | 15s 循环单通道 | **双通道**：dense（`nvidia-smi -lms 500`，主判读）+ legacy（v1 同款 15s，对照） |
| 稳态统计 | sbatch 内嵌 python，报中位 util | `analyze_gpu_util.py`，禁止中位数做标题结论 |

**判读口径（AGENTS.md 第 16 条）**：标题结论只用稳态窗口（丢前 50 步）的 util **均值**、**0% 采样占比**、**慢步/非慢步分层均值**；中位数只进附录。v1 的教训：中位 100% 掩盖了均值 69-70%（约 1/3 采样 <100%、大量为 0%）。dense 采样 500ms 已到有效密度上限——`utilization.gpu` 本身是 NVML 约 1/6~1 秒内部周期的均值，更密只是重复读数。

## 三档 job（全部 4×A40 / b64 / 16C / 96G / 600 步，需用户逐次特批）

| run_name | workers | seed | 提交批次 |
|---|---|---|---|
| `v1-e2efix-w8c16` | 8 | 210 | 第一批（与 w16 同时） |
| `v1-e2efix-w16c16` | 16 | 211 | 第一批 |
| `v1-e2efix-w12c16` | 12 | 212 | 第二批（延后，缓解 8×A40 配额挤占） |

seed 各档互异且避开 42/200-205：防同节点先后复跑时 page cache 跨 job 污染（步时不依赖 seed，可比性不受影响）。

## 成功判据

- GPU util 均值 69-70% → **≥85%**，0% 采样占比大幅下降；
- 步时中位 6.933 s → **≤5.2 s**（compute-only 下界 4.778 s），慢步墙钟占比 33% 大幅下降；
- epoch(6,176 步) 推算落入 ~8.2-8.9 h；
- 三档对比给出正式训练最终 `num_workers`。

## 跑法

```bash
# 第一批 w8+w16（经 gl_submit.py；ControlMaster 失效时需 GLPW + GLOTP=TOTP）：
bash scripts/bottleneck-bench-v2/submit_fix_jobs.sh
# 第二批 w12（前两档入队后延后提交）：
BATCH=2 bash scripts/bottleneck-bench-v2/submit_fix_jobs.sh
# 结果复算 / 回归 v1 旧数据（缺 dense 文件时自动退回 legacy 通道并告警）：
uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py v1-store/bench/bottleneck/v1-e2efix-w8c16 --steps 600
```

## 记录文件（`v1-store/bench/bottleneck/<run_name>/`）

`metrics.jsonl`（逐步）、`gpu_util_dense.csv`（`本地时区毫秒时间戳,卡号,util%,显存MiB`）、`gpu_util.csv`（legacy，`epoch秒,卡号,util%,显存MiB`）、`nfs_read.csv`、`param_checksums.jsonl`（末步一次）、`env.json`。stdout 末尾 `RESULT …` 行给出结论数字。

按 AGENTS.md 第 12/17 条：三个 run 均从 clean HEAD 起跑，起跑与结果留档在 `docs/training-doc/<run_name>/`；结果并入 `docs/v1-nfs-bottleneck-analysis.md` 汇总。
