# records 归档说明（v1-prod-trend-10h）

## 与既有 bench 留档的差别：dense CSV 是**降采样版**

此前所有 bench 记录（`docs/training-doc/v1-framesamp-e2e/records/` 等）都是从
`v1-store/bench/bottleneck/` **逐字节原样拷贝**——因为 600 步的 run 只产出约 1 MB dense CSV。

本 run 跑 7000 步约 9.5 小时，dense 通道（4 卡 × 2 Hz）产出约 **12 MB / 29 万行**。
AGENTS 12 要求把 git 无法还原的记录归档进本目录（进 git），AGENTS 14 又要求大产物收敛到
不进 git 的 `v1-store/`——两条在长训尺度上冲突。**处置（用户 2026-08-28 拍板「dense 采样
文件降采样归档」）**：

| 文件 | 本目录内 | 全分辨率原件 |
|---|---|---|
| `gpu_util_dense.ds10.csv` | **每卡每 10 个采样保留 1 个**（500 ms → 5 s），约 1.2 MB | `v1-store/train-records/v1-prod-trend-10h/gpu_util_dense.csv` |
| 其余四路（legacy / nfs_read / meminfo / compute_apps） | 原样（15 s 采样，本就很小） | 同上目录 |

**判读一律以全分辨率原件为准。** 降采样版只供归档与人工翻阅——5 秒间隔对 4.7 秒的步时
不满足 AGENTS 16「采样间隔必须显著小于步时」的要求，**不得用它复算任何 util 结论**。
`result.md` 记录了原件绝对路径与降采样倍率；降采样由
`scripts/train-prod/downsample_util_csv.py` 完成（按卡分组抽稀，各卡采样点数保持一致、
首末样本恒保留以保住时间跨度）。

## 文件清单

| 文件 | 内容 |
|---|---|
| `env.json` | run 口径与 provenance；analyzer 从中读 `log_interval` / `epoch_steps` / `batch_size` / `num_workers` |
| `metrics.jsonl` | 每 100 步一行（`log_interval=100` 的区间均值），含 loss / grad_norm / llm_grad_norm / mem_enc_norm / param_norm 的十进制与 IEEE hex 双精度 + `wall_time` |
| `run_meta.json` | 真实 argv、jax 编译缓存事件计数、入口标识 |
| `gpu_util_dense.ds10.csv` | dense GPU util（降采样 10×），列：`时间戳,卡号,util%,显存MiB` |
| `gpu_util.csv` | legacy 15 s 通道，列：`epoch秒,卡号,util%,显存MiB` |
| `nfs_read.csv` | NFS 累计读字节，列：`epoch秒,normal_read,server_read` |
| `meminfo.csv` | 列：`epoch秒,Cached(kB),pgmajfault,anon` —— 比 bench 版多一列 `anon`（cgroup 真实不可回收内存，128G 档位复核用） |
| `compute_apps.csv` | CUDA 进程存证，列：`epoch秒,pid,显存MiB` |

**不归档**：checkpoint 权重（AGENTS 12 明禁；约 13 GB/份，留在
`v1-store/train-runs/mme_vla_suite/v1-prod-trend-10h/`）。
