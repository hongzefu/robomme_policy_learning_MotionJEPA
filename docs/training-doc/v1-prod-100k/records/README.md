# records 归档说明（v1-prod-100k）

归档集合与 `docs/training-doc/v1-prod-trend-10h/records/` 先例相同（AGENTS 12：只归档 git 无法
还原的日志 / 指标 / 结果；AGENTS 14：大产物留在 `v1-store/`）。**判读一律以全分辨率原件为准**，
原件目录：`v1-store/train-records/v1-prod-100k/`。

## 与先例的一处差别：dense 通道降采样倍率 10 → 100

先例 run 跑 7000 步（9.5 h），dense 通道 `ds10` 约 1.2 MB。本 run 跑 100k 步（134.7 h），
dense 原件 **151 MB / 387.9 万行**，`ds10` 仍有 15.1 MB——进 git 历史不可逆，故本目录改存
**`gpu_util_dense.ds100.csv`**（每卡每 100 个采样留 1 个，500 ms → 50 s，1.51 MB / 38,796 行），
由同一脚本 `scripts/training/util/downsample_util_csv.py --every 100` 生成（按卡分组抽稀、首末
样本恒保留）。`ds10` 版本保留在原件目录（sbatch 收尾自动生成）。50 s 间隔远大于 4.85 s 步时，
**不得用它复算任何 util 结论**（AGENTS 16），只供归档与人工翻阅曲线走势。

## 文件清单

| 文件 | 体积 | 内容 |
|---|---|---|
| `env.json` | 1 KB | run 口径与 provenance（实际 git_head、节点、jax 版本、store_meta sha256 等） |
| `run_meta.json` | 1 KB | 真实 argv、jax 编译缓存事件计数、入口标识 |
| `metrics.jsonl` | 0.4 MB | 每 100 步一行（1000 行），loss / grad_norm / llm_grad_norm / mem_enc_norm / param_norm 十进制 + IEEE hex，含 `wall_time` |
| `gpu_util_dense.ds100.csv` | 1.5 MB | dense GPU util 降采样 100×，列 `时间戳,卡号,util%,显存MiB` |
| `gpu_util.csv` | 3.2 MB | legacy 15 s 通道原样，列 `epoch秒,卡号,util%,显存MiB` |
| `nfs_read.csv` | 1.3 MB | NFS 累计读字节原样，列 `epoch秒,normal_read,server_read` |
| `meminfo.csv` | 1.2 MB | 原样，列 `epoch秒,Cached(kB),pgmajfault,anon` |
| `compute_apps.csv` | 3.2 MB | CUDA 进程存证原样，列 `epoch秒,pid,显存MiB` |
| `submit.md` | — | 两次提交实录（首次被起跑闸拦下、第二次去闸） |

**不归档**：10 份 checkpoint（129 GB，`v1-store/train-runs/mme_vla_suite/v1-prod-100k/`）、
`train_console.log`（5.2 MB，与 slurm 日志 `v1-store/logs/v1-prod-100k-59620943.log` 内容重叠）、
dense 原件与 `ds10`。
