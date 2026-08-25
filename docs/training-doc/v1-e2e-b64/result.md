# v1-e2e-b64 结果留档（实验 4：4×A40 官方口径 b64 端到端真实吞吐）

起跑 commit `d0fa27e`，job 58627883（gl1505），2026-08-24 完成 300 步训练。

**结局说明**：job 最终态 `OUT_OF_MEMORY`（host 64G，`sacct` 确认）——被杀的是**末步参数校验和**（`device_get` 需把 params+EMA 约 28 GB 拉回 host，叠加 dataloader worker 缓冲超出 64G cgroup 限额）。**300 步训练与全部逐步记录在被杀前已完整落盘**（metrics.jsonl 恰 300 行、loss 全有限 0.648→0.035），吞吐结论无损；稳态分析在本机对同一 metrics.jsonl 复算（与 sbatch 内嵌逻辑逐行相同）。

## 核心结果

| 项 | 值 |
|---|---|
| e2e 稳态（丢前 50 步，n=249 中位） | **6.933 s/step**（p10 5.008 / p90 8.537，尾部宽） |
| 实际吞吐 | **9.2 样本/s** |
| **epoch（6,176 步）实测口径** | **42,815 s ≈ 11.9 小时** |
| 稳态窗口 NFS 实测读 | 122 MB/s（公式口径 173 MB/s） |
| GPU 利用率中位（nvidia-smi 粗粒度） | 100% |
| 对比 compute-only（4.778 s/step） | **慢 45%** |

## 瓶颈归因（与实验 1/3 交叉验证）

- **不是 NFS 存储侧**：e2e 期间实际只消耗 122 MB/s，远低于同节点族 standalone 供给 398-628 MB/s；供给本身是需求 251 MB/s 的 1.6-2.5 倍。
- **是 job 内 CPU 配比**：8 个 CPU 同时供 jax 主进程（驱动 4 卡）与 4 个 dataloader worker；worker 被挤到连在线所需的 13.4 样本/s 都供不上（standalone 同 4 worker 配足 CPU 可达 27-36 样本/s）。证据链：p10 = 5.008 ≈ compute-only（最快的步是计算受限的）、p90 = 8.537（慢步在等数据）、供给实验显示每 worker CPU 预算 +1 可使供给 +30%。
- **改进方向**（预计可把步时压回 ~4.8-5.2 s、epoch 回到 ~8.2-8.9 h）：正式训练 sbatch 把 `--cpus-per-task` 提到 ≥12-16，和/或 `num_workers` 提到 8-16（供给侧证明 NFS 有余量）；这是启动脚本参数，无需改代码。
- **附带教训**：4 卡 run 若开参数校验和，host 内存须 ≥96G（28 GB device_get 峰值），或像本实验的正确做法——e2e 吞吐测试就不该开校验和（save-interval 已设 1000，但 train.py 末步强制保存触发了它）。

产物：本目录 `records/`（metrics.jsonl 300 行 / gpu_util.csv / nfs_read.csv / env.json，git 归档）；工作副本 `v1-store/bench/bottleneck/v1-e2e-b64/`；日志 `v1-store/logs/v1-e2e-b64-58627883.log`。run 目录与节点 jax 缓存已由 sbatch 清理（OOM 发生在清理之前的训练进程内，sbatch 外层清理照常执行）。
