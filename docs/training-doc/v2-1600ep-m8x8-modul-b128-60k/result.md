# 正式训练进行中：起跑验收与 300 步复核

`v2-1600ep-m8x8-modul-b128-60k` 已于 `2026-09-15T05:48:27Z` 从 clean HEAD `dd07f18fc385b01eb52db7563fe5f202997b9706` 启动，使用 GPU 4,5,6,7。本文是起跑阶段记录，60k 步训练尚未结束，最终退出码、12 个 checkpoint 和结束后的 V10 复查均待完成。

[wandb 运行](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/6ubtaf9l) 在线同步；训练会话 `m8-prod`、前 30 分钟密采会话 `m8-prod-dense`。完整启动配置见 [launch.md](launch.md)，实际命令、进程 PID、缓存与开发环境信息见 [records/launch.actual.json](records/launch.actual.json)。

## 起跑前验证全部通过

F2 的 3454 个样本/19 键逐字节一致，F3 的 100 步五标量、800 个训练索引、6 份输入与完整 TrainState 均逐位一致。modulation 两档 A/A 和 A/B 共四组比较通过：61 个初态叶子、三类 loss hex、38 个可训练梯度叶子完全相同。

b128 四卡 smoke 的 20 步全部有限并正常退出，checkpoint 参数树 61/61 精确匹配，六条 modulation 专属参数路径齐全，motion 关闭，新库 norm_stats SHA 与 checkpoint 中保存的文件一致。正式 run 的 25 项 preflight 全通过，原文见 [records/preflight.log](records/preflight.log)。

## 300 步数值与代码隔离

| 日志步 | loss（显示值） | grad_norm（显示值） | mem_enc_norm（显示值） |
|---|---:|---:|---:|
| 0 | 0.0894 | 1.2680 | 0.1092 |
| 100 | 0.0563 | 0.5048 | 0.0416 |
| 200 | 0.0292 | 0.1914 | 0.0122 |
| 300 | 0.0240 | 0.2083 | 0.0125 |

正式档 `log_interval=100`，除 step 0 外表中值是对应日志区间的统计，不冒充逐个单步记录。四次日志的全部五项标量均有限，精确 dec/hex 见 [records/startup_metrics.jsonl](records/startup_metrics.jsonl)。当前没有运行错误，不根据这些训练 loss 推断策略评估效果。

主副本 `src/`、`scripts/`、`packages/` 已锁只读，开发全部转到 `-temp`。开发副本使用独立 `.venv`，共享数据仅通过已批准的 `v1-store` symlink 访问。300 步后的复核确认主副本仍为原 HEAD、Git 状态干净且三份源码 SHA 未变，见 [records/startup_check.json](records/startup_check.json) 与 [records/lock_sha256.txt](records/lock_sha256.txt)。这不替代训练结束后的再次复查。

## 起跑稳态性能与待查问题

存储为 AWS 本地 NVMe RAID（`/dev/md0`），batch 128、worker 8、fsdp 4。排除前 100 步，实际窗口为 tqdm 进度 102→302，共 200 次更新、484.184 秒；相邻进度区间共 38 个。GPU 每 500ms 采样，窗口内 3872 条记录，每卡 968 条。

| 指标 | 实测 |
|---|---:|
| 稳态步时均值 | 2.42092 秒 |
| 吞吐 | 52.87 samples/s |
| GPU util 均值 | 70.14% |
| 0% 采样占比 | 27.30% |
| 慢区间 util 均值 / 0% 占比 | 52.31% / 44.47% |
| 其他区间 util 均值 / 0% 占比 | 77.92% / 19.81% |

**GPU 尚未吃满。** 慢区间利用率更低，表明当前有较多等待；这份统计不能单独定位到 CPU、I/O 或同步环节。性能问题已向用户反馈，当前保持已授权的训练运行，不擅自改动 worker、batch、学习率或其他设置。

分层按相邻 tqdm 进度区间的平均步时进行，慢区间阈值为 3.58993 秒（区间步时中位数的 1.5 倍），共 13 个慢区间。由于进度点通常跨多个训练步，这里明确称「区间」，不声称逐个训练步分层；中位数仅用于分层阈值。完整数值与四卡分表见 [records/startup_performance.json](records/startup_performance.json)，原始起跑日志、密采快照和分析脚本一并归档。

按该窗口外推，总计算时间约 **40.35 小时**，预计 **2026-09-16 22:13 UTC** 左右完成。此 ETA 不含后续 checkpoint 开销、epoch 边界和资源争用变化，后续应随实际训练更新。它替换计划中的旧档位约 65 小时估计，不据不同配置的两条 run 宣称某项改动带来等比例提速。

## 后续收尾

保持训练运行并保留全部已有 checkpoint；最终应产出 5000 至 55000 的 11 个周期 checkpoint 和 59999 的最终 checkpoint。结束后记录训练 `EXIT_CODE`，恢复主副本代码写权限，再做 Git 状态与源码 SHA 复查、全程指标归档和提交同步。评估另立任务。
