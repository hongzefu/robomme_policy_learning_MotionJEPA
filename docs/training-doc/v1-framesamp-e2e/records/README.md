# v1-framesamp-e2e 原始记录归档说明

2026-08-28 按 AGENTS 12 补归档（用户批准，见 `v1-95util.md` 双审计修复轮）。此前本 run 只有
`launch.md` / `result.md` 二次汇总，`result.md` 与 `v1-95util.md` 诊断节引用的原始数据
（89.2% util、97.4%/71.6% 快慢步分层、慢步相位、冷热缓存对比）无法从 git 复核——Codex 审计
据此判「无冻结证据」，Claude 对抗审计用 `v1-store` 原件复算后确认数字属实、只是未归档，故补齐。

各子目录逐字节拷贝自 `v1-store/bench/bottleneck/` 下同名目录：

| 子目录 | 用途 |
|---|---|
| `v1-framesamp-e2e-w4c16` / `-w8c16` / `-w12c16` | worker 档位对比（w8 为主诊断档：89.2%、慢步相位 `step%8∈{0,3}`） |
| `v1-framesamp-e2e-w4c16-coldlike` / `-hot` | COLDHOT 冷热缓存对比（5.224 vs 5.180 s；注意 `result.md` 标题「+0.7%」与展示值换算的 +0.85% 不符，以换算为准） |

每档内容：`metrics.jsonl`（逐步 wall_time 等）、`env.json`（配置指纹）、`gpu_util.csv`
（15 s legacy 采样）、`gpu_util_dense.csv`（500 ms dense 采样）、`compute_apps.csv`、
`meminfo.csv`、`nfs_read.csv`、`run_meta.json`。
