# finalckpt-modul16frame50000-gl — 结果（1024（16 帧 / budget 1024） @50000，692 条）

起跑口径见 [`launch.md`](../finalckpt-modul16frame50000-gl/launch.md)（四个 run 共用），四模型两档对照见主结果页 [`binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。

`MERGE_OK shards=10 episodes=692 successes=291 errors=9`，**291/692 = 42.05%**，macro 0.4202。
落在 512 @50000 的 30.49% 与 2048 @50000 的 45.09% 之间；79999 比 50000 反而低 0.72 pp（291→286），属噪声级波动。

9 条 error 与前几轮完全相同（VideoRepick / VideoUnmaskSwap 的 reset 卡死集，起跑前预标），计入分母按失败处理，成功率是下界。

## 14 组逐格

| 组 | 1024 @50000 |
|---|---|
| BinFill/easy | 40/49 = 81.6% |
| BinFill/medium | 28/49 = 57.1% |
| BinFill/hard | 13/44 = 29.5% |
| RouteStick/easy | 49/50 = 98.0% |
| RouteStick/medium | 45/50 = 90.0% |
| RouteStick/hard | 30/50 = 60.0% |
| RouteStick/xhard | 1/50 = 2.0% |
| VideoRepick/easy | 14/50 = 28.0%（err 1） |
| VideoRepick/medium | 16/50 = 32.0%（err 3） |
| VideoRepick/xhard | 17/50 = 34.0%（err 3） |
| VideoUnmaskSwap/easy | 12/50 = 24.0% |
| VideoUnmaskSwap/medium | 17/50 = 34.0% |
| VideoUnmaskSwap/hard | 4/50 = 8.0% |
| VideoUnmaskSwap/xhard | 5/50 = 10.0%（err 2） |


## 收尾

40 单元 `rc=0`、40 个 `SHARD_PASS`、4 个 `MERGE_OK episodes=692 errors=9`；四个 run 的 `DEMO_INJECT` 全 PASS、注入合计各恰 142，`MOTION_WINDOWS` 全 SKIP（非 motion）。40 单元均值 0.62 h/片（0.53–0.69），合计 24.8 GPU·h；8 卡墙钟 **3 h 15 min**（17:24:50Z → 20:39:35Z）。64 帧分片并不比 16 帧慢（0.62 h vs 0.63 h），主干序列长度不是评测瓶颈。

8 个占位作业在 `EV1K4K_POOL_ALL_DONE` 与 4 个 `MERGE_OK` 之后于 2026-09-25T20:40:20Z 逐个 `scancel`（写全 jobid，一次一个）：删前队列 10 个作业（8 个 `ev1k4k-hold-*` + 用户自己的 `v6gen-hold-1/2`），删后恰剩 `v6gen-hold-1/2` 两个，差集恰为 8 个目标。占位实际使用 3 h 37 min（提交 17:02:55Z → 取消 20:40:20Z），远小于 48 h 上限。

本轮起过的 tmux 会话均已自然退出：本机 `ev-1k4k-dl-1024` / `ev-1k4k-dl-4096`，登录节点 `ev-1k4k-smoke64` / `ev-1k4k-pool`。

`records/`：`manifest.tsv`（40 行）、`preseeded-errors.json`、`shard-checks.json`（40 片两道闸汇总）、`merged/{log,progress}.json`。
