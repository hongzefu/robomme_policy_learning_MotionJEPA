# finalckpt-modul64frame79999-gl — 结果（4096（64 帧 / budget 4096） @79999，692 条）

起跑口径见 [`launch.md`](../finalckpt-modul16frame50000-gl/launch.md)（四个 run 共用），四模型两档对照见主结果页 [`binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。

`MERGE_OK shards=10 episodes=692 successes=318 errors=9`，**318/692 = 45.95%**，macro 0.4589。
与 2048 @79999 的 316/692 = 45.66% 几乎持平（+0.29 pp）：感知上下文从 2048 翻到 4096 没有带来收益。

9 条 error 与前几轮完全相同（VideoRepick / VideoUnmaskSwap 的 reset 卡死集，起跑前预标），计入分母按失败处理，成功率是下界。

## 14 组逐格

| 组 | 4096 @79999 |
|---|---|
| BinFill/easy | 42/49 = 85.7% |
| BinFill/medium | 29/49 = 59.2% |
| BinFill/hard | 13/44 = 29.5% |
| RouteStick/easy | 50/50 = 100.0% |
| RouteStick/medium | 50/50 = 100.0% |
| RouteStick/hard | 44/50 = 88.0% |
| RouteStick/xhard | 20/50 = 40.0% |
| VideoRepick/easy | 12/50 = 24.0%（err 1） |
| VideoRepick/medium | 17/50 = 34.0%（err 3） |
| VideoRepick/xhard | 10/50 = 20.0%（err 3） |
| VideoUnmaskSwap/easy | 11/50 = 22.0% |
| VideoUnmaskSwap/medium | 9/50 = 18.0% |
| VideoUnmaskSwap/hard | 3/50 = 6.0% |
| VideoUnmaskSwap/xhard | 8/50 = 16.0%（err 2） |


## 产物

`records/merged/{log,progress}.json`；manifest、预标清单与 40 片两道闸汇总见 [`finalckpt-modul16frame50000-gl/records/`](../finalckpt-modul16frame50000-gl/records/)。
