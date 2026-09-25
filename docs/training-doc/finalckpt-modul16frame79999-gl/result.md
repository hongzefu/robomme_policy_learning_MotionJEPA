# finalckpt-modul16frame79999-gl — 结果（1024（16 帧 / budget 1024） @79999，692 条）

起跑口径见 [`launch.md`](../finalckpt-modul16frame50000-gl/launch.md)（四个 run 共用），四模型两档对照见主结果页 [`binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。

`MERGE_OK shards=10 episodes=692 successes=286 errors=9`，**286/692 = 41.33%**，macro 0.4128。
落在 512 @59999 的 32.95% 与 2048 @79999 的 45.66% 之间。与 2048 差距最大的是 RouteStick/xhard（2/50 vs 22/50）与 RouteStick/hard（26/50 vs 42/50）。

9 条 error 与前几轮完全相同（VideoRepick / VideoUnmaskSwap 的 reset 卡死集，起跑前预标），计入分母按失败处理，成功率是下界。

## 14 组逐格

| 组 | 1024 @79999 |
|---|---|
| BinFill/easy | 39/49 = 79.6% |
| BinFill/medium | 31/49 = 63.3% |
| BinFill/hard | 11/44 = 25.0% |
| RouteStick/easy | 49/50 = 98.0% |
| RouteStick/medium | 46/50 = 92.0% |
| RouteStick/hard | 26/50 = 52.0% |
| RouteStick/xhard | 2/50 = 4.0% |
| VideoRepick/easy | 15/50 = 30.0%（err 1） |
| VideoRepick/medium | 16/50 = 32.0%（err 3） |
| VideoRepick/xhard | 18/50 = 36.0%（err 3） |
| VideoUnmaskSwap/easy | 9/50 = 18.0% |
| VideoUnmaskSwap/medium | 11/50 = 22.0% |
| VideoUnmaskSwap/hard | 5/50 = 10.0% |
| VideoUnmaskSwap/xhard | 8/50 = 16.0%（err 2） |


## 产物

`records/merged/{log,progress}.json`；manifest、预标清单与 40 片两道闸汇总见 [`finalckpt-modul16frame50000-gl/records/`](../finalckpt-modul16frame50000-gl/records/)。
