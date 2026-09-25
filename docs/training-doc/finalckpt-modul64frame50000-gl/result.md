# finalckpt-modul64frame50000-gl — 结果（4096（64 帧 / budget 4096） @50000，692 条）

起跑口径见 [`launch.md`](../finalckpt-modul16frame50000-gl/launch.md)（四个 run 共用），四模型两档对照见主结果页 [`binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。

`MERGE_OK shards=10 episodes=692 successes=313 errors=9`，**313/692 = 45.23%**，macro 0.4524。
与 2048 @50000 的 312/692 = 45.09% 持平（+0.14 pp）；训到 79999 再 +0.72 pp，与其他 run「50k 后基本收敛」一致。

9 条 error 与前几轮完全相同（VideoRepick / VideoUnmaskSwap 的 reset 卡死集，起跑前预标），计入分母按失败处理，成功率是下界。

## 14 组逐格

| 组 | 4096 @50000 |
|---|---|
| BinFill/easy | 43/49 = 87.8% |
| BinFill/medium | 29/49 = 59.2% |
| BinFill/hard | 16/44 = 36.4% |
| RouteStick/easy | 49/50 = 98.0% |
| RouteStick/medium | 50/50 = 100.0% |
| RouteStick/hard | 42/50 = 84.0% |
| RouteStick/xhard | 15/50 = 30.0% |
| VideoRepick/easy | 8/50 = 16.0%（err 1） |
| VideoRepick/medium | 14/50 = 28.0%（err 3） |
| VideoRepick/xhard | 16/50 = 32.0%（err 3） |
| VideoUnmaskSwap/easy | 13/50 = 26.0% |
| VideoUnmaskSwap/medium | 12/50 = 24.0% |
| VideoUnmaskSwap/hard | 2/50 = 4.0% |
| VideoUnmaskSwap/xhard | 4/50 = 8.0%（err 2） |


## 产物

`records/merged/{log,progress}.json`；manifest、预标清单与 40 片两道闸汇总见 [`finalckpt-modul16frame50000-gl/records/`](../finalckpt-modul16frame50000-gl/records/)。
