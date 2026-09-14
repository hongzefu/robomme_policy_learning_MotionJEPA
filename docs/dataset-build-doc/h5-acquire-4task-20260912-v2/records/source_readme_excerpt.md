# 源 README 溯源与分布摘录

来源为固定 revision 的 snapshot/README.md；形状错误段未作为事实引用。

# RoboMME 四任务 h5 数据集（运行 20260912-contract-v3-10）

本仓库发布 RoboMME 新值注入专项运行 `20260912-contract-v3-10` 的 h5 轨迹与配套录像。
四个任务 BinFill / RouteStick / VideoUnmaskSwap / VideoRepick，每个任务 400 条正式 h5，
共 **1600** 条 primary、**196** 条 spare 备件、
**1** 条 smoke 样例，另带 **1949** 个 mp4 录像。

## 条数分布

| 任务 | 难度 | primary | spare | smoke |
| --- | --- | ---: | ---: | ---: |
| BinFill | easy | 134 | 16 | 0 |
| BinFill | medium | 133 | 16 | 0 |
| BinFill | hard | 133 | 6 | 0 |
| RouteStick | easy | 100 | 15 | 1 |
| RouteStick | medium | 100 | 15 | 0 |
| RouteStick | hard | 100 | 15 | 0 |
| VideoUnmaskSwap | easy | 100 | 15 | 0 |
| VideoUnmaskSwap | medium | 100 | 15 | 0 |
| VideoUnmaskSwap | hard | 100 | 15 | 0 |
| VideoRepick | easy | 134 | 16 | 0 |
| VideoRepick | medium | 133 | 13 | 0 |
| RouteStick | xhard | 100 | 15 | 0 |
| VideoUnmaskSwap | xhard | 100 | 13 | 0 |
| VideoRepick | xhard | 133 | 11 | 0 |

## 生成口径

* 注入契约 **v3**（`meta/configs/injection_contract_v3.json`，sha256 `bfcf4c5984fb89f28c8f04ae918513514ec3d726edb0735d7a22e8587b450da6`）；
* 候选规格按每 100 条一个 **block** 扩容冻结，组内 episode 号连续；
* 每档**实跑目标 × 1.15** 留失败余量（`meta/configs/delivery_400.json`）；
* **严格交付**：每组按 episode 升序取**前 N 条通过**为 primary，其余通过条降级为 spare，未通过条进 `MANIFEST.json` 的 `failures`；
* 每档另有 **50 条只做 env-check、不出 h5** 的额外候选，其结果在 `meta/env_check/**.jsonl` 里 `delivered=true` 的行。
