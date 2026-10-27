# 简化弹性队列与20k早期试评估结果

## 当前结论

真实单局已完整跑通：`BinFill:0` 执行342个动作后正常终局，任务失败，成功率为0/1。随后两卡十局全部完成：1成功、9失败、0超时，成功率10%；无遗漏、基础设施错误或重试。成功任务为 `ButtonUnmask:0`。单局冒烟与两卡试评估分别计分。

这次评估使用Stage2 VLA训练20000步的checkpoint及epoch105 EMA encoder，只是早期小规模试评估；不是60000步、800局正式成绩。

## 版本与实现

最终执行代码固定为 `789ef0e5e9c241dd0f8bca11f2c3b885edc52e39`，NFS release与benchmark gitlink `856bc3a189d4172f3f47dbee4424d585f8d78db3` 固定；模型、环境、缓存和完整启动入口见同目录 `launch.md`。所有新结果位于原运行根的 `simple-1027/`，旧失败产物保留。

简化后保留单控制器文件队列、每五局一块、动态加入与drain、逐局结果和确认旧step停止后的有限基础设施重试。每卡常驻模型，每块新建仿真进程。删除旧复杂资源账本与证明前置，队列相关变更净删1590行。未改benchmark、模型权重、训练配置、动作执行口径或精度。

服务入口短测4项通过；最终整合后的队列与真实WebSocket定向4项通过，耗时0.24秒。B、C及必要心跳修复均经独立固定差异复核通过。没有重跑旧平台全测试矩阵，短测不替代下面的真实运行证据。

## 真实单局

| 项目 | 实测 |
| --- | --- |
| 任务 | BinFill:0，seed540000，easy |
| 完整运行 | 美东10月27日01:20:19—01:23:06，167秒 |
| 资源 | 64855345.step2，gl1504，1 A40／4 CPU／24 GiB |
| 终局 | outcome=fail，任务成功0/1 |
| 动作与局内耗时 | 342个动作，107.77秒 |
| 首次推理 | 约54.44秒，含冷编译；之后约0.12秒 |
| 退出状态 | 客户端、Slurm均为0，视频产物正常 |

`SINGLE_EP_SMOKE=PASS task_success=0/1 actions=342`。链路通过与任务成功分别报告，未为得到成功结果重跑正常失败局。

## 两卡共享队列

worker-1于美东01:24:20从64855345.step3启动，先领取第一块五局；worker-2于01:26:32从64855348.step0加入，并于01:28:21现场确认领取另一块。两个worker均使用上述固定版本，从同一共享队列动态领取，没有预先按卡固定分配。

| worker | 节点 | 领取成员 |
| --- | --- | --- |
| worker-1 | gl1504 | BinFill:0、StopCube:0、PickXtimes:0、SwingXtimes:0、ButtonUnmask:0 |
| worker-2 | gl1520 | VideoUnmask:0、VideoUnmaskSwap:0、ButtonUnmaskSwap:0、PickHighlight:0、VideoRepick:0 |

GPU实测覆盖新增worker加入及不同块领取；drain、已完成项保留、故障归还和一次重试通过CPU定向测试，不把这些短测写成GPU故障注入实测。两个块均已关闭，客户端各完整执行五局并退出0；控制器状态为 `COMPUTE_DONE`，活动块为0。十个视频记录均无产物错误。

| 任务 | 任务终局 | 动作数 | 局内秒数 |
| --- | --- | ---: | ---: |
| BinFill:0 | fail | 342 | 43.708 |
| StopCube:0 | fail | 286 | 22.565 |
| PickXtimes:0 | fail | 643 | 52.217 |
| SwingXtimes:0 | fail | 321 | 27.493 |
| ButtonUnmask:0 | success | 221 | 18.035 |
| VideoUnmask:0 | fail | 113 | 52.178 |
| VideoUnmaskSwap:0 | fail | 132 | 38.609 |
| ButtonUnmaskSwap:0 | fail | 382 | 32.005 |
| PickHighlight:0 | fail | 200 | 18.633 |
| VideoRepick:0 | fail | 272 | 61.617 |

`TWO_WORKER_EVAL=PASS completed=10 unique=10 success=1 fail=9 timeout=0 infra=0 retries=0`。任务成功由每局 `outcome` 独立计算，不从程序退出码推断。

## 实施中遇到的问题

1. 首次客户端启动时，仿真解释器缺少通信依赖，`started_episode_keys=[]`，未开始官方局。按benchmark已有 `pyproject.toml`／`uv.lock` 的server组，通过 `uv sync --only-group server --inexact --frozen --no-install-project` 只补齐dm-tree0.1.8、msgpack1.1.2、websockets15.0.1，其余包与依赖声明未改。
2. 补包后首次已开始局，在首次infer期间被默认WebSocket心跳提前断开；实际动作0，无终局。客户端与服务端各设置一行 `ping_interval=None`，保留600秒RPC超时；随后使用该已开始局的一次基础设施重试完成运行。未新增开关、重试平台或数值对拍前置。

前两次失败分别保存在 `simple-1027/smoke1-client/`、`simple-1027/smoke1-client-retry1/`；完成单局保存在 `simple-1027/smoke1-episode-retry1/`，互不覆盖。更早的独立恢复检查失败也保留，未充当评估成绩。

## 用户决定与范围

用户原话：「同意，按这个简化方案开工，保留弹性 worker 队列，不要再堆叠功能和校验。尽快先跑通一局，再做两卡分片评估。」按此范围实施，没有启动正式800局；跨硬件数值一致性沿用户此前决定不作为要求。

## 资源与归档

本轮仅新申请64855345与64855348两份占位，旧作业已取消。两个worker分别于美东01:34:01、01:34:14退出0；唯一CPU控制器通过精确PID收到SIGTERM后正常退出0。本轮两份占位已逐个取消，调度队列已无这两个JobID，本轮tmux全部退出，其他会话未操作。

`RESOURCE_RELEASE=PASS jobs=64855345,64855348 queue_entries=0 own_tmux=0`。权重、视频和完整原始日志保留在运行根；逐局终局、队列汇总、视频元数据及两次必要故障原文已归入本目录 `records/evaluation.json`，具名结果摘录见 `records/verification.summary.txt`。原始资源回收输出保存在运行根 `simple-1027/resource-release.log`。本次阶段一与十一局试评估已完成，不自动启动正式八百局。
