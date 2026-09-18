# 200 次真实 reset：budget 160 上界检查通过

四任务各 50 集全部完成，episode 集合完整、无重复。最长 demo 为 383 帧，1300 步评估口径下最多需要 103 个 motion 窗，预算 160 余量 57 窗；四任务和总退出码均为 0。

用户原话：「采用现有仿真环境和单卡渲染，保留 200 次真实 reset」。起止均为 clean f6915f2a09443d48c1bb57e9b5f83e95400704fd。由 uv 调用既有 micromamba/robomme 环境，GPU 7 渲染，四任务分进程，不加载策略。完整命令及环境见 [launch.md](launch.md)。

| 任务 | 集数 | 最大 es | make_env/reset 累计实测 |
|---|---:|---:|---:|
| BinFill | 50 | 0 | 401.7 秒 |
| RouteStick | 50 | 350 | 1542.8 秒 |
| VideoRepick | 50 | 383 | 234.3 秒 |
| VideoUnmaskSwap | 50 | 216 | 154.1 秒 |

总挂钟为 2026-09-17 21:45:49–22:25:40 UTC，2391 秒；与表中累计的差包含解释器启动、导入和聚合。使用 GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192，未出现历史 Vulkan TLS 失败；各任务独立进程和缓存落点均按启动记录执行。

聚合器独立重算预算，没有直接信任 worker 自报 k。判定为 `EVAL_ES_BOUND=PASS tasks=4 episodes=200 es_max=383 budget=160 headroom=57`。RoboMME 提交、test metadata SHA、软件版本、每集 es 与耗时、原始退出码均保存在 [records](records/)。

该结果用于起跑前预算检查；正式策略成功率评估按计划另行执行。
