# 2048 before 20步数值回归结果

已于2026-09-23 00:31:28 UTC启动，00:53:36 UTC主动中断，EXIT_CODE=130，不能认定20步通过。工具锚点e1b97169d9a1cdc838af5f27bce0bca6a7a513e9，训练源码742f2d894d25abe802a26b1c118852b0b7676e44；二者起跑均clean。tmux为modul-sweep-20260923T003111Z，PID1197660。

真实8卡、batch128、worker16、FSDP8。首个完整初态201叶均有限，checksum_seconds=615.913，state_digest前缀85a829759b07d809；正常训练标量只到step0/1，第二次更新后的状态摘要中途停止。完整状态、指标、索引、环境及清洗日志在records；不把交集或部分前缀当成完整20步验收。

用户原话及原始命令见[总档案](../modul-sweep-20260923-a/launch.md)。取证开销导致本轮中断，后续[b批次](../modul-sweep-20260923-b/launch.md)使用相同字节协议的有界并行取证重新运行，原现场保留。
