# 功能检查首次失败与处置

本次run失败，EXIT_CODE=1，不能作为V5通过证据。CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`，按用户并行决定改用GPU2，于04:08:00Z启动；04:13:15Z在逐位置输入梯度执行处报错。

三次A/A全部38个可训练梯度叶已一致，但JIT闭包把整棵权重捕获为常量，CompilationResultProto膨胀到10039569987字节，超出protobuf的2GB上限；随后新增75497472字节常量分配失败。完整[原始错误](records/driver.summary.log)保留。该错误属于验证包装，不是数值比较失败，未回滚生产或放宽阈值。

修复将input_function与act的权重改为显式参数。先做保留源与SHA的[探索性诊断](../m2048-func-probe1/result.md)，再提交 `31d4bd0b03cc1031eee04b95a69cd6a8aa6a1222`，从clean提交[正式重测](../m2048-func-retry1/result.md)全通过，问题已闭合。本run原失败状态不改写。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
