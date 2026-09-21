# 512与2048初态比较结果

完整初态比较通过：同seed42的init_train_state分别构造512与2048配置，61个参数叶的路径、shape、dtype和字节SHA完全一致，mismatches=0。预算没有进入参数形状这一点已由实际初始化验证。

CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`，CPU运行，04:08:00Z→04:10:07Z，127秒、EXIT_CODE=0；未启动参数更新。原始逐叶结果在[init.json](records/init.json)，判定见[日志](records/driver.summary.log)。该结论不意味着不同记忆长度的前向输出相同。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
