# 旧8×64候选的第1次更新补跑

补跑成功，补齐20步主run没有落盘的state_step=1。CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`，04:32:00Z→04:39:49Z，GPU4、5、batch8、worker4、FSDP2、seed42，与[主run启动口径](../m2048-b20-cand-8x64/launch.md)相同。

实得状态0/1两行、完整201叶、1行五标量、80个索引/10批。初态、首批与主run首标量逐位一致，且与BASE对应补跑逐位一致；合并后的0..20状态完整性由[双侧检查](records/complete-comparison.summary.log)确认。清洗日志保留唯一Step行，驱动EXIT_CODE=0。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
