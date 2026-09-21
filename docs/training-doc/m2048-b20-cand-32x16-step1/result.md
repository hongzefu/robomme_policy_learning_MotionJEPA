# 旧32×16候选第1次更新补跑

补跑成功，05:43:22Z→05:51:06Z，464秒，驱动EXIT_CODE=0。CAND为0c877c7495dfe5db8b83f033442013c6d6fd8552，GPU4、5、b8/w4/FSDP2/seed42，遵循[主run启动命令](../m2048-b20-cand-32x16/launch.md)的1步分支。

实得状态0/1、201叶、1行五标量、80索引/10批；与本侧主run初态/首批/首标量一致，并与BASE对应补跑逐位一致。合并状态0..20的完整性通过，见[双侧判定](records/complete-comparison.summary.log)。用户「保持原计划，完整取证（推荐）」要求的旧档两侧补跑已全部收齐。

数据、依赖、确定性条件与主run相同，AWS /dev/md0 XFS本地NVMe RAID。原记录压缩回读SHA和唯一Step日志行数已核验，见[实际运行](records/launch.actual.json)。
