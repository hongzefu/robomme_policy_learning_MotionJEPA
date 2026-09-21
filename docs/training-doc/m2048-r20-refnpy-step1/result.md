# 2048参考链第1次更新补跑

补跑成功。实际为2026-09-21T05:13:26Z→05:21:21Z，475秒，驱动EXIT_CODE=0。源码是clean CAND 0c877c7495dfe5db8b83f033442013c6d6fd8552，GPU0、1、batch8、worker4、FSDP2、seed42；命令由[主run启动档案](../m2048-r20-refnpy/launch.md)的1步分支还原。

实得state_step=0/1两行、201叶、1行五标量、80个索引/10批；初态、首批及首标量与主20步一致。合并状态0..20通过，见[完整性结果](records/complete-states.json)。本补跑为用户「保持原计划，完整取证（推荐）」要求的一部分，尚待packed对应补跑完成后双侧比较。

环境与主run相同：AWS A100、/dev/md0 XFS NVMe RAID、1600ep源NPY及原norm_stats。原始指标、状态与索引压缩回读SHA通过，实际版本和退出信息见[运行记录](records/launch.actual.json)。
