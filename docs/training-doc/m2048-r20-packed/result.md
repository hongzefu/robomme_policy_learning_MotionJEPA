# 2048参考与packed真实训练完整对拍

V7b全部通过。两侧同为2048，从主副本clean CAND 0c877c7495dfe5db8b83f033442013c6d6fd8552在GPU0、1顺序执行，b8/w4/FSDP2/seed42。主20步的五标量hex、输入逐键摘要和全部201叶状态均相同；两侧各1步补跑补齐更新1后，完整状态0..20仍逐位相等，mismatches=0。

packed主run05:22:51Z→06:26:40Z，3829秒；补跑06:26:40Z→06:34:42Z，482秒，驱动EXIT_CODE=0。主232索引/29批、160个参与更新，补跑80索引/10批。两次compare_baseline及官方完整性比较全部通过，见[判定原文](records/complete-comparison.summary.log)；主/补初态、首批与首标量也一致。

packed侧十个记忆叶的Adam nu与参数均实际改变、全程有限，七个缺失/错相位/非有限/同漏叶重算摘要等反例全拒。各20批mask_sum=16384，补跑同样满历史；该项只记形制，不代替[最终位置与mask功能证据](../m2048-func-retry2/result.md)。状态和索引原件压缩回读SHA见[清单](records/archive_manifest.json)，版本与退出见[实际启动](records/launch.actual.json)。

用户要求「保持原计划，完整取证（推荐）」「能并行的尽可能并行 8个gpu你可以用」。本项沿用全部叶、步数和判据，同卡对照不变；仅物理组从原4/5调整为共同0/1。AWS A100、/dev/md0 XFS NVMe RAID，1600ep数据与规范化资产未改。本结果证明同2048配置下两条输入链的更新等价，不要求512与2048等价，也不以含摘要的耗时估计生产速度。
