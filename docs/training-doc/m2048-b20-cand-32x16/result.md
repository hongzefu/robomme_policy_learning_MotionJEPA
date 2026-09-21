# 旧32×16候选训练回归结果

完整20步与1步补跑全部通过，至此旧两档的输入与训练回归均已完成。BASE为29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6，CAND为0c877c7495dfe5db8b83f033442013c6d6fd8552；全部五标量、输入和201叶TrainState逐位相同，合并后state_step=0..20齐全，mismatches=0。

GPU4、5、batch8、worker4、FSDP2、seed42；主run04:40:55Z→05:43:22Z，3747秒；补跑05:43:22Z→05:51:06Z，464秒。主232个索引/29批、160个实际参与更新，补跑80个索引/10批；首批与首标量一致。两次compare_baseline以及官方完整记录检查均通过，见[判定原文](records/complete-comparison.summary.log)。驱动EXIT_CODE=0，tmux自然结束，GPU已释放。

用户选择「保持原计划，完整取证（推荐）」，故保留全叶、每步与补跑，不因约3分钟/份摘要成本缩减证据。环境为AWS A100-SXM4-80GB、/dev/md0 XFS NVMe RAID，1600ep的framesamp 4×4库、norm_stats和依赖不变。本档仅证明旧32×16行为不变，不要求512与2048前向等价，也不据此估算生产速度。完整[启动记录](records/launch.actual.json)与[原SHA](records/archive_manifest.json)可复核。
