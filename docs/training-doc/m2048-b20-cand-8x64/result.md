# 旧8×64候选训练回归结果

完整20步与1步补跑均通过，与BASE逐位相同。BASE `29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6`、CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`；每次更新的五标量、输入、参数、EMA及优化器状态均一致。合并补跑后state_step=0..20完整，共201叶，无非有限值或差异。

主run于03:29:36Z→04:32:00Z运行，耗时3744秒；补跑04:32:00Z→04:39:49Z，469秒。GPU4、5，batch8、worker4、FSDP2、seed42，确定性设置与BASE相同。驱动EXIT_CODE=0，tmux自然结束、GPU释放。约3分钟/份的完整摘要是验证成本，不用于训练ETA。

主run有20行五标量、状态0及2..20、232个索引/29批，其中160个实际参与训练；补跑有状态0/1、80个索引/10批。官方check_modul_train_records.py核同初态、首批、首标量及合并状态完整性，判定见[完整比较](records/complete-comparison.summary.log)。每个子run的compare_baseline也全部通过，清洗日志保留20个Step行。

本档证明生产改动没有改变旧8×64的输入与前20次更新数值；不将其作为2048与512数值等价或策略质量提高的证据。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
