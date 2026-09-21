# m2048-dump-cand-32x16 结果

旧档输入回归通过。BASE为 `29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6`，CAND为 `0c877c7495dfe5db8b83f033442013c6d6fd8552`；1600集、605611个执行样本身份与1192918帧采样索引均相同，3000个定点样本与200个batch的逐键dtype、shape和原字节摘要完全一致，mismatches=0。

CPU从2026-09-21T03:29:35Z启动，正常退出；结束时间按日志mtime留存，约5.3分钟。两侧完整fixture计划一致、limit=0、per_step=200，manifest逐文件SHA全部核验。origin_mode=per_episode_offset、covers_zero_pad=false，真实库全部满历史；这些组名不表示覆盖了真实短历史。

官方compare_fixture_dumps.py判定原文见[对拍日志](records/comparison.summary.log)。原记录与压缩回读SHA见[归档清单](records/archive_manifest.json)。本档只证明同配置输入交付不变；训练状态由对应20+1步回归另行证明。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
