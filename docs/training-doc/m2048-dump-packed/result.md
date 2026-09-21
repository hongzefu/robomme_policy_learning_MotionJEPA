# m2048-dump-packed 结果

新2048档的源NPY与packed完整对拍通过。两侧均从clean CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552` 运行，完整遍历1600集、605611个执行样本及1192918帧；3000个定点样本、200个batch逐键dtype、shape和原字节摘要完全相同，mismatches=0。

本侧实现为packed，CPU运行并以EXIT_CODE=0结束。packed于04:09:18Z启动、日志末次mtime约04:25:22Z；refnpy于04:09:20Z启动、末次mtime约04:26:59Z。它们是并发输入取证的墙钟，不能当作生产吞吐对照。

先完成四任务400个有界真实样本对拍，再跑本项全量fixture。manifest完整性、计划相等、limit=0与200批均由官方比较器核验，见[判定原文](records/comparison.summary.log)；[压缩清单](records/archive_manifest.json)保留原SHA。真实样本恒满32帧，补零/短历史另由m2048-input与正式功能门验证，不用全True mask代替功能证据。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
