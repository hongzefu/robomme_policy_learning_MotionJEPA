# V4：3232 个关键样本逐位通过

从 clean 81bd0217f890218a1134b135adcf3490cc68dd51 启动，2026-09-18 02:56:23–02:57:04 UTC，41 秒、EXIT_CODE=0。命令和输入见 [launch.md](launch.md)。

HAND_CALC_8FRAME=PASS samples=3232 mismatches=0。覆盖全部1600个冷启动、1600个首exec窗、全部32个k≥140样本和16种尾帧余数，最大k=141。motion表行、时间码、右填充、mask和mem_order的独立手算与真实Dataset交付逐位一致。样本清单和覆盖细节保存在 [report.json](records/report.json)，日志见 [run.summary.log](records/run.summary.log)。

用户要求按计划实施并保持逐位判据；本任务已完成。该检查使用组件级state统计夹具，真实训练norm_stats的交付另由V8/V9验证，不据此宣称策略效果。
