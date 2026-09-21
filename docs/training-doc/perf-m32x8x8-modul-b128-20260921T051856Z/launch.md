# 2048八卡独立1000步测速

run名在2026-09-21T05:18:56Z准备档案时生成，不是实际起跑时间；实际HEAD和UTC另由runner记录。会话为m2048-perf-20260921T051856Z。必须在所有正确性与容量门通过、其他本轮GPU/IO任务结束后，从主副本clean HEAD独占八卡起跑。用户原话：「1000步测速后如果占用率高于50% 并且没有其他的问题 可以直接启动训练」。

完整生产模型与mme_vla_suite_b128_80k原样，b128/w16/FSDP8/seed42、LR5e-5/warmup5000/EMA0.999；仅启动覆盖1000步与关闭W&B，保留log_interval100/save_interval5000。数据、清单、norm_stats和YAML与m2048-smoke相同；仅使用1600ep的framesamp-8x8。XLA_FLAGS明确unset，与正式运行相同；无profiler、逐步状态摘要或逐批字节摘要。

预热0..99，主窗100..899共800步，尾段900..998观察趋势，999实际保存。轻量包装保留真实train.py、loader与保存，只在99/899末尾及999保存前共同步三次。独立GPU采样500ms、主机与磁盘采样保留原始记录；均值、0%占比、慢步/其他步分层和八个100步块同时报告，不把异步主线程时间冒充设备kernel耗时。

入口在detached tmux运行，runner负责pipefail、PYTHONUNBUFFERED、tee、EXIT_CODE及采样器PID回收。记录根为v1-store/bench/m2048/perf-m32x8x8-modul-b128-20260921T051856Z，日志为同run名的.driver.log，checkpoint为v1-store/train-runs/mme_vla_suite_b128_80k下同名目录；均须全新。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
M2048_PERF_HEAD=$(git rev-parse HEAD)
bash scripts/training/prod/run_modul2048.sh perf perf-m32x8x8-modul-b128-20260921T051856Z "$M2048_PERF_HEAD" 42813c7e06840e511f8a382f14954edcaad8d4fd5b320b4323fea94979cda4de
```

退出后执行check_modul_speed.py report，传records、同名.gpu.csv、driver日志和全新report.json。P为进程开始至99末同步，μ为800步同步主窗均值，B为真实保存调用阻塞，C为保存开始至管理器等待结束及收尾的上界；80k中心估计P+79900μ+15B+C，保守保存情景P+79900μ+16C，并报告八块速度情景范围。它们是外推，不是实跑80k；W&B、长期缓存演化与周期保存争用的限制必须说明。

要求SPEED_WINDOW、八卡GPU_COVERAGE、shm≤0.70和真实checkpoint全部通过，缺采样、明显漂移或并发污染时报告INCOMPLETE。只有SPEED_REPORT=READY、全部前置门通过且无未解决问题、800步主窗八卡总体平均util严格>50%时，才落实用户条件授权，绑定实际报告SHA/runner SHA/TRAIN_HEAD/正式run名，重跑V10并直接起跑v2-1600ep-m32x8x8-modul-b128-80k。未达条件停止并报告；不预填成功结果或报告SHA。
