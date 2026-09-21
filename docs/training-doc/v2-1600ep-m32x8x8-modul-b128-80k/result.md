# 正式80k起跑与前300步复核

## 1. 结论与指标

**已按用户条件授权启动，前300步复核通过，训练继续运行。** 第100→300步原生日志窗口均值1.032397秒/步、123.9834 samples/s；八卡GPU均值98.3656%、0%采样占比0.3650%。与独立1000步测速的步时比为1.001664，起跑阶段未见明显变慢。80k中心估计仍约23小时，按driver起点外推结束时间为2026-09-22约06:21 UTC，不是完成承诺。

## 2. 版本与代码状态

正式Beta为 `55647ff33c8ddb9ec324fdbcee8bd1491456725b`（commitV11.2Beta），起跑时porcelain为空。PERF_HEAD `9a7acc81b93855032a5f177a03adffe2c2209318` 到Beta的src/scripts/packages/pyproject.toml/uv.lock无差异。前300步通过后将主副本src、scripts、packages共594个路径移除写权限，保留执行权限，Git仍clean、HEAD不变；锁前权限保存在本run忽略目录。主环境没有sync或改动，后续档案在独立副本提交，主库运行期间不拉取新文档。

## 3. 启动与配置还原

2026-09-21 07:20:38 UTC在本机detached tmux `m2048-prod` 启动。V10的26门全部通过，USER_APPROVAL、GPU_IDLE与LAUNCH_CONFIG均PASS。训练PID3999606，独立500ms采样PID3999598。具体命令见 [launch.actual.json](records/launch.actual.json)，全部原始门见 [preflight.log](records/preflight.log)。

用 `git show 55647ff33c8ddb9ec324fdbcee8bd1491456725b:scripts/training/prod/run_modul2048.sh` 还原入口，以同提交的training/config.py和perceptual-framesamp-modul-32frame-8x8.yaml还原配置。CLI只覆盖已定run名、资产/数据/输出路径与新history YAML；不接续测速或短验证权重，重新从pi05_base初始化。实际解析与摘要见 [approval.actual.json](records/approval.actual.json)，config SHA为b2acd4a95c6c9fb5649fe0791b24ab891292ffeaa56befb747221cf90179bbb8。

## 4. 数据集与划分

使用1600ep四任务的verified/full framesamp-8x8，605611执行样本、1192918帧；不使用40ep/400ep旧库。本库历史均满32帧，短历史正确性由此前合成测试单独覆盖。清单、norm_stats、store_meta与YAML摘要绑定在实际授权和preflight中，训练数据及划分未变。

## 5. 关键超参

原样mme_vla_suite_b128_80k：batch128、worker16、FSDP8、seed42、80000步、warmup5000、peak_lr=decay_lr=5e-5、decay_steps80000、EMA0.999、clip1.0、log100、save5000、keep5000。生产bf16，2048记忆token，32帧×64，modulation，motion关闭；全局默认没有修改。正式W&B开启，XLA_FLAGS unset、TRAIN_TIMING_STEPS=0，无profiler或新增逐步同步。

## 6. 硬件与耗时

8×A100-SXM4-80GB，AWS本地NVMe RAID /dev/md0 XFS，运行时环境见 [runtime.json](records/runtime.json)。首次第0步指标UTC07:25:06.580454；复核窗口UTC07:26:49.352375至07:30:15.831688，共200步206.4793秒。正式窗口采用原生log100时间戳，与独立测速的三同步800步窗口不是完全相同测法。

GPU500ms原始样本和各卡均值、0%比例、采样数、最大间隔、显存峰值见 [startup300.json](records/startup300.json)。无逐步计时，不能在正式窗口重做单步慢步分层；独立测速慢步87.3664%、其他98.8127%的分层证据见[测速报告](../perf-m32x8x8-modul-b128-20260921T051856Z/result.md)。NVML重复读数不视作独立新增证据。

## 7. 训练过程行为

第0、100、200、300步的loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm全部有限，记忆编码器梯度均非零。第300步loss=0.0239966679、grad_norm=0.207800344、mem_enc_norm=0.0124392305。真实输入static_mask为(128,2048) bool。原生指标记录保留于 [metrics.startup300.jsonl.gz](records/metrics.startup300.jsonl.gz)，清洗日志为 [startup300.summary.log](records/startup300.summary.log)。

本快照时训练尚未结束，也尚未到首次周期checkpoint；日志没有EXIT_CODE属运行中，不能标记全长成功。采样器由runner在最终退出时回收，原始日志和全部后续指标继续写本run的v1-store目录。

## 8. 训练后评估

尚未开展。前300步只验证起跑行为，loss下降不证明策略成功率或完整80k收敛；完整训练结论和与60k/motion80k的策略评估按原计划后续另立任务。

## 9. 用户决定记录

「保持原计划，完整取证（推荐）」；「能并行的尽可能并行 8个gpu你可以用」；「1000步测速后如果占用率高于50% 并且没有其他的问题 可以直接启动训练」；「你要推送到hongzefu的位置 不可以用yinpei tri」。

条件判定采用预先说明的800步稳态八卡总体均值>50%，全部正确性、容量通过且测速READY，无未解决问题。实际98.6104%满足条件，因此固定Beta后直接起跑。approval的approved_at是条件成立后的授权落档时间，不冒充用户当时新增回复。Git推送沿用主副本已验证的hongzefu SSH命令，不修改凭据或upstream。

## 10. 计划外事件与处置

正式起跑后未发生错误。起跑前独立配置解析首次未source paths.sh，触发与预览不一致的断言，未写授权或启动训练；按真实入口加载同一paths.sh后精确匹配，V10又独立通过。测速归档核验曾发现清洗driver日志的original_sha字段误填清洗结果，Beta提交前已以真实原日志校正并回读验证；测量本身不变。更早的两次验证器修复及完整重测记录见总实施档案。

## 11. 结论与下一步

本轮实现、完整取证、容量、1000步测速、条件起跑及前300步复核均完成。正式训练继续，不在本轮等待80k结束。查看 [W&B](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/u5qogzm9) 或主仓v1-store/logs下同名.driver.log；完成后应核对退出码、checkpoint与全长指标再写正式V11.2结论。

## 12. 归档文件

records保存launch.actual、approval.actual、preflight、runtime、startup300、原生指标/GPU压缩快照、清洗日志和source_lock。原始产物仍在主库v1-store/bench/m2048及train-runs对应run；无权重或bash/yaml副本入Git。完整文件SHA与gzip回读SHA见 [archive_manifest.startup.json](records/archive_manifest.startup.json)。本档案的后续docs提交不改变正在运行的Beta锚点。
