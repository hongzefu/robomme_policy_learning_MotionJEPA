# 正式80k训练完成、最终权重验收与副本合并

## 1. 结论与指标

**80,000步已于2026-09-22 06:19:30 UTC正常完成，EXIT_CODE=0，总耗时22小时58分52秒。** 16份checkpoint全部保存，最终79999已真实CPU加载通过，参数树61/61、配置与数据同源；800条记录的4000个标量全部有限。完整结果见 [training_completion.json](records/training_completion.json)。

正式step100→79900共79800次更新，平均1.030577秒/步、124.2023 samples/s；同窗八卡GPU利用率均值98.4911%、0%采样占比0.2321%。原先约23小时的预测与实测接近，现以实际完成时间为准。

起跑时第100→300步原生日志窗口均值1.032397秒/步、123.9834 samples/s；GPU均值98.3656%、0%采样占比0.3650%。该早期记录继续保留，与下方全程统计分开。

## 2. 版本与代码状态

正式Beta为 `55647ff33c8ddb9ec324fdbcee8bd1491456725b`（commitV11.2Beta），起跑时porcelain为空。PERF_HEAD `9a7acc81b93855032a5f177a03adffe2c2209318` 到Beta的src/scripts/packages/pyproject.toml/uv.lock无差异。前300步通过后将主副本src、scripts、packages共594个路径移除写权限，保留执行权限，Git仍clean、HEAD不变；锁前权限保存在本run忽略目录。主环境没有sync或改动，后续档案在独立副本提交，主库运行期间不拉取新文档。

结束验收从上述clean Beta执行。合并前将313份Git管理的源码和依赖文件逐字节核对到Beta blob，全部相同；随后按锁前记录恢复594个路径的原权限，将已推送的起跑档案快进到621134a0fba435e28b51e4afcbf28537f4b3cfb4。本次完整结论与Beta配对提交commitV11.2，不amend或合并掉Beta。主环境没有重装依赖。

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

全程统计仍为同一8卡/b128/w16/FSDP8和本地NVMe。step100→79900耗时82240.0494秒，含周期checkpoint，排除初始化、起跑前100步和最后99步。每卡163851条500ms读数，共1310808条；平均间隔0.50192秒，最大0.688秒，各卡时间覆盖约99.9995%。详情见 [performance.full.json](records/performance.full.json) 和 [完整密采压缩文件](records/gpu_util_500ms.full.csv.gz)。

正式日志只保留每100步时间戳，分层只能针对100步区间：超过区间均值1.5倍（1.545866秒/步）的区间为0，慢区间采样为无样本；这不能说明不存在单个慢步，也不替代独立测速的逐步分层。不根据高util单独认定无瓶颈。

## 7. 训练过程行为

第0、100、200、300步的loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm全部有限，记忆编码器梯度均非零。第300步loss=0.0239966679、grad_norm=0.207800344、mem_enc_norm=0.0124392305。真实输入static_mask为(128,2048) bool。原生指标记录保留于 [metrics.startup300.jsonl.gz](records/metrics.startup300.jsonl.gz)，清洗日志为 [startup300.summary.log](records/startup300.summary.log)。

上述前300步快照在当时尚未到首次周期保存；完整训练随后正常结束。原生指标现在覆盖0、100、…、79900共800行，每项dec/hex一致，最后记录loss=0.001246184343、grad_norm=0.02127394639、mem_enc_norm=0.001548738568；最后10条loss均值0.001250365062。step79900是对应日志区间的统计，不能称作最终79999单步loss。

checkpoint集合为5000至75000每5000一步，加79999，共16份、190052280479字节。每份提交时间有效，61叶参数元数据形状一致、OCDBT清单与数据文件非空、norm_stats文件SHA一致；清单见 [checkpoint_inventory.json](records/checkpoint_inventory.json)。最终79999于2026-09-22 14:49:05→14:49:26 UTC经既有check_config_provenance.py真实CPU读取和模型加载，tmux全名m2048-finish-l0-20260922，21秒、EXIT_CODE=0；NORM_STATS_SAME、LIB_PROVENANCE_MATCH、MOTION_STORE_PATH、CKPT_PARAM_TREE、TIC_L0全PASS。原始dtype为38个float32可训练叶和23个bfloat16冻结图像叶。见 [加载结果](records/completion_l0.json) 和 [加载日志](records/completion-l0.summary.log)。

完整指标见 [metrics.full.jsonl.gz](records/metrics.full.jsonl.gz)，清洗日志 [train.complete.summary.log](records/train.complete.summary.log) 保留800条Step记录及唯一EXIT_CODE=0，原日志压缩见 [driver.full.log.gz](records/driver.full.log.gz)。训练会话m2048-prod、训练PID3999606、GPU采样PID3999598均已自然退出；W&B确认同步完成。未删除或覆盖任何训练checkpoint。

## 8. 训练后评估

策略评估尚未开展。完整训练与最终权重加载已经通过，但loss下降不证明策略成功率；与60k/motion80k的策略比较按原计划后续另立任务，不在本轮合并与收尾中启动rollout。

## 9. 用户决定记录

「保持原计划，完整取证（推荐）」；「能并行的尽可能并行 8个gpu你可以用」；「1000步测速后如果占用率高于50% 并且没有其他的问题 可以直接启动训练」；「你要推送到hongzefu的位置 不可以用yinpei tri」。

2026-09-22用户要求「完成合并」，并补充「和训练的收尾」。据此完成最终加载、全部指标和保存验收、档案提交、主副本同步和本轮临时开发副本清理。

条件判定采用预先说明的800步稳态八卡总体均值>50%，全部正确性、容量通过且测速READY，无未解决问题。实际98.6104%满足条件，因此固定Beta后直接起跑。approval的approved_at是条件成立后的授权落档时间，不冒充用户当时新增回复。Git推送沿用主副本已验证的hongzefu SSH命令，不修改凭据或upstream。

## 10. 计划外事件与处置

正式起跑后未发生错误。起跑前独立配置解析首次未source paths.sh，触发与预览不一致的断言，未写授权或启动训练；按真实入口加载同一paths.sh后精确匹配，V10又独立通过。测速归档核验曾发现清洗driver日志的original_sha字段误填清洗结果，Beta提交前已以真实原日志校正并回读验证；测量本身不变。更早的两次验证器修复及完整重测记录见总实施档案。

主仓解除写保护后，另一会话新增了scripts/dataset/hf_export/run_m2048_80k_ckpt_export.sh与配套说明，随后独立提交并推送为54a48d3fd37312bc706500bb52d24f3e8d5b4253（commitV11.3）。本轮完整保留该提交，只提交训练档案与计划状态，不把其导出实施或结果算入本轮验证。merge_verification保留首次复核时两文件仍在暂存区的真实状态及后续提交信息。训练结论仍与原Beta配对为V11.2，不因并发V11.3而更改配对编号。

## 11. 结论与下一步

本轮实现、完整取证、容量、1000步测速、正式80k及训练收尾均完成。主仓已恢复为可开发的权威副本，全部既有分支提交通过快进合并，没有代码冲突。最终曲线见 [W&B](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/u5qogzm9)；策略评估仍待后续。

本轮v1-store/workspaces/m2048-tool-fix独立副本已删除：删除前确认Git clean、无stash/独有分支/额外worktree/占用进程，全部提交在主仓与hongzefu远端；其中237个专用模型和缓存文件（逻辑12449071571字节）原样移到v1-store/bench/m2048/merged-tool-fix-store-20260922，目录inode保持不变。主仓v1-store inode和16份checkpoint元数据、norm_stats均未变。详见 [合并前源码核验](records/merge.before.json) 与 [合并清理记录](records/merge_cleanup.json)。

旧的/scratch/hongze/robomme_policy_learning_MotionJEPA-temp停在5a66e4e0eee7083beb6f85f8c01460b3a9775a39，全部提交已经包含在主仓。核对时该目录仍被PID3710284、3831858的其他会话使用，因此保留其目录、HEAD和v1-store链接，不结束会话。它没有待合并的独有改动；待会话迁出后才可另行清理。

## 12. 归档文件

records保存launch.actual、approval.actual、preflight、runtime、startup300、原生指标/GPU压缩快照、清洗日志和source_lock。原始产物仍在主库v1-store/bench/m2048及train-runs对应run；无权重或bash/yaml副本入Git。完整文件SHA与gzip回读SHA见 [archive_manifest.startup.json](records/archive_manifest.startup.json)。本档案的后续提交保留起跑Beta锚点。

结束归档新增training_completion、checkpoint_inventory、completion_l0及日志、metrics.full、gpu_util_500ms.full、performance.full、driver.full、train.complete.summary、run_meta.final、merge.before和merge_cleanup；前300步快照均保留。结束产物SHA与gzip回读摘要见 [completion_archive_manifest.json](records/completion_archive_manifest.json)，合并后的复核见 [merge_verification.json](records/merge_verification.json)。原始大权重继续留在v1-store。
