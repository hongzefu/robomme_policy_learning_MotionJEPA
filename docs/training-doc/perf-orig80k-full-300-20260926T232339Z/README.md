# perf-orig80k-full-300-20260926T232339Z：normal300步perf实际结果

## 1. 一句话结论与指标速览

本run已完成normal300步、真实保存299及61EMA叶CPU恢复，联合report、measurement及已批准B预算通过。稳态100–299共200步，192.639424秒，**66.445381 samples/s**。两库合计按窗口并集计算，不直接相加单侧速度；本结果不改写旧normal逐位重复性FAIL。

真实Beta `70a641a827eb2559c3acb69430f55c1c53babaeb`，UUID `9bc82a7e-1f03-496a-824c-600d5eabd6d6`，W&B ID `ar2nmtrt`。准备TAG `20260926T232339Z`不是实际开始时间。

## 2. 版本与代码状态

两run与八个阶段实际使用clean Beta `70a641a827eb2559c3acb69430f55c1c53babaeb`（commitV11.17Beta）；运行中源码及venv冻结，最终完成快照于2026-09-27T04:24:31Z记录后解除冻结。现行实现锚为aec86db，确定性20运行锚c71、结果归档29f71均保留自身范围；本阶段受保护源码/依赖没有新增变化，归档提交不替代运行版本。

INPUT仍为3a1582d、P1/100仍为00bd/ecf，已批准最外退出限制保留。旧normal20及off复验FAIL未被本次性能/恢复结果改写；确定性20的严格PASS仅作为已批准包装正确性闸门。

## 3. 启动与配置还原

[共享launch](../orig80k-perf300-0927/launch.md)锁定完整8阶段控制正文、父适配及实际CLI。该run的内部调用是 `run_orig80k.sh perf "$RUN" "$GPUS" "$LIB" "$ASSETS"`，只覆盖perf步数300，不带额外训练超参。固定配置展开如下，为实际参数还原：

```bash
RUN=perf-orig80k-full-300-20260926T232339Z
GPUS=0,1,2,3
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
```

实际训练session为 `orig80k-perf-full-20260926T232339Z`，window/pane `@581/%581`，pane PID `1325331`；起止UTC为 `2026-09-27T02:18:30Z` → `2026-09-27T02:27:56Z`。所有阶段经f8 guard先retain和identity后释放，记录原命令文件及整文件SHA。guard仍执行原v1-store冻结副本，每阶段核正式Beta Git源、工作树和runtime字节相同，并将绑定写入外部launch/capture记录；不能直接把GUARD移出v1-store。

8阶段为两GPU runner → 两侧都结束并通过退出验收 → full CPU恢复 → count CPU恢复 → 联合report → 两个measurement receipt → budget。measurement为两个独立阶段，因此总数为8。两个CPU恢复顺序串行，不能在另一侧GPU训练尚未结束时抢先执行；其I/O和CPU工作不得污染对方稳态。

## 4. 数据集与划分口径

沿用1600集、768897帧、476857执行样本的既有库，建库实际Beta为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`；不改变划分或减少数据以提速。训练读 `$LIB/framesamp`（4×4），同时绑定 `$LIB/source`和 `$LIB/meta/episode_manifest.json`。来源、packed全量校验与子集边界见[本库档案](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/dataset-build-doc/16task-pub-1600ep/README.md)。

norm实际路径是 `$ASSETS/robomme/norm_stats.json`，期望SHA为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。正式使用原版f332统计量，不用完整库自算c5d45b…替换。 两run之间参数不要求相等；每run恢复只能与其自身训练末步EMA比较。

## 5. 关键超参

| 参数 | 固定值 |
|---|---|
| 配置 / 初始化 | mme_vla_suite / v1-store内pi05_base |
| steps / batch / FSDP / workers / seed | 300 / 64 / 4 / 4 / 42 |
| history | perceptual-framesamp-modul.yaml；modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024 |
| history SHA256 | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| warmup / peak_lr / decay_lr / decay_steps | 10000 / 5e-5 / 5e-5 / 100000 |
| optimizer / EMA | AdamW clip1.0 / 0.999 |
| log / save / keep | 100 / 10000 / 10000 |
| W&B | online，project_name=openpi |

原history的streaming_obs_horizon16保持，不与最多32 memory帧混写。启动清除CPU平台遗留、确定性XLA_FLAGS、旧TRAIN_TIMING_STEPS及两个selector ORIG80K_SMOKE_EQ_MODE/ORIG80K_TIMING_EQ_PROFILE；不覆盖继承的JAX_ENABLE_X64，启用时让preflight拒绝。实际GPU_ENV与CPU_ENV都含 `env -u ORIG80K_TIMING_EQ_PROFILE`，不靠tmux继承控制shell。perf不安装20步共同输入/全state取证层，也不使用100步摘要器取消真实保存。normal显式描述符为 `{"name":"normal","xla_flags":""}`；launch完整环境需两个selector为未设、实际XLA_FLAGS未设或空，speed_start/speed_run必须schema2且档位、flags、HEAD和完整argv一致，不能缺字段回退normal。

## 6. 硬件、调度与性能口径

AWS8×A100-SXM4-80GB，md0 XFS本地NVMe；本run固定GPU 0–3，两侧独立输出/cache并行。实际稳态为100–299，共200步，192.639424秒、平均0.963197秒/步、66.445381 samples/s；窗口与吞吐按正式report同步边界计算。

逐卡util均值：GPU0 98.764%, GPU1 98.569%, GPU2 98.501%, GPU3 98.808%；0%样本占比均0。本侧没有超过稳态中位host步时2倍的慢步，slow层样本0、均值null，不能写成慢步利用率0；其他步层即上述均值。重复相邻读数及实际间隔见[联合报告](../orig80k-perf300-0927/records/paired-speed.json)，高util不证明不存在瓶颈。

初始化/JIT/预热263.200秒，save dispatch 6.555秒，save至入口收尾63.631秒；最后一项包含额外收尾，不叫纯磁盘写时。原公式80k ETA约21.592小时，只是一次300步和末步保存的外推。

## 7. 训练过程、计时与磁盘采样

实际speed_start/run均schema2且显式normal，300条step_timing闭合，真实save与wait完成。普通metrics步为[0,100,200]，末尾201…299的99步五标量完整有限；不把未逐步记录的其余loss虚写成已有日志。最终EMA61叶有限，perf未安装20步共同输入/完整201叶观察层。

磁盘目标0.5秒，实际1004样本，间隔均值0.512782秒、最大3.757910秒，missed ticks 26；最大扫描3.519036秒。P=16039718912 B、F=11879968768 B；它们均整run根，峰值只是离散非原子观测下界。路径竞态样本0，正式工具按原口径接受且保留该事实，不能隐去采样盲区。

## 8. 真恢复、联合报告与measurement验收

双方GPU结束并原生退出后才顺序执行full/count CPU恢复。本侧恢复UTC `2026-09-27T02:28:00Z` → `2026-09-27T02:29:14Z`，唯一checkpoint[299]/state300，61EMA叶真实加载后逐叶等于自身final；不要求两库参数互相相等。

联合report与[清理前measurement](records/checkpoint_measurement.json)已通过：单299目录allocated=11879944192 B，整个run根=11879968768 B，两者不能混称单份大小。receipt绑定原始小文件/采样/原生元数据，且在权重仍存在时生成；没有删除后补造或再次运行测量。见[group八阶段结果](../orig80k-perf300-0927/result.md)。

## 9. 用户决定记录

用户目标为原版完整80k与纯counting80k、4卡+4卡并跑并先对拍；「尽可能并行做」「你有8张卡」落实为两份GPU perf同时进行。100的S1/S3真单跑判据保持，不因为利用率把旧基线改成并跑。

「允许仅对拍关闭 W&B」只适用于P1/100，perf仍online。用户批准「补充 0.5 秒只读采样」，要求报告观察峰值并保守预留；用户随后已明确选择B余量，详见第11节及group结果；没有把短测峰值当硬上界。用户允许确定性档用于20步正确性对照，perf/prod继续normal，逐位判据不改；即使确定性档通过也不改写旧normal20及追加off差异结果。此前「继续工作 一路做到起泡前 有问题问用户」作为历史范围保留；用户最新明确「确认无误后可以直接开始两个训练」。因此最终共同Beta/clean、输出和全部前置核查通过后，root已获准直接启动两80k；本perf档案不冒称正式训练已启动。

## 10. 历史限制与失败处置

旧INPUT/P1退出限制、原wrapper 15/16测试与日志0盲区保持原史。本轮八任务均有原生pane退出0、capture父实际0、根调度宿主0；不靠日志自证。

八份wrapper第9行WRAPPER_COMMAND各有一个未转义末尾分隔空格。用户已批准原字节保留；仅[精确八文件清单](../orig80k-perf300-0927/records/wrapper-log-policy.json)允许这一类尾白，逐条shlex argv相同但实际没有删空格。其余进度/W&B清洗及所有其他文件的空白检查不放宽。原PENDING快照留存，不改写成事前批准。

## 11. 当前结论、已批准预算与下一步

normal300性能、真实恢复、联合报告、测量及第八预算均PASS。用户选择B：16GiB/份×16份，另加32GiB采样盲区和64GiB日志/cache工程余量；增长余量84798963712 B，采样余量34359738368 B，日志余量68719476736 B。正式所需383488663552 B（357.151649GiB），执行时可用708412715008 B。

同一[预算JSON](../orig80k-perf300-0927/records/prod-disk-budget.json) SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`供两正式run引用，起跑时仍重核空间及最终共同Beta/clean。按用户最新「确认无误后可以直接开始两个训练」推进，正式80k尚未由本档案执行。

## 12. 归档文件清单

[17成员核心包](records/core.records.tar.gz) 166292 B，SHA `2bffe6d445d1fd62f998b8084de5cdcdb0e43b11e290c0c5bc7f64a25b37ddf0`；[最终检查](records/archive_checks.json)和[独立measurement](records/checkpoint_measurement.json)保留原字节绑定。

[driver摘要](logs/driver.summary.log)及[清洗账](logs/log_checks.json)另存，wrapper和独立退出在group的[八阶段索引](../orig80k-perf300-0927/records/final-perf-archive-index.json)。只归档Git不能还原的小记录；原始日志、checkpoint、cache保留v1-store，没有为归档删除权重或复制.sh/.yaml。
