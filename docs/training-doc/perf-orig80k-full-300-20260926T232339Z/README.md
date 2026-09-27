# perf-orig80k-full-300-20260926T232339Z：300步perf起跑前档案（待V11.17Beta，未运行）

## 1. 一句话结论与指标速览

**本run仅完成名称及起跑前档案准备，尚未启动300步perf，实际Beta待commitV11.17Beta生成。** 根代理已采用名称 `perf-orig80k-full-300-20260926T232339Z`；准备标签 `20260926T232339Z`不是实际开始UTC。本run拟与另一库各4卡同时训练300步，W&B online、真实保存并CPU恢复299。没有本run吞吐、ETA、GPU/磁盘峰值或通过结论。

| 项目 | 计划值或当前状态 |
|---|---|
| 数据 / GPU | 公开16任务全集 / `0,1,2,3` |
| steps / batch / FSDP / workers | 300 / 64 / 4 / 4 |
| 预热 / 稳态 | 0…99 / 100…299，共200稳态步 |
| checkpoint / 真实恢复 | 末步299；当前未产生 |
| perf TRAIN_HEAD、UUID、W&B ID、START/END | 均待真实记录 |

## 2. 版本与代码状态

正式perf必须另建独立Beta，两个run与后续8阶段共用其实际完整40位TRAIN_HEAD并保持clean；该SHA尚未生成，不能将准备时主仓、旧100或新20提交当作本run起跑版本。现行接口锚点为已提交实现 `aec86db64e5178e63d9e7f77d3f5bc235db16390`：runner/contract/speed/observer及3份测试共7文件已纳入显式档位/schema2，377项核心验证通过。不得再把旧00bd/b0量具当作新接口，或声称相对旧normal版本只有文档变化。未来perf Beta与aec及实际确定性20版本的受保护代码/依赖要求精确相同，数据/资产和环境指纹另行核对。

既有INPUT保持 `3a1582db39c723c735e04752e5027bfe40ecc3e1`，P1/100量具与B保持 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A保持 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。两库确定性20实际Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`，准备TAG `20260927T012426Z`；两库原正式judge均为schema2/deterministic100 PASS，root已核七phase宿主/原生/capture均0；最终独立报告及归档路径/SHA由root补齐。见[确定性20实测结果](../orig80k-timing20-det-0927/result.md)。旧normal失败不改写，本normal perf仍未执行。

本档案用于独立V11.17Beta起跑前登记；当前未运行perf，没有实际run UUID、W&B ID或起止UTC。新Beta必须仅含获准文档，运行时源码和venv冻结。

## 3. 启动与配置还原

[共享launch](../orig80k-perf300-0927/launch.md)锁定完整8阶段控制正文、父适配及实际CLI。该run的内部调用是 `run_orig80k.sh perf "$RUN" "$GPUS" "$LIB" "$ASSETS"`，只覆盖perf步数300，不带额外训练超参。固定配置展开如下，仅是变量定义：

```bash
RUN=perf-orig80k-full-300-20260926T232339Z
GPUS=0,1,2,3
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
```

训练session拟为 `orig80k-perf-full-20260926T232339Z`；实际session/window/pane/PID与开始UTC由新门闩执行时记录。所有阶段经f8 guard先retain和identity后释放，记录原命令文件及整文件SHA。guard仍执行原v1-store冻结副本，每阶段核正式Beta Git源、工作树和runtime字节相同，并将绑定写入外部launch/capture记录；不能直接把GUARD移出v1-store。

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

现场目标为AWS 8×A100-SXM4-80GB，/scratch为md0 XFS本地NVMe；本run固定物理GPU `0,1,2,3`，两库同时4+4，分别隔离记录、checkpoint及JAX/CUDA/W&B缓存。实际GPU UUID、占用/进程、CPU、内存、SHM、IO压力、存储和可用字节仍须起跑前重测，历史快照不是资源保证。

稳态从step99完成同步到step299训练完成且保存之前的同步，单侧吞吐为 `200*64/steady_seconds`。GPU按0.5秒目标间隔记录均值、0%占比及慢步/其他步分层，慢步为host耗时超过稳态中位数2倍；中位数只用于分类。两库合计用两个稳态窗口并集跨度计算，不能简单相加不同时间窗的单侧速率。必须证实实际稳态正重叠，否则不能作本轮4+4结论。

实际数值全部待report生成。初始化/JIT/预热、保存提交、保存至收尾分别报告；含W&B收尾的跨度不叫纯磁盘写时，一次末步保存外推的80k ETA不是承诺。

## 7. 训练过程、计时与磁盘采样

runner原样构造训练ARGV，由speed.run将计时钩子接入HostTiming后进程内执行原入口；不调用JAX profiler。同步只用于既定窗口边界，单步host提交耗时不冒充GPU内核时间。保留真实save_state和wait，300步结束只留checkpoint299。

0.5秒只读采本run整个checkpoint根与/scratch可用量，首尾补采并覆盖异步wait结束。分配量按 `st_blocks*512`及device/inode去重，临时/最终文件都计入、不跟随软链；保留采样间隔、扫描延迟、漏tick、路径竞态及错误。它不是原子扫描，也不识别XFS reflink共享extent；采样最大值是观察下界，不是连续真实峰值。

保存窗口必须绑定本run原生commit纳秒与wait/UUID/HEAD，不能把W&B收尾样本冒充保存期样本；真实I/O错误或保存期无覆盖不能放行预算。本perf所有记录均未生成；确定性20和旧normal差异各自保留其真实原档案，不回写数值或宣称根因已解决。

## 8. 真恢复、联合报告与measurement验收

必须先等两个GPU run都结束，各自原生pane退出、capture父真实0、driver及wrapper终态均通过，再按full→count顺序执行CPU完成器。要求state_step300/loop_step299、checkpoint集合[299]、metrics步[0,100,200]和尾窗201…299共99步五标量完整有限；run/HEAD/UUID/W&B身份与异步wait、norm一致。真正恢复299/params并与该run末步EMA逐叶比较，不用文件名或旧恢复记录替代。

GPU收尾后先核本run launch与两份speed schema2显式normal，包含真实300步、来源、argv、成功状态和保存计数；两侧恢复均通过才生成含peer的联合perf报告。报告后再按checkpoint_root唯一归属核两侧normal descriptor、实际flags与HEAD，才能继续测量。report本身无schema字段，不能把speed的schema2要求误写到report外层；两份真实299仍在时，分别调用 `orig80k_contract.py measure-checkpoint`，回执写各自REC/checkpoint_measurement.json，不能放通用RESULT_ROOT。回执绑定report、completion、原始采样、launch及原生元数据；小证据归档核验之前不讨论清理。本档案不安排删除，恢复/report/measurement/预算均未执行。

## 9. 用户决定记录

用户目标为原版完整80k与纯counting80k、4卡+4卡并跑并先对拍；「尽可能并行做」「你有8张卡」落实为两份GPU perf同时进行。100的S1/S3真单跑判据保持，不因为利用率把旧基线改成并跑。

「允许仅对拍关闭 W&B」只适用于P1/100，perf仍online。用户批准「补充 0.5 秒只读采样」，要求报告观察峰值并保守预留；本档案不选择余量倍率或容量。用户允许确定性档用于20步正确性对照，perf/prod继续normal，逐位判据不改；即使确定性档通过也不改写旧normal20及追加off差异结果。「继续工作 一路做到起泡前 有问题问用户」把本轮终点限定为正式80k起跑前，perf通过或预算通过都不直接起80k。

## 10. 历史限制与失败处置

旧INPUT/P1最外final append之后实际退出不可追补，分别按「同样补记限制，沿用INPUT结果（推荐）」及「补记P1限制，补独立退出取证后继续（推荐）」保留原始证据及限制；没有实际失败新证据，不补造wait。原perf wrapper的16例纯shell注入中15例符合预期、append_after反例确认日志0不能独立证明最外成功，此历史测试不改写。

本阶段所有8任务使用新guard门闩、原生pane、独立capture实际返回和sidecar联合验收；还须保存宿主观测的父适配真实返回。失败或PENDING停止依赖步骤并保留所有产物，不自动覆盖、改名、清理或放宽数值/采样判据。wrapper日志与driver分别保留，外层不追加driver终态。

## 11. 当前结论、预算待填与下一步

当前仅有起跑前档案。det双judge PASS已确认，根代理仍须补齐最终归档与无损独立报告、四run完成器和六退出链的引用/SHA，并按共享launch在任何mkdir前执行固定d3f只读预检；独立V11.17Beta；现场资源与完整剩余空间；实际START/END、run UUID和W&B ID。现有perf代码300GiB最低保护不能冒充完整新增占用预算，须结合新20实际checkpoint和现场既有占用核算，不据旧11GiB结果自动通过。

真实perf、恢复、report及receipt之后，三项margin分别登记 `checkpoint_growth_margin_bytes`（非负整数）、`save_sampling_margin_bytes`（正整数）、`logs_cache_margin_bytes`（正整数）及可定位basis；**本档案不填写任何数值或默认倍率**。预算公式为两侧各8份单checkpoint真实分配量加增长余量、两侧采样额外量下界加采样余量，再加日志/cache余量，最终与300GiB取max。相同schema2预算和SHA供未来两prod共同引用；预算实际来源和人工合理性核对不能省略。

## 12. 归档文件清单

本档案保留起跑前README及[共享launch](../orig80k-perf300-0927/launch.md)；实际run输出尚未创建。未来runner记录根为 `v1-store/bench/orig80k/perf-orig80k-full-300-20260926T232339Z`、checkpoint根为 `v1-store/train-runs/mme_vla_suite/perf-orig80k-full-300-20260926T232339Z`、driver为 `v1-store/logs/perf-orig80k-full-300-20260926T232339Z.driver.log`，各自cache/W&B目录按run名隔离。

待真实运行后归档launch/runtime/run_meta、metrics/final、speed_start/speed_run、step_timing/host_samples/disk_samples、gpu.csv/.err、completion、checkpoint_measurement、原始及清洗日志、命令/源码/runtime摘要、guard identity/native receipt/父sidecar及检查链；联合report、三margin来源和schema2预算放公共结果根。保留原始小记录供权重清理后重算审查，不能用清洗日志覆盖其绑定源。Git能还原的脚本/yaml及大权重、venv、缓存不复制入docs；实际时间、UUID、W&B ID和结果只按现场回填。
