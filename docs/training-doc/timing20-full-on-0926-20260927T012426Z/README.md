# timing20-full-on-0926-20260927T012426Z

实际训练Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`（commitV11.16Beta），实施锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`。377项核心合测属于起跑前工具验证；本档案另记录本run及确定性配对的真实结果，后续归档提交不替代运行锚点。

## 1. 结论与指标速览

本run为full/on确定性20步包装对照，实际完成21次取批、20次模型输入/更新与末步真实保存；checkpoint19/state20的61个EMA叶CPU恢复一致。本库正式off/on判定PASS：100个标量对及201叶完整TrainState零差异，输入与RNG逐位相同。结论仅限本次deterministic100及共同观察层，旧normal失败保留，不作性能结论。

## 2. 版本与代码状态

本轮四run和两judge均在clean Beta `c71d5255597db2f26930b7b5684a1d5b2994cf75`下执行；运行期间源码及uv环境冻结，主代理在2026-09-27T02:07:11Z记录完成快照后解除冻结。原始记录继续绑定该Beta，不改写为归档后的工作区。

INPUT原HEAD为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1/100量具与B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。沿用范围需由新Beta前置核对模块、依赖和数据证明，不宣称在本新HEAD重跑历史INPUT/100。旧normal结果仅引用[group历史边界](../orig80k-timing20-det-0927/README.md)。

## 3. 启动与配置还原

实际执行须使用[group launch完整函数与六任务命令](../orig80k-timing20-det-0927/launch.md)的门闩、独立退出及联合验收，下面列本run实际固定参数，不能绕开group控制链直接起跑：

```bash
TRAIN_HEAD='c71d5255597db2f26930b7b5684a1d5b2994cf75'
RUN=timing20-full-on-0926-20260927T012426Z
TIMING=on
GPUS=0,1,2,3
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
```

run runtime第一行由Bash `printf %q`生成 `set -- "$1" <固定参数...>`，第二行固定 `export ORIG80K_TIMING_EQ_PROFILE=deterministic100`，之后附原45b4模板完整字节；不依赖tmux继承控制shell环境。整份runtime新SHA、两行前缀及原模板余体SHA均留证。模板真实调用 `run_orig80k.sh smoke "$RUN" "$GPUS" "$LIB" "$ASSETS"`，显式 `ORIG80K_SMOKE_EQ_MODE=on`；成功后串行执行真实CPU completion。COMMON始终为未加前缀的原模板实体，judge runtime第二行unset selector，只从记录判定档位。

Beta形成后用 `git show <TRAIN_HEAD>:scripts/training/prod/run_orig80k.sh`、`git show <TRAIN_HEAD>:scripts/training/tests/timing20-wrapper.sh`及 `git show <TRAIN_HEAD>:docs/training-doc/orig80k-timing20-det-0927/launch.md`还原。正式源、工作树及v1-store runtime逐字绑定；两模板SHA45b4/17c0与guard f8不变。

## 4. 数据集与划分口径

本run只读 `$LIB/framesamp` 的4×4 packed与原manifest。full为1600集、768897帧、476857执行样本；继续使用原版f332 norm，不能替换为full自算c5统计。 构建锚点为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`；见[数据档案](../../dataset-build-doc/16task-pub-1600ep/README.md)。

norm实体为 `$ASSETS/robomme/norm_stats.json`，期望SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。不新建划分，不缩数据量、不改变采样顺序或精度规则。

## 5. 关键超参

| 配置 | 固定值 |
|---|---|
| steps / batch / FSDP / workers / seed | 20 / 64 / 4 / 4 / 42 |
| history | modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024 |
| history SHA256 | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| warmup / peak_lr / decay_lr / decay_steps | 10000 / 5e-5 / 5e-5 / 100000 |
| optimizer / EMA | AdamW clip1.0 / 0.999 |
| log / save / keep | 100 / 10000 / 10000 |
| W&B / 初始权重 | online、project openpi / pi05_base |
| timing_eq_profile | `deterministic100` |
| 实际XLA_FLAGS | `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` |

仅smoke覆盖steps20；`streaming_obs_horizon=16`维持原值，与最多32 memory帧区分。新增档位仅获准用于20步off/on对照；perf/prod仍normal。清除CPU平台遗留，不盖写继承的JAX_ENABLE_X64，实际True必须拒绝。

## 6. 硬件、调度与耗时

AWS8×A100-SXM4-80GB、/scratch md0 XFS；本run固定GPU 0–3，同库off→on按真实恢复及独立退出依赖推进。UTC为 `2026-09-27T01:53:39Z` → `2026-09-27T02:04:25Z`。

会话 `orig80k-timing20-full-on-0926-20260927T012426Z`，window/pane `@578/%578`，pane PID `1296146`；UUID `4f142027-87fa-45ea-9462-ae6e4f7e73f6`，W&B ID `bngtdcfw`。这些身份由[原生identity](../orig80k-timing20-det-0927/records/exit/orig80k-timing20-full-on-0926-20260927T012426Z/identity.json)和原始记录绑定；共同取证存在同步/CPU读回开销，不把本窗口用作吞吐或ETA。

## 7. 训练过程行为

on运行speed.run(smoke20)，speed_start/run均schema2，profile与实际flags相同；真实同步、主机/磁盘/GPU采样保留。实际取批21次、仅前20批进入模型并更新；一次原save_state返回，末态step20/loop19。完整状态为params61、EMA61、optimizer78、step1，共201叶，所有标量和状态有限。

log100的step0与final尾窗1…19覆盖20×5原始标量，未改变日志间隔。独立核验覆盖52个项目模块及208项依赖，launch与metadata均显式deterministic100；实际flags为精确两项。

## 8. 保存、恢复与配对验收

[本run独立报告](../orig80k-timing20-det-0927/records/verification/full-on.independent-verification-v2.json)确认checkpoint19/state20实际恢复61个EMA叶，与final及共同完整状态的EMA一致；恢复范围不冒称包含全部optimizer。

[本库正式judge](../orig80k-timing20-det-0927/records/judges/full.json)逐位通过21fetch、20model/RNG、100个标量对及201叶完整状态，schema2与deterministic100均明确。四run和两judge的原生pane均退出0且无信号，capture父实际返回0；七调度阶段的父进程与宿主返回0另由[最终controller快照](../orig80k-timing20-det-0927/records/controller.completed.snapshot.json)绑定。日志0不单独作为最外成功证据。

## 9. 用户决定记录

用户要求「两库各跑一对 20 步（推荐）」「尽可能并行做」「你有8张卡」，并已同意本轮20步采用既有100步确定性档；不改逐位判据，perf/prod正常环境不变。P1/100仅对拍关闭W&B的授权不延伸到本run，本run保持online与真保存。

「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及四None键等价授权保留其CPU输入范围，不放宽本训练对照。「继续工作 一路做到起泡前 有问题问用户」限定本轮止于正式80k起跑前。

## 10. 计划外事件与退出边界

旧normal off/on及追加off诊断的差异保留原始FAIL，只引用既有档案，不归因autotune或包装、也不预言本档位必过。旧INPUT/P1最外final append实际退出不可追补，分别按「同样补记限制，沿用INPUT结果（推荐）」及「补记P1限制，补独立退出取证后继续（推荐）」处置；其内容与子命令证据保留，不补造独立wait。

本轮六任务都由f8先保留自有窗口并落identity后释放，结束联合capture及真实父返回；日志0不能单独证明最外实际成功。本run没有训练/恢复失败。后续独立报告封装的大整数精度事件见[group结果](../orig80k-timing20-det-0927/result.md)：原记录、原checker stdout及真实判定未改，四份修正v2报告单独保留，首版不导入。

## 11. 当前结论与下一步

本run真实完成及本库确定性包装对照PASS。该结论只证明同一额外观察层、指定两flags、固定模型与输入下本次20步逐位相等；不能推广normal环境、一般确定性、无观测运行或性能。后续仍须normal300步perf、真实测盘预算及正式起跑前条件，本轮不自行启动80k。

## 12. 归档文件清单

[核心原字节包](records/core.records.tar.gz)共22成员，170114 B，SHA `8fed9d894f75f140d79eec11b913371e7d219ae5d2d7f3519c55af66634d349c`；[成员检查](records/archive_checks.json)记录逐成员源bytes/SHA、完整解压CRC及退出证据。

清洗日志为[driver](logs/driver.summary.log)、[wrapper](logs/wrapper.summary.log)、[completion](logs/completion.summary.log)，并附[清洗检查](logs/log_checks.json)。仅删除已识别独立进度行，Git副本仅处理获批纯W&B装饰尾白；其他行及EOF保真，关键行零丢失。group保留六任务独立退出链、父过程、最终结果和精确v2报告。权重留v1-store，脚本/配置/运行命令不作为独立docs附件。
