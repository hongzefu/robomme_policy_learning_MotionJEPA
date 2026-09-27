# timing20-full-on-0926-20260927T012426Z

已形成并推送的实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`；它锁定本轮已批准档位实现，377项核心合测通过。下一档案Beta确定为 `commitV11.16Beta`，其实际TRAIN_HEAD仍待生成；实施锚点不能冒充尚未发生的训练起跑版本。

## 1. 结论与指标速览

本run为确定性档20步包装对照的full/on侧，**尚未起跑，没有训练、恢复或等价结论**。准备标签 `20260927T012426Z` 不表示实际起跑时间。预期验收21次取批、20次模型输入/更新、五标量20步逐位比较及完整TrainState，真实保存checkpoint19并CPU恢复EMA；实际结果待回填。

## 2. 版本与代码状态

本轮TRAIN_HEAD待新Beta真实生成后填写完整40位SHA，预计编号 `commitV11.16Beta`，以实际Git记录为准。运行前HEAD必须与另外三run、两个judge一致且工作区clean；运行中源码和uv环境冻结。不能把准备时HEAD、归档提交或旧normal版本写成这次运行锚点。

INPUT原HEAD为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1/100量具与B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。沿用范围需由新Beta前置核对模块、依赖和数据证明，不宣称在本新HEAD重跑历史INPUT/100。旧normal结果仅引用[group历史边界](../orig80k-timing20-det-0927/README.md)。

## 3. 启动与配置还原

实际执行须使用[group launch完整函数与六任务命令](../orig80k-timing20-det-0927/launch.md)的门闩、独立退出及联合验收，下面仅列本run待展开的固定参数，不能绕开group控制链直接起跑：

```bash
TRAIN_HEAD='<待新Beta生成后填写实际完整40位SHA>'
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

现有机器记录为AWS 8×A100-SXM4-80GB及/scratch md0 XFS，起跑前须重核现场空闲与预算。本run固定GPU `0,1,2,3`，同库配对侧用相同四卡；full/count各自off→on独立推进。必须等本库off真实恢复及独立退出全部通过后起跑，不等待另一库链。

会话名固定 `orig80k-timing20-full-on-0926-20260927T012426Z`；实际START/END_UTC、window/pane/PID、UUID、W&B ID及耗时全部待运行记录。共同观测存在同步和CPU哈希开销，不将20步耗时写成吞吐、ETA或性能结论。

## 7. 训练过程行为

on在相同共同取证层下执行speed.run(mode=smoke)，保留HostTiming与原同步点，step19保存前同步；0.5秒磁盘采样含异步保存wait完成后的末次样本。 两侧均必须真实保存，不用摘要器替代save_state。预期共同记录21次fetch、20次model/RNG，最后一批未用于训练；完整末态覆盖params/EMA/opt_state/step及每叶结构、dtype与原始字节摘要。

log100的step0与final尾窗1…19覆盖全部20×5标量；不改log_interval，不用四位小数或W&B摘要替代原始hex逐位判定。此节为待执行判据，尚无实际训练行为。

## 8. 保存、恢复与配对验收

完成器须真实恢复checkpoint19的EMA，与训练末步及共同完整状态EMA摘要相等；EMA恢复范围不冒称覆盖全部optimizer/params状态。completion必须为smoke/state20/final19/checkpoints[19]，绑定run/HEAD/UUID、GPU窗口和全部小文件。

同库[配对off侧](../timing20-full-off-0926-20260927T012426Z/README.md)完成后运行原正式judge，严格比较21fetch、20model/RNG、100对标量和完整TrainState。不加容差、不改BASE或跨HEAD正式判据。联合验收还必须核launch/metadata及judge结果的 `timing_eq_profile={name:deterministic100,xla_flags:精确两flags}`，on两份speed JSON为schema2且与记录一致；不能凭任意profile的PASS放行。

每run和judge均要原生pane实际退出、capture父实际返回、父适配宿主退出及原日志内容联合通过。全部结果当前待验证。

## 9. 用户决定记录

用户要求「两库各跑一对 20 步（推荐）」「尽可能并行做」「你有8张卡」，并已同意本轮20步采用既有100步确定性档；不改逐位判据，perf/prod正常环境不变。P1/100仅对拍关闭W&B的授权不延伸到本run，本run保持online与真保存。

「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及四None键等价授权保留其CPU输入范围，不放宽本训练对照。「继续工作 一路做到起泡前 有问题问用户」限定本轮止于正式80k起跑前。

## 10. 计划外事件与退出边界

旧normal off/on及追加off诊断的差异保留原始FAIL，只引用既有档案，不归因autotune或包装、也不预言本档位必过。旧INPUT/P1最外final append实际退出不可追补，分别按「同样补记限制，沿用INPUT结果（推荐）」及「补记P1限制，补独立退出取证后继续（推荐）」处置；其内容与子命令证据保留，不补造独立wait。

本轮六任务都由f8先保留自有窗口并落identity后释放，结束联合capture及真实父返回；日志0不能单独证明最外实际成功。新意外尚无；失败保留现场、停止依赖，不覆盖或自动清理。

## 11. 当前结论与下一步

当前仅完成启动候选准备，本run尚无PASS/FAIL结果。须先形成统一新Beta、核前置沿用和现场资源，再执行本run、同库配对和正式judge；只有确定性档全部严格通过才能讨论后续normal300步perf及预算。不能把确定性档结果回写成旧normal逐位通过，不自动启动80k。

## 12. 归档文件清单

本候选现有文件仅README和[group launch](../orig80k-timing20-det-0927/launch.md)。未来run根为 `v1-store/train-runs/mme_vla_suite/timing20-full-on-0926-20260927T012426Z`，records为 `v1-store/bench/orig80k/timing20-full-on-0926-20260927T012426Z`，driver为 `v1-store/logs/timing20-full-on-0926-20260927T012426Z.driver.log`，wrapper为 `v1-store/logs/orig80k-timing20-full-on-0926-20260927T012426Z.wrapper.log`；均要求新目录/新文件，当前不得预建。

待真实产生后归档launch/runtime、metrics/final、timing_eq四记录及manifest、completion、GPU采样、speed/host/disk记录、清洗日志及原字节检查链，另保留六任务identity/native receipt/父capture及实际返回。runtime命令和cache留v1-store；docs只纳Git不可还原实测，不复制.sh/.yaml/venv或大权重。待产生清单不表示已有记录。
