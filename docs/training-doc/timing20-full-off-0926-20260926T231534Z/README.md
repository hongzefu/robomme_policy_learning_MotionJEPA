# timing20-full-off-0926-20260926T231534Z

## 1. 结论与指标速览

**本run的20步训练、真实保存与EMA恢复通过；同库off/on逐位等价失败。** 本run是公开16任务全集的测速包装off侧，完成21次取批、20次模型使用与更新、checkpoint 19保存；完整TrainState记录201叶且均有限，真实恢复覆盖61个EMA叶。原生pane和capture实际返回均为0。

与[配对on侧](../timing20-full-on-0926-20260926T231534Z/README.md)相比，21批输入及20次模型实参/RNG全部相同，但100对标量中80对不同，完整末态147/201叶摘要不同；正式judge为FAIL，不能宣称包装等价或放行300步perf/80k。名称中的标签是准备标签，实际起止UTC见第6节。

## 2. 版本与代码状态

实际运行Beta为 `commitV11.14Beta`：`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`。四个run均从同一clean Beta起跑，运行及验收期间冻结源码和uv环境；之后的归档提交不替代这一运行版本。起跑前已重核原INPUT的113项文件引用、运行模块/依赖及两库资产小记录；每个runner另完成25项前置与完整资产检查。

已有INPUT为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1/100量具与B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。六run/四judge的100步证据保持原HEAD，本次沿用核验不声称新Beta重跑过旧INPUT/100；见[group版本与前置](../orig80k-timing20-0926/README.md)。

## 3. 启动与配置还原

启动按Beta所锁定的[group launch完整控制正文](../orig80k-timing20-0926/launch.md)，其中包含源码/runtime绑定、参数生成、guard门闩、独立capture及验收函数。本run实际使用的固定展开值如下；用于还原，不是再次启动指令：

```bash
TRAIN_HEAD=b0efbde61e38411fb1b9114eec8485d36a4ee9a0
RUN=timing20-full-off-0926-20260926T231534Z
TIMING=off
GPUS=0,1,2,3
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
```

运行模板调用 `run_orig80k.sh smoke "$RUN" "$GPUS" "$LIB" "$ASSETS"`，显式 `ORIG80K_SMOKE_EQ_MODE=off`；runner成功后同一wrapper串行执行 `check_orig80k_completion.py --mode smoke`真实CPU恢复。实际session为 `orig80k-timing20-full-off-0926-20260926T231534Z`，实际PID/起止时间见第6节。不能跳过group门闩而直接调用内部runner。

还原完整入口与控制正文：

```bash
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:scripts/training/prod/run_orig80k.sh
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:scripts/training/tests/timing20-wrapper.sh
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:docs/training-doc/orig80k-timing20-0926/launch.md
```

正式Git模板为 `scripts/training/tests/timing20-wrapper.sh`，SHA256 `45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f`；本run runtime在v1-store加固定参数前缀后，整文件SHA为 `5e915cf19ecc31b5b047f4d361ed934e723d0847e1767c0bbe04f87b8c8389d8`。guard执行原v1-store f8副本，Beta Git源/工作树/runtime逐字绑定，不复制.sh附件入docs。

## 4. 数据集与划分口径

本库为1600集、768897帧、476857执行样本，构建Beta为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`。训练读取 `$LIB/framesamp`的4×4 packed，source与 `meta/episode_manifest.json`由runner显式绑定。来源和构建实测见[本库档案](../../dataset-build-doc/16task-pub-1600ep/README.md)。

norm文件为 `$ASSETS/robomme/norm_stats.json`，期望SHA为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。继续使用原版f332统计量，不替换为完整库自算c5d45b…文件。 不新增划分，不改变输入、sampler或数据量来缩短测试。

## 5. 关键超参

| 配置 | 固定值 |
|---|---|
| steps / batch / FSDP / workers / seed | 20 / 64 / 4 / 4 / 42 |
| history | perceptual-framesamp-modul.yaml；modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024 |
| history SHA256 | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| warmup / peak_lr / decay_lr / decay_steps | 10000 / 5e-5 / 5e-5 / 100000 |
| optimizer / EMA | AdamW clip1.0 / 0.999 |
| log / save / keep | 100 / 10000 / 10000 |
| W&B / 初始权重 | online、project openpi / pi05_base |

history的 `streaming_obs_horizon=16`仍为原值，不与最多32 memory帧混写。仅smoke覆盖步数20；log/save/keep及其他原版超参不改。清除CPU平台遗留，不静默覆盖继承的JAX_ENABLE_X64；实际启用x64时由runner拒绝。

## 6. 硬件、调度与耗时

环境为AWS 8×A100-SXM4-80GB、/scratch本地md0 XFS；本run固定使用 `0,1,2,3`四卡，UUID已对起跑前指纹逐项核验。两库各自off→真实保存/CPU恢复/独立退出→on推进，两个库不等待无关阶段。

| 实测项 | 值 |
|---|---|
| START_UTC | `2026-09-27T00:09:50Z` |
| END_UTC | `2026-09-27T00:21:15Z` |
| wrapper总墙钟 | 685秒，包含前置、训练、保存与CPU恢复 |
| tmux window / pane | `@567` / `%567` |
| pane及wrapper PID / body PID | 1158488 / 1158499 |
| 训练uv子进程 / 实际训练PID | 1159390 / 1159400 |
| run UUID | `3522f36c-417c-4fe9-901f-85cbe85eb855` |
| W&B run | [ewcj0x4h](https://wandb.ai/hongzefu-university-of-michigan/openpi/runs/ewcj0x4h) |

W&B ID及链接取自本机driver日志，日志记录online同步完成。本侧没有speed的五份计时/主机/磁盘记录，符合off路径。 本次20步共同观测包含同步与CPU哈希开销，只验正确性，不产生吞吐、ETA或性能优劣结论。

## 7. 训练过程行为

off在共同只读取证层下直接runpy原train.py，没有经过speed.run。 实测判定行为：

```text
TIMING_EQ_RUN=PASS timing=off fetched=21 used=20 state_step=20 real_save=1
```

实际取21批、用于模型和更新20批，最后一批未用于训练；完整TrainState摘要包括params 61叶、EMA 61叶、opt_state 78叶和step 1叶，共201叶。保留原始save_state与异步wait，没有用摘要器替代真实保存。

log100只产生step 0的常规标量日志，末尾1…19的五标量由final尾窗补齐。W&B的Run summary对应已记录的step 0，不能把它读作loop_step 19的末步指标。

## 8. 保存、恢复与配对验收

本run只有checkpoint 19，state_step=20、loop_step=19。完成器实际恢复 `19/params`中的61个EMA叶，与训练末步和完整状态EMA组逐叶对应；201叶完整状态另由共同摘要验证，不能把EMA恢复范围扩写成全201叶均被恢复。判定原文：

```text
RUN_COMPLETED=PASS run=timing20-full-off-0926-20260926T231534Z checkpoints=1 final=19
```

completion绑定本run的11项文件证据、HEAD/UUID、GPU采样及wait记录；runner、completion及其tee、identity/version检查、wrapper各层退出均为0，原生pane退出0和capture实际返回0已联合核实。[本run独立报告](../orig80k-timing20-0926/records/verification/both-off.independent-verification.json)的 `output.results.full`记录这一单次运行结论，文件SHA为 `1a17d5e21c5960f912af96adf97abf79aa540981a13deb17faac66a821ffea61`。

同库正式配对judge已完成，真实判定为：

```text
TIMING_EQ_JUDGE=FAIL reason=20步五标量不是逐位一致
JUDGE_EXIT_CODE=1
WRAPPER_EXIT_CODE=1
```

21次取批树及20次模型实参/RNG均相同；loss、grad_norm、llm_grad_norm、mem_enc_norm各20步不同，param_norm 20步相同，即80/100标量对不同。原状态结构与叶集合相同，params 37/61、EMA 36/61、opt_state 74/78叶摘要不同，step 0/1叶不同，共147/201叶不同。两份真实恢复均成功也不能使配对等价通过。judge原生退出1、capture实际1，原PASS守卫按非零拒绝后续；没有生成成功的 `full.json`。详见[group实测结论](../orig80k-timing20-0926/result.md)及[完整比较记录](../orig80k-timing20-0926/records/verification/final-comparison.independent.json)。CPU INPUT的主机dtype/四None键授权没有放宽本训练对照。

## 9. 用户决定记录

用户选择「两库各跑一对 20 步（推荐）」，并要求「尽可能并行做」「你有8张卡」；因此采用两条4卡库链。用户「允许仅对拍关闭 W&B」仅针对P1/100，本run保持online与真实保存恢复。

「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」和四键missing/None授权保留其原CPU范围，训练逐位要求不变。「继续工作 一路做到起泡前 有问题问用户」限定本轮推进至正式80k起跑前，不因本run完成而直接起80k。

发现差异后，用户对「两库各追加一次相同配置20步off、4+4并行、严格判据、仅诊断」回复「同意」。追加运行须另立新Beta和唯一名称；本档案不预写其结果，不以诊断复验改写本次FAIL，也不据此放行perf/80k。

## 10. 计划外事件与退出边界

运行中已发现第0步四项标量不同，而首批输入/RNG摘要相同；完整记录最终确认所有20步的上述四键均不同。原因尚未确定，不把相同输入或一次off/on差异归因到某个机制。原始标量、状态、失败日志和回执均保留，不调整容差。

独立验收首次用`uv run --no-sync python`调用纯verify时，实际解释器名为`.venv/bin/python3`，与原父适配锁定的`.venv/bin/python`字面不同，检查返回1。随后改用同一uv环境的原父适配路径，完整核验实际返回0；原始失败调用及修正均保留在独立报告，未改任何回执、expect或训练，也未重训。

本run在起跑前通过f8门闩设置remain-on-exit并持久化identity，结束后核实原生pane及capture父真实返回。日志中的0不单独作为最外成功证明；单run成功与正式judge非零失败分别记录。旧INPUT/P1最外final append不可追补的限制，仍按「同样补记限制，沿用INPUT结果（推荐）」及「补记P1限制，补独立退出取证后继续（推荐）」保留；此次新20结果不回写那些历史记录。

## 11. 当前结论与下一步

本run的20步训练、真实保存与EMA恢复已完成，来源及独立退出验收通过；同库off/on逐位等价结论为FAIL。当前停止300步perf和正式80k的放行。已获准的下一项仅是两库各一次相同配置off诊断复验，须从新Beta执行并独立留证；其结果尚未产生，不能覆盖本次差异或预设差异原因。

## 12. 归档文件清单

原始run根为 `v1-store/train-runs/mme_vla_suite/timing20-full-off-0926-20260926T231534Z`，记录根为 `v1-store/bench/orig80k/timing20-full-off-0926-20260926T231534Z`，driver为 `v1-store/logs/timing20-full-off-0926-20260926T231534Z.driver.log`，wrapper为 `v1-store/logs/orig80k-timing20-full-off-0926-20260926T231534Z.wrapper.log`；各自cache及W&B目录保持独立名称。

下列归档目标已按白名单导入并逐项核对来源字节，实际复制范围见[group导入回执](../orig80k-timing20-0926/records/import-receipt.json)：

- 本run小记录：[core.records.tar.gz](records/core.records.tar.gz)、[archive_checks.json](records/archive_checks.json)。含launch/runtime、metrics/final、timing_eq四记录及manifest、completion、GPU采样。
- 清洗日志：[driver](logs/driver.summary.log)、[wrapper](logs/wrapper.summary.log)、[completion](logs/completion.summary.log)与[检查链](logs/log_checks.json)。
- 本run退出链：[identity](../orig80k-timing20-0926/records/exit/orig80k-timing20-full-off-0926-20260926T231534Z/identity.json)、[原生receipt](../orig80k-timing20-0926/records/exit/orig80k-timing20-full-off-0926-20260926T231534Z/exit.json)、[capture父回执](../orig80k-timing20-0926/records/exit/orig80k-timing20-full-off-0926-20260926T231534Z/capture.json)；同目录保留launch与capture原始输出。
- [单run联合独立报告](../orig80k-timing20-0926/records/verification/both-off.independent-verification.json)（取 `output.results.full`）、[全20步差异及正式FAIL链](../orig80k-timing20-0926/records/verification/final-comparison.independent.json)、[group result](../orig80k-timing20-0926/result.md)。比较报告SHA为 `113cec3b17a19e85bd55c7a2b4420f150eac17dbea69a2b3c3e6c38d62265e99`。

原始输出未混入timing_eq manifest管辖目录；只归档Git不能还原的实测记录。脚本/yaml/venv/缓存及大权重不进入docs，真实checkpoint继续留在run根。
