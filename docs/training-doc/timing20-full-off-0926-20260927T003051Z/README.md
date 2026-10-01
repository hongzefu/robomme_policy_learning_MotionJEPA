# timing20-full-off-0926-20260927T003051Z：追加off重复性诊断实测

## 1. 一句话结论与指标速览

**本run的20步训练、真实保存与EMA恢复通过；与同库旧off的严格重复性结果为DIFFER / FAIL。** 实测21次取批、20次模型使用与更新，完整TrainState共201叶且有限，checkpoint 19真实保存并恢复61个EMA叶；wrapper、原生pane和capture实际返回均为0。

与[旧off基线](../timing20-full-off-0926-20260926T231534Z/README.md)相比，21批输入树及20次模型实参/RNG全部相同，但100对标量中80对不同、201叶末态中147叶摘要不同。诊断比较程序读完返回0，`comparison_status=FAIL`；不能从执行成功推导重复一致，也不改写旧off/on失败或放行perf/80k。

| 项目 | 实际值 |
|---|---|
| run_name | `timing20-full-off-0926-20260927T003051Z` |
| 旧off基线 | `timing20-full-off-0926-20260926T231534Z` |
| GPU / steps / batch | `0,1,2,3` / 20 / 64 |
| 实际Beta | `b60ec2b59e0ba012cae0998c5f8713b2aef07b2e` |

名称中的TAG是准备标签，实际UTC与运行身份见第6节。

## 2. 版本与代码状态

实际运行Beta为 `commitV11.15Beta`：`b60ec2b59e0ba012cae0998c5f8713b2aef07b2e`，从clean状态起跑，运行与验收期间冻结源码和uv环境。旧off仍固定 `b0efbde61e38411fb1b9114eec8485d36a4ee9a0`；V11.14失败已先归档，新旧差异仅docs/计划，训练源码、runner、量具、模板及依赖逐字不变。三规则继续固定 `d710d8489b88aa75770af3452fd7c9b374deaef7`。

原INPUT/100沿用预检、旧off固定基线/read_side与空间复核均已通过；每run另完成25项前置及完整资产检查。新旧数据、资产、history、模块与依赖指纹分别核对，原records中的HEAD及旧文件SHA未改，归档提交不替代各自真实运行版本。

## 3. 启动与配置还原

完整命令、Git差异/新输出守卫、原样模板及guard父适配见[共享launch](../orig80k-off-repeat-0927/launch.md)，不能绕过它直接运行内部runner。实际使用的固定展开值如下，用于还原而非再次启动：

```bash
TRAIN_HEAD=b60ec2b59e0ba012cae0998c5f8713b2aef07b2e
RUN=timing20-full-off-0926-20260927T003051Z
TIMING=off
GPUS=0,1,2,3
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
```

实际session为 `orig80k-timing20-full-off-0926-20260927T003051Z`，实际UTC及PID见第6节。原模板 `scripts/training/tests/timing20-wrapper.sh`完整SHA为 `45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f`，不改正文。runtime在新v1-store命令目录加固定参数前缀，整文件SHA为 `b1aeed016e67fc1980218a018643bf24608c88241d6768f4b6f5944af5937023`；f8仍用原runtime实体，新Beta Git源/工作树/runtime字节联合校验。

还原实际Beta中的入口、模板与控制正文：

```bash
git show b60ec2b59e0ba012cae0998c5f8713b2aef07b2e:scripts/training/prod/run_orig80k.sh
git show b60ec2b59e0ba012cae0998c5f8713b2aef07b2e:scripts/training/tests/timing20-wrapper.sh
git show b60ec2b59e0ba012cae0998c5f8713b2aef07b2e:docs/training-doc/orig80k-off-repeat-0927/launch.md
```

## 4. 数据集与划分口径

公开16任务全集为1600集、768897帧、476857执行样本，与旧off消费同一实体库：packed为 `$LIB/framesamp`，source及episode manifest显式绑定，数据和划分不改。既有构建Beta为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`，见[构建档案](../../dataset-build-doc/16task-pub-1600ep/README.md)。

norm为 `$ASSETS/robomme/norm_stats.json`，期望SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。继续使用原版f332，不替换为完整库自算c5d45b…统计量。 不换数据、采样种子、worker数或精度来追求相等。

## 5. 关键超参

| 参数 | 固定值 |
|---|---|
| steps / batch / FSDP / workers / seed | 20 / 64 / 4 / 4 / 42 |
| history | perceptual-framesamp-modul.yaml，modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024 |
| history SHA | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| warmup / peak / decay / decay_steps | 10000 / 5e-5 / 5e-5 / 100000 |
| optimizer / EMA | AdamW clip1.0 / 0.999 |
| log / save / keep | 100 / 10000 / 10000 |
| 模式 / W&B / 初始化 | off、normal env / online / pi05_base |

原streaming_obs_horizon16保持。仅smoke步数覆盖20；不加确定性XLA_FLAGS、不变容差、不设置JAX_ENABLE_X64绕过实际False要求。各run独立缓存保持旧流程，不复用旧cache来制造结果。

## 6. 硬件、调度与耗时

AWS 8×A100-SXM4-80GB、md0 XFS；本run固定 `0,1,2,3`四卡，full/count新off按4+4并行。各自用原45b模板在训练后立即串行CPU恢复，与旧off流程相同。设备UUID、依赖及输入身份已对前置记录核验。

| 实测项 | 值 |
|---|---|
| START_UTC | `2026-09-27T00:54:30Z` |
| END_UTC | `2026-09-27T01:05:30Z` |
| wrapper总墙钟 | 660秒，包含前置、训练、保存和CPU恢复 |
| tmux window / pane | `@573` / `%573` |
| pane及wrapper PID / body PID | 1220920 / 1220931 |
| 训练uv子进程 / 实际训练PID | 1221536 / 1221541 |
| run UUID | `18ec5653-957e-48ab-9092-14998cfd3500` |
| W&B run | [y0vco81d](https://wandb.ai/hongzefu-university-of-michigan/openpi/runs/y0vco81d) |

W&B ID及链接取自本机driver日志，日志记录online同步完成。前置资源与剩余空间按实际快照核对；此run没有speed的五份采样记录，符合off路径。上述墙钟包含共同观测/哈希开销，不用于吞吐、ETA或性能优劣结论。

## 7. 训练过程行为

与旧off一样，在共同只读取证层下直接runpy原train.py，没有经过speed.run。实际判定原文：

```text
TIMING_EQ_RUN=PASS timing=off fetched=21 used=20 state_step=20 real_save=1
```

21批中20批用于模型与更新，最后一批未使用。保留真实save_state及异步wait，checkpoint 19已保存。完整末态记录params 61叶、EMA 61叶、opt_state 78叶、step 1叶，共201叶；常规log100提供step 0，final尾窗补齐1…19的五标量。

初始完整state未记录。param_norm只覆盖筛选kernel的聚合值，不能据相同norm证明初始params/EMA/优化器逐位相同。W&B的Run summary仅对应常规记录的step 0，不能当作loop_step 19的末步指标。

## 8. 真恢复与独立重复性比较

本run真实保存checkpoint 19，state_step=20、loop_step=19；完成器实际恢复61个EMA叶并与本run末步/完整状态EMA摘要相同。201叶完整状态由共同摘要核验，不能把EMA恢复范围扩写为201叶均从checkpoint恢复。完成器判定为：

```text
RUN_COMPLETED=PASS run=timing20-full-off-0926-20260927T003051Z checkpoints=1 final=19
```

completion的11项文件引用、run/HEAD/UUID、GPU采样和异步wait完整；runner、completion/tee、identity/version、wrapper各层退出均0，原生pane及capture实际0。[两新run独立报告](../orig80k-off-repeat-0927/records/verification/both-off.independent-verification.json)中 `output.results.full`记录本run，SHA为 `b0428ec4a9a229b35307387c570d05fd29ff6f5c63fe7ac9c4e245ed9bf58f90`。

随后按[独立CPU方案](../orig80k-off-repeat-0927/launch.md)分别以旧b0和新b60真实HEAD读取两侧。仅配置身份字段exp_name/derived checkpoint_dir不同，provenance仅在比较投影中排除已分别核实的git_head_of_cwd；fingerprint、loader、toolchain、其余来源及原保存函数均相同。没有调用原同HEAD正式judge或修改旧记录。

```text
OFF_REPEAT_DIAGNOSTIC_READ=COMPLETE
OFF_REPEAT_DIAGNOSTIC_EQ=FAIL
```

比较进程实际返回0表示两侧验证与比较正常读完，结果JSON为 `status=DIFFER`、`comparison_status=FAIL`。21次fetch tree和20次model tree/RNG全部相同；loss、grad_norm、llm_grad_norm、mem_enc_norm各20步不同，param_norm 20步相同，共80/100标量对不同。原结构和叶集合相同，params 37/61、EMA 36/61、opt_state 74/78、step 0/1叶不同，总147/201叶摘要不同；原treedef、EMA路径和步数相同。完整结果见[group result](../orig80k-off-repeat-0927/result.md)及[独立比较核验](../orig80k-off-repeat-0927/records/verification/repeat-comparison.independent.json)。

## 9. 用户决定记录

用户对两库各追加一次同配置20步off、4+4并行、仅诊断的请求回复「同意」。两个新名称由根代理采用，准备TAG不代表开始时间。

既有「尽可能并行做」「你有8张卡」继续落实；「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」不放宽本次训练payload判据。W&B保持online，原no-W&B授权仅P1/100适用。本轮仍按「继续工作 一路做到起泡前 有问题问用户」停在正式80k起跑前。

结果回填后，用户已明确同意下一阶段仅在包装正确性20步对照使用两项指定确定性flags，其他训练参数和逐位判据不变，perf/prod仍normal。本run本身仍为b60的normal off，不回填为用了新档。

## 10. 计划外事实与处置

旧off/on每库21 fetch、20 model/RNG一致，但100标量80对不同，末态147/201叶不同；四run自身保存/恢复及原生退出通过，两正式judge真实FAIL。该V11.14结论已归档，完整报告SHA `113cec3b17a19e85bd55c7a2b4420f150eac17dbea69a2b3c3e6c38d62265e99`保持不变。

本次追加off在首步再次出现四项标量不同，已单独保存首完整行bytes/SHA证据；结束后完整比较确认同库两次off也不逐位相同。[机理说明](../orig80k-timing20-0926/records/diagnostics/timing20-step0-mechanism-note-20260927.md)所列边界仍成立：初始完整state未记录，不能确定具体初始化、编译或计算机制，也不能证明一般确定性。原数据、记录和判据均未修改，没有自动重跑或降低标准。

## 11. 当前结论与下一步

本run已完成训练、真实保存恢复及原生退出验收；旧off/新off严格重复性诊断为DIFFER / FAIL。两次均为off也出现差异，不据此指定某个机制为根因；旧off/on正式FAIL保持原样。

300步perf和正式80k继续等待新严格闸门。确定性20步方案已获用户批准，正在另行实现与验证，尚无新GPU实测；后续使用新代码Beta和全新run，不改本run原始结果。

## 12. 归档文件清单

原始records为 `v1-store/bench/orig80k/timing20-full-off-0926-20260927T003051Z`，权重根为 `v1-store/train-runs/mme_vla_suite/timing20-full-off-0926-20260927T003051Z`，driver为 `v1-store/logs/timing20-full-off-0926-20260927T003051Z.driver.log`，wrapper为 `v1-store/logs/orig80k-timing20-full-off-0926-20260927T003051Z.wrapper.log`。旧off仍保留b0的原目录与SHA，新run保留b60真实身份，比较输出在独立 `v1-store/bench/orig80k-offrepeat-results-20260927T003051Z/repeat-comparison.json`，SHA为 `92d8726e27819374b0a7feeee1824e35e42215c1749b0fafc6549bcfbe3ce125`。

按已验收清单由主代理统一导入以下目标；本次正文回填时，尚未复制的链接随group结果落地，不把目标路径当作已导入证明：

- 本run小记录：[core.records.tar.gz](records/core.records.tar.gz)、[archive_checks.json](records/archive_checks.json)。
- 清洗日志：[driver](logs/driver.summary.log)、[wrapper](logs/wrapper.summary.log)、[completion](logs/completion.summary.log)、[检查链](logs/log_checks.json)。
- 原生退出链：[identity](../orig80k-off-repeat-0927/records/exit/orig80k-timing20-full-off-0926-20260927T003051Z/identity.json)、[receipt](../orig80k-off-repeat-0927/records/exit/orig80k-timing20-full-off-0926-20260927T003051Z/exit.json)、[capture父回执](../orig80k-off-repeat-0927/records/exit/orig80k-timing20-full-off-0926-20260927T003051Z/capture.json)；同目录保留launch/capture原始输出。
- [单run独立报告](../orig80k-off-repeat-0927/records/verification/both-off.independent-verification.json)、[重复比较独立报告](../orig80k-off-repeat-0927/records/verification/repeat-comparison.independent.json)、[group result](../orig80k-off-repeat-0927/result.md)。后者独立报告SHA为 `c557f7403a09b706f192666e76814ab9b89b669f5d0fd43d7131957eae5b5e54`。

仅归档Git不能还原的小型实测记录；不把脚本/yaml、权重、venv或cache复制进docs。真实checkpoint保留在原run根，原始产物和旧失败记录不改写。
