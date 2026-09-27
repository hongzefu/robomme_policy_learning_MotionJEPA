# 追加off重复性诊断结果：关闭包装仍未逐位自复现

## 1. 结论与指标速览

**两库旧off与新off均为DIFFER / comparison_status=FAIL。** 两次新运行的真实保存、CPU恢复、原生退出与capture均通过；比较进程返回0只表示读取和诊断正常完成，不表示逐位相等。

| 每库比较项 | full | counting |
|---|---:|---:|
| 21次取批树、20次模型输入/RNG差异 | 0 | 0 |
| 五标量不等 / 总数 | 80 / 100 | 80 / 100 |
| 完整状态不等叶 / 总数 | 147 / 201 | 147 / 201 |
| 新run原生退出 / capture返回 | 0 / 0 | 0 / 0 |

[原始诊断JSON](records/repeat-comparison.json) SHA256为`92d8726e27819374b0a7feeee1824e35e42215c1749b0fafc6549bcfbe3ce125`，[独立复核](records/verification/repeat-comparison.independent.json)为`c557f7403a09b706f192666e76814ab9b89b669f5d0fd43d7131957eae5b5e54`。这证明所测normal流程关闭计时包装时也存在运行间差异，尚不能确定具体机制；原off/on失败保留。

## 2. 版本与代码状态

旧off实际HEAD为`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`，新off实际`commitV11.15Beta`为`b60ec2b59e0ba012cae0998c5f8713b2aef07b2e`。二者受保护训练源码、工具、依赖逐字一致，差异仅在docs/原计划和实测归档；三规则文件继续固定d710。新运行及完整独立核验结束前HEAD/venv保持冻结，之后源码实现或结果提交不得替换这两个运行锚点。

启动前用[baseline-pins](records/baseline-pins.json)固定旧manifest、completion、identity/native/capture、UUID及独立报告，重新按原SHA核验实体文件。实际[沿用预检](records/preflight/offrepeat-reuse-preflight-20260927.json)与[旧基线/空间检查](records/preflight/offrepeat-baseline-and-space-20260927.json)均完成，原始旧records没有改头或重新定义摘要。

## 3. 启动与配置还原

新run为`timing20-full-off-0926-20260927T003051Z`、`timing20-count-off-0926-20260927T003051Z`；准备TAG不等于真实起跑时间。会话为run前加`orig80k-`，full位于`@573/%573`、pane/wrapper PID1220920，count位于`@574/%574`、PID1220989。两训练PID分别1221541/1221543；完整身份见两个run README和原生日志。

```bash
git show b60ec2b59e0ba012cae0998c5f8713b2aef07b2e:docs/training-doc/orig80k-off-repeat-0927/launch.md
git show b60ec2b59e0ba012cae0998c5f8713b2aef07b2e:scripts/training/tests/timing20-wrapper.sh
git show b60ec2b59e0ba012cae0998c5f8713b2aef07b2e:scripts/training/prod/run_orig80k.sh
```

实际四个父调度载体从Beta对应代码块提取，只把TRAIN_HEAD展开为b60，分别执行start、两run结束验收和compare。原45b4训练wrapper及原f8运行实体不变，新命令/记录根独立；没有新增on运行、重跑原正式judge或变更同HEAD判据。

## 4. 数据集与划分口径

full仍为公开16任务×100集，共1600集、768897帧、476857执行样本；counting仍为同源四任务×100集，共400集、189035帧及执行样本。两个库的4×4 packed、source、manifest、history、tokenizer及pi05_base初始化资产沿用原实体，数据和划分未变。

full norm为原版`f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`，counting为自身`a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`；history为`823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。具体路径按launch及两run档案还原，不重新下载或构建。

## 5. 关键超参

两run均smoke20、b64/fsdp4/workers4/seed42、modul512/4×4/max32、原warmup10000、peak/decay5e-5、decay_steps100000、EMA0.999、log100/save10000/keep10000，W&B在线。mode为off、normal环境，没有确定性XLA_FLAGS；实际x64=False。仅沿用smoke20覆盖，模型、optimizer、采样、dtype及误差阈值未改。

## 6. 硬件、耗时与空间

AWS本机8×A100-SXM4-80GB，md0 XFS本地NVMe。两run均于2026-09-27 00:54:30 UTC起跑；full使用0–3，01:05:30结束，共660秒，W&B ID`y0vco81d`；count使用4–7，01:05:12结束，共642秒，W&B ID`4qxclhfu`。耗时包含初始化/编译、完整状态哈希、真实保存和CPU恢复，不作吞吐或80k ETA结论。

诊断前scratch可用748.324GiB；按前四次相同配置已完成run的文件分配、日志/cache作新增量估计22.137GiB，扣估计后726.187GiB，仍高于既定300GiB预留。此为诊断资源估计，不是连续峰值上界或正式prod schema2预算。初次只读估算器误将checkpoint的全链接拒绝用于W&B内部日志别名而返回1；核实16项均在v1-store内后，日志/cache按lstat计链接本身、不穿透，实体去重，checkpoint拒绝链接不变。原始记录均未修改。

## 7. 实际训练、保存与恢复

两新run各完成21次fetch、20次模型调用、20步五标量与201叶末态摘要，state.step=20，loop/checkpoint=19。两个checkpoint均实际写入并由CPU恢复61个EMA叶，摘要与各自末步EMA一致；不是从checkpoint恢复了完整201叶TrainState。两训练wrapper、恢复及相关tee/身份/版本核对均返回0，原生pane和capture也为0。

full UUID为`18ec5653-957e-48ab-9092-14998cfd3500`，count为`5e8178e6-fce4-48a7-b1a3-1c3ffef0d6e8`。完整单run独立核验见[两新off报告](records/verification/both-off.independent-verification.json)，SHA`b0428ec4a9a229b35307387c570d05fd29ff6f5c63fe7ac9c4e245ed9bf58f90`。

## 8. 诊断比较与完整差异

按Beta内联CPU比较于01:05:58—01:05:59 UTC执行，原输出为：

```text
OFF_REPEAT_DIAGNOSTIC_READ=COMPLETE
OFF_REPEAT_DIAGNOSTIC_EQ=FAIL
```

比较进程和父执行返回0，JSON为`read_validation_completed=true`、`comparison_status=FAIL`、`status=DIFFER`。两侧分别以自身真实HEAD调用原read_side；完整配置仅排除原IDENTITY_FIELDS的exp_name与派生checkpoint_dir两项，provenance只在比较投影排除已分别验证的git_head_of_cwd。旧记录未改，原compare_sides/judge没有被调用或改宽。

两库均所有取批/模型输入/RNG、fingerprint、loader、toolchain、其余来源与原保存函数相同。loss、grad_norm、llm_grad_norm、mem_enc_norm各20项不同，param_norm20项相同。params37/61、EMA36/61、opt74/78、step0/1叶摘要不同，总147/201；原treedef、EMA路径、state.step和loop_step一致。严格不等是诊断结果，不能因为程序返回0就写成等价PASS。

## 9. 用户决定记录

本次追加来自明确提问：两库各追加一次相同配置20步off，4+4并行，保持严格判据，仅用于判断关闭包装时是否也有差异，不替代正式对拍。用户答复「同意」。原有「尽可能并行做」「你有8张卡」「继续工作 一路做到起泡前 有问题问用户」继续适用。

其后针对新的验证范围提问：将包装正确性闸门改为两库各一对确定性档20步off/on，输入、标量及完整状态仍逐位一致；仅此启用`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，通过后再进入normal 300步perf，旧normal FAIL保留，其他训练参数不变。用户再次答复「同意」。这是下一阶段授权，本次b60运行仍是normal off，不能改写为用了新档。

## 10. 原因边界与证据保存

首步早期差异已在[独立首行证据](records/diagnostics/offrepeat-step0-early-independent-20260927.json)固定，只绑定首完整行bytes/SHA，未把在写整文件当最终快照。完整结论来自实际结束后的全部20步、末态及两份独立报告。

关闭包装也不重复，说明计时开关不是差异出现的必要条件；尚未证明具体初始state、编译算法或时序机制。初始完整state没有摘要，param_norm只聚合特定kernel，相同norm不足以证明初态相同。获准前冻结的[确定性候选说明](records/diagnostics/timing20-deterministic-validation-option-20260927.md)保留其当时“未获准”文字，最新批准以第9节和总计划为准；不能将新验证档称为根因修复。

## 11. 当前结论与下一步

本次已完成且严格重复比较失败，原normal off/on失败继续保留。下一阶段按新授权实现显式smoke-only确定性档、完成开关隔离/记录绑定/严格拒绝测试，再从新代码Beta重取两库各一对20步。该通过结论只覆盖确定性验证档，不证明normal包装无数值影响；perf/prod环境仍保持原normal。

当前没有新确定性20的GPU结果，300步perf和80k均未启动。工具测试或本次单run恢复通过不能代替新严格闸门；本轮仍止步正式80k起跑前。

在本次b60诊断结束并解除冻结后，新档位工具已落实于runner、contract、timing_equiv、speed四文件及三份定向测试；模型/训练循环/数据链、同步/采样/真实保存转发及原模板/guard保持不变。四组合测377项通过（53.59秒），pytest、tee及宿主返回0，Ruff/Bash/diff检查通过。见[整合验证](records/tool-validation/deterministic20-integrated-validation-20260927.json)和[原测试日志](records/tool-validation/deterministic20-integrated-tests-20260927.log)。该记录明确基准HEAD与尚未提交的源码摘要，不把新实现或其测试冒称b60原诊断所用代码；GPU验证须另从新Beta进行。

## 12. 归档文件清单

[原白名单](records/formal-import-whitelist.json)47项加[两报告增量](records/independent-reports-import-addendum.json)2项，共49项、863446 B，逐项源/目标bytes/SHA相同，见[导入回执](records/import-receipt.json)。另保留首行证据、获准前候选及其唯一直接机理引用。两核心包共34成员，CRC/成员字节与原件核验通过；两run的records/logs各自独立，两个任务的8项原生退出链在group records/exit下。

三类日志保留raw/stage/git摘要；只清洗明确进度行及精确`wandb:`纯装饰行的尾空白，完成/失败/退出关键行不丢。原JSON里的v1-store绝对来源路径不改写。四个父阶段的实际返回、stdout/stderr、旧固定基线与现场前置均保存；旧V11.14核心包不重复复制，代码、运行脚本、大权重、cache和动态控制文件不进入归档。
