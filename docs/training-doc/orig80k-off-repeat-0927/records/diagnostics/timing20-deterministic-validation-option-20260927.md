# 仅20步包装正确性对照采用既有确定性档：未获准候选

固定审查HEAD为 `b60ec2b59e0ba012cae0998c5f8713b2aef07b2e`，开工porcelain为空。**本说明不是实施授权；没有改源码、环境、flags、判据或正在运行的新off，也没有新增GPU任务。** 根代理待完整复验闭环后再询问用户是否采用。既有normal off/on正式FAIL保留；不能把切验证档称作已修复根因或预言一定通过。

根代理已提供[追加off首步早期记录](offrepeat-step0-early-independent-20260927.json)，SHA256 `5a86c541ab960ff25c8653329618477423ee40ba4a46b7acad0ba6778df9a8dc`：两库旧off/新off第0步各4/5标量hex仍不同，首批input/RNG相同。它说明该前缀已不重复，但不替代完整20步、末态、恢复和退出的最终验收。

## 已有拦截位置与最小实现边界

拟仅为“smoke20且显式ORIG80K_SMOKE_EQ_MODE=off/on”提供默认关闭的验证档选择，例如 `ORIG80K_TIMING_EQ_PROFILE=deterministic100`（拟议接口，当前不存在）。只有这一显式选择才在新Python进程启动前设置既有100步的精确字符串：

```text
--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0
```

不能通过删除所有XLA_FLAGS检查、接受任意flags、只看继承的两项flags就自动选档，或在JAX已初始化后再改环境实现。未选择时保留现有normal清理行为；prod/perf或未安装共同观测层的普通smoke只要显式请求新档就拒绝，未知/空档名也拒绝。两项flags取自现有 `run_entry_equiv.sh`，不另加第三项；x64=False、W&B online、全部超参/输入/保存与逐位判据不变。

| 最小候选文件与稳定锚点 | 当前行为 | 若获准的必要小改动 |
|---|---|---|
| [run_orig80k.sh](/scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/prod/run_orig80k.sh)模式前置与body环境段 | 无条件unset XLA_FLAGS；外层45b模板不会替它保留flags | 增加显式档位合法性检查，仅上述smoke off/on分支设置精确两项flags；在CPU配置解析和GPU训练进程创建前完成。默认normal及prod/perf继续清理遗留flags。不要依赖secret/env文件重新注入未登记flags |
| [orig80k_contract.py](/scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/orig80k_contract.py)::check_launch/parse_config | check_launch要求XLA_FLAGS和JAX_PLATFORMS都空；CPU解析仍继承flags | 将XLA条件改成闭合的“模式＋共同观测开关＋显式档位＋精确flags”校验，JAX_PLATFORMS仍禁止泄漏；记入launch实际档位/flags。可在此放一个小的纯函数/常量供下面入口复用，不新增模块框架或CLI |
| [check_orig80k_timing_equiv.py](/scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/check_orig80k_timing_equiv.py)::run | 在装observer前拒绝非空XLA_FLAGS | 复用同一闭合校验；元数据明确记录验证档位与实际flags，TRAIN_TIMING_STEPS入口仍须0。共同输入/RNG/状态观察、同步点和真实save调用不动 |
| 同文件::validate_fingerprint/read_side/validate_completion/validate_speed | 指纹强制flags为空；读回已绑定launch、metadata、speed、实际代码SHA | 用**记录中的**档位核验fingerprint.environment.XLA_FLAGS，并与launch及on侧speed_run的声明/实际flags交叉绑定；不要取CPU judge当前shell的flags作历史证据。normal必须空，确定性档必须精确两项；缺/伪造/混合档拒绝 |
| [check_orig80k_speed.py](/scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/check_orig80k_speed.py)::run | 一进函数即要求flags为空，包括mode=smoke的on路径 | 必须先识别模式，再只允许“smoke20＋共同观测on＋显式新档”的精确flags；perf300始终拒绝新档。新档/实际flags入speed记录；报告入口继续只收normal perf300，不能拿确定性20记录当性能或预算依据 |

不改 `train.py/train_step`、模型、optimizer、`step_timing/HostTiming`、磁盘采样、真实保存/恢复或100步harness。45b外层模板本身可保持字节不变，档位通过新launch的明确环境传入；f8和独立退出协议不变。通用 `preflight_train_launch.py`未见额外XLA空值门禁，先通过CPU定向解析验证，不预先扩大修改范围。

元数据档位只用于声明验证环境，不把它加入config的身份差异白名单。`compare_sides`继续同HEAD、相同完整fingerprint/toolchain/provenance，21次fetch、20次model/RNG、20×5 hex和完整state严格逐位比较；两侧档位不同即失败。新元字段必须可核，不能因旧记录缺字段就猜测它使用了确定性档。

## 定向测试，仅在获准实施后执行

1. 扩展 `test_orig80k_launch.py::test_real_shell_propagates_failures_and_keeps_cpu_scoped`：默认三模式仍清遗留flags、训练无CPU平台；新档仅smoke off/on能进入，两侧看到精确两项flags；prod/perf、普通smoke、非法/空档、额外flags均在训练前拒绝。用真实shell＋外部桩记录实际环境，不只查源码字符串。
2. 扩展 `test_orig80k_speed.py::test_run_rejects_deterministic_flags_before_outputs`及 `test_wrapper_forwards_real_entry_and_save_and_restores_state`：perf仍拒绝新档/非空flags，未声明档位的smoke仍拒绝；显式新档smoke20能原样转发argv/save、禁profiler，并在异常后恢复已有hooks。既定20/300同步与真save次序测试保持。
3. 扩展 `test_orig80k_timing_equiv.py`的入口/指纹/配对负例：同档精确flags可读回；normal与新档混配、改标签不改flags、只改launch或speed记录、x64开启、漏元字段均拒绝。原输入、RNG、任一hex、optimizer/state、同HEAD及nonfinite拒绝测试不得删除或改宽。
4. 真实CPU配置解析确认两项GPU flags能通过该环境的CPU preflight，且profile/flags记录不丢失、x64仍False；若解析失败先报告，不静默清除它们掩盖差异。通过这些最小测试后，才在新clean Beta、全新名称下实跑两库各一对20步真保存/恢复及正式judge；本次不执行任何这些验证。

## 版本与证据不能跨档偷换

这项候选会改变runner/contract/speed/observer源码，因此现有“perf Beta相对b0ef只改文档、所有工具字节不变”的预制放行前提不再成立。若用户批准实施，必须以**新实现的新20 Beta**重新取得两对off/on证据，再为后续normal perf核代码版本、档位差异及指纹；不能仍写相对b0ef源码未变。现有read_side比较当前toolchain文件SHA，旧normal记录不能通过删除该检查或改旧SHA来混入新验证；历史FAIL继续使用其原锚点与归档证据。

即使新档20通过，得到的也只是“既定确定性验证档下，该20步包装观测一致”，**不是normal档无数值影响的证明**。perf/prod保持normal且需真实保存/恢复、资源测量与原完成闸门；是否接受这一验证范围变化，应明确交用户决定。它不能证明先前差异由autotune造成，也不承诺normal轨迹逐位稳定。

## 若完整追加off反而重复一致，先补只读证据

不得从一次off重复一致直接断言包装是根因。先对旧off/旧on/新off三方完成记录逐组核：原始step hex首次分岔和全部20步、201叶按params/EMA/opt_state/step分组差异、完整fingerprint与loader、实际模块/工具SHA、命令argv与有效环境、GPU/driver、独立cache路径和已有时序/采样记录。只读已存在的编译日志或缓存元信息可作为线索；未留编译/算法证据就明确写“没有”，不补造或现在开启探针。

仍须保留[初始state/norm边界](timing20-step0-mechanism-note-20260927.md)：目前没有初始完整state摘要，param_norm只是部分kernel的多对一聚合。缺这类证据时不能静态排除初态或运行时数值差异；若进一步需要新取证或隔离实验，应另列最小实验由用户批准，不在此切flags、拆包装或追加GPU运行。
