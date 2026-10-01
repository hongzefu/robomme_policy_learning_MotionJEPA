# 两库normal300 perf与B预算实际结果

## 1. 一句话结论与指标速览

两库normal300步、两次61EMA真实恢复、联合report、两measurement和已批准B预算共八阶段全部PASS。full **66.445381 samples/s**，count **66.092631 samples/s**；两稳态窗口正交集191.588853秒，按并集194.718151秒计算合计**131.472078 samples/s**。不能直接相加单侧速率。

预算B实际所需383488663552 B（357.151649GiB），正式消费者当时可用708412715008 B，04:20:41 UTC完成。旧normal20/off复验FAIL不改写，本结果不证明normal逐位重复性或模型质量。用户最新明确「确认无误后可以直接开始两个训练」；最终正式Beta/clean及前置确认后可直接起两80k，本档案没有伪记正式训练已启动。

## 2. 版本与代码状态

真实perf Beta `70a641a827eb2559c3acb69430f55c1c53babaeb`（commitV11.17Beta）；实现锚aec86db，确定性20运行锚c71、结果归档29f71各自保留。整个运行期源码/uv环境冻结，04:24:31 UTC最终快照后解除；归档提交不替换运行HEAD。INPUT3a、P1/100的00bd/ecf及已批准历史退出限制保持原记录。

## 3. 启动与配置还原

[共享launch](launch.md)锁定原八阶段控制正文：两GPU并行，均结束后依次full/count恢复，再联合report、两measurement和预算。固定d3f只读前置在任何mkdir前实际执行PASS，报告SHA `c47062e8e37e49933fb75e7c92d407983bce74c1339530ec5a8e8ff9796aaec0`。实际GPU起点均2026-09-27T02:18:30Z，TAG `20260926T232339Z`不是起跑时间。参数、来源、原命令SHA、门闩identity和父返回由原件绑定；不复制脚本或运行命令附件。

## 4. 数据集与划分口径

full1600集/768897帧/476857执行样本，count400集/189035帧与执行样本，原4×4 packed/source/manifest不改，无缩样本或新划分。full沿原norm f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5，count沿授权自算a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9。

## 5. 关键超参

300步、b64/fsdp4/workers4/seed42，modul512/4×4/max32、streaming_obs_horizon16；warmup10000、peak/decay_lr5e-5、decay_steps100000、EMA0.999、log100/save10000/keep10000，pi05_base初始化，W&B online。profile显式normal，两selector/XLA_FLAGS清除，x64实际禁用且没有静默覆盖。perf没有20步共同输入/完整201叶观察层，计时不调用JAX profiler。

## 6. 硬件、调度与稳态性能

AWS8×A100-SXM4-80GB、本地md0 XFS；full0–3、count4–7。两GPU UTC窗口分别02:18:30→02:27:56、02:18:30→02:27:51；随后才开始CPU恢复，未用恢复I/O污染另一侧steady。

| 指标 | full | count |
|---|---:|---:|
| steady100–299，共200步秒数 | 192.639424 | 193.667581 |
| 平均稳态步秒数 | 0.963197 | 0.968338 |
| samples/s | 66.445381 | 66.092631 |
| 初始化/JIT/预热秒 | 263.200 | 259.271 |
| save dispatch秒 | 6.555 | 6.990 |
| save至收尾秒 | 63.631 | 62.879 |
| 原公式80k ETA小时 | 21.592 | 21.703 |

窗口按step99同步完成至step299训练后/保存前同步定义。ETA含初始化估计、80000稳态步及8次末步保存收尾外推，仅一次save观测，不是承诺；save至收尾不是纯磁盘写时。正交集不夸成两侧所有200步完全并行。

## 7. GPU、主机与磁盘采样

GPU目标0.5秒，逐卡steady采样如下；相邻重复读数不视作新增独立证据。两侧slow_steps均为空，慢步阈值为稳态host中位数2倍，slow层样本0/均值null，other层为表中均值；不能把null写成0利用率，也不因高util断言无瓶颈。

| GPU | util均值% | 0%占比% | 样本数 | 实际间隔均值/最大s | 相邻相同读数 |
|---|---:|---:|---:|---|---:|
| 0 | 98.764 | 0.0 | 385 | 0.500516 / 0.505000 | 240 |
| 1 | 98.569 | 0.0 | 385 | 0.500516 / 0.504000 | 350 |
| 2 | 98.501 | 0.0 | 385 | 0.500516 / 0.504000 | 262 |
| 3 | 98.808 | 0.0 | 385 | 0.500513 / 0.505000 | 350 |
| 4 | 98.499 | 0.0 | 387 | 0.500992 / 0.569000 | 235 |
| 5 | 98.904 | 0.0 | 387 | 0.500992 / 0.569000 | 377 |
| 6 | 98.543 | 0.0 | 387 | 0.500992 / 0.569000 | 243 |
| 7 | 98.837 | 0.0 | 387 | 0.500992 / 0.569000 | 308 |

full/count host RSS总和峰值114735075328/64864890880 B，含共享页重复计数，不是独占物理RAM；SHM峰值均4620455936 B，不将同一共享对象重复相加。MemAvailable最低值1009290399744/1009351651328 B。

磁盘样本1004/999，实际间隔均值0.512782/0.511879s、最大3.757910/3.725057s，missed26/24；扫描非原子，count有1行中间路径竞态，错误0且最终稳定。P分别16039718912/13251739648 B，F分别11879968768/11879948288 B，E合计5531541504 B仅为观测下界。0.5秒目标不是连续峰值证明。

## 8. 真实恢复、报告、receipt与预算

两侧各唯一299/state300，metrics[0,100,200]及尾窗201…299共99步五标量完整有限；各自真实恢复61EMA叶等于自身final，两库之间不要求参数相等。两receipt均在权重仍在时生成，C（单299）为11879944192/11879923712 B，不能以F整run根冒充单份。联合[paired-speed](records/paired-speed.json) SHA `dd9bdf145be4642099e4b285b1e5fcd984f9feba5187a134441a4067b6b5f9d8`。

用户已选择B：每份16GiB×16份保留目标274877906944 B，增长余量84798963712 B；采样盲区余量34359738368 B，加E得保存额外39891279872 B；日志/cache余量68719476736 B。总需求与300GiB取max后为383488663552 B。当前单checkpoint逻辑量与zstd/OCDBT不能推出16GiB是数学硬上界；这些是用户审阅批准的工程保守余量，采样下界限制不变。

[margin-source](records/margin-source.json) SHA `09ea4e4739e11e2851a013b00e9a2f3322d0b6d9b7859f9b749495be90d375bf`；[schema2预算](records/prod-disk-budget.json) SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`。正式消费者记录available708412715008 B，独立只读验收观察708412710912 B，时点不同各自保留；未改成同值。第八阶段原生/capture/父/宿主均0，权重未为此删除，receipt原始evidence不改写。

## 9. 用户决定记录

用户保留两库原版80k/4+4目标，要求「尽可能并行做」「你有8张卡」；P1/100可关闭W&B不延伸到perf，0.5秒磁盘采样及det包装闸门均已按批准执行。用户已明确选择B=16GiB/份、32GiB盲区、64GiB日志/cache；本轮八阶段wrapper COMMAND末尾原1B空格保留，只对明确八文件允许该尾白，其他异常不放宽。

旧「继续工作 一路做到起泡前 有问题问用户」保留为历史范围。最新原话为「确认无误后可以直接开始两个训练」，已覆盖此前stop-before；在V11.17归档、独立正式Beta/clean及最终前置确认后，root可直接启动两组80000步。本结果只记录已完成perf/预算，不把新授权写成80k已经起跑。

## 10. 意外与历史限制

首次GPU暂存被日志守卫拒绝，原因是wrapper第9行WRAPPER_COMMAND末尾1B未转义分隔空格，不是训练失败。后续确认七份已结束和第八预算同模板均如此。用户批准后，八条原命令行实际未删字节，shlex前后argv相同；[精确日志政策](records/wrapper-log-policy.json)逐文件/逐行登记。进度行及既有纯W&B装饰清洗规则不变，关键行和EOF保真；Git仅对列出的八份摘要禁blank-at-eol，其余默认严格检查。

原PENDING阶段索引/中间controller保留，不追写事前批准；最终快照另存。[旧normal20](../orig80k-timing20-0926/result.md)及[off复验](../orig80k-off-repeat-0927/result.md)FAIL不变；[确定性20](../orig80k-timing20-det-0927/result.md)PASS仅限其档位。旧INPUT/P1最外退出限制也不被本轮新原生证据追补。

## 11. 结论与下一步

normal perf及用户B预算全部完成，已取得稳态正交集、实际吞吐/资源/保存恢复和八阶段独立退出证据。两正式run须引用同一固定预算和SHA，按新的共同正式Beta、clean、原版配置、资产/输入沿用、全新输出与实时资源/空间核对。遵照最新「确认无误后可以直接开始两个训练」，确认无误后直接起两80k，无需重复请求已授授权；此处不虚构未来run身份或运行结果。

## 12. 归档文件清单

本批白名单131项、2284191 B（不含白名单/导入回执自身）。两run各17成员核心包、独立measurement、driver及检查；group包含八独立报告、八wrapper摘要、64原生侧证、八父调度/预算装配返回、前置、共享report、批准余量/预算及[最终索引](records/final-perf-archive-index.json)。八任务和八根phase由[最终controller](records/controller.completed.snapshot.json)绑定；历史中间PENDING快照单独保留。

另有[导航副本补充](records/navigation-alias-addendum.json)：批准前余量说明原样保留，按其原相邻链接添加同字节日志/cache统计副本；原MD、原JSON及白名单SHA不变，没有新增测量或重跑。

原JSON/纳秒整数逐字节保存，不经JavaScript重序列化；原文件路径不改写。原始wrapper/driver留v1-store，不以清洗副本覆盖原件。源码、.sh/.yaml、cache和大权重不作为docs附件；两run入口见[full](../perf-orig80k-full-300-20260926T232339Z/README.md)、[count](../perf-orig80k-count-300-20260926T232339Z/README.md)。
