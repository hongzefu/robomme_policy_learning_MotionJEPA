# 八卡80k已起跑，按用户确认口径完成分窗性能报告

2026-09-18 05:47:05 UTC从clean `2f10473161b760f16d9240d3c2959ff326cde66b`（commitV10.2Beta）启动 `v2-1600ep-m8x8-modul-motion-b128-80k`。30项preflight全部通过；运行时实际8×A100-SXM4-80GB、batch128、fsdp8、workers16、mesh(1,8)，source/norm/motion表指纹与起跑档案一致。完整配置与启动命令见 [launch.md](launch.md)，实际进程、参数和独立空卡检查见 [launch.actual.json](records/launch.actual.json)。训练仍在运行，不把起跑结果称为80k训练完成。

tmux全名为mv2-prod，Python训练PID545524，WandB为 [uzv8avpq](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/uzv8avpq)。GPU全程15秒采样，另有mv2-dense以500ms采八卡、按原计划运行1800秒；统计严格限制在声明的步骤窗口。密集采样与原始trace保存在同名v1-store/bench目录。

模型初始化已完成并进入训练。第108步时保存了290份受Git管理的源码SHA及原权限，将主仓src/scripts/packages锁为只读；锁前后字节SHA一致，HEAD仍是Beta，porcelain为空，见 [source_lock.json](records/source_lock.json) 与 [完整锁定记录](records/source_lock.before.json)。文档回写在独立开发副本完成，主仓代码和依赖保持固定。

## 第100至299步的稳态主线程和利用率

环境为8×A100-SXM4-80GB、AWS本地NVMe RAID `/dev/md0` XFS，batch128、workers16、fsdp8、mesh(1,8)、seed42。排除0–99步预热，正式窗口为2026-09-18 05:53:10.733674→05:56:28.224496 UTC，共200步。主线程300行计时全部完整、completed全部为true。数据见 [主线程与NVML统计](records/perf_step300_host_nvml.json)、[逐步计时](records/step_timing.jsonl)、[500ms采样快照](records/gpu_util_500ms.step300.snapshot.csv)。

| 指标 | 实测 |
|---|---:|
| 平均墙钟步时 | 0.987454秒 |
| 吞吐 | 129.626样本/秒 |
| 主线程取batch等待 | 0.143053秒/步，占主线程步时14.4906% |
| 主线程train_dispatch | 0.837123秒/步 |
| 主线程logging | 0.006968秒/步 |
| checkpoint | 此窗口没有保存 |
| 八卡GPU利用率均值 | 98.5777% |
| GPU利用率为0的采样占比 | 0% |
| 慢步采样均值 / 0%占比 | 95.0536% / 0%，56条 |
| 其他步采样均值 / 0%占比 | 98.6415% / 0%，3096条 |

慢步定义为host_step超过均值的1.5倍，阈值1.480819秒，命中step100和200这两个记录标量的步骤。500ms共取得3152条卡级读数，即每卡394条；NVML自身存在采样周期，相同连续读数不视为新增独立证据。各卡均值98.2538%–98.7208%，各卡0%占比均为0。

完整密集采样于05:46:56→06:16:56 UTC按1800秒自然结束，`SAMPLER_EXIT_CODE=124`为timeout正常到时，驱动核验后`DENSE_CAPTURE=PASS`、`EXIT_CODE=0`；未提前终止。每卡3579条、总28632条，首尾实际跨度1798.265–1798.270秒，平均间隔0.502590秒，最大间隔0.878秒；stderr为空。完整CSV为1077226字节，SHA256为`7ea2685075c59589ad6abacd62e85c01cbf32e2e1bba4c38c26f48b4a3af764f`。第300步快照是完整文件的精确字节前缀，因此上述稳态统计输入与完整采样同源；不将包含初始化和导出停顿的30分钟全程均值当稳态利用率。见 [完整CSV](records/gpu_util_500ms.full.csv)、[采样验收](records/dense_capture_complete.json) 与 [采样日志](records/dense.summary.log)。

`train_dispatch`含异步提交和可能的背压，不是GPU计算耗时。取batch等待也可能与设备工作重叠，14.4906%不能解释为GPU空闲或可直接节省的时间。由于下述设备覆盖缺口，当前窗口不能计算等待期间全部GPU同时空闲的时间，也不能据高util判断计算/通信瓶颈或决定调整worker。

按0.987454秒/步外推，第300步剩余79700步约21.861小时；从恢复训练时刻估算约2026-09-19 03:51 UTC完成。该估算不含后续checkpoint、潜在波动及其他停顿，不是完成承诺。正式超参保持用户决定。

## 第300步的一次性停顿与恢复

step299结束于05:56:28.224496 UTC，trace结束后的进度于05:59:29.119恢复，间隔180.894504秒，包含一次flush、trace导出和日志收尾。这期间训练主线程处理CPU导出，GPU暂时为0%；05:59:30左右实查八卡均99%，随后Step300及后续步骤继续，无训练EXIT_CODE。进度条一度显示1223小时，是将该一次性停顿混入滑动步速造成；它不作为ETA依据。证据见 [导出与恢复记录](records/trace_export_resume.json)。

导出留下739736011字节的原始XPlane与356898751字节的JSON压缩文件。两者的SHA、位置与恢复时间完整保存，原始大trace不进Git。默认关闭的计时仅在本轮前300步启用，没有加入逐步强制同步，之后不再采trace。

## 设备事件完整性失败与证据边界

`mv2-perf`从clean Beta启动，以 `summarize_step_timing.py --warmup-steps 100 --end-step 299` 汇总，05:59:42→06:01:20 UTC，EXIT_CODE=1：稳态窗口没有物理stream设备事件，未生成误导性的通过报告。完整失败见 [汇总日志](records/perf-step300.summary.log)。

JSON包含26162279个事件，低于已提高到100000000的查看器上限，主线程0–299步均在。原始XPlane的errors/warnings均为空，但直接解析原始GPU planes与JSON逐卡比较，1979015个物理设备事件的数量及终点完全一致：八卡均在相对146.09–146.17秒停止，落在step25中；step100–299为相对219.53–417.02秒，八卡设备事件数均为0。**原始采集已缺失，重导出不能补回；errors/warnings为空也不足以证明完整。** 证据见 [JSON覆盖](records/trace_coverage_diagnostic.json) 与 [原始XPlane覆盖复核](records/xplane_coverage_corrected.json)。

JAX0.5.3对应的 [XLA固定提交](https://github.com/jax-ml/jax/blob/jax-v0.5.3/third_party/xla/workspace.bzl) 为 `df971129bd82e381954da0185b534220e21798a4`；其 [CuptiTracerCollectorOptions](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/backends/profiler/gpu/cupti_collector.h) 将callback/activity上限各设为2097152，[GpuTracer::DoStart](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/backends/profiler/gpu/device_tracer_cuda.cc) 直接使用默认值。结合本轮约198万设备事件后同时截止，采集层上限是高度吻合的原因推断；未将其冒充为日志中的显式丢弃计数。此次没有改依赖、没有注入正在运行的进程，也没有重启训练。

第一次原始protobuf探索把已归零的XPlane时间戳误写成1970年UTC；未据此改训练或得出性能结论。后续改用与JSON一致的相对微秒，并逐卡核对数量和终点，`xplane_coverage_corrected.json`取代该探索记录中的时间列和跨时钟比较。探索记录仍保留在v1-store。

开发副本随后为 `summarize_step_timing.py::require_gpu_events_each_step` 增加逐GPU、逐步骤的物理事件覆盖检查，拒绝仅窗口前半段有事件以及中间整步缺失；允许一个异步kernel跨越多个主线程步骤。它不改变训练计时、超参或正在运行的源码，也不声称仅凭每步有事件即可证明逐kernel零丢失。`uv run --no-sync pytest -q scripts/training/tests/test_step_timing_export.py scripts/training/tests/test_motion_v2_tools.py` 为14项通过/1.22秒；同一真实trace的5–24步在修补后重新汇总，八卡各20步覆盖、所有既有数值完全不变，06:07:49.792262→06:09:38.509974 UTC，见 [真实验证记录](records/coverage_guard_validation.json) 和 [增加覆盖字段的JSON](records/perf_early_step5_24_guarded.json)。验证时开发副本尚有未提交修补，记录源码SHA，不将其冒充为Beta内的代码。

## 第5至24步的独立早期短窗诊断

为保留可用的真实设备证据，另外运行相同Beta汇总器，显式改统计窗口为5–24步；此操作仅分析已存在的文件，不运行新的训练。`mv2-perf-short`于06:04:29→06:06:15 UTC完成，EXIT_CODE=0。原始XPlane与JSON在八卡逐卡数量、时间终点一致，该短窗每卡有192906–194384个物理stream事件；没有将XLA派生轨道重复计时。结果见 [早期短窗JSON](records/perf_early_step5_24.json) 与 [日志](records/perf-early.summary.log)。日志通用标签`STEADY_PERF=PASS`仅表示传入的20步窗口汇总成功，**不能解释成原定100–299步已通过**。

该窗口为05:51:37.635010→05:51:57.061970 UTC，共20步，均值0.971348秒/步；它只跳过前5步，窗口较短，不取代正式稳态窗口或用来估计全程ETA。按kernel名称/HLO标签归类，每卡每步GEMM累计621.311ms、通信129.367ms、传输/清零2.323ms、其他188.800ms，物理stream忙碌区间的并集为941.256ms/步。不同stream可重叠，累计分类、busy并集和主线程阶段不能直接相加。

主线程取batch等待均值139.945ms/步；20步中等待与所有GPU同时无事件重叠总计3.787ms，即0.189ms/步。短窗内取batch等待主要与GPU工作重叠，设备分类中GEMM用时最大；该观察的适用范围仅为这20步，不能外推原稳态窗的瓶颈。500ms读数每卡38条、共304条，util均值98.8553%、0%占比0；无步骤超过1.456627秒的慢步阈值，因此慢步分层为无样本，不写0%。

## 用户决定与后续

用户原话「继续工作」「纳入计时与 trace，按计划提供实测分解」。发现原始设备采集缺口后，用户进一步确认「保持训练，接受明确标注的分窗报告（推荐）」。据此采用第100–299步的步时、主线程等待、NVML统计，加第5–24步的独立设备短窗诊断；正式训练保持运行。**这是用户接受的分窗口径，第100–299步设备分解仍未满足原判据，不因批准分窗而将缺失写成完整。**

80k训练继续运行，未宣称已训完。正式策略评估及同权重motion全遮消融仍另起计划；当前loss或运行正常均不证明motion提升策略成功率。主仓保持Beta和只读源码，全部档案修改在独立开发副本完成。

本轮正式起跑、用户接受的分窗报告和完整采样归档已完成。06:17:01 UTC收尾核查时，训练PID545524仍在，最新标量记录至step1300，各已记录loss/梯度范数/参数范数均有限，主仓仍为clean Beta；`mv2-dense`已自然退出，`mv2-prod`继续保留。训练结束后再补全程指标和配对V10.2结论，不预填成功或策略效果。

## 本次归档与复现

`records/launch.actual.json`和`preflight.log`固定真实起跑；`runtime.json`固定卡数、batch、mesh及worker；`source_lock.before.json`、`source_lock.json`和`source_recheck_step900.json`证明290份Git源码在锁定及第900步复核时均未变、主仓clean且只读。第900步指标快照与清洗日志只描述当时进度，不包含或预设80k结束状态。

`step_timing.jsonl`、`step_trace_metadata.json`、`gpu_util_500ms.step300.snapshot.csv`和`perf_step300_input.json`固定主线程300步及稳态采样输入；主线程与NVML结果、失败日志、短窗JSON、原始覆盖复核和真实回归分别按上文链接保存。`trace_export_resume.json`记录大trace的SHA与位置，原始文件保留在v1-store，不进Git；checkpoint同理。

`gpu_util_500ms.full.csv`、`dense_capture_complete.json`、`dense.summary.log`补全30分钟采样证据；`metrics.at_dense_end.snapshot.jsonl`与`train.at_dense_end.summary.log`保存采样结束时的指标和清洗日志，不覆盖第900步快照。汇总器覆盖修补与启动归档提交为`70ca65c2c9a09ebcc4536cd64d7e551fa851f4bd`；此后的文档提交只固定用户分窗决定和采样收尾，不进入正在训练的主仓。

实际短窗命令为 `uv run --no-sync python scripts/training/tests/summarize_step_timing.py --records v1-store/bench/v2-1600ep-m8x8-modul-motion-b128-80k --gpu-csv v1-store/bench/v2-1600ep-m8x8-modul-motion-b128-80k/gpu_util_500ms.step300.snapshot.csv --warmup-steps 5 --end-step 24 --out v1-store/bench/v2-1600ep-m8x8-modul-motion-b128-80k/perf_early_step5_24.json`。主线程与原始覆盖的独立诊断脚本保存在 [step300-host-and-coverage.py](records/step300-host-and-coverage.py)，实际执行位置为主仓 `v1-store/logs/mv2-step300-host-and-coverage.py`；恢复到该位置后才能使用其相对路径。所有输出均拒绝覆盖，复跑应使用独立输出位置。
