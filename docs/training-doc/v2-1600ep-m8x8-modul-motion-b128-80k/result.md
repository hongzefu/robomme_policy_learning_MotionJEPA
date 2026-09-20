# 八卡motion训练80k完成，最终checkpoint真实加载通过

**80,000步已于2026-09-19 03:24:35 UTC（9月18日23:24:35 EDT）正常结束，EXIT_CODE=0，总墙钟耗时21小时37分30秒。** 最终79999 checkpoint已完成异步保存，全部16个预期checkpoint在位；最终checkpoint的真实CPU加载及配置同源核验通过。训练从2026-09-18 05:47:05 UTC的clean `2f10473161b760f16d9240d3c2959ff326cde66b`（commitV10.2Beta）起跑，30项preflight全部通过，实际8×A100-SXM4-80GB、batch128、fsdp8、workers16、mesh(1,8)。完整配置与启动命令见 [launch.md](launch.md)，起跑原始状态见 [launch.actual.json](records/launch.actual.json)，结束实测见 [training_completion.json](records/training_completion.json)。

原训练tmux全名为mv2-prod、Python PID545524，已自然退出；WandB [uzv8avpq](https://wandb.ai/hongzefu-university-of-michigan/robomme-framesamp/runs/uzv8avpq)在结束日志中确认同步。GPU全程15秒采样；mv2-dense另以500ms采八卡、按原计划运行1800秒。以下性能统计严格保留各自声明的窗口，不把两种采样或初始化停顿混作同一稳态结果。

起跑后第108步保存了290份受Git管理的源码SHA及原权限，将主仓src/scripts/packages锁为只读；当时锁前后字节SHA一致、HEAD为Beta且porcelain为空，见 [source_lock.json](records/source_lock.json) 与 [完整锁定记录](records/source_lock.before.json)。09-20收尾复核的290份既有源码和uv.lock仍与起跑指纹一致。当前主仓另有6份未跟踪的`hf_export/`文件，属于其他在途工作，本轮原样保留、不暂存，不能将当前整个主仓称为clean。结束文档及配对commitV10.2仍在独立开发副本完成。

## 完成判据与最终指标

| 项目 | 实测 |
|---|---|
| runner起止UTC | 09-18 05:47:05 → 09-19 03:24:35 |
| runner总墙钟 | 77850秒，即21小时37分30秒，包含初始化、trace导出、存盘与退出收尾 |
| 完成步数 / 最终checkpoint | 80000 / 79999 |
| 60k checkpoint完成保存 | 09-18 18:01:58.713219 EDT |
| 最终checkpoint完成保存 | 09-18 23:24:19.723562 EDT |
| 退出码 | 0 |
| checkpoint集合 | 5000至75000每5000一步，加79999，共16份 |
| 指标覆盖 | step0至79900，每100步一条，共800条；五类标量共4000个均有限，hex与十进制一致 |
| 最后记录loss / grad_norm | 0.001474518794566393 / 0.020426495000720024（step79900） |
| 最后10条记录loss均值 | 0.00144775704247877 |

指标行按训练入口的记录区间汇总，step79900不是step79999的单步loss；末尾99步没有另外写出标量。80k完成依据来自完整进度、79999最终保存完成、checkpoint提交元数据和EXIT_CODE=0。完整数据见 [逐次训练指标](records/metrics.full.jsonl)、[全程清洗日志](records/train.complete.summary.log) 与 [checkpoint清单](records/checkpoint_inventory.json)。清洗前日志1233164字节，清洗后263631字节、1707行，800条Step指标行及唯一EXIT_CODE=0均保留；原始/归档文件SHA见 [归档核验](records/completion_archive_manifest.json)。

最终79999在09-20 19:55:55→19:56:22 UTC经既有 `scripts/training/g0/check_config_provenance.py` 实际CPU恢复，诊断启动于开发副本clean `d7370f5d40bf51488e039fe290740773130d8d68`，tmux为`mv2-finish-l0-20260920`，27秒、EXIT_CODE=0。`NORM_STATS_SAME`、`LIB_PROVENANCE_MATCH`、`MOTION_STORE_PATH`、`CKPT_PARAM_TREE`全部通过，参数树65/65、missing=extra=0、motion参数存在；原始dtype为42个float32可训练叶、23个bfloat16冻结图像叶。此验证真实读取最终权重并装入模型，没有启动策略rollout。见 [加载核验JSON](records/completion_l0.json) 与 [核验日志](records/completion-l0.summary.log)。

16份checkpoint逐一核实提交时间、65叶元数据、OCDBT清单、非空数据文件与相同norm_stats SHA；60000保留为后续主比较锚点，79999保留为最终模型。checkpoint文件共190161501591字节，继续留在v1-store，不进Git；本轮没有重写或清理任何checkpoint。

训练loss下降只说明训练目标的表现，不能据此认定motion提高策略成功率。正式策略评估、同权重motion全遮消融和60000对基线59999的比较仍另起计划。本节完成核验不改变下文已获用户接受的分窗报告，也不补写原始trace未覆盖的设备数据。

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

历史第300步按0.987454秒/步外推，剩余79700步约21.861小时，预计2026-09-19 03:51 UTC完成；实际runner于03:24:35 UTC结束。保留该当时估计以区分预测与实测，正式超参全程沿用用户决定。

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

09-20用户继续要求「跑完了吗 结束后commit」。本轮已确认80k正常结束并补齐最终档案，按起跑Beta配对创建commitV10.2，不合并或改写Beta历史。主仓保持起跑版本，其他在途导出文件原样保留，档案提交及推送均从独立开发副本完成。

历史首段收尾时间为09-18 06:17:01 UTC，当时训练仍在、指标至step1300且主仓clean Beta，`mv2-dense`已自然退出。此次完整训练收尾以本页开头及完成判据为准；策略效果尚未评估，不将训练完成写成评估通过。

## 本次归档与复现

`records/launch.actual.json`和`preflight.log`固定真实起跑；`runtime.json`固定卡数、batch、mesh及worker；`source_lock.before.json`、`source_lock.json`和`source_recheck_step900.json`证明290份Git源码在锁定及第900步复核时均未变、主仓clean且只读。第900步指标快照与清洗日志只描述当时进度，不包含或预设80k结束状态。

`step_timing.jsonl`、`step_trace_metadata.json`、`gpu_util_500ms.step300.snapshot.csv`和`perf_step300_input.json`固定主线程300步及稳态采样输入；主线程与NVML结果、失败日志、短窗JSON、原始覆盖复核和真实回归分别按上文链接保存。`trace_export_resume.json`记录大trace的SHA与位置，原始文件保留在v1-store，不进Git；checkpoint同理。

`gpu_util_500ms.full.csv`、`dense_capture_complete.json`、`dense.summary.log`补全30分钟采样证据；`metrics.at_dense_end.snapshot.jsonl`与`train.at_dense_end.summary.log`保存采样结束时的指标和清洗日志，不覆盖第900步快照。汇总器覆盖修补与启动归档提交为`70ca65c2c9a09ebcc4536cd64d7e551fa851f4bd`，分窗口径与密集采样归档为`d7370f5d40bf51488e039fe290740773130d8d68`；这些运行期间的开发副本提交均未回拉到训练主仓。

本次新增`training_completion.json`、`checkpoint_inventory.json`、`metrics.full.jsonl`、`run_meta.final.json`、`gpu_util_15s_full.csv`、`completion_l0.json`、`completion-l0.summary.log`、`train.complete.summary.log`和`completion_archive_manifest.json`。15秒全程采样只保留监控证据，不据此新增步骤级稳态或慢步瓶颈结论。全部既有首段快照保留，完整日志和指标另用新文件名归档。

实际短窗命令为 `uv run --no-sync python scripts/training/tests/summarize_step_timing.py --records v1-store/bench/v2-1600ep-m8x8-modul-motion-b128-80k --gpu-csv v1-store/bench/v2-1600ep-m8x8-modul-motion-b128-80k/gpu_util_500ms.step300.snapshot.csv --warmup-steps 5 --end-step 24 --out v1-store/bench/v2-1600ep-m8x8-modul-motion-b128-80k/perf_early_step5_24.json`。主线程与原始覆盖的独立诊断脚本保存在 [step300-host-and-coverage.py](records/step300-host-and-coverage.py)，实际执行位置为主仓 `v1-store/logs/mv2-step300-host-and-coverage.py`；恢复到该位置后才能使用其相对路径。所有输出均拒绝覆盖，复跑应使用独立输出位置。
