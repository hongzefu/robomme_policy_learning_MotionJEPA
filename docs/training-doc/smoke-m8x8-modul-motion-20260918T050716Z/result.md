# 八卡20步smoke通过，trace导出上限问题已修复

从clean `5b3b812961ed089f94546edfda8c116d1e7564c1` 起跑，2026-09-18 05:12:33→05:18:15 UTC，训练342秒、EXIT_CODE=0。实际8×A100-SXM4-80GB、batch128、fsdp8、workers16、mesh(1,8)，存储AWS本地NVMe RAID `/dev/md0` XFS。准确命令和进程见 [launch.actual.json](records/launch.actual.json)，启动口径见 [launch.md](launch.md)。20步五项标量完整且有限，最终loss为0.0843的日志显示值；本项只证明生产档位可运行。

## 训练与checkpoint验收

30项preflight全部通过。参数树双向精确匹配65/65叶，missing/extra/shape_mismatch均0；六个modulation参数叶和四个motion参数叶齐全。checkpoint保留42个可训练f32叶及23个冻结图像bf16叶。配置同源检查五项全部通过，budget在run快照、仓库YAML与固定期望三侧均为160，norm_stats原始SHA仍为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。

```text
PREFLIGHT=PASS n=30
SMOKE20=PASS steps=20 finite=1 exit_code=0
PARAM_TREE_EXACT=PASS n_model=65 n_ckpt=65 missing=0 extra=0 shape_mismatch=0
MEM_PARAMS=PASS n=10
TIC_L0=PASS
BUDGET_CONSISTENT=PASS snapshot=160 repo_yaml=160 expected=160
```

参数树与同源明细见 [param_tree.json](records/param_tree.json)、[config_provenance.json](records/config_provenance.json)，实际设备配置见 [runtime.json](records/runtime.json)，20步原始指标见 [metrics.jsonl](records/metrics.jsonl)。两个真实入口失败即停负例（占卡、错误YAML SHA）见启动页及对应日志。

## trace意外、修复与验证

用户要求「纳入计时与 trace，按计划提供实测分解」。20步训练与checkpoint验收均成功，但原始stage在计时汇总时EXIT_CODE=1：JAX0.5.3默认查看器JSON仅保留100万个时段事件，被首步编译事件占满，只有第0步注解和GPU元数据，没有完整设备事件。原始日志原样保留失败，不改写为成功。

原始XPlane为332375058字节，SHA `ffcae70e84d4eaf44ef3f3d076d573bd2f17e6f883b6960fbcad1d17ab752a04`，包含12个plane，errors/warnings均为空。使用同一JAX导出器提高查看器上限后恢复11530267个JSON事件对象，全部20步和8张GPU事件齐全，首步时间与原始JSON相同。该上限属于 [XLA查看器导出逻辑](https://raw.githubusercontent.com/openxla/xla/main/xla/tsl/profiler/convert/xplane_to_trace_events.cc)；原始采集数据没有重采。

修复只作用于计时与后处理：`step_timing.py::complete_trace_export` 在导出期间把 `TF_PROFILER_TRACE_VIEWER_MAX_EVENTS` 至少设为100000000，结束后恢复原值，记录实际上限；汇总器达到上限即拒绝结论，只计物理GPU stream，避免派生XLA注解重复累计。`reexport_step_trace.py` 用原始XPlane与导出JSON的双SHA绑定恢复目录，汇总器拒绝误用其他run或已改变的JSON。训练数学、超参、采集步数和原有收尾一次同步保持原口径。

14项CPU测试通过（0.32秒）；真实GPU探针故意预设查看器上限为1，修补后完整保留步骤0/1/2与406个GPU事件、9328个JSON对象，收尾同步仅一次，外部上限恢复为1，EXIT0。probe使用修补工作树，结果见 [trace-export-probe.json](records/trace-export-probe.json)。在本次真实八卡XPlane上调用新的重导出CLI和汇总器，05:34:03→05:35:56 UTC，共113秒、EXIT0；脚本SHA与工作树状态记录在 [trace-check.summary.log](records/trace-check.summary.log)，不将原训练HEAD冒充修补后的源码版本。

## 计时功能验收与保留范围

第5–19步窗口共15步，完整覆盖八卡物理stream及376个500ms NVML样本，派生轨道排除数为0。主线程平均取batch等待0.1304秒，设备区间平均忙碌960.75毫秒/步，分类累计为GEMM618.03、通信128.60、传输26.39、其他188.34毫秒/步。这些异步量不能相加为互斥占比。窗口包含第19步checkpoint和逐步日志，平均步时1.5713秒、GPU均值61.80%、0%样本17.29%；仅用于检验计时工具，不用于正式80k的性能或ETA。正式报告按第100–299步重新测量。

完整结果见 [timing_smoke.json](records/timing_smoke.json) 和 [trace_reexport.json](records/trace_reexport.json)。原始XPlane、初次截断JSON及恢复后的完整trace均保留在 `v1-store/bench/smoke-m8x8-modul-motion-20260918T050716Z/`，不入Git。所有验收通过后仅清理本轮smoke的run权重与快照，指标、trace和日志保留；清理完成状态另记于 [cleanup.json](records/cleanup.json)。正式训练尚未起跑。
