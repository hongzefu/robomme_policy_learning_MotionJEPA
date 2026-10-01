# 两库追加20步off重复性诊断

## 1. 结论与指标速览

**两库追加off均已完成，严格重复比较为DIFFER/FAIL。** 新run真实保存、CPU恢复及独立退出0均通过；与各自旧off相比，全部21次取批、20次模型输入/RNG及记录指纹一致，但每库80/100标量、147/201状态叶摘要不同。关闭计时包装的正常流程也出现了运行间差异，具体机制未确定，见[result](result.md)。原[off/on判定失败](../orig80k-timing20-0926/result.md)保留，perf与80k尚未放行。

| 库 | 新run档案 | GPU |
|---|---|---|
| full | [timing20-full-off-0926-20260927T003051Z](../timing20-full-off-0926-20260927T003051Z/README.md) | 0–3 |
| counting | [timing20-count-off-0926-20260927T003051Z](../timing20-count-off-0926-20260927T003051Z/README.md) | 4–7 |

## 2. 版本与代码状态

旧off基线HEAD为`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`，其失败组结果已在`c6d726f3834f871fcef10f33fa2530321bee534f`归档。新诊断实际`commitV11.15Beta`为`b60ec2b59e0ba012cae0998c5f8713b2aef07b2e`；相对b0只有docs/原计划及归档变化，三规则内容固定d710，受保护训练源码、工具和依赖逐字一致。两个run及完整独立核验完成前保持HEAD与venv冻结，后续提交不替换原运行锚点。

## 3. 启动与配置还原

[完整launch](launch.md)包含参数绑定、两off并行启动、原生退出、真实恢复及独立CPU比较正文。继续使用原45b4训练wrapper及原f8 runtime；Git源码/工作树/runtime绑定不变。旧基线固定摘要见[baseline-pins](records/baseline-pins.json)，起跑前重新核对实际文件，不改旧records。

## 4. 数据集与划分口径

沿用公开16任务1600集/476857执行样本，以及同源counting四任务400集/189035执行样本。两库4×4 packed、source/manifest、history、tokenizer和pi05_base均与旧off相同。full使用原版f332…norm，counting使用自身a770…norm；全部完整路径与SHA见两run档案及launch。

## 5. 关键超参

smoke20、b64/fsdp4/workers4/seed42、modul512/4×4/max32、warmup10000、peak/decay5e-5、decay_steps100000、EMA0.999、log100/save10000/keep10000不变。W&B在线，normal env，不增加确定性flags或放宽容差；仅运行off，两run分别真保存和CPU恢复checkpoint19。

## 6. 硬件、调度与耗时

AWS本机8×A100-SXM4-80GB、md0 XFS本地NVMe。full0–3/count4–7均于2026-09-27 00:54:30 UTC起跑，count于01:05:12结束（642秒）、full于01:05:30结束（660秒）。输出/cache分别全新，准备TAG不代替这些实际时间。该耗时包含取证、编译及保存恢复，不用于吞吐或ETA结论。

## 7. 实际执行与记录

每run实际记录21次fetch、20次模型输入/RNG、20步五标量与201叶完整末态，真实保存19并恢复61个EMA叶。两侧内容、真实恢复及原生退出/capture/父实际返回均通过；没有将完整201叶摘要写成201叶均从checkpoint恢复。原始记录保留，无配置或容差改动。

## 8. 独立诊断判定

CPU比较分别以真实旧b0/新Beta调用原read_side，重核各自来源、finite、输入与恢复。完整配置仅允许原IDENTITY_FIELDS的run/输出身份差异；provenance只投影掉已分别验证的git_head_of_cwd，其他来源及payload严格相同。原compare_sides/judge的同HEAD要求不修改，也不冒用其正式PASS。

`OFF_REPEAT_DIAGNOSTIC_READ=COMPLETE`和进程0只说明读取/比较正常完成；严格比较由`OFF_REPEAT_DIAGNOSTIC_EQ=PASS|FAIL`及JSON的MATCH/DIFFER分别说明。不同就是诊断比较FAIL；一次MATCH也不证明一般确定性，不抵销原off/on失败。

实际比较于01:05:58—01:05:59 UTC完成，程序与外部父返回0，输出`OFF_REPEAT_DIAGNOSTIC_EQ=FAIL`，JSON为DIFFER/comparison_status=FAIL。两库均4种标量各20项不同、param_norm全部相同；params37/61、EMA36/61、opt74/78、step0/1叶摘要不同。原同HEAD judge没有被调用或修改。

## 9. 用户决定记录

根代理提出在当前取证结束后“两库各追加一次相同配置的20步off复验（4+4并行），保持严格判据，仅用于诊断，不替代正式对拍闸门”，用户答复「同意」。既有「尽可能并行做」「你有8张卡」「继续工作 一路做到起泡前 有问题问用户」继续有效，本阶段不启动perf或80k。

随后针对normal重复性失败提出：包装正确性闸门改为两库各一对确定性档20步off/on，输入、标量与完整状态仍逐位一致；仅此对拍启用`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，通过后再进入normal 300步perf。用户答复「同意」。此授权改变下一阶段验证环境与结论范围，不改写本次normal失败；其他训练参数不变。

## 10. 已知差异与原因边界

旧off/on每库输入/RNG全同，但100标量80项不同、末态201叶147叶不同。只读核查未证明首步数学函数被改，也未证明具体后端算法或时序是原因。完整初态没有摘要，param_norm只聚合特定kernel，不能把相同norm当作完整初态一致证据。详见[机理记录](../orig80k-timing20-0926/records/diagnostics/timing20-step0-mechanism-note-20260927.md)。

## 11. 当前结论与下一步

本次证明所测正常流程在关闭包装时也未逐位自复现；计时开关不是差异出现的必要条件，仍不能确定具体机制。下一阶段按新增明确授权实现smoke-only确定性验证档与严格隔离测试，再从新代码Beta重跑两库各一对20步；perf/prod仍保持normal，旧normal FAIL永久保留。新验证尚无实测结论，当前不启动perf或80k。

## 12. 归档文件清单

起跑前文本由b60 Beta还原；本README、launch、两run README和result追加实测。已按[白名单](records/formal-import-whitelist.json)及[独立报告增量](records/independent-reports-import-addendum.json)导入49项、863446 B，见[导入回执](records/import-receipt.json)。比较JSON、清洗日志、原生退出及父侧证分别保留，权重留run根；旧b0与新b60真实身份不改。获准前的确定性候选说明保留其当时文字，最新批准以第9节为准。
