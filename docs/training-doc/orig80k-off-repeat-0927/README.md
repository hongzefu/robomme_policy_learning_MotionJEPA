# 两库追加20步off重复性诊断

## 1. 结论与指标速览

本阶段尚未启动。用户已批准两库各追加一次相同配置20步off、4+4并行，仅诊断关闭计时包装时是否也有运行间差异。原[off/on判定失败](../orig80k-timing20-0926/result.md)保留，perf与80k不放行。

| 库 | 新run档案 | GPU |
|---|---|---|
| full | [timing20-full-off-0926-20260927T003051Z](../timing20-full-off-0926-20260927T003051Z/README.md) | 0–3 |
| counting | [timing20-count-off-0926-20260927T003051Z](../timing20-count-off-0926-20260927T003051Z/README.md) | 4–7 |

## 2. 版本与代码状态

旧off基线HEAD为`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`，其失败组结果已在`c6d726f3834f871fcef10f33fa2530321bee534f`归档。新诊断拟由`commitV11.15Beta`锁定，完整SHA待实际生成；相对b0只允许docs/原计划变动，三规则内容固定d710，训练源码、工具与依赖保持字节一致。运行期间冻结HEAD和venv。

## 3. 启动与配置还原

[完整launch](launch.md)包含参数绑定、两off并行启动、原生退出、真实恢复及独立CPU比较正文。继续使用原45b4训练wrapper及原f8 runtime；Git源码/工作树/runtime绑定不变。旧基线固定摘要见[baseline-pins](records/baseline-pins.json)，起跑前重新核对实际文件，不改旧records。

## 4. 数据集与划分口径

沿用公开16任务1600集/476857执行样本，以及同源counting四任务400集/189035执行样本。两库4×4 packed、source/manifest、history、tokenizer和pi05_base均与旧off相同。full使用原版f332…norm，counting使用自身a770…norm；全部完整路径与SHA见两run档案及launch。

## 5. 关键超参

smoke20、b64/fsdp4/workers4/seed42、modul512/4×4/max32、warmup10000、peak/decay5e-5、decay_steps100000、EMA0.999、log100/save10000/keep10000不变。W&B在线，normal env，不增加确定性flags或放宽容差；仅运行off，两run分别真保存和CPU恢复checkpoint19。

## 6. 硬件、调度与耗时

AWS本机8×A100-SXM4-80GB、md0 XFS本地NVMe。full0–3/count4–7同时起跑，输出及cache分别全新。准备TAG`20260927T003051Z`不是开始时间；现场GPU/内存/SHM/可用空间与实际起止待记录。共同取证开销不用于吞吐或ETA结论。

## 7. 预定执行与记录

每run记录21次fetch、实际20次模型输入/RNG、20步五标量与201叶完整末态，真实保存19并恢复61个EMA叶。每侧须先完成内容、恢复、原生日志及父实际返回联合核验，再接受为有效诊断输入；失败保留现场，不自动改配置或重跑。

## 8. 独立诊断判定

CPU比较分别以真实旧b0/新Beta调用原read_side，重核各自来源、finite、输入与恢复。完整配置仅允许原IDENTITY_FIELDS的run/输出身份差异；provenance只投影掉已分别验证的git_head_of_cwd，其他来源及payload严格相同。原compare_sides/judge的同HEAD要求不修改，也不冒用其正式PASS。

`OFF_REPEAT_DIAGNOSTIC_READ=COMPLETE`和进程0只说明读取/比较正常完成；严格比较由`OFF_REPEAT_DIAGNOSTIC_EQ=PASS|FAIL`及JSON的MATCH/DIFFER分别说明。不同就是诊断比较FAIL；一次MATCH也不证明一般确定性，不抵销原off/on失败。

## 9. 用户决定记录

根代理提出在当前取证结束后“两库各追加一次相同配置的20步off复验（4+4并行），保持严格判据，仅用于诊断，不替代正式对拍闸门”，用户答复「同意」。既有「尽可能并行做」「你有8张卡」「继续工作 一路做到起泡前 有问题问用户」继续有效，本阶段不启动perf或80k。

## 10. 已知差异与原因边界

旧off/on每库输入/RNG全同，但100标量80项不同、末态201叶147叶不同。只读核查未证明首步数学函数被改，也未证明具体后端算法或时序是原因。完整初态没有摘要，param_norm只聚合特定kernel，不能把相同norm当作完整初态一致证据。详见[机理记录](../orig80k-timing20-0926/records/diagnostics/timing20-step0-mechanism-note-20260927.md)。

## 11. 当前结论与下一步

旧阶段已结束归档，本阶段等待新Beta及现场前置，尚无新运行或比较结果。新off重复不同可证明计时开关不是差异出现的必要条件，但不能直接确定机制；重复相同也只是进一步线索。结果出来后再按实测确定处置，原严格闸门继续保留。

## 12. 归档文件清单

起跑前为本README、launch、baseline-pins与两run README。运行后另归档实际小记录、清洗日志、原生退出与父侧证、独立诊断JSON；代码由Beta还原，不复制脚本、配置、权重或cache。旧基线保留原b0路径与SHA，新记录保留实际新HEAD，比较输出独立于两run记录根。
