# 8帧8×8训练验收总览

12条正式轨迹全部完成，C32、M32、C8、M8四组完整1000步gate均PASS。每组先证明A1/A2可逐位重复，再证明参考A1与候选B逐位一致；五项标量、全部11份完整TrainState、11份输入摘要、前8000个样本索引、有限性、活性、导入来源及真实norm_stats检查全部通过。训练轨迹与四profile三batch全梯度均已通过；真实checkpoint推理仍待后续完成。

## 版本与执行边界

REF为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，CAND为 `c08ec2060a544af1869c1e24f755e536150569ca`。12条正式轨迹运行期间，主树始终保持clean `38f0db46db19a3b645613e04fba3f6e635b47054`，期间未写入或提交文档；A侧在REF工作树并显式设置PYTHONPATH，B侧在主树，实际来源逐项检查。两波按计划串行，C组只用物理GPU4,5，M组只用6,7，同profile的A1→A2→B始终同卡对。

数据为400ep库；32帧组读旧4×4库，8帧参考直接读源npy，8帧候选读新8×8库。全部显式400ep norm_stats（文件SHA `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`），实际loader数组与文件解析数组逐项匹配。batch8、worker4、seed42、fsdp2、1000更新、log/save interval1均是启动覆盖。每条1000×8=8000个样本用于更新，采样记录包含预取，共8072项。

用户要求“一口气全做完”，并已确认四GPU、双重norm_stats与checkpoint配置驱动评估长度。用户后续同意“排错关闭完整状态摘要（推荐）”，只适用于四个-s100；排错完成后已清理记录、run根和日志，不作正式等价证据。详见[用户决定](decisions.md)。

## 正式运行结果

首条开始于 `2026-09-14T04:41:00.000Z`，末条成功结束于 `2026-09-14T11:24:14.000Z`，总墙钟6小时43分14秒。下表步时与util为各run独立稳态统计，不把共享主机上的数值当作性能收益证明。

| run | 输入实现 | 墙钟 | 均值秒/步 | util均值 | 末步loss |
|---|---|---:|---:|---:|---:|
| [t8-c32-a1](../t8-c32-a1/result.md) | packed | 1小时8分6秒 | 1.5555 | 96.26% | 0.028470 |
| [t8-c32-a2](../t8-c32-a2/result.md) | packed | 1小时1分36秒 | 1.5554 | 96.32% | 0.028470 |
| [t8-c32-b](../t8-c32-b/result.md) | packed | 1小时0分50秒 | 1.5553 | 96.34% | 0.028470 |
| [t8-m32-a1](../t8-m32-a1/result.md) | packed | 1小时11分33秒 | 1.6866 | 96.34% | 0.029662 |
| [t8-m32-a2](../t8-m32-a2/result.md) | packed | 1小时2分47秒 | 1.6861 | 96.52% | 0.029662 |
| [t8-m32-b](../t8-m32-b/result.md) | packed | 1小时2分43秒 | 1.6863 | 96.38% | 0.029662 |
| [t8-c8-a1](../t8-c8-a1/result.md) | refnpy | 1小时4分38秒 | 1.5550 | 96.39% | 0.031813 |
| [t8-c8-a2](../t8-c8-a2/result.md) | refnpy | 1小时3分31秒 | 1.5555 | 96.43% | 0.031813 |
| [t8-c8-b](../t8-c8-b/result.md) | packed | 1小时3分15秒 | 1.5577 | 96.09% | 0.031813 |
| [t8-m8-a1](../t8-m8-a1/result.md) | refnpy | 1小时8分5秒 | 1.6862 | 96.51% | 0.031016 |
| [t8-m8-a2](../t8-m8-a2/result.md) | refnpy | 1小时7分37秒 | 1.6914 | 96.36% | 0.031016 |
| [t8-m8-b](../t8-m8-b/result.md) | packed | 1小时5分4秒 | 1.6863 | 96.59% | 0.031016 |

完整判定原文：[C32](records/gate-c32.log)、[M32](records/gate-m32.log)、[C8](records/gate-c8.log)、[M8](records/gate-m8.log)。四组分别包含BASELINE_REPEAT_EXACT、TRAIN_1000_EXACT、FINAL_STATE_EXACT、FINITE_ALIVE、IMPORT_ORIGIN、NORM_STATS_ACTUAL与GATE_8X8的PASS行。未用四舍五入后的loss作比较，实际比较[各run的hex投影与逐叶摘要](records/training-summary.json)。

## 覆盖与限制

库层与输入层另见[t8-fixture](../t8-fixture/result.md)：40ep和400ep全量verify、独立xgrid、四profile完整fixture、worker矩阵、29项拒绝与独立手算均已完成。训练只覆盖一个epoch中的前8000个随机索引，不能替代123044行的全量库核验。

C8/M8的999目录已保存1000更新后的EMA参数，属于本轮测试模型。[四profile全梯度](../t8-gradient/result.md)已完成且全部逐位通过，后续仍需完成C8/M8的七关与各48集闭环；闭环成功率按计划不作为模型能力指标。真实8×8在线池化已提前通过，见[M8推理分阶段记录](../t8-infer-m8/result.md)。

## 归档

各run的records保留Git无法重建的完整指标与清洗日志，BASELINE_MANIFEST按源文件SHA核对。中央records汇总四组gate、自重复检查、运行表与一次共享主机环境观测。不复制配置或启动脚本，不归档权重。配置与命令由各launch及上述提交还原。
