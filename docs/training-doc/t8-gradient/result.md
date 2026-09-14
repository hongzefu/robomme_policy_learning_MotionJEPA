# 四种配置的全梯度验收结果

四个profile全部通过：C32/C8每种batch有32个可训练梯度叶子，M32/M8有36个，三种batch的参考与候选逐叶SHA、loss hex、样本索引和叶集合完全一致。全部梯度统计有限，四份日志均以EXIT_CODE=0结束。连同[十二条1000步轨迹](../t8-training/result.md)，计划内训练侧验收已完成；checkpoint推理与闭环另行执行。

## 版本、输入和用户决定

用户原话：“一口气全做完！”、“注意你只能用4个gpu”。norm_stats采用用户确认的“文件SHA＋解析后数组摘要双重检查”。“排错关闭完整状态摘要（推荐）”仅适用于四个-s100，本项没有关闭或删减任何完整初态或梯度摘要。

A在REF `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，显式PYTHONPATH指向REF/src；B在主树 `ec6a43dc30e3cc6d8f3baaf2799a02441cd18e8b`，清除该覆盖。候选生产代码仍为CAND `c08ec2060a544af1869c1e24f755e536150569ca`，其后只有docs/变更。第一波外层从clean `9006d4a49cd99dd4d99ab376d8053341ff99e3ba`启动，期间增加推理启动文档，B起跑时已是clean ec6a43d；第二波直接从clean ec6a43d启动。各侧实际HEAD均在GRAD_START中记录，不把第一波外层HEAD当作其B侧版本。

输入为同一400ep源数据：32帧组两侧都是旧4×4 packed；8帧组A直接解码源npy，B读新8×8 packed。全部显式400ep norm_stats，文件SHA `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`。启动命令和完整env见[launch.md](launch.md)，参数仍为启动覆盖：seed42、batch8、worker4、fsdp2及确定性XLA两开关。num-train-steps1000只为还原配置，本脚本在共同初态上计算三种batch梯度，不额外执行训练更新。

## 逐项证据

每侧先与自己对应的1000步轨迹核环境指纹，再现场初始化TrainState，与该轨迹state_step0逐叶比较；C组177叶、M组193叶均通过。因此梯度不是从另一个随机初态算出。mixed1、allshort、allfull各8样本，两侧batch_meta中的16个交付键、dtype、shape、None与数组摘要全部相同，归档时再次逐项确认。

梯度使用与真实train_step相同的trainable_filter，对全部可训练叶取原始SHA，不排除任何不确定叶，不放宽容差。三种batch间不更新参数；每种分别比较索引、loss hex、完整叶集合与全部叶SHA，另外核loss、每叶max_abs/l2为有限值。

| profile | 物理GPU | A→B墙钟 | 每kind梯度叶数 | mixed1 / allshort / allfull loss（两侧同值） | 证据 |
|---|---|---:|---:|---|---|
| C32 | 4,5 | 32分7秒 | 32 | 0.633969307 / 0.263696939 / 0.559610307 | [完整日志](records/c32/gradient.summary.log) / [A摘要](records/c32/a/grad_summary.json) / [B摘要](records/c32/b/grad_summary.json) |
| M32 | 6,7 | 32分22秒 | 36 | 0.619058847 / 0.263696939 / 0.567672908 | [完整日志](records/m32/gradient.summary.log) / [A摘要](records/m32/a/grad_summary.json) / [B摘要](records/m32/b/grad_summary.json) |
| C8 | 4,5 | 14分31秒 | 32 | 0.611530423 / 0.704216182 / 0.580537736 | [完整日志](records/c8/gradient.summary.log) / [A摘要](records/c8/a/grad_summary.json) / [B摘要](records/c8/b/grad_summary.json) |
| M8 | 6,7 | 16分31秒 | 36 | 0.620160639 / 0.704216182 / 0.597755611 | [完整日志](records/m8/gradient.summary.log) / [A摘要](records/m8/a/grad_summary.json) / [B摘要](records/m8/b/grad_summary.json) |

四个profile分别产生以下判定行；完整原文在各自日志中，不能将一组PASS替代其他组：

```text
C32: GRAD_EQ=PASS kinds=3 leaves=32 mismatches=0
M32: GRAD_EQ=PASS kinds=3 leaves=36 mismatches=0
C8:  GRAD_EQ=PASS kinds=3 leaves=32 mismatches=0
M8:  GRAD_EQ=PASS kinds=3 leaves=36 mismatches=0
GRAD_FINITE=PASS profile=<上述各profile> kinds=3
GRAD_INPUT_EXACT=PASS profile=<上述各profile> kinds=3
```

loss表只方便阅读，实际判定使用表内各侧grad_summary.json的loss_hex和per_leaf；[汇总](records/summary.json)保留版本、索引和loss hex，没有用表中四舍五入的小数。32帧与8帧的fixture索引与内容有意不同，不做跨档loss相等断言。

## 资源、耗时和计划外事件

环境为AWS A100-SXM4-80GB，本地NVMe RAID /dev/md0。第一波C32/M32并行，第二波C8/M8并行；始终只占4、5、6、7，每profile的A→B沿用同一对卡。最早开始 `2026-09-14T11:29:04Z`，最后结束 `2026-09-14T12:18:22Z`，总墙钟49分18秒。会话名依次为t8-grad-c32、t8-grad-m32、t8-grad-c8、t8-grad-m8，均正常退出，未清理其他会话。

首次完整初态摘要与梯度统计较慢。只读栈检查确认在原始字节SHA和float64统计中推进，没有修改哈希口径或跳过叶子。每kind记录的seconds包含编译、GPU回传、SHA和CPU统计，不是纯反向计算时间；各档墙钟受缓存和这些一次性成本影响，不能解释为8帧的吞吐收益。生产吞吐口径仍见十二轨迹中的稳态分层记录。

## 结论与归档

旧32帧路径和新8帧路径均完成输入、1000步真实训练、完整初态和全梯度闭合。模型能力不由这项一致性检查判断，推理和48集闭环另见[C8](../t8-infer-c8/result.md)及[M8](../t8-infer-m8/result.md)。

records/按四profile和A/B保存env.json、BASELINE_MANIFEST.json、grad_summary.json、三份batch_meta、清洗日志及早期输入核对报告；[source-sha256.json](records/source-sha256.json)核对与原始文件相同。完整batch数组留在v1-store/fixtures/8x8/grad，不归档权重或大数组。[summary.json](records/summary.json)汇集实际版本、时间、索引和loss hex。
