# 验证包装修复的探索性诊断

本项只用于定位JIT常量问题，标记EXPLORATORY_ONLY，不替代正式V5。主库仍冻结在CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`；忽略目录中的独立探针通过apply_patch复制func，并只将input_function/act权重改为显式参数。源探针见[留存载体（gzip）](records/func_constant_probe.py.gz)，压缩回读保持原字节，SHA与命令载体SHA记录在launch.actual.json。

GPU2，04:18:56Z→04:23:03Z，247秒，EXIT_CODE=0。三次A/A的38个梯度叶相同；image与pos各2048位置非零；删63/token反例被拒；32帧带扰动都高于零噪声，最小delta=0.0006692409515380859；n1/8/31的mask外垃圾不改变loss、动作或全部38叶梯度，无效位置梯度零，有效位负对照改变loss。

正式修复的func与本载体AST一致，随后另从clean修复提交重跑并通过，见[正式结果](../m2048-func-retry1/result.md)。各run功能判定使用各自三次A/A控制噪声；本项没有被当作训练数值回归的另一侧。数值阈值、样本数和叶集合未改变。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
