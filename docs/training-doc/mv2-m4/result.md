# 完整 V5：六项模型接线检查全部通过

用户原话「采用兼容权重加载，保留 gemma_150m 替身」。从 clean 81bd0217f890218a1134b135adcf3490cc68dd51 启动，2026-09-18 02:56:23–03:00:20 UTC，237 秒、EXIT_CODE=0。GPU4，完整参数与命令见 [launch.md](launch.md)。

四类样本的有效motion数为0、39、141、160。三次确定性探针一致，全部42个可训练叶均被覆盖，nondeterministic_leaves为空、excluded=0。结果：

- MASK_INVARIANCE、PAD_CONTENT_INVARIANCE通过：padding垃圾不改变loss、动作或任何可训练梯度。
- GRAD_LEAK通过：padding输入梯度为零；motion全遮时四个motion叶梯度为零，有效输入时非零。
- MOTION_CONTENT_EFFECT通过：改变有效motion内容会改变loss。
- ORDER_EFFECT通过：并列与交错的loss绝对差为0.00404310226。
- ROW_PERM_INVARIANCE通过：对应内容与时间一起置换后loss逐位相同，max_abs_diff=0。

base与padding的loss hex同为0x1.48f9960000000p+2。modulation不同记忆长度的全遮等价按计划记LEN_EQUIV_NA；本轮验证的是固定budget下的接线与不变性。

两侧各加载40个pi05_base兼容叶、19个动作专家参数叶齐全。开启态保留随机25叶，关闭态21叶，完整名单、门参数及资产指纹见 [report.json](records/report.json)。本任务没有执行优化器，四个motion叶的实际训练更新由后续V8验证；模型质量和策略成功率仍需后续评估。
