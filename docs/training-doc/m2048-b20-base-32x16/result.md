# m2048-b20-base-32x16 结果

本档改前证据完整，已与同配置CAND完成20+1步逐位对拍，全部201叶、五标量和输入一致；详见[候选结果](../m2048-b20-cand-32x16/result.md)。此结论仅针对旧32×16，不单独证明2048完整验收。

用户原话：「开始实现 有问题立刻越早问用户越好 一路实现到测速结束报告用户为止」；对较长摘要成本明确选择「保持原计划，完整取证（推荐）」。20次更新完成；20行五标量、20行状态（0及2..20）、201叶，全部有限，记忆梯度严格为正；232个索引、29批。对应1步补跑完成并与初态/首批/首标量逐位一致，完整更新0..20已补齐。

起跑代码为 `29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6`，生产src与初始c31b0509be3f76a6cc262fd655f5f0080b12c266相同；运行期间HEAD与源码保持不变。起点2026-09-21T02:10:45Z，终点2026-09-21T03:13:06Z，驱动记录的墙钟约3741.0秒。GPU4、5，batch8、worker4、FSDP2、seed42。环境B：8×A100-SXM4-80GB，/dev/md0 XFS本地NVMe RAID。训练摘要每次约183–185秒，是验证开销，不用于估计正式训练速度。

仅使用1600ep库及其source/framesamp/framesamp-8x8，norm_stats文件SHA为856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173；实际导入路径、配置、数组摘要、输入索引和退出记录见[运行记录](records/launch.actual.json)。真实样本均满历史，allshort/mixed1组名不构成短历史覆盖。原始数据、norm_stats、权重与依赖均未修改。

计划外事件：完整状态摘要耗时显著高于计划排期；向用户及时询问后按原方案继续，未优化算法、缩减步数或排除任何叶。本档外层补充项目内CUDA/W&B/XDG数据缓存位置及显式TRAIN_TIMING_STEPS=0，详情见launch.actual.json；其数值对照只与对应CAND同配置进行。

验证使用真实工具输出与逐项检查：check_modul_train_records.py的validate/with_step1在本次真实记录上核对步集合、201叶结构、有限性、norm_stats和索引完整性；BASE_WITH_STEP1=PASS。这些开发期检查将在已提交CAND检查器上再次用于正式双侧比较。

归档仅包含不可由Git还原的运行JSON、压缩JSONL/TSV及清洗日志；压缩前后SHA见[归档清单](records/archive_manifest.json)。大型参数未在本档另行保存；配置按启动提交及launch.md的覆盖项还原。下一步为候选同配置逐位对拍。
