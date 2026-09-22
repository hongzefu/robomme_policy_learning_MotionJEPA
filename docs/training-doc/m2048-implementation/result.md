# 2048实施与短验证记录

**完成更新：** 正式80k已从Beta `55647ff33c8ddb9ec324fdbcee8bd1491456725b` 正常完成，耗时22小时58分52秒、EXIT_CODE=0；16份checkpoint和最终61叶CPU加载通过。用户要求「完成合并」「和训练的收尾」后，已完成档案、主仓快进、源码解锁及本轮独立副本清理。[完整训练与合并报告](../v2-1600ep-m32x8x8-modul-b128-80k/result.md)记录全部实测；旧temp因仍有其他会话占用而保留，策略评估另立任务。以下保留此前实施过程与当时状态。

生产扩容、全部正确性、容量及1000步测速已完成：输入、初态、旧两档回归、新2048参考/packed完整20+1、十记忆叶更新、功能门及100步保存加载全通过。[正确性汇总](records/correctness.complete.json)与[容量结果](../m2048-smoke/result.md)均留档。[测速](../perf-m32x8x8-modul-b128-20260921T051856Z/result.md)为1.030681秒/步、124.1897 samples/s，GPU均值98.6104%、0%采样占比0.1219%，SPEED_REPORT=READY。正式80k条件授权已满足，正在固定Beta并执行最终起跑门；约23小时为外推。

## 用户决定与范围

用户原话：「开始实现 有问题立刻越早问用户越好 一路实现到测速结束报告用户为止」。首次完整TrainState摘要实测184.428秒后，用户明确选择「保持原计划，完整取证（推荐）」。因此保留两种旧档各侧20步、新档参考两侧20步、每侧1步补跑、100步保存加载，以及8卡容量和1000步独立测速。

实施中用户又要求「能并行的尽可能并行 8个gpu你可以用」，并授权「1000步测速后如果占用率高于50% 并且没有其他的问题 可以直接启动训练」。本轮按800步稳态窗口八卡总体平均GPU利用率严格大于50%、全部正确性与容量门通过、SPEED_REPORT=READY且无未解决问题判断；满足后绑定实际报告/runner/HEAD与已定run_name，重跑V10并直接起跑，不重复询问。条件未满足则停在起跑前报告。

本轮遵循[0920计划](../../../0920-32frame-8x8-modul-2048-plan.md)。生产提交commitV11.0只含三个Python文件、新YAML和直接守卫用例；其余工具单独提交，便于数值回归失败时只撤销生产改动。AGENTS.md、全局训练配置、采样/归一化/collate数值实现、MemoryAttention、优化器及checkpoint实现均未改。

## 版本与环境

起点c31b0509be3f76a6cc262fd655f5f0080b12c266；工具准备后的BASE为29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6。BASE的生产src和依赖与起点相同。全部BASE任务已完成，运行期间HEAD和工作区保持不变。候选生产提交为commitV11.0（f54b1a6）；验证工具提交完成且porcelain为空后固定CAND，各后续run在启动日志记录完整实际HEAD。

环境B：8×NVIDIA A100-SXM4-80GB，/scratch为/dev/md0 XFS本地NVMe RAID，初始余量约1.7TB；旧/data和NFS不存在。按并行决定，旧两档BASE/CAND始终GPU4、5；新参考/packed始终GPU0、1且两侧顺序；功能GPU2；100步拟GPU6、7。CPU检查禁用CUDA，性能阶段独占八卡。依赖锁和主环境未变，运行使用uv run --no-sync；本轮独立工具修复副本另以uv sync --offline --frozen创建自己的环境，未同步或复用主环境。

CAND固定为0c877c7495dfe5db8b83f033442013c6d6fd8552。工具修复31d4bd0b03cc1031eee04b95a69cd6a8aa6a1222在独立副本提交；其生产src树4a16a51e8fd57b25c60d47836a2ac31250b71b89与CAND相同，208个安装分发版本相同。主副本继续从原CAND完成在途取证，随后才快进工具和档案。

## 改动与数据流

FrameSampDataset只额外放行整数(2048,64,1)、modulation、motion关闭；旧两个512档继续沿原int转换条件。PerceptualMemory显式检查image/pos/state的batch与长度；HistoryPi0在关闭态要求mask非None、bool且精确为(B,budget)，拒绝(B,1)广播形状。四处inputs_spec预算转int，所有新增拒绝路径在-O下保留。

链路及字节账见计划第二部分「改动前后链路与字节账」。数据仍来自同一verified/full packed表与源pkl；旧8×64交付静态四键3703296 B/样本，新32×64为14813184 B/样本。image保持bf16，pos f32，归一化state f64，mask bool；JAX端既有state f64→f32不变。新档选帧数与RoPE query位置有意变化，不能要求512与2048前向数值等价。

## 已实跑的短验证

均在本机CPU执行，外层CUDA_VISIBLE_DEVICES为空、JAX_PLATFORMS=cpu；UV/XDG/HF缓存与pytest临时目录显式置于v1-store，候选短测另显式设置pytest cache_dir。隔离的启动夹具只模拟设备查询及Git状态，不启动训练。

| 验证 | 实测结果 |
|---|---|
| BASE工具：uv run --no-sync pytest scripts/training/tests/test_padding_dtype.py -q -s | 18 passed，7.62秒；两帧预算FIXTURE_COMPAT/FIXTURE_NEW通过，1600ep每偏移候选四任务各400 |
| BASE工具：完整test_pack_guards.py -x -q | 29 passed，28.06秒，无skip |
| 生产候选：完整test_pack_guards.py -x -q | 41 passed，32.63秒；包含2048真实行、spawn、类型/组合拒绝及旧预算转换 |
| check_32frame_modul.py guards | 普通及-O均GUARD_2048=PASS、STATIC_SHAPE=PASS；正确2048通过，3类型/4组合反例及mask(B,1)等拒绝 |
| check_32frame_modul.py yaml | YAML_DIFF=PASS，仅budget从512变2048，类型int，motion关闭 |
| pytest test_modul_speed.py test_modul_launch_guards.py -x -q | 46 passed，124.47秒；真实CLI、全部preflight门的隔离夹具、错误配置/同意绑定、零训练入口、三处同步、无profiler及证据缺失拒绝 |
| 真实BASE记录的结构检查 | 两档主run及补跑均完整，合并state_step0..20；201叶，主232索引/29批，补跑80索引/10批 |
| 真实BASE记录派生反例 | 7/7拒绝，包括所有行一致漏同一叶并重算摘要，防止两侧同时缺叶假通过 |
| 静态与格式 | git diff --check、bash -n，以及ruff --no-cache的F821/F822/F823均通过 |

pytest显式设置MMEVLA_TEST_SOURCE/MANIFEST指向1600ep，并使用全新的--basetemp。新静态守卫通过实际PerceptualMemory和HistoryPi0接口调用验证；启动负例把最终训练入口换为计数桩，失败时调用0次，正例1次。

## 开发阶段发现与处置

首次CPU完整dump漏传exp-name，CLI在读数据前以2退出；保留失败日志，补齐后在同一clean BASE重跑成功。两档CPU完整取证均完成，3200/3000个样本及200批，完整身份覆盖1600集、1192918帧、605611个执行样本。原始与gzip回读SHA一致。

独立numpy oracle草稿曾因numpy直接f32求幂的1 ULP误差失败；位置2048会放大频率舍入误差。改为独立f64求幂后舍入f32，后续阶段仍f32，未改被测代码或容差。CPU开发自测六组数值、单key不变、垃圾不变、非退化RoPE敏感性及统一query位置均通过；2048/64的max误差1.4603e-6，小于预设9.6622e-6。在线开发桩两种池化分支也通过，位置表805306368 B。两项仍会在clean CAND正式执行，不能以开发草稿代替最终证据。

按BASE真实参数形状推算，两份完整TrainState原数组约93.305 GiB，而非旧注释的14GB量级；当前空间充足，V8将记录实际bin字节数。摘要耗时约3分钟/份，已向用户说明并获得继续完整取证的决定，不从该验证时间外推正式训练吞吐。

## 后续判据

已完成：旧两档完整输入回归、新档NPY/packed的3000样本及200批逐位对拍、400样本有界装配、32种pad、在线5时刻两池化分支、混合共享内存与worker0/4交付、六组独立oracle、512/2048的61叶同初态、旧8×64完整20+1训练回归及正式V5。详细数据见总索引各run，CPU与功能结果不能替代仍在进行的训练对拍。

功能检查首次因JIT捕获权重而编译出约10GB常量失败。探索性探针247秒全通过后，仅修复检查器显式权重参数；clean修复提交正式重测276秒全通过，原失败保留。独立副本首次推送用了默认yinpei-tri身份而被拒，已停止并询问；用户明确「你要推送到hongzefu的位置 不可以用yinpei tri」。临时沿用主副本既有SSH命令，ssh -T确认Hi hongzefu后成功推送到hongzefu/robomme_policy_learning_MotionJEPA，未修改密钥或upstream。

随后复核补齐同一JIT路径的帧带与有效位基线，clean e4a50672fc68770429654af4bc96c31e531a5195的retry2在154秒内全通过。实测两条旧loss路径有7.05481e-4差值；排除该差后32帧带最小变化仍为1.04189e-4>同路径噪声0，全部2048位置及mask门仍通过。最终功能证据以[retry2](../m2048-func-retry2/result.md)为准，既有记录不改写。新参考链20+1也已完成自身完整性、十叶更新与7个记录反例，尚待packed双侧比较。

旧32×16的20+1也已与BASE完整逐位通过，至此旧档回归收齐。100步在GPU6、7完成并全部通过：原数组各50092740044字节、共201叶与现场SHA一致；TIC L0通过，61叶bf16加载零差异，十记忆叶均区别于初态，动作RMS=0、实际mem_len2048。大bin与checkpoint保留v1-store，Git只归档元数据和实测。该结果不支持恢复优化器状态续训。

新参考/packed的20+1双侧完整比较于06:35:37Z通过，全部五标量、输入和状态0..20的201叶逐位同；packed十叶更新与7个记录反例也通过。所有本轮GPU取证任务自然退出、八卡显存全0；主副本可解除CAND冻结并快进到已提交修复和档案，再从clean HEAD做容量/测速。

8卡20步只验容量，1000步独立测速取0..99预热、100..899稳态、900..999尾段与真实保存；不启用profiler，不逐步同步，GPU500ms采样，主机及并发IO独立记录。80k用时只标估计；满足上方已获条件授权后执行V10和正式起跑，否则报告未满足项。
