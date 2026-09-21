# 2048实施与短验证记录

生产扩容和验证工具已实现，短验证通过；完整旧档回归、新档真实训练及独立测速仍待执行，当前不能宣称2048完整验收结束。

## 用户决定与范围

用户原话：「开始实现 有问题立刻越早问用户越好 一路实现到测速结束报告用户为止」。首次完整TrainState摘要实测184.428秒后，用户明确选择「保持原计划，完整取证（推荐）」。因此保留两种旧档各侧20步、新档参考两侧20步、每侧1步补跑、100步保存加载，以及8卡容量和1000步独立测速；正式80k仍须报告后的明确同意。

本轮遵循[0920计划](../../../0920-32frame-8x8-modul-2048-plan.md)。生产提交commitV11.0只含三个Python文件、新YAML和直接守卫用例；其余工具单独提交，便于数值回归失败时只撤销生产改动。AGENTS.md、全局训练配置、采样/归一化/collate数值实现、MemoryAttention、优化器及checkpoint实现均未改。

## 版本与环境

起点c31b0509be3f76a6cc262fd655f5f0080b12c266；工具准备后的BASE为29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6。BASE的生产src和依赖与起点相同。全部BASE任务已完成，运行期间HEAD和工作区保持不变。候选生产提交为commitV11.0（f54b1a6）；验证工具提交完成且porcelain为空后固定CAND，各后续run在启动日志记录完整实际HEAD。

环境B：8×NVIDIA A100-SXM4-80GB，/scratch为/dev/md0 XFS本地NVMe RAID，初始余量约1.7TB；旧/data和NFS不存在。验证用GPU4、5，CPU检查明确禁用CUDA。依赖由既有uv.lock管理，所有Python命令使用uv run --no-sync，本轮没有同步或更改依赖。

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

先完成同配置BASE/CAND的完整输入及20+1步逐位回归，再执行新档独立输入、逐位置功能、参考训练、原数组到checkpoint动作对拍。随后8卡20步只验容量，1000步独立测速取0..99预热、100..899稳态、900..999尾段与真实保存；不启用profiler，不逐步同步，GPU500ms采样，主机及并发IO独立记录。80k用时只能标估计，报告与可检查正式runner完成后保持USER_APPROVAL=PENDING。
