# 原版80k正式CPU输入对拍档案

## 1. 结论与指标速览

**full和counting两组正式CPU输入对拍均PASS，四份collector与两份judge已结束。** 实际运行于2026-09-26 17:07:21—19:41:35 UTC，B及取证工具锚定`3a1582db39c723c735e04752e5027bfe40ecc3e1`，A固定上游`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。完整判定、UTC/PID、原始记录与归档SHA见[result](result.md)。

| 组 | 全量身份 | 每侧定点样本 | 每侧真实内容与顺序 | 结论 |
|---|---|---:|---|---|
| full | 1600集、768897帧、476857执行样本 | 6906 | 104个b64批；两epoch完整索引 | INPUT_EQ PASS |
| counting | 400集、189035帧及执行样本 | 2400 | 104个b64批；两epoch完整索引 | INPUT_EQ PASS |

此结论截至`collate_before_jax`，不等于GPU训练标量/参数/EMA/优化器状态逐位等价。P1、100步、单跑/并跑、新测速包装20步开关对照、300步perf及80k尚未实跑。

## 2. 版本与代码状态

实际INPUT_HEAD为`3a1582db39c723c735e04752e5027bfe40ecc3e1`；后续归档或工具提交不得替换该运行锚点。四侧分别从各自clean根启动，起止provenance逐字相同、porcelain为空，父进程和worker模块来源分别留证。A使用独立上游uv环境，B使用主仓库uv环境；Python均为3.11.15，起跑前208项包版本一致。输入工具SHA256为`0e52ad01d2de5f9250de465bc7c1fb6550377d564e8037ad476c9ec1150a7305`。

数据与既有20步实际Beta为`49a333eb18e8d6ff1143bf7871ef7c498ab91579`，数据组归档提交为`ec6c9e35784a96f636a587dd737dffa294b616ad`，均与本次INPUT运行版本分开。CPU采集期间主副本与两侧环境保持冻结，六会话全部结束后才解除本阶段冻结。

## 3. 启动与配置还原

实际标签`20260926T170654Z`，四份collector独立并行，每侧独立输出、日志、tmux及编译缓存。同组A/B成功后即可启动该组judge，因此count judge在full仍运行时先完成。具体命令及包装按[launch](launch.md)还原；六份实际命令文件SHA、会话全名、wrapper/body PID、UTC起止见[result第三节](result.md#3-启动与配置还原)。

CPU环境固定`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`JAX_ENABLE_X64=0`、`HF_HUB_OFFLINE=1`；清除`PYTHONPATH/PYTHONHOME/JAX_PLATFORM_NAME/XLA_FLAGS`，不覆盖HOME。`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`四侧一致；缓存落本仓库scratch的`v1-store/cache`，六个JAX缓存分别隔离。B使用`uv run --no-sync`，本阶段没有安装或同步依赖、模型初始化、训练或W&B调用。

## 4. 数据集与划分口径

full为公开16任务各100集；counting为其中BinFill、PickXtimes、StopCube、SwingXtimes各100集。双方同组共享source、manifest和norm_stats；A读取原版source，B读取同库4×4 packed，实际数据与划分未改变。full仍使用原版norm SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`，counting使用自算SHA `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。

前置档案为[full库](../../dataset-build-doc/16task-pub-1600ep/README.md)、[counting库及严格SUBSET](../../dataset-build-doc/4task-counting-pub-400ep/README.md)、[full20保存/恢复](../smoke-orig80k-full-0925/README.md)和[count20保存/恢复](../smoke-orig80k-count-0925/README.md)。这些前置PASS与本次INPUT PASS各有独立范围。

## 5. 输入参数与判定口径

固定b64/workers4/prefetch2/seed42、drop_last=True、action_horizon20、modul 4×4（budget512、16 token/帧）和motion关闭。每集按合法执行首尾、交界和30/31/32步定点；真实内容为epoch0前100批及末2批、epoch1首2批，共104批、6656个批内样本位置。full每epoch7450完整批、丢弃57项；counting2953批、丢弃43项。索引探针和真实内容两遍均完整保存并比较两epoch的索引、generator起止状态、base_seed和drop_last尾项；不能把104批内容称为两个epoch全量内容解码。

协议`schema=2`、`host_numeric_exact_motion_none_v1`：主机dtype可不同，支持数值叶的无损摘要必须精确相同，signed zero、有限性、shape和顺序严格。仅`motion_emb/motion_pos/motion_mask/mem_order`允许缺失与严格None等价；任何非None或其他未获准字段差异失败。四键登记于`equivalent_motion_none_keys`，原始ALL字段、dtype和raw SHA保留，实际输入不补键、不cast。训练五标量及完整状态bitwise要求不变。

## 6. 硬件、资源与耗时

AWS单机，`/scratch`为`/dev/md0` XFS，本次只用CPU。起跑前96个可见CPU，MemAvailable约1064.49GiB、scratch可用793.11GiB、SHM可用560.90GiB，cgroup OOM事件0；这些是快照。资源记录共6份离散快照，末份19:46:07 UTC时INPUT进程已全部结束、另有独立CPU单测，不能当作输入峰值。指定时点、进程树PSS、线程/FD和SHM见[result第六节](result.md#6-硬件资源与耗时)，不把稀疏快照称为全程峰值。

count-A/B包装耗时5123/4515秒，full-A/B为9223/8292秒；count/full judge分别6/16秒。本阶段不测训练吞吐，不用这些并行取证耗时推断loader速度比。预建约50GiB工作集和4.78GiB SHM只是估计，不能写成本次实测峰值。

## 7. 采集过程行为（本阶段无训练）

四侧均于17:07:21 UTC启动；count-B/A先后于18:22:36/18:32:44完成，count judge在18:33:44—18:33:50执行。full-B/A分别于19:25:33/19:41:04完成，full judge在19:41:19—19:41:35执行。每侧完成17项记录文件及record_manifest，同组judge在两侧完整成功后运行。

六份日志各有唯一工具PASS，以及各自唯一的`COMMAND_EXIT=0`、`TEE_EXIT=0`、`FOOTER_PRINTF_EXIT=0`、`FOOTER_TEE_EXIT=0`、`EXIT_CODE=0`。现场头尾还记录会话、进程、命令自SHA、INPUT_HEAD及UTC。六会话结束状态与退出记录共同核对，会话消失本身不算成功；原始collector目录没有混入tee或judge日志。

## 8. 输入判定与后续评估状态

两组judge均输出`INPUT_EQ=PASS ... comparison=host_numeric_exact_motion_none_v1`，full为1600/476857/6906/104/2，counting为400/189035/2400/104/2（依次为集、执行样本、定点、批、epoch）。原始记录保留，结论只覆盖本次输入与顺序。

本次没有执行P1、100步或单跑/并跑训练，没有执行新测速包装20步开关对照、300步perf和80k。既有两次可读性20步的checkpoint19保存/恢复PASS不能替代新包装对照。没有进行策略评估。

## 9. 用户决定记录

用户原话：「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」；实施授权：「开始实现该计划 有问题立刻问用户 不要自己决策」。暂不建库的阶段性范围后来由「恢复计划中的建库，严格按前置闸门推进」覆盖，其他前置要求保留。

比较口径沿用「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及「允许这四键缺失与 None 等价」。此前按「先保留严格判据并取证，结果出来后再决定」留下的三样本严格失败仍保留于[三样本档案](../orig80k-schema-0925/result.md)，不改写历史。

用户已批准「允许仅对拍关闭 W&B」（仅P1/100步A/B，本地完整证据保留）、「补充 0.5 秒只读采样」（perf checkpoint分配字节与scratch可用字节，预算保留保守余量）及「补记限制，继续 CPU 输入取证」。用户另选择「两库各跑一对 20 步（推荐）」验证新测速包装开关，真实保存和W&B仍开启；该新20步尚未执行。20步/perf/80k的W&B开启要求不变，批准不能代替实跑PASS。

## 10. 计划外事件与处置

新INPUT包装起跑前修复了“先写成功EXIT_CODE、后检查footer tee”的缺口，两方各6例纯shell失败传播验证通过，实际CPU使用修正版。历史四次数据/20步最外footer和整体包装退出0缺独立观测的边界仍见[exit-record-audit](exit-record-audit.md)，子任务与产物PASS未被反例证伪，原记录未改，也未用新包装测试补造旧观测。

原始INPUT日志没有tqdm中间态；归档时只在独立Git副本删除COMMAND行末一个未转义ASCII分隔空格，解析argv及判定/退出原文保持一致，原日志和collector记录不改。记录无损压缩包解码后逐成员校验SHA/bytes及原manifest字节，检查链随档案保存。具体日志与包摘要见[result第十二节](result.md#12-归档文件清单与摘要)。

## 11. 当前结论与下一步

正式CPU INPUT闸门两组均通过，后续继续既定P1→100步/单跑与并跑→新20步包装对照及perf等前置链。INPUT运行期间，预算对接和测速包装工具仅为ignored候选；解除冻结后已落实源码并完成工具合测，具体验证由总计划记录，不能将工具测试外推为真实训练通过。[P1启动档](p1-launch.md)和[100步/单跑/并跑启动档](train100-launch.md)均已预建、尚未执行。80k仍须所有对应前置通过。

后续沿用证据须同时记录原INPUT_HEAD和实际TRAIN_HEAD，并核对固定提交diff、取证工具、父/worker输入模块、依赖、参数、数据、资产与history指纹一致；不能重写原记录HEAD或仅重复judge便声称新HEAD已重采。取证工具变化需双方同工具重采，输入相关变化或影响不明先报告。详见[launch沿用边界](launch.md#后续提交变化与证据沿用边界)。

## 12. 归档文件清单

[launch.md](launch.md)保留起跑前机制和本次实际展开值；[result.md](result.md)保存完整12节实测与SHA；[exit-record-audit.md](exit-record-audit.md)保留历史边界。四侧无损记录位于`records/{full-a,full-b,count-a,count-b}.records.tar.gz`，各自`*.archive_checks.json`绑定原始目录/manifest及解码校验；六份Git日志及检查链位于`logs/`。资源和控制器快照分别为`records/resources.jsonl`、`records/runtime-controller.json`。

四侧无损包及六份日志的stage检查均已通过，最终归档副本由主代理统一复制核验；资源文件SHA已记录，controller固定副本SHA在[result](result.md)明确待复制时填写，不编造。源码、脚本和yaml由固定提交还原，不复制到档案；权重、缓存和凭据不归档。原始运行路径及全部collector内部文件字节保持不变。
