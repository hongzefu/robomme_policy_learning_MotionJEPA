# 原版80k正式CPU输入对拍档案（预建，尚未运行）

## 1. 结论与指标速览

**本档案仅完成 CPU 输入阶段的预建；尚无本次采集或判定结果，当前因历史最外包装终态处置待用户答复而暂缓启动。** 两个公开库、严格子集内容检查及两次20步保存/恢复的子任务和产物验收PASS未被该日志反例证伪；但最外footer及整体进程真实退出0不能仅凭日志独立证明。本阶段将另做两组上游/当前完整输入对拍，不以此前三样本或工具单测代替。P1、100步训练轨迹、单跑/并跑、perf及80k均不在本次CPU执行范围内。

| 对象 | 本阶段要求 | 当前实际状态 |
|---|---|---|
| full A/B | 各自核对1600集、476857个执行样本身份 | 未采集 |
| counting A/B | 各自核对400集、189035个执行样本身份 | 未采集 |
| 每侧定点样本 | 执行首尾/交界及合法历史边界，数量由实际计划计算 | 未采集 |
| 每侧真实批次 | b64/workers4，共104批、6656个批内样本位置 | 未采集 |
| 每侧采样顺序 | 两个完整epoch索引、drop_last尾部及generator状态 | 未采集 |
| 两组输入判定 | 分别由同组A/B记录进行精确数值及获准None等价判定 | 待结果 |

完整启动设置、四份collector及两份judge命令见[launch.md](launch.md)。本表只列要求与当前状态，不是通过判定。

## 2. 版本与代码状态

数据组归档提交为 **`ec6c9e35784a96f636a587dd737dffa294b616ad`**；两库构建及两次20步检查的实际运行Beta为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`。它们是已经完成的前置证据版本，不能填写为尚未执行的CPU输入采集版本。

A固定上游 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，使用 `v1-store/worktrees/orig-ecf086c/.venv/bin/python`。B的 **`INPUT_HEAD` 待包含本README及launch的clean提交创建后填写**，实际命令须传40位Git SHA字面量；本档案不预填未来提交，不通过动态读取当前HEAD代替期望锚点。

正式启动时再记录双方实际HEAD、clean状态、解释器、取证工具和项目模块摘要、完整argv/env及时间。采集期间冻结主副本和两侧环境，不改tracked文件、不提交结果、不同步依赖。结果结束后另行归档提交，不改写真实启动版本。

## 3. 启动与配置还原

入口为 `scripts/training/tests/check_orig80k_inputs.py` 的 `collect` 和 `judge`，当前只保留待执行模板，须先等待历史最外终态处置答复并落实。既定执行方式仍为四侧collector独立并行，每侧拥有新输出、日志、tmux会话与JAX缓存；full或counting任一组的A/B均成功后，该组即可运行judge，**无需等无关组结束**。具体六份命令及退出状态包装均在[launch](launch.md#公共命令设置与有退出记录的启动包装)保留，不额外复制脚本或yaml。

CPU环境固定 `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`JAX_ENABLE_X64=0`、`HF_HUB_OFFLINE=1`，清除 `PYTHONPATH/PYTHONHOME/JAX_PLATFORM_NAME/XLA_FLAGS`，不覆盖用户HOME。`OPENPI_DATA_HOME`指向主副本只读共享模型资产；依赖缓存、日志和六份独立编译缓存均在本仓库 `v1-store/`。A与B各用自己的uv环境，B使用 `uv run --no-sync`，不安装或同步依赖。

`INPUT_TAG`、最终输出路径、实际六份会话名和日志名均待启动时展开、核实唯一并记录；目前未创建任何采集输出或会话。每份实际命令先保存在 `v1-store/` 内独立命令文件，现场计算SHA256，再由tmux执行该同一文件并核对SHA，避免长内联命令限制。日志头留会话全名、wrapper/body PID、声明的INPUT_HEAD、命令文件路径/期望与实际SHA及START_UTC。尾部先记录END_UTC、命令及正文tee状态，捕获footer printf/tee状态后，才以受检查的直接append写两项footer状态和唯一综合退出码；不得在检查footer tee之前写成功终态。当前不预写任何实际摘要或时间。tee日志和命令文件都在collector输出目录之外，不能混入其 `record_manifest.json` 所约束的文件集合。

## 4. 数据集与划分口径

full为公开16任务各100集，共1600集、768897帧、476857个执行样本；counting为同一集合中BinFill、PickXtimes、StopCube、SwingXtimes各100集，共400集、189035帧及执行样本。本阶段不修改划分、不重建或裁剪数据。

本阶段采用的前置档案均已在数据组提交中归档：[full正式库](../../dataset-build-doc/16task-pub-1600ep/README.md)、[counting正式库及严格子集检查](../../dataset-build-doc/4task-counting-pub-400ep/README.md)、[full20保存/恢复](../smoke-orig80k-full-0925/README.md)、[count20保存/恢复](../smoke-orig80k-count-0925/README.md)。其通过范围与本次输入闸门分开，不互相替代。

同组A/B消费同一source、manifest与norm_stats；A从source读取原版格式，B从同库4×4 packed读取历史特征。full使用原版norm SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`；counting使用自算norm SHA `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。资产父目录、两库manifest完整SHA与只读路径详列于[launch的输入表](launch.md#已核实的输入与资产)。

## 5. 输入参数与判定口径

固定b64、workers4、原生prefetch2、seed42、`drop_last=True`、action_horizon20、modul 4×4（budget512、16 token/帧）及motion关闭。四侧并行只改变调度，不改任何这些值、数据范围或判据。

每集定点范围为合法执行步集合 `{exec_start_idx, exec_start_idx+1, num_timesteps-1, 30, 31, 32}` 的排序去重结果。每侧真实内容取首epoch的前100批与末2批，以及次epoch首2批，共104批；full每epoch7450完整批并丢弃57个尾样本，counting为2953批和43个尾样本。两遍真实sampler还要完整消费并核对两epoch的全部索引、base_seed和generator起止状态，不能将104批内容误称为全量两epoch内容解码。取证终点为 `collate_before_jax`。

协议为 `schema=2`、`host_numeric_exact_motion_none_v1`。主机dtype可不同，但无损数值摘要须精确一致，signed zero、有限性、shape、顺序与未获准字段保持严格。仅 `motion_emb/motion_pos/motion_mask/mem_order` 的缺失与严格None等价，完整记录 `equivalent_motion_none_keys`；任何非None值或其他未获准字段差异失败。原始 `raw_all/transformed_all/inputs_all`、dtype和raw SHA均保留，实际输入不补键、不改值。训练五标量及参数/EMA/优化器状态的bitwise要求不在本阶段放宽。

## 6. 硬件、资源与耗时

环境为AWS单机，存储 `/dev/md0` XFS，本阶段显式使用CPU。预建前只读快照为96个可见CPU、MemAvailable约1068.17GiB、可用SHM约560.90GiB；相关cgroup没有CPU quota或内存硬限。完整快照及限制说明见[launch资源段](launch.md#cpu内存与输出预检)，起跑时仍须重新核实。

按源码shape、预取与复制路径估算，四侧应用数据工作集约50GiB量级，两份B的SHM约4.78GiB；这些不是实测峰值、硬上限或容量保证。上游两份A的历史读取线程可合计达到256条，另有JAX CPU线程，不能凭CPU数量宣称没有瓶颈。运行中按明确PID监测进程树RSS/PSS、线程/FD、MemAvailable、SHM和CPU/IO压力；不以重复共享页计数冒充独占内存，不改参数来消除资源失败。

本阶段尚无实际起止时间、耗时或资源峰值。它不测吞吐，也不承诺四侧并行的性能收益；后续仅记录本次诊断耗时及资源行为，不替代perf。

## 7. 采集过程行为（本阶段无训练）

尚未启动collector。后续在独立detached tmux中运行四份采集，以流式日志观察各侧全量身份扫描、定点样本与批次进度。每侧须记录完整会话名、实际进程、起止时间、工具退出码和tee退出码；会话消失本身不算成功。

四份collector彼此无内容依赖，可连续发出启动命令并行运行；同组judge只依赖本组两份成功且记录清单完整，不等待无关组。放行必须同时具备唯一工具完成判定，以及日志中各自唯一的 `COMMAND_EXIT=0`、正文 `TEE_EXIT=0`、`FOOTER_PRINTF_EXIT=0`、`FOOTER_TEE_EXIT=0` 和 `EXIT_CODE=0`；缺失、重复、任何非零或终态追加失败均不算成功，两份judge自身也适用同一要求。任一失败保留原始输出和日志，停止相关依赖步骤并报告；不覆盖重试、不改名破坏provenance、不往collector目录追加外部日志。

## 8. 输入判定与后续评估状态

full和counting两组judge均待执行。collector成功只说明该侧取证完整，仍需同组A/B的来源、完整范围、精确数值与获准None等价全部满足才形成该组输入结论。实际判定行、退出记录、失配定位和各组范围将在结果文件中原样留证。

P1、上游/当前100步、同入口单跑/并跑、300步perf及正式80k均未由本阶段验证。用户对P1/100步是否允许关闭W&B，以及perf磁盘采样两个问题尚未答复；这两项仍不影响只做Dataset/transforms/loader取证的CPU内容，也不能据CPU阶段独立而跳过后续待决事项。另有第三项历史最外包装终态处置待答，因此CPU采集现在暂缓，不能默认用户认可历史最外终态证据。本次不启动任何训练或策略评估。

## 9. 用户决定记录

用户要求「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」，并授权「开始实现该计划 有问题立刻问用户 不要自己决策」。数据阶段按后续决定「恢复计划中的建库，严格按前置闸门推进」完成，CPU输入闸门已准备，但当前尚未启动，等待新发现的历史终态证据边界处置。

比较口径沿用用户原话：「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及「允许这四键缺失与 None 等价」。四键白名单、原始字段保留和训练bitwise约束按第5节执行，不扩大为一般schema或容差豁免。四侧并行是已明确的本阶段调度安排，不改变b64/workers4/prefetch2/种子或取证覆盖范围。W&B与perf磁盘采样两项问题继续待答；第三项已向用户提出“补记限制继续CPU输入”或“先制定额外复验方案”，当前同样未答。不能把任一待决选项写成用户已批准，也不自行选择方案。

## 10. 计划外事件与处置

本阶段尚未采集，但预建审查发现新启动包装会先写成功EXIT_CODE再检查footer tee。纯shell反例证明，footer tee可完整输出成功文本后返回非零，从而留下不能代表wrapper实际成功的日志。当前INPUT包装已修复为先捕获footer printf/tee状态、合并非零返回，再以受检查的直接append写两项footer状态和唯一综合退出码；放行要求命令、正文tee、footer printf/tee及综合状态各唯一且全0。

实现方和独立审查方各完成6例纯shell替身验证：正常、主命令exit7、正文tee失败、footer完整输出后失败、自SHA不匹配、日志冲突均正确；两方footer失败案例分别得到非零终态1与29，没有残留成功EXIT_CODE。未运行真实tmux、collector或训练，临时载体已清理。这是新包装验证，不是历史运行退出码的新增独立观测。

历史最外包装终态的证据边界已补充在[exit-record-audit.md](exit-record-audit.md)。数据与两次20步子任务和产物验收PASS未被该反例证伪，也不能因新包装已修复就推定历史整体进程真实退出0已获独立证明。原始记录保持不变，不将旧日志退出0重新标成独立观测；该边界的处置已询问用户，答复前暂停CPU起跑，不自行决定补记即继续或追加复验。

后续如资源不足、来源污染、原始字段或精确数值不一致、缺记录或非零退出，应保留实际失败证据及当时版本，明确对应侧与阶段，不以修改阈值、删除记录或缩小范围制造通过。

历史三样本曾按当时严格schema规则留有失败，后来仅按用户批准的四None键规则更新比较结论，详见[三样本档案](../orig80k-schema-0925/result.md)；本阶段不重写该历史，也不拿其小范围结果顶替正式两库输入验收。

## 11. 当前结论与下一步

当前结论为数据/产物证据已归档、CPU输入内容与W&B/perf两项待决独立，但历史最外终态处置另待用户答复，故本阶段仅预建、暂不启动。先等待并落实第三项答复；满足相应前置后，再按包含本档案的clean提交填实INPUT_HEAD和唯一标签，复核环境/资源/路径并执行既定四侧并行流程。每组A/B都成功才运行本组judge，结果事后回写。现在不填任何实际会话、运行时间或输入通过结论，不预设用户将选择哪种处置。

后续仅改文档或无关测速工具时，旧输入记录仍须锚定原INPUT_HEAD；沿用前应记录新旧提交diff和取证工具、输入链模块、依赖、数据及资产指纹一致性，不能改写记录HEAD或声称新提交已重跑。取证工具变化要求两侧同工具重采；输入相关变化或影响不明先报告再确定重测范围。训练S1/S3与S2仍须同HEAD。完整边界见[launch](launch.md#后续提交变化与证据沿用边界)。

## 12. 归档文件清单与待更新项

本README和[launch.md](launch.md)均为起跑前文件；[历史最外终态说明](exit-record-audit.md)记录独立审查及新包装验证，不是本阶段输入实测结果。实际运行结束后才新增CPU结果正文及 `records/`，保存四侧原始采集记录与record_manifest、两组judge日志、各侧原始/清洗日志、环境/资源记录和真实起止/退出证据。collector子目录保持文件字节与清单完整，外部解释和日志放独立层级。

待补内容首先是历史边界处置的用户答复原话及落实范围；实际获准启动后再补INPUT_HEAD、INPUT_TAG、六份展开命令及运行时命令文件路径/真实SHA、会话/日志路径、wrapper/body PID与起止时间、四份collector状态、每侧定点样本数和批次/索引覆盖、两组judge判定、耗时、资源行为及计划外事件。仅归档Git不能还原的实测记录；源码和配置由固定提交还原，不复制脚本/yaml、运行时命令文件、模型权重或敏感凭据，命令正文按launch和运行日志还原。本次无CPU结果文件或通过记录需要提前创建。
