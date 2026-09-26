# 原版80k正式CPU输入对拍结果

后续阶段结果另见[P1](p1-result.md)及[100步](train100-result.md)：P1内容验收和100步四项judge现均通过，实际B/量具版本为`00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`。本文件的INPUT原始证据仍锚定`3a1582db39c723c735e04752e5027bfe40ecc3e1`，不改写为后续运行版本。

本次正式CPU阶段于2026-09-26 17:07:21—19:41:35 UTC完成。四侧collector与两份judge均通过，各自命令、正文tee、footer printf、footer tee及综合退出均唯一为0。实际运行版本与原始记录保持不变；本文是运行结束后的结果回写，不把归档提交冒称为起跑版本。启动命令见[launch](launch.md)，总览见[README](README.md)。

## 1. 结论与指标速览

**full与counting两组正式CPU输入对拍均通过。** full覆盖1600集、476857个执行样本身份与每侧6906个定点样本；counting覆盖400集、189035个执行样本身份与每侧2400个定点样本。每侧均有104个真实b64批次及两epoch完整sampler索引，比较终点为`collate_before_jax`。此结果不证明GPU训练五标量、参数/EMA/优化器状态逐位相同，也不代表P1、100步、测速包装20步对照、300步perf或80k已通过。

| 对象 | 已核实范围 | 最终CPU状态 |
|---|---|---|
| counting A/B | 400集、189035执行样本；每侧2400定点样本、104真实批、两epoch全索引 | 两侧采集及count judge均PASS |
| full A/B | 1600集、768897帧、476857执行样本；每侧6906定点样本、104真实批、两epoch全索引 | 两侧采集及full judge均PASS |

两组实际判定原文为：

```text
INPUT_EQ=PASS episodes=1600 samples=476857 boundary_samples=6906 batches=104 epochs=2 comparison=host_numeric_exact_motion_none_v1
INPUT_EQ=PASS episodes=400 samples=189035 boundary_samples=2400 batches=104 epochs=2 comparison=host_numeric_exact_motion_none_v1
```

## 2. 版本与代码状态

本次B及判定工具的实际 **INPUT_HEAD为`3a1582db39c723c735e04752e5027bfe40ecc3e1`**，A固定上游 **`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`**。四侧从clean版本启动，运行期间没有同步依赖或修改tracked文件。四侧的`provenance_start.json`与各自`provenance_end.json`均逐字相同，`porcelain`为空。后续归档提交不得回填为实际采集版本。

A解释器为主仓库下`v1-store/worktrees/orig-ecf086c/.venv/bin/python`，独立上游根；B为主仓库uv管理的`.venv`，由`uv run --no-sync python -B`调用。两侧Python均为3.11.15；起跑前比对208个已安装包的版本记录一致，版本清单SHA256为`a31fe021dbb20937aa93ce68d59a7d2f7d38d92a7e9514256ca2da617f5274dc`。collector元数据另保留JAX/JAXlib 0.5.3、torch 2.7.1、NumPy 1.26.4、Flax 0.10.2等实际关键版本与每侧解释器前缀，不因包版本相同就省略模块来源证据。

四侧使用同一主仓库取证工具`check_orig80k_inputs.py`，文件47669 B、SHA256为`0e52ad01d2de5f9250de465bc7c1fb6550377d564e8037ad476c9ec1150a7305`；项目Dataset/transforms/loader模块分别来自各自根，worker端也留来源记录。数据前置运行Beta为`49a333eb18e8d6ff1143bf7871ef7c498ab91579`，数据组归档提交为`ec6c9e35784a96f636a587dd737dffa294b616ad`；它们不是本次INPUT_HEAD。

## 3. 启动与配置还原

`INPUT_TAG=20260926T170654Z`。主仓库根为`/scratch/hongze/robomme_policy_learning_MotionJEPA`；以下运行产物路径均相对此根。四份collector于 **2026-09-26T17:07:21Z**分别在独立detached tmux起跑，彼此无内容依赖；每组judge只等待本组两侧成功，因此count judge没有等待仍在运行的full组。

完整命令按实际INPUT_HEAD下的`docs/training-doc/orig80k-equiv-0925/launch.md`及本次日志中的`COMMAND=`还原。起跑时该launch文件SHA256为`d4f6f991f92721d19af2cc9575fdb539720b3fef102bbe49df0d390b38f673b0`；运行调度载体`v1-store/bench/orig80k-build-preflight-0925/input-dispatch-20260926T170654Z.command.txt`的SHA256为`1cb4fe7642e7a7bca2c43a15c5eb66e96c31eb334fd5141f06378c2cd5362d84`。不将运行命令载体复制为档案内.sh或yaml。

四侧和judge均显式使用CPU：`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`JAX_ENABLE_X64=0`、`HF_HUB_OFFLINE=1`、`PYTHONUNBUFFERED=1`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；清除`PYTHONPATH/PYTHONHOME/JAX_PLATFORM_NAME/XLA_FLAGS`，未覆盖HOME。只读共享资产位于主仓库`v1-store/models`；uv/HF/XDG缓存均显式位于`v1-store/cache`。六个JAX缓存分别为`v1-store/cache/orig80k-input-20260926T170654Z/{full-a,full-b,count-a,count-b,full-judge,count-judge}`，不共用同一编译缓存。

每份实际命令先写独立`.command`文件，tmux执行同一文件前核对自身SHA。所有命令文件位于`v1-store/bench/orig80k-equiv-0925/commands-20260926T170654Z/`；文件名为下表会话名加`.command`。日志位于`v1-store/logs/`，文件名为同一会话名加`.log`，均在collector输出根之外。

| 会话全名 | wrapper/body PID | 命令文件SHA256 |
|---|---|---|
| `orig80k-input-full-a-20260926T170654Z` | 31409 / 31412 | `58f86ee7d62e39b1294bf23d321c0cd3a85033e786bac6c75b7e22134932f28d` |
| `orig80k-input-full-b-20260926T170654Z` | 31422 / 31426 | `35f9e8d5d1a5df0e63d036f136a3899d56e03845d1c62790f150eb21b02fceaa` |
| `orig80k-input-count-a-20260926T170654Z` | 31435 / 31440 | `f20abe885cb4cd444b248e6c7520d62e8b32f6c3869b32a99fc9753b338ad432` |
| `orig80k-input-count-b-20260926T170654Z` | 31448 / 31450 | `367f50c0a020742121dd01bece128ba0eaea4486a0ef5b96b68bf362213c629c` |
| `orig80k-input-count-judge-20260926T170654Z` | 392319 / 392321 | `ab2f9269afd97127719e0ecce6801bcece33a6b58d26a2e82ae2ed4a7d535519` |
| `orig80k-input-full-judge-20260926T170654Z` | 566576 / 566578 | `3597b818656915a4c6b45474437c4e8d55b48d745d55d63cc2b5a5c50ca848e9` |

四份collector的pane PID分别为31406、31419、31432、31446；full judge的pane PID为566574、pane为`%547`。count judge在6秒内结束，查询pane前会话已自然退出，未补造pane PID。

## 4. 数据集与划分口径

本次不建库、不裁剪数据。full为公开16任务各100集，共1600集、768897帧、476857个执行样本；counting为其中BinFill、PickXtimes、StopCube、SwingXtimes各100集，共400集、189035帧及执行样本。两库与严格SUBSET已通过，前置两次20步真实保存/恢复也已通过，详见[full库](../../dataset-build-doc/16task-pub-1600ep/README.md)、[counting库](../../dataset-build-doc/4task-counting-pub-400ep/README.md)、[full20](../smoke-orig80k-full-0925/README.md)和[count20](../smoke-orig80k-count-0925/README.md)。这些通过范围与本次INPUT分开。

同组A/B共享source、episode manifest和norm_stats。A读取`RoboMMEDataset(source)`，经该侧`TransformedDataset`和原生pickle collate；B读取同库`FrameSampDataset(framesamp)`，经该侧transforms与SHM collate；两条链均在JAX前取证，实际输入不补键、不cast、不改值。source及packed根分别为`v1-store/datasets/<库名>/{source,framesamp}`。

| 资产/身份 | full | counting |
|---|---|---|
| 库名 | `16task-pub-1600ep` | `4task-counting-pub-400ep` |
| 原episode manifest文件SHA256 | `222d84f58fb82073603fda19992c4f82e542475294176bea99e8868036b8a11a` | `fa0535335751910451c483d198cda151f04d25df36b75467c7460d118e25ad52` |
| 全量执行样本身份摘要`identity_sha256` | `aa17a64cd43c443a375df391380f871e5002838cd61a94c856d1d827abb3ba4b` | `9e34dc982130f73d906b70a825aa7ae5b1ee5cf51e9e1f466066d3c66bff294b` |
| norm_stats SHA256 | `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

full的norm路径为`v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`；counting为`v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep/robomme/norm_stats.json`。history文件SHA256为`823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`，tokenizer文件SHA256为`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。原文件摘要、collector规范化manifest摘要和执行身份摘要是不同口径，不混写为同一SHA。

## 5. 输入参数与实际覆盖

四侧固定batch64、workers4、原生prefetch2、seed42、`drop_last=True`、action_horizon20、`perceptual-framesamp-modul.yaml`，modul 4×4、budget512、16 token/帧及motion关闭。四侧并行仅改变调度，不改变参数、种子、sampler或内容判据。

每集定点为合法执行步集合`{exec_start_idx, exec_start_idx+1, num_timesteps-1, 30, 31, 32}`的排序去重结果。count-A/B实际均为2400点，full-A/B均为6906点。每侧实际内容窗口为epoch0批0…99及末2批、epoch1批0和1，共104批、6656个批内样本位置。counting末2批为2951/2952；full末2批为7448/7449。

两遍分别是等长索引探针与真实内容loader，均委托原生RandomSampler/BatchSampler：counting每epoch记录189035个索引、2953个完整批及43个drop_last尾样本；full为476857个索引、7450批及57个尾样本。两epoch分别完整记录`indices/base_seed/generator_before/generator_after/drop_last_tail`，并核对两遍序列；四侧均已完整保存并经同组judge核对。104批是实际解码内容范围，不能说成两epoch全部样本都已解码或全部内容逐项对拍。

协议为`schema=2`、`host_numeric_exact_motion_none_v1`。用户允许主机dtype不同，但无损数值摘要精确相同；shape、顺序、signed zero、有限性及未豁免字段继续严格。仅`motion_emb/motion_pos/motion_mask/mem_order`允许缺失与值严格为None等价；任何非None值（包括False、零数组）或其他键差异失败。元数据完整登记`equivalent_motion_none_keys`，原始`raw_all/transformed_all/inputs_all`、dtype与raw SHA保留；既有recur四键的严格None规则亦保留，不扩大白名单。训练五标量、参数/EMA/优化器状态仍要求bitwise，本次不修改该要求。

## 6. 硬件、资源与耗时

环境为AWS单机，`/scratch`为`/dev/md0`上的XFS本地NVMe；本次显式用CPU，起跑前8张A100-SXM4-80GB均空闲。起跑前只读快照为96个可见CPU，MemAvailable `1142987542528 B`（约1064.49GiB），scratch可用`851594649600 B`（约793.11GiB），SHM可用`602265333760 B`（约560.90GiB），当时cgroup OOM事件0。它们是现场快照，不是整个运行的资源上限。

资源记录为`v1-store/bench/orig80k-equiv-0925/resources-20260926T170654Z.jsonl`，归档副本为`records/resources.jsonl`，共6份离散快照。运行中首份为17:08:27.636713 UTC；其中18:52:49.236571 UTC这一份的MemAvailable为`1127615328256 B`、SHM使用`2310602752 B`、scratch可用`851238481920 B`，full-A/B进程树PSS分别为`10949635072/4762867712 B`，线程数209/215，当时counting已结束。第6份为19:46:07 UTC的结束后快照，所有INPUT进程均已结束，当时有独立CPU单测，不能归因为输入占用或输入峰值。只报告这些实际时点，不将稀疏记录中的最大值称为实测全程峰值，不把RSS共享页重复计数当成独占内存。

预建时约50GiB应用工作集及约4.78GiB SHM是按shape/prefetch推算的估计，不能改写成本次观测。此阶段不测训练吞吐，不根据A/B采集总耗时推出loader速度比或GPU瓶颈结论。

| 任务 | START_UTC | END_UTC | 包装日志起止耗时 |
|---|---|---|---|
| count-A | 2026-09-26T17:07:21Z | 2026-09-26T18:32:44Z | 5123秒（85分23秒） |
| count-B | 2026-09-26T17:07:21Z | 2026-09-26T18:22:36Z | 4515秒（75分15秒） |
| count judge | 2026-09-26T18:33:44Z | 2026-09-26T18:33:50Z | 6秒 |
| full-B | 2026-09-26T17:07:21Z | 2026-09-26T19:25:33Z | 8292秒（138分12秒） |
| full-A | 2026-09-26T17:07:21Z | 2026-09-26T19:41:04Z | 9223秒（153分43秒） |
| full judge | 2026-09-26T19:41:19Z | 2026-09-26T19:41:35Z | 16秒 |

## 7. 采集过程与退出回执

count-B先完成，之后count-A完成；确认同组两侧唯一collector PASS、五项唯一退出0、会话已结束及record_manifest完整后，于18:33:44 UTC启动count judge。judge输出上述唯一INPUT_EQ PASS，并于18:33:50 UTC完成。full-B于19:25:33 UTC完成，full-A于19:41:04 UTC完成；两侧17项文件的bytes/SHA和文件集合均核对完整。full judge于19:41:19 UTC启动，19:41:35 UTC输出结束回执，正式INPUT_EQ PASS。六份本轮会话均已结束，没有清理其他会话。

四份collector及两份judge的原始日志均已重读核对，以下五行各自恰出现一次且全0：

```text
COMMAND_EXIT=0
TEE_EXIT=0
FOOTER_PRINTF_EXIT=0
FOOTER_TEE_EXIT=0
EXIT_CODE=0
```

日志头还保留会话全名、wrapper/body PID、声明的INPUT_HEAD、命令文件期望/实际SHA及START_UTC，尾部保留SESSION_END、INPUT_HEAD_END和END_UTC。修正后的包装先核footer printf/tee状态再直接追加唯一综合退出码；没有把会话消失单独当作成功。四侧原始输出与六份原始日志均保留；会话结束状态与日志终态一起核验。

上述五行是原始日志实测值，不是最外整体进程独立wait回执。固定源码直接捕获六个collector/judge子命令的返回，所以COMMAND_EXIT=0及工具/内容验收保留；最终direct append之后的最外真实退出码未独立保存，现不可追补。该后续发现与用户已批准的沿用处置见第9、10节，原五行和全部原始文件不改。

## 8. 输入判定与后续评估状态

两组PASS分别覆盖本次各组记录的共同来源/资产、全量身份和索引顺序、full的6906点或counting的2400点，以及104个真实批的精确数值与授权None等价。原始dtype、schema和raw字节摘要仍可查；PASS不意味着所有原始摘要都逐位相同，也不是全量两epoch内容解码。

两份judge都保持`--expect-head-a=ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`、`--expect-head-b=3a1582db39c723c735e04752e5027bfe40ecc3e1`。正式结论来自原始完整collector目录；无损归档只保存证据，不重新命名或改写其provenance。

以下保留CPU首次结果回写时的阶段范围：P1两步入口自检、上游/当前100步、同入口单跑/并跑、两库各一对新测速包装20步开关对照、300步perf及两个80k均尚未执行。既有两次数据可读性20步确曾真实保存并恢复checkpoint19，但不是新测速包装20步等价对照。此阶段未运行策略评估。

## 9. 用户决定记录

用户原始目标为「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」，执行授权为「开始实现该计划 有问题立刻问用户 不要自己决策」。公开全集16×100与counting4×100同源子集、原版modul/训练超参、counting重算norm及正式两组名称沿用。阶段性「本轮只实现和验证工具，暂不建库」后来由「恢复计划中的建库，严格按前置闸门推进」覆盖；已经完成的数据前置不自动豁免后续验证。

输入口径沿用「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」。motion最初按「先保留严格判据并取证，结果出来后再决定」留下三样本严格schema失败事实；其后用户明确「允许这四键缺失与 None 等价」，仅按第五节四键生效。历史失败不删除，当前PASS不扩大为其他字段或数值容差豁免。

用户随后批准「允许仅对拍关闭 W&B」和「补充 0.5 秒只读采样」。前者仅覆盖P1/100步A/B，仍留完整本地标量与状态；既有/后续20步、300步perf和80k仍开启W&B。后者只读采本run checkpoint分配字节及`/scratch`可用字节，后续预算保留保守余量，不把离散采样最大值冒称连续真实峰值。

历史最外footer证据边界提出后，用户选择「补记限制，继续 CPU 输入取证」，因此本轮按新包装继续，保留旧数据与20步子任务/产物PASS，既不补造旧最外进程退出观测，也不重训已验收产物。用户另已选择「两库各跑一对 20 步（推荐）」，用于新测速包装关闭/开启的真实等价覆盖；两侧仍真实保存且W&B开启，**截至本CPU结果回写尚未执行**。上述决定完整保留，均不是相应闸门已经通过。

后续用户分别明确「补记P1限制，补独立退出取证后继续（推荐）」和「同样补记限制，沿用INPUT结果（推荐）」。两项决定各自作用于P1与INPUT的历史证据边界，不补造旧最外退出回执，也不放宽输入数值/顺序/来源或训练逐位判据。用户另要求「继续工作 一路做到起泡前 有问题问用户」，本轮据此只推进至正式80k起跑前；「尽可能并行做」「你有8张卡」按保留判据的4+4调度执行。

## 10. 计划外事件与处置

起跑前新INPUT包装曾存在先写成功EXIT_CODE再检查footer tee的确定缺口。现已修正，并由实现方及独立审查方分别完成6例纯shell替身测试（正常、命令exit7、正文tee失败、footer完整输出后失败、自SHA不匹配、日志冲突）；两方footer失败例分别返回1与29，未残留成功终态。这些测试不能补成旧四次数据/20步最外层的独立退出观测。历史边界详见[exit-record-audit.md](exit-record-audit.md)；原始历史记录保持不变。

后续发现另一层边界：最终direct append可能已经完整写出日志0，随后自身返回非零。固定INPUT源码在子shell最后直接执行collector/judge并捕获其PIPESTATUS，所以六份COMMAND_EXIT=0仍是实际子命令返回0；四份collector、manifest及两组INPUT_EQ内容验收保留。已结束INPUT/P1没有保存最外整体进程独立wait/pane退出码，现不可追补；日志EXIT_CODE=0不是该最外实际返回码的独立证明。目前没有这些真实任务失败的新证据，纯shell反例也不写成它们的真实退出值。

[INPUT静态取证补充](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/input-outer-exit-scope-supplement-20260926.md)的SHA256为 `bd8c37ba659b2dca0db52f1ebf54c5ff9e925416acbe9e7f8b0e3b46dfb2fd8a`，后续[INPUT批准说明](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/input-outer-exit-scope-approved-20260926.md)为 `c9b3149476fea956f1baf2b473667b2c5221e9e24c0922ea4004fe521d4b093a`；[P1已批准限制说明](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/archive-staging/p1-result-draft.md)为 `aa2bf921ffbc676bc1444c00bf42ff9f6378fccc3ff5d336c30085e2ea76b1c4`。这些追加说明不改变原collector、数值/索引结果、SHA、raw日志或归档包，也不扩大[exit-record-audit.md](exit-record-audit.md)原固定审计范围。

P1四侧入口、finite、分段与来源记录完成；A保留tentative/main两段状态，不去重，也没有以P1执行100步轨迹judge。按用户对P1的独立批准保留这些证据，不把原日志0标成最外实退0。后续必须先保留本任务pane并持久化identity后才释放启动门闩；结束后联合验原生pane退出、capture父进程真实返回及sidecar文件绑定，再核原有内容判据。父适配本身的真实返回由宿主/controller另留证；工具或日志自报0、session消失均不能代替独立退出证据。

本次六份原始日志均没有tqdm中间态需要过滤，第一份stage summary与各自原日志逐字相同。随后为满足Git行尾空白检查，在独立`git-ready/`副本中仅删除每份日志`COMMAND=`行末一个未转义ASCII分隔空格（0x20），不改变解析argv；不是删掉参数字符。每份各减少1 B，各18条关键行规范化一致、6条判定/退出行保留原始字节。原始日志、旧stage/checks、collector记录和tar包均未改，没有借日志清洗删去失败或进度记录。

无损记录包只规范化tar成员元信息（顺序、uid/gid、mode/mtime及gzip头），不改变成员JSON/JSONL字节。每侧18成员包含17项record_manifest登记文件加清单自身，解码后逐项核对SHA/bytes及原清单字节一致，并核原目录未变化；归档PASS不等于重新执行collector或INPUT judge。

## 11. 当前结论与下一步

两组正式INPUT_EQ现均PASS，六会话结束后已解除本阶段tracked冻结；完整原始证据继续锚定实际INPUT_HEAD。CPU采集期间预算/测速包装工具仅在ignored路径准备候选，后续工具集成与测试单独留证，不能把本输入结果当作这些候选已经通过真实训练的依据。

CPU阶段解除冻结后，预算对接和测速包装20步对照工具已落实到源码并完成工具合测，具体实施验证记录见[总计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)。[P1预建记录](p1-launch.md)与[100步及单跑/并跑预建记录](train100-launch.md)保留00bd起跑前文本，后续真实执行不回写成当时已知。其后P1按实际量具/B提交`00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`完成并按批准补记限制；上游锚点仍按原档案。100步六侧训练和四项judge均通过，来源、有限值、500个标量及2×201叶完整状态逐位判据均满足，并独立核验十个最外进程原生退出0。新20步、perf和正式预算仍待完成；正式预算仍须真实perf/恢复、末步stat、采样证据及保守余量来源，80k全部前置保留。

最新调度保持完整前置：100步S1/S3分别独占训练窗口，不与另一组GPU任务并跑；S2以GPU0–3和4–7形成真实训练重叠并按同HEAD/环境判据验收。新20步为两条并行库链，各库off训练、真保存/恢复和独立退出验收后才进入同4卡on，再及时进行本库CPU judge，两库互不等待无关阶段；300步perf按4+4并跑并核稳态重叠。20步只验正确性，不据其耗时作吞吐结论。每阶段保留各自全新输出/cache、W&B既定口径、真实恢复、磁盘预算和独立退出要求；两份正式launch最终同Beta/clean就绪后仍停在放行前，不启动80k。

仅改文档或无关测速/预算工具时，旧输入证据继续锚定原INPUT_HEAD；沿用前留新旧提交diff以及取证工具、输入链模块、依赖、数据和资产指纹一致性证明，不改原记录HEAD或声称新HEAD已重跑。取证工具变化要求两侧同工具重采；输入相关变化或影响不明先报告，训练S1/S3/S2仍要求同HEAD。

工具集成后的[提交前指纹复核](records/input-fingerprint-bridge-precommit.json)已通过：从四侧44份父进程/worker/元记录汇总113份实际文件引用，逐项核对当前字节数与SHA；主环境208项依赖版本及Python构建仍同输入阶段，`src/packages/shared`、依赖配置/锁与输入取证器对INPUT_HEAD无差异。该文件42374 B、SHA256为`8c1b10ce39593b42bd579b22b7d9d061d071dc1e1165681b821a96ec14828805`。这是尚未提交工作区的复核，实际TRAIN_HEAD仍须由起跑前固定提交复核记录给出，不宣称新HEAD已重采。

## 12. 归档文件清单与摘要

原始collector根为`v1-store/bench/orig80k-equiv-0925/input-20260926T170654Z/`，各侧子目录为`count/a`、`count/b`、`full/a`、`full/b`。原始清单逐项绑定17个文件的SHA/bytes；下表payload不含清单自身，不能与压缩包字节混用。

| 侧 | 原始payload B | record_manifest SHA256 | 无损包及大小 |
|---|---:|---|---|
| count-A | 49264791 | `4adabc7d5e4f959682530360c6ea06acb30222cc5419ffd16b4e19ec9706c24e` | `count-a.records.tar.gz`，5328311 B |
| count-B | 49198475 | `ac66670e25f5c9f9b01e088b5c15d1577f62d6cc11563c67c4be54b6fbc69430` | `count-b.records.tar.gz`，5313445 B |
| full-B | 138462397 | `e70f4dc1c66e60ac97b1dd3f7fc116a82bb783f4bb4e00dd34c236572808ddf0` | `full-b.records.tar.gz`，14720809 B |
| full-A | 138642249 | `b95d89845d2e7219a6102cd05ffe26d5941ebcf66179f5c423539e1d710dbc05` | `full-a.records.tar.gz`，14740202 B |

已完成无损包均位于`v1-store/bench/orig80k-equiv-0925/archive-staging-20260926T170654Z/`：count-A包SHA256为`2ce614db05bf0c7c2fa348bf274df3a8155bfa3b066e5c0a062e21e2013b8e12`，count-B为`d41ca9f527718b188252fb4410d26cd83fc95c2ea09668e06abe7eeda5928140`。对应`count-a.archive_checks.json`和`count-b.archive_checks.json`的SHA256分别为`02ff448600576399f8d9ee133888e4fd31fb3ed0be6779b503877b7a5f0fc095`、`fe7aa8774e747324b625bc31cc77efcc9b8262fa2e2f0479ce1d9f72658247cf`。

| 原始日志（`v1-store/logs/`） | 字节 | 原始SHA256 |
|---|---:|---|
| `orig80k-input-count-a-20260926T170654Z.log` | 8543 | `907324d97d4d7b3a57bcb34488bfdbac93cee687d3b090ddb0418e429df5b509` |
| `orig80k-input-count-b-20260926T170654Z.log` | 8720 | `0c0569957bded1da5106191f25bd747fe67f31b4e2e0af4a5de3aa101668ee29` |
| `orig80k-input-count-judge-20260926T170654Z.log` | 2255 | `6b69b0cdaf34e04e086324d014540d47d9a6dbc71088cb7c96846b3d9a2b91b0` |
| `orig80k-input-full-b-20260926T170654Z.log` | 11796 | `6caf0915328a89227a1b2214141789523357a9a4e2fc3dd974a00408613d49b9` |
| `orig80k-input-full-a-20260926T170654Z.log` | 11619 | `5630af6c6f3a8bfda4f67dc114be3f1315ebe4dc4fad1fa2ded69852627a5e80` |
| `orig80k-input-full-judge-20260926T170654Z.log` | 2250 | `282a827f091ae72f4ad390f54fc1be3e2ee829f45c18ec55b26e5da4fa9cec32` |

count三份Git副本位于同stage下`git-ready/`，文件为`count-a.summary.log`、`count-b.summary.log`、`count-judge.summary.log`，SHA256分别为`f3a23050075686ebd31437dea5bb777b9c97667be3dbaf3602048af5c57a8278`、`be3cdd378e9c2395d656d440e76dab86ad6f3a1d01a828336b298df856350808`、`d324f0b8ab7cc8cbff163de95a0fd363e74b996674757e8bf1c7f310c2830fd9`。原日志处理检查`count.logs.archive_checks.json`的SHA256为`9b9b00e4c75d252840ed6c38edad29a9b7a557af68b8298e46747edb65a4d131`；`git-ready/final-log-checks.json`为`d9445271bed4a080e5072d9cd1b91db6e9fe473e99ea4ce01f705cc4a8e37522`。上述五个文件最终均按同名归档到`logs/`；其中`logs/final-log-checks.json`只约束count三份，full检查各自使用带侧名前缀的文件，不能相互覆盖。

正式归档位置为`records/{full-a,full-b,count-a,count-b}.records.tar.gz`及同名`*.archive_checks.json`、`logs/*.summary.log`和日志检查链；资源快照为`records/resources.jsonl`，控制器快照为`records/runtime-controller.json`。资源最终记录17605 B、SHA256为`78a635779eea0ab869bbad10a655ef89a36dc69959c5da20938ffafb1b05f66a`。控制器固定导出27629 B、SHA256为`888cd2d0fa17139b6c6e56e1f79c97e9b0b7c16750cd206bc37c41475c60d1d0`，它是INPUT结束后的跨阶段快照，含当时工具集成状态，不冒充启动时快照。含提交前指纹复核在内，23份副本共40294291 B，均与对应stage或导出源逐字节相同；另附7769 B的[归档索引](records/archive-index.json)，SHA256为`981c36f728568b079bffaead0ae6285fe9cf084736778000f83b944adc2452d7`，逐项记录来源、字节数和摘要。只归档Git不能还原的记录，保持collector内部清单完整；不复制源码、命令脚本/yaml、权重、缓存或凭据。运行命令正文由实际日志和固定版本launch还原。

以下full归档摘要均来自已完成的stage检查；实际CPU采集、两组judge、四侧无损打包及正式副本核验均已完成。归档提交由包含本文的Git历史确认，不预写未来提交。不得为归档改写原始collector、判定或运行HEAD。

| 归档文件（本档案内约定路径） | 字节 | SHA256 |
|---|---:|---|
| `records/full-a.records.tar.gz` | 14740202 | `0a6b3b4f0f7db98c134f4721506a13ac4f43a8d5e9de01bbf8b98d87c6cb36b1` |
| `records/full-a.archive_checks.json` | 9108 | `d9c0769784f749d0ad7ad8557abfa43b2eda772367c596eefe0781487981881b` |
| `records/full-b.records.tar.gz` | 14720809 | `7e9b7ebc24db5ea8e9ff29ada0154637cc3ea61a190e539dcee1354366486d25` |
| `records/full-b.archive_checks.json` | 9127 | `eee8a1be4c9d3a27eda8fc8a30eabbd8075c34ad2f49f89f2e6fdac595ea4738` |
| `records/full-judge.archive_checks.json` | 1218 | `477a3564aec1752e6bdf9d6461a341d3bd22ed573c1940e26830b73deba8414c` |
| `logs/full-a.summary.log` | 11618 | `1c33c61ea5c8a94f813fd52ad0df07fc5acbcb357e88ce6908c4b797dddf45fd` |
| `logs/full-b.summary.log` | 11795 | `a928d95c2b096fa90257bceb539130fa45785f61f5c13c24f1d3650019e253a3` |
| `logs/full-judge.summary.log` | 2249 | `9331fe3daf1523d8671d76acfd03beeae3291c40104c7e451b49e8f1ee5e6e7b` |
| `logs/full-a.final-log-checks.json` | 2039 | `d2cdef2c3415e8fa790b70c8235aed14332539aa872fe3d76cf956dd2f3256b8` |
| `logs/full-b.final-log-checks.json` | 2055 | `d5aba512f9673148410c8c764643597a4548eae8a6ca299b239c1b6d545618ca` |
| `logs/full-judge.final-log-checks.json` | 2056 | `34567e7522d37baf9d4f896428c4312482d73117620c8147f7e6ccb78b326c11` |
