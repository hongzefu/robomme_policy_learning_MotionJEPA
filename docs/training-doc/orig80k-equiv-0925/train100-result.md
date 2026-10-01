# 原版80k前置100步2a/2b结果

本轮六run、两组upstream及两组same-entry正式judge全部通过根与独立验收，4+4主机完成间隔重叠满足既定门槛。最后两组same-entry独立复核记录于23:11:53 UTC，主代理于23:13:22解除源码冻结；之后的工作区修改不计入旧运行快照。正式导入只含已完成实测资料，未改原记录，见[导入回执](records/post-eq100-import-receipt.json)。

## 1. 一句话结论与指标速览

**本轮100步2a/2b阶段完成：full与counting的upstream及same-entry对拍均在本机固定确定性档通过，4+4重叠达到既定门槛。** 每组逐步检查100步×5键，共500个标量比较，hex失配为0；main的state标签50、99各含201叶，两行状态摘要失配为0，有限性、配置和来源均通过。

| 项目 | full | counting | 当前结论范围 |
|---|---|---|---|
| 上游A run完整性 | tentative12/main100，state50/99各201叶 | 同左 | 已结束，原生退出及独立capture证据通过 |
| B单跑基线 | S1 main100，state50/99各201叶 | S3 main100，state50/99各201叶 | 已结束，作为同HEAD的solo基线保留 |
| upstream正式judge | 500个标量比较、2行状态，失配均0 | 同左 | 两组各自ENTRY_EQ=PASS |
| S2完整run | main100、state50/99各201叶 | 同左 | 原生退出、capture及来源完整性通过 |
| same-entry正式judge | 500个标量比较、2行状态，失配均0 | 同左 | 两组各自ENTRY_EQ=PASS |
| S2重叠 | complete_steps=[90,85] | complete_steps=[85,90] | 交集631.9925267696381秒，双方完整间隔均≥50 |

## 2. 版本与代码状态

实际共同 `TRAIN_HEAD=00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，真实CPU输入版本为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`。A源码位于主仓 `v1-store/worktrees/orig-ecf086c`，使用该工作树独立uv环境；B使用主仓独立uv环境。记录中的B锚点不替代A的实际源码HEAD。

本轮标签为 `20260926T210415Z`。从两A于21:06:08 UTC起跑，至六run/四judge完成和独立复核结束，主仓tracked、上游源码、两侧venv及训练输入资产持续冻结；各run起止provenance和runtime分别留证。已完成A各记录49个项目模块、B各52个，运行依赖均208项。主代理于23:13:22 UTC明确解除冻结，此后开始新代码/文档整合；本结果继续只绑定旧00bd/ecf及原始测量，不把后续dirty工作区或新提交冒充旧运行版本。[收尾controller不可变快照](records/train100/same-entry-increment-20260926/completed-controller.snapshot.json)为87690字节，SHA `b7cac28e1ed8c7ec90f4888c4491876f9a5c0b445b5192f6cf6cef3114b96c03`。

训练/判定代码锚点为[已提交train100 launch](train100-launch.md)及 `run_entry_equiv.sh::common_args/run-a/run-b/judge`、`entry_equiv.py::_run/_judge/_concurrent_overlap`。后加入的退出守卫是单独冻结的ignored工具，SHA256为 `f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b`；它没有变更上述Git源码或训练参数。

## 3. 启动与配置还原

记录根为 `v1-store/bench/orig80k-eq100-20260926T210415Z`，命令及身份根为 `v1-store/bench/orig80k-eq100-commands-20260926T210415Z`。六run的exp_name统一为 `eq100-orig80k-<下表侧名>-20260926T210415Z`，会话统一为 `orig80k-eq100-<侧名>-20260926T210415Z`。

| 侧名 | 记录相对根 | 物理GPU | execution_role |
|---|---|---|---|
| full-a | full/a | 0–3 | upstream |
| count-a | count/a | 4–7 | upstream |
| full-s1 | full/solo | 0–3 | solo |
| count-s3 | count/solo | 4–7 | solo |
| full-s2 | full/concurrent | 0–3 | concurrent |
| count-s2 | count/concurrent | 4–7 | concurrent |

原配置与入口可从固定提交还原；下面仅为只读还原命令，不启动训练：

```bash
git show 00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8:docs/training-doc/orig80k-equiv-0925/train100-launch.md
git show 00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8:scripts/training/tests/run_entry_equiv.sh
git show 00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8:scripts/training/tests/entry_equiv.py
```

实际展开命令的path/bytes/SHA由各identity和归档清单绑定，不在docs另存.command正文。主环境覆盖保持 `JAX_ENABLE_X64=0`、OMP/OPENBLAS线程各1、HF离线；GPU阶段清除遗留CPU平台和record/timing变量。驱动固定 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、显存比例0.95、按GPUS设置CUDA_VISIBLE_DEVICES，以及原UV/HF/XDG缓存路径。每个run独立JAX/CUDA/W&B路径，S2显式覆盖自己的RECORD_B、EXP_B、CHECKPOINT_B和MMEVLA_JAX_CACHE_DIR。

六份后续dispatcher与保留旧版逐字对比，唯一变更是launch末层由裸tmux改为f8门闩及独立记录；原body、命令生成、训练/四judge argv/env、缓存新根与顺序检查未变。证据见[退出启动层变更账](records/train100/background/eq100-exit-guard-dispatch-change-20260926T210415Z.json)，SHA `cdcb5e9c6c348758274578acfb9f1a1563466161c508c03a7b12795c12d216a4`。CPU judge实际依赖调度另见第10节。

## 4. 数据集与划分口径

两库由建库Beta `49a333eb18e8d6ff1143bf7871ef7c498ab91579` 构建并留档。本阶段没有新训练/验证划分或重建数据；A读取source，B读取同库packed framesamp，通过已绑定的source/manifest保证来源映射。full训练身份为1600集、476857样本；counting为400集、189035样本，四任务为BinFill、PickXtimes、StopCube、SwingXtimes，严格子集验收已在建库阶段完成。

| 组 | 数据库（相对v1-store/datasets） | assets-dir父目录（相对v1-store/train-assets） | 实际norm SHA256 |
|---|---|---|---|
| full | 16task-pub-1600ep | mme_vla_suite | `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` |
| counting | 4task-counting-pub-400ep | mme_vla_suite/4task-counting-pub-400ep | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

full沿用原版norm，counting使用其自身建库统计。asset-id均为robomme；history为 `perceptual-framesamp-modul.yaml`，SHA `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。初始化参数和tokenizer由既有资产锁核验，不因本轮归档重新下载。

[100步起跑前置](records/train100/background/eq100-preflight-20260926T210415Z.json)SHA为 `12219976ca38d58acd4a9b6f2b20f2a17ae91f488fab29bef051f1e7e1ae83d1`，绑定P1、实际资产校验返回码0、包指纹、库小元数据和现场资源。原INPUT结果继续保留原INPUT_HEAD及实际覆盖；用户对其最外退出证据限制的处置见第9节。

## 5. 关键超参与验证口径

| 参数 | 本轮实际值 |
|---|---|
| main训练步数 | 100，索引0…99；A额外tentative12，B无tentative |
| batch / workers / fsdp / seed | global64 / 4 / 4 / 42 |
| history | modul，512 / 4×4 |
| 日志与状态采样 | log_interval=1，save_interval=50，状态标签50和99 |
| 原训练配置 | warmup10000、学习率5e-5、EMA0.999、keep10000 |
| 在线W&B | 仅本P1/100步对拍关闭，完整本地记录保留 |
| 五标量 | loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm |
| 一致性 | 每组500个main标量hex逐项一致；两状态点的完整参数/EMA/优化器/step摘要逐位一致且有限 |

harness临时替换save_state记录完整状态摘要，不调用原真实保存。因此本100步不是checkpoint写盘、真实恢复或吞吐验证；存在工作目录不代表保存过权重。A tentative在step11结束且未到状态保存分支，main只有50/99两行状态，未把P1的重复step1口径带入本轮。

## 6. 硬件与耗时

实际环境为ENV_B：AWS单机8×NVIDIA A100-SXM4-80GB，驱动595.71.05；`/scratch`为本机NVMe RAID `/dev/md0`、XFS。Python为3.11.15。全组使用各自固定四卡UUID；S1/S3保持本轮GPU单跑基线，S2使用0–3与4–7两组互不重叠四卡并行。

下表时间均为2026-09-26 UTC，耗时来自完整run记录，不作为稳态性能结论：

| run | 实际开始 | 实际结束 | 墙钟秒 | harness PID / pane |
|---|---|---|---:|---|
| full-a | 21:06:08 | 21:40:15 | 2047 | 704727 / %552 |
| count-a | 21:06:08 | 21:40:19 | 2051 | 704740 / %553 |
| full-s1 | 22:11:16 | 22:29:49 | 1113 | 1049115 / %559 |
| count-s3 | 22:30:39 | 22:49:33 | 1134 | 1067732 / %560 |
| full-s2 | 22:50:53 | 23:09:33 | 1120 | 1085946 / %563 |
| count-s2 | 22:50:54 | 23:09:56 | 1142 | 1085982 / %564 |

两A耗时包含官方tentative/main两段和初始化/JIT；B耗时也包含初始化、编译及状态取证。完整perf仍须另按300步稳态和0.5秒采样执行。[S2起跑前置](records/train100/background/eq100-s2-prelaunch-20260926T210415Z.json)在22:50:49记录八卡无活动训练进程、显存0和scratch可用851067650048字节；这是当时资源观测，不是正式80k磁盘预算PASS。

## 7. 训练过程行为与退出证据

两A各完成tentative12+main100，初始化所有权调用2次；S1/S3各main100、初始化1次且不覆盖旧工作根。四run的主记录步集合完整、state50/99各201叶有限，来源和起止环境指纹核验通过。原始与独立报告分别绑定真实上游或B锚点，没有把声明的TRAIN_HEAD字段当作额外git观测。

发现最外append证据缺口后，主代理于21:18:27仅为仍存活的两A窗口开启remain-on-exit，identity标记enabled_mid_run=true；两A结束后实际取得dead=1、status=0、signal为空的原生回执及capture外部实际返回0。S1/S3及两份upstream judge则使用冻结f8：先创建唯一门闩、为本窗口保留退出、落identity，再释放原命令，enabled_mid_run=false。命令文件、身份、原始tmux输出、原生退出字段、guard SHA和外部capture侧证均分别绑定；日志0没有被当作原生退出证明。

[S2 state50预检](records/train100/background/eq100-s2-state50-preview-20260926T230222Z.json)SHA为 `5f30c35398cbb25aefb21958ce1851c1574bd2ce83330708e5c16c9f2af4bde1`。23:02:22时两侧均已出现201叶的state50，mismatched_fields为空；full strict_global为 `3ee1d1dfbbf92d0204aa5cb1933cd33217137e639f61b0efa3a394e021193bfe`，count为 `e413d3adde35d0a5b2331dc2aa2d9fbd3d7025ccc8d3e81f94d168b30989caad`，分别等于solo。该前缀观测不覆盖step99、完整100步、最终退出或并行窗口。

| S2最终验收 | full-s2 | count-s2 |
|---|---|---|
| 分段与有限性 | main100/tentative0，全部有限 | 同左 |
| 状态 | 50、99各201叶，完整 | 同左 |
| 来源/环境 | B00bd、52项目模块、208依赖，起止指纹通过 | 同左 |
| 原生退出/capture | dead1/status0/signal空，capture实际0 | 同左 |
| 最终run报告 | 双侧根报告SHA `dfe371a3a8e3a38e49e61074dda8166261c9d90f7adb4afbce8a78071a3ef173` | 同一报告count-s2分支 |
| 独立run报告 | SHA `3a976996abb7943e616c3d369d4d8a44ba851fcc04160f244e8baf541afd94dd` | 同一报告count-s2分支 |

S2最终验收在其真实结束后完成，先前state50预检仍作为当时的前缀证据保留。六run总报告为33785字节、SHA `f23eb261938939958e122cf5138d604175ffc922f2f2552293351e3b0d379648`，不以预检替代完整结果。

## 8. 训练后正式判定

两份upstream与两份same-entry正式judge均已通过，scope限定本机确定性档、各自前100步；后两份另验证solo/S2来源及实际4+4重叠。每组500个标量比较失配0；两行state摘要失配0，每行201叶；配置、来源、finite通过。历史外部锚点显式跳过，未借旧A40/1000步结果作本轮依据。

| 判定 | 实际左右记录 | 结果 | 两侧scalar TSV共同SHA256 |
|---|---|---|---|
| full upstream | full/a → full/solo | PASS | `dac4e9e1ac5ab5e7308e9282cdf2fec140408c726c60b41b0a397158401cc5ec` |
| count upstream | count/a → count/solo | PASS | `d37c6b4eb6aeb58207e92657d70a7d0b71d50d27fbcfc5dcefa881d2236904e9` |
| full same-entry | full/solo → full/concurrent；peer=count/concurrent | PASS | `dac4e9e1ac5ab5e7308e9282cdf2fec140408c726c60b41b0a397158401cc5ec` |
| count same-entry | count/solo → count/concurrent；peer=full/concurrent | PASS | `d37c6b4eb6aeb58207e92657d70a7d0b71d50d27fbcfc5dcefa881d2236904e9` |

| 正式judge证据 | 根报告SHA256 | 独立报告SHA256 |
|---|---|---|
| full | `cb19125eea2f71f1536ac8b4490d6cd513861812470ec8c0b5a86dc0c11e8163` | `e95102d9121294e0fb80fc6334e32ef1feff332b5edf389a00c627512f22f94f` |
| count | `382c9bf19f8ded3d15c6307a4c8d78141ba3a2786d8cd961fa807464e4ee6792` | `22bfa7a15872ca2f5d2d4f35f031a1adef52dd3a159950028fbf9084f737a820` |
| full same-entry | `f3f52bef9e75f3e3556adf950cfb912359369a1b240ea5e8cd406e603a7d5062` | `9fcf54005bdb5766eade0aea4aa89d18cefdf507c3bdded83691e92c9f255e3f`（合并独立报告） |
| count same-entry | `bfa8908f236974dbc2d362c16b06be06345b5f9cbf381ead8c1b9709d56fe063` | 同一合并报告count分支 |

same-entry必须保持HEAD_A=HEAD_B=本轮00bd、tentative期望0，使用对应solo与concurrent而非上游A。除ENTRY_EQ/SCALARS/STATE/FINITE/RESOLVED_CFG/PROVENANCE外，还须真实ENTRY_CONCURRENT=PASS。工具从step10…99的主机完成间隔 `[time(step-1),time(step)]` 求窗口交集；两侧各至少50个完整间隔，四卡UUID集合不重叠，不能用两个进程同时存活替代。

| 已完成重叠字段 | full same-entry报告 | count same-entry报告 |
|---|---|---|
| window_steps | [10,99] | [10,99] |
| begin / end | 1790463191.363476 / 1790463823.3560028 | 相同 |
| seconds | 631.9925267696381 | 相同 |
| complete_steps | [90,85]，本侧full/peer=count | [85,90]，本侧count/peer=full |
| peer与GPU | 另一组S2，互不重叠的四卡UUID | 同左 |
| judge原生退出/capture | dead1/status0/signal空，capture实际0 | 同左 |

这些主机时间不是GPU内核时间或吞吐测量；重叠不足时保留证据并报告，不改阈值或复用输出掩盖失败。

## 9. 用户决策记录

用户原话「允许仅对拍关闭 W&B」用于本P1/100步，完整本地标量和状态保留；20步真实保存对照、300步perf和80k仍开启W&B。

用户于21:18:27批准「补记P1限制，补独立退出取证后继续（推荐）」。历史P1的入口、分段、有限性和来源结论保留，同时如实记录最外最终append缺独立等待回执；没有把故障注入的非零当成历史实际失败，也没有补造已消失pane的状态。

用户于22:11:03批准「同样补记限制，沿用INPUT结果（推荐）」。四份CPU collector和两份judge的子命令及INPUT内容证据继续沿用，最外退出缺项仍保留；这项批准不改写原INPUT_HEAD或原始文件。

用户另要求「尽可能并行做」「你有8张卡」。实际保留S1/S3 GPU单跑基线，S2按4+4并行，独立CPU判定和归档按真实依赖并行推进。上述原话、时间及范围见[已完成控制记录快照](records/train100/same-entry-increment-20260926/completed-controller.snapshot.json)。早前读取动态控制文件时SHA为 `184c5d9a23958a78085920298d04d465f2244221a787347e108f6ad60c992f22`；最终归档快照为第2节b7cac摘要，两者是不同时间点，原动态文件仍可更新。

## 10. 计划外事件与实际调度调整

原强wrapper的纯shell注入出现“最终append已完整写出0、调用却返回非零”的反例。历史日志没有因此被判定为实际失败；本轮新增独立原生退出链，保留原wrapper/训练命令正文，所有后续阶段使用f8门闩，结束后联合验收原生receipt、capture真实返回码、原日志与内容记录。原记录和历史限制均未重写。

已提交launch原排布为S3完成后再发两份upstream judge。根据并行要求，实际拆成按各组依赖独立调度：full A与S1完成后，full CPU judge于22:32:10运行；当时S3已于22:30:39起跑、至22:49:33结束。count CPU judge在自身S3完成后于22:50:02运行。两份judge的日志起止时间分辨率为1秒，不能由相同秒值推断精确零耗时。

此次只移除full CPU judge对无关count S3完成的等待，原judge argv/env、记录映射、HEAD、tentative与逐位/来源判据不变。S1和S3两项GPU训练仍依次单跑；S2仍等待两组upstream和四run验收均通过后才于22:50:53/54按4+4起跑。独立dispatcher的实际SHA分别为full `4331d4871e538c5aaaa5c4a391b2aa449b8fccfb0d4cd772c9b3da76734520d8`、count `93a4af95997bccad255855daaf9e7e3120a933ea577eb645c4f2f496c5ae031f`，由控制记录的eq100_independent_upstream_judges登记。

## 11. 当前结论与下一步

已完成六run完整性、两组upstream与两组same-entry的正式及独立验收：每份judge均覆盖500个标量比较和50/99两状态点，S2重叠满足各侧至少50完整间隔。六run及四judge均有原生退出和外部capture链。本结论只覆盖本机确定性档前100步，不作吞吐或80k收敛结论。

正式实测资料已归档，原始运行版本仍保留00bd。随后须按新的clean Beta推进两库各一对off/on真保存20步及完整输入/状态/恢复判定；再执行4+4的300步perf、0.5秒GPU/磁盘采样、真实299保存/恢复和清理前measurement receipt。三类margin与正式预算取真实测量，不从本100步推算。本轮持续工作至正式80k起跑前，不自动启动80k。

## 12. 归档文件与来源清单

实测资料已按明确白名单导入records/train100和logs/train100；十个任务的身份、原生退出和capture侧证位于records/train100/exit。以下四个原始清单分别保留原字节核心包、成员检查、原始来源、清洗/git-ready日志、报告以及独立退出引用：

| 原始清单 | SHA256 | 已覆盖范围 |
|---|---|---|
| [两A pre-judge基线](records/train100/eq100-a-archive-references.json) | `e4c7fa4b759e025ea909dfa7738e95886222827f659b2f93eb4f830d0f8fbc05` | 两A各13核心文件、tent12/main100、state50/99、退出证据；保留其当时尚未judge的快照 |
| [full upstream增量](records/train100/full-upstream-increment-20260926/eq100-full-upstream-archive-increment.json) | `1c023563d384342b085f794900783501d93ee278def78e1f39c4d98eee1f1a67` | S1 12核心文件、full judge、两份TSV及原生退出；旧15文件未改 |
| [count upstream增量](records/train100/count-upstream-increment-20260926/eq100-count-upstream-archive-increment.json) | `32fa8fd5f5ad323ea6bd986be9018c92514fe1daced11e46d1b88f55d145ed16` | S3 12核心文件、count judge、两份TSV及原生退出；此前33文件未改 |
| [same-entry收尾增量](records/train100/same-entry-increment-20260926/eq100-same-entry-archive-increment.json) | `02979af73418881cd14b422a10fc309b7e75b3340c80c91dcf5a37591bdf5321` | 两S2各12核心文件、两same judge、四份TSV、四条原生退出链及controller固定快照；旧51产物未改 |

run完整性原始/独立报告已随清单原字节保存：[两A根报告](records/train100/eq100-both-a.root-verification.json)与[两A独立报告](records/train100/eq100-both-a.independent-verification.json)、[S1根报告](records/train100/full-upstream-increment-20260926/full-s1.root-verification.json)与[S1独立报告](records/train100/full-upstream-increment-20260926/full-s1.independent-verification.json)、[S3根报告](records/train100/count-upstream-increment-20260926/count-s3.root-verification.json)与[S3独立报告](records/train100/count-upstream-increment-20260926/count-s3.independent-verification.json)。两组正式判定报告为[full根报告](records/train100/full-upstream-increment-20260926/full-upstream-judge.root-verification.json)、[full独立报告](records/train100/full-upstream-increment-20260926/full-upstream-judge.independent-verification.json)、[count根报告](records/train100/count-upstream-increment-20260926/count-upstream-judge.root-verification.json)、[count独立报告](records/train100/count-upstream-increment-20260926/count-upstream-judge.independent-verification.json)。

S2和same-entry报告已另存：[S2根报告](records/train100/same-entry-increment-20260926/both-s2.root-verification.json)、[S2独立报告](records/train100/same-entry-increment-20260926/both-s2.independent-verification.json)、[full same根报告](records/train100/same-entry-increment-20260926/full-same-judge.root-verification.json)、[count same根报告](records/train100/same-entry-increment-20260926/count-same-judge.root-verification.json)、[same合并独立报告](records/train100/same-entry-increment-20260926/both-same-judge.independent-verification.json)和[六run根总报告](records/train100/same-entry-increment-20260926/six-runs.root-verification.json)。same judge按既有实现重写了两solo原TSV，SHA/字节未变、mtime/ctime更新；旧归档的source_stat描述当次归档前后稳定，旧包和检查自身保持不变。

新off/on20、perf、实际预算和80k仍须各自前置与实测结果。此次导入保留原始与清洗SHA区别；没有复制权重、cache、独立.sh/.yaml/.command或准备载体源码。完整源→目标关系见[白名单](records/post-eq100-import-whitelist.json)，复制结果见导入回执。
