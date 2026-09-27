# 两库测速包装20步结果：保存恢复完成，严格等价失败

## 1. 结论与指标速览

**full与counting的正式off/on判定均为FAIL，300步perf不放行。** 四次20步训练均完成真实保存、CPU恢复与独立原生退出0；这与包装是否保持训练逐位一致是不同判据。最终独立核验见[完整比较报告](records/verification/final-comparison.independent.json)，其SHA256为`113cec3b17a19e85bd55c7a2b4420f150eac17dbea69a2b3c3e6c38d62265e99`。

| 每库比较项 | full | counting |
|---|---:|---:|
| 21次取批树不等数 | 0 | 0 |
| 20次模型输入与RNG不等数 | 0 | 0 |
| 五标量不等项 / 总项 | 80 / 100 | 80 / 100 |
| 末态不等叶 / 总叶 | 147 / 201 | 147 / 201 |
| 正式judge / 原生pane退出码 | 1 / 1 | 1 / 1 |

以上状态差异按叶摘要计数，不是不同浮点元素个数。原因尚未确定；未通过的原判据和原始记录均保留。用户已批准追加两库各一次相同配置的off诊断复验，不能以该诊断替代本次正式闸门。

## 2. 版本与代码状态

实际运行锚点为`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`（`commitV11.14Beta`）。四run和两judge均从该clean版本运行，源码与uv环境冻结至最终独立核验完成。结果提交与新诊断Beta不回填为本次运行版本。

已有CPU INPUT保留`3a1582db39c723c735e04752e5027bfe40ecc3e1`，P1/100量具及B保留`00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A保留`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。本阶段起跑前的[证据沿用预检](records/diagnostics/timing20-reuse-preflight-20260927.json)实查113项输入引用、源码、两侧解释器/208项依赖、数据资产小记录和现场资源；[父进程回执](records/diagnostics/timing20-reuse-preflight-process-20260927.json)记录实际返回0。它不声称新HEAD重跑过旧INPUT/100。

远端仅三规则文件的更新由`40de98a1b3b60072e80bcf82dc4fbdf64d329b6c`普通双父合并并推送。只读预检新增三文件固定`d710d8489b88aa75770af3452fd7c9b374deaef7`内容登记，原训练保护、六工具SHA和113项引用不放宽；[验证](records/diagnostics/timing20-reuse-rules-20260927-validation.json)与[差异清单](records/diagnostics/timing20-reuse-rules-20260927-delta.json)分别留证。

## 3. 启动与配置还原

准备标签为`20260926T231534Z`，实际执行在2026-09-27 UTC。四个run完整名称为`timing20-{full,count}-{off,on}-0926-20260926T231534Z`；训练会话在run前加`orig80k-`，两个judge会话为`orig80k-timing20-{full,count}-judge-20260926T231534Z`。各自pane、进程、UUID、W&B ID及运行命令见[四run索引](README.md#1-结论与指标速览)。

```bash
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:docs/training-doc/orig80k-timing20-0926/launch.md
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:scripts/training/tests/timing20-wrapper.sh
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:scripts/training/tests/timing20-judge-wrapper.sh
git show b0efbde61e38411fb1b9114eec8485d36a4ee9a0:scripts/training/prod/run_orig80k.sh
```

七个父调度载体按Beta中对应代码块提取，TRAIN_HEAD展开为实际b0，见[调度来源清单](records/diagnostics/timing20-dispatch-manifest-20260927.json)。训练wrapper仍逐项调用`run_orig80k.sh smoke <run> <gpus> <lib> <assets>`，仅用`ORIG80K_SMOKE_EQ_MODE=off|on`选择计时包装；真实保存、恢复与W&B保持开启。guard先保留窗口和持久化identity，再释放命令；每次动作绑定Git源、工作树和原f8 runtime。现有输出不得复用或覆盖。

## 4. 数据集与划分口径

full为公开16任务各100集，共1600集、768897帧、476857执行样本；counting为同一公开集合的BinFill、PickXtimes、StopCube、SwingXtimes各100集，共400集、189035帧及执行样本。本阶段不重建、裁剪或划分数据，均消费各库4×4 packed，并绑定同库source/episode manifest。

full使用`v1-store/datasets/16task-pub-1600ep`及原版norm SHA`f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`；counting使用`4task-counting-pub-400ep`及自身norm SHA`a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。history SHA为`823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a`。资产、tokenizer与初始化参数沿用原本机副本，未重新下载。

## 5. 关键超参

四run均为`mme_vla_suite`、global batch64、fsdp4、workers4、seed42，modul budget512、4×4、最多32帧，motion关闭。smoke仅把训练步数覆盖为20；log100/save10000/keep10000、warmup10000、peak/decay lr均5e-5、decay_steps100000、EMA0.999保持原值。W&B online，实际x64=False，正常计算环境没有P1/100确定性档的XLA_FLAGS。

两侧安装相同只读取证层，记录取批、模型实参/RNG、五标量及完整末态；on另安装既定计时与0.5秒主机/磁盘观测层。没有为获得相等而改变dtype、确定性设置、种子或误差阈值。

## 6. 硬件、调度与耗时

AWS单机8×A100-SXM4-80GB，`/scratch`为`/dev/md0` XFS本地NVMe RAID。full固定GPU0–3，counting固定GPU4–7；两off同启，各库通过恢复及原生退出后立即进入本库on，不等待另一库无关阶段。

| run | 开始UTC | 结束UTC | 包装墙钟秒 | W&B ID |
|---|---|---|---:|---|
| full-off | 00:09:50 | 00:21:15 | 685 | `ewcj0x4h` |
| count-off | 00:09:50 | 00:20:35 | 645 | `b0frcaff` |
| full-on | 00:21:48 | 00:32:55 | 667 | `jj211w0u` |
| count-on | 00:20:57 | 00:31:59 | 662 | `7mcrk704` |

这些时间来自各自日志起止，包含初始化/JIT、device_get/哈希、训练、真实保存、W&B收尾和CPU恢复，不能用作吞吐优劣或80k ETA。count/full judge分别于00:33:10—00:33:11、00:33:33—00:33:34运行。主机时间记录不冒充GPU内核耗时。

## 7. 训练过程行为与真实恢复

每run均完成21次取批、20次模型调用、一次真实save，state.step=20、loop/checkpoint=19。普通metrics的step0与final尾窗1…19组成完整20步五标量；所有取证叶及标量有限。完整末态共201叶：params61、EMA61、opt78、step1。**真实checkpoint恢复的是61个EMA叶，不把201叶摘要误写成201叶均从checkpoint恢复。**

四个训练wrapper均原生退出0，capture实际0；完成器确实加载各自`19/params`，dtype/shape/字节摘要与该run末步EMA一致。on的采样和原生checkpoint commit绑定已核验；采样最大分配占用只是离散观测，不是连续峰值。单run联合核验见[两off](records/verification/both-off.independent-verification.json)与[两on](records/verification/both-on.independent-verification.json)。

## 8. 正式判定与完整差异

两份正式判定器实际均输出：

```text
TIMING_EQ_JUDGE=FAIL reason=20步五标量不是逐位一致
JUDGE_EXIT_CODE=1
WRAPPER_COMMAND_EXIT=1
WRAPPER_EXIT_CODE=1
```

两个pane均`dead=1/status=1/signal为空`，f8记录`receipt.status=FAIL`，capture进程真实返回1；既定finish调度在原生退出闸门返回2。正文tee及footer printf/tee返回0。两个成功结果JSON没有生成，未补造PASS或空成功文件。读取失败证据的独立程序返回0只说明取证完成，不改变实验FAIL。

独立比较进一步核对全部记录：每库21个fetch.tree全同、20个model.tree与RNG全同；loss、grad_norm、llm_grad_norm、mem_enc_norm各20项不等，param_norm的20项相等。状态treedef与叶键集合相同，params37/61、EMA36/61、opt74/78叶摘要不同，step的1叶相同，总计147/201。正式judge先在标量处拒绝，状态全量差异是独立后验分析，不冒称原judge已执行其后的状态比较。

## 9. 用户决定记录

沿用用户原话「两库各跑一对 20 步（推荐）」「尽可能并行做」「你有8张卡」，本次确实按两条独立4卡链执行。「允许仅对拍关闭 W&B」仅限P1/100，本次四run保持在线记录。「继续工作 一路做到起泡前 有问题问用户」继续限定本轮终点为正式80k起跑前。

发现首步差异后立即询问：「是否同意在当前取证结束后，两库各追加一次相同配置的 20 步 off 复验（4+4 并行），保持严格判据，用于判断未开启计时包装时是否也存在运行间差异？该复验只用于诊断，不替代正式对拍闸门。」用户答复「同意」。该决定新增诊断运行，不授权放宽当前失败的逐位判据，也不授权启动perf或80k。

## 10. 计划外事件与原因边界

首次结果push的远端非快进拒绝已按用户决定以普通merge解决；三规则文件合入没有改变训练源码或依赖，详见[group README](README.md#10-历史限制与新阶段处置)。本阶段起跑、所有旧运行版本和原始SHA保持实际值。

独立核验首次用uv入口调用，`sys.executable`字面为python3，与Beta父适配实际python不符，因而被argv守卫拒绝；随后按已锁定的原解释器重核通过。原失败调用、修正和实际返回码均保留，不把这次核验调用错误算作训练失败，也没有改回执或expect。

首步差异的[早期证据](records/diagnostics/timing20-step0-early-mismatch-independent-20260927.json)只绑定各原文件第一完整行含LF的字节和SHA，不把在写文件的当时摘要冒称最终整文件摘要。[只读机理记录](records/diagnostics/timing20-step0-mechanism-note-20260927.md)没有发现可证明的首步数学参数改动，也没有证明XLA算法选择或包装时序就是原因。完整初始state没有被摘要，param_norm又只聚合特定kernel，因此相同norm不能证明完整初态逐位相同。

## 11. 当前结论与下一步

本次四run保存恢复正常，严格off/on等价失败。INPUT及确定性档100步的原有结论各保留其真实范围，不能用它们覆盖本次正常计算环境下的失败。300步perf、正式磁盘预算和80k继续受闸门约束。

下一步按用户新授权，用两个全新run`timing20-full-off-0926-20260927T003051Z`与`timing20-count-off-0926-20260927T003051Z`各复验20步off，4+4并行。新TAG仍只是准备标签；新Beta仅改文档时，分别保留旧b0和新运行HEAD并核对受保护代码、依赖、数据与配置，再进行独立诊断比较。原正式judge的同HEAD规则不改，诊断正常读完与是否逐位一致分别报告。

## 12. 归档文件清单

[正式导入白名单](records/formal-import-whitelist.json)列出81个原字节目标，共1665572 B，另有起跑前置、规则登记、首步证据、机理说明和调度来源7份必要小记录；[导入回执](records/import-receipt.json)区分`archive_integrity=PASS`与`timing_eq_verdict=FAIL`。四run各自的`records/core.records.tar.gz`与`archive_checks.json`保留原JSON/JSONL/CSV及两个checkpoint小元数据，权重不入Git。公共`records/exit/`保存六任务身份、原生回执、launch/capture父侧证及其原始输出。

七个父调度阶段的process/stdout/stderr及一份完成后控制快照共22个补充原件位于`records/dispatch/`，[补充回执](records/dispatch/import-receipt.json)逐项列来源与摘要，并单列宿主实际观察的父返回值。两个judge-finish按原守卫返回2，其余五阶段返回0；这些阶段退出与六个tmux任务退出分别记录。动态控制文件、脚本载体未复制。

四run各自driver/wrapper/completion清洗日志与检查链位于各run的`logs/`。原日志不改，只去明确进度行；对精确匹配`^wandb:[ \t]+$`的纯装饰行在Git副本删尾白，保留`wandb:`并逐行登记变换、删除字节与raw/clean/git摘要。所有SHA引用中的原始绝对路径不重写。源码、运行时command、缓存和大checkpoint不复制；两份失败judge保留日志及真实非零链，不制造成功结果文件。
