# 两库确定性档20步包装对照结果

## 1. 结论与指标速览

两库各一对20步off/on严格逐位通过。每组21次fetch树、20次模型输入与RNG相同，100个标量对零差异，末态201叶完整TrainState零差异；四run各自checkpoint19/state20真实CPU恢复61个EMA叶通过。六任务原生退出/capture为0，七调度阶段父/宿主实际返回为0。

**结论仅限本次deterministic100档与共同观察层。** 旧normal off/on及off重复诊断仍FAIL；本结果不证明normal数值不变、一般确定性、吞吐或模型质量。normal300步perf、正式预算及80k起跑前条件尚未完成。

## 2. 版本与代码状态

真实Beta `c71d5255597db2f26930b7b5684a1d5b2994cf75`（commitV11.16Beta），实施提交 `aec86db64e5178e63d9e7f77d3f5bc235db16390`。运行期间源码/两venv冻结，2026-09-27T02:07:11Z生成最终快照后解除；记录不改写为随后归档提交。实施阶段377项核心测试是工具验证，与这里的GPU实跑证据分别保留。

原INPUT的3a1582d、P1/100的00bd/A ecf、normal20的b0ef及追加off的b60各自保留旧版本。新schema2不向缺profile的旧记录提供自动normal兼容。

## 3. 启动与配置还原

[固定启动档](launch.md)锁定四run、两judge、原45b4/17c0模板、f8门闩与两行参数/profile前缀。TAG `20260927T012426Z`只是准备标签；真正起点为2026-09-27T01:42:29Z。新preflight在创建命令根之前执行，报告SHA `a69075ffbba2f92e89eee34cecf10cc5e428b14e50f4f210b3d8bcfe1ed8c345`，实际子/父/宿主均0；见[原报告](records/preflight/timing20-det-actual-preflight-20260927.json)及[父记录](records/preflight/timing20-det-actual-preflight-20260927.process.json)。它校验实施→Beta仅docs、原113输入引用、2venv/208依赖、库/小资产、GPU与输出冲突，不冒称重跑旧INPUT/100。

两个off背靠背释放，各库off恢复与独立退出通过后立即启动本库on；各自on结束后及时启动CPU judge，不等待另一库。真实运行ARGV与完整命令SHA在launch/identity及父记录中绑定，代码由Beta还原，不复制.sh/.yaml。

## 4. 数据集与划分口径

full为1600集、768897帧、476857执行样本；count为400集、189035帧及执行样本。两库沿已验收4×4 packed、source/manifest，无新划分或缩样本。full原norm为`f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`，count授权自算norm为`a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。

## 5. 关键超参

20步、b64、fsdp4、workers4、seed42；modul512/4×4/max32 memory帧；warmup10000、peak/decay_lr5e-5、decay_steps100000、EMA0.999；log100/save10000/keep10000；pi05_base初始化，W&B在线。明确profile为`deterministic100`，实际flags精确为`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`。flags仅用于本包装对拍，normal perf/prod不继承；未改训练config身份白名单或掩盖x64。

## 6. 硬件、调度与耗时

AWS8×A100-SXM4-80GB，/scratch md0 XFS本地NVMe；full0–3/count4–7。四份新checkpoint的起跑前实存估计为47,537,762,304 B；当时可用779,734,368,256 B，扣估计后732,196,605,952 B，约681.911GiB，高于既定300GiB保护。此[阶段估计](records/preflight/timing20-det-space-preflight-20260927.json)不是连续峰值上界或正式80k预算。

| 任务 | 开始→结束UTC（2026-09-27） | 身份/结果 |
|---|---|---|
| full-off | 01:42:29→01:53:35 | 4a290d47-fbed-4d6d-8d62-4fc01a1c9805；完成/恢复/退出PASS |
| count-off | 01:42:29→01:52:53 | 7ac2fc1b-e087-4815-b4f0-2e0630f30d99；完成/恢复/退出PASS |
| full-on | 01:53:39→02:04:25 | 4f142027-87fa-45ea-9462-ae6e4f7e73f6；完成/恢复/退出PASS |
| count-on | 01:53:14→02:03:52 | 1d572c9d-8b26-4f48-9f3e-0926c3f8308a；完成/恢复/退出PASS |
| full judge | 02:04:39→02:04:39 | 严格逐位PASS；同秒时间戳不解释为耗时0 |
| count judge | 02:04:28→02:04:29 | 严格逐位PASS |

各run完整窗口含取证、真实保存、CPU恢复及W&B收尾；不用于吞吐/ETA。

## 7. 训练过程行为

四run均21次取批、20次真正模型调用及更新，最后一批只预取不用；一次真实save_state返回，末态step20/loop19。step0原metrics加final尾窗1…19组成20×5精确标量，未改log_interval。全部标量和状态有限，52项目模块/208依赖等来源核验通过。

off直调原入口且没有speed五文件；on经过speed.run(smoke)，起止JSON均schema2并显式记录相同profile/flags。主机、GPU和0.5秒磁盘采样原件均保留；实际间隔/漏tick和保存窗口见speed记录，离散峰值仍为下界。本轮共同device_get/CPU摘要有同步和耗时开销，不能替代无观测normal perf。

## 8. 保存、恢复与正式判定

四份checkpoint19均实际恢复61个EMA叶，与训练final和完整状态中的EMA一致；61叶为23个bf16和38个f32。恢复证明范围与完整TrainState严格比较分开，未声称从权重恢复全部optimizer。

| 库 | fetch/model+RNG | 标量差异 | 完整末态差异 | 正式结果 |
|---|---|---:|---:|---|
| full | 21/20全同 | 0/100 | 0/201 | [full.json](records/judges/full.json) PASS |
| count | 21/20全同 | 0/100 | 0/201 | [count.json](records/judges/count.json) PASS |

201叶覆盖params61、EMA61、optimizer78、step1，连同treedef/keyset/dtype/shape/raw SHA核验。六个原生pane明确status0且无信号，capture父actual_exit_code0；[最终controller快照](records/controller.completed.snapshot.json)绑定七phase的父实际返回与宿主实际0。日志0单独不承担最外成功证明。

## 9. 用户决定记录

用户选择「两库各跑一对 20 步（推荐）」，要求「尽可能并行做」「你有8张卡」，并明确批准将包装正确性闸门切到两库各一对确定性20，input/scalar/fullstate仍逐位，perf/prod仍normal，旧FAIL保留。原「继续工作 一路做到起泡前 有问题问用户」限定本轮不启动正式80k。P1/100关闭W&B的授权不延伸，本阶段在线并真保存。

## 10. 计划外事件与历史限制

四份新独立报告初版经JavaScript JSON.parse重封装时，各有两个纳秒大整数被舍入；二次精确校验出现AssertionError。合并shell中随后Git检查的总体0不作为该失败校验成功依据。原训练记录、原checker stdout及真实退出未改，四首版留在ignored原处、不导入。修正v2直接嵌入原stdout字节，用Python精确整数重核output、judge嵌入与源引用通过；没有重跑训练、恢复或正式judge。完整[事件说明](records/verification/report-serialization-incident.json)保留失败输出及边界。

off归档首准备版复用旧helper的日志scope文字带有历史FAIL，另存v2纠正；核心tar原字节/SHA完全相同，首准备版不导入。所有原日志保留；清洗只删除已识别进度行，Git副本只处理获批纯W&B装饰尾白，其他行与EOF保真，不为美化diffcheck改原记录。

旧[normal off/on](../orig80k-timing20-0926/result.md)与[off重复](../orig80k-off-repeat-0927/result.md)各库80/100标量、147/201状态叶差异保持FAIL；本次确定性通过不确定具体根因。旧INPUT/P1最外退出限制及用户沿用批准保持原样，不追补历史wait。

## 11. 结论与下一步

确定性档的本次两库20步包装正确性闸门完成。该结论不推广normal档、一般确定性或性能；下一步须从对应clean版本执行normal300步perf、真实保存恢复和资源测量，再形成显式三项margin及完整磁盘预算供确认。正式80k仍停在起跑前，不因本阶段PASS自动启动。

## 12. 归档文件清单

本批明确白名单115项，共1,987,125 B（不含导入白名单/回执自身）。四run分别保留17/22成员原字节核心包、12份run清洗日志及检查；group保留两正式judge结果/日志、6×8原生侧证、七phase父过程/宿主证据、前置、最终快照与六份独立报告。off原阶段清单记录当时PENDING，保持不改；本[最终增量清单](records/final-increment-index.json)与正式judge记全阶段det PASS。

独立报告为[full-off](records/verification/full-off.independent-verification.json)、[count-off](records/verification/count-off.independent-verification.json)、[full-on v2](records/verification/full-on.independent-verification-v2.json)、[count-on v2](records/verification/count-on.independent-verification-v2.json)、[full judge v2](records/verification/full-judge.independent-verification-v2.json)、[count judge v2](records/verification/count-judge.independent-verification-v2.json)。原报告实际路径不改写，整数不经JavaScript重序列化；权重/cache/runtime留v1-store，源码/配置由Beta还原，不复制.sh/.yaml或旧normal嵌套档案。
