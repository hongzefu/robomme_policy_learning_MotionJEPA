# 两库确定性档20步包装对照（已完成）

实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`，实际起跑Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`（`commitV11.16Beta`）。两库确定性档off/on正式判定均通过；完整实测与证据导读见[result](result.md)。

## 1. 结论与指标速览

两库各off/on一对20步真保存对照均通过：每库21次fetch、20次模型输入/RNG、100对标量和201个完整TrainState叶逐位相等；四run全部实际保存19/state20并恢复61叶EMA。确定性档仅用于包装正确性闸门，不构成normal环境数值不变、性能或模型质量结论。

## 2. 版本与代码状态

四run和两judge均从同一clean `c71d5255597db2f26930b7b5684a1d5b2994cf75`启动。实际运行与独立验收期间冻结tracked及两venv，2026-09-27 02:07:11 UTC验收完成后才解除；原训练/model源码、超参、依赖均未改。本轮先前批准的工具改动集中于runner/contract/speed/observer的显式档位守卫与对应测试，377项核心合测通过。

INPUT源HEAD为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；[P1](../orig80k-equiv-0925/p1-result.md)及[100步](../orig80k-equiv-0925/train100-result.md)的量具与B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。原normal20锚 `b0efbde61e38411fb1b9114eec8485d36a4ee9a0` 与追加off锚 `b60ec2b59e0ba012cae0998c5f8713b2aef07b2e` 保持原样；新schema2不提供缺profile旧记录自动normal的豁免。

## 3. 启动与配置还原

[launch完整正文](launch.md)锁定生成器、父适配、联合验收及六任务CLI。准备TAG `20260927T012426Z` 不是实际起跑时间。四run分别为：

| 库 | off | on | GPU |
|---|---|---|---|
| full | [full-off](../timing20-full-off-0926-20260927T012426Z/README.md) | [full-on](../timing20-full-on-0926-20260927T012426Z/README.md) | 0,1,2,3 |
| count | [count-off](../timing20-count-off-0926-20260927T012426Z/README.md) | [count-on](../timing20-count-on-0926-20260927T012426Z/README.md) | 4,5,6,7 |

固定参数由printf %q写成runtime第一行，run第二行显式export selector，judge第二行unset selector；其后原模板完整字节保持。COMMON原45b4实体不加前缀。正式Git源=工作树=runtime绑定，f8运行副本仍用v1-store既有实体；不能把旧模板SHA冒充加前缀后整份命令SHA。

## 4. 数据集与划分口径

复用已验收full1600与count400库的4×4 packed和原manifest，不缩数据量或新划分。full使用原norm `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`；count使用授权重算norm `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`，路径见launch。新Beta起跑前核数据/资产/依赖指纹及旧INPUT沿用边界，不把历史记录当作新HEAD重新运行结果。

## 5. 关键超参

steps20、b64、fsdp4、workers4、seed42；history modulation/budget512/4×4/最多32 memory帧，streaming_obs_horizon16；warmup10000、peak/decay_lr均5e-5、decay_steps100000、AdamW clip1.0、EMA0.999；log100、save/keep10000，pi05_base初始化、W&B online不变。

本阶段新增显式 `ORIG80K_TIMING_EQ_PROFILE=deterministic100`；实际flags精确为 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`。normal perf/prod不得继承此selector；这两flags不是全局默认。继承x64开启仍须fail loud，不用覆写来绕过。

## 6. 硬件、调度与耗时

实测环境AWS8×A10080、本地md0 XFS，两库各4卡独立off→on。本库off完成真实恢复与独立退出后启动本库on，再独立CPU judge。两off均于01:42:29 UTC开始，count/full分别于01:52:53/01:53:35结束；两on于01:53:14/01:53:39开始，分别于02:03:52/02:04:25结束。两judge于02:04:29/02:04:39结束，完整身份见四run档案及result。20步不作吞吐或ETA。

## 7. 训练过程行为

off共同观测原train入口，on共同观测speed.run(smoke20)，两侧都保留真实save_state及wait。log100 step0加尾窗1…19覆盖全部标量，完整末态params/EMA/optimizer/step按结构、dtype/shape/原始SHA严格比较。on保留原HostTiming/同步点与0.5秒只读磁盘采样；采样峰值不冒称连续真实峰值，不能据20步推正常perf性能。

## 8. 保存、恢复与正式判定

四run的checkpoint19/state20真实EMA恢复、完整来源/UUID/HEAD及两on采样证据均通过。两正式judge满足原严格输入/RNG/100标量/完整状态逐位判据；额外联合守卫确认launch、metadata及judge结果均显式deterministic100，on两speed JSON均为schema2且实际flags一致。任意normal PASS不能满足本阶段。

六任务的f8原生pane退出均0且无信号，capture父真实返回0；七段根调度及其宿主实际返回均0，原日志内容联合通过。六份独立只读验收均通过，终态controller快照锁定七段过程与六任务身份；仅文本0或session消失不构成本结论。

## 9. 用户决定记录

用户选择「两库各跑一对 20 步（推荐）」，要求「尽可能并行做」「你有8张卡」，并已批准本阶段使用既有100步确定性档，逐位判据不变，perf/prod仍normal。用户「继续工作 一路做到起泡前 有问题问用户」将本轮终点限定为正式80k起跑前。原CPU主机dtype数值等价及四None键授权不放宽训练逐位判据；P1/100 no-W&B不延伸到本阶段。

## 10. 计划外事件与历史限制

旧normal结果分别见[off/on失败](../orig80k-timing20-0926/result.md)及[off复验](../orig80k-off-repeat-0927/result.md)。本阶段不改其原FAIL，不声称已定位根因或确定性flags必能解决。

旧INPUT/P1最外final append实际退出不可追补，子命令0、manifest/输入/标量/状态内容证据仍保留。用户分别批准「同样补记限制，沿用INPUT结果（推荐）」和「补记P1限制，补独立退出取证后继续（推荐）」。此次六任务从起跑前就使用强独立退出链，不能补造历史wait。

独立报告首轮外封装经JavaScript解析后舍入了部分纳秒大整数；原训练记录、判定器stdout及真实PASS未变。保留四份首版失败封装，正式归档仅使用原stdout字节直接嵌入的v2，并经Python逐值复核。事件说明和校验限制随result归档；未因此重跑训练或judge。

## 11. 当前结论与下一步

确定性档包装正确性闸门已完成。下一步执行normal300步perf、据实完成磁盘预算及正式起跑前条件；本轮仍止于正式80k起跑前。旧normal失败不改写，确定性档的通过不能推导为normal包装数值不变或已解释原差异根因。

## 12. 归档文件清单

本README、[launch](launch.md)和四run README共同锁定启动口径；[候选原验证](records/validation.json)、[on两speed验收补充](records/validation-on-speed-addendum.json)和[9项shell绑定实测](records/runtime-profile-bindings.results.tsv)只证明控制文本和绑定。实际训练证据、白名单及事件记录见[result](result.md)。六任务runtime/identity/native receipt/父capture保留在`v1-store/bench/orig80k-timing20-commands-20260927T012426Z`，两judge结果位于独立`v1-store/bench/orig80k-timing20-results-20260927T012426Z`。

仅归档Git不能还原的小JSON/JSONL、清洗日志、检查链和退出回执；权重留run根，cache/runtime留v1-store，不复制.sh/.yaml入docs。原记录及六个已结束tmux窗口均保留，不以清理产物换取通过。
