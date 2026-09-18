# V-online三档通过：全量装配与次序一致，3139窗真实编码逐位一致

用户确认方案B，并要求「采用预生成并校验的 GPU 位置表，CPU 验证装配与次序」。三档全部从clean `782696c231aace21c20200638ae302a5a1c7f277` 启动，期间零提交、零版本化文件修改，起止源码检查均通过。分片规则为g%8，八片episode和窗口无重复、无遗漏。环境为AWS本地NVMe RAID `/dev/md0` XFS、8×A100-SXM4-80GB，使用现有uv环境、`--no-sync`，具体命令见 [launch.md](launch.md)。

## 覆盖与实测结果

| 档位 | UTC起止（2026-09-18） | 挂钟 | 核验结果 |
|---|---|---|---|
| assembly，八CPU分片 | 04:28:28→04:31:56 | 208秒 | 全71316窗的33帧原始字节SHA相同，1600补帧窗集合完整 |
| stub，八CPU分片及stub sidecar | 04:32:05→04:43:03 | 658秒 | 全1600集起点、motion_pos/mask/mem_order逐位相同 |
| real，八CPU主进程及GPU0–7 sidecar | 04:43:49→05:06:18 | 1349秒 | 全1600补帧窗及23必含整集，共3139窗真实重编逐位相同；其余行注入离线表 |

每档八个SHARD_EXIT_CODE及总EXIT_CODE均为0，所有模式的mismatch_total均为0。CPU位置表缓存SHA为 `9069eb6e5349241fa5a5a4d2ac7369066aab2e83bbdd7ca0e09e615b88ad415b`；八片均复核缓存内容、组件源码和离线表前缀身份，严格字节判据保留。该缓存只在验证中替换已验证的组件输出，生产策略没有因此读取MotionStore。

最终判定：

```text
ONLINE_INPUT_SHA=PASS windows=71316 padded=1600 mismatches=0
ONLINE_START_SET=PASS
ONLINE_POS=PASS
ONLINE_ORDER=PASS
ONLINE_STRATA=PASS padded=1600/1600 kmax_ep=367 es_ge1000=4 pad_r=16/16 per_task_min=1
SHARD_SET=PASS shards=8 eps=1600 disjoint=1
P5_ONLINE=PASS episodes=1600 windows=71316 assembly_checked=71316 compared_real=3139 padded_covered=1600 stub=False
```

## 真实编码口径

必含整集的g为 `[0,1,2,3,4,5,7,9,11,16,17,19,28,30,31,55,268,273,281,367,400,800,1200]`，含最大k的g367、4条es≥1000、全部16种(es−17)%16余数和四任务。23个整集共1562窗，其中23窗也属于补帧集合；所以真实编码量为1600+1562−23=3139，实际sidecar调用数与该去重集合一致。逐片真实窗数为409/638/288/340/442/282/232/508，均逐位通过；负载差异与挂钟如实保留，不缩减覆盖。

承诺口径落实为：「在线侧 33 帧装配对全部 71,316 窗与离线逐位一致；起点集合 / 时间码 / 交错次序对全部 1,600 集逐位一致；真编码器 token 对全部 1,600 个补帧窗与23个整集共3,139窗逐位一致，其余行由离线表注入、不计入 token 判据。」全量71316行encoder数值检查另由建库D3完成，此处的真在线重编范围不混称全库。

Wan VAE使用fp32，encoder依epoch72配置使用bf16 autocast，最终token为float32。原始环境与数组提示保留在清洗日志，钉版数值入口未改。验证成功不代表策略成功率提升，正式策略评估及同权重motion全遮消融仍按另起计划执行。

## 归档

结构化汇总为 [assembly.aggregate.json](records/assembly.aggregate.json)、[stub.aggregate.json](records/stub.aggregate.json)、[real.aggregate.json](records/real.aggregate.json)，每档8份分片报告在同目录；汇总器只消费结构化字段并独立重算集合与分层条件。清洗日志为 [assembly.summary.log](records/assembly.summary.log)、[stub.summary.log](records/stub.summary.log)、[real.summary.log](records/real.summary.log)，全部24个分片退出码与三个总退出码均保留。前置数据和数值证明见 [建库档案](../../dataset-build-doc/4task-v2-1600ep-motion-demopad17/result.md)。
