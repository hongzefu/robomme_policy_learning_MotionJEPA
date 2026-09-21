# m2048-dump-base-32x16 结果

本档改前证据完整，尚待同配置CAND对拍，不能单凭本档成功宣称改动等价或2048功能通过。

用户原话：「开始实现 有问题立刻越早问用户越好 一路实现到测速结束报告用户为止」；对较长摘要成本明确选择「保持原计划，完整取证（推荐）」。3000个定点样本、200个batch完整导出；全量身份覆盖1600集、605611个执行样本、1192918帧。两个旧档身份摘要相同；采样摘要随8/32帧预算而不同，符合配置差异。

起跑代码为 `29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6`，生产src与初始c31b0509be3f76a6cc262fd655f5f0080b12c266相同；运行期间HEAD与源码保持不变。起点2026-09-21T00:59:21Z，终点2026-09-21T01:10:33.192587+00:00，终点取日志mtime近似，墙钟约672.2秒。仅CPU，未占用GPU。环境B：8×A100-SXM4-80GB，/dev/md0 XFS本地NVMe RAID。训练摘要每次约183–185秒，是验证开销，不用于估计正式训练速度。

仅使用1600ep库及其source/framesamp/framesamp-8x8，norm_stats文件SHA为856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173；实际导入路径、配置、数组摘要、输入索引和退出记录见[运行记录](records/launch.actual.json)。真实样本均满历史，allshort/mixed1组名不构成短历史覆盖。原始数据、norm_stats、权重与依赖均未修改。

计划外事件：首次启动漏传CLI必填--exp-name，在读取数据前以2退出，失败日志保留。补齐该参数后从同一clean BASE重新启动，最终EXIT_CODE=0；修正命令及载体归档时SHA在launch.actual.json中注明，不伪称该SHA已进入起跑日志。

验证使用真实工具输出与逐项检查：DUMP_DONE及EXIT_CODE=0；DUMP_MANIFEST逐文件复验，gzip解压后的字节SHA与原始摘要文件相同。这些开发期检查将在已提交CAND检查器上再次用于正式双侧比较。

归档仅包含不可由Git还原的运行JSON、压缩JSONL/TSV及清洗日志；压缩前后SHA见[归档清单](records/archive_manifest.json)。summary文件解压后可恢复samples/summary.jsonl和batches/summary.jsonl的原始字节。下一步为候选同配置逐位对拍。
