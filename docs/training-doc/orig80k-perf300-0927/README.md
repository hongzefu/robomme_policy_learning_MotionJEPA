# 两库normal300测速与正式磁盘预算（已完成）

两库normal300、真实CPU恢复、联合报告、两份checkpoint测量及用户选定B档预算全部通过，完整证据见[result](result.md)。实际Beta为 `70a641a827eb2559c3acb69430f55c1c53babaeb`（commitV11.17Beta）。[共享launch](launch.md)保留启动时控制正文，两份十二节档案分别为[full](../perf-orig80k-full-300-20260926T232339Z/README.md)、[count](../perf-orig80k-count-300-20260926T232339Z/README.md)。准备TAG `20260926T232339Z`不是实际起跑时间；两GPU任务均于2026-09-27 02:18:30 UTC开始。

## 实现与运行版本

实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`，runner/contract/speed/observer及3测试形成显式档位/schema2，377核心合测通过。确定性20实际Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`，结果归档 `29f71d44f3cd96a772f5fc8122d09037bd45970d`；两库输入、100对标量和201叶状态逐位通过，见[确定性20结果](../orig80k-timing20-det-0927/result.md)。旧normal20及off复验的FAIL保持原样，不能改写为normal逐位重复，也不把旧readability20替代本闸门。

本轮从同一clean `70a641a827eb2559c3acb69430f55c1c53babaeb`启动；受保护源码与aec/c71逐字一致，数据/资产、两venv208依赖及113输入引用重新核对通过。tracked和两venv冻结到八阶段独立核验结束，于04:24:31 UTC形成最终快照后解除。旧INPUT3a、P1/100的00bd/ecf锚点及已批准最外退出限制不改，不把新版本回填旧证据。

## 相对冻结旧稿的必要变化

冻结基础为[perf300-ready旧launch](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-ready-candidate-20260926/launch.md)，SHA `86b6140521928a2298391334abf377ebf73b4faec47d8c1db6347f0a080cca8c`。旧四MD及原报告不改。与冻结候选相比，原launch十个Bash块保持逐字；仅适配正式档案链接、已知det结果和mkdir前只读预检说明。既有schema2控制内容为：

- 实际GPU_ENV和CPU_ENV都增加 `-u ORIG80K_TIMING_EQ_PROFILE`，继续清掉旧SMOKE_EQ与确定性flags；不依赖tmux继承、不覆盖x64。
- 训练终态后只读查launch及两份speed schema2记录的显式normal、实际flags、HEAD/300步/argv和源码；report完成后再核按本run checkpoint_root定位的报告段。缺字段/旧schema/混配/确定性档均拒绝，不读报告进程当前selector补证。
- 源码说明改以aec实现及实际确定性20版本为基准，保留未来perf实际Beta和结果占位。新增只读核验函数内联在launch，不开发独立执行框架。

原8阶段子入口和训练参数、wrapper body、f8父适配、完成器、统计/采样/measurement与预算公式保持。顺序为两GPU runner并行→双方真实退出并联合验收→full CPU恢复→count CPU恢复→联合report→两个measurement→budget；不能提前CPU恢复污染另一侧steady。所有8阶段都有独立原生退出与capture父真实返回，原15/16故障记录保留，日志0不单独证明最外成功。

## 实测与B档预算

原d3f只读前置实际PASS，父/宿主返回0。两GPU分别于02:27:56/02:27:51 UTC结束，两份299恢复在02:30:42前完成；联合report于02:30:47通过，两measurement于02:31:44完成。用户确认B档后，预算阶段于04:20:41通过；等待用户决定的时间不计入吞吐或ETA。

AWS8×A100、本地NVMe RAID、每run b64/workers4；预热0–99，稳态100–299各200步。full/count吞吐分别66.445381/66.092631 samples/s，稳态交集191.588853秒，按194.718151秒并集计算合计131.472078 samples/s。80k外推约21.59/21.70小时，包含报告约定的初始化及保存成本，仍是短窗估计。各卡GPU均值98.50%–98.90%，0%采样占比均0%；两侧均未触发报告定义的慢步阈值。实际采样间隔、重复读数、host/SHM及保存证据完整保留在result。

用户原话“方案 B：16 / 32 / 64 GiB”。两侧各8份checkpoint按每份16GiB规划，共256GiB；相对实测基数的增长余量84798963712B。采样增量下界5531541504B另加32GiB盲区余量，日志/cache另留64GiB；原公式得到所需383488663552B，即357.151649GiB。正式消费者核验时可用708412715008B，独立只读复核另记其当时可用量，均通过。预算路径为 `v1-store/bench/orig80k-perf300-results-20260926T232339Z/prod-disk-budget.json`，固定SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`，两正式run共用。

磁盘目标采样0.5秒，实际最大间隔约3.76/3.73秒且count有1条路径竞态；采样峰值与额外量只作下界。B档是用户明确选定的规划余量，不是未来体积或连续峰值保证。用户随后明确“确认无误后可以直接开始两个训练”，覆盖此前停在起跑前的范围；两正式run仍须共同Beta与最终前置复核通过，才按该新授权启动。

## 验证与归档边界

[准备验证](records/preparation/validation.json)记录本次语法、链接、八阶段参数保持、小记录正反例和环境unset验证的实际结果及四MD摘要。验证不运行GPU、tmux、runner、completion、report、measurement或预算消费者，不改两venv。小测试源码与原source留v1-store，docs只纳[复制清单](records/preparation/records-source-manifest.json)中的验证JSON及其来源引用，不复制脚本/yaml/运行命令附件或大权重。固定d3f预检的[契约](records/preparation/preflight-contract.json)和[12项小验证](records/preparation/preflight-small-tests.json)已备，不代表实际preflight已执行。

八个实际阶段的原生退出/capture、八段根调度与宿主返回及独立验收均通过，详见result及终态快照。用户已批准“保留原字节，限定豁免本轮 8 阶段 wrapper 日志（推荐）”；各wrapper仅第9行WRAPPER_COMMAND末尾1字节未转义分隔空格纳入精确豁免，其余清洗规则不变。未删除任何权重/cache，也未再跑20/100/300。下一步完成正式共同Beta及只读前置后直接启动两80k；本perf结果收尾时两正式run尚未启动。
