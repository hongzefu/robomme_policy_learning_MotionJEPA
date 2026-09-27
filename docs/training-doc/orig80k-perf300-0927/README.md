# 两库normal300 perf起跑前档案（待V11.17Beta，未执行）

本档案包含[共享launch](launch.md)及两run十二节README：[full](../perf-orig80k-full-300-20260926T232339Z/README.md)、[count](../perf-orig80k-count-300-20260926T232339Z/README.md)。名称仍为 `perf-orig80k-{full,count}-300-20260926T232339Z`；准备TAG不是实际起跑时间。实际perf run尚未创建，未启动训练、恢复、tmux或任何perf阶段；未来TRAIN_HEAD必须是独立commitV11.17Beta的实际完整SHA。

## 实现与运行版本

实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`，runner/contract/speed/observer及3测试已形成显式档位/schema2，377核心合测通过。确定性20实际Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`，准备TAG `20260927T012426Z`；两库正式judge已schema2/deterministic100 PASS，root已核七phase宿主/原生/capture均0；最终独立报告及归档引用由root补齐，见[确定性20结果](../orig80k-timing20-det-0927/result.md)。旧normal20及off复验的FAIL保持原样，不能改写为normal逐位重复，也不把旧readability20替代本闸门。

未来perf须另立实际Beta、共同clean HEAD并核指纹。受保护scripts/training、src、pyproject.toml和uv.lock应与aec以及实际确定性20代码精确相同；若不同停止处理，不能借本档案授权新源码改动。旧INPUT3a、P1/100的00bd/ecf锚点及已批准最外退出限制不改，不把新版本回填旧证据。

## 相对冻结旧稿的必要变化

冻结基础为[perf300-ready旧launch](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-ready-candidate-20260926/launch.md)，SHA `86b6140521928a2298391334abf377ebf73b4faec47d8c1db6347f0a080cca8c`。旧四MD及原报告不改。与冻结候选相比，原launch十个Bash块保持逐字；仅适配正式档案链接、已知det结果和mkdir前只读预检说明。既有schema2控制内容为：

- 实际GPU_ENV和CPU_ENV都增加 `-u ORIG80K_TIMING_EQ_PROFILE`，继续清掉旧SMOKE_EQ与确定性flags；不依赖tmux继承、不覆盖x64。
- 训练终态后只读查launch及两份speed schema2记录的显式normal、实际flags、HEAD/300步/argv和源码；report完成后再核按本run checkpoint_root定位的报告段。缺字段/旧schema/混配/确定性档均拒绝，不读报告进程当前selector补证。
- 源码说明改以aec实现及实际确定性20版本为基准，保留未来perf实际Beta和结果占位。新增只读核验函数内联在launch，不开发独立执行框架。

原8阶段子入口和训练参数、wrapper body、f8父适配、完成器、统计/采样/measurement与预算公式保持。顺序为两GPU runner并行→双方真实退出并联合验收→full CPU恢复→count CPU恢复→联合report→两个measurement→budget；不能提前CPU恢复污染另一侧steady。所有8阶段都有独立原生退出与capture父真实返回，原15/16故障记录保留，日志0不单独证明最外成功。

## 仍需真实补齐

根代理须将已完成确定性20的最终归档/六份独立报告和期末不可变controller快照路径/SHA填入共享launch的完整preflight CLI（四个on/judge用无损v2）；再按固定d3f工具取得实际stdout及宿主返回0。未来perf Beta、现场GPU/CPU/内存/SHM/IO/存储及剩余空间；真正的START/END/PID/run UUID/W&B ID。没有perf吞吐、ETA或采样峰值实测。

三项margin仍未选数值：growth为明确非负整数，sampling及logs/cache为明确正整数，均须真实receipt/采样/日志来源与推导basis。没有默认倍率或GiB数值，不把0.5秒采样最大值说成连续峰值，不缩减保守余量。预算PASS也不自动执行prod；本轮终点仍为正式80k起跑前。

## 验证与归档边界

[准备验证](records/preparation/validation.json)记录本次语法、链接、八阶段参数保持、小记录正反例和环境unset验证的实际结果及四MD摘要。验证不运行GPU、tmux、runner、completion、report、measurement或预算消费者，不改两venv。小测试源码与原source留v1-store，docs只纳[复制清单](records/preparation/records-source-manifest.json)中的验证JSON及其来源引用，不复制脚本/yaml/运行命令附件或大权重。固定d3f预检的[契约](records/preparation/preflight-contract.json)和[12项小验证](records/preparation/preflight-small-tests.json)已备，不代表实际preflight已执行。

起跑前文档不是运行证明；所有真实perf状态仍待逐阶段产生。旧normal20/off复验保持原FAIL，确定性档不会自动将其改写为normal逐位等价或根因结论。
