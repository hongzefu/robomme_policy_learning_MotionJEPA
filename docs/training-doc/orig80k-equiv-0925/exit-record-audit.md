# 历史最外层退出记录边界与后续包装修正

## 结论与当前范围

**数据、训练及CPU恢复子任务的成功记录没有被此次反例证伪；四个历史最外层包装的最终进程退出码缺少独立持久化观测。** 原始日志中记录的`EXIT_CODE=0`保留，但不能单独将它升级为“最外层最后一次tee及整个包装进程真实退出0均已独立证明”。当前没有证据表明历史末尾tee实际失败，也不能用日志未检出错误反推它一定成功。

本审计的固定代码基线为`ec6c9e35784a96f636a587dd737dffa294b616ad`，历史实际运行Beta为`49a333eb18e8d6ff1143bf7871ef7c498ab91579`。开审时在途范围为训练索引和新建的`orig80k-equiv-0925/`输入启动文档；这些未提交内容没有被冒称为ec6代码。历史部分只读四份已保存命令、六份原始日志及小型实测记录，没有重跑建库、训练、恢复或修改原始产物。

新CPU输入包装已修复并完成两轮独立纯shell验证，但**CPU输入采集尚未启动**。用户要求“有问题立刻问用户 不要自己决策”；已就历史记录处理询问“补记限制，继续CPU输入取证”或“先制定额外复验方案”，尚未收到答复。因此暂不自行选择处置，也不把历史观测缺项写成已经补齐。另两项P1/100步W&B与perf磁盘采样仍分别待答。

## 缺口的机制

四个历史最外层包装均采用以下顺序：先将根据主体任务和正文tee计算的`rc`写入末尾`printf ... | tee -a`，其中包含`EXIT_CODE=0`或`READABILITY20=PASS`；再检查这条末尾管道自身的`PIPESTATUS`。代码发现末尾管道失败会真实返回非零，但没有独立保存该最后状态。

由此存在已复现的反例：主体工具成功，正文tee成功，末尾tee完整写出成功文本后自身返回非零。包装进程失败，但仅检查日志中的工具PASS、`COMMAND_EXIT=0`、正文`TEE_EXIT=0`和`EXIT_CODE=0`仍可能误放行。`TEE_EXIT`和`WRAPPER_TEE_EXIT`描述的是此前正文管道，并非输出这些字段的末尾tee。

固定基线的`scripts/training/prod/run_orig80k.sh`也有这种footer顺序。它与历史Beta中对应文件的Git blob相同：`d51521e66964bfb5838eaa134b080eef0369b2d5`。但两次历史20步检查由外层`readability_body()`直接捕获整个runner的真实返回码，已覆盖runner自身末尾检查；剩余观测限制只在最外一层，不能笼统说训练进程退出码没有记录。

## 哪些证据已经存在

| 层级 | 已核实的实际证据 | 边界 |
|---|---|---|
| full数据子阶段 | `run_stage()`捕获10个真实返回码，全部0；主体`TASK_EXIT=0`、正文`TEE_EXIT=0` | 不等于最外footer状态有独立回执 |
| counting数据子阶段 | 12个阶段真实返回码均0，主体和正文tee记录0 | 同上 |
| 严格子集检查 | 工具与其tee的两项`PIPESTATUS`均0；400集、189035样本，两类失配0 | 是内容验收证据，不是最外包装OS退出观测 |
| 两次实际训练进程 | runner的`wait "$train_pid"`捕获返回0，GPU采样收尾也通过 | 训练进程成功有直接证据 |
| 两次完整runner | read20外层捕获runner的`$?`，记录`stage=train20 code=0` | 覆盖runner自己的footer检查 |
| 两次CPU恢复器 | 外层真实捕获恢复器及其tee的两项状态，均0；恢复结果PASS | 完成器没有只凭文件名猜测成功 |
| 两个数据包装与两个read20包装的最末footer | 代码检查了自身管道，但未独立落档最后管道或包装进程的最终退出观测 | 仅凭最后一行0不足以排除上述反例 |

两份完成器结果仍为20次更新、唯一checkpoint19，各61个恢复EMA叶与各自训练末步摘要完全相同。原始完成结果SHA保持不变：

- full：`340f8b53543041c36456f8c4aebacf6182139711caf5a3121ca8b307c248f897`。
- counting：`97cd2cdc2c6feb6c9c5612813c64f871cadcb53259a44829ba58bef72549b268`。

完整内容与边界分别见[完整库](../../dataset-build-doc/16task-pub-1600ep/README.md)、[counting库](../../dataset-build-doc/4task-counting-pub-400ep/README.md)、[full20](../smoke-orig80k-full-0925/README.md)和[count20](../smoke-orig80k-count-0925/README.md)。本文不改写这些实测小记录，也不以修正包装后的新代码冒充旧运行版本。

## 历史命令与观测溯源

四份命令都位于`v1-store/bench/orig80k-build-preflight-0925/`；本次核对其文件SHA与已有运行记录一致。它们是运行时shell文本，不复制成档案内独立脚本。

| 运行 | 实际命令文件 | SHA256 |
|---|---|---|
| full建库 | `full_build_command.txt` | `02e47a92d5753f2e649860bdd72a5bf74c2b256ee422ab93ef27f6b91b0182bb` |
| counting建库 | `cnt-pub400-20260926T004948Z.command.txt` | `35c0af03c1a8f58e3022eece8b0231521c97dcc863b6e3fab2c2d3a9c7fb6f2f` |
| full20 | `orig80k-full-read20-20260926T004046Z.command.txt` | `7206c3d1ed54b7f05f5a28b25aa70c9621280b71f2f923eeb7baa6564c4d63eb` |
| count20 | `orig80k-count-read20-20260926T014535Z.command.txt` | `f487451f08b6cce8e77f9881b298878dd03508ae2ffcf7f7b2c7750de1b6e3a1` |

四个历史tmux会话均已不存在。当前没有找到当时留存的父进程wait结果或`pane_dead_status`来独立证明这四个最外包装的最终退出码。controller中`exit_code=0`、`complete_exit_0`源自当时日志判读，不能自动升级为这种独立观测；监听日志的tail/grep工具退出0也只证明监听器结束。

六份原始日志未检出tee/write-error匹配，但最外末尾tee的stderr未必进入这些文件，因此未检出不是排除失败的证明。这里没有声称历史tee发生过失败，更没有据反事实重写数据或训练结果。

## 新包装修正与实际验证

[CPU输入launch的`launch_input()`](launch.md#公共命令设置与有退出记录的启动包装)先通过末尾tee写不含成功终态的记录，再立即取得末尾printf及tee的两个返回码；任一非零令综合结果非零。随后使用受检查的直接追加写出`FOOTER_PRINTF_EXIT`、`FOOTER_TEE_EXIT`和唯一`EXIT_CODE`，不在检查末尾tee之前留下成功退出标记。

collector或judge的日志放行必须同时满足：工具判定成功，命令、正文tee、footer printf、footer tee以及综合退出记录均唯一且为0。修正只实现既有“失败停止、tee失败不能算成功”要求；四份采集和两份judge的参数、CPU环境、并行调度、数值精度与四None键判据逐字未改。

两方各自从文档提取实际shell包装，在`/tmp`隔离目录中执行了六类纯shell替身检查：正常完成、主命令非零、正文tee失败、末尾tee完整输出后失败、命令文件自SHA不匹配、已有日志拒绝覆盖；没有调用真实tmux、collector、训练或依赖同步。两轮均通过，临时测试载体按归属清理。

独立重审的关键反例将末尾tee设置为写完正文后返回29。修复后的实测为：包装返回29、`FOOTER_TEE_EXIT=29`、唯一`EXIT_CODE=29`，不含成功退出0；另一轮以返回1重复也得到非零。Bash语法及文档空白检查通过。修正不逆向补造历史记录，后续GPU任务也不能仅依赖旧runner日志末行判定最外层成功。

## 待用户决定与后续边界

当前已请求用户决定：补充上述历史说明后，沿用数据内容和20步保存恢复子任务的成功证据继续CPU输入取证；或先制定额外复验方案。没有默认选择，也没有启动CPU采集、P1、100步、perf或80k。

若要求补证，只能使用当时实际留下的独立监督记录；不存在的历史退出状态无法靠修改日志、更新controller或再次读取成功标记重建。任何新复验都是新事件，必须另留版本、命令、真实退出与产物记录，不能冒充旧包装的退出证明。现有部分verify入口会更新数据元信息，不能未经核实就称为“只读复验”；复验范围与实现应先形成具体方案，不自行重建或覆盖两个库及checkpoint。

P1/100步的W&B覆盖与perf保存磁盘峰值采样是另外两项待决，仍分别处理。主机dtype许可、四键缺失/None许可和训练标量/状态逐位要求均不改变。
