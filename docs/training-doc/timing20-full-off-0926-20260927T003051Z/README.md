# timing20-full-off-0926-20260927T003051Z：追加off重复性诊断启动档

## 1. 一句话结论与指标速览

**本run尚未启动，仅为用户已批准的同配置20步off追加诊断。** 目的是比较本库旧off与新off是否严格重复一致；不替代、修改或放行原off/on正式judge。准备TAG `20260927T003051Z`不是未来实际开始UTC，当前没有新run的PASS、MATCH、吞吐或质量结论。

| 项目 | 值或当前状态 |
|---|---|
| run_name | `timing20-full-off-0926-20260927T003051Z` |
| 旧off基线 | `timing20-full-off-0926-20260926T231534Z` |
| GPU / steps / batch | `0,1,2,3` / 20 / 64 |
| 新TRAIN_HEAD、START/END、UUID、W&B ID | 全部待真实记录 |

## 2. 版本与代码状态

旧off固定 `b0efbde61e38411fb1b9114eec8485d36a4ee9a0`；新run预计由独立 `commitV11.15Beta`锁定档案，完整40位SHA待真实生成。必须先完成当前V11.14失败归档，再从新clean Beta启动，源代码/工具/依赖逐字不变，新旧差异仅本轮docs/计划。三规则继续固定 `d710d8489b88aa75770af3452fd7c9b374deaef7`。

旧records.head和归档原SHA不改，不把新提交回填成旧off运行版本。新旧数据、资产、history、模块与依赖指纹分别核对；声明的相同配置不能替代实际指纹。本次只准备启动文档，不改量具或venv；实际新Beta与运行结果待产生。

## 3. 启动与配置还原

完整命令、Git差异/新输出守卫、原样模板及guard父适配见[共享launch](../orig80k-off-repeat-0927/launch.md)，不能绕过它直接运行内部runner。固定展开值如下，仅是配置定义：

```bash
RUN=timing20-full-off-0926-20260927T003051Z
TIMING=off
GPUS=0,1,2,3
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
```

实际session为 `orig80k-timing20-full-off-0926-20260927T003051Z`，实际开始时间/PID待记录。原模板 `scripts/training/tests/timing20-wrapper.sh`完整SHA为 `45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f`，不改正文。runtime在新v1-store命令目录加固定参数前缀并绑定整文件SHA；f8仍用原runtime实体，新Beta Git源/工作树/runtime字节联合校验。

## 4. 数据集与划分口径

公开16任务全集为1600集、768897帧、476857执行样本，与旧off消费同一实体库：packed为 `$LIB/framesamp`，source及episode manifest显式绑定，数据和划分不改。既有构建Beta为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`，见[构建档案](../../dataset-build-doc/16task-pub-1600ep/README.md)。

norm为 `$ASSETS/robomme/norm_stats.json`，期望SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。继续使用原版f332，不替换为完整库自算c5d45b…统计量。 不换数据、采样种子、worker数或精度来追求相等。

## 5. 关键超参

| 参数 | 固定值 |
|---|---|
| steps / batch / FSDP / workers / seed | 20 / 64 / 4 / 4 / 42 |
| history | perceptual-framesamp-modul.yaml，modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024 |
| history SHA | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| warmup / peak / decay / decay_steps | 10000 / 5e-5 / 5e-5 / 100000 |
| optimizer / EMA | AdamW clip1.0 / 0.999 |
| log / save / keep | 100 / 10000 / 10000 |
| 模式 / W&B / 初始化 | off、normal env / online / pi05_base |

原streaming_obs_horizon16保持。仅smoke步数覆盖20；不加确定性XLA_FLAGS、不变容差、不设置JAX_ENABLE_X64绕过实际False要求。各run独立缓存保持旧流程，不复用旧cache来制造结果。

## 6. 硬件、调度与耗时

AWS 8×A100-SXM4-80GB、md0 XFS；本run固定 `0,1,2,3`四卡，full/count新off按4+4并行。各自用原45b模板在训练后立即串行CPU恢复，和旧off行为相同，不采用perf的双侧训练结束屏障改变其流程。

实际GPU UUID、占用、CPU/内存/SHM、磁盘、起止时间与耗时待记录。新两份checkpoint及临时/cache/log需求需现场核算，不能仅凭300GiB最低保护放行。此诊断的观测/哈希耗时不用于性能结论。

## 7. 训练过程行为

与旧off一样：共同只读取证层下直接runpy原train.py，不经过speed.run；取21批、实际训练20批，记录输入tree/RNG、step0及tail1…19五标量和末步params/EMA/optimizer/step。真实save_state及异步wait不替换为摘要器，checkpoint19必须真保存。

初始完整state仍不新增摘要；现有观察器仅末步记录完整状态。param_norm只覆盖筛选kernel的聚合值，不能据相同norm证明初始params/EMA/优化器逐位相同。这项证据边界不因追加一次off自动消失。

## 8. 真恢复与独立重复性比较

先要求本run真实恢复checkpoint19、state_step20/loop19、完整有限值、run/HEAD/UUID、GPU采样及异步wait一致；完成器/tee、wrapper、原生pane、capture父实际返回和宿主父适配返回各自核对。单run成功不等于两次重复一致。

两新run各自验收后，按[独立CPU方案](../orig80k-off-repeat-0927/launch.md)分别用read_side读取旧b0和新真实Beta。它验证各自存档HEAD并重核当前代码/资产字节，未要求当前HEAD等于旧HEAD；不修改旧records，不猴补检查。完整配置仅放行现有IDENTITY_FIELDS两项run/输出差异，provenance只在比较投影中排除已分别核实的git_head_of_cwd，其他模块/来源字段严格相等。

同库21 fetch、20 model/RNG、20×5原始hex及完整state payload严格比较、无容差。读完工具进程0与comparison_status分开：不同就是独立诊断比较FAIL；即使MATCH也不输出原正式TIMING_EQ=PASS或抵销旧off/on失败。当前比较未执行。

## 9. 用户决定记录

用户已批准两库各追加一次同配置20步off、4+4并行、仅诊断；此处是授权范围复述，不另造用户原话。两个新名称由根代理采用，准备TAG不代表开始时间。

既有「尽可能并行做」「你有8张卡」继续落实；「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」不放宽本次训练payload判据。W&B保持online，原no-W&B授权仅P1/100适用。本轮仍按「继续工作 一路做到起泡前 有问题问用户」停在正式80k起跑前。

## 10. 计划外事实与处置

旧off/on每库全21 fetch、20 model/RNG一致，但100标量80个不同，末态147/201叶不同；四run自身真保存/恢复及原生退出通过，两正式judge真实FAIL。完整核验报告SHA `113cec3b17a19e85bd55c7a2b4420f150eac17dbea69a2b3c3e6c38d62265e99`由根代理归档，原失败不得删除或改写。

[机理说明](../orig80k-timing20-0926/records/diagnostics/timing20-step0-mechanism-note-20260927.md)未证实原因。若旧/新off仍不同，包装开关不是差异所必需条件；若重复相同，只是进一步线索，不能证明唯一根因或一般确定性。本run发生任何运行/验收错误均保留现场并报告，不自动重跑、改环境、改名、覆盖或放宽判据。

## 11. 当前结论与下一步

旧V11.14失败结果已归档，本run等待独立新Beta及现场前置；未启动、未恢复、未比较。所有新时间、UUID、W&B ID与结果待真实记录。

两off诊断及独立比较之后由根代理/用户依据实测决定后续；原正式off/on失败仍保留，perf冻结不启动。诊断MATCH不能自动放行perf或80k，DIFFER不能自动归因具体数值机制。

## 12. 归档文件清单

当前起跑前档案为本README及[共享launch](../orig80k-off-repeat-0927/launch.md)，尚未创建run输出或实测结果。未来records为 `v1-store/bench/orig80k/timing20-full-off-0926-20260927T003051Z`、权重根为 `v1-store/train-runs/mme_vla_suite/timing20-full-off-0926-20260927T003051Z`、driver为 `v1-store/logs/timing20-full-off-0926-20260927T003051Z.driver.log`。

待运行后保存实际launch/runtime/run_meta、metrics/final、timing_eq记录及manifest、completion/清洗日志、GPU采样、命令与模板SHA、Git源/runtime绑定、identity/native receipt/父sidecar和独立比较报告。旧off保留b0的原目录/文件SHA，新off保留其真实新Beta；比较输出在独立results根。只归档Git不能还原的测量记录，不复制脚本/yaml、权重、venv或cache入docs。
