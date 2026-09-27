# timing20-count-on-0926-20260926T231534Z

## 1. 结论与指标速览

**本run尚未启动，是公开counting四任务子集测速包装on的20步真保存对照。** 准备标签 `20260926T231534Z`只用于唯一名称，不是实际起跑UTC；真实START/END、PID、UUID、W&B ID和结果均待运行记录。正式入口为smoke20，GPU `4,5,6,7`，b64/fsdp4/workers4；没有当前run的PASS、吞吐或质量结论。

## 2. 版本与代码状态

本阶段拟由主代理以 `commitV11.14Beta`锁定group launch、四run档案及正式模板；完整TRAIN_HEAD尚待该提交真实形成后填写。四个run使用同一clean Beta，执行期冻结源码和uv环境，不能用旧100的HEAD冒充新Beta。

已有INPUT为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1/100量具与B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。六run/四judge的100步证据已通过，原HEAD不改，具体来源和新Beta指纹复核要求见[group版本与前置](../orig80k-timing20-0926/README.md)。

## 3. 启动与配置还原

唯一实际启动依据是[group launch完整控制正文](../orig80k-timing20-0926/launch.md)，其中包含源码/runtime绑定、参数生成、guard门闩、独立capture及验收函数。该run的固定展开值如下；这段只定义配置，不执行任务：

```bash
RUN=timing20-count-on-0926-20260926T231534Z
TIMING=on
GPUS=4,5,6,7
LIB=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep
ASSETS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep
NORM_STATS_SHA256=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9
```

运行模板调用 `run_orig80k.sh smoke "$RUN" "$GPUS" "$LIB" "$ASSETS"`，显式 `ORIG80K_SMOKE_EQ_MODE=on`；runner成功后同一wrapper串行执行 `check_orig80k_completion.py --mode smoke`真实CPU恢复。实际session为 `orig80k-timing20-count-on-0926-20260926T231534Z`，具体PID/起止时间由运行记录提供。不能跳过group门闩而直接调用内部runner。

正式Git模板为 `scripts/training/tests/timing20-wrapper.sh`，SHA256 `45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f`；实际runtime在v1-store加固定参数前缀后绑定其整文件新SHA。guard执行原v1-store f8副本，Beta Git源/工作树/runtime逐字绑定，不复制.sh附件入docs。

## 4. 数据集与划分口径

本库为400集、189035帧、189035执行样本，构建Beta为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`。训练读取 `$LIB/framesamp`的4×4 packed，source与 `meta/episode_manifest.json`由runner显式绑定。来源和构建实测见[本库档案](../../dataset-build-doc/4task-counting-pub-400ep/README.md)。

norm文件为 `$ASSETS/robomme/norm_stats.json`，期望SHA为 `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。按用户授权使用本counting库重算的a770统计量；该库为公开集合中BinFill、PickXtimes、SwingXtimes、StopCube各100集的严格子集。 不新增划分，不改变输入、sampler或数据量来缩短测试。

## 5. 关键超参

| 配置 | 固定值 |
|---|---|
| steps / batch / FSDP / workers / seed | 20 / 64 / 4 / 4 / 42 |
| history | perceptual-framesamp-modul.yaml；modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024 |
| history SHA256 | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| warmup / peak_lr / decay_lr / decay_steps | 10000 / 5e-5 / 5e-5 / 100000 |
| optimizer / EMA | AdamW clip1.0 / 0.999 |
| log / save / keep | 100 / 10000 / 10000 |
| W&B / 初始权重 | online、project openpi / pi05_base |

history的 `streaming_obs_horizon=16`仍为原值，不与最多32 memory帧混写。仅smoke覆盖步数20；log/save/keep及其他原版超参不改。清除CPU平台遗留，不静默覆盖继承的JAX_ENABLE_X64；实际启用x64时由runner拒绝。

## 6. 硬件、调度与耗时

环境为AWS 8×A100-SXM4-80GB、/scratch本地md0 XFS；本run固定使用 `4,5,6,7`四卡。full与count两条库链并行，同库off/on严格串行：必须等前run真保存、CPU恢复及独立退出全部验收后才能推进。另库可按自身进度进入下一步，不等待无关阶段。

本run实际设备UUID、利用率、内存和耗时待记录。两库共享CPU、内存和磁盘，起跑仍核现场资源和完整剩余预算。20步共同观测包含同步/哈希开销，只验正确性，不产生吞吐、ETA或性能优劣结论。

## 7. 训练过程行为

on在同一共同只读取证层下调用真实speed.run(mode=smoke)，只在末步19保存前做既定边界同步，并采0.5秒磁盘元信息。 两侧均真正训练20批但取21批，最后一批不用于训练；记录完整模型实参/RNG、20步五标量和末步params/EMA/optimizer/step摘要。

保持真实save_state及异步wait，不用entry_equiv摘要器替代保存。普通log100只产生step0，末尾1…19的五标量由final尾窗补齐。以上是待执行行为与判据，当前没有本run数值或状态记录。

## 8. 保存、恢复与配对验收

本run必须只有checkpoint19，state_step20/loop_step19，全部标量和完整末态有限；CPU完成器真正加载19/params并与本run末步EMA摘要一致，run/HEAD/UUID及GPU采样、wait记录完整。完成器和tee都须真实返回0，wrapper与独立guard退出链共同通过。

本库off/on都通过各自完成验收后，CPU judge再核21次取批、20次模型输入/RNG、20步五标量、完整TrainState结构/dtype/shape/bytes逐位相同及两份真实恢复。CPU INPUT授权的主机dtype/四None键投影不用于放宽此训练对照。当前未执行本run恢复或配对judge。

## 9. 用户决定记录

用户选择「两库各跑一对 20 步（推荐）」，并要求「尽可能并行做」「你有8张卡」；因此采用两条4卡库链。用户「允许仅对拍关闭 W&B」仅针对P1/100，本run保持online与真实保存恢复。

「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」和四键missing/None授权保留其原CPU范围，训练逐位要求不变。「继续工作 一路做到起泡前 有问题问用户」限定本轮推进至正式80k起跑前，不因本run完成而直接起80k。

## 10. 计划外事件与退出边界

准备时没有本run运行事件。旧INPUT/P1最外final append退出不可追补，分别按「同样补记限制，沿用INPUT结果（推荐）」和「补记P1限制，补独立退出取证后继续（推荐）」保留原始内容证据及限制；没有实际失败新证据，不补造独立wait。

本run从起跑前就使用f8门闩，必须先retain并持久化identity再释放；结束后核原生pane、capture父真实返回、sidecar以及宿主记录的父适配实际返回，日志成功行不能替代。失败保留全部产物并停止本库依赖阶段，报告问题，不自行改名、覆盖或放宽判据。

## 11. 当前结论与下一步

当前状态为准备完成、待新Beta与现场放行，未启动。本run及同库off全部真实保存/恢复与独立退出验收通过后，立即可启动本库CPU judge；两库judge均通过才完成group阶段。 新20结果不会自动放行perf、预算或80k，完整后续闸门保留。

## 12. 归档文件清单

当前仅本README和[group README/launch](../orig80k-timing20-0926/README.md)为起跑前档案，未产生结果文件。实际run根将为 `v1-store/train-runs/mme_vla_suite/timing20-count-on-0926-20260926T231534Z`，records为 `v1-store/bench/orig80k/timing20-count-on-0926-20260926T231534Z`，driver为 `v1-store/logs/timing20-count-on-0926-20260926T231534Z.driver.log`；各自cache和W&B目录按group列出的新名称隔离。

待运行后才归档实际launch/runtime/run_meta、metrics/final、timing_eq四份记录及manifest、completion/清洗日志、GPU采样、on侧磁盘/计时记录、命令及模板摘要、source/runtime绑定、identity/native receipt/capture父侧证及其检查链。原始输出不混入manifest管辖子目录。只复制Git不能还原的测量记录，不复制脚本/yaml/venv/缓存或权重；真实UUID、W&B ID、UTC和结果届时按实填写。
