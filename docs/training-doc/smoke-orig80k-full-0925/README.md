# 公开16任务库20步可读性与保存恢复：通过记录

本run按已提交的[启动记录](launch.md)完成20步训练、真实保存及CPU恢复，小型实测记录和日志已落到本目录`records/`。本组结果随`commitV11.11`与起跑Beta配对归档，不预填未知最终提交SHA。

下文`$REC`为原始运行目录`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-full-0925`，`$RUN_ROOT`为checkpoint根`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite/smoke-orig80k-full-0925`。正式小记录位于`records/`，权重留在原运行根。

## 1. 结论与指标速览

**公开16任务正式库的20步可读性、真实checkpoint保存和CPU恢复验收均通过。** 实际更新次数为20，循环末步及唯一checkpoint为19；普通日志step0与尾窗step1–19完整且全部有限。完成器当时真实加载`19/params`，61个EMA叶的dtype、shape、字节数及SHA均与本次保存现场一致，包含23个bfloat16叶和38个float32叶。

| 项目 | 已完成事实 |
|---|---|
| run | `smoke-orig80k-full-0925` |
| run UUID | `a56dd89a-ffbe-415d-9dd1-d5e205e14c25` |
| 更新/保存 | `state_step=20`，`loop_step=checkpoint_step=19`，checkpoint集合`[19]` |
| 标量覆盖 | step0一条普通日志；step1–19共19条末段记录，五标量均有限 |
| 真实恢复 | 61个EMA叶全部有限，恢复摘要与保存现场逐叶相等 |
| 整体终态 | `READABILITY20=PASS`、外层`EXIT_CODE=0` |
| 完成器判定 | `RUN_COMPLETED=PASS run=smoke-orig80k-full-0925 checkpoints=1 final=19` |

本结论仅证明这次短训练能够读取数据、执行更新、保存和恢复。正式`INPUT_EQ`、上游100步对拍、单跑/并跑对照及300步perf均未由本run验证，也不据20步作吞吐、ETA或模型质量结论。

## 2. 版本与代码状态

完整起跑Beta为 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**（`commitV11.11Beta`）。runner起跑时检查HEAD相等且`git status --porcelain`为空，wrapper结束时仍记录同一HEAD；`launch.json`、`final/start.json`、`final/final.json`和`completion.json`均绑定这个提交和同一run身份。本组结果以`commitV11.11`配对收尾；归档提交不是起跑代码，不替换、squash或amend此Beta锚点。

版本还原使用该提交中的`src/mme_vla_suite/training/config.py::_CONFIGS`的`mme_vla_suite`条目、`TrainConfig`默认值、`src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml`、`scripts/training/prod/run_orig80k.sh`、`orig80k_contract.py::parse_in_process()`、`train.py::main()`、`final_record.begin()/finish()/committed()`及`check_orig80k_completion.py::check()/restore_arrays()`。例如`git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/training/prod/run_orig80k.sh`可还原runner；不使用易漂移的代码行号。

归档准备和正式落地的核验见[archive_checks.json](records/archive_checks.json)。事后文档修改不作为原起跑证据；原证据仍是本run记录及driver的`CHECK_REPO_HEAD=PASS`、`CHECK_REPO_CLEAN=PASS`。

## 3. 启动与配置还原

实际tmux完整名称为`orig80k-full-read20-20260926T004046Z`，pane为`%539`，pane/包装PID为`3915445`，body PID为`3915450`。driver另记训练wrapper PID `3915973`、真实训练PID `3915984`及GPU采样PID `3915972`。本run使用独立输出根，启动时日志、records和checkpoint根都不存在；没有使用overwrite或resume。

wrapper日志为`v1-store/logs/orig80k-full-read20-20260926T004046Z.wrapper.log`，driver日志为`v1-store/logs/smoke-orig80k-full-0925.driver.log`。前者记录整体调度与恢复，后者由runner独占写入，外层没有向driver追加终态。以下为日志中已执行的实质命令，用于还原历史调用；归档整理未执行它们。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TRAIN_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  HISTORY_CONFIG_SHA256=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a \
  NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5 \
  bash scripts/training/prod/run_orig80k.sh \
  smoke smoke-orig80k-full-0925 0,1,2,3 \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite
```

训练driver成功后，同一tmux内串行执行CPU完成器，stdout另以tee写入`$REC/completion.summary.log`，并分别核对完成器和tee退出码：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
env -u JAX_PLATFORM_NAME CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
  XDG_CACHE_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/xdg \
  HF_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/hf \
  OPENPI_DATA_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models \
  uv run --no-sync python scripts/training/tests/check_orig80k_completion.py \
  --mode smoke --run smoke-orig80k-full-0925 \
  --head 49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  --records /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-full-0925 \
  --run-root /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite/smoke-orig80k-full-0925 \
  --log /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/smoke-orig80k-full-0925.driver.log \
  --out /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-full-0925/completion.json
```

runner展开后的实际训练argv保存在`$REC/run_meta.json`，完整解析配置在`$REC/launch.json`和`final/start.json`。实际环境中`JAX_PLATFORMS`、`XLA_FLAGS`和`JAX_ENABLE_X64`均未设置，CPU配置解析确认`jax_enable_x64=false`；本次没有通过显式覆盖继承的x64设置绕过拒绝。所有缓存及W&B目录位于本仓库`v1-store/`，没有覆盖HOME。运行时wrapper命令记录在ignored区域，本档案保留本文命令和实测记录，不复制独立bash/sh载体。

## 4. 数据集与划分口径

数据为公开`Yinpei/robomme_data_h5`的16任务×100集正式库`v1-store/datasets/16task-pub-1600ep`，共1600集、768897总帧、476857执行样本。模型当前帧、动作等读取该库source，历史特征读取`framesamp-4x4-v1`。此处不能混用此前私有四任务1600集库，也不能把总帧数当作执行样本数。

起跑前该库的来源pin、构建、finalize、全量packed verify和统计量比较均已完成。本run按seed42从全库执行样本shuffle取批；20步没有覆盖全部样本，不能充当独立验证集或正式输入对拍。

`launch.json`记录episode manifest文件SHA为`222d84f58fb82073603fda19992c4f82e542475294176bea99e8868036b8a11a`，store_meta文件SHA为`a554a7859ba35c883ec805244c98764a9eb10c1ddbc244850c5f213db4da669b`。manifest文件SHA与规范内容SHA不是同一字段，引用时保持区分。

训练实际使用原版norm_stats，路径为`v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`，完整SHA为`f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。完整库自算文件SHA为`c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd`，只作既定比较，没有替换原版。两者差异不在本次20步中另行决定。

## 5. 关键超参与观测配置

只在runner的smoke模式把`num_train_steps`覆盖为20；数据/输出/资产路径及run身份按预建档案传入，训练超参其余沿用原版条目。

| 配置项 | 实际值 |
|---|---|
| 配置、batch、FSDP、workers | `mme_vla_suite`，global batch64，FSDP4，workers4 |
| seed、步数 | 42、20 |
| history | `perceptual-framesamp-modul.yaml`，budget512，每帧16 token（4×4），最多32帧，modulation，memory_token_dim1024 |
| 模型 | pi05、action_horizon20、action_dim32、use_history；配置dtype为bfloat16，motion关闭 |
| 初始化 | `v1-store/models/openpi-assets/checkpoints/pi05_base/params` |
| 学习率 | warmup10000，peak_lr=decay_lr=5e-5，decay_steps100000 |
| 优化器 | AdamW：b1=0.9，b2=0.95，eps=1e-8，weight_decay=1e-10，clip_gradient_norm=1.0 |
| EMA | 0.999 |
| log/save/keep | 100/10000/10000；仍触发末步19保存 |
| 冻结与精度 | 原版freeze_filter；实际保存EMA为23个bfloat16叶和38个float32叶，不把配置dtype概括成所有参数均为bf16 |
| W&B | 开启，project=`openpi`，运行ID=`770cbaaz`；凭据不进入档案 |
| XLA显存比例 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` |

采用runner已有的结构化metrics和`final_record`末步记录、外部GPU采样，没有启用perf包装或JAX profiler。常规log100仅写step0是预期行为；后续19步由末段记录补齐。

## 6. 硬件、存储与耗时

硬件为AWS机器上的4×A100-SXM4-80GB（物理GPU0–3），实际mesh为`(batch=1, fsdp=4)`。底层存储为`/dev/md0` XFS、挂载`/scratch`。运行记录的依赖包括PyTorch2.7.1、JAX/jaxlib0.5.3、NumPy1.26.4、ml_dtypes0.4.1、Flax0.10.2、Optax0.2.4。

| 时间段 | UTC | 墙钟跨度及口径 |
|---|---|---|
| 整体wrapper | 2026-09-26 00:41:03→00:47:20 | 377秒，包含前检、训练、保存记录及CPU完成验收 |
| 训练runner | 00:41:03→00:46:19 | 316秒，包含CPU前检、初始化/JIT、20步、保存、EMA取证和W&B收尾 |
| CPU完成器 | 00:46:19→00:47:20 | 61秒，包含检查记录及真实恢复比较 |
| checkpoint保存提交 | 14.491722秒 | 引用controller已有的checkpoint提交时长记录，不作为单步耗时 |

controller已有测量记录checkpoint逻辑字节`11879851680 B`、实际分配`11879907328 B`，约11.064GiB。本次整理没有重新遍历或读取权重文件；该终态大小不是保存期间磁盘峰值，14.491722秒也不等于最后一训练步或全部保存取证的端到端耗时。

GPU采样通过完成器的完整性检查：每卡在记录的训练窗口内519个样本，平均间隔约0.50025秒，最大间隔约0.508秒；窗口为`00:41:49.930060Z`至`00:46:09.290245Z`。这只是运行覆盖与采样有效性的证据，未定义性能预热/稳态窗口。不能用316秒除20或日志中tqdm速率推断稳态吞吐、80k ETA或资源瓶颈。

## 7. 训练过程行为

`train.py::main()`实际构造四卡mesh并读取b64 batch，运行20次更新，`final_record.finish()`记录`state_step=20`、`loop_step=checkpoint_step=19`。常规metrics只有step0，末段记录严格覆盖1–19，五标量的有限性和十六进制记录自洽均由完成器检查。

| 标量 | step0实测 | step1–19均值 |
|---|---:|---:|
| loss | 0.099929943681 | 0.089858042566 |
| grad_norm | 1.507178068161 | 1.360645827494 |
| llm_grad_norm | 1.491889476776 | 1.347938951693 |
| mem_enc_norm | 0.129764407873 | 0.115255177805 |
| param_norm | 1815.546508789062 | 1815.546508789062 |

表中小数仅作展示，完整浮点值与`float.hex()`在现有JSON中保留。这里检查数值有限与记录覆盖；各步消费不同batch，不能从这一列短程统计推导模型质量、motion有效性或正式学习曲线。

真实训练进程的现场UUID为`a56dd89a-ffbe-415d-9dd1-d5e205e14c25`。末段记录时间为`2026-09-26T00:46:09.278599+00:00`，`checkpoint_wait_done.json`记录等待保存完成时间`00:46:09.290245+00:00`；同一UUID、HEAD和checkpoint路径在开始、结束与恢复记录中一致。

## 8. 训练后检查与尚未验证项

当时的CPU完成器调用`restore_params(..., restore_type=np.ndarray, dtype=None)`真实读取checkpoint，保持原始dtype；61个恢复叶全部有限，dtype/shape/字节数/SHA均等于本run最终EMA。检查结果文件现位于`$REC/completion.json`，其SHA为`340f8b53543041c36456f8c4aebacf6182139711caf5a3121ca8b307c248f897`。归档时只读该已保存结果并与`final/final.json`核对，没有重新执行恢复。

driver中训练及tee退出0，采样器以`reason=driver_stop`正常收尾、终点采样退出0；完成器及其tee退出0，wrapper任务及tee退出0，最后为`READABILITY20=PASS`。driver只有自己的唯一`EXIT_CODE=0`；wrapper会转录driver输出，判断整体结果须看wrapper末尾独立的`WRAPPER_TASK_EXIT/WRAPPER_TEE_EXIT/READABILITY20/EXIT_CODE`，不能按全文件仅出现一条EXIT_CODE要求wrapper。

以下仍未由本次检查完成：正式库全范围`INPUT_EQ`、上游P1与100步训练对拍、同入口单跑/并跑对拍、300步perf及策略/仿真评估。恢复权重与本次EMA一致也不意味着训练轨迹已与上游逐位一致；后者仍须原定独立验证。

## 9. 用户决定记录

本run沿用[原计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)中的用户原话：「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」，以及后续「恢复计划中的建库，严格按前置闸门推进」。正式库完成后的20步可读性、真实保存和加载检查已在预建launch中明确，本次使用原版统计量和原版W&B开启状态。

用户允许「主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」，并明确「允许这四键缺失与 None 等价」。四键仅为`motion_emb/motion_pos/motion_mask/mem_order`，其非None值或其他键不能顺带豁免；主机原始dtype/raw SHA仍留证，精确数值和signed zero要求保留。该输入比较许可不把本run变成已经完成的正式输入或训练对拍。

两个事项尚未获得用户答复：一是仅在P1/100步对拍关闭W&B的覆盖是否允许，二是是否为perf新增0.5秒只读磁盘峰值采样或暂停受影响的后续perf。当前20步W&B实际开启；不能把本次PASS写成用户已经批准对拍关闭W&B或批准磁盘采样改动。

## 10. 计划外事件与处置

已读日志含uv配置弃用提示、未启用ROCm/TPU后端的探测提示及部分buffer donation不可用提示。实际runtime记录CUDA四卡，训练和恢复均成功退出；本次没有为绕过这些提示修改训练源码、学习率、batch、精度或设备配置。正式归档日志保留这些提示，不因包含“Failed”字样直接删去或误写成CUDA训练失败。

保存期间磁盘峰值没有采集，是已明确报告的证据缺项；终态11.064GiB和保存提交时长不能替代峰值。该缺项不改变本次实际保存/恢复通过的事实，但不能据此填完正式80k的保存峰值预算。补采方案仍待用户决定。

controller含整体调度及历史模板状态。本档案只提取已完成的`full_readability20_result`及对应会话，见[controller.full20.snapshot.json](records/controller.full20.snapshot.json)，并与本run日志及completion核对；不把其他run或未同步的模板标签当成本run终态，没有修改原controller来补造历史。

## 11. 当前结论与下一步

full库20步读取、更新、真实保存及CPU恢复这道闸门通过。checkpoint与原始运行记录继续保留，归档没有删除、改名、续训或另起任务。完整库仍使用原版f332统计量，不切换到自算c5文件。

本组counting库及其[20步检查](../smoke-orig80k-count-0925/README.md)也已完成。下一阶段仍须从明确版本锚点完成正式输入及训练对拍，两项未决事项按用户答复处理。输入/训练对拍、并跑验证、perf及正式预算全部满足前，不把本run的PASS扩写为80k起跑闸门已全部通过。

本目录已按白名单归档小记录并清洗验核，来源、ignored暂存及正式目标映射见第12节。结果随本组`commitV11.11`配对归档；训练Beta永久保留，不使用归档后的HEAD冒充起跑代码。

## 12. 已归档文件清单与格式核验

本目录`records/`已包含**16个文件，共348373 B**，不含README和launch正文。来源、ignored暂存与正式目标的字节数/SHA、读回核验及凭据筛查封装在[archive_checks.json](records/archive_checks.json)，该最终清单SHA为`2a3bce6eee4f2bcb4ec4e50f481b6df995747504845eb8f50b943b59cd744f8d`。

| 已有源 | 正式归档文件 | 内容 |
|---|---|---|
| launch/runtime/run_meta | [launch.actual.json](records/launch.actual.json)、[runtime.json](records/runtime.json)、[run_meta.json](records/run_meta.json) | 实际解析配置、设备及argv |
| metrics | [metrics.jsonl](records/metrics.jsonl) | step0五标量 |
| final三份记录 | [start.json](records/final/start.json)、[final.json](records/final/final.json)、[checkpoint_wait_done.json](records/final/checkpoint_wait_done.json) | 身份、尾窗、EMA及异步保存完成 |
| 完成器结果与日志 | [completion.json](records/completion.json)、[completion.summary.log](records/completion.summary.log) | 真实恢复判定 |
| GPU采样与错误流 | [gpu.csv](records/gpu.csv)、[gpu.csv.err](records/gpu.csv.err) | 采样覆盖证据，错误流为空 |
| driver/wrapper原始日志 | [train.summary.log](records/train.summary.log)、[wrapper.summary.log](records/wrapper.summary.log) | 清洗后的过程与退出记录 |
| checkpoint小型元数据 | [_CHECKPOINT_METADATA](records/checkpoint_19/_CHECKPOINT_METADATA) | 原生保存提交时间 |
| controller相关小记录 | [controller.full20.snapshot.json](records/controller.full20.snapshot.json) | 本run的一次读取快照 |
| 两阶段归档核验 | [archive_checks.json](records/archive_checks.json) | 暂存检查与正式文件映射 |

归档准备阶段保存12份原字节副本、2份清洗日志和1份controller提取快照，回车归一后删除`%|`类进度中间态；本run实际删除0行，关键行序列完整。原ignored暂存核验SHA为`f06c462beddfc144581ef6e5be5c5fe342fc06186c0ae9a320a58f9e891130d1`，其内容已封装在最终检查清单中，原暂存没有被重写。

正式落地时仅对`.log`副本去掉行尾空格和制表符：`train.summary.log`与`wrapper.summary.log`合计移除**168个行尾空白字符**。原始日志和ignored暂存日志均未修改；所有行在同一行尾规范化后逐行相等，关键判定、时间及指标数值没有变化。最终清单保留规范化前后摘要与目标映射，不能将格式清洗后的文件误记为与原日志原字节相同。

controller以单次读取保存本run结果、会话及源SHA/读取时刻，不要求可变controller此后的SHA保持不变。其余小记录按白名单复制并复核，正式归档没有再次训练、恢复或读取大权重。

不归档H5、features、packed数组、checkpoint权重、脚本、yaml或独立`command.txt`/bash载体。权重留在`$RUN_ROOT/19`；生产代码与配置由固定Beta还原，实质命令在正文中记录。本组结果以`commitV11.11`与`commitV11.11Beta`配对收尾，最终提交SHA由Git历史记录，不能替代实际起跑锚点。
