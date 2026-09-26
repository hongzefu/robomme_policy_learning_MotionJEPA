# 公开counting库20步可读性与保存恢复：通过记录

本run按已提交的[启动记录](launch.md)完成20步训练、真实保存及CPU恢复，小型实测记录和日志已落到本目录`records/`。本组结果随`commitV11.11`与起跑Beta配对归档，不预填未知最终提交SHA。

下文`$REC`为原始运行目录`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-count-0925`，`$RUN_ROOT`为checkpoint根`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite/smoke-orig80k-count-0925`。正式小记录位于`records/`，权重留在原运行根。

## 1. 结论与指标速览

**公开counting正式库的20步可读性、真实checkpoint保存和CPU恢复验收全部通过。** `state_step=20`，循环末步和唯一checkpoint均为19；step0普通日志及step1–19末段记录覆盖完整、五标量全部有限。此前完成器真实加载`19/params`，61个恢复叶的dtype、shape、字节数及SHA与本run保存现场EMA一致，包含23个bfloat16叶和38个float32叶。

| 项目 | 实测结果 |
|---|---|
| run | `smoke-orig80k-count-0925` |
| run UUID | `ba82f509-8189-4002-809a-021f11d58d97` |
| GPU/配置 | 物理GPU4–7，batch64/FSDP4/workers4，seed42 |
| 更新与保存 | 20次更新，checkpoint集合`[19]` |
| 标量 | 常规日志仅step0；末段1–19完整、全部有限 |
| 恢复 | 61叶全部有限且与保存现场EMA逐叶一致 |
| 完成器 | `RUN_COMPLETED=PASS run=smoke-orig80k-count-0925 checkpoints=1 final=19` |
| 日志终态记录 | `READABILITY20=PASS`，最后`EXIT_CODE=0`；最外层进程退出证据边界见第8节 |

这道检查没有完成正式`INPUT_EQ`、上游100步对拍、单跑/并跑对照或300步perf；也不构成策略评估、模型质量或吞吐/ETA结论。

## 2. 版本与代码状态

完整起跑Beta为 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**（`commitV11.11Beta`）。wrapper和runner检查起跑HEAD及clean工作区，结束仍记录同一HEAD；`launch.json`、三个final记录及`completion.json`均绑定该提交和同一run UUID。本组结果以`commitV11.11`配对收尾；归档提交不是起跑代码，不替换、合并或改写Beta。

稳定代码锚点为`src/mme_vla_suite/training/config.py::_CONFIGS`的`mme_vla_suite`条目及`TrainConfig`默认值，`scripts/training/prod/run_orig80k.sh`、`orig80k_contract.py::parse_in_process()`、`train.py::main()`、`final_record.begin()/finish()/committed()`及`check_orig80k_completion.py::check()/restore_arrays()`。以`git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:<对应路径>`还原，不使用硬编码代码行号。

归档准备和正式落地的核验分别封装在[archive_checks.json](records/archive_checks.json)中。事后整理没有改变生产代码，也不以整理时的工作区状态替代run本身的起跑记录。

## 3. 启动与配置还原

会话全名`orig80k-count-read20-20260926T014535Z`，pane `%541`，pane/包装PID `3957023`，body PID `3957028`；训练wrapper PID `3957599`，真实训练PID `3957603`，GPU采样PID `3957598`。wrapper日志为`v1-store/logs/orig80k-count-read20-20260926T014535Z.wrapper.log`，driver日志为`v1-store/logs/smoke-orig80k-count-0925.driver.log`，两者分别记录整体与训练终态。

训练前先只读核对counting建库唯一成功终态、clean HEAD、新输出根、`build_sizes.json`与建库日志一致、manifest绑定、严格子集结果及统计量SHA。实际得到`COUNT_METADATA=PASS`，build_sizes SHA为`69f36813bf9168ff08f57b64f6e6587c5746e3466caf84d529f38bde84b40f7e`，manifest规范SHA为`6b309822aa604a31f6eb0dfb5f02d5482573dc5fa11dc5c4f917880aa92ba986`。norm期望来自已验收记录并与实际文件独立核对，填写完整字面量，不是边算边默认接受，也没有为这一客观测量值再次请求用户批准。

以下为已执行的实质命令，用于历史还原；归档整理未执行这些训练或恢复命令。展开训练argv保存在`$REC/run_meta.json`。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TRAIN_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  HISTORY_CONFIG_SHA256=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a \
  NORM_STATS_SHA256=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9 \
  bash scripts/training/prod/run_orig80k.sh \
  smoke smoke-orig80k-count-0925 4,5,6,7 \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep
```

训练driver成功后，同一tmux串行执行CPU完成器，stdout单独经tee写入`$REC/completion.summary.log`，并分别检查完成器和tee退出码：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
env -u JAX_PLATFORM_NAME CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
  XDG_CACHE_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/xdg \
  HF_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/hf \
  OPENPI_DATA_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models \
  uv run --no-sync python scripts/training/tests/check_orig80k_completion.py \
  --mode smoke --run smoke-orig80k-count-0925 \
  --head 49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  --records /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-count-0925 \
  --run-root /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite/smoke-orig80k-count-0925 \
  --log /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/smoke-orig80k-count-0925.driver.log \
  --out /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-count-0925/completion.json
```

实际训练环境中`JAX_PLATFORMS`、`XLA_FLAGS`和`JAX_ENABLE_X64`均未设置，配置解析确认`jax_enable_x64=false`；没有覆盖继承的x64设置来绕过守卫。所有缓存、记录和W&B路径均位于本仓库`v1-store/`，未覆盖HOME。本档案没有复制独立wrapper/bash载体。

## 4. 数据集与划分口径

输入是公开全集的四个counting任务BinFill、PickXtimes、SwingXtimes、StopCube，各100集，合计400集、189035帧和189035执行样本；无demo段，所以总帧数与执行样本数相同。库根`v1-store/datasets/4task-counting-pub-400ep`，历史特征为4×4 packed，其他训练输入读取同库source。

起跑前已完成来源、构建、finalize、全量pack/verify、独立norm_stats和严格子集内容比较；本次metadata前检也绑定了这些已完成记录。20步从该库按seed42 shuffle取批，没有覆盖全库，也没有验证正式样本/batch对上游的完整交付等价。

`launch.json`记录manifest文件SHA为`fa0535335751910451c483d198cda151f04d25df36b75467c7460d118e25ad52`，store_meta文件SHA为`6c0ffe3968149fd666091e1809fa55a4c1540132db906ec27a40a584cb6aa748`。文件SHA与第3节manifest规范内容SHA分开引用。

训练使用本库自算`v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep/robomme/norm_stats.json`，SHA为`a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。`assets-dir`传其上一级`4task-counting-pub-400ep`，asset-id仍为robomme；没有使用完整库原版f332或自算c5统计量替代。

## 5. 关键超参与观测配置

除了数据/统计量/运行身份与路径，本run仅在smoke入口把总步数覆盖为20，其余训练参数沿用原版`mme_vla_suite`。

| 项目 | 实际值 |
|---|---|
| batch/FSDP/workers/seed | global batch64、FSDP4、workers4、seed42 |
| 模型 | pi05、action_horizon20、action_dim32、use_history、配置dtype=bfloat16 |
| history | modul，budget512，16 token/帧（4×4），最多32帧，memory_token_dim1024；motion关闭 |
| history文件SHA | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| 初始化 | `v1-store/models/openpi-assets/checkpoints/pi05_base/params` |
| 学习率 | warmup10000，peak_lr=decay_lr=5e-5，decay_steps100000 |
| 优化器 | AdamW，b1=0.9，b2=0.95，eps=1e-8，weight_decay=1e-10，clip_gradient_norm=1.0 |
| EMA、冻结 | EMA0.999，原版freeze_filter |
| log/save/keep | 100/10000/10000；末步19仍触发保存 |
| W&B | 开启，project=openpi，run ID=`iuk34x8g`；不归档凭据 |
| 显存比例 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` |

未启用perf包装或JAX profiler。只使用已有metrics、末步EMA/尾窗记录及外部GPU采样。保存树的23个bfloat16叶和38个float32叶按实际记录留证，不能把模型配置dtype写成全树统一bf16。

## 6. 硬件、存储与耗时

AWS单机4×A100-SXM4-80GB，物理GPU4–7，实际mesh为`(batch=1, fsdp=4)`；局部CUDA设备编号不等同于物理编号。存储为`/dev/md0` XFS，挂载`/scratch`。运行记录包含PyTorch2.7.1、JAX/jaxlib0.5.3、NumPy1.26.4、ml_dtypes0.4.1、Flax0.10.2和Optax0.2.4。

| 阶段 | 2026-09-26 UTC | 耗时口径 |
|---|---|---|
| 整体wrapper | 01:45:54→01:51:37 | 343秒，包括metadata检查、训练和CPU验收 |
| 训练runner | 01:45:55→01:50:36 | 281秒，包括前检、初始化/JIT、20步、保存取证及W&B收尾 |
| CPU完成器 | 01:50:36→01:51:37 | 61秒，包括记录核验和当时真实恢复 |
| checkpoint提交 | `16201301633 ns = 16.201301633 s` | 原生checkpoint元数据的commit与init时间差 |

小型`19/_CHECKPOINT_METADATA`中init为`1790387353747557070`、commit为`1790387369948858703`纳秒，本次只读该元数据核对上述差值。controller已有du测量为物理分配`11879915520 B`、逻辑`11879876416 B`，约11.064GiB；本次没有重新扫描或读取权重。终态大小和提交时长均不是保存期磁盘峰值。

GPU采样通过完成器的覆盖检查：每卡452个窗口内样本，平均间隔约0.500300秒，最大间隔约0.512秒；窗口为`01:46:41.036776Z`至`01:50:26.990629Z`。这些用于确认采样有效，不是性能稳态窗口。不用281秒除20、不用tqdm速率，也不与full20墙钟时间比较来推导吞吐、ETA或瓶颈。

## 7. 训练过程行为

`train.py::main()`完成20次更新，唯一末步保存为19；普通metrics只有step0，`final_record.finish()`记录1–19共19步末段。本次小记录核对确认五标量字段完整、数值有限、十进制与`float.hex()`自洽，尾窗均值与逐步值一致。

| 标量 | step0实测 | step1–19均值 |
|---|---:|---:|
| loss | 0.091358602047 | 0.090672967465 |
| grad_norm | 1.500042080879 | 1.430209862558 |
| llm_grad_norm | 1.487161517143 | 1.417042010709 |
| mem_enc_norm | 0.126513749361 | 0.120219538478 |
| param_norm | 1815.546508789062 | 1815.546508789062 |

展示值已舍入，完整浮点与十六进制值保留在原始及正式归档JSON。此表服务于有限性及覆盖检查，不构成模型质量结论；不同batch的短程均值不能当成等价样本的学习曲线，counting与full的数据分布及norm_stats也不同，loss不可直接比较。

末步记录时间为`2026-09-26T01:50:26.981786+00:00`，保存等待完成为`01:50:26.990629+00:00`。开始、末步、等待及完成器均为UUID `ba82f509-8189-4002-809a-021f11d58d97`及同一Beta。

## 8. 训练后检查与尚未验证项

此前完成器通过`restore_params(..., restore_type=np.ndarray, dtype=None)`真实加载checkpoint，保持原dtype；61个恢复叶全部有限，且完整叶描述与保存现场EMA相同。`completion.json`的SHA为`97cd2cdc2c6feb6c9c5612813c64f871cadcb53259a44829ba58bef72549b268`，已在归档准备中核对小记录哈希及叶描述相等，没有重跑恢复。

driver中训练与正文tee退出0，整个runner的真实返回0另由wrapper捕获，包含runner自身footer检查。completion任务及其tee、wrapper主体及正文tee也分别返回0。driver保留唯一成功终态；wrapper转录driver输出，不把中途转录的一行EXIT_CODE当作wrapper结束。

补记[历史最外层退出记录审计](../orig80k-equiv-0925/exit-record-audit.md)：wrapper末尾先写`WRAPPER_TASK_EXIT/WRAPPER_TEE_EXIT/READABILITY20/EXIT_CODE`，再检查该footer自身printf/tee，最后状态未独立持久化。末行0不能单独证明最外层包装实际退出0；目前没有证据表明历史footer失败。训练、真实保存和CPU恢复PASS保留，原始records不改写。

正式全范围`INPUT_EQ`、上游P1/100步、同入口单跑/并跑、300步perf和策略评估仍未由本run完成。保存恢复对自身EMA逐叶相等，不等于训练标量与状态已对上游逐位验证。

## 9. 用户决定记录

沿用[原计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)中的用户原话：「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」、「纯counting跑4*100 公开集合的子集合」及「恢复计划中的建库，严格按前置闸门推进」。本库独立重算norm_stats已获授权，实际a770摘要是建库完成后的客观测量值，不是另行需要用户批准的参数变更。

用户允许「主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」，并批准「允许这四键缺失与 None 等价」。仅`motion_emb/motion_pos/motion_mask/mem_order`缺失与显式None可等价，不扩展到其他键或非None值；原始dtype/raw SHA、精确数值与signed zero仍留证和严判。

两个问题尚未获得用户答复：仅对P1/100步对拍关闭W&B是否允许，以及是否为perf补充0.5秒只读磁盘峰值采样或暂缓受影响步骤。本次20步W&B保持原版开启；不能以其PASS推断上述覆盖或新采样已获批准。

## 10. 计划外事件与处置

本run没有以修改训练参数、数据统计量、x64或源码绕过检查，真实训练、保存和完成器均成功。日志中的环境/框架提示保留在正式归档，不因出现“Failed”等字样就删除或单独判为训练失败；最终结论结合CUDA runtime、完整记录及退出码。

保存期间磁盘占用峰值未采集；11.064GiB是终态分配量，16.201301633秒是元数据提交时长，均不能替代峰值证据。该缺项不否认本次真实保存/恢复已成功，但正式80k预算仍须解决已提出的峰值取证问题。

归档准备已进行敏感JSON字段、凭据赋值、常见令牌模式及当前环境已有凭据值筛查，未发现匹配；没有读取凭据文件或输出凭据内容。源文件保持不变，controller只读取一次，避免把后续调度变化混入该快照。

## 11. 当前结论与下一步

counting20步读取、更新及真实保存恢复通过；[full20](../smoke-orig80k-full-0925/README.md)的成功另有独立记录。这两次可读性检查不替代正式输入、上游轨迹、并跑和perf闸门。下一步按明确版本锚点完成这些检查，并等待两项用户决定；全部必要闸门满足前不把20步PASS扩写为80k已具备全部起跑证据。

本目录已按白名单归档小记录和日志并完成必要格式规范化及核验，见第12节。本组以`commitV11.11`配对收尾，不预填最终提交SHA；没有清理checkpoint，起跑Beta和源记录保持其原有含义。

## 12. 已归档文件清单与格式核验

本目录`records/`已包含**16个文件，共340204 B**，不含README和launch正文。来源、ignored暂存与正式目标的字节数/SHA、读回核验及凭据筛查封装在[archive_checks.json](records/archive_checks.json)，该最终清单SHA为`7625a23a55ad539468b78513f57ed6bf4502d5477cd94e8941eb3853dcaf7762`。

| 已有源 | 正式归档文件 | 内容 |
|---|---|---|
| launch/runtime/run_meta | [launch.actual.json](records/launch.actual.json)、[runtime.json](records/runtime.json)、[run_meta.json](records/run_meta.json) | 实际解析配置、设备及argv |
| metrics | [metrics.jsonl](records/metrics.jsonl) | step0五标量 |
| final三份记录 | [start.json](records/final/start.json)、[final.json](records/final/final.json)、[checkpoint_wait_done.json](records/final/checkpoint_wait_done.json) | 身份、尾窗、EMA及异步保存完成 |
| 完成器结果与日志 | [completion.json](records/completion.json)、[completion.summary.log](records/completion.summary.log) | 真实恢复判定 |
| GPU采样与错误流 | [gpu.csv](records/gpu.csv)、[gpu.csv.err](records/gpu.csv.err) | 采样覆盖证据，错误流为空 |
| driver/wrapper原始日志 | [train.summary.log](records/train.summary.log)、[wrapper.summary.log](records/wrapper.summary.log) | 清洗后的过程与退出记录 |
| checkpoint小型元数据 | [_CHECKPOINT_METADATA](records/checkpoint_19/_CHECKPOINT_METADATA) | 原生保存提交时间 |
| controller相关小记录 | [controller.count20.snapshot.json](records/controller.count20.snapshot.json) | 本run的一次读取快照 |
| 两阶段归档核验 | [archive_checks.json](records/archive_checks.json) | 暂存检查与正式文件映射 |

归档准备阶段保存12份原字节副本、2份清洗日志和1份controller提取快照，回车归一后删除`%|`类进度中间态；本run实际删除0行，关键行序列完整。原ignored暂存核验SHA为`2a07cf82f30e3050eb7386f4e74f04d0d500477fe7de435237033208e7980c34`，其内容已封装在最终检查清单中，原暂存没有被重写。

正式落地时仅对`.log`副本去掉行尾空格和制表符：`train.summary.log`与`wrapper.summary.log`合计移除**169个行尾空白字符**。原始日志和ignored暂存日志均未修改；所有行在同一行尾规范化后逐行相等，关键判定、时间及指标数值没有变化。最终清单保留规范化前后摘要与目标映射，不能将格式清洗后的文件误记为与原日志原字节相同。

controller于`2026-09-26T01:58:03.169905+00:00`仅读一次，完整源SHA为`02e868d652459b1c857e13c538f76a5278e30c4aabcc0433924b1ec425911b00`，只提取`build_head`、`counting_readability20_result`及对应会话；后续controller更新不回填这次快照。其余小记录按白名单复制并复核，正式归档没有再次训练、恢复或读取大权重。

不归档H5、features、packed数组、checkpoint权重、脚本、yaml或独立`command.txt`/bash载体。权重留在`$RUN_ROOT/19`；生产代码与配置由固定Beta还原，实质命令在正文中记录。本组结果以`commitV11.11`与`commitV11.11Beta`配对收尾，最终提交SHA由Git历史记录，不能替代实际起跑锚点。
