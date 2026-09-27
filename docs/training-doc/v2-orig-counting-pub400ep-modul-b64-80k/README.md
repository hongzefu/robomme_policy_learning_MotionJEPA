# v2-orig-counting-pub400ep-modul-b64-80k：正式80k起跑前README（尚未启动）

最新授权：用户明确「确认无误后可以直接开始两个训练」。共同V11.18Beta、clean状态和全部起跑前检查真实通过后，由root直接启动两条既定80k，无需再次请求批准。此前stop-before要求仅作为历史决定保留；检查器本身仍只读并返回STOP_BEFORE_PROD_LAUNCH，root随后独立调用f8 launch。当前两个prod尚未启动。

## 1. 一句话结论与指标速览

**公开纯counting子集正式run的名称与原版配置已确认，当前尚未启动80k；最终确认全部检查无误后按最新授权直接起跑。** 数据、CPU INPUT、P1与完整100步六run/四judge已有各自原版本证据；确定性20已双judge PASS，normal300两训练/两恢复/报告/两receipt共七实测阶段正式与独立验收均PASS；第8预算已正式PASS，最终独立复核已通过，已归档到 `bafecc658c45fbd940fb1655e7057d2b1d2c0593`。本文不代替最终检查实测，也没有本run的训练指标、吞吐、ETA或模型质量结论。

| 项目 | 已确认值或当前状态 |
|---|---|
| run_name | `v2-orig-counting-pub400ep-modul-b64-80k` |
| 目标 | 80000次更新，batch 64，4卡FSDP |
| 数据 | 400集、189035帧、189035执行样本 |
| 固定GPU组 | `4,5,6,7`；与[另一正式run](../v2-orig-16task-pub1600ep-modul-b64-80k/README.md)按4+4并跑 |
| 正式TRAIN_HEAD、UUID、W&B ID、起跑UTC | 均待真实产生并记录，不预填 |
| 正式阶段结论 | 未启动、未验收 |

## 2. 版本与代码状态

本候选接口实现锚点为 `aec86db64e5178e63d9e7f77d3f5bc235db16390`，已包含显式档位/schema2；确定性20实际Beta为 `c71d5255597db2f26930b7b5684a1d5b2994cf75`，其两库正式judge已PASS。normal perf实际Beta为 `70a641a827eb2559c3acb69430f55c1c53babaeb`，前七实测阶段已正式与独立验证通过；该HEAD也不是未来共同V11.18Beta。这两个提交都不是未来正式80k Beta。两个固定run必须共享独立正式Beta和clean状态，运行中冻结源码/uv环境；Beta、UUID、W&B ID和实际起跑时间全部待真实填入。受保护训练代码与依赖须对实际perf来源复核，不能把旧00bd工具接口或“仅文档变化”当作未核实的兼容保证。

证据保留各自原始版本：两正式库从clean `49a333eb18e8d6ff1143bf7871ef7c498ab91579`构建；CPU INPUT的量具/B为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1及当前100步量具/B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`；上游A始终为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。后续提交不能回填为这些运行的源码版本。正式Beta确定后，应复核输入、资产、依赖及计算相关文件指纹；若影响既定100步可比较性，不能直接复用旧结果。

## 3. 启动与配置还原

入口是[run_orig80k.sh](/scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/prod/run_orig80k.sh)的实际5参数接口：`prod <run_name> <gpus> <lib绝对路径> <assets_dir绝对路径>`。正式prod不传步数、batch、学习率、log/save或workers覆盖项；详见[共享launch候选](../orig80k-prod-0927/launch.md)。以下仅为未来版本还原方式，当前占位符不能执行：

```bash
TRAIN_HEAD='<待正式Beta生成后填写完整40位SHA>'
git show "$TRAIN_HEAD":src/mme_vla_suite/training/config.py
git show "$TRAIN_HEAD":src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml
git show "$TRAIN_HEAD":scripts/training/prod/run_orig80k.sh
git show "$TRAIN_HEAD":scripts/training/orig80k_contract.py
git show "$TRAIN_HEAD":uv.lock
```

正式运行使用主仓独立uv环境 `/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv`，每run独立记录、checkpoint、JAX/CUDA/W&B缓存。保留 `PYTHONUNBUFFERED=1`，runner自带pipefail与tee；新会话必须通过冻结f8 guard先保留窗口、持久化身份再释放，完成后取得独立pane回执和父进程真实返回sidecar。命令文件及完整SHA、session/window/pane/PID、实际开始/结束UTC均待真实记录，不从候选命令推造。

## 4. 数据集与划分口径

同一公开16任务集合中的 BinFill、PickXtimes、SwingXtimes、StopCube 各100集。严格子集检查为 `SUBSET_EQ=PASS episodes=400 samples=189035 feature_mismatch=0 sample_mismatch=0`。 建库的来源pin、1024点特征复算最大绝对差0和packed全量字节校验通过；本库source与packed合计实占 `202152935424 B`。正式不新增训练/验证划分，不替换为其他本地数据或历史数据量。参见[本库构建档案](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/dataset-build-doc/4task-counting-pub-400ep/README.md)。

| 输入与资产 | 实体路径或期望摘要 |
|---|---|
| 原始公开H5 | `/scratch/hongze/robomme_data_h5` |
| 库根 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep` |
| 训练packed | 库根下 `framesamp`，格式 `framesamp-4x4-v1` |
| 对应source / episode manifest | 库根下 `source` / `meta/episode_manifest.json` |
| episode manifest SHA256 | `6b309822aa604a31f6eb0dfb5f02d5482573dc5fa11dc5c4f917880aa92ba986` |
| assets父目录 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep` |
| 真正norm文件 | 上行目录下 `robomme/norm_stats.json` |
| norm SHA256 | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |

按用户授权对本counting库重算norm；文件由建库 `build_sizes.json.self_norm_stats_sha256`绑定。计算实际消费1476批、188928个样本，末107个样本按原工具drop_last未进入统计；这是已归档计算口径，不改写为覆盖全部189035个执行样本。 `assets-dir`传父目录，不能再多加一层 `robomme`。pi05初始化固定为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models/openpi-assets/checkpoints/pi05_base/params`。

CPU INPUT已按 `host_numeric_exact_motion_none_v1`通过本库400集、189035执行样本、2400边界样本、104批与2个epoch索引检查；104批不是对全量两epoch样本内容解码的声明。原始dtype/raw SHA及ALL字段均保留；仅精确数值比较和四个获准missing/None键等价，终点为 `collate_before_jax`。训练状态和标量逐位要求不因此放宽，参见[INPUT结果](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-equiv-0925/result.md)。

## 5. 关键超参

这些值来自 `config.py::_CONFIGS[mme_vla_suite]`、`TrainConfig`默认和既定history；prod只提供身份/路径/history覆盖，全部训练超参沿用原版。

| 参数 | 正式值 |
|---|---|
| 更新步数 / batch / seed | 80000 / 64 / 42 |
| FSDP / dataloader workers | 4 / 4 |
| warmup / peak_lr | 10000 / 5e-5 |
| decay_steps / decay_lr | 100000 / 5e-5；peak与decay相同，warmup后为5e-5 |
| optimizer / clip | AdamW / 1.0 |
| EMA | 0.999 |
| log_interval / save_interval / keep_period | 100 / 10000 / 10000 |
| history | `perceptual-framesamp-modul.yaml` |
| history SHA256 | `823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a` |
| memory | budget512，每帧4×4=16 token，最多32帧；modulation，memory_token_dim1024 |
| history其他字段 | num_views1、streaming_obs_horizon16、mean pooling；不把该horizon误写为32 |
| 初始化 / W&B | pi05_base / 开启，project_name=openpi |
| x64 / 显存比例 | 实际解析必须禁用x64 / XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 |

正式入口保持W&B online，不启用perf包装或对拍摘要器；实际命令显式unset `ORIG80K_TIMING_EQ_PROFILE`、`ORIG80K_SMOKE_EQ_MODE`、`XLA_FLAGS`、`TRAIN_TIMING_STEPS`及CPU平台遗留。normal描述符必须明确为 `{"name":"normal","xla_flags":""}`，不能缺字段默认接受。继承的 `JAX_ENABLE_X64`不被重置，实际True由runner拒绝。

已完成的CPU参数预览来自70a，原报告SHA `4cf995473747d6ad9e462e4dda710f433fe6409b01e71566c54b8b2c3643f7a6`，状态为PARAMETER_PREVIEW_PASS：两个正式run均normal/80000步/x64 False，主机实际返回0，六正式输出仍不存在。它不包含预算/启动许可；未来共同V11.18Beta仅文档变化时按源文件与依赖链复核，最终只读/CPU检查见共享launch。

## 6. 硬件、资源与耗时

已核实环境为AWS单机8×NVIDIA A100-SXM4-80GB，`/scratch`位于 `/dev/md0` XFS本地NVMe RAID；这不是正式起跑时设备空闲或磁盘足够的保证。该run固定GPU `4,5,6,7`，两个正式run按4+4尽量并行，分别保持b64/fsdp4/workers4，共享CPU、内存、/dev/shm及底层磁盘。

本run实际开始/结束UTC、GPU UUID、PID、耗时、稳态吞吐、GPU利用率均待正式记录。不能用P1/A100摘要耗时、旧20步或历史其他batch的速度推算本run。性能依据来自已经完成的本轮300步并跑实测，报告稳态、GPU均值/0%占比、慢步分层、主机及共享内存，并保留0.5秒磁盘采样盲区和保守余量。

## 7. 已有过程证据与正式训练待记录项

[P1正式记录](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-equiv-0925/p1-result.md)仅证明入口、分段、finite与来源；[100步正式结果](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-equiv-0925/train100-result.md)保留六run/四judge逐位通过、S1/S3真单跑与S2并跑主机记录窗口证据。它们使用原00bd/ecf锚点，不能改成未来prod版本或作为性能结论。

[旧normal20](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-timing20-0926/result.md)与[追加off复验](/scratch/hongze/robomme_policy_learning_MotionJEPA/docs/training-doc/orig80k-off-repeat-0927/result.md)的差异和FAIL保留。用户已同意后续20仅正确性对照采用既有100步确定性档；其两库正式judge已通过，仍不能据此宣称normal数值重复性根因已解决。未来perf和本prod继续normal，训练逐位判据未放宽。

本正式run的每100步五标量、尾99步、保存/异步wait、GPU采样及W&B记录均未产生，不复制任何验证run指标充当本run结果。

## 8. 正式完成验收与评估范围

正式完成要求checkpoint恰为 `10000,20000,30000,40000,50000,60000,70000,79999`共8份，普通结构化日志恰为 `range(0,80000,100)`共800条；尾窗恰为79901…79999共99步，五标量完整且有限。末态必须 `state_step=80000`、`loop_step=checkpoint_step=79999`，run/HEAD/UUID与异步 `checkpoint_wait_done.json`一致。

[完成器](/scratch/hongze/robomme_policy_learning_MotionJEPA/scripts/training/tests/check_orig80k_completion.py)须真正加载79999/params，验证全部EMA叶有限且dtype/shape/字节摘要与本run训练末态相同，并核对GPU采样和唯一driver成功终态；同时独立guard退出链通过。具体prod CLI见launch。当前没有任何本run完成器结果，不填EMA叶数或最终权重大小。本轮不做策略评估，不把训练有限性当作模型质量结论。

## 9. 用户决定与本轮边界

用户目标为「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」，并明确「开始实现该计划 有问题立刻问用户 不要自己决策」。公开16×100全集和4×100 counting子集、两组run名称、modul及原版超参已确认。

用户要求「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」和「允许这四键缺失与 None 等价」；四键仅 `motion_emb/motion_pos/motion_mask/mem_order`，其他字段及非None仍严格。用户「允许仅对拍关闭 W&B」，只作用于P1/100；正式80k保持开启。

此前「继续工作 一路做到起泡前 有问题问用户」限定止于起跑前；最新「确认无误后可以直接开始两个训练」已授权检查无误后直接启动。最新「尽可能并行做」「你有8张卡」按4+4用于允许并行阶段；100步S1/S3仍须真单跑以保留单跑/并跑判据，不能为利用率跳过。两个正式launch须同Beta、同clean状态一并就绪，最终检查通过后按最新授权直接起跑。

## 10. 计划外事件、已批准限制与处置

旧INPUT与P1的最终direct append可能完整写出日志0后自身返回非零，历史最外wait/pane退出码未留存，现不可追补。当前没有真实任务失败的新证据；各子命令返回、记录内容和对应已核验来源仍保留，不把缺失最外退出证据写成独立实退0。

用户对P1明确「补记P1限制，补独立退出取证后继续（推荐）」，对INPUT另明确「同样补记限制，沿用INPUT结果（推荐）」。参见[INPUT原取证补充](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/input-outer-exit-scope-supplement-20260926.md)（SHA256 `bd8c37ba659b2dca0db52f1ebf54c5ff9e925416acbe9e7f8b0e3b46dfb2fd8a`）和[批准说明](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/input-outer-exit-scope-approved-20260926.md)；原记录、日志和包不改。六个CPU collector/judge实际子命令返回0及manifest/INPUT_EQ内容证据按批准沿用；这不豁免后续任何轨迹、恢复或预算判据。

后续强制冻结f8 guard、新唯一session、命令与identity SHA绑定、原生退出回执及外部实际返回sidecar。失败保留现场，只停止依赖阶段；不自动删除、改名、重训、修改超参或降低逐位判据。

## 11. 起跑前闸门与下一步

| 闸门 | 当前证据范围 |
|---|---|
| 两库来源/构建/packed/counting子集 | 已完成，保留49a实际证据 |
| CPU INPUT、P1 | 按用户批准沿用内容与来源，旧最外退出限制保留；P1不冒称轨迹等价 |
| 100步上游及当前、S1/S3单跑与S2并跑 | 六run/四judge已通过，实际00bd/ecf锚点与独立退出、主机记录窗口保持原档案 |
| 确定性20 | 实际Beta c71，已完成两库正式judge/四恢复/六独立退出验收 |
| normal300 perf、真实恢复、联合报告及measurement | 实际70a前七阶段已PASS；保留显式normal/schema2、0.5秒采样及独立退出；第8预算与最终索引待回填 |
| 三类余量与共用预算 | B档16/32/64GiB已确认；同一预算路径/SHA已固定，正式预算PASS，最终归档bafecc6已完成 |
| 共同正式Beta/clean、现场资源与最终复核 | 待共同V11.18Beta及最终检查通过，随后root直接起跑 |

正式预算仍为 `max(300 GiB, 两侧各8份checkpoint实测分配量+增长余量+两侧采样额外量下界+正采样余量+正日志/cache余量)`。0.5秒离散采样不代表连续峰值，保守余量不缩水、不自选数值。逐项既定要求见[起跑前清单](../orig80k-prod-0927/checklist.md)，其待填项没有被本候选放行。

B档已由用户确认：每份checkpoint按16GiB、两侧共16份保留，即checkpoint_total_bytes=274877906944；增长余量84798963712B；保存采样余量34359738368B（32GiB），另加真实E合计5531541504B，save_temporary_peak_bytes=39891279872；日志/cache余量68719476736B（64GiB）。三项合计383488663552B，约357.151649GiB。预算实体固定为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-perf300-results-20260926T232339Z/prod-disk-budget.json`，真实budget SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`，margin-source SHA `09ea4e4739e11e2851a013b00e9a2f3322d0b6d9b7859f9b749495be90d375bf`。第8阶段于04:20:41完成正式预算PASS，所需383488663552B、当时可用708412715008B，原生/capture/父宿主均0；该可用量不是未来正式起跑保证，最终独立复核已通过，已归档到 `bafecc658c45fbd940fb1655e7057d2b1d2c0593`。

## 12. 归档文件清单

当前只有本README、起跑前清单和[共享launch候选](../orig80k-prod-0927/launch.md)，它们位于ignored准备目录；既有数据/INPUT/P1/100证据通过上文真实链接引用，没有复制成当前run结果。

未来正式目标为 `docs/training-doc/v2-orig-counting-pub400ep-modul-b64-80k/`；正式起跑前先落README版本/启动配置两节与共享launch完整正文，同一Beta锁定两个run，并更新总索引。待实际产生后才归档 `launch.json/launch.actual.json/runtime.json/run_meta.json`、`metrics.jsonl`、final三份JSON、completion结果与清洗日志、GPU采样、W&B ID、预算/资源/guard身份及退出sidecar/检查链、全程及稳态实测记录；各文件路径、大小、SHA和清洗关键行完整性按真实内容填写。

只归档Git无法还原的测量记录；不复制独立.sh/.yaml/.command脚本载体、venv、缓存或权重入docs。配置与命令由正式Beta和本文代码块还原，checkpoint留在本run的 `v1-store/train-runs/mme_vla_suite/v2-orig-counting-pub400ep-modul-b64-80k`。本轮未创建该输出根、records根或driver日志。
