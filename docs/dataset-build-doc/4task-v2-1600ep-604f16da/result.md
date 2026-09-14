# 1600 集正式建库与训练可读性验收通过

本轮计划全部交付并验收：1600 条 primary，1,192,918 个源时步，605,611 个执行样本。合并 H5、源摘要、全部 H5 数值/结构对拍、独立输入指纹、SigLIP、finalize、两档 framesamp 全量验证和跨网格检查均通过，新 norm_stats 已生成并核验有限。两档真实 batch 及 20 步真实训练也已通过，checkpoint 收尾成功，临时测试 run 已清理。

## 合并实测

| 任务 | episode | timestep | 原 H5 总字节 | 合并文件字节 |
|---|---:|---:|---:|---:|
| BinFill | 400 | 513858 | 341043214908 | 341047005852 |
| RouteStick | 400 | 214900 | 143317131528 | 142668708960 |
| VideoRepick | 400 | 304453 | 202977750392 | 202058926776 |
| VideoUnmaskSwap | 400 | 159707 | 106477388744 | 105994839080 |

源总字节 793,815,485,572 B。H5 容器大小不要求与原单集文件之和相同；内容与结构是否一致由独立 full 对拍判定。四份 map 均有 400 行 `h5_sha256_local`，全部等于生成侧清单摘要，且 `role` 全为 primary。

阶段 1/2 从 `2026-09-14T17:50:56Z` 到 `18:15:03.823676Z`，共 1447.82 秒；前检 9 秒，merge 约 1438.82 秒（23 分 58.82 秒）。会话正常结束，`EXIT_CODE=0`，没有强杀。启动版本、实际命令、进程归属及输出目录见 [launch.md](launch.md)。判定行原文如下。

```text
PLAN_OK tasks=4 selected=1600 role=primary:1600 max_timesteps=2304 prechecks=PASS
MERGE_TASK=PASS task=VideoUnmaskSwap episodes=400
MERGE_TASK=PASS task=RouteStick episodes=400
MERGE_TASK=PASS task=VideoRepick episodes=400
MERGE_TASK=PASS task=BinFill episodes=400
EXIT_CODE=0
```

## 与在跑训练共处

以计划指定的 874.0 samples/s 为参考，暂停线为 865.26。epoch 56 实测 871.4952877793485 samples/s、875.6627955436707 秒，epoch 57 为 872.0038647435658 samples/s、875.1520845890045 秒，分别较参考低约 0.29% 和 0.23%，均未触发暂停。epoch 57 完整时间窗落在合并阶段内。这里只对训练吞吐作共处检查，不引用训练侧相对 loss 作为效果证据，也不由 GPU 单次利用率推断没有瓶颈。

epoch 58 在 full 校验阶段结束，吞吐 867.9841933298775 samples/s、879.2049508094788 秒，较参考低约 0.69%，未触发暂停。其窗口同时覆盖合并尾段与并行对拍，不能单独归因于其中一段。

epoch 59 为 871.9092604383924 samples/s、875.2470407485962 秒，较参考低约 0.24%，同样未触发暂停。该窗口覆盖数值对拍尾段和输出文件摘要计算。

独立输入指纹阶段的 epoch 60 为 870.8126569688189 samples/s、876.3492283821106 秒，epoch 61 为 872.9121273456386 samples/s、874.241491317749 秒，分别低约 0.36% 和 0.12%，仍均通过阈值检查。

## 全量数值与结构对拍

阶段 3/4 于 `2026-09-14T18:15:24Z` 从 `eb25f839c836de584f3a982c790df292304ba25a` 启动，工作区干净，会话 `v2b-verify-20260914T174147Z`，pane PID/PGID `1067561`。48 个 `(task, episode 区间)` 分片的最后一条 `VERIFY_SHARD` 于约 `18:26:34.97Z` 写出，合计覆盖 1600 条，所有 `mismatches=0`；数值/结构部分约 671 秒。随后四文件并行 SHA256 约 860 秒，`18:40:55Z` 进入 scan。扫描耗时约 188 秒，整个阶段 3/4 于 `18:44:03.848191Z` 正常结束，合计约 1719.85 秒，`EXIT_CODE=0`。监视只核验 epoch 58 起的新增吞吐记录，未发送暂停信号。

```text
MERGE_VERIFY=PASS level=full tasks=4 episodes=1600 timesteps=1192918 role=primary:1600 mismatches=0
episode=1600  timestep=1192918  执行样本=605611
sha256=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918
EXIT_CODE=0
```

完整扫描的执行样本数为 605,611，替代计划中约 617,463 的抽样外推，不是丢失 episode。`first_execution_step` 逐帧读取 demo 前缀，因此全量 scan 不是计划预估的 4–8 秒；本轮保留原入口与语义，记录实际耗时。canonical_order 为 BinFill、RouteStick、VideoRepick、VideoUnmaskSwap；global episode 区间依次为 0–399、400–799、800–1199、1200–1599。

MERGE_DONE 的 level、source_pin、四份 map 的当前 SHA256、选取参数、1600 条本地源摘要计数和四个输出摘要已再次核对。该完成标记及正式 episode manifest 已入 records，后续输入指纹须与这四个输出摘要一致。

## 独立输入指纹

阶段 5 从 `2026-09-14T18:46:07Z` 的 `e193c6dc1b4a8cbd6779dbfb2b1f6aa077686c6d` 启动，当时工作区干净；会话 `v2b-hash-20260914T174147Z`，pane PID/PGID `1087992`。原 `finalize_checks.py hash-inputs` 串行重读全部四个 H5，未复用 MERGE_DONE 内的摘要；于 `19:18:33.385790Z` 正常结束，共 1946.39 秒，`EXIT_CODE=0`。后验逐项检查 `input_manifest.files` 的文件集合、大小和 SHA256，全部与 MERGE_DONE 和实体文件一致，`INPUT_MANIFEST_IDENTITY=PASS files=4`。

该阶段期间新增的 Git 提交仅为后续训练可读性启动文档，实际消费代码未改变。GPU 4–7 在起跑 SigLIP 前再次实测空闲；磁盘可用约 2.79 TB，仍足以容纳后续 source 和两档 packed。下一步 SigLIP 起跑至 finalize 通过之间本轮不提交。

## 既有链路差异的准确口径

比较旧 400 集 SigLIP 起跑提交 `8093ebda23ec566533067e319bab506baaf80de5` 与本轮验证起跑提交 `eb25f839c836de584f3a982c790df292304ba25a`，`src/mme_vla_suite/dataset_builder/` 无差异。既有 `pack_framesamp_store.py` 是布局参数化：模块常量改由 `spec.*` 查表，默认仍为 `framesamp-4x4-v1`；另外保留磁盘预检下限和 resume 布局不符报错。实际 `git diff --numstat` 为新增 50 行、删除 37 行，修正计划文字的计数；不把“布局参数化”写成所有脚本零改动。

本轮新增合并脚本与测试，并按用户“允许最小改动，实现四文件并行”修改 `finalize_checks.check_inputs` 的摘要调度。没有修改 builder、dataloader、训练模型或全局超参。计划记载的 `is_completed` 后缀沿用上一帧 subgoal 属既有行为，本轮不改；计划中的 37,219 步统计不是本轮重新统计的结果。

## 正式 SigLIP 与 finalize

SigLIP 于 `2026-09-14T19:22:45Z` 从 `cd99ce44872e8730b9383cca5add8c4b0a2dac88` 起跑，工作区干净，会话 `v2b-siglip-20260914T174147Z`，pane PID/PGID `1110303`，采样器 PID `1110324`。GPU 固定 4–7，启动使用现有 `--require-free-mib 70000` 参数核实空闲显存。于 `20:28:50.691321Z` 正常结束，3965.69 秒（约 66 分 6 秒），sampler 由本轮精确 PID 清理。四片实际处理 episode 数为 399/404/397/400，总和 1600；源时步总和 1,192,918。

```text
STAGE_DONE stage=siglip workers=4 items=1600 elapsed=3965s
feature 目录缺失=0  pkl 实得=605611 期望=605611
sidecar=4 覆盖 episode=1600 残留 claim=0
FINALIZE_EXIT_CODE=0
EXIT_CODE=0
```

finalize 会话为 `v2b-finalize-20260914T174147Z`，从 `0233f17b1f91a875b0005f66189b0757f1e530db` 于 `20:29:38Z` 启动，pane PID/PGID `1217259`。保留 `--input_level sha256 --spot_check 1024`，仅重算抽检时使用 GPU 7。五项检查全部通过，于 `20:44:48.679535Z` 退出 0，耗时 910.68 秒；`source.stats` 最终为 `execution_samples=605611`、`total_samples=1192918`。

### 版本并行事件与检查口径

本轮遵守 SigLIP 至 finalize 之间不提交。HF 任务在此期间产生 `73928536b3c0f22e0d74184070e1e777562c8a30` 和 `0233f17b1f91a875b0005f66189b0757f1e530db`；diff 仅涉及 HF 导出脚本与其档案，实际消费代码和依赖无差异。`build_shard.main` 在结束处采集 Git HEAD，因此四个 worker 均记录 `0233f17b…`，与启动 `cd99ce4…` 的采集时刻不同。

代理的一次附加检查错误地要求“结束 HEAD 等于启动 HEAD”，触发 `AssertionError`。随后核对四片结束版本唯一、处理数与覆盖数正确、实际代码未变；原 finalize 的跨片一致性和全部数据守卫均通过。没有修改生产守卫，也没有改写 worker 记录。事件详情保存在 `version-precheck-event.json`，起跑原始记录保存在 `siglip.start.json`。

### 吞吐共处与采样

SigLIP 启动时已有的 epoch 62 只作为基线：原始监视首行虽标 PASS，其时间窗口早于本阶段，不能用于证明共处。真正的新 epoch 63/64/65/66 吞吐依次为 872.5478001944947、871.8584157365709、871.7257345026377、872.1236876850226 samples/s，均未低于 865.26；finalize 阶段 epoch 67 为 872.7780757243082，同样通过。整个阶段没有发送暂停信号。

GPU 记录请求间隔 500 ms，实际平均约 0.50087 秒；以启动后 120 秒为预热区间，稳态窗口为 `19:24:45.471Z` 至约 `20:28:50.157Z`，每卡 7677 条读数。下表为时间加权利用率均值、0% 读数占比与显存峰值；相邻 NVML 读数可能来自同一内部窗口，不视为独立证据。本轮未采逐步耗时，不据这些量给出慢步/其他步分层或瓶颈结论。

| GPU | 利用率均值 | 0% 读数占比 | 显存峰值 MiB |
|---|---:|---:|---:|
| 4 | 37.40% | 1.211% | 62907 |
| 5 | 37.95% | 1.016% | 62907 |
| 6 | 37.42% | 1.172% | 62907 |
| 7 | 37.58% | 1.042% | 62907 |

`_shard*.json` 中源时步的稳态处理速率为每 worker 76.06–77.16 step/s，初次 episode 已由原脚本统计口径排除。这些是当前 AWS 本地 NVMe RAID、四 worker、现有 SigLIP 入口的实测，不与旧 NFS 数字混比。

## 归档文件与恢复指引

records 已包含来源 pin、选取计划、全部四份 map、任务目标变体、`merge.measurements.json`、MERGE_DONE、正式 episode/input manifest、source stats/provenance、四片元数据、两档 store_meta、norm_stats 摘要、全部建库阶段的清洗日志与吞吐监视、SigLIP GPU 原始采样及汇总、版本检查事件。完整日志保留在 v1-store，清洗日志保留所有完成与退出判定行。20 步可读性结果见 [训练检查档案](../../training-doc/v2b-read20-20260914T174147Z/result.md)。

## 两档 packed 与独立统计量

三个阶段均从 `3b2864f7441a965250bbf18de7a6f00a63b08ab7` 的干净工作区启动，使用原有入口和 CPU 隔离环境，退出码均为 0。

| 阶段 | 起跑 UTC | 结束 UTC | 墙钟秒 | 结果 |
|---|---|---|---:|---|
| 4×4 pack + verify | 2026-09-14 20:49:33 | 20:53:02.724 | 209.72 | 32 个 part，全 1,192,918 行零差异 |
| norm_stats | 2026-09-14 20:55:08 | 21:00:32.208 | 324.21 | 4731 个 batch 完成，统计量有限 |
| 8×8 pack + verify + xgrid | 2026-09-14 21:01:13 | 21:05:16.610 | 243.61 | 32 个 part，全行及独立跨网格检查通过 |

```text
VERIFY_PACK=PASS scanned=1192918 mismatches=0
VERIFY_PACK=PASS scanned=1192918 mismatches=0
IMAGE_NPY_SPOT=PASS frames=512 mismatches=0
MOTION_POS_XGRID=PASS t=2304 npy=64 mismatches=0
```

两行 VERIFY_PACK 依次对应 4×4 和 8×8；两份 store_meta 均为 `status=verified`，pack 锁均已释放，manifest 为 `4cd5a170…`。8×8 使用 `--layout framesamp-8x8-v1 --reader decode`；512 帧源 NPY 对拍不依赖 packer 的规格表，时间码同时覆盖两档完整位置表。

统计量位置为 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`，SHA256 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。state/actions 的原始统计维度均为 8，mean/std/q01/q99 全有限，std 非负且 q01 不大于 q99。原脚本按 batch 128 和 `len(dataset)//batch_size` 迭代 4731 个完整批次，实际纳入 605568 个样本，末尾不足批次的 43 个样本未纳入这次统计；训练数据仍保留全部 605611 个执行样本，未改脚本口径。

epoch 69 的吞吐为 869.0484657440462 samples/s、878.1282403469086 秒，较参考低约 0.57%，未触发暂停；其窗口覆盖 packed、统计量及相邻阶段，不单独归因于某一步。

最终复核另从原训练日志读取到 epoch 70，吞吐为 870.7770815382041 samples/s，较参考低约 0.37%，仍高于暂停线。该窗口覆盖 20 步检查与相邻时间段，不将变化单独归因于短测。完整性能字段保存在 `coexistence.train-perf.json`。

## 产物体积与真实 batch

以下为实测 `du -sb`，含各目录内元数据，未把日志/Git 计入主要数据体积。

| 产物 | 字节 |
|---|---:|
| 合并 RAW 目录 | 791770288140 |
| source | 958903249848 |
| framesamp 4×4 | 78349610344 |
| framesamp 8×8 | 313226571153 |
| 新统计量目录 | 2143 |

两档分别使用对应 history 配置，经真实 dataloader 加载新统计量取得 batch 64。state `(64,32)`、actions `(64,20,32)` 均为 float32 且有限；static_image_emb 为 `(64,512,2048)` bfloat16；`motion_emb/motion_pos/motion_mask/mem_order` 均为 None。CPU 输入检查使用 worker 0、seed 42；20 步真实训练使用默认 worker 4 并已通过。证据保存在训练检查档案的 `records/batch4.json` 与 `records/batch8.json`。

## 20 步训练与最终状态

从干净 HEAD `81a6c7580507f82a4ad19cf4c651ccd8a3c80336` 于 `2026-09-14T21:11:15Z` 启动，使用 GPU 4–7、默认 batch 64 / worker 4 / FSDP 4 / seed 42、关闭 motion 的 4×4 history 配置和本轮新统计量。20 个 step（0–19）全部完成，逐步 loss/梯度记录有限，checkpoint 19 保存收尾成功，`21:14:36.544738Z` 退出 0，耗时 201.54 秒。checkpoint 内统计量 SHA256 与输入文件一致。

本检查用于验证可读取和可训练，不据 20 步结果判断策略质量。数据集、RAW、源 extracted、新统计量及日志保留；28 集冒烟的两个数据目录、20 步检查的 checkpoint/bench 临时目录均已在身份核对和归档后清理。没有清理其他任务的会话或产物。

用户已决定本轮先不做 motion 阶段，私有 sim 的评估适配另案处理；本档案的完成范围为计划明确的阶段 1–10、来源补档、冒烟及训练可读性验证。

## JSON 档案身份索引

下表给出每份 JSON 与本轮库的关联。源清单是原始 MANIFEST（前缀 df992cdf），库清单是 episode_manifest（前缀 4cd5a170）；并非每份文件都内嵌这两个字段，关联由本轮启动记录与校验建立。GPU 和原训练性能文件只表示本轮资源监视关联，不属于训练样本。

| 文件 | 关联库 | 清单前缀 |
|---|---|---|
| [MERGE_DONE.json](records/MERGE_DONE.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [_shard0of4.json](records/_shard0of4.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [_shard1of4.json](records/_shard1of4.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [_shard2of4.json](records/_shard2of4.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [_shard3of4.json](records/_shard3of4.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [coexistence.train-perf.json](records/coexistence.train-perf.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [episode_manifest.json](records/episode_manifest.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [final-check.json](records/final-check.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [framesamp-8x8.store_meta.json](records/framesamp-8x8.store_meta.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [framesamp.store_meta.json](records/framesamp.store_meta.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [input_manifest.json](records/input_manifest.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [merge.measurements.json](records/merge.measurements.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [merge_plan.json](records/merge_plan.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [norm_stats.sha256.json](records/norm_stats.sha256.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [record_dataset_BinFill_episode_map.json](records/record_dataset_BinFill_episode_map.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [record_dataset_RouteStick_episode_map.json](records/record_dataset_RouteStick_episode_map.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [record_dataset_VideoRepick_episode_map.json](records/record_dataset_VideoRepick_episode_map.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [record_dataset_VideoUnmaskSwap_episode_map.json](records/record_dataset_VideoUnmaskSwap_episode_map.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [siglip.gpu-summary.json](records/siglip.gpu-summary.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [siglip.start.json](records/siglip.start.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [source.provenance.json](records/source.provenance.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [source.stats.json](records/source.stats.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |
| [source_pin.json](records/source_pin.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [task_goal_variants.json](records/task_goal_variants.json) | 4task-v2-1600ep-604f16da | 源 df992cdf… |
| [version-precheck-event.json](records/version-precheck-event.json) | 4task-v2-1600ep-604f16da | 库 4cd5a170… |

JSONL 的阶段吞吐监视、SigLIP GPU CSV 和各阶段清洗日志同属本轮库；训练检查的 JSON/JSONL 记录位于独立训练档案，并明确绑定同一 4cd5a170 库清单与 856c75ea 统计量摘要。
