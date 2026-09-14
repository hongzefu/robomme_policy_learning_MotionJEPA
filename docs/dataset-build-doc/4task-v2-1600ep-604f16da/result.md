# 1600 集正式建库进行中

当前已完成物理合并、full 验真及独立输入指纹：1600 条 primary 的本地源 SHA256 全部与固定 MANIFEST 相符，四任务各 400 条，源 timestep 合计 1,192,918。四个合并 H5 合计 791,769,480,668 B；48 个分片的全部数值与结构对拍零差异，MERGE_DONE 与随后独立计算的 input_manifest 四文件摘要全部一致。正式清单确认 605,611 个执行样本。SigLIP、framesamp、norm_stats 和训练可读性尚未执行；本文件不将阶段性通过写成最终交付通过。

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

## 已归档与待补

records 已包含来源 pin、选取计划、全部四份 map、任务目标变体、`merge.measurements.json`、MERGE_DONE、正式 episode/input manifest、合并/full 验真/输入指纹清洗日志和相应吞吐监视记录。完整日志保留在 v1-store，清洗日志保留所有完成与退出判定行。后续追加两档 store_meta、norm_stats 摘要、各阶段日志及 20 步可读性结果，并更新本文件为最终结论。
