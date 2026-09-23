# 4096 有界输入、在线装配、padding、RoPE及初始化

当前为预建档案，尚未启动。本档由[顺序实施总档案](../modul-sweep-20260923-c/launch.md)管理；不得把预建时间当作实际起跑时间。

## 用户决定与范围

用户最终决定：「可以优化取证方式 但要保证改前后都是用的一种取证方式 这个是用户最终决策」。主跑、补跑及对照两侧须通过取证源码摘要和全部设置的一致性检查；任何混用均拒绝。

用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间」。据此执行[0922计划](../../../0922-4096-1024-8gpu-training-plan.md)；4096完整验收后才开始1024的GPU验证与训练。

## 版本、环境与启动

实施前BASE为 `742f2d894d25abe802a26b1c118852b0b7676e44`。起跑锚点为包含本档案的Beta提交，实际完整HEAD、UTC、PID、tmux全名和逐阶段命令在 `v1-store/bench/modul-budget-sweep/modul-sweep-20260923-c/start.json` 与 `events.jsonl` 现场记录。

本机8×A100-SXM4-80GB，`/scratch` 为 `/dev/md0` XFS NVMe RAID。所有过程从clean HEAD启动，源码与依赖保持不变；只读旧源码快照通过PYTHONPATH导入，沿用主副本uv环境，只执行 `uv run --no-sync --project`，不在快照中同步或安装依赖。

执行 `check_32frame_modul.py` 的 yaml、frames、guards、assembly、pad、online、collate、oracle、init。assembly只读取1600集首/中/末及前20批去重后的索引；实际读取量与索引清单留在JSON。init使用八卡FSDP8。

数据为 `v1-store/datasets/4task-v2-1600ep-604f16da/` 下的source、meta与framesamp-8x8；四任务1600集、605611执行样本、1192918帧。norm_stats来自同名train-assets。数据与输出均为本仓库scratch内实体目录，不覆盖已有run。

取证使用 `BENCH_CHECKSUM_WORKERS=8`，仅控制状态读取及哈希线程，数据worker仍为16。原始dtype、C序字节、叶顺序、SHA256和完整状态判据保持不变；取证器CPU及八卡真实传输验证已通过。

## 判据与产物

要求有界真实样本逐键shape/dtype/字节完全一致、实际训练索引前缀一致；短历史补零与mask垃圾不变性、独立RoPE oracle、61叶初态检查通过。

完整运行记录保留在本批次 `v1-store/`，结果与清洗日志结束后归入本目录；不复制脚本、YAML或权重进Git。实测结论见 [result.md](result.md)。
