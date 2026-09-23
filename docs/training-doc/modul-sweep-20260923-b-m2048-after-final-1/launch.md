# 2048 after-final 1步数值回归

本段已完成，以下保留起跑前计划；实际运行信息见末节。本档由[顺序实施总档案](../modul-sweep-20260923-b/launch.md)管理；不得把预建时间当作实际起跑时间。

## 用户决定与范围

用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间」。据此执行[0922计划](../../../0922-4096-1024-8gpu-training-plan.md)；4096完整验收后才开始1024的GPU验证与训练。

## 版本、环境与启动

实施前BASE为 `742f2d894d25abe802a26b1c118852b0b7676e44`。起跑锚点为包含本档案的Beta提交，实际完整HEAD、UTC、PID、tmux全名和逐阶段命令在 `v1-store/bench/modul-budget-sweep/modul-sweep-20260923-b/start.json` 与 `events.jsonl` 现场记录。

本机8×A100-SXM4-80GB，`/scratch` 为 `/dev/md0` XFS NVMe RAID。所有过程从clean HEAD启动，源码与依赖保持不变；只读旧源码快照通过PYTHONPATH导入，沿用主副本uv环境，只执行 `uv run --no-sync --project`，不在快照中同步或安装依赖。

全局batch128、worker16、FSDP8、seed42、1步；history文件为 `perceptual-framesamp-modul-32frame-8x8.yaml`。验证仅由启动覆盖缩短步数并关闭W&B；正确性取证使用log1/save1与确定性XLA_FLAGS，容量及测速清除XLA_FLAGS。

初始化仍为pi05_base；lr warmup5000、peak/decay 5e-5、EMA0.999、clip1.0，其余继承 `mme_vla_suite_b128_80k`。

数据为 `v1-store/datasets/4task-v2-1600ep-604f16da/` 下的source、meta与framesamp-8x8；四任务1600集、605611执行样本、1192918帧。norm_stats来自同名train-assets。数据与输出均为本仓库scratch内实体目录，不覆盖已有run。

取证使用 `BENCH_CHECKSUM_WORKERS=8`，仅控制状态读取及哈希线程，数据worker仍为16。原始dtype、C序字节、叶顺序、SHA256和完整状态判据保持不变；取证器CPU及八卡真实传输验证已通过。

## 判据与产物

要求同预算输入、五标量hex、完整TrainState的201叶摘要一致；20步与独立1步共同覆盖状态0..20。100步档另保存初末原数组和真实checkpoint，按既有6.8e-5动作RMS上限验收。

完整运行记录保留在本批次 `v1-store/`，结果与清洗日志结束后归入本目录；不复制脚本、YAML或权重进Git。实测结论见 [result.md](result.md)。

## 实际运行

实际UTC为 2026-09-23T03:56:51.980314+00:00 → 2026-09-23T04:04:33.468638+00:00，耗时 461.488 秒，退出码0。取证器提交 `5e97d52a6420fdfd85620773b0496526bbf589a4`，训练源码提交 `5e97d52a6420fdfd85620773b0496526bbf589a4`；tmux为 `modul-sweep-20260923T011058Z`。

用户最终决定：「可以优化取证方式 但要保证改前后都是用的一种取证方式 这个是用户最终决策」。 真实命令、身份和环境见[launch.actual.json](records/launch.actual.json)，逐步证据及实测结论见[result.md](result.md)。事后留档不改变原始启动提交。
