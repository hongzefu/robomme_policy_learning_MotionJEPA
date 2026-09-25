# 4096 八卡生产容量验证

当前为预建档案，尚未启动。本档由[顺序实施总档案](../modul-sweep-20260923-c/launch.md)管理；不得把预建时间当作实际起跑时间。

## 用户决定与范围

用户最终决定：「可以优化取证方式 但要保证改前后都是用的一种取证方式 这个是用户最终决策」。主跑、补跑及对照两侧须通过取证源码摘要和全部设置的一致性检查；任何混用均拒绝。

用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间」。据此执行[0922计划](../../../0922-4096-1024-8gpu-training-plan.md)；4096完整验收后才开始1024的GPU验证与训练。

## 版本、环境与启动

实施前BASE为 `742f2d894d25abe802a26b1c118852b0b7676e44`。起跑锚点为包含本档案的Beta提交，实际完整HEAD、UTC、PID、tmux全名和逐阶段命令在 `v1-store/bench/modul-budget-sweep/modul-sweep-20260923-c/start.json` 与 `events.jsonl` 现场记录。

本机8×A100-SXM4-80GB，`/scratch` 为 `/dev/md0` XFS NVMe RAID。所有过程从clean HEAD启动，源码与依赖保持不变；只读旧源码快照通过PYTHONPATH导入，沿用主副本uv环境，只执行 `uv run --no-sync --project`，不在快照中同步或安装依赖。

全局batch128、worker16、FSDP8、seed42、20步；history文件为 `perceptual-framesamp-modul-64frame-8x8.yaml`。验证仅由启动覆盖缩短步数并关闭W&B；正确性取证使用log1/save1与确定性XLA_FLAGS，容量及测速清除XLA_FLAGS。

初始化仍为pi05_base；lr warmup5000、peak/decay 5e-5、EMA0.999、clip1.0，其余继承 `mme_vla_suite_b128_80k`。

数据为 `v1-store/datasets/4task-v2-1600ep-604f16da/` 下的source、meta与framesamp-8x8；四任务1600集、605611执行样本、1192918帧。norm_stats来自同名train-assets。数据与输出均为本仓库scratch内实体目录，不覆盖已有run。

取证使用 `BENCH_CHECKSUM_WORKERS=8`，仅控制状态读取及哈希线程，数据worker仍为16。原始dtype、C序字节、叶顺序、SHA256和完整状态判据保持不变；取证器CPU及八卡真实传输验证已通过。

## 判据与产物

要求八卡b128/w16/FSDP8真实20步与末步保存、状态step20、现场EMA摘要与真实读回逐叶相同、全叶和固定动作有限、共享内存峰值≤70%。

完整运行记录保留在本批次 `v1-store/`，结果与清洗日志结束后归入本目录；不复制脚本、YAML或权重进Git。实测结论见 [result.md](result.md)。

## 实际起跑与完成

已从 `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb` 正常完成；实际命令、UTC、环境和退出码见[launch.actual.json](records/launch.actual.json)，结论见[result.md](result.md)。上文未启动标记为起跑前记录。
