# 相同取证方式下继续4096→1024顺序验证与训练

本批次尚未起跑。执行范围仍为[0922计划](../../../0922-4096-1024-8gpu-training-plan.md)，两个正式run名称保持不变。用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间」。

用户最终决定：「可以优化取证方式 但要保证改前后都是用的一种取证方式 这个是用户最终决策」。同一对照组的改前、改后、开关最终记录和全部补跑必须使用同一取证器源码及设置。检查器从完整Git提交读取四份取证相关源码，计算SHA256，并比较线程数、三个记录开关、间隔及两类附加步集合；不一致即拒绝。取证仍使用b批次的固定八线程实现，没有再次改变拷贝或哈希算法，20+1步和201叶逐位判据保持。

## 版本与基线复用条件

b批次起跑提交为 `5e97d52a6420fdfd85620773b0496526bbf589a4`，改前源码为 `742f2d894d25abe802a26b1c118852b0b7676e44`。本批次的起跑锚点为包含此档案的下一Beta提交，实际完整HEAD、UTC、PID、tmux名称在 `v1-store/bench/modul-budget-sweep/modul-sweep-20260923-c/start.json` 中记录。起跑前主副本必须干净；运行中不改源码、依赖或HEAD。

仅在b批次三组2048的20+1步及两个完整比较全部通过后，才允许复用整组原记录。 `Workflow.reuse_baseline` 检查提交之间只变化列明的验收器、调度器、测试与文档；取证源码、训练源码、配置和依赖任何变化均拒绝。还须比较完整环境指纹，重算manifest/store_meta/norm三个摘要，并用新检查器重读全部六份记录。记录本轮采用的基线run、两侧版本、所有记录文件摘要和指纹比对结论到 `baseline-reuse.json`。未满足条件则停止，不拼接不同方法的记录。a批次串行前缀不参与正式等价结论。

此轮还修正 `check_32frame_modul.py::sample_indices_oracle` 的参考公式。实际选帧一直使用float64步长后取整；之前的理想整数公式在64帧档、真实帧号0–2303中有67处不同，例如t=153。验证器现在保留既有浮点舍入及末端固定；三个预算全域6912例一致，实际采样、在线实现、数据、模型与训练数学未改。前后训练数据流仍见[原实施档案](../modul-sweep-20260923-b/launch.md#实施与数据流)。

## 环境、命令与产物

沿用AWS本机8×A100-SXM4-80GB、`/dev/md0` XFS NVMe RAID、uv管理的主副本环境。全部持久化运行产物落本仓库 `v1-store/`。数据为 `datasets/4task-v2-1600ep-604f16da`，四任务1600集、605611样本、1192918帧；初始化为 `pi05_base/params`。三份固定数据摘要和所有训练超参沿用原计划。正式两档独立初始化，global batch128、数据worker16、FSDP8、seed42，各80000步。

准备阶段命令如下；`TRAIN_HEAD` 在提交后取实际完整HEAD，不把此文档的创建时间当作运行时间：

```bash
bash scripts/training/prod/run_modul4096_then1024.sh \
  modul-sweep-20260923-c "$TRAIN_HEAD" \
  742f2d894d25abe802a26b1c118852b0b7676e44 \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/modul-budget-sweep/modul-sweep-20260923-b
```

命令须放入本轮唯一的detached tmux，使用原入口的PYTHONUNBUFFERED、pipefail、tee和EXIT_CODE。展开的命令及每阶段覆盖值写入events。顺序仍为2048完整对照重验 → 4096输入与模型检查 → NPY/packed各20+1步 → 100步保存加载 → 容量20步 → 测速1000步 → 正式80k及完成验收 → 1024相同链路。任何失败停止，不自动降配或续训。

两档测速后各提供ETA，正式约300步复核；不把逐叶摘要耗时外推为正式训练步时。正式完成要求16份checkpoint、800条log100、末99步有限、真实末态EMA摘要匹配及固定noise/10步动作有限。4096完整完成才准进入1024。短测及待运行档案见[总索引](../README.md)，验证结果见[result.md](result.md)。权重不进Git，结束后仅归档不可由Git还原的运行记录。
