# 顺序训练完成

4096与1024均完成80000步和最终验收，先4096后1024，全部阶段退出0。起跑锚点 `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb`；旧源码对照 `742f2d894d25abe802a26b1c118852b0b7676e44`。

用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间」。全部展开命令、环境、UTC及阶段退出见records/events.jsonl，两档独立测速与初始化/稳态/保存的ETA见records/m4096-speed.json、records/m1024-speed.json。正式结果在两个run子档案。

取证优化保持原dtype/C序字节/SHA、全部201叶及20+1步判据；没有减少验证或改变正式超参。原a批次的中断和615.913秒初态摘要保留。策略rollout与外部权重导出未执行。

用户取证最终决定：「可以优化取证方式 但要保证改前后都是用的一种取证方式 这个是用户最终决策」。同组取证方式的检查记录与完整比较结果一同归档。
