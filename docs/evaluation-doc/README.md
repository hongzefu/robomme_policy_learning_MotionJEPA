# 评估档案索引

本目录记录官方评估实现和运行证据；大型权重、日志及视频留在各任务的 `v1-store`。

| 档案 | 范围 | 状态 |
| --- | --- | --- |
| [StopCube 关键时间与动作第一阶段](stopcube-decision-1104/result.md) | motion 59999、seed42；原计划正常50局及悬停延长 | 用户已中止；28局终局、13成功15失败，accepted25局；父程序143退出，全部本轮进程已停，第一阶段未完成 |
| [阶段一实现](official-stage2-phase1-1026/README.md) | 按1027简化方案保留弹性队列、常驻模型与逐块仿真 | 简化实现完成，实际新增worker领取不同块通过 |
| [20k试评估](official-stage2-20k-probe-1026-a/result.md) | 固定Stage2 VLA20k、epoch105、单局及双卡十局 | 单局0/1；两卡1/10（10%）；两个worker退出0，资源已释放 |
| [o16ctx 60k 官方800](o16ctx-60k-800/result.md) | 官方16×100 context split 59999，seed 42，800局 | 311/800（38.88%）；资源已释放 |
| [三模型三seed](o16-3seed-1031/result.md) | 官方16×100 modulation split、context split、context concat 1×1024 的 59999，seed 7／42／0 各800局 | modul 43.42±0.52；ctx split 38.42±1.25；ctx concat 37.33±0.59；10份占位已释放 |
| [attention 第一阶段](attn-ctx512-vs-motion-1103/result.md) | 官方 ctx512/79999 vs ctx512+motion concat 59999，ButtonUnmaskSwap「揭开」单次推理 attention，本机 2×A6000 | ep8 两模型成功；关键推理 #18：帧流裸露窗占比 3.6% vs 17.0%（均匀 12.5%），motion 交换区间 30.5%（均匀 35.0%，重叠/33 口径，无偏好）；帧流总质量仅 0.9%/1.35%；另含因果实验 12 局，完整报告见 [report.html](attn-ctx512-vs-motion-1103/report.html) |
| [成功回合细致注意力分析](attn-buttonunmaskswap-success-seed42-1103/result.md) | ButtonUnmaskSwap，带 motion token 的 VLA 12 局、不带 motion token 的 VLA 10 局，模型 seed42；开局帧窗口与完整 motion token | 已完成22条采集与分析；本轮实际成功分别11/12、5/10；全部token留存、96张图目视复核及独立内容核验通过 |
