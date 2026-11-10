# 评估档案索引

本目录记录官方评估实现和运行证据；大型权重、日志及视频留在各任务的 `v1-store`。

| 档案 | 范围 | 状态 |
| --- | --- | --- |
| [逐子目标位置交换（全部长度）+ 交换任务按第 k 次交换](sp-swap-1110/result.md) | orig100 MotionMem 与 Frozen Wan 59999；PatternLock m=1–7、RouteStick m=2–7 每个 (m,j) 换第 j 段示范 motion token（Bsame/Bcross/Z），三交换任务按第 k 次交换（Bsame_k/Z_k）读第一次拿起；GreatLakes 8×A40 + aspen，约 5000 局 | 完成；MotionMem 第 j 步跟随供体减基线 RS Bsame +0.08 [+0.02,+0.13]、Bcross +0.11 [+0.02,+0.20]、PL Bcross +0.10，集中在 j=1；Wan ≈0；无对角结构（MotionMem 前序步约一半改变，与 seed 一致率 20/40 相当）；交换任务换交换段 token 后两模型都很少改拿「没交换」答案 |
| [RouteStick 第一子目标交换 + Frozen Wan 对比](rs-swap-full-1109/result.md) | orig100 MotionMem 与 Frozen Wan 59999；easy 两步局只换只读第一个子目标（1 路径 × 1 绕向，16 类型），N/Z/Rnd/A/Bsw/Bside/C/D1/D2/W0 + seed 复核，约 870 局 | 完成；MotionMem 绕向跟供体 15/18、左右跟供体 16/20、D2 donor_dir 39/124 对 donor_abs 0/124；Wan 交换 237/237 走自己（不读内容），但置零后第一步 1/13 |
| [PatternLock 全因子单步交换 + Frozen Wan 对比](pl-swap-full-1109/result.md) | orig100 MotionMem（concat 1×1024）与 Frozen Wan pooling 59999；新 env_seed 80 局覆盖 40 种（起点格，方向），N/Z/Rnd/A/B/C/D + seed 复核，共 2433 局 | 完成；MotionMem Z 10/37、Rnd 11/37、B 跟供体方向 136/150、D donor_dir 147/378 对 donor_abs 61/378；Wan 交换与扰动 0 局跟供体（token 几乎不影响动作） |
| [PatternLock 单步 motion token 交换](pl-single-step-swap-1108/result.md) | motion 59999；easy 单步成功 8 局 × 5 seed；只换示范段唯一 motion 窗口：原样 / 置零 / 位置方向同 / 位置同方向异 / 位置异方向同，共 130 局 | 完成；B 15/15 走到供体目标格（10 局可区分）；C 位置异方向同仍走对方向 15/25（60%），2 对确定由 token 驱动、2 对走错，0/25 走到供体绝对目标；置零 16/40 |
| [MoveCube 强推恢复](movecube-force-recovery-seed42-1107/result.md) | 官方 ctx512 79999 与 motion 59999；拿 cube 21 局强推到 peg 上方、拿 peg 12 局强推到 cube 上方，0～80 步放手，抓起正确物体即成功，共 396 局，不录注意力 | 完成；k=16 约六到七成恢复，k≥32 几乎全部抓错误物体，两模型无差异 |
| [PatternLock、RouteStick 按子目标的注意力](subgoal-rollout-attn-seed42-1106/result.md) | 官方 ctx512 79999 与 motion 59999；两任务官方 50 局全程模型控制，922 次推理全算，按 current_task_index 分组 | 成功 motion 34/100、基线 18/100；12 组对位读数无「明显」；motion 帧流回看示范开头与 Swap 方向相同，基线文字读示范末尾；资源已释放 |
| [两个 Swap 任务第一次取 bin](swap-first-bin-attn-seed42-1105/result.md) | ButtonUnmaskSwap、VideoUnmaskSwap 各 50 局；官方 ctx512 79999 与 motion 59999；靠近到闭合的注意力，motion 模型 4 个替换条件 | 成功 motion 29/100、基线 20/100；motion 回看开局是两任务共同 pattern，与选对选错无可见关系；Button 换 motion 后相对 E0 下降（17→7），Video 无显著变化；E4 未跑 |
| [StopCube 真实 rollout](stopcube-rollout-1105/result.md) | 官方 ctx512 79999 与 motion 59999；官方 50 局全程模型控制，触发分布、注意力与进度探针 | 基线 7/50、motion 22/50；1527 次推理重放误差 0；资源已释放 |
| [StopCube Oracle 下压与机制分析](stopcube-oracle-1104/result.md) | motion 59999 与官方 ctx512 79999；50局原历史询问、真实校准及同输入分析 | 7460询问点、74次校准、100次固定放手和16输入160原噪声机制分析完成；基线18/50、motion29/50；35图已逐张目视，非官方标准整局成绩 |
| [StopCube 关键时间与动作第一阶段](stopcube-decision-1104/result.md) | motion 59999、seed42；原计划正常50局及悬停延长 | 用户已中止；28局终局、13成功15失败，accepted25局；父程序143退出，全部本轮进程已停，第一阶段未完成 |
| [阶段一实现](official-stage2-phase1-1026/README.md) | 按1027简化方案保留弹性队列、常驻模型与逐块仿真 | 简化实现完成，实际新增worker领取不同块通过 |
| [20k试评估](official-stage2-20k-probe-1026-a/result.md) | 固定Stage2 VLA20k、epoch105、单局及双卡十局 | 单局0/1；两卡1/10（10%）；两个worker退出0，资源已释放 |
| [o16ctx 60k 官方800](o16ctx-60k-800/result.md) | 官方16×100 context split 59999，seed 42，800局 | 311/800（38.88%）；资源已释放 |
| [三模型三seed](o16-3seed-1031/result.md) | 官方16×100 modulation split、context split、context concat 1×1024 的 59999，seed 7／42／0 各800局 | modul 43.42±0.52；ctx split 38.42±1.25；ctx concat 37.33±0.59；10份占位已释放 |
| [attention 第一阶段](attn-ctx512-vs-motion-1103/result.md) | 官方 ctx512/79999 vs ctx512+motion concat 59999，ButtonUnmaskSwap「揭开」单次推理 attention，本机 2×A6000 | ep8 两模型成功；关键推理 #18：帧流裸露窗占比 3.6% vs 17.0%（均匀 12.5%），motion 交换区间 30.5%（均匀 35.0%，重叠/33 口径，无偏好）；帧流总质量仅 0.9%/1.35%；另含因果实验 12 局，完整报告见 [report.html](attn-ctx512-vs-motion-1103/report.html) |
| [成功回合细致注意力分析](attn-buttonunmaskswap-success-seed42-1103/result.md) | ButtonUnmaskSwap，带 motion token 的 VLA 12 局、不带 motion token 的 VLA 10 局，模型 seed42；开局帧窗口与完整 motion token | 已完成22条采集与分析；本轮实际成功分别11/12、5/10；全部token留存、96张图目视复核及独立内容核验通过 |
