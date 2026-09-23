# 4096→1024八卡顺序实施

本批次已于2026-09-23 00:31:21 UTC从e1b97169d9a1cdc838af5f27bce0bca6a7a513e9启动，于00:53:36 UTC主动中断慢取证，阶段退出130、队列退出1。两个正式run均未启动。后续见[b批次](../modul-sweep-20260923-b/launch.md)。用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间」。

## 版本与硬件

实施前BASE：`742f2d894d25abe802a26b1c118852b0b7676e44`，主副本工作区最初干净。完整配置参照Beta：`55647ff33c8ddb9ec324fdbcee8bd1491456725b`。计划原文与原2048记录保留，不回写旧实测。

实测本机8×A100-SXM4-80GB、八卡初始显存0MiB；`/scratch` 为 `/dev/md0` XFS NVMe RAID，开工可用1274259423232字节；`/dev/shm` 总602265600000字节。无/data及NFS目录。uv与主副本独立.venv存在，本轮不新增正式依赖。

## 实施与数据流

新增16/64frame YAML，仅budget为1024/4096。生产和独立NPY参考链只扩精确形制白名单；在线采样、HistoryPi0、collate、训练优化器和具名默认不改。代码变化集中在完整配置记录、八卡验证、最终保存取证与顺序调度。

改前：同一source/manifest/framesamp-8x8 → 每样本最多32帧 → image(2048,2048) bf16、pos(2048,768) f32、state(2048,8) f64、mask(2048) bool，共14813184字节 → batch128共享内存collate → JAX将state转f32，共14747648字节/样本 → memory(128,2048,1024) → 20个动作query。

改后：同一来源 → 4096选64帧、1024选16帧 → 四键第一轴分别4096/1024，dtype保持 → collate逻辑量29626368/7406592字节/样本 → 同一JAX转换为29495296/7373824字节/样本 → memory(128,budget,1024) → 同一动作监督。保留帧的特征数值不变，选帧集合与query位置预期不同；等价仅比较同预算NPY与packed及2048改前/改后。

清单SHA为df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，store_meta为f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a，norm_stats为856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173，开工重算均相同。

## 执行与留档

入口是 `scripts/training/prod/run_modul4096_then1024.sh`，内部调用 `modul_budget_workflow.py`。参数依次为本批次名 `modul-sweep-20260923-a`、固定Beta完整HEAD、上述实施前BASE。实际tmux全名由启动时UTC生成，并通过MODUL_TMUX_SESSION传入现场记录。入口使用PYTHONUNBUFFERED、pipefail、tee和EXIT_CODE；每阶段独立日志与events记录。主副本训练期间不改源码、依赖或HEAD。

先补2048 Beta完整配置基线及旧档20+1步改前/改后/最终记录开关回归；再4096输入、模型语义、NPY/packed各20+1步、100步保存加载、20步生产容量、1000步测速、正式80000步及完成验收；通过后才进入1024相同链路。任何失败保留现场并停止，用户需要决定的问题立即上报，不自动降batch、改worker、改预算或续训。

所有命令原文、实际HEAD/时间/PID/环境、每档配置与报告SHA写入 `v1-store/bench/modul-budget-sweep/modul-sweep-20260923-a/`；正式输出在原有 `v1-store/train-runs/mme_vla_suite_b128_80k/` 两个新run下。初始空间约1.27TB；两档正式权重约380GB，加验证checkpoint和初末数组，起跑逐档仍要求可用≥400GB。保留原run。

## 验证状态

已完成的CPU短测及实际中断记录在[result.md](result.md)列出。长验证只完成2048初态和前两步，不以它替代完整八卡对拍。正式训练未启动；b批次继续全部闸门，测速1000步后提供各档ETA，正式约300步复核。

## 本批次档案

- [modul-sweep-20260923-a-m2048-before-20](../modul-sweep-20260923-a-m2048-before-20/launch.md)：已启动后中断，退出130，不满足20步判据。
- [modul-sweep-20260923-a-m2048-before-1](../modul-sweep-20260923-a-m2048-before-1/launch.md)：2048 before 1步数值回归，预建未启动。
- [modul-sweep-20260923-a-m2048-after-20](../modul-sweep-20260923-a-m2048-after-20/launch.md)：2048 after 20步数值回归，预建未启动。
- [modul-sweep-20260923-a-m2048-after-1](../modul-sweep-20260923-a-m2048-after-1/launch.md)：2048 after 1步数值回归，预建未启动。
- [modul-sweep-20260923-a-m2048-after-final-20](../modul-sweep-20260923-a-m2048-after-final-20/launch.md)：2048 after-final 20步数值回归，预建未启动。
- [modul-sweep-20260923-a-m2048-after-final-1](../modul-sweep-20260923-a-m2048-after-final-1/launch.md)：2048 after-final 1步数值回归，预建未启动。
- [modul-sweep-20260923-a-m4096-refnpy-20](../modul-sweep-20260923-a-m4096-refnpy-20/launch.md)：4096 refnpy 20步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m4096-refnpy-1](../modul-sweep-20260923-a-m4096-refnpy-1/launch.md)：4096 refnpy 1步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m4096-packed-20](../modul-sweep-20260923-a-m4096-packed-20/launch.md)：4096 packed 20步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m4096-packed-1](../modul-sweep-20260923-a-m4096-packed-1/launch.md)：4096 packed 1步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m4096-save-100](../modul-sweep-20260923-a-m4096-save-100/launch.md)：4096 100步真实保存与加载，预建未启动。
- [modul-sweep-20260923-a-m4096-capacity](../modul-sweep-20260923-a-m4096-capacity/launch.md)：4096 八卡生产容量验证，预建未启动。
- [modul-sweep-20260923-a-m4096-perf](../modul-sweep-20260923-a-m4096-perf/launch.md)：4096 八卡独立测速，预建未启动。
- [modul-sweep-20260923-a-m4096-input](../modul-sweep-20260923-a-m4096-input/launch.md)：4096 有界输入、在线装配、padding、RoPE及初始化，预建未启动。
- [modul-sweep-20260923-a-m1024-refnpy-20](../modul-sweep-20260923-a-m1024-refnpy-20/launch.md)：1024 refnpy 20步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m1024-refnpy-1](../modul-sweep-20260923-a-m1024-refnpy-1/launch.md)：1024 refnpy 1步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m1024-packed-20](../modul-sweep-20260923-a-m1024-packed-20/launch.md)：1024 packed 20步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m1024-packed-1](../modul-sweep-20260923-a-m1024-packed-1/launch.md)：1024 packed 1步真实输入、五标量与201叶取证，预建未启动。
- [modul-sweep-20260923-a-m1024-save-100](../modul-sweep-20260923-a-m1024-save-100/launch.md)：1024 100步真实保存与加载，预建未启动。
- [modul-sweep-20260923-a-m1024-capacity](../modul-sweep-20260923-a-m1024-capacity/launch.md)：1024 八卡生产容量验证，预建未启动。
- [modul-sweep-20260923-a-m1024-perf](../modul-sweep-20260923-a-m1024-perf/launch.md)：1024 八卡独立测速，预建未启动。
- [modul-sweep-20260923-a-m1024-input](../modul-sweep-20260923-a-m1024-input/launch.md)：1024 有界输入、在线装配、padding、RoPE及初始化，预建未启动。
- [v2-1600ep-m64x8x8-modul-b128-80k](../v2-1600ep-m64x8x8-modul-b128-80k/launch.md)：4096八卡80000步正式训练，预建未启动。
- [v2-1600ep-m16x8x8-modul-b128-80k](../v2-1600ep-m16x8x8-modul-b128-80k/launch.md)：1024八卡80000步正式训练，预建未启动。
