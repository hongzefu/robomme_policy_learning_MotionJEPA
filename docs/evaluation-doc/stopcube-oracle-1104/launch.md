# StopCube Oracle 下压比例起跑档案

## 目的与用户授权

用户原话：「/data/hongzefu/robomme_policy_learning_MotionJEPA/1104-stopcube-oracle-press-plan.html」「开工」。批准计划提交为 `48be54c0414a362bfc5684d1357f383881103f92`，计划见[完整方案](../../../1104-stopcube-oracle-press-plan.html)。本次含官方 ctx512 79999 与 concat motion 59999；不恢复旧计划、不使用旧计划运行产物，不跑正常局，不启动第二阶段归因。

状态机登记尚未获答复，按计划不写入 `docs/handoff/current.json`，采用人工交接。

资源补充授权原话：「允许使用剩余显存，不操作其他进程」。现场两张卡有其他用户的小显存进程；本轮允许使用剩余容量，不停止、暂停或修改他人进程，不新增高频查询。

## 环境与版本

Aspen 本地环境 A，主检出 `/data/hongzefu/robomme_policy_learning_MotionJEPA`，分支 `v4.1-interpretMotion`，两张 RTX A6000。不连接集群。主检出有用户未跟踪文件 `TODO_RoboMME_Qwen3VL_FrameSamp_Context.en (1).html`，不动、不提交；真实运行另用干净实现工作树并记录精确提交。

benchmark 锁定 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。策略环境为主检出 `.venv`，仿真为 `v1-store/envs/official-sim`，motion 编码为 `v1-store/envs/stage2`。基线规格沿用[既有资产与规格档案](../attn-ctx512-vs-motion-1103/launch.md)，本轮运行前重新核实实际指向与参数。

静态核对三套解释器均为 CPython 3.11.14。主环境 editable 指向主检出 `src` 和 `packages/openpi-client/src`；仿真环境 `robomme.pth` 指向主检出锁定 benchmark。执行工作树以 `PYTHONPATH` 优先指定其自身代码。motion 配置名采用本计划所引用旧起跑档案第三节的 `mme_vla_suite_orig16_b128_60k`；基线采用 `mme_vla_suite_b128`。当前 `training/config.py` 已注册两者，模型历史由检查点配置快照恢复。此前依据 11-03 历史陈述误写“未注册”，已回源码纠正，未启动模型、未发生运行配置变更。缓存须由外层显式指定到本机 `v1-store/cache`。

## 口径、执行顺序与截止

50 局官方 StopCube，环境 seed 不变，模型 seed 42；首问 `t=1`，随后 `t=1+4k≤M·N`，每点 10 预测、两个模型使用相同噪声。询问预测不执行，环境动作表覆盖官方终局。先两局最小试跑，再 50 局动作表，先询问分层 12 局并校准，向用户报告校准结果，继续其余 38 局询问，余量够才补放手。

截止为美东 2026-11-05 04:52 EST，不重新起算；截止前 10 秒终止本轮运行并保留非零退出证据。运行前完整命令、工作树、提交、GPU、端口与会话名由主会话写入本档案；未填入前不视为已经起跑。

## 审查与代码分工

`PLAN_CONSISTENCY_REVIEW=PASS base=48be54c0414a362bfc5684d1357f383881103f92 findings=2 blocking=0`。审查修正文案两处：第 0 步约 2 小时；补明首包 `D+1` 与首次询问 `t=1`，均不改既定口径。独立工作树 S1 动作表、S2 运行器、S3 询问存盘与离线、S4 分析；依次独立审查、保留子提交合并、短测、推送。

## 产物与盯盘

本轮产物根 `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-oracle-1104/`，各局各目录；子目录为 `oracle/`、`smoke/`、`query/`、`calibrate/`、`release/`、`analysis/`、`logs/`。大数组和视频不进 git。具名验证写入 `records/verification.summary.txt`，失败不改阈值、不为成功重试。

## 本轮会话清单

尚未启动运行会话。后续只使用 `so-` 前缀并逐项在此登记。
