# motion 接入：实施中短测记录

用户原话「开始实施 有问题尽早问用户」。本页记录 [0916 计划](../../../0916-motion-modul-8x8-plan.md) 步骤 2b 的工作区短测，尚未完成候选侧关闭态对拍、建库、正式开启态验收或长训练。

## 已完成的验证

`uv run --no-sync pytest scripts/dataset/test_motion_demopad.py scripts/dataset/test_guards.py -q`：77 项通过，12.86 秒。包括新旧布局、独立网格与短 demo 边界、真实 H5 读取及生产补帧函数、逐段汇总和补帧账目、两段 token 续跑、输入变化拒绝、三方 raw_dir 绑定，以及独立抽样比较器的删行负例。

`uv run --no-sync pytest scripts/training/tests/test_motion_v2_tools.py -q`：10 项通过，0.06 秒。使用真实 BASE 记录验证完整性检查，覆盖缺步、重复步、缺标量、NaN、错误 hex；五项 motion preflight、设备事件重叠计时、200 集预算聚合均有拒绝用例。

`motion_v2_checks.py --gate basic` 的三项全部通过：80,000 步学习率逐位相同；真实 policy 构造传入 10 个 motion 键、budget 160，并通过两集真实 reset 重建；RoPE 探针在固定 fp32 输入、seed 42、4 头 × 256 头维下，budget 144→160 的概率总变差为 **0.3375600576**，输出相对 L2 为 **0.7787887454**，记忆长度 512→672 的概率总变差为 **0.4333174825**，同长度 padding 垃圾影响为 0。数字是固定探针的机制示例，不作为训练后效应量。

`motion_v2_checks.py --gate init` 真调用 `init_train_state(..., resume=False)`：CPU 上 116.4 秒完成，关闭态 61 个公共参数叶与开启态逐位一致，开启态只多 4 个 motion 叶，六条 modulation 叶全部在比较集合中。命令设置 240 秒上限，实际正常退出 0。完整数组摘要见 [init-common.json](records/init-common.json)。

在线 P2/P3/P4 在新 YAML 下通过；P1 首次因测试仍匹配旧报错文字失败，更新匹配后独立重跑通过，错帧仍使 sidecar 返回 4。真实 policy 形状探针确认 V5 使用 pi05、VLM gemma_150m、action expert gemma_300m，65 参数叶及真实 motion 输出宽 1024。

新在线比较器在一集真实旧库上，用 GPU 位置编码验证三条 CLI 路径：正例退出 0，错误 stub token 与把 `+0.0` 改为 `-0.0` 两个负例均退出 1，分别记录 1 个失配；每次覆盖 17 窗。它们证明比较器不会因 stub 模式短路而吞掉失败。

计时模块已用实际 GPU kernel 测试：记录前 3 步，后续 2 步不再写计时，trace 包含逐步与取 batch/提交注解以及 GPU 事件；trace 收尾只等待一次，不逐步同步。无需新增依赖。

## 发现的环境差异与待决策事项

CPU 与 GPU 的 `PosEmb3D` 不是逐位相同：帧 16 的 256 维时间码有 47 个元素不同，帧 32 有 51 个不同；两侧均为 float32，并非只有符号零差异。GPU 计算对旧库 `(586,16,768)` 和新库 `(2304,64,768)` **整张位置表逐位一致**。因此计划中“CPU 主进程现场计算位置表并与离线 GPU 表逐位比较”不能直接成立。已向用户提出“预生成并校验 GPU 位置表后供 CPU 验证使用”或“在线验证主进程同样用 GPU”两种选择，尚待回复，不降低逐位判据。

真实仿真环境是 `/scratch/hongze/micromamba/envs/robomme`，可由 `uv run --no-project --python <该环境>/bin/python` 调用；RoboMME 依赖位于本仓库 `third_party/robomme_benchmark`，提交 `856bc3a189d4172f3f47dbee4424d585f8d78db3`，工作区干净。实际 `EnvRunner` 请求 RGB，ManiSkill 默认 GPU 渲染，不能把 200 次 reset 记作零 GPU。已询问使用单卡渲染及逐任务进程隔离，尚未执行该预检。

## 版本与适用边界

短测期间 HEAD 为 `3fd117dded502543e9c240931461a83b83efea83`，生产与测试代码处于本轮未提交的实施工作区；该 HEAD 本身不锚定这些改动。正式候选 V1/V2/V6/V7 必须在生产及验证工具提交后的 clean CAND 上重新取证。依赖未变，未执行 `uv add` 或同步，`uv.lock` 仍为 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`。

上述测试不替代 V3 全量建库、V4 全库交付、V5 真 GPU 模型梯度、V8 开启态 A/A、V-online 三档与 V9 八卡 smoke。实际 GPU 权重推理、正式训练和策略评估仍按原计划各自执行。
