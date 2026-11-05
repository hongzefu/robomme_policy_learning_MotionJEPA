# StopCube 关键时间与动作：第一阶段启动档案

## ① 目的与当前状态

按[完整计划](../../../1104-stopcube-decision-point-plan.html)，正常运行官方测试集全部 50 局，对提前按错的局另外执行悬停延长，产出逐局关键时间、动作记录和人工确认页。本阶段到人工确认处停止，不运行 32 次噪声采样、梯度、逐层投影或探针。

当前处于实现与验证阶段，正式采集尚未启动。固定绝对截止为美东 **2026-11-05 04:52 EST**（`2026-11-05T09:52:00Z`），从美东 11 月 4 日 20:52 起八小时，包含计划、实现、运行与人工等待；不因恢复任务重置。

## ② 版本、环境与代码状态

- 主机为 `sled-aspen.eecs.umich.edu`，本机两张 NVIDIA RTX A6000，数据盘 `/data`。本任务不使用集群。
- 主检出 `/data/hongzefu/robomme_policy_learning_MotionJEPA`，分支 `v4.1-interpretMotion`。用户未跟踪文件 `TODO_RoboMME_Qwen3VL_FrameSamp_Context.en (1).html` 排除，不读、不改、不提交。
- 初始审查锚点 `f520a29d9c5827754c1955ad53a476a3ade3a664`；修订计划 `afe46fcf52c6f398da1ec1d4243c726e280d230b` 的恢复接口冲突经用户裁决后，由 `652409aeece5a6d111d05263102a99d06cba104d` 修订。
- `PLAN_CONSISTENCY_REVIEW=PASS base=652409aeece5a6d111d05263102a99d06cba104d findings=0 blocking=0`。这只是静态计划审查，不是运行通过。
- benchmark 固定 gitlink `856bc3a189d4172f3f47dbee4424d585f8d78db3`；模型为 `orig100-ctx32x4x4-motion-s2concat1x1024-60k`，checkpoint `59999`，模型 seed `42`。
- 正式运行须从整合审查通过的干净执行工作树启动；该实现提交与实际导入路径在起跑时补记，不能用本段的计划提交代替。

## ③ 启动与配置还原

解释器均复用本机既有 uv 管理环境，不改依赖：

| 用途 | 绝对路径 |
| --- | --- |
| 策略 | `/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv/bin/python` |
| 官方仿真 | `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/envs/official-sim/bin/python` |
| motion 编码 | `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/envs/stage2/bin/python` |
| checkpoint | `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/assets/vla-o16cc-59999/59999` |
| 运行合同 | `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/contracts/runtime-mot.json` |

拟用配置 `mme_vla_suite_orig16_b128_60k`、`model_kind=stage2_pool4_concat`、`purpose=validation`、`ablate=none`、`save_prefix=false`。正常与延长各自使用独立输出目录和身份；延长接管表的内容指纹进入身份。上述为明确运行口径，完整命令在实现 CLI 固定后、起跑之前补记。

缓存显式落在主检出 `v1-store/cache/`：uv、HF、XDG、Matplotlib 分项指定；不覆盖 HOME。固定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不下载模型；策略与仿真 PYTHONPATH 优先执行工作树源码，并只读主检出的锁定 benchmark 源码。实际导入位置须起跑前校验。

资产已按既有固定 manifest 全量核对：`manifests/step-59999.json` 的 sha256 为 `bd35a857e8d5134068539ab03cd01c03c67b61aa22c942477b44df1fe8f97f08`，与先前 Git 留档的锚点一致；逐项比对其中 38 个文件的字节数与 sha256。实测 `ASSETS=PASS files=38 bytes=11976241540 mismatches=0 seconds=10.85`。本轮原始记录在产物根的 `preflight/assets-verified.json`，该结果不代表跨硬件数值逐位一致。

## ④ 数据、时刻与轨迹口径

使用官方 StopCube test 元数据全部 50 局，不换环境 seed，不因任务失败重试挑成功。历史 21 成功/29 失败只是旧结果，不把 29 当作本轮提前按错局数。

时间以环境 `elapsed_steps` 为准，另存模型初始化帧数、推理序号、执行行号和偏移。正常轨迹记录模型命令与实际执行动作；延长轨迹只在接管区间执行下压前实测的悬停关节/夹爪位置。环境真实画面和记忆持续更新，在最接近合适下压时间的合法 16 步新推理边界恢复，禁止重复/漏帧、旧记忆或积压动作。

按钮触发、TCP 真正下降、命令下压与终局分别记录。机械臂提前量从命令实际执行到按钮触发测量；关节目标的 FK 高度只用于目标动作判定，不冒充实际轨迹。官方方块运动使用 smoothstep，时间窗口从实际同 seed 仿真位置读取，不套线性窗口近似。

## ⑤ 数量、校准与验收

第一阶段独立轨迹为正常 50 份，加提前按错局的延长轨迹 E 份，总计 `50+E`，最多 100 份；辅助窗口仿真、冒烟及未来两次重放另外计数。采样数、梯度数等属于人工确认后阶段，本轮不执行。

悬停参考从锁定任务定义及可用示范/真实运动核实。已查四任务 H5 落点没有 StopCube 示范，不据此声称全盘没有。对完整观测到悬停、下压命令与真实按钮触发的失败局，允许同局姿态适用的耗时测量，记 `calibration_source=same_episode_measured` 并保留 `task_success=false`。独立审查 `ADJUST_REVIEW=PASS scale=same findings=0`；它不改变目标时机、官方成功判据或窗口成功验证，也不保证恢复后新动作具有相同耗时。

第一阶段具名验收依次为 `TIMING_CALIBRATION`、`WINDOW_SIM`、`RUN_LOCAL`、`HOLD_REPLAY`、`STAGE1_REVIEW_READY`；只有真实验证后才能写 PASS，缺校准或缺轨迹如实保留，不靠空分母或合成记录通过。确认页保持 `approval=pending`，不自动生成 approved 文件。

## ⑥ 资源、句柄与顺序

最多两张 A6000；起跑前只做必要的低频资源核对，不新增高频 GPU 监控。先单局正常/接管冒烟，再放大正常 50 局；正常结果完成后识别提前按错局，生成接管表，执行延长，再汇总人工确认页。生产者退出并核退出码和完整性后才分析其产物。

本轮已启动的长计算 tmux 清单：**无**。实际起跑时补入唯一 `sd-` 会话名、命令、日志、GPU、代码 sha 和截止控制；不能把预留名称写成已启动。到绝对截止停止计算，保留超时或非零退出码和部分结果，不自动延期。

## ⑨ 用户决定原话

1. 「你别用这个ABC来跟我说，我现在看不懂。」
2. 「子代里要映射成这个Codex的子代里，就是 Opus对应 Sol。」
3. 「对的，延长动作就是重复下压前的悬停位置。然后恢复模型的控制，同意。」
4. 「同意定8小时」。
5. 「同意修改计划文件。」
6. 「文档改完之后同意开工，直接开始进入第一阶段，就是一直做到这个关键时间和动作片段之前，就是做到需要人工确认的位置之前。」
7. 状态机选择：「不登记，按本会话人工交接」。本轮不修改 `docs/handoff/current.json`。
8. 主会话完整解释最近合法新推理边界恢复后，用户：「好的，就是这样。」「准备开工」「同意开工」。

完成后在 `result.md` 补实际覆盖、正常与延长的独立成绩、时刻与动作一致性、人工确认入口和未完成项；轻量证据进 `records/`，大数组、视频、完整轨迹留在 `v1-store`。
