# StopCube 关键时间与动作：第一阶段启动档案

## ① 目的与当前状态

按[完整计划](../../../1104-stopcube-decision-point-plan.html)，正常运行官方测试集全部 50 局，对提前按错的局另外执行悬停延长，产出逐局关键时间、动作记录和人工确认页。本阶段到人工确认处停止，不运行 32 次噪声采样、梯度、逐层投影或探针。

实现和 74 项定向 CPU 联测已完成；美东 2026-11-04 22:08:54 开始的第 16 局正常冒烟已成功结束（26 次推理、416 动作、退出 0），正式 50 局尚未启动。固定绝对截止为美东 **2026-11-05 04:52 EST**（`2026-11-05T09:52:00Z`），从美东 11 月 4 日 20:52 起八小时，包含计划、实现、运行与人工等待；不因恢复任务重置。

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

下面保留单局正常冒烟的固定启动规格；实际起跑及终态见本档案末尾回执，不以预留名称替代现场句柄：

- 干净执行工作树为 `/home/hongzefu/.codex/worktrees/stopcube-runner/robomme_policy_learning_MotionJEPA`，实现提交 `65395f2524b5ac9e9debf4374d6226638d1bb3f5`，与主检出已推送的整合代码一致。三次合并后审查均通过；整合定向验证 `74 passed in 14.36s`，退出 0。
- 官方清单已签封：`MANIFEST=PASS episodes=50 chunks=50 seal=9401b7b3ff89a02d351460c388739a28c8c3312e01f9797e29803584d825f4f6`，文件位于产物根 `preflight/manifest-all.json`。
- 单局为 `StopCube:16`，run 为 `smoke-normal-ep16-a`，使用物理 GPU 1。起跑前 GPU 0 出现其他用户计算，本任务先用 GPU 1，不操作其进程；最多两卡不是强制同时占满两卡。
- 预留 tmux 名 `sd-smoke-normal-1104a`；日志 `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/logs/sd-smoke-normal-1104a.log`；只有收到起跑回执后才转入实际句柄清单。

完整起跑命令为：

```bash
tmux new-session -d -s sd-smoke-normal-1104a \
  'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/smoke_entry.sh'
```

入口的实质命令为下面这条，并通过 `pipefail`、`tee`、`PYTHONUNBUFFERED=1` 和 `EXIT_CODE=` 尾行留存结果：

```bash
/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv/bin/python \
  -m scripts.interpret.run_local \
  --spec /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/spec-smoke-normal.json \
  --episodes StopCube:16
```

spec 的额外值为 `cuda="1"`、`xla_mem_fraction=0.55`、`startup_timeout_seconds=900`、`rpc_timeout_seconds=600`、`stopcube_mode="normal"`、`stopcube_config=<产物根>/normal_config.json`（精确内容 `{"schema_version":1,"episodes":{}}` 后接换行）、`manifest=<产物根>/preflight/manifest-all.json`；`repo` 为上述执行工作树，`attn_out=<产物根>/smoke-normal-ep16-a/attn`、`run_root=<产物根>/smoke-normal-ep16-a/run`。其余模型和解释器值见第三节。正常冒烟不保存额外 prefix 全矩阵和 motion 生成历史，既有推理记录仍保存实际 motion 输入。

起跑脚本每次以固定截止减去当前时间计算剩余秒数，不设置新的八小时。提前 10 秒向当前任务发 TERM，至绝对截止执行 KILL 上限；现有 `run_local` 的信号处理和 `finally` 收尾关闭自己的服务端及客户端，原失败日志保留。任何超时按非零/超时状态报告，不能当成功。

## ⑨ 用户决定原话

1. 「你别用这个ABC来跟我说，我现在看不懂。」
2. 「子代里要映射成这个Codex的子代里，就是 Opus对应 Sol。」
3. 「对的，延长动作就是重复下压前的悬停位置。然后恢复模型的控制，同意。」
4. 「同意定8小时」。
5. 「同意修改计划文件。」
6. 「文档改完之后同意开工，直接开始进入第一阶段，就是一直做到这个关键时间和动作片段之前，就是做到需要人工确认的位置之前。」
7. 状态机选择：「不登记，按本会话人工交接」。本轮不修改 `docs/handoff/current.json`。
8. 主会话完整解释最近合法新推理边界恢复后，用户：「好的，就是这样。」「准备开工」「同意开工」。

## 实际起跑回执

运行代理于美东 **2026-11-04 22:08:54 EST** 启动唯一会话 `sd-smoke-normal-1104a`，启动命令退出 0，启动前会话不存在，执行树 HEAD 为 `65395f2524b5ac9e9debf4374d6226638d1bb3f5` 且工作区干净。实用 GPU 1；本轮实际长计算会话清单目前仅此一项。

日志出现 `RUN_LOCAL_BEGIN model=stage2_pool4_concat ... chunks=1 cuda=1 ablate=none` 和 `CHUNK_BEGIN ... episodes=StopCube:16 port=42969`；模型服务已监听并与客户端建立连接。运行身份为 `3d213e0b02306a0b498eeb07b30a67c012c80fac2397b406af3085652c4be5eb`。这只证明真实任务已启动，推理、终局和物理记录尚待检查，不写为冒烟通过。

正常冒烟终态已核实：`NORMAL_SMOKE=PASS episodes=1 infers=26 actions=416 task_success=1 client_exit=0`，父日志 `RUN_LOCAL=PASS model=stage2_pool4_concat episodes=1 success=1 npz=26`、`EXIT_CODE=0`；逐步记录 416 行，命令与执行动作一致，26 份实际输入和动作 attention 有限，正常回执/视频已写。任务实际评估耗时 `87.67911048373207` 秒；它不包含完整冷启动耗时。会话已自动退出，本用户无残留模型/运行器进程。

### 单局几何与时间校准冒烟规格

夹爪单位修正经前后独立审查后，执行工作树快进到 `d69081fb7f98ed1984802234f79356d18c33e401`，仍干净；修正后的 S1/S3 定向联测 `54 passed in 0.13s`。正常记录入口未改，已完成的正常轨迹仍为上述原实现锚点。

下一份唯一会话预留为 `sd-window-probe16-1104a`，GPU 1，只做第 16 局同 seed 无按键几何仿真，不运行策略。日志 `<产物根>/logs/sd-window-probe16-1104a.log`，输出 `<产物根>/window-probes/smoke16-a/`。环境变量沿第六节，另外显式 `CUDA_VISIBLE_DEVICES=1`，完整程序命令为：

```bash
/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/envs/official-sim/bin/python \
  -m scripts.interpret.stopcube_windows \
  --probe-episode 16 --episode-key StopCube:16 \
  --benchmark-root /data/hongzefu/robomme_policy_learning_MotionJEPA/third_party/robomme_benchmark \
  --output-dir /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/window-probes/smoke16-a
```

程序外层同样使用绝对截止、`tee` 与退出码尾行。主会话生成的固定入口为 `<产物根>/probe16_entry.sh`，运行代理只执行 `tmux new-session -d -s sd-window-probe16-1104a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/probe16_entry.sh'`；启动前核会话不存在，不自动重试或覆盖旧目录。`WINDOW_GEOMETRY` 只证明采到了几何窗口，正式 `WINDOW_SIM` 要结合正常成功事件另验。

几何探针已于美东 22:21:18 唯一启动并正常结束，`WINDOW_GEOMETRY=PASS episode=StopCube:16`、`EXIT_CODE=0`；下一步单独在相同干净提交做独立 FK，环境与截止沿用探针，预留会话 `sd-fk16-1104a`，日志 `<产物根>/logs/sd-fk16-1104a.log`，具体程序命令为：

```bash
/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/envs/official-sim/bin/python \
  -m scripts.interpret.stopcube_windows \
  --trace /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/smoke-normal-ep16-a/run \
  --trace /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/window-probes/smoke16-a/window_probe.jsonl \
  --task-definitions /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/window-probes/smoke16-a/task_definitions.json \
  --fk --benchmark-root /data/hongzefu/robomme_policy_learning_MotionJEPA/third_party/robomme_benchmark \
  --output-dir /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/calibration-smoke16-a
```

起跑入口为 `<产物根>/fk16_entry.sh`，仍由唯一 tmux 加日志退出码包裹；不得将 FK 目标高度说成实际 TCP。输出校准与窗口之后，先核真实成功事件，再构造一局独立悬停恢复冒烟，不与正式成绩混合。

同局 FK 与成功窗口已通过：`TIMING_CALIBRATION=PASS sources=2 unresolved=0`、`WINDOW_GEOMETRY=PASS episodes=1 unresolved=0`、`WINDOW_SIM=PASS successful=1 validated=1 missing_success_groups=none`、`EXIT_CODE=0`。第 16 局命令下压步 403、按钮触发/官方停步 416，运动耗时 13 步；物体目标窗口 `[412,428]`，合适命令范围 `[399,415]`。该数字只对本局实测姿态和动作有证据，不外推所有局。

### 单局悬停恢复功能冒烟

此冒烟仍用第 16 局正常成功记录，单独验证接管/恢复功能，不将该局伪标为正式提前按错局。固定配置 `<产物根>/hold_smoke_config.json`：实测悬停来源第 375 步，`takeover_step=375`、`release_step=401`（现场首步 1 加 25×16），理想恢复为 407，偏差 −6 步；前七维取该行真实状态，夹爪命令取同一行原执行值 `-1.0039340752248764`，原 trace 与其 sha256 写入配置。控制保持到第 401 步的新推理边界，之后执行模型新输出。

正常真实命令为 float64，悬停入口不得再强转 float32 舍入；这一原样保持命令的局部修复已通过 CPU7项及独立前后审查，执行树固定 `731df5ed5c4d663a15557c224b5a8b0e520a4d86`，不改变正常动作、RPC 或物理控制层转换。

预留唯一会话 `sd-smoke-hold16-1104a`，GPU1，日志 `<产物根>/logs/sd-smoke-hold16-1104a.log`；启动命令为 `tmux new-session -d -s sd-smoke-hold16-1104a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/hold_smoke_entry.sh'`。实际程序为第三节解释器运行 `-m scripts.interpret.run_local --spec <产物根>/spec-smoke-hold.json --episodes StopCube:16`，该 spec 复用正常冒烟全部模型/解释器/合同覆盖，仅模式改 `hold_extended`、接管表改上述文件、输出分为 `<产物根>/smoke-hold-ep16-a/{attn,run}`。缓存和固定截止沿用正常入口，正常失败或成功都记录，不为成功重试。

该功能冒烟于美东 22:30:02 唯一起跑并正常结束，程序 `RUN_LOCAL=PASS episodes=1 success=1 npz=27`、`EXIT_CODE=0`。主会话逐项读回：`HOLD_SMOKE=PASS prefix_actions=374 held_steps=26 takeover=375 release=401 restore_infer=25 infers=27 task_success=1 mismatches=0`。恢复推理的模型步为 400，实际环境步 401，最新 frame_times 为 400；输入 state 与现场 state_before 完全一致，12 个有效 motion 槽，记录为本轮真实输入证据。它只验证这份受测轨迹，不泛称所有局或跨硬件逐位等价。

### 正式分块的实施细节

正式 50 局拟使用已有 `chunk_size=5`，共 10 块，每五局复用一次服务，各局仍按官方循环新建环境与评估器、reset 记忆及模型 seed42。正式入口只消费明确 50 局的清单，不传 `--episodes` 造成重新拆单局；单局冒烟仍使用原始一局入口。独立静态审查 `ADJUST_REVIEW=PASS scale=same findings=0`：这是计划未固定的调度细节，不增加任务、协议、运行规模或恢复机制，真实多局行为仍待实跑。当前 GPU0 有其他用户计算，先使用 GPU1，最多两卡的资源上限不变。

正式清单为 `<产物根>/preflight/manifest-all-b5.json`，明确包含 StopCube:0～49 各一次：`MANIFEST=PASS episodes=50 chunks=10 seal=05313f4116785920f732984648e7958aecbeeb07592379ebcf6997539b3e2fdd`。主会话另核每块恰 5 局、模型 seed42、normal 模式、GPU1 和新输出目录，`FORMAL_SPEC=PASS episodes=50 chunks=10 cuda=1 outputs=new`。

正式 run 为 `normal-gpu1-1104a`，固定干净执行提交 `731df5ed5c4d663a15557c224b5a8b0e520a4d86`，预留 tmux `sd-normal50-gpu1-1104a`，日志 `<产物根>/logs/sd-normal50-gpu1-1104a.log`。实际程序命令为：

```bash
/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv/bin/python \
  -m scripts.interpret.run_local \
  --spec /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/spec-normal-gpu1.json
```

该 spec 与正常冒烟模型/环境/合同覆盖相同，改为上述五局分块清单与新输出 `<产物根>/normal/gpu1/attn`、`<产物根>/normal/gpu1/run`。启动命令 `tmux new-session -d -s sd-normal50-gpu1-1104a 'bash /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/stopcube-decision-1104/normal50_entry.sh'`，入口仍固定绝对截止与退出码，不自动重试；只有收到实际起跑回执才写为正在运行。

完成后在 `result.md` 补实际覆盖、正常与延长的独立成绩、时刻与动作一致性、人工确认入口和未完成项；轻量证据进 `records/`，大数组、视频、完整轨迹留在 `v1-store`。
