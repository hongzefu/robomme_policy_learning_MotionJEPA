> 创建日期：2026-11-03（美东）。
> 工作副本：`/data/hongzefu/robomme_policy_learning_MotionJEPA`；分支：`v4.1-interpretMotion`。
> 只读核实的代码基线：`80a5e7ef9e7ee1ce29dfd1bee7d3a2d705a63777`。后续实施和实际运行的提交另行固定。
> 本轮只将聊天中的完整方案写入 Markdown，不启动实验、不修改代码、不提交作业。用户本次明确要求 Markdown，作为本文件对通用 HTML 计划格式的豁免。
> 本任务尚未取得开工令；是否登记进 `docs/handoff/current.json` 也尚待用户确认。本文件及其中的命令不构成开工授权。

# 第一部分（给人看）

## 一、要做什么与统一名称

重新分析 `ButtonUnmaskSwap`：两种 VLA 各自重跑模型 seed=42 下已经成功的回合，沿用[现有注意力报告](docs/evaluation-doc/attn-ctx512-vs-motion-1103/report.html)的方法，将细致分析从少数运行扩展到全部入选回合，并归纳共同模式和反例。

所有计划正文、进度、图例、表格、报告和结论统一使用下列名称；代码标识符、配置名和历史原文保留原名。

| 固定名称 | 对象与身份 |
|---|---|
| **带 motion token 的 VLA** | 既有报告的 context concat 1×1024 模型，`orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k/59999`；配置 `mme_vla_suite_orig16_b128_60k`；`model_kind=stage2_pool4_concat` |
| **不带 motion token 的 VLA** | 既有报告的官方 `perceptual-framesamp-context/79999`；配置 `mme_vla_suite_b128`；`model_kind=history` |
| **motion token** | 带 motion token 的 VLA 接收的运动编码输入，不能用这个词单独代称模型 |

`seed=42` 是模型采样噪声的种子编号，不是任务数，也不是环境 seed。环境 seed 和难度按 test 元数据逐局核对。任务种类始终只有 `ButtonUnmaskSwap`。

## 二、成功回合与运行范围

沿用刚才方案的统一口径：**同一权重、模型 seed=42 的已有正常运行中，曾成功的回合按局号去重取并集**。选局依据是历史成功，不要求本次重跑必然成功。

| 模型 | 局数 | 回合编号 |
|---|---:|---|
| 带 motion token 的 VLA | 12 | `4,8,11,12,17,18,20,26,34,37,38,40` |
| 不带 motion token 的 VLA | 10 | `8,12,14,16,18,20,26,34,38,47` |
| 合计 | 22 条模型与回合组合 | 两组分别使用自己的清单，不强制使用同一批局 |

带 motion token 的 VLA 的 12 局来自既有报告采用的官方评估成功集合。不带 motion token 的 VLA 的完整 50 局历史记录有 9 个成功回合：`8,12,14,16,18,20,26,34,47`；本机后续正常组又成功过第 38 局，因此并集为 10 局。不能将后续预选 12 局中成功的 `8,12,18,38` 当作它全部的成功集合。

来源分别为：

- 带 motion token 的 VLA：[既有结果档案](docs/evaluation-doc/attn-ctx512-vs-motion-1103/result.md)的“因果实验：遮挡与 motion 内容替换（12 局）”。
- 不带 motion token 的 VLA 的完整 50 局记录：[逐局 JSON](docs/training-doc/eval-3seed-context-vs-motion/records/per_episode_by_seed.json)，筛选 `group=official`、`task=ButtonUnmaskSwap`、`results["42"].success=true`。
- 不带 motion token 的 VLA 的第 38 局：[既有结果档案](docs/evaluation-doc/attn-ctx512-vs-motion-1103/result.md)的正常组，以及本机 `v1-store/evaluation/interpret-1103/ablate12/ctx/none/run/accepted/` 的终局记录。

本次只做正常运行的采集与分析，不新增遮挡、置零或内容替换实验。历史消融结果仅作为背景，不能冒充本轮验证结果。

## 三、只关注的两处

| 关注对象 | 分析口径 | 保留的细节 |
|---|---|---|
| 开局帧窗口 | 沿用裸露窗口 `VIS=[0,32)`，另列第 0 帧结果 | 逐帧及帧内 token；核对该局实际可见性与时间边界 |
| motion token | 逐个分析全部有效 token，按覆盖窗口关联交换过程 | 完整向量、窗口起点、有效掩码、输入顺序与注意力位置映射 |

不带 motion token 的 VLA 没有运动编码输入，其 motion token 指标写“不适用”，不能用 0 冒充测量结果。其他 token 类别只作为解释间接通路所需的中转，不扩展成新的分析主题。

沿用原报告中的注意力总质量、区间占比、相对均匀分配的富集、逐层逐头分工、时间线、去除注意力汇聚点后的稳健结果，以及间接通路近似。motion 窗口固定按**与目标区间重叠的帧数除以 33**计权，不能恢复已经纠正的“沾边即整窗计入”口径。

每局按实际“靠近、抓取、抬起”阶段对齐；不把第 8 局的第 18 次推理套用到其他局。没有发生的阶段如实记缺失，不凭推理编号补造事件。

## 四、所有 motion token 都要记录

带 motion token 的 VLA 每次推理均保存实际输入的完整 `motion_emb`、`motion_pos`、`motion_mask`、`motion_times`、`mem_order`，连同局身份、`infer_seq`、`step_idx` 和动作记录，使每个有效 token 都能映射回它的内容、时间窗口及注意力 key 位置。

同时保留编码器生成的全部窗口编码及调用顺序，为之后切换不同 motion token 做比较。既有 `MotionContentClient` 按起点保存首次编码的记录，不能单独代替本次“逐次实际输入”的完整记录。

原始注意力保留去噪步、层、头、动作 query 和 key 维度；保存时不平均，不只留排名靠前的 token。完整 token 内容与注意力权重分别保存：只保存热图或注意力权重不足以支持后续内容替换。

## 五、共同模式怎么总结

分别回答两种 VLA 的成功回合是否重复出现同一类读取行为：开局窗口是否被回看、哪些头反复参与、这些行为发生在什么动作阶段；带 motion token 的 VLA 是否稳定关注相同类型的运动窗口。

每条结论报告支持局数、可计算局数、逐局分布和反例。先逐局计算，再跨局汇总，避免长回合因推理次数多而主导结论。分别列出全部历史入选回合的重跑状态，以及本轮实际成功回合的模式；重跑失败回合单列，不混入本轮成功模式的分母。

两组成功集合不同，组间总体差异只作描述；同局比较仅使用交集，并标明本轮双方的实际结局。没有失败对照，不能声称某种模式能区分成功与失败；注意力也不能直接证明因果作用。

`L3H0+L16H0` 的含义固定为同时遮挡两个注意力头的输出。层和头从 0 编号，分别对应第 4 层第 1 个头、第 17 层第 1 个头；不是 motion token 的编号。本轮检查它们跨局是否稳定参与，不默认它们是通用关键头，也不新增遮挡实验。

## 六、验收与实施步骤

下表均为待实施判据，不是已经通过的结果。程序完成与任务成功分别报告，`completed=22` 不表示成功 22 局。

| 阶段 | 检查内容与方式 | 通过说明什么 | 判定行 |
|---|---|---|---|
| 开工审查 | 独立只读子代理核对最终计划全文 | 无待裁决的实施矛盾 | `PLAN_CONSISTENCY_REVIEW=PASS` |
| 清单核对 | 逐条核对权重、模型 seed、环境 seed、成功来源与局号 | 22 条组合身份正确，无漏选和重复 | `SUCCESS_MANIFEST=PASS models=2 episodes=22` |
| 最小冒烟 | 每种 VLA 各一局，核对记录完整与采集不改变动作 | 核心采集路径可用 | `CAPTURE_SMOKE=PASS` |
| 全量采集 | 核对每条组合的确定终局及逐次原始记录；通过的冒烟计入清单 | 所有入选组合都有结果及完整记录 | `ATTN_COVERAGE=PASS expected=22 completed=22` |
| token 验收 | 逐次核对有效位、完整向量、时间与 key 位置映射 | 后续可精确定位要替换的 token | `MOTION_TOKEN_COVERAGE=PASS missing=0` |
| 报告交付 | 核对逐局结果、共同模式、反例、图及原始数据索引 | 结论可追溯；图经渲染目视复查 | `ATTN_REPORT=PASS` |

## 七、子代理分工与合并（简述）

采集修改和批量分析分别由独立工作树中的代码子代理负责，不共写同一文件；审查子代理只读核对固定提交。主会话先整合采集改动及验证，再整合分析入口，确认接口一致后运行。运行代理只执行确定版本和命令，不修代码；资源、监听、收尾和最终验收归主会话。

运行拟在本机 Aspen，最多使用两张空闲 GPU，不提交集群作业。预计超过 5 分钟的运行使用独立 detached tmux。当前实测是 2×RTX A6000；旧环境表的 GPU 型号与实测不同，实施前需明确本轮环境适用口径，不能将旧表视为本轮硬件证明。

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

1. 当前只有计划文档写入授权，没有开工令；不改代码、依赖或运行配置，不下载、不启动仿真或推理、不占用 GPU。
2. 模型、checkpoint、模型 seed 和选局范围按第一部分固定。不得混入 context split、自训纯帧模型或其他训练轮次。
3. 重跑正常失败如实保留，不为得到成功而重试；基础设施故障与正常任务失败分开，重试必须记录原因和次数。
4. 不修改训练语义、模型计算或第三方环境；只扩展记录和离线分析。采集不改变动作须验证，不能仅凭设计推断。
5. 不覆盖既有原始产物和报告。工作区现有未跟踪文件 `TODO_RoboMME_Qwen3VL_FrameSamp_Context.en (1).html` 不属于本任务，不读取、不暂存、不修改。

## 一、逐文件改动清单与数据契约

| 文件或入口 | 拟改内容 | 关闭态／既有行为 | 开启态／新增行为 |
|---|---|---|---|
| `scripts/interpret/serve_policy_attn.py::AttnRecorder.install`、`build_record` | 捕获并保存本次实际运动输入 | 原有采样和动作输出 | 追加完整输入记录，采样和动作输出不变 |
| `scripts/interpret/serve_policy_attn.py::MotionContentClient` | 保留全部生成窗口的内容与调用顺序 | 原编码调用与缓存语义 | 编码内容可追溯，不改变返回给策略的编码 |
| `scripts/interpret/run_local.py` | 沿用驱动，检查两个模型各自的清单及记录出口 | 既有逐局运行方式 | 每条模型与回合组合有独立结果和目录 |
| `scripts/interpret/analysis/success_attention.py`（拟新增） | 读取清单，批量调用既有分析函数并汇总 | 原有单局分析保持可复现 | 支持两组各自成功集合和逐局阶段对齐 |
| `scripts/interpret/test_motion_capture.py`（拟新增） | 定向验证完整记录与无行为改变 | 不参与生产入口 | 覆盖空窗口、有效位映射、内容保真与输入不被改写 |
| `docs/evaluation-doc/<新运行名>/` | 启动档案、结果、报告和小型证据 | 既有档案保留 | 新目录独立交付 |

复用 `aggregate_attn.py::split_streams`、`analysis/x1_layer_head.py::infer_metrics`、`analysis/x3_intervals.py` 的区间口径，以及 `analysis/r2_memory_internal.py`、`r3_sink.py`、`r4_layer_paths.py` 中对应的计算逻辑。需参数化的局身份、区间和阶段显式从本轮清单传入，不能继承脚本内硬编码的三局或固定关键推理。

实际运动输入记录以推理边界捕获的张量为准，维度和 dtype 从本轮 checkpoint 与真实观测读回并写入档案；有效 token 的数量由 mask 决定，不能把预算槽位全部算作有效 token。原始内容不为画图而降精度；attention 沿用既有存储精度并记录。

## 二、子代理分配表

| 子任务 | 目标与可写集合 | 禁触范围／接口依赖 | 合并与验收 | 资源与归属 |
|---|---|---|---|---|
| S1 采集 | `serve_policy_attn.py`、`test_motion_capture.py` | 不改模型、训练、第三方环境和 S2 文件；提供完整 token 记录契约 | 先合入；定向验证及 `CAPTURE_SMOKE` | 独立工作树，代码角色 `gpt-6.1-sol`、`high`；不自行启动长任务 |
| S2 分析 | 拟新增 `analysis/success_attention.py` | 只读 S1 接口与既有分析；不改 S1 文件 | S1 接口固定后合入；逐局与既有计算对拍 | 独立工作树，代码角色 `gpt-6.1-sol`、`high`；离线 CPU 为主 |
| R 审查 | 只读最终计划及固定 BASE/TIP | 不写文件、不运行任务 | 开工一致性、整合差异及验收口径审查 | `gpt-6.1-sol`、`high` |
| E 运行 | 执行主会话给定版本、完整命令与确定目录 | 不改代码、配置、参数或依赖；不自行取消资源 | 先最小冒烟，再清单内剩余组合 | `gpt-6.1-sol`、`high`；命令、GPU、tmux、预算与日志派发前固定 |
| 主会话 | 清单、运行规格、计划、档案与最终报告 | 不碰用户在途改动 | 确认归属后整合，统一验证、提交和推送 | 唯一资源、监听、清理和共享档案负责人 |

上表是未来实施分工，不是本轮派发实施的授权。长任务运行委派的完整命令、确定代码提交、tmux、日志、起跑判据、最长等待及基础设施重试上限，需在实际运行前落入启动档案；未确定的规格不能交运行代理自行决定。

## 三、闸门与执行手册

1. 用户确认完整方案并明确开工后，先对最终计划提交做独立一致性审查；解决所有待裁决项后实施。
2. 固定成功清单和两份权重身份，逐条核对 `(模型身份, model_seed, task, episode_id, env_seed)`，生成两组独立 manifest。
3. 修改并验证采集记录：用同一输入、同一随机状态核对记录前后动作与输入内容不变，检查空窗口及有效位映射。
4. 合入批量分析入口，针对旧第 8 局记录核对既有指标，确认重叠／33 口径和去汇聚点口径一致。
5. 确认全新运行名称、真实代码提交、运行规格及实体输出目录后，从 clean HEAD 启动。拟用名称 `attn-buttonunmaskswap-success-seed42-1103`；若已存在，不覆盖，先更新计划中的名称。
6. 每种 VLA 先运行一局并完成采集冒烟；通过的完整记录计入 22 条组合，不因“冒烟”身份重复运行。冒烟的采集完整性失败时不放大运行。
7. 补齐清单内其余组合；逐条保存真实终局，核对原始 attention、prefix attention、视频及 motion token 记录。前缀记录供既有间接通路和汇聚点分析使用。
8. 逐局离线分析，再按实际动作阶段汇总共同模式、支持比例和反例；报告重跑失败与缺失阶段。
9. 主会话目视复核最终图，完成档案、验收和提交推送；只清理由本轮明确登记的会话与资源。

未来运行命令沿用现有入口，以下仅是结构示意，`<已固定规格>` 必须先落实为真实文件：

```bash
UV_CACHE_DIR=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
  uv run --frozen --no-sync python scripts/interpret/run_local.py --spec '<已固定规格>'
```

所有新原始数据、运行规格、缓存、日志与临时验证产物落在本机 `v1-store/`；原始运行根拟为 `v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103/`。每种 VLA 与每局各自拥有独立输出，不能并发覆盖进度。运行前检查实际磁盘余量，耗时和存储量以最小冒烟估算，尚未测量时不承诺完成时刻。

## 四、风险登记与盲区

| 风险或局限 | 处置与结论边界 |
|---|---|
| 历史成功在本机不复现 | 保留该局及真实结局；历史入选覆盖和本轮成功模式使用不同分母，不为成功重跑 |
| 旧评估环境与本机硬件、attention 展开实现存在差异 | 记录代码、依赖、硬件、权重与实现口径；不把跨环境差异全部归因于模型或 token 内容 |
| 两组按各自成功选择，样本不同 | 总体差异只作描述，同局比较单列交集；不能用本轮重跑成功率代表完整 test 成功率 |
| 单个高富集头实际质量极小 | 沿用质量、富集与超额质量共同报告，以及原有低质量解释限制 |
| 注意力汇聚点制造表面偏好 | 保留原始结果与既有去汇聚点结果，不为得到模式改阈值 |
| 颜色和阶段并不固定 | 不能把第 8 局绿色目标或固定推理编号硬套其他局，按该局元数据、动作和视频核实 |
| 全量 prefix attention 占用较大 | 逐次读取并释放；启动前以冒烟测实际磁盘及耗时，不静默删减采集维度 |
| 注意力概率不能直接表示贡献 | 不宣称因果作用；没有记录值向量时，不声称完成值范数加权分析 |
| 只选历史成功回合、单模型 seed | 不推断失败机制、总体性能或跨 seed 泛化；本轮不扩大至其他任务 |

## 五、交接顺序与状态机登记

本任务拟用任务 id `buttonunmaskswap-success-attention-seed42`，**尚未登记**。用户本轮仅要求写 Markdown，未回答是否登记，不改 `docs/handoff/current.json`。

- 交接前必须完成：清单、完整采集、全部 motion token 记录、离线分析与报告，分别使用第一部分的具名判定行，证据位置写入新运行档案。
- 长尾：本方案没有上传、下载或其他长尾任务；不新增对应句柄。
- 上游依赖：已有权重、环境元数据和历史成功证据；现场核对身份与可用性后方可消费，不凭档案路径推断文件仍存在。
- 下游：报告及记录验收完成后，可作为未来 motion token 切换比较的输入；本方案不授予下游实验开工权。
- 如用户同意登记，按仓库规则记录计划提交、开工令、owner、句柄及状态；最终全部前置项通过且运行句柄清空后才可标记 `done`。

## 六、留档、文档验证与提交纪律

本次文档写入验证：检查两个一级标题、名称、22 条组合及来源链接；执行 `git diff --check`，只逐路径暂存本计划，中文 `docs:` 提交后立即推送到当前既有 upstream。用户未跟踪文件保持原状。

未来运行在新档案目录写 `launch.md`、`result.md` 和 `records/`；启动时记录代码、权重、完整命令、参数、模型与环境 seed、清单、存储、tmux、日志和预算，结束后补真实终局与验收。不复制可由 Git 还原的脚本或配置，不把大体积权重和原始 attention 提交进 Git。实施中调整按通用规则先审查并留档，不能静默更换口径。

## 七、实施追加记录（2026-11-04，美东）

用户在完整计划保存并推送后明确指令：「开始实现」。本指令是本计划的开工令，覆盖上文创建时的“尚未开工”历史状态；状态机登记仍未单独获准，因此不改 `docs/handoff/current.json`。

开工代码基线为 `b7ad0560e4315a8b1ecb669a86cb584e0eff68d7`，计划与原保存提交之间无内容变化。独立子代理审查结论：`PLAN_CONSISTENCY_REVIEW=PASS base=b7ad0560e4315a8b1ecb669a86cb584e0eff68d7 findings=0 blocking=0`。

### 实施中调整

1. **执行位置**：主检出含用户未跟踪 HTML，正式运行改用同一整合提交的干净工作树 `/home/hongzefu/.codex/worktrees/button-attn-run/robomme_policy_learning_MotionJEPA`；所有写产物仍落主检出本轮 `v1-store` 新运行根。复用已有只读环境、权重和锁定 benchmark；实际核对 VLA 导入来自运行工作树，benchmark 导入来自主检出已有子模块，不初始化第二份 benchmark。没有移动、暂存或删除用户文件。审查结论：`ADJUST_REVIEW=PASS scale=same findings=0`。
2. **测试夹具**：S1 可写集合追加 `tests/interpret/test_motion_replace.py`、`tests/interpret/test_ablate.py` 和 `tests/interpret/test_run_local.py`。它们的假观测缺少本次必须记录的 `motion_emb`、`motion_pos`，需补齐真实字段，并更新带／不带运动输入两态的精确键集合、shape 与 dtype 断言。保留原有命令、错误退出、nonce、identity、mask、动作与调用次数检查，不为旧假夹具新增生产回退，prefix 独立记录合同不变。两次按实际发现范围进行的独立审查均为 `ADJUST_REVIEW=PASS scale=same findings=0`。
3. **关节状态回放命令**：第一次离线回放额外设置 `CUDA_VISIBLE_DEVICES=''`，导致既有 SAPIEN 渲染设备不可见并在自动回退 CPU 渲染后退出 139。保留该失败日志，不将它算作策略回合失败。独立审查后只将该诊断命令恢复为 `CUDA_VISIBLE_DEVICES=0`，脚本、物理实现和渲染实现均不改；现场读回 `SIM_RESET ... gpu=0`（CPU 物理）及 `SIM_EEF_POSE=PASS infers=20`、`EXIT_CODE=0`。GPU 可见不等于使用 GPU 物理；回放只 reset 后设置已有 qpos，不调用策略或 `env.step`，不增加或重试 22 条策略回合。审查结论为 `ADJUST_REVIEW=PASS scale=same findings=0`。
4. **多目标与分析执行位置**：实际第 11 局先取藏蓝色方块的容器，再取藏绿色方块的容器；hard 固定取物两次，easy 也允许两次。第六 query 族因此按实际提示中的全部已核目标色文本行求均值，仍保留一个 `q_target_color` 族；记录顺序、各自 piece／行号，任一目标缺行则整组未核实，不以子集冒充完整族。单目标与原式逐元素一致，六族及所有 R3 阈值不变。阶段证据逐次保留 `pickup_index`，不只展示最后一次取物。独立同规模调整及修复复审均通过。GPU 采集继续固定 `capture_commit=e56bbceca5fe1c734c2553e83b8b6de1db71ca08`；离线分析使用已有干净 S2 工作树固定 `analysis_commit=c55700b63b89de451adfc0c459bd99ecefce8443`，不切换正在采集的工作树，不重跑策略回合。
5. **离线 CPU 分片**：503 次完整推理按独立回合分给 4 个 CPU 进程；各进程直接调用固定分析版本的 `episode_analysis`，每局独立目录，父进程按原清单顺序收集后调用原 `summarize` 和 `figures_summary`。指标、公式、归约顺序、22 条组合和无重试口径均不改变；不使用 GPU，每进程 BLAS／OMP／MKL 线程各为 1。包装脚本和完整命令、源码哈希留在本轮运行根及档案，独立审查结论为 `ADJUST_REVIEW=PASS scale=same findings=0`。

以上调整保持两份权重、seed、22 条模型与回合组合、全部采集维度及验收判据不变。最终运行命令、提交与资源记录于本任务独立启动档案。

### 最终实测与交付（美东2026-11-04）

22条历史成功入选组合全部完成：带 motion token 的 VLA 12局，本轮实际成功11局；不带 motion token 的 VLA 10局，本轮实际成功5局。两者集合不同，不把这两个比例用作总体性能比较；正常失败全部保留，没有为成功重试。采集版本为 `e56bbceca5fe1c734c2553e83b8b6de1db71ca08`，完整离线分析版本为 `c55700b63b89de451adfc0c459bd99ecefce8443`。

最终判定如下，均附原始证据于[结果报告](docs/evaluation-doc/attn-buttonunmaskswap-success-seed42-1103/result.md)及其records目录：

```text
ATTN_COVERAGE=PASS expected=22 completed=22 infers=503 prefixes=503
MOTION_TOKEN_COVERAGE=PASS episodes=12 generated=130 uses=1531 missing=0
PHASE_ANNOTATIONS=PASS episodes=22 pickups=25 annotations=75
SUCCESS_ATTENTION=PASS episodes=22 complete=1 r3_six_families=1 diagnostic=0 workers=4 seconds=1966.14
EXIT_CODE=0
REPORT_CONTENT_REVIEW=PASS base=22fed762e5dd981162885069c5ede59ed2998ab7 findings=0
ATTN_REPORT=PASS episodes=22 figures=96 findings=0
RUN_HANDLES=PASS known=8 live=0
```

完整motion记录包含130次编码调用、130个去重的局号与窗口组合、279次实际输入快照及1531次跨推理有效使用；96个槽位的完整向量、mask、时间、位置及真实key映射均保存，供未来切换实验选择。共通观察为L3H0持续回看开局、L16H0在抬起阶段更一致，以及全12局存在主要由L17H1承担的固定 `[160,193)` 窗口峰；失败局也有该峰，不作为交换追踪或成功因果证明。详见[完整数据说明](docs/evaluation-doc/attn-buttonunmaskswap-success-seed42-1103/motion-tokens.md)。

96张PNG已由主会话目视，独立内容审查1008项阶段数值完全一致、720项跨局均值最大差约 `4.44e-16`，48项头级计数及门槛一致。全部运行结束，采集和分析工作树已在应用归档；运行工作树因应用认定为主检出而保留。原始数据约218GiB仍在本轮主检出 `v1-store`，不进Git。没有新增下游替换实验或状态机登记。
