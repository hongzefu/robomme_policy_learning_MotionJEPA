# 新库 motion 表构建 + modulation 8×8 接入 motion 计划（v2-motion）

> **状态：80k训练已于2026-09-19 03:24:35 UTC（9月18日23:24:35 EDT）完成，EXIT_CODE=0；16个checkpoint齐全，最终79999真实CPU加载通过。09-20按用户要求回写配对commitV10.2。分窗报告保留原稳态设备覆盖缺口；策略评估另起计划。** 2026-09-16 起草，09-17 经五轮审查修订（第二轮九路核验 32 条外部清单；第三轮三路对抗 52 条；第四轮九路对抗审计 108 条、0 条被推翻、P0 3 条，报告在 `v1-store/reports/audit/0916-sec7-audit-report.md`，不进 git；**第五轮 09-17 按用户指令把正式 run 从 4 卡改为 8 卡**——依据是 `0917-collate-shm-8gpu-plan.md` 的 collate 共享内存改动已并入 `v2-motionmem`（`3868cc9` `commitV9.10`），该轮修订另经 Codex 只读审计、锚定 `26863de`、6 条意见逐条处置）。环境 B（AWS 单机 8×A100），主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`；基线 `v2-1600ep-m8x8-modul-b128-60k` 已跑完，本轮八卡正式训练也已结束，无 turbo、无 GreatLakes。
>
> **第一部分只讲做什么、为什么、哪些不变**；命令、判定行、代码锚点、推导与数字出处全部在第二部分（A–H 按模块，I 节收第一部分精简时移出的细节）。正本 `docs/motion-memory.md` 已补充 modulation 的布局、位置、模型与验证口径，历史 context 记录保留其适用范围。

**用户已拍板**

**09-18 节奏回放裁决**：用户原话「采用35例，保留其他全部覆盖（推荐）」。`eval_rhythm_gates.py --replay-cases representative --expect-replay-cases 35 --expect-real-es 411` 只缩减 rhythm/termination 重复回放：16种余数各取最短/最长，并保留四任务最长及全局最大es1152，共35个真实例。全部411种长度的预算扫描、1296/1297两侧完整1300步、TAU_LONG、V4、V-online和200次reset全部保留。默认all兼容原行为。原411例任务运行1334秒后按该决定中止、EXIT_CODE=143，保留日志且不计通过；5项测试及实际边界短测通过，完整代表例已从修补后的clean HEAD重跑通过，见节奏档案。

**09-18 终止参考裁决**：用户确认「修正验证参考并补用例（推荐）」。35例首次重跑发现g317在16步整边界终止时，eval控制流/通用参考/独立生成点数为67/68/67；只修`eval_rhythm_gates.py::gate_rhythm`的参考帧数截止与`gate_termination`公式，不改生产eval、通用_drive或任何其他覆盖。环境终止推理次数为ceil((T−1−es)/16)，即floor((T−2−es)/16)+1；1296/1297预算边界与411长度扫描保留。19项测试和真实g317短测已通过，完整重跑已从clean782696c通过，生产eval和通用_drive保持不变。

| 项 | 决定 | 日期 |
|---|---|---|
| 接入目标 | modulation 8×8，与基线 `v2-1600ep-m8x8-modul-b128-60k` 同 YAML、只多 `motion` 节 | 09-16 |
| demo 段窗口 | 全覆盖 + 真实帧 ≥ 17；越过段尾的部分用 demo 最后一帧重复填满 33 帧 | 09-16 |
| exec 段窗口 | 不变 | 09-16 |
| 新库 motion 表 | 单独构建 | 09-16 |
| 正式 run 步数 | 80k | 09-17 |
| 80k 落点 | 新具名配置 `mme_vla_suite_b128_80k`；60k 与官方条目一字不动 | 09-17 |
| `motion.budget` | 160 | 09-17 |
| encoder | 换 `wan-full1600-filter2-b176x4-72ep-a` 的 `checkpoint_epoch_72.pt` | 09-17 |
| V-online 口径 | 方案 B「分层解耦」（替换第三轮的「抽 10% 集」） | 09-17 |
| motion 全遮消融评估 | 加入（只做评估侧，不加训练对照 run） | 09-17 |
| encode 续跑 bug | 本轮顺带修 + 加单测 | 09-17 |
| D2 抽样边界 | 只抽 VAE 前向，四道元数据检查全量 | 09-17 |
| 旧配置兼容 | reader 侧默认值：两键同缺按 `(33, "none")`；训练侧与在线侧同口径 | 09-17 |
| 主比较锚点 | 80k run 的 `60000` vs 基线终点 `59999` | 09-17 |
| V5 测试模型 | GPU 上跑，VLM 主干用 `gemma_150m` 替身（乙） | 09-17 |
| 训练卡数 | 4 卡 → **8 卡**（`fsdp_devices=8`、mesh (1,8)、per-device batch 16）；batch 128 与其余超参一字不变 | 09-17 |
| `fsdp_devices=8` 落点 | 写进新具名配置 `mme_vla_suite_b128_80k`，**不走 runner CLI 覆盖**（第 10 条要求先定落点） | 09-17 |
| `num_workers` | 8 → **16**（0917 那次 0.972660 s/步 实测档位即 w16），同样写进新具名配置 | 09-17 |
| 与基线成对可比 | **接受降级、不重跑 8 卡基线**；主比较锚点仍是 80k run 的 `60000` vs 基线 `59999` | 09-17 |
| `run_name` | `v2-1600ep-m8x8-modul-motion-b128-80k`（本轮用户确认采用） | 09-17 |

**历史实施起点（09-17；最终进度以本页顶部和下方最新进度为准）**：用户指令原话「开始实施 有问题尽早问用户」；正式名称确认原话「采用 v2-1600ep-m8x8-modul-motion-b128-80k」。已开始步骤 0–1，起始提交为 `f9f668920923c58e5d7b721c2827da17b6afa865`，主副本工作区干净；环境 B，本地 `/dev/md0` XFS，8 × A100-SXM4-80GB，显存占用均为 0，磁盘剩余约 1.1 TB，`/dev/shm` 剩余 561 GiB。数据目录在位，内容核验尚未执行，所有验证与训练结果均待实测。

用户另确认「采用建议顺序，保留三方强校验」：建库顺序修正为 **Wan → encode → oracle 重算及汇总 → pack/verify → compare → 五项账目**。原因是 pack 要读取 oracle 生成的 `vae_report.json`，必须先生成该报告；oracle 重算只依赖清单与 latent，不依赖 motion 整表。Wan 与 encode 之间零 commit，三个来源的 `raw_dir` 校验继续保留。

步骤 1b 同时纳入基线驱动缓存落点修补，用户原话「一并修补，在修补后的同一驱动上取前后基线」：`run_2gpu_epoch_bench.sh` 去除 `$HOME/.cache` 软链接的创建与删除，改向训练入口传入 `MMEVLA_JAX_CACHE_DIR`，仍指向 `v1-store/cache/jax/<EXP_NAME>`。前后基线均显式设 `KEEP_JAX_CACHE=1`，保留相同缓存。生产训练代码与超参不因该修补改变；BASE 在这次修补提交后取得。

**实施补充裁决（执行时优先于下方原计划表述）**：① 用户原话「纳入修补，确保所有起跑检查失败即停」，新 smoke / 正式 runner 的任务主体使用启用 `set -euo pipefail` 的子 shell，外层继续写 EXIT_CODE；内外日志使用不同路径。② 用户原话「采用真实 skipped 计数及完整性验收」，取消首次建库 `skipped=0` 的预设，汇总真实 skipped，另记起跑前完成段数；以 3,200 段、完整窗口集合与无重复处理验收。③ 用户原话「纳入计时与 trace，按计划提供实测分解」，增加默认关闭的主线程计时与 JAX 设备 trace，本轮 smoke / 正式 run 取前 300 步（smoke 取实际 20 步），不增加逐步设备同步；仅在 trace 收尾等待一次，单独区分主线程取 batch 等待、异步提交和设备 kernel 时间。

**步骤 2a 完成**：BASE=`2126b1b1c436166662ff89a629985ecd4524fc42`；V1 为 3,200 样本 / 200 batch，V2 为 1,200 样本 / 200 batch，V6 为 61 初态叶 / 三类各 38 梯度叶，V7 为 100 步 / 5 状态摘要 / 7 输入摘要 / 872 索引，均退出码 0。详情见 [V1](docs/training-doc/mv2-v1-dump/result.md)、[V2](docs/training-doc/mv2-v2-legacy/result.md)、[V6](docs/training-doc/mv2-v6-grad/result.md)、[V7](docs/training-doc/mv2-v7-guard-base/result.md)。此处记录当时改前基线，后续生产改动和对拍已完成。

---

## 第一部分（给人看）

**最终收尾（09-20回写）**：用户要求「跑完了吗 结束后commit」。80k已于09-19 03:24:35 UTC正常结束，总墙钟21小时37分30秒；60k checkpoint于09-18 18:01:58.713219 EDT完成保存，最终79999于23:24:19.723562 EDT完成保存。5000至75000每5000一步及79999共16份齐全；800条训练指标、4000个标量全部有限，最后记录step79900的loss为0.001474518794566393。最终checkpoint经真实CPU恢复，65/65参数叶完整，配置、norm和库同源全部通过，27秒、EXIT_CODE=0。完成证据、指标、日志和checkpoint清单已补入[训练档案](docs/training-doc/v2-1600ep-m8x8-modul-motion-b128-80k/result.md)。主仓290份既有源码及uv.lock指纹仍与Beta一致；主仓另有6份其他工作的未跟踪导出文件，原样保留，结束提交仅在开发副本完成。策略评估及同权重消融仍另起计划。

**历史起跑进度（09-18；最终收尾见下）**：关闭态V1/V2/V6/V7、71316行建库全量验收、3232样本V4、完整V5和200次真实reset均通过。修正后的35例节奏回放、411长度扫描及1296/1297边界全部通过。开启态V8两轮100步的五标量、5份217叶完整状态、7份输入摘要及800训练索引逐位一致，四个motion参数叶全部更新。V-online三档同源于clean `782696c231aace21c20200638ae302a5a1c7f277`：71316窗输入SHA全量一致；1600集起点/时间码/次序一致；全部1600补帧窗与23个整集去重后的3139真编码窗逐位一致，含g367、4条es≥1000、16种余数及四任务。参见 [建库](docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/result.md)、[V8](docs/training-doc/mv2-aa/result.md)、[V-online](docs/training-doc/mv2-online/result.md)、[节奏](docs/training-doc/mv2-rhythm/result.md) 与 [200次reset](docs/training-doc/mv2-evalbound/result.md)。两个实际runner拒绝负例（占卡、错误YAML SHA）均在训练之前停止；八卡smoke20步、65叶参数树、10个记忆参数叶及配置同源全部通过；原始XPlane恢复了查看器截断的完整20步八卡trace，导出上限修复已通过14项CPU测试与真实GPU探针。正式run从 `2f10473161b760f16d9240d3c2959ff326cde66b` clean起跑、30项preflight全部通过，训练PID545524；主仓290份源码SHA在锁为只读前后不变，回写改在独立开发副本。

**第300步报告的已测与未测**：第100–299步共200步，平均0.987454秒/步、129.626样本/秒；主线程取batch等待0.143053秒/步（14.4906%，不能视为GPU空闲比例）。500ms八卡采样均值98.5777%、0%占比0；慢步/其他步均值95.0536%/98.6415%。第299步后一次trace导出停顿180.894504秒，GPU暂时0%，随后恢复训练。主线程300步完整，但原始XPlane设备事件到第25步内截止，不能恢复100–299步设备分类。用户已确认「保持训练，接受明确标注的分窗报告（推荐）」：100–299步报告步时/取batch等待/利用率，5–24步独立报告设备分解，同时保留原稳态设备覆盖未满足的说明。按稳态步时，第300步剩余约21.86小时，不含后续保存开销。完整证据见 [正式起跑与性能记录](docs/training-doc/v2-1600ep-m8x8-modul-motion-b128-80k/result.md)，80k训练已正常结束，策略评估另起计划。

**V5 初始化裁决已确认**：用户原话「采用兼容权重加载，保留 gemma_150m 替身」。`motion_gates_model.py::_make_models` 对 modulation 测试加载 `pi05_base` 中名称与形状均兼容的参数；`compatible_pi05_weights.py::merge_compatible` 独立要求动作专家及六个 AdaRMS 参数齐全，检查 18 层门不全零，报告保留随机的参数名单。生产初始化和门值不改。此前随机零门导致的失败是测试初态问题，不属于关闭态回归失败。

**历史实施短测（09-17）**：步骤 2b 的生产及验证代码已完成首轮实现，87 项 CPU 自动测试通过；CPU 初态真调用 `init_train_state` 完成 61 个公共参数叶逐位比较，仅增加 4 个 motion 叶，耗时 116.4 秒。新旧布局读取、真实 H5 补帧、独立抽样删行负例、跨集 reset、GPU 计时及比较器错误 token/符号零负例均已验证。详见 [实施短测记录](docs/training-doc/mv2-implementation/result.md)。后续已分别提交生产链路与验证工具，并完成 clean CAND 的 V1/V2/V6/V7 逐位对拍。

**位置表与仿真环境裁决已确认**：用户原话「采用预生成并校验的 GPU 位置表，CPU 验证装配与次序」与「采用现有仿真环境和单卡渲染，保留 200 次真实 reset」。`verified_gpu_posemb.py` 在 GPU 运行原 `PosEmb3D`，逐位核对整张离线位置表并记录源码、配置、表字节 SHA；`compare_online_motion.py --gpu-posemb-cache` 在 CPU 复核这些证据后注入已验证组件输出，继续严格比较装配和次序。仿真由 uv 调用 `/scratch/hongze/micromamba/envs/robomme/bin/python`，GPU 7 渲染，设置 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`，四任务各独立进程完成 50 次真实 reset，不加载策略模型。输入 SHA 重锚以 `df6fdcc4f9424f00901a107a66d98e19e679a7ba` 完成后，先提交这次验证修补，再用新的 clean BUILD_HEAD 启动 Wan；Wan 与 encode 之间仍为零 commit。

### 这轮做什么

给 1600 集新库（BinFill / RouteStick / VideoRepick / VideoUnmaskSwap，605,611 个执行样本）建一张 motion token 表，然后用已跑完的 modulation 8×8 基线配置**只加一个 `motion` 节**，训练一条 80k 步的「+motion」模型，与基线成对可比。顺序：① demo 段补帧规则 → ② 建表七步 → ③ 新 run 与基线的差异 → ④ context 改 modulation 意味着什么 → ⑤ 训练链路改什么 → ⑥ 推理侧改什么 → ⑦ budget → ⑧ 怎么证明没改坏 → ⑨ 执行顺序。

### 一、demo 段怎么补：数轴

**做什么**：一个 motion 窗口 = 连续 33 帧，喂给冻结的 MotionJEPA encoder 得一个 token。起点钉在段内网格 0, 16, 32, … 上，demo 段和 exec 段各自算网格、不跨段。现行规则 demo 尾部整窗放不下就不编，demo 最后 0–15 帧的运动没有任何窗覆盖。以 RouteStick 一集 `es = 100`（demo 帧 0–99）为例：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
现行规则    [0 ══════════════ 32]                                        s=0   真 33
                     [16 ═════════════ 48]                               s=16  真 33
                              [32 ═════════════ 64]                      s=32  真 33
                                       [48 ═════════════ 80]             s=48  真 33
                                                [64 ═════════════ 96]    s=64  真 33
                                                         ✗ s=80: 80+32=112 > 99，整窗越段，不编
                                                            → 帧 97–99 的运动没有任何窗覆盖
```

新规则：起点只要还剩 ≥ 17 帧真实帧就编一窗，越过段尾的部分用 demo 最后一帧重复填满：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
新规则      前 5 窗与现行完全相同（字节不变）
                                                         [80 ═══════ 99│99 99 … 99]   s=80  真 20 + 补 13（第 99 帧重复）
                                                                  ✗ s=96: 真实帧只剩 4 < 17，不编
```

**为什么**：补的是静止帧，这是对 encoder 输入的事实描述——「前 20 帧真实运动、后 13 帧定格在第 99 帧」；本文不断言 encoder 会把它编码成「动作结束」，那需要另行验证。`es ≥ 17` 的集恰好多 1 窗、且补帧窗永远是 demo 段最后一个窗（可由段长直接推出，不必查表）——这条性质被建库账目与在线对拍反复依赖。

**哪些不变**：exec 段一字不改（当前帧之后的帧根本不存在，没有段尾可补）。

**规模**：新库 demo 窗 34,313 → 35,913，exec 35,403 不变，总行 71,316，补帧窗恰 1,600（= 集数）。这是整表重建、不是增量补行——demo 每段多 1 窗会平移其后所有行号，旧表任何一行都不能复用。同一个网格公式在代码里被独立重写了 10 遍、调用点散在 15 个文件，逐处裁决见 A1。

### 二、建表：七步（0–6），哪步没做

**做什么**：建表链路照抄 400ep 库，链路本身不改，只换口径与规模；另加两步前置（encoder 换型、输入重锚）。现状：新库只有帧特征库，motion 相关的五样（latent、token、整表、oracle、留档）一样都没有；原始 H5、wan 子 venv 在位。

| 步 | 模块 | 改不改 | 状态 | 预计耗时（8 卡） |
|---|---|---|---|---|
| 0 | encoder 换型 | 新 encoder 落盘、资产锚点换新；全仓 7 处写死旧 run 名逐项裁决（A9） | 未做 | 分钟级 |
| 1 | 输入重锚 | 建 motion 前重核四个 H5 的内容 sha256 == 帧库 finalize 时的清单 | 未做 | ≈ 15 min |
| 2 | 改代码 | 见第二部分 A1–A9 | 未做 | CPU 单测 < 5 min |
| 3 | Wan 抽取 | 逻辑不改，只是切出来的 demo 尾窗多了补帧 | 未跑 | ≈ 3.6 h |
| 4 | encode | 数值核心不改；只修续跑完成性判断的路径 bug | 未跑 | ≈ 3 min（按实测回填） |
| 5 | pack / verify | 拼表逻辑不改，只多写新布局名与新契约键 | 未跑 | 分钟级 |
| 6 | oracle 与账目 | MotionJEPA 仓库自己的 VAE / encoder 独立重算逐位比：encoder 全量，VAE 抽样（补帧窗全部 + 其余 10%）；账目五项 | 未跑 | ≈ 1.2 h |

**为什么 encoder 要换**：新 encoder 在本批 1600 集全库上训成；架构与旧 ckpt 一致、可直接加载。真正的坑是全仓 7 处写死了旧 run 名或目录，4 处在本轮必经路径上——最致命的是打包器把 run 名硬编码成旧值，不改的话训练侧核对 provenance 时必 raise，而且要等到建库全绿、留档写完之后才炸。逐项裁决见 A9。

**哪些不变**：Wan VAE、encoder 推理脚本（钉版）、pack / verify 的拼表逻辑、oracle 的独立重算链路。Wan 抽取与 encode 之间不能 commit（provenance 要求两阶段同一 commit，400ep 曾因此重抽全量）。新表布局换带 `demopad17` 的新名，旧 40ep / 400ep 表照旧可读。

### 三、新 run 与刚跑完的基线：逐项差异

**做什么**：基线是 modulation 8×8、motion 关闭的 60k run；新 run 是同一条链路只开 motion + 步数改 80k。

**只有这 11 行不同**（计数订正：第四轮正文写「9 行」，但表内实际只有 8 个数据行；8 行 + 第五轮新增的 3 行 = 11 行）：

| 项 | 基线 | 新 run |
|---|---|---|
| 具名配置 | `mme_vla_suite_b128_60k` | `mme_vla_suite_b128_80k`（改 `num_train_steps`、`decay_steps`、`fsdp_devices`、`num_workers` **四项**，第二部分 C 节） |
| history YAML | `perceptual-framesamp-modul-8frame-8x8.yaml` | 同名加 `-motion`（= 基线 + 一个 `motion` 节） |
| `--history-config-sha256` | 基线 YAML 的 sha | 新 YAML 的 sha |
| `run_name` | 基线名 | `v2-1600ep-m8x8-modul-motion-b128-80k` |
| `--run-root` 子目录 | `…/mme_vla_suite_b128_60k/<RUN>` | `…/mme_vla_suite_b128_80k/<RUN>`（随配置名派生） |
| tmux 会话 | `m8-prod` / `m8-prod-dense` | `mv2-prod` / `mv2-dense` |
| `LOG` / `TRAIN_RECORD_DIR` / `MMEVLA_JAX_CACHE_DIR` | 随 `$RUN` 派生 | 同式，照抄 runner 即生效 |
| 卡数 / mesh | `fsdp_devices=4`、mesh (1,4)、per-device batch 32 | **`fsdp_devices=8`、mesh (1,8)、per-device batch 16** |
| `num_workers` | 8 | **16** |
| GPU 可见范围与采样范围 | `CUDA_VISIBLE_DEVICES=4,5,6,7`，两路 `nvidia-smi --id=4,5,6,7` | **全部 `0,1,2,3,4,5,6,7`**（runner 内共 6 处，清单见 G 节） |
| preflight 的 motion 期望值参数 | 无 | 新增 4 个参数（第四轮新增的第 11 行） |

**为什么是「只多这些」**：lr 逐步完全相同（`peak_lr == decay_lr`，改 `decay_steps` 不改任何一步的 lr，V0 直接证明）；数据顺序与步数无关（索引序列只由 seed、batch size、数据集长度决定）；启动时的三个 `env` 变量不在 runner 里、要照抄基线的 `launch.actual.json`，否则缓存落到 `$HOME`。

**换卡数改了什么、没改什么（限定命题）**：这里能断言的只有一句——**固定模型定义、输入、初始参数与随机流后，把分片从 4 卡换到 8 卡，不改变 global batch 128 所对应的优化目标**。五条结构性依据：① loss 是 `scripts/training/train.py::loss_fn` 的 `jnp.mean(chunked_loss)`，对整个 global batch 取均值，全局 batch 仍是 128，数学定义与分片数无关；② 全仓无 `BatchNorm` / `batch_stats`，归一化都是 per-sample 的 LayerNorm / RMSNorm、不跨样本，per-device batch 32→16 不改变任何单个样本的前向数值定义；③ 梯度裁剪是 `optax.clip_by_global_norm(1.0)`（`src/openpi/training/optimizer.py`），全局范数与分片方式数学无关；④ `train_rng = jax.random.fold_in(rng, state.step)`，随机数只由 seed 与 step 决定，JAX 的 SPMD 语义下同 key 同形状与 mesh 无关，EMA 0.999 是逐步参数级操作、同样与卡数无关；⑤ 索引序列由主进程 sampler 生成，只由 seed、batch size、数据集长度决定，与 `fsdp_devices` 和 `num_workers` 都无关——⑤ 是 torch `DataLoader` 的**结构性事实**，0917 的 `INDEX_SEQ=PASS n=17024` 是同 w16 下的实测，**w8→w16 这一跳本轮不另测**，不得当作实测引用。

**两处不能写强**：㈠ 差异来源不是「纯粹的归约次序」——换分片同时可能改变归约次序、kernel 选择、算子融合与编译重排（JAX 官方 FAQ「jit changes the exact numerics of outputs」），正确表述是**可能产生并累积浮点差异、不承诺跨拓扑逐位一致**；㈡ 不断言这些差异「必然放大到曲线可见」，只保留可操作的那一条——**新 run 与基线不能再用逐位判据比较，只能做统计比较**。㈢ 以上命题**只覆盖换卡数**，不覆盖新 run 与基线的关系：新 run 开了 motion，模型函数与参数空间本身就变了（参数叶 61→65、记忆区 512→672），那正是本轮要测的东西。

**其余一字不动**：batch 128、EMA 0.999、AdamW 裁剪 1.0、冻结过滤、`pi05_base` 权重、每 5k 保存、seed 42、warmup 5k / lr 5e-5、assets 与数据集路径、`norm_stats`（沿用基线 sha，**不为开启态重算**）、XLA 内存比例、线程数、runner 外壳（**采样卡号随可见卡同改**，6 处清单见 G 节）。

**开 motion 带来的差异**（这是要测的东西，不是配置差异）：参数叶 61 → 65（多 1.77 M）；记忆区长度 512 → 672；每样本多交付四个键约 658 KB（b128 约 84 MB/batch）；dataloader 多常驻 219 MB × **16 worker ≈ 3.5 GB** 的 motion 表。`/dev/shm` 峰值按 16 worker × prefetch 2 × 约 597 MB ≈ **19 G** 估；0917 F4 实测当时余量 561 GiB，那是**历史余量**，起跑前重查。

**步时**：8 卡关闭态实测 **0.972660 s/步**（b128、w16、mesh (1,8)，稳态窗口 step 100→290，出处 `docs/training-doc/bench-collate-shm-8gpu/result.md`），80,000 × 0.972660 s = **21.6 h**（精确 21.6147 h），**这是下界**。**前提**：起跑源码必须含 `3868cc9`（`commitV9.10`）的 collate 共享内存交付，否则 8 卡回落到 1.77 s/步。相对 4 卡 52.0 h 的 2.41 倍是**相对历史 4 卡配置的综合改善**，同时含共享内存、`num_workers` 8→16、卡数 4→8 三项，不可归因于任何单一因素。改后 util 均值 97.87% 只说明「几乎总有 kernel 在执行」，**不足以定位瓶颈归属**（NVML `utilization.gpu` 不区分计算与通信），且该数字是 **motion 关闭态**；300 步处按 `AGENTS.md` 第 16 条口径重估 ETA。细节见 I.9。

**评估侧交接**（评估口径另起计划）：基线从未做过策略评估、且没有 60000 ckpt（末点 59999）——基线评估是下一轮必做项；新 run 评 60000、基线评 59999，同 seed / 同 1300 步 / 同 test split；评估前先做 `EVAL_ES_BOUND`（第七节）。

### 四、context 改 modulation：改了什么、哪里等价、哪里不等价

**结论**：motion memory 最早只在 context 上实现，正本全文是 context 口径。改到 modulation 上，数据侧一个字节没变，模型怎么用这份 motion 彻底换了一套——两者**不等价**；最要紧的副作用是 `motion.budget` 从「缓冲区大小」变成了**模型语义参数**。

**哪些一样（六层）**：关闭态逐位相同；离线表、切窗、起点集合、补帧规则、四键交付在训练与在线是同一份代码；在线装配代码里没有一行按 integration 分支；sidecar 协议不变。所以在线链路可以直接复用，由 V-online 证明。

**哪些不一样（模型侧）**：
- 注入点：以前 motion token 和帧 token 一起排进主干序列最前面，和图像 / 文字 / 动作 token 在 18 层里互相看；现在主干里没有记忆区（长度和基线一样），记忆单独算一次，每一层让 20 个动作 token 做一次 cross-attention 去读它，读出来的向量拿去调制 FFN 前的归一化。
- 谁看得到：以前文字和动作都看得到 motion；现在图像和文字完全看不到，只有动作 token 看。
- 记忆是否被逐层加工：以前走 18 层残差流；现在同一份记忆广播给 18 层，记忆 token 之间没有 attention。
- 位置编码：以前 padding 不占号；现在 padding 占号，记忆区一变长，所有 query 的位置整体平移。
- 宽度：2048 → 1024，新参数 3.35 M → 1.77 M。
- 训练动力学：motion 梯度只来自 20 个动作 token，通道又被一个近零初值的门挡着——能不能学出来不能靠「接线对了」推断。
- 推理：主干 kv_cache 不含记忆、形状不随 motion 变；cross-attention 每个去噪步每层重算、没有缓存，但折合每集只有约 +9 ms，评估耗时的大头仍是 sidecar 逐窗编码。
- 一条不要误修的：modulation 下 `mask_na` 恒等空操作，训练与推理传不传都等价，不构成训推不一致。

**为什么 budget 变成模型参数**：`history_gemma.py::MemoryAttention` 的动作 query 位置是 `512+budget+i`，有效 memory key 位置是交错后的 `p_j`，因此 RoPE 距离为 `512+budget+i-p_j`。帧路有效位满512时，`budget−k`（本库19..154）是尾部padding长度；最后一个motion后还可能有采样帧，所以它不等于动作到最后motion的距离。固定有效key排列时增大budget会把这些距离整体平移，均值增加、方差保持不变。这会改变模型语义；训练、评估和在线三侧的budget必须逐值相同，而参数树形状不受budget影响，单靠参数树校验无法发现差异。

**由此新增什么**：
- `INIT_COMMON`（秒级 CPU）：6 条 modulation 叶 + 4 条 motion 叶与基线是否同初值，现有验证没有一项能核到。
- `ROPE_LEN_EFFECT`（秒级 CPU）：第三轮引用的五个「位置平移效应」数字没有留档来源，重取后再写进正本。
- `EVAL_ES_BOUND`（起跑前，单卡渲染）：把「评估集 es 不可预知」变成实测，见第七节。
- **motion 全遮消融评估**（用户拍板）：训完在同一 ckpt 上把 motion 全遮再评一次，同权重、同长度、只是看不到 motion 内容，Δ 即 motion 内容净贡献——没有它，新 run 与基线的差分不清是「motion 有用」还是「记忆区 512→672 改了位置编码」。本轮承诺的是实现等价，不是因果归因。

**由此改口的表述**：两处「帧 token 彼此的 RoPE 相对距离改变」不成立（记忆 token 之间没有 attention），改为「每个记忆 token 的绝对位置改变」；「唯一 width=1024 的是 `gemma_300m`」是计数错（有三个）；「唯一拦得住 budget 的是快照 sha 核对」归因错（真正的结构性保证是评估侧只认 run 内快照）；`scripts/motion-variance` 的 +5.00pp 结论是 context 口径，结构性不适用、不得引用。差异矩阵（含代码锚点）、三条新验证的判据、措辞纠错表与正本待改段落清单见 I.1–I.3。

### 五、训练链路：哪些不动，哪些改

**不动的**：帧路的数据交付（基线 batch 的帧路内容逐字节不变）；dataloader 的 motion 取样逻辑（公式在契约层）；模型的两层 motion 投影与 `embed_memory`（context 版已写好，modulation 复用；参数只在开启时创建，关闭态模型与基线完全相同；逻辑不动但注释要改）；modulation 的消费路径（出口维正好 1024，不用动）；基线 YAML、60k 配置、官方配置、Wan VAE、变体表；`norm_stats`（与 motion 解耦，沿用基线，不得为开启态重算）。

**要改的**：
- motion 表契约：加规格表，让「demo 最少真帧数 / 尾部补帧方式」成为布局的属性，挂在数据对象上而不是改函数签名（25+ 个调用点零改动、自动对每张表用对口径）；新旧表共存；表 schema 不升版（升了旧表在回归里必 FAIL）。
- dataloader 守卫：budget 从写死 96 放开为 16 的倍数；加两键三态核对（同缺 → 历史契约；缺一 → raise；齐备 → 与表逐值相等），位置必须在表加载之后。
- 参考侧 `ref_npy_dataset.py`：除 budget 守卫外还有一条「必须是 context」的形制闸，放开 modulation。
- 模型闸：`HistoryPi0.__init__` 现在 motion 只放行 context，改成也放行 modulation，全仓就这一处。
- 新 YAML = 基线 YAML + `motion` 节；新具名配置 = 复制 60k 条目改**四项**：`num_train_steps`、`decay_steps`、`fsdp_devices=8`、`num_workers=16`（C 节；只改步数会静默继承 `fsdp_devices=4`）。

**为什么**：这些改动全部是「让新表 / 新 budget / modulation 能被读到」，不改任何数值路径——关闭态逐位不变由第八节 A 组证明。budget 作为模型语义参数的一致性由 V9 的跨源比对与在线前置门守住；RoPE 语义的技术表述见 I.4，代码级改法见 A–D 节。

### 六、推理侧：哪些不动，哪些改

**先画界线**：在线装配层对 integration 完全不敏感，本节改动全在装配层；模型消费层的差异在第四节，本轮不改也不要「修」。

**不动的**：sidecar 进程、线协议（仍是 33 帧一包）、客户端、exec 段增量编码、policy 的记忆装配。补帧发生在「凑齐 33 帧」之前，sidecar 收到的永远是 33 帧。

**要改的**：
- `FrameSampMemory`：demo 判据从「整窗放得下」改成「还剩 ≥ 17 帧」，凑窗时用 demo 最后一帧补齐；该谓词在文件里出现 4 次、两次是同一条件写两遍，极易只改前者。
- policy 层：透传两个补帧口径键，缺键按历史默认。
- encoder 目录传递：sidecar 默认目录写死旧 run，两个构造点都不传目录，换 encoder 后握手必失败——必须传。
- 两处 stub 帧号校验放宽（sidecar 不改的唯一例外）：合法补帧窗的尾部是重复帧号，现行严格连续校验会误判；这一改动不触及协议 sha，既有 provenance 记录一字不动。
- 在线侧 fail-loud：eval 不构造 dataset，训练侧的两键核对拦不住在线；在 `FrameSampMemory` 也加一道，且必须与「旧配置兼容」同口径按对放行，否则历史快照回归必 FAIL。

**怎么证明同源（方案 B）**：本轮唯一新增的机制是 demo 段 min_real 33→17 + 补帧，风险落在「输入 33 帧是否装配正确」；encoder 的「同输入 ⇒ 同输出」已被数值钉版、握手 provenance、跨卡逐位实测反复确认。所以用零 GPU 手段（在线装配的 33 帧算 sha 与离线表已有的 sha 逐窗比）100% 覆盖全部 71,316 窗，真 encoder 只跑全部 1,600 个补帧窗与必含集（8 卡约 7 min）。第三轮「抽 10% 集」的前提用错了单卡口径（8 卡全量其实只要 3.5–4 h），且实算漏掉 k 最大与全部 `es ≥ 1000` 的集。三段判据见第八节 V-online，证据与成本出处见 I.5。

新任务的策略评估口径本轮不定，另起计划；成对可比的交接约束在第三节末。

### 七、budget：已拍板 160

**做什么**：按 ≥ 17 规则逐样本实算合法窗数（605,611 个）：均值 43.9、中位 39、P90 80、P95 93、P99 115、最大 141（BinFill ep367，es 1152）。budget 超限从不截断、一律 raise，所以 budget 必须 ≥ 141；160 是 16 的倍数里第一个留有裕度的。

| budget | 超限样本 | 占比 | 超限 episode | 平均填充率 | 1300 步评估口径下末段 raise 的 episode |
|---:|---:|---:|---:|---:|---:|
| 96 | 24,366 | 4.02% | 112 | 45.2% | 951 |
| 112 | 7,817 | 1.29% | 43 | 39.1% | 271 |
| 128 | 488 | 0.08% | 6 | 34.3% | 112 |
| 144 | 0 | 0 | 0 | 30.5% | 6 |
| **160** | 0 | 0 | 0 | **27.4%** | **0** |

**为什么不是 176**：budget 是模型语义参数（第四节）。固定有效key排列时，160→176使每个动作query到有效key的RoPE距离增加16；不能推断该平移让距离方差增加。K/V计算量也随之增加，实际训练成本由本轮第300步分解测量。

**评估集**：最后一列度量的是训练集外推（1300 步下最大 151 ≤ 160），评估集走新 seed、es 不受训练 manifest 约束。零截断的充要条件是评估 `es ≤ 1296`；训练集上界 1152、裕度 12.5%，唯一风险线是 BinFill hard。**起跑前用 `EVAL_ES_BOUND` 实测**（200 次真实 `env.reset()`，单卡 GPU 渲染），把赌变成测。`es ≥ 1297` 的真实后果不是「该集记 error」而是整轮评估中止并触发全量重跑，评估计划另起时须先修评估脚本的错误处理。推导备注（填充率定义、τ_max 修正、评估脚本控制流）见 I.6。

### 八、怎么证明没改坏

**六组、12 项，每项一句话讲证什么、为什么；命令与判定行全在 F 表（每行标「实测」或「新写」），每项五层叙述在 I.7。** 本轮承诺的是实现等价与可复现，不是因果归因；「motion 有没有用」由消融臂回答。

**A. 关闭态：改前 vs 改后一个比特都不能变**（三项都在旧库小档位上证，生产档位属推论——关闭态下本轮改的代码大多根本不执行）
- V1 dataset 交付逐位：同一批 3,200 样本改前改后逐字节比，排除交付层被波及。
- V6 定点梯度逐位：同一固定 batch 的单步梯度逐叶逐字节比，排除数值路径被波及。
- V7 100 步守卫：真训练 100 步逐步 loss / 参数摘要 / 输入摘要逐字节比，排除累积差异。这是 `AGENTS.md` 第 18 条第二块唯一一项。
- 两侧源码身份三个比较器都不核对，必须人工断言；FAIL 先分辨是 harness 失配还是数值失配，数值失配才回滚。

**B. 旧东西还能用**（V2 三条）：旧表能读并解析出历史契约；旧 YAML + 旧表的交付改前改后逐位不变；历史 run 快照真走一遍 policy 构造（起 stub sidecar、不占 GPU）。三条能过的前提是第五节的 schema 不升版、三新键按 layout 兜底、在线守卫按对放行。

**C. 新表本身对不对**（V3，建库侧）：输入重锚 → Wan → encode → pack / verify → oracle 独立重算（encoder 全量逐位、VAE 抽样只缩前向）→ 账目五项。本轮 D1 不跑，其「像素同源」位由输入重锚顶替。

**D. 开启态新东西对不对**
- V4 交付逐位：dataloader 交付的 motion 行与表字节逐位比（复用现成的 `hand_calc_8frame.py`），样本集写死——1,600 冷启动、1,600 首 exec 窗、32 个 k≥140、16 个补帧档。第三轮只有计数、没有比较谓词，是 P0。
- V5 接线正确：padding 塞垃圾不影响 loss 与全部可训练参数梯度、motion 内容变了 loss 必变、4 叶有梯度、次序有效；带确定性探针（仓库有过同类叶不确定的实测）；VLM 主干用 `gemma_150m` 替身。
- V8 开启态 A/A：同一命令跑两次 100 步逐位可复现（含四键摘要），并确认 4 条 motion 叶 100 步内确有更新。开启态没有旧链路，第 18 条第二块不适用，登记为盲区。
- V-online 三段（方案 B）：① 在线装配 33 帧 sha vs 离线表 sha，全部 71,316 窗零 GPU；② stub 档全 1,600 集比起点集合 / 时间码 / 次序；③ 真 encoder 只跑全部 1,600 补帧窗 + 必含集。承诺口径原话在 F 表。
- V9 **8 卡** 20 步 smoke：跑通、参数树 65 叶精确匹配、真实 ckpt 走生产口径加载、budget 跨源一致、norm_stats 未变。
- V10 起跑 preflight：30 项，其中 5 项 motion 检查的期望值由 runner 显式传参。

**E. 秒级 CPU 单测**：V0 lr 逐步相同、`INIT_COMMON`、`ROPE_LEN_EFFECT`、A8 离线边界用例表（驱动全部 10 处独立实现）、`EVAL_ES_BOUND`。

**F. 跑完后**：motion 全遮消融评估 + 基线 59999 评估（下一轮）。

**两块划分**：第一块（非训练轻量对拍）= V1 / V2 / V4 / V-online；第二块 = 只有 V7；V6 单列单步对拍；V8 登记盲区；其余不属两块。第二块不通过不得宣称改动等价。链路图见 H 节。

### 九、commit 与执行顺序：从现在到 80k 起跑

主副本上顺序执行，每个 commit 后立即 push。建库 8 卡全开；**V-online ③、smoke 与正式 run 都独占 8 卡（GPU 0–7）**。固定卡号的几项与生产卡数无关、保持原样：**V6 / V7 关闭态 2 卡 b8 对拍（GPU 4,5）、V8 开启态 A/A（GPU 4,5）、V1 CPU 交付验证（无 GPU）、V5 1 卡模型档（GPU 4）**。tmux 会话名清单、清理纪律、每步的工具链与参数见 G 节与 I.8。

| 步 | 做什么 | 为什么 | commit |
|---|---|---|---|
| 0 | 确认 `run_name`；磁盘 ≥ 400 G | 第 6 条 | — |
| 1 | 本文入库 | — | `docs:` |
| 1b | 只改验证工具白名单与形制闸，不触生产链路；之后取 `BASE` | 不先入库，取「改前」证时白名单与 clean-source 校验互斥 | `fix:` |
| 2a | 在 clean BASE 取「改前」证：V1、V2②、V6、V7 | 关闭态等价的基线侧证据；卡号钉死 4,5 | — |
| 2b | 实施 A–E 生产代码 + F 节验证工具 + CPU 单测 | 工作区 dirty，不跑带校验的工具，禁 `uv add` | — |
| 2c | 两个 commit：生产改动 + 验证工具 | 回滚只碰生产改动 | `commitV10.0` + `commitV10.1` |
| 2d | 取「改后」证并对拍 V1 / V2 / V6 / V7 | 任一数值项 FAIL → 只 revert V10.0，停下交用户；harness 失配先排查 | — |
| 3 | encoder 换型 + 资产锚点 + 输入重锚 | — | `fix:` |
| 4 | 建库四步 + 账目五项 | Wan 与 encode 之间零 commit | — |
| 5 | 建库留档 | — | `docs:` |
| 6 | 开启态验证，串行标卡号：6a V4 ‖ V5 → 6b V8 → 6c V-online ①② → 6d V-online ③（独占 8 卡）→ 6e smoke | 每档结束确认前一档进程已退，避免残留占卡 | `docs:` |
| 7 | 起跑留档 + `EVAL_ES_BOUND` | — | `docs:` |
| 8 | 生成 runner（含 G 节 6 处卡号改写）→ 独立占卡闸 `GPU_IDLE`（**0–7**）→ preflight → 训练；300 步重估 ETA | 独立空卡复核，加严格子shell保证每项起跑检查失败即停 | — |
| 8b | 起跑后归档 | 第 12 条 | `docs:` |
| 9 | 训完：消融评估 + 基线评估 | 归因 | 另起计划 |

## 第二部分（技术细节，供 agent 追踪）

### A. 契约与公式

**A1 `src/mme_vla_suite/datastore/motion_store.py`**

三个新契约键：`demo_min_real_frames`、`exec_min_real_frames`、`demo_tail_pad`。**YAML 只承载 demo 两键**；`exec_min_real_frames` 由规格表固定为 33、不进 YAML、不参与 dataloader 的 `_req` 比对。

- **常量与规格表**：`LAYOUT = "motion-768-grid16-demopad17-v1"`；`LAYOUT_SPECS = {"motion-768-grid16-v1": LayoutSpec(demo_min_real=33, exec_min_real=33, demo_tail_pad="none"), "motion-768-grid16-demopad17-v1": LayoutSpec(17, 33, "repeat_last")}`。**`LayoutSpec` 必须是模块级 `@dataclass(frozen=True)`**——`IndexEntry` 是 frozen dataclass，其 `__eq__`/`__hash__` 逐字段递归，且 `FrameSampDataset` 持有 `MotionMeta`（含 entries）并定义了 `__getstate__`，8 个 worker 要 pickle 它。
- **导入守卫必须同改**（不改则模块 import 即 RuntimeError）：现有 `if not LAYOUT.endswith(f"-{LAYOUT_GRID_SUFFIX}-v1"): raise` 对新名求值为 False。改为遍历 `LAYOUT_SPECS`，对每个 key 断言 `f"-{LAYOUT_GRID_SUFFIX}-" in key and key.endswith("-v1")`，并断言 `LAYOUT in LAYOUT_SPECS`。
- **两处 layout 等值校验必须同改**（初稿只提了 `load`，漏了 `parse_index`；而 `MotionMeta.load` 内部正是调 `parse_index`，不改则 V2「旧表可读」必 FAIL）：`parse_index` 与 `MotionMeta.load` 里的 `if need("layout") != LAYOUT: raise` 一律改为 `raw["layout"] not in LAYOUT_SPECS` → raise。
- **`parse_index` 的逐段公式核对**（`int(s["num_chunks"]) != seg_num_chunks(L) or int(s["num_grid"]) != seg_num_grid(L)`）必须用**本表 layout 查出的 spec**，不得用模块默认值——不改这里，新库的 `motion_index` 一加载就 ValueError。
- **公式签名**：`seg_num_chunks(L, min_real=33)`、`seg_num_grid(L, min_real=33)`、`segment_grid_starts(L, min_real=33)`（该函数全仓无真实调用点，改它无收益但无害）。
- **`spec` 的传递方式 —— 挂在数据对象上，不加进各函数签名**：
  - `IndexEntry` 增 `spec: LayoutSpec` 字段，由两条产生路径各自填好——writer 走 `build_index_entries(manifest, spec)`（**必填位置参数，不给默认值**），reader 走 `parse_index`（从 `raw["layout"]` 查表写进每个 entry）。**`parse_index` 的返回值类型不变**（初稿写「spec 随 entries 一起返回」措辞会被理解成改返回签名，那样 4 个调用点全断）。
  - **`MotionMeta` 增 `spec: LayoutSpec` 字段**（初稿漏了，而 A6/C 三次引用 `meta.spec`），由 `MotionMeta.load` 从 `parse_index` 的结果填入。A5 里 `_independent_visible` 读的 `store_meta.demo_min_real_frames` 是**第三个口径**，须统一：`store_meta` 也写三新键，**但 reader 按 layout 分档读取、不塞进现有 `need()` 常量循环**（第四轮修正：该循环缺键即 raise，两张旧表都没有三新键，照第三轮字面加进去 V2① 同样必 FAIL、报的是「缺字段」）——新布局必须三键齐备且与 `LAYOUT_SPECS[layout]` 逐值相等；旧 layout 允许缺失并按 `LayoutSpec(33, 33, "none")` 兜底、只缺部分即 raise（与 C 节三态同口径）。
  - 于是 `visible_motion_rows(entry, t)` 与 `max_visible_count(entry)` **签名一字不改**，25+ 个调用点零改动且自动对每张表用对口径。
  - `visible_motion_rows`：demo 条件用 `s + (entry.spec.demo_min_real − 1) ≤ es − 1`；exec 条件不变。
  - **`pack_motion_store::dataclass_tuple` 同步加 `spec` 字段**——理由不是「防止漏掉规格差异」（`cmd_verify` 两侧 spec 同源、该比对对 spec 恒真），而是保持逐字段完备、避免日后加字段时漏。
- **`index_payload` 必须加 `spec` 与 `layout` 形参**（初稿只说「写三新键」）：现行它取模块常量 `LAYOUT`。A8 要求 `test_guards.py` 分「旧 spec 组 + 新 spec 组」，旧组用 `build_index_entries(m, OLD_SPEC)` 造 entries 而 `index_payload` 写新 `LAYOUT` + 新三键、`parse_index` 再按新 spec 校 `num_grid` ⇒ 必 `ValueError`。签名改为 `index_payload(manifest, entries, *, spec, layout, mj_repo_commit)`。
- **`INDEX_SCHEMA` / `META_SCHEMA` 不升版（第四轮改口，三路审计独立命中）**：两处 reader（`parse_index` 与 `MotionMeta.load`）对 schema 是严格等值且**先于** layout 检查，升版即让两张 `schema=1` 的旧表在 V2① 必 FAIL、并让 V9 的 `MOTION_STORE_PATH` 负例臂以「错误的报错」FAIL；第三轮的升版理由「旧解析器读新表不报错只错读」也不成立——旧解析器会先在 layout 等值处 raise。本轮只升 A3 的 `METADATA_SCHEMA`（2→3）与 A2b 的 aggregate schema（2→3）。

**同一网格公式的 10 处独立重写 —— 逐处裁决**（初稿的「四处同改」低估）：

| # | 位置 | 本轮处置 |
|---|---|---|
| 1 | `motion_store.seg_num_grid` / `seg_num_chunks` | 参数化（本节） |
| 2 | `wan_common.seg_num_grid` / `seg_num_chunks` | 参数化（A2） |
| 3 | `oracle_driver.expected_segments` | 参数化，demo 17 / exec 33 分段取值（A4） |
| 4 | `framesamp_memory` 的 4 处 demo 谓词 | 同改（E） |
| 5 | `motion_checks._independent_visible` | **按段分支**改写，禁止 import 公式（A5） |
| 6 | `extra_checks.visible` | 本轮不跑 a8/a9enc，**显式标注「不适用新库、不跑」**，避免留下不一致的独立实现被误用 |
| 7 | `motion_gates_model.oracle_visible` | 按 demo `min_real` 参数化（F 表 V5 适配） |
| 8 | ★ `summarize_eval_probe.motion_frames_formula` | **必改**：它是阻断级判据 `EVAL_K_FORMULA` 的期望侧（文件自述「本文件内独立重写 demo+exec 起点公式」），不改则下一轮评估直接 FAIL。本轮不跑，但口径必须同步 |
| 9 | ★ `hand_calc_8frame.expected_motion` | 必改（dataloader 手算参照） |
| 10 | ★ `eval_rhythm_gates.predict_k` | 必改（1300 步评估口径与 budget 的唯一守门，见 E 节） |

**全仓调用点裁决**（`★` = 初稿未提及；总计 `visible_motion_rows` ≥25 处、`max_visible_count` 5 处、`seg_num_grid`/`seg_num_chunks` 十余处，散在 **15 个文件**）：

| 符号 | 调用点 | 处置 |
|---|---|---|
| `build_index_entries`（8 处） | `pack_motion_store::cmd_pack` | writer：传**目标** spec |
| | `pack_motion_store::cmd_verify` | reader：传 `meta.spec` |
| | `motion_checks::cmd_a7` | reader：传 `--motion` 表的 spec |
| | ★ `scan_16task_memory` | 无表可读，**必须显式传规格**；本轮不跑，但签名改后不传即 TypeError |
| | ★ `test_guards`（3）、★ `motion_gates_online::_entry` | 单测/用例：新旧两套规格各一组 |
| `visible_motion_rows`（≥25 处 / 13 文件） | `framesamp_dataset`、`motion_checks::cmd_a9set`、`motion_store::max_visible_count`、★`compare_online_motion`、★`compare_siglip_replay`、★`compare_train_infer_obs`（2）、★`ref_npy_dataset`、★`eval_rhythm_gates`（2）、★`motion_gates_model`、★`motion_gates_online`（2）、★`mv_points`、★`scan_16task_memory`、★`test_guards`（10） | 签名不变，**随 `entry.spec` 自动生效**；只有 `test_guards` 的硬编码期望值要按新旧两套分组 |
| `max_visible_count`（5 处） | `framesamp_dataset`、★`ref_npy_dataset`、★`scan_16task_memory`（2）、★`test_guards` | 同上 |
| `seg_num_grid` / `seg_num_chunks` | `motion_store` 内部、`wan_common`、`extract_wan`、★`scan_16task_memory`、★`compare_online_motion`、★`motion_gates_online`、★`test_guards` | **纯公式调用，一律显式传 `min_real`** |

**A2 `scripts/dataset/wan/wan_common.py`**（子 venv 独立副本）
- 同名常量与 `seg_num_chunks(L, min_real=33)` / `seg_num_grid(L, min_real=33)`。
- `list_segments`：demo 项 `min_real = 17`、exec 项 `33`，每项增 `min_real`；**`num_chunks` 必须用该项 `min_real` 计算**（初稿只提了 `num_grid`，漏了 `num_chunks` → 会写出 `num_chunks=68` 与 `num_grid=6` 这种单文件内部就不自洽的 metadata），并断言 `num_grid == len(range(0, num_chunks, 16))`。

**A2b `scripts/dataset/run_local.py`**（初稿完全未列此文件）
- `aggregate_segments` 的 payload `schema` 2 → **3**，增 `demo_min_real_frames` / `exec_min_real_frames` / `demo_tail_pad` 三键（取自 `wan_common` 常量）**与 `raw_dir`**（A6 的三方断言要用它，现在 payload 里根本没有这个字段），逐段 segs 增 `min_real`。
- 消费端 `oracle_driver::cmd_vae` 的 `schema != 2` 改为「`schema ∈ {2,3}`；为 3 时三键必须与 oracle 自身口径相等，为 2 时必须是旧口径」。
- `aggregate_segments` 补两条集合级检查：逐段 `len(rows) == num_grid`、`rows` 的 `m` 集合 `== range(num_grid)`（现在完全不查，原样搬运）。
- **汇总 8 个 worker 的 `skipped` 打进 `STAGE_DONE`**（现行格式是 `STAGE_DONE stage={} workers={} items={} elapsed={:.0f}s`，**没有 `skipped` 字段**。第四轮订正两条理由：`skipped` 在 `extract_wan.py` 与 `encode_motion.py` **两个** worker 的 `WORKER_DONE` 上都打，B 节 Monitor 正则也**已含** `WORKER_DONE`——真正的问题是那是 8 份分散数字、Monitor 里读不成账，故在 `run_local.py::pump` 里解析累加后汇总进 `STAGE_DONE` 一行；不要改成全量转发 `WORKER_DONE`，会撞 CLAUDE.md Monitor 第 5 条）。

**A3 `scripts/dataset/wan/extract_wan.py::process_segment`**
```python
off = wc.GRID_STRIDE * m
real = min(wc.WINDOW_FRAMES, item["seg_len"] - off)
if real < item["min_real"]:            # exec 段 min_real=33 → 与现行「窗口越段」raise 数学等价
    raise RuntimeError(...)
window = frames[off:off + real]
if real < wc.WINDOW_FRAMES:            # 只有 demo 段能进这里
    last = frames[item["seg_len"] - 1:item["seg_len"]]
    window = np.concatenate([window, np.repeat(last, wc.WINDOW_FRAMES - real, axis=0)])
window = np.ascontiguousarray(window)  # (33,256,256,3) uint8；后续 sha / encode_chunk 不变
```
- `frames` 是**段内切片**，故 `frames[seg_len-1]` 对 demo 段而言全域帧号正是 `es − 1`，与 `pad_source_frame` 定义一致。
- `metadata.json` 每行增 `real_frames` / `pad_frames` / `pad_source_frame`（demo 补帧行为全域帧号 `es − 1`，demo 未补帧行与 exec 段一律 `null`）；`input_frames_sha256` 仍对最终 33 帧算；`METADATA_SCHEMA` 升 3；写入前核对 `num_grid` 与 `num_chunks` 互洽。
- **resume 纪律**：`wan_common::segment_outputs_complete` 只查字节数 / sha / JSON 可解析，**不查 schema 与字段**。exec 段规则不变 ⇒ 残留的旧 schema 2 的 exec 产物字节完全匹配、会被当成「已完成」跳过，其 metadata 永远缺三新字段，一路带到 a11 才以 KeyError 形态炸。处置：给该函数增可选 `expect_schema` / `require_keys`，**且** launch.md 里写死「新库建在空 `wan-latents/` 上，起跑前 `ls wan-latents | wc -l` 必须为 0」。

**A3b `scripts/dataset/wan/encode_motion.py`（修既有 bug，用户已拍板本轮修）**
- 现状：调用 `wc.segment_outputs_complete(out_dir, key + ".f32", …)`，该函数内部拼 `m = out_dir / f"{key}.metadata.json"`，于是去找 `<key>.f32.metadata.json`；而实际写出的是 `<key>.metadata.json` ⇒ **函数恒返回 False**，`and` 右边那个写对了的检查永远短路不到。`extract_wan` 用裸 `key` 调同一函数，三个后缀全对，所以 bug 只在 encode。
- 实证（统一引 `v1-store/logs/p3-post.attempt1-mixedcommit.log` 一份日志）：8 个 worker 的 `WORKER_DONE` items 合计 **4,529** 段，而 400ep 全库 `totals.segments = 600`（≈7.5×）；单 worker 跑到 `windows=6,799` ≈ 全库 6,832 的 **99.5%**，即每个 worker 几乎重编了整个库。
- 改法：完成性检查分别接受 binary 与 metadata 两个路径（或给 `segment_outputs_complete` 增 `bin_stem` / `meta_stem` 两参）；**取得 claim 之后、`unlink` 三件套之前**再复核一次（放反了等于没做）；并把输入 `wan-latents/<key>.bin` 的 sha256 与 metadata 里的 `input_latent_sha256` 绑定核对——**该字段在现行 schema 1 里已经写了**（初稿担心的兼容问题不存在），只需在 skip 判据里加读取与比对，全库 39 GiB 重算 sha ≈ 30 s 量级。
- **stale claim**：`try_claim` 的 claim 文件只在 `finally: release_claim` 时删，worker 被 SIGKILL 会留下 stale claim，该段之后永远没人认领且当场无人报错（到 `aggregate_segments` 的残留 claim 检查才以整段缺产物的 SystemExit 暴露）。续跑前先 `ls motion-tokens/_claims | wc -l` 必须为 0。
- 单测：`scripts/dataset/test_guards.py` 加一条——造两段假产物，断言 skip 生效。**这条单测是 bug 已修的唯一证据**：建库是全新库，`skipped=0` 与「bug 没修」时的输出完全一样，判定行本身零信息量。

**A4 `scripts/dataset/wan/oracle_driver.py`**
- **`expected_segments` 是 vae / encoder / aggregate 三个子命令共用的唯一枚举函数**（初稿只说改 `vae`，照字面改会让 D3 在编码前就 SystemExit：`cmd_encoder` 用 `len(s["starts"]) * CHUNK_BYTES` 校 latent 字节，新 demo 段多一块 latent 即不符）。改为 `expected_segments(manifest, *, demo_min_real, exec_min_real)`，**demo 项用 17、exec 项用 33**（绝不能用单一全局 `min_real`，否则 exec 段被按 17 切、整个 exec 表全错），三个子命令共用同一签名。连带更新：`cmd_encoder` 的字节判据、`row_map` 行数、`aggregate` 的重排顺序（**口径错会让行序静默错乱，比 FAIL 更危险**）。
- **口径参数必须在命令行显式注入**：三个子命令各加 `--demo-min-real` / `--exec-min-real`（**不给默认值**，漏传即 argparse 报错，不静默用旧口径）。
- **独立实现补帧**：`vae` 子命令独立重算起点并**独立实现**补帧（不 import `extract_wan` / `wan_common` 的切窗函数）。
- **抽样协议（用户拍板：只抽 VAE 前向）**：
  - 新增 `--sample-spec "padded:all,rest:0.10,seed:0"`。抽样判据用**确定式** `sha256(f"{seed}:{key}:{m}")` 前 8 字节取模，与分片数、遍历顺序、并发度全部无关；补帧窗（demo 段最后一个 m，可由 `seg_len` 直接推）恒抽中。改 seed 或比例即换库版本。
  - **段文件仍为 `num_grid` 行定长**，未抽中的行写零；**零抽中的段仍产出 `.bin` 与 `.bin.sha256`**，使 `cmd_aggregate` 现有的「段集合全等 + 每段文件必在」两条断言不必放宽（按 `rest:0.10` 逐窗抽样，exec 段平均 22 窗，一段都不中的概率 `0.9^22 ≈ 9.8%` → 约 157 个 exec 段会完全没有文件，不这样做 aggregate 直接 SystemExit）。**代价：oracle latent 与被测同大（39.2 GiB），已计入磁盘预算。**
  - **分片产物必须带分片名**：`sampled_windows.shard<i>of<n>.json`（现行分片一律走 `_shard_name`；8 个进程写同一个 `sampled_windows.json` + `atomic_write` 的 `os.replace` ⇒ 最后一个赢、抽样集合丢 7/8）。`aggregate --kind vae` 合并成 `sampled_windows.json` 并断言 `len(merged) == report["windows"]`；各片报告加 `sample_spec`，aggregate 用现成的 `_same_across` 校各片同 spec。
  - **四道廉价检查保持全量遍历、不受抽样影响**：段集合双向相等、逐段 `num_grid`/`seg_len`/`len(rows)`、逐行 `m`/`seg_offset`/`start_global_frame`、逐窗 `input_frames_sha256`（读 h5 本来就是整段读，成本低）；另加新增的三键全量核对。`vae_report.json` 增 `metadata_rows_checked` / `windows_encoded` **与 `raw_dir`**（A6 三方断言要用），**并把这两个计数打进 `ORACLE_VAE=DONE` 的 print**（现行 print 没有 `metadata_rows=` 字段），`cmd_aggregate` 跨片求和。
- `encoder` 子命令不抽样、全量。
- **耗时提醒**：`cmd_vae` 是**整段读** h5（`read_frames(..., s["start"], s["len"])`），与抽了几个窗无关；本库 1,192,918 帧 ≈ 218 GiB 一帧不少，1.2 h 的估计取决于 h5py 读速，留档里按实测回填。

**A5 `scripts/dataset/motion_checks.py`**
- **`_independent_visible` 必须按段分支改写**（初稿写「换公式」，字面照做会出错）：现行 demo 与 exec **共用**谓词 `if f + 32 <= t`。旧规则下 demo 行恒真；**新规则下 demo 补帧行 `f = es − r`（`r ∈ [17,32]`）⇒ `f + 32 ∈ [es, es+15]`，冷启动 `t = es` 时除 `r = 32` 外全为假 → 独立实现漏掉补帧窗 → `a9set` 在冷启动样本上必 FAIL**。而若把 32 统一换成 `min_real − 1`，demo 侧碰巧对、**exec 侧被放宽成错的**。正确写法：demo 行判 `s ≤ es − spec.demo_min_real`（与 `t` 无关），exec 行判 `f + (EXEC_MIN_REAL − 1) ≤ t`。**禁止从 `motion_store` import 任何公式或常量**——该函数的价值就在独立性，`min_real` 只从 `store_meta.demo_min_real_frames` 读数值。
- **`a9set` 改分层采样**：现行 `--n 500` 纯随机，而 F 表 V4 要求「样本集必须含每个 episode 的 `t = es`」（冷启动是唯一能区分正确谓词与错误统一谓词的样本，全库占比仅 `1600/605611 = 0.264%`，随机 500 条有 26.6% 概率一个都打不中）。加 `--cold-all` 分层参数，判定行改为 `A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0`。
- **`a10` 改为真正独立**：新增 `--manifest` 与 `--demo-min-real`，`want` 从清单的 `num_timesteps` / `exec_start_idx` 现算（`demo: len(range(0, max(0, es−(min_real−1)), 16))`、`exec: len(range(0, max(0, T−es−32), 16))`），`min_real` 取**命令行参数**、一律不读 `store_meta`（不符时单独报一条失配）；**`--expect-*` 写死** `--expect-rows 71316 --expect-exec 35403 --expect-demo 35913 --expect-episodes 1600`，不得从 `motion_index.json` 的 totals 现算（那样等号两侧同源、该半个判据恒真）。
- **新增 `a11`（补帧账，集合级而非只算总数）**：从 `episode_manifest.json` 独立枚举完整窗口集合 `{(段 key, m)}`，与 `wan-latents/*.metadata.json` 的实际 rows 做**集合相等**（双向差集）；断言每段 `len(rows) == num_grid`、`m` 集合 `== range(num_grid)`、`seg_offset == 16m`、`start_global_frame == seg_start + 16m`；demo 逐行 `real_frames == min(33, es − 16m)`、`pad_frames == 33 − real_frames`、补帧行 `pad_source_frame == es − 1`、**未补帧行 `pad_source_frame is None`**；exec 逐行 `real_frames == 33 and pad_frames == 0 and pad_source_frame is None`；并断言 **`es ≥ 17` 的集数 == 1600**；聚合 `metadata.json` 与逐段文件一致。判定行 `A11_PAD=PASS segs=3200 rows=71316 padded=1600 set_diff=0`（**段数是 3200 = 1600 集 × 2 段**，初稿写 6400 是错的，且与 B 节自己的 `items=3200` 打架）。
- **`a5` 不跑**，真实理由是**对照物在本环境不可得**：其默认路径 `--raw-dir-400ep` 与 `--mj-data-raw` 都写死在 `/data/hongzefu/...`，环境 B 下不存在（第 13 条环境 B 段）。它留下的「像素同源」空位由 B 节的 `INPUT_REANCHOR` 填补。
- **`a6` 拆分、不整条砍掉**：前半「新清单 40 条 vs 旧 400ep 清单同身份」只对 40ep 库有意义、不适用；**后半适用且重要**——四方 `manifest_sha256` 绑定 + `check_index_against_manifest` 逐 episode 五字段互校，是**唯一的建库期**双库同源闸。新增子命令 `a6set`（或给 a6 加 `--skip-legacy`），判定行 `A6_SAMESOURCE=PASS`。

**A6 `scripts/dataset/pack_motion_store.py`**
- **`gather_provenance` 的 `run_name` 硬编码必须改（本轮最致命的一条）**：现行返回体里写死 `"encoder": {"run_name": "wan-v8-filter10-72ep-a", …}`，而 `dataloader::_motion_gates` 会把它与新 YAML 的 `source_run` 解析结果逐字段比对（`run_name` / `checkpoint_name` / `epoch` / `state_key`）⇒ 换 encoder 后必 raise，**暴露时点（第四轮订正）是建库四步全绿、建库留档写完之后、步 6 第一个构造 dataloader 的开启态验证（V4 / V8 / V-online 任一）**——不是第三轮写的「preflight 过后」；V10 另加 `CHECK_MOTION_SOURCE_RUN` 兜底。改为从 `args.encoder_run_dir`（或新增 `--encoder-run-name`）推导，并断言「推导出的 run_name == ckpt 路径所在目录名」。
- `cmd_pack` 写三新键与新 `layout`；`cmd_verify` 的 `build_index_entries` 传 `meta.spec`。
- **`gather_provenance` 增两个字段**：`raw_dir`（绝对路径）与 `input_manifest_sha256`，把输入重锚绑进新库 provenance。
- `cmd_pack` 新增三方断言：`wan-latents/metadata.json`（A2b 新加该字段）、`oracle/wan-mj/vae_report.json`（A4 新加该字段）记录的 `raw_dir` 与 `meta/input_manifest.json` 的 `raw_dir` 相同。**第四轮提前**：`run_local.py`（wan / encode 起手）与 `oracle_driver vae` 读完清单后**立即**断言 `realpath(--raw-dir) == realpath(manifest["raw_dir"])`，不等近 5 h GPU 跑完才在 pack 期报；pack 期断言降级为兜底。现在 `run_local.py --raw-dir` 与 `oracle_driver vae --raw-dir` 是两处独立传入、**无任何一处断言二者相同**，一次手误就会让 Wan 与 oracle 读不同目录而对拍照样通过。判定行并入 `PACK_MOTION_DONE`。

**A7 `scripts/dataset/wan/compare_wan.py`（初稿未列此文件，但 B 节要跑它）**
- 现行 `cmd_latents` 三处硬编码「oracle 是全量」：段集合双向相等、`.bin` 大小必须 `== ng * CHUNK_F32`、`for m in range(ng)`，末行还要 `compared == rep["windows"]`。按 A4 的「定长写零」方案，前两条自动满足；仍需：新增 `--sampled <sampled_windows.json>` **与 `--sample-spec "padded:all,rest:0.10,seed:0"`（与 oracle 侧逐字相同、由人写两遍；第三轮 B 节命令漏了它，比较器拿不到重算参数只能退回照单比对）**，`for m in` 改为遍历该段的抽样 m 列表，`compared` 改为按抽样集合计数。
- **比较器必须自己按同一确定式规则重算整张抽样表**，只用 `sampled_windows.json` 做交叉核对（而非当作真值）——否则 sampler 漏抽、分片文件互相覆盖，comparer 跟随同一缩水清单会一起 PASS。并**独立重算补帧窗集合**（demo 段最后一个 m 由 `seg_len` 直接推），断言恰好 1,600 个且各比一次。**三条同时成立才 PASS**：自算集合 == `sampled_windows.json` 双向差集为 0；`compared == len(自算集合)`；记录的 `sample_spec` 与命令行逐字相等。并注入反向用例（人工删行确认非零退出）。

**A8 测试（落点纠正）**
- **`scripts/dataset/test_guards.py` 是 motion 网格公式唯一的 CPU 单测，初稿全文未提**；而初稿点名的 `scripts/training/tests/test_pack_guards.py` 601 行里**没有任何 motion 用例**（全是 framesamp），规格表用例放在那里是放错了文件。
- `test_guards.py` 至少四组用例按旧口径写死，须分成「旧 spec 组（沿用现有期望值）+ 新 spec 组（新期望值）」两套：
  - `test_wan_common_constants_match_motion_store`：`for L in range(0,1300)` 逐 L 比 `wc` 与 `ms` 两份实现——这是 wan 子 venv 与主 venv **互为对照的唯一守卫**，参数化后必须 `for mr in (17, 33)` 各扫一遍，否则新的 17 口径零覆盖。锚点补 `seg_num_grid(100,17)==6`、`seg_num_grid(100,33)==5`、`seg_num_grid(16,17)==0`、`seg_num_grid(17,17)==1`。
  - `test_motion_index_roundtrip_and_totals`、`test_visible_motion_rows_boundaries`、`test_wan_common_list_segments_matches_index`：旧组显式传旧 spec 保留原断言，新组按新规则给新期望。
- **新增离线边界用例表**（纯公式，不依赖真实数据；本库 `es` 全部 ≥ 100，全量对拍覆盖不到短 demo 边界）：

  | es | demo 窗数 | 末窗 真/补 |
  |---:|---:|---|
  | 0、16 | 0 | 无 |
  | 17 | 1 | 17 / 16 |
  | 32 | 1 | 32 / 1 |
  | 33 | 2 | 17 / 16 |
  | 66 | 4 | 18 / 15 |
  | 114 | 7 | 18 / 15 |

  每档再按 `t − es ∈ {0, 16, 31, 32, 33, 48}` 核可见窗集合；**构造夹具时让 exec 首帧像素 ≠ demo 末帧像素**，以捕获「误取第 `es` 帧而非第 `es−1` 帧」这一类差一错。**这张表必须驱动上面清单里的全部 10 处独立实现**（初稿只说四处，把最易写错的 `_independent_visible` 排除在外——它是 a9set 的对照侧、A5 刚重写，且每档都要有 `t = es` 这一行）。
- `a11` 与 pack 三键要有 writer→parser→verify 三段各自的负例（三键缺一 / 三方不同值 / 旧布局文件带新键 / 新布局文件缺新键）；schema 不升版（A1），不需要 schema 正反例。

**A9 encoder 换型（用户 09-17 拍板）**
- 落点：`v1-store/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/{checkpoint_epoch_72.pt,config.yaml}`，源 `/scratch/hongze/MotionJEPA/runs/wan-full1600-filter2-b176x4-72ep-a/`。
- 实测锚点：ckpt `477,432,433 B`、sha256 `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca`；config `2,008 B`、sha256 `4a505440b7c5ff0f1b0d7ec4680800e9296f5ede1df622767be3d5c3aee4c9b4`。
- **全仓写死旧 run 名 / 旧 run 目录的清单与逐项裁决**（本轮必经路径标 ⚠）：

  | # | 位置 | 裁决 |
  |---|---|---|
  | 1 | ⚠ `pack_motion_store::gather_provenance` 的 `run_name` | **必改**，见 A6（不改则训练起跑必 raise） |
  | 2 | ⚠ `run_local.py::ENCODER_RUN_DIR_DEFAULT` | **必改或 B 节显式传 `--encoder-run-dir $ENC`**（`--expected-ckpt-sha256` 会跟 ASSETS_LOCK 换新 sha，用旧目录的 ckpt 对新 sha ⇒ `load_encoder` SystemExit） |
  | 3 | ⚠ `oracle_driver.py::ENCODER_RUN_DIR_DEFAULT` | 同上，D3 起不来 |
  | 4 | ⚠ `motion_client.py` 的 sidecar 默认 `--encoder-run-dir` | **必改**：两个构造点（`compare_online_motion.py`、`policy_config.py`）都只传 `expected_ckpt_sha256`、不传目录 ⇒ V-online 与所有 eval 握手失败。建议给 `MotionEncoderClient` 加「从 YAML `source_run` 推导 run 目录」的路径 |
  | 5 | `paths.sh` 的 `readonly ENCODER_RUN_DIR` / `ENCODER_CKPT` / `v1_require_encoder` | 列入改动清单（B 节自定义 `ENC=` 绕开了它，不会立刻炸，但 `readonly` 让 runner 无法覆盖） |
  | 6 | `extra_checks.py`、`probe_wan.py` | 本轮不跑，显式标注「仍指旧 run，不适用新库」 |
  | 7 | `scripts/motion-variance/eval_shard_mv.sh` | 同上 |

- **`ASSETS_LOCK.json` 要改的字段**（初稿清单不全）：`motionjepa_ckpt` / `motionjepa_config` 两条的 `dest`、`bytes`、`sha256`、`headtail`、`related.run_name`、**`related.mj_train_commit`**（旧值 `7388a42` 是训练旧 ckpt 的 commit，新 ckpt 由不同 commit 训出）、**`source.filename`**（含旧 run 名），以及顶层 `sha256` 重算。
- **⚠ 换锚点会打挂一条现存单测**：`scripts/assets/test_assets_lock.py::test_lock_cross_checks_motion_store_provenance` 断言「40ep 库 `store_meta.json` 里记的 `encoder.checkpoint_sha256` == `expected_sha256("motionjepa_ckpt")`」，而 40ep 库记的是旧 ckpt 的 `bae96037…`，skipif 条件（40ep 库存在）在本机为真 ⇒ **该测试会跑且必 FAIL**。处置：把这条交叉互证改成「按 `related.run_name` 分档」或「对历史库用历史锚点」，并在步骤 3 的验证里点名跑一次 `test_assets_lock.py`。
- **HF 上传缺口**：`source` 段指向 HF 私有仓 `HongzeFu/MotionJEPA` 的 revision + tag，**新 ckpt 尚未上传**。本环境有本地原件不影响建库，但「异地无 NFS 机器从零复刻」路径会断；**另立一项补上传，不阻塞本轮**。同时注意旧 ckpt 一旦失去 lock 条目，所有仍指旧目录的消费者（上表 5/6/7）会同时失效——这些本轮都不跑，但要在留档里写明。
- 新 YAML 的 `source_run: wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder`。
- `scripts/dataset/wan/SOURCE_PIN.json` **不动**：它钉的是推理脚本 `wan_motion_infer.py` 的 sha（`af67fdd9…`），与 ckpt 无关；已核 MJ 活仓 HEAD 虽已前进到 `f43b38f`，该脚本 sha 仍与复制件逐字节相同，D3 的 `module_sha256` 比对不受影响。`paths.sh` 里 `readonly MJ_COMMIT` 指 `2a484ad9`（推理侧钉版），与活仓 HEAD 是两回事，留档里要区分开。

### B. 建库命令序列（主副本，clean HEAD）

每个 tmux 阶段都走同一个 runner 外壳（第 7 条）：`set -o pipefail` + `PYTHONUNBUFFERED=1` + `tee` + `rc=${PIPESTATUS[0]}` + 收尾 `printf 'END_UTC=%s\nEXIT_CODE=%s\n'`，runner 落 `v1-store/logs/mv2-<stage>-runner.sh`（不进 git）。

> 编号说明：B 节按命令顺序自编号（第 1–5 步 + 账目），与第一部分第二节的建表七步（0–6）不是同一套；凡引用「不能 commit」的约束一律用具名表述「Wan 抽取与 encode 之间」，不引编号。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA; source scripts/dataset/paths.sh; v1_prepare_dirs
LIB=$V1_STORE/datasets/4task-v2-1600ep-604f16da; RAW=$V1_STORE/raw-h5/4task-20260912-v2; MJ=/scratch/hongze/MotionJEPA
ENC=$V1_STORE/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a
SHA=0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca
ls -ld $LIB $LIB/meta $RAW $ENC; df -h /scratch      # 实体目录；Avail ≥ 400 G
BUILD_HEAD=$(git rev-parse HEAD); git status --porcelain   # 必须为空
ls $LIB/wan-latents 2>/dev/null | wc -l                    # 必须为 0（A3 的 resume 纪律）
ls $LIB/motion-tokens/_claims 2>/dev/null | wc -l           # 必须为 0（A3b 的 stale claim）
```

**第 1 步 输入重锚**（≈15 min，791.8 GB）——runner 内用一次性 `uv run python -c` 调 `finalize_checks.check_inputs(manifest, raw_dir, input_manifest, level="sha256")`（纯函数、返回 errs、不写任何文件、不拉 jax），**绝不重跑 `finalize_checks.py check`**：后者在全通过时会**覆写** `<lib>/source/meta/provenance.json`（`git_commit` 被换成 motion 建库时的 commit，`finalize_host` / `jax` / `gpu_device_kind` 一并重写），等于用这次核验的指纹污染帧库的建库留档。判定行 `INPUT_REANCHOR=PASS files=4` 由该 wrapper 打印，不新增仓库文件。

```bash
# 第 2 步 Wan（tmux mv2-wan；日志 $LIB/logs/mv2-wan.log）
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1,2,3,4,5,6,7 --raw-dir $RAW
# 红线：wan 与 oracle vae 必须显式传 --raw-dir $RAW。paths.sh 的 RAW_H5_DIR 是普通赋值、没有 export，
#       子进程看不到；run_local.py 的缺省是代码写死的 /data/hongzefu/robomme_data_h5（本环境不存在，
#       漏传即打开失败、不会静默——第三轮把因果说反了）。真正的静默误用风险来自 `export RAW_H5_DIR`
#       或手写 --raw-dir /scratch/hongze/robomme_data_h5（该目录真实存在，是 v1 旧库、同名不同内容的
#       16 任务 h5，误用会在 episode_{j} KeyError 处失败）。禁止 export RAW_H5_DIR。兜底：run_local /
#       oracle 读清单后立即断言 realpath(--raw-dir) == realpath(manifest["raw_dir"])（A6），pack 期三方断言是最后一道。
# 第 3 步 encode（tmux mv2-encode）——与 wan 之间零 commit
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB --gpus 0,1,2,3,4,5,6,7 \
  --encoder-run-dir $ENC --expected-ckpt-sha256 $SHA
# 第 4 步 oracle 重算及汇总（MotionJEPA venv，--mj-repo $MJ 是顶层参数、必须在子命令之前；tmux mv2-oracle）
#   三个子命令都必须显式传口径（--demo-min-real / --exec-min-real 无默认值，漏传即报错）
CUDA_VISIBLE_DEVICES=7 … oracle_driver.py --mj-repo $MJ encoder --manifest $LIB/meta/episode_manifest.json --latents $LIB/wan-latents --out $LIB/oracle/wan-mj \
  --encoder-run-dir $ENC --expected-ckpt-sha256 $SHA --demo-min-real 17 --exec-min-real 33
for i in 0..7: CUDA_VISIBLE_DEVICES=$i … oracle_driver.py --mj-repo $MJ vae --manifest … --raw-dir $RAW --latents $LIB/wan-latents --out $LIB/oracle/wan-mj \
  --shard-idx $i --num-shards 8 --demo-min-real 17 --exec-min-real 33 --sample-spec "padded:all,rest:0.10,seed:0"
… oracle_driver.py aggregate --manifest … --out $LIB/oracle/wan-mj --num-shards 8 --kind vae --demo-min-real 17 --exec-min-real 33
# 第 5 步 pack / verify（tmux mv2-pack；此时 vae_report.json 已生成，三方 raw_dir 校验不能跳过）
uv run --no-sync python scripts/dataset/pack_motion_store.py pack   --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion --encoder-run-dir $ENC --raw-dir $RAW
uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store $LIB/motion --resume
# 最后做数值 compare，输入来自已完成的 oracle 与 pack。
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents $LIB/wan-latents --oracle $LIB/oracle/wan-mj --sampled $LIB/oracle/wan-mj/sampled_windows.json --sample-spec "padded:all,rest:0.10,seed:0"
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens  --store $LIB/motion --oracle $LIB/oracle/wan-mj
# 账目 a6set / a7 / a9set / a10 / a11（a10 的 --expect-* 写死，不取 motion_index.json）
```

> B 节落地时必须展开成完整 runner（真实 interpreter、绝对路径、八片并发启动与逐片退出码收拢）：**全部分片成功才允许 aggregate，aggregate 成功才允许 compare**；不能只靠日志过滤器看到一个 PASS 就继续。

Monitor 过滤管道（每级行缓冲）：`tail -n +1 -F <日志> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E 'STAGE_DONE|STAGE_FAIL|WORKER_DONE|INPUT_REANCHOR=|PACK_MOTION_DONE|VERIFY_MOTION=|ORACLE_.*=DONE|BITEXACT=|A1[01]_|A[679]_|Traceback|out of memory|EXIT_CODE='`。

判定行：`INPUT_REANCHOR=PASS files=4`、`STAGE_DONE stage=wan workers=8 items=3200 skipped=0 elapsed=…s`、`STAGE_DONE stage=encode workers=8 items=3200 skipped=0 elapsed=…s`（`skipped=` 是 A2b 新加到 `STAGE_DONE` 的字段；**注意首次全新建库 `skipped=0` 与 bug 未修时输出相同，bug 已修的唯一证据是 A3b 的单测**）、`PACK_MOTION_DONE=1`、`VERIFY_MOTION=PASS scanned=71316 mismatches=0`、`ORACLE_ENCODER=DONE rows=71316`、`ENCODER_BITEXACT=PASS compared=71316 mismatches=0`、`ORACLE_VAE=DONE metadata_rows=71316 windows=<抽样数> frame_mismatches=0 metadata_mismatches=0`（`metadata_rows=` 是 A4 新加到 print 的字段）、`WAN_BITEXACT=PASS compared=<抽样数> padded_covered=1600 frame_mismatches=0 latent_mismatches=0`、`A6_SAMESOURCE=PASS`、`A7_BYTES=PASS`、`A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0`、`A10_ROWS=PASS rows=71316 exec=35403 demo=35913 episodes=1600`、`A11_PAD=PASS segs=3200 rows=71316 padded=1600 set_diff=0`。

留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/{launch.md,result.md,records/}`，`records/` 只放清洗后日志与判定行，不放 `.sh` / `.yaml`。

### C. dataloader / 配置

- **`src/mme_vla_suite/training/framesamp_dataset.py::__init__`**
  - `_req(int(mcfg.budget) % 16 == 0 and int(mcfg.budget) >= 16, …)`（160 = 10 × 16 通过）。
  - **两键的三态判定（用户拍板：reader 侧默认值兼容）**——必须用 `mcfg.get(k, None)`，**不能用 `mcfg[k]` 或属性式**：实测 omegaconf（非 struct 模式）上前者抛 `ConfigKeyError`、后者抛 `ConfigAttributeError`，都会在比对之前先炸。
    ```
    want = self._motion_meta.spec                      # LayoutSpec(demo_min_real, exec_min_real, demo_tail_pad)
    got_min, got_pad = mcfg.get("demo_min_real_frames", None), mcfg.get("demo_tail_pad", None)
    两键同缺 → (got_min, got_pad) = (33, "none")        # 历史契约；旧 YAML + 旧表照跑，旧 YAML + 新表在末行 raise
    只缺一键 → raise（配置半残）
    if (got_min, got_pad) != (want.demo_min_real, want.demo_tail_pad): raise   # 显式字段比较（第四轮：第三轮伪码把二元组与三字段 dataclass 直接比、字面实现恒不等）；exec_min_real 不参与
    ```
    **位置硬约束**：这段必须写在 `self._motion_meta = ms.MotionMeta.load(...)` **之后**，不能并进现有那段 `_req` 块（那时 `self._motion_meta` 还是 `None`，取 `.spec` 会 AttributeError）。若要严格区分「键不存在」与「键存在但值为 null」，用 `k in mcfg` 判存在性而非 `.get` 的默认值。
  - `__getitem__` 调 `ms.visible_motion_rows(entry, step)`（签名不变，spec 随 entry）；`__init__` 的零截断预检 `ms.max_visible_count(e)` 同理。
  - 三处「32 帧 / 96 / 608」注释改按变量描述。注意 `percep_mem.py` 里还有一个同名但不同义的 `self.config.budget`（帧路 512 位），注释里要区分这两个 budget。
- ★ **`scripts/training/tests/ref_npy_dataset.py`**：硬断言 `(self._motion_budget, self._motion_pos_dim, int(mc.stride), int(mc.window_frames)) != (96, 256, 16, 33) → raise`。它是 V1（`dump_fixture_samples` import 它）与 V8（`BENCH_DATASET_IMPL=refnpy`）的参考侧，**budget=160 会直接 ValueError**。**改法不能写成 `(mcfg.budget, 256, 16, 33)`**——`self._motion_budget` 本来就等于 `int(mc.budget)`，那样 budget 那项变成自比恒真、守卫失效。正确写法：`self._motion_budget % 16 == 0 and self._motion_budget >= 16`，其余三项 `(pos_dim, stride, window_frames) != (256, 16, 33)` 照旧 raise，与 `framesamp_dataset._req` 同口径。**第四轮补：该文件还有一条在 budget 断言之前、无条件执行的形制闸** `(representation_type, integration_type, perceptual_memory.type) != ("perceptual", "context", "frame_sampling") → raise ValueError("参考链形制不符")`，两份 modulation YAML 必撞、且撞在 budget 守卫之前；放开为 `integration_type in ("context", "modulation")`，其余两项照旧严格（放开不降低守卫强度，该类里与 integration 相关的只有这条形制断言），**落在 1b commit**。V1 / V8 显式设 `DTYPE_DUMP_IMPL=refnpy` / `BENCH_DATASET_IMPL=refnpy`（默认 packed）时会用到它。
- `src/mme_vla_suite/training/dataloader.py::_motion_gates`：逻辑不变。注意 `root = os.environ.get("MMEVLA_MOTION_STORE") or str(mcfg.store_path)`，相对路径按 `_REPO_ROOT`（不是 cwd）解析，所以 YAML 写相对路径安全；生产 runner 保持该变量 unset ⇒ **motion 表路径的唯一来源是新 YAML 的 `store_path`**，而 `store_path` 参与 `history_config.resolved.sha256`，改路径即改 YAML sha，preflight 的 `--history-config-sha256` 要同步更新。
- **`src/mme_vla_suite/training/config.py`**
  - `RepackTransform` 注释改 `(b, budget, …)` / `(b, 512 + budget)`。
  - `_CONFIGS` 新增 `mme_vla_suite_b128_80k` = 复制 `mme_vla_suite_b128_60k`，**改四项**：`num_train_steps=80_000`、`decay_steps=80_000`、**`fsdp_devices=8`**（60k 为 4）、**`num_workers=16`**（60k 为 8）；注释写明这四项差异与「因 `peak_lr == decay_lr == 5e-5`，两条配置的 lr 在**任意 step** 逐步相同」。**fail-loud 提示**：若只改步数而漏改 `fsdp_devices`，8 卡可见时 JAX 不会报错、会静默得到 mesh **(2,4)**，per-device batch 与通信拓扑全变而日志毫无提示——必须在 V9 smoke 与起跑留档里显式核对实际 mesh 与 `fsdp_devices`。文件末尾的名字唯一性断言与 `tyro` 精确 subcommand 匹配保证新增条目不影响既有三条。
- **新 YAML** `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml` = 基线 `perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，一字不动）+ `motion` 节：`enabled: true`、`budget: 160`、`demo_min_real_frames: 17`、`demo_tail_pad: repeat_last`、`store_path: v1-store/datasets/4task-v2-1600ep-604f16da/motion`、`source_run: wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder`，其余（`dim 768` / `stride 16` / `window_frames 33` / `window_direction forward` / `grid_origin segment_start` / `pos_dim 256` / `frame_size 256` / `online_gpu`）照抄 `perceptual-framesamp-context-8frame-8x8-motion.yaml`。
  - ★ 备注：`scripts/training/tests/g0_gate.py` 对关闭态 YAML 的 `motion` 节做**精确集合相等**校验。本轮新 YAML 是开启态、不受影响；但日后若往关闭态 YAML 也补这两个新键，那条断言会挂。
- **budget 复核式（评估口径变更时用）**——初稿两处都错，修正后：
  ```
  τ_max = es + 16·⌊max_steps/16⌋                 # 不是 ⌊(max_steps−4)/16⌋
  E(δ)  = 0 if δ < 32 else (δ − 32)//16 + 1      # 等价于 seg_num_grid(δ+1, min_real=33)，直接复用 A1 的函数更稳
  k_eval = demo_num_grid(es, min_real=17) + E(τ_max − es)
  ```
  `max_steps = 1300` 时 `τ_max − es = 1296`、exec 项恒 80，新库 `k_eval` 最大 151 ≤ 160；16 任务全集口径复核同为 151。
- `scripts/training/compute_norm_stats.py::_NONE_KEYS` 不变；norm_stats 沿用新库 `856c75ea…`，**不得为开启态重算**。
- ★ **写死 `96` / `608` 的代码清单**（非注释，逐项裁决）：
  - **必改（本轮跑到）**：`motion_gates_model.py`（`MOTION_BUDGET = 96` 及 m3/m4 里的 32/16/96/608 字面量，V5 的工具本身）、`ref_npy_dataset.py`（见上）、`motion_gates_online.py`、`eval_rhythm_gates.py`（见 E）。
  - **必改（本轮不跑但含独立网格公式，口径不同步会让下一轮直接 FAIL）**：**`summarize_eval_probe.py`** —— 不只是 `--budget` 默认 96，它的 `motion_frames_formula` 是 `EVAL_K_FORMULA`（**阻断级判据**）的期望侧、内含旧 demo 谓词 `while s + (window−1) <= es−1`；**`hand_calc_8frame.py::expected_motion`** —— `[s for s in range(0, es, 16) if s + 32 < es]`，dataloader 手算参照；**第四轮改为本轮必跑（V4 复用它做逐位判据，不另新写）**，三处必改：`for es in (0, 66, 114)` 在新库 StopIteration（es 最小 100、es=66 零集）→ 按 `(es−17) mod 16` 残差分层现挑；YAML stem 写死 context → 参数化到 modul-motion；两处 `96` 字面量（不在下面「写死 96」清单里）→ `budget`。
  - **点名但本轮不动**：`mv_common.py`（`MEM_LEN, FRAME_SLOTS, MOTION_SLOTS = 608, 512, 96`，motion 利用率链路）、`scan_16task_memory.py`（`MOTION_BUDGET = 96`）。**`scripts/motion-variance/*` 是结构性不适用**（`mv_common.py` 的 `PREFIX_LEN  # 1184`、`mv_model_adapter.py` 两处 `integration_type="context"` 硬编码），不只是字面量；本轮不改，且**不得引用其结论（`docs/motion-utilization.md` 的 normal−mask +5.00pp）作为新 run 的 motion 利用率证据**。
  - **纯注释**：`history_observation.py`、`config.py`、`percep_mem.py`、`history_pi0.py`、★`robomme_policy.py`（四行 `(96,768)`/`(608,)` 注释）。第四轮补三处过时注释：`percep_mem.py::__call__` 的「(b, 512+96, 2048)」与 `__init__` 的「1536 → 2048」「177 → 193」、`history_pi0.py::embed_memory` 的「(b,608,2048)」与「长度自动跟随 608」——modulation + budget=160 下应为 1024 / 672。

### D. 模型侧

- **`history_pi0.py::HistoryPi0.__init__`**：`if self.mem_encoder.motion_enabled and self.integration_type not in ("context", "modulation"): raise`。全仓 grep 确认这是 `integration_type` 与 `motion_enabled` 联用的**唯一**断言，「就这一处」成立。另一条相关守卫 `framesamp_dataset` 里已同时允许 `("modulation", 1024)`，无需改。
- `history_observation.py`、`percep_mem.py`、`robomme_policy.py` 只改注释。**特别点名**：`percep_mem.py` 里 `motion_encoder_static` 的中文注释写的是「1536 → 2048，W 1536×2048 + b 2048」——那是 context 版（`memory_token_dim: 2048`）的数字，modulation 下是 **1024**，本轮必须改。另三处（`percep_mem.py::__call__` 的「(b, 512+96, 2048)」、`__init__` 的「177 → 193」、`history_pi0.py::embed_memory` 的「(b,608,2048)」「长度自动跟随 608」）见 C 节「纯注释」补记。
- **motion 四个参数叶**（从模块定义推算，供 V9 断言）：`mem_encoder/motion_pos_proj/kernel (256,768)` + `bias (768,)`、`mem_encoder/motion_encoder_static/kernel (1536,1024)` + `bias (1024,)`，合计 `1,771,264`。基线实测 61 叶 + 4 = **65**。6 条 modulation 叶的来源已定位：`MemoryAttention` 的 `q_einsum_mem` / `kv_einsum_mem` / `out_einsum_mem` / `mem_rms_norm.scale` 共 4 条，加 `HistoryBlock` 里 `MemoryRMSNorm(name="mem_rms_norm_ffn")` 带 cond 分支产生的 `Dense(2*width)` kernel+bias 2 条 ⇒ `MEM_PARAMS` 从 `n=6` 变 `n=10`。
- **6 条 modulation 叶与基线初值是否同源，现有 V 表核不到**（第四轮）：基线 `records/startup.log` 实测 6 条 `Merging missing weight: …mem_attn/{mem_rms_norm/scale, kv_einsum_mem/w, q_einsum_mem/w, out_einsum_mem/w}`、`…mem_rms_norm_ffn/Dense_0/{kernel, bias}`（`weight_loaders.py::_merge_params` 对 checkpoint 缺失键保留随机初值）；两个 motion `nnx.Linear` 建在 llm 之前，若 `ToNNX.lazy_init` 在 `mem_mods=[False, True]` 下从共享 RNG 计数器取键，新 run 的约 85 M mem 参数与基线从不同随机抽样起步。处置：2b 的 `INIT_COMMON` CPU 单测（I.2、F 表）。
- `scripts/training/train.py::init_history_config`：`motion_provenance.json` 的字段按 `enabled` 填、与 integration 无关，**预期无需改**（已核无 modulation 专属漏写）。
- **两个参数树工具的开关名不同，别抄串**：`scripts/training/legacy-eval/check_ckpt_param_tree.py` 用 `--config`（默认 `mme_vla_suite`，须显式带 `mme_vla_suite_b128_80k`），只产出 `PARAM_TREE_EXACT`，且读的是**仓库** YAML、用文件名字符串建模型，完全绕开 run 内 `history_config.resolved.yaml` 快照 ⇒ 它证明不了「部署加载已验证」。走生产口径的是 `scripts/training/g0/check_config_provenance.py::gate_ckpt_param_tree`（用 `_load_resolved_snapshot` + `model.load(remove_extra_params=False)` + `_assert_param_tree_exact`），开关名是 **`--train-config`**，判定行 `CKPT_PARAM_TREE=PASS missing=0 extra=0 leaves=<n> model_leaves=<n> ckpt_has_motion=1`。**它的 `--ckpt/--lib/--train-config/--store-subdir/--neg-lib` 默认值全部指向旧库旧 run，V9 必须逐个显式传**（见 F 表）。
- **budget 不进参数树，参数树校验对 budget 完全失明**：四个 motion 叶的形状只依赖 `pos_dim/dim/pos.hidden_dim/memory_token_dim`，与 budget 无关 ⇒ 拿 budget=96 的 YAML 加载 budget=160 训的 ckpt，`PARAM_TREE_EXACT` 与 `_assert_param_tree_exact` 都会 PASS 而语义已静默改变。**V9 必须显式加一条 `BUDGET_CONSISTENT`**：从 `history_config.resolved.yaml` 读出的 `motion.budget` == 160。**必须跨源**（第四轮修正：第三轮写的「快照 == dataloader 实际值 == 在线 `_motion_cfg`」三侧在代码里同源自一个 DictConfig——`train.py` 只按 CLI 文件名解析一次、同一对象装入 model config 并直传 dataloader；`policy.py` 的 `_motion_cfg` 取自 `create_trained_policy` 换进来的同一份快照——恒等不可能 FAIL，且 smoke 不构造 policy）：判定行 `BUDGET_CONSISTENT=PASS snapshot=160 repo_yaml=160 expected=160`，比 run 内 `history_config.resolved.yaml` 的 `motion.budget`、仓库新 YAML 的 `budget`、常量 160；在线侧 budget 归 V-online 的 CPU 前置门。生产评估侧由 `create_trained_policy` 丢弃传入 history_config、整体换成 run 内快照**结构性保证**；sha 三方核对（三方同目录、由 `init_history_config` 一次写出）只防 run 目录被事后编辑；`train.py` 硬禁 `--resume`。真实暴露面是 `check_ckpt_param_tree.py` 这类按仓库 YAML 文件名建模型的 `--yaml` 驱动工具。

### E. 在线侧

- **`src/mme_vla_suite/policies/framesamp_memory.py`**
  - `motion_cfg` 增 `demo_min_real_frames` / `demo_tail_pad`，读取一律用 `.get(k, 历史默认)`。
  - **demo 谓词 `+ (W−1) <= es − 1` 在本文件出现 4 次（不是初稿写的 3 处）**：`_encode_ready_windows` 的 demo `while`、其后 `keep_from` 收缩的 `if`、同一条件在列表推导里的**第二份拷贝**、`visible_motion_frames` 的 demo `while`。四处全改为 `+ (demo_min_real − 1) <= es − 1`，并把重复的两处抽成局部变量 `demo_unfinished` 消掉拷贝。exec 的两处谓词不动。
  - `_encode_window(f)`：对 `f < es`（等价于「这是 demo 起点」，因 demo 起点 `s ≤ es−17 < es`、exec 起点 `≥ es`）取 `[f, min(f+32, es−1)]` 并 `np.repeat` 第 `es−1` 帧补齐。边界已核：`es = 0` 时 demo 窗数为 0、分支永不进入；`es = 17` 时 `[0, 16]` 真 17 补 16；满窗时 `min(f+32, es−1) = f+32`、补 0 帧。
  - `_raw_frames` 清理语义**确认不变**：新规则下 demo 仍在首批一次编完 ⇒ `keep_from = es` ⇒ 删光 `k < es`，与旧规则同结果；补帧源帧 `es−1` 的存活性成立（`_encode_ready_windows` 在 `add_buffer` 写完 `_raw_frames` 之后才调，清理发生在编码之后）；`max_buf ≤ 32+16+1` 的上界不变。
  - ★ **新增 fail-loud 校验（第四轮改为「按对放行」）**：不能仿照 `window_direction/grid_origin` 那种「默认值 == 唯一合法值」的硬等值——两个新键不是这样，硬等值 `repeat_last` 会让被哈希冻结的历史快照（`awsprod40k-b128-motion` 的 `motion:` 节实测无两新键、只能取默认 `"none"`）在 V2③ 必 FAIL，并让所有旧 context motion run（含正本第九章的 `awsprod40k-b128-motion`、`scripts/motion-variance/mv_model_adapter.py` 的 `create_trained_policy(..., motion_stub=False)`）永久起不了在线评估。改成与 C 节 dataloader 同形的三态判定：两键同缺 → `(33, "none")` 放行；只缺一键 → raise；两键齐备 → 只允许 `(window_frames, "none")` 与 `(1 ≤ x ≤ window_frames, "repeat_last")` 两种组合，其余 raise。理由：训练侧的 `_req` 在 `FrameSampDataset.__init__`，而 **eval 根本不构造 dataset** —— 不加这道闸，YAML 写错口径会变成「训练 raise、在线照跑」，训推口径不一致而无人发现。
- **`src/mme_vla_suite/policies/policy.py::__init__`**：`_motion_cfg` 的 8 键字典推导增两键，且这两键用 `mcfg.get(k, 历史默认)`（旧 YAML 与被哈希冻结的历史 run 快照都缺它们，**快照不能补键**）。
- ★ **同型写死的 `motion_cfg` 生产者共三处（另有一处吃整节），必须同改**：`policy.py`、`scripts/training/g0/compare_online_motion.py`（同样的 8 键推导——所以初稿写的「脚本本身逻辑不变」是错的）、`scripts/training/tests/motion_gates_online.py::MOTION_CFG`；`scripts/training/tests/online_mem_regress.py` 直接吃旧 YAML 的整节，受上条「按对放行」校验约束、走 `(33, "none")` 历史契约分支。
- **两处 stub 帧号校验放宽（「sidecar 不改」的唯一例外）**：`motion_gates_online.py::_stub_enc_local` 与 `scripts/dataset/wan/motion_sidecar.py` 都断言 `ids == list(range(start, start+33))`；`stub_decode` 把全域帧号写进像素、逐帧解回，补帧窗尾部会解出重复帧号 ⇒ 前者 assert 失败、后者 `return 4` 退出，还会与 P1 的「错帧→rc=4」用例混同。放宽为「连续真实前缀 + 合法重复尾」：存在 `r ∈ [demo_min_real, 33]` 使 `ids[:r] == range(start, start+r)` 且 `ids[r:]` 全等于 `ids[r−1]`；**继续拒绝**内部错帧、`r < demo_min_real`、尾部重复的不是 `ids[r−1]`、以及 exec 段出现任何 `r < 33`。**第四轮订正：这一改动不触及 `protocol_sha256`**——它只哈希 `motion_protocol.py` 自身（`hashlib.sha256(pathlib.Path(__file__).read_bytes())`），两处被改文件都不是它；握手断言与既有 provenance 记录**一字不动**（G 节明令不改冻结文件；若确要让 stub 语义变更被握手捕获，就把放宽逻辑实现成 `motion_protocol.py` 里的共享函数 `stub_validate_ids(ids, start, min_real)`，此时该 sha 才真会变）。另一处订正：「exec 段出现任何 `r < 33` 继续拒绝」在 `motion_sidecar.py` **不可实现**（线协议无段身份、无 `demo_min_real`），拆成 sidecar 侧段无关形状校验（连续真实前缀 + 合法重复尾，`r ∈ [1, 33]`）+ 知道 `es` 的一侧（`motion_gates_online.py::_stub_enc_local` / `FrameSampMemory`）做段相关断言（demo `r ≥ demo_min_real`、exec `r == 33`）。 注意 `scripts/training/tests/eval_rhythm_gates.py` 直接复用 `G._stub_enc_local`，会跟着生效。
- **`scripts/training/tests/motion_gates_online.py`**：`MOTION_CFG` 从 `--yaml` 的 `motion` 节整体读入（不再硬写字面量，与 `compare_online_motion` 同一份键列表）；所有 `96` / `608` / `(96,768)` 断言改用 `MOTION_CFG["budget"]` 与 `HIST_BUDGET + budget` 表达（8×8 档还须 `TOKEN_PER_IMAGE=64`，现文件写死 `512, 16, 1`）；`--yaml` 分支现在只更新帧路四个量、必须一并更新 `MOTION_CFG`。**超预算用例失效需重造**：原用例取 `es = 1600`，新规则下 demo 窗数 99（`⌊1583/16⌋+1`）< 160，`_prepare_motion` 不再 raise ⇒ 改用**专门的小 budget 实例**（如 budget=8 + es=200），不要靠拉长 es 去撞 budget。`assert mem2.motion_encode_calls == ms.seg_num_grid(es) > 96` 同时踩「公式默认值」与「96 字面量」两颗雷，一并改。P2/P4 用例加「es=114 首批编 7 窗、第 7 窗补 15 帧」断言。
- ★ **`scripts/training/tests/eval_rhythm_gates.py`（初稿完全没提这个文件）**：`predict_k` 里写死 `demo = ((es-33)//16 + 1) if es >= 33 else 0`，新规则应为 `((es-17)//16 + 1) if es >= 17 else 0`；`BUDGET = G.MOTION_CFG["budget"]` 随 budget 改 160 一起变，于是文件里写死的 `es=288/289` 边界失效——**新边界由计划自己的式子直接算出：`k > 160 ⇔ demo > 80 ⇔ es ≥ 1297`，故 `288/289` 改为 `1296/1297`**，与第七节「充要条件是评估 `es ≤ 1296`」自洽。该文件用的 `tau = es + 1296` 本来就与修正后的 `τ_max` 一致。**注意它对边界两侧各跑一遍完整 1300 步 eval 控制流，`es=1297` 意味着要缓冲 1297 帧 256×256×3 原始帧（≈255 MB），须先估内存。** 它是 1300 步评估口径与 budget 的唯一守门，必须和 budget=160 的结论一起重算。**第四轮补三条，缺一即架空**：① 同一函数里的扫描区间 `lo, hi = 200, 320` 必须改成覆盖新边界（如 `(1280, 1312)`）——只改常量不改区间，`[200,320)` 内无解 ⇒ `predicted is None` 且 `first_raise is None`，`ES_BOUNDARY` 以未捕获 `StopIteration` 崩掉、上游 `if first_raise != predicted` 恒不触发；`fails` 里加 `predicted is None → FAIL`。② 该文件没有 `--yaml` / `--budget` 入口，`BUDGET = G.MOTION_CFG["budget"]` 是 `import motion_gates_online as G` 的**模块级字面量**（96 与 4×4 context 档的 512/16/1），`G.main()` 从不被调用，与上条「`MOTION_CFG` 改从 `--yaml` 整体读入」互不兼容——补 `--yaml`（默认新 motion YAML），在 `main()` 里先跑与 `motion_gates_online.main()` 同一份初始化把 `G.MOTION_CFG` / `G.HIST_BUDGET` / `G.TOKEN_PER_IMAGE` / `G.NUM_VIEWS` / `G.MAX_FRAMES` 设好再取 `BUDGET`，并把实际跑在哪个 YAML / budget 上写进判定行。③ `--lib` 默认仍指旧 400ep 库（旧口径下 `margin` 约 67、新库是 9），显式传新库并纳入 F 表。
- **`scripts/training/g0/compare_online_motion.py`**：读 YAML `motion` 节透传两键；`visible_motion_rows` 调用随 `entry.spec` 自动生效；`first_batch_windows` 的 `ms.seg_num_grid(es)` 必须显式传 `min_real=17`（否则静默少记 1 窗/集）；**必须传 `encoder_run_dir`**（见 A9 第 4 条）。另见 F 节 V-online 行的三档（`--assembly-hash` / `--stub` / 真 encoder 路由）、比较器加固与分片改造。
- `motion_protocol.py`（除协议 sha 随 stub 变更）、`motion_client.py` 的**数值逻辑**不改，但其默认 encoder 目录必须改（A9）。

### F. 验证与判定行

> **判定行标注约定（第四轮）**：第 4 列每条判定行标「实测」（既有工具原样输出，并给出产出者）或「新写」（本轮定义、须先落地自测）。初稿有 7 个判定行名在代码里根本不存在，第二轮已换用工具真实输出（`SAMPLE_RAW_EXACT` / `MASK_INVARIANCE` / `A10_ROWS` / `A9_INDEXSET` / `ONLINE_POS`）；第三轮那句「每一行都已与工具实际输出对表两轮」**只对标「实测」的行成立**，本轮新定义的判定行名清单在I.7 末。**第 18 条两块划分**：第一块 = V1 / V2 / V4 / V-online；第二块 = **只有 V7**；V6 单列单步数值对拍；V8 是 A/A 可复现基线、开启态无旧链路、第二块不适用并登记为盲区。每项的「证什么 / 不过怎么办」在 I.7，本表只写命令级细节。

| 编号 | 内容 | 工具与关键参数 | 判定行 | 顺序步 |
|---|---|---|---|---|
| V0 | lr 曲线逐步相同 | 新写数行 `uv run python -c`：`config.lr_schedule.create()` 在 `range(0, 80_000)` 上逐步比 60k 与 80k 两条配置（`peak_lr == decay_lr == 5e-5` ⇒ optax `alpha = 1`，余弦段恒等、`decay_steps` 之后 clamp 也相等；warmup 段两配置同 5,000 步） | 新写 `GUARD_LR=PASS steps=80000 mismatches=0` | 2b |
| `INIT_COMMON` | 6 条 modulation 叶 + 4 条 motion 叶与基线初值同源 | 同 `seed=42`、`JAX_PLATFORMS=cpu`，分别用关闭态与新 motion YAML 只做 `init_train_state(..., resume=False)`，按叶名比 sha256（I.2） | 新写 `INIT_COMMON=PASS common_mismatches=0 open_only=4` | 2b |
| `ROPE_LEN_EFFECT` | 替换I.4 五个无留档数字 | 固定 token 内容、只改 `mem_len`，直接构造 `MemoryAttention` 或复算 `_apply_rope` + masked softmax；json 落 `v1-store/reports/motion/rope_len_effect.json` | 新写 `ROPE_LEN_EFFECT=PASS tv_144_160=… relL2_144_160=… tv_512_672=… pad_content_diff=0` | 2b |
| V1 | 关闭态 dataset 交付逐位（改前 vs 改后） | `dump_fixture_samples.py` ×2 + `compare_fixture_dumps.py`；**数据集必须用 `4task-motion-400ep/framesamp-8x8`**（新库 `exec_start_idx` 全 ≥100，`fixture_per_step` 找不到 `es=0` 的 episode 必 raise）；环境 `DTYPE_DUMP_DIR`（两侧各一空目录）/`DTYPE_DUMP_GIT_HEAD`/`DTYPE_DUMP_MODE=both`/`DTYPE_DUMP_ARRAYS=1`/**不设** `DTYPE_DUMP_LIMIT`/`JAX_PLATFORMS=cpu`；白名单见下；**该比较器不比较两侧 `DUMP_MANIFEST.json` 的 `git_head`**，对拍前 `jq -r .git_head` 断言两侧等于 `$BASE` / `$CAND` | 实测（`dump_fixture_samples.py` / `compare_fixture_dumps.py`）：单侧 `DUMP_DONE samples=3200 batches=200 out=<dir>`；对拍 `SOURCE_IDENTITY=PASS episodes=400 samples=101066` / `FRAME_INDEX_EXACT=PASS … max_frames=8` / `SAMPLE_RAW_EXACT=PASS samples=3200 per_step=200 mismatches=0` / `BATCH_RAW_EXACT=PASS batches=200 mismatches=0`。失败是抛异常 + 退出码 1，无 FAIL 行 | 2a/2d |
| V2 | 旧表 + 旧 YAML + 历史快照三重回归 | ① **新写 `scripts/training/tests/legacy_regress.py`**：`MotionMeta.load` 两张旧表（40ep rows=772、400ep rows=6832，`schema=1`、无三新键）解析出 `LayoutSpec(33, 33, "none")`——前提 A1 的 schema 不升版与三新键按 layout 兜底；② **2a 与 2d 各跑一次** `dump_fixture_samples.py` + `compare_fixture_dumps.py`：**未经修改的** `perceptual-framesamp-context-motion.yaml`（已在 `_EXPECTED_HISTORY_CONFIGS`）+ 40ep 库（含 `exec_start_idx == 0` 的集），N 同 V1；③ `legacy_regress.py` 用历史快照 `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/5000` 走 `create_trained_policy(..., **motion_stub=True**)`（仍 `Popen` 一个 stub sidecar 并握手、不加载 encoder、不占 GPU；前置 `ls v1-store/venvs/wan/bin/python`；不加该参数会真去拉 sidecar、占 GPU、加载 40ep encoder；该目录**不得改动任何文件**；快照 `motion:` 节无两新键 ⇒ 走 `(33,"none")`，E 节在线守卫必须放行） | 新写 `LEGACY_LAYOUT=PASS tables=2 rows=772/6832 spec=(33,33,none)`；② 实测同 V1 四行 + 新写汇总 `LEGACY_YAML=PASS`；新写 `LEGACY_POLICY=PASS spec=(33,none) stub=1` | ① ③ 2d；② 2a/2d |
| V6 | 关闭态定点梯度（改前 vs 改后） | `single_step_grad_fixed.py` ×2 + `compare_fixed_grad.py`；**`DTYPE_BATCH_FIXTURE_DIR=v1-store/fixtures/8x8/grad/c8-b`**（8 帧档、12 个数组键 + 4 个 `none` motion 键，满足工具「必须恰好 12 数组键」的断言；`m8-*` 是 16 键会 raise）；`--batch-size 8 --fsdp-devices 2 --seed 42 --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml`；`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`；`DTYPE_GRAD_DIR` 预先不存在；`unset DTYPE_BASELINE_CHECKSUMS`；`DTYPE_SOURCE_COMMIT=$BASE` / `$CAND` + porcelain 为空。**该工具把关的是 `_CONFIGS` 与 `main()` 的 `raise ValueError("只接受 modulation 关闭态两档配置")`，modul YAML 已在其中、无需改**。**硬约束**：比较器对 `environment` 整体等值（含 `cuda_visible_devices`、`uv_lock_sha256`），两侧 `export CUDA_VISIBLE_DEVICES=4,5`，2b 禁 `uv add` | 实测：单侧 `FIXTURE=PASS kind=… keys=12 batch=8`（×3）/ `INIT_PARAMS=PASS leaves=<N>` / `GRAD_KIND=PASS kind=…`（×3）/ `GRAD_DONE kinds=3`；对拍 `INIT_EQ=PASS leaves=<N> mismatches=0` / `GRAD_EQ=PASS kinds=3 leaves=<N> mismatches=0` | 2a/2d |
| V7 | 关闭态 100 步守卫 | 驱动 `scripts/training/g0/run_2gpu_epoch_bench.sh`，一行式照 `docs/training-doc/aws-t3-closed-s100/launch.md`：`STEPS=100 SAVE_INTERVAL=25 EXTRA_DIGEST_STEPS=99 WORKERS=4 HISTORY_CONFIG=perceptual-framesamp-modul-8frame-8x8.yaml EXP_NAME=<run> RUN_TAG=<tag> BENCH_GPUS=4,5 DATASET_PATH=v1-store/datasets/4task-motion-400ep/framesamp-8x8 XLA_FLAGS='…'`（驱动默认 `DATASET_PATH=${GL_DATASET}=4task-gl` 本环境不存在；`BENCH_ROOT=v1-store/bench/2gpu-epoch-bench` 写死，产物落 `<BENCH_ROOT>/<RUN_TAG>`；驱动的 `case "${HISTORY_CONFIG}"` 白名单是第三处、1b 放行）。链路：`check_baseline_env.py dump`（两侧）→ `manifest`（2a，产 `BASELINE_MANIFEST.json`）→ `check --base … --steps 100 --batch-size 8`（2d）→ bench（两侧）→ `project_scalars.py`（两侧，产 `scalars_hex.tsv`）→ `compare_baseline.py`。摘要步 `{0,25,50,75,99}`、输入摘要步 `{0,1,2,25,50,75,99}`（正本 100 步口径）。基线 60k run 的 `metrics.jsonl` 只有每 100 步一行且无 digests / env.json，**不能复用**。约 30–35 min × 2，tmux `mv2-bench-base` / `mv2-bench-cand`，**按第 17 条留档** | 实测（`check_baseline_env.py` / `compare_baseline.py`）：`BASELINE_ENV=PASS` / `SCALARS steps=100 keys=5 hex_mismatch_steps=0` / `INDEX_SEQ=PASS n=<共同前缀长度，≥800 含 prefetch 余量，历史 872>` / `STATE_DIGEST rows=5 mismatch=0` / `BATCH_DIGEST rows=7 mismatch=0` / `BATCH_DIGEST_CANONICAL rows=7 mismatch=0` / `CANON_CHECK=PASS steps=7` / `DET_CHECK=…`；新写（`finish_check.py`）`INDEX_TRAIN=PASS n=800 recorded_n=<n>` / `GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5` | 2a/2d |
| V3 | 新表：输入重锚 / D3 全量 / D2 抽样 / a6set a7 a9set a10 a11 | B 节 | 见 B 节判定行清单 | 3（重锚）/ 4（其余） |
| V4 | 开启态 dataloader 交付逐位 | **复用并参数化 `scripts/training/tests/hand_calc_8frame.py`**（交付 vs 表字节 + `mem_order` 手算；三处必改见 C 节）。样本集**写死**：冷启动 `t==es` 全 1,600、首 exec 窗 `t==es+32` 全 1,600、`k≥140` 的 32 个全取、16 个 `(es−17) mod 16` 残差档各 ≥ 2。`a9set --cold-all` 只在步 4 跑一次，此处引用 | 实测 `HAND_CALC_8FRAME=PASS samples=<n> mismatches=0`；新写 `V4_COVER=PASS cold=1600 first_exec=1600 kmax=141 kge140=32 pad_r=16/16`（四数等值断言）；引用 `A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0` | 6a |
| V5 | 模型侧 mask 正确性（modulation） | `motion_gates_model.py` 适配见下；模型档已拍板用乙 `gemma_150m` 替身（见下）；确定性四项 + 流式逐叶（见下）；tmux `mv2-m4`，GPU 4，**按第 17 条留档** | 新写 / 改口径：`MASK_INVARIANCE=PASS` / `GRAD_LEAK=PASS` / `ORDER_EFFECT=PASS` / `ROW_PERM_INVARIANCE=PASS max_abs_diff=…` / `PAD_CONTENT_INVARIANCE=PASS det_probes=3 nondeterministic_leaves=[] excluded=0 covered=<n>/<m>` / `LEN_EQUIV_NA`（显式不做并写明 RoPE 原因） | 6a |
| V8 | 开启态 A/A 100 步可复现 + motion 叶更新 | 同 V7 驱动 ×2（同一侧跑两次），YAML `perceptual-framesamp-modul-8frame-8x8-motion.yaml`，`DATASET_PATH` 指新库 `framesamp-8x8`，`MMEVLA_MOTION_STORE` unset，GPU 4,5，tmux `mv2-aa`，**按第 17 条留档** | 实测：`SCALARS steps=100 keys=5 hex_mismatch_steps=0` / `INDEX_SEQ=PASS n=<…>` / `STATE_DIGEST rows=5 mismatch=0` / `BATCH_DIGEST rows=7 mismatch=0` / `BATCH_DIGEST_CANONICAL rows=7 mismatch=0` / `CANON_CHECK=PASS steps=7`；新写 `AA_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5`（汇总器另断言四键 `per_key` 逐步相等且均非 null）/ `MOTION_PARAMS_UPDATED=PASS n=4 first_step=0 last_step=99`（从 `param_checksums.jsonl` 比 4 叶 sha 随步变化） | 6b |
| V-online | 在线 vs 表（**方案 B 分层解耦**，三档） | `compare_online_motion.py`：① `--assembly-hash $LIB/wan-latents`（零 GPU，8 片 `g % 8`，tmux `mv2-vonl-a`）；② `--stub`（零 GPU，全 1,600 集，tmux `mv2-vonl-b`）；③ 真 encoder + `motion_enc_fn` 路由（选中窗 = 全部补帧窗 + 必含集；8 卡 `--gpu 0..7` 各起一个 sidecar，`--shard-idx/--num-shards`，`aggregate` 子命令，tmux `mv2-vonl-c`）；**必须显式 `--out v1-store/reports/motion/p5_online_v2_1600ep.shard<i>of8.json`**；主进程 `export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`；**必须传 `encoder_run_dir`**（A9 第 4 条）；比较器加固见下 | 新写 / 改口径：① `ONLINE_INPUT_SHA=PASS windows=71316 padded=1600 mismatches=0`；② `ONLINE_START_SET=PASS` / `ONLINE_POS=PASS` / `ONLINE_ORDER=PASS`；③ 各片 `ONLINE_ENC_BITEXACT=PASS compared_real=<n> mismatches=0` / `PROVENANCE=PASS` / `YAML_STORE_PATH=PASS`，aggregate 后 `ONLINE_STRATA=PASS padded=1600/1600 kmax_ep=367 es_ge1000=<n≥3> pad_r=16/16 per_task_min=<n≥1>` / `SHARD_SET=PASS shards=8 eps=1600 disjoint=1`；汇总 `P5_ONLINE=PASS episodes=1600 windows=71316 assembly_checked=71316 compared_real=<n> padded_covered=1600 stub=False` | 6c（①②）/ 6d（③） |
| V9 | **8 卡** b128 20 步 smoke（开启态） | 基线 `records/smoke-runner.sh` 同法，换 YAML / 配置名 `mme_vla_suite_b128_80k`（`fsdp_devices=8` / `num_workers=16` 由该配置自带，不在 CLI 覆盖）；**必须同改 smoke runner 第 23 行 `CUDA_VISIBLE_DEVICES` 与第 33 行占卡 `test` 为 `0,1,2,3,4,5,6,7`**，否则 `fsdp_devices=8` 与四卡可见冲突；`PARAM_TREE_EXACT` / `MEM_PARAMS` / `BUDGET_CONSISTENT` 等判定行不受卡数影响；tmux `mv2-smoke`；**基线同档实测 5 分 18 秒是 4 卡数字、8 卡耗时待测，一律按第 17 条以完整 run 留档**（不赌它 < 5 min），run_name `smoke-m8x8-modul-motion-<UTC>`，验完删 run 产物、留档保留。`check_config_provenance.py` **逐个显式传** `--ckpt <smoke run>/19 --lib v1-store/datasets/4task-v2-1600ep-604f16da --train-config mme_vla_suite_b128_80k --store-subdir framesamp-8x8 --neg-lib v1-store/datasets/4task-motion-400ep --norm-stats <新库 norm_stats.json>`（默认值全指旧库旧 run）；**新验收器必须新写**，不得沿用 `smoke-m8x8-modul-20260915T054007Z/records/finish_check.py`（硬编码 `motion_enabled is False`、`n_model=n_ckpt=61`、`PREFLIGHT=PASS n=25` / `== 25`、`mme_vla_suite_b128_60k`） | 新写 `SMOKE20=PASS steps=20 finite=1 exit_code=0`；实测 `PREFLIGHT=PASS n=30`；实测 `PARAM_TREE_EXACT=PASS … n_model=65 n_ckpt=65 missing=0 extra=0 shape_mismatch=0`；新写 `MEM_PARAMS=PASS n=10`（6 modulation 叶 + 4 motion 叶）；实测 `check_config_provenance.py` 全部五条 `NORM_STATS_SAME=PASS` / `LIB_PROVENANCE_MATCH … all_equal=1` / `MOTION_STORE_PATH … guard=raise infer_reads_store=0 same_manifest=1` / `CKPT_PARAM_TREE=PASS … leaves=65 ckpt_has_motion=1` / 观察 `CKPT_DTYPE_PROFILE` / 汇总 `TIC_L0=PASS`；新写 `BUDGET_CONSISTENT=PASS snapshot=160 repo_yaml=160 expected=160`（跨源，D 节）；新写 `NORM_STATS=PASS sha256=856c75ea…` | 6e |
| V10 | 起跑 preflight | `preflight_train_launch.py`（runner 内，与 train 共用同一 `TRAIN_ARGS`）。**`n=25` 属实但前提是 runner 必须传 `--run-root`**。新增 **5** 项，期望值由 runner 显式传参（preflight 无 YAML 解析器）：`--motion-store` → `CHECK_MOTION_STORE_PATH`、`--motion-store-meta-sha256` → `CHECK_MOTION_STORE_META_SHA256`、`--motion-layout` → `CHECK_MOTION_LAYOUT`（== `motion-768-grid16-demopad17-v1`）、`--motion-rows` → `CHECK_MOTION_ROWS`（== 71316，来自建库留档、不从被测 store 现算）、`CHECK_MOTION_ENV_UNSET`（`MMEVLA_MOTION_STORE` 为 unset，无参）；建议加 `CHECK_MOTION_SOURCE_RUN` | 实测口径 `PREFLIGHT=PASS n=30`（25 + 5 无条件计入） | 8 |
| `EVAL_ES_BOUND` | 评估集 es 上界实测 | 遍历 4 任务 × 全部 test 集，只做 `EnvRunner.make_env(i)` + `get_init_obs()`（200 次 `env.reset()`，无策略、无 GPU） | 新写 `EVAL_ES_BOUND=PASS tasks=4 episodes=<n> es_max=<m> budget=160 headroom=<160 − k_eval_max>` | 7 |
| `GPU_IDLE` | 起跑前独立占卡闸 | `nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits` 求和为 0 且 `tmux ls` 中本轮会话只剩 `mv2-prod`；不依赖 runner 内的 `test`（`body()` 无 `set -e`） | 新写 `GPU_IDLE=PASS gpus=0,1,2,3,4,5,6,7 used_mib=0 sessions=<清单>` | 8 |

**工具白名单（1b commit，三处 + 一处形制闸）**：`dump_fixture_samples.py` 与 `bench_train_steps.py` 各有一份 `_EXPECTED_HISTORY_CONFIGS`，当前只含 4 个 context YAML，对 modul YAML 会 raise；`run_2gpu_epoch_bench.sh` 另有独立 `case "${HISTORY_CONFIG}" in perceptual-framesamp-context.yaml|perceptual-framesamp-context-motion.yaml)` 白名单（第三处，第三轮漏了）。**三处各加两个**：`perceptual-framesamp-modul-8frame-8x8.yaml`（V1/V7 关闭态用）与本轮新增的 `perceptual-framesamp-modul-8frame-8x8-motion.yaml`（**V8 开启态用**）。不可理解成「`modul.yaml` + `modul-8frame-8x8.yaml`」，否则 V8 起跑即 raise。`ref_npy_dataset.py` 的形制闸放开 modulation 也在 1b（C 节）。这些改动必须**先于 2a commit**，否则 V1/V7 起手 raise 与 V6 的 porcelain 清洁闸互斥。

**V5 模型档：为何 CPU dummy 不行、为何 VLM 主干可以换替身**：`MemoryAttention` 里 `num_heads, num_kv_heads, head_dim, width = (4, 1, 256, 1024)` 并 `assert mem_width == x_width == width`，其中 **`x_width` 来自最后一条 expert 流（action expert）的宽度**，不是 `memory_token_dim`；同时 `Module.setup` 断言 `all(config.depth == configs[0].depth)`。现有变体里 `dummy` = width 64 / depth 4，CPU dummy 撞宽度断言。第三轮写「唯一 width=1024 的是 `gemma_300m`」是计数错：width=1024 / depth=18 的有三个——`gemma_300m`（mlp 4096）、`gemma_300m_lora`、`gemma_150m`（mlp 2048，注释「memory expert to be integrated in pi05」），**没有浅层 width-1024 变体**。由此两档：
- **甲（09-17 第一次拍板原文，已被乙替换、不采用）**：真实配置，`paligemma_variant=gemma_2b`（width 2048 / mlp 16384 + 257152×2048 词表，约 2.6 B、fp32 约 10 GB）+ `action_expert_variant=gemma_300m`。modulation 记忆路一个字节都不经过 VLM 主干。
- **乙（第四轮推荐，用户 09-17 拍板采用）**：`paligemma_variant="gemma_150m"` + `action_expert_variant="gemma_300m"` + 真 `memory_token_dim=1024` / 真 YAML / 真库 / 真 budget=160。结构可行：openpi `Attention` 只要求各 expert 的 `head_dim / num_heads / num_kv_heads` 相等（同为 256 / 8 / 1），`Module.setup` 只要求同 depth（同为 18），`history_pi0.py` 的 `_siglip.Module(num_classes=paligemma_config.width, …)` 自动跟随。留档写明「VLM 主干用 `gemma_150m` 替身；`MemoryAttention` 的 4 头 / 256 头维 / 1024 宽 / 18 层全部为真」。

**V5 工具适配（`motion_gates_model.py`，初稿只写「新增 `--integration` 开关」，实际至少九层）**
1. `_make_models` 的 dummy 宽度问题（见上）：改为接受真实配置路径与变体名，在 GPU 上建真模型。
2. `cmd_m3`/`cmd_m4` 写死 32 帧 / 16 token 每帧 / 96 motion / 608 总长的**字面量**（连现有 8×8 context 口径的 `FRAME_BUDGET=8`、`TOKENS_PER_FRAME=64` 都已对不上），全部换成常量表达。
3. `_fixture_batch` 固定找 `(6,0)`/`(32,11)`/`(32,max)`，而 `(6,0)` 在新库**物理不可达**（新库 `exec_start_idx` 最小 100 ⇒ `k = min(t+1,32)` 恒为 32）；`(32,"max")` 在新库 m 最大 141 > 96 也会 raise。**spec 分两类**（第三轮写「四个都按数据集真实交付挑」，字面实现必 `SystemExit`）：库内可得——`m` 中位与 `m=max=141`（写死 `BinFill ep367, es=1152`，或用 `manifest_expected_k` 向量化定位、避免 605,611 样本 Python 全扫）；必须合成——`m=0`（照 `cmd_m4` 的 `obs_empty`）与 `m=budget=160`（`ds[i]` 返回后改字典）。新库真实 `m ∈ [6, 141]`。
4. `_t3_main` 把 YAML stem 写死成 `perceptual-framesamp-context…`，`--lib` 默认旧 40ep 库，`--store-subdir` 默认 `framesamp`：新增 `--integration {context,modulation}` 决定 stem，V5 显式传新库与 `framesamp-8x8`。
5. **`MOTION_BUDGET` 不在 `_t3_main` 的 `global` 列表里**，`--budget` 无从注入 ⇒ 必须把它纳入 global 并新增 `--motion-budget`。
6. `_synthetic_entry` 用旧公式造合成 index、且 `parse_index` 的 payload 写 `"layout": ms.LAYOUT` 但**不带三新键** ⇒ 换 `LAYOUT` 常量后该用例会撞 A1 的「新布局三键缺一即 raise」。改为显式写布局名与三键、`_synthetic_entry` 增 `min_real` 形参；其 oracle `oracle_visible` 的 demo 判据也要按 demo `min_real` 参数化。
7. **确定性纪律（第四轮 P0）**：`cmd_m4::param_grads` 现在 `nnx.split(m_open)` 后直接 `jax.grad`、无过滤、无探针。照抄 `cmd_t3mechanism` 已验证的做法：`--det-probes`（默认 3）同 obs 探针、`nondeterministic_leaves` 单列并排除（硬闸：loss R 次逐位相同；不确定叶含 `mem_encoder` / motion 叶即 FAIL）、比较集合取 `params.filter(config.trainable_filter)` 即 `get_freeze_filter()` 的补集（非 lora 档 `PathRegex(".*img.*")`）、挂与 V6 同一组 `XLA_FLAGS`。
8. **显存纪律**：逐叶比较改流式（每叶算 sha / max_abs 后立即释放，禁止同时持有两棵完整梯度树）；ema / opt_state 在初态校验后即释放（同文件既有注释：否则第二次 `value_and_grad` OOM，2026-09-03 实测 44 GB）。
9. `ROW_PERM_INVARIANCE` 现只 `fails.append`、无 `=PASS/FAIL` 打印，加标准格式；顺带清 (d) 段写死的 `np.arange(96)` / `pad_times(frames, 96)` 与循环内重复解析 manifest / index。

**V5 判据的两处修正**
- `ZERO_MOTION_EQUIV`（开启态 motion 全 mask ⇒ 等价于关闭态）在 context 下成立（记忆进 prefix、位置走 `cumsum(input_mask)−1`，全 mask 的 motion 位不占号）；**modulation 下比的是 `mem_len=672` 与 `mem_len=512` 两个模型，q 位置差 160 位，`tol=1e-4·|loss|` 不可能过，且 FAIL 不代表实现有错**（差异量级由 `ROPE_LEN_EFFECT` 重取）。拆成两条不同承诺：`PAD_CONTENT_INVARIANCE`（**同一 budget** 下 padding 内容不影响结果）与 `LEN_EQUIV_NA`（不同 memory 长度的两模型不等价，显式不做该断言并写明 RoPE 原因）。
- 现有垃圾对拍**只比 loss 与 actions 的字节**，其后的梯度检查跑在 clean 输入上、只看零/非零性质，**从未比较垃圾前后的参数梯度**。`PAD_CONTENT_INVARIANCE` 的判据改为：固定模型参数、固定 rng key、固定 actions，分别算 clean 与 garbage 的 `jax.grad`，**逐叶比全部可训练参数**（不只 4 条 motion 叶，且带第 7 条的探针与过滤）。明确「垃圾 = 有限数值（1e3 量级正态）」，NaN/Inf 不在同一承诺内。保留正向检查：真实 motion 内容改变必须改变 loss；4 条 motion 叶在有 motion 时梯度非零（fp32 下 `max_abs > 0`，不设阈值）。**「20 步 smoke 内这 4 叶的参数值确有更新」从 V5 移到 V8 的 `MOTION_PARAMS_UPDATED`**（V5 只走 `jax.grad`、不落优化器；`bench_train_steps.py` 本就写 `param_checksums.jsonl`）。
- `ORDER_EFFECT` 口径说明：modulation 下记忆 token 之间没有 attention，该判据只证「`mem_order` 经 `k_position` 改变 cross-attn logits」，不再是 context 下「改主干 RoPE 位次」那个承诺。

**V-online 比较器的加固**（第三轮五条保留，第四轮加三条）
1. `verdict = start_ok and pos_ok and order_ok and (args.stub or (not tok_mism and all_rows))` —— `args.stub` 短路让**已记录的 `stub_token` 失配**永远不进判定，同时那一行还被打印成 `ONLINE_ENC_BITEXACT=SKIP(stub) … mismatches=N`，即 stub 档 token 全错仍 `P5_ONLINE=PASS`。改为**任何模式下已检测到的失配都必须清零**，`SKIP` 只允许出现在「真离线 token 对拍」这一项的打印上、不得进入 verdict。
2. 四处 `np.array_equal` 比浮点：`+0.0 == -0.0` 为真（放过符号零）、float32 vs float64 同值也为真（放过 dtype 漂移），而判定行却叫 `BITEXACT`。换成先核 dtype/shape 再按 `uint8` 视图比原始字节（`motion_gates_model.py` 里已有现成的 `_bytes_equal` 可抄）。
3. 改完必须**注入两个反向用例证明比较器会 FAIL**：把某个 stub token 改成错常数、把某行 `+0.0` 换成 `-0.0`，各跑一次确认非零退出。
4. ★ 另补一项**秒级 CPU 前置门**（现有 P 系列全部用 `MME_VLA_Policy.__new__` 手工赋 `_motion_cfg`，真 `__init__` 的键透传与真 `reset()`（**重建对象**，不是 `mem.clear()`）从未被跑到）：用真 YAML → `get_history_config` → 真 `MME_VLA_Policy.__init__`（注入 stub `motion_enc_fn`）构造 policy，断言 `_motion_cfg` 的键集合与值逐项正确（含 `budget == 160`——这是在线侧 budget 一致性的落点）；连跑两条 es 不同的 episode，中间调**真 `pol.reset()`**，断言新 `mem_buffer` 是新对象、补帧窗数正确。别让 20+ min 的 V-online 去当第一道发现者。
5. ★ V-online 会被 `_make_dataset` 设置的 `MMEVLA_MOTION_STORE` 隐式覆盖 YAML 路径（生产 runner 保持 unset），所以**对拍前先 `unset MMEVLA_MOTION_STORE` 用真 YAML 解析一次路径并断言 `realpath == $LIB/motion`**（判定行 `YAML_STORE_PATH=PASS`）。注意这是**验证覆盖缺口，不是数据正确性风险**：生产 unset 时路径写错会被 `check_same_source` fail-loud raise，不会静默用错库。
6. ★ **shard 输出结构化字段（第四轮）**：现行 `out.write_text(json.dumps({"lines": …, "per_episode": …, "mismatches": mismatches[:200], …}))`——三个布尔只以文本行存在于 `lines`，`len(rows_seen)` 只作为 `covered=<n>` 出现，`mismatches` 截前 200 条。新增 `rows_seen`（排序列表或其 sha256 + 计数）/ `start_ok` / `pos_ok` / `order_ok` / `n_compared` / `n_steps` / `padded_covered` / `mismatch_total`（不截断）；`aggregate` 只消费这些字段、禁止解析 `lines`，任一字段缺失即 FAIL、不按 True 兜底。
7. ★ **期望侧独立与分片完整性（第四轮）**：现行 `all_rows = len(rows_seen) == total_rows` 里 `total_rows = mmeta.num_rows`（打包器写的表头）与在线遍历不同源，这条独立性不能丢——窗数期望改用 `motion_index` 的 `SegmentInfo.num_grid` 求和（不从在线遍历现算）；`padded_covered` 两侧各算一遍（在线按 es 推、离线取 `pad_frames>0` 的行）断言集合相等；aggregate 断言恰 8 个分片、shard 索引集合 == `range(8)`、各片 episode 两两不交且并集 == 1,600（`SHARD_SET`）。分片切分写死 `g % num_shards`（连续切块实测 BinFill 前后段 12,750 / 18,558 窗，挂钟拉到 2.08 倍）。
8. ★ **`motion_enc_fn` 路由（第四轮，③ 档）**：`framesamp_memory.py` 的 `motion_enc_fn` 是 `__init__` 形参，`compare_online_motion.py` 现行直接 `motion_enc_fn=client`；包一层路由（选中 → 真 sidecar，未选中 → 直接返回离线表行），`_prepare_motion` 只查 `f in self._history_feats_motion` 不关心来源，装配与次序检查不受影响；报告 `compared_real` / `assembly_checked` 分列。

### G. commit 约束

- 第一部分第九节的表是唯一执行顺序（步级细节见 I.8）；每个 commit 后立即 `git push`；只 `git add` 本轮明确文件，禁 `git add .` / `-A` / `-a`。
- **「改→验→commit」的顺序硬闸**：`single_step_grad_fixed.py` 要求 `git rev-parse HEAD` 逐字等于 `DTYPE_SOURCE_COMMIT` 且 `git status --porcelain` 为空（唯一例外是一条为历史树写死的单行 `??`，**不得拿来放行本轮改动**）。故拆成 1b（只改验证工具白名单与形制闸，`fix:` commit）→ 2a（在 1b 之后的 clean BASE 取基线证，`BASE=$(git rev-parse HEAD)` **不写死 sha**）→ 2b（改码 + 验证工具 + CPU 单测，dirty，不跑带校验的工具，禁 `uv add`）→ 2c（`commitV10.0` 生产改动 + `commitV10.1` 验证工具，push，porcelain 为空）→ 2d（取候选证 + 对拍；FAIL 先分流，数值项失配才 `git revert --no-commit` + 中文 `revert:` subject、只 revert V10.0）。第三轮把 2a 写成「clean HEAD 跑 V1+V6+V7」物理上不成立：白名单不改 V1/V7 起手 raise，改了 V6 的 porcelain 清洁闸又不过。**绝不允许临时放宽 clean-source 校验。**
- ★ **三个比较器都不核对两侧源码身份**（第四轮订正，第三轮只点了 V1、还把 V6 说成有覆盖）：`compare_fixture_dumps.py` 不读 `DUMP_MANIFEST.json` 的 `git_head`；`compare_fixed_grad.py` 只比 `schema / seed / fsdp_devices / xla_flags / environment` 与 `history_config_sha256`、不比 `source["head"]`（`single_step_grad_fixed.py::_source_record` 在 `DTYPE_SOURCE_COMMIT=$BASE` 写法下恒真，有效的是紧随的 porcelain 清洁闸）；`compare_baseline.py` 同。两侧误跑在同一 commit 时三项全绿。处置：2a / 2d 各记 `git rev-parse HEAD` 与 porcelain 入留档；对拍前统一 `jq` 断言两侧 `.git_head` / `.source.head` / `.start_head` 等于预期 SHA、`start_status` 为空，并显式断言 `$BASE != $CAND`。
- 步骤 4 的 Wan → encode 之间零 commit；步骤 2c 结束后到步骤 4 起跑前 `git status --porcelain` 必须为空；步骤 7 的 `TRAIN_HEAD` 记在 docs commit 之后、runner 生成之前，写进 launch.md 正文；步骤 8b 起跑后归档 `records/launch.actual.json` / `records/preflight.log` / `records/prod-runner.sh` 副本另起 `docs:` commit。
- 正式 run 的 runner 与 preflight 参数照抄 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/records/prod-runner.sh` **加上 `records/launch.actual.json::launch_command` 的 `env` 前缀**（`CUDA_CACHE_PATH` / `WANDB_DATA_DIR` / `XDG_DATA_HOME` 三个不在 runner 里），差异只有第一部分第三节列的那 **11** 处（第 11 处 = preflight 新增的 motion 期望值参数）；runner 落 `v1-store/logs/mv2-prod-runner.sh`（不进 git）；`MMEVLA_MOTION_STORE` 保持 unset。
- **卡号改写清单（照抄时必改的 6 处）**——基线留档里的这几个文件是**历史记录、一律不改**（第 12 条），改的是照抄生成的新 runner：

  | 源文件 | 位置 | 原值 | 新值 |
  |---|---|---|---|
  | `records/prod-runner.sh` | 第 20 行 `export CUDA_VISIBLE_DEVICES` | `4,5,6,7` | `0,1,2,3,4,5,6,7` |
  | `records/prod-runner.sh` | 第 28 行 占卡 `test`（`body()` 无 `set -e`，本就是无效闸，另设独立 `GPU_IDLE`） | `--id=4,5,6,7` | `--id=0,1,2,3,4,5,6,7` |
  | `records/prod-runner.sh` | 第 57 行 15 秒 util 采样 | `--id=4,5,6,7` | `--id=0,1,2,3,4,5,6,7` |
  | `records/launch.actual.json` | `dense_sampling`（`timeout 1800 nvidia-smi … -lms 500`） | `--id=4,5,6,7` | `--id=0,1,2,3,4,5,6,7` |
  | `records/smoke-runner.sh` | 第 23 行 `export CUDA_VISIBLE_DEVICES` | `4,5,6,7` | `0,1,2,3,4,5,6,7` |
  | `records/smoke-runner.sh` | 第 33 行 占卡 `test` | `--id=4,5,6,7` | `--id=0,1,2,3,4,5,6,7` |

  漏改采样两行的后果是**新增的四张卡完全没有 util 记录**，第 16 条的稳态判读直接失效。新 run 的 `launch.actual.json` 里 `gpu_ids` 记 `0-7`。`fsdp_devices` / `num_workers` **不在 runner 里覆盖**，由新具名配置承载（C 节）。
- **不得修改**：`perceptual-framesamp-context-motion.yaml`、`perceptual-framesamp-context-8frame-8x8-motion.yaml`、任何 run 内 `history_config.resolved.yaml`、`train.py` 的保存条件、`checkpoints.py` 的保留策略、`utils.get_config` 的变体表。两键的向后兼容靠 reader 侧默认值实现，**不靠回填旧配置**。

### H. 重构前后链路图（第 18 条强制）

**关闭态（基线，本轮必须逐位不变）** —— 初稿把 `memory_order` → `mem_order (512,)` 当成基线的一步是错的；第三轮又把 `ftimes (8,) int64` 留在图里，而它与 `memory_order` **同在 `if self._motion_enabled:` 块内**（`ftimes = pad_times(frames_arr, self._max_frames)`），关闭态同样不执行。第四轮订正后：

```
episode_manifest.json ──► FrameSampDataset.__getitem__(g, step)
  framesamp-8x8 store ──► static_image_emb (512, 2048) bf16   [(8,64,2048) 只是 _pad 的中间量]
                          static_mask      (512,)      bool   [np.repeat(mask, 64)]
                          static_pos_emb   (512, 768)  f32
                          static_state_emb (512, 8)    f64
                          frames_arr       (n,)        int64  [even_sampling_indices 原始帧号，未右填充；仅内部中间量]
     ★ 关闭态从 static_mask 之后直接跳到模块级 _NONE_KEYS：
       mem_order / motion_emb / motion_pos / motion_mask 四键恒为 None，
       pad_times() 与 memory_order() 都只在 if self._motion_enabled: 块内调用，中间不产生任何 motion 侧中间量
  ──► RepackTransform ──► HistAugObservation ──► PerceptualMemory.__call__  [帧路 encode 得 (b,512,1024)；motion_enabled=False 早返回]
  ──► HistoryPi0.embed_memory  [关闭态编译期早返回：不 concat、不 take_along_axis，返回 (tokens, static_mask, [False]*512, [False]*512)]
  ──► mem_seq (b, 512, 1024) ──► MemoryAttention  k_pos=arange(512), q_pos=arange(512, 512+x_len)
```

**开启态（本轮新增，只多 motion 一路）** —— 第三轮把重排挂在不存在的 `PerceptualMemory.embed_memory` 上，实际是两层：`PerceptualMemory.__call__` 出并列序，`HistoryPi0.embed_memory` 做 concat mask 与两次 `take_along_axis`。

```
motion 表 motion_token.f32.bin (71316, 768) f32  ← pack/verify 产出，整表 219 MB（208.9 MiB）
  └► visible_motion_rows(entry, step)   [entry.spec = (17, 33, repeat_last)]
       demo 行：s ≤ es − 17（含补帧窗）      exec 行：u + 32 ≤ t − es
  └► 取行 → motion_emb  (k, 768) f32 ──右填充──► (160, 768) f32  = 491,520 B  [填充位内容不影响结果，V5 判据]
            motion_pos  (k, 256) f32 ──右填充──► (160, 256) f32  = 163,840 B
            motion_mask (k,)     bool──右填充──► (160,)     bool =     160 B
            ftimes (8,) int64 = pad_times(frames_arr, 8)；mtimes (160,) int64 = _pad_motion 右填充后长度是 budget
  └► memory_order(ftimes, 64, mtimes) ──► mem_order (672,) int32 = 2,688 B
       ★ 交错：motion token 按时刻插进帧 token 之间 ⇒ 改变的是每个记忆 token 的绝对 k_position（arange(mem_len)）
         modulation 下记忆 token 只作 key/value、彼此之间没有 attention，「帧 token 彼此的相对距离」不进入任何计算
                                      合计每样本新增 658,208 B ≈ 658 KB；b128 ≈ 84 MB/batch
  ──► RepackTransform ──► HistAugObservation
  ──► PerceptualMemory.__call__
        帧路 encode                     ┐
        motion_pos_proj      (256→768)  ├ 两层 motion 投影（新增 4 叶 1,771,264 参数）
        motion_encoder_static(1536→1024)┘ → 与帧路在长度轴 concat 得并列序 (b, 512+160, 1024)
  ──► HistoryPi0.embed_memory
        input_mask = concat([static_mask, motion_mask], axis=1) → (b, 672) bool
        tokens     = take_along_axis(tokens, mem_order[:, :, None], axis=1)
        input_mask = take_along_axis(input_mask, mem_order, axis=1)     [时间序；True 占前 16k+m 位]
        返回 (mem_seq, mem_mask, ar_mask, na_mask)；compute_loss 的 modulation 分支只取前两个
  ──► mem_seq (b, 672, 1024) ──► MemoryAttention  k_pos=arange(672), q_pos=arange(672, 672+x_len)
       ★ mem_len 512→672 ⇒ 全部 query 的 RoPE 位置整体 +160 ⇒ query 到每个真实 key 的相对距离全部 +160
       ★ 同一 mem_seq 广播给 18 层（nn.scan in_axes=nn.broadcast），只出 K/V，层间不更新
```

**在线（推理侧，第四轮补——本轮改的正是在线窗口调度与补帧）**

```
env pre_traj ──► pack_buffer ──► MME_VLA_Policy.add_buffer(es 下传) ──► FrameSampMemory.add_buffer
  └► _raw_frames[k] = 帧 k (256,256,3) uint8
  └► _encode_ready_windows：
       demo：while next + (demo_min_real − 1) ≤ es − 1     [本轮 33→17；该谓词在文件里 4 处，E 节]
       exec：while next + 32 ≤ 末帧 − es                    [不变]
  └► _encode_window(f)：f < es 时截到 [f, min(f+32, es−1)] 并 np.repeat 第 es−1 帧补齐 33 ──► (33,256,256,3) uint8
  └► motion_enc_fn(window, f) ──► sidecar（33 帧一包，B=1 fp32 关 TF32）──► (768,) f32
       [V-online ①：这里对 33 帧算 sha256 与离线 metadata.input_frames_sha256 逐窗比，零 GPU]
       [V-online ③：motion_enc_fn 外包路由，未选中窗直接返回离线表行]
  └► _history_feats_motion[f] = token
MME_VLA_Policy.infer ──► _prepare_history ──► _prepare_motion（k > budget → raise；右填充到 160）
  ──► memory_order(ftimes, 64, mtimes) ──► 四键 ──► HistAugObservation ──► PerceptualMemory.__call__ ──► HistoryPi0.embed_memory
  ──► mem_seq (1, 672, 1024)
  ──► sample_actions（modulation 分支）：
       prefix（image + language，不含记忆区）一次前缀 pass 装进 kv_cache   [kv_cache 形状不随 motion 变]
       10 步去噪：每步传 mem_seq=[None, mem_seq]，每层重算 mem_rms_norm(mem_seq) 与 kv_einsum_mem（无 K/V 缓存）
  每 16 环境步一次 infer ⇒ 1300 步下每集 82 次
```

**每一跳有没有改数**：取行、右填充、`memory_order` 置换、`RepackTransform`、`take_along_axis`、在线切窗与补帧全是搬运与重排，**不改数值**；唯一改数的是两层 Linear 投影与 `MemoryAttention` 本身（含其 `mem_rms_norm`）。帧路在**数据交付层**逐字节不变，但在**模型内**因 `mem_order` 改变每个记忆 token 的绝对 `k_position`、且 `mem_len` 512→672 使 query 位置整体平移而不再等价——这正是本轮要测的东西，不是回归；与 context 版「交错改变主干 RoPE 位次」是两套不同的机制（I.1 的 4.2 c/d 行）。

### I. 第一部分精简时移入的技术细节（第四轮）

> 第一部分只保留「做什么 / 为什么 / 哪些不变」，本节按原节顺序收纳被移出的差异矩阵、判据细节、推导备注与步级叙述，供 agent 追踪；与 A–H 节互为补充，冲突以 A–H 与 F 表的命令级描述为准。

#### I.1 context vs modulation 差异矩阵（含六层等价的代码锚点）


1. **关闭态逐位**：`percep_mem.py::PerceptualMemory.__init__` 的 `if self.motion_enabled:` 只在开启时创建两层投影，`__call__` 关闭态早返回；`history_pi0.py::HistoryPi0.embed_memory` 关闭态是「编译期 Python 分支 + 早返回，四处一个元素都不追加、不重排」；`framesamp_dataset.py` 的 `_NONE_KEYS` 四键恒 `None`。关闭态模型与基线完全相同，这是I.7 A 组 V1/V6/V7 的前提。
2. **数据交付层**：`shared/sampling.py::memory_order` / `pad_times`、`motion_store.py::visible_motion_rows` 是 context / modulation / 训练 / 在线共用的同一份函数，四键（`motion_emb` / `motion_pos` / `motion_mask` / `mem_order`）的构造与 context 版逐字相同；唯一随 integration 变的是 `mem_order` 长度（context 32 帧档 608 → 本轮 512 + 160 = 672）。
3. **在线装配层**：`src/mme_vla_suite/policies/` 目录下 grep `integration_type` **零命中**；`policy.py::_prepare_history` 的 motion 分支只看 `if self.motion_enabled:`。在线链路可以直接复用，由 V-online 证明。
4. **切窗与起点集合**：在线 `framesamp_memory.py::visible_motion_frames` 与训练侧 `visible_motion_rows` 同式。
5. **sidecar 线协议**：`(33,256,256,3) uint8 → (768,) f32`，与 integration 无关。
6. **补帧规则训推同源**：离线按 `entry.spec` 取行、在线按同一 `min(f+32, es−1)` 补帧（E 节改动后）。

**4.2 九项不等价（训练侧）**

| # | 维度 | context（正本口径） | modulation（本轮） | 代码锚点 |
|---|---|---|---|---|
| a | 注入点 | 记忆区 concat 进主干 prefix，走 18 层 self-attention | 每层对最后一条 expert 流（action expert）在 FFN 前做一次 cross-attention，输出作为 `MemoryRMSNorm` 的 cond 产生 scale / shift | `history_pi0.py::embed_prefix` 的 `if self.integration_type == "context":`；`history_gemma.py::HistoryBlock.__call__` 的 `if i == len(xs) - 1 and self.integration_type == "modulation": mem_mod_vec = mem_attn(x, mem_seq[-1], mem_mask[-1]); x = MemoryRMSNorm(name="mem_rms_norm_ffn")(x, mem_mod_vec)` |
| b | 谁看得到记忆 | 文本与动作 token 能 attend 记忆区，图像被 `na_mask` 挡住 | 图像 / 文本 / 整条主干完全接触不到；只有 `action_horizon=20` 个 action token 经 cross-attn **单向**读取 | 同上；正本 3.4 |
| c | 记忆是否被逐层加工 | 记忆 token 走 18 层残差流（「记忆只看记忆」） | `embed_memory` 算一次的同一条 `mem_seq` 原样广播给 18 层，只出 K/V；**记忆 token 之间没有 attention** | `history_gemma.py::Module.setup` 的 `nn.scan(..., in_axes=(0, nn.broadcast, …))`，`mem_seq` / `mem_mask` 走 `nn.broadcast` |
| d | 位置编码 | `positions = cumsum(input_mask) − 1`，padding 不占号 | `k_positions = arange(mem_len)`、`q_positions = arange(mem_len, mem_len + x_len)`，**padding 占号**、query 位置随 `mem_len` 整体平移 | `history_gemma.py::MemoryAttention.__call__` |
| e | mask 语义 | `make_attn_mask(input_mask, ar_mask, na_mask)` 三项 | 只有一维 K 侧屏蔽 `attn_mask = mem_mask[:, None, None, None, :]`；`compute_loss` 的 modulation 分支把 `embed_memory` 返回的 `ar_mask` / `na_mask` 直接丢弃（`mem_seq, mem_mask, _, _ = self.embed_memory(observation)`） | 同上；`history_pi0.py::compute_loss` |
| f | 出口维与参数量 | `memory_token_dim: 2048` ⇒ `motion_encoder_static` 1536→2048，新增 3,345,152 参数 | `1024` ⇒ 1536→1024，新增 `256×768+768 + 1536×1024+1024 = 1,771,264`，motion 表征宽度减半 | 两份 YAML；`percep_mem.py::PerceptualMemory.__init__` |
| g | 梯度来源与初始门 | motion 四叶从整条 prefix 的全部 query 收梯度 | 只从 20 个 action query 收梯度；整条记忆通道被 `MemoryRMSNorm` 里 `nn.Dense(x.shape[-1] * 2, kernel_init=kernel_init_out_proj)` 门住，`kernel_init_out_proj = nnx.initializers.normal(stddev=0.002)` ⇒ **初始贡献近零** | `representation/utils.py::kernel_init_out_proj`；`history_gemma.py::MemoryRMSNorm` |
| h | 主干序列长度 | 开 motion 后记忆区 608、prefix 1184、全序列 1204（正本 5.4） | 主干 = 2 视角 × 256 + 文本 64（prefix 576）+ 动作 20 = **596**，与基线完全相同；记忆（672）不进主干 | `robomme_policy.py::RoboMMEInputs`；`HistoryPi0Config.max_token_len = 64` |
| i | RMSNorm 共用 | — | `rms_norm = MemoryRMSNorm(name="mem_rms_norm")` 先作用于 `x` 再作用于 `mem_seq`，同一 `scale` 被 action-expert 隐状态与记忆序列共用 | `MemoryAttention.__call__` |

**两条训练动力学后果**（由 g 直接推出，不是猜测）：motion 梯度只来自 20 个 action query；记忆表征在 18 层之间不更新。motion 能否被学到，要靠F 表 V8 的 `MOTION_PARAMS_UPDATED` 与训完后的全遮消融来回答，不能靠「接线对了」推断。

**4.3 推理侧差异**

| # | 维度 | context | modulation |
|---|---|---|---|
| j | 在线数据装配 | — | **完全不敏感**：`policies/` 零 `integration_type` 分支（4.1 第 3 条） |
| k | `embed_memory` 本身 | 同一份代码 | 同一份代码（含同一次 `take_along_axis` 重排），唯一差别是返回值去了哪里与出口宽度 |
| l | 消费与缓存 | 记忆 K/V 一次烙进 kv_cache，10 步去噪不再碰 motion | `embed_memory` 在去噪循环**外**只算一次，但 `mem_seq=[None, mem_seq]` **每步**传入，`MemoryAttention` 每步每层重算 `mem_rms_norm(mem_seq)` 与 `kv_einsum_mem`（18 × 10 = 180 次 / infer），**无 K/V 缓存** |
| m | kv_cache 长度 | 随 motion 变长（记忆区进前缀） | **完全不变**（前缀只有 image + language） |
| n | `mask_na` | 让 vision 看不到记忆区，是实质机制 | **恒等空操作**：`embed_prefix` 在 modulation 下 `na_mask` 首元素即图像的 `True` ⇒ `cumsum(mask_na) <= 0` 全 False；`compute_loss` 传 na_mask、`sample_actions` 不传，两者等价，**不构成训推不一致，不要当 bug 去「修」** |
| o | 推理频次与算量 | — | `eval.py::eval_each_episode` 每 16 环境步调一次 `get_action_chunk` ⇒ `max_steps=1300` 下每集 ⌊1296/16⌋ + 1 = **82** 次 infer；S=672 时 `MemoryAttention` 每层每步约 421.8 MMAC（K/V 投影占 83.5%）× 18 × 10 = 75.9 GMAC / 次，S=512 为 59.7 GMAC ⇒ motion 带来 **+16.3 GMAC / 次（+27%）**、每集 +1.34 TMAC ≈ **+9 ms / 集**（按 A100 fp32 峰值推算，未实测）；同期 sidecar 每集最多 151 窗 × 1421.5 ms ≈ **215 s / 集**，比模型侧增量大四个数量级，新 demo 规则只让每集多 1 窗（+1.4 s） |


#### I.2 由此新增的三条验证与消融臂（判据细节）


- **`INIT_COMMON`（2b CPU 单测，秒级）**：开启态下 6 条 modulation 叶（`mem_attn/{mem_rms_norm/scale, kv_einsum_mem/w, q_einsum_mem/w, out_einsum_mem/w}` + `mem_rms_norm_ffn/Dense_0/{kernel, bias}`，18 层堆叠约 85 M 参数，全部只能来自随机初始化——基线 `records/startup.log` 实测 6 条 `Merging missing weight`）与基线是否同一初值，现有 V 表**没有任何一项能核到**：V6 的 `INIT_EQ` 比的是「同一 YAML 改前 vs 改后」，V8 是同侧 A/A。`percep_mem.py` 自写「flax nnx 单条 default RNG 流按调用顺序 fold_in，插在前面会改变帧路的初始化值」，而两个 motion `nnx.Linear` 正建在 llm 之前；`0901` 计划的「RNG 消耗序」表与其闸门 `T3_COMMON_INIT` 是在 context（`mem_mods=[False, False]`，llm 内无随机初始化叶）下建立的，**不能默认沿用**。判据：同 `seed=42`、`JAX_PLATFORMS=cpu`，分别用关闭态与新 motion YAML 只做 `init_train_state(..., resume=False)`，按叶名比 6 条 modulation 叶与 4 条 motion 叶的 sha256，`INIT_COMMON=PASS common_mismatches=0 open_only=4`；不相同则先量差异再定处置。
- **`ROPE_LEN_EFFECT`（2b CPU 探针，几十行）**：I.4 引用的五个数字（144→160 总变差 0.33 / 相对 L2 0.76；512→672 为 0.87；gap 非 0 时总变差饱和 ≈ 0.4；改成不占号或去 RoPE 后 1e-17）在仓库里**没有任何可追溯的脚本或留档**（全仓 grep 零命中）。机制本身可核且成立（`src/openpi/models/gemma.py::_apply_rope` 是标准 RoPE；`shared/sampling.py::pad_times` 用 `MEM_ORDER_SENTINEL` 把 padding 排到尾部 ⇒ 真实 key 位置不随 budget 变、只有 query 整体平移），但数字必须重取：固定 token 内容、只改 `mem_len`，直接构造 `MemoryAttention` 或复算 `_apply_rope` + masked softmax，输出 `ROPE_LEN_EFFECT=PASS tv_144_160=… relL2_144_160=… tv_512_672=… pad_content_diff=0`，json 落 `v1-store/reports/motion/`，写入正本前用它替换五个数字。
- **`EVAL_ES_BOUND`（起跑前，单卡渲染）**：见第七节。
- **motion 全遮消融评估（用户已拍板加入）**：80k 训完后在同一 ckpt 上把 `motion_mask` 整列置 False 再评一次。`MemoryAttention` 的 `masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)` 保证被 mask 的 K 位 softmax 概率恰为 0，故该臂是「同权重、同 `mem_len=672`、同 q 位置，只是看不到 motion 内容」，Δ 即 motion 内容净贡献（帧路 `static_mask` 恒有 True，不存在全 mask 导致 softmax 退化的边界）。成本一次评估；训练侧对照（budget=160 且全库 mask 恒 False 的 80k run）另需约 **21.6 h**（8 卡下界口径，I.9），**本轮不做**。第八节据此明确：**本轮承诺的是实现等价，不是因果归因**；归因由消融臂承担。


#### I.3 措辞纠错表与正本待改段落清单


| 原表述 | 问题 | 改为 |
|---|---|---|
| 「`mem_order` 交错 ⇒ 帧 token 彼此的 RoPE 相对距离改变」（I.4 与 H 节两处） | modulation 下记忆 token 只作 key/value，彼此之间没有 attention，它们的相对距离不进入任何计算 | 「`mem_order` 改变的是每个记忆 token 的绝对 `k_position`（`arange(mem_len)`），从而改变它到全部 query 的 RoPE 相对距离」；`ORDER_EFFECT` 在 modulation 下只证「次序经 `k_position` 改变 cross-attn logits」，是比 context 弱的承诺 |
| 「唯一 width=1024 的是 `gemma_300m`」 | 变体表里 width=1024 / depth=18 的有三个：`gemma_300m`、`gemma_300m_lora`、`gemma_150m` | 「没有浅层 width-1024 变体，CPU dummy（宽 64）仍撞 `assert mem_width == x_width == width`；但 VLM 主干可换 `gemma_150m` 替身」——V5 已拍板用乙（`gemma_150m` 替身） |
| 「唯一拦得住的是 `_load_resolved_snapshot` 的快照 sha 三方核对」+ 「训练、resume、评估、在线四侧」 | 三方 sha 全在同一 run 根、由 `init_history_config` 一次写出，只能查该目录被篡改；真正使评估侧不可能用错 budget 的是 `create_trained_policy` 直接丢弃传入的 history_config、整体换成 run 内快照；`train.py` 硬禁 `--resume`，不存在 resume 侧 | 两层表述：生产评估由快照替换**结构性保证**；sha 三方核对只防事后编辑；真实暴露面是按仓库 YAML 文件名建模型的 `check_ckpt_param_tree.py` 与一切 `--yaml` 驱动的工具 + 「参数树对 budget 失明」的组合 |
| `scripts/motion-variance/*` 只当「写死 96 / 608」的字面量处理 | 结构性不适用：`mv_common.py` 的 `PREFIX_LEN  # 1184`、两处 `integration_type="context"` 硬编码 | 本轮不改，**且不得引用其结论（`docs/motion-utilization.md` 的 normal−mask +5.00pp）作为新 run 的 motion 利用率证据** |
| 正本第十章第 12 条「未覆盖 expert / modulation」 | 实施后过时 | 改成「已覆盖 modulation（本文），expert 仍未覆盖」，不整条删除 |

**正本 `docs/motion-memory.md` 待改段落清单**（实施后另立 `docs:`）：3.1 / 3.4 / 3.5（交错口径只对 context 成立，补 modulation 的 `k_position` 段）、4.2（模型层）、5.3 / 5.4（608 / 1184 / 1204 是 context 32 帧档数字）、6.4（`gap` 改名以区分 `rope_gap`）、第八章两项 FAIL 的适用性、第十章第 12 条。


#### I.4 训练链路：budget 成为模型语义参数的技术表述

**一条必须写进正本、且初稿写错了的语义（第四轮再次订正）**：modulation 的 `MemoryAttention` 用 `q_positions = arange(mem_len, mem_len + x_len)`、`k_positions = arange(mem_len)`，而 `_apply_rope` 是**纯相对位置**旋转。padding 全在尾部、真实 key 位置不随 budget 变，但 **`mem_len = 512 + budget` 一变，所有 query 的位置整体平移，query 到每个真实 key 的相对距离全部 +160**。同一 budget 下只把 padding 位塞垃圾，差异严格为 0（`masked_logits = where(mem_mask, logits, -2.38e38)` 保证 padding 位 softmax 概率恰为 0）。第三轮引用的五个数值实验数字（0.33 / 0.76 / 0.87 / ≈0.4 / 1e-17）**无留档来源**，本轮以 2b 的 `ROPE_LEN_EFFECT` 探针重取后再写入正本（I.2）。

结论：**`motion.budget` 是模型语义参数，不是纯容量上限。** 训练、评估、在线三侧必须逐值相同（`train.py` 硬禁 `--resume`，不存在 resume 侧），任何一侧改 budget 都等同换模型。这条**不受参数树保护**——motion 四个参数叶的形状只依赖 `pos_dim/dim/pos.hidden_dim/memory_token_dim`，与 budget 无关，拿 budget=96 的 YAML 去加载 budget=160 训的 checkpoint，`PARAM_TREE_EXACT` 会照样 PASS 而语义已静默改变。生产评估侧由 `create_trained_policy` 直接丢弃传入的 history_config、整体换成 run 内快照**结构性保证**用对 budget；`_load_resolved_snapshot` 的 sha 三方核对只防 run 目录被事后编辑。真实暴露面是按仓库 YAML 文件名建模型的 `check_ckpt_param_tree.py` 这类 `--yaml` 驱动工具，所以F 表 V9 的 `BUDGET_CONSISTENT` 改成**跨源**比对（run 内快照 vs 仓库新 YAML vs 160）。

#### I.5 推理侧算量、「同输入 ⇒ 同输出」证据与 V-online 成本口径

**推理侧算量（I.1 的 4.3 o 行的结论）**：主干 kv_cache 不随 motion 变长；`MemoryAttention` 每步每层重算，S 512→672 约 +27%，折合每集约 +9 ms（推算未实测）；评估耗时的主导项是 sidecar 逐窗编码（每集 ≤ 151 窗 × 1421.5 ms ≈ 215 s），本轮新 demo 规则只让每集多 1 窗。「评估会被 modulation 拖垮」这个担心不成立。

**怎么证明同源**：既有的在线对拍工具按 eval 节奏驱动 `FrameSampMemory` + 真 sidecar，逐窗与离线表比逐位。**成本口径第四轮订正**：1421.5 ms/窗 出自 `v1-store/reports/motion/p5_online.json` 的 `ENC_MS_PER_WINDOW mean`（40ep 库、旧 ckpt、A100 独占一卡、含 6.3 MB IPC；离线探针 A2 为 1413.5 ms），全量 71,316 窗 × 1.4215 s ÷ 8 卡 = **3.52 h**（不是第三轮写的「单卡 30–32 h 不可承受」），与 Wan 抽取的 8 卡 3.6 h 同量级；40ep 那次全部非编码开销 ≤ 71 s，h5 读取不是瓶颈。用户据此拍板改为**方案 B「分层解耦」**：本轮唯一新增的机制是 demo 段 `min_real` 33→17 + `np.repeat` 补帧，风险落在「输入 33 帧是否装配正确」，而 encoder 的「同输入 ⇒ 同输出」已被 `wan_motion_infer.pin_numerics()`（`cudnn.benchmark=False` / `allow_tf32=False` / `float32_matmul_precision=highest`）、`check_versions(strict=True)`、`motion_client` 握手 provenance 逐键比对（含 `gpu_name` / `compute_cap` / `sm_count` / `driver` / `flags` / `module_sha256`）、`0901` 计划记的 `A3_CROSSGPU=PASS compared=64 latent_bitwise=64 token_bitwise=64 max_abs_diff=0.000e+00` 与正本记的 40ep「772 窗逐位相同」反复确认——所以用零 GPU 手段 100% 覆盖「输入是否相同」，用最贵手段只覆盖新代码路径的全部补帧窗。三段判据与承诺口径见F 表 V-online 行；换新 ckpt 后「同输入 ⇒ 同输出」只需重取一次小样本确认，不需要全量。

#### I.6 budget 推导备注

按 ≥ 17 规则逐样本实算合法窗数（605,611 个）：均值 **43.9**，中位 39，P90 80，P95 93，P99 115，最大 **141**（BinFill ep367，es 1152，nt 2304 → demo 71 + exec 70）。第四轮反驳者用 `episode_manifest.json` 独立复算：total_samples=605611、mean k=43.906481、kmax=141（g=367）、k≥140 恰 32、`t==es` 恰 1600、`δ==32` 恰 1600、over96=24366 / over112=7817 / over128=488 / over144=0、涉及 episode 112 / 43 / 6 / 0、中位 39、P90/95/99 = 80/93/115、填充率 27.442%——与本节与I.7 V4 期望值逐项吻合。

> **填充率定义**：`mean(min(k, budget)) / budget`（截断后实际占位）。初稿未写定义且整张表用了一个在 `δ < 32` 时多算一窗的 exec 公式（`len(range(0, max(0, δ−32)+1, 16))` 在 `δ<32` 时返回 1，正确值是 0），导致均值与填充率系统性偏高。正确 exec 可见窗数：`E(δ) = 0 if δ < 32 else (δ − 32)//16 + 1`，`δ = t − es`。受影响的恰是每集 exec 段前 32 个样本共 51,200 个，`51,200/605,611 = 0.08454` 与两版均值差 `43.99102 − 43.90648 = 0.08454` 吻合。**截断数、截断 episode 数、中位、P90/P95/P99、最大值不受影响。**

> 列名用「超限」而非「截断」：代码里 budget 超限**从不截断，一律 raise**（`FrameSampDataset.__init__` 的零截断契约预检、`__getitem__`、右填充三处各一道）。budget=96 时这个数据集根本构造不出来，不是「4.02% 的样本被截断」。

**最后一列度量的是训练集，不是评估集。** 它按 `k_eval = demo_num_grid(es) + E(τ_max − es)` 外推，`τ_max = es + 16·⌊max_steps/16⌋`（`max_steps=1300` → exec 项恒 80），训练集 `es` 最大 1152 ⇒ `k_eval` 最大 **151**；按正本红线在 16 任务全集口径复核，同为 **151**（`VideoPlaceOrder` ep10，es=1140）。

> 初稿 C 节写的 `τ_max = es + 16·⌊(max_steps − 4)/16⌋` 里那个「−4」不成立：`⌊(M−4)/16⌋ = ⌊M/16⌋` 当且仅当 `M mod 16 ≥ 4`，而 `1300 mod 16 = 4` 恰好卡在边界。反例：M=1296 正确 1296、该式给 1280；M=1312 正确 1312、该式给 1296；M=800 正确 800、该式给 784。代码里 `eval_rhythm_gates.py` 本来用的就是 `es + 1296`，与修正式一致。

**评估集不可预知 → 起跑前测出来（第四轮）**：评估将走**新的 seed**，仅「1300 步」这一条固定。当前 `examples/robomme/env_runner.py` 硬编码 `dataset="test"`，`es` 由 `DemonstrationWrapper.reset` 现场 rollout 生成、`examples/robomme/eval.py` 现取，训练 manifest 约束不到它；现有 test split（4 任务 × 50 集，seed 540000+）与训练库 seed 4000–18500 完全不相交，且训练集里 `VideoRepick` 只有 easy/medium/xhard、没有 test split 里的 hard 档。因此：

- budget 160 在 1300 步下零截断的**充要条件是评估 `es ≤ 1296`**；
- 训练集实测上界 1152，裕度 144 帧（12.5%），唯一风险线是 BinFill hard（训练 es 578–1152）；
- **新增预检 `EVAL_ES_BOUND`（起跑前、单卡渲染）**：遍历 4 任务 × 全部 test 集，只做 `EnvRunner.make_env(i)` + `get_init_obs()`（无策略、单卡 GPU 渲染，共 200 次 `env.reset()`），收集 es 分布，判定行 `EVAL_ES_BOUND=PASS tasks=4 episodes=<n> es_max=<m> budget=160 headroom=<160 − k_eval_max>`；把「es 不可预知」从赌变成测，`headroom < 0` 即回到本节重定 budget。
- **`es ≥ 1297` 的真实后果（第四轮订正，比第三轮写的「该集记 error」严重得多）**：raise 点唯一，在 `FrameSampMemory._prepare_motion`（`_encode_ready_windows` **不做上限检查**，超限的集会先白编一堆窗再在第一次 infer 时炸）；服务端 `websocket_policy_server._handler` 发回 traceback、关连接后 raise，server 进程存活；但 `examples/robomme/eval.py::evaluate` 的控制流是——episode 异常只写 `"error"` 后 `if success_flag == "unknown": … return` 整轮提前返回；只要 progress 里出现过一个 `"error"`，收尾的 `success_rate = sum(...)` 混入字符串抛 `TypeError` 被吞、`log.json` 永不落盘、外层 `while not os.path.exists(...)` 把全部任务重跑一遍，且续评跳过判据用 `str(episode_id)` 而运行中写的是 int 键 ⇒ 重跑一集也不跳。代价是几十 GPU·小时白跑且拿不到成功率。评估计划另起时须先修 `success_flag` 作用域与 `success_rate` 对 `"error"` 的处理，或把超 budget 的集显式降级；`EVAL_ES_BOUND` 是它的前置门。
- 评估口径或任务集变更时须按第二部分 C 节的式子重核。

**代价栏**（初稿写的「只是多 16 个被 mask 的 K/V 位」是错的，已删）：K/V 长度 512 → 672（+31%），`MemoryAttention` 在 18 层里每层都对整条 mem_seq 做 K/V 投影、`sample_actions` 每个去噪步再算一遍，且 remat 策略是 `nothing_saveable`（反向要重算）——但按I.1 的 4.3 o 行推算，这一项折合每集只有约 +9 ms；训练步时里的占比在 **4 卡数据受限档**下是毫秒级量级，但 8 卡 + collate 共享内存之后数据路已不再是瓶颈，**新档位下的瓶颈归属尚未实测**（0917 只测 motion 关闭态，且 NVML util 不区分计算与通信），由 300 步分解裁决（I.9）。位置编码的代价也需要单独考虑：`rope_gap(i,j)=512+budget+i-p_j`。本库 `k` 为6..141，满512个有效帧位时 `budget-k` 为19..154，这描述padding长度；query到最后motion的距离还取决于交错位置 `p_j`。固定内容与排列时增大budget使距离整体平移、方差不变，不能将19..154写成准确的query-key距离。固定fp32探针实测144→160的概率总变差为0.3375600576、输出相对L2为0.7787887454，同budget垃圾padding影响为0，详见实施记录。预算仍按用户决定固定为160。

#### I.7 验证体系每项五层叙述（原第八节全文，与 F 表互为补充：F 是命令级，本节是每项「证什么 / 怎么证 / 判定行 / 耗时 / 不过怎么办」）

**先说口径，再说每一项**。本节按「A 关闭态不能变 / B 旧东西还能用 / C 新表对不对 / D 开启态对不对 / E 秒级 CPU 单测 / F 跑完后」六组、12 个 V 项一个不漏，每项五层：证什么 → 怎么证（命令、夹具、档位内联）→ 判定行（标「实测」= 既有工具原样输出并给产出者，或「新写」= 本轮定义、须先落地自测）→ 耗时 / GPU / 留档 → 不过怎么办。`AGENTS.md` 第 18 条的两块划分放在本节末尾单独说，不再与六组分类叠着写。命令级细节在 F 表，本节内联的是读者不翻代码就能核对的那部分。

**两条贯穿全节的纪律**
- **两侧源码身份必须机器核对**（第四轮 P1）：V1 / V6 / V7 三个比较器**都不比 `git_head`**（`compare_fixture_dumps.py::compare` 不读 `DUMP_MANIFEST.json` 的 `git_head`；`compare_fixed_grad.py::compare` 只比 `schema / seed / fsdp_devices / xla_flags / environment` 与 `history_config_sha256`，不比 `source["head"]`；`compare_baseline.py` 同），两侧误跑在同一 commit 时三项全绿。所以 2a / 2d 各记 `git rev-parse HEAD` 与 `git status --porcelain` 入留档，对拍前统一 `jq` 断言两侧 `.git_head` / `.source.head` / `.start_head` 等于预期 SHA、`start_status` 为空，并显式断言 `$BASE != $CAND`。
- **FAIL 先分流再处置**：`reason=运行口径不同：…`、白名单 raise、目录已存在这类 harness 失配先排查；只有 `SAMPLE_RAW_EXACT` / `INIT_EQ` / `GRAD_EQ` / `STATE_DIGEST` 这类**数值项**失配才走第九节 2d 的 revert，不得先改判据。

##### A. 关闭态：改前 vs 改后一个比特都不能变

**口径行（第四轮补，之前读起来像「在生产配置上证明了」）**：三项都在旧库小档位上证——V1 用 400ep 旧库 8×8 档、V6 用 8 帧夹具 `c8-b` 2 卡 b8 fsdp2、V7 用 2 卡 b8；**生产 8 卡 b128 + 1600ep 新库上的关闭态等价属推论、不是实测**（第五轮改 8 卡后档位差距比原先更大，推论性质不变）。推论成立的理由：关闭态下本轮改的代码（`MotionMeta.load` / `parse_index` 的 layout 查表、`FrameSampDataset.__init__` 的三态判定、`ref_npy_dataset` 守卫）大多根本不执行（`MotionMeta.load` 只在 motion 开启时调），能执行到的只有 `motion_store` 模块级 import 守卫与 `_NONE_KEYS`。所以这三项证的是「模块可 import + 关闭态交付 / 初态 / 梯度 / 前 100 步逐位不变 + 四键恒 None」；本轮新逻辑的覆盖来源逐项点名：旧表读取 = V2①②③、网格公式与边界 = E 组的 A8 离线边界用例表、新表交付 = V3 + V4。

**V1 关闭态 dataset 交付逐位**
- 证什么：同一 3,200 个样本 / 200 个 batch 的 dataset 交付改前 vs 改后逐字节相同，排除交付层被波及。
- 怎么证：`dump_fixture_samples.py` 2a / 2d 各一次 + `compare_fixture_dumps.py`。库 `4task-motion-400ep/framesamp-8x8`（新库 `exec_start_idx` 全 ≥ 100，`fixture_per_step` 找不到 `es=0` 的 episode 必 raise）；环境 `DTYPE_DUMP_DIR`（两侧各一空目录）/ `DTYPE_DUMP_GIT_HEAD` / `DTYPE_DUMP_MODE=both` / `DTYPE_DUMP_ARRAYS=1` / **不设** `DTYPE_DUMP_LIMIT` / `JAX_PLATFORMS=cpu`；YAML `perceptual-framesamp-modul-8frame-8x8.yaml`（白名单已在 1b 放行）。
- 判定行（实测，产出者 `compare_fixture_dumps.py`）：单侧 `DUMP_DONE samples=3200 batches=200 out=<dir>`；对拍 `SOURCE_IDENTITY=PASS episodes=400 samples=101066` / `FRAME_INDEX_EXACT=PASS … max_frames=8` / `SAMPLE_RAW_EXACT=PASS samples=3200 per_step=200 mismatches=0` / `BATCH_RAW_EXACT=PASS batches=200 mismatches=0`。失败是抛异常 + 退出码 1，无 FAIL 行。
- 耗时 / GPU / 留档：CPU；> 5 min，tmux `mv2-v1`，留档 `docs/training-doc/mv2-v1-dump/`。
- 不过：按开头分流；`SAMPLE_RAW_EXACT` / `BATCH_RAW_EXACT` 失配才 revert。

**V6 关闭态定点梯度逐位（单步、固定夹具、不启动训练）**
- 证什么：同一固定 batch 的单步梯度改前 vs 改后逐叶逐位相同，排除公式参数化改动波及关闭态数值。
- 怎么证：`single_step_grad_fixed.py` ×2 + `compare_fixed_grad.py`。夹具 `DTYPE_BATCH_FIXTURE_DIR=v1-store/fixtures/8x8/grad/c8-b`（8 帧档、12 个数组键 + 4 个 `none` motion 键，满足工具「必须恰好 12 数组键」的断言；`m8-*` 是 16 键会 raise）；`--batch-size 8 --fsdp-devices 2 --seed 42 --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml`；`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`；`DTYPE_GRAD_DIR` 预先不存在；`unset DTYPE_BASELINE_CHECKSUMS`；`DTYPE_SOURCE_COMMIT=$BASE` / `$CAND` + porcelain 为空。**硬约束（第四轮）**：比较器对 `environment` **整体等值、无白名单**（含 `cuda_visible_devices` 与 `uv_lock_sha256`），故两侧必须 `export CUDA_VISIBLE_DEVICES=4,5`、2b 禁止 `uv add`（`uv.lock` sha 三处各记一次入留档）；确需换卡须重跑 2a 而非放宽比较器（`0908` 计划记 2026-09-04 实际发生过跨卡 gate FAIL）。该工具把关的是 `_CONFIGS` 与 `main()` 的 `raise ValueError("只接受 modulation 关闭态两档配置")`（不是 `_EXPECTED_HISTORY_CONFIGS`），modul YAML 已在其中、无需改。
- 判定行（实测，产出者 `single_step_grad_fixed.py` / `compare_fixed_grad.py`）：单侧 `FIXTURE=PASS kind=… keys=12 batch=8`（×3）/ `INIT_PARAMS=PASS leaves=<N>` / `GRAD_KIND=PASS kind=…`（×3）/ `GRAD_DONE kinds=3`；对拍 `INIT_EQ=PASS leaves=<N> mismatches=0` / `GRAD_EQ=PASS kinds=3 leaves=<N> mismatches=0`。
- 耗时 / GPU / 留档：约 15 min，GPU 4,5；留档 `docs/training-doc/mv2-v6-grad/`。
- 不过：`reason=运行口径不同：environment` 是 harness 失配先排查；`INIT_EQ` / `GRAD_EQ` 数值失配才 revert。

**V7 关闭态 100 步守卫（第 18 条第二块，唯一一项）**
- 证什么：真训练前 100 步，逐步 loss / 参数摘要 / 输入摘要改前 vs 改后逐位相同，排除多步累积差异。
- 怎么证（第四轮补全，第三轮只写了 `bench_train_steps.py` 一步、照写跑不起来）：驱动 `scripts/training/g0/run_2gpu_epoch_bench.sh`，一行式照 `docs/training-doc/aws-t3-closed-s100/launch.md`，显式给 `STEPS=100 SAVE_INTERVAL=25 EXTRA_DIGEST_STEPS=99 WORKERS=4 HISTORY_CONFIG=perceptual-framesamp-modul-8frame-8x8.yaml EXP_NAME=<run> RUN_TAG=<tag> BENCH_GPUS=4,5 DATASET_PATH=v1-store/datasets/4task-motion-400ep/framesamp-8x8 XLA_FLAGS='…'`（驱动默认 `DATASET_PATH=${GL_DATASET}=4task-gl` 本环境不存在；`BENCH_ROOT` 写死 `v1-store/bench/2gpu-epoch-bench`，产物落 `<BENCH_ROOT>/<RUN_TAG>`；驱动自带第三处 history config `case` 白名单，1b 一并放行；驱动的 `SAVE_INTERVAL` / `EXTRA_DIGEST_STEPS` 会转成工具读的 `BENCH_DIGEST_INTERVAL` / `BENCH_EXTRA_DIGEST_STEPS`）。完整链路：`check_baseline_env.py dump`（两侧）→ `manifest`（2a 侧，产 `BASELINE_MANIFEST.json`，**缺它 `BASELINE_ENV` 恒 FAIL**）→ `check --base … --steps 100 --batch-size 8`（2d 侧）→ bench（两侧）→ `project_scalars.py`（两侧，产 `scalars_hex.tsv`）→ `compare_baseline.py`。摘要步 `{0,25,50,75,99}`、输入摘要步 `{0,1,2,25,50,75,99}`：state 族 = 步 0 强记 + save 调度经 gate 过滤，batch 族恒含 `{0,1,2}` 与末步（第三轮说「`[0,1,2,24,49,99]` + rows=6 任何参数下都产不出」是误判——那正是 `t8-c8-guard-s100` 用 `BENCH_DIGEST_INTERVAL=1000` + `EXTRA=1,2,24,49` + `--save-interval 1` 的实际产出）。
- 判定行（实测，产出者 `compare_baseline.py` / `check_baseline_env.py`）：`BASELINE_ENV=PASS` / `SCALARS steps=100 keys=5 hex_mismatch_steps=0` / `INDEX_SEQ=PASS n=<共同前缀长度>（≥ 800，含 prefetch 余量；同档位历史实测 872，不是常量）` / `STATE_DIGEST rows=5 mismatch=0` / `BATCH_DIGEST rows=7 mismatch=0` / `BATCH_DIGEST_CANONICAL rows=7 mismatch=0` / `CANON_CHECK=PASS steps=7` / `DET_CHECK=…`。汇总（新写，`finish_check.py` 本轮定义）：`INDEX_TRAIN=PASS n=800 recorded_n=<n>`（`base_indices[:800] == cand_indices[:800]`）与 `GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5`。**两个 800 来自不同工具、含义不同，不可统一成一个数。**
- 耗时 / GPU / 留档：30–35 min × 2，GPU 4,5，tmux `mv2-bench-base` / `mv2-bench-cand`；留档 `docs/training-doc/mv2-v7-guard-base/` 与 `mv2-v7-guard-cand/`（先把 `metrics.jsonl` / `*digests.jsonl` / `param_checksums.jsonl` / `env.json` 拷入 `records/` 再删 run 产物，第 6 条）。
- 不过：同 V6 分流。

##### B. 旧东西还能用（V2，三条）

- **V2① 旧表可读**：`MotionMeta.load` 两张旧表（40ep rows=772、400ep rows=6832，实测 `schema=1`、`layout="motion-768-grid16-v1"`、无三新键）且解析出 `LayoutSpec(demo_min_real=33, exec_min_real=33, demo_tail_pad="none")`。**能过的前提是第四轮两处修正**：`INDEX_SCHEMA` / `META_SCHEMA` 不升版（A1）、`store_meta` 三新键按 layout 分档兜底（A1）。判定行（新写，`legacy_regress.py`）`LEGACY_LAYOUT=PASS tables=2 rows=772/6832 spec=(33,33,none)`。秒级。
- **V2② 旧 YAML + 旧表交付逐位**：**改成 2a 与 2d 各跑一次 `dump_fixture_samples.py` + `compare_fixture_dumps.py`**（第三轮写「2d 跑新写的 `legacy_regress.py`」，改前对照物不存在）：`_EXPECTED_HISTORY_CONFIGS` 已含 `perceptual-framesamp-context-motion.yaml`，40ep 库同时有 `framesamp` / `motion` 两套产物且清单含 `exec_start_idx == 0` 的集；N 沿用 V1（`per_step=200` + 默认 `N_RANDOM`）。判定行（实测）同 V1 四行；汇总（新写）`LEGACY_YAML=PASS`。这条是唯一覆盖「旧 YAML 两键同缺 → `(33,"none")` 兜底 → 旧表 spec 相等」分支的交付层判据。> 5 min，tmux `mv2-v2`，留档 `docs/training-doc/mv2-v2-legacy/`。
- **V2③ 历史 run 快照真走 policy 构造**：`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/5000` 走 `create_trained_policy(..., motion_stub=True)`。**表述订正**：`motion_stub=True` 仍会 `Popen` 一个 stub sidecar 子进程（`uv run --project scripts/dataset/wan --no-sync python motion_sidecar.py --stub`）并走握手，只是不加载 encoder、不占 GPU；前置 `ls v1-store/venvs/wan/bin/python`。该快照 `history_config.resolved.yaml` 的 `motion:` 节实测无两新键 ⇒ 只能走 `(33,"none")` 分支，**在线守卫必须放行这一对**（E 节；第三轮的硬等值 `repeat_last` 会让这条必 FAIL）。该目录不得改动任何文件。判定行（新写）`LEGACY_POLICY=PASS spec=(33,none) stub=1`。秒级。

##### C. 新表本身对不对（V3，建库侧，第九节步 3 / 4）

链路与判定行全在 B 节，这里只给读者一条线：输入重锚 `INPUT_REANCHOR=PASS files=4`（4 个 H5 内容 sha256 == 帧库 finalize 时的 `input_manifest.json`）→ Wan `STAGE_DONE stage=wan workers=8 items=3200 skipped=0` → encode（同式）→ pack / verify `PACK_MOTION_DONE=1` / `VERIFY_MOTION=PASS scanned=71316 mismatches=0` → oracle：D3 encoder **全量** `ENCODER_BITEXACT=PASS compared=71316 mismatches=0`；D2 VAE **抽样只缩前向**（补帧窗全部 + 其余 10%，段集合 / 行数 / 逐行 m·offset / 帧 sha 四道检查全量）`ORACLE_VAE=DONE metadata_rows=71316 windows=<抽样数> …` / `WAN_BITEXACT=PASS compared=<抽样数> padded_covered=1600 frame_mismatches=0 latent_mismatches=0`——**比较器必须带与 oracle 逐字相同的 `--sample-spec`**，自算抽样集合与 `sampled_windows.json` 双向差集为 0、`compared == len(自算集合)`（A7；第三轮 B 节命令漏了这个参数，比较器只能退回照单比对）→ 账目**五项**（第三轮写「六项」是计数错）：`A6_SAMESOURCE=PASS` / `A7_BYTES=PASS` / `A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0`（**只在步 4 跑一次**，D 组 V4 引用其判定行、不重跑）/ `A10_ROWS=PASS rows=71316 exec=35403 demo=35913 episodes=1600`（`--expect-*` 写死、不取 `motion_index.json`；第四轮反驳者用 manifest 独立复算 demo 35,913 + exec 35,403 = 71,316，按任务 BinFill 31,308 / VideoRepick 18,205 / RouteStick 12,626 / VideoUnmaskSwap 9,177，与本文吻合）/ `A11_PAD=PASS segs=3200 rows=71316 padded=1600 set_diff=0`。本轮 **D1 不跑**（帧库不重建），其「像素同源」位由 `INPUT_REANCHOR` 顶替；`D1–D3` 与 `a*` 代号的出处是正本 `docs/motion-memory.md` 7.5 节与第八章。

##### D. 开启态新东西对不对（第九节步 6）

**V4 开启态 dataloader 交付逐位（第四轮从「只有计数」改成逐位判据，P0）**
- 证什么：dataloader 交付的 `motion_emb` 每一有效行与 `motion_token.f32.bin` 对应行逐字节相同、`motion_pos` 取对列、`motion_mask` == `arange(budget) < k`、`mem_order` 与独立手算置换相同、填充区全零。第三轮的 `V4_COVER` 全是计数字段，取行偏移错一行 / pos 切错列 / `_pad_motion` 错位 / `mem_order` 置换错向都照样 PASS。
- 怎么证：**复用并参数化 `scripts/training/tests/hand_calc_8frame.py`**（它就是「交付 vs 表字节 + `mem_order` 手算」：`f.seek(row*768*4)` + `item["motion_emb"][i].tobytes() != f.read(768*4)` → raise，再比 `motion_mask` 与 `mem_order != expected_order(...)`；第三轮把它列成「必改但本轮不跑」并另新写工具重造同一判据）。三处必改：`for es in (0, 66, 114)` 在新库 StopIteration（es 最小 100、es=66 零集）→ 按 `(es−17) mod 16` 残差分层现挑；YAML stem 写死 context → 参数化到 modul-motion；两处 `96` 字面量 → `budget`。样本集**写死**：冷启动 `t==es` 全取 1,600、首 exec 窗 `t==es+32` 全取 1,600、`k≥140` 的 32 个全取、16 个残差档各取 ≥ 2（分布 r=0..15：55/186/49/136/49/112/54/365/52/126/55/86/34/129/39/73）。
- 判定行：`HAND_CALC_8FRAME=PASS samples=<n> mismatches=0`（实测，既有）+ 新写 `V4_COVER=PASS cold=1600 first_exec=1600 kmax=141 kge140=32 pad_r=16/16`——四个数做**等值断言**（第三轮留成 `<n>` 占位符，任何 n 含 0 都 PASS）；`pad_r` 是覆盖统计、不构成交付判别，补帧正确性举证在 C 组的 `A11_PAD` 与 D2 / D3。
- 耗时 / 留档：CPU，> 5 min，tmux `mv2-v4`，留档 `docs/training-doc/mv2-v4-deliver/`。

**V5 模型侧接线正确性（modulation）**
- 证什么：同一 budget 下 padding 内容（有限值，1e3 量级正态）不影响 loss 与**全部可训练参数**的梯度；真实 motion 内容改变必改变 loss；4 条 motion 叶在有 motion 时梯度非零；`mem_order` 次序经 `k_position` 改变 cross-attn logits（`ORDER_EFFECT` 在 modulation 下只证这一点，比 context 弱）；行置换不变性。`ZERO_MOTION_EQUIV`（全 mask ⇒ 等价关闭态）在 modulation 下**不适用**——比的是 `mem_len=672` 与 `512` 两个模型，标 `LEN_EQUIV_NA` 并写明 RoPE 原因（I.4）。
- 怎么证：`motion_gates_model.py` 六层适配见 F 表。**模型档已拍板用乙**（用户 09-17）：`paligemma_variant="gemma_150m"` + `action_expert_variant="gemma_300m"`——modulation 记忆路只经 action expert 流，`MemoryAttention` 的 4 头 / 256 头维 / 1024 宽 / 18 层、真 YAML、真库、真 budget=160 全为真；不用甲档（`gemma_2b` 主干约 2.6 B 参数、fp32 约 10 GB + 编译，记忆路一个字节都不经过它）；留档写明「VLM 主干用 `gemma_150m` 替身」。**确定性纪律（第四轮 P0）**：照抄 `cmd_t3mechanism` 已验证的四项——R≥3 同 obs 探针、不确定叶单列并排除（硬闸保留「loss R 次逐位相同」与「不确定叶含 `mem_encoder` / motion 叶即 FAIL」）、比较集合取 `get_freeze_filter()` 的补集（非 lora 档 `PathRegex(".*img.*")`）、挂与 V6 同一组 `XLA_FLAGS`；正本 8.3 记 A100 上 `['PaliGemma']['llm']['embedder']['input_embedding']` 叶曾不确定（09-07 后未复现，属间歇），不加探针会在与 padding 无关的叶上误报 FAIL。**显存纪律**：逐叶比较改流式（每叶算 sha / max_abs 后立即释放，禁止同时持有两棵完整梯度树；同文件已有「ema / opt_state 在初态校验后即释放，否则第二次 `value_and_grad` OOM——2026-09-03 实测 44 GB」先例）。样本 spec 分两类：库内可得 `m` 中位与 `m=max=141`（写死 `BinFill ep367, es=1152`）；必须合成 `m=0`（照 `obs_empty`）与 `m=160`（`ds[i]` 返回后改字典）——新库真实 `m ∈ [6, 141]`，0 与 160 只能合成（第三轮写「四个 spec 都按数据集真实交付挑」，按字面 `_fixture_batch` 必 `SystemExit`）。
- 判定行（全部新写或改口径）：`MASK_INVARIANCE=PASS` / `GRAD_LEAK=PASS` / `ORDER_EFFECT=PASS` / `ROW_PERM_INVARIANCE=PASS max_abs_diff=…`（代码里现只 `fails.append`、无 `=PASS/FAIL` 打印，须加）/ `PAD_CONTENT_INVARIANCE=PASS det_probes=3 nondeterministic_leaves=[] excluded=0 covered=<n>/<m>` / `LEN_EQUIV_NA`。「20 步内 4 叶参数值确有更新」这条**从 V5 移到 V8**（V5 只走 `jax.grad`、不落优化器；第三轮写了承诺没写产出者）。
- 耗时 / GPU / 留档：1 张 GPU（4），tmux `mv2-m4`；> 5 min 留档 `docs/training-doc/mv2-v5-gates/`。

**V8 开启态 A/A 可复现 + motion 叶确有更新**
- 证什么：同一侧同一命令跑两次 100 步逐位可复现（含四键的 batch 摘要），且 4 条 motion 叶 100 步内参数值确有变化。V8 是 A/A，不能支持「新旧等价」——开启态没有旧链路。
- 怎么证：V7 同一驱动 ×2；YAML `perceptual-framesamp-modul-8frame-8x8-motion.yaml`（三处白名单已放行）；`DATASET_PATH` 指新库 `4task-v2-1600ep-604f16da/framesamp-8x8`，`MMEVLA_MOTION_STORE` unset（表路径唯一来源是 YAML `store_path`）；GPU 4,5；`bench_train_steps.py` 本就写 `param_checksums.jsonl`。
- 判定行（实测，产出者同 V7）：`SCALARS steps=100 keys=5 hex_mismatch_steps=0` / `INDEX_SEQ=PASS n=<…>` / `STATE_DIGEST rows=5 mismatch=0` / `BATCH_DIGEST rows=7 mismatch=0` / `BATCH_DIGEST_CANONICAL rows=7 mismatch=0` / `CANON_CHECK=PASS steps=7`（第三轮漏了后三行——`_install_batch_digest_recorder` 的 `per_key` 对每键落哈希，开启态下正是四键的真实字节摘要）。新写汇总 `AA_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5`（字段集与 `GUARD_GRAD_100` 相同），汇总器另断言两侧 `batch_digests.jsonl` 里 `motion_emb` / `motion_pos` / `motion_mask` / `mem_order` 的 `per_key` 逐步相等且**均非 null**（关闭态四键为 null，同时反证 YAML 真开了 motion）；新写 `MOTION_PARAMS_UPDATED=PASS n=4 first_step=0 last_step=99`（从 `param_checksums.jsonl` 比 4 叶 sha 是否随步变化；`t3_smoke.py` 的同类判据写死 `_EXPECT_N_LEAVES=(193,177)` / `_EXPECT_N_KEYS=(16,12)` 是 32 帧 context 口径，不复用）。
- 耗时 / GPU / 留档：30–35 min × 2，tmux `mv2-aa`，留档 `docs/training-doc/mv2-v8-aa/`。

**V-online 在线 vs 表（方案 B「分层解耦」，用户拍板；三段）**
- 证什么：① 在线侧对**全部 71,316 窗**装配出的 33 帧与离线逐位一致（含 1,600 个补帧窗的补帧数与补帧源帧 `es−1`）；② **全 1,600 集**的起点集合 / `motion_pos` / `mem_order` 与 dataset 四键逐位一致；③ 真 encoder token 对**全部 1,600 个补帧窗 + 必含集**逐位一致。为什么这样拆：本轮唯一新增的机制是 demo 段 `min_real` 33→17 + `np.repeat` 补帧，风险落在「输入 33 帧是否装配正确」；encoder 的「同输入 ⇒ 同输出」已被反复确认（I.5 列了五条证据），换 ckpt 后只需小样本重取。第三轮的「按 episode 抽 10%」实算（seed=0、`% 10 == 0`）漏掉 k 最大的 g=367 与全部 24 个 `es ≥ 1000` 的集，且判定行写死 `episodes=160` 与取模规则不相容（恰中 160 概率约 3.3%）。
- 怎么证：`compare_online_motion.py` 三档，全部前置：a11 与 oracle 四道全量检查已 PASS（① 以离线 metadata 为真值）、E 节的 stub 帧号校验放宽已落地（②）。
  - ① `--assembly-hash $LIB/wan-latents`：在线 `_encode_window` 产出的 `np.ascontiguousarray(np.stack(frames))` 算 `hashlib.sha256(arr.tobytes())`，与离线 `wan-latents/*.metadata.json` 每行已有的 `input_frames_sha256`（同一 `wan_common.sha256_bytes` 口径；A3 保证补帧窗也对最终 33 帧算）逐窗比；同时断言「离线 `pad_frames>0` 的行集合」== 「在线按 es 独立推出的补帧窗集合」。**零 GPU**；h5 读 1,192,918 帧 + 71,316 次 6.3 MB sha，8 片（切分写死 `g % 8`，连续切块实测把挂钟拉到 2.08 倍）约 15–20 min。
  - ② `--stub`：sidecar stub 档（不加载模型、不 import torch、帧号写进像素解回），全 1,600 集，token 判据 SKIP、只验起点集合 / pos / order；零 GPU。
  - ③ 真 encoder：给 `motion_enc_fn` 外包一层路由——选中窗 → 真 sidecar，未选中 → 直接返回离线表行（`framesamp_memory.py` 的 `motion_enc_fn` 是 `__init__` 形参、`_prepare_motion` 只查 `f in self._history_feats_motion` 不关心来源），装配与次序检查不受影响（第三轮说「窗级抽样一秒也省不下来」只是现行工具把真 sidecar 直接当 `motion_enc_fn` 传进去的后果，不是链路性质）；选中集合 = 全部 1,600 个补帧窗（每集恰一个、由 `seg_len` 直接推）+ 必含整集（g=367；`es ≥ 1000` 的 24 集里 ≥ 3 条；16 个 `(es−17) mod 16` 档各 ≥ 1 条；四任务各 ≥ 1 条）。8 卡 `g % 8` 分片各起一个 sidecar，`--out v1-store/reports/motion/p5_online_v2_1600ep.shard<i>of8.json`（默认路径已有 40ep 历史结果），主进程显式 `export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`，**必须传 `encoder_run_dir`**（A9）；每片约 200 个补帧窗 × 1421.5 ms ≈ 284 s + 必含集，**8 卡约 7 min**。报告把 `compared_real` 与 `assembly_checked` 分列，注入行不计入 token 判据。
  - 比较器加固（第三轮五条保留 + 第四轮三条）：任何模式下已检测到的失配必须清零、`SKIP` 只准出现在打印不进 verdict；四处 `np.array_equal` 改先核 dtype / shape 再按 `uint8` 视图比字节（抄 `motion_gates_model._bytes_equal`）；两个反向用例（stub token 改错常数、`+0.0` 换 `-0.0`）确认非零退出；秒级 CPU 前置门（真 YAML → `get_history_config` → 真 `MME_VLA_Policy.__init__` 注入 stub `motion_enc_fn`，断言 `_motion_cfg` 键集合与值（含 `budget == 160`，即在线侧 budget 的落点），连跑两条 es 不同的集、中间真 `pol.reset()`，断言新 `mem_buffer` 是新对象、补帧窗数正确）；`unset MMEVLA_MOTION_STORE` 用真 YAML 解析路径断言 `realpath == $LIB/motion`。第四轮新增：**shard 输出加结构化字段** `rows_seen`（排序列表或其 sha256 + 计数）/ `start_ok` / `pos_ok` / `order_ok` / `n_compared` / `n_steps` / `padded_covered` / `mismatch_total`（不截断；现行只落 `lines` 文本与前 200 条 mismatches），`aggregate` 只消费这些字段、禁止解析 `lines`，任一字段缺失即 FAIL 不按 True 兜底；**期望侧独立**：窗数期望用打包器写进 `motion_index` 的 `SegmentInfo.num_grid` 求和（不从在线遍历现算，否则等号两侧同源），`padded_covered` 两侧各算一遍（在线按 es 推、离线取 `pad_frames>0` 的行）断言集合相等；**分片完整性** `SHARD_SET`。
- 判定行（全部新写或改口径）：① `ONLINE_INPUT_SHA=PASS windows=71316 padded=1600 mismatches=0`；② `ONLINE_START_SET=PASS` / `ONLINE_POS=PASS` / `ONLINE_ORDER=PASS`（stub 档、全 1,600 集）；③ 各片 `ONLINE_ENC_BITEXACT=PASS compared_real=<n> mismatches=0`、`PROVENANCE=PASS`、`YAML_STORE_PATH=PASS`、aggregate 后 `ONLINE_STRATA=PASS padded=1600/1600 kmax_ep=367 es_ge1000=<n≥3> pad_r=16/16 per_task_min=<n≥1>`、`SHARD_SET=PASS shards=8 eps=1600 disjoint=1`；汇总 `P5_ONLINE=PASS episodes=1600 windows=71316 assembly_checked=71316 compared_real=<n> padded_covered=1600 stub=False`。
- **承诺口径原话（原样写进留档）**：「在线侧 33 帧装配对全部 71,316 窗与离线逐位一致；起点集合 / 时间码 / 交错次序对全部 1,600 集逐位一致；真编码器 token 对全部 1,600 个补帧窗与 <N> 个整集共 <M> 窗逐位一致，其余行由离线表注入、不计入 token 判据。」
- 耗时 / GPU / 留档：①② 零 GPU 约 20 min（tmux `mv2-vonl-a` / `mv2-vonl-b`）；③ **独占 8 卡**约 7 min（tmux `mv2-vonl-c`，第九节步 6d 单独一档）；留档 `docs/training-doc/mv2-v-online/`。

**V9 8 卡 b128 20 步 smoke（开启态）**
- 证什么：开启态生产档位跑通且有限；参数树 61 → 65 叶精确匹配（多 `mem_encoder/motion_pos_proj/{kernel (256,768), bias (768,)}` + `mem_encoder/motion_encoder_static/{kernel (1536,1024), bias (1024,)}`）；真实 checkpoint → policy 加载走生产口径；budget 跨源一致；norm_stats 未变。
- 怎么证：基线 `records/smoke-runner.sh` 同法换 YAML / 配置名 `mme_vla_suite_b128_80k`，tmux `mv2-smoke`，run_name `smoke-m8x8-modul-motion-<UTC>`，**独占 GPU 0–7**（smoke runner 第 23 / 33 行卡号同改，见 G 节清单）；基线同档实测 5 分 18 秒是 4 卡数字、**8 卡耗时待测**，一律按第 17 条完整留档、验完删 run 产物。`check_config_provenance.py` **逐个显式传** `--ckpt <smoke run>/19 --lib v1-store/datasets/4task-v2-1600ep-604f16da --train-config mme_vla_suite_b128_80k --store-subdir framesamp-8x8 --neg-lib v1-store/datasets/4task-motion-400ep --norm-stats <新库 norm_stats.json>`（默认值全指旧库旧 run）；负例臂 `MOTION_STORE_PATH` 判的是报错内容含「绑定的清单不同」，依赖「旧 layout 的表仍能 load 到 `check_same_source` 那一步」——schema 不升版即满足。**新验收器必须新写**，不得沿用 `smoke-m8x8-modul-20260915T054007Z/records/finish_check.py`（硬编码 `motion_enabled is False`、`n_model=n_ckpt=61`、`PREFLIGHT=PASS n=25` / `== 25`、`mme_vla_suite_b128_60k`）。
- 判定行：`SMOKE20=PASS steps=20 finite=1 exit_code=0`（新写）/ `PREFLIGHT=PASS n=30`（实测）/ `PARAM_TREE_EXACT=PASS … n_model=65 n_ckpt=65 missing=0 extra=0 shape_mismatch=0`（实测）/ `MEM_PARAMS=PASS n=10`（6 modulation 叶 + 4 motion 叶；新写）/ `check_config_provenance.py` **全部五条**（实测；第三轮只列了一条）`NORM_STATS_SAME=PASS` / `LIB_PROVENANCE_MATCH … all_equal=1` / `MOTION_STORE_PATH … guard=raise infer_reads_store=0 same_manifest=1` / `CKPT_PARAM_TREE=PASS … leaves=65 ckpt_has_motion=1` / 观察行 `CKPT_DTYPE_PROFILE` / 汇总 `TIC_L0=PASS`（任一 FAIL 整脚本非零退出）/ `BUDGET_CONSISTENT=PASS snapshot=160 repo_yaml=160 expected=160`（新写、**跨源**：run 内 `history_config.resolved.yaml` vs 仓库新 YAML vs 160。第三轮写的「快照 == dataloader 实际值 == 在线 `_motion_cfg`」三侧在代码里同源自一个 DictConfig、恒等不可能 FAIL，且 smoke 不构造 policy；在线侧 budget 归 V-online 的 CPU 前置门）/ `NORM_STATS=PASS sha256=856c75ea…`（新写）。
- 耗时 / GPU / 留档：耗时待测（4 卡同档 5 分 18 秒）+ 验收，**独占 GPU 0–7**，留档 `docs/training-doc/smoke-m8x8-modul-motion-<UTC>/`。

**V10 起跑 preflight**
- `preflight_train_launch.py`（runner 内，与 train 共用同一 `TRAIN_ARGS`）；`n=25` 的前提是 runner 传 `--run-root`。新增 5 项 motion 检查——**期望值全部由 runner 显式传参**（preflight 只 import 标准库、无 YAML 解析器，第三轮没写期望值从哪来）：`--motion-store` / `--motion-store-meta-sha256` / `--motion-layout`（== `motion-768-grid16-demopad17-v1`）/ `--motion-rows`（== 71316，值来自建库留档、**不从被测 store 现算**，否则等号两侧同源）+ `CHECK_MOTION_ENV_UNSET`（无参，`MMEVLA_MOTION_STORE` 为 unset）。由此 runner 与基线的差异从 10 处变 **11 处**（第三节；第五轮新增卡数 / mesh、`num_workers`、GPU 可见与采样范围三行后，非 preflight 部分为 10 处）。建议再加 `CHECK_MOTION_SOURCE_RUN`（provenance `run_name` 与 YAML `source_run` 一致），把 A6 那条「起跑必 raise」提前到 preflight。
- 判定行 `PREFLIGHT=PASS n=30`（实测口径，5 项无条件计入）。

##### E. 秒级 CPU 单测（第九节 2b，全部新写）

`GUARD_LR=PASS steps=80000 mismatches=0`（`config.lr_schedule.create()` 在 `range(0, 80_000)` 逐步比 60k 与 80k 两条配置）；`INIT_COMMON=PASS common_mismatches=0 open_only=4`（I.2）；`ROPE_LEN_EFFECT=PASS tv_144_160=… relL2_144_160=… tv_512_672=… pad_content_diff=0`（I.2）；A8 离线边界用例表驱动全部 10 处独立实现（`test_guards.py` 新旧两组 + `for mr in (17, 33)`，负例含「旧 layout 带新键 / 新 layout 缺新键 / 三方不同值」）；`EVAL_ES_BOUND=PASS tasks=4 episodes=<n> es_max=<m> budget=160 headroom=<…>`（第七节，起跑前）。

##### F. 跑完后

motion 全遮消融评估（I.2，用户拍板加入）+ 基线 `59999` 评估（第三节末交接约束，下一轮）。

##### 归因唯一化与两块划分

**归因**：V1 排除交付层、V6 排除单步数值、V7 排除 100 步累积——三项同过的唯一解释是关闭态语义未变；V2 排除旧物被改坏；V3 排除新表本身错；V4 + V-online ① 排除交付与装配错；V5 排除接线错；V8 排除不可复现与 motion 叶不更新。**本轮承诺的是实现等价与可复现，不是因果归因**；「motion 有没有用」由消融臂回答。

**`AGENTS.md` 第 18 条两块划分（第四轮重划）**：第一块（非训练轻量对拍）= V1 / V2 / V4 / V-online；第二块（本机训练梯度一致）= **只有 V7**；V6 单列「单步数值对拍（固定夹具、不启动训练）」；V8 是 A/A 可复现性基线，开启态没有旧链路，**第 18 条第二块对开启态不适用**，本轮以 V8 + V9 + V-online 替代并如实登记为盲区；V0 / V3 / V5 / V9 / V10 不属两块（分别是配置守卫 / 建库对拍 / 模型接线 / 冒烟 / 起跑闸）。**第二块不通过不得宣称改动等价。** 两张链路图（外加第四轮补的在线图）见 H 节。

**判定行标注约定**：F 表每条判定行标「实测」或「新写」。在 `AUDIT_BASE=530e841` 下 grep 零命中、本轮新定义的判定行名：`GUARD_LR`、`INIT_COMMON`、`ROPE_LEN_EFFECT`、`EVAL_ES_BOUND`、`INDEX_TRAIN`、`GUARD_GRAD_100`、`AA_100`、`MOTION_PARAMS_UPDATED`、`LEGACY_LAYOUT` / `LEGACY_YAML` / `LEGACY_POLICY`、`V4_COVER`、`PAD_CONTENT_INVARIANCE`、`LEN_EQUIV_NA`、`ONLINE_INPUT_SHA`、`ONLINE_STRATA`、`SHARD_SET`、`YAML_STORE_PATH`、`A11_PAD`、`A6_SAMESOURCE`、`INPUT_REANCHOR`、`BUDGET_CONSISTENT`、`SMOKE20`、`MEM_PARAMS`、`NORM_STATS`、5 条 `CHECK_MOTION_*`、`GPU_IDLE`。第三轮那句「每一行都已与工具实际输出对表两轮」**只对标「实测」的行成立**。


#### I.8 执行顺序步级细节（原第九节全文，含 tmux 会话名清单与清理纪律）

主副本上顺序执行，每个 commit 后立即 push。建库 8 卡全开；**V-online ③、smoke 与正式 run 都独占 8 卡（GPU 0–7）**。固定卡号的几项与生产卡数无关、保持原样（分类按各自形制，不统称「二卡对拍」）：**V6 / V7 关闭态 2 卡 b8 对拍（GPU 4,5）、V8 开启态 A/A 2 卡（GPU 4,5）、V1 CPU 交付验证（无 GPU）、V5 1 卡模型档（GPU 4）**。

**本轮 tmux 会话名清单**（第 7 条要求先记清单、清理时只按全名逐个 kill；两两之间无前缀关系）：`mv2-wan`、`mv2-encode`、`mv2-pack`、`mv2-oracle`、`mv2-v1`、`mv2-v2`、`mv2-v4`、`mv2-vonl-a`、`mv2-vonl-b`、`mv2-vonl-c`、`mv2-bench-base`、`mv2-bench-cand`、`mv2-aa`、`mv2-m4`、`mv2-smoke`、`mv2-prod`、`mv2-dense`。**清理命令统一写 `tmux kill-session -t =<全名>`**（`=` 强制精确匹配）。第三轮说「`tmux -t` 做前缀匹配，`mv2-prod` 会误杀 `mv2-prod-dense`」把顺序说反了：tmux 先精确名再前缀，**目标会话已结束时**前缀匹配才会命中同前缀的其他会话——所以仍不叫 `mv2-prod-dense`，且一律用 `=`。

| 步 | 做什么 | commit |
|---|---|---|
| 0 | 用户确认 `run_name`（V5 模型档已拍板用乙）；`df -h /scratch` ≥ 400 G 才起跑 | — |
| 1 | 本文入库 | `docs:` |
| **1b** | **只改验证工具白名单、不触生产链路**（第四轮新增，解 2a 与 2b 的互斥）：`dump_fixture_samples.py` 与 `bench_train_steps.py` 的 `_EXPECTED_HISTORY_CONFIGS` 各加 `perceptual-framesamp-modul-8frame-8x8.yaml` 与 `perceptual-framesamp-modul-8frame-8x8-motion.yaml`；`run_2gpu_epoch_bench.sh` 的 `case "${HISTORY_CONFIG}"` 白名单同加；`ref_npy_dataset.py` 形制闸放开 `integration_type in ("context", "modulation")`。commit + push 后 **`BASE=$(git rev-parse HEAD)`（不写死 sha）**。第三轮把 2a 写成「clean HEAD 跑 V1+V6+V7」物理上不成立：白名单不改 V1/V7 起手 raise，改了 V6 的 porcelain 清洁闸又不过 | `fix:`（取证工具白名单） |
| **2a** | **在 1b 之后的 clean BASE 取「改前」证**，`DTYPE_SOURCE_COMMIT=$BASE`：V1 dump、V2② dump（`perceptual-framesamp-context-motion.yaml` + 40ep 库）、V6 定点梯度、V7（`dump` + `manifest` + bench + `project_scalars`）；卡号钉 `CUDA_VISIBLE_DEVICES=4,5`；记 `git rev-parse HEAD` 与 porcelain 入留档；产物落 `v1-store/bench/2gpu-epoch-bench/<RUN_TAG>-base/`（V7）与各自 `DTYPE_*_DIR`。留档目录 `docs/training-doc/mv2-v1-dump/`、`mv2-v2-legacy/`、`mv2-v6-grad/`、`mv2-v7-guard-base/`；V7 真起训练进程，run 名记入清单、**先把 metrics / digests / env.json 拷入 `records/` 再删 run 产物**（第 6 / 17 条） | — |
| **2b** | 实施第二部分 A–E 全部生产代码 + F 节全部验证工具（`legacy_regress.py`、`hand_calc_8frame.py` 参数化、`motion_gates_model.py` 六层适配、`compare_online_motion.py` 三档与 aggregate、`finish_check.py`、preflight 5 项、`eval_rhythm_gates.py` 区间与 `--yaml`）+ CPU 单测（`GUARD_LR` / `INIT_COMMON` / `ROPE_LEN_EFFECT` / A8 边界表 / `test_guards.py` 新旧两组）；此阶段工作区 dirty，**不跑任何带 clean-source 校验的工具**；**禁止 `uv add`**（V6 比较器比 `uv_lock_sha256`） | — |
| **2c** | **两个 commit**（第四轮拆分，让回滚只碰生产改动）：`commitV10.0: demo 段补帧网格、modulation 接入 motion 与 80k 配置`（A–E 生产链路）→ `commitV10.1: 第四轮验证工具与 CPU 单测`（F 节工具）→ push；记 `CAND=$(git rev-parse HEAD)`；**`git status --porcelain` 必须为空才能进 2d** | `commitV10.0` + `commitV10.1` |
| **2d** | 取「改后」证并对拍（`DTYPE_SOURCE_COMMIT=$CAND`，卡号仍 4,5）：V1 / V2② compare、V6 compare、V7 `check --base` + bench + `project_scalars` + `compare_baseline`（tmux `mv2-bench-cand`）、V2① / V2③ 秒级。对拍前 `jq` 断言两侧源码身份（I.7 开头）。**FAIL 分流**：harness 失配（`reason=运行口径不同…`、目录已存在、白名单 raise）先排查；数值项失配 → `git revert --no-commit <commitV10.0 sha>` + 手写 subject `revert: 撤销 commitV10.0（2d 对拍 <判定行> FAIL）` + push（裸 `git revert` 产出英文无前缀 commit，违反第 11 条；**只 revert V10.0、保留 V10.1 工具**），停下交用户。第二轮规则：`DTYPE_SOURCE_COMMIT` 取 revert 后新 commit、产物目录 / `RUN_TAG` 加 `-r2`；基线证在 `uv.lock` / `packages` / `CUDA_VISIBLE_DEVICES` / GPU 列表不变时可跨轮复用（`compare_fixed_grad.py` 不比 `source["head"]`，但 `environment` 任一变化即 raise） | — |
| 3 | encoder 换型落盘 + `ASSETS_LOCK.json` 更新 + `test_assets_lock.py` 通过 + 输入重锚（`INPUT_REANCHOR=PASS files=4`） | `fix:`（资产锚点） |
| 4 | 建库顺序（tmux `mv2-wan` → `mv2-encode` → `mv2-oracle` 重算及汇总 → `mv2-pack` 打包及数值 compare；**Wan 与 encode 之间零 commit**）+ 账目五项；`--raw-dir` 三方断言在 `run_local.py` / `oracle_driver vae` **读完清单后立即做**，pack 再核三方（A6，不等 5 h GPU 跑完） | — |
| 5 | 建库留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/` | `docs:` |
| **6** | 开启态验证，**串行子步、标卡号、每档结束 `tmux ls` + `nvidia-smi` 确认前一档进程已退**（第四轮 P0：第三轮五项并列在一格、V-online 要 8 卡与表头「用 4–7」冲突）：**6a** V4（CPU，tmux `mv2-v4`）‖ V5（GPU 4，tmux `mv2-m4`）→ **6b** V8（GPU 4,5，tmux `mv2-aa`）→ **6c** V-online ①②（零 GPU，tmux `mv2-vonl-a` / `mv2-vonl-b`）→ **6d** V-online ③（**独占 GPU 0–7**，tmux `mv2-vonl-c`，约 7 min）→ **6e** smoke（**独占 GPU 0–7**，smoke runner 卡号同改，tmux `mv2-smoke`，run_name `smoke-m8x8-modul-motion-<UTC>`，一律完整留档）。第五轮起表头已统一为 8 卡，上面括注里「V-online 要 8 卡与表头『用 4–7』冲突」的那处历史冲突随之消失。本步所有工具改动已在 2c 入库 | `docs:` 守卫留档 |
| 7 | 起跑留档 `docs/training-doc/<run_name>/launch.md`（沿用「起跑前留档不嵌入自身提交 SHA」）；此刻记 `TRAIN_HEAD` 写进 launch.md 正文；同步做 `EVAL_ES_BOUND`（单卡渲染，第七节） | `docs:` |
| 8 | 生成 runner 落 **`v1-store/logs/mv2-prod-runner.sh`（不进 git）**：照抄基线 runner + `launch.actual.json` 的 `env` 前缀，只改第三节那 **11** 处（其中卡号相关 6 处的逐行清单见 G 节；`fsdp_devices` / `num_workers` 不在 runner 里覆盖、由新具名配置承载） → **独立占卡闸**（不依赖 runner 内的 `test`——两个基线 runner 的 `body()` 内没有 `set -e`，第 33 行 `test "$(nvidia-smi …)" = "0"` 失败后照样往下走）：`nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits` 求和为 0 且 `tmux ls` 中本轮会话只剩待起的 `mv2-prod`，判定行 `GPU_IDLE=PASS gpus=0,1,2,3,4,5,6,7 used_mib=0 sessions=<清单>` 写进 launch 留档 → tmux `mv2-prod` → preflight `PREFLIGHT=PASS n=30` → 训练；Monitor 盯日志；**300 步处按第 16 条口径重估 ETA 并回报**，分记 dataloader 等待占步时比例与 step compute | — |
| **8b** | **起跑后归档**（第四轮补，第 12 条要求的 commit / 命令 / 配置 / 输出路径此前只留在终端与 v1-store）：`records/launch.actual.json`（`train_head` / `launch_command` / `tmux_sessions` / `train_pid`）+ `records/preflight.log` + `records/prod-runner.sh` 副本 | `docs:` |
| 9 | 训完：motion 全遮消融评估 + 基线 `59999` 评估（评估口径另起计划，第三节末交接约束） | 另起计划 |


#### I.9 步时三因与 8 卡口径

**步时必须重测，21.6 h 只是下界**：8 卡关闭态实测 **0.972660 s/步**（`docs/training-doc/bench-collate-shm-8gpu/result.md`，b128 / w16 / mesh (1,8)，稳态窗口 step 100→290 共 190 步），80,000 × 0.972660 s = **21.6147 h**。对照出处：4 卡基线 2.34033 s/步（`result.md` 的 step 100→59900 实测）= **52.0 h**。**两个百分比不得混用**——历史 8 卡 1.818 s 相对 4 卡 2.340 s 是 **−22.3%**（`bench-m8x8-8gpu-worker`，共享内存改动之前），0917 对照的 old 侧 1.772729 s 相对 4 卡是 **−24.25%**；而 52.0 h → 21.6 h 的 2.41 倍是**相对历史 4 卡配置的综合改善**，同时含共享内存交付、`num_workers` 8→16 与卡数 4→8 三项，不可归因于任何单一因素。

**三个增量的排序本轮降级为待验证假设**（第四轮那份排序是 4 卡数据受限档下的结论，档位已变）：collate 改走共享内存后数据路负担被大幅压低（CPU 导出口径下主进程每批等待 0.783745 s → 0.002647 s；**该数字逐批摘要会与预取重叠，不外推训练吞吐**），因此**预期**主要增量转到 `MemoryAttention` 在 18 层里每层对整条 mem_seq 做的 K/V 投影（512→672 即 +31%，remat 策略 `nothing_saveable` 反向还要重算），其次才是数据路的 84 MB/batch 与 16 worker 常驻 3.5 GB，最后是每样本一次 672 长稳定排序（numpy，微秒级）。**这只是假设，不是结论**。

**为什么不能用 util 下结论**：NVML 的 `utilization.gpu` 度量的是「该采样周期内有 kernel 在执行的时间占比」，它不区分计算、通信与等待，**单独不能定位瓶颈归属**；0917 那个 97.87% 又只是 **motion 关闭态**的数字，开启态没有测过。所以执行顺序表的硬节点改为：推进到 300 步后按 `AGENTS.md` 第 16 条口径（稳态窗口均值 + 0% 采样占比 + 慢/非慢分层，**禁用中位数**）**联合三项**裁决并回报——① 步时稳态均值与吞吐；② dataloader 等待占步时比例；③ step compute 分解。三项指向数据路才调 `num_workers`（属第 10 条超参，须另行确认落点），指向模型侧则属本节预期的增量、不必调 worker。第四轮写的「步时若上升先看数据路」与「上升 > 15% 先评估 `num_workers`」两条按这三项分解替换。

**为什么改用 8 卡、代价是什么**：用户 09-17 拍板。8 卡的性能前提是起跑源码含 `3868cc9`（`commitV9.10`）的 collate 共享内存交付——没有它，8 卡相对 4 卡只剩约 22% 的改善（`bench-m8x8-8gpu-worker` 实测 1.818 s，util 均值 51.6%、0% 采样占比 43.5%），当初钉死 4 卡正是因为这个。代价是与基线不再逐位可比：`fsdp_devices` 4→8 让 per-device batch 从 32 变 16、mesh 从 (1,4) 变 (1,8)。

能断言的只有第三节那条**限定命题**——固定模型定义、输入、初始参数与随机流后，换分片不改变 global batch 128 所对应的优化目标（loss 为全局均值、无 batch 依赖算子、全局范数裁剪、RNG 与 EMA 与卡数无关、索引序列与 `fsdp_devices`/`num_workers` 无关）。**不能**把它说成「差异纯粹来自归约次序」：换分片同时可能改变归约次序、kernel 选择、算子融合与编译重排，正确表述是**可能产生并累积浮点差异、不承诺跨拓扑逐位一致**；也**不断言**这些差异必然放大到曲线可见。可操作的结论只有一条：**新 run 与基线不能再用逐位判据比较，只能做统计比较**。该命题**不覆盖**新 run 与基线的整体关系——新 run 开了 motion，模型函数与参数空间本身就变了。

用户同时拍板**不重跑 8 卡基线**（60k × 0.97266 s ≈ 16.2 h），主比较锚点仍是 80k run 的 `60000` vs 基线 `59999`，比较降级为统计比较。这不改变本轮的承诺范围：第八节承诺的本来就是**实现等价与可复现**，而不是逐位复现基线；关闭态逐位对拍 V1 / V6 / V7 跑在固定夹具档上（V1 CPU、V6/V7 2 卡 b8），与生产卡数无关，一项都不受影响。
