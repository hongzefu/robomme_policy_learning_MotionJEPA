# RouteStick 进度探针与注意力图：结果

**一句话结论**：在 200 局新生成的 RouteStick 专家局（benchmark `b4e97f2`，两个模型都没见过）上，两个四任务 modulation 模型的动作 token 都能读出「下一步往哪走、怎么绕」，明显超过时间基线（A +0.241、B +0.210）。读得最清楚的是动作 token 每层从记忆里读到的向量（记忆调制组）：A 0.668，B 0.552。进度比例 τ 则不能超出时间：A 动作组 0.539，低于只看时间的 0.706。注意力上，A 执行第 j 步时读示范第 j 段的 motion token 只有轻微的对角偏好（D = 1.10 [1.09, 1.11]），低于预注册的 1.2；B 没有（1.00）。

起跑与运行细节见 [launch.md](launch.md)，图与汇总 JSON 在 `records/`。

## 判定行

| 判定 | 原文 |
|---|---|
| 生成 | `GEN_SPEC=PASS eps=200 mismatches=0 dropped=0 irregular=0 missing=0`；`DATA_SYNC=PASS files=4 mismatches=0` |
| 前缀看不到记忆 | `MEM_BLIND=PASS prefix_maxabs=0 action_maxabs=73.8765`（A），`… action_maxabs=63.8488`（B） |
| 捕获路径 = 原生采样 | `REPLAY_PARITY=PASS points=7 maxabs=1.78813934e-07 tol=0.001`（A），`maxabs=2.38418579e-07`（B） |
| 喂帧不变性 | `HIDDEN_PRECISION=PASS` 六组，`path_maxabs=0`（A、B） |
| 回放完整 | `REPLAY_SET=PASS model=A episodes=200/200 points=3421 expected=3421`，B 同 |
| 注意力聚合 | `ATTN_AGG=PASS rowsum_maxerr=1.27e-07 memmod_maxabs=0.00e+00 min_cell_eps=17` |
| 无同局泄漏 | `GROUP_LEAK=PASS leak=0` |
| 正对照（当前这一步走法，动作组 L17） | `POSITIVE_CONTROL=PASS model=A layer=17 acc=0.939`，`model=B acc=0.947` |
| 主读数（PASS 只表示流程完整） | `PROBE=PASS model=A label=tau group=action value=0.539 minus_u=-0.167`；`PROBE=PASS model=A label=next group=action value=0.498 minus_u=0.241` |
| 严格档配对 | 0 对，见「与计划不符」 |

## 探针（训练折内选层，按局 bootstrap 2000 次）

时间基线：执行偏移 u 独热。τ R² 0.706，下一步走法 0.257，剩余步数 R² 0.119。`[u, L]` oracle：τ 1.000，剩余步数 1.000，下一步走法 0.220。

| 组 | 模型 | τ（R²）−[u] | 下一步走法（准确率）−[u] | 剩余步数（R²）−[u] |
|---|---|---|---|---|
| 前缀（图像+文字） | A | 0.436，−0.271 | 0.382，+0.126 [0.081, 0.174] | 0.034，−0.086 |
| | B | 0.425，−0.282 | 0.382，+0.125 | 0.013，−0.107 |
| 动作 token | A | **0.539，−0.167** [−0.190, −0.144] | **0.498，+0.241** [0.198, 0.285] | 0.230，+0.111 |
| | B | 0.488，−0.218 | 0.467，+0.210 [0.169, 0.253] | 0.198，+0.078 |
| 记忆调制 | A | 0.549，−0.157 | 0.668，+0.412 [0.367, 0.458] | 0.417，+0.298 |
| | B | 0.840，+0.134 [0.111, 0.158] | 0.552，+0.296 [0.248, 0.346] | 0.688，+0.568 |
| 输入级 motion 均值 | A | 0.447，−0.259 | 0.498，+0.241 | 0.429，+0.309 |
| | B | 0.926，+0.220 | 0.440，+0.184 | 0.775，+0.656 |
| 输入级最近 motion | A | 0.417，−0.289 | 0.371，+0.114 | 0.115，−0.004 |
| | B | 0.589，−0.118 | 0.337，+0.080 | 0.516，+0.397 |
| 输入级帧均值 | A | 0.180，−0.526 | 0.358，+0.101 | 0.085，−0.034 |
| | B | 0.191，−0.515 | 0.355，+0.098 | 0.077，−0.042 |

下一步走法按难度分层（准确率，括号内为 [u] 基线）：

| 组 | easy | medium | hard | xhard |
|---|---|---|---|---|
| A 记忆调制 | 0.817 (0.257) | 0.734 (0.264) | 0.633 (0.264) | 0.488 (0.242) |
| B 记忆调制 | 0.779 | 0.621 | 0.439 | 0.370 |
| A 动作 | 0.540 | 0.567 | 0.476 | 0.407 |
| B 动作 | 0.547 | 0.516 | 0.416 | 0.389 |

- **记忆里有路线，动作 token 读得到**：前缀组看不到记忆（`MEM_BLIND`），它读下一步只有 0.382，主要来自当前画面里手臂的位置和朝向；动作组和记忆调制组都明显更高。A 比 B 高的主要在 hard/xhard（记忆调制 0.633/0.488 对 0.439/0.370）。
- **τ 读不出超过时间的部分**：同一难度内 L 的变化不大，执行时间本身已决定了大部分进度（[u] 0.706）。A 的动作组与记忆调制组都低于它。
- **B 的输入级 motion 均值读 τ 达 0.926、读剩余步数 0.775，原因未核实**：一个可能的解释是，记忆里示范段有 50L 帧、执行段有 u 帧，motion 窗口中执行窗口所占的比例近似 τ；Frozen Wan 的池化 token 若能区分示范窗与执行窗（例如画面里有没有手臂），均值就能线性读出这个比例。这一项不能当作「编码器理解进度」，写论文时要么排除，要么先做「只用示范段窗口」的对照。B 记忆调制组的 τ 高值可能是同一来源。
- 层：A 的动作组与记忆调制组在训练折内一致选 L12–L13；B 记忆调制读 τ 选 L6、读下一步选 L9。

## 注意力图

`records/figs/attn-a-routestick.png`（模型 A）、`attn-a-routestick-B.png`（模型 B，附录）、`attn-b-routestick.png`。

- **(a) 块①**：A 的动作 token 在 L0–L6 偏读帧（action←motion 低于均分），L11–L16 转为偏读 motion（最高在 L14）；B 从 L0 起就偏读 motion（份额÷均分全程大于 1）。
- **(a) 块②**：记忆对 FiLM 的相对改变量，A 在 L9 有一个 370% 的离群格（已截断并在角标标出），其余集中在 L11–L16（数十到 200%）；B 只在 L12–L14 有 20–30%。也就是说，A 的记忆在后半段层对动作专家的影响远大于 B。motion share 行是 ‖Σmotion αV‖÷‖Σall αV‖，是向量模长之比，不是严格的占比，可以超过 100%（B L7 为 143%）。
- **(b)**：A 读 motion 有轻微对角（D = 1.102 [1.094, 1.108]），第 1 列（示范第 1 段）在所有行都偏高；读帧没有对角（0.977）。B 读 motion D = 1.002，读帧 0.968，帧流明显偏读示范第 1 段。与原论文 context 版的结论方向一致：注意力读哪一段并不随当前步数调整，只有 A 的 motion 流有很弱的对角。
- 原图的 frames←motion、lang←motion、lang←frames 三行在 modulation 下按结构为零，不画。

## 新版 Fig. 4：只画下一步走法（11-17 追加，`8645958`）

用户：「如果只看这个下一步往哪儿走的话，怎么画图呢」「用 Fable 重构一下，就是我要体现出它更强」。τ 读不出超出时间的信息，新 Fig. 4 只画下一步走法（4 类，随机 0.25，时间基线 0.257）。脚本和图在 [fig4-next/](fig4-next/README.md)，复现命令：`uv run --no-project --with matplotlib --with numpy python docs/evaluation-doc/rs-progress-probe-1117/fig4-next/plot_fig4_next.py`（只读 `records/rsprog-probe-summary.json`）。判定行 `FIG4_NEXT=PASS a_sel=0.668 b_sel=0.552 diffs=+0.112,+0.195,+0.118`，`PRE_MERGE_REVIEW=PASS findings=2`（两条措辞问题已在 README 修正）。

- (a) 逐层：A、B 叠在同一坐标轴。A 的记忆读出在 L12–L13 达到峰值（L13 为 0.668，B 同层 0.455），其余层大多与 B 重叠，L9 是 B 高（0.552 对 0.418）。动作 token 在 18 层上 A 全部高于 B，差约 0.01–0.05。
- (b) 分难度的记忆读出（各自训练折内选层：A L13，B L9）：easy +0.04（CI 重叠），medium +0.11、hard +0.19、xhard +0.12（CI 都不重叠）。差值不随难度单调变化。
- 图注建议：「中等以上难度 A 明显更强」，同时如实说明记忆读出的优势集中在中后层、动作 token 每层都更高。不写「越难越大」，也不写 p 值。

注意力 (b) 对角线的现状：A 读 motion D = 1.10，未达预设门槛 1.2；最显眼的是第 1 列整列偏高。A 读帧 0.98，B 读 motion 1.00、读帧 0.97，都没有对角线。可以写成「注意力不按当前步逐段对齐，更像整体读取示范」，不能写成「出现对角线」。

## 归档与跨机器接续

论文所需的图和汇总已全部进 git（本目录）：

| 图 / 数据 | 路径 |
|---|---|
| 新版 Fig. 4 | `fig4-next/fig4-next.{png,pdf}`，脚本 `fig4-next/plot_fig4_next.py` |
| 注意力 (a)：A 主图、B 附录 | `records/figs/attn-a-routestick{,-B}.{png,pdf}` |
| 注意力 (b)：对角线矩阵 | `records/figs/attn-b-routestick.{png,pdf}` |
| 旧版 τ 图（不再用作主图） | `records/figs/progress-{fig4,layerwise,fit}.{png,pdf}` |
| 汇总 | `records/rsprog-probe-summary.json`、`records/rsprog-attn-summary.json`、`records/rsprog_points.json`（取点，sha256 前 16 位 `65a357c75eb740c9`）、`records/pool.json` |

其他机器上 clone 本分支即可重画图 4，不需要大文件。重跑探针或注意力分析则需要以下不在 git 里的产物，都在 sled-aspen 上：
- 回放 npz：`/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/rs-progress-probe-1117/replay-all/`（3.3 G）。
- 数据 H5：`v1-store/datasets/rs-progress-1117/`（66 G）。另有一份副本在 turbo `…MotionJEPA-eval/v1-store/datasets/rs-progress-1117`，`DATA_SYNC=PASS`。
- 需要时用 `rsync` 从 aspen 拉取。

## 与计划不符的地方（如实记录）

1. **严格档配对为空**：「同一按钮、两视角逐像素差 ≤8」的候选配对 566 对，全部超阈（前视最大差中位数 222.5、最小 170），所以严格档 0 对。原因是 RouteStick 里同一按钮可能从左或右到达，手臂与棍子姿态不同，不会像 StopCube 那样出现画面相同的帧。阈值没有放宽，严格档记为不可用。
2. **τ 主读数不超时间基线**：计划预先写死的主读数 1 结论为否；如实报告，不换口径。
3. 打乱对照为同难度内错位换路线（审查后 R3 实现），比计划写的「整局换路线」更保守。
4. `HIDDEN_PRECISION` 实际量的是 16 帧与 8 帧喂法的不变性；`MEM_BLIND` 的前缀差为 0 由适配器结构保证，独立证据是 `REPLAY_PARITY` 与源码守卫 `_guard_modulation_source`。
5. film_rel 的基线是 F(x, 0)（cond 置零时 MemoryRMSNorm 只剩偏置），不是计划 §6 写的 RMSNorm(x)；`cond=None` 分支用的是另一个 scale 参数，生产路径从不走。

## 实施中调整（不改语义的工程处置）

| 旧 | 新 | 理由 |
|---|---|---|
| 生成单组 spec 加 `--difficulty <d>` | 加清单 manifest | 生成脚本 `--difficulty` 只接受 easy/medium/hard 比例串，xhard 只能经清单 |
| 用 `official-sim` 环境生成 | 按 `b4e97f2` 锁文件新建 `v1-store/envs/bench-b4e97f2` | 旧环境缺 `pebble`；GL 上同样建环境失败（缺 Python.h），因单局 hard 只要 37.6 s，生成全部改在 aspen |
| 生成输出在子模块里 | benchmark `git archive` 副本里跑 | 脚本要求输出目录在 benchmark 仓库根内，避免弄脏子模块 |
| 冒烟显存比例 0.40 | A 0.75、B 0.62（aspen），GL 0.80 | bf16 policy 加 f32 模型 0.40 放不下；B 在 0.70 时 sidecar 无显存 |
| 默认 provenance 严格比对 | `MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count,driver` | 离线库在 A100 上产，本机 A6000/A40；沿用 steer-4task-1115 aspen spec 的同一放行口径，编码器权重 sha 照常核对 |
| aspen 两片 + GL 四片 | 用户「所有的gpu任务全部改到gl上跑」后 aspen 两片停掉，剩余局接在 GL | aspen 被其他用户压满，单局慢 3–4 倍 |
| 剩余局列表含已完成的 A k=0,1,2 | 去掉已存在的 npz | 指纹含 checkpoint 路径字符串，aspen 与 GL 路径不同而拒绝覆盖；两处权重同一份 SHA256SUMS，npz 有效 |
| R2 motion `mem_t1` 截到示范末 | 存未截断 t0+32 | 合并前审查发现截断会降低跨示范尾窗口的归段门槛，与「≥22/33」不符 |
| (a) 块①色标、块②共用色标 | 稀疏刻度；块②拆两行各自色标并截断离群格 | 真实数据出图后目视发现刻度重叠、两行单位不同 |

## 盲区

- 没有做 motion 通道消融或干预，结论只到「可解码」；与 11-15 steering（换段 token 后 85–89% 跟随）合看才有因果含义。
- B 输入级 motion 均值读 τ 的高值，可能是示范/执行窗口比例的混淆，未核实。
- 专家轨迹不是策略自己的轨迹；原论文注意力图用的是 50 局真实 rollout。
- 每模型只有一个 checkpoint（step 50000），A、B 训练配方不同，跨模型比较只作描述。

## 下一步（需另给开工令）

- 「只用示范段窗口」的输入级对照，排除 τ 的比例混淆。
- 把新版 Fig. 4 与注意力 (a)(b) 放进论文 `5_interpretability.tex`，图注按本文件口径改写。
