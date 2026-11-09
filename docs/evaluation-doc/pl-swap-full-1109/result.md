# PatternLock 全因子单步 motion 交换结果（pl-swap-full-1109）

## ① 一句话结论与指标速览

MotionMem 的示范窗口 motion token 承载了「这一步往哪个方向走」：去掉或换成同范数随机向量后，原样成功的接收局只剩约 3 成能成功；同起点换成别的方向时，150 局里有 136 局按供体方向走；起点和方向都不同时，按「从自己起点走供体方向」到达的格（147/378）明显多于「走到供体的绝对目标格」（61/378）。Frozen Wan pooling 则几乎不看这个 token：置零、随机、B/C/D 交换后 17 个原样成功接收局全部仍走自己的目标（交换组 0 局跟供体），逐步动作与原样局的差异中位数只有 MotionMem 的约 1/50。

最终判定行（原文，见 `records/motion/readout.md`、`records/wan/readout.md`、`records/intersect.md`）：

- `PL_SWAP_FULL=PASS model=motion runs=876 invalid=0 N=74/80 Z=10/37 Rnd=11/37 A=37/37 B_donor=136/150 C_own=86/157 D_dir=147/378 D_abs=61/378`
- `PL_SWAP_FULL=PASS model=wan runs=427 invalid=0 N=34/80 Z=17/17 Rnd=17/17 A=17/17 B_donor=0/57 C_own=71/71 D_dir=0/168 D_abs=0/168`
- `PL_SWAP_INTERSECT models=motion,wan recv=17 motion:Z=5/17,Rnd=7/17,A=17/17,B_donor=51/57,C_own=44/71,D_dir=59/168,D_abs=22/168 wan:Z=17/17,Rnd=17/17,A=17/17,B_donor=0/57,C_own=71/71,D_dir=0/168,D_abs=0/168`

判定行 PASS 只表示数据完整、无效局 0；各类占比是读数，不是验收项（计划第四节）。

## ② 闸门与验收一览

| 判定 | 结果 |
|---|---|
| 开工审查 | `PLAN_CONSISTENCY_REVIEW=FAIL base=1ed8838 findings=11 blocking=5`，按用户「尽可能不要问我自己决策」「一路做到底」由主代理裁决（计划第二部分〇′，`ff5a48e`） |
| 实施中调整 | `ADJUST_REVIEW=PASS scale=smaller findings=2`：motion 复用本地 `vla-o16cc-59999`（即主表 orig100 concat） |
| 合并 | W3 `PRE_MERGE_REVIEW=PASS … findings=4`；W1 `PRE_MERGE_REVIEW=PASS … files=440 commits=181 findings=3`；W2 `PRE_MERGE_REVIEW=PASS … files=6 findings=2`。三次合并后短测仅既有失败 `test_rollout_hidden::test_hidden_values_match_independent_rollout`（全目录顺序下数值偏差，BASE 同样失败） |
| 资产 | `ASSETS=PASS model=motion files=38 bytes=11976241540 mismatches=0`；`VLA_DOWNLOAD=PASS step=59999 files=35 bytes=11620193417`（wan，逐文件 sha256） |
| 局池 | `PL_POOL=PASS eps=80 combos=40 overlap=0`；`PL_PAIRS=PASS A=40 B=160 C=168 D=1232 D_sel=400 seedcheck=60` |
| 模型装对 | `MODEL_REPRO=PASS model=motion eps=50 mismatch=0 succ_diff=0 ref=subgoal-rollout-attn-seed42-1106/mot`（同机 A6000 逐局，18/50 对 18/50）；`MODEL_REPRO=PASS model=wan eps=50 mismatch=na succ_diff=1 ref=ctx-wan-timepool1024-3seed-1107 seed42`（AWS A100 只有局数，6/50 对 5/50，按〇′ F4） |
| 示范窗口 | `DEMO_WINDOWS=PASS model=motion eps=80 not_one=0`；`DEMO_WINDOWS=PASS model=wan eps=80 not_one=0` |
| 多局服务端 | `MULTI_SWAP_EQ=PASS runs=12 differ=0`（motion、wan 各一次） |
| 替换范围 | `SWAP_SCOPE=PASS model=motion runs=937 log_all_00=1 count_mismatch=0`；`SWAP_SCOPE=PASS model=wan runs=426 log_all_00=1 count_mismatch=0` |
| 供体 | `DONOR_TRIM=PASS donors=40 zeros=40 randoms=40`、`DONOR_DISTINCT=PASS donors=40 identical_pairs=0`（两模型各一次） |
| Wan 门槛 | `N_RESULT r_success=17/40 … gate=ok`（≥ 12，交换照常全跑） |

## ③ MotionMem 读数（seed 42）

| 组 | 有效局 | 成功 | own | donor_dir | donor_abs | other |
|---|---|---|---|---|---|---|
| N 原样 | 80 | 74 | 74 | 0 | 0 | 6 |
| Z 置零 | 37 | 10 | 10 | 0 | 0 | 27 |
| Rnd 同范数随机 | 37 | 11 | 11 | 0 | 0 | 26 |
| A 位置同·方向同 | 37 | 37 | 37 | 0 | 0 | 0 |
| B 位置同·方向异 | 150 | 5 | 4 | 136 | 0 | 10 |
| C 位置异·方向同 | 157 | 86 | 86 | 0 | 15 | 56 |
| D 位置异·方向异 | 378 | 59 | 59 | 147 | 61 | 111 |

- **Z 与 Rnd 几乎一样**（10/37 对 11/37）：掉分来自示范方向信息缺失，不是分布外输入；补上了先导轮缺的扰动对照。
- **A 37/37**：换成真正不同局的同组合 token 没有副作用（先导轮的 A 对编码逐位相同，这次 40 个供体两两不同）。
- **B 136/150 跟供体方向**：与帧记忆冲突时几乎总跟 token 走（先导轮 15/15 的结论在 40 种组合上成立）。
- **C 按位置差分层**：起点差一格时多数仍走自己的目标（例如 dpos=(0,±1) 共 40 局里 34 局成功），差两格时基本全失败（多记 other）；方向信息能跨一格搬运，跨两格不能。
- **D 分层**：供体方向从接收起点出发仍在盘内的 286 局里，147 局到 donor_dir、39 局到 donor_abs、28 局到自己目标；出盘的 92 局 donor_dir 无定义，22 局到 donor_abs、31 局到自己目标。即模型更像「在自己的位置上执行供体方向」，而不是「去供体那个绝对格子」。
- **seed 复核**：`SEED_REACH pairs=57 agree=46`（81%），比先导轮「seed 几乎不影响到达格」弱，约两成配对的到达格随模型采样 seed 改变，读主 seed 结果时应计入这部分噪声。

## ④ Frozen Wan pooling 读数（seed 42）

| 组 | 有效局 | 成功 | own | donor_dir | donor_abs | other |
|---|---|---|---|---|---|---|
| N 原样 | 80 | 34 | — | — | — | — |
| Z / Rnd | 17 / 17 | 17 / 17 | 17 / 17 | 0 | 0 | 0 |
| A | 17 | 17 | 17 | 0 | 0 | 0 |
| B | 57 | 57 | 57 | 0 | 0 | 0 |
| C | 71 | 71 | 71 | 0 | 0 | 0 |
| D | 168（盘内 116、出盘 52） | 168 | 168 | 0 | 0 | 0 |

- 原样成功率 34/80（R 17/40、S 17/40），高于它官方 PatternLock 的 14%：新局全是单步局，比官方多步局容易。
- 所有交换与扰动都不改变到达格。为排除「替换没生效」：每局 `motion_content.n_replaced=1`、`replace_log=[[0,0]]`；逐步 `command_action` 与原样局比较，Z 组 17 局无一逐位相同，但最大差的中位数 Wan 为 0.0036（随机 0.0052），MotionMem 为 0.18（随机 0.19）。token 送进了模型，只是对动作几乎没有影响。
- `SEED_REACH pairs=26 agree=18`。

## ⑤ 交集并排（两模型都原样成功的 17 个接收局）

| 组 | 局数 | MotionMem 成功 / 跟供体 | Wan 成功 / 跟供体 |
|---|---|---|---|
| Z | 17 | 5 / — | 17 / — |
| Rnd | 17 | 7 / — | 17 / — |
| A | 17 | 17 / — | 17 / — |
| B | 57 | 4 / donor_dir 51 | 57 / 0 |
| C | 71 | 44 / donor_abs 5 | 71 / 0 |
| D | 168 | 37 / donor_dir 59、donor_abs 22 | 168 / 0 |

计划第三节事先写下的三种可能里，结果对应第二种：「Wan 置零/随机后成功率不怎么掉——它没怎么用这个 token，靠帧记忆在走」。这也解释了 Wan 在官方 PatternLock 上弱于 MotionMem（14.00 ± 4.00 对 32.67 ± 1.15）：在需要从示范里读方向的单步局上，它不读示范窗口的 motion，只能依赖帧记忆。

## ⑥ 硬件与耗时

aspen 本机 2 × RTX A6000，每卡 2 worker、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`，每卡显存约 34 GB；GPU 0 上另有他人常驻进程约 0.3 GB，未动。运行 11-09 美东 12:03–15:15，约 3.2 小时（两模型共 2433 局，其中官方 50 局重评每个模型约 50 分钟）。单步局单局 rollout 约 7–19 秒。

## ⑦ 用户决策记录

见 `launch.md` ②。实施中按「自己决策」作出的裁决全部列在计划〇′，另有：

1. motion 资产复用（`ADJUST_REVIEW=PASS scale=smaller`）。
2. 合并顺序：W3 先于 W1 合入（只写新文件，接口约定由主代理先定）。
3. W1 解冲突按 F1 取本分支规则文件会丢掉用户 10-27 定的 P-LINEAGE，主会话从 `68bb958:AGENTS.md` 原样补回（`fix:` 提交）。
4. 运行调度：两模型显存只够一次 4 个 worker，按 worker 空出逐个接替（Wan repro 与 MotionMem 交换/seed 复核交叠），不改 spec。

## ⑧ 计划外事件与处置

- 单步局计数：按约定范围（3950 个 seed）实扫 1232，计划第二节写 1247；W3 核实 1247 对应多扫 1300000–1304900。40 种组合每种 ≥ 10 局，选局不受影响。
- `env.sh` 会覆盖调用方的 `O` 变量，两次 donors/readout 调用因路径拼成 `motion/motion`、`wan/wan` 直接报 FileNotFoundError；核实未产生目录，改用显式路径重跑，结果不受影响。
- 本机无 Wan 官方逐局记录（在 AWS），Wan 的 `MODEL_REPRO` 按〇′ F4 只比成功局数。
- 已知局限（沿用先导轮判法）：`first_reached` 排除接收局起点格，D 组里供体绝对目标格恰为接收起点的情形不能记为 donor_abs。

## ⑨ 结论与下一步

- MotionMem：示范窗口 token 编码的是相对方向，执行时在自己的位置上展开；跨一格位置仍能搬运，跨两格失效。
- Wan：示范窗口 token 对动作几乎无影响，PatternLock 上靠帧记忆。
- 下游：稿件仓库把 `PL_SWAP_FULL` 两行、交集表与 D 分层写进 sources.md 与附录（计划第六节）。

## ⑩ 归档文件清单（`records/`）

`pool.json`、`pairs.json`、`record_dataset_PatternLock_metadata.json`、`intersect.{md,json}`、`motion/{readout.md,readout.json,plan.json}`、`wan/{readout.md,readout.json,plan.json}`、`logs/*.summary.log`（47 份，含全部 42 份 plf- 队列日志，均以 `EXIT_CODE=0` 结尾）。运行产物（spec、录制 npz、供体文件、逐局轨迹）留在 `v1-store/evaluation/pl-swap-full-1109/`。
