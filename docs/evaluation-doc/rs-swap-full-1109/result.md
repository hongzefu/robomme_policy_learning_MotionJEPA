# RouteStick 第一子目标 motion 交换结果（rs-swap-full-1109）

## ① 一句话结论与指标速览

MotionMem 在 RouteStick 上也从示范窗口 token 读第一步「往哪边走、怎么绕」：只换绕向时 18 局里 15 局按供体绕向绕杆，只换左右时 20 局里 16 局往供体那一侧走，起点不同也能搬运（同动作异起点 40/54 走到自己的目标、异起点只换绕向 32/54 跟供体绕向），走的是相对方向而不是供体的绝对目标格（D2 donor_dir 39/124、donor_abs 0/124）。Frozen Wan pooling 不读 token 的内容，但需要「有一个真 token」：任何真实供体换进去第一步都 100% 成功、0 局跟供体（A/Bsw/Bside/C/D1/D2/W0 合计 237/237），把窗口置零后第一步只剩 1/13 成功，同范数随机 6/13。

最终判定行（原文，见 `records/motion/readout.md`、`records/wan/readout.md`、`records/intersect.md`）：

- `RS_SWAP_FULL=PASS model=motion runs=410 invalid=0 N=29/48 Z=8/18 Rnd=8/18 A=13/18 Bsw_swing=15/18 Bside_dir=16/20 C_own=40/54 D1_swing=32/54 D2_dir=39/124 D2_abs=0/124 W0_swing=9/18 W0_dir=7/20`
- `RS_SWAP_FULL=PASS model=wan runs=311 invalid=0 N=18/48 Z=1/13 Rnd=6/13 A=13/13 Bsw_swing=0/13 Bside_dir=0/16 C_own=39/39 D1_swing=0/39 D2_dir=0/88 D2_abs=0/88 W0_swing=0/13 W0_dir=0/16`
- `RS_SWAP_INTERSECT models=motion,wan recv=10 motion:Z=6/10,Rnd=6/10,A=9/10,Bsw_swing=8/10,Bside_dir=8/10,C_own=24/30,D1_swing=16/30,D2_dir=18/70,D2_abs=0/70,W0_swing=3/10,W0_dir=4/10 wan:Z=1/10,Rnd=4/10,A=10/10,Bsw_swing=0/10,Bside_dir=0/10,C_own=30/30,D1_swing=0/30,D2_dir=0/70,D2_abs=0/70,W0_swing=0/10,W0_dir=0/10`

判定行中 N 为整局成功、Z/Rnd/A 为第一步成功，其余为跟随计数；PASS 只表示数据完整、无效局 0。

## ② 闸门与验收

| 判定 | 结果 |
|---|---|
| 开工审查 | `PLAN_CONSISTENCY_REVIEW=FAIL base=9e1038d findings=7 blocking=1`，按用户「尽可能自己做 不要阻塞」裁决（计划〇′，`2d83b00`） |
| 合并 | `PRE_MERGE_REVIEW=PASS base=c8a56a9 tip=aad32e6 files=3 commits=1 findings=3`；合并后短测 57 passed |
| 复算与局池 | `RS_REPLAY=PASS seeds=20 mismatches=0`；`RS_POOL=PASS eps=48 types=16 overlap=0`；`RS_PAIRS=PASS A=32 Bsw=32 Bside=48 C=96 D1=96 D2=208 W0=80 seedcheck=60` |
| 示范结构 | `RS_DEMO_LAYOUT=PASS model=motion eps=48 bad=0`；`RS_DEMO_LAYOUT=PASS model=wan eps=48 bad=0`（100 帧、窗口 0/32/64、第一步第 0–49 帧） |
| 替换范围 | `SWAP_SCOPE=PASS model=motion runs=467 count_mismatch=0 scope_bad=0`；`SWAP_SCOPE=PASS model=wan runs=324 count_mismatch=0 scope_bad=0` |
| 多局服务端 | `MULTI_SWAP_EQ=PASS runs=12 differ=0`（两模型各一次） |
| 供体 | `DONOR_TRIM=PASS donors=16`、`DONOR_DISTINCT=PASS donors=16 identical_pairs=0`（两模型各一次） |
| 读数与环境判定 | `OFFICIAL_STEP1` motion 410/410 一致；wan 311 局中 310 一致（1 局读数判失败、环境判第一子目标完成，见⑦） |
| Wan 门槛 | `N_RESULT r_success=13/32 gate=ok`（≥ 8，交换全跑） |
| 官方 50 局重评 | 按用户「不用这个！」取消 |

## ③ MotionMem 读数（seed 42，只读第一步）

| 组 | 有效局 | 第一步成功 | own | donor_dir | donor_abs | other | own_swing | donor_swing | other_swing |
|---|---|---|---|---|---|---|---|---|---|
| N 原样 | 48 | 31 | 43 | 0 | 0 | 5 | 33 | 0 | 15 |
| Z 置零 | 18 | 8 | 11 | 0 | 0 | 7 | 13 | 0 | 5 |
| Rnd 同范数随机 | 18 | 8 | 11 | 0 | 0 | 7 | 13 | 0 | 5 |
| A 同类型 | 18 | 13 | 17 | 0 | 0 | 1 | 14 | 0 | 4 |
| Bsw 只换绕向 | 18 | 3 | 15 | 0 | 0 | 3 | 3 | **15** | 0 |
| Bside 只换左右 | 20 | 2 | 4 | **16** | 0 | 0 | 11 | 5 | 4 |
| C 同动作异起点 | 54 | 30 | **40** | 0 | 6 | 8 | 40 | 0 | 14 |
| D1 同左右异绕向 | 54 | 13 | 42 | 0 | 6 | 6 | 22 | **32** | 0 |
| D2 起点左右都不同 | 124 | 48 | 85 | **39** | **0** | 0 | 69 | 21 | 34 |
| W0 只换起点 0 窗口 | 38 | 20 | 30 | 7 | 0 | 1 | 24 | 11 | 3 |

- **绕向跟 token 走**：Bsw 15/18 按供体绕向；跨起点仍成立（D1 32/54）。
- **左右跟 token 走**：Bside 16/20 往供体那一侧。
- **相对方向**：D2 里供体方向从接收起点出发仍在盘内的 60 局，39 局到 donor_dir、0 局到 donor_abs；出盘的 64 局全部走回自己的目标格（own 64/64，但绕向 24 局错）。
- **第一步尾巴窗口的作用**：只换起点 0 窗口（W0）时，跟供体绕向从 15/18 降到 9/18，跟供体左右从 16/20 降到 7/20；起点 32 窗口（第一步后 18 帧 + 第二步头 15 帧）承载了第一步信息的相当一部分。
- **置零 / 随机**：两组第一步成功都是 8/18，18 局里 16 局的成败与到达、绕向完全相同；A 组只有 13/18，说明 MotionMem 在 RouteStick 第一步本身就不太稳。
- **seed 复核**：`SEED_FIRST_STEP pairs=39 agree=26`（67%），比 PatternLock（81%）低，单局读数噪声更大。

## ④ Frozen Wan pooling 读数（seed 42，只读第一步）

| 组 | 有效局 | 第一步成功 | own | other | none | own_swing | donor_swing | other_swing | none |
|---|---|---|---|---|---|---|---|---|---|
| N 原样 | 48 | 19 | 29 | 19 | 0 | 23 | 0 | 25 | 0 |
| Z 置零 | 13 | 1 | 8 | 5 | 0 | 3 | 0 | 10 | 0 |
| Rnd 同范数随机 | 13 | 6 | 8 | 0 | 5 | 6 | 0 | 2 | 5 |
| A / Bsw / Bside / C / D1 / D2 / W0 | 13 / 13 / 16 / 39 / 39 / 88 / 29 | 全部成功 | 全部 | 0 | 0 | 全部 | 0 | 0 | 0 |

- 换进任何真实供体的 token，第一步都按自己的路径和绕向走，跟供体 0 局（237/237 own）。
- 但把窗口置零后只剩 1/13 成功（多数走错绕向），同范数随机 6/13（5 局没到达任何目标）。即 Wan 需要窗口里有一个「像真 token 的东西」，但不读它的方向内容；方向信息来自帧记忆。
- 与 PatternLock 的差别：PatternLock 上 Wan 置零、随机都不掉分（17/17），这里置零掉到 1/13。
- `SEED_FIRST_STEP pairs=17 agree=11`。

## ⑤ 交集并排（两模型都原样成功的 10 个接收局）

| 组 | 局数 | MotionMem 第一步成功 / 跟供体 | Wan 第一步成功 / 跟供体 |
|---|---|---|---|
| Z | 10 | 6 / — | 1 / — |
| Rnd | 10 | 6 / — | 4 / — |
| A | 10 | 9 / — | 10 / — |
| Bsw | 10 | 2 / 绕向 8 | 10 / 0 |
| Bside | 10 | 2 / 左右 8 | 10 / 0 |
| C | 30 | 22 / — | 30 / — |
| D1 | 30 | 11 / 绕向 16 | 30 / 0 |
| D2 | 70 | 38 / donor_dir 18、donor_abs 0 | 70 / 0 |

## ⑥ 硬件与耗时

aspen 2 × RTX A6000，每卡 2 worker、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`。运行 11-09 美东 16:55–19:39，约 2.7 小时（两模型共约 870 局，含冒烟、N、多局等价、交换、W0、seed 复核）。两步局单局 rollout 约 20–60 秒。

## ⑦ 计划外事件与处置

- **取消官方 50 局重评**：用户「不用这个！」；运行器代码与上一轮相同，上一轮在 `efc3a19` 上两模型已 `MODEL_REPRO=PASS`。
- **中间读数误报 `RS_SWAP_FULL=FAIL invalid=19`**：W0 组跑到一半时出读数，把尚未写终局的在途局计为无效（全部 `no_terminal`）；全部跑完后 `invalid=0`。
- **审查提出的轨迹段隐患不成立**：冒烟核实执行段第一步时杆尖已在起点格（`first_at_start=0`），第一步轨迹段不含拿杆段。
- **Wan 1 局读数与环境判定不一致**：读数判第一步失败、环境判第一子目标完成（`OFFICIAL_STEP1 agree=310/311`）；不影响任何跟随计数。
- **调度**：多局等价之后每个 worker 位置挂一条顺序链（`rsf-chain-<w>`），各段仍单独日志与 `EXIT_CODE`，规模与 spec 未变。
- **D2 盘内 / 出盘局数**：按接收局原样成功过滤后，MotionMem 实跑 D2 盘内 60、出盘 64（计划全量为 144 / 64）。

## ⑧ 结论与下一步

- MotionMem 在两个任务上的结论一致：示范窗口 token 编码第一步的方向（RouteStick 上包括左右与绕向），执行时以自己的位置为参照展开；PatternLock 跨一格可搬运，RouteStick 跨起点也能搬运。
- Wan 在两个任务上都不读 token 内容；RouteStick 上它对「窗口里有没有真 token」敏感（置零崩），PatternLock 上不敏感。
- 下游：稿件仓库写入 sources.md 与附录。

## ⑨ 归档文件清单（`records/`）

`pool.json`、`pairs.json`、`record_dataset_RouteStick_metadata.json`、`intersect.{md,json}`、`motion/{readout.md,readout.json,plan.json}`、`wan/{readout.md,readout.json,plan.json}`、`logs/*.summary.log`（49 份，含 46 份 rsf- 日志，均以 `EXIT_CODE=0` 结尾）。运行产物留在 `v1-store/evaluation/rs-swap-full-1109/`。
