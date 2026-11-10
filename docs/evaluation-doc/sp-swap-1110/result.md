# 逐子目标位置 motion 交换结果（sp-swap-1110）

## ① 一句话结论

在正式多步局上，MotionMem 换掉第 j 个子目标对应的示范 motion token 后，第 j 步跟随供体的比例高于不干预时的基线（RouteStick 同位置 +0.08 [+0.02,+0.13]、跨位置 +0.11 [+0.02,+0.20]；PatternLock 跨位置 +0.10 [+0.02,+0.19]、同位置 +0.04 [−0.01,+0.10]），效应主要集中在第 1 个子目标；Frozen Wan pooling 在所有对比上都接近 0。但 MotionMem 没有对角结构：换第 j 段时它之前已执行的步也有约一半与原样不同，与它换采样 seed 时的一致率（20/40）相当，说明 MotionMem 的轨迹对任何扰动都敏感，而 Wan 前序步几乎全保持（因为它不读 token）。三个交换任务上，换掉或置零第 k 次交换段的 motion token 后，MotionMem 大多仍拿对第一个物体，很少改拿「没发生该次交换」的物体——交换任务的记忆主要不靠交换段 motion token。

## ② 判定行（原文）

多步任务（`records/{motion,wan}/readout.md`、`records/intersect.md`）：

- `SP_REPLAY=PASS model=motion eps=240 steps=201 mismatches=0`；`SP_REPLAY=PASS model=wan eps=240 steps=144 mismatches=0`
- `SP_SEGMENT=PASS`（两模型 × 两任务，各 120 局 bad=0）
- `SWAP_SCOPE=PASS model=motion runs=1042 scope_bad=0 count_mismatch=0`；`SWAP_SCOPE=PASS model=wan runs=935 scope_bad=0 count_mismatch=0`
- `MULTI_SWAP_EQ=PASS runs=6 differ=0`（两模型各一次）
- `SEED_STEPS pairs=40 agree=20`（motion）；`SEED_STEPS pairs=40 agree=22`（wan）
- `SP_MATRIX=PASS model=motion task=PatternLock runs=453 invalid=0 Bsame_follow=29/181 Bsame_keep=34/83 Bcross_follow=22/91 Bcross_keep=20/40 Z_correct=48/181 Z_keep=51/83`
- `SP_MATRIX=PASS model=motion task=RouteStick runs=545 invalid=0 Bsame_follow=67/218 Bsame_keep=95/158 Bcross_follow=42/109 Bcross_keep=65/90 Z_correct=77/218 Z_keep=123/158`
- `SP_MATRIX=PASS model=wan task=PatternLock runs=405 invalid=0 Bsame_follow=23/162 Bsame_keep=47/55 Bcross_follow=12/81 Bcross_keep=24/29 Z_correct=34/162 Z_keep=7/55`
- `SP_MATRIX=PASS model=wan task=RouteStick runs=488 invalid=0 Bsame_follow=52/195 Bsame_keep=114/114 Bcross_follow=20/98 Bcross_keep=59/61 Z_correct=47/195 Z_keep=46/114`
- `SP_MATRIX_INTERSECT task=PatternLock positions=146 …`；`task=RouteStick positions=165 …`

交换任务（`records/fp/{motion,wan}/readout.md`）见 ⑤。

## ③ 多步任务主分析：第 j 步跟随率减基线

基线 = 同一 (局, j) 集合上不干预运行「碰巧到达同一供体目标」的比例；区间为按接收局重采样 2000 次的 95% 区间；干预只做在不干预运行走到第 j 步起点的 (局, j) 上，不按第 j 步本身是否正确再筛选（「补充」列为只看不干预时第 j 步正确的局）。

| 任务 / 组 | MotionMem | Frozen Wan |
|---|---|---|
| PatternLock Bsame（同位置供体） | +0.04 [−0.01, +0.10]，n=181；补充 +0.09 | −0.03 [−0.06, −0.01]，n=162；补充 0.00 |
| PatternLock Bcross（不同位置供体） | **+0.10 [+0.02, +0.19]**，n=91；补充 +0.19 | −0.01 [−0.04, 0.00]，n=81 |
| RouteStick Bsame | **+0.08 [+0.02, +0.13]**，n=218；补充 +0.21 | +0.01 [−0.01, +0.02]，n=195 |
| RouteStick Bcross | **+0.11 [+0.02, +0.20]**，n=109；补充 +0.29 | +0.01 [−0.02, +0.05]，n=98 |

按位置 j 合并（MotionMem；Wan 各位置差均在 −0.12 至 +0.11 之间且多数为 0）：

| j | PatternLock Bsame | PatternLock Bcross | RouteStick Bsame | RouteStick Bcross |
|---|---|---|---|---|
| 1 | +0.08 [+0.03,+0.15] n=120 | +0.17 [+0.08,+0.27] n=59 | +0.09 [+0.03,+0.16] n=120 | +0.17 [+0.07,+0.28] n=54 |
| 2 | −0.05 [−0.16,+0.07] n=44 | −0.04 [−0.20,+0.12] n=25 | 0.00 [−0.12,+0.12] n=59 | −0.03 [−0.19,+0.12] n=32 |
| 3 | 0.00 n=12 | 0.00 n=6 | +0.19 [+0.05,+0.38] n=21 | +0.15 [−0.15,+0.46] n=13 |
| 4 | 0.00 n=5 | 0.00 n=1 | +0.13 [−0.20,+0.47] n=15 | +0.12 [−0.25,+0.50] n=8 |
| 5 | — | — | 0.00 n=3 | +0.50 n=2 |

- 第 1 个子目标上效应清楚；第 2 个子目标没有；RouteStick 第 3、4 个子目标点估计为正但样本少、区间宽。PatternLock 第 3 步以后 n ≤ 12，走到的局本身就很少（MotionMem m≥2 的局约一半第 1 步就失败）。
- 同位置与跨位置供体都能带偏第 1 步，跨位置的点估计不比同位置小——token 带来的方向信息没有表现出与示范中的时序位置绑定。

## ④ 特异性矩阵（i≠j）

图：`records/matrix_{PatternLock,RouteStick}_{Bsame,Bcross}.png`（两模型并排，红框为对角，格内为 k/n）。

| 任务 / 组 | i<j 与原样同结果：MotionMem | Frozen Wan |
|---|---|---|
| PatternLock Bsame | 34/83 | 47/55 |
| PatternLock Bcross | 20/40 | 24/29 |
| RouteStick Bsame | 95/158 | 114/114 |
| RouteStick Bcross | 65/90 | 59/61 |

- MotionMem 没有对角结构：换第 j 段时，第 j 步之前已执行的步也常常改变（约一半）。示范 token 在开局就全部可见，这些变化在时间上发生在第 j 步之前，所以不是「第 j 步被带偏后起点变了」造成的。
- 这一比例与 MotionMem 只换采样 seed 时的一致率（20/40）相当，所以前序步改变更像是 MotionMem 轨迹对任何输入扰动都敏感（混沌式发散），不是 token 内容的选择性作用。Wan 的前序步几乎全保持，原因是它不读 token（与前两轮单步实验一致）。
- 与审稿意见对照：注意力分析没有对角线（B.2），这里的干预同样没有对角线，两者一致，但都不能当作「段级选择性控制」的证据。

## ⑤ 交换任务：按第 k 次交换只换该次交换段

接收局为不干预时第一次就拿对的局；swap_blind_k = 「若第 k 次交换没发生」时目标所在物体；基线为全部原样局中拿到对应 swap_blind 物体的比例。

MotionMem：`FIRST_PICK_LABEL=PASS eps=270 on_turn=268 wrong_but_success=0 correct_but_fail=0`；`SWAP_SCOPE=PASS runs=468 scope_bad=0`；`SEED_FIRST_PICK pairs=60 agree=45`。

| 任务 | Bsame_k（同类交换段供体）仍拿对 / 拿 swap_blind | Z_k（置零）仍拿对 / 拿 swap_blind | 原样局 swap_blind 率 |
|---|---|---|---|
| VideoUnmaskSwap | 47/53 / 1 | 43/53 / 3 | 35/100 |
| ButtonUnmaskSwap | 53/74 / 6 | 46/74 / 10 | 32/101 |
| VideoRepick | 69/77 / 3 | 62/77 / 5 | 31/119 |

Frozen Wan：`FIRST_PICK_LABEL=PASS eps=270 on_turn=269 wrong_but_success=0 correct_but_fail=0`；`SWAP_SCOPE=PASS runs=412 scope_bad=0`；`SEED_FIRST_PICK pairs=60 agree=46`。

| 任务 | Bsame_k 仍拿对 / 拿 swap_blind | Z_k 仍拿对 / 拿 swap_blind | 原样局 swap_blind 率 |
|---|---|---|---|
| VideoUnmaskSwap | 53/55 / 0 | 30/55 / 0 | 24/100 |
| ButtonUnmaskSwap | 47/53 / 2 | 22/53 / 2 | 38/101 |
| VideoRepick | 68/68 / 0 | 61/68 / 1 | 45/119 |

原样（N）第一次拿对：MotionMem VUS 28/90、BUS 34/90、VR 40/90；Wan 27/90、26/90、33/90。

逐 (n,k) 的行在 `records/fp/motion/readout.md`。结论：换掉或抹掉第 k 次交换段的 motion token，MotionMem 多数仍拿对（72%–90%），改拿「没交换」答案的很少（1%–14%），置零比换供体掉得稍多（ButtonUnmaskSwap 最明显）。即交换任务上第一次拿起主要不依赖交换段 motion token 的内容，记忆更可能来自帧记忆。Wan 换成别的交换段几乎不受影响（拿对 47/53–68/68），但置零后明显掉（VUS 30/55、BUS 22/53），且掉下来的局也不拿 swap_blind（0–2 局）——与 RouteStick 上一样，Wan 需要窗口里「有一个真 token」，但不读其内容。

## ⑥ 硬件与耗时

- GreatLakes 8 × A40 单卡占位（65812859–65812866）+ aspen 2 × A6000（4 worker）。美东 11-10 00:1x 提交占位，00:45 起跑 N，05:4x 收尾。
- 局数：多步 N 480、干预 MotionMem 1050 + Wan 945；交换任务 N 540、按 k 干预 468 + 412；另冒烟、重跑。

## ⑦ 计划外事件与处置（实施中调整）

1. **局号上限 80→120**（`commitV4.3.2`，`ADJUST_REVIEW=PASS scale=same`）。
2. **ButtonUnmaskSwap 开局抬起容器被误判为拿起**：环境 elapsed 0–64 自动抬起全部容器；容器拿起判据加抓取（`commitV4.4.4`）。
3. **供体全同检查拦截**：PatternLock 6 对不同局同位置示范编码逐位相同（子目标也相同），改为 WARN 并新增「供体与接收局同编码则跳过」（`commitV4.4.5`，`ADJUST_REVIEW=PASS scale=same`；实际 same_as_recv=0）。
4. **GreatLakes 运行方式**：最初每个队列一个 srun 作业步，作业步结束后同卡新起服务端拿不到 GPU（`no supported devices`），叠加一占位 3–4 进程的 OOM，导致多批队列中途失败；改为每占位一个长作业步、内含两条通道。期间三次重排优先级时，未跑完的 spec 整体归档后重跑（run_local 不支持续跑），损失约数百局计算，全部数据以重跑结果为准，归档在 `runs-failed/`、`logs/*/failed/`。
5. **两条通道同时启动撞端口**（Wan g0a 一组 5 局），之后通道错开 45 秒启动。
6. **空闲占位接手**：通道里尚未开始的队列复制给空闲占位，原队列文件改写为空脚本（`STOLEN_TO_*`），避免重复。
7. **aspen 尾段补跑**：aspen w00 的 Wan 干预后段改到 1 号卡补跑，前两次因 spec 写死 `cuda: "0"` 仍落 0 号卡 OOM，第三次用 cuda=1 的 spec 副本成功。
8. **交换任务帧号对应偏差**：`FP_FRAMEMAP=FAIL bad=120`——ButtonUnmaskSwap 第 0 个动作的 env_step 为 2（帧号与 env_step 偏 1），VideoUnmaskSwap 30 局示范长度正好止于最后交换 end（gap=−1）。W_k 的交换帧区间按 [start+1, end+1] 计，实际约早 1–2 帧，影响仅限紧挨交换开始、只含 1–2 帧交换画面的那一个窗未被替换；交换主体窗均已替换。结果按已知局限报告，未重跑。
9. **按 k 跑法没有多局等价阶段**，`MULTI_SWAP_EQ=FAIL runs=0` 为不适用。
10. **反向对照（换帧 token、保留 motion）未跑**：服务端已实现（`commitV4.4.1`，frame_swap_exact），run_local 接入与帧供体生成未做，按计划优先级 4 放弃。
11. **W_j core 版未跑**：读数按「W_j 是否含跨界窗」分组报告（readout.md 中「无跨界 / 含跨界」两行）；PatternLock 绝大多数位置含跨界窗（173/181）。

## ⑧ 结论与限制

- 多步任务上 MotionMem 的示范 token 对第 1 个子目标有可测的定向作用（效应 +0.08 至 +0.17），Wan 没有。这把前两轮「简化单步局」的结论推进到正式多步局的第一步；后续位置样本太少，不能下结论。
- 没有对角结构；MotionMem 对扰动整体敏感（seed 一致率约 50%），所以单局读数噪声大，结论只在按接收局聚合后成立。
- 交换任务上交换段 motion token 的内容不是主要信息来源（两模型换供体后改拿「没交换」答案的都很少）；MotionMem 置零掉分小，Wan 置零掉分大但不跟内容。
- 局限：新 seed 局池，不与官方分数直接比；主分析限定在不干预能走到第 j 步的位置；跨界窗普遍存在；交换任务 W_k 边界偏 1–2 帧；反向对照未做。

## ⑨ 归档

`records/`：`motion/`、`wan/`（多步 readout.md/json、自动热力图）、`matrix_*.png` 与 `plot_matrix.py`（两模型并排矩阵，已目视复查）、`intersect.{md,json}`、`sp-pool.json`、`fp/{motion,wan}/readout.{md,json}`。运行产物在 NFS `…-eval/v1-store/evaluation/{sp-swap-1110,fp-swap-1110}/`。
