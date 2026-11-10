# 逐子目标位置 motion 交换启动档案（sp-swap-1110）

## ① 目的

回应审稿意见「干预只在简化的单子目标局上」：在多步局每个长度、每个子目标位置 j 上，只替换覆盖第 j 个子目标的示范 motion 窗口，读第 j 步是否跟供体，并读其他步是否不受影响（「干预位置 j × 被影响子目标 i」矩阵）。PatternLock（m=1–7）、RouteStick（m=2–7）；三个交换任务（VideoUnmaskSwap、ButtonUnmaskSwap 三档，VideoRepick easy/medium）按第 k 次交换读第一次拿起。两模型：MotionMem（orig100 concat 1×1024）与 Frozen Wan pooling。

计划：根目录 `1109-subgoal-position-swap-all-lengths-plan.html`（`ac6ff45`，修订一 `c79053b` 含开工审查裁决 F1–F12）。取代并吸收 `1109-swap-tasks-first-pick-motion-swap-plan.html`（任务 `fp-swap-1110`，已并入）。

## ② 用户决策原话（按时间，美东 11-09 晚—11-10 凌晨）

1. 「我还需要再测两个任务，就是 video and mask swap 和 button and mask swap，然后都只测第一次第一次拿起来，然后我们要测的是这个就是不同的 swap 对它这个模型有影响。」
2. 「然后那个 video repick 任务它也是有 swap 的，你也要查，它是 easy 和 medium 两个任务。」
3. 「你先暂停，我需要对于这个呃回答进行扩展。」并贴两份审查意见。
4. 「现在的这个 swapping 任务以及 pattern lock rootstake 任务的长度是多少？就是我需要在每一个长度上都做分析。但是分析的都是一个 sub goal 的 steerable 的可能性，但是我每一个长度位置都要分析。……尽可能压缩在……明天早上之前能够完成。……你可以 query 八个 Great Lakes 的 job，呃，单卡的 job 来完成。」
5. 「Video Repick Hard 这个不考虑，因为它这个没有交换。」
6. 开工令：「开工呢哦，有问题尽可能在十五分钟内问我，之后就尽可能自己做。」
7. 「我现在要睡觉了，之后有问题都留待我裁决，尽可能不要停，一路跑到明天早上八点」

## ③ 版本与代码状态

- 分支 `v4.2-plswap-full`。合并链：`commitV4.4.1`（SP-W3 frame_swap_exact）→ `commitV4.4.2`（SP-W2 交换任务按第 k 次交换）→ `commitV4.4.3`（SP-W1 多步扫描/分段/供体/读数矩阵）→ `commitV4.4.4`（FP-W5 容器拿起判据加抓取）→ `commitV4.4.5`（SP-W4 供体全同改记录）。此前 `commitV4.3.1–4.3.3`（FP-W1/W3/W2）。
- 运行代码：GreatLakes release `…-eval/releases/14debb7d6f6f736ba0e5bdc873123e44ef10bfce`（`git archive` + 子模块 archive，`GL_RELEASE=PASS files=5543 bench_src=106`，`.release_sha` 严格比对）；aspen 运行树 `v1-store/worktrees/sp-run`（`14debb7`，detached clean）。`commitV4.4.5` 只改 donors 生成逻辑（主检出 `e25fa45` 上执行），运行代码不变。
- 闸门：`PLAN_CONSISTENCY_REVIEW=FAIL base=ac6ff45 findings=12 blocking=2`（主代理裁决，计划〇′）；各块 `PRE_MERGE_REVIEW=PASS`；`ADJUST_REVIEW=PASS scale=same`（局号上限 80→120、供体全同改记录）。

## ④ 被跑对象

| 模型 | aspen checkpoint | GreatLakes checkpoint | model_kind |
|---|---|---|---|
| motion | `v1-store/assets/vla-o16cc-59999/59999` | `…-eval/v1-store/assets/vla-o16cc-59999-bd35a857…/59999` | `stage2_pool4_concat` |
| wan | `v1-store/assets/vla-o16wan/vla59k-8bf3feb9…/59999` | `…-eval/v1-store/assets/vla-o16wan/vla59k-8bf3feb9…/59999`（rsync 后 `VLA_SYNC=PASS model=wan files=35 sha_mismatch=0`） | `wan_fixedpool1024` |

policy_config 均为 `mme_vla_suite_orig16_b128_60k`；GL 冒烟 `GL_SMOKE`：两模型 RouteStick:0/33 与 aspen 结果一致（2/2 成功）。

## ⑤ 局池

- PatternLock：seed 2300000 起，`SP_SCAN=PASS seeds=159 replay_mismatch=0 segment_bad=0`；`SP_POOL=PASS eps=120 m=1:8,2:12,3:18,4:20,5:20,6:21,7:21`。
- RouteStick：seed 2400000 起，`SP_SCAN=PASS seeds=158 …`；`SP_POOL=PASS eps=120 m=2:15,3:15,4:20,5:20,6:25,7:25`。
- 交换任务：VUS 2000000/2050000(hard)、BUS 2100000/2150000(hard)、VR 2200000 起；`FP_POOL=PASS` 三任务各 90 局，n1:30,n2:30,n3:30。VR 300 个 seed 中 17 个场景生成失败跳过。

## ⑥ 运行方式与资源

- GreatLakes 占位 8 个（`sps-hold-1..8`，JobID 65812859–65812866，1×A40、4 CPU、64G、48h；超出默认 4 个与 1 CPU/24G 由开工令放行）。
- 最终跑法：**每个占位只起一个 srun 作业步，内含两条通道顺序跑队列**（第 4 轮 `sps-hold{0..7}-r4`）。前几轮「每个队列一个作业步」导致作业步结束后同卡新服务端拿不到 GPU（`no supported devices`）与显存叠加 OOM，见 result.md ⑦。
- aspen：2×A6000，4 worker；会话 `sps-chain-aspen-n-w*`（N）、`sps-iv-aspen-w*`、`sps-ivwan-aspen-w*`（干预）。
- 产物与日志：NFS `…-eval/v1-store/evaluation/{sp-swap-1110,fp-swap-1110}/`、`…-eval/v1-store/logs/…`；aspen 日志 `v1-store/logs/{sp,fp}-swap-1110/`。

## ⑦ 命令（摘要）

```bash
O=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/sp-swap-1110
$SIM -m scripts.interpret.sp_swap_prep scan --out $O --task PatternLock|RouteStick --workers 8
python -m scripts.interpret.sp_swap_prep pool --out $O
python -m scripts.interpret.sp_swap_prep specs --out $O/<M> --model <M> --commit 14debb7… --gl-jobs 8 --gl-per-job 2 --gl-release <release> --gl-checkpoint <…> --gl-target-contract <…>
python -m scripts.interpret.sp_swap_prep donors --out $O/<M>
python -m scripts.interpret.analysis.sp_swap_readout --root $O/<M>
# 交换任务（GL 目录 …/fp-swap-1110）
python -m scripts.interpret.fp_swap_prep pool --out <FPO> --by-swap-times
python -m scripts.interpret.fp_swap_prep specs --out <FPO>/<M> --gl-store <GLSTORE> --run-tree <release> …
python -m scripts.interpret.fp_swap_prep donors --out <FPO>/<M> --by-k
```
