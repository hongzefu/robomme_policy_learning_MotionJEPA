# StopCube motion 速度替换启动档案（sc-speed-1110）

## ① 目的

测 StopCube 上「把 motion token 换成另一种方块速度的真实 token」时，模型按按钮的时刻是否跟着变。方块单程步数 M∈{60,80,120}，第 N 次（N∈2..5）经过目标时按下为正确，正确时刻 T(M,N)=M·N−M/2。供体 token 的造法是：同局同种子，把环境方块单程步数改成 M′，回放接收局原样动作到就位后悬停不按，在这个过程中录下 motion token。正式回合的环境与画面帧都不变，只把 motion token 逐窗口替换掉。两个模型：MotionMem concat 1×1024 与 Frozen Wan pooling。

计划：根目录 `1110-stopcube-motion-speed-swap-plan.html`（`a48af69` 补成功判断条件 K11；`57e19e9` 补开工审查裁决〇′），任务 id `sc-speed-1110`。

## ② 用户决策原话（美东 2026-11-10）

1. 「给出方案再测一下这个 stopcube 任务，就是如果我们去调整这个 motion token 的速度，会不会导致它按按 button 的速度改变？还是可以用8卡」
2. 「所以说你你是把你是单独创建的任务，然后这个任务，嗯、其他的配置都是一样的，只是改变了单层的部署，是吗？」（已答：只有录供体回合改单程步数，正式回合环境不变、只换 motion token）
3. 「好的开始吧，你你最好还要给我把这个判断条件写出来，就是如果在一个新的速度下，能不能做到 success」→ 计划 K11、§3.3
4. 「不用再来找我了直接开始吧」→ 开工令；开工审查 blocking 项由主会话自裁（计划〇′）

## ③ 版本与代码状态

- 分支 `v4.2-plswap-full`。合并链：`commitV4.5.1`（SC-W1 speed_donor 与 plfull 放行）→ `commitV4.5.2`（SC-W2 准备/读数）→ `commitV4.5.3`（SC-W2 修补）→ `commitV4.5.4`（SC-W3 干预回合关超时，实施中调整）。
- 运行代码分两批：
  - n 阶段与 MotionMem 供体录制：GL release `…-eval/releases/a2a762a8e7d90a86d53fb621abea0f7ba19a2b19`，aspen 运行树 `v1-store/worktrees/sc-speed-run`（`a2a762a`）。
  - 冒烟干预重跑、全部干预、Wan 供体录制：GL release `…/releases/17bfb50e85027eb90bf478c90752ce0661b54163`（与 `3802c24` 只差 `docs/handoff/current.json`），以及 `…/releases/3802c24c…`（冒烟），aspen 运行树 `v1-store/worktrees/sc-speed-run2`。
  - 每个 release 都核对过：`GL_RELEASE files=5578 expect=5578 bench_src=106`，`.release_sha` 严格比对。
- 闸门：
  - `PLAN_CONSISTENCY_REVIEW=FAIL base=a48af69 findings=10 blocking=5`（主会话裁决，计划〇′）
  - `PRE_MERGE_REVIEW=PASS`（SC-W1 fea43c3、SC-W2 7522b05、修补 5e75e69、SC-W3 6208b56）
  - `ADJUST_REVIEW=PASS scale=same findings=5`（干预回合关超时）

## ④ 被跑对象

| 模型 | checkpoint（aspen / GL） | model_kind |
|---|---|---|
| MotionMem | `v1-store/assets/vla-o16cc-59999/59999` / `…-eval/v1-store/assets/vla-o16cc-59999-bd35a857…/59999` | `stage2_pool4_concat` |
| Frozen Wan | `v1-store/assets/vla-o16wan/vla59k-8bf3feb9…/59999` / `…-eval/v1-store/assets/vla-o16wan/vla59k-8bf3feb9…/59999` | `wan_fixedpool1024` |

policy_config `mme_vla_suite_orig16_b128_60k`。所有 spec 都 `save_attn=false`，不存注意力。

## ⑤ 局池

`SC_POOL=PASS eps=72 cells=12 per_cell=6`：种子从 2500000 起，排除官方 train/test/val；difficulty 取官方众数 easy；局号 0–71 按 (M,N) 分配；冒烟局为 6、30、54（M=60/80/120，N=3）。见 `records/sc-pool.json`。

## ⑥ 运行方式与资源

- GreatLakes 8 个占位 `scs-hold-1..8`（JobID 65847954–65847961，1×A40、4 CPU、64G、48h）。
  - 全程只有 65847954–57 四个排到；58–61 一直排队，MotionMem 跑完后取消。
  - 每个占位只起一个 srun 作业步，内含两条通道（错开 45 秒）。排不到的占位名下的队列，由空出来的已运行占位依次接着跑。
- aspen 2×A6000，4 个 worker：跑冒烟、Wan 的 n/供体/干预阶段。
- 每个模型 72 个接收局 × 8 个回合：N（噪声种子 42，录 motion）、S（噪声种子 1）、3 个供体录制、3 个干预。
- 产物在 NFS `…-eval/v1-store/evaluation/sc-speed-1110/{motion,wan}/`，日志在 `…-eval/v1-store/logs/sc-speed-1110/` 与 aspen `v1-store/logs/sc-speed-1110/`。
- 本轮 tmux 会话：
  - aspen：`scs-motion-smoke-{n,donor,iv}-w*`、`scs-wan-{n,donor,iv}-w*`。
  - GL：`scs-motion-n-h0..7`、`scs-motion-n-a2g`、`scs-motion-n-a2gB2/B3`、`scs-motion-donor-h0..7`、`scs-motion-smoke-iv2`、`scs-motion-iv-h0..3`、`scs-wan-donor-h0/1`、`scs-wan-iv-h0/1`。
  - 收尾时全部已退出。

## ⑦ 命令（摘要）

```bash
O=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/sc-speed-1110
uv run --no-sync python -m scripts.interpret.sc_speed_prep pool --out $O
uv run --no-sync python -m scripts.interpret.sc_speed_prep specs --out $O/<M> --model <M> --stage n|donor|iv --commit <sha> [--smoke] [--no-aspen] --gl-jobs <k> --gl-per-job 2 --gl-release <release> --gl-checkpoint <…> --gl-target-contract <…>
uv run --no-sync python -m scripts.interpret.sc_speed_prep donors --out $O/<M>              # 生成 speed_donor 配置
uv run --no-sync python -m scripts.interpret.sc_speed_prep donors --out $O/<M> --finalize   # 合成供体、REPLAY_EQ、MOVE_OVERRIDE
uv run --no-sync python -m scripts.interpret.analysis.sc_speed_readout --root $O/<M>
# GL：srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared bash holds/<stage>-hold<j>.sh
```
