# RouteStick 第一子目标 motion 交换启动档案（rs-swap-full-1109）

## ① 目的

在 RouteStick easy 档恰好两步的局上，只换第一个子目标（1 个路径 × 1 个绕向）对应的示范 motion 窗口（起点 0、32），只读第一步走到哪个目标、绕杆顺时针还是逆时针，按 16 种第一步类型做全因子受控交换；MotionMem（orig100 concat 1×1024）与 Frozen Wan pooling 各跑一遍。计划 `1109-routestick-swap-full-factorial-plan.html`（第二版 `9e1038d`，修订一 `2d83b00`）。

## ② 用户决策原话（按时间）

1. 「给出方案 再测一下routestick任务」
2. 「routestick也是只做单步」「我们只测一个subgoal的更随性」「一个subgoal就是1个路径*1个绕向」
3. 开工令：「同意开工并登记」；「同意开始 尽可能自己做 不要阻塞」
4. 「不用这个！」——取消官方 50 局重评（`MODEL_REPRO`）

## ③ 版本与代码状态

- 分支 `v4.2-plswap-full`，运行代码钉在 `8d7d60320be745ebef7c906f495e94b1d67f4e9c`（`commitV4.2.4`，合入 RS-W1），分离工作树 `v1-store/worktrees/rs-full-run`（detached，clean）。
- 闸门：`PLAN_CONSISTENCY_REVIEW=FAIL base=9e1038d findings=7 blocking=1`（按用户指令主代理裁决，计划〇′）；`PRE_MERGE_REVIEW=PASS base=c8a56a9 tip=aad32e6 files=3 commits=1 findings=3`；合并后短测 57 passed。
- 运行器代码（run_local、serve_policy_attn、eval.py）沿用上一轮 `efc3a19`，S0 核实 plfull 口径无任务硬编码，未改。

## ④ 被跑对象

| 模型 | checkpoint | policy_config | model_kind |
|---|---|---|---|
| motion | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` | `stage2_pool4_concat` |
| wan | `v1-store/assets/vla-o16wan/vla59k-8bf3feb9…/59999` | `mme_vla_suite_orig16_b128_60k` | `wan_fixedpool1024` |

资产已于 pl-swap-full-1109 核验（`ASSETS=PASS model=motion files=38`、`VLA_DOWNLOAD=PASS files=35`），本轮不下载。

## ⑤ 局池

env_seed 1400000 起步长 100 共 4000 个（与 RouteStick train/test/val 无交集），两步局 1999 个，每类型 ≥ 74 局；每类型取 3 局：R 局号 2k、2k+1，S 局号 32+k。
`RS_REPLAY=PASS seeds=20 mismatches=0`；`RS_POOL=PASS eps=48 types=16 overlap=0`；`RS_PAIRS=PASS A=32 Bsw=32 Bside=48 C=96 D1=96 D2=208 W0=80 seedcheck=60`。

## ⑥ 命令与执行顺序

```bash
O=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/rs-swap-full-1109
python -m scripts.interpret.rs_full_prep pool --out $O                      # official-sim 解释器
python -m scripts.interpret.rs_full_prep specs --out $O/motion --model motion --commit 8d7d603… --run-tree <rs-full-run>
python -m scripts.interpret.rs_full_prep specs --out $O/wan --model wan --commit 8d7d603… --run-tree <rs-full-run> \
  --checkpoint <wan 59999> --policy-config mme_vla_suite_orig16_b128_60k --model-kind wan_fixedpool1024 --target-contract none
bash $O/<M>/launch_stage.sh smoke | n
python -m scripts.interpret.rs_full_prep donors --out $O/<M> [--wan-gate 8]
# multieq 起，每个 worker 位置挂一条顺序链（tmux rsf-chain-<w>）：等本位置 motion multieq 结束后依次跑
#   motion swap → motion w0 → motion seeds → wan multieq → wan swap → wan w0 → wan seeds，
#   每段队列单独 tee 日志 rsf-<m>-<stage>-<w>.log 并写 EXIT_CODE= 尾行（链脚本在 $O/chain.sh，不入档）
python -m scripts.interpret.analysis.rs_full_readout --root $O/<M>
python -m scripts.interpret.analysis.rs_full_readout --root $O --intersect motion,wan
```

每卡 2 worker、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`，环境变量见 `$O/<M>/env.sh`。

## ⑦ 产物与日志

产物根 `v1-store/evaluation/rs-swap-full-1109/`；日志 `v1-store/logs/rs-swap-full-1109/`。

## ⑧ 本轮 tmux 会话清单（全部已自行退出，日志末行 `EXIT_CODE=0`）

| 会话 | 用途 | 时段（美东 11-09） |
|---|---|---|
| `rsf-motion-smoke-w00`、`rsf-wan-smoke-w00` | 冒烟 | 16:55 – 16:59 |
| `rsf-motion-n-{w00,w01,w10,w11}` | MotionMem N | 17:01 – 17:10 |
| `rsf-wan-n-{w00,w01,w10,w11}` | Wan N（接替空出位置） | 17:0x – 17:16 |
| `rsf-motion-multieq-{w00,w01,w10,w11}` | MotionMem 多局等价 | 17:1x – 17:23 |
| `rsf-chain-{w00,w01,w10,w11}` | 顺序链（其内各段日志 `rsf-{motion,wan}-{swap,w0,seeds,multieq}-<w>`） | 17:15 – 19:39 |
