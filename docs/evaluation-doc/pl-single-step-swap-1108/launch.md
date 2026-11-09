# PatternLock 单步 motion token 交换：起跑记录

## 目的

在 PatternLock easy 档的单步成功局上，只换示范段唯一的 motion 窗口（或置零），帧记忆、文字、seed 不动，看第一步跟谁走。计划：[`1108-patternlock-single-step-motion-swap-plan.html`](../../../1108-patternlock-single-step-motion-swap-plan.html) 第三版（49397af）。

## 运行环境与版本

- 环境 A：sled-aspen，2 × RTX A6000。本仓库 `/data/hongzefu/robomme_policy_learning_MotionJEPA`，分支 `v4.1-interpretMotion`。
- 冒烟到阶段 3a 首次：运行工作树 `v1-store/worktrees/pl-swap-run` @ `f503a07ed04f40199126d1e2034f668804af690b`（`commitV4.1.47`）。阶段 3a 重跑起：@ `d6bb48f6e8e1ebd9fd501686291e2668f6686481`（`commitV4.1.48`，seed≠42 改走 testx）。两版只差 `pl_swap_prep.write_specs` 的一处；seed 42 的 spec 逐字相同。
- 模型：`v1-store/assets/vla-o16cc-59999/59999`，配置 `mme_vla_suite_orig16_b128_60k`，`model_kind=stage2_pool4_concat`。benchmark gitlink `856bc3a`。
- 环境变量照抄 1106 `env.sh`：`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`HF_HUB_OFFLINE=1`、缓存在 `v1-store/cache/`；每卡 2 个 worker，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`。

## 分组与执行顺序

| 阶段 | 内容 | 局数 | 起跑（美东） |
|---|---|---:|---|
| smoke1 | seed 42 原样录局 5、20 | 2 | 21:05 |
| （手动） | `pl_swap_prep donors --episodes 5,20` | — | 21:07 |
| smoke2 | seed 42 B 20←5、Z 局 20 | 2 | 21:08 |
| stage2 | seed 42 原样其余 6 局（录 motion） | 6 | 21:11 |
| （手动） | `pl_swap_prep donors`（全部） | — | 21:16 |
| stage3a | seed 1 原样 8 局 | 8 | 21:16（契约失败）、21:23（OOM）、21:25（完成） |
| stage3b | 其余 112 局 | 112 | 21:29 → 22:08 |

## 完整命令

```bash
W=/data/hongzefu/robomme_policy_learning_MotionJEPA
T=$W/v1-store/worktrees/pl-swap-run
O=$W/v1-store/evaluation/pl-single-step-swap-1108
git -C $W worktree add --detach $T <commit>
cd $T
env PYTHONPATH=$T:$W/third_party/robomme_benchmark/src $W/v1-store/envs/official-sim/bin/python -m scripts.interpret.pl_swap_prep paths --out $O
env PYTHONPATH=$T:$T/src $W/.venv/bin/python -m scripts.interpret.pl_swap_prep specs --out $O --commit <40 位 sha>
bash $O/launch_stage.sh smoke1      # 每个有局的 worker 起 tmux pls-<阶段>-<worker>，日志 $O/logs/<阶段>-<worker>.log
env PYTHONPATH=$T:$W/third_party/robomme_benchmark/src $W/v1-store/envs/official-sim/bin/python -m scripts.interpret.pl_swap_prep donors --out $O --episodes 5,20
bash $O/launch_stage.sh smoke2
bash $O/launch_stage.sh stage2
env PYTHONPATH=$T:$W/third_party/robomme_benchmark/src $W/v1-store/envs/official-sim/bin/python -m scripts.interpret.pl_swap_prep donors --out $O
bash $O/launch_stage.sh stage3a
env PYTHONPATH=$T:$T/src $W/.venv/bin/python -m scripts.interpret.analysis.pl_swap_readout --root $O --seeds 42,1   # 中间检查
bash $O/launch_stage.sh stage3b
env PYTHONPATH=$T:$T/src $W/.venv/bin/python -m scripts.interpret.analysis.pl_swap_readout --root $O
```

`launch_stage.sh` 是运行产物目录里的小脚本：对每个 worker 起
`tmux new-session -d -s pls-<阶段>-<worker> "set -o pipefail; bash <队列> 2>&1 | tee <日志>; echo EXIT_CODE=\$? >> <日志>"`。

## 产物路径

- 运行产物：`v1-store/evaluation/pl-single-step-swap-1108/`（`specs/`、`queues/`、`<组>/seed<s>/<worker>/{attn,run}`、`record/`、`donor_trim/`、`zero/`、`readout/`、`logs/`）。
- 档案：本目录 `records/`（readout、paths、plan、清洗后日志）。

## 盯盘项

每份日志一个 Monitor，过滤 `ENV_CHECK=FAIL|rc=[1-9]|PLS_QUEUE|EXIT_CODE=|Traceback|out of memory|Killed|ContractError`。

## 本轮 tmux 会话清单

`pls-smoke1-w00`、`pls-smoke1-w10`、`pls-smoke2-w00`、`pls-smoke2-w10`、`pls-stage2-w00/w01/w10/w11`、`pls-stage3a-w00/w01/w10/w11`（同名起过 3 次，前两次由主会话按精确名停止），以及 `pls-stage3b-w00/w01/w10/w11`。全部已正常退出，收尾时 `tmux ls` 中已没有 `pls-` 会话。
