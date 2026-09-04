# v1-prod-60k / ckpt 59999 / seed 7 在线评估起跑档（双卡并行，4 任务 × 50 episodes）

> 用户 2026-09-04 指令：「跑完的最后一个 ckpt 如何 eval？官方 eval 的脚本是什么」→
> 确认范围「只评训练过的 4 任务」「只 59999，单 seed=7」，随后追加「尽可能并行 2 个 GPU
> 各一个模型」→ 定为两张卡各跑一条完整流水线（policy server + eval client 同卡），
> 4 个任务劈成两半并行，墙钟时间减半。

## 可复现锚

- 起跑 commit：`442a7b9847de531d9316c542a5e0880ee442c81a`（工作区 clean，AGENTS 12）。
- 被评 checkpoint：`v1-store/train-runs/mme_vla_suite/v1-prod-60k/59999`
  - 来源 run `v1-prod-60k`（slurm job 59345249，wandb run `utktmnx4`，2026-09-03 09:56 收官，
    `EXIT_CODE=0`），起跑 commit `d88185ca037b2e2dbf49373ee0c52aadd1cf2b42`。
  - 形状：Orbax OCDBT，`params/`（≈13.5 GB，**只有 EMA 权重**，`ema_decay=0.999`）+
    `assets/robomme/norm_stats.json`；`_CHECKPOINT_METADATA` 只声明 `assets` / `params`
    两个 handler，**无 `train_state`**——可推理、不可续训。
  - 名为 `59999` 而非 `60000`：`train.py` 的保存条件是
    `step % save_interval == 0 and step > start_step` 或 `step == num_train_steps - 1`，
    最后一步强制存盘。
  - `history_config.txt` 位于 ckpt 的 **parent** 目录（`.../v1-prod-60k/`），内容
    `perceptual-framesamp-context.yaml`；`policy_config.create_trained_policy` 读的正是这一层，
    与 `train_config.model.history_config` 不一致时以它为准。
- 入口（官方链路，无离线 eval，唯一指标是在线 rollout 成功率）：
  - server：`scripts/training/serve_policy.py`（仓库 uv venv / JAX）
  - client：`examples/robomme/eval.py`（micromamba `robomme` 环境 / SAPIEN 仿真）
  - 二者经 WebSocket 对接；`scripts/training/eval.sh` 是同一对命令的单流水线 tmux 封装，
    本轮因需双路并行未使用，命令行口径与之等价。

## 评测集口径

- split：**test**（`examples/robomme/env_runner.py` 里 `BenchmarkEnvBuilder(dataset="test")` 写死）。
- episode 数：每任务 **50**，来自 test metadata `record_count`；4 任务合计 **200 episodes**。
- difficulty 分布（四任务一致）：easy 26 / medium 12 / hard 12。
- **两个 seed 不是一回事**：
  - 每个 episode 的**环境 seed** 由 test metadata 写死（ButtonUnmask 580000–584900、
    VideoUnmask 560000–564900、ButtonUnmaskSwap 570000–574900、VideoUnmaskSwap 550000–554900），
    不受任何命令行参数影响，任何人任何次跑都是同一套场景。
  - `--seed=7` / `--args.model_seed=7` 是**策略侧采样种子**，只影响 policy server 推理与结果目录名。
- 单 episode 上限 `max_steps=1300` 仿真步，策略每 16 步推理一次
  （`exec_horizon = obs_horizon = 16`，`utils.check_args` 硬 assert `obs_horizon == 16`）。

## 双卡切分

| | 卡 0（splitA） | 卡 1（splitB） |
|---|---|---|
| server 端口 | 8021 | 8022 |
| 任务 | ButtonUnmask, VideoUnmask | ButtonUnmaskSwap, VideoUnmaskSwap |
| episodes | 100 | 100 |
| `--args.policy_name` | `v1-prod-60k-splitA` | `v1-prod-60k-splitB` |

- 劈分刻意每边一个 Button 类 + 一个 Video 类：Video 类有前置 video demo 帧、单 episode 更长，
  混搭比按类分组更均衡。
- `policy_name` **必须两边不同**：`eval.py` 的结果目录是
  `save_dir / policy_name / ckpt{id} / seed{seed}`，只由这三个参数决定；两条 client 若同名，
  会同时全量覆盖同一个 `progress.json`，互相抹掉结果。跑完再合并成标准布局。
- 显存：每卡 46 GB。`serve_policy.py` 不设 `XLA_PYTHON_CLIENT_MEM_FRACTION`，JAX 默认预分配 75%
  （≈34.5 GB）会挤压同卡 SAPIEN 渲染，故显式压到 **0.7**（≈32 GB），给仿真侧留 ≈14 GB。
- 本机资源：32 核 EPYC 9334 / 377 GB 内存 / 起跑前 load ≈2.6；两卡占用 1017 MiB、9 MiB。

## 启动命令（钉死口径）

```bash
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA

# server A（卡 0 / 8021）；server B 只换 CUDA_VISIBLE_DEVICES=1、--port=8022、日志名 -b
tmux new-session -d -s prod60k-eval-server-a \
  "set -o pipefail; PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
   uv run scripts/training/serve_policy.py --seed=7 --port=8021 \
     policy:checkpoint \
     --policy.dir=v1-store/train-runs/mme_vla_suite/v1-prod-60k/59999 \
     --policy.config=mme_vla_suite \
   2>&1 | tee v1-store/logs/prod60k-eval-server-a.log; \
   echo \"EXIT_CODE=\$?\" >> v1-store/logs/prod60k-eval-server-a.log"

# client A（卡 0 / 8021）；client B 只换 CUDA_VISIBLE_DEVICES=1、--args.port=8022、
# --args.policy_name=v1-prod-60k-splitB、--args.only_tasks=ButtonUnmaskSwap,VideoUnmaskSwap、日志名 -b
tmux new-session -d -s prod60k-eval-client-a \
  "set -o pipefail; PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
   /home/hongzefu/micromamba/envs/robomme/bin/python examples/robomme/eval.py \
     --args.port=8021 --args.policy_name=v1-prod-60k-splitA \
     --args.model_ckpt_id=59999 --args.model_seed=7 \
     --args.only_tasks=ButtonUnmask,VideoUnmask \
   2>&1 | tee v1-store/logs/prod60k-eval-client-a.log; \
   echo \"EXIT_CODE=\$?\" >> v1-store/logs/prod60k-eval-client-a.log"
```

- **cwd 必须是仓库根**：`--args.save_dir` 默认 `v1-store/evaluation` 是相对路径；
  `eval.py` 的 `from utils import ...` 靠 `sys.path[0]`（脚本所在目录）解析，
  `python examples/robomme/eval.py` 这种写法两者同时成立。
- client 用 micromamba 环境的**绝对路径 python**，不走 `micromamba activate`（tmux 里依赖 shell hook，
  不可靠）。这是 AGENTS 3「禁裸 python」的既有豁免：仿真依赖不在 uv venv，`eval.sh` 内已有同款注释
  （2026-08-30 用户拍板）。
- **不传 `--args.overwrite`**：它会 `shutil.rmtree` 整个 seed 目录；中断续跑时传它会删掉已评结果。
- `--args.use_history` 默认 True、`--args.obs_horizon` 默认 16，与训练侧 `--model.use-history` +
  `perceptual-framesamp-context.yaml` 一致，不覆盖。

## 环境事实（起跑前实测）

- micromamba `robomme`（`/home/hongzefu/micromamba/envs/robomme`，Python 3.11.15）可 import
  `robomme` 与 `openpi_client`。
- 但二者是 editable 安装，**指向另一个仓库副本** `/data/hongzefu/robomme_policy_learning-vqa-test/`，
  不是本仓库。本仓库的 `third_party/robomme_benchmark` submodule 为空（`git submodule status` 前缀 `-`），
  因 editable 指向别处，**不影响评估**；也**不得**为此 `git submodule update --init` 后重装 `robomme`——
  那会改变 editable 指向、引入版本差异。
- 已逐字节 diff 核对：`/data` 副本的 `openpi_client/websocket_client_policy.py` 与本仓库
  `packages/openpi-client/` 那份**完全相同**，`MMEVLAWebsocketClientPolicy` 的
  `reset` / `add_buffer` / `infer` 接口一致，两侧不会协议错配。

## Monitor 判据

- server 就绪判定行 **`Creating server`**：`serve_policy.py::main` 中该 `logging.info` 排在
  `create_policy(args)` 之后，即 13.5 GB 权重读完、模型建好才打印，紧接 `serve_forever()` 监听。
  以它为门，不用固定 `sleep`。
- client 进度标记：`[robomme] env for task <T> episode <N> setup finished`（每 episode 起点），
  用于数进度、量单 episode 耗时并外推总时长。
- 两条流水线四份日志各挂一个 Monitor（AGENTS：一份日志一个，禁止一条 tail 挂多文件）。

## 产物路径

```
v1-store/evaluation/v1-prod-60k-splitA/ckpt59999/seed7/{progress.json,log.json,videos/}
v1-store/evaluation/v1-prod-60k-splitB/ckpt59999/seed7/{progress.json,log.json,videos/}
                      ↓ 收尾合并（一次性脚本，不进仓库）
v1-store/evaluation/v1-prod-60k/ckpt59999/seed7/{progress.json,log.json,videos/}
v1-store/logs/prod60k-eval-{server,client}-{a,b}.log
```

合并后两个 split 原目录保留作原始凭据。**汇总不用 `scripts/training/compute_results.py`**：
它的 `TASK_NAME_LIST` 硬编码全 16 任务、缺失按 `success_rate.get(task_name, 0)` 补 0，
只评 4 任务时算出的 `Overall` 会被 12 个 0 稀释成假数；单 seed 时 std 还是 NaN。
直接读合并后的 `log.json`（其 `total_success_rate` 只对实际评过的任务求平均）。

## 中断与续评

`eval.py` 每评完一个 episode 就写 `progress.json`，重跑自动跳过已评 episode（README Q2：
长 horizon 任务 WebSocket 可能因大 video 帧断开）。注意两点：
`log.json` 一旦生成，重跑该侧 client 会因外层 `while not os.path.exists(save_dir / "log.json")`
直接空转退出，想重评必须先删它；某 episode 返回 `"unknown"` 时 `eval.py` 打印
`API calling error, aborting...` 并 return，那是该侧 server 挂了的信号，须先查 server 日志。
