# StopCube 共同轨迹 Fig. 4 重跑（补动作 token）起跑档案

## 目的与授权

计划见 [1108-stopcube-common-trajectory-action-plan.html](../../../1108-stopcube-common-trajectory-action-plan.html)。稿件仓库原计划 `probe/1108-stopcube-common-trajectory-probe-plan.html`（`7007bc7`）。

用户原话（2026-11-08 美东，按时间顺序）：
1. 「上一次是在这里aspen跑的吗 给出细化方案 审查一下有什么阻塞 我起泡之后就要睡觉明天8点前不能打扰我」
2. 「你现在的读数是什么 画图给我」
3. 「按推荐开工」：推荐项为 FrameSamp 用 79999；冒烟 FAIL 停在冒烟、全量 FAIL 跑到底；沿用 1105 取点；不登记状态机。
4. 目标：「尽可能找到角度找到优势 在明天8点不要停止 不要来问我」「在明天8点尽可能穷尽解决方案」。

状态机：不登记，按人工交接。

开工审查：`PLAN_CONSISTENCY_REVIEW=FAIL base=6facf5f findings=7 blocking=3`。Finding 1（冒烟 FAIL 处理）与原话 4 冲突，按原话 4 裁定为「修复后重新冒烟，最多 2 轮，不停等」；Finding 2、3 在计划里写死定义；其余做文字统一。修订版 `0a38617`。

## 运行环境

环境 A，sled-aspen，两张 RTX A6000（46 GB），不连接集群。基线 ctx 用 GPU0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`），motion 和它的编码器侧车用 GPU1（`0.55`）。`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。

起跑时两张卡上都有其他用户（daiyp 的 robomme2 runner 与 mikasa_robo，frankcui 的 server）的进程，每张卡合计约 4–5 GB。本轮没有动这些进程；ctx 的配额从计划写的 0.90 降到 0.80，留出余量。

## 版本

| 阶段 | 版本 |
| --- | --- |
| 计划 | `6facf5f`，审查修订后 `0a38617` |
| W1 合入（回放加动作捕获） | `1e8b5df`（`commitV4.1.49`） |
| W2 合入（Fig. 4 三表导出） | `5f8a87b9a664b6835c9f66c122954272eb9f23fb`（`commitV4.1.50`） |
| 冒烟、全量、分析 | `5f8a87b`，执行树 `v1-store/worktrees/stopcube-common-run`（分离 HEAD，起跑前核对 HEAD 并确认 porcelain 为空） |

## 被跑对象

| 标记 | 权重 | 配置 |
| --- | --- | --- |
| 基线 ctx | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `mme_vla_suite_b128` |
| motion mot | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` |

模型 seed 42；motion 的编码器走真实侧车，运行契约 `v1-store/evaluation/interpret-1103/contracts/runtime-mot.json`。

输入：`/data/hongzefu/data_0226/record_dataset_StopCube.h5`（1105 判定行 `H5_SOURCE=PASS sha256=f8be65f8…`）；取点 `v1-store/evaluation/stopcube-hover-probe-1105/hover_points.json`，共 2344 个点、98 局有点。

## 完整命令

产物根 `O=v1-store/evaluation/stopcube-common-fig4-1108`，环境和启动脚本都在这里（`env.sh`、`smoke-*.sh`、`run-*.sh`、`repro-1105.sh`）。

```bash
python -m scripts.interpret.hover_replay check --points $H1105/hover_points.json --h5 $H5 --ckpt <权重> --config <配置> --label <模型> \
  --output $O/smoke --seed 42 --stage2-python <解释器> [--eval-target-contract <契约>] --episode 0 --rollout-attn <rollout 局目录> \
  --action-samples 5 --prefix-reference $H1105/replay

python -m scripts.interpret.hover_replay run --points $H1105/hover_points.json --h5 $H5 --ckpt <权重> --config <配置> --label <模型> \
  --output $O/replay --seed 42 --stage2-python <解释器> [--eval-target-contract <契约>] --action-samples 5

python -m scripts.interpret.analysis.common_traj_fig4 --points $H1105/hover_points.json --replay-root <库根> --output <目录> --workers 8 \
  [--repro-summary $H1105/analysis/hover-summary.json]
```

## 时间线（美东 2026-11-09）

| 时刻 | 事件 |
| --- | --- |
| 00:02 | 冒烟 `cc-cf-smoke-ctx`、`cc-cf-smoke-mot`，复现 `cc-cf-repro-1105` |
| 约 00:09 | 复现结束：`REPRO_1105=PASS maxabs=6.661e-16 compared=137 missing=0`，390 秒 |
| 约 00:10 | 两份冒烟全部 PASS；单局 42 个点，ctx 146 秒，mot 150 秒（动作部分 119／103 秒） |
| 00:11 / 00:12 | 全量 `cc-cf-ctx`、`cc-cf-mot` 起跑 |

## 本轮 tmux 会话

`cc-cf-explore1`、`cc-cf-smoke-ctx`、`cc-cf-smoke-mot`、`cc-cf-repro-1105`、`cc-cf-ctx`、`cc-cf-mot`（后续新增会话补记在 result.md）。

## 冒烟判定行

ctx：`REPLAY_PARITY=PASS layout_mismatch=0 points=21 maxabs=0`；`HIDDEN_PRECISION=PASS`（历史视觉 2598、当前视觉 4216、文字 6578）；`ACTION_PARITY=PASS points=21 maxabs=0`；`HIDDEN_PRECISION=PASS group=action min_dist_over_err=inf path_maxabs=0`；`PREFIX_MATCH=PASS points=42 maxabs=0`；`REPLAY_RUN=PASS model=ctx points=42 expected=42`。

mot：`REPLAY_PARITY=PASS layout_mismatch=0 points=21 maxabs=0`；`HIDDEN_PRECISION=PASS`（历史视觉 8625、motion 23951、当前视觉 5194、文字 6070、最近 motion 42693）；`ACTION_PARITY=PASS points=21 maxabs=0`；`HIDDEN_PRECISION=PASS group=action min_dist_over_err=inf path_maxabs=0`；`PREFIX_MATCH=PASS points=42 maxabs=0`；`REPLAY_RUN=PASS model=mot points=42 expected=42`。

## 加跑：MotionMem split 变体（美东 2026-11-09）

- 目的：检验三个模型的表征强弱是否与 StopCube 成功率对应（result.md 第三节 J）。对抗验证 `ADJUST_REVIEW=PASS scale=same findings=3`。
- 被跑对象：`orig100-ctx32x4x4-motion-s2split4x256-b128-60k` 59999，只读引用 turbo `robomme_policy_learning_MotionJEPA-eval/v1-store/assets/vla-o16ctx-59999-ada0653d…/59999`；运行合同 `official-stage2-1026/o16ctx-60k-seed42/contracts/runtime.json`；配置 `mme_vla_suite_orig16_b128_60k`。代码版本同上（`5f8a87b` 执行树）。
- 脚本：`smoke-split.sh`、`run-split-g0.sh`、`run-split-g1.sh`、`run-split-g1b.sh`（GPU1 接手 GPU0 末尾 41–46 局）。产物 `replay-split/mot/`。
- 时间线：04:00 前后冒烟，04:07 两卡全量起跑，约 05:31 全部结束。
- tmux 会话：`cc-cf-smoke-split`、`cc-cf-split-g0`、`cc-cf-split-g1`、`cc-cf-split-g1b`、`cc-cf-split-ana`，全部自然退出。
