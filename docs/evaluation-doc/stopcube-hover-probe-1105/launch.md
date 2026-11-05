# StopCube 悬停期逐层探针起跑档案

## 目的与授权

计划见 [1105-stopcube-hover-probe-plan.html](../../../1105-stopcube-hover-probe-plan.html)。

用户原话：「细化计划 并且做对抗验证 不要添加额外的内容 不确定就问用户」（原文「永固」为错字，随后更正为「用户」）、「全部按推荐」「不登记状态机 开工」。

状态机：用户选择不登记，按人工交接。

## 运行环境

环境 A，sled-aspen，两张 RTX A6000，不连接集群。基线 ctx 用 GPU0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`），motion 和它的编码器侧车都用 GPU1（`0.55`）。输入和产物都在本机 `/data` NVMe。`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。现场有其他用户的小显存进程，没有操作。

## 版本

| 阶段 | 版本 |
| --- | --- |
| 计划（开工版本） | `cd469e2`，文字统一后 `f2811bf` |
| 标签生成、第一轮冒烟 | `87b45d0` |
| 第二轮 motion 冒烟、基线自检、全量回放、分析 | `84a6fe1141709373fbf10b0cb1a9d79912b9cbdb` |

执行树是干净的分离工作树 `v1-store/worktrees/stopcube-hover-run`，脚本起跑前核对 HEAD 和 `git status --porcelain` 为空。benchmark 使用主检出 `third_party/robomme_benchmark`，gitlink `856bc3a`。

## 被跑对象

| 标记 | 权重 | 配置 |
| --- | --- | --- |
| 基线 ctx | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `mme_vla_suite_b128` |
| motion mot | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` |

模型 seed 42。motion 的编码器走真实侧车，解释器 `v1-store/envs/stage2/bin/python`，运行契约 `v1-store/evaluation/interpret-1103/contracts/runtime-mot.json`（与上一轮 rollout 相同）。

输入 `/data/hongzefu/data_0226/record_dataset_StopCube.h5`，`H5_SOURCE=PASS sha256=f8be65f89eda55f41e51a148576f8ada4898d1a463abe87fd73cd0d471e1cadd`。

## 代码改动

W1 选点与标签（`commitV4.1.32`），W2 回放入口与前缀捕获（`V4.1.33`），W3 探针与出图（`V4.1.34`），W2-F1 传运行契约（`V4.1.35`）。全部经合并前后审查，`src/`、`third_party/`、`scripts/training/`、`scripts/evaluation/` 零改动。

## 完整命令

产物根 `O=v1-store/evaluation/stopcube-hover-probe-1105`，环境和启动脚本都在这里（`env.sh`、`smoke-*.sh`、`selftest-ctx.sh`、`run-*.sh`、`run-probe.sh`）。

```bash
python -m scripts.interpret.hover_points --h5 "$H5" --metadata <元数据> --output "$O/hover_points.json" --figures "$O/label-figures"

python -m scripts.interpret.hover_replay check --points "$O/hover_points.json" --h5 "$H5" --ckpt <权重> --config <配置> --label <模型> \
  --output "$O/smoke" --seed 42 --stage2-python <解释器> [--eval-target-contract <契约>] --episode 0 --rollout-attn <rollout 局目录>

python -m scripts.interpret.hover_replay run --points "$O/hover_points.json" --h5 "$H5" --ckpt <权重> --config <配置> --label <模型> \
  --output "$O/replay" --seed 42 --stage2-python <解释器> [--eval-target-contract <契约>]

python -m scripts.interpret.analysis.hover_probe --points "$O/hover_points.json" --replay-root "$O/replay" --output "$O/analysis" --workers 8
```

## 时间线（美东 2026-11-05）

| 时刻 | 事件 |
| --- | --- |
| 16:00 | 第一轮冒烟：`cc-hp-smoke-ctx`、`cc-hp-smoke-mot` |
| 16:12 | 基线 `--selftest` 在配额 0.45 下 OOM；基线 `check` 和单局回放继续通过 |
| 约 16:13 | motion 握手报 numerics 不符，原因是漏传运行契约 |
| 16:17 | 合并修复后重跑：`cc-hp-smoke-mot2`、`cc-hp-selftest-ctx`（配额 0.90） |
| 16:22 | 全量回放：`cc-hp-ctx`、`cc-hp-mot` |
| 约 16:43 | 基线回放结束，累计约 21 分钟 |
| 17:01 | motion 回放结束，累计约 38 分钟；分析 `cc-hp-probe` 起跑 |
| 17:05 | 分析结束，192 秒 |

## 本轮 tmux 会话

`cc-hp-smoke-ctx`、`cc-hp-smoke-mot`、`cc-hp-smoke-mot2`、`cc-hp-selftest-ctx`、`cc-hp-ctx`、`cc-hp-mot`、`cc-hp-probe`，全部自然退出。

## 产物

隐状态库 `$O/replay/{ctx,mot}/episode_<k>.npz`，各 1.7 GB；分析在 `$O/analysis/`。`records/` 归档标签表、汇总 JSON、三张图、两张标签抽查图和清洗后的日志。
