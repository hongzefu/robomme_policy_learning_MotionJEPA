# PatternLock、RouteStick 按子目标的真实 rollout 注意力：起跑档案

## 目的与授权

计划见 [1106-patternlock-routestick-subgoal-attention-plan.html](../../../1106-patternlock-routestick-subgoal-attention-plan.html)，修订记录见其第二部分第九节。

用户关键原话（美东 2026-11-06），按时间顺序：

1. 「两个 Unmask Swap 任务的这个注意力分析不是很明显……重新做一下这个分析。但这次我们分析 Pattern Lock 和 Root Stake 这两个任务。」
2. 「能不能够不分析关键点，就是在所有的动作的时候都分析。就不需要再造造这个关键位置了」「通过rollout」
3. 「你通过 sub-goal 不就可以看出来了吗？……它的 sub-goal 就是呃很明确的语义的边界，就我们只关注它不同 sub-goal 之间的这个动作。」
4. 「好的，可以开始了」
5. 「你能够在单卡上同时启动两个 worker 吗？然后尽可能先跑完一个模型，再跑另外一个模型，然后做简单的分片」「如果不够的话就把 worker 数量降低」
6. 「审查完毕之后直接开始，不用再来等我批准」

状态机：用户未答复是否登记，且要求不再等批准，所以不登记，按人工交接。

## 运行环境

环境 A，sled-aspen，两张 NVIDIA RTX A6000（每卡 46068 MiB），不连接集群。判据表写的是 `RTX 6000 Ada Generation`，与实测不符，已向用户报告，未裁决。输入与产物都在本机 `/data` NVMe。`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。现场有其他用户的进程，没有操作。

## 版本

| 项 | 版本 |
| --- | --- |
| 全部运行 | `6994a5943715a66f0d9cc0d9b7340179c379d164`，干净执行树 `v1-store/worktrees/subgoal-rollout-run` |
| benchmark | 主检出 `third_party/robomme_benchmark`，gitlink `856bc3a189d4172f3f47dbee4424d585f8d78db3` |

## 被跑对象

| 标记 | 权重 | 配置 | model_kind |
| --- | --- | --- | --- |
| 基线 ctx | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `mme_vla_suite_b128` | `history` |
| motion mot | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` | `stage2_pool4_concat` |

模型 seed 42。官方 test 局 0～49，两任务共 100 局，每个模型每局跑 1 次。

## 分片

| 片 | 局号 `episode_id%16` | 卡 | 局数（easy/medium/hard） |
| --- | --- | --- | --- |
| s0 | 0～3 | GPU0 | 28（16/6/6） |
| s1 | 4～7 | GPU0 | 24（12/6/6） |
| s2 | 8～11 | GPU1 | 24（12/6/6） |
| s3 | 12～15 | GPU1 | 24（12/6/6） |

先跑完 motion，再跑基线。每个 worker 显存比例 0.30。

## 冒烟结果

判定行见 [records/verification.summary.txt](records/verification.summary.txt)。GPU1 同时起 2 个 motion worker，比例 0.30，峰值 35274 MiB，其中其他用户约 1499 MiB；推算其他用户回到约 10 GB 时约 43.8 GB，低于 46068 MiB。这两局推理时段没有重叠，同卡提速未测到。

## 完整命令

产物根 `O=v1-store/evaluation/subgoal-rollout-attn-seed42-1106`，脚本 `$O/env.sh`、`$O/run_rollout.sh`、`$O/run_offline.sh`、`$O/gen_specs.py`、`$O/reset_hook_parity.py`。

```bash
bash $O/run_rollout.sh <mot|ctx> <gpu> <s0～s3> all
bash $O/run_offline.sh <mot|ctx> <gpu> <s0～s3> all
python -m scripts.interpret.analysis.subgoal_attention --root $O --out $O/analysis
```

## 时间线（美东 2026-11-06）

| 时刻 | 事件 |
| --- | --- |
| 17:29 | 钩子比较起跑 |
| 17:30 | 冒烟起跑：GPU1 同时 2 个 motion worker |
| 约 17:36 | 冒烟与分析全部通过 |

## tmux 会话清单

`sra-hook`、`sra-mem`、`sra-smoke-mot-s0`、`sra-smoke-mot-s1`、`sra-smoke-ctx-s0`、`sra-smoke-ctx-s1`、`sra-smoff-mot-s0`、`sra-smoff-mot-s1`、`sra-smoff-ctx-s0`、`sra-smoff-ctx-s1`（冒烟）；正式 `sra-roll-<模型>-<片>`、`sra-off-<模型>-<片>`、`sra-analysis`。

## 产物路径

`$O/<模型>/<片>/{attn,run}`、`$O/<模型>/offline`、`$O/analysis`，不进 git。git 只归档 `records/` 下的统计、图、清洗后日志和一次性诊断脚本。
