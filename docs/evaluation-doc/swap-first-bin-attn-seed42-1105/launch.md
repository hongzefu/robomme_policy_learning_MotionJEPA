# 两个 Swap 任务第一次取 bin 的注意力：起跑档案

## 目的与授权

计划见 [1105-swap-first-bin-attention-plan.html](../../../1105-swap-first-bin-attention-plan.html)，实施中调整见其第二部分末节。

用户关键原话，按时间顺序：

1. 「能不能所有五十局都跑关键决策点……一路往下找到这个第一个第一个病，并且把它拿起来的这么一个过程。」
2. 「就是按照第一个bin选对还是选错来判断……你应该把它截断到靠近、抓取，就是假爪闭合就算结束了」
3. 「然后你需要再跑一下 video and mask swap 任务也要跑」
4. 「然后总结一下这两个任务的这个注意力机制，看看能不能找到一些一些这个一些 pattern。」
5. 「不应该并行两个模型，而是应该一个模型分在两张卡跑」
6. 「静态的交替的分片就可以了，不要做动态的分配机制」
7. 「注意一下有没有其他的这个注意力头的这么一个峰值。在测试我们现在两个任务的时候，这个峰值可能会改变。」
8. 「这次用户批准可以直接开始，不需要再来问用户。但是呢，你需要看一下有什么问题，有问题的尽可能继续做下去，不要阻塞，然后再扫一遍有没有什么问题，尽可能在今天晚上十二点之前告诉我还有什么，还有些什么问题，然后之后的话就尽可能不要再阻塞。」
9. 「如果有其他用户占了卡的话，你就尽可能地避开。尽可能还是做完」

状态机：用户未答复是否登记，且要求不阻塞，所以不登记，按人工交接。

## 运行环境

环境 A，sled-aspen，两张 NVIDIA RTX A6000（每卡 46068 MiB），不连接集群。判据表写的是 `RTX 6000 Ada Generation`，与实测不符，已向用户报告，未裁决。输入与产物都在本机 `/data` NVMe。另一位用户 daiyp 同时使用两张卡（实测 2～8 GB），不操作其进程。

`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。显存比例 motion 0.55、基线 0.40。缓存环境变量与 `stopcube-rollout-1105` 相同。

## 版本

| 阶段 | 版本 |
| --- | --- |
| rollout、离线注意力 | `7c63418901f3eea016b40d09cf4588d7cc4b4377`，干净执行树 `v1-store/worktrees/swap-first-bin-run` |
| 窗口、分析、对照 | `3404326332dbefc9afdf757bcbeb36c57d01a0bb`，干净执行树 `v1-store/worktrees/swap-first-bin-analysis` |
| benchmark | 主检出 `third_party/robomme_benchmark`，gitlink `856bc3a189d4172f3f47dbee4424d585f8d78db3` |

## 被跑对象

| 标记 | 权重 | 配置 | model_kind |
| --- | --- | --- | --- |
| 基线 ctx | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `mme_vla_suite_b128` | `history` |
| motion mot | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` | `stage2_pool4_concat` |

模型 seed 42；环境 seed 与难度逐局取官方 test 元数据。

## 分片与顺序

| 清单 | 内容 | 封签 |
| --- | --- | --- |
| `manifest-g0.json` | 两任务偶数局各 25，每任务 5 块 × 5 局 | `b188d4e7a569e32a37d881f9b688e6ff2b41c3f3a35dc3cd7da21b5783da7467` |
| `manifest-g1.json` | 两任务奇数局各 25，每任务 5 块 × 5 局 | `899c56e947d8c33792851ff5d4bf9f59dfd158540bfe6788a8c0a1cbd8e487b6` |
| `manifest-smoke.json` | ButtonUnmaskSwap:8、VideoUnmaskSwap:0 | `d27ca35fb0226ebf8314bdcb9dac13100cd48503fbf0691939a23032716b3a4f` |

一次一个模型占两卡：先 motion（g0 → GPU0，g1 → GPU1），两片退出后再基线；离线相同。

## 完整命令

产物根 `O=v1-store/evaluation/swap-first-bin-attn-seed42-1105`，环境见 `$O/env.sh`。

```bash
# 正式 rollout（规格 $O/spec-<片>-<模型>.json：bin_trace=true、rollout_record=true、save_prefix=false）
python -m scripts.interpret.run_local --spec "$O/spec-g0-mot.json"

# 窗口与标签
python -m scripts.interpret.first_bin_windows --run-root "$O/mot/g0/run" --run-root "$O/mot/g1/run" --output "$O/mot/windows"

# 离线注意力（每片一个进程）
python -m scripts.interpret.first_bin_offline --windows "$O/mot/windows/windows.json" \
  --run-root "$O/mot/g0/run" --run-root "$O/mot/g1/run" --attn-root "$O/mot/g0/attn" --attn-root "$O/mot/g1/attn" \
  --output "$O/mot/offline" --ckpt <权重> --config <配置> --seed 42 --shard g0

# 上一轮对照（CPU）
python scripts/interpret/analysis/first_bin_attention.py --bridge-old22 v1-store/evaluation/attn-buttonunmaskswap-success-seed42-1103 --out "$O/bridge" --workers 4
```

## 本轮 tmux 会话清单

`sfb-smoke-mot`、`sfb-smoke-ctx`、`sfb-offsmoke-mot`、`sfb-offsmoke-ctx`、`sfb-an-smoke`、`sfb-bridge`、`sfb-roll-mot-g0`、`sfb-roll-mot-g1`，后续预留 `sfb-roll-ctx-g0`、`sfb-roll-ctx-g1`、`sfb-off-mot-g0`、`sfb-off-mot-g1`、`sfb-off-ctx-g0`、`sfb-off-ctx-g1`、`sfb-analysis`。

## 冒烟结果（美东 2026-11-05）

| 项目 | 判定行或结果 |
| --- | --- |
| motion rollout | `RUN_LOCAL=PASS model=stage2_pool4_concat episodes=2 success=2 npz=37` |
| 基线 rollout | `RUN_LOCAL=PASS model=history episodes=2 success=1 npz=27` |
| motion 标签 | `FIRST_BIN_LABEL=PASS episodes=2 correct=2 wrong=0 no_pick=0 before_buttons=0 geom_mismatch=0 wrong_but_success=0` |
| 基线标签 | `FIRST_BIN_LABEL=PASS episodes=2 correct=1 wrong=1 no_pick=0 before_buttons=0 geom_mismatch=0 wrong_but_success=0` |
| motion 离线 | `ROLLOUT_REPLAY=PASS points=8 same_path_maxabs=0 tol=1e-6`；`ATTN_WINDOW=PASS records=8 rowsum_err=0.000409603119 attn_path_maxabs=0.00390625` |
| 基线离线 | `ROLLOUT_REPLAY=PASS points=8 same_path_maxabs=0 tol=1e-6`；`ATTN_WINDOW=PASS records=8 rowsum_err=0.000390172005 attn_path_maxabs=0.00390625` |
| 分析链路 | `PATTERN_SUMMARY=PASS tasks=2 models=2 readouts=12`（只验证链路，2 局不解读） |

主会话目视：Video 第 0～31 帧方块露出，第 32 帧起被容器盖住，演示段（红框）到第 113 帧，第 114 帧起模型控制。基线 Video 第 0 局抬起后露出红色方块而目标是绿色，确认 `wrong`；motion Button 第 8 局露出绿色方块，确认 `correct`。Video 第 0 局第 178 步 motion 窗起点为 0、32、64、96、114、146，都是全局帧号，且不含 160。

计划外事件：第一次冒烟时 `HF_HOME` 写错，停掉 tmux 后基线策略服务（PID 2113726）成为孤儿并占 18 GB，导致重跑的基线冒烟 SAPIEN 报 `CUDA ... out of memory`（`client_exit=-11`）。核实归属后只结束该 PID，修正环境变量后重跑通过。失败的第一次产物已删除，日志未保留。

正式 motion 两片于美东 23:42 起跑。

## 替换实验（第二阶段）

用户原话：「做完了之后，可以尝试像上一次一样，就是做一些 masking 和替换的工作……只能接受把 motion token 换成别的相邻的 token 这种形式。」

版本 `81c7c674076a1a547248516034db5811ca6d5016`，干净执行树 `v1-store/worktrees/swap-first-bin-exp`。只跑 motion 模型，显存比例 0.55。供体由 `records/derive_motion_donors.py` 从正式 rollout 提取，映射为同任务、同交换次数、同 bin 数组内循环的下一局。

| 条件 | `ablate` | 规格 |
| --- | --- | --- |
| E0 | 不设 | `exp/spec-e0-g<n>.json` |
| E1 | `motion_swap:exp/donors/full` + `swap_map` | `exp/spec-e1-g<n>.json` |
| E3 | `motion_shift` | `exp/spec-e3-g<n>.json` |
| E2 | `motion_swap_exact:exp/donors/swap` + `swap_map` | `exp/spec-e2-g<n>.json` |

驱动 `exp/run_exp_chain.sh` 依次运行，美东 07:30 后不起新条件。tmux：`sfb-e1-smoke`、`sfb-e2-smoke`、`sfb-e3-smoke`、`sfb-exp-chain`、`sfb-off-chain`、`sfb-analysis`。

计划外事件：基线和 motion 的 `.log` 被 gitignore 拦下，清洗后日志改存 `.summary.txt`；合并时一次重复使用版本号 `commitV4.1.35`，历史未改写，后续跳过 36。
