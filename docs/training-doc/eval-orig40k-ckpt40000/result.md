# eval-orig40k-ckpt40000 — 结果

起跑记录见 [launch.md](launch.md)。评估代码与上一轮 [eval-orig80k-modul-vs-official](../eval-orig80k-modul-vs-official/result.md) 完全相同。本轮 HEAD 为 `3ed41ae`（commitV11.21Beta，clean），16 片驱动日志首行均记录 `HEAD=3ed41ae…`。

## 一、一句话结论与指标速览

**两条原版 run 的第 40000 步 checkpoint 都已评完。16 任务 run 从 40k 到 80k 总体提升了 4.0pp（40.62% → 44.62%，95% CI [+0.88, +7.00]，McNemar p=0.013，显著），提升主要在 easy 档（+7.2pp，p=0.0014）。40k 比官方 modul 80k 低 7.25pp（p=2.1e-5），其中 Behavior 类低 13.5pp、Counting 类低 10.5pp。counting run 从 40k 到 80k 总体 +3.0pp（67.0% → 70.0%，p=0.31），未检出差异，只有 StopCube 显著上升（52% → 68%，p=0.0078）。**

注意：这里的「40k」是两条 **80k 训练**（batch 64，总长 80000 步）中途第 40000 步保存的 checkpoint，不是一条单独训练 40k 步的 run。

## 二、五个 checkpoint 的完整分任务成功率（test split，seed 42，每任务 50 集）

「官方 80k」「16任务 80k」「counting 80k」三列来自上一轮评估；「16任务 40k」「counting 40k」两列是本轮结果。counting run 只评了它训练过的 4 个任务。

| 类别 | 任务 | 官方 80k | 16任务 40k | 16任务 80k | counting 40k | counting 80k |
|---|---|---|---|---|---|---|
| Counting | BinFill | 23/50 = 46% | 18/50 = 36% | 21/50 = 42% | 22/50 = 44% | 19/50 = 38% |
| Counting | StopCube | 27/50 = 54% | 13/50 = 26% | 21/50 = 42% | 26/50 = 52% | 34/50 = 68% |
| Counting | PickXtimes | 47/50 = 94% | 45/50 = 90% | 44/50 = 88% | 39/50 = 78% | 43/50 = 86% |
| Counting | SwingXtimes | 47/50 = 94% | 47/50 = 94% | 48/50 = 96% | 47/50 = 94% | 44/50 = 88% |
| Persistent | ButtonUnmask | 14/50 = 28% | 11/50 = 22% | 11/50 = 22% | — | — |
| Persistent | VideoUnmask | 15/50 = 30% | 17/50 = 34% | 16/50 = 32% | — | — |
| Persistent | VideoUnmaskSwap | 13/50 = 26% | 13/50 = 26% | 11/50 = 22% | — | — |
| Persistent | ButtonUnmaskSwap | 9/50 = 18% | 10/50 = 20% | 14/50 = 28% | — | — |
| Referential | PickHighlight | 15/50 = 30% | 8/50 = 16% | 13/50 = 26% | — | — |
| Referential | VideoRepick | 17/50 = 34% | 13/50 = 26% | 14/50 = 28% | — | — |
| Referential | VideoPlaceButton | 26/50 = 52% | 29/50 = 58% | 29/50 = 58% | — | — |
| Referential | VideoPlaceOrder | 22/50 = 44% | 20/50 = 40% | 23/50 = 46% | — | — |
| Behavior | MoveCube | 44/50 = 88% | 34/50 = 68% | 36/50 = 72% | — | — |
| Behavior | InsertPeg | 4/50 = 8% | 2/50 = 4% | 2/50 = 4% | — | — |
| Behavior | PatternLock | 29/50 = 58% | 17/50 = 34% | 21/50 = 42% | — | — |
| Behavior | RouteStick | 31/50 = 62% | 28/50 = 56% | 33/50 = 66% | — | — |
| **汇总** | Counting 类 | 144/200 = 72.0% | 123/200 = 61.5% | 134/200 = 67.0% | 134/200 = 67.0% | 140/200 = 70.0% |
| **汇总** | Persistent 类 | 51/200 = 25.5% | 51/200 = 25.5% | 52/200 = 26.0% | — | — |
| **汇总** | Referential 类 | 80/200 = 40.0% | 70/200 = 35.0% | 79/200 = 39.5% | — | — |
| **汇总** | Behavior 类 | 108/200 = 54.0% | 81/200 = 40.5% | 92/200 = 46.0% | — | — |
| **汇总** | **总体（16 任务）** | 383/800 = 47.9% | 325/800 = 40.6% | 357/800 = 44.6% | 仅 4 任务 | 仅 4 任务 |

按难度拆分（easy 26 / medium 12 / hard 12 集，每格为成功集数）见 `records/{full16,count4}/summary.txt`。16 任务 run 的难度汇总如下：

| 难度 | 官方 80k | 16任务 40k | 16任务 80k |
|---|---|---|---|
| easy（416 集） | 56.73% | 47.36% | 54.57% |
| medium（192 集） | 45.83% | 40.62% | 41.67% |
| hard（192 集） | 30.73% | 26.04% | 26.04% |

## 三、验收判定项（全部通过）

```
PARAM_TREE_EXACT=PASS ×2（full16@40000 / count4@40000，n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0）
smoke：EXIT_CODE=0 ×2（HEAD 655dfc6），GPU0 峰值 47,973 MiB
正式：EXIT_CODE=0 ×16（records/logs/*.summary.log）
EVAL_ORIG40K_FULL16=DONE tasks=16 episodes=800/800 errors=0 … mean_rate=0.4062 timeout=21/800 fail=454/800
EVAL_ORIG40K_COUNT4=DONE tasks=4 episodes=200/200 errors=0 … mean_rate=0.6700 timeout=0/200 fail=66/200
SPLIT_SEED_MATCH=PASS ×2（n=800 / 200，mismatch=0 unlogged=0）
VIDEOS=PASS full16=800 count4=200
```

## 四、配对比较（`compare_eval_runs.py`，逐集配对，delta = 前者 − 后者）

完整判定行见 `records/compare_full16.txt` 与 `records/compare_count4.txt`。主要几行：

```
CMP_FULL16_40K_VS_FULL16_80K scope=overall n=800 a=40.62 b=44.62 delta=-4.00 ci=[-7.00,-0.88] p=0.01281 a_only=62 b_only=94 verdict=B_BETTER
CMP_FULL16_40K_VS_FULL16_80K scope=diff:easy n=416 a=47.36 b=54.57 delta=-7.21 ci=[-11.30,-2.88] p=0.001404 a_only=27 b_only=57 verdict=B_BETTER
CMP_FULL16_40K_VS_FULL16_80K scope=suite:Counting n=200 delta=-5.50 ci=[-11.50,+0.00] p=0.08953 verdict=NOT_DETECTED
CMP_FULL16_40K_VS_FULL16_80K scope=suite:Behavior n=200 delta=-5.50 ci=[-11.00,+0.50] p=0.08953 verdict=NOT_DETECTED
CMP_FULL16_40K_VS_OFFICIAL scope=overall n=800 a=40.62 b=47.88 delta=-7.25 ci=[-10.50,-4.00] p=2.054e-05 a_only=62 b_only=120 verdict=B_BETTER
CMP_FULL16_40K_VS_OFFICIAL scope=suite:Behavior n=200 delta=-13.50 ci=[-19.50,-7.50] p=1.43e-05 verdict=B_BETTER
CMP_FULL16_40K_VS_OFFICIAL scope=suite:Counting n=200 delta=-10.50 ci=[-16.50,-4.50] p=0.00145 verdict=B_BETTER
CMP_FULL16_40K_VS_OFFICIAL scope=task:StopCube n=50 delta=-28.00 p=0.001312 / task:PatternLock delta=-24.00 p=0.001831 / task:MoveCube delta=-20.00 p=0.01294
CMP_COUNT4_40K_VS_COUNT4_80K scope=overall n=200 a=67.00 b=70.00 delta=-3.00 ci=[-7.50,+2.00] p=0.3075 verdict=NOT_DETECTED
CMP_COUNT4_40K_VS_COUNT4_80K scope=task:StopCube n=50 a=52.00 b=68.00 delta=-16.00 ci=[-26.00,-6.00] p=0.007812 a_only=0 b_only=8 verdict=B_BETTER
CMP_COUNT4_40K_VS_COUNT4_80K scope=task:PickXtimes n=50 delta=-8.00 ci=[-16.00,-2.00] p=0.125 a_only=0 b_only=4 verdict=B_BETTER（不一致集只有 0 对 4，以 McNemar 为准，记作未检出差异）
CMP_COUNT4_40K_VS_OFFICIAL scope=overall n=200 a=67.00 b=72.00 delta=-5.00 ci=[-12.00,+2.00] p=0.2116 verdict=NOT_DETECTED
```

解读口径与上一轮相同：只有一个 seed；逐任务检验没有做多重比较校正（16 个任务时 Bonferroni 阈值约 0.0031）；不一致集很少时 bootstrap CI 偏窄，以 McNemar 为准。按这个口径，经校正后仍然显著的有：

- 16任务 40k vs 80k 的 overall（p=0.013，只有一次检验）和 easy 档（p=0.0014）；
- 16任务 40k vs 官方的 overall、Behavior 类、Counting 类，以及 StopCube 和 PatternLock 两个任务（p≈0.0013–0.0018，16 个任务里低于 0.0031）；
- counting run 的 StopCube 从 40k 到 80k 上升（p=0.0078，4 个任务的阈值是 0.0125）。

## 五、回放视频（全部保留）

| 模型 | 视频目录 | 个数 | 字节 |
|---|---|---|---|
| 16任务 40k | `v1-store/evaluation/eval-orig40k-full16-w{0..7}/ckpt40000/seed42/videos/` | 800 | 1,048,125,007 |
| counting 40k | `v1-store/evaluation/eval-orig40k-count4-w{0..7}/ckpt40000/seed42/videos/` | 200 | 253,902,616 |

文件名格式为 `<Task>_ep<n>_<success|fail>_<goal>_<difficulty>.mp4`。与 80k 那一轮同 (task, ep) 的环境种子相同，可以直接并排对照。

## 六、硬件与耗时

- 16:03:47 UTC 起跑，16:04:27 时 16 片全部就绪，最后一片（full16 w3，大部分时间在跑 RouteStick 长集）于 18:21:30 结束，**总墙钟约 2 h 18 min**。
- count4 各片 16.5–25.1 min；full16 各片 70.0–137.7 min。
- 每卡 2 个 server（`POLICY_MEM_FRACTION=0.28`），显存峰值 47.9–48.2 GB/卡。
- GPU 稳态利用率（16:30–17:10）均值 6.6–7.9%，0% 采样占比 6.7–8.2%。机器 CPU 约 96% 空闲。
- **用户在跑动中指出「你为什么不能一起跑 现在gpu没吃满」。** 核实后确认是本轮并行度设低了：每个模型只开 8 个 worker（一卡一个），而每个 worker 是「仿真 → 推理 → 仿真」的串行循环，机器大部分资源空着。已记为教训：下次每个模型开 24–32 个 worker，先用 smoke 验证显存再上。本轮已跑 70%，所以没有中途重启。

## 七、用户决策记录

1. 「给我完整的分task*成功率的结果 三个model 注意两个是40k说清楚」→ 已核实并说明：上一轮三个模型评的都是 79999 步（80k），不是 40k。
2. 「再评一下两个run的40k checkpoint」→ 本轮。
3. 「还要跑多久」「跑到哪里了」→ 期间汇报进度。
4. 「你为什么不能一起跑 现在gpu没吃满」→ 见第六节。

## 八、计划外事件与处置

- smoke 在 `655dfc6` 上跑。之后才提交 Beta `3ed41ae`，两者之间只差文档，评估代码相同。
- smoke 脚本的 awk 按字符串比较取显存峰值，误报 859 MiB。改成数值比较后为 47,973 MiB；上一轮留档的 71,932 MiB 用数值比较复核后不变。
- 没有任何重跑。GPU 采样会话 `ev40-gpu` 按确切名删除，删前删后 `tmux ls` 的差集只有它。

## 九、结论与下一步

- 16 任务 run 从 40k 到 80k 仍在明显进步（+4pp，显著），主要体现在 easy 档。到 80k 时，与官方的差距从 −7.25pp 缩小到 −3.25pp（后者不显著）。继续训练或许还能再缩小差距，但这是推测，未验证。
- counting run 在 40k 时已接近它 80k 的水平，只有 StopCube 在后半程明显上升。
- 可选后续（未授权）：评估 60k / 70k 等中间 checkpoint 画出曲线（按新的并行度，预计每个 checkpoint 30–45 min）；多 seed 复测。

## 十、归档文件清单（`records/`）

- `param_tree_{full16,count4}.json`
- `{full16,count4}/summary.txt`、`per_episode.json`
- `compare_full16.txt`（40k vs 80k、40k vs 官方）、`compare_count4.txt`
- `logs/{ev40full,ev40cnt}-w{0..7}.summary.log`（16 份清洗后的驱动日志）
- `smoke/`（两片 smoke 日志与 GPU 0 采样）
- `gpu_summary.txt`（原始 csv 留在 `v1-store/logs/ev40-gpu.csv`）
