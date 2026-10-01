# eval-orig80k-modul-vs-official — 结果

起跑记录见 [launch.md](launch.md)。评估代码锚点 HEAD `1af91f9`（commitV11.20Beta），smoke 与 24 片正式评估都在这个 clean HEAD 上跑，每片驱动日志首行 `=== EVAL_SHARD … HEAD=1af91f9…` 可证。

## 一、一句话结论与指标速览

**自训的 16 任务模型总体比官方 modul 80k 低 3.25pp（44.62% vs 47.88%，95% CI [−6.62, +0.12]，McNemar p=0.065），未检出显著差异。差距集中在 Behavior 类：低 8.00pp（CI [−13.5, −3.0]，p=0.0052），主要来自 MoveCube（−16pp）和 PatternLock（−16pp）。Persistent、Referential 两类与官方基本持平。只在 counting 4 任务上训练的模型，在这 4 个任务上与官方（70.0% vs 72.0%）和 16 任务模型（70.0% vs 67.0%）都未检出差异；唯一例外是它的 StopCube 明显好于 16 任务模型（68% vs 42%，p=0.00098）。**

| 模型 | 任务 | 集数 | 总体成功率 | Counting | Persistent | Referential | Behavior | timeout |
|---|---|---|---|---|---|---|---|---|
| 官方 modul 80k | 16 | 800 | **47.88%** | 72.00% | 25.50% | 40.00% | 54.00% | 20 |
| full16（`v2-orig-16task-pub1600ep-modul-b64-80k`） | 16 | 800 | **44.62%** | 67.00% | 26.00% | 39.50% | 46.00% | 21 |
| count4（`v2-orig-counting-pub400ep-modul-b64-80k`） | 4 counting | 200 | **70.00%** | 70.00% | — | — | — | 1 |

上表 suite 与总体的数字，由对比脚本和上游 `scripts/training/compute_results.py`（`--ckpt_list ckpt79999 --seed_list seed42`）分别算出，两边完全一致。

## 二、验收判定项（全部通过）

```
PARAM_TREE_EXACT=PASS ×3（official / full16 / count4，n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0）
smoke：EXIT_CODE=0 ×3，GPU0 峰值 71,932 MiB（< 75 GB）
正式：EXIT_CODE=0 ×24（records/logs/*.summary.log 逐份可核）
EVAL_ORIG80K_OFFICIAL=DONE tasks=16 episodes=800/800 errors=0 … mean_rate=0.4788 timeout=20/800 fail=397/800
EVAL_ORIG80K_FULL16=DONE tasks=16 episodes=800/800 errors=0 … mean_rate=0.4462 timeout=21/800 fail=422/800
EVAL_ORIG80K_COUNT4=DONE tasks=4 episodes=200/200 errors=0 … mean_rate=0.7000 timeout=1/200 fail=59/200
SPLIT_SEED_MATCH=PASS ×3（n=800 / 800 / 200，mismatch=0 unlogged=0）
VIDEOS=PASS official=800 full16=800 count4=200（与集数相等）
CMP_*：见第四节，每组都有 overall / suite / diff / task 四个层面
```

## 三、逐任务成功率（test split，seed 42，每任务 50 集）

| 任务 | 官方 | full16 | full16−官方 | count4 | count4−官方 | count4−full16 |
|---|---|---|---|---|---|---|
| BinFill | 46% | 42% | −4 | 38% | −8 | −4 |
| StopCube | 54% | 42% | −12 | 68% | +14 | **+26**（p=0.00098） |
| PickXtimes | 94% | 88% | −6 | 86% | −8 | −2 |
| SwingXtimes | 94% | 96% | +2 | 88% | −6 | −8（p=0.125，见注） |
| ButtonUnmask | 28% | 22% | −6 | | | |
| VideoUnmask | 30% | 32% | +2 | | | |
| VideoUnmaskSwap | 26% | 22% | −4 | | | |
| ButtonUnmaskSwap | 18% | 28% | +10 | | | |
| PickHighlight | 30% | 26% | −4 | | | |
| VideoRepick | 34% | 28% | −6 | | | |
| VideoPlaceButton | 52% | 58% | +6 | | | |
| VideoPlaceOrder | 44% | 46% | +2 | | | |
| MoveCube | 88% | 72% | **−16**（p=0.021） | | | |
| InsertPeg | 8% | 4% | −4 | | | |
| PatternLock | 58% | 42% | **−16**（p=0.021） | | | |
| RouteStick | 62% | 66% | +4 | | | |

按难度档（full16 vs 官方）：easy 54.57% vs 56.73%（−2.16），medium 41.67% vs 45.83%（−4.17），hard 26.04% vs 30.73%（−4.69），CI 都跨 0。各档都略低于官方，没有哪一档特别突出。

## 四、配对比较口径与解读注意

`scripts/training/legacy-eval/compare_eval_runs.py`：按 (task, ep) 逐集配对，三个模型的 env seed 已逐集核对一致。delta 取 A − B，95% CI 用配对 bootstrap（按集重采样 10000 次，seed 0），p 用精确 McNemar（双侧）。全部判定行见 `records/compare_full16_vs_official.txt` 与 `records/compare_count4.txt`，其中关键几行：

```
CMP_FULL16_VS_OFFICIAL scope=overall n=800 a=44.62 b=47.88 delta=-3.25 ci=[-6.62,+0.12] p=0.06503 a_only=79 b_only=105 verdict=NOT_DETECTED
CMP_FULL16_VS_OFFICIAL scope=suite:Behavior n=200 a=46.00 b=54.00 delta=-8.00 ci=[-13.50,-3.00] p=0.005223 a_only=7 b_only=23 verdict=B_BETTER
CMP_FULL16_VS_OFFICIAL scope=task:MoveCube n=50 a=72.00 b=88.00 delta=-16.00 ci=[-28.00,-4.00] p=0.02148 a_only=1 b_only=9 verdict=B_BETTER
CMP_FULL16_VS_OFFICIAL scope=task:PatternLock n=50 a=42.00 b=58.00 delta=-16.00 ci=[-28.00,-4.00] p=0.02148 a_only=1 b_only=9 verdict=B_BETTER
CMP_COUNT4_VS_OFFICIAL scope=overall n=200 a=70.00 b=72.00 delta=-2.00 ci=[-9.00,+5.00] p=0.6835 a_only=25 b_only=29 verdict=NOT_DETECTED
CMP_COUNT4_VS_FULL16 scope=overall n=200 a=70.00 b=67.00 delta=+3.00 ci=[-3.00,+9.50] p=0.4296 a_only=23 b_only=17 verdict=NOT_DETECTED
CMP_COUNT4_VS_FULL16 scope=task:StopCube n=50 a=68.00 b=42.00 delta=+26.00 ci=[+12.00,+40.00] p=0.0009766 a_only=14 b_only=1 verdict=A_BETTER
CMP_COUNT4_VS_FULL16 scope=task:SwingXtimes n=50 a=88.00 b=96.00 delta=-8.00 ci=[-16.00,-2.00] p=0.125 a_only=0 b_only=4 verdict=B_BETTER
```

解读时要注意三点：

1. **verdict 只看 CI 是否跨 0，而且没有做多重比较校正。** 16 个任务逐一检验时，Bonferroni 阈值是 0.05/16≈0.0031。MoveCube、PatternLock 的 p=0.021 过不了这个阈值，只能当作「值得看回放的线索」。Behavior 类的 p=0.0052 能过 4 个 suite 的阈值 0.0125。count4 vs full16 的 StopCube（p=0.00098）能过 4 个任务的阈值 0.0125。
2. **SwingXtimes（count4 vs full16）的 B_BETTER 不可信。** 不一致集只有 0 对 4，这种情况下 bootstrap 百分位区间偏窄，而精确 McNemar p=0.125，未达显著。以 McNemar 为准，记作未检出差异。脚本判据保持原样不改。
3. 只有 seed 42 一个 seed。「未检出差异」不等于两者等价。

## 五、回放视频（全部保留）

| 模型 | 视频目录（8 个分片各一） | 个数 | 字节 |
|---|---|---|---|
| 官方 | `v1-store/evaluation/eval-orig80k-official-modul-w{0..7}/ckpt79999/seed42/videos/` | 800 | 1,035,710,325 |
| full16 | `v1-store/evaluation/eval-orig80k-full16-w{0..7}/ckpt79999/seed42/videos/` | 800 | 1,037,271,087 |
| count4 | `v1-store/evaluation/eval-orig80k-count4-w{0..7}/ckpt79999/seed42/videos/` | 200 | 249,656,316 |

文件名格式为 `<Task>_ep<n>_<success|fail>_<goal>_<difficulty>.mp4`。同一任务的同一 ep，三个模型用的是同一个 env seed，可以直接并排看，例如 `MoveCube_ep*`、`PatternLock_ep*` 两个任务中官方成功而 full16 失败的 9 集。每集是哪个分片、成功与否，见 `records/<模型>/per_episode.json`。另外，merge 已把合并后的 `progress.json`、`log.json`、`shards.json` 写入 `v1-store/evaluation/<RUN_NAME>/ckpt79999/seed42/`，视频不做移动。

## 六、硬件与耗时

- 起跑 03:10:52 UTC，24 片 server 在 03:11:47 全部就绪，最后一片（官方 w1）在 05:22:48 结束，**总墙钟约 2 h 12 min**。
- count4 八片 20.9–30.9 min（每片 24–28 集）；官方八片 89.3–131.9 min，full16 八片 89.3–130.7 min（每片 96–112 集）。16 任务平均每集约 65–70 s，w3、w7 等片落到的长集较多，所以更慢。
- 每卡驻留 3 个 policy server（`POLICY_MEM_FRACTION=0.28`）加 3 个仿真进程，显存峰值 71.9–72.3 GB/卡。
- GPU 利用率：稳态窗口 03:45–04:35 UTC 均值 16.5–20.0%，0% 采样占比 1.9–4.8%（`nvidia-smi -lms 500`，见 `records/gpu_summary.txt`）。评测瓶颈在仿真端，不在 GPU。
- 每卡 3 进程并发时，推理 `infer` 中位约 180 ms（历史上单卡单 server 为 66 ms），每片的 TIMING 见各 summary.txt。
- smoke（GPU 0 三片各 1 集）：server 约 1 min 就绪；BinFill 一集约 95–105 s，RouteStick 一集约 220 s；smoke 总墙钟 241 s。

## 七、用户决策记录

1. 「给出方案 … 和官方80k的modul的eval 8卡并行 看差距 跑完保留回放」
2. counting 模型评测范围：「只评 4 个 counting 任务」
3. 评估代码落地方式：「拷回 examples/robomme 并提交」
4. 「没看懂 还是分两部分 评估代码从哪里来 训练结果放在policy learning这个repo？」→ 方案改为两部分体例后获批

## 八、计划外事件与处置

- **smoke 与 Beta 的先后顺序调换**：`eval_shard.local.sh` 的 clean HEAD 硬闸对 smoke 同样生效，所以改为先打 Beta，在同一 HEAD 上跑 smoke，通过后直接起正式评估，其间没有任何 commit。
- **smoke 产物已删**：本轮 3 个 `v1-store/evaluation/smoke-orig80k-*` 目录与 6 份日志已删除。删之前把清洗后的日志和 GPU 采样留进了 `records/smoke/`。9 月 25 日的两份 `smoke-orig80k-*-0925.driver.log` 属于别的任务，没有动。
- **监听换挂时漏报一个事件**：ev16off-w7 的结束事件恰好落在轮询监听换挂的间隙，事后核对其日志为 `EXIT_CODE=0`，24 片退出码逐一核对无误。
- **没有任何重跑**：没有基础设施故障，也没有为成绩重跑。
- **tmux**：24 个评估会话都自行退出。GPU 采样会话 `ev-orig80k-gpu` 按确切名 `kill-session` 删除，删前删后 `tmux ls` 的差集只有它，其他会话一个不少。

## 九、结论与下一步

- 在同一评测口径下，自训 16 任务模型的总体成功率比官方低约 3pp，未达显著。差距主要来自 Behavior 类的 MoveCube 和 PatternLock，这两个任务可以先对照回放（官方成功而 full16 失败的各 9 集）找原因。
- counting 专项模型在它的 4 个任务上与两者都未检出总体差异。它在 StopCube 上显著好于 16 任务模型，说明多任务混训可能稀释了 StopCube 这类计时停止行为。这一点只是单 seed 线索，要确认需要多 seed。
- 可选的后续（均未授权）：多 seed（如 0、7）复测以收紧 CI；评估 70k 等中间 checkpoint，看是否仍在上升；V11.18 两条训练的结果归档（训练 README）仍待另行补写。

## 十、归档文件清单（`records/`）

- `param_tree_{official,full16,count4}.json`：起跑前参数树核对
- `{official,full16,count4}/summary.txt`、`per_episode.json`：merge 输出，含逐任务 / 难度 / 分片墙钟 / TIMING 与逐集结果
- `compare_full16_vs_official.txt`、`compare_count4.txt`：全部 `CMP_*` 判定行
- `logs/{ev16off,ev16full,ev4cnt}-w{0..7}.summary.log`：24 份驱动日志的清洗版（tqdm 中间态已去掉，`EVAL_EPISODE` 与 `EXIT_CODE=` 行完整）
- `smoke/`：smoke 三片的清洗后日志与 GPU 0 采样
- `gpu_summary.txt`：8 卡显存峰值、利用率均值与 0% 占比（原始 csv 留 `v1-store/logs/ev-orig80k-gpu.csv`）
