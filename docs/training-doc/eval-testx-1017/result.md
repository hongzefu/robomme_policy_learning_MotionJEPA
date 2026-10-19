# eval-testx-1017 — 结果（60k 两模型；官方中止）

本文件是 AGENTS 第 12 条留档的第二段，接 [`launch.md`](launch.md)（起跑记录与冒烟）。时间一律美东。正式评估锚点 commitV11.27Beta `4aea1b29`。

## 一、一句话结论与指标速览

**在同一份 1200 局 test 布局清单上，512+motion@60k 全部成功率 32.0%，比 512@60k 的 25.8% 高 6.2 个百分点；xhard 五档（624 局）上 23.7% 对 16.7%，提升比 base 三档（41.0% 对 35.6%）更大。官方 FrameSamp 79999 按用户决定中途放弃，不出完整成功率。**

| 判定行 | 512@60k | 512+motion@60k |
|---|---|---|
| 分片合并 | `SHARDS_UNION=PASS total=1200` | `SHARDS_UNION=PASS total=1200` |
| seed 对齐 | `SEED_MATCH=PASS rows=1200` | `SEED_MATCH=PASS rows=1200` |
| 视频 | `EVAL_VIDEO=PASS videos=1200` | `EVAL_VIDEO=PASS videos=1200` |
| 终态 | `EVAL_512@60k=DONE episodes=1200 success=309 fail=797 timeout=94 error=0 downsampled=0` | `EVAL_512+motion@60k=DONE episodes=1200 success=384 fail=736 timeout=80 error=0 downsampled=1` |

两模型共用判定：`MANIFEST_SHARED=PASS models=2 manifest_sha256=20d4fbfb8430`、`EVAL_TABLE=DONE models=2 cells=100`。

| 口径 | 局数 | 512@60k | 512+motion@60k | 差 |
|---|---|---|---|---|
| base（easy／medium／hard） | 576 | 35.6%（205） | 41.0%（236） | +5.4 pp |
| xhard（x12–x20） | 624 | 16.7%（104） | 23.7%（148） | +7.0 pp |
| 全部 | 1200 | 25.8%（309） | 32.0%（384） | +6.2 pp |
| 全部，easy 计两次加权 | 1392 | 28.4%（395） | 34.6%（482） | +6.2 pp |

## 二、各任务各档成功率（%，「512 / 512+motion」）

每格 12 局；「—」表示该任务没有该档（xhard 只在 test metadata 追加过 seed 的任务上有）。任务按官网编号排序（P6）。

| 编号 | 任务 | easy | medium | hard | xhard12 | xhard14 | xhard16 | xhard18 | xhard20 |
|---|---|---|---|---|---|---|---|---|---|
| 1.1 | BinFill | 83 / 50 | 42 / 17 | 17 / 0 | 8 / 17 | 0 / 17 | 0 / 8 | 0 / 8 | 8 / 8 |
| 1.2 | PickXtimes | 83 / 92 | 92 / 58 | 25 / 17 | 33 / 67 | 8 / 42 | 25 / 17 | 25 / 17 | 17 / 17 |
| 1.3 | SwingXtimes | 100 / 75 | 100 / 100 | 100 / 100 | 83 / 58 | 83 / 42 | 58 / 50 | 58 / 67 | 58 / 42 |
| 1.4 | StopCube | 33 / 75 | 50 / 58 | 33 / 67 | — | 67 / 92 | 50 / 75 | 58 / 83 | 58 / 75 |
| 2.1 | VideoUnmask | 58 / 58 | 25 / 25 | 8 / 0 | — | 0 / 8 | — | — | — |
| 2.2 | ButtonUnmask | 33 / 50 | 17 / 25 | 17 / 0 | — | 0 / 0 | — | — | — |
| 2.3 | VideoUnmaskSwap | 17 / 33 | 25 / 33 | 25 / 17 | 0 / 0 | 0 / 8 | 8 / 8 | 8 / 8 | 0 / 0 |
| 2.4 | ButtonUnmaskSwap | 25 / 25 | 58 / 8 | 0 / 8 | 8 / 17 | 0 / 0 | 8 / 8 | 8 / 0 | 0 / 8 |
| 3.1 | PickHighlight | 42 / 33 | 17 / 25 | 0 / 25 | 0 / 0 | — | 0 / 17 | 0 / 17 | — |
| 3.2 | VideoRepick | 8 / 25 | 8 / 33 | 25 / 17 | 25 / 42 | 8 / 42 | 25 / 17 | 0 / 17 | 25 / 25 |
| 3.3 | VideoPlaceButton | 50 / 33 | 25 / 58 | 25 / 25 | 17 / 50 | 25 / 25 | — | — | — |
| 3.4 | VideoPlaceOrder | 17 / 8 | 25 / 25 | 17 / 42 | 0 / 17 | — | — | — | — |
| 4.1 | MoveCube | 92 / 83 | 75 / 83 | 83 / 67 | — | — | — | — | — |
| 4.2 | InsertPeg | 17 / 25 | 0 / 33 | 8 / 17 | — | — | — | — | — |
| 4.3 | PatternLock | 8 / 58 | 25 / 42 | 0 / 17 | 0 / 0 | 0 / 8 | 0 / 0 | 0 / 0 | 0 / 0 |
| 4.4 | RouteStick | 50 / 92 | 8 / 67 | 17 / 42 | 0 / 42 | 0 / 42 | 0 / 0 | 0 / 8 | 0 / 0 |

逐格成功数、timeout 数与 motion 抽稀局数见 `summary-512-vs-512m/summary.md`（归档清单）。

## 三、各档合计

| 档 | 局数（任务数） | 512@60k | 512+motion@60k |
|---|---|---|---|
| easy | 192（16） | 44.8% | 51.0% |
| medium | 192（16） | 37.0% | 43.2% |
| hard | 192（16） | 25.0% | 28.6% |
| xhard12 | 132（11） | 15.9% | 28.0% |
| xhard14 | 144（12） | 16.0% | 27.1% |
| xhard16 | 120（10） | 17.5% | 20.0% |
| xhard18 | 120（10） | 15.8% | 22.5% |
| xhard20 | 108（9） | 18.5% | 19.4% |

xhard 合计按任务看，512+motion 只在 SwingXtimes 上明显退步（68.3% → 51.7%，41/60 → 31/60）；其余有 xhard 档的任务持平或上升，升幅最大的是 StopCube（+22.9 pp）与 RouteStick（0/60 → 11/60）。

衰减图（benchmark 仓库主检出 `/data/hongzefu/robomme_benchmark_newtask-v3-MotionJepa1006/artifacts/test-xhard/eval-figs/`，在 sled-aspen 本机、不进 git）：

- `testx-decay-512-vs-512m.png`：按档（easy → xhard20）的成功率衰减曲线，两模型对照；
- `testx-decay-total-512-vs-512m.png`：同口径的总体衰减曲线（两模型合计行）。

两张图各有同名 `.svg`。

## 四、终态构成

| 模型 | success | fail | timeout | error | motion 抽稀局 |
|---|---|---|---|---|---|
| 512@60k | 309 | 797 | 94 | 0 | 0 |
| 512+motion@60k | 384 | 736 | 80 | 0 | 1（VideoPlaceOrder hard，按 T13 `MMEVLA_MOTION_OVERFLOW=resample` 抽稀到 160 窗） |

timeout 集中在 BinFill 高档、PickHighlight xhard16／xhard18、ButtonUnmaskSwap 与 InsertPeg；error 两模型均为 0。

## 五、官方 FrameSamp 79999（中止）

用户 2026-10-18 原话：「512+motion可以优先跑吗 官方的放弃不跑了」。处置：官方 10 片中 s0–s4 已完成（600 局，各单元 rc=0，`queue/done/` 有标记）；s5–s8 已被 worker 领取、在停官方时中途停止（结果目录 `testx-official/s5`–`s8` 为不完整半成品，不计入任何统计）；s9 以 `queue/claims/testx-official-s9/ABANDONED`（内容「blocked official-s9（用户 2026-10-18 放弃官方模型）」）挡住、未领取。**不出官方的完整成功率，也不把 s0–s4 的局部数字与两模型对比。** 状态机里 `EVAL_OFFICIAL` 记 `skipped`。

## 六、资源与释放

12 个占位 job（64233473–64233480、64233525–64233528，清单 `v1-store/logs/hold-jobs-testx.txt`）全部 CANCELLED，分两批：

1. 停官方时释放 4 个：64233478、64233526、64233527、64233528；
2. 用户随后原话「不许cancel」「那不动了」，其余 8 个保留；之后用户原话「站位的卡可以停了。」「跑完了就可以停。」，在两模型全部单元完成后释放其余 8 个。

两批都在对应 job 内全部 worker 退出、无 srun step 后才按清单逐个 scancel。

## 七、执行中调整与事件

1. **10k 冒烟 sidecar 锁旧 encoder**：V11.24Beta 下 512+motion@10k 第一次冒烟，sidecar 因 `ASSETS_LOCK` 仍指旧 encoder 触发 `SystemExit`。处置：V11.25Beta 把锁改指新 encoder `wan-mixed8-200-filter2-stop075-b88x8-72ep-a`，第一次的产物以 `.fail-v1124beta` 后缀保留；对抗验证 `ADJUST_REVIEW=PASS`。证据见 `launch.md` 七节。
2. **官方 sha 判据前缀误抄**：计划里官方 ckpt 判据前缀写成 `387d4bd5`（context 版对象），与实际下载的权重对象不符。处置：改核 HF LFS 对象前缀 `86ef2b63`，`OFFICIAL_CKPT_SHA=PASS files=19`。
3. **视频名 ENAMETOOLONG**：V11.26Beta 正式轮视频文件名含 task goal，部分任务超 255 字节报 `ENAMETOOLONG`，512 的 s0–s2 rc=1。处置：停 12 个 worker；15 个已领取单元（512 s0–s9、512+motion s0–s4）的 claim 与结果目录改名 `.abort-v1126beta` 保留（claim 移入 `queue/retired/`）；FIX2 把视频名改为计划格式 `<Task>_ep<n>_<终态>_<难度>.mp4`，合入为 V11.27Beta（`ADJUST_REVIEW=PASS`），正式单元从头重跑。
4. **跨主机改名与登录节点 NFS 缓存**：上条改名与 GL 侧 worker 不在同一主机，加上登录节点的 NFS 缓存，导致新一轮的部分 done 标记与单元日志写进了改名后的旧文件——`testx-512-60k-s2`、`testx-512m10k-smoke2-s0` 的 done 标记，`testx-512m-60k-s2`／`s4` 的单元日志写进 `*.abort-v1126beta.log`。处置：done 标记已在 GL 侧补回（`queue/done/` 现有 27 个标记，含这两条）；结果目录与每局产物不受影响。查这两片的单元日志时去 `.abort-v1126beta.log` 里找新一轮的内容。
5. **10k 冒烟规模**：按用户「10k 100 局冒烟降低到和官方一样的2局」改为 2 局（BinFill easy ep0 与 xhard12 ep50）。
6. **占位 job 8 → 12**：按用户「同意你再追加四个 job 去并行。」追加 64233525–64233528。

## 八、用户决策原话（本段新增）

1. 2026-10-18：「512+motion可以优先跑吗 官方的放弃不跑了」
2. 2026-10-18：「不许cancel」「那不动了」
3. 2026-10-18：「站位的卡可以停了。」「跑完了就可以停。」
4. 2026-10-18：「80k的两组也要跑 优先跑motion跑完了再跑512 用10卡 发信息和训练session沟通等预估马上跑完之前开始排队job提前15分钟」

起跑前的决策原话见 `launch.md` 二节。

## 九、结论与下一步

- 60k 上 motion 分支整体有效：全部 +6.2 pp，xhard 上 +7.0 pp，且 xhard12／xhard14 上差距最大（+12.1／+11.1 pp）；xhard20 两者接近（18.5% 对 19.4%）。SwingXtimes 是唯一明显退步的任务，值得在 80k 上复查。
- 每格只有 12 局，单格差 1 局即 8.3 pp，逐格比较只作参考，结论以档合计与全部合计为准。
- 下一步：评估 80k（两条 run 的 step 79999），状态机任务 `eval-testx-80k`；顺序 512+motion 在前、512 在后，10 卡；训练侧预计写出前 15 分钟提交占位 job。

## 十、归档清单（NFS，不进 git）

根目录 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/`：

| 路径 | 内容 |
|---|---|
| `testx/summary-512-vs-512m/summary.md`、`summary.json` | 两模型 100 格对比表与逐格计数 |
| `testx/summary-512only/summary.md`、`summary.json` | 只含 512@60k 的表 |
| `testx/eval-testx-1200.jsonl`、`manifest.tsv`、`manifest_meta.json` | 共用清单（sha256 `20d4fbfb8430…`）与单元队列 |
| `testx/queue/{claims,done,retired}/` | 单元领取、完成与作废标记（含官方 s9 的 `ABANDONED`） |
| `testx-512-60k/`、`testx-512m-60k/` | 每局 mp4、`steps/*.jsonl` 逐步动作日志、分片结果 |
| `testx-official/s0`–`s4` | 官方已完成的 5 片（不进统计）；`s5`–`s8` 为中止半成品 |
| `../logs/policy-eval/` | worker 与单元日志 |
| `../logs/hold-jobs-testx.txt` | 12 个占位 JobID 清单 |

## 十一、80k 与 40k 补充结果（2026-10-19 补写）

同一份 1200 局清单（`MANIFEST_SHARED=PASS models=6 manifest_sha256=20d4fbfb8430`）、同一链路（seed 7、max_steps 2000），在 10 个占位 job 64243171–64243180 上依次跑完四组，40 个单元全部 `rc=0`：

| 组 | 起止（美东，括注 UTC） | 判定行 |
|---|---|---|
| 512+motion@80k（V11.28Beta `a8c9f03`） | 10-18 16:06 → 19:47（20:06:16Z → 23:47:24Z） | `EVAL_512+motion@80k=DONE episodes=1200 error=0` |
| 512@80k | 10-18 19:13 → 20:55（23:13:59Z → 00:55:39Z） | `EVAL_512@80k=DONE episodes=1200 success=321 fail=790 timeout=89 error=0` |
| 512+motion@40k（V11.29Beta `f98a988`） | 10-18 20:56 → 10-19 00:23（00:56:51Z → 04:23:04Z） | `EVAL_512+motion@40k=DONE episodes=1200 success=366 fail=770 timeout=64 error=0 downsampled=1` |
| 512@40k | 10-18 23:58 → 10-19 01:35（03:58:51Z → 05:35:03Z） | `EVAL_512@40k=DONE episodes=1200 success=276 fail=829 timeout=95 error=0` |

四组均 `SHARDS_UNION=PASS total=1200`、`SEED_MATCH=PASS rows=1200`、`EVAL_VIDEO=PASS videos=1200`；`EVAL_TABLE=DONE models=6 cells=100`。

六模型按档成功率（%；局数 easy／medium／hard 各 192，xhard12 132、xhard14 144、xhard16 与 xhard18 各 120、xhard20 108）：

| 档 | 512@40k | 512@60k | 512@80k | 512+motion@40k | 512+motion@60k | 512+motion@80k |
|---|---|---|---|---|---|---|
| easy | 47.9 | 44.8 | 43.2 | 47.9 | 51.0 | 50.0 |
| medium | 31.2 | 37.0 | 40.1 | 43.8 | 43.2 | 40.6 |
| hard | 20.8 | 25.0 | 22.4 | 26.6 | 28.6 | 29.7 |
| xhard12 | 13.6 | 15.9 | 21.2 | 27.3 | 28.0 | 24.2 |
| xhard14 | 18.1 | 16.0 | 19.4 | 21.5 | 27.1 | 25.7 |
| xhard16 | 9.2 | 17.5 | 20.0 | 17.5 | 20.0 | 26.7 |
| xhard18 | 11.7 | 15.8 | 15.8 | 23.3 | 22.5 | 26.7 |
| xhard20 | 13.9 | 18.5 | 17.6 | 21.3 | 19.4 | 16.7 |
| base 三档 | 33.3 | 35.6 | 35.2 | 39.4 | 41.0 | 40.1 |
| xhard 五档 | 13.5 | 16.7 | 18.9 | 22.3 | 23.7 | 24.2 |
| **全部** | **23.0** | **25.8** | **26.8** | **30.5** | **32.0** | **31.8** |
| easy 加权两倍 | 26.4 | 28.4 | 29.0 | 32.9 | 34.6 | 34.3 |

结论：三个 step 上 motion 都领先（全部 +7.5／+6.2／+5.0 pp，xhard +8.8／+7.0／+5.3 pp）；512 随步数单调上升（23.0 → 25.8 → 26.8），motion 在 60k 与 80k 持平（32.0 对 31.8）、40k 已到 30.5。逐格表见 `v1-store/evaluation/testx/summary-6models/summary.md`。

用户决策原话：2026-10-18「这两个评估完了之后，再跑四十 K 的两个」「必须先跑这个带 motion 的。」

下一步：官方 eval seed 800 局（seed 42、max_steps 1300）评 512+motion 的 80k 与 40k，见 `docs/training-doc/eval-off800-1019/`。
