# gl692-ctx1024-50k 评估结果（第二段，2026-11-20 美东）

> 第一段见同目录 [`launch.md`](launch.md)；被评对象是训练 run [`v2-1600ep-m16x8x8-ctx-b128-50k-gl`](../v2-1600ep-m16x8x8-ctx-b128-50k-gl/result.md) 的 50000 步 ckpt；计划为仓库根 `1116-gl-4task400-ctx-692-eval-plan.html`。姊妹评估见 [`../gl692-ctx512-50k/result.md`](../gl692-ctx512-50k/result.md)。

## ① 一句话结论与指标速览

context 1024（16 帧 × 8×8，无 motion）@50000 在 692 局上成功 **160/692 = 23.12%**，比同库、同 692 局、同 seed 7、同为 GL A40 的 modulation 1024 @50000 对照（[`finalckpt-modul16frame50000-gl`](../finalckpt-modul16frame50000-gl/result.md)，291/692 = 42.05%）低 **18.93 pp**。本次 `new_errors=0`，按计划 V7 可以和对照直接比较。差距来自 RouteStick（低 32.5 pp）、BinFill（低 31.0 pp）和 VideoRepick（低 14.7 pp），VideoUnmaskSwap 两者持平。context 1024 与 context 512（158/692 = 22.83%）基本相同：modulation 从 512 到 1024 涨了 11.6 pp，context 加倍 budget 几乎没有收益。

```
EVAL692=PASS episodes=692 success=160 rate=0.2312 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=0
MERGE_OK shards=8 episodes=692 successes=160 errors=9
```

## ② 版本与代码状态（与第一段的差异）

和 ctx512 相同：正式评估实际用的代码是 **`4ba8b49`**，不是 launch.md 写的 97f237ac。评估副本目录名仍是 `.../eval-wt/97f237ac`，已 `checkout --detach 4ba8b49`；链前缀用 `/usr/bin/env`，并加了 `PYTHONPATH=$W`。这三处修复都在 ctx512 评估时完成，见 `../gl692-ctx512-50k/result.md` ⑩。本条链在第 1 关等待期间已换上修复后的脚本，全程没有再出故障。

## ③ 运行过程（美东）

| 时刻 | 事件 |
|---|---|
| 11-17 18:32 | 换脚本后链重起，第 1 关 `CHAIN_WAIT stage=train` |
| 11-20 06:30 | 训练 `EXIT_CODE=0` |
| 06:31 | `TRAIN_DONE=PASS ckpts=10` |
| 06:31–06:36 | g6b：`EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=35.6`（单片 174 s，同卡两片墙钟 127 s） |
| 06:38 | 压测：`EVAL_CAPACITY=PASS per_gpu=2 peak_mem_gb=35.6 free_gb=9.4 host_delta_gb=25.5`、`HOST_MEM=PASS projected_gb=102.0 limit=115`、`CAP_TIMEOUT=PASS wall_s=122 projected_chunk_s=4880 limit=14400` → `CHAIN_EVAL per_gpu=2 frac=0.38 shards=8` |
| 06:38 | `PLAN_OK shards=8 total=692 groups=14`、`PRESEED=PASS episodes=9 placed=9 duplicates=0`，8 片起跑 |
| 约 07:30 | 8 片 `SHARD_EXIT_CODE=0`，8 片 `CHECK_SHARD ... SHARD_PASS` |
| 07:33 | `MERGE_OK`、`EVAL692=PASS`、`CHAIN_DONE tag=1024`，链 `EXIT_CODE=0` |
| 07:34 | 按 R3（PASS 且 new_errors=0）`scancel 65943151`，sacct `CANCELLED+`，job 共运行 8-00:53:27 |

正式 692 局约 55 分钟跑完：每卡 2 片，共 8 片，每片约 87 局。

## ⑦⑧ 结果：逐格与对照

对照是一次跑完的 692 局（`finalckpt-modul16frame50000-gl`，GL A40、每卡 1 片、10 片）。14 个组的局号集合与本次完全一致（`KEYSET_MISMATCH_GROUPS=0`），对照的 9 个 error 局与本次预标的 9 局为同一批。

**按任务**

| 任务 | 局数 | context 1024 | modulation 1024 对照 | 差 | 两者都成功 | context 512（参考） |
|---|---|---|---|---|---|---|
| BinFill | 142 | 37 (26.1%) | 81 (57.0%) | −31.0 pp | 28 | 57 (40.1%) |
| RouteStick | 200 | 60 (30.0%) | 125 (62.5%) | −32.5 pp | 59 | 25 (12.5%) |
| VideoRepick | 150 | 25 (16.7%) | 47 (31.3%) | −14.7 pp | 14 | 38 (25.3%) |
| VideoUnmaskSwap | 200 | 38 (19.0%) | 38 (19.0%) | +0.0 pp | 18 | 38 (19.0%) |
| **合计** | **692** | **160 (23.12%)** | **291 (42.05%)** | **−18.93 pp** | 119 | 158 (22.83%) |

**按难度**

| 难度 | 局数 | context 1024 | modulation 1024 |
|---|---|---|---|
| easy | 199 | 86 (43.2%) | 115 (57.8%) |
| medium | 199 | 44 (22.1%) | 106 (53.3%) |
| hard | 144 | 12 (8.3%) | 47 (32.6%) |
| xhard | 150 | 18 (12.0%) | 23 (15.3%) |

**14 组逐格**

| 组 | 局数 | context 1024 | modulation 1024 | 差 |
|---|---|---|---|---|
| BinFill/easy | 49 | 30 (61.2%) | 40 (81.6%) | −20.4 pp |
| BinFill/medium | 49 | 6 (12.2%) | 28 (57.1%) | −44.9 pp |
| BinFill/hard | 44 | 1 (2.3%) | 13 (29.5%) | −27.3 pp |
| RouteStick/easy | 50 | 33 (66.0%) | 49 (98.0%) | −32.0 pp |
| RouteStick/medium | 50 | 21 (42.0%) | 45 (90.0%) | −48.0 pp |
| RouteStick/hard | 50 | 6 (12.0%) | 30 (60.0%) | −48.0 pp |
| RouteStick/xhard | 50 | 0 (0.0%) | 1 (2.0%) | −2.0 pp |
| VideoRepick/easy | 50（含 1 error） | 12 (24.0%) | 14 (28.0%) | −4.0 pp |
| VideoRepick/medium | 50（含 3 error） | 6 (12.0%) | 16 (32.0%) | −20.0 pp |
| VideoRepick/xhard | 50（含 3 error） | 7 (14.0%) | 17 (34.0%) | −20.0 pp |
| VideoUnmaskSwap/easy | 50 | 11 (22.0%) | 12 (24.0%) | −2.0 pp |
| VideoUnmaskSwap/medium | 50 | 11 (22.0%) | 17 (34.0%) | −12.0 pp |
| VideoUnmaskSwap/hard | 50 | 5 (10.0%) | 4 (8.0%) | +2.0 pp |
| VideoUnmaskSwap/xhard | 50（含 2 error） | 11 (22.0%) | 5 (10.0%) | +12.0 pp |

逐局配对：只有 context 成功的 41 局，只有 modulation 成功的 172 局，两者都成功的 119 局。

**可比性说明**（计划 §六 / R5）：

- 训练：对照 `v2-1600ep-m16x8x8-modul-b128-80k` 在 AWS 8×A100 训练（80k 配置取 50000 步）；本 run 在 GL 4×A40 训练，每卡 16 × 累积 2，fsdp 4。
- 评估：对照用 9 月的 GL 评估栈，每卡 1 片；本次用 AWS 的 local_692_pool.sh + v17 客户端（4ba8b49），每卡 2 片，显存比例 0.38。新驱动没有在 GL 已评过的 ckpt 上复现过历史数字。
- 692 局二项分布的标准误约 1.6–1.9 pp，18.93 pp 的总差远超噪声。单组 50 局的标准误约 6–7 pp，逐格差异不宜过度解读。
- context 512 与 1024 的总成功率只差 2 局（0.29 pp），在噪声内；但分任务方向相反：1024 在 RouteStick 高 17.5 pp，在 BinFill 低 14.1 pp、VideoRepick 低 8.7 pp。

## ⑨ 用户决策记录

- 11-16：「给出方案 跑完了直接在本地原job评估 参考aws的评估方式」「加上并开工」「按推荐改」。
- 11-17：「这些问题你自己来修理。就除了会影响评估结果语义的问题，其他的你都自己来决策。」「不要每次都阻塞来问我」「好的，大概评估要多久？就先这样评估吧。」
- 11-20 巡检指令：「若 EXIT_CODE=0：核 ckpt {5000..50000} 共 10 份，写 result.md、提交推送、更新 current.json；不 scancel 65943151，占位留给 1116-gl-4task400-ctx-692-eval-plan.html 的 692 局评估」。

## ⑩ 计划外事件与处置

本条链没有计划外事件；评估脚本的三处工程修复都在 ctx512 评估时完成并合入（见 ②）。

## ⑪ 结论与下一步

- 在相同训练库、相同 50k 步数下，context 融合 budget 1024 比 modulation 融合低 18.93 pp，差距比 512 档（−7.66 pp）更大。原因是 modulation 从 512 涨到 1024 时获益明显（30.49% → 42.05%），而 context 几乎不变（22.83% → 23.12%）。差距最大的是 RouteStick medium/hard（各 −48 pp）和 BinFill medium（−44.9 pp）。
- 占位 65943151 已 scancel；1116 计划的两条评估都已完成。下一步按计划 E6：把评估产物从评估副本迁到 `<GL_REPO>/v1-store/evaluation/`，再移除评估副本 worktree 与 E1/F1 子代理 worktree。

## ⑫ 归档文件清单（`records/`）

`EVAL692.txt`、`preseeded-errors.json`、`retries.json`、`merged-{log,errors,progress,shards,retry_plan}.json`、`g4c-ev1024-chain.summary.log`（链日志，tqdm 已清洗）。逐片原始输出（client/server log、视频）在 `<GL_REPO>/v1-store/eval-wt/97f237ac/v1-store/evaluation/gl692-ctx1024-50k/`，E6 时迁到 `<GL_REPO>/v1-store/evaluation/`。
