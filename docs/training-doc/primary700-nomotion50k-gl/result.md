# primary700-nomotion50k-gl 与 primary700-motion50k-gl — 结果（无 motion 50000 vs motion 50000，700 条）

**环境 A**（GreatLakes spgpu A40，gpu-hold-03…10 八个既有作业，各 1×A40 / 1 CPU / 24G，驱动 595.71.05）。执行 HEAD `d33c0ba`（clean），
两轮同一套调用（`gl_hold_pool.sh` → `gl_hold_queue.sh` → `srun --overlap` → `gl_eval_shard.sbatch` → `run_shard.sh`，见 `launch.md`）：
同一套 10 个分片计划（身份 sha `3b4de03a…`）、策略 seed 7、`max_steps=2000`、`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、`CHUNK_EPISODES=20`、
benchmark gitlink `b4e97f2`，只差 `RUN_NAME / CKPT / POLICY / POLICY_CONFIG` 四个值。墙钟 2026-09-18 22:24:29Z → 2026-09-19 03:22:48Z（**4 h 58 min**，
含 18 次卡死等待与 13 次续跑）。本文件是两轮的共同结果档；`../primary700-motion50k-gl/result.md` 只放 motion 专属条目并回链到这里。

## 结论先行

```
MERGE_OK nomotion50k shards=10 episodes=700 successes=182 errors=9  macro=micro=0.2600
MERGE_OK motion50k   shards=10 episodes=700 successes=285 errors=9  macro=micro=0.4071
```

- **motion 50000 比无 motion 50000 高 14.7 个百分点**（40.71% vs 26.00%；同批 691 对非 error 集上 41% vs 26%，仅 motion 成 173 集、仅无 motion 成 70 集）。
- **无 motion 50000（26.00%）与老基线 59999（27.57%）同一水平**：差距来自 motion，不是步数。
- **收益几乎全在 RouteStick 与 hard / xhard**：RouteStick 四档 84% vs 35%（xhard 58% vs 2%，「仅无 motion 成」一列全为 0）；
  按难度 easy +4 pp、medium +12 pp、hard +23 pp、xhard +25 pp。
- **BinFill 实打实退化**：12% vs 29%，三档全负且是双向翻转（easy 有 8 集仅 motion 成、18 集仅无 motion 成），不是少量掉分。需单独排查。
- **VideoRepick +17 pp**（43% vs 26%），**VideoUnmaskSwap +4 pp**（18% vs 14%，各格差值 ±6 集以内，噪声范围）。
- 两侧 error 各 9 条且**逐条相同**，全部是看门狗标记的卡死集（也与老基线的 9 条相同），不计入成功、计入分母；配对表已剔除。

## 14 组成功率（三列：无 motion 50000 / motion 50000 / 老基线 59999）

| 任务 | 难度 | 无 motion 50000 | motion 50000 | 差（motion − 无） | 老基线 59999 |
|---|---|---|---|---|---|
| BinFill | easy | 23/50（46%） | 13/50（26%） | -10 | 25/50（50%） |
| BinFill | medium | 17/50（34%） | 5/50（10%） | -12 | 16/50（32%） |
| BinFill | hard | 4/50（8%） | 0/50（0%） | -4 | 2/50（4%） |
| RouteStick | easy | 41/50（82%） | 50/50（100%） | +9 | 44/50（88%） |
| RouteStick | medium | 20/50（40%） | 47/50（94%） | +27 | 23/50（46%） |
| RouteStick | hard | 8/50（16%） | 41/50（82%） | +33 | 10/50（20%） |
| RouteStick | xhard | 1/50（2%） | 29/50（58%） | +28 | 1/50（2%） |
| VideoRepick | easy | 13/50（26%） | 22/50（44%） | +9 | 9/50（18%） |
| VideoRepick | medium | 16/50（32%） | 19/50（38%） | +3 | 14/50（28%） |
| VideoRepick | xhard | 10/50（20%） | 23/50（46%） | +13 | 13/50（26%） |
| VideoUnmaskSwap | easy | 8/50（16%） | 8/50（16%） | +0 | 13/50（26%） |
| VideoUnmaskSwap | medium | 11/50（22%） | 17/50（34%） | +6 | 13/50（26%） |
| VideoUnmaskSwap | hard | 2/50（4%） | 7/50（14%） | +5 | 3/50（6%） |
| VideoUnmaskSwap | xhard | 8/50（16%） | 4/50（8%） | -4 | 7/50（14%） |
| **微平均** | | **182/700（26%）** | **285/700（41%）** | +103 | 193/700（28%） |
| **宏平均（14 组均值）** | | 26.00% | 40.71% | +14.71 pp | 27.57% |
| error 条目 | | 9 | 9 | | 9 |

### 按任务

| 任务 | 无 motion 50000 | motion 50000 | 差 | 老基线 59999 |
|---|---|---|---|---|
| BinFill | 44/150（29%） | 18/150（12%） | -17 pp | 43/150（29%） |
| RouteStick | 70/200（35%） | 167/200（84%） | +48 pp | 78/200（39%） |
| VideoRepick | 39/150（26%） | 64/150（43%） | +17 pp | 36/150（24%） |
| VideoUnmaskSwap | 29/200（14%） | 36/200（18%） | +4 pp | 36/200（18%） |

### 按难度

| 难度 | 无 motion 50000 | motion 50000 | 差 | 老基线 59999 |
|---|---|---|---|---|
| easy | 85/200（42%） | 93/200（46%） | +4 pp | 91/200（46%） |
| medium | 64/200（32%） | 88/200（44%） | +12 pp | 66/200（33%） |
| hard | 14/150（9%） | 48/150（32%） | +23 pp | 15/150（10%） |
| xhard | 19/150（13%） | 56/150（37%） | +25 pp | 21/150（14%） |

## 同批 episode 配对（两侧均非 error 的集）

| 任务 | 难度 | 配对数 | motion 成功 | 无 motion 成功 | 差 | 都成 | 仅 motion 成 | 仅无 motion 成 | 都败 |
|---|---|---|---|---|---|---|---|---|---|
| BinFill | easy | 50 | 13（26%） | 23（46%） | -10 | 5 | 8 | 18 | 19 |
| BinFill | medium | 50 | 5（10%） | 17（34%） | -12 | 2 | 3 | 15 | 30 |
| BinFill | hard | 50 | 0（0%） | 4（8%） | -4 | 0 | 0 | 4 | 46 |
| RouteStick | easy | 50 | 50（100%） | 41（82%） | +9 | 41 | 9 | 0 | 0 |
| RouteStick | medium | 50 | 47（94%） | 20（40%） | +27 | 20 | 27 | 0 | 3 |
| RouteStick | hard | 50 | 41（82%） | 8（16%） | +33 | 8 | 33 | 0 | 9 |
| RouteStick | xhard | 50 | 29（58%） | 1（2%） | +28 | 1 | 28 | 0 | 21 |
| VideoRepick | easy | 49 | 22（45%） | 13（27%） | +9 | 7 | 15 | 6 | 21 |
| VideoRepick | medium | 47 | 19（40%） | 16（34%） | +3 | 10 | 9 | 6 | 22 |
| VideoRepick | xhard | 47 | 23（49%） | 10（21%） | +13 | 7 | 16 | 3 | 21 |
| VideoUnmaskSwap | easy | 50 | 8（16%） | 8（16%） | +0 | 2 | 6 | 6 | 36 |
| VideoUnmaskSwap | medium | 50 | 17（34%） | 11（22%） | +6 | 6 | 11 | 5 | 28 |
| VideoUnmaskSwap | hard | 50 | 7（14%） | 2（4%） | +5 | 1 | 6 | 1 | 42 |
| VideoUnmaskSwap | xhard | 48 | 4（8%） | 8（17%） | -4 | 2 | 2 | 6 | 38 |
| **合计** | | 691 | 285（41%） | 182（26%） | +103 | 112 | 173 | 70 | 336 |

## error 条目

- 无 motion 50000（9）：VideoRepick/easy ep189, VideoRepick/medium ep155, VideoRepick/medium ep167, VideoRepick/medium ep172, VideoRepick/xhard ep168, VideoRepick/xhard ep197, VideoRepick/xhard ep201, VideoUnmaskSwap/xhard ep139, VideoUnmaskSwap/xhard ep164
- motion 50000（9）：VideoRepick/easy ep189, VideoRepick/medium ep155, VideoRepick/medium ep167, VideoRepick/medium ep172, VideoRepick/xhard ep168, VideoRepick/xhard ep197, VideoRepick/xhard ep201, VideoUnmaskSwap/xhard ep139, VideoUnmaskSwap/xhard ep164
- 老基线 59999（9）：VideoRepick/easy ep189, VideoRepick/medium ep155, VideoRepick/medium ep167, VideoRepick/medium ep172, VideoRepick/xhard ep168, VideoRepick/xhard ep197, VideoRepick/xhard ep201, VideoUnmaskSwap/xhard ep139, VideoUnmaskSwap/xhard ep164

## 计时与资源（A40，1 CPU / 24G，两轮同档位）

server.log 的 TIMING 行（续跑会覆盖 server.log，多次续跑的分片只剩最后一次的样本；`records/shards/s*-server.excerpt.txt`）：

| 环节 | 无 motion 50000 | motion 50000 |
|---|---|---|
| 16 帧 `add_buffer` 稳态 median（各片） | 62–64 ms | **1,651–1,742 ms**（含 sidecar 一窗 ≈1.64 s） |
| `infer` 稳态 median | 124–125 ms | 124–127 ms |
| 首批 demo 段 `add_buffer` median（各片） | 0.7–1.7 s | 17–33 s |
| `MOTION_PROV_RELAXED` 行 / 每次 server 启动 | 0 | 恰 6（gpu_name / compute_cap / sm_count × vae / encoder） |
| 一片 70 集净墙钟（无卡死的片） | 33–38 min | 94–107 min |

分片资源（`records/shards/` 同目录的 `v1-store/evaluation/<run>/s*/records/*.csv`，15 s cgroup 采样 + 500 ms GPU 采样；
卡死等待期间 GPU 空转，s2 / s4 / s5 / s8 / s9 的 util 均值被显著拉低，不能当稳态看）：

| run | cgroup anon 峰（max / mean of 10 片） | CPU 核当量 | GPU 显存峰 | GPU util 均值（无卡死片） | util 0% 采样占比（无卡死片） |
|---|---|---|---|---|---|
| 无 motion 50000 | 16.34 / 14.86 GiB | 0.89–0.95 | 32,771 MiB | 14–16% | 44–51% |
| motion 50000 | 18.24 / 15.48 GiB | 0.93–0.97 | 35,762 MiB | 65–71% | 15–19% |

1 CPU 是两轮共同的硬约束（核当量 ≈0.9–0.97 贴满）；无 motion 轮 GPU 大部分时间在等仿真，motion 轮 sidecar 把 GPU 填到约 70%。

## 动态队列执行记录（单元 → 作业 / 节点 / 起止 UTC / 时长 / rc）

| 队列 | 单元 | 作业 | 节点 | 开始 | 结束 | 分钟 | rc |
|---|---|---|---|---|---|---|---|
| queue | motion-s0 | 61495578 | gl1523 | 22:24:29 | 00:10:31 | 106 | 0 |
| queue | motion-s1 | 61512013 | gl1524 | 22:24:29 | 00:10:13 | 106 | 0 |
| queue | motion-s2 | 61495576 | gl1522 | 22:24:29 | 01:35:53 | 191 | 1 |
| queue | motion-s3 | 61495575 | gl1526 | 22:24:29 | 23:59:03 | 95 | 0 |
| queue | motion-s4 | 61512014 | gl1511 | 22:24:29 | 01:35:54 | 191 | 1 |
| queue | motion-s5 | 61495577 | gl1522 | 22:24:29 | 01:35:54 | 191 | 1 |
| queue | motion-s6 | 61495431 | gl1517 | 22:24:29 | 00:11:33 | 107 | 0 |
| queue | motion-s7 | 61495432 | gl1523 | 22:24:29 | 00:02:06 | 98 | 0 |
| queue | motion-s8 | 61495575 | gl1526 | 23:59:03 | 01:37:19 | 98 | 1 |
| queue | motion-s9 | 61495432 | gl1523 | 00:02:06 | 01:35:53 | 94 | 1 |
| queue | nomotion-s0 | 61512013 | gl1524 | 00:10:13 | 00:43:13 | 33 | 0 |
| queue | nomotion-s1 | 61495578 | gl1523 | 00:10:31 | 00:45:23 | 35 | 0 |
| queue | nomotion-s2 | 61495431 | gl1517 | 00:11:33 | 01:35:54 | 84 | 1 |
| queue | nomotion-s3 | 61512013 | gl1524 | 00:43:13 | 01:17:05 | 34 | 0 |
| queue | nomotion-s4 | 61495578 | gl1523 | 00:45:23 | 01:35:54 | 51 | 1 |
| queue | nomotion-s5 | 61512013 | gl1524 | 01:17:05 | 01:50:34 | 33 | 1 |
| queue | nomotion-s6 | 61495576 | gl1522 | 01:35:53 | 02:14:12 | 38 | 0 |
| queue | nomotion-s7 | 61495432 | gl1523 | 01:35:53 | 02:09:17 | 33 | 0 |
| queue | nomotion-s8 | 61512014 | gl1511 | 01:35:54 | 02:11:36 | 36 | 1 |
| queue | nomotion-s9 | 61495577 | gl1522 | 01:35:54 | 02:11:36 | 36 | 1 |
| queue-resume | motion-s2 | 61495578 | gl1523 | 01:37:18 | 02:19:52 | 43 | 0 |
| queue-resume（重抢前） | motion-s4 | 61495431 | gl1517 | 01:37:18 | 01:56:35 | 19 | 1 |
| queue-resume | motion-s5 | 61495575 | gl1526 | 01:38:17 | 02:05:33 | 27 | 0 |
| queue-resume（重抢前） | motion-s9 | 61512013 | gl1524 | 01:51:18 | 02:38:40 | 47 | 1 |
| queue-resume | nomotion-s2 | 61495431 | gl1517 | 01:56:35 | 02:17:23 | 21 | 0 |
| queue-resume（重抢前） | motion-s4 | 61495575 | gl1526 | 02:05:33 | 02:29:39 | 24 | 1 |
| queue-resume（重抢前） | nomotion-s4 | 61495432 | gl1523 | 02:09:17 | 02:26:38 | 17 | 1 |
| queue-resume | motion-s8 | 61495577 | gl1522 | 02:12:18 | 02:39:02 | 27 | 0 |
| queue-resume | nomotion-s5 | 61512014 | gl1511 | 02:12:18 | 02:29:49 | 18 | 0 |
| queue-resume | nomotion-s8 | 61495576 | gl1522 | 02:16:33 | 02:31:21 | 15 | 0 |
| queue-resume（重抢前） | nomotion-s9 | 61495431 | gl1517 | 02:17:24 | 02:44:41 | 27 | 1 |
| queue-resume（重抢前） | nomotion-s4 | 61495575 | gl1526 | 02:29:39 | 02:47:41 | 18 | 1 |
| queue-resume（重抢前） | motion-s4 | 61495578 | gl1523 | 02:32:38 | 03:05:43 | 33 | 1 |
| queue-resume | motion-s9 | 61512013 | gl1524 | 02:41:42 | 02:44:55 | 3 | 0 |
| queue-resume | nomotion-s9 | 61495432 | gl1523 | 02:47:40 | 02:50:30 | 3 | 0 |
| queue-resume（重抢前） | nomotion-s4 | 61512014 | gl1511 | 02:50:41 | 03:14:43 | 24 | 1 |
| queue-resume | motion-s4 | 61512013 | gl1524 | 03:09:00 | 03:15:18 | 6 | 0 |
| queue-resume | nomotion-s4 | 61495431 | gl1517 | 03:17:42 | 03:22:48 | 5 | 0 |

首个单元开始 2026-09-18T22:24:29Z，末个单元结束 2026-09-19T03:22:48Z。

## 看门狗标记（卡死在 make_env/reset 的 episode，写为 error）

| 单元 | 组 | episode | 静默分钟 | 作业 | 标记时刻 |
|---|---|---|---|---|---|
| motion-s2 | VideoRepick/medium | 155 | 128 | 61495576 | 01:35:50 |
| motion-s4 | VideoRepick/easy | 189 | 132 | 61512014 | 01:35:50 |
| motion-s5 | VideoRepick/xhard | 168 | 119 | 61495577 | 01:35:51 |
| motion-s9 | VideoRepick/medium | 172 | 19 | 61495432 | 01:35:51 |
| nomotion-s2 | VideoRepick/medium | 155 | 64 | 61495431 | 01:35:51 |
| nomotion-s4 | VideoRepick/easy | 189 | 32 | 61495578 | 01:35:51 |
| motion-s8 | VideoRepick/xhard | 201 | 12 | 61495575 | 01:37:17 |
| nomotion-s5 | VideoRepick/xhard | 168 | 13 | 61512013 | 01:50:32 |
| motion-s4 | VideoRepick/medium | 167 | 13 | 61495431 | 01:56:32 |
| nomotion-s8 | VideoRepick/xhard | 201 | 12 | 61512014 | 02:11:33 |
| nomotion-s9 | VideoRepick/medium | 172 | 13 | 61495577 | 02:11:34 |
| nomotion-s4 | VideoRepick/medium | 167 | 14 | 61495432 | 02:26:36 |
| motion-s4 | VideoRepick/xhard | 197 | 13 | 61495575 | 02:29:36 |
| motion-s9 | VideoUnmaskSwap/xhard | 164 | 12 | 61512013 | 02:38:37 |
| nomotion-s9 | VideoUnmaskSwap/xhard | 164 | 12 | 61495431 | 02:44:38 |
| nomotion-s4 | VideoRepick/xhard | 197 | 13 | 61495575 | 02:47:39 |
| motion-s4 | VideoUnmaskSwap/xhard | 139 | 14 | 61495578 | 03:05:40 |
| nomotion-s4 | VideoUnmaskSwap/xhard | 139 | 12 | 61512014 | 03:14:41 |

## 盲区诚实清单

- 单 seed（7）、单 checkpoint（两侧各 50000）；老基线 59999 列仅供参考，它用 sbatch array、1 CPU / 24G（`sacct -j 61466496` 实测 `cpu=1,mem=24G`；本档早先误记为 2 CPU / 32G，那是 sbatch 头部默认值、提交时被 CLI 覆盖）、`EPISODE_WALL_S` 默认 900，与本轮墙钟不同。
- 9 条 error 是环境侧 reset 卡死（策略尚未动作），两侧对称；若视作失败，两侧成功率不变（分母已含），只是不能再说「700 条全部评完」。
- 同一分片的两轮不在同一节点 / 同一张卡上跑（20 单元动态抢单），但均为 A40 + 同驱动、调用逐字相同。
- BinFill 退化未看视频归因；VideoRepick / VideoUnmaskSwap 的正差在各格 ±6 集内，不作单格结论。
- 成功率来自环境 `status` 判定，本轮未人工抽看视频。

## 产物

- 合并结果：`v1-store/evaluation/primary700-{nomotion,motion}50k-gl/merged/{log,progress,errors,shards,retry_plan}.json`（副本 `records/merged/` 与
  `../primary700-motion50k-gl/records/merged/`）。
- 各片 `check.json`、`checkpoint.log`、server / driver 日志摘录：`records/shards/`（motion 侧在 `../primary700-motion50k-gl/records/shards/`）。
- 队列：`records/queue/`（`items.txt`、`items-resume.txt`、`hang_marks.txt`、每单元 `done-*.txt` / `resume-done-*.txt` / `resume-history-*.txt`）。
- worker 与看门狗日志：`records/worker-{h,r,q}0K.txt`、`records/watchdog.txt`、`records/watchdog.sh.txt`。
- 视频与 server 全量日志留在 `v1-store/`（不进 git）。
