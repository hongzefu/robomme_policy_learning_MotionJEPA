# gl692-ctx512-50k 评估结果（第二段，2026-11-17 美东）

> 第一段见同目录 [`launch.md`](launch.md)；被评对象是训练 run [`v2-1600ep-m8x8-ctx-b128-50k-gl`](../v2-1600ep-m8x8-ctx-b128-50k-gl/result.md) 的 50000 步 ckpt；计划为仓库根 `1116-gl-4task400-ctx-692-eval-plan.html`。

## ① 一句话结论与指标速览

context 512（8 帧 × 8×8，无 motion）@50000 在 692 局上成功 **158/692 = 22.83%**，比同库、同 692 局、同 seed 7、同为 GL A40 的 modulation 512 @50000 对照（211/692 = 30.49%）低 **7.66 pp**。本次 `new_errors=0`，按计划 V7 可以和对照直接比较。差距主要来自 RouteStick（低 22.5 pp）和 BinFill（低 11.3 pp）。VideoUnmaskSwap 上 context 高 4.5 pp，VideoRepick 两者基本持平。

```
EVAL692=PASS episodes=692 success=158 rate=0.2283 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=0
MERGE_OK shards=8 episodes=692 successes=158 errors=9
```

## ② 版本与代码状态（与第一段的差异）

正式评估实际用的代码是 **`4ba8b49`**，不是 launch.md 写的 97f237ac。评估副本目录名仍是 `.../eval-wt/97f237ac`，其中已 `checkout --detach 4ba8b49`。相对 97f237ac，评估路径上只改了 `examples/robomme/{utils_v17,env_runner_v17,eval_v17}.py` 三个文件；子模块 b4e97f22 不变，server/client 环境复用，没有重建。链前缀也改了两处：`env` → `/usr/bin/env`，并加了 `PYTHONPATH=$W`。原因见 ⑩。

## ③ 运行过程（美东）

| 时刻 | 事件 |
|---|---|
| 18:11 | 训练 `EXIT_CODE=0`，第 1 关 `TRAIN_DONE=PASS ckpts=10` |
| 18:11 / 18:18 / 18:21 | g6b 连续三次失败（工程故障，见 ⑩），各自修复后重起 |
| 18:32 | 第四次起跑 |
| 18:35–18:37 | g6b：`EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=35.6`（单片 156 s，同卡两片墙钟 127 s） |
| 18:39 | 压测：`EVAL_CAPACITY=PASS per_gpu=2 peak_mem_gb=35.6 free_gb=9.4 host_delta_gb=23.1`、`HOST_MEM=PASS projected_gb=92.4 limit=115`、`CAP_TIMEOUT=PASS wall_s=124 projected_chunk_s=4960 limit=14400` → `CHAIN_EVAL per_gpu=2 frac=0.38 shards=8` |
| 18:39 | `PRESEED=PASS episodes=9 placed=9 duplicates=0`，8 片起跑 |
| 约 19:40 | 8 片 `SHARD_EXIT_CODE=0`，8 片 `CHECK_SHARD ... SHARD_PASS` |
| 19:46 | `MERGE_OK`、`EVAL692=PASS`、`CHAIN_DONE tag=512`，链 `EXIT_CODE=0` |
| 19:48 | 按 R3（PASS 且 new_errors=0）`scancel 65943150`，sacct `CANCELLED+`，job 共运行 5-14:54:42 |

正式 692 局约 67 分钟跑完：每卡 2 片，共 8 片，每片约 87 局，平均每片约 46 s/局。

## ⑦⑧ 结果：逐格与对照

对照 512 由两次评估拼成：`primary700-nomotion50k-gl` 的 550 局非 BinFill，加上 `binfilldemo-nomotion50k-gl` 的 142 局 BinFill。14 个组的局号集合与本次完全一致（`KEYSET_MISMATCH_GROUPS=0`）。对照的 9 个 error 局与本次预标的 9 局为同一批。

**按任务**

| 任务 | 局数 | context 512 | modulation 512 对照 | 差 | 两者都成功 |
|---|---|---|---|---|---|
| BinFill | 142 | 57 (40.1%) | 73 (51.4%) | −11.3 pp | 45 |
| RouteStick | 200 | 25 (12.5%) | 70 (35.0%) | −22.5 pp | 22 |
| VideoRepick | 150 | 38 (25.3%) | 39 (26.0%) | −0.7 pp | 17 |
| VideoUnmaskSwap | 200 | 38 (19.0%) | 29 (14.5%) | +4.5 pp | 13 |
| **合计** | **692** | **158 (22.83%)** | **211 (30.49%)** | **−7.66 pp** | 97 |

**按难度**

| 难度 | 局数 | context 512 | modulation 512 |
|---|---|---|---|
| easy | 199 | 86 (43.2%) | 100 (50.3%) |
| medium | 199 | 41 (20.6%) | 73 (36.7%) |
| hard | 144 | 13 (9.0%) | 19 (13.2%) |
| xhard | 150 | 18 (12.0%) | 19 (12.7%) |

**14 组逐格**

| 组 | 局数 | context 512 | modulation 512 | 差 |
|---|---|---|---|---|
| BinFill/easy | 49 | 41 (83.7%) | 38 (77.6%) | +6.1 pp |
| BinFill/medium | 49 | 13 (26.5%) | 26 (53.1%) | −26.5 pp |
| BinFill/hard | 44 | 3 (6.8%) | 9 (20.5%) | −13.6 pp |
| RouteStick/easy | 50 | 16 (32.0%) | 41 (82.0%) | −50.0 pp |
| RouteStick/medium | 50 | 6 (12.0%) | 20 (40.0%) | −28.0 pp |
| RouteStick/hard | 50 | 3 (6.0%) | 8 (16.0%) | −10.0 pp |
| RouteStick/xhard | 50 | 0 (0.0%) | 1 (2.0%) | −2.0 pp |
| VideoRepick/easy | 50（含 1 error） | 14 (28.0%) | 13 (26.0%) | +2.0 pp |
| VideoRepick/medium | 50（含 3 error） | 13 (26.0%) | 16 (32.0%) | −6.0 pp |
| VideoRepick/xhard | 50（含 3 error） | 11 (22.0%) | 10 (20.0%) | +2.0 pp |
| VideoUnmaskSwap/easy | 50 | 15 (30.0%) | 8 (16.0%) | +14.0 pp |
| VideoUnmaskSwap/medium | 50 | 9 (18.0%) | 11 (22.0%) | −4.0 pp |
| VideoUnmaskSwap/hard | 50 | 7 (14.0%) | 2 (4.0%) | +10.0 pp |
| VideoUnmaskSwap/xhard | 50（含 2 error） | 7 (14.0%) | 8 (16.0%) | −2.0 pp |

逐局配对：只有 context 成功的 61 局，只有 modulation 成功的 114 局，两者都成功的 97 局。

**可比性说明**（计划 §六 / R5）：

- 训练：对照在 AWS 8×A100 训练；本 run 在 GL 4×A40 训练，每卡 16 × 累积 2，fsdp 4。
- 评估：对照用 9 月的 GL 评估栈，每卡 1 片；本次用 AWS 的 local_692_pool.sh + v17 客户端（4ba8b49），每卡 2 片，显存比例 0.38。新驱动没有在 GL 已评过的 ckpt 上复现过历史数字。
- 692 局二项分布的标准误约 1.6–1.8 pp，7.66 pp 的总差明显超出噪声。单组 50 局的标准误约 6–7 pp，逐格差异不宜过度解读。

## ⑨ 用户决策记录

- 11-16：「给出方案 跑完了直接在本地原job评估 参考aws的评估方式」「加上并开工」「按推荐改」。
- 11-17：「这些问题你自己来修理。就除了会影响评估结果语义的问题，其他的你都自己来决策。」「不要每次都阻塞来问我」「好的，大概评估要多久？就先这样评估吧。」

## ⑩ 计划外事件与处置

g6b 关连续三次失败，都是工程故障，各自修复后重起。失败日志保留在 `records/g4c-ev512-chain.{envfail-1811,pyfail-1821,clientfail-1824}.summary.log`，对应的运行目录改名为 `gl692-ctx512-g6b.<后缀>` 保留。

1. **18:11 rc=13**：`execve(): /home/hongzefu/.local/bin/env: Permission denied`。链里裸写的 `env` 在 GL 上按 PATH 命中 uv 的 shell 片段。改为 `/usr/bin/env`，并在占位里冒烟 `ENV_OK`。
2. **18:18 server `ModuleNotFoundError: No module named 'scripts'`**：`serve_policy.py` 以脚本方式启动，sys.path 里没有仓库根，而 `websocket_policy_server` 要 import `scripts.evaluation.runtime_contract`。链 env 加 `PYTHONPATH=$W`，只影响模块搜索路径。
3. **18:21 client `AttributeError: 'Args' object has no attribute 'episode_stride'`**：合并 95faf61 把 v17 的 `eval.py` 搬成了 `eval_v17.py`，但它 import 的是本分支 v4.2 版 `utils.py`/`env_runner.py`。两版还有三处行为差异：`clear_buffers` 是否把 exec_start_idx 归零、`pack_buffer` 的校验、`env.step` 异常是否记为 error 局。AWS 692（commitV17.5，v17 分支 5809a5b）用的是 v17 原版。
   - 修法：新增 `utils_v17.py`、`env_runner_v17.py`（95faf61^2 原样），`eval_v17.py` 改三处 import，合并提交 `4ba8b49`。
   - 审查：`ADJUST_REVIEW=PASS scale=same findings=4`，`PRE_MERGE_REVIEW=PASS base=765b474 tip=f98490d files=3 commits=1 findings=0`，`POST_MERGE_REVIEW=PASS merge=4ba8b49`。
   - server 侧已核对：97f237ac 的 `policy.add_buffer` 与 v17 相同，把 exec_start_idx=0 解释为沿用已保存值；无 motion run 也不会把它传给 mem_buffer，所以不改。

## ⑪ 结论与下一步

- 在相同训练库、相同 50k 步数下，context 融合（budget 512）比 modulation 融合低 7.66 pp。主要差在 RouteStick（easy 档差 50 pp）和 BinFill medium/hard。context 只在 VideoUnmaskSwap 略好。
- 占位 65943150 已 scancel。ctx1024 训练预计 11-20 约 06:00 美东结束，其评估链 `g4c-ev1024-chain` 已换上新脚本，在第 1 关等待；对照为 291/692 = 42.05%。
- 两条评估都已完成；E6 已于 11-20 执行：评估产物迁到 `<GL_REPO>/v1-store/evaluation/gl692-ctx512-*`，评估副本 worktree 已移除。

## ⑫ 归档文件清单（`records/`）

`EVAL692.txt`、`preseeded-errors.json`、`retries.json`、`merged-{log,errors,progress,shards,retry_plan}.json`、`g4c-ev512-chain.summary.log`（第四次、成功的链日志，tqdm 已清洗），以及三次失败的链日志清洗版。逐片原始输出（client/server log、视频）已于 E6 迁到 `<GL_REPO>/v1-store/evaluation/gl692-ctx512-50k/`。
