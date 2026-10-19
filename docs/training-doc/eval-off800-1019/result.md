# eval-off800-1019 — 结果

起跑记录见 [launch.md](launch.md)。评估代码锚点 commitV11.31Beta `f8138e0`，三组 30 个单元都在这个 clean HEAD 上跑。时间一律美东，括注 UTC。

## 一、一句话结论与指标速览

**自训 mixed8 512+motion run 在官方 eval seed 800 局上：40k 40.6%、60k 42.5%、80k 43.1%，随步数小幅上升；与历史 full16（原版数据、无 motion）同步数结果 40.62%／44.62% 基本打平到略低，比历史官方 modul 80k 的 47.88% 低 4.8 pp。** 普通难度下换 mixed8 数据加 motion 没有带来提升；motion 的优势只出现在 1017 testx 的 xhard 长 episode 上（见 `eval-testx-1017/result.md` 十一节）。

| 模型 | 局数 | 成功 | 失败 | 超时 | error | 成功率 | easy | medium | hard |
|---|---|---|---|---|---|---|---|---|---|
| 512+motion@40k | 800 | 325 | 439 | 36 | 0 | **40.6%** | 45.7% | 43.2% | 27.1% |
| 512+motion@60k | 800 | 340 | 422 | 38 | 0 | **42.5%** | 49.8% | 41.7% | 27.6% |
| 512+motion@80k | 800 | 345 | 416 | 39 | 0 | **43.1%** | 50.0% | 42.2% | 29.2% |

判定行：

```
EVAL_512+motion@40k=DONE episodes=800 success=325 fail=439 timeout=36 error=0 downsampled=0
EVAL_512+motion@60k=DONE episodes=800 success=340 fail=422 timeout=38 error=0 downsampled=0
EVAL_512+motion@80k=DONE episodes=800 success=345 fail=416 timeout=39 error=0 downsampled=0
SHARDS_UNION=PASS total=800、SEED_MATCH=PASS rows=800、EVAL_VIDEO=PASS videos=800（三组各一行）
MANIFEST_SHARED=PASS models=3 manifest_sha256=702028a2bbf6
EVAL_TABLE=DONE models=3 cells=48
```

## 二、逐任务（每任务 50 局，P6 顺序）

| 编号 | 任务 | 40k | 60k | 80k | 历史官方 80k | 历史 full16 80k |
|---|---|---|---|---|---|---|
| 1.1 | BinFill | 32% | 38% | 28% | 46% | 42% |
| 1.2 | PickXtimes | 54% | 50% | 56% | 94% | 88% |
| 1.3 | SwingXtimes | 88% | 88% | 84% | 94% | 96% |
| 1.4 | StopCube | 56% | 64% | 58% | 54% | 42% |
| 2.1 | VideoUnmask | 28% | 22% | 30% | 30% | 32% |
| 2.2 | ButtonUnmask | 36% | 28% | 32% | 28% | 22% |
| 2.3 | VideoUnmaskSwap | 22% | 22% | 34% | 26% | 22% |
| 2.4 | ButtonUnmaskSwap | 22% | 20% | 22% | 18% | 28% |
| 3.1 | PickHighlight | 24% | 30% | 24% | 30% | 26% |
| 3.2 | VideoRepick | 20% | 26% | 20% | 34% | 28% |
| 3.3 | VideoPlaceButton | 44% | 38% | 40% | 52% | 58% |
| 3.4 | VideoPlaceOrder | 28% | 30% | 26% | 44% | 46% |
| 4.1 | MoveCube | 78% | 82% | 92% | 88% | 72% |
| 4.2 | InsertPeg | 22% | 20% | 20% | 8% | 4% |
| 4.3 | PatternLock | 38% | 50% | 52% | 58% | 42% |
| 4.4 | RouteStick | 58% | 72% | 72% | 62% | 66% |
| | **总计** | **40.6%** | **42.5%** | **43.1%** | **47.88%** | **44.62%** |

历史两列取自 `eval-orig80k-modul-vs-official/result.md`；历史 full16 40k 为 40.62%（`eval-orig40k-ckpt40000/result.md`）。

- 随步数上升主要来自模仿类：MoveCube 78→82→92、PatternLock 38→50→52、RouteStick 58→72→72。
- 明显落后于历史两模型的是计数类 PickXtimes（50–56% 对 88–94%）与 BinFill，以及参考类 VideoRepick、VideoPlaceButton、VideoPlaceOrder。
- 领先的是 StopCube、Unmask 系列（部分）、InsertPeg（20% 对 4–8%）。
- 每任务 50 局，单任务差 1 局即 2 pp，逐任务差异只作线索；本轮未做配对 bootstrap／McNemar。

⚠ 口径差异：历史 800 局用 `scripts/training/legacy-eval/robomme-local/` 旧 eval.py，本轮用现行 `examples/robomme/eval.py` + `episode_plan`；episode、env seed、策略 seed 42、max_steps 1300 四项一致，循环实现不同。

## 三、过程与耗时

| 组 | 起止（美东，括注 UTC） | 每片用时 |
|---|---|---|
| 80k | 10-19 01:39 → 03:13（05:39:53Z → 07:13:17Z） | 71–93 min |
| 40k | 02:51 → 04:43 前后（06:51:25Z → 约 08:43Z） | 73–90 min |
| 60k | 04:12 → 06:11（08:12:14Z → 10:11:45Z） | 约 80–100 min |

30 个单元全部 `rc=0`；严重错误行（`SERVER_DIED`／`Traceback`／`EVAL_PARAMS=FAIL` 等）计数 0。10 个 worker 起跑回显 `seed=42 max_steps=1300`，30 个单元 `EVAL_PARAMS` 均为 1300。

## 四、计划外事件与处置

1. **追加 60k**：用户 2026-10-19 在起跑后要求「40k跑完把60k也跑了」。主会话在 `official800/manifest.tsv` 队尾追加 10 行 `off800-512m-60k`（`UNITS=PASS added=10 total=30`），分片计划 `SHARD_PLAN=PASS per_shard=80`；worker 每轮重读 manifest，40k 领完后自动接 60k。step 60000 已在 RUNS 登记、ckpt 已在 1017 轮核过，不改代码，HEAD 不动。
2. **完成通知延迟**：60k 在 06:11 跑完，主会话监听到期未续挂，直到用户 09:07 询问才发现；不影响结果。
3. **占位 job 释放**：三组跑完、确认 GL 登录节点无 `xh-off800-pool-*` 会话、每个 job 只剩 batch 步后，美东 09:10 前后逐个 `scancel` 64243171–64243180，复核 `squeue -u hongzefu` 为空。

## 五、用户决策原话（本段新增）

1. 2026-10-19：「40k跑完把60k也跑了」
2. 2026-10-18（沿用）：「站位的卡可以停了。」「跑完了就可以停。」

## 六、结论与下一步

- 在官方普通难度上，mixed8 + motion 与原版数据无 motion 的 full16 同步数基本持平，比官方模型低约 5 pp，差距集中在计数类（PickXtimes、BinFill）与参考类。
- 若要对比是否显著，可用 `scripts/training/legacy-eval/compare_eval_runs.py` 按 (task, ep) 配对做 bootstrap 与 McNemar（需对齐历史 run 的结果目录）。

## 七、归档清单（NFS，不进 git）

| 路径 | 内容 |
|---|---|
| `v1-store/evaluation/official800/eval-official-800.jsonl`、`manifest_meta.json` | 800 局清单（sha256 `702028a2…`） |
| `v1-store/evaluation/official800/manifest.tsv`、`queue/` | 30 单元队列与领取／完成标记 |
| `v1-store/evaluation/official800/summary-3steps/summary.md`、`summary.json` | 三模型 48 格对比表 |
| `v1-store/evaluation/off800-512m-{40k,60k,80k}/` | 每局 mp4、`steps/*.jsonl`、分片结果 |
| `v1-store/logs/policy-eval/off800-pool-<jobid>.log`、`off800-512m-*-s*.log` | worker 与单元日志 |
