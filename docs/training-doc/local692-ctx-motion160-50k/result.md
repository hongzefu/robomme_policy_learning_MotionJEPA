# local692-ctx-motion160-50k：本机 692 局评估结果

## ① 结论

`EVAL692=PASS episodes=692 success=254 rate=0.3671 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=1`，**254/692 = 36.71%**；`MERGE_OK shards=24 episodes=692 successes=254 errors=9`；`EXIT_CODE=0`。

| 任务 | 成功 | 局数 | 成功率 |
|---|---|---|---|
| BinFill | 65 | 142 | 45.8% |
| RouteStick | 112 | 200 | 56.0% |
| VideoRepick | 35 | 150（含预标 error 7） | 23.3% |
| VideoUnmaskSwap | 42 | 200（含预标 error 2） | 21.0% |

分组：BinFill easy/medium/hard 40/49、21/49、4/44；RouteStick easy/medium/hard/xhard 47/50、38/50、24/50、3/50；VideoRepick easy/medium/xhard 13/50、11/50、11/50；VideoUnmaskSwap easy/medium/hard/xhard 13/50、16/50、7/50、6/50。

对照（@50000）：原 encoder motion modulation 54.62%（GL A40）；v11 Wan 池化 modulation 43.06%（本机）；无 motion 512 modulation 30.49%（GL）。跨环境数字只作参考。

## ② 被评对象与版本

- ckpt：`v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-ctx-motion-b128-50k/50000`（训练 TRAIN_HEAD `fd20feb`，见 [`../v2-1600ep-m8x8-ctx-motion-b128-50k/`](../v2-1600ep-m8x8-ctx-motion-b128-50k/result.md)）。
- 评估代码：主检出 v17 分支（起跑时 HEAD `5809a5b`，工作区干净；相对 `fd20feb` 的 `src/`、`scripts/` 零差异）。评估栈 `scripts/evaluation/local_692_pool.sh`（v15 `c0826be` 移植，S2 合入 `55ddfac`）；`check_checkpoint.py` 按 run 登记 `integration=context`。
- 在线编码：`v1-store/venvs/wan` 当时为 motion-jepa `2a484ad`（encoder_src `d00267e1`，与离线表一致）。
- benchmark 候选 `b4e97f2`；BinFill 演示前缀库 `v1-store/demo-prefix/binfill-3b4de03a0b46`（`E0_GATE=PASS DEMO_PREFIX_SRC=PASS`）。

## ③ 启动

```bash
tmux new-session -d -s v17-eval "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && set -o pipefail; PYTHONUNBUFFERED=1 EVAL_PER_GPU=3 EVAL_SHARDS=24 bash scripts/evaluation/local_692_pool.sh /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-ctx-motion-b128-50k/50000 7 2>&1 | tee v1-store/logs/v17-eval.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/v17-eval.log"
```

口径：seed 7、max_steps 2000、`MMEVLA_MOTION_OVERFLOW=resample`、`EXPECT_MOTION_STATS=1`、`EXPECT_DEMO_INJECT=1`、每卡 3 份、每份 XLA 预占 0.2、24 片（每片约 28–29 局；每卡 4 份压测 FAIL 后按用户「不行的话就每卡三份」降级）。`PLAN_OK shards=24 total=692 per_shard=28 disjoint=True`；`PRESEED=PASS episodes=9 placed=9 duplicates=0`（与 v11、v16 同一组 9 局）。

## ⑥ 耗时

`LOCAL692_START` 2026-11-17T07:43:33Z（美东 02:43）→ `LOCAL692_DONE` 2026-11-17T09:30:34Z（美东 04:30），约 1 h 47 m；24 片全部 `SHARD_EXIT_CODE=0`、`CHECK_SHARD` 全部通过，无新 error、无重跑。

## ⑦ motion 统计

683 局有 motion 统计：超预算重采样 1 局，`motion_k_max` 最大 175（预算 160）。

## ⑫ 归档

`records/`：`EVAL692.txt`、`merged-progress.json`（逐局）、`merged-errors.json`、`merged-shards.json`、`merged-motion-stats.json`、`preseeded-errors.json`、`retries.json`、`v17-eval.summary.log`。视频与逐分片产物留在 `v1-store/evaluation/local692-ctx-motion160-50k/`（约 1.0 GB）。
