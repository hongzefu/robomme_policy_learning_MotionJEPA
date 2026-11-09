# local692-wantp768-50k 评估结果

## ① 一句话结论

Wan 重叠窗零参数池化 768（8 帧时间均值 + 自适应 6×8）替换 motion encoder 的 4task modulation 模型，@50000 在本机 692 局评估上成功 **298/692 = 43.06%**：比无 motion 的 512 帧记忆（30.49%）高 12.6 pp，与加宽帧记忆的 1024/2048（42.05% / 45.09%）相当，明显低于上次学习型 encoder motion（54.62%，跨环境历史参考）。

```
EVAL692=PASS episodes=692 success=298 rate=0.4306 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=1
MERGE_OK shards=10 episodes=692 successes=298 errors=9
```

## 与历史 @50000 对照（历史数字来自 origin/v2-eval-0918-motion，GreatLakes A40，跨环境参考）

| 模型 @50000 | 总体 | BinFill | RouteStick | VideoRepick | VideoUnmaskSwap |
|---|---|---|---|---|---|
| 512（8 帧，无 motion） | 30.49% | 51.4% | 35.0% | 26.0% | 14.5% |
| 1024（16 帧） | 42.05% | 57.0% | 62.5% | 31.3% | 19.0% |
| 2048（32 帧） | 45.09% | 59.2% | 75.5% | 30.7% | 15.5% |
| 512 + encoder motion（768） | 54.62% | 78.2% | 83.5% | 42.7% | 18.0% |
| **512 + Wan 池化 768（本次，本机 A100）** | **43.06%** | **59.2%** | **69.0%** | **25.3%** | **19.0%** |

分组（本次）：BinFill easy/medium/hard 89.8% / 61.2% / 22.7%；RouteStick easy/medium/hard/xhard 100% / 96.0% / 54.0% / 26.0%；VideoRepick easy/medium/xhard 24.0% / 32.0% / 20.0%（含预标 error 1/3/3）；VideoUnmaskSwap easy/medium/hard/xhard 28.0% / 30.0% / 8.0% / 10.0%（xhard 含预标 error 2）。按难度：easy 60.3%、medium 54.8%、hard 28.5%、xhard 18.7%。

读法（描述，不作因果结论）：池化 motion 相对无 motion 的提升主要在 RouteStick（+34 pp）与 BinFill（+7.8 pp）；而上次 encoder motion 拉开差距的 VideoRepick（42.7%），池化版本 25.3% 与无 motion 持平。环境（A100 vs A40、本机重生成前缀）不同，逐格差异只作参考。

## ⑥ 耗时

`w768-eval` 美东 11-09 03:01:24 起跑，约 06:09 结束（日志最后写入 11:08:57Z），约 3 h 08 m。同卡 2 片的 s0/s1/s8/s9 最慢，决定总墙钟。

## ⑦ 过程行为

- 起跑闸：`E0_GATE=PASS`、`PLAN_OK shards=10 total=692 per_shard=69`、`PRESEED=PASS episodes=9 placed=9 duplicates=0`；10 片 6 秒内全部占位。
- 10 片 `SHARD_EXIT_CODE=0`；`RETRY_INTEGRATE=PASS retried=0`（除预标 9 局外无新 error）；10 片 `check_shard.py` 全部 `SHARD_PASS`。
- motion 统计：`MOTION_MERGED episodes=683 k_max=179 es_max=1107 downsampled_episodes=2`（超 160 预算按 resample 降采样的 2 局成功 0/2，其余 298/681；两组有选择偏倚，仅作描述）。
- 显存：GPU 0/1 各 2 片约 78.6 GB，其余各约 39 GB。

## ⑨ 用户决策

「只拿 54.62% 作参考」「9局预标+新error重跑一次」「放宽闸门同卡2片」「它这个前缀库只要你是用同样的代码去生成的就可以…」「692局 seed7 单次」「50k」。

## ⑫ 归档（records/）

`EVAL692.txt`、`merged-progress.json`（逐局结果）、`merged-shards.json`、`merged-errors.json`、`preseeded-errors.json`、`retries.json`、`demo-prefix-diff.json`、`E0.txt`、`w768-eval.summary.log`（清洗后驱动日志）。视频与逐分片产物留在 `v1-store/evaluation/local692-wantp768-50k`。
