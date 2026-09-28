# 官方 MME-VLA test-hard 评估（2026-09-28）：结果

## 一句话结论

1100 局（xhard1/2/3 各 13 任务 × 20 局 + xhard4 16 任务 × 20 局）全部得到正常终态；逐局回注绑定全过、按档步数上限逐局生效。四档成功率都在 2.7%～5.6%，远低于同一批身份上 SimpleMemVLA 的 11.5%～35.0%。

## 判定行（`records/mmevla-gates.txt`）

```text
EVAL_ROUND1=PASS policy=mmevla episodes=550 normal=550 error_left=0 retries=47/55(max_per_shard=9) shards=10 per_shard=55-55 shape=55x10
EVAL_ROUND2=PASS policy=mmevla episodes=550 normal=550 error_left=0 retries=0/55(max_per_shard=0) shards=10 per_shard=55-55 shape=55x10
EVAL_IDENTITY_SET=PASS policy=mmevla rounds=2 shards=10 per_shard=55 episodes=1100 missing=0 extra=0 dup=0
EVAL_BINDING=PASS policy=mmevla episodes=1100 replay=1100 injected_mismatch=0 recorded_drift=62 max_abs=1.2e-07 unused=0
EVAL_TIER_CAP=PASS policy=mmevla episodes=1100 mismatch=0
EVAL_DEMO_FRAMES=INFO policy=mmevla episodes=1100 exact=1054 within_5=1096 max_diff=19
```

`retries=47` 是合并脚本按「每身份多出的记录」计：其中 20 条是更正记录（见下），实际重评 27 局，均在每轮 55 次上限内。

## 分档成功率

| 档 | 局数 | success | fail | timeout | 成功率 |
|---|---|---|---|---|---|
| xhard1 | 260 | 14 | 233 | 13 | 5.4% |
| xhard2 | 260 | 11 | 232 | 17 | 4.2% |
| xhard3 | 260 | 7 | 232 | 21 | 2.7% |
| xhard4 | 320 | 18 | 268 | 34 | 5.6% |

逐格明细 `records/mmevla-cells.json`；逐局 `records/episodes-r<轮>-s<片>.jsonl`（每身份以最后一条为准）。`timeout` 的步数是上限 + 1（官方判断为 `count > max_steps`）。

## 计划外事件与处置（第一轮）

1. **server 显存不足**：VideoPlaceButton／VideoPlaceOrder 新值档演示约 1419 帧，`add_buffer` 需 6.8～8.5 GB，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`（约 19 GB）不够，server 回字符串错误、客户端报 `a bytes-like object is required, not 'str'`。改 0.75 后第二轮零异常。只改显存分配上限，不改模型、推理或数值。
2. **记录缺陷（本分支引入，已修）**：官方 `eval.py` 捕获单局异常后 `success_flag` 保留上一局的值，`episodes.jsonl` 因此把 20 个异常局误记成上一局终态；若异常落在某任务本片第一局则 `success_flag` 为 `unknown`，触发官方整轮中止（第 9 片之后各任务当时未评）。`7fc7128` 修复后，对日志中出过异常的 20 个身份追加更正记录（原字段保留、`status=error`、`correction` 写明原因），以续评方式重评。
3. **心跳超时**：续评第一遍在每片首个长演示局触发 server 端 websocket keepalive 超时（server 在事件循环里同步做首次大形状计算）；同一 server 进程第二遍编译已缓存，全部评完。未改 server 代码（计划 R17）。

## 视频

官方每局都存视频（1100 条，约 2.4 GB）；按计划每格只保留 candidate 最小的一局（55 条，清单 `records/mmevla-videos-kept.txt`），其余 1046 条删除，保留的 55 条在 benchmark 仓库 `artifacts/hard-split/eval-out-mmevla/`（不进 git）。
