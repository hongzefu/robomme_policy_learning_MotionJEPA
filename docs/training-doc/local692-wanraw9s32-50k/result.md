# local692-wanraw9s32-50k 评估结果

起跑口径见 [launch.md](launch.md)；训练档案 [v2-1600ep-m8x8-modul-wanraw9s32-b128-50k](../v2-1600ep-m8x8-modul-wanraw9s32-b128-50k/result.md)；对照 [local692-wantp768-50k](../local692-wantp768-50k/result.md)。

## ① 一句话结论

Wan 原始 9 latent 不池化（每窗 9 个 16384 维 latent 各经同一 `Linear(16384+768→1024)` 成 1 个记忆 token）+ 窗口 stride 32 的 4task modulation 模型，@50000 在本机 692 局评估上成功 **262/692 = 37.86%**，比 v11 Wan 池化 768（stride 16）的 43.06% 低 5.2 pp，差距主要在 RouteStick（−12.0 pp）与 BinFill（−5.0 pp）。本轮同时改了「不池化」与「stride 16→32」两个变量（计划 D3，用户接受混杂），差异无法归因到其中任一项。

```
EVAL692=PASS episodes=692 success=262 rate=0.3786 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=1
MERGE_OK shards=32 episodes=692 successes=262 errors=9
  macro_success_rate=0.3780  micro_success_rate=0.3786
```

## 与 v11 及历史 @50000 对照

分母均含预标 error（按失败计）。v11 与本次同为本机 A100、同一前缀库、同候选、同 seed 7；历史三行来自 origin/v2-eval-0918-motion（GreatLakes A40），只作跨环境参考。

| 模型 @50000 | 总体 | BinFill | RouteStick | VideoRepick | VideoUnmaskSwap |
|---|---|---|---|---|---|
| 512（8 帧，无 motion，A40） | 30.49% | 51.4% | 35.0% | 26.0% | 14.5% |
| 512 + encoder motion（768，A40） | 54.62% | 78.2% | 83.5% | 42.7% | 18.0% |
| 512 + Wan 池化 768，stride 16（v11，本机） | 43.06% | 59.2% | 69.0% | 25.3% | 19.0% |
| **512 + Wan raw9 不池化，stride 32（本次，本机）** | **37.86%** | **54.2%（77/142）** | **57.0%（114/200）** | **25.3%（38/150）** | **16.5%（33/200）** |
| 本次 − v11 | −5.2 pp | −5.0 pp | −12.0 pp | 0.0 pp | −2.5 pp |

分组（本次，从 `merged/progress.json` 逐局重算，与 `merge.log` 一致；括号内为 v11）：

| 任务 | easy | medium | hard | xhard |
|---|---|---|---|---|
| BinFill | 41/49 = 83.7%（89.8%） | 27/49 = 55.1%（61.2%） | 9/44 = 20.5%（22.7%） | — |
| RouteStick | 47/50 = 94.0%（100%） | 45/50 = 90.0%（96.0%） | 20/50 = 40.0%（54.0%） | 2/50 = 4.0%（26.0%） |
| VideoRepick | 12/50 = 24.0%（24.0%），含预标 error 1 | 15/50 = 30.0%（32.0%），含 3 | — | 11/50 = 22.0%（20.0%），含 3 |
| VideoUnmaskSwap | 10/50 = 20.0%（28.0%） | 11/50 = 22.0%（30.0%） | 5/50 = 10.0%（8.0%） | 7/50 = 14.0%（10.0%），含预标 error 2 |

按难度：easy 110/199 = 55.3%（v11 60.3%）、medium 98/199 = 49.2%（54.8%）、hard 34/144 = 23.6%（28.5%）、xhard 20/150 = 13.3%（18.7%）。

读法（只作描述，不作因果结论）：降幅集中在 RouteStick hard / xhard（−14 pp / −22 pp）与 BinFill 各档（−2～−6 pp）；VideoRepick 与 v11 持平，VideoUnmaskSwap easy / medium 降、hard / xhard 小幅升。「不池化」与「stride 32」两个改动同时生效，且 stride 32 下执行段前 32 帧无 exec motion token（计划风险第 9 条），哪一项造成差距本轮无法区分。motion token 激活方差在训练中近线性增长到约 7279（训练档案 ⑦），是否与此相关同样未验证。

## ⑥ 耗时

`r9-eval32` 美东 11-13 12:30:31 起跑，`LOCAL692_DONE 2026-11-13T18:43:27Z`（美东 13:43:27），约 **73 分钟**（v11 10 片每卡 ≤2 片约 3 h 08 m）。分片最早 s29 于美东约 13:19 完成，主批最晚 s1 约 13:38；补跑的 s5 美东 12:41:57 起、13:30:45 完成（约 49 分钟），未拖长总墙钟。每卡 4 片显存约 79 GB/卡，无 OOM。

## ⑦ 过程行为

- 起跑闸：`E0_GATE=PASS`、`PLAN_OK shards=32 total=692 per_shard=21 disjoint=True groups=14`、`PRESEED=PASS episodes=9 placed=9 duplicates=0`；32 片 `SLOT_REGISTERED` 均在起跑时登记（gpu-slots 每卡 4 个）。
- 32 片 `SHARD_EXIT_CODE=0`；`SHARD_SCAN` 全部 `verdict=ok`（预标 error 落在 s6/s9/s10×2/s19/s22/s24/s26/s27，`new_errors=0`）；`RETRY_INTEGRATE=PASS retried=0`；32 片 `check_shard.py` 全部 `SHARD_PASS`。
- motion 统计：`MOTION_MERGED episodes=683 k_max=96 es_max=1107 downsampled_episodes=4`（可见窗超 budget 80 按 resample 降采样的 4 局成功 0/4，其余 262/679；两组有选择偏倚，仅作描述）。v11 对应为 `k_max=179 downsampled_episodes=2`（stride 16，budget 160）。

## ⑨ 用户决策

按时间：「给出方案 我之前训了wan直接pool作为modulation的 4task每个task400 成功率43。1的 我现在训一个不pool的 直接9个latent都在 每个latent单读投影 然后stride原来是16 现在改为32 应该怎么做？」「训完也是直接eval」「同意写完整方案 1234都按照你说的 data成瓶颈的话你看能不能修复？现在这个投影是什么意思？」「bf16我同意 没问题」「登记」「开工」「同意开工，按照推荐开工，你可以直接开始了。不用再找我」「修复然后起跑，有这种问题你都要自己解决，你不能每次在这种情况下就停。」「之后持续监控训练有没有问题，如果有问题的话，尽可能自己解决，然后报告用户。」「以后这种工程问题你都要自己解决，不要再来问我，不要停止，你这样停止非常浪费时间。」「hugging face 上传的是公开 bucket」「我同意放行，你可以再试一下」「能不能改成16分片？分片不需要对比」「每卡 4 个 worker），预计能压到 1.5 小时左右 这个同意」「实现每卡4worker worker现存等以后都不需要保持一致」。

## ⑩ 计划外事件与处置

1. **改为每卡 4 片（32 片）**：用户指令，`commitV15.18`（66eb506，合入 sub/EVAL32 0e63c49）把片数 / 每卡片数 / 显存占比做成可配置（默认 32 / 4 / 0.2）；`PRE_MERGE_REVIEW=PASS base=3addd7c tip=0e63c49 files=3 commits=1 findings=9 blocking=0`，合并后单测 16 passed。链脚本用占位 `r9-eval.log` 挡住自动 10 片评估（`CHAIN=STOP reason=eval_log_exists`，预期）。
2. **s5 首跑误判退出**：同卡 4 片并发启动时，s5 的进程归属检查报「所选 GPU 上的计算进程 1242253 不属于已登记分片的进程树（根=1238170 1238170 1238221 1238281 1238367），停止起跑」，`END_UTC=2026-11-13T17:33:42Z`、`EXIT_CODE=2`、0 局（records/`s5.try1.log`）。原因判断：slot 文件里同卡另一片尚处 `pending`、还没换成 PID 的竞争窗口内，其 server 进程已出现在该 GPU 上，被当成外来进程。主会话按原流程手动补跑 s5（tmux `r9-eval32-s5`，`S5_RERUN_REGISTERED pid=1288620 2026-11-13T17:41:57Z` → `S5_RERUN_DONE rc=0 2026-11-13T18:30:45Z`），环境逐项一致；另起看守 `r9-eval32-guard`，防止其余分片先完成时驱动在缺 s5 的情况下合并，最终未触发（`GUARD_DONE s5done=1 others_done=18 stopped=0`）。驱动随后读到 `s5.rc=0`，s5 记 `SHARD_EXIT_CODE=0`、`SHARD_SCAN … verdict=ok`、`SHARD_PASS`。该竞争属待修缺陷（见 ⑪）。
3. **计划闸 G9 的评估结果上传未执行**：计划 RB-G2 要求把逐局结果与 merge 结果传同一 bucket 的 `eval/local692-wanraw9s32-50k/` 并读回（`UPLOAD_READBACK=PASS`）；本轮未执行，逐局结果以本目录 records/ 进 git 归档替代（ckpt 部分已 `HF_VERIFY=PASS ckpts=10`）。如需补传由主会话另行处理。

## ⑪ 结论与下一步

评估交付完成，状态机 `4task-wanraw9s32-eval` → closed。下一步（均为非授权建议）：① 修 slot 登记竞争（`run_slot` 的 pending→pid 窗口导致同卡并发启动误判）；② 可选：raw latent 归一化，或把「不池化」与「stride 16→32」拆成两个单变量对照，以便归因。

## ⑫ 归档（records/）

`EVAL692.txt`、`merged-progress.json`（逐局结果）、`merged-log.json`（分组与分任务汇总）、`merged-shards.json`、`merged-errors.json`、`preseeded-errors.json`、`retries.json`、`eval32.summary.log`（清洗后驱动日志）、`merge.log`、`scan.log`、`integrate.log`、`preseed.log`、`s5.try1.log`（s5 误判原始日志）、`eval32-s5.log`（补跑驱动尾行）、`eval32-guard.log`、`r9-eval.placeholder.log`（挡链脚本的占位文件）。E0 判定与 v11 同一前缀库（`v1-store/demo-prefix/binfill-3b4de03a0b46/E0.txt`，v11 records 已收）。视频与逐分片产物留在 `v1-store/evaluation/local692-wanraw9s32-50k`。
