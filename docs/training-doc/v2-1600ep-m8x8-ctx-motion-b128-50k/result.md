# v2-1600ep-m8x8-ctx-motion-b128-50k 结果

## ① 一句话结论

54.62% 那条 512+motion160（v7 ep72 encoder）modulation run 只改 context 注入、其余一致，训满 50k（`TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10`，`EXIT_CODE=0`），本机 692 局 seed 7 评 step 50000：**254/692 = 36.71%**（`EVAL692=PASS … new_errors=0`），低于原 modulation 的 54.62%（GL A40）与本机 v11 Wan 池化 modulation 的 43.06%。

| 模型 @50000 | 总体 | BinFill | RouteStick | VideoRepick | VideoUnmaskSwap | 评估环境 |
|---|---|---|---|---|---|---|
| 512＋encoder motion，**modulation**（原 run） | 54.62% | 78.2% | 83.5% | 42.7% | 18.0% | GL A40 |
| 512＋Wan 池化 768，modulation（v11） | 43.06% | 59.2% | 69.0% | 25.3% | 19.0% | 本机 A100 |
| 512＋encoder motion，**context**（本 run） | **36.71%** | **45.8%** | **56.0%** | **23.3%** | **21.0%** | 本机 A100 |
| 512，无 motion，modulation | 30.49% | 51.4% | 35.0% | 26.0% | 14.5% | GL |

评估细节见 [`../local692-ctx-motion160-50k/result.md`](../local692-ctx-motion160-50k/result.md)。

## ⑥ 硬件与耗时

AWS p4de 8×A100-SXM4-80GB，FSDP8、batch 128、workers 16。tmux `v17-train`，起跑 2026-11-16T04:12:47Z（美东 11-15 23:12），结束约 2026-11-17T07:43Z（美东 11-17 02:43），总墙钟约 27 h 30 m；稳态 2.0 s/步（与 0928 同组合测速 1.979 s/步一致）。

## ⑦ 训练过程行为

| step | loss | grad_norm | mem_enc_norm |
|---|---|---|---|
| 0 | 0.6635 | 23.42 | 16.18 |
| 100 | 0.3007 | 9.69 | 6.78 |
| 200 | 0.0714 | 1.03 | 0.70 |
| 300 | 0.0356 | 0.50 | 0.26 |
| 1000 | 0.0169 | 0.28 | 0.10 |
| 5000 | 0.0076 | 0.060 | 0.024 |
| 10000 | 0.0049 | 0.043 | 0.018 |
| 20000 | 0.0030 | 0.034 | 0.014 |
| 30000 | 0.0024 | 0.028 | 0.0097 |
| 40000 | 0.0021 | 0.026 | 0.0072 |
| 50000 | 0.0018 | 0.026 | 0.0073 |

前 300 步 loss 与 0928 同组合测速逐点吻合（0.663759 / 0.300722 / 0.071265 / 0.035547）；全程无 NaN、无报错，跨过第 300 步计时窗口结束点无异常。记忆编码器梯度全程非零。

读回：`CKPT_LOAD=PASS step=50000 integration=context motion=1 leaves=59 resolved_sha256=21791b29…`。

备份：公开 bucket `HongzeFu/robomme-vla-4task-m8x8-ctx-motion160-v7-50k-v1`，10 份逐份 `STEP_VERIFY=PASS`（每份约 11.59 GB，19–22 个文件，文件数差异来自 OCDBT 分块数，已对本地逐目录核对），末行 `HF_VERIFY=PASS ckpts=10`、`EXIT_CODE=0`（tmux `v17-ckptup`）。

## ⑨ 用户决策记录

- 「给出方案 54.62% 那条 … 但是用context模式其他保持一致」
- 「同意选 A，然后每卡四份同意，如果不行的话就每卡三份。上传还是公开的 Hugging Face。修改计划」
- 「同意开工」
- 「a」（提前切主检出）；「尽可能让他自己来做，不要让我们新的对话来做这件事情。」（v16 上传由 v16 会话自管，其后报告 `EVAL_HF_VERIFY=PASS`、`EXIT_CODE=0`）

## ⑩ 计划外事件与处置

1. 评估编码环境 `v1-store/venvs/wan` 于 10-21 被 v16 线换成 MotionJEPA `ee719fb`（encoder_src `40d8a33b`），与 v7 离线表 `d00267e1` 不符；按用户选的方案 A 于 11-15 按 `c55a6bc` 锁文件修复为 `2a484ad`（`WAN_ENV=PASS`），评估后于 11-17 美东约 04:35 按 `b5bdccf` 锁文件还原为 `ee719fb`（tmux `v17-wanrestore`，`EXIT_CODE=0`，`direct_url` 读回 `ee719fb`）。
2. 试评估 `EVAL_LOAD_SMOKE=FAIL … share2=FAIL peak_mem_gb=78.9`（`--g6b` 写死每份 0.45 预占，2 份必超 80 GB，渲染器 OOM）；`ADJUST_REVIEW=PASS scale=same findings=3` 后改由压测承担并发子项。压测 `EVAL_CAPACITY=FAIL per_gpu=4 peak_mem_gb=79.0 free_gb=1.0`、`EVAL_CAPACITY=PASS per_gpu=3 peak_mem_gb=59.6 free_gb=20.4`，正式评估用每卡 3 份、24 片。均已写入计划 K 节（`24d7b8e`）。
3. 训练进行中，另一会话（nativemem 训练）在主检出 v17 分支上做了 6 个文档提交（`e6cd6bf`…`5809a5b`），主检出 HEAD 由 Beta `fd20feb` 移到 `5809a5b`；净改动只有计划 HTML 与已删除的 NativeMEM 计划文件，`src/`、`scripts/` 零改动，训练与评估代码与 Beta 一致。TRAIN_HEAD 仍记为 `fd20feb`。
4. `v16-up-eval-m` 在主检出切分支时仍在运行，由 v16 会话确认未受影响。

## ⑪ 结论与下一步

- 在本组输入（512 帧记忆＋160 窗 v7 motion、8×8、50k、本机评估）下，context 注入（36.71%）明显不如 modulation（原 run 54.62% 跨环境；同环境的 v11 池化 modulation 43.06%），差距主要在 BinFill（45.8% vs 78.2%）与 RouteStick（56.0% vs 83.5%）；VideoUnmaskSwap 略高（21.0% vs 18.0%）。
- 严格的同环境对比需在本机重评原 modulation run @50000（其私有 bucket 当前令牌不可见，计划 D3 未做）。
- seed 42／0 补评未做（计划 D4）。

## ⑫ 归档文件

`records/`：`v17-train.summary.log`（清洗后训练日志）、`v17-ckptup.summary.log`、`v17-checkonly.summary.log`、`v17-smoke.summary.log`、`v17-g6b.summary.log`、`v17-cap4.summary.log`、`v17-cap3.summary.log`、`v17-wanenv.summary.log`、`v17-wanrestore.summary.log`。ckpt 在 `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-ctx-motion-b128-50k/` 与上述公开 bucket。
