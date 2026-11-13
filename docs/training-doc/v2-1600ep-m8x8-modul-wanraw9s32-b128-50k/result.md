# v2-1600ep-m8x8-modul-wanraw9s32-b128-50k 训练结果

起跑档案见 [launch.md](launch.md)；计划见仓库根目录 [1112-4task-modul-wanraw9-s32-50k-plan.html](../../../1112-4task-modul-wanraw9-s32-50k-plan.html)；评估档案见 [local692-wanraw9s32-50k](../local692-wanraw9s32-50k/result.md)。

## ① 一句话结论

第三次起跑（`commitV15.17Beta` 3addd7c）50,001 步正常跑完，10 份 checkpoint（5000…50000）齐全，最终 50000 真实加载通过；loss 单调下降、全程无 nan / OOM；10 份 ckpt 已传公开 bucket 并读回校验通过。策略成功率 262/692 = 37.86%（评估另档），低于 v11 Wan 池化 768 的 43.06%。

```
CKPT_LOAD=PASS step=50000 leaves=65 resolved_sha256=819a001c58b00d4a4110835b913ac1fb58238fb71d272eaa33a429f6dcb53062
TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10
EXIT_CODE=0
HF_VERIFY=PASS ckpts=10
```

链脚本随后输出 `CHAIN=STOP reason=eval_log_exists 拒绝覆盖评估日志 …/r9-eval.log`（`EXIT_CODE=1`）：这是主会话按用户「每卡 4 worker」指令预先放的占位文件挡住自动 10 片评估，属预期，不是训练失败（见 ⑩ 第 6 条）。

## ⑥ 硬件与耗时

AWS 8×A100-SXM4-80GB，本地 NVMe RAID `/dev/md0`，FSDP8、batch 128、num_workers 16、seed 42（`run_meta.json` / `runtime.json`）。

| 项 | 值 |
|---|---|
| tmux | `r9-train`（第三次起跑） |
| `START_UTC` | 2026-11-13T01:42:01Z（美东 11-12 20:42:01） |
| `Step 0` 完成 | 美东 11-12 20:44:55（含编译约 2 分 54 秒） |
| step 50000 存盘 | 美东 11-13 12:29:20 |
| `END_UTC` | 2026-11-13T17:30:09Z（美东 11-13 12:30:09），总墙钟约 15 h 48 m |
| 平均步速 | (step 50000 存盘 − step 0) / 50000 = **1.133 s/步**；tqdm 全程报 1.1–1.2 s/it |
| 前 300 步计时 | `step_timing.jsonl` 300 行，`STEP_TIME` 均值 1.162 s/步，`dl_wait` 均值 0.076 s（约 6.5%）；起跑前末级测速 `BENCH300=PASS s_per_step=1.1467 dl_wait_frac=0.2854 gpu_util_mean=94.44` |
| wandb | run `tgm7yn5f`（项目 robomme-framesamp） |

各 ckpt 存盘时刻（美东 11-12 / 11-13）：5000 22:18、10000 23:51、15000 01:26、20000 03:02、25000 04:37、30000 06:13、35000 07:47、40000 09:20、45000 10:54、50000 12:29，每 5000 步约 93–96 分钟。

**/dev/shm**（`r9-shmmon` 每 60 s 采样，964 行，records/`shmmon.log`）：起跑 2 分钟内升到约 202–216 GB 后稳定在 **约 209–225 GB 平台**（峰值 225 GB，远低于 561 GB tmpfs 与 300 GB 告警线，`SHM_ALERT` 0 次）；在若干 ckpt 保存前后短暂回落（50–192 GB 各 1–2 个采样点）；训练结束美东 12:30 起回落到 1 GB。说明 FIX-SHM 的有界队列生效，第二次起跑的无界积压未再出现。

## ⑦ 训练过程行为

| step | loss | grad_norm | motion_enc_grad_norm | motion_enc_kernel_norm | mem_enc_norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0881 | 0.9649 | 0.0336 | 83.84 | 0.0805 |
| 5000 | 0.0084 | 0.0571 | 0.0038 | 84.22 | 0.0097 |
| 10000 | 0.0056 | 0.0361 | 0.0022 | 85.33 | 0.0052 |
| 15000 | 0.0044 | 0.0318 | 0.0017 | 86.33 | 0.0035 |
| 20000 | 0.0036 | 0.0304 | 0.0014 | 87.29 | 0.0030 |
| 25000 | 0.0033 | 0.0343 | 0.0011 | 88.21 | 0.0025 |
| 30000 | 0.0029 | 0.0267 | 0.0010 | 89.10 | 0.0020 |
| 35000 | 0.0027 | 0.0260 | 0.0007 | 89.95 | 0.0015 |
| 40000 | 0.0024 | 0.0248 | 0.0006 | 90.76 | 0.0014 |
| 45000 | 0.0023 | 0.0247 | 0.0006 | 91.58 | 0.0012 |
| 50000 | 0.0021 | 0.0232 | 0.0005 | 92.36 | 0.0011 |

与 v11（Wan 池化 768，同库同超参）对照：v11 @50000 loss 0.0020、grad_norm 0.0232，本轮 0.0021 / 0.0232，曲线形状几乎相同。全程无 nan、OOM、Killed、Traceback。

**motion token 激活统计**（每份 ckpt 落盘后主会话跑 `--mode act-stats`，64 样本、12,798 个 motion token；records/`act-stats.log` 原文）：

| step | mean | var |
|---:|---:|---:|
| 5000 | 0.0276 | 142.3 |
| 10000 | 0.389 | 780.4 |
| 15000 | 0.468 | 1378.7 |
| 20000 | 0.752 | 2022.2 |
| 25000 | 0.838 | 2701.2 |
| 30000 | 1.069 | 3534.9 |
| 35000 | 1.162 | 4450.5 |
| 40000 | 1.373 | 5331.2 |
| 45000 | 1.608 | 6395.9 |
| 50000 | 1.993 | 7279.2 |

方差随步数近线性增长（约每 5000 步 +700～+1000），与 `motion_enc_kernel_norm` 83.8→92.4 的缓慢增长同向；loss 正常、无 nan，按计划风险登记第 1 条只观察不处置。下一轮可考虑对 raw latent 做归一化（计划口径为「零额外处理」，本轮未做）。

## ⑩ 计划外事件与处置（按时间）

1. **第一次起跑 abort**（`commitV15.13Beta` fac3502，美东 11-12 16:57 起，`START_UTC=2026-11-12T21:57:58Z`）：第 299–300 步 `terminate called after throwing an instance of 'absl::lts_20230802::BadStatusOrAccess'`，runner `EXIT_CODE=134`，`CHAIN=STOP reason=train_exit runner_exit=134`。根因：`TRAIN_TIMING_STEPS=300` 令 `StepTiming` 前 300 步开 `jax.profiler` trace，`stop_trace()` 后与 L1 数据线程冲突。修复 FIX-PROF（`commitV15.14` 59acd84，合入 fc4a0e5）：新增 `TRAIN_TIMING_PROFILE`，raw9 runner 置 0；冒烟 `r9-smoke20-l2` `STEP_TIMING_DONE steps=10 profile=0`、`SMOKE20=PASS`。
2. **第二次起跑 /dev/shm 写满**（`commitV15.15Beta` 7ad7890，美东 11-12 17:42 起）：约第 7560 步（美东约 20:06）DataLoader worker `RuntimeError: unable to write to file </torch_…>: No space left on device (28)`，runner `EXIT_CODE=1`、`CHAIN=STOP reason=train_exit runner_exit=1`。根因：L1 `_Releaser` 队列无界（`queue.SimpleQueue`）积压共享内存批。修复 FIX-SHM（`commitV15.16` 8382c80，合入 132d549）：改 `queue.Queue(maxsize=1)` 反压；`SHM_LEAK=PASS batches=240 rel_q_max=1`、`PRE_MERGE_REVIEW=PASS base=7ad7890 tip=132d549 blocking=0`。
3. **train.py 不支持 resume**：ckpt 只存 params，两次作废尝试均从头重跑；作废产物以 `.try1` / `.try2` 后缀留存（`v1-store/logs/r9-train.try{1,2}.log`、`*.try{1,2}.runner.log`、`v1-store/bench/<run>/train-try{1,2}/`、`v1-store/train-runs/…/<run>.try1`），本档案只摘判定行，不整份归档。
4. **闸 G5 训推一致对拍挪到冒烟后**：用末级冒烟 ckpt 跑，`ADJUST_REVIEW=PASS scale=same findings=2`；结果见 launch.md ⑤节 G5 行。
5. **HF 私有存储额度 403**：ckpt 上传到 step 20000 时 HF 返回 `403 Forbidden: You need to set up automatic credit recharge to use more storage`（3 次重试均 403，records 未收此日志，原文在 `v1-store/logs/r9-ckptup.quota403.log`；此前 5000/10000/15000 已 `STEP_VERIFY=PASS`）。用户指出「hugging face 上传的是公开 bucket」并放行「我同意放行，你可以再试一下」，bucket `HongzeFu/robomme-4task-modul-wanraw9-s32-50k-v1` 改为公开（`private=False`）后续传；`r9-ckptup.log` 中 20000…50000 共 7 份 `STEP_VERIFY=PASS`，合计 10 份，末行 `HF_VERIFY=PASS ckpts=10`、`EXIT_CODE=0`（约美东 12:34 完成）。作废 try2 上传的 5000/ 前缀此前已删，新 run 重新上传并读回。上传脚本 `run_wanraw9s32_ckpt_export.sh` 的建 bucket 已同步改为公开（不带 `--private`）。
6. **评估改每卡 4 片**：用户要求「每卡 4 worker」，链脚本正在 `r9-train` 中运行不可改，主会话放占位 `v1-store/logs/r9-eval.log` 挡住链脚本自动起的 10 片评估（`CHAIN=STOP reason=eval_log_exists`，预期），训练结束后手动起 32 片评估（评估档案）。

## ⑪ 结论与下一步

训练侧交付完成，ckpt 公开备份并读回校验通过；状态机 `4task-wanraw9s32-train` → closed。下一步（非授权建议）：raw latent 归一化或 LayerNorm 的对照；把「不池化」与「stride 16→32」拆成两个单变量 run，以便归因（本轮按计划 D3 同时改两项，混杂）。

## ⑫ 归档文件（records/）

| 文件 | 内容 |
|---|---|
| `train.summary.log` | `r9-train.log` 按第 12 条命令 `tr '\r' '\n' \| grep -vE '%\|'` 清洗（8,282 行、约 1.0 MB；本 run 的进度条是 `tqdm_logging` 的 `Progress on:` 日志行而非 `%\|` 形态，清洗几乎不减行，照原口径保留；另去掉行尾空白以过 `git diff --check`）；`grep -c 'Saving checkpoint at step'` = 10 |
| `metrics.jsonl` | 逐 100 步全指标（501 行，含十进制与 hex 原值） |
| `act-stats.log` | 每份 ckpt 的 motion token 激活均值 / 方差 |
| `step_timing.jsonl` | 前 300 步逐步计时（`TRAIN_TIMING_PROFILE=0`，不含 profiler） |
| `shmmon.log` | `/dev/shm` 每 60 s 采样（`r9-shmmon`；去掉行尾空白，内容不变） |

未归档：runner 日志 `v1-store/logs/v2-1600ep-m8x8-modul-wanraw9s32-b128-50k.runner.log`（与 `r9-train.log` 内容重叠）、上传日志 `r9-ckptup.log`（1.7 MB）与 `r9-ckptup.quota403.log`、GPU 15 s 采样 `v1-store/bench/<run>/gpu_util_15s_full.csv`、作废尝试日志；checkpoint 权重留在 `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-wanraw9s32-b128-50k/` 与公开 bucket。
