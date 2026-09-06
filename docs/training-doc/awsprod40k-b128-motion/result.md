# awsprod40k-b128-motion — 结果

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。起跑 commit `934ccea`（clean HEAD），配置 `mme_vla_suite_b128`，
8 卡 w16、global batch 128、40k step、seed 42；起止与配置细节见 `launch.md`。

## 结论先行

- **跑完**：`2026-09-04 20:13:14 → 2026-09-06 00:46:09`，**28 h 33 min**（预估 30 h），`EXIT_CODE=0`；8 个 checkpoint 齐全
  （`5000, 10000, …, 35000, 39999`，每个 11 G，共 87 G，`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/`），
  run 快照 `history_config.resolved.sha256 = 94d8660…927fb`、wandb run `441nhaiz`。
- **训练 loss 收敛到 1e-3 量级**（下表）；`grad_norm` 从 28 降到 0.02，`param_norm` 1803 → 1869（`freeze_filter='.*img.*'` 的 SigLIP 不动，见
  `siglip-ab-replay-40k` 的 `CKPT_IMG_FROZEN=PASS`）。这是训练 loss，**没有验证集**——launch.md 已述的 50.6 epoch 过拟合风险只能由闭环评估回答。
- **步时**：checkpoint 区间实测 **2.570 s/step**（5k→10k，12,852 s）与 **2.562 s/step**（35k→40k，12,808 s），比 bench-b128-util 8 卡 w16 档外推的 2.656 快约 3%；
  全程含编译与 8 次保存折合 2.569 s/step。
- **GPU util（AGENTS 16 口径，15 s 采样，8 卡合并）**：训练窗口 `20:13:13 → 00:46:10` 均值 **71.94%**、0% 采样占比 **26.4%**（54,808 样本）；
  稳态窗口 `20:45 → 00:40` 均值 **72.20%**、0% 占比 **26.1%**；逐卡均值 71.9–72.3% 无离群。与 bench 档位实测 70.25% 一致——8 卡档 27% 空转是已知的
  数据侧瓶颈（`bench-b128-util/result.md`），本轮不作优化。显存峰值 80.0–80.2 GB/卡（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 预分配，非真实占用）。
- **下一步**：正式评估 4 任务 × 50 集 test split，留档 `docs/training-doc/eval-awsprod40k-b128-motion/`。本轮无 nomotion 对照，单侧数字不回答 motion 是否有效。

## loss 里程碑（`records/run.txt`，`log_interval=100`）

| step | loss | grad_norm | llm_grad_norm | mem_enc_norm | param_norm |
|---|---|---|---|---|---|
| 0 | 0.5666 | 28.27 | 21.11 | 18.58 | 1803.46 |
| 100 | 0.1961 | 11.07 | 8.21 | 7.36 | 1803.46 |
| 5000 | 0.0045 | 0.0388 | 0.0307 | 0.0118 | 1808.48 |
| 10000 | 0.0024 | 0.0260 | 0.0205 | 0.0048 | 1819.91 |
| 20000 | 0.0015 | 0.0241 | 0.0200 | 0.0029 | 1838.32 |
| 30000 | 0.0012 | 0.0216 | 0.0180 | 0.0034 | 1854.43 |
| 39000 | 0.0010 | 0.0192 | 0.0160 | 0.0015 | 1867.70 |
| 39900 | 0.0010 | 0.0205 | 0.0173 | 0.0015 | 1868.97 |

## checkpoint 落盘时刻（`Finished asynchronous save` 行）

| step | 时刻 | 保存耗时 |
|---|---|---|
| 5000 | 09-04 23:58:58 | 13.35 s（首次） |
| 10000 | 09-05 03:33:10 | 3.95 s |
| 15000 | 09-05 07:09:28 | 3.24 s |
| 20000 | 09-05 10:35:05 | 4.36 s |
| 25000 | 09-05 14:01:23 | 3.76 s |
| 30000 | 09-05 17:28:36 | 4.19 s |
| 35000 | 09-05 21:02:41 | 3.85 s |
| 39999 | 09-06 00:46:09 | 4.67 s |

## 产物（`records/`）

- `run.txt`：训练全程日志（8,718 行，含 `EXIT_CODE=0`）。
- `gpu_util_lms500_first30min.csv`：前 30 min `nvidia-smi -lms 500` 密采（28,672 行）。
- `gpu_util_15s_full.csv`：全程 15 s 采样（56,112 行；采样器于 09-06 01:35 手动停止，00:46 之后为空转，统计时截到 `00:46:10`）。
- 统计脚本：一次性 python（csv 逐行、按时间窗过滤、逐卡均值与 `util==0` 占比），口径同 `scripts/training/util/analyze_gpu_util.py`。
