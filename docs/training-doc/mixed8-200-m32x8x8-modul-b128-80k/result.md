# mixed8-200 32 帧 8×8 modulation 2048（无 motion）80k 正式训练——结果

本页是 AGENTS 第 12 条两段式档案的第二段，承载十二节中的 ①⑦⑧⑨⑩⑪⑫；版本与代码状态、配置与启动覆盖、数据与身份、起跑前闸门（含实施中调整）、训练中增量上传与 tmux 会话清单都在同目录 [launch.md](launch.md)，此处不重复。同一计划 [`1018-mixed8-2048-context-modul-plan.md`](../../../1018-mixed8-2048-context-modul-plan.md) 下 GPU 0–3 的 context 2048 已于 455 步叫停作测速留档，见 [mixed8-200-m32x8x8-context-b128-80k](../mixed8-200-m32x8x8-context-b128-80k/result.md)。结构与口径照搬上一轮 512 档案 [mixed8-200-m8x8-modul-b128-80k](../mixed8-200-m8x8-modul-b128-80k/result.md)。时刻一律写美东（EDT，UTC−4）；引用日志原文中的 UTC 时间戳时原样保留并括注。

## ① 一句话结论与指标速览

**80k 步训练一次跑完、`EXIT_CODE=0`，16 份 checkpoint 全部落盘，训练中已逐份公开在 `HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1`（bucket 自美东 2026-10-18 20:25 由增量会话以公开方式创建），训练结束后经正式驱动全量回读校验通过；训练 loss 从 step 0 的 0.0874 平稳降到末段约 0.0015，全程无 NaN、OOM、Traceback，稳态 1.856 s/步（比 512 的 1.718 s/步慢约 8%）。** 这只是训练 loss，没有验证集，策略好坏要等评估侧的闭环评估回答（见 ⑧）。

| 项 | 值 |
|---|---|
| 起跑 | 美东 2026-10-18 17:45:01（日志 `START_UTC=2026-10-18T21:45:01Z`）；训练循环首条 `Progress on` 在 17:46:34（日志 21:46:34 UTC），首条带步数的心跳在 17:48:36（`7.00it/80.0kit rate:6.1s/it`） |
| 结束 | 美东 2026-10-20 11:03:31（日志 `END_UTC=2026-10-20T15:03:31Z`），日志末行 `EXIT_CODE=0` |
| 总时长 | 41 h 18 min 30 s（148,710 s，含初始化、编译、16 次保存与退出）；tqdm 记录的训练循环耗时 `elapsed:41:16:39` |
| 起跑代码 | `GIT_HEAD=1c95c0606885df167bcfb7eb6bce73d7ae060b9f`（commitV11.28Beta），config `mme_vla_suite_b128_80k`，history yaml `perceptual-framesamp-modul-32frame-8x8.yaml`（32 帧 × 8×8 = 2048 token，无 motion） |
| 设备 | GPU 4–7（`CUDA_VISIBLE_DEVICES=4,5,6,7`，见 launch.md），`TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8`；GPU 0–3 同期先后跑过 context 2048（已叫停）与 context 512+motion（已取消），见 ⑦「GPU 0–3 并行任务」 |
| 稳态步时 | 相邻 checkpoint 区间 1.8547–1.8622 s/步（除含 44.77 s 长保存的 5000→10000 外，14 个区间都在 1.8547–1.8578），5000→79999 合计 74,999 步平均 **1.8559 s/步**（约 69.0 samples/s）；tqdm 心跳 13,350 条中 `rate:1.8s/it` 8,274 条、`rate:1.9s/it` 5,016 条 |
| 最终 loss | `Step 79900: grad_norm=0.0216, llm_grad_norm=0.0186, loss=0.0015, mem_enc_norm=0.0017, param_norm=1870.6018`；step 79000–79900 十条日志 loss 均值 0.00150，75000–79900 五十条均值 0.00150 |
| wandb | 项目 robomme-framesamp，run `plp07t4d` |
| checkpoint | `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m32x8x8-modul-b128-80k/` 下 5000…75000 每 5000 一份加 79999，共 16 份，每份约 11.88 GB，合计 190,054,399,659 B（驱动 `README_RENDERED steps=16 ckpt_bytes=190054399659`） |
| 公开 bucket | `HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1`，346 个文件、190,056,497,175 B，`private=False`、匿名可读 |
| 上传前校验清单 | `SHA256SUMS.pre.txt` 345 行、40,638 B、sha256 `08b681a214d59fcdfa8db9433195ea1b79f5c8f19f57912cc4d27b6147d3daba`（本地 stage 副本复核一致，原样存于 [records/SHA256SUMS.pre.txt](records/SHA256SUMS.pre.txt)） |
| 只读锁 | 本 run 收尾时按 1019 计划 Q3 裁决**不恢复**，移交 GPU 0–3 的 context+motion run（handoff 任务 `mixed8-ctxmotion-train`）；该 run 于美东 10-20 11:44:33 应用户要求取消后，主会话按 `v1-store/m8v-scripts/m2k-chmod-before.txt`（395 行）+ `cmo-chmod-added.txt`（5 行）共 400 行逐文件恢复并逐个 stat 核对：`PERM_RESTORED=OK files=400`（handoff 提交 `7eb95e8`，美东 11:46:12；合并 `1342e48`） |

loss 节点（`Step N:` 行原值；openpi 按 `log_interval=100` 对区间内各步取均值后打印，step 0 除外，口径同 512 档案；最后 79901–79999 共 99 步没有独立标量日志）：

| 日志步 | loss | grad_norm | llm_grad_norm | mem_enc_norm | param_norm |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0874 | 1.3995 | 1.3866 | 0.1190 | 1815.5465 |
| 100 | 0.0568 | 0.5424 | 0.5359 | 0.0450 | 1815.5463 |
| 1000 | 0.0149 | 0.1385 | 0.1355 | 0.0120 | 1815.5583 |
| 5000 | 0.0079 | 0.0627 | 0.0521 | 0.0256 | 1817.0952 |
| 10000 | 0.0051 | 0.0363 | 0.0304 | 0.0096 | 1821.1737 |
| 20000 | 0.0036 | 0.0287 | 0.0243 | 0.0048 | 1828.8014 |
| 30000 | 0.0029 | 0.0248 | 0.0210 | 0.0032 | 1836.2345 |
| 40000 | 0.0023 | 0.0227 | 0.0195 | 0.0026 | 1843.4998 |
| 50000 | 0.0021 | 0.0217 | 0.0185 | 0.0025 | 1850.5728 |
| 60000 | 0.0017 | 0.0227 | 0.0198 | 0.0022 | 1857.4291 |
| 70000 | 0.0016 | 0.0225 | 0.0199 | 0.0021 | 1864.1499 |
| 75000 | 0.0016 | 0.0199 | 0.0168 | 0.0019 | 1867.4177 |
| 79000 | 0.0015 | 0.0225 | 0.0196 | 0.0017 | 1870.0275 |
| 79900 | 0.0015 | 0.0216 | 0.0186 | 0.0017 | 1870.6018 |

全程最低 loss 为 0.0014（4 位小数并列），出现在 step 76400、77900、78200、78500、78600、78800。每 1000 步一行的完整表见 [records/loss_every_1k.csv](records/loss_every_1k.csv)。

checkpoint 落盘时刻（`Finished asynchronous save (blocking + background)` 行，日志时间为 UTC）：

| step | 落盘（美东） | 日志原文（UTC） | 保存耗时 | 距上一份 |
|---|---|---|---:|---|
| 5000 | 10-18 20:23:28 | 00:23:28.732 | 11.72 s | 首份（含编译） |
| 10000 | 10-18 22:58:39 | 02:58:39.710 | 44.77 s | 9311.0 s，1.8622 s/步 |
| 15000 | 10-19 01:33:20 | 05:33:20.599 | 12.13 s | 1.8562 s/步 |
| 20000 | 10-19 04:07:57 | 08:07:57.232 | 18.46 s | 1.8553 s/步 |
| 25000 | 10-19 06:42:32 | 10:42:32.582 | 11.48 s | 1.8551 s/步 |
| 30000 | 10-19 09:17:08 | 13:17:08.828 | 19.63 s | 1.8552 s/步 |
| 35000 | 10-19 11:51:42 | 15:51:42.175 | 12.77 s | 1.8547 s/步 |
| 40000 | 10-19 14:26:18 | 18:26:18.191 | 22.81 s | 1.8552 s/步 |
| 45000 | 10-19 17:00:54 | 21:00:54.089 | 23.78 s | 1.8552 s/步 |
| 50000 | 10-19 19:35:29 | 23:35:29.867 | 28.12 s | 1.8552 s/步 |
| 55000 | 10-19 22:10:18 | 02:10:18.900 | 32.11 s | 1.8578 s/步 |
| 60000 | 10-20 00:44:57 | 04:44:57.858 | 29.04 s | 1.8558 s/步 |
| 65000 | 10-20 03:19:31 | 07:19:31.523 | 21.53 s | 1.8547 s/步 |
| 70000 | 10-20 05:54:09 | 09:54:09.844 | 28.36 s | 1.8557 s/步 |
| 75000 | 10-20 08:28:48 | 12:28:48.037 | 28.21 s | 1.8556 s/步 |
| 79999 | 10-20 11:03:22 | 15:03:22.315 | 28.44 s | 9274.3 s / 4999 步，1.8552 s/步 |

上传与校验：16 份 step 目录在训练中由 `m2k-inc-modul` 逐份增量上传（`INC_END run=mixed8-200-m32x8x8-modul-b128-80k done=16`，`EXIT_CODE=0`，末份 79999 于美东 10-20 11:07 `INC_UPLOADED`），训练与 inc 都退出后跑正式驱动 `scripts/dataset/hf_export/run_mixed8_200_vla_modul2048_80k_ckpt_export.sh`（未改代码）。SMOKE 会话 `cc-m2k-upsmoke` 美东约 11:46 起（主会话记录），日志末次写入 11:50:51（文件 mtime）；全量会话 `cc-m2k-upload` 约 11:51 起（stage `README.md` mtime 11:51:03），日志末次写入 12:08:42（文件 mtime）。判定行原文：

```text
# cc-m2k-upsmoke（SMOKE=1，日志 m2k-ckptmodul-upload-smoke.log）
PROVENANCE_MATCH=PASS motion=False fs8_store_meta=ace85246419836d3 resolved=91512306b3aaaaa5
README_RENDERED steps=16 ckpt_bytes=190054399659 git_head=1c95c0606885
LOCAL_FILES=346 LOCAL_BYTES=190056497175 PRE_LINES=345 SRC_FILES=343
HYGIENE=PASS scanned=88 hits=214 ungated=0
SMOKE=PASS（到阶段 3 为止，未碰网络与凭据）
全部完成
EXIT_CODE=0

# cc-m2k-upload（全量，日志 m2k-ckptmodul-upload.log）
PROVENANCE_MATCH=PASS motion=False fs8_store_meta=ace85246419836d3 resolved=91512306b3aaaaa5
README_RENDERED steps=16 ckpt_bytes=190054399659 git_head=1c95c0606885
LOCAL_FILES=346 LOCAL_BYTES=190056497175 PRE_LINES=345 SRC_FILES=343
HYGIENE=PASS scanned=88 hits=214 ungated=0
HF_WHOAMI=HongzeFu
BUCKET_CREATED=HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1 private=False
BUCKET_FILES_AT_START=338 BUCKET_BYTES_AT_START=190054399659
UPLOAD_DONE batches=17 root_files=7
BUCKET_FILES=346 BUCKET_BYTES=190056497175
DF_CHECK avail=3957903929344 readback_need=190056497175 verify_have=0 margin=400000000000
SHA256_PRE_POST_DIFF=0
RESULT=PASS
SRC_UNCHANGED=OK checked=343
BUCKET_PUBLIC_FINAL private=False
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
全部完成
EXIT_CODE=0
```

计数对账：16 个 step 目录在源侧共 338 个文件（各 step 文件数见 ⑩ 表），即驱动起跑时 bucket 内的 338 个；`BUCKET_BYTES_AT_START` 与 `ckpt_bytes` 逐字节相等（190,054,399,659 B），且 inc 日志里没有任何根文件上传记录（对 `history_config|motion_provenance|README|ROOT_` 计数为 0，16 条 `Sync plan` 的目标全部是 step 目录），说明本 run **没有**像 512 那样提前公开根文件，起跑时 bucket 里是纯 ckpt。驱动补传 `_run-meta/train.log` 1 个（2,050,642 B）与根下平文件 7 个（`README.md` 5,007 B、`SHA256SUMS.pre.txt` 40,638 B、`history_config.resolved.sha256` 65 B、`history_config.resolved.yaml` 406 B、`history_config.txt` 43 B、`motion_provenance.json` 707 B、`wandb_id.txt` 8 B）后为 346 个，增量 2,097,516 B 恰为 190,056,497,175 − 190,054,399,659；`SRC_FILES=343` 是 338 个 step 文件加 5 个源 run 根下平文件，`PRE_LINES=345` 是 346 个文件除去 `SHA256SUMS.pre.txt` 自身。

## ⑦ 训练过程行为

**loss 与梯度。** 起步 loss 0.0874（与 512 同值），前 1000 步降到 0.0149，warmup 结束的 5000 步为 0.0079，此后总体缓降（逐 1000 步表有 0.0001 级小回升 9 处，如 11000→12000 的 0.0047→0.0048、76000→77000 的 0.0015→0.0016；最大一处是 41000→42000 的 0.0022→0.0025）：10k 0.0051、20k 0.0036、40k 0.0023、60k 0.0017、末段 0.0015。5000 步以后 loss 与 `grad_norm` 的最大值都出现在 step 5000 本身（0.0079 / 0.0627），之后没有明显反弹或尖峰。`grad_norm` 从 1.40 降到 0.02 左右后基本持平（70k–79.9k 一百条均值 0.0215），`llm_grad_norm` 与之同步、数值接近；`mem_enc_norm`（按字段名推测为 memory encoder 部分的梯度范数）从 0.119 降到约 0.0017（79000–79900 十条均值 0.00181）。`param_norm` 1815.55 → 1870.60，增长平滑（`freeze_filter='.*img.*'` 冻结 SigLIP，可训练参数 2886.9 MB、memory 相关 84.4 MB，见日志 `Trainable Model Size` / `Memory-related  Size` 行，与 512 同值）。学习率是常数 5e-5（peak==decay==5e-5，warmup 5000），末段逐 1000 步区间均值（`loss_bin_mean`）60000 行 0.00177、70000 行 0.00161、78000 行 0.00146、79000 行 0.00150，每 10k 步仍降约 0.00015，没有到平台。

**与 512 的训练 loss 对照（仅训练 loss、同库同超参同步数，不代表策略好坏）。** 同步数节点上本 run 一律更低：10k 0.0051 对 0.0062、40k 0.0023 对 0.0030、79.9k 0.0015 对 0.0019；同目录 512+motion 末 1k 步均值为 0.00142（见 [总索引](../README.md)）。三条都没有验证集，差异是否转化为闭环成功率由评估回答。

**异常检测。** 对原始日志 `tr '\r' '\n'` 后 `grep -ciE 'traceback|nan|out of memory|RESOURCE_EXHAUSTED|Killed'` 为 0；800 条 `Step N:` 行五项标量全部有限。训练进程未重启、未 resume，`Found 0 checkpoint steps` 后从 pi05_base 初始化一次跑到底。

**步时。** 首个 7 步含编译（`rate:6.1s/it`），13 步时 2.4 s/it，之后稳定在 1.8–1.9 s/it。以 checkpoint 落盘时刻计，15 个相邻区间在 1.8547–1.8622 s/步之间，最慢的 5000→10000 含 step 10000 那次 44.77 s 的长保存，其余 14 个在 1.8547–1.8578，前后段没有漂移；比起跑档案按 20 步短测估的 1.84 s/步、约 41 h 慢约 0.9%，实际总时长 41.31 h。比同库 512（1.7183 s/步）慢约 8.0%，即记忆 token 从 512 增到 2048 在 modulation 接入下只带来约 8% 步时开销（同条件 context 2048 为 10.09 s/步，见 context 档案）。tqdm 心跳的 `rate` 分布：1.8 s/it 8,274 条、1.9 s/it 5,016 条、2.0 s/it 19 条、2.1–2.3 s/it 8 条、2.4–14.3 s/it 31 条，另有 2 条为 `rate:-`（训练循环刚启动、尚无速率）。这 31 条的构成：23 条落在保存 checkpoint 那一刻（5000…75000 与 79999 各 1–2 条，见 ⑩ 第 3 条），2 条是起步编译期（7 步 6.1 s/it、13 步 2.4 s/it），其余 6 条是 10.8k、21.6k、32.4k、43.2k、54.1k、75.7k 附近 2.4–3.7 s/it 的单点抖动，下一条心跳即回落；这些位置大致每隔约 10.8k 步一次，与 512 档案记下的 10.8k、21.6k、43.2k 抖动位置一致，原因未核实。2.1–2.3 s/it 的 8 条未逐条归因。

**保存。** 16 次异步保存耗时 11.48–44.77 s（阻塞部分 4.51–40.06 s，如 79999 `Finished blocking save in 19.84 seconds`），后半程略长于前半程（50k 起 21.5–32.1 s）；最长一次是 step 10000（阻塞 40.06 s、合计 44.77 s），原因未核实。对区间步时影响见上段，单次长保存只把所在区间拉高约 0.007 s/步。keep_period=5000 使 5000…75000 全部保留（79999 保存时日志逐条 `Preserving Checkpoint[step=…]`）。

**训练中增量上传对训练的影响。** inc 上传每份约 1 分钟（`INC_STEP_BEGIN` 到 `INC_UPLOADED` 0–1 min），都在每次保存完成后约 1–4 分钟开始；各区间步时差异不超过 0.4%，看不出上传影响。

**GPU 0–3 并行任务。** 本 run 在 GPU 4–7 跑的 41 小时里，GPU 0–3 先后有：context 2048（`m2k-train-ctx`，美东 10-18 17:43:57 起，19:09:36 应用户要求叫停，在本 run 首份 checkpoint 之前）；context 512+motion160 的 G4 短测（美东 10-19 00:42:42–00:48:46）与正式训练 `mixed8-200-m8x8-context-motion-b128-80k`（日志 `START_UTC=2026-10-19T04:52:22Z` 即美东 10-19 00:52:22 起，10-20 11:44:33 应用户要求在 32.4k/80k 取消，日志尾行 `EXIT_CODE=CANCELLED`）。后者起跑落在本 run 10000→15000 区间内，该区间 1.8562 s/步、其后各区间 1.8547–1.8578，与起跑前的区间没有可见差别；本 run 结束（11:03）早于该 run 取消（11:44）。

**GPU 利用率。** 本 run 未另做 GPU util 采样归档，本档案不给 util 结论（AGENTS 第 16 条口径的数字只能来自专门采样）；每卡显存峰值只有起跑前 20 步短测的 67907 MiB（launch.md G4）。

## ⑧ 训练后评估

训练侧不做评估；评估由评估侧会话另档（本档案写作时未核实评估侧是否已排期）。训练侧只负责 checkpoint 齐全并公开在 bucket（见 ①），除此之外未改任何仓库或 bucket。

## ⑨ 用户决策记录

起跑前的决策（2048 context 与 modulation 4+4 分别跑、边训练边上传、直接写 Markdown 计划、都用 32 帧 8×8、OOM 即报错、「开工」等）原话已在 [launch.md](launch.md)，本节只记训练期间与收尾阶段的。

**GPU 0–3 的 context 2048 叫停（美东 2026-10-18 约 19:09）**：「停止context2048 作为测速留档」「大产物删除」。只涉及 GPU 0–3 的 context run 与本计划三份短测 checkpoint，本 run 照常继续（见 context 档案）。

**只读锁移交（美东 2026-10-19）**：1019 context+motion 计划 Q3 推荐「2048 收尾时不恢复，改由 context+motion run 结束后按 `m2k-chmod-before.txt` + `cmo-chmod-added.txt` 一次恢复」，用户「1234都按照推荐的」「再审查一遍 没问题就开始」，即 Q3 按推荐执行（见 1019 计划实施进展第 1 条、handoff 该任务 `pre_handoff` 第 3 项）。

**收尾阶段（美东 2026-10-20）**，按时间：

1. 约 11:42「检查现在正在跑的两个训练 跑完了吗」——本 run 已于 11:03:31 `EXIT_CODE=0` 结束、inc 已 `INC_END`。
2. 约 11:4x（GPU 0–3 那条）「取消 GPU 零零到三的这个任务，留档把只把文件关键文件 log 留档这些大的ckpt都删掉」——context+motion run 于 11:44:33 取消（原话据 handoff `mixed8-ctxmotion-train`）。
3. 约 11:45「统一恢复状态记录也要更新。」——主会话按 Q3 以 400 行原权限逐文件恢复，`PERM_RESTORED=OK files=400`，并更新 handoff（`7eb95e8`，美东 11:46:12 提交）。
4. 约 11:46（晚于 handoff `7eb95e8` 提交、早于 SMOKE 起跑）「Modulation 2048这个 Hugging Face 都上传完毕了吗？」——主会话只读核查：16 份 ckpt 都在 bucket，逐 step 文件数与字节和本地一致（`LOCAL_EQ_HF=PASS`，主会话现场比对 `hf buckets ls -R` 逐 step 文件数与字节 vs 本地 `find`，不在日志文件中），但正式驱动未跑、bucket 缺 README / 校验清单 / 训练日志。
5. 约 11:46「做好这个收尾的上传驱动，全部做好。」——主会话跑 SMOKE=1 → 全量，判定行见 ①。

## ⑩ 计划外事件与处置

**1. 训练中增量上传。** 本计划起跑前即定为边训练边上传（launch.md「训练中增量上传」节，脚本沿用 512 那轮的 `m8v-ckpt-inc.sh`），不是实施中调整。收尾实测：inc 会话 `m2k-inc-modul` 美东 2026-10-18 17:45:30 `INC_START`，首份 5000 上传时以公开方式创建 bucket（`BUCKET_CREATED=HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1 private=False`），16 份每份 `HYGIENE=PASS scanned=5 hits=0 ungated=0`、`Sync plan` 中 `Skips: 0`，无 `UPLOAD_FAIL`、无重试中断，`INC_END run=mixed8-200-m32x8x8-modul-b128-80k done=16 训练会话已退出；剩余补传与全量验收交正式驱动`，`EXIT_CODE=0`。正式驱动阶段 5 对 16 个 step 全部打出「已完成，跳过」，只补传 `_run-meta` 与根下平文件；最终验收判定行见 ①。逐份记录（inc 日志原文，美东）：

| step | `INC_STEP_BEGIN` | `INC_UPLOADED` | files | bytes |
|---|---|---|---:|---:|
| 5000 | 10-18 20:25 | 20:26 | 21 | 11,879,755,989 |
| 10000 | 10-18 23:01 | 23:02 | 20 | 11,879,359,961 |
| 15000 | 10-19 01:37 | 01:37 | 23 | 11,879,275,593 |
| 20000 | 10-19 04:12 | 04:13 | 22 | 11,879,139,241 |
| 25000 | 10-19 06:43 | 06:44 | 22 | 11,879,097,198 |
| 30000 | 10-19 09:19 | 09:20 | 21 | 11,878,832,821 |
| 35000 | 10-19 11:55 | 11:55 | 24 | 11,878,757,360 |
| 40000 | 10-19 14:30 | 14:31 | 22 | 11,878,544,415 |
| 45000 | 10-19 17:01 | 17:02 | 21 | 11,878,336,965 |
| 50000 | 10-19 19:37 | 19:38 | 21 | 11,878,126,269 |
| 55000 | 10-19 22:13 | 22:13 | 20 | 11,877,929,564 |
| 60000 | 10-20 00:48 | 00:49 | 18 | 11,877,788,923 |
| 65000 | 10-20 03:19 | 03:20 | 20 | 11,877,590,385 |
| 70000 | 10-20 05:55 | 05:56 | 19 | 11,877,457,161 |
| 75000 | 10-20 08:31 | 08:32 | 22 | 11,877,289,970 |
| 79999 | 10-20 11:07 | 11:07 | 22 | 11,877,117,844 |

合计 338 个文件、190,054,399,659 B，与驱动 `ckpt_bytes`、`BUCKET_BYTES_AT_START` 一致。各 step 文件数 18–24 不等（推测是 orbax ocdbt 数据分片数不同，未逐份核实），不是缺文件；驱动阶段 0 的「16 个 step 目录齐全且均已提交」与阶段 8 的逐行 sha256 对照覆盖了全部文件。inc 日志的日期由 `INC_START` 的 10-18 起按时刻回绕推出（日志行本身只记时分）。

**2. 与 512 的差异：没有提前公开根文件。** 512 那轮为评估侧 10k 冒烟提前两次上传了 run 根文件，驱动起跑时 bucket 是 337 = 334 + 3。本 run 训练期间没有这类请求，inc 只传 step 目录，驱动起跑时 `BUCKET_FILES_AT_START=338` 恰为纯 ckpt（对账见 ①）；README 与校验清单随正式驱动一次补齐。

**3. 存 checkpoint 时的瞬时步时。** 日志里 tqdm 恰在保存那一刻的单条心跳显示 2.5–14.3 s/it（step 10000 为 14.3 s/it，对应那次 40.06 s 的阻塞保存；25000 只有 2.5 s/it；其余 14 次在 3.6–8.6 s/it），10000、50000、55000、60000、70000、75000 之后紧跟的一条为 2.4–3.2 s/it，数分钟内回到 1.8–1.9 s/it。属于预期行为，区间平均步时（①表）不受影响，未作处置。

**4. 驱动阶段 6 首次计数不符。** 全量驱动阶段 6 打出「首次不符（files=345 bytes=190056497167），统计接口可能异步滞后，120s 后复查」，复查得 `BUCKET_FILES=346 BUCKET_BYTES=190056497175` 一致。这是 bucket 服务端统计异步滞后（AGENTS 第 15 条「服务端统计不采信」），驱动按设计复查；真正的验收是阶段 7–8 的全量回读与逐行 sha256 对照。与 512 那次（files=341 → 342，差 8 B）同一现象。

**5. 日志清洗不减体积。** 与 512 相同，本 run 的 tqdm 进度通过 logging 输出成独立的 `Progress on:` 行（共 13,350 条），不含 `%|`，所以 AGENTS 第 12 条 (3) 的清洗命令对训练日志几乎不删内容：驱动阶段 1 记为「`_run-meta/train.log` 清洗: 2050642 B → 2050642 B」，records 里的 `train.summary.log`（另去 ANSI 与行尾空白）为 16,602 行、2,050,578 B。为此另出只含 `Step N:` 行、保存完成行与起止横幅的 `train.steps.log`（口径见 ⑫）。

**6. 只读锁未在本 run 收尾时恢复。** 按 Q3 裁决移交 context+motion run，由主会话在该 run 取消后统一恢复（见 ① 表与 ⑨ 第 3 条）。本 run 自起跑到结束代码冻结在 `1c95c06`，正式驱动运行时锁也仍在（驱动只读代码、不改代码）。

## ⑪ 结论与下一步

**结论。** 本 run 按起跑配置一次跑完 80k 步，代码自起跑到结束冻结在 `1c95c060`，16 份 checkpoint 齐全，公开 bucket 经全量回读、逐行 sha256 对照、源侧第三遍 sha256 与匿名读确认（`RESULT=PASS`、`SHA256_PRE_POST_DIFF=0`、`SRC_UNCHANGED=OK checked=343`、`BUCKET_PUBLIC=True`、`PUBLIC_ANON_READ=OK`）。2048 modulation 的步时只比 512 慢约 8%（1.856 对 1.718 s/步），训练 loss 收敛平稳、同步数下低于 512；它不回答策略成功率，也不回答 2048 记忆比 512 好不好——那要等评估侧同口径评估。

**下一步与待用户定。**

1. 本机回读副本 `v1-store/exports/hf-ckpt-mixed8-200-vla-modul2048-80k/verify`（`du -sb` 190,056,614,156 B，`du -sh` 178G）是否删除——待用户定；bucket 校验已 PASS，删本地副本不影响远端。
2. 评估由评估侧另档，回来后在评估档案里与 512、512+motion 并排对照。
3. 交接：handoff 任务 `mixed8-2048-train` 由主会话推进——`pre_handoff`（训练 `EXIT_CODE=0`、两份 result.md 回写并 push、只读锁移交已 pass）→ done，`long_tail` 两项（`m2k-inc-modul` 的 `INC_END`、正式驱动 `SHA256_PRE_POST_DIFF=0`）均已完成 → closed。

## ⑫ 归档文件清单

`records/` 只放 git 还原不出的实测数据，不含 .sh / .yaml、不含权重。清洗命令同 512 档案：在 AGENTS 第 12 条 (3) 的基础上另去掉 ANSI 转义与行尾空白（保证 `git diff --check` 干净），三份上传日志再去掉 hf 进度条行（`Processing Files` / `New Data Upload` 与空行），保留全部判定行、`Sync plan` 与 `Uploads/Downloads/Deletes/Skips` 计数：

```bash
LG=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs
tr '\r' '\n' < $LG/mixed8-200-m32x8x8-modul-b128-80k.log | grep -vE '%\|' \
  | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g; s/[[:space:]]+$//' > records/train.summary.log
grep -E '^(RUN|CONFIG|HC|FSDP_DEVICES|NUM_WORKERS|START_UTC|GIT_HEAD|TRAIN_CMD|END_UTC|EXIT_CODE)=|^TRAIN_MESH |^Step |Finished asynchronous save' \
  records/train.summary.log > records/train.steps.log
# <src>:<dst> = m2k-ckptmodul-upload:upload-modul2048、m2k-ckptmodul-upload-smoke:upload-modul2048-smoke、m2k-inc-modul:inc-modul2048
tr '\r' '\n' < $LG/<src>.log | grep -vE '%\|' | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g; s/[[:space:]]+$//' \
  | grep -vE '^(Processing Files|New Data Upload)|^$' > records/<dst>.summary.log
```

| 文件 | 来源与口径 | 规模 |
|---|---|---|
| [records/train.summary.log](records/train.summary.log) | 训练日志 `v1-store/logs/mixed8-200-m32x8x8-modul-b128-80k.log`（2,050,642 B）清洗；起止横幅、TrainConfig 回显、800 条 `Step N:`、13,350 条 `Progress on:`、16 次保存、`END_UTC` 与 `EXIT_CODE=0` 全保留。核验：`grep -c '^Step '` 清洗后 800 = 原日志 800 | 16,602 行，2,050,578 B |
| [records/train.steps.log](records/train.steps.log) | 从上一份再筛：runner 横幅（RUN/CONFIG/HC/FSDP_DEVICES/NUM_WORKERS/START_UTC/GIT_HEAD/TRAIN_CMD）、`TRAIN_MESH`、全部 `Step N:` 行、16 条 `Finished asynchronous save`、`END_UTC`、`EXIT_CODE=` | 827 行，90,680 B |
| [records/loss_every_1k.csv](records/loss_every_1k.csv) | 由 `Step N:` 行解析；每 1000 步一行（step 0、1000、…、79000，共 80 行）。`loss,grad_norm,llm_grad_norm,mem_enc_norm,param_norm` 是该 step 那条日志的原值（本身是 100 步区间均值）；`n_bin` 与 `*_bin_mean` 是 [step, step+1000) 内 10 条日志（step, step+100, …, step+900）的算术平均。step 79900 的末条不在 79000 行原值里，但计入 79000 行的区间均值 | 81 行（含表头） |
| [records/upload-modul2048-smoke.summary.log](records/upload-modul2048-smoke.summary.log) | 正式驱动 `SMOKE=1`，tmux `cc-m2k-upsmoke`，日志 `v1-store/logs/m2k-ckptmodul-upload-smoke.log` | 23 行，1,548 B |
| [records/upload-modul2048.summary.log](records/upload-modul2048.summary.log) | 正式驱动全量，tmux `cc-m2k-upload`，日志 `v1-store/logs/m2k-ckptmodul-upload.log`；阶段 0–10 全部判定行 | 117 行，6,250 B |
| [records/inc-modul2048.summary.log](records/inc-modul2048.summary.log) | 训练中增量上传，tmux `m2k-inc-modul`，日志 `v1-store/logs/m2k-inc-modul.log`（原始 261,394 行、5,334,313 B，绝大部分是 hf 进度条）；`INC_START`、`BUCKET_CREATED`、16 组 `INC_STEP_BEGIN` / `HYGIENE` / `Sync plan` / `INC_UPLOADED`、`INC_END` 与 `EXIT_CODE=0` | 210 行，12,579 B |
| [records/SHA256SUMS.pre.txt](records/SHA256SUMS.pre.txt) | 驱动阶段 2 生成、已公开在 bucket 根下的上传前校验清单，从 `v1-store/exports/hf-ckpt-mixed8-200-vla-modul2048-80k/stage/` 原样复制（`cmp` 一致），供异地按路径核对 sha256 | 345 行，40,638 B，sha256 `08b681a2…7daba` |

不归档：checkpoint 权重（留在 `v1-store/train-runs/…` 与公开 bucket）、wandb 本地目录、驱动 stage 与 verify 副本、`m8v-ckpt-inc.sh`（全文见 512 档案 launch.md）、各 step 的 `hygiene-inc-<step>.json` 与 `hygiene-stage.json` 体检报告（在 `v1-store/exports/hf-ckpt-mixed8-200-vla-modul2048-80k/logs/`，判定行已在 summary 日志里）。
