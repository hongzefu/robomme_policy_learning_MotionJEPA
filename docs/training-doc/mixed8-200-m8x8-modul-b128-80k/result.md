# mixed8-200 8×8 modulation 512（无 motion）80k 正式训练——结果

本页是 AGENTS 第 12 条两段式档案的第二段，承载十二节中的 ①⑦⑧⑨⑩⑪⑫；②版本与代码状态、③启动与配置还原、④数据集与划分口径、⑤关键超参、⑥硬件与耗时预估，以及训练中途追加的「实施中调整：训练中增量上传 checkpoint」一节，都在同目录 [launch.md](launch.md)，此处不重复。另一条同时训练的 512+motion run 见 [mixed8-200-m8x8-modul-motion-b128-80k](../mixed8-200-m8x8-modul-motion-b128-80k/launch.md)。时刻一律写美东（EDT，UTC−4）；引用日志原文中的 UTC 时间戳时原样保留并括注。

## ① 一句话结论与指标速览

**80k 步训练一次跑完、`EXIT_CODE=0`，16 份 checkpoint 全部落盘并经正式驱动全量回读校验后公开在 `HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1`；训练 loss 从 step 0 的 0.0874 平稳降到末段约 0.0019，全程无 NaN、OOM、Traceback，稳态 1.718 s/步。** 这只是训练 loss，没有验证集，策略好坏要等评估侧的闭环评估回答（见 ⑧）。

| 项 | 值 |
|---|---|
| 起跑 | 美东 2026-10-17 01:00:36（日志 `START_UTC=2026-10-17T05:00:36Z`）；训练循环首条 `Progress on` 在 01:02:02（日志 05:02:02 UTC） |
| 结束 | 美东 2026-10-18 15:15:45（日志 `END_UTC=2026-10-18T19:15:45Z`），日志末行 `EXIT_CODE=0` |
| 总时长 | 38 h 15 min 09 s（137,709 s，含初始化、编译、16 次保存与退出）；tqdm 记录的训练循环耗时 `elapsed:38:13:25` |
| 起跑代码 | `GIT_HEAD=62a8ef15b4be267b56612d50562fc445160d041f`（commitV11.24Beta），config `mme_vla_suite_b128_80k`，history yaml `perceptual-framesamp-modul-8frame-8x8.yaml` |
| 设备 | GPU 0–3，`TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8`；GPU 4–7 同时在跑 motion 那条 run（4+4） |
| 稳态步时 | 相邻 checkpoint 区间 1.7176–1.7190 s/步，5000→79999 合计 74,999 步平均 **1.7183 s/步**（约 74.5 samples/s）；tqdm 心跳 13,339 条中 13,191 条为 `rate:1.7s/it` |
| 最终 loss | `Step 79900: grad_norm=0.0238, llm_grad_norm=0.0210, loss=0.0019, mem_enc_norm=0.0016, param_norm=1874.2955`；step 79000–79900 十条日志 loss 均值 0.00192，75000–79900 五十条均值 0.00192 |
| wandb | 项目 robomme-framesamp，run `jq08aey9` |
| checkpoint | `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-b128-80k/` 下 5000…75000 每 5000 一份加 79999，共 16 份，每份约 11.88 GB，合计 190,051,975,877 B（驱动 `README_RENDERED steps=16 ckpt_bytes=190051975877`） |
| 公开 bucket | `HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1`，342 个文件、190,054,069,413 B，`private=False`、匿名可读 |
| 上传前校验清单 | `SHA256SUMS.pre.txt` 341 行、40,118 B、sha256 `c4a2f32e55eeb08e5b99f0985619c164e79334d7e102bef595d468fc9496eee2`（本地 stage 副本复核一致，原样存于 [records/SHA256SUMS.pre.txt](records/SHA256SUMS.pre.txt)） |
| 只读锁 | 两条训练都写出 `EXIT_CODE=0` 后，美东约 16:00 按 `v1-store/m8v-scripts/m8v-chmod-before.txt`（385 行）逐文件恢复原权限，逐行 diff 为空：`PERM_RESTORED=OK` |

loss 节点（`Step N:` 行原值；openpi 按 `log_interval=100` 对区间内各步取均值后打印，step 0 除外，口径同 v2-1600ep-m8x8-modul-b128-60k 档案；最后 79901–79999 共 99 步没有独立标量日志）：

| 日志步 | loss | grad_norm | llm_grad_norm | mem_enc_norm | param_norm |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0874 | 1.3947 | 1.3818 | 0.1186 | 1815.5465 |
| 100 | 0.0568 | 0.5412 | 0.5347 | 0.0449 | 1815.5463 |
| 1000 | 0.0150 | 0.1382 | 0.1352 | 0.0119 | 1815.5583 |
| 5000 | 0.0094 | 0.0622 | 0.0549 | 0.0172 | 1817.1246 |
| 10000 | 0.0062 | 0.0371 | 0.0313 | 0.0089 | 1821.4897 |
| 20000 | 0.0043 | 0.0325 | 0.0279 | 0.0046 | 1829.7678 |
| 30000 | 0.0036 | 0.0273 | 0.0232 | 0.0031 | 1837.7623 |
| 40000 | 0.0030 | 0.0268 | 0.0234 | 0.0027 | 1845.5100 |
| 50000 | 0.0026 | 0.0244 | 0.0210 | 0.0024 | 1853.0403 |
| 60000 | 0.0023 | 0.0258 | 0.0226 | 0.0021 | 1860.3513 |
| 70000 | 0.0021 | 0.0246 | 0.0218 | 0.0019 | 1867.4668 |
| 75000 | 0.0020 | 0.0227 | 0.0197 | 0.0023 | 1870.9484 |
| 79000 | 0.0019 | 0.0250 | 0.0219 | 0.0017 | 1873.6769 |
| 79900 | 0.0019 | 0.0238 | 0.0210 | 0.0016 | 1874.2955 |

全程最低一条为 `Step 76700` 的 loss 0.0018。每 1000 步一行的完整表见 [records/loss_every_1k.csv](records/loss_every_1k.csv)。

checkpoint 落盘时刻（`Finished asynchronous save (blocking + background)` 行，日志时间为 UTC）：

| step | 落盘（美东） | 日志原文（UTC） | 保存耗时 | 距上一份 |
|---|---|---|---:|---|
| 5000 | 10-17 03:27:44 | 07:27:44.705 | 15.28 s | 首份（含编译） |
| 10000 | 10-17 05:50:59 | 09:50:59.337 | 13.58 s | 8594.6 s，1.7189 s/步 |
| 15000 | 10-17 08:14:12 | 12:14:12.240 | 14.26 s | 1.7186 s/步 |
| 20000 | 10-17 10:37:23 | 14:37:23.404 | 13.91 s | 1.7182 s/步 |
| 25000 | 10-17 13:00:35 | 17:00:35.294 | 13.71 s | 1.7184 s/步 |
| 30000 | 10-17 15:23:48 | 19:23:48.142 | 14.11 s | 1.7186 s/步 |
| 35000 | 10-17 17:46:59 | 21:46:59.882 | 14.47 s | 1.7183 s/步 |
| 40000 | 10-17 20:10:14 | 00:10:14.309 | 17.09 s | 1.7189 s/步 |
| 45000 | 10-17 22:33:22 | 02:33:22.561 | 15.14 s | 1.7177 s/步 |
| 50000 | 10-18 00:56:37 | 04:56:37.413 | 18.61 s | 1.7190 s/步 |
| 55000 | 10-18 03:19:45 | 07:19:45.806 | 17.49 s | 1.7177 s/步 |
| 60000 | 10-18 05:42:56 | 09:42:56.281 | 17.92 s | 1.7181 s/步 |
| 65000 | 10-18 08:06:05 | 12:06:05.876 | 16.00 s | 1.7179 s/步 |
| 70000 | 10-18 10:29:14 | 14:29:14.122 | 15.78 s | 1.7176 s/步 |
| 75000 | 10-18 12:52:24 | 16:52:24.200 | 16.88 s | 1.7180 s/步 |
| 79999 | 10-18 15:15:37 | 19:15:37.037 | 19.67 s | 8592.8 s / 4999 步，1.7189 s/步 |

上传与校验：16 份 step 目录在训练中由 `m8v-inc-512` 逐份增量上传（`INC_END run=mixed8-200-m8x8-modul-b128-80k done=16`，`EXIT_CODE=0`，末份 79999 于美东 15:19 `INC_UPLOADED`），训练与 inc 都退出后照原计划跑正式驱动 `run_mixed8_200_vla_512_80k_ckpt_export.sh`（计划 A1b 原驱动，未改代码）。SMOKE 会话 `m8v-ckpt512-upload-smoke` 美东 15:39 起，全量会话 `m8v-ckpt512-upload` 15:41 起，日志末次写入约 15:58（文件 mtime）。判定行原文：

```text
# m8v-ckpt512-upload-smoke（SMOKE=1）
PROVENANCE_MATCH=PASS motion=False fs8_store_meta=ace85246419836d3 resolved=804b25382af668d6
README_RENDERED steps=16 ckpt_bytes=190051975877 git_head=62a8ef15b4be
LOCAL_FILES=342 LOCAL_BYTES=190054069413 PRE_LINES=341 SRC_FILES=339
HYGIENE=PASS scanned=88 hits=214 ungated=0
SMOKE=PASS（到阶段 3 为止，未碰网络与凭据）
全部完成
EXIT_CODE=0

# m8v-ckpt512-upload（全量）
PROVENANCE_MATCH=PASS motion=False fs8_store_meta=ace85246419836d3 resolved=804b25382af668d6
README_RENDERED steps=16 ckpt_bytes=190051975877 git_head=62a8ef15b4be
LOCAL_FILES=342 LOCAL_BYTES=190054069413 PRE_LINES=341 SRC_FILES=339
HYGIENE=PASS scanned=88 hits=214 ungated=0
HF_WHOAMI=HongzeFu
BUCKET_CREATED=HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1 private=False
BUCKET_FILES_AT_START=337 BUCKET_BYTES_AT_START=190051977053
UPLOAD_DONE batches=17 root_files=7
BUCKET_FILES=342 BUCKET_BYTES=190054069413
DF_CHECK avail=4149691445248 readback_need=190054069413 verify_have=0 margin=400000000000
SHA256_PRE_POST_DIFF=0
RESULT=PASS
SRC_UNCHANGED=OK checked=339
BUCKET_PUBLIC_FINAL private=False
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
全部完成
EXIT_CODE=0
```

计数对账：16 个 step 目录在源侧共 334 个文件（各 step 文件数见 ⑩ 表），加训练前提前公开的 3 个根文件（`history_config.resolved.yaml`、`history_config.resolved.sha256`、`motion_provenance.json`，见 ⑩）即驱动起跑时 bucket 内的 337 个；`BUCKET_BYTES_AT_START` 比 `ckpt_bytes` 多出的 1,176 B 正是这 3 个文件 405 + 65 + 706 B。驱动补传 `_run-meta/train.log` 1 个与根下平文件 7 个（其中 3 个内容相同、重传覆盖）后为 342 个；`SRC_FILES=339` 是 334 个 step 文件加 5 个源 run 根下平文件，`PRE_LINES=341` 是 342 个文件除去 `SHA256SUMS.pre.txt` 自身。

## ⑦ 训练过程行为

**loss 与梯度。** 起步 loss 0.0874，前 1000 步降到 0.0150，warmup 结束的 5000 步为 0.0094，此后单调缓降：10k 0.0062、20k 0.0043、40k 0.0030、60k 0.0023、末段 0.0019。5000 步以后 loss 与 `grad_norm` 的最大值都出现在 step 5000 本身（0.0094 / 0.0622），之后没有反弹或尖峰。`grad_norm` 从 1.39 降到 0.02–0.03 区间后基本持平（70k–79.9k 一百条均值 0.0234），`llm_grad_norm` 与之同步，说明梯度绝大部分来自 LLM 分支；`mem_enc_norm`（memory encoder 梯度范数）从 0.12 降到约 0.0017。`param_norm` 1815.55 → 1874.30，增长平滑（`freeze_filter='.*img.*'` 冻结 SigLIP，可训练参数 2886.9 MB、memory 相关 84.4 MB，见日志 `Trainable Model Size` / `Memory-related  Size` 行）。学习率是常数 5e-5（peak==decay==5e-5，warmup 5000），末段 loss 仍以每 5000 步约 0.0001 的速度下降，没有到平台。

**异常检测。** 对原始日志 `tr '\r' '\n'` 后 `grep -ciE 'traceback|nan|out of memory|RESOURCE_EXHAUSTED|Killed'` 为 0；800 条 `Step N:` 行五项标量全部有限。训练进程未重启、未 resume，`Found 0 checkpoint steps` 后从 pi05_base 初始化一次跑到底。

**步时。** 首个 8 步含编译（`rate:5.1s/it`），约 20 步后稳定在 1.7 s/it。以 checkpoint 落盘时刻计，15 个相邻区间全部在 1.7176–1.7190 s/步之间，前后段没有漂移，比起跑档案按 20 步短测估的 1.84 s/步、约 41 h 快约 7%，实际总时长 38.25 h。tqdm 心跳的 `rate` 分布：1.6 s/it 90 条、1.7 s/it 13,191 条、1.8 s/it 19 条、1.9 s/it 14 条，2.0–5.4 s/it 共 24 条——后者几乎都落在保存 checkpoint 那一刻（见 ⑩ 第 3 条），另有 10.8k、21.6k、43.2k 附近三条 2.1–2.3 s/it 的单点抖动，下一条心跳即回到 1.7。

**保存。** 16 次异步保存耗时 13.58–19.67 s（阻塞部分约 10 s，如 79999 `Finished blocking save in 10.73 seconds`），后半程略长于前半程（40k 起 15–19 s），对区间步时无可见影响。keep_period=5000 使 5000…75000 全部保留（79999 保存时日志逐条 `Preserving Checkpoint[step=…]`）。

**训练中增量上传对训练的影响。** inc 上传每份约 1 分钟、`ionice -c 3 nice -n 19` 限流；上传窗口所在的区间（如 35000→40000、40000→45000）步时 1.7189 / 1.7177 s/步，与其他区间无差别。

**GPU 利用率。** 本 run 未另做 GPU util 采样归档，本档案不给 util 结论（AGENTS 第 16 条口径的数字只能来自专门采样）。

## ⑧ 训练后评估

训练侧不做评估（起跑档案「本轮不做 VLA 评测」，口径 15）；评估由 sled-aspen 评估会话按 1017-test-xhard5-12per 计划做 10k 冒烟 / 60k / 79999 正式评估，结果另档。训练侧为其做的只有两件：提前公开 run 根文件（见 ⑩ 第 2 条）与提供只读信息（见 ⑩ 第 6 条），除此之外未改任何仓库或 bucket。

## ⑨ 用户决策记录

起跑前的决策（512 改 80k、4 卡跑、`.git/info/exclude`、崩溃停下问用户、上传口径等）原话已在 [launch.md](launch.md)，本节只记训练期间的。

**训练中增量上传（美东 2026-10-17）**，原口径「训练结束后再上传」改为「训练中逐份上传」，详细见 launch.md「实施中调整」节：

1. 约 18:50「你现在会上传吗？就是训练完的 checkpoint 会上传吗？」
2. 约 19:00「你现在就可以开始上传了。」
3. 约 19:00（同一轮中途追加）「有一个 checkpoint 就上传。」
4. 约 19:00（同一轮中途追加）「然后你看一下还有哪些没有上传的？」
5. 约 19:05「先回答我，还有哪些产物没上传？」
6. 约 20:28「把你现在执行情况全部归档commit push，用子代理来做」

**根文件提前上传（美东 2026-10-18 约 01:05）**：评估侧请求为 10k 冒烟提前拿到 run 根文件，用户「同意上传」。

**跨会话沟通授权（美东 2026-10-18）**：

- 01:12「两个对话之间的沟通，我都同意，然后尽可能在不违反用户要求的情况下，尽可能满足对话的要求，并且每次收到对方的信息都要回复收到。」
- 约 01:20「纯回执不用再管了。」
- 约 01:25「对话之间的沟通都不需要用户批准。只要用户没有拒绝、明确拒绝的任务都可以继续沟通并且实现」

**是否续训到 100k（美东 2026-10-18 约 12:45）**：用户问「是否可以让带motion的继续训练下去到100k」。主会话分析：不能原地续——`train.py` 禁 `--resume`/`--overwrite`（「续跑语义有损」），checkpoint 只存 params + assets、没有优化器状态，且红线 18 冻结期不能改代码；常数学习率（peak==decay==5e-5）本身不是障碍。给出两个方案：A 从 79999 warm-start 再训 20k；B 从头训 100k。用户答「先不管了」，未执行。问题针对 motion 那条，本 512 run 同样按 80k 结束、未续训。

## ⑩ 计划外事件与处置

**1. 训练中增量上传（实施中调整，经用户授权）。** 起因、旧新口径对照、对抗验证 `ADJUST_REVIEW=PASS scale=same findings=5` 与脚本全文都在 launch.md「实施中调整」节。收尾实测：inc 会话 `m8v-inc-512` 美东 2026-10-17 19:08:05 `INC_START`，16 份每份 `HYGIENE=PASS scanned=5 hits=0 ungated=0`，无 `UPLOAD_FAIL`、无重试中断，`INC_END run=mixed8-200-m8x8-modul-b128-80k done=16 训练会话已退出；剩余补传与全量验收交正式驱动`，`EXIT_CODE=0`。正式驱动阶段 5 对 16 个 step 全部打出「已完成，跳过」，只补传 `_run-meta` 与根下平文件；最终验收判定行与原计划完全相同（见 ①）。逐份记录（inc 日志原文，美东）：

| step | `INC_STEP_BEGIN` | `INC_UPLOADED` | files | bytes |
|---|---|---|---:|---:|
| 5000 | 10-17 19:08 | 19:08 | 21 | 11,879,834,870 |
| 10000 | 10-17 19:08 | 19:09 | 22 | 11,879,321,777 |
| 15000 | 10-17 19:09 | 19:10 | 21 | 11,879,188,089 |
| 20000 | 10-17 19:10 | 19:10 | 21 | 11,879,035,793 |
| 25000 | 10-17 19:10 | 19:11 | 20 | 11,878,896,713 |
| 30000 | 10-17 19:11 | 19:12 | 20 | 11,878,709,178 |
| 35000 | 10-17 19:12 | 19:12 | 21 | 11,878,549,753 |
| 40000 | 10-17 20:12 | 20:13 | 21 | 11,878,348,505 |
| 45000 | 10-17 22:33 | 22:34 | 20 | 11,878,180,333 |
| 50000 | 10-18 00:59 | 01:00 | 21 | 11,877,983,521 |
| 55000 | 10-18 03:20 | 03:20 | 19 | 11,877,784,879 |
| 60000 | 10-18 05:45 | 05:46 | 23 | 11,877,611,306 |
| 65000 | 10-18 08:06 | 08:07 | 25 | 11,877,419,076 |
| 70000 | 10-18 10:32 | 10:32 | 20 | 11,877,203,671 |
| 75000 | 10-18 12:52 | 12:53 | 19 | 11,877,030,810 |
| 79999 | 10-18 15:18 | 15:19 | 20 | 11,876,877,603 |

合计 334 个文件、190,051,975,877 B，与驱动 `ckpt_bytes` 一致。各 step 文件数 19–25 不等（推测是 orbax ocdbt 数据分片数不同，未逐份核实），不是缺文件；驱动阶段 0 的「16 个 step 目录齐全且均已提交」与阶段 8 的逐行 sha256 对照覆盖了全部文件。

**2. 根文件提前上传，两次，均经对抗验证。** 评估侧会话（sled-aspen「Great Lakes 评估与 F1 计划修订」）为 10k 冒烟要求提前拿到 run 根文件，用户「同意上传」（美东 10-18 约 01:05）。

- 第一次（美东 01:06）：对抗验证 `ADJUST_REVIEW=PASS scale=same findings=5`，采纳「上传前 cmp 源文件」「whoami + private==false 断言」。本 run 上传 `history_config.resolved.yaml`（405 B，sha256 `804b2538…2f92`）与 `history_config.resolved.sha256`（65 B，`9ce42131…7706`）；两条 run 合计 5 个文件 `ROOT_READBACK=OK`（匿名读回 sha 一致），`ROOT_PRE_UPLOAD=PASS`。体检 `HYGIENE=PASS`。README 披露晚于文件公开，由正式驱动阶段 5 补上 README。
- 第二次（美东 01:20）：评估侧更正——其加载代码 `policy_config._load_resolved_snapshot` 对任何带 resolved yaml 的 run 都需要 `motion_provenance.json`，512 也不例外。复核 `ADJUST_REVIEW=PASS scale=same findings=2`；补传本 run `motion_provenance.json`（706 B，sha256 `99a726c9718b00977428560e4172dd3e226f1c76ccea486782d0cd5f4836cdea`），`COPY_IDENTICAL=OK`、`HYGIENE=PASS scanned=1 hits=1`（P9 `framesamp_root` 本机路径，账本 accept）、`ROOT_READBACK=OK`、`ROOT_PRE_UPLOAD=PASS`。

这 3 个文件后来被正式驱动按 stage 内容重传（sha256 与 `SHA256SUMS.pre.txt` 中对应行相同），计入最终 342 个文件与 `SHA256_PRE_POST_DIFF=0`。

**3. 存 checkpoint 时的瞬时步时。** 主会话心跳在保存 checkpoint 的那几步记到 1.8–1.9 s/it；日志里 tqdm 恰在保存那一刻的单条心跳显示 4.7–5.4 s/it（阻塞保存约 10 s 拉高平滑速率），下一条为 1.9–2.0 s/it，数分钟内回到 1.7 s/it。属于预期行为，区间平均步时（①表）不受影响，未作处置。

**4. 驱动阶段 6 首次计数不符。** 全量驱动阶段 6 打出「首次不符（files=341 bytes=190054069405），统计接口可能异步滞后，120s 后复查」，复查得 `BUCKET_FILES=342 BUCKET_BYTES=190054069413` 一致。这是 bucket 服务端统计异步滞后（AGENTS 第 15 条「服务端统计不采信」），驱动按设计复查；真正的验收是阶段 7–8 的全量回读与逐行 sha256 对照。

**5. 日志清洗不减体积。** 本 run 的 tqdm 进度通过 logging 输出成独立的 `Progress on:` 行（约每 10 s 一条，共 13,339 条），不含 `%|`，所以 AGENTS 第 12 条 (3) 的清洗命令对训练日志几乎不删内容：驱动阶段 1 记为「`_run-meta/train.log` 清洗: 2047760 B → 2047760 B」，records 里的 `train.summary.log` 也与原日志同为 16,592 行。为此另出只含 `Step N:` 行、保存完成行与起止横幅的 `train.steps.log`（口径见 ⑫）。

**6. 向评估侧提供的只读信息（不改任何仓库或 bucket）。** 本 run 相关：60000 逐文件清单 23 行（清单 sha256 `41930f22…f827`）、79999 逐文件清单 20 行（`edda8b46…df1d`）；另有两条 run 共用的 ASSETS_LOCK 口径说明（policy 仓库 `scripts/assets/ASSETS_LOCK.json` 仍指旧 full1600 encoder，建库靠显式传参满足检查）。motion 侧的清单、encoder 远端位置与 ASSETS_LOCK 细节记在 motion 那份档案。

**7. MotionJEPA-backup 退役。** 已在 MotionJEPA `v7.0-mixed8`（`f25f9d3`）与 policy `de86240` 登记，此处只引用；launch.md 中引用的计划 HTML 路径以那两处登记为准。

## ⑪ 结论与下一步

**结论。** 本 run 按起跑配置一次跑完 80k 步，代码自起跑到结束冻结在 `62a8ef15`（红线 18 只读锁，结束后 `PERM_RESTORED=OK`），16 份 checkpoint 齐全，公开 bucket 经全量回读、逐行 sha256 对照、源侧第三遍 sha256 与匿名读确认（`RESULT=PASS`、`SHA256_PRE_POST_DIFF=0`、`SRC_UNCHANGED=OK checked=339`、`BUCKET_PUBLIC=True`、`PUBLIC_ANON_READ=OK`）。训练 loss 收敛平稳、末段仍缓降；它不回答策略成功率，也不回答 motion 有没有用——后者要等评估侧对本 run 与 512+motion run 的同口径评估。

**下一步与待用户定。**

1. 本机回读副本 `v1-store/exports/hf-ckpt-mixed8-200-vla-512-80k/verify`（约 190 GB，`du` 178 GiB）是否删除——待用户定；bucket 校验已 PASS，删本地副本不影响远端。
2. 是否另起 100k（⑨ 的方案 A warm-start 20k / B 从头 100k）——用户 12:45 答「先不管了」，留作待定，不自行起跑。
3. 评估结果由 sled-aspen 评估会话另档，回来后在评估档案里与 motion run 并排对照。
4. 交接：handoff 任务 `mixed8-vla-train` 由主会话按 pre_handoff（两条 `EXIT_CODE=0` + 档案回写）→ done、long_tail 四项（`m8v-inc-512`、`m8v-inc-motion`、`m8v-ckpt512-upload`、`m8v-ckptmotion-upload` 全 PASS）→ closed 推进。

## ⑫ 归档文件清单

`records/` 只放 git 还原不出的实测数据，不含 .sh / .yaml、不含权重。清洗命令在 AGENTS 第 12 条 (3) 的基础上另去掉 ANSI 转义与行尾空白（保证 `git diff --check` 干净），三份上传日志再去掉 hf 进度条行（`Processing Files` / `New Data Upload` 与空行），保留全部判定行、`Sync plan` 与 `Uploads/Downloads/Deletes/Skips` 计数：

```bash
LG=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs
tr '\r' '\n' < $LG/mixed8-200-m8x8-modul-b128-80k.log | grep -vE '%\|' \
  | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g; s/[[:space:]]+$//' > records/train.summary.log
grep -E '^(RUN|CONFIG|HC|FSDP_DEVICES|NUM_WORKERS|START_UTC|GIT_HEAD|TRAIN_CMD|END_UTC|EXIT_CODE)=|^TRAIN_MESH |^Step |Finished asynchronous save' \
  records/train.summary.log > records/train.steps.log
# <src>:<dst> = m8v-ckpt512-upload:upload-512、m8v-ckpt512-upload-smoke:upload-512-smoke、m8v-inc-512:inc-512
tr '\r' '\n' < $LG/<src>.log | grep -vE '%\|' | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g; s/[[:space:]]+$//' \
  | grep -vE '^(Processing Files|New Data Upload)|^$' > records/<dst>.summary.log
```

| 文件 | 来源与口径 | 规模 |
|---|---|---|
| [records/train.summary.log](records/train.summary.log) | 训练日志 `v1-store/logs/mixed8-200-m8x8-modul-b128-80k.log` 清洗；起止横幅、TrainConfig 回显、800 条 `Step N:`、13,339 条 `Progress on:`、16 次保存、`END_UTC` 与 `EXIT_CODE=0` 全保留。核验：`grep -c '^Step '` 清洗后 800 = 原日志 800 | 16,592 行，2,047,696 B |
| [records/train.steps.log](records/train.steps.log) | 从上一份再筛：runner 横幅（RUN/CONFIG/HC/FSDP_DEVICES/NUM_WORKERS/START_UTC/GIT_HEAD/TRAIN_CMD）、`TRAIN_MESH`、全部 `Step N:` 行、16 条 `Finished asynchronous save`、`END_UTC`、`EXIT_CODE=` | 827 行，90,624 B |
| [records/loss_every_1k.csv](records/loss_every_1k.csv) | 由 `Step N:` 行解析；每 1000 步一行（step 0、1000、…、79000，共 80 行）。`loss,grad_norm,llm_grad_norm,mem_enc_norm,param_norm` 是该 step 那条日志的原值（本身是 100 步区间均值）；`n_bin` 与 `*_bin_mean` 是 [step, step+1000) 内 10 条日志（step, step+100, …, step+900）的算术平均。step 79900 的末条不在 79000 行原值里，但计入 79000 行的区间均值 | 81 行（含表头） |
| [records/upload-512-smoke.summary.log](records/upload-512-smoke.summary.log) | 正式驱动 `SMOKE=1`，tmux `m8v-ckpt512-upload-smoke`，日志 `v1-store/logs/m8v-ckpt512-upload-smoke.log` | 23 行 |
| [records/upload-512.summary.log](records/upload-512.summary.log) | 正式驱动全量，tmux `m8v-ckpt512-upload`，日志 `v1-store/logs/m8v-ckpt512-upload.log`；阶段 0–10 全部判定行 | 117 行 |
| [records/inc-512.summary.log](records/inc-512.summary.log) | 训练中增量上传，tmux `m8v-inc-512`，日志 `v1-store/logs/m8v-inc-512.log`（原始 297,106 行，绝大部分是 hf 进度条）；`INC_START`、16 组 `INC_STEP_BEGIN` / `HYGIENE` / `Sync plan` / `INC_UPLOADED`、`INC_END` 与 `EXIT_CODE=0` | 206 行 |
| [records/SHA256SUMS.pre.txt](records/SHA256SUMS.pre.txt) | 驱动阶段 2 生成、已公开在 bucket 根下的上传前校验清单，从 `v1-store/exports/hf-ckpt-mixed8-200-vla-512-80k/stage/` 原样复制（`cmp` 一致），供异地按路径核对 sha256 | 341 行，40,118 B，sha256 `c4a2f32e…6eee2` |

不归档：checkpoint 权重（留在 `v1-store/train-runs/…` 与公开 bucket）、wandb 本地目录（口径 44）、驱动 stage 与 verify 副本、`m8v-ckpt-inc.sh`（全文已作为代码块贴在 launch.md）、各 step 的 `hygiene-inc-<step>.json` 体检报告（判定行已在 `inc-512.summary.log`）。
