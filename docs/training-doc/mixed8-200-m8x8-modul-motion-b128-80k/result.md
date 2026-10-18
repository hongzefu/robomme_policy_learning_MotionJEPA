# mixed8-200 8×8 modulation 512+motion 80k 正式训练——结果

本页是 AGENTS 第 12 条两段式档案的第二段，承载十二节中的①⑦⑧⑨⑩⑪⑫；②版本与代码状态、③启动与配置还原、④数据集与划分口径、⑤关键超参、⑥硬件与耗时预估，以及训练中增量上传的实施中调整，都在同目录 [launch.md](launch.md)（commitV11.25Beta `ae6e9cdee494e0cfa815c5cde04d255e0cb14cfd`）。同批 4+4 并行的 512（无 motion）结果见 [mixed8-200-m8x8-modul-b128-80k/result.md](../mixed8-200-m8x8-modul-b128-80k/result.md)。时间一律美东（括号内给 UTC），日志里的原始时间戳是 UTC，引用时原样保留。

## ① 一句话结论与指标速览

**一句话结论**：512 modulation + motion 的 80k 正式训练在 GPU 4–7 上一次跑完、`EXIT_CODE=0`，全程无报错、无非有限数、无显存溢出；16 份 checkpoint 训练中逐份增量上传、训练后由正式驱动补齐并全量回读校验，已公开在 `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1`；训练 loss 从 0.0869 降到末 1k 步均值 0.00142，同步数下始终比 512 低约四分之一（这是训练 loss，不代表评估成功率）。

| 指标 | 实测 |
|---|---|
| 起跑 | 美东 2026-10-17 01:01:44（`START_UTC=2026-10-17T05:01:44Z`），`GIT_HEAD=ae6e9cdee494e0cfa815c5cde04d255e0cb14cfd`（V11.25Beta） |
| 结束 | 美东 2026-10-18 15:39:03（`END_UTC=2026-10-18T19:39:03Z`），日志末行 `EXIT_CODE=0` |
| 总时长 | 38 小时 37 分 19 秒（runner 起止）；训练进度条自身计时 `elapsed:38:35:26` |
| 稳态步时 | 5000→75000 共 70000 步平均 **1.7346 s/步**（含 14 次保存）；每 5000 步区间 8,668.7–8,678.9 秒，即 1.7337–1.7358 s/步；进度心跳 13,332 行中 13,211 行（99.1%）为 `rate:1.7s/it` |
| 与短测预估对比 | 20 步短测估约 1.90 s/步、约 42 小时，实测快约 9%（launch.md 已注明短测偏保守） |
| loss 节点（`Step N:` 行，每 100 步均值） | 0：0.0869；1000：0.0148；10000：0.0046；20000：0.0032；40000：0.0022；60000：0.0017；79900：0.0015；末 1k 步（79000…79900 共 10 行）均值 0.00142 |
| 梯度范数 | `grad_norm` 1.2855 → 0.0217（79900）；`param_norm` 1815.74 → 1859.49 |
| 异常 | 训练日志 `Traceback` / `nan` / `out of memory` / `Killed` 零命中 |
| checkpoint | 5000…75000 每 5000 一份加 79999，共 16 份，合计 329 个文件、190,164,311,721 字节（每份约 11.88–11.89 GB），根目录 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-b128-80k` |
| bucket | `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1`（公开）：`BUCKET_FILES=337 BUCKET_BYTES=190166411781` |
| SHA256SUMS.pre.txt | 336 行、39,469 字节、sha256 `1855c5ef9735f2b50a514e0a96a6a6466b670caf0c5926adfeb756775ede4357` |
| wandb | 项目 robomme-framesamp，run `gxanzsmk`（本地目录不上传，口径 44） |
| 只读锁 | 两条训练都写出 `EXIT_CODE=0` 后，主会话美东约 16:00 按 `v1-store/m8v-scripts/m8v-chmod-before.txt`（385 行）逐文件恢复原权限，恢复后与记录逐行比对为空：`PERM_RESTORED=OK` |

正式驱动 `scripts/dataset/hf_export/run_mixed8_200_vla_motion_80k_ckpt_export.sh` 的全部判定行原文（先 `SMOKE=1`，tmux `m8v-ckptmotion-upload-smoke`，美东 15:42 起、15:43 结束；再全量，tmux `m8v-ckptmotion-upload`，美东 15:44 起、16:02 结束；驱动代码未改）：

```text
# SMOKE=1（records/upload-motion-smoke.summary.log）
PROVENANCE_MATCH=PASS motion=True fs8_store_meta=ace85246419836d3 resolved=53f070ff5363efff encoder=bec423fdfa0c9494
README_RENDERED steps=16 ckpt_bytes=190164311721 git_head=ae6e9cdee494
LOCAL_FILES=337 LOCAL_BYTES=190166411781 PRE_LINES=336 SRC_FILES=334
HYGIENE=PASS scanned=88 hits=217 ungated=0
SMOKE=PASS（到阶段 3 为止，未碰网络与凭据）
全部完成
EXIT_CODE=0

# 全量（records/upload-motion.summary.log）
PROVENANCE_MATCH=PASS motion=True fs8_store_meta=ace85246419836d3 resolved=53f070ff5363efff encoder=bec423fdfa0c9494
README_RENDERED steps=16 ckpt_bytes=190164311721 git_head=ae6e9cdee494
LOCAL_FILES=337 LOCAL_BYTES=190166411781 PRE_LINES=336 SRC_FILES=334
HYGIENE=PASS scanned=88 hits=217 ungated=0
HF_WHOAMI=HongzeFu
BUCKET_CREATED=HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1 private=False
BUCKET_FILES_AT_START=332 BUCKET_BYTES_AT_START=190164316322
UPLOAD_DONE batches=17 root_files=7
BUCKET_FILES=337 BUCKET_BYTES=190166411781
DF_CHECK avail=3959619260416 readback_need=190166411781 verify_have=0 margin=400000000000
SHA256_PRE_POST_DIFF=0
RESULT=PASS
SRC_UNCHANGED=OK checked=334
BUCKET_PUBLIC_FINAL private=False
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
全部完成
EXIT_CODE=0
```

`PROVENANCE_MATCH` 一行同时钉住了 framesamp-8x8 的 store_meta（`ace85246…`）、resolved 配置快照（`53f070ff…`）与 motion 表所用 encoder（`encoder=bec423fdfa0c9494`，即 launch.md 记的 `checkpoint_epoch_72.pt` EMA-only，sha256 `bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04`）。

## ⑦ 训练过程行为

### 起止与步时

runner 打印 `START_UTC=2026-10-17T05:01:44Z`，训练脚本初始化、加载 `pi05_base` 权重与 motion 表（日志「motion memory 开启」两行，rows 128342）后，进度条于 05:03:23（UTC，美东 01:03:23）出现，首个心跳 `8.00it/80.0kit rate:5.0s/it` 是编译期。约 100 步后进入稳态，`rate:1.7s/it` 一直保持到结束。心跳中偶见 3.5–6.3 s/it 的瞬时值，全部落在每 5000 步保存的那一刻（进度条瞬时速率被保存阻塞拉高），不是持续减速；每 5000 步区间的实耗（下表「区间」列）几乎恒定，可见保存只占每区间约 10–19 秒。

motion 比 512 稳态略慢：同口径（5000→75000 平均）512 为 1.7183 s/步、motion 为 1.7346 s/步，慢约 0.95%；两条起跑相差约 1 分钟，motion 比 512 晚约 23 分钟结束（512 美东 15:15:45、motion 15:39:03）。差值来自 motion 分支多出的 motion token 计算与读表，量级很小。

本档案未收 GPU 利用率采样，不对利用率下结论（AGENTS 第 16 条）。

### loss 走势（训练 loss）

`Step N:` 行是训练脚本每 100 步把这 100 步的指标取均值后打印的一行（第 0 步只含自身），全程 800 行，与原始日志逐行一致（见⑫）。每 1000 步一行的数值在 [records/loss_every_1k.csv](records/loss_every_1k.csv)（81 行：0、1000…79000 与最后一行 79900）。

| step | loss | grad_norm | llm_grad_norm | mem_enc_norm | param_norm |
|---|---|---|---|---|---|
| 0 | 0.0869 | 1.2855 | 1.2726 | 0.1082 | 1815.7416 |
| 100 | 0.0570 | 0.5072 | 0.5007 | 0.0412 | 1815.7412 |
| 1000 | 0.0148 | 0.1295 | 0.1264 | 0.0109 | 1815.7537 |
| 5000 | 0.0069 | 0.0532 | 0.0466 | 0.0137 | 1817.1903 |
| 10000 | 0.0046 | 0.0368 | 0.0306 | 0.0108 | 1820.7826 |
| 20000 | 0.0032 | 0.0289 | 0.0236 | 0.0079 | 1827.2040 |
| 30000 | 0.0027 | 0.0271 | 0.0225 | 0.0065 | 1833.1312 |
| 40000 | 0.0022 | 0.0231 | 0.0193 | 0.0054 | 1838.7587 |
| 50000 | 0.0020 | 0.0214 | 0.0178 | 0.0046 | 1844.2310 |
| 60000 | 0.0017 | 0.0238 | 0.0204 | 0.0050 | 1849.4606 |
| 70000 | 0.0016 | 0.0232 | 0.0202 | 0.0042 | 1854.5642 |
| 75000 | 0.0015 | 0.0210 | 0.0178 | 0.0038 | 1857.0610 |
| 79000 | 0.0014 | 0.0220 | 0.0188 | 0.0038 | 1859.0488 |
| 79900 | 0.0015 | 0.0217 | 0.0186 | 0.0038 | 1859.4856 |

区间均值（每 100 步一行取平均）：40000…49900 为 0.00209；60000…69900 为 0.00164；70000…79900 为 0.00147；79000…79900 为 0.00142。后 20k 步仍在缓慢下降（每 10k 步约降 10%），没有到平台；学习率按 launch.md 是 5e-5 常数（peak 与 decay 相同），下降来自持续拟合而非学习率退火。全程只有训练集，没有验证集，loss 下降不能说明泛化，过拟合与否只能由闭环评估回答（见⑧）。

### 与 512（无 motion）的同步数对照

数据来自两份训练日志的 `Step N:` 行（512 为 `v1-store/logs/mixed8-200-m8x8-modul-b128-80k.log`），两条同库、同 config、同 seed、同 batch 128、同 4 卡，差别只在 history yaml 是否接入 motion。下表是**训练 loss**，两条的输入信息量不同，motion 更低只说明它在训练集上拟合得更好，**不代表评估成功率**更高。

| step | 512 loss | 512+motion loss | motion / 512 |
|---|---|---|---|
| 0 | 0.0874 | 0.0869 | 0.99 |
| 10000 | 0.0062 | 0.0046 | 0.74 |
| 20000 | 0.0043 | 0.0032 | 0.74 |
| 30000 | 0.0036 | 0.0027 | 0.75 |
| 40000 | 0.0030 | 0.0022 | 0.73 |
| 50000 | 0.0026 | 0.0020 | 0.77 |
| 60000 | 0.0023 | 0.0017 | 0.74 |
| 70000 | 0.0021 | 0.0016 | 0.76 |
| 79900 | 0.0019 | 0.0015 | 0.79 |
| 末 1k 步均值（79000…79900） | 0.00192 | 0.00142 | 0.74 |

第 0 步两者几乎相同（同一初始化），从第一个 10k 起 motion 稳定低约 25%，比值在 0.73–0.79 之间小幅波动、没有随训练收窄或拉大的趋势。最后一行的 `grad_norm`：512 为 0.0238、motion 为 0.0217。

### checkpoint 落盘与增量上传

落盘时刻取训练日志 `Finished asynchronous save` 行（UTC 原样，美东换算在旁），文件数与字节数取增量上传日志 `INC_UPLOADED` 行，上传时刻为美东。

| step | 落盘完成（UTC） | 落盘完成（美东） | 保存耗时 | 区间（距上一份） | files | bytes | `INC_UPLOADED`（美东） |
|---|---|---|---|---|---|---|---|
| 5000 | 10-17 07:30:34 | 10-17 03:30:34 | 19.04 s | — | 20 | 11,886,310,995 | 10-17 19:08 |
| 10000 | 10-17 09:55:13 | 10-17 05:55:13 | 12.31 s | 8,678.9 s | 22 | 11,886,011,575 | 10-17 19:09 |
| 15000 | 10-17 12:19:46 | 10-17 08:19:46 | 9.33 s | 8,674.1 s | 20 | 11,885,899,560 | 10-17 19:10 |
| 20000 | 10-17 14:44:20 | 10-17 10:44:20 | 11.81 s | 8,671.2 s | 23 | 11,885,835,971 | 10-17 19:10 |
| 25000 | 10-17 17:08:50 | 10-17 13:08:50 | 10.59 s | 8,670.9 s | 20 | 11,885,686,389 | 10-17 19:11 |
| 30000 | 10-17 19:33:26 | 10-17 15:33:26 | 11.10 s | 8,675.3 s | 21 | 11,885,610,505 | 10-17 19:12 |
| 35000 | 10-17 21:57:57 | 10-17 17:57:57 | 11.33 s | 8,671.4 s | 21 | 11,885,478,798 | 10-17 19:12 |
| 40000 | 10-18 00:22:27 | 10-17 20:22:27 | 10.53 s | 8,670.5 s | 20 | 11,885,380,854 | 10-17 20:23 |
| 45000 | 10-18 02:46:56 | 10-17 22:46:56 | 10.71 s | 8,668.7 s | 21 | 11,885,236,872 | 10-17 22:49 |
| 50000 | 10-18 05:11:33 | 10-18 01:11:33 | 14.07 s | 8,678.2 s | 19 | 11,885,091,065 | 10-18 01:17 |
| 55000 | 10-18 07:36:04 | 10-18 03:36:04 | 10.99 s | 8,669.3 s | 20 | 11,884,973,049 | 10-18 03:38 |
| 60000 | 10-18 10:00:39 | 10-18 06:00:39 | 13.61 s | 8,672.8 s | 21 | 11,884,801,637 | 10-18 06:04 |
| 65000 | 10-18 12:25:11 | 10-18 08:25:11 | 10.19 s | 8,675.2 s | 21 | 11,884,723,137 | 10-18 08:29 |
| 70000 | 10-18 14:49:42 | 10-18 10:49:42 | 10.04 s | 8,672.1 s | 20 | 11,884,500,635 | 10-18 10:50 |
| 75000 | 10-18 17:14:19 | 10-18 13:14:19 | 13.85 s | 8,675.9 s | 19 | 11,884,454,816 | 10-18 13:16 |
| 79999 | 10-18 19:38:54 | 10-18 15:38:54 | 11.69 s | 4,999 步 | 21 | 11,884,315,863 | 10-18 15:42 |

「区间」列取 orbax `Preserving Checkpoint` 行记录的各 step 保存起点之差。16 份合计 329 个文件、190,164,311,721 字节，与正式驱动 `README_RENDERED … ckpt_bytes=190164311721` 一致。5000…35000 在增量上传会话美东 2026-10-17 19:08 起跑时已落盘，19:08–19:12 一次补齐；之后每份在落盘后约 1–6 分钟内传完（扫描间隔 300 秒）。每份上传前都 `HYGIENE=PASS scanned=5 hits=0 ungated=0`（16 次）。增量上传会话 `m8v-inc-motion` 收尾行原文：

```text
INC_UPLOADED step=79999 files=21 bytes=11884315863 15:42 美东
INC_END run=mixed8-200-m8x8-modul-motion-b128-80k done=16 训练会话已退出；剩余补传与全量验收交正式驱动
EXIT_CODE=0
```

### 正式驱动的计数核对

bucket 文件数前后可以逐项对上，说明增量上传、根文件提前上传与正式驱动三路写入的内容没有重叠错位：

- `BUCKET_FILES_AT_START=332` = 增量上传的 329 个 ckpt 文件 + 美东 10-18 01:06 提前上传的 3 个根文件（`history_config.resolved.yaml` 811 B、`history_config.resolved.sha256` 65 B、`motion_provenance.json` 3725 B，见⑩）；字节 190,164,311,721 + 811 + 65 + 3725 = 190,164,316,322，与 `BUCKET_BYTES_AT_START=190164316322` 逐字节相等。
- 正式驱动阶段 5 跳过 16 个已 `.done` 的 step，只传 `_run-meta/train.log`（2,050,777 字节）与 7 个根下平文件（`README.md`、`SHA256SUMS.pre.txt`、`history_config.resolved.sha256`、`history_config.resolved.yaml`、`history_config.txt`、`motion_provenance.json`、`wandb_id.txt`；其中 3 个与提前上传的同名同内容）：`UPLOAD_DONE batches=17 root_files=7`。
- 终态 `BUCKET_FILES=337` = 329 + 1 + 7，等于本地 `LOCAL_FILES=337`；`PRE_LINES=336` 是除 `SHA256SUMS.pre.txt` 自身外的全部文件；`SRC_FILES=334` = 329 个 ckpt 文件 + run 根下 5 个平文件，与阶段 9 `SRC_UNCHANGED=OK checked=334` 对应。
- 阶段 6 首次计数为 `files=336 bytes=190166411773`，驱动按设计等 120 秒复查后一致（服务端统计异步滞后，AGENTS 第 15 条不把服务端统计当验收证据）；最终验收以阶段 7 全量回读 337 个文件、阶段 8 `SHA256_PRE_POST_DIFF=0` 与 `sha256sum -c` 336 行为准。

## ⑧ 训练后评估

训练侧本轮不做评估（launch.md「用户决定与结论边界」，口径 15：本轮不做 VLA 评测、不做纯 encoder 对照）。本 run 的 10000 / 60000 / 79999 三个 checkpoint 由评估侧会话 sled-aspen（「Great Lakes 评估与 F1 计划修订」）从公开 bucket 取用并评估，结果在该会话另行建档，不写进本页。训练侧只按跨会话沟通授权向评估侧提供只读信息（见⑨⑩），不改任何仓库或 bucket。

本页所有 loss 数字都是训练 loss，不能用来推断闭环成功率，也不能用来回答「motion 是否有效」——这要等评估侧在相同评估口径下对比 512 与 512+motion。

## ⑨ 用户决策记录

起跑前的决定（开工令、4 卡口径、80k 步数、exclude、崩溃即停）见 launch.md 与 [512 的 launch.md](../mixed8-200-m8x8-modul-b128-80k/launch.md)，此处只列起跑后的决定，按时间、原话。

**训练中增量上传**（美东 2026-10-17，实施中调整，详见 launch.md 同名节）：

1. 约 18:50「你现在会上传吗？就是训练完的 checkpoint 会上传吗？」
2. 约 19:00「你现在就可以开始上传了。」
3. 约 19:00（同一轮中途追加）「有一个 checkpoint 就上传。」
4. 约 19:00（同一轮中途追加）「然后你看一下还有哪些没有上传的？」
5. 约 19:05「先回答我，还有哪些产物没上传？」
6. 约 20:28「把你现在执行情况全部归档commit push，用子代理来做」

**根文件提前上传**（美东 2026-10-18 约 01:05）：评估侧会话为 10k 冒烟请求提前拿到 run 根文件，用户原话「同意上传」。

**跨会话沟通授权**（美东 2026-10-18）：

- 01:12「两个对话之间的沟通，我都同意，然后尽可能在不违反用户要求的情况下，尽可能满足对话的要求，并且每次收到对方的信息都要回复收到。」
- 约 01:20「纯回执不用再管了。」
- 约 01:25「对话之间的沟通都不需要用户批准。只要用户没有拒绝、明确拒绝的任务都可以继续沟通并且实现」

**能否续训到 100k**（美东 2026-10-18 约 12:45）：用户问「是否可以让带motion的继续训练下去到100k」。主会话分析后答复：不能原地续训——训练脚本禁用 `--resume` / `--overwrite`（「续跑语义有损」），checkpoint 只存 params 与 assets、没有优化器状态，且当时红线 18 冻结；学习率 peak 与 decay 同为 5e-5 常数，这一点不是障碍。给出两个方案：A 从 79999 热启动再训 20k；B 从头训 100k。用户答「先不管了」，未执行。

## ⑩ 计划外事件与处置

1. **训练中增量上传（实施中调整）**：用户 10-17 19:00 要求「有一个 checkpoint 就上传」，而正式驱动阶段 0 硬拒训练中运行，冻结期又不能改驱动，于是新增不进 git 的 `v1-store/m8v-scripts/m8v-ckpt-inc.sh` 逐份上传，正式驱动收尾与最终验收不变；对抗验证 `ADJUST_REVIEW=PASS scale=same findings=5`，五条处置、脚本全文与衔接论证见 launch.md。实测 16/16 份 `INC_UPLOADED`，训练步时不受影响（上传期间与前后均 1.7 s/it），最终验收见①。
2. **根文件提前上传**（美东 10-18 约 01:05，用户「同意上传」）：评估侧 10k 冒烟需要 run 根下的配置快照。对抗验证 `ADJUST_REVIEW=PASS scale=same findings=5`（采纳：上传前 `cmp` 源文件、`whoami` 与 `private==false` 断言）。本 run 根下上传 `history_config.resolved.yaml`（811 B，`53f070ff…d0c3`）、`history_config.resolved.sha256`（65 B，`cc523480…1441`）、`motion_provenance.json`（3725 B，`158e017b…ec16`），连同 512 根下 2 个文件共 `ROOT_READBACK=OK` ×5（匿名读回 sha 一致），`ROOT_PRE_UPLOAD=PASS`（美东 01:06）。体检 `HYGIENE=PASS`：本 run 的 `motion_provenance.json` 有 2 处本机路径（P9），按账本 accept。副作用：bucket 根下这 3 个文件公开时还没有 README 说明，README 由正式驱动阶段 5 补上（披露晚于文件公开，已记录、无需处置）。同晚 01:20 512 根下补传 `motion_provenance.json`（`ADJUST_REVIEW=PASS scale=same findings=2`），属 512 那条，详见其 result.md。
3. **向评估侧提供只读信息**（跨会话沟通授权下，不改仓库与 bucket）：本 run 10000 逐文件清单（22 行，清单 sha256 `8d9b04df…a1e4`）、60000（21 行，`71173939…9e74`）、79999（21 行，`74266818…b00d`）；encoder 远端位置（公开 bucket `HongzeFu/motionjepa-wan-mixed8-200-72ep-v1` 根下 `checkpoint_epoch_72.pt` 477,433,329 B、sha256 `bec423fd…9e04`，`config.yaml` 2816 B、`bdd8bb04…de51`，`SHA256SUMS.pre.txt` 252 行）；`scripts/assets/ASSETS_LOCK.json` 口径：该锁在 `82e5175` / `62a8ef1` / `ae6e9cd` / HEAD 均仍指旧 full1600 encoder（`motionjepa_ckpt` sha `0c198629…`，顶层 `0a654077…`），本轮建库是用 `--encoder-run-dir` 指向新 encoder 并以 `--expected-ckpt-sha256 bec423fd…` 显式传参满足检查，训练不走旁路配置文件；为评估分支改锁算出的新条目首尾片段为 ckpt `d7139efb5090e4aa778adc707ac5b064`、config `a9357aaf3ecc0bd9a42e71ca45a45135`。
4. **正式驱动阶段 6 首次计数不符**：`首次不符（files=336 bytes=190166411773）`，驱动按设计等 120 秒复查后 `BUCKET_FILES=337 BUCKET_BYTES=190166411781`，是服务端统计异步滞后，不是缺文件；回读与 sha256 对照全部通过。
5. **续训 100k 的询问**：见⑨，用户「先不管了」，未执行，列入待定事项。
6. **MotionJEPA-backup 退役**：已在 MotionJEPA `v7.0-mixed8`（`f25f9d3`）与本仓库 `de86240` 登记，此处只引用；交接文件里的计划路径已改为主检出 `/scratch/hongze/MotionJEPA/1016-mixed8-vla-plan.html`。

训练本身没有计划外事件：无崩溃、无非有限数、无显存溢出、无掉卡，未触发口径 32「80k训练崩了停下来问我」。

## ⑪ 结论与下一步

**结论**：512+motion 80k 训练按计划完成，16 份 checkpoint 公开且经全量回读逐文件 sha256 校验（`RESULT=PASS`、`SHA256_PRE_POST_DIFF=0`、`SRC_UNCHANGED=OK checked=334`、`BUCKET_PUBLIC=True`、`PUBLIC_ANON_READ=OK`）；源 checkpoint 未被改动；红线 18 只读锁已解除（`PERM_RESTORED=OK`）。训练 loss 同步数下比 512 稳定低约 25%，但这只是训练集拟合，motion 是否带来闭环收益要看评估。交接任务 `mixed8-vla-train` 的交接前清单与长尾四项均已完成，状态写为 `closed`。

**下一步**：

- 评估：由 sled-aspen 评估会话对 10000 / 60000 / 79999 做评估并另行建档（训练侧不做）。
- 待用户定：本 run 与 512 的回读副本 `v1-store/exports/hf-ckpt-mixed8-200-vla-{512,motion}-80k/verify`（各约 190 GB）是否删除；100k 是否另起（方案 A 从 79999 热启动 20k，或方案 B 从头 100k，均需新计划与新 run 名）。

## ⑫ 归档文件清单

`records/` 只放 git 还原不出来的实测日志与指标，不放 checkpoint、配置拷贝或脚本（AGENTS 第 12 条）。

| 文件 | 来源与清洗口径 | 核验 |
|---|---|---|
| [records/train.summary.log](records/train.summary.log) | `v1-store/logs/mixed8-200-m8x8-modul-motion-b128-80k.log` 按 AGENTS 第 12 条 (3) 命令 `tr '\r' '\n' \| grep -vE '%\|'` 清洗，另去行尾空白（只为通过 `git diff --check`，不删行）。本日志的进度条走日志行而非回车刷新，清洗前后都是 16,602 行 | `^Step ` 行 800 = 原日志 800；含 `START_UTC=`、`GIT_HEAD=`、`END_UTC=`、`EXIT_CODE=0` |
| [records/train.steps.log](records/train.steps.log) | 上一文件中全部 `^Step ` 行（每 100 步一行） | 800 行，与 `grep '^Step '` 原日志逐行一致 |
| [records/loss_every_1k.csv](records/loss_every_1k.csv) | 从 `Step N:` 行解析，取 step 为 1000 倍数的行加最后一行 79900；列为 step、loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm；每行数值是该步前 100 步的均值（第 0 步只含自身），原样照抄日志小数位 | 81 行数据 |
| [records/inc-motion.summary.log](records/inc-motion.summary.log) | `v1-store/logs/m8v-inc-motion.log`（41.8 万行，绝大部分是 hf 上传进度条）：回车拆行、去终端颜色码、去 hf 进度条行与空行 | 206 行；`INC_`、`HYGIENE`、`BUCKET_`、`EXIT_CODE=` 等判定行 53 行与原日志相等 |
| [records/upload-motion-smoke.summary.log](records/upload-motion-smoke.summary.log) | `v1-store/logs/m8v-ckptmotion-upload-smoke.log`，同上清洗 | 判定行 16 行与原日志相等 |
| [records/upload-motion.summary.log](records/upload-motion.summary.log) | `v1-store/logs/m8v-ckptmotion-upload.log`，同上清洗 | 117 行；判定行 41 行与原日志相等 |

不归档：checkpoint 权重（留在 `v1-store/train-runs/…` 与公开 bucket）、wandb 本地目录（口径 44）、`m8v-ckpt-inc.sh`（全文已以代码块贴入 launch.md）、导出目录下的 `SHA256SUMS.pre.txt`（已随 bucket 公开，本页记行数、字节与 sha256）。
