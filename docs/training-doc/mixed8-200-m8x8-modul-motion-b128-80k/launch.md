# mixed8-200 8×8 modulation 512+motion 80k 正式训练

本页是 commitV11.25Beta 起跑前档案（AGENTS 第 12 条两段式第一段）；跑完回写同号 commitV11.25 与 result.md。起跑 HEAD 即本 Beta 提交，由 preflight 的 `REPO_HEAD` 与 runner 打印的 `GIT_HEAD=` 固定。与 512 同批起跑（4+4 并行），512 档案见 [mixed8-200-m8x8-modul-b128-80k/launch.md](../mixed8-200-m8x8-modul-b128-80k/launch.md)（commitV11.24Beta `62a8ef15b4be267b56612d50562fc445160d041f`，美东 2026-10-17 01:00 起跑），版本、代码冻结、exclude、数据身份、监控与用户决定各节与之相同，本页只写差异。

## 与 512 的差异

- run 名 `mixed8-200-m8x8-modul-motion-b128-80k`；GPU 4–7（`CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8`），全局 batch 128、每卡 32。9 月 8 卡 motion run 每卡 16，本轮每卡样本翻倍、吞吐约减半。
- history yaml `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml`，sha256 `213bbe809baa286e951e873a49f6fc153a0517f939ac263cefcfc45c02380794`：复制自 `…-8x8-motion.yaml`，只改 `motion.store_path: v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion` 与 `motion.source_run: wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt#encoder`（budget 160、stride 16、window 33、demo 至少 17 帧 repeat_last）。
- motion 表：`v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion`，rows 128342（exec 81611 + demo 46731），补帧窗 1800；store_meta `c9f8275fe12d6395ea7cc29191e39a1fe5df7b569c579b5acdfea07a71bdde2e`，layout `motion-768-grid16-demopad17-v1`。encoder `checkpoint_epoch_72.pt`（EMA-only）sha256 `bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04`。窗口预算 `TRAIN_ES_BOUND=PASS max_windows=123 budget=160 headroom=37`。
- 对拍：`VERIFY_MOTION=PASS scanned=128342`、`WAN_BITEXACT=PASS compared=14480`、`ENCODER_BITEXACT=PASS compared=128342`、A6/A7/A9/A10/A11 全 PASS（[判定行汇总](../../dataset-build-doc/16task-mixed8-200-3200ep-e3c7564f/records/verdicts-b1-b4.md)）。
- 20 步短测：`MOTION_4GPU_SMOKE=PASS run=motion steps=20 exit=0 finite_losses=20/20 peak_mem_gib=38.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8`；稳态约 1.90 s/步，80k 预计约 42 h（含 4+4 互扰）。
- checkpoint 根 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-b128-80k`；日志 `v1-store/logs/mixed8-200-m8x8-modul-motion-b128-80k.log`；训练后上传 `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1`（公开，不含 wandb/）。

```bash
bash v1-store/m8v-scripts/m8v-preflight-motion.sh      # 不传 --launch-mode；四个 --motion-* 一起给
tmux new-session -d -s m8v-train-motion "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/mixed8_200_runner/train-motion-80k.sh mixed8-200-m8x8-modul-motion-b128-80k"
```

## 两条训练之间工作区复查（口径 40 模板）

2026-10-17 01:01 美东，git status --porcelain：空（512 已于美东 01:00 起跑后复查）。exclude 仍只含口径 40 授权的 3 个路径与 1 行注释，未扩大。本 Beta 只新增本档案文件；起 motion 前另断言 `git diff --stat 62a8ef15b4be267b56612d50562fc445160d041f..HEAD -- src pyproject.toml uv.lock` 为空（红线 18）。

## 实施中调整：训练中增量上传 checkpoint（美东 2026-10-17）

本节是起跑后追加的实施中调整记录（AGENTS 第 2 条「执行中调整计划」要求不停工也必须告知并留档；第 12 条两段式档案），前文起跑档案一字未改。前文「用户决定与结论边界」一节写的是训练结束后再上传全部 16 份 ckpt，那是旧口径；现改为训练进行中逐份增量上传已提交的 step，训练结束后的正式驱动收尾与最终验收不变，二者冲突处以本节为准。另一条训练的同名小节见[512 档案](../mixed8-200-m8x8-modul-b128-80k/launch.md)。

### 用户原话（美东 2026-10-17，按时间）

1. 约 18:50「你现在会上传吗？就是训练完的 checkpoint 会上传吗？」
2. 约 19:00「你现在就可以开始上传了。」
3. 约 19:00（同一轮中途追加）「有一个 checkpoint 就上传。」
4. 约 19:00（同一轮中途追加）「然后你看一下还有哪些没有上传的？」
5. 约 19:05「先回答我，还有哪些产物没上传？」
6. 约 20:28「把你现在执行情况全部归档commit push，用子代理来做」

### 旧口径 → 新口径与理由

按计划口径 29/30、A1b 与 runbook 长尾段，原定每条训练出现 `EXIT_CODE=0` 之后才起正式驱动 `scripts/dataset/hf_export/run_mixed8_200_vla_motion_80k_ckpt_export.sh`（共用体 `scripts/dataset/hf_export/_mixed8_200_vla_ckpt_export_common.sh`），先 `SMOKE=1` 再全量，一趟跑完。用户 19:00 要求「有一个 checkpoint 就上传」，而正式驱动在阶段 0 有三道硬拒：训练进程还在跑、训练日志末行不是 `EXIT_CODE=0`、step 目录不恰为 16 个——训练中任何一条都满足不了，驱动不可能提前跑。同时红线 18 冻结期内 `src`/`scripts`/`packages` 已 `chmod a-w`（原权限记在 `v1-store/m8v-scripts/m8v-chmod-before.txt`），不能改驱动去放宽这三道检查。因此新增一个不进 git 的脚本 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`（`v1-store/` 整体 gitignored，全文见本节末尾），只做「训练中把已由 orbax 提交的 step 目录逐个传上去」这一件事，其余全部仍交给正式驱动。

| 项 | 旧口径 | 新口径 |
|---|---|---|
| 何时开始上传 | 训练 `EXIT_CODE=0` 之后 | 训练进行中，每出现一个已提交 step（`_CHECKPOINT_METADATA` 带 `commit_timestamp_nsecs`）就传 |
| 谁传 step 目录 | 正式驱动阶段 5 | inc 脚本先传；驱动阶段 5 跳过已 `.done` 的 step、补传其余 |
| 谁传 `_run-meta/`、根下平文件（README.md、SHA256SUMS.pre.txt 等） | 正式驱动 | 不变，仍是正式驱动 |
| 公开前体检 | 驱动对 stage 跑 `scan_hygiene.py` | inc 对每个 step 目录跑同一个 `scan_hygiene.py`、同一份 allowlist（`--scope stage`），体检仍在上传之前；驱动收尾照旧体检 |
| 建 bucket | 驱动阶段 4 | inc 在首个 step 体检通过之后建，`hf buckets settings --public` 并断言 `private==false`，首传前写 `uploaded.marker` |
| 重试 | 首试 + 3 次（口径 27） | 不变：首试 + 3 次，间隔 60 s，耗尽打 `UPLOAD_FAIL` 退出 |
| 训练中限流 | 无（训练已结束） | `hf` 调用前加 `ionice -c 3 nice -n 19`（F2） |
| 最终验收 | 驱动阶段 6–10 | 不变（见下） |

### 与正式驱动的衔接

inc 与正式驱动用同一个 bucket（`HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1`）、同一个 bucket 内路径 `$BUCKET/<step>`。驱动阶段 5 同步的是 `$STAGE/<step>`，而 stage 是源 run 目录的硬链接，内容与 inc 直接从源目录 `hf sync` 上去的完全相同。inc 每传完一个 step 就写驱动阶段 5 自己使用的完成标记 `v1-store/exports/hf-ckpt-mixed8-200-vla-motion-80k/logs/uploaded/<step>.done`，首传前写 `logs/uploaded.marker`（驱动阶段 4 据此认定 bucket 里的内容属于本轮上传，不会因 bucket 非空而拒绝）。驱动阶段 1 `rm -rf STAGE` 不删 `logs/uploaded`，`hf sync` 只读源，阶段 6/7/8 的计数、回读与校验不依赖由谁上传——验收强度不变。

操作顺序（F4）：本 run 训练 `EXIT_CODE=0`，且 inc 日志出现 `INC_END`、`tmux has-session -t '=m8v-inc-motion'` 为假之后，才照原计划起正式驱动：先 `SMOKE=1`（tmux `m8v-ckptmotion-upload-smoke`），再全量（tmux `m8v-ckptmotion-upload`）。驱动跳过已 `.done` 的 step，补传剩余 step、`_run-meta/` 与根下平文件，并做阶段 6–10 的计数、全量回读、上传前后 sha256 逐行对照与公开性确认。最终验收判定行不变：`RESULT=PASS`、`SHA256_PRE_POST_DIFF=0`、`SRC_UNCHANGED=OK`、`BUCKET_PUBLIC=True`、`PUBLIC_ANON_READ=OK`、`全部完成`、`EXIT_CODE=0`。inc 日志里的任何行都不算最终验收证据。

### 对抗验证（AGENTS 第 2 条执行中调整）

调整前派了一个只读 sonnet 审查子代理，结论 `ADJUST_REVIEW=PASS scale=same findings=5`（无 blocking）。要点：① 调整必要（驱动三道硬拒）；② 衔接成立（stage 是源硬链接、sync 只读源、阶段 1 `rm -rf STAGE` 不删 `logs/uploaded`、阶段 4 marker 语义、阶段 6/7/8 不依赖谁上传，验收强度不变）；③ 最小改法、无多余保护层；④ 未触硬停清单。五条 finding 及处置：

| 编号 | 级别 | 内容 | 处置 |
|---|---|---|---|
| F1 | 低 | inc 原末尾打印「全部完成」，会与正式驱动的完成信号混淆 | 已采纳：删去，只留 `INC_END` 行 |
| F2 | 中 | 训练中上传未限流 | 已采纳：`hf()` 前加 `ionice -c 3 nice -n 19`（照搬 `run_motion80k_ckpt_export.sh` 的 THROTTLE 先例） |
| F3 | 低 | provenance 交叉锚点要到驱动阶段 0 才核 | 已采纳：首传前主会话只读核对，两行判定见下 |
| F4 | 操作约束 | 正式驱动启动前必须先见 inc 日志 `INC_END` 且 `tmux has-session -t '=m8v-inc-motion'` 为假 | 已采纳，写入上节操作顺序与 handoff `next_steps` |
| F5 | 信息 | 训练若崩溃，公开 bucket 会停在部分 step、没有 README/校验文件，驱动永远拒绝运行 | 记为风险：届时写进 `pending_decisions` 交用户裁决（保留或清理），并核 `INC_END done=<n>` |

F3 首传前的 provenance 核对（resolved yaml sha == `.sha256` == `provenance.resolved_sha256`；framesamp-8x8 store_meta sha；motion 另核 motion store_meta 与 encoder ckpt sha），两条训练各一行：

```text
PROVENANCE_MATCH=PASS run=mixed8-200-m8x8-modul-b128-80k motion=False []
PROVENANCE_MATCH=PASS run=mixed8-200-m8x8-modul-motion-b128-80k motion=True []
```

本 run 对应的是 `PROVENANCE_MATCH=PASS run=mixed8-200-m8x8-modul-motion-b128-80k motion=True []`（resolved yaml sha == .sha256 == provenance.resolved_sha256，framesamp-8x8 store_meta sha 一致，另核 motion store_meta 与 encoder ckpt sha）。起跑前的只读体检试跑做在 512 的 step 5000 上（`HYGIENE=PASS scanned=5 hits=0 ungated=0`），本 run 未单独试跑，正式上传时每个 step 各自体检（见下文实测）。

### 本 run 的 inc 句柄与起跑命令

| 项 | 值 |
|---|---|
| tmux 会话 | `m8v-inc-motion`（本轮新起；被盯的训练会话 `m8v-train-motion`） |
| 日志 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8v-inc-motion.log` |
| 五个参数 | `mixed8-200-m8x8-modul-motion-b128-80k` `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1` `hf-ckpt-mixed8-200-vla-motion-80k` `scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_motion_80k.json` `m8v-train-motion` |
| 完成标记目录 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/exports/hf-ckpt-mixed8-200-vla-motion-80k/logs/uploaded/` |
| 起跑 | 美东 2026-10-17 19:08:05（日志 `INC_START`） |
| 完成行 | `INC_END run=mixed8-200-m8x8-modul-motion-b128-80k done=<n> …`（训练会话退出后的最后一轮扫描之后打出）+ `EXIT_CODE=0`；inc 不打「全部完成」 |

扫描逻辑：每轮先记下训练 tmux 是否存活，再按 5000…75000、79999 顺序检查未 `.done` 且已提交的 step；本轮开始时训练已退出则扫完即打 `INC_END` 退出，否则睡 300 s 再扫。完整起跑命令：

```bash
tmux new-session -d -s m8v-inc-motion "cd /scratch/hongze/robomme_policy_learning_MotionJEPA; set -o pipefail; bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m8x8-modul-motion-b128-80k HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1 hf-ckpt-mixed8-200-vla-motion-80k scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_motion_80k.json m8v-train-motion 2>&1 | tee /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8v-inc-motion.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8v-inc-motion.log"
```

### 本轮 tmux 会话清单（AGENTS 第 7、12 条）

前文起跑档案未单列会话清单，这里补齐本 run 相关的全部会话名；清理只以此清单为依据，一次一个 `tmux kill-session -t '=确切会话名'`。

| 会话名 | 用途 | 状态（美东 2026-10-17 20:29） |
|---|---|---|
| `m8v-train-motion` | 本 run 80k 训练 | 运行中 |
| `m8v-inc-motion` | 训练中增量上传 ckpt | 运行中，已传 8/16 |
| `m8v-ckptmotion-upload-smoke` | 正式驱动 `SMOKE=1` | 未起（等 F4 条件） |
| `m8v-ckptmotion-upload` | 正式驱动全量 | 未起（SMOKE 通过后） |

### 盯盘项（inc 日志）

inc 日志单独挂一个 Monitor（一份日志一个），关注行：`INC_UPLOADED`（每传完一份）、`INC_END`（训练退出后的收尾）、`UPLOAD_FAIL`（重试耗尽）、`HYGIENE=FAIL`（体检不过、不会上传）、`错误`（脚本自检失败）、`Traceback`、`EXIT_CODE=`。inc 退出码非 0 或出现 `UPLOAD_FAIL` / `HYGIENE=FAIL` 时停下交证据，不自行改脚本重跑。

```bash
tail -n +1 -F /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8v-inc-motion.log | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "INC_UPLOADED|INC_END|UPLOAD_FAIL|HYGIENE=FAIL|错误|Traceback|EXIT_CODE="
```

### 实测（截至美东 2026-10-17 20:29，日志原文判定行）

`HF_WHOAMI=HongzeFu`；19:08 首个 step 体检通过后 `BUCKET_CREATED=HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1 private=False`。每个 step 都 `HYGIENE=PASS scanned=5 hits=0 ungated=0`。已传 8 份（5000…40000），本地 `.done` 标记 8 个，合计 167 个文件、95,086,214,647 字节：

| step | `INC_STEP_BEGIN`（美东） | `INC_UPLOADED`（美东） | files | bytes |
|---|---|---|---|---|
| 5000 | 19:08 | 19:08 | 20 | 11,886,310,995 |
| 10000 | 19:08 | 19:09 | 22 | 11,886,011,575 |
| 15000 | 19:09 | 19:10 | 20 | 11,885,899,560 |
| 20000 | 19:10 | 19:10 | 23 | 11,885,835,971 |
| 25000 | 19:10 | 19:11 | 20 | 11,885,686,389 |
| 30000 | 19:11 | 19:12 | 21 | 11,885,610,505 |
| 35000 | 19:12 | 19:12 | 21 | 11,885,478,798 |
| 40000 | 20:22 | 20:23 | 20 | 11,885,380,854 |

速度：5000 那份日志 `New Data Upload 9.26GB / 9.26GB, 333MB/s`（xet 去重后 9.26 GB 新数据），单份约 1 分钟。bucket 服务端统计：19:13 时 files=147 / 83.2 GB（与本地 7 个 step 文件数之和 147 一致）；20:29 时 files=167、bytes=95086214647（8 个 step）。服务端统计是异步的，只作参考，不作验收证据（AGENTS 第 15 条）；验收以正式驱动的全量回读与 sha256 对照为准。

训练未受影响：上传期间与前后训练日志 rate 均为 1.7 s/it；20:29 时本 run 进度 40.2k/80k（remaining 19:05:39），预计结束约美东 2026-10-18 15:35，到 60k 约 10-18 05:49。（前文起跑档案按 20 步短测估的步时与总时长偏保守，以此处实测为准。）

### 风险与下一步

每 5000 步自动传一份（45000 约美东 2026-10-17 22:30）。F5：训练若崩溃、NaN、OOM、掉卡，按口径 32 停下问用户、不 resume；此时公开 bucket 停在部分 step、没有 README 与校验文件，正式驱动会拒绝运行，需把 bucket 部分状态与 `INC_END done=<n>` 写进 handoff `pending_decisions` 交用户裁决保留还是清理。正常路径：训练 `EXIT_CODE=0` 且 inc `INC_END`、inc 会话已退出后，起 `m8v-ckptmotion-upload-smoke` → `m8v-ckptmotion-upload`，核全部最终判定行，回写 result.md 与正式 commit，全部结束后按 `m8v-chmod-before.txt` 恢复权限，handoff 任务 `mixed8-vla-train` 走 done/closed。

### m8v-ckpt-inc.sh 全文

脚本位于 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`，`v1-store/` 不进 git，git 无法从别处还原，故原样贴入（AGENTS 第 12 条只禁止在 docs 下放 .sh/.yaml 文件，代码块允许）。两条训练共用同一份脚本。

```bash
#!/usr/bin/env bash
# m8v-ckpt-inc：训练中增量上传已由 orbax 提交的 step 目录（用户 2026-10-17「你现在就可以开始上传了」「有一个 checkpoint 就上传」）。
# 不改仓库代码（红线 18 冻结期）。与正式驱动 _mixed8_200_vla_ckpt_export_common.sh 衔接：
#   同一 bucket、同一 bucket 内路径（$BUCKET/<step>，与驱动阶段 5 的 sync "$STAGE/<step>" 相同内容：stage 是源的硬链接）；
#   每传完一个 step 写驱动阶段 5 的 $EXPORT_ROOT/logs/uploaded/<step>.done，首传前写 uploaded.marker（驱动阶段 4 据此认本轮上传）。
# 训练 EXIT_CODE=0 后照计划跑正式驱动（SMOKE=1 → 全量）：已 .done 的 step 跳过，其余 step、_run-meta/、根下平文件补传，
# 阶段 6–10 照旧做计数、全量回读、上传前后 sha256 逐行对照与公开性确认——最终验收不变。
# 每个 step 上传前对该 step 目录跑同一个 scan_hygiene.py 与同一份 allowlist（公开前体检仍在上传之前）；建 bucket 在首个 step 体检之后。
# 用法：bash m8v-ckpt-inc.sh <RUN_NAME> <BUCKET_ID> <EXPORT_NAME> <ALLOWLIST 相对仓库路径> <训练 tmux 名>
set -euo pipefail
RUN_NAME="$1"; BUCKET_ID="$2"; EXPORT_NAME="$3"; ALLOWLIST="$4"; TRAIN_TMUX="$5"
REPO=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$REPO"
SRC="$REPO/v1-store/train-runs/mme_vla_suite_b128_80k/$RUN_NAME"
EXPORT_ROOT="$REPO/v1-store/exports/$EXPORT_NAME"
BUCKET="hf://buckets/$BUCKET_ID"
EXPECT_STEPS="$(seq 5000 5000 75000 | tr '\n' ' ')79999"
UPLOAD_RETRIES=3; MAX_ATTEMPTS=$((UPLOAD_RETRIES + 1))
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE || true
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }
hf() { ionice -c 3 nice -n 19 uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }   # 训练中限流，照搬 run_motion80k_ckpt_export.sh 的 THROTTLE
bucket_stat() {  # 与驱动同一写法
  curl -sf -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/buckets/HongzeFu" | python3 -c "
import sys, json
for b in json.load(sys.stdin):
    if b['id'] == '$BUCKET_ID':
        print(b.get('$1')); break
else:
    sys.exit('目标 bucket $BUCKET_ID 不在返回列表里')
"
}
assert_public() {
  local p; p="$(bucket_stat private)"; echo "$1 private=$p"
  case "$p" in False|false) : ;; *) echo "错误: bucket 不是公开的（private=$p）"; exit 1 ;; esac
}
[ -d "$SRC" ] || { echo "错误: 源 run 目录不存在 $SRC"; exit 1; }
[ -L "$EXPORT_ROOT" ] && { echo "错误: $EXPORT_ROOT 是符号链接"; exit 1; }
mkdir -p "$EXPORT_ROOT/logs/uploaded"
echo "INC_START run=$RUN_NAME bucket=$BUCKET_ID train_tmux=$TRAIN_TMUX $(TZ=America/New_York date '+%F %T') 美东"
WHO="$(hf auth whoami --format agent 2>&1 | tr '\n' ' ' | tr -s ' ')"
case "$WHO" in *HongzeFu*) echo "HF_WHOAMI=HongzeFu" ;; *) echo "错误: 身份不是 HongzeFu（$WHO）"; exit 1 ;; esac
BUCKET_READY=0
ensure_bucket() {  # 与驱动阶段 4 同一逻辑；只在首个 step 体检通过后调用
  [ "$BUCKET_READY" = 1 ] && return 0
  if bucket_stat totalFiles >/dev/null 2>&1; then
    local n; n="$(bucket_stat totalFiles)"
    if [ "$n" != "0" ] && [ "$n" != "None" ] && [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
      echo "错误: 目标 bucket 非空（files=$n）且本地无本轮上传标记，停止交用户裁决"; exit 1
    fi
    echo "  bucket 已存在，跳过创建"
  else
    hf buckets create "$BUCKET_ID"
  fi
  hf buckets settings "$BUCKET_ID" --public
  assert_public "BUCKET_CREATED=$BUCKET_ID"
  touch "$EXPORT_ROOT/logs/uploaded.marker"
  BUCKET_READY=1
}
while true; do
  alive=0; tmux has-session -t "=$TRAIN_TMUX" 2>/dev/null && alive=1
  for s in $EXPECT_STEPS; do
    DONE="$EXPORT_ROOT/logs/uploaded/$s.done"
    [ -f "$DONE" ] && continue
    [ -f "$SRC/$s/_CHECKPOINT_METADATA" ] || continue
    python3 -c "
import json, sys
m = json.load(open('$SRC/$s/_CHECKPOINT_METADATA'))
sys.exit(0 if m.get('commit_timestamp_nsecs') else 1)
" || continue
    nf=$(find "$SRC/$s" -type f | wc -l); nb=$(find "$SRC/$s" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
    echo "INC_STEP_BEGIN step=$s files=$nf bytes=$nb $(TZ=America/New_York date '+%H:%M') 美东"
    uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
      --root "$SRC/$s" --scope stage --allowlist "$ALLOWLIST" \
      --report "$EXPORT_ROOT/logs/hygiene-inc-$s.json"
    ensure_bucket
    ok=0
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
      if hf sync "$SRC/$s" "$BUCKET/$s"; then ok=1; break; fi
      echo "  step $s 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
      if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then sleep 60; fi
    done
    [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$s/ attempts=$MAX_ATTEMPTS"; exit 1; }
    touch "$DONE"
    echo "INC_UPLOADED step=$s files=$nf bytes=$nb $(TZ=America/New_York date '+%H:%M') 美东"
  done
  if [ "$alive" = 0 ]; then
    echo "INC_END run=$RUN_NAME done=$(ls "$EXPORT_ROOT/logs/uploaded" | grep -c '\.done$') 训练会话已退出；剩余补传与全量验收交正式驱动"
    exit 0   # 不打「全部完成」：那是正式驱动的完成信号
  fi
  sleep 300
done
```
