# hf-export-m2048-80k-ckpt-v1 · launch

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 6.9 T 本地 NVMe RAID）**

- 任务：把训练 run `v2-1600ep-m32x8x8-modul-b128-80k`（modulation + **motion 关闭**、
  32 帧 × 8×8 = `budget 2048`）的全部 16 个 orbax checkpoint（329 文件 / 190,052,281,699 字节）
  上传到 HuggingFace **private** bucket `hf://buckets/HongzeFu/robomme-vla-modul-2048-80k-v1`，
  上传前 / 上传后两遍独立 sha256 逐行对照验收。
- 动机：与前三次 ckpt 导出同理——这批权重此前只存在于本机 NVMe，无任何异地副本；
  环境 B 无 turbo / GreatLakes 可用。本条是 2048 token 预算那组的正式 80k 产物。
- 起跑 commit：`54a48d3`（commitV11.3，新增本条导出链路；见下方「起跑前状态」）。
- 入口：`bash scripts/dataset/hf_export/run_m2048_80k_ckpt_export.sh`
- tmux 会话：`exp2048-ckpt`（本轮起过的会话仅此一个）。
- 日志：`v1-store/exports/hf-ckpt-v2-1600ep-m32x8x8-modul-b128-80k/logs/export.log`

## 源与落点

| 项 | 值 |
|---|---|
| 源 | `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m32x8x8-modul-b128-80k/`（16 个 step：5000…75000 + 79999） |
| 目标 | `hf://buckets/HongzeFu/robomme-vla-modul-2048-80k-v1`（private，起跑前 `files=0 size=0`，本轮新建） |
| stage / verify | `v1-store/exports/hf-ckpt-v2-1600ep-m32x8x8-modul-b128-80k/{stage,verify}` |
| 凭据 | `v1-store/secrets/hf.env`（`HF_TOKEN`，600，在 `.gitignore` 的 `/v1-store/` 下） |

## 被导出的这条 run

训练已于 `2026-09-22T06:19:30Z` 正常结束（`EXIT_CODE=0`，起跑 `2026-09-21T07:20:38Z`，
耗时 22 小时 58 分 52 秒），故本次导出**直接走趟 B**（补传 + 全量验收），不经趟 A。

| 项 | 值 |
|---|---|
| history config | `perceptual-framesamp-modul-32frame-8x8.yaml`（raw sha256 `42813c7e…`） |
| resolved history config sha256 | `91512306b3aaaaa541d248bc5dfaf8be2849632cd3b9e89a226207c09c6501c0` |
| budget / token_per_image / num_views | 2048 / 64 / 1 |
| motion | **关闭**（`motion_provenance.json` 的 `motion_enabled=false`、`encoder=null`） |
| 训练起跑 commit | `55647ff33c8ddb9ec324fdbcee8bd1491456725b`（commitV11.2Beta） |
| wandb run id | `u5qogzm9` |

各 step 的文件数不等（OCDBT 的 `d/` 分块数随内容变）：
21 / 21 / 19 / 20 / 19 / 20 / 20 / 19 / 21 / 23 / 20 / 21 / 20 / 20 / 21 / 19，合计 324，
加 run 根 5 个 sidecar（`history_config.resolved.sha256`、`history_config.resolved.yaml`、
`history_config.txt`、`motion_provenance.json`、`wandb_id.txt`）共 **329**。

## 与蓝本 run_motion80k_ckpt_export.sh 的差异

driver 的六阶段骨架与趟 A / 趟 B 机制照抄蓝本，两处实质差异：

1. **没有 `_motion-encoder/` 这一段。** 本 run motion 关闭，训练全程不涉及 MotionJEPA encoder，
   蓝本里「按 `motion_provenance.json` 的 `encoder.checkpoint_sha256` 校验 encoder 本体并上传」
   的整块逻辑在本脚本中不存在，阶段 6 的剔除清单相应少一项。阶段 0 新增 `motion_enabled` 断言，
   读出 `True` 即停，防止 `RUN_NAME` 指错 run。
2. **训练日志文件名是 `<run>.driver.log`**（本 run 由 `scripts/training/prod/run_modul2048.sh`
   起跑，日志带 `.driver` 中缀），蓝本那条是 `<run>.log`。

另新增两处防护：阶段 0 断言 `$REPO/v1-store` 不是 symlink（`-temp` 开发副本整条 `v1-store`
是指向主副本的 symlink，本脚本只允许在主副本执行）；训练存活判定改用括号技巧
`pgrep -f "[t]rain\.py …"` 破坏自匹配。

README 的「训练结果」一节仍由 `motion80k_result_block.py` 从训练日志现读现生成——该脚本只依赖
日志格式与 `<!-- TODO-PASS-B-RESULT -->` 哨兵、与 motion 开关无关，直接复用，不另立一份。

## 起跑前状态（已核实）

- `hf auth whoami` → 含 `HongzeFu`（driver 阶段 0 打印 `HF_WHOAMI=HongzeFu`）。
  环境里原有的 `HF_TOKEN` 属别的账号，写不进 `HongzeFu`，driver 内由 `hf.env` 覆盖。
- 目标 bucket 本轮由 driver 以 `hf buckets create --private` 新建，
  阶段 0 复查为 `files=0 size=0`，非空即停的断言已通过。
- 磁盘：`/scratch` 余量 1.4 T。stage 走硬链接不额外占盘，阶段 4 的回读副本约 178 GiB，余量够。
- 主副本工作区当时有**其他 agent 的在途改动**（本条 run 的训练归档文档）。按 `AGENTS.md`
  第 11 条只逐个 `git add` 了本轮自己的两个新文件，未触碰任何在途改动；
  故起跑时 `git status --porcelain` 非空，driver 生成的 `_run-meta/git-commit.txt` 会如实记录。

## 验收判定行

日志里必须同时出现，缺一条即不算通过：

```
HF_WHOAMI=HongzeFu
未检测到训练进程 => 趟 B（补传 + 全量验收）
PRE_PARTS_CONSISTENT=OK                  # 分片合并出的 pre 清单 == 整树重算
LOCAL_FILES=<n> LOCAL_BYTES=<n> PRE_LINES=<n-1>
BUCKET_FILES=<n> BUCKET_BYTES=<n>        # 与 LOCAL_* 逐字相等
SHA256_PRE_POST_DIFF=0                   # 上传前 / 上传后两份独立清单 diff 为空 ← 核心判据
RESULT=PASS
SRC_UNCHANGED=OK                         # 源侧第三遍 sha256 与上传前一致（硬链接未改动原件）
全部完成
EXIT_CODE=0
```

预期 `LOCAL_FILES` = 329（源）+ 12（`_run-meta/`：`train.log` + wandb 9 件 + `git-commit.txt`
+ `train-cmd.txt`）+ 1（`README.md`）+ 1（`SHA256SUMS.pre.txt`）= **343**，`PRE_LINES` = 342。
实测值以 `result.md` 为准。
