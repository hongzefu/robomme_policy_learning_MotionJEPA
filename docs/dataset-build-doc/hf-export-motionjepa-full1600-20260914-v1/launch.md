# hf-export-motionjepa-full1600-20260914-v1 — 起跑

把 MotionJEPA 在本机 8×A100 上建出的 full1600 产物发布到**公开** HF bucket
`hf://buckets/HongzeFu/robomme-4task-motion-full1600-20260914-v1`。

## 口径（用户 2026-09-14 拍板）

| 项 | 决定 |
|---|---|
| 范围 | 只传最终产物 + 溯源：`dataset-token/` 的 `PUBLISHED.json` 清单 2804 件 + 清单自身 + `control/` 溯源 |
| 不传 | `data-raw/`(121 G)、`reference/`+`reference-fp32/`(163 G)、`smoke28/`(21 G)、`logs/`，以及 `dataset-token/` 下 5618 个清单外边车 |
| 可见性 | 公开 |
| 命名 | 沿用现有风格 |
| `metadata.json` 里的 AWS 内部主机名 | 在被明确告知含私有 IP 与 region 后，拍板**原样传 + README 披露**（改字节会让 `PUBLISHED.json` 的 sha256 校验链整体失配） |

## 起跑信息

- **起跑 commit**：`04d9ac421cb7793737acd08128c83b680a28b077`（日志首行记录）
- **起跑时刻**：2026-09-14T17:41:58+00:00
- **tmux 会话**：`hfup-full1600`（本轮只起这一个 + 批 2 的 `hfup-h5v2`）
- **工作区状态**：起跑时工作区存在**他人的在途改动**（`scripts/dataset/finalize_checks.py` 被修改、
  `scripts/dataset/merge_v2_h5.py` 与 `scripts/dataset/test_merge_v2_h5.py` 未跟踪），属另一条
  h5 合并任务，按 AGENTS 第 11 条未予触碰。这些文件不在本轮导出链路上，不影响本轮语义。
- **源**：`/scratch/hongze/dataset-local-a100-4task-full1600/`
  （`dataset-token/` 2804 件 479,479,189,461 B；`control/` 2985 件）
- **落点**：`v1-store/exports/hf-motionjepa-full1600/{stage,verify,tmp,logs}`

## 命令原文

```bash
# 起跑（detached tmux）
tmux new-session -d -s hfup-full1600 \
  "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/exports/hf-motionjepa-full1600/logs/launch.sh"

# launch.sh 内容
set -o pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
export PYTHONUNBUFFERED=1
export PATH="/home/ec2-user/.local/bin:$PATH"   # tmux server 的 PATH 不含 ~/.local/bin
LOG=v1-store/exports/hf-motionjepa-full1600/logs/run.log
echo "=== 起跑 $(date -Is) commit=$(git rev-parse HEAD) ===" | tee -a "$LOG"
bash scripts/dataset/hf_export/run_motionjepa_full1600_export.sh 2>&1 | tee -a "$LOG"
echo "EXIT_CODE=$?" | tee -a "$LOG"

# 起跑前的 SMOKE（阶段 0–3，不碰网络写）
SMOKE=1 bash scripts/dataset/hf_export/run_motionjepa_full1600_export.sh
```

## 十一个阶段

| 阶段 | 做什么 | 判定行 |
|---|---|---|
| 0 | 凭据 + whoami | `HF_WHOAMI=HongzeFu` |
| 1 | 硬链接组装 stage + 边车/符号链接断言 | `STAGE_DONE` / `STAGE_SIDECARS=0` |
| 2 | 全量 sha256 + 对 `PUBLISHED.json` 自带清单逐条核对 | `PUBLISHED_SHA_MATCH=2804 mismatches=0` |
| 3 | 公开前体检（**在 create 之前**） | `HYGIENE=PASS … ungated=0` |
| 4 | 建公开 bucket + 非空断言 | `BUCKET_CREATED=… PRIVATE=False` |
| 5 | `hf sync --dry-run` 预演，断言无 delete | `SYNC_PLAN_DELETES=0` |
| 6 | 按字节预算装箱后分批上传 | `PLAN_BATCHES` / `UPLOAD_DONE` |
| 7 | 计数层（REST，异步滞后复查一次） | `BUCKET_FILES` / `BUCKET_BYTES` |
| 8 | 全量回读 479.5 GB | sync 摘要 |
| 9 | post 重算 + 与 pre 逐行 diff + tar 成员抽样 | `SHA256_PRE_POST_DIFF=0` / `MEMBERS_SPOT=PASS` |
| 10 | 源侧第三遍 sha256 | `SRC_UNCHANGED=OK` |
| 11 | 公开性 + 清 token 匿名读 | `BUCKET_PUBLIC=True` / `PUBLIC_ANON_READ=OK` |

## 两处与既有 400ep 链路的实质差别

1. **多一层「清单层」**：`PUBLISHED.json` 带着建库侧当时写下的 2804 个 sha256，是一份**第三方
   账本**。stage 组装后重算与它逐条对，能抓「建库之后源文件被动过」——这是回读校验抓不到的。
   实测 `PUBLISHED_SHA_MATCH=2804 expected=2804 missing=0 mismatches=0`。
2. **分批按字节预算而非个数**：`.bin` 是重尾分布（最小 33.7 MiB、中位 132 MiB、最大 612 MiB），
   固定「每批 25 个」最坏会到 15 GB，越过 2026-09-08 实测的稳过上限（10 GB）。改为
   ≤8 GiB / ≤256 件贪心装箱（`plan_upload_batches.py`），得 58 批、最大批 8,588,427,264 B；
   计划落盘防止重跑时分组漂移导致 `--include` 失配。
