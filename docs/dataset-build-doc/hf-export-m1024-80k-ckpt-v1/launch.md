# hf-export-m1024-80k-ckpt-v1 · launch

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 6.9 T 本地 NVMe RAID）**

- 任务：把训练 run `v2-1600ep-m16x8x8-modul-b128-80k`（modulation + **motion 关闭**、
  16 帧 × 8×8 = `budget 1024`）的全部 16 个 orbax checkpoint（327 文件 / 190,050,687,155 字节）
  上传到 HuggingFace **private** bucket `hf://buckets/HongzeFu/robomme-vla-modul-1024-80k-v1`，
  上传前 / 上传后两遍独立 sha256 逐行对照验收。
- 用户原话：「1024的跑完了吗也上传」。
- 起跑 commit：新增本条导出链路的 commitV11.8Beta（在 `-temp` 开发副本提交并推送，完整 sha 见
  导出日志与 bucket 内 `_run-meta/git-commit.txt`）。
- 执行位置：`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`。主副本本地有其他会话的未推送提交
  `6c94989`（训练归档），与 origin 分叉，本轮不动主副本；脚本把 `v1-store` 先 `readlink -f` 成
  主副本实体路径再用，写入落点与在主副本执行完全相同。
- 入口：`EXPORT_THROTTLE=0 bash scripts/dataset/hf_export/run_m1024_80k_ckpt_export.sh`
  （1024 训练已结束、同机无训练在跑，不限流）
- tmux 会话：`hfx-m1024-<UTC>`（本轮起过的会话仅此一个，完整名写入 result.md）。
- 日志：`v1-store/exports/hf-ckpt-v2-1600ep-m16x8x8-modul-b128-80k/logs/<会话名>.log`

## 源与落点

| 项 | 值 |
|---|---|
| 源 | `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m16x8x8-modul-b128-80k/`（16 个 step：5000…75000 + 79999） |
| 目标 | `hf://buckets/HongzeFu/robomme-vla-modul-1024-80k-v1`（private，本轮新建；起跑前不存在，阶段 0 断言为空） |
| stage / verify | `v1-store/exports/hf-ckpt-v2-1600ep-m16x8x8-modul-b128-80k/{stage,verify}` |
| 凭据 | `v1-store/secrets/hf.env` |
| 磁盘 | 起跑前 `/scratch` 空闲 402 G；stage 走硬链接不占盘，verify 约 177 GiB |

## 被导出的这条 run

训练于 `2026-09-24T12:44:27Z` 起跑、`2026-09-25T10:45:41Z` 正常结束（`EXIT_CODE=0`），
完成器判定 `RUN_COMPLETED=PASS run=v2-1600ep-m16x8x8-modul-b128-80k budget=1024 steps=80000 checkpoints=16 final=79999`。

| 项 | 值 |
|---|---|
| history config | `perceptual-framesamp-modul-16frame-8x8.yaml`（raw sha256 `ec1308fe…`） |
| resolved history config sha256 | `9d7a18fe2cdb720dce21dc7d1c03b7a905fc6f65d98758a582268e5fa9e5390b` |
| budget / token_per_image / num_views | 1024 / 64 / 1 |
| motion | **关闭**（`motion_enabled=false`） |
| 训练起跑 commit | `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb`（commitV11.6Beta） |
| wandb run id | `0ayufnrr` |
| norm_stats | 16 份 sha256 全部为 `856c75ea…`，与 4096、2048 同值 |

## 与蓝本 run_m4096_80k_ckpt_export.sh 的差异

逻辑逐行相同，非注释差异只有 5 处：`RUN_NAME`、`BUDGET=1024`、`BUCKET_ID`、train-cmd 的
`--model.history-config`（16frame）、README 模板路径（`m1024_80k_bucket_README.md`）。
提交前验证：`bash -n` 通过；阶段 0 完成断言用 1024 真实记录作正例 rc=0，拿 4096 的完成记录作负例 rc=1；
`motion80k_result_block.py` 用真实 driver.log 生成 README 无残留哨兵（耗时 22 小时 1 分 14 秒）。

## 验收判据

- 阶段 3：bucket REST 统计的 `totalFiles` / `size` 与本地 stage 一致。
- 阶段 5：`SHA256_PRE_POST_DIFF=0`、`sha256sum -c --strict` 全 OK、`RESULT=PASS`。
- 阶段 6：`SRC_UNCHANGED=OK`。
- 导出日志末尾 `EXIT_CODE=0`。
