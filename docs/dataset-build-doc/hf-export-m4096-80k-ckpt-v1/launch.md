# hf-export-m4096-80k-ckpt-v1 · launch

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 6.9 T 本地 NVMe RAID）**

- 任务：把训练 run `v2-1600ep-m64x8x8-modul-b128-80k`（modulation + **motion 关闭**、
  64 帧 × 8×8 = `budget 4096`）的全部 16 个 orbax checkpoint（316 文件 / 190,052,719,306 字节）
  上传到 HuggingFace **private** bucket `hf://buckets/HongzeFu/robomme-vla-modul-4096-80k-v1`，
  上传前 / 上传后两遍独立 sha256 逐行对照验收。
- 用户原话：「给出方案把4096训练完的上传huggingface」；方案确认时选定「现在限流上传」，
  即与同机在跑的 1024 训练并行、全程限流，不等 1024 结束。
- 动机：这批权重此前只存在于本机 NVMe，无任何异地副本；环境 B 无 turbo / GreatLakes 可用。
- 起跑 commit：新增本条导出链路的 commitV11.7Beta（在 `-temp` 开发副本提交并推送，完整 sha 见
  导出日志与 bucket 内 `_run-meta/git-commit.txt`）。
- 执行位置：`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`（1024 训练期间主副本锁定，
  HEAD 保持 `0571fea`）。脚本把 `v1-store` 先 `readlink -f` 成主副本实体路径再用，
  写入落点与在主副本执行完全相同；脚本不含 `--force` / rmtree，也不需要 `.venv`。
- 入口：`EXPORT_THROTTLE=1 bash scripts/dataset/hf_export/run_m4096_80k_ckpt_export.sh`
- tmux 会话：`hfx-m4096-<UTC>`（本轮起过的会话仅此一个，完整名写入 result.md）。
- 日志：`v1-store/exports/hf-ckpt-v2-1600ep-m64x8x8-modul-b128-80k/logs/<会话名>.log`

## 源与落点

| 项 | 值 |
|---|---|
| 源 | `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m64x8x8-modul-b128-80k/`（16 个 step：5000…75000 + 79999） |
| 目标 | `hf://buckets/HongzeFu/robomme-vla-modul-4096-80k-v1`（private，本轮新建；阶段 0 断言起跑前为空） |
| stage / verify | `v1-store/exports/hf-ckpt-v2-1600ep-m64x8x8-modul-b128-80k/{stage,verify}` |
| 凭据 | `v1-store/secrets/hf.env`（`HF_TOKEN`，在 `.gitignore` 的 `/v1-store/` 下） |
| 磁盘 | 起跑前 `/scratch` 空闲 745 G；stage 走硬链接不占盘，verify 约 177 GiB |

## 被导出的这条 run

训练于 `2026-09-23T07:15:33Z` 起跑、`2026-09-24T09:51:29Z` 正常结束（`EXIT_CODE=0`），
完成器判定 `RUN_COMPLETED=PASS run=v2-1600ep-m64x8x8-modul-b128-80k budget=4096 steps=80000 checkpoints=16 final=79999`
（`v1-store/bench/modul-budget-sweep/modul-sweep-20260923-c/v2-1600ep-m64x8x8-modul-b128-80k.completed.json`，
run_uuid `93521f4e-f25a-4f89-ada8-98cc7ce18437`）。

| 项 | 值 |
|---|---|
| history config | `perceptual-framesamp-modul-64frame-8x8.yaml`（raw sha256 `e5274831…`） |
| resolved history config sha256 | `9df78b4dac2acdc43fbce5040a88014805df1b50bb803feba027559fc31f96a3` |
| budget / token_per_image / num_views | 4096 / 64 / 1 |
| motion | **关闭**（`motion_enabled=false`、`encoder=null`） |
| 训练起跑 commit | `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb`（commitV11.6Beta） |
| wandb run id | `iuw13maz` |
| norm_stats | 16 份 sha256 全部为 `856c75ea…`，与 2048 那条同值 |

各 step 文件数：19 / 19 / 20 / 20 / 20 / 19 / 19 / 21 / 19 / 19 / 19 / 19 / 18 / 19 / 20 / 21，合计 311，
加 run 根 5 个 sidecar 共 **316**。

## 与蓝本 run_m2048_80k_ckpt_export.sh 的差异

六阶段骨架照抄 2048 那条，四处实质差异：

1. **run / bucket 名、train-cmd 的 history YAML（64frame）、README 模板**（`m4096_80k_bucket_README.md`）换成 4096。
2. **删掉趟 A 与 `pgrep` 判定，只走趟 B。** 蓝本用 `pgrep -f "[t]rain\.py mme_vla_suite_b128_80k"` 判断
   「本 run 还在训练」；1024 run 用同一个 config 名且正在训练，照抄会误判走趟 A——丢掉 `79999`、不回读却打印
   `PASS=A`。改为按本 run 自身硬断言：driver.log 尾部 `EXIT_CODE=0`，完成 JSON `status=PASS`、run_name / budget
   对得上、checkpoints 恰 16 个且末步 79999。任一不满足阶段 0 即 `exit 1`（提交前以真实记录作正例、
   6 个伪造记录作负例验证：正例 rc=0，负例全部 rc=1）。
3. **限流由 `EXPORT_THROTTLE` 决定（默认 1）**：`ionice -c 3 nice -n 19`、sha256 并发 `NPROC=2`，
   包住上传、回读下载与全部 sha256；`0` 为不限流、`NPROC=8`。
4. **`_run-meta/` 多带完成证据**：`completed.json` 与 `train-record/`（保存现场记录 `final/`、
   `run_meta.json`、`runtime.json`、`metrics.jsonl`，合计 <0.5 MB），逐文件硬链接进 stage。

## 与 1024 训练共存的约定

1024 run `v2-1600ep-m16x8x8-modul-b128-80k` 同机八卡训练中（起跑前进度 7.65k/80k，步时约 1.0 s/it）。
导出不碰 GPU、不碰 1024 的数据与输出目录。上传期间单独挂 Monitor 盯 1024 driver.log 的 `rate:`；
若步时持续比上传前慢 20% 以上，按确切会话名 `kill-session` 停掉导出，等 1024 结束后以
`EXPORT_THROTTLE=0` 原样重跑（stage、分片清单、已传批次均幂等续传）。漂移数据写入 result.md。

## 验收判据

- 阶段 3：bucket REST 统计的 `totalFiles` / `size` 与本地 stage 一致（早停信号）。
- 阶段 5：`SHA256_PRE_POST_DIFF=0`、`sha256sum -c --strict` 全 OK、`RESULT=PASS`（最终判据）。
- 阶段 6：`SRC_UNCHANGED=OK`（源侧第三遍 sha256 与 pre 清单去掉 README / `_run-meta` 后逐行相同）。
- 导出日志末尾 `EXIT_CODE=0`。
