# hf-export-m4096-80k-ckpt-v1 · result

**结论：通过。** 训练 run `v2-1600ep-m64x8x8-modul-b128-80k` 的全部 16 个 orbax checkpoint
已完整上传到 HuggingFace private bucket `HongzeFu/robomme-vla-modul-4096-80k-v1`。
上传前与上传后两遍独立 sha256 逐行对照，**diff 为 0**；源文件没有被硬链接连带改动。
上传与同机 1024 训练并行，全程限流，1024 的步时没有受到可见影响。

- 导出 commit：`1a08afb`（commitV11.7Beta），从 `-temp` 开发副本执行
- tmux 会话：`hfx-m4096-20260924T145634Z`（本轮起过的唯一会话；脚本退出后按确切名称 `kill-session`，
  删除前后两次 `tmux ls` 的差集恰好只有它）
- 起止：`2026-09-24 14:56:34` → `15:36:53`（UTC），**约 40 分钟**
- 限流：`THROTTLE_MODE=idle NPROC=2`（`ionice -c 3 nice -n 19`）
- 日志：`v1-store/exports/hf-ckpt-v2-1600ep-m64x8x8-modul-b128-80k/logs/hfx-m4096-20260924T145634Z.log`
  （清洗版见 `records/run.clean.log.txt`）

## 验收判定行（逐条实测）

```
THROTTLE_MODE=idle NPROC=2
TRAIN_COMPLETED=OK
HF_WHOAMI=HongzeFu
  目标 bucket 现状: files=0 size=0
LOCAL_FILES=337 LOCAL_BYTES=190081207647 PRE_LINES=336
PRE_PARTS_CONSISTENT=OK
UPLOAD_DONE batches=16
BUCKET_FILES=337 BUCKET_BYTES=190081207647
SHA256_PRE_POST_DIFF=0
  sha256sum -c 校验行数: 336
RESULT=PASS
  pre(剔 README/_run-meta)=316 行, src-after=316 行, _run-meta=19 行
SRC_UNCHANGED=OK
全部完成
EXIT_CODE=0
```

`LOCAL_FILES=337` 的组成：源 316 个文件、`_run-meta/` 19 个、`README.md` 1 个，
以及 `SHA256SUMS.pre.txt` 本身 1 个。`_run-meta/` 比 2048 那次多 7 个，
是新增的 `completed.json` 与 `train-record/` 下 6 个文件。
bucket 的 `totalFiles` / `size` 第一次核对就与本地相等，没有触发 120 s 复查。
起跑后 bucket 经 REST 确认为 `private: True`。

## 耗时分解

阶段边界取自导出根下各标记文件的 mtime。

| 阶段 | 起 | 止 | 用时 |
|---|---|---|---|
| 阶段 0–1（预检、建 bucket、硬链接 stage、逐 step 分片 sha256） | 14:56:34 | 15:02:13 | 约 5 分 39 秒 |
| 阶段 2（16 批上传，190 GB，零重试） | 15:02:13 | 15:11:58 | 约 9 分 45 秒 |
| 阶段 2b–5（补传 `_run-meta`/README/清单、计数核对、全量回读、post 清单比对） | 15:11:58 | 15:25:46 | 约 13 分 48 秒 |
| 阶段 6（源侧第三遍 sha256） | 15:25:46 | 15:36:53 | 约 11 分 07 秒 |

限流下仍比 2048 那次不限流的 51 分钟快，瓶颈始终不在带宽。

## 对并行 1024 训练的影响

1024 run `v2-1600ep-m16x8x8-modul-b128-80k` 同机八卡训练。步时由其 driver.log 里 tqdm `Progress` 行的
时间戳和 `x.xxkit` 步数（精度 10 步）取首尾差计算：

| 窗口（UTC） | 导出所处阶段 | 步数 | s/步 | 相对基线 |
|---|---|---:|---:|---:|
| 14:26:00–14:56:00 | 导出前（基线） | 1810 | 0.9856 | — |
| 15:02:13–15:11:58 | 阶段 2 上传 | 570 | 0.9997 | +1.4% |
| 15:11:58–15:25:46 | 阶段 2b–5 回读与校验 | 830 | 0.9859 | +0.0% |
| 15:25:46–15:36:53 | 阶段 6 源侧 sha256 | 680 | 0.9645 | −2.1% |
| 14:56:34–15:36:53 | 导出全程 | 2460 | 0.9797 | −0.6% |

最大的偏移是阶段 2 的 +1.4%，远低于预设的 20% 停止阈值。全程平均步时与基线在噪声范围内，
导出没有中途停止。

## 产物与磁盘

- bucket：`hf://buckets/HongzeFu/robomme-vla-modul-4096-80k-v1`（private，337 文件 / 190,081,207,647 B）
- 本地 stage 由硬链接组成，不额外占盘；按 2048 那次的惯例保留 `verify/`（178 G）
- 导出后 `/scratch` 空闲 557 G，1024 余下 checkpoint 约需 170 GB

## records/

| 文件 | 内容 |
|---|---|
| `run.clean.log.txt` | 导出日志清洗版（去掉 `\r` 进度条），已检查不含 token |
| `SHA256SUMS.pre.txt` | 上传前清单，336 行，与 bucket 根下同名文件相同 |
| `SHA256SUMS.post.txt` | 回读侧独立重算的清单，336 行，与 pre 的 diff 为 0 |
| `SHA256SUMS.src-after.txt` | 源侧第三遍重算，316 行 |
| `bucket-README.uploaded.md` | 实际上传到 bucket 的 README（训练结果一节由 driver.log 现场生成） |
