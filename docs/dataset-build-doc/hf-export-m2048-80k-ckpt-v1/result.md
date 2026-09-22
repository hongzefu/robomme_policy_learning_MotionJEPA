# hf-export-m2048-80k-ckpt-v1 · result

**结论：通过。** 训练 run `v2-1600ep-m32x8x8-modul-b128-80k` 的全部 16 个 orbax checkpoint
已完整上传到 HuggingFace private bucket `HongzeFu/robomme-vla-modul-2048-80k-v1`，
上传前 / 上传后两遍独立 sha256 逐行对照 **diff 为 0**，源文件未被硬链接连带改动。

- 导出 commit：`54a48d3`（commitV11.3）+ `0b8ca74`（起跑留档）
- tmux 会话：`exp2048-ckpt`（本轮起过的唯一会话，脚本正常退出后自行结束）
- 起止：`2026-09-22 14:52:46` → `15:43:48`（UTC），**约 51 分钟**
- 日志：`v1-store/exports/hf-ckpt-v2-1600ep-m32x8x8-modul-b128-80k/logs/export.log`

## 验收判定行（逐条实测）

```
未检测到训练进程 => 趟 B（补传 + 全量验收）
HF_WHOAMI=HongzeFu
LOCAL_FILES=343 LOCAL_BYTES=190076831330 PRE_LINES=342
PRE_PARTS_CONSISTENT=OK
BUCKET_FILES=343 BUCKET_BYTES=190076831330
SHA256_PRE_POST_DIFF=0
  sha256sum -c 校验行数: 342
RESULT=PASS
  pre(剔 README/_run-meta)=329 行, src-after=329 行, _run-meta=12 行
SRC_UNCHANGED=OK
全部完成
EXIT_CODE=0
```

`LOCAL_FILES=343` 与起跑留档的预估**逐字吻合**：329（源）+ 12（`_run-meta/`）+ 1（`README.md`）
+ 1（`SHA256SUMS.pre.txt`）。`BUCKET_FILES` / `BUCKET_BYTES` 与本地一次核对即相等，
未触发 120 秒异步滞后复查。

## 落点现状（buckets REST API 实测）

| 项 | 值 |
|---|---|
| bucket id | `HongzeFu/robomme-vla-modul-2048-80k-v1` |
| private | `True` |
| totalFiles | 343 |
| size | 190,076,831,330 字节（约 177.0 GiB） |
| createdAt | `2026-09-22T14:52:47Z`（本轮由 driver 新建） |

## 耗时分解

| 阶段 | 起 | 止 | 用时 |
|---|---|---|---|
| 阶段 0–1（预检、建 bucket、硬链接 stage、逐 step 分片 sha256） | 14:52:46 | 14:58:02 | 约 5 分 16 秒 |
| 阶段 2（16 批上传，190 GB） | 14:58:02 | 15:27:51 | 约 29 分 49 秒 |
| 阶段 2b–6（补 `_run-meta`/README、计数核对、全量回读、post 清单比对、源侧第三遍 sha256） | 15:27:51 | 15:43:48 | 约 15 分 57 秒 |

上传平均约 106 MB/s（存储为 AWS 本地 NVMe RAID `/dev/md0`）。

## 中途的一次 TimeoutError（已由重试自愈）

第 9 批（step 45000）上传时命中蓝本注释里记录过的**已知间歇性失败**：字节已全部传完，
却在 `huggingface_hub/hf_api.py` 的 `_batch_bucket_files → session.new_upload_commit`
抛出

```
TimeoutError: Timeout: Request error: error decoding response body, domain: no-url
```

driver 的 8 次重试机制生效，**第 1 次重试即成功**（Xet 内容寻址，服务端已有相同块，
重试不重传字节）。其余 15 批一次通过。这条印证了蓝本注释的判断——该超时是间歇性的、
不是确定性失败，重试次数定在 8 次而非 3 次是对的。

## 与蓝本 motion80k 导出的对照

| 项 | motion80k（`robomme-vla-modul-motion-80k-v1`） | 本次（`robomme-vla-modul-2048-80k-v1`） |
|---|---|---|
| 趟次 | 趟 A（训练在跑，限流）+ 趟 B | **只有趟 B**（训练已结束才起跑） |
| bucket 文件数 | 344 | 343 |
| bucket 字节数 | 190,662,023,078 | 190,076,831,330 |
| `_motion-encoder/` | 有（456 MB encoder + config） | **无**（本 run motion 关闭） |
| 限流 | 趟 A 走 `ionice -c 3 nice -n 19`、`NPROC=2` | 全程不限流、`NPROC=8` |

本次不限流是因为起跑时八卡 GPU 全空闲、无训练进程争抢 `/dev/md0` 读带宽，
driver 的存活判定自动选择了趟 B 路径。

## 归档物

`records/` 下留四份（均为 Git 无法还原的产物）：

| 文件 | 说明 |
|---|---|
| `SHA256SUMS.pre.txt` | 上传前逐文件 sha256，342 行，路径相对 bucket 根 |
| `SHA256SUMS.post.txt` | 回读侧独立重算，342 行，与 pre **逐行相同** |
| `SHA256SUMS.src-after.txt` | 源侧第三遍重算，329 行，证明硬链接未改动原件 |
| `run.clean.log.txt` | 导出日志去掉 tqdm 进度回车帧后的可读版，291 行 |
| `bucket-README.uploaded.md` | 实际上传到 bucket 根的 README 正本（训练结果一节由日志现读现生成） |

大模型权重本身不归档（第 12 条），bucket 即异地副本。

## 遗留与建议

- `v1-store/exports/` 下的 `verify/` 回读副本共占约 178 GiB，`stage/` 是硬链接不额外占盘。
  验收已通过，**`verify/` 可删**以回收空间；本轮未删，交用户决定。
  同理，此前三次导出的 `verify/` 也仍在盘上（`/scratch` 当前余量约 1.2 T）。
- 本 run 的策略评估尚未开展，bucket README 已显式声明「训练 loss 不代表策略评估成功率」。
