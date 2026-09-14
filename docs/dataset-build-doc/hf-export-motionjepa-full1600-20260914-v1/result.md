# hf-export-motionjepa-full1600-20260914-v1 — 结果

**终判 PASS。** MotionJEPA full1600 建库产物已上传到**公开** bucket
`hf://buckets/HongzeFu/robomme-4task-motion-full1600-20260914-v1`：**2991 个对象 /
479,545,649,918 字节**（`PUBLISHED.json` 清单的 2804 件 + 清单自身 + `control/` 溯源 184 项
+ README 与 pre 清单）。上传前与回读后两遍独立重算的 sha256 逐行 diff 为空，覆盖全部 479.5 GB
字节；匿名（无 token）可读已验证。起跑口径见 [`launch.md`](launch.md)。

执行窗口 2026-09-14T17:41:58+00:00 → 19:36:03（1 h 54 min），起跑 commit
`04d9ac421cb7793737acd08128c83b680a28b077`，tmux `hfup-full1600`。**本轮共起过
`hfup-full1600` 与 `hfup-h5v2` 两个会话**；用户自有的 6 个会话（`0`、`1`、`claude-private`、
`codex`、`codex-repo`、`tr-wan-full1600-filter2-b176x4-72ep-a`，最后一个是在跑的 72 ep 训练）
全程未动。

## 一、判定行（全部命中，原文）

```
HF_WHOAMI=HongzeFu
STAGE_DONE files=2991 bytes=479545649918
STAGE_SIDECARS=0
PUBLISHED_SHA_MATCH=2804 expected=2804 missing=0 mismatches=0
LOCAL_FILES=2991 LOCAL_BYTES=479545649918
PRE_LINES=2990
HYGIENE=PASS scanned=175 hits=28186 ungated=0
BUCKET_CREATED=HongzeFu/robomme-4task-motion-full1600-20260914-v1 PRIVATE=False
BUCKET_FILES_AT_START=0 BUCKET_BYTES_AT_START=0
SYNC_PLAN_LINES=2992 SYNC_PLAN_DELETES=0
UPLOAD_DONE batches=58
BUCKET_FILES=2991 BUCKET_BYTES=479545649918
SHA256_PRE_POST_DIFF=0
  sha256sum -c 校验行数: 2990
MEMBERS_SPOT=PASS tar_members=2800 checked=200 mismatches=0
RESULT=PASS
SRC_UNCHANGED=OK checked=2804 mismatches=0
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
EXIT_CODE=0
PLAN_BATCHES=58 PLAN_FILES=2991 PLAN_BYTES=479545649918 ROOT_FILES=6 MAX_BATCH_BYTES=8588427264 MAX_BATCH_FILES=184 PLAN_SHA=0d81533db32e6d4f
```

## 二、五层校验闭环

| 层 | 做什么 | 结果 |
|---|---|---|
| **清单层** | `PUBLISHED.json` 自带的 2804 个文件 sha256（建库侧当时写下的**第三方账本**）对 stage 重算结果逐条比对 | `PUBLISHED_SHA_MATCH=2804 missing=0 mismatches=0` |
| **计数层** | buckets REST API 的 `totalFiles` / `size` 对本地 `find` 统计 | `2991 / 479,545,649,918`，**逐字相等**，未触发异步滞后复查 |
| **回读层** | `hf sync bucket → verify/` 全量取回，**从零独立重算** post 清单，与 pre 清单 `diff` | `SHA256_PRE_POST_DIFF=0`，覆盖全部 479.5 GB |
| **内容层** | 在 verify 侧再叠一遍 `sha256sum -c --strict` | 2990 行全 OK |
| **成员抽样** | `control/content_hashes.tar` 随机抽 200 个成员解包，逐字节对源文件复算 | `MEMBERS_SPOT=PASS tar_members=2800 checked=200 mismatches=0` |
| **收尾层** | 源侧第三遍 sha256，反证硬链接全程未改动原件 | `SRC_UNCHANGED=OK checked=2804 mismatches=0` |

**回读是真实网络下载，不是本地缓存重建**——这一点专门核过：sync 摘要为
`Downloads: 2991 / Skips: 0`，而本地 `v1-store/cache/hf-xet` 只有 2.4 GB，
不可能靠它重建 479.5 GB。

**清单层是本链路相对 400ep 那条多出来的一层。** 其余各层证明的都是「bucket 里的东西 = 我这次
从源目录读到的东西」；只有 `PUBLISHED.json` 这份建库时就写好的账本，能证明「上传的就是建库时
验收过的那批字节」——即数据自建库以来一字未动。

## 三、分段耗时与吞吐

| 段 | 窗口 | 耗时 | 吞吐 |
|---|---|---|---|
| stage 组装（硬链接 2989 项） | 17:41:58→ | **0.8 s** | 纯元数据操作，零拷贝 |
| 阶段 2 全量 sha256（479.5 GB，16 并行） | →17:43:12 | **74 s** | 约 6.5 GB/s |
| 体检 + 建 bucket + dry-run | →17:44 | < 1 min | — |
| 阶段 6 分批上传 58 批 | 17:44→18:23 | **约 39 min** | 约 205 MB/s |
| 阶段 7 计数层 | 18:23→18:26 | 约 3 min | — |
| 阶段 8 全量回读 479.5 GB | 18:26→18:52 | **约 26 min** | 约 307 MB/s（实测瞬时 356 MB/s） |
| 阶段 9 post 重算 + diff | 18:52→18:57 | 约 5 min | — |
| 阶段 9 `sha256sum -c`（单进程 479.5 GB） | 18:57→19:16:37 | **约 19 min** | 单进程，是本轮最慢的一段 |
| 阶段 10 源侧第三遍 + 阶段 11 匿名读 | →19:36:03 | 约 19 min | — |

**吞吐口径**：本环境为 AWS 单机本地 NVMe RAID（`/dev/md0`），与历史的 turbo NFS、旧本机
`/data` NVMe 数字**不得混比**（AGENTS 第 13 条）。上传期间批 2 的搬迁与回读并行占用带宽，
205 MB/s 不代表链路上限（2026-09-08 单独跑 400ep 时实测 546 MB/s）。

## 四、分批按字节预算：本轮最实在的一处改进

2026-09-08 那轮的失败模式是服务端 `new_upload_commit` 随**批内字节量**增长而超时
（1.62 GB×10=16.2 GB 过、2 GB×10=20 GB 挂、2 GB×5=10 GB 稳）。本轮的 2800 个 `.bin` 是
**重尾分布**：

```
min 33.7 MiB   中位 132 MiB   p90 269 MiB   max 612 MiB
```

按原计划「每批 25 个」，碰上大 latent 那批就是 25 × 612 MiB ≈ **15 GB**，正好落进已知会挂的
区间。改为 ≤8 GiB / ≤256 件贪心装箱（`plan_upload_batches.py`）后：

| | 值 |
|---|---|
| 批数 | **58** |
| 每批字节 min / 中位 / max | 0.07 / 8.49 / **8.59 GB**（预算上限 8 GiB = 8.59 GB） |
| 每批件数 min / max | 18 / 184 |
| **重试次数** | **0**（对照：400ep 那轮 25 批里 3 批需重试、其中一批连挂三次） |
| 计划哈希 | `0d81533db32e6d4f…`（落盘防重跑分组漂移） |

中位 8.49 GB 贴着 8.59 GB 的上限，说明贪心装箱几乎填满每个箱而不越界。

## 五、公开前体检与已披露事项

`HYGIENE=PASS scanned=175 hits=28186 ungated=0`，闸门跑在 `hf buckets create` **之前**。

| 模式 | 含义 | 命中 | 处置 |
|---|---|---|---|
| P5 | 凭据（token / AWS key / 私钥 / wandb key） | **0** | — |
| P1/P2/P3 | NFS turbo、`coe-chaijy`、环境 A 路径 | **0** | — |
| P4/P6/P8/P11 | Slurm id、GitHub 地址、GPU UUID、私有录制 | **0** | — |
| P9 | 构建机路径 `/scratch/hongze` | 25,380 | accept + README 披露 |
| P7 | 构建机内部主机名 | 2,801 | accept + README 披露 |
| P10 | `MotionJEPA` 仓库名 | 5 | accept + README 披露 |

裁决账本 [`hygiene_allowlist_motionjepa_full1600.json`](../../../scripts/dataset/hf_export/hygiene_allowlist_motionjepa_full1600.json)
的 7 条 `expected_count` 合计 **28,186**，与实测 `total_hits` 逐字相等——账本因此是闸门而非注释。

**P7 需要专门说明**：`wan_chunk_latents/metadata.json` 的每条记录带
`"hostname": "ip-10-242-11-177.us-west-2.compute.internal"`（2800 条同一台机器），即公开后会
暴露构建机的 **AWS 私有 IP 与 region**。该文件是发布清单必需件（下游靠它读每段 latent 的
shape / dtype / 偏移），排除它等于让 2800 个 `.bin` 无法使用；而改字节会让 `PUBLISHED.json`
的整套 sha256 校验链失配。用户 2026-09-14 在被明确告知含私有 IP 与 region 后拍板
**原样传 + README 披露**。10.x 是 RFC 1918 私有地址，对外不可路由、非凭据。

**8400 处 P7 内部主机名被「清单驱动」自然挡在门外**：它们全部落在 `*.complete.json` 边车上，
而边车不在 `PUBLISHED.json` 清单里。stage 组装严格按清单逐条硬链接，driver 另有硬断言
`STAGE_SIDECARS=0` 兜底——若有人图省事写成 `cp -al dataset-token stage`，这些主机名就上了
公开库，而公开过就是公开过、不可逆。

## 六、一处实现缺陷（已修，记录经过）

阶段 10「源侧第三遍 sha256」最初写成**单进程 Python 逐文件 hashlib**。运行中实测：18 分 51 秒
只读了 43.4 GB，即约 **38 MB/s**，照此 479.5 GB 需要约 3.5 小时——而阶段 2 用
`xargs -P 16 sha256sum` 算同样的 479.5 GB 只花了 **74 秒**（约 6.5 GB/s）。慢 170 倍，却换不到
任何信息增量。

期间为了不空等，我用并行方式独立算了一遍 stage（stage 每项都是源文件的硬链接、同 inode，读
stage 即读源文件），与阶段 2 落盘的 pre 清单逐行比对：**2990 行全同**。结论与阶段 10 要证明的
完全一致，且覆盖面更宽（2990 项含 `control/` 与 README，阶段 10 只核发布清单的 2804 项）。

**这次并行重算顺带把数据预热进了页缓存**，单线程那段随后从缓存读，于 19:16:37→19:36:03 约
19 分钟内跑完，打印 `SRC_UNCHANGED=OK checked=2804 mismatches=0`。也就是说它最终自己完成了，
但那 19 分钟是靠缓存偶然加速的，冷缓存下的真实代价是 3.5 小时。

脚本已改为复用 `hash_tree` 并行，判据同时从 2804 项放宽到 stage 全部 2990 项
（commit `fix: 源侧第三遍 sha256 改为并行，单线程版慢 170 倍`）。**本轮留档里的
`SRC_UNCHANGED=OK checked=2804` 是旧单线程版打印的**，与并行版的 2990 行结论互相印证。

另一处可优化但本轮未动：阶段 9 的 `sha256sum -c --strict` 是单进程读 479.5 GB，用了
18:57:28→19:16:37 **约 19 分钟**，而它与同阶段的 `diff pre post` 用的是同一份回读字节、结论
重复。下次可并入 `hash_tree` 的并行重算，一次读盘出两个结论。

## 七、本地产物的处置

- `v1-store/exports/hf-motionjepa-full1600/stage/`：**全部是硬链接**（2989 项指向
  `dataset-local-a100-4task-full1600/` 的原 inode），不额外占盘。删除 stage 不会动原件，
  但原件在、stage 就能秒级重建。
- `v1-store/exports/hf-motionjepa-full1600/verify/`：479.5 GB 回读副本，校验已用完，
  **可回收**（源目录与 bucket 都在，不像批 2 那份是唯一副本）。
