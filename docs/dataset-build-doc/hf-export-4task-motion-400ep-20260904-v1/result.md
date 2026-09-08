# hf-export-4task-motion-400ep-20260904-v1 — 结果

**终判 PASS。** `v1-store/datasets/4task-motion-400ep` 全库已上传到**公开** bucket
`hf://buckets/HongzeFu/robomme-4task-motion-400ep-20260904-v1`：**122 个对象 /
130,589,124,350 字节 / tar 内成员 229,323 个**，上传前与回读后两遍独立重算的 sha256
清单 diff 为空，匿名（无 token）可读已验证。起跑口径见 [`launch.md`](launch.md)。

末轮执行窗口 2026-09-08 04:23:49 → 04:42:05（**18 min 16 s**），起跑 commit `1c5b387`，
tmux `hf-ds400ep-export`。**本轮共起过的 tmux 会话仅此一个**；用户自有会话 `0`、`1`、
`claude-private` 全程未动。

## 一、判定行（全部命中，原文）

```
HF_WHOAMI=HongzeFu
HYGIENE=PASS scanned=3425 hits=4897 ungated=0            ← 阶段 1，源库侧
BUCKET_CREATED=HongzeFu/robomme-4task-motion-400ep-20260904-v1 PRIVATE=False
HYGIENE=PASS scanned=19 hits=50 ungated=0                ← 阶段 4，stage 侧
LOCAL_FILES=122 LOCAL_BYTES=130589124350
PRE_LINES=121
BUCKET_FILES=122 BUCKET_BYTES=130589124350
SHA256_PRE_POST_DIFF=0
  sha256sum -c 校验行数: 121
MEMBERS_SPOT=PASS tars=4 members=9839 mismatches=0
RESULT=PASS
  直传件核对行数: 49
SRC_UNCHANGED=OK
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
EXIT_CODE=0
```

打包侧（阶段 3，首轮实产、末轮全量跳过）：

```
ANCHOR_OK=3
PACK_DONE shards=67 plain=49 members=229323 bytes=130567459201
```

`BUCKET_BYTES` 比 `PACK_DONE bytes` 多 21,665,149 字节，是阶段 4 才生成的 `README.md`、
`SHA256SUMS.pre.txt`、`manifest/upload_manifest.json` 与三份 `checksums/`（其中
`sha256-source-files.txt` 229,323 行约 21 MB 占绝大部分）。

## 二、四层校验闭环

| 层 | 做什么 | 结果 |
|---|---|---|
| **成员层** | 打包时单遍读的同一份字节流逐源文件算 sha256 → bucket 内 `checksums/sha256-source-files.txt` | 229,323 行 |
| **计数层** | buckets REST API 的 `totalFiles` / `size` 对本地 `find` 统计 | 122 / 130,589,124,350，**逐字相等**，未触发异步滞后复查 |
| **回读层** | `hf sync bucket → verify/` 全量取回，**从零独立重算** post 清单，与 pre 清单 `diff` | `SHA256_PRE_POST_DIFF=0`，覆盖全部 130 GB 字节 |
| **内容层** | 在 verify 侧再叠一遍 `sha256sum -c --strict` | 121 行全 OK |
| **成员抽样** | 从 4 类 tar 各抽 1 个，流式解包逐成员复算对清单（不落地） | `MEMBERS_SPOT=PASS tars=4 members=9839 mismatches=0` |
| **收尾层** | 源侧第三遍 sha256，断言硬链接未改动原件 | `SRC_UNCHANGED=OK`，49 个直传件全对 |

**回读是真实的网络下载，不是本地缓存重建**——这一点专门核过：阶段 7 的 sync 摘要为
`Downloads: 122 / Skips: 0`，`verify/` 实际落盘 122 G / 123 个文件，而本地
`v1-store/cache/hf-xet` 只有 1.5 G，不可能靠它重建 130 GB。

## 三、分段耗时（末轮，由产物 mtime 推断）

| 阶段 | 时刻 | 耗时 |
|---|---|---|
| 0 预检 + 1 体检（源库侧 3,425 个文本文件） | 04:23:49 → 04:23:56 | 7 s |
| 2 建 bucket + 公开性断言 | → 04:23:57 | 1 s |
| 3 打包 | — | 0 s（67 片全部由 `pack_progress.jsonl` 跳过） |
| 4 README + stage 体检 + 上传前 sha256（130 GB，`xargs -P8`） | 04:23:57 → 04:25:16 | 79 s（≈1.65 GB/s） |
| 5 分批上传 25 批 | 04:25:16 → 04:30:40 | **5 min 24 s** |
| 7 全量回读 130 GB | 04:30:40 → 04:35:08 | **4 min 28 s**（≈487 MB/s） |
| 8 post sha256 + `sha256sum -c` + 成员抽样 | 04:35:08 → 04:36:25+ | ≈1 min 17 s + 单进程 `-c` |
| 9 收尾断言 + 10 公开性确认 | → 04:42:05 | 余下 |
| **末轮合计** | | **18 min 16 s** |

首轮的打包（本轮唯一一次实产）：03:08:15 → 03:13:51，**5 min 36 s** 写出 67 个 tar 共
122.3 GB（其中 3 片由 smoke 提前打好）；`pack_and_hash.py` 自报净耗时 2.1 分钟。

介质为 **AWS 本地 NVMe RAID（`/dev/md0`）**，与环境 A 的 turbo NFS 数字不混比。

## 四、公开性体检与裁决

两遍都过：源库侧 `scanned=3425 hits=4897 ungated=0`，stage 侧 `scanned=19 hits=50 ungated=0`。

**零命中（安全）**：P5 凭据（HF token / AWS key / OpenAI key / 私钥 / wandb key）、
P3 `/data/hongzefu`、P4 Slurm job id、P6 GitHub 地址、P11 未公开数据集表述。

**有命中、已逐条裁决**（账本 `scripts/dataset/hf_export/hygiene_allowlist.json`，29 条，
每条带 `expected_count`，源库变动导致数量对不上即停机）：

| 命中 | 处 | 裁决 |
|---|---:|---|
| P1/P2/P10 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA` | 1 | **accept，原样保留 + README 披露**（用户 2026-09-08 拍板） |
| P10 `"mj_repo": "/scratch/hongze/MotionJEPA"`（oracle report） | 9 | accept，同性质、同裁决，README 一并披露 |
| P7/P8 构建机主机名 + GPU UUID | 2,441 + 1,224 | accept：EC2 私有 DNS 不可路由；这些文件全部在 D2 逐位对拍范围内 |
| P9 构建机绝对路径 / 仓库名 | 1,220 | accept：字段进了 `manifest_sha256 92fa17e9…` 哈希范围，改一字符断三处绑定 |
| `logs/` 19 个建库日志 | — | **exclude，不上传**（不参与任何 sha 绑定、不被任何脚本读，泄露密度最高） |

**没有修改任何数据字节。** `scan_hygiene.py` 是只读工具、无 `--fix`——本库
store_meta / episode_manifest 的整文件 sha256 是训练 provenance 门与建库契约链的锚点，
改一个字节公开出去的库就再也过不了自己的门，处置只有「排除」与「披露」两种。

## 五、公开性确认

- `hf buckets create` 后显式 `hf buckets settings --public`（不依赖 create 的默认值），
  REST API 回 `private=False`，阶段 2 与阶段 10 各断言一次 → `BUCKET_PUBLIC=True`。
- 匿名可读实证：`env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN` 起子 shell 拉 `README.md`
  成功且非空 → `PUBLIC_ANON_READ=OK`。**清 token 是必需的**——带着自己的凭据拉下来
  证明不了任何「公开」的事。

## 六、插曲：122 个对象一次性 batch commit 超时（本轮主要成本）

**现象**：首版 driver 用一次 `hf sync "$STAGE" "$BUCKET"` 提交全部 122 个对象。三次尝试
全部在**数据传完之后**的提交步骤失败：

```
第 1 次  New Data Upload 79.0GB/79.0GB 100%  →  TimeoutError
第 2 次  New Data Upload 18.6GB/18.6GB 100%  →  TimeoutError
第 3 次  New Data Upload 17.8GB/17.8GB 100%（速率 546 MB/s）→ TimeoutError
上传三次仍失败 / EXIT_CODE=1，bucket 始终 files=0
```

错误原文 `TimeoutError: Timeout: Request error: error decoding response body, domain: no-url`，
栈顶 `_execute_plan → api.batch_bucket_files → session.new_upload_commit`。

**判读**：传输速率 546 MB/s，瓶颈不在带宽；每次重试的待传量在收敛（79 → 18.6 → 17.8 GB，
Xet 内容寻址让服务端已有的块不必重传）但 commit 负担不变，所以重试解决不了。是服务端
一次性提交 122 个 Xet 对象的 batch commit 过重。

**对策与验证**：改为按 stage 子目录 + `--include` 分批。第一批（10 个 data 分片 16 GB）
一次通过、`files=10` —— 分批假设成立。随后发现两件事：

1. features 分片单个约 2 GB（data 分片 1.62 GB），10 个一批仍偏重：data 三批全部一次过，
   features 三批里两批要重试、`features-0002*` 连挂三次。→ features 细分到**每批 5 个**。
2. 这个超时是**间歇性**的：`features-0000*` 第 2 次就过、`features-0001*` 一次过。
   待传字节在重试间稳定不再收敛，说明数据早在服务端、纯粹是 commit 请求超时。
   → 重试次数 **3 提到 8**。

改完后末轮 **25 批全部完成、0 次重试**。这坐实了根因是批大小而非网络或凭据。

**顺带修的一处逻辑缺陷**：`uploaded.marker` 原本在阶段 5 末尾才 touch，导致中途失败后
重跑会被阶段 2 的「bucket 非空即停」断言拦下——那个断言本是防 repo_id 打错污染他库的，
不该误伤自己的续传。已移到阶段 5 开头，语义变为：首次跑 marker 不存在、bucket 必须为空；
续跑 marker 存在、允许非空。

**另一处停机（按设计）**：首轮阶段 4 的 stage 侧体检以 `HYGIENE=FAIL` 停在 `README.md:133`，
命中是 README「已知事项」段解释字段格式时写出的示例 `ip-….compute.internal`，属说明文字
而非真实主机名。闸门在上传任何字节之前生效，补 4 条 scope=stage 的裁决后通过。

## 七、A100 / 本地 NVMe 数字（只记录，不与环境 A 混比）

| 项 | 实测 |
|---|---|
| 打包（读 122 GB + 写 122 GB + 双重 sha256，8 worker） | 5 min 36 s |
| sha256 聚合（`xargs -P8`，130 GB） | 79 s ≈ 1.65 GB/s |
| 上传（25 批，含 Xet 去重） | 5 min 24 s；单批峰值速率 546 MB/s |
| 回读（130 GB） | 4 min 28 s ≈ 487 MB/s |
| Xet 去重效果 | 128 GB 的批里实际新数据 79 GB；重试时进一步降到 17.8 GB |

## 八、留底与可回收

- `docs/dataset-build-doc/hf-export-4task-motion-400ep-20260904-v1/records/`：
  `SHA256SUMS.{pre,post}.txt`、`upload_manifest.json`、`features_plan.json`、
  `hygiene.json`、`hygiene-stage.json`、`run.clean.log.txt`（539 行，去掉了进度条覆写行）。
  **不放** `sha256-source-files.txt`（21 MB / 229,323 行，bucket 内已有一份）。
- `v1-store/exports/hf-dataset-4task-motion-400ep/`：
  - `verify/`（122 G）**验收已完成，可整目录删除**。
  - `stage/`（122.3 G）里 67 个 tar 是实体文件（**删了下次要重打包 5.5 分钟**），
    49 个直传件是硬链接（删它们不回收空间、也不动源库）。是否保留取决于还要不要再传。
  - `logs/`、`SHA256SUMS.src-after.txt` 建议留。
- bucket 内自带完整自验材料：`SHA256SUMS.pre.txt`（121 行）+ `checksums/` 三份
  （67 行分片 / 49 行直传件 / 229,323 行逐源文件）+ `manifest/upload_manifest.json`。

## 九、已知盲区（诚实清单）

1. **`SHA256SUMS.pre.txt` 自身不在校验范围内**（`hash_tree` 用 `-not -name 'SHA256SUMS.*'`
   排除）。它被篡改无法自证——但下游可用 bucket 内 `manifest/upload_manifest.json` 与
   `checksums/sha256-shards.txt` 交叉核对同一批 sha256。
2. **成员级只抽了 4 个 tar / 9,839 个成员**（占 229,323 的 4.3%）。全量解包复算需再落地
   122 GB 且收益为零——L1/L2 已逐字节覆盖同样这些字节，成员抽样防的是另一类错误
   （成员命名或清单错位），那种错误在任何一个分片上都会暴露。
3. **计数层曾是已知的弱判据**：buckets REST 统计历史上两次异步滞后。本轮首查即吻合、
   未触发 120 s 复查，但这不改变「计数层不单独作判据」的口径。
4. **匿名可读只验证了 `README.md` 一个文件**。它证明 bucket 的访问控制确实是公开的，
   但没有对全部 122 个对象逐个做匿名拉取验证。
5. **公开体检是模式匹配，不是语义理解**。P1–P11 覆盖的是已知的泄露形态（凭据、内部路径、
   主机名、非公开仓库名）。库里若存在这些模式之外的敏感信息，扫描器看不见。
6. 本库全部逐位结论都在 A100-SXM4-80GB 上得出，**跨 GPU 架构不保证逐位一致**。
