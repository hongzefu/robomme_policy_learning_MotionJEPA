# hf-export-h5v2-rehost-20260914 — 起跑

把公开 dataset repo `HongzeFu/robomme-4task-h5-20260912-v2`（2057 件 / 126.6 GB，
revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`）原样搬进**同名公开 bucket**
`hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2`，校验通过后删除原 dataset repo。

## 为什么搬

用户判定 2026-09-12 那批数据当初传成 dataset repo 的形式选错了，要改成 bucket（对象存储）。

## 口径（用户 2026-09-14 拍板）

| 项 | 决定 |
|---|---|
| 可见性 | 公开（与原 repo 一致） |
| 命名 | 与原 dataset repo 同名（bucket 与 repo 命名空间独立，账号下 `HongzeFu/MotionJEPA` 已有 bucket 与 model repo 同名并存的先例） |
| 原 repo 处置 | **校验通过后直接删除**。删除前已告知：repo 有 30 天内 281 次下载、讨论区只有 1 个 bot 自动条目（无人类互动），删除会让外部的 `load_dataset` 与固定 revision 引用全部失效 |

## 起跑信息

- **起跑 commit**：`f4a22bb`（日志首行记录实际值）
- **起跑时刻**：2026-09-14T17:45:29+00:00
- **tmux 会话**：`hfup-h5v2`
- **源**：dataset repo `HongzeFu/robomme-4task-h5-20260912-v2` @ `604f16da…`；
  另有本机 2026-09-13 独立下载的 14 个 primary tar.xz（`/scratch/hongze/robomme-4task-h5-20260912-v2/snapshot/`，90.5 GB）用作 L5 异地副本对拍
- **落点**：`v1-store/exports/hf-h5v2-rehost/{verify,tmp,logs}`

## 搬迁方式：服务端零字节复制

实测 repo 的 2057 个文件里 **1978 个（126,588,285,456 B，即全部 `.tar.xz` 与 `.mp4`）带
xetHash**，可经 `batch_bucket_files(copy=[(repo_type, repo_id, xet_hash, dest)])` 让 Hub
服务端按内容哈希直接挂到目标 bucket（文档原文 "This is a server-side operation — no data is
downloaded or re-uploaded"）；其余 79 个普通 git blob（22,116,164 B）走客户端中转。

**实测效果：18 批 / 126.59 GB 在约 25 秒内完成。** 对照原方案（本地下载缺的 36 GB + 上传
126.6 GB ≈ 40 分钟），这一条既省时间，又把「两侧是同一份内容」变成内容寻址层面的直接事实。

bucket 里保留 repo 全部 2057 个文件**逐位原样**（含 `.gitattributes` 与未改动的 `README.md`），
迁移说明另放 `MIGRATION.md`——这是本轮唯一新增的对象。之所以不改 README，是因为原 repo 建成
后要删除，例外越少，「删掉的东西在 bucket 里一件不缺」这句话越站得住。

## 起跑前的 smoke（阶段 3，已通过）

两个未知一次验完，都是「不提前验就只会在搬完 126 GB 后的回读阶段才炸」的那类：

```
SMOKE_TARGET bytes=183418 basename_bytes=247
  videos/VideoRepick/medium/success_NO_OBJECT_VideoRepick_ep1_seed9100_…__HASH__0cb4ced1022e.mp4
SERVERSIDE_COPY=OK
LONGNAME_ROUNDTRIP=OK bytes=183418 sha256=73f194354afde8e9…
```

1. **服务端复制可用性**：`hf cp` 的帮助文本明写 remote-to-remote 只在同一 storage region 内
   有效，而新建 bucket 落在哪个区不由我们控制。
2. **最长文件名**：本 repo 里 basename 最长的是 **247 字节**，而 `/scratch` 是 xfs、
   `NAME_MAX=255`，只剩 8 字节余量。smoke 刻意挑这个文件，验证它能原样往返落盘。

## 六层校验

| 层 | 做什么 | 判定行 |
|---|---|---|
| L1 清单 | bucket 递归列举 ⟷ repo 清单，逐条路径与字节；允许且仅允许 `MIGRATION.md` 一个 extra | `INVENTORY_DIFF=0` |
| L2 XetHash | 1978 条内容寻址哈希逐条对拍（零字节） | `XETHASH_MATCH=1978 MISMATCH=0` |
| L3 回读 | 全量 126.6 GB 拉回本地，从零独立重算 | `VERIFY_FILES` / `VERIFY_BYTES` |
| L4 账本 | repo 自带的**两份互相独立**的账本：`SHA256SUMS`(2055 行) 与 `MANIFEST.json` 的 `archives`(29) + `videos`(1949) | `SHA256SUMS_OK` / `MANIFEST_ARCH_OK` / `MANIFEST_VIDEO_OK` |
| L5 异地副本 | bucket 取回的 14 个 primary tar.xz 与本机 09-13 独立下载的那份**逐字节** `cmp` | `SNAPSHOT_IDENTICAL=14` |
| L6 匿名 | 清 token + 全新空缓存匿名读 | `PUBLIC_ANON_READ=OK` |

L5 是删除授权的关键一层：它排除「源 repo 本身在某个时点被改过、而我们把改过的版本原样搬走了」
这一种情况——两份来源不同、时间不同的副本都对得上，才谈得上放心删。

## 删除前已抓走的元信息（`records/`）

bucket 是对象存储，没有 card / revision / discussions 概念。以下在 repo 删除后永久不可得，
已在任何改动之前抓好（`SRC_SNAPSHOT_SAVED files=2057 sha=604f16da36d6 xet=1978 commits=16`）：

| 文件 | 内容 |
|---|---|
| `revision.json` | revision 全量元数据（每个 blob 的 oid/size/lfs） |
| `tree.json` | 递归文件树（xetHash + 每文件 lastCommit） |
| `commits.json` | 16 个提交的 id / title / date |
| `paths_info.json` | 全部路径的 paths-info |
| `discussions.json` | 讨论区与 PR |
| `croissant.json` | Croissant 元数据（dataset viewer 的机器可读描述） |
| `README.raw.md` | README 原文，**含 YAML front matter**（`license: apache-2.0`、`task_categories`、`tags`、`pretty_name`）——bucket 不渲染 dataset card，这段机器可读的 license 声明事实上会消失 |
| `gitattributes.raw.txt` | LFS 追踪规则 |
| `summary.json` | 下载数 281 / likes / 创建时间 / sha / usedStorage 的一次性快照 |

## 删除（独立一步，不在 driver 里）

```bash
# 只做前置断言、不删：
bash scripts/dataset/hf_export/delete_h5v2_dataset_repo.sh
# 确认后真删：
CONFIRM_DELETE=1 bash scripts/dataset/hf_export/delete_h5v2_dataset_repo.sh
```

最危险的一点已在脚本里设防：`hf repos delete <id> --repo-type dataset` 与
`hf buckets delete <id>` 的 `<id>` **字符串完全相同**，且前者的 `--repo-type` 默认是 `model`。
故删除前断言 dataset 与 bucket 两者都在、bucket 文件数等于 2058；删除后断言 dataset 变 404/401
而 **bucket 仍在、文件数与字节数一个不少**，并再匿名读一次。
