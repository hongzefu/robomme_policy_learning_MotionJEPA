# hf-export-h5v2-rehost-20260914 — 结果

**终判 PASS，原 dataset repo 已删除。**

四任务 h5 已从 dataset repo `HongzeFu/robomme-4task-h5-20260912-v2`（revision
`604f16da36d6b6d175884df8fb687dc08e0a36eb`）原样搬入同名**公开** bucket
`hf://buckets/HongzeFu/robomme-4task-h5-20260912-v2`：**2058 个对象 / 126,610,404,806 字节**
（原 repo 的 2057 件逐位原样 + 新增的 `MIGRATION.md`）。六层校验全过后执行删除，
删除后 bucket 文件数与字节数一个不少、匿名仍可读。

起跑口径见 [`launch.md`](launch.md)。执行窗口 2026-09-14T17:45:29+00:00 起，`EXIT_CODE=0`，
tmux `hfup-h5v2`（本轮共起过 `hfup-h5v2` 与 `hfup-full1600` 两个会话，用户自有的 6 个会话
`0`、`1`、`claude-private`、`codex`、`codex-repo`、`tr-wan-full1600-filter2-b176x4-72ep-a`
全程未动）。

## 一、判定行（全部命中，原文）

```
HF_WHOAMI=HongzeFu
SRC_SNAPSHOT_SAVED files=2057 sha=604f16da36d6 xet=1978 commits=16
PROBE_FILES=2057 XET=1978 PLAIN=79 XET_BYTES=126588285456 PLAIN_BYTES=22116164
SMOKE_TARGET bytes=183418 basename_bytes=247
SERVERSIDE_COPY=OK
LONGNAME_ROUNDTRIP=OK bytes=183418 sha256=73f194354afde8e9…
BUCKET_CREATED=HongzeFu/robomme-4task-h5-20260912-v2 PRIVATE=False
BUCKET_FILES_AT_START=0
COPY_DONE xet=1978 plain=79 total=2057
INVENTORY_DIFF=0 BUCKET_FILES=2058 REPO_FILES=2057 MISSING=0 BADSIZE=0 EXTRA=['MIGRATION.md']
XETHASH_MATCH=1978 EXPECTED=1978 NOT_REPORTED=0 MISMATCH=0
VERIFY_FILES=2058 VERIFY_BYTES=126610404806
SHA256SUMS_OK=2055
MANIFEST_ARCH_OK=29/29 MANIFEST_VIDEO_OK=1949/1949
SNAPSHOT_IDENTICAL=14 DIFFERENT=0
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK anon_files=2
REHOST_RESULT=PASS
EXIT_CODE=0
```

删除（`delete_h5v2_dataset_repo.sh`，`CONFIRM_DELETE=1`）：

```
DELETE_PRECHECK=PASS
  dataset repo HTTP=200 → 删除 → HTTP=404
  bucket files=2058 bytes=126610404806 private=False（删除前后逐字相同）
DATASET_DELETED=OK
BUCKET_INTACT=2058 PRIVATE=False
POST_DELETE_ANON_READ=OK
DELETE_RESULT=PASS
```

## 二、搬迁方式：服务端零字节复制（本轮最大的一处改进）

repo 的 2057 个文件里 **1978 个（126,588,285,456 B，全部 `.tar.xz` 与 `.mp4`）带 xetHash**，
经 `batch_bucket_files(copy=[(repo_type, repo_id, xet_hash, dest)])` 由 Hub 服务端按内容哈希
直接挂载；其余 79 个普通 git blob（22,116,164 B）走客户端中转。

**实测 18 批 / 126.59 GB 在约 25 秒内完成**，对照原计划的「本地补下载 36 GB + 上传
126.6 GB ≈ 40 分钟」。除了快，它还把「两侧是同一份内容」从「我下下来又传上去、但愿中间没坏」
变成内容寻址层面的直接事实（`XETHASH_MATCH=1978 MISMATCH=0`）。

**取 xetHash 必须走 REST**：`POST /api/datasets/<repo>/paths-info/<rev>` 带
`{"paths":[...], "expand":true}`。SDK 的 `pi.xet_file_data.hash` 取不到（全返回 None），
这一点一度让我误判成「0 个文件可服务端复制」。

## 三、六层校验

| 层 | 覆盖什么 | 结果 |
|---|---|---|
| **L1 清单** | 全部 2057 条路径与字节，逐条比对 | `INVENTORY_DIFF=0`，extra 恰为 `['MIGRATION.md']` |
| **L2 XetHash** | 1978 条内容寻址同一性（零字节、零下载） | `1978/1978`，`MISMATCH=0` |
| **L3 回读** | 126.61 GB 全量拉回本地，从零独立重算 sha256 | `VERIFY_FILES=2058` |
| **L4 账本** | repo 自带的**两份互相独立**的已发布账本 | `SHA256SUMS` 2055 行全 OK；`MANIFEST.json` 的 `archives` 29/29 + `videos` 1949/1949 |
| **L5 异地副本** | bucket 取回的 14 个 primary tar.xz ⟷ 本机 2026-09-13 独立下载的那份，逐字节 `cmp` | `SNAPSHOT_IDENTICAL=14 DIFFERENT=0` |
| **L6 匿名** | 清 token + 全新空缓存，陌生人视角取用 | `PUBLIC_ANON_READ=OK` |

**L5 是删除授权真正的依据。** 前四层都只能证明「bucket 里的东西 = 我这次从 repo 读到的东西」；
只有拿一份**来源不同、时间不同**（09-13 下载）的副本对上，才排除「源 repo 本身在某个时点被改
过、而我们把改过的版本原样搬走了」这一种情况。

## 四、起跑前 smoke 挡下的两个坑

两个都属于「不提前验就只会在搬完 126 GB 之后的回读阶段才炸」：

1. **服务端复制的 region 约束**：`hf cp` 帮助文本明写 remote-to-remote 只在同一 storage
   region 内有效，而新建 bucket 落在哪个区不由我们控制。实测 `SERVERSIDE_COPY=OK`。
2. **最长文件名**：repo 里 basename 最长 **247 字节**，而 `/scratch` 是 xfs、`NAME_MAX=255`，
   只剩 8 字节余量。smoke 刻意挑这个文件做往返，实测 `LONGNAME_ROUNDTRIP=OK`。
   （顺带查清：`hf download --local-dir` 模式会写同名 + `.metadata` 的边车，247+9=256 会
   `ENAMETOOLONG`；而 bucket 的 `download_bucket_files` 直接写目标路径、不加后缀，故安全。）

## 五、同名 bucket 与 dataset repo 可以并存（此前的未验证假设）

建 bucket 时同名 dataset repo 仍在，两者并存无冲突（`BUCKET_CREATED=… PRIVATE=False` 与
`dataset repo HTTP=200` 同时成立）。原先准备的「撞名就先用 `-bucket` 临时名、删完再
`hf buckets move` 改回」逃生门没有用上。

## 六、迁移损失了什么（已记录，不可恢复）

bucket 是对象存储，没有 card / revision / discussions 概念。以下随 repo 删除永久消失，
删除前已完整抓进 [`records/`](records/)：

| 留档文件 | 内容 |
|---|---|
| `revision.json` (996 KB) | revision 全量元数据 |
| `tree.json` (1.5 MB) | 递归文件树（xetHash + 每文件 lastCommit） |
| `paths_info.json` (1.5 MB) | 全部路径的 paths-info |
| `commits.json` | 16 个提交的 id / title / date |
| `discussions.json` | 讨论区（只有 1 个 bot 的 "Conversion to Parquet"，无人类互动） |
| `croissant.json` | Croissant 元数据 |
| `README.raw.md` | README 原文，**含 YAML front matter**：`license: apache-2.0`、`task_categories: [robotics]`、`tags: [robotics, manipulation, maniskill, robomme, hdf5]`、`pretty_name` |
| `gitattributes.raw.txt` | LFS 追踪规则 |
| `summary.json` | 下载数 **281**、likes 0、创建时间、sha、usedStorage 的一次性快照 |

**机器可读的 license 声明事实上消失了**（bucket 不渲染 dataset card）。补救：front matter 原文
仍在 bucket 的 `README.md` 开头，且 `MIGRATION.md` 正文里重新用人可读的方式声明了 Apache-2.0。

另：`load_dataset()`、`hf_hub_download(repo_type="dataset")`、`snapshot_download()` 与任何钉住
`revision=604f16da…` 的引用**全部失效**。repo 删除前 30 天内有 281 次下载记录。

## 七、本地副本的处置

`v1-store/exports/hf-h5v2-rehost/verify/`（126.61 GB）是回读校验的产物，**暂不清理**——
`videos/`(26.3 GB) 与 `spare/`(9.7 GB) 此前本机从未有过完整副本，删除 repo 之后 bucket 是唯一
来源，保留 `verify/` 作为过渡期的本地第二副本。确认 bucket 稳定后可回收。
（`/scratch/hongze/robomme-4task-h5-20260912-v2/snapshot/` 那 90.5 GB 与 `extracted/` 那
793.8 GB 不属本轮范围，原样保留。）
