# hf-export-4task-motion-400ep-20260904-v1 — 起跑记录

把派生数据集 `v1-store/datasets/4task-motion-400ep`（130 GB / 229,390 个文件）上传到
**新建的公开** HF bucket `hf://buckets/HongzeFu/robomme-4task-motion-400ep-20260904-v1`。
结果见 [`result.md`](result.md)。

## 环境

环境 B（AWS 单机）。判定输出：仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`；
`/nfs/turbo/coe-chaijy-unreplicated/hongzefu` 与 `/data/hongzefu` 均不存在；无 `~/.ssh/config`；
8 × `NVIDIA A100-SXM4-80GB`。介质 AWS 本地 NVMe RAID（`/dev/md0`，6.9 T / 已用 1.1 T / 可用 5.8 T）。
本轮不用 GPU。

## 动机

这批派生库目前**只有本机 NVMe 一份，没有任何异地副本**。根目录 `external-assets-lock.md`
第五节「异地无 NFS 机器从零复刻」写明：原始 H5 在公开集 `Yinpei/robomme_data_h5` 拿得到，
但派生库需在异地用 `scripts/dataset/` 重建、**全程要 GPU**。把它传上 HF 正是为了消除这条阻塞。

生产 run `awsprod40k-b128-motion` 就是用这个库训的（起跑 2026-09-04 20:13:14），
与建库日期同为 2026-09-04 —— bucket 名里的 `20260904` 由此而来，两种读法结果相同。

## 源与落点

| 项 | 值 |
|---|---|
| 源库 | `v1-store/datasets/4task-motion-400ep` |
| 附带交付件 | `v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`（`750a8e9b…`）→ bucket 内 `assets/robomme/norm_stats.json` |
| 显式不上传 | 源库 `logs/`（19 个文件 / 12,300 行建库日志） |
| 目标 | `hf://buckets/HongzeFu/robomme-4task-motion-400ep-20260904-v1`，**创建时即公开**（用户 2026-09-08 拍板） |
| 暂存 | `v1-store/exports/hf-dataset-4task-motion-400ep/{stage,verify,tmp,logs}` |
| 凭据 | `v1-store/secrets/hf.env` 的 `HF_TOKEN` |
| 入口 | `bash scripts/dataset/hf_export/run_dataset400ep_export.sh` |
| tmux | `hf-ds400ep-export`（本轮起过的会话仅此一个；用户自有会话不动） |

## 起跑前状态（已核实）

源库实测（本轮现场量，非估值）：

| 子目录 | 文件数 | 字节数 |
|---|---:|---:|
| `source/data/` | 101,066 | 39,977,897,218 |
| `source/features/` | 123,444 | 74,191,541,170 |
| `source/meta/` | 5 | 9,550 |
| `framesamp/` | 36 | 8,098,545,523 |
| `wan-latents/` | 1,801 | 4,034,988,786 |
| `oracle/wan-mj/` | 1,211 | 4,051,677,329 |
| `motion-tokens/` | 1,801 | 27,929,423 |
| `motion/` | 5 | 21,329,950 |
| `meta/` | 2 | 119,943 |
| `logs/`（不上传） | 19 | 1,376,294 |
| **合计** | **229,390** | **130,405,415,186（121.4 GiB）** |

- `hf auth whoami` → `HongzeFu`（orgs `umich`）。
- **环境里原有的 `HF_TOKEN` 属 `yinpei-tri`，写不进 HongzeFu 命名空间**，由 `hf.env` 在 driver 里覆盖；
  本机 `~/.cache/huggingface/` 下**没有 token 文件**（只有 `hub/` 与 `xet/`），所以只能走环境变量注入。
- 空间账：stage 实体 tar +122.29 GB（49 个直传件走 `ln -f`，额外占盘 0）+ verify 全量回读 +130.41 GB
  = **峰值 +252.72 GB**，峰值时余量 5.55 T。空间不构成约束。
- `v1-store/exports/hf-ckpt-awsprod40k-b128-motion/verify/` 那 87 G 按 ckpt 留档已可删，但本轮
  **不作为前置闸门**（删它只回收 0.085 T，本轮需求 0.247 T、余量 5.8 T，删不删都跑得完）。

## 打包策略

判据只有一条：**平均文件大小是否小到会让上传退化成逐文件 HTTP 往返**，以 ~50 MB 为界。

- **打 tar（67 片）**：`source/data/`（平均 395.6 KB，4096 个/片 → 25 片）、`source/features/`
  （平均 601 KB，按 episode 贪心 ~2 GiB/片 → **37 片**，实测）、`wan-latents/`（2 片）、
  `oracle/`（2 片）、`motion-tokens/`（1 片）。后三个目录里有大量 65 字节的 `.sha256` 边车文件。
- **原样直传（49 个对象）**：`framesamp/`（31 个 ~260 MB part bin，本身就是理想 Xet 对象；
  打 tar 会毁掉「下下来即训练可直读 store」）、`motion/`、两处 `meta/`、`norm_stats.json`。

分片参数 `DATA_PER_SHARD=4096` / `FEAT_SHARD_TARGET=2 GiB` **沿用 480 GB 那次不动**，两条依据：
① 后端同构（同为 Xet bucket），且本环境 ckpt 导出已实测同一 CLI 1.30.0 的 `hf sync` 语义一致；
② 粒度数字撞上了——4096 × 395.6 KB ≈ 1.62 GB/片，与环境 A 逐字吻合（两库 pkl 同 schema 同尺寸）。

分片可复现由 `pack_and_hash.py` 的 `ti.mtime=0 / uid=gid=0 / uname=gname="" / 定序写入` 保证。

**预期目标布局：122 个 bucket 对象 / ~130.5 GB / tar 内成员 229,323 个。**

## 公开性体检（起跑前已跑，结论如下）

工具 `scripts/dataset/hf_export/scan_hygiene.py` + 裁决账本 `hygiene_allowlist.json`。
**只读、无 `--fix`**：本库 store_meta / episode_manifest 的整文件 sha256 是训练 provenance 门
（`scripts/training/g0/check_config_provenance.py` 的 `motion_store_meta_sha256`）与建库契约链的
锚点，改一个字节公开出去的库就再也过不了自己的门。处置只有「排除」与「披露」两种。

| ID | 检查项 | 严重度 | 实测 |
|---|---|---|---|
| P5 | HF token / AWS key / OpenAI key / 私钥 / wandb key | 命中即硬停 | **0** ✔ |
| P3 | `/data/hongzefu` | gate | **0** ✔ |
| P4 | Slurm job id | gate | **0** ✔（`provenance.json` 里是空值 `"slurm_job": ""`） |
| P6 | `github.com/...` | gate | **0** ✔ |
| P11 | 未公开数据集（环境 A 私有录制版） | gate | **0** ✔ |
| P1/P2 | `/nfs/turbo` + `coe-chaijy` | gate | **1 处**：`motion/meta/store_meta.json` 的 `source_pin.mj_repo_path` |
| P10 | 非公开仓库名（独立引用） | gate | 10 处：上面那条 1 处 + `oracle/wan-mj/*.json` 的 `"mj_repo": "/scratch/hongze/MotionJEPA"` 9 处 |
| P7/P8 | 构建机主机名 / GPU UUID | notice | 2,441 + 1,224 处，集中在 1,218 个 metadata json |
| P9 | 构建机绝对路径 / 仓库名 | notice | 1,220 处 |

判定行：`HYGIENE=PASS scanned=3425 hits=4897 ungated=0`。

逐条裁决（全文见 `hygiene_allowlist.json`，每条带 `expected_count`，源库若变动数量对不上即停机）：

- **P1/P2/P10 的 turbo 那条 → accept，原样保留字节 + README 显式披露**（用户 2026-09-08 拍板）。
  路径不含凭据、对外不可达，`coe-chaijy` 是公开可查的院系组名；真正起绑定作用的是同段
  `source_sha256 af67fdd9…`。oracle 那 9 处是同性质的本机路径，按同一裁决处理。
- **P7/P8/P9 → accept**。EC2 私有 DNS（RFC1918 派生）不可路由；这 1,218 个 metadata json 全部
  参与 D2 逐位对拍（`metadata_mismatches=0`），改它们等于毁掉 `WAN_BITEXACT=PASS` 的可复核性。
  各 `raw_dir` / `manifest_path` 进了 `manifest_sha256 92fa17e9…` 的哈希范围，改一个字符断三处绑定。
- **`logs/` → exclude，不上传**。19 个文件含完整 `uv run --project /scratch/...` 命令行与 venv
  内部布局，不参与任何 sha 绑定、不被任何脚本读，验证价值为零而泄露密度最高。

体检在 driver 里跑**两遍**：阶段 1 扫源库（`--scope source`，严格校验计数），阶段 4 扫 stage
（`--scope stage`）——后者才看得到我们自己新写的 `README.md`、`manifest/upload_manifest.json`、
`checksums/*.txt`（`upload_manifest.json` 原本会写构建机绝对路径，本轮已改为仓库相对路径）。

**体检闸门放在 `hf buckets create` 之前**：用户拍板创建时即公开，bucket 一建出来就没有
「发现问题还来得及」的窗口，发现问题的成本必须在写下任何字节之前付清。

## 工具链

PATH 上没有 `hf`；项目 `.venv` 的 `huggingface_hub` 是 0.32.3（openpi 钉死、不能升、无 bucket
子命令）。唯一可用路径：

```bash
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }
```

1.30 相对老脚本用的版本已重构：`hf buckets sync` → 顶层 `hf sync`；`hf buckets info` →
`hf repos ls`（只给 "130.4 GB" 这种人类可读值，精确字节/文件数/可见性改走 buckets REST API）。
bucket 子命令签名在本环境尚未实证过，driver 阶段 0 会把 `hf buckets {--help,create --help,
settings --help}` 原样 tee 进日志，实测结果补进 result.md。

## 阶段清单与幂等性

```
阶段 -1  smoke（SMOKE=1 单独跑，本地不碰网络）
阶段  0  预检：落点断言 / bucket 子命令 help / whoami        —— bucket 尚未创建
阶段  1  公开性体检 scan_hygiene.py --scope source           ★ 闸门，不过不建库
阶段  2  hf buckets create（公开）+ settings --public + totalFiles=0 断言
阶段  3  打包 122.3 GB（pack_progress.jsonl 断点续跑）
阶段  4  README + 符号链接断言 + stage 侧体检 + SHA256SUMS.pre
阶段  5  上传（3 次重试 + 60s 退避）
阶段  6  计数层核对（首次不符 sleep 120 复查）
阶段  7  全量回读 130.5 GB
阶段  8  内容层验收：pre/post diff + sha256sum -c + tar 成员抽样
阶段  9  收尾断言：源侧第三遍 sha256
阶段 10  公开性确认：private=false + 匿名无 token 拉 README
```

全部阶段幂等可重跑：阶段 3 靠 `pack_progress.jsonl` + `*_plan.json`（分组不漂移），
阶段 5/7 靠 `hf sync` 增量语义。

## smoke 结果（起跑前已通过）

`SMOKE=1 bash scripts/dataset/hf_export/run_dataset400ep_export.sh`，本地 11 s：

```
features 计划：400 episode → 37 片，共 74.2 GB
分片完成 source/data_tars/data-00000.tar（4096 个成员，1.62 GB）
分片完成 source/data_tars/data-00001.tar（4096 个成员，1.62 GB）
分片完成 source/features_tars/features-00000.tar（3187 个成员，1.92 GB）
ANCHOR_OK=3
MEMBERS_SPOT=PASS tars=3 members=11379 mismatches=0
SMOKE_BYTEWISE checked=30 mismatches=0
```

三层含义分开看：`ANCHOR_OK=3` 是三条 store_meta 交叉锚点（`pos_emb 3176ac09…`、
`state_emb 80ecd422…`、`motion_token 6e70604d…`，**均从本库自己的 store_meta 现读**，
不抄 480 GB 那次的常量——本库 `state_emb` 是 `80ecd422…`，与那次的 `8445ff9c…` 不同，
抄了会得到恒失败或恒通过的假验证）；`MEMBERS_SPOT` 是拿打包时逐成员算的 sha 对 tar 内实际
内容重算；`SMOKE_BYTEWISE` 才是独立证据——把成员解出来跟源文件真比一次（前两者出自同一遍
读，自洽但不能证明与源一致）。smoke 产物由 `pack_progress.jsonl` 记账，全量跑直接续用。

## 验收判定行（起跑前写死，缺一条不算过）

```
HF_WHOAMI=HongzeFu
HYGIENE=PASS scanned=<n> hits=<n> ungated=0
BUCKET_CREATED=HongzeFu/robomme-4task-motion-400ep-20260904-v1 PRIVATE=False
BUCKET_FILES_AT_START=0 BUCKET_BYTES_AT_START=0
ANCHOR_OK=3
PACK_DONE shards=67 plain=49 members=229323 bytes=<n>
LOCAL_FILES=122 LOCAL_BYTES=<n>
PRE_LINES=121
BUCKET_FILES=122 BUCKET_BYTES=<n>          （与 LOCAL_* 逐字相等）
SHA256_PRE_POST_DIFF=0                     ★ 核心判据，覆盖全部 130 GB 字节
MEMBERS_SPOT=PASS tars=4 members=<n> mismatches=0
RESULT=PASS
SRC_UNCHANGED=OK
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK                        ★ env -u HF_TOKEN 匿名拉 README 成功
EXIT_CODE=0
```

## 已知坑（已写进脚本）

1. `hf auth whoami` 输出格式随 tty 而变（tmux 有 tty → 多行）。ckpt 那次因 `tail -1` 取到
   orgs 行导致首跑 `EXIT_CODE=1`。本脚本 `tr '\n' ' ' | tr -s ' '` 整体合并后 `case` 匹配。
2. buckets REST 统计**异步滞后**（480 GB 那次一度显示 312 GB / 200 文件；87 G 那次首查 0/0）。
   计数层首次不符 `sleep 120` 复查，且**计数层不单独作判据**，真判据在阶段 8。
3. `scripts/dataset/paths.sh` 会 export `HF_HUB_OFFLINE=1`。driver 不 source 它，但仍显式
   `unset`——不 unset 的话所有网络操作静默走缓存，看起来成功实际什么都没传。
4. stage 里不得有符号链接：上传端遍历不跟随符号链接**目录**，其内容会被静默整体漏传。
5. **禁止覆盖 `HOME`**（AGENTS 第 14 条），缓存变量逐项指向 `v1-store/cache/`。
6. `verify_model_repo.py` 写死 `info.private is not True → return 1`，那是给 model repo 用的、
   口径是必须保持 private。本轮是 bucket 且目标 public，阶段 2/10 的断言方向相反、各自显式、
   不共用函数——不要为了「统一」把它改回强制 private。
7. **绝对禁止 `tmux kill-server`**（AGENTS 第 7 条红线）。清理只用
   `tmux kill-session -t hf-ds400ep-export`，名字写全。
