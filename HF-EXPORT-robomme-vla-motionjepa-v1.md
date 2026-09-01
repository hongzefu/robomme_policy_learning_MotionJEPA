# HF Bucket 导出实录：robomme-vla-motionjepa-v1（sha256 全量回读校验 PASS）

两个训练数据库已完整备份到 HuggingFace private bucket
**`hf://buckets/HongzeFu/robomme-vla-motionjepa-v1`**，端到端 sha256 校验闭环通过：
**480,205,065,887 字节 / 293 个文件，回读重算哈希与打包时清单逐行一致，`RESULT=PASS`。**
执行日期 2026-08-31 23:12 → 2026-09-01 01:55（EDT，总 2 小时 43 分）。

## 一、备份对象与体量（实测）

| 内容 | 源路径（turbo） | 实测体量 | bucket 内形态 |
|---|---|---|---|
| packed 库 | `v1-store/datasets/4task-gl-framesamp` | 36.9 GB（30 个 ~1GB part bin + 2 小 bin + meta 3 件） | `packed/4task-gl-framesamp/` 原样文件 |
| 原始库 data/ | `v1-store/datasets/4task-gl/data`（395,289 个 pkl，单个 ~395 KB） | 156.3 GB | `source/4task-gl/data_tars/` 97 个 tar（每片 4096 个、按 idx 升序） |
| 原始库 features/ | `v1-store/datasets/4task-gl/features`（1600 episode、逐 episode `token_emb_*.npy`+`kept_indices.json`） | 291.4 GB | `source/4task-gl/features_tars/` 144 个 tar（按 episode 分组 ~2GiB/片） |
| 原始库 meta/ | `v1-store/datasets/4task-gl/meta` | 10 个小 json | `source/4task-gl/meta/` 原样 |
| 合计 | — | **480.2 GB，tar 内成员 880,180 个** | 293 个文件 |

注：源库 `_claims/` 为空目录，未上传；features 体量为打包时逐文件 stat 实测（此前口径「原始库整体 ~156 GB」实际仅覆盖 data/）。

## 二、方案要点

1. **不走「裸复制到 /data 再打包」**：打包、哈希、上传至少要读数据 2–3 遍，而 395k+47万 个小文件在 NFS 上每遍都要付 open/close 元数据往返。实际方案是**单遍顺序读 NFS**：每个源文件只 `read()` 一次，同一份字节流同时喂给 sha256 与 tar/复制目标，产物直接落本机 NVMe `/data`，之后上传、回读、校验全在本机盘。
2. **小文件必须打 tar 分片**：直接传 88 万个小文件会退化成逐文件 HTTP 往返；tar 成员 `mtime=0`、uid/gid 归零、定序写入，分片可复现。
3. **断点续跑**：进度账本 `pack_progress.jsonl` 逐分片记录 sha256/字节数，重跑自动跳过已完成项；features 分组计划持久化（`features_plan.json`），续跑分组不漂移；`hf buckets sync` 本身按 size+mtime 增量，重跑同一命令即续传。
4. **三层 sha256**：逐源文件（880,180 行）、逐 tar 分片/平文件（241+47 行）、以及 packed 库两个小 bin 与 `store_meta.json` 内置 sha256 的交叉锚点（`pos_emb=3176ac09…`、`state_emb=8445ff9c…`，打包时核对一致）。

## 三、执行步骤与实测结果

工具链：`scripts/dataset/hf_export/`（`pack_and_hash.py` / `verify_download.py` / `run_export.sh` / `bucket_README.md`）。
起跑时仓库 HEAD `1d1de7c`（脚本为本轮新增，随导出后的 commit 入库）。driver 在 detached tmux `hf-export-v1` 内执行，日志 tee 到 `/data` 暂存区。

### 步骤 0：smoke 验证（132 秒，通过后产物直接续用）

```bash
uv run scripts/dataset/hf_export/pack_and_hash.py --workers 8 --limit-shards 2
```

- 打 2 个 data 分片 + 1 个 features 分片（共 11,379 个成员，5.16 GB）。
- 验收三项全过：解包后系统 `sha256sum -c` 独立复算 11,379 行全 OK；与 NFS 源逐字节对拍 0 不一致；两小 bin 与 store_meta 记录一致。
- 另用小文件试传确认 `hf buckets sync <目录> <bucket>` 语义是「目录内容 → bucket 根」（试传后已删，bucket 复归为空）。

### 步骤 1：打包+哈希（72.3 分钟）

```bash
uv run scripts/dataset/hf_export/pack_and_hash.py --workers 8
```

8 线程并发，NFS 单遍读 480 GB / ~88 万文件；smoke 已完成的 3 片自动跳过。
产出 241 个 tar + 47 个平文件 + 三份清单（`sha256-shards.txt` 241 行、`sha256-packed.txt` 47 行、`sha256-source-files.txt` 880,180 行）+ `manifest/upload_manifest.json`。

### 步骤 2：上传（约 45 分钟，聚合峰值 ~230 MB/s）

```bash
hf buckets sync /data/.../stage hf://buckets/HongzeFu/robomme-vla-motionjepa-v1
```

- 插曲一：driver 把 `HF_HOME` 指到 /data 后 CLI 找不到默认位置的登录 token，首次尝试 401；把 `~/.cache/huggingface/{token,stored_tokens}` 复制进新 `HF_HOME` 后，60 秒自动重试即成功（`run_export.sh` 已固化该步）。
- 插曲二：sync 成功后 `hf buckets info` 一度显示 312 GB / 200 文件（统计接口异步滞后）。用上传方向 `--dry-run` 核实：**293/293 全部 `skip: identical`、`uploads: 0`**，确认无缺文件；稍后 info 追平为 480,205,065,887 / 293。

### 步骤 3：全量回读（约 44 分钟）

```bash
hf buckets sync hf://buckets/HongzeFu/robomme-vla-motionjepa-v1 /data/.../verify
```

480 GB 完整下载到本机 `verify/`，293 个文件。

### 步骤 4：sha256 校验（约 2 分钟，PASS）

```bash
uv run scripts/dataset/hf_export/verify_download.py --workers 6
```

- 文件集合一致：stage 与 verify 均 293 个文件；
- 清单/说明类小文件逐字节一致；
- `sha256sum -c` 对回读副本重算全部 288 行清单（241 分片 + 47 平文件，覆盖全部 480 GB 字节），6 路并行，**失败 0**；
- 终判：`RESULT=PASS`，driver `EXIT_CODE=0`。

### 步骤 5：清理与留底

- `/data/hongzefu/hf-export-robomme-vla-motionjepa-v1/`（stage + verify + hf-home ≈ 960 GB 临时暂存）已整体删除，/data 恢复 3.0 T 可用。
- 三份 sha256 清单、`upload_manifest.json`、`features_plan.json`、`pack_progress.jsonl`、清洗后的 run/smoke 日志留底于
  `v1-store/exports/hf-export-robomme-vla-motionjepa-v1/`（约 65 MB）；bucket 内 `checksums/` 也带同一套清单。
- docs 留档：`docs/dataset-build-doc/hf-export-robomme-vla-motionjepa-v1/`。

## 四、以后如何下载与自验

bucket 根有 `README.md`（内容同 `scripts/dataset/hf_export/bucket_README.md`），核心三步：

```bash
hf buckets sync hf://buckets/HongzeFu/robomme-vla-motionjepa-v1 ./robomme-vla-motionjepa-v1
cd robomme-vla-motionjepa-v1
sha256sum -c checksums/sha256-shards.txt && sha256sum -c checksums/sha256-packed.txt
```

解包 `source/4task-gl/{data_tars,features_tars}/*.tar` 即还原 `data/`、`features/` 原目录结构，解包后可用
`checksums/sha256-source-files.txt`（路径相对 `source/4task-gl/`）做逐源文件自验；
`packed/4task-gl-framesamp/` 无需解包即训练可读 store（异地使用时以 `MMEVLA_FRAMESAMP_SOURCE` 指向还原出的 `4task-gl`）。
