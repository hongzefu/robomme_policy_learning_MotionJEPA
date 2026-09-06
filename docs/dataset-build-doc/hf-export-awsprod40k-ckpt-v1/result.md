# hf-export-awsprod40k-ckpt-v1 · result

**终判 PASS**：HF bucket 与本机源字节级一致，源文件未被改动。

- 目标：`hf://buckets/HongzeFu/robomme-motionjepa-vla-v1`（private），起跑前 `files=0 size=0`。
- 内容：训练 run `awsprod40k-b128-motion` 的 8 个 orbax checkpoint（step 5000/10000/15000/
  20000/25000/30000/35000/39999）+ run 根 5 个元文件 + `README.md` + `SHA256SUMS.pre.txt`。
- 最终：**166 个文件 / 92,689,314,972 字节（86.3 GiB）**，与本地 stage 逐字节相等。
- 执行窗口：2026-09-06 06:46:13 → 06:56:07（EDT），**9 min 54 s**，tmux `hf-ckpt-export`。
- 起跑 commit：`b28f7d8`（clean HEAD）。driver：`scripts/dataset/hf_export/run_ckpt_export.sh`。

## 判定行（全部命中）

```
HF_WHOAMI=HongzeFu
LOCAL_FILES=166 LOCAL_BYTES=92689314972
PRE_LINES=165
BUCKET_FILES=166 BUCKET_BYTES=92689314972
SHA256_PRE_POST_DIFF=0
  sha256sum -c 校验行数: 165
RESULT=PASS
SRC_UNCHANGED=OK
EXIT_CODE=0
```

## 三层校验闭环

| 层 | 做法 | 结果 |
|---|---|---|
| 计数层 | buckets REST API 的 `totalFiles` / `size` 对 `find`/`%s` 求和 | 166 / 92,689,314,972 逐字节相等 |
| 回读层 | `hf sync hf://buckets/... verify/` 全量拉回 87 G 到空目录 | 完整落盘 |
| 内容层 | **上传前** 对 stage 算 `SHA256SUMS.pre.txt`（165 行）；**上传后** 在 `verify/` 从零重算 `SHA256SUMS.post.txt`（165 行）；两份 `diff` | **diff 为空**，另叠 `sha256sum -c --strict` 165 行全 OK |
| 收尾 | 源目录侧第三遍 sha256（164 行）与 pre 清单（剔除 stage 专有的 `README.md`）比对 | 一致，硬链接未改动原件 |

两份清单存于 `records/SHA256SUMS.{pre,post}.txt`，可离线复核到具体某一行。

## 分段耗时（由各清单文件 mtime 推断）

| 段 | 时刻 | 耗时 |
|---|---|---|
| 阶段 0–1（预检 + stage 硬链接 + 上传前 sha256） | 06:46:13 → 06:46:55 | 42 s |
| 阶段 2–4 + 上传后 sha256 | 06:46:55 → 06:51:52 | 4 min 57 s（含阶段 3 的 120 s 滞后等待） |
| 阶段 5 余下 `-c` 复核 + 阶段 6 源侧 sha256 | 06:51:52 → 06:56:07 | 4 min 15 s |

sha256 单遍 87 GB 约 42 s（`xargs -P 8`，NVMe RAID + SHA-NI，聚合约 2 GB/s）；
`sha256sum -c` 那道是单进程复核，故明显慢于并行的两遍。

## 插曲两则

1. **首跑停在阶段 0（`EXIT_CODE=1`，未上传任何字节）**。`hf auth whoami` 的输出格式随有无 tty
   而变：非 tty 是单行 `user=X orgs=Y`，tmux 的 tty 下是多行 `user: HongzeFu` / `  orgs: umich`。
   driver 原用 `tail -1` 取字段，在 tmux 里取到 orgs 行，把合法身份判成不符。已固化修复为整体
   合并后匹配（commit `b28f7d8`），bucket 未受影响。
2. **统计接口异步滞后复现**。阶段 3 首次查 buckets REST API 得 `files=0 bytes=0`，而此时
   `Sync completed.` 已打印。driver 内置的 120 s 复查后追平为 166 / 92,689,314,972
   （`updatedAt=2026-09-06T06:48:04Z`）。与上次 480 GB 导出时「一度显示 312 GB / 200 文件」
   同源，故计数层不能单独作为判据。

## 工具链偏差

`huggingface_hub` **1.30.0** 相对老脚本（`run_export.sh`）用的版本已重构命令：
`hf buckets sync` → 顶层 `hf sync`，`hf buckets info` → `hf repos ls`（只给 "92.7 GB" 这类
人类可读值，故本 driver 的精确字节改走 buckets REST API）。项目 `.venv` 的 0.32.3 由 openpi
依赖钉住、无 bucket 子命令，全程未动，新 CLI 由 `uvx --python 3.11` 独立拉取。

## 留底

- `v1-store/exports/hf-ckpt-awsprod40k-b128-motion/`
  - `stage/`：166 个文件，其中 164 个 ckpt 文件是源的**硬链接**（额外占盘 0）。
  - `verify/`：回读的 87 G 实体副本，**验收已完成，可整目录删除**。
  - `SHA256SUMS.src-after.txt`：源侧第三遍清单（164 行）。
- 本目录 `records/`：两份 sha256 清单 + 清洗后 driver 日志（49 行）。
