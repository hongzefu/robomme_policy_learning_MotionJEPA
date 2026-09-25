# 磁盘清理留档：第一档冗余副本（2026-09-25）

环境 B（AWS 8×A100，`/scratch` = `/dev/md0` 6.9T）。用户批准删除「第一档：冗余副本」。

## 效果

| 时点 | 已用 | 可用 | 占用率 |
|---|---|---|---|
| 删前 | 6.5T | 402G | 95% |
| 删后 | 5.6T | 1.3T | 82% |

## 已删除

| 路径 | 大小 | 可恢复来源 |
|---|---|---|
| `/scratch/hongze/robomme-4task-h5-20260912-v2/extracted`（1600 个 h5） | 740G | 同目录 `snapshot/` 的 14 个 `*.h5.tar.xz`（85G，含 `SHA256SUMS` 与 `control/verified_*.json`）本地解压即可；另有 HF bucket `HongzeFu/robomme-4task-h5-20260912-v2`。合并后的 `v1-store/raw-h5/4task-20260912-v2` 是独立实体副本，不受影响 |
| `/scratch/hongze/robomme_data_h5/*.h5.tar.xz`（16 个） | 53G | 同目录解压后的 16 个 `.h5` 保留；公开源 `Yinpei/robomme_data_h5` |
| `v1-store/exports/hf-dataset-4task-motion-400ep` | 115G | 上传日志 `全部完成`、`EXIT_CODE=0`、`PUBLIC_ANON_READ=OK`；HF bucket `HongzeFu/robomme-4task-motion-400ep-20260904-v1`。其中 49 个与 `v1-store/datasets/4task-motion-400ep` 的硬链接只减链接数，库本身完整（153G） |
| `v1-store/attic`（`400ep-attempt{1,2}-mixedcommit`） | 8G | 无需恢复（混合 commit 的废弃尝试） |

## 按预检结果保留

- `dataset-local-a100-4task-full1600/smoke28`（21G）：MotionJEPA `scripts/dataset-build/local_dataset_build.py` 断言 `smoke28/control/acceptance_post.json` 为 PASS，`local_dataset_policy.py` 以它为默认 smoke 根；删除会使建库流水线无法重跑，待用户另行裁决。

## 注意

- MotionJEPA `scripts/dataset-filter-vis/README.md` 的示例 `--source-root` 仍指向已删的 `extracted`，复用前需先从 `snapshot/` 解压恢复。
- 预检口径：逐项 `ls -ld` 为实体目录、目录内无 symlink、无进程占用（`lsof`）。

## 第二轮：HF 回读校验副本（2026-09-25）

用户目标：可用空间 ≥1.5T。第一轮之后 m1024 导出的回读校验又把 16 个 ckpt 从 HF 下载回 `verify/`，可用空间从 1.3T 降到 1138G。

| 时点 | 已用 | 可用 | 占用率 |
|---|---|---|---|
| 删前 | 5863G | 1138G | 84% |
| 删后 | 5332G | 1669G | 77% |

已删除 `v1-store/exports/hf-ckpt-v2-1600ep-{m16x8x8,m32x8x8,m64x8x8}-modul-b128-80k/verify/`（3×178G）。它们是从 HF bucket 回读、只用于 sha256 比对的副本，三次导出日志均为 `RESULT=PASS`、`EXIT_CODE=0`；删前核实共 1031 个文件无硬链接共享、无 symlink、无进程占用。`train-runs/` 下源 ckpt 删后仍各 16 步。

注意：`scripts/dataset/hf_export/run_m*_ckpt_export.sh` 每次导出都会留下约 178G 的 `verify/`，校验通过后应清理（脚本改动另立任务）。
