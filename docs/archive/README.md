# docs/archive/ —— 环境 A 历史留档归档

> 2026-09-07 建立（commitV7.0）。本目录只收「环境 A（GreatLakes / turbo + 本机 2×RTX 6000 Ada，2026-09-03 及以前）」产出、且在环境 B（AWS 单机 8×A100，2026-09-04 起）已无对照价值或无法复现的 run / 数据集留档。**全部为 `git mv` 迁入，一个文件都没有删除**，`git log --follow` 可追到原路径。

## 归档口径

- **划线规则**：凡被真读写代码、Markdown 链接或现行 docs 正文引用的目录原地保留；只有「仅在冻结的根目录计划文件里以反引号文本出现、且属环境 A 的 v1 吞吐 / 确定性 / 中间对拍系」的目录才归档。
- **数字不混比**：归档内的吞吐、步时、util、存储介质数字全部属环境 A（turbo NFS 或本机 NVMe），按 `AGENTS.md` 第 13 条不得与环境 B 的数字并列比较。
- **产物不可得**：这些留档引用的 `v1-store/` 产物（`v1-prod-*` run、`4task-gl` 库、`env.json` 指纹、固化梯度数组）在环境 B 不存在，凡依赖它们的对拍口径按 `AGENTS.md` 第 13 条环境 B 段视作失效。
- **新 run 落点不变**：新的训练 / 评估 / 对拍 run 仍留档在 `docs/training-doc/<run_name>/`，数据集构建仍在 `docs/dataset-build-doc/<dataset_name>/`；`docs/archive/` 只进不出、不接收新 run。
- **空壳例外**：`eval-official-framesamp-modul` 是环境 B 目录，但只含一个 `download.sh`、没有 launch / result，从未跑成；按「未完成空壳」归档，不代表该评估已完成。
- **软引用有意保留**：根目录冻结的计划文件（`motion-memory-plan.md`、`v1-gradient-baseline.md`、`v1-95util.md`、`greatlakes.md` 等）与归档目录内部互相引用的反引号路径仍写着归档前位置，这些文件按仓库规则不再改动；查找时把 `docs/training-doc/<name>` 换成 `docs/archive/training-doc/<name>` 即可。Markdown 形式的链接已全部修正，断链为 0。

## 清单（training-doc 34 项 + dataset-build-doc 2 项）

| 原路径 | 新路径 | 组 | 归档理由 |
|---|---|---|---|
| `docs/training-doc/v1-gl-dlbench/` | `docs/archive/training-doc/v1-gl-dlbench/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-2gpu-epoch-bench-b8/` | `docs/archive/training-doc/v1-2gpu-epoch-bench-b8/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-coldcache-b8/` | `docs/archive/training-doc/v1-coldcache-b8/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-computeonly-b64/` | `docs/archive/training-doc/v1-computeonly-b64/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-e2e-b64/` | `docs/archive/training-doc/v1-e2e-b64/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-e2efix-w8c16/` | `docs/archive/training-doc/v1-e2efix-w8c16/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-e2efix-w12c16/` | `docs/archive/training-doc/v1-e2efix-w12c16/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-e2efix-w16c16/` | `docs/archive/training-doc/v1-e2efix-w16c16/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-g0-speed/` | `docs/archive/training-doc/v1-g0-speed/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-g0-speed-r2/` | `docs/archive/training-doc/v1-g0-speed-r2/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-g1-speed/` | `docs/archive/training-doc/v1-g1-speed/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-framesamp-cmp/` | `docs/archive/training-doc/v1-framesamp-cmp/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-framesamp-dl/` | `docs/archive/training-doc/v1-framesamp-dl/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-framesamp-e2e/` | `docs/archive/training-doc/v1-framesamp-e2e/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-prod-trend-10h/` | `docs/archive/training-doc/v1-prod-trend-10h/` | A | turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在 |
| `docs/training-doc/v1-det-d0-r1/` | `docs/archive/training-doc/v1-det-d0-r1/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d0-r2/` | `docs/archive/training-doc/v1-det-d0-r2/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d1-r1/` | `docs/archive/training-doc/v1-det-d1-r1/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d1-r2/` | `docs/archive/training-doc/v1-det-d1-r2/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d2-r1/` | `docs/archive/training-doc/v1-det-d2-r1/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d2-r2/` | `docs/archive/training-doc/v1-det-d2-r2/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d2cold-r1/` | `docs/archive/training-doc/v1-det-d2cold-r1/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-det-d2cold-r2/` | `docs/archive/training-doc/v1-det-d2cold-r2/` | B | D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用 |
| `docs/training-doc/v1-dtype-p3-dump-pre/` | `docs/archive/training-doc/v1-dtype-p3-dump-pre/` | C | P3 dump / P4 对拍 / 修复后 1000 步 A-B；结论固化进 `docs/v1-phase2-dtype-unify-report.md`；判据锚 `v1-dtype-p5-grad` 原地保留 |
| `docs/training-doc/v1-dtype-p4-cmp/` | `docs/archive/training-doc/v1-dtype-p4-cmp/` | C | P3 dump / P4 对拍 / 修复后 1000 步 A-B；结论固化进 `docs/v1-phase2-dtype-unify-report.md`；判据锚 `v1-dtype-p5-grad` 原地保留 |
| `docs/training-doc/v1-dtype-ab-post-r1/` | `docs/archive/training-doc/v1-dtype-ab-post-r1/` | C | P3 dump / P4 对拍 / 修复后 1000 步 A-B；结论固化进 `docs/v1-phase2-dtype-unify-report.md`；判据锚 `v1-dtype-p5-grad` 原地保留 |
| `docs/training-doc/v1-upstream-eq/` | `docs/archive/training-doc/v1-upstream-eq/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/v1-gl-entry-eq/` | `docs/archive/training-doc/v1-gl-entry-eq/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/v1-prod-60k/` | `docs/archive/training-doc/v1-prod-60k/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/v1-prod-100k/` | `docs/archive/training-doc/v1-prod-100k/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/v42-smoke5-n2/` | `docs/archive/training-doc/v42-smoke5-n2/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/v43-n3-grad/` | `docs/archive/training-doc/v43-n3-grad/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/v46-smoke5-preg3/` | `docs/archive/training-doc/v46-smoke5-preg3/` | D | 上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu` |
| `docs/training-doc/eval-official-framesamp-modul/` | `docs/archive/training-doc/eval-official-framesamp-modul/` | E | 环境 B 目录，但只有 `download.sh`、无 launch/result，从未跑成留档；作为空壳归档，若日后补跑请在 `docs/training-doc/` 下新建 run 目录 |
| `docs/dataset-build-doc/4task-gl-400ep/` | `docs/archive/dataset-build-doc/4task-gl-400ep/` | F | GreatLakes 8×1GPU 产出的 678 GB 原版 PKL+NPY 库（`v1-store/datasets/4task-gl`），环境 B 不可得；正文报告 `docs/v1-gl-dataset-consistency-report.md` 原地保留 |
| `docs/dataset-build-doc/framesamp-original-4task-400ep/` | `docs/archive/dataset-build-doc/framesamp-original-4task-400ep/` | F | 2026-08-23 已弃用档案（commit `d951aef` 脚本从未运行），仅作「此路不通」记录 |

## 分组说明

- **A 环境 A 吞吐 / 瓶颈基准**（15 项）：turbo NFS + 4×A40 / 本机 RTX 6000 Ada 的吞吐、util、epoch 外推与 worker/CPU 档位实测；AGENTS 第 13 条禁止与环境 B 混比，且所引 `v1-store` 产物在环境 B 不存在。
- **B 确定性四档预备实验**（8 项）：D0/D1/D2/D2-cold 各两轮 100 步，结论已固化进 `docs/v1-phase1-gradient-baseline-report.md` 九节；原始 records 只作复核用。
- **C dtype 统一中间产物**（3 项）：P3 dump / P4 对拍 / 修复后 1000 步 A-B；结论固化进 `docs/v1-phase2-dtype-unify-report.md`；判据锚 `v1-dtype-p5-grad` 原地保留。
- **D v5 入口重构对拍与 GreatLakes 正式 run / smoke**（7 项）：上游等价、集群入口等价、60k/100k 正式训练（turbo 产物）、V4.x smoke 与 n3 梯度短测；全部依赖 `/nfs/turbo` 或本机 `/data/hongzefu`。
- **E 未完成空壳（环境 B 例外）**（1 项）：环境 B 目录，但只有 `download.sh`、无 launch/result，从未跑成留档；作为空壳归档，若日后补跑请在 `docs/training-doc/` 下新建 run 目录。
- **F 数据集构建档案（环境 A）**（2 项）：`4task-gl-400ep`——GreatLakes 8×1GPU 产出的 678 GB 原版 PKL+NPY 库（`v1-store/datasets/4task-gl`），环境 B 不可得；正文报告 `docs/v1-gl-dataset-consistency-report.md` 原地保留；`framesamp-original-4task-400ep`——2026-08-23 已弃用档案（commit `d951aef` 脚本从未运行），仅作「此路不通」记录。

## 原地保留的环境 A 留档（对照用，不归档）

- 梯度 / 对拍判据锚：`docs/training-doc/v1-grad-baseline-g0/`、`v1-grad-baseline-g0b/`、`v1-dtype-p5-grad/`、`v1-framesamp-g2/`、`v1-postclean-g3/`、`v1-singlerun-g0/`、`v1-l0-gauge/`——被 `docs/v1-phase1-gradient-baseline-report.md` / `docs/v1-phase2-dtype-unify-report.md` 与 `scripts/training/g0/` 直接引用。
- motion 接入依据：`docs/training-doc/motion-*/` 8 项与 `docs/dataset-build-doc/4task-motion-40ep/`、`4task-gl-framesamp/`——`docs/motion-memory.md` 的实测数字出处。
- 环境 A 评估：`docs/training-doc/eval-3seed-context-vs-motion/` 等——三 seed 成功率对照的唯一来源，正文标注环境。

完整索引见 [`docs/README.md`](../README.md) 与 [`docs/training-doc/README.md`](../training-doc/README.md)。
