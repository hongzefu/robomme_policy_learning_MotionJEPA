# docs/ 索引

> 环境 B（AWS 单机 8×A100，2026-09-04 起）现行。本目录顶层只剩三份正本与本索引；上游原文、图、环境 A 只读报告与历史 run 留档全部在 `archive/`；
> 根目录的各 `*-plan.md` 是过程档案，与本目录正本冲突时以正本为准（对照表见末节）。

## 一、正式正本（先读这三份）

| 文件 | 内容 |
|---|---|
| [`motion-memory.md`](motion-memory.md) | MotionJEPA motion token 接入 HistoryPi0 的现行正本：窗口口径、交错次序、三层代码、训练 / 推理链路逐跳、离线表与建库、闸门体系、生产 run 与评估、已知未知 |
| [`dataloader-restructure.md`](dataloader-restructure.md) | dataloader 四阶段重构的现行正本：三张连续大表、`store_meta.json` 契约、index 派生与读路径、dtype 统一、破坏性单一化、`train.py` 单跑、对拍体系与吞吐分环境 |
| [`train-infer-consistency.md`](train-infer-consistency.md) | 训练 / 推理一致性验证：结论与待拍板项、两侧调用链、六关结果、三个发现（编码器精度差、整段 vs 缓存数值差、测试集 prompt 训练未见）；Vulkan 第 28 次建环境崩溃的根因与修法见 `training-doc/tic-vulkan-makeenv/` |

## 二、已归档的上游原文与图（`archive/`，2026-09-08 迁入）

- 上游 MME-VLA 自带、未改：[`archive/docker_installation.md`](archive/docker_installation.md)、[`archive/manual_evaluation.md`](archive/manual_evaluation.md)、[`archive/training_curve_sample.md`](archive/training_curve_sample.md)、`archive/wandb.png`。
- 图（正本仍引用，链接已指向 archive）：[`archive/motion-memory-mask-axis.svg`](archive/motion-memory-mask-axis.svg)（记忆区 608 位 mask 轴与交错示意）、[`archive/motion-memory-online-timeline.svg`](archive/motion-memory-online-timeline.svg)（在线推理每批 16 帧的时刻线与 motion 窗）、[`archive/eval-success-by-task-length.png`](archive/eval-success-by-task-length.png)（四任务成功率对照图，三 seed，环境 A；由 `scripts/training/legacy-eval/plot_eval_success_by_length.py` 生成）。

## 三、已归档的环境 A 只读报告（`archive/`，数字不与环境 B 混比）

| 文件 | 内容 | 状态 |
|---|---|---|
| [`archive/v1-phase1-gradient-baseline-report.md`](archive/v1-phase1-gradient-baseline-report.md) | 训练确定性定档与梯度对拍黄金基线（G0 / G0b） | 只读，结论仍是对拍链依据 |
| [`archive/v1-phase2-dtype-unify-report.md`](archive/v1-phase2-dtype-unify-report.md) | dtype 统一修复与两块验证 | 只读，结论仍有效 |
| [`archive/v1-nfs-bottleneck-analysis.md`](archive/v1-nfs-bottleneck-analysis.md) | 4 卡 b64 的 NFS 瓶颈判定 | 只读历史（turbo 已退役） |
| [`archive/v1-gl-dataset-consistency-report.md`](archive/v1-gl-dataset-consistency-report.md) | GreatLakes 四任务建库与一致性验证 | 只读历史（链路已删） |
| [`archive/v1-gl-resource-tier-bench.md`](archive/v1-gl-resource-tier-bench.md) | 集群作业 CPU / mem 档位实测 | 只读历史 |

## 四、（并入第二、三节）

## 五、留档目录

- [`training-doc/`](training-doc/README.md)：训练 / 评估 / 对拍 run 留档，每个 run 一个目录（`launch.md` / `result.md` / `records/`）；索引按「环境 B 现行 / 环境 A motion 依据 / 环境 A 基线链 / 环境 A 评估 / 已归档」五档。
- [`dataset-build-doc/`](dataset-build-doc/README.md)：数据集构建与 HF 导出留档。

## 六、归档（`docs/archive/`）

环境 A 的吞吐 / 确定性 / dtype 中间产物 / v5 对拍与集群正式 run 留档共 36 项已于 2026-09-07（commitV7.0）`git mv` 入
[`archive/`](archive/README.md)，零删除；`git log --follow` 可追原路径。根目录冻结计划文件里的反引号路径仍写归档前位置，按下表重定向：

| 原路径（docs/ 下） | 现路径（docs/ 下） | 组 |
|---|---|---|
| `training-doc/v1-gl-dlbench/` | [`archive/training-doc/v1-gl-dlbench/`](archive/training-doc/v1-gl-dlbench/) | A |
| `training-doc/v1-2gpu-epoch-bench-b8/` | [`archive/training-doc/v1-2gpu-epoch-bench-b8/`](archive/training-doc/v1-2gpu-epoch-bench-b8/) | A |
| `training-doc/v1-coldcache-b8/` | [`archive/training-doc/v1-coldcache-b8/`](archive/training-doc/v1-coldcache-b8/) | A |
| `training-doc/v1-computeonly-b64/` | [`archive/training-doc/v1-computeonly-b64/`](archive/training-doc/v1-computeonly-b64/) | A |
| `training-doc/v1-e2e-b64/` | [`archive/training-doc/v1-e2e-b64/`](archive/training-doc/v1-e2e-b64/) | A |
| `training-doc/v1-e2efix-w8c16/` | [`archive/training-doc/v1-e2efix-w8c16/`](archive/training-doc/v1-e2efix-w8c16/) | A |
| `training-doc/v1-e2efix-w12c16/` | [`archive/training-doc/v1-e2efix-w12c16/`](archive/training-doc/v1-e2efix-w12c16/) | A |
| `training-doc/v1-e2efix-w16c16/` | [`archive/training-doc/v1-e2efix-w16c16/`](archive/training-doc/v1-e2efix-w16c16/) | A |
| `training-doc/v1-g0-speed/` | [`archive/training-doc/v1-g0-speed/`](archive/training-doc/v1-g0-speed/) | A |
| `training-doc/v1-g0-speed-r2/` | [`archive/training-doc/v1-g0-speed-r2/`](archive/training-doc/v1-g0-speed-r2/) | A |
| `training-doc/v1-g1-speed/` | [`archive/training-doc/v1-g1-speed/`](archive/training-doc/v1-g1-speed/) | A |
| `training-doc/v1-framesamp-cmp/` | [`archive/training-doc/v1-framesamp-cmp/`](archive/training-doc/v1-framesamp-cmp/) | A |
| `training-doc/v1-framesamp-dl/` | [`archive/training-doc/v1-framesamp-dl/`](archive/training-doc/v1-framesamp-dl/) | A |
| `training-doc/v1-framesamp-e2e/` | [`archive/training-doc/v1-framesamp-e2e/`](archive/training-doc/v1-framesamp-e2e/) | A |
| `training-doc/v1-prod-trend-10h/` | [`archive/training-doc/v1-prod-trend-10h/`](archive/training-doc/v1-prod-trend-10h/) | A |
| `training-doc/v1-det-d0-r1/` | [`archive/training-doc/v1-det-d0-r1/`](archive/training-doc/v1-det-d0-r1/) | B |
| `training-doc/v1-det-d0-r2/` | [`archive/training-doc/v1-det-d0-r2/`](archive/training-doc/v1-det-d0-r2/) | B |
| `training-doc/v1-det-d1-r1/` | [`archive/training-doc/v1-det-d1-r1/`](archive/training-doc/v1-det-d1-r1/) | B |
| `training-doc/v1-det-d1-r2/` | [`archive/training-doc/v1-det-d1-r2/`](archive/training-doc/v1-det-d1-r2/) | B |
| `training-doc/v1-det-d2-r1/` | [`archive/training-doc/v1-det-d2-r1/`](archive/training-doc/v1-det-d2-r1/) | B |
| `training-doc/v1-det-d2-r2/` | [`archive/training-doc/v1-det-d2-r2/`](archive/training-doc/v1-det-d2-r2/) | B |
| `training-doc/v1-det-d2cold-r1/` | [`archive/training-doc/v1-det-d2cold-r1/`](archive/training-doc/v1-det-d2cold-r1/) | B |
| `training-doc/v1-det-d2cold-r2/` | [`archive/training-doc/v1-det-d2cold-r2/`](archive/training-doc/v1-det-d2cold-r2/) | B |
| `training-doc/v1-dtype-p3-dump-pre/` | [`archive/training-doc/v1-dtype-p3-dump-pre/`](archive/training-doc/v1-dtype-p3-dump-pre/) | C |
| `training-doc/v1-dtype-p4-cmp/` | [`archive/training-doc/v1-dtype-p4-cmp/`](archive/training-doc/v1-dtype-p4-cmp/) | C |
| `training-doc/v1-dtype-ab-post-r1/` | [`archive/training-doc/v1-dtype-ab-post-r1/`](archive/training-doc/v1-dtype-ab-post-r1/) | C |
| `training-doc/v1-upstream-eq/` | [`archive/training-doc/v1-upstream-eq/`](archive/training-doc/v1-upstream-eq/) | D |
| `training-doc/v1-gl-entry-eq/` | [`archive/training-doc/v1-gl-entry-eq/`](archive/training-doc/v1-gl-entry-eq/) | D |
| `training-doc/v1-prod-60k/` | [`archive/training-doc/v1-prod-60k/`](archive/training-doc/v1-prod-60k/) | D |
| `training-doc/v1-prod-100k/` | [`archive/training-doc/v1-prod-100k/`](archive/training-doc/v1-prod-100k/) | D |
| `training-doc/v42-smoke5-n2/` | [`archive/training-doc/v42-smoke5-n2/`](archive/training-doc/v42-smoke5-n2/) | D |
| `training-doc/v43-n3-grad/` | [`archive/training-doc/v43-n3-grad/`](archive/training-doc/v43-n3-grad/) | D |
| `training-doc/v46-smoke5-preg3/` | [`archive/training-doc/v46-smoke5-preg3/`](archive/training-doc/v46-smoke5-preg3/) | D |
| `training-doc/eval-official-framesamp-modul/` | [`archive/training-doc/eval-official-framesamp-modul/`](archive/training-doc/eval-official-framesamp-modul/) | E |
| `dataset-build-doc/4task-gl-400ep/` | [`archive/dataset-build-doc/4task-gl-400ep/`](archive/dataset-build-doc/4task-gl-400ep/) | F |
| `dataset-build-doc/framesamp-original-4task-400ep/` | [`archive/dataset-build-doc/framesamp-original-4task-400ep/`](archive/dataset-build-doc/framesamp-original-4task-400ep/) | F |

## 七、根目录计划文件对照表

| 根目录文件 | 性质 | 对应正本 / 处置 |
|---|---|---|
| `motion-memory-plan.md`、`motion-memory-interleave.md` | motion 接入计划与交错设计（过程档案，冻结） | [`motion-memory.md`](motion-memory.md) |
| `v2-framesamp-restructure-plan.md`、`v3-destructive-restructure-plan.md`、`v5.0-train-entry-restructure-plan.md` | dataloader 三阶段计划（过程档案，冻结） | [`dataloader-restructure.md`](dataloader-restructure.md) |
| `v1-framesamp-restructure-plan.md`、`v1-framesamp-restructure-adversarial-review.md`、`v1-post-restructure-roadmap.md` | 已被 v2 取代的首版计划、对抗审查与 roadmap（三个加速项不立项） | 历史，只读 |
| `v1-gradient-baseline.md`、`v1-dtype-unify-plan.md`、`v1-95util.md` | 第一 / 二阶段与 util 计划（环境 A） | 报告见第三节（已归档） |
| `v5.1-prod-60k-wandb-plan.md` | GreatLakes 60k 正式训练计划（环境 A） | 留档已归档：`archive/training-doc/v1-prod-60k/` |
| `greatlakes.md` | 集群操作指引 | 环境 B 下只读存档（AGENTS 第 8 条） |
| `env-b-aws-replication.md` | 环境 B 从零复刻实录 | 现行，与本目录正本互补 |
| `external-assets-lock.md`、`HF-EXPORT-robomme-vla-motionjepa-v1.md` | 外部资产锁与 HF 发布实录 | 现行 |
