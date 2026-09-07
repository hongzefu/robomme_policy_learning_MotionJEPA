# 数据集构建档案索引

> 环境 B（AWS 单机 8×A100，2026-09-04 起）现行。环境 A（GreatLakes / turbo）时期的两份档案已于 2026-09-07 迁入
> [`docs/archive/dataset-build-doc/`](../archive/dataset-build-doc/)（见 [`docs/archive/README.md`](../archive/README.md)）。
> 库本体一律在 `v1-store/datasets/<name>/`（不进 git）；本目录只放 launch / result / records。
> 数据集格式与建库链路的正文见 [`docs/motion-memory.md`](../motion-memory.md)（motion 表）与
> [`docs/dataloader-restructure.md`](../dataloader-restructure.md)（framesamp 三表）。

## 环境 B 现行库（训练 / 对拍实际使用）

| 档案 | 库 | 内容 | 状态 |
|---|---|---|---|
| [`4task-motion-400ep/`](4task-motion-400ep/launch.md) | `v1-store/datasets/4task-motion-400ep` | 四任务 × 100 episode 全量：SigLIP framesamp 三表 + Wan latent + motion token 表；生产 run `awsprod40k-b128-motion` 与本轮 `tic-*` 对拍所用 | 已完成（[result](4task-motion-400ep/result.md)） |
| [`4task-motion-40ep-aws/`](4task-motion-40ep-aws/launch.md) | `v1-store/datasets/4task-motion-40ep` | 40 ep 测试库在环境 B 的从零复刻，测试脚本默认路径；M1–M5 / T3 等闸门在此库跑 | 已完成（[result](4task-motion-40ep-aws/result.md)） |
| [`16task-h5-scan/`](16task-h5-scan/launch.md) | `/scratch/hongze/robomme_data_h5/` | 公开集 `Yinpei/robomme_data_h5` 16 任务 H5 的下载与扫描（episode 数、帧数、sha256 清单） | 已完成（[result](16task-h5-scan/result.md)） |

## 环境 A 时期的建库依据（原地保留，数字不与环境 B 混比）

| 档案 | 内容 | 为什么保留 |
|---|---|---|
| [`4task-motion-40ep/`](4task-motion-40ep/launch.md) | motion 40 ep 测试库的首次构建（S0 先验 / oracle、S1 重抽与建库） | `docs/motion-memory.md` 窗口口径与先验数字的出处 |
| [`4task-gl-framesamp/`](4task-gl-framesamp/README.md) | framesamp 三表全量打包（v2 计划 S4） | `docs/dataloader-restructure.md` 三表契约与打包判据的出处 |

## HF 导出（环境 B）

| 档案 | 内容 |
|---|---|
| [`hf-export-awsprod40k-ckpt-v1/`](hf-export-awsprod40k-ckpt-v1/launch.md) | `awsprod40k-b128-motion/39999` checkpoint 上传 |
| [`hf-export-motionjepa-encoder-v1/`](hf-export-motionjepa-encoder-v1/launch.md) | MotionJEPA encoder + decoder 上传（含 [model card](hf-export-motionjepa-encoder-v1/model-card.md)） |
| [`hf-export-robomme-vla-motionjepa-v1/`](hf-export-robomme-vla-motionjepa-v1/launch.md) | 整体 VLA + motion 发布仓；实录在根目录 `HF-EXPORT-robomme-vla-motionjepa-v1.md` |

## 已归档（环境 A，`docs/archive/dataset-build-doc/`）

| 档案 | 处置 |
|---|---|
| [`4task-gl-400ep/`](../archive/dataset-build-doc/4task-gl-400ep/README.md) | GreatLakes 8×1GPU 产出的 678 GB 原版库，环境 B 不可得；正文报告 [`docs/v1-gl-dataset-consistency-report.md`](../v1-gl-dataset-consistency-report.md) 原地保留 |
| [`framesamp-original-4task-400ep/`](../archive/dataset-build-doc/framesamp-original-4task-400ep/README.md) | 2026-08-23 已弃用档案，仅作「此路不通」记录 |
