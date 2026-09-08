# RoboMME Policy Learning + MotionJEPA

## 当前状态（2026-09-07）

当前工作分支为 `v2-motionmem`，运行环境为 **环境 B**（AWS 单机 8×A100-80GB，仓库根
`/scratch/hongze/robomme_policy_learning_MotionJEPA`，无 GreatLakes / turbo 访问；判定方法与两套环境的判据表见
[`AGENTS.md`](AGENTS.md)「运行环境判定」一节）。

已完成的两条主线：

- **dataloader 已重构**：训练数据链路从原版 PKL + per-step NPY 改为 framesamp 三张连续大表（v2 → v3 破坏性单一化 → v5 入口统一），
  训练语义不变由 G0/G1/G2/G3 梯度对拍链证明。现行正本 [`docs/dataloader-restructure.md`](docs/dataloader-restructure.md)。
- **MotionJEPA motion token 已接入**：记忆区 512 帧路 token 与最多 96 个 motion token 按时刻稳定交错成 608 位，训练读离线表、推理由 sidecar 现算，
  两侧共用同一套排序与采样函数。现行正本 [`docs/motion-memory.md`](docs/motion-memory.md)。

生产 run `awsprod40k-b128-motion`（40k 步、global batch 128、400 ep 库）已在环境 B 跑完并评估；
与官方 `perceptual-framesamp-context` 的三 seed 成功率对照**无可辨别差异**（该三 seed 数字出自环境 A 的 2×RTX 6000 Ada，见
`docs/training-doc/eval-3seed-context-vs-motion/`）。针对「推理侧喂给模型的记忆是否与训练侧同一份」的逐层对拍见
[`docs/train-infer-consistency.md`](docs/train-infer-consistency.md)。

文档总入口：[`docs/README.md`](docs/README.md)。根目录的各 `*-plan.md` 是过程档案，与 `docs/` 正本冲突时以正本为准。

## 项目总体目标

本仓库以 MME-VLA 的 `perceptual-framesamp-context` 为起点，修改其训练数据链路并接入
[MotionJEPA](https://github.com/hongzefu/MotionJEPA) 生成的 motion token。两步都已落地（见「当前状态」）；
后续 scope（预算消融、SigLIP 编码器精度闭合、8 卡 200 集评估扩展）见 `docs/train-infer-consistency.md`「没做的事」一节，
**不代表当前实施授权**。

## v1 计划范围（历史）

`v1-dataloader-Restructure` 分支只负责重构 dataloader，目标是在不改变模型训练语义的前提下尽可能优化训练吞吐；
该分支的工作已合入 `v2-motionmem`。v1 起只关注以下四个任务，至今未变：

- `ButtonUnmask`
- `VideoUnmask`
- `ButtonUnmaskSwap`
- `VideoUnmaskSwap`

## 仓库位置（重要）

**环境 B（当前）**：仓库工作副本位于本地 NVMe RAID（`/dev/md0`，6.9 T）：

```text
/scratch/hongze/robomme_policy_learning_MotionJEPA
```

所有持久化文件只能落 `/scratch/hongze/` 下；原始 16 任务 H5 从公开集 `Yinpei/robomme_data_h5` 获取到
`/scratch/hongze/robomme_data_h5/`（sha256 清单见 `docs/dataset-build-doc/16task-h5-scan/`）。

**环境 A（历史，2026-09-03 及以前）**：仓库单副本在 NFS turbo
`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA`，本机 `/data/hongzefu` 保留原始 H5，
GreatLakes 计算节点通过 turbo 共享代码与数据；集群操作细节见根目录 [`greatlakes.md`](greatlakes.md)。
这两处路径在环境 B **不存在**，既有文档里出现它们的段落都是环境 A 历史，`AGENTS.md` 第 13、15 条禁止在环境 B 新建指向它们的引用。

## 项目内路径布局

除原始 H5 外，所有派生产物一律收敛到仓库内单一根 `v1-store/`（整体不进 git，环境 B 下全部是实体目录、无 symlink 外链）：

```text
v1-store/models/                         # OPENPI_DATA_HOME：SigLIP / PaliGemma tokenizer / pi05_base
v1-store/external/motionjepa/            # MotionJEPA encoder + wan_decoder（--encoder-run-dir 默认值）
v1-store/datasets/4task-motion-400ep/    # 四任务 × 100 ep 生产库：framesamp 三表 + wan latent + motion 表
v1-store/datasets/4task-motion-40ep/     # 40 ep 测试库（闸门与对拍脚本默认路径）
v1-store/train-assets/mme_vla_suite/     # norm stats（配置 mme_vla_suite_b128 绑定 robomme-400ep）
v1-store/train-runs/<config>/<run_name>/ # checkpoint 与 run 产物（motion_provenance.json 记四张表指纹）
v1-store/reports/                        # 对拍 / 审计 / 评估的 json 与日志
v1-store/cache/                          # uv / XDG / wandb / HF / JAX 缓存
v1-store/logs/                           # tmux 长任务日志
```

**不覆盖 `HOME`**——改为逐项显式设置缓存类环境变量指向 `v1-store/cache/`。环境 A 时期的 `v1-store/datasets/4task-gl/`、
`ref-*/`、`bench/` 等布局已随该环境退役，只在 `docs/archive/` 的留档里出现。

## 外部权重从哪拉

五个外部模型资产（SigLIP、PaliGemma tokenizer、pi05_base、Wan2.1 VAE、MotionJEPA encoder+decoder）的身份
钉死在 [`scripts/assets/ASSETS_LOCK.json`](scripts/assets/ASSETS_LOCK.json)（进 git，顶层自哈希防篡改）。
异地机器一条命令取齐并校验：

```bash
export HF_TOKEN=hf_…                                     # 仅私有的 MotionJEPA 权重需要
uv run python scripts/assets/fetch_assets.py plan        # 不联网，看缺什么
uv run python scripts/assets/fetch_assets.py fetch       # 取回后自动全量复校
uv run python scripts/assets/fetch_assets.py verify --level full   # 判定行 ASSETS=PASS
```

MotionJEPA 的 encoder + wan_decoder（run `wan-v8-filter10-72ep-a`、epoch 72）备份在 HuggingFace **private**
model repo [`HongzeFu/MotionJEPA`](https://huggingface.co/HongzeFu/MotionJEPA)，其余四项都能从公开 HF/GCS 取回。
模型本身的一切事实（结构、训练 commit、数据集、超参、数值合同、加载示例）以该 repo 的 model card 为准，
仓库内正本是
[`docs/dataset-build-doc/hf-export-motionjepa-encoder-v1/model-card.md`](docs/dataset-build-doc/hf-export-motionjepa-encoder-v1/model-card.md)；
上传过程与验收记录见同目录 `launch.md` / `result.md`。

ckpt sha256 必须是 `bae96037…c15a`，与本机 `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/` 那份相同
（后者是 `--encoder-run-dir` 的默认值，**不要用下载件覆盖它**）。

资产锁的设计原理、链路六处接入的改前改后、以及异地无 NFS 机器从零复刻的完整步骤，见
[`external-assets-lock.md`](external-assets-lock.md)。

环境 B（AWS 8×A100，2026-09-04）从零复刻 motion-memory 全部测试（≤100 步）与 4 任务 × 100 ep 完整库的完整过程——代码适配、判定行原文、A100 数字、事故与待裁决项——见
[`env-b-aws-replication.md`](env-b-aws-replication.md)。

## 固定入口

- **建库**：`scripts/dataset/`（SigLIP framesamp 三表打包 `pack_framesamp_store.py`、Wan latent 抽取 `wan/`、motion 表打包 `pack_motion_store.py`、本机多 GPU 调度 `run_local.py`）；逐段命令与判据见 [`docs/motion-memory.md`](docs/motion-memory.md) 第七章与 `docs/dataset-build-doc/4task-motion-400ep/launch.md`。
- **训练**：`scripts/training/train.py`（配置 `mme_vla_suite_b128`），生产命令见 `docs/training-doc/awsprod40k-b128-motion/launch.md`。
- **闸门与对拍**：`scripts/training/tests/motion_gates_model.py` / `motion_gates_online.py`（M/P/T 系闸门）、`scripts/training/g0/`（G0 梯度对拍、SigLIP 重放、训练/推理一致性对拍 `compare_train_infer_obs.py` 等）；对拍体系说明见 [`docs/train-infer-consistency.md`](docs/train-infer-consistency.md)。
- **评估**：主线不再维护评估链路，评估脚本在 `scripts/training/legacy-eval/`（含 `.local` / `.remote` 两套，见其 README）；主线 `examples/robomme/eval.py` 保持 4b7a710 版。单进程建第 28 个仿真环境必崩（Vulkan）的根因见 `docs/training-doc/tic-vulkan-makeenv/result.md`，commitV7.2 起评估启动已加 `GLIBC_TUNABLES` 修法。
- **motion 利用率评估**：`scripts/motion-variance/`（四条件闭环矩阵、开环逐层分析、donor bank、汇总出图；正本 [`docs/motion-utilization.md`](docs/motion-utilization.md)）。
- **GPU 利用率观测**：`scripts/training/util/`。

集群链路（`scripts/dataset/gl/`）已于 commitV6.2 删除，环境 A 的建库方案报告
[`docs/archive/v1-gl-dataset-consistency-report.md`](docs/archive/v1-gl-dataset-consistency-report.md) 保留为只读历史。

## 已弃用

commit `d951aef` 引入的 `scripts/v1_dataloader_restructure/` 与 `scripts/smoke_train_once.py`
**经判定不可靠，已删除**（从未实际运行过）。其定义的路径约定与固定入口一并作废，
勿从 git 历史里翻出重新采用。说明见
[`docs/archive/dataset-build-doc/framesamp-original-4task-400ep/README.md`](docs/archive/dataset-build-doc/framesamp-original-4task-400ep/README.md)。

## 存储介质与吞吐口径

- 环境 B 的一切训练、建库、评估都在本机 8×A100 上跑，本机数字即最终指标；吞吐基准必须记录介质「AWS 本地 NVMe RAID（`/dev/md0`）」与 batch size、worker 数、warmup 和稳态统计。
- 环境 A（本机 2×RTX 6000 Ada 与 turbo NFS）的历史数字**不与环境 B 混比**；相关留档已迁入 [`docs/archive/`](docs/archive/README.md)。

## 环境与协作约定

- Python 环境和命令统一使用 uv（`UV_LINK_MODE=copy uv run --no-sync …`）。
- 所有计划、文档、进度和总结使用简体中文。
- 详细工作规则见 [`AGENTS.md`](AGENTS.md)，Claude Code 独有机制见 [`CLAUDE.md`](CLAUDE.md)。
