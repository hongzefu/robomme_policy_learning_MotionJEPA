# RoboMME Policy Learning + MotionJEPA

## 当前状态

当前工作分支为 `v1-dataloader-Restructure`。

本轮正在做的是：把仓库整体迁到 NFS turbo、弃用上一版不可靠的本地链路、
把四任务全量数据处理改到 GreatLakes 上以 8×1GPU job array 完成，并给出集群产物与
本地产物的分层一致性验证。dataloader 本身**尚未**重构，MotionJEPA motion token **尚未**接入。

## 项目总体目标

本仓库以 MME-VLA 的 `perceptual-framesamp-context` 为起点，后续将修改该链路并接入
[MotionJEPA](https://github.com/hongzefu/MotionJEPA) 生成的 motion token。

## v1 计划范围

`v1-dataloader-Restructure` 只负责重构 dataloader，目标是在不改变模型训练语义的前提下
尽可能优化训练吞吐。v1 只关注以下四个任务：

- `ButtonUnmask`
- `VideoUnmask`
- `ButtonUnmaskSwap`
- `VideoUnmaskSwap`

## 仓库位置（重要）

**仓库单副本位于 NFS turbo**，本机不再保留任何副本：

```text
/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
```

这样本机与 GreatLakes 计算节点看到的是同一份代码、同一个 venv、同一份数据，
不存在两侧同步问题。计算节点唯一可见的共享路径就是 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/`，
详见根目录 [`greatlakes.md`](greatlakes.md)。

**本机 `/data/hongzefu` 只保留最初的全局原始 H5**：

```text
/data/hongzefu/robomme_data_h5_v2_4env400ep     # 401 GB 级原件，永久保留
```

集群作业期间会在 turbo 上放一份同源副本（逐文件 sha256 核对）供计算节点读取；
**该副本是临时暂存，全流程验收通过后删除**。

## 项目内路径布局

除最初的全局原始 H5 外，所有派生产物一律收敛到仓库内单一根 `v1-store/`：

```text
v1-store/models/               # OPENPI_DATA_HOME：SigLIP / PaliGemma tokenizer / pi05_base
v1-store/datasets/4task-gl/    # GreatLakes 8×1GPU 全量产物（1600 episodes）
v1-store/datasets/ref-*/       # 一致性验证用的本地对照产物
v1-store/train-assets/         # norm stats（--assets-base-dir）
v1-store/train-runs/           # checkpoint 与 smoke run 目录（--checkpoint-base-dir）
v1-store/cache/                # uv / XDG / wandb / HF / JAX 缓存
v1-store/logs/                 # sbatch --output 与本地 tmux 日志
v1-store/bench/                # CPU/mem 档位实测采样与对照表
```

`v1-store/` 整体不进 git。**不覆盖 `HOME`**——覆盖会让 ssh 找不到 `~/.ssh/config` 与
ControlMaster socket、直接打断集群提交；改为逐项显式设置缓存类环境变量。

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

## 固定入口

本机数据处理链路（SigLIP / Wan VAE / MotionJEPA encoder 三阶段）全部在 `scripts/dataset/`（集群链路已于 v2-motionmem 删除），本地 G0 对拍量具在 `scripts/training/g0/`，GPU 利用率观测族在 `scripts/training/util/`。
逐段命令、续跑口径与全部实测数字见方案报告：

- [`docs/v1-gl-dataset-consistency-report.md`](docs/v1-gl-dataset-consistency-report.md)

## 已弃用

commit `d951aef` 引入的 `scripts/v1_dataloader_restructure/` 与 `scripts/smoke_train_once.py`
**经判定不可靠，已删除**（从未实际运行过）。其定义的路径约定与固定入口一并作废，
勿从 git 历史里翻出重新采用。说明见
[`docs/dataset-build-doc/framesamp-original-4task-400ep/README.md`](docs/dataset-build-doc/framesamp-original-4task-400ep/README.md)。

## 本地与 NFS

- 本机 2× RTX 6000 Ada 只用于：一致性验证的本地对照产物、CPU/mem 档位实测、功能性 smoke run。
- 本机得到的 batch latency 或 samples/s **不作为吞吐结论**。
- 正式吞吐结论在 NFS 数据副本上测，并记录数据位置、batch size、worker 数、warmup 与稳定态统计。

## 环境与协作约定

- Python 环境和命令统一使用 uv；NFS 上执行 uv 操作必须 `UV_LINK_MODE=copy`。
- 所有计划、文档、进度和总结使用简体中文。
- 详细工作规则见 [`AGENTS.md`](AGENTS.md)，Claude Code 独有机制见 [`CLAUDE.md`](CLAUDE.md)。
