# 四对运动 token 训练链路冒烟

本次把 v10 的原始 `8×256` 运动表示接入训练 loader，在唯一入口固定相邻拼接为 `4×512`。完整模型仍使用原有 modulation 和 RoPE；160 窗形成 640 个 motion token，加 512 个 frame token，总容量 1152。验收只覆盖单窗真实提取、一次训练更新及模型权重严格回读。

用户指令原话：“/scratch/hongze/MotionJEPA/1020-paired4-modulation-plan.html 开始实施 有问题问用户”。实施依据为 MotionJEPA 的 [获批计划](../../../../MotionJEPA/1020-paired4-modulation-plan.html)，计划提交 `ee719fbedea88f8a6c8e6a0c280fd332c0a36af9`。`PLAN_CONSISTENCY_REVIEW=PASS base=ee719fbedea88f8a6c8e6a0c280fd332c0a36af9 findings=0 blocking=0`。

## 版本、环境与分工

实施前 policy 代码为 `84cef6af56df77eceef6f6f685a9e9c3d865b337`；规则同步后基线为 `2b2954cf7d04f10f1b0826a3e72a0324bdda08d2`，分支 `v2-motionmem`。本文件建立时尚未启动 GPU；最终运行代码为 `4c83a702878ef6c601e30a5103d02e1cbb1fe8f9`，结果及首次初始化修复见 [result.md](result.md)。

环境为 AWS 单机，`/scratch` 位于 `/dev/md0` 的 XFS 本地 NVMe RAID；8 张 A100-SXM4-80GB。开工时八卡显存使用均为 0 MiB，磁盘空闲约 3.4 TiB。不存在历史 `/data/hongzefu`、NFS 或集群连接。主环境和 Wan 子环境分开，保持根依赖锁不变。

全部子代理使用 `gpt-6-astra / ultra`。四个独立工作树分别负责：T1 离线 source、pack/store 与唯一配对；T2a 训练 loader 与 YAML；T1b 33 帧夹具与单窗提取；T2b CPU/模型检查入口。主会话负责共同接口、整合、资源、运行、证据与清理；独立代理审查固定提交和运行结果。

## 数据与固定参数

| 项 | 固定值 |
|---|---|
| 父 VLA 库 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f` |
| 父 manifest SHA | `d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad` |
| 真实 RGB | `/scratch/hongze/dataset-local-a100-16task-mixed8-200/data-raw/BinFill_ep0/video_exec.h5` 的 `frames[0:33]` |
| 原 encoder | `/scratch/hongze/MotionJEPA/runs/wan-mixed8-200-v10-inseq256q8-b352x8-trend-20261020-a/checkpoint_final.pt`，epoch6/step1884，EMA |
| checkpoint SHA | `146efd425f7923428d630dfb7ee2a8f010f19b207d36519dba49454aaa373d30`，开工现场完整复算一致 |
| 注册 | 独立 `v1-store/external/motionjepa/<原run名>/checkpoint_epoch_6.pt` 原样副本，config 同源 |
| 统计 | `v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f/robomme/norm_stats.json` |
| 模型 | `gemma_2b + gemma_300m`，`pi05=true`，horizon20，batch1，seed42，num_workers0，原 optimizer/LR 日程 |
| 输入 | 正常 loader 按顺序消费 33 批，更新 idx32；有效 motion 为 4，保留父 PKL32 原动作序列 |
| 输出根 | `v1-store/smoke/paired4-epoch6-20261020-a/`，开工前不存在 |
| 预算 | GPU 阶段累计最多 30 分钟；Torch 使用 GPU1 并退出后，JAX 使用 GPU0；到限保留证据并报告未完成 |

## 执行入口

从 policy 主仓根执行，以下命令使用完整整合后的源码。先运行 CPU 与 prepare，再运行单窗提取、finalize、模型和 summary。GPU 前置检查通过前不启动 GPU 阶段；Wan 依赖更新须先按提交的子项目 lock 同步既有专用环境。

```bash
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
source scripts/training/paths.sh
HC=perceptual-framesamp-modul-pair4-epoch6-original-smoke.yaml
SMOKE_ROOT="$PWD/v1-store/smoke/paired4-epoch6-20261020-a"
CHECK=scripts/training/tests/check_pair4_contract.py
ENCODER_RUN="$PWD/v1-store/external/motionjepa/wan-mixed8-200-v10-inseq256q8-b352x8-trend-20261020-a"
ENCODER_SHA=146efd425f7923428d630dfb7ee2a8f010f19b207d36519dba49454aaa373d30
JAX_PLATFORMS=cpu uv run --frozen --no-sync python "$CHECK" --suite cpu --root "$SMOKE_ROOT"
JAX_PLATFORMS=cpu uv run --frozen --no-sync python "$CHECK" --suite prepare-fixture --root "$SMOKE_ROOT"
CUDA_VISIBLE_DEVICES=1 UV_PROJECT_ENVIRONMENT="$PWD/v1-store/venvs/wan" uv run --project scripts/dataset/wan --frozen --no-sync python scripts/training/tests/pair4_source_smoke.py --root "$SMOKE_ROOT" --max-windows 1 --encoder-run-dir "$ENCODER_RUN" --checkpoint checkpoint_epoch_6.pt --expected-ckpt-sha256 "$ENCODER_SHA"
JAX_PLATFORMS=cpu uv run --frozen --no-sync python "$CHECK" --suite finalize-fixture --root "$SMOKE_ROOT"
CUDA_VISIBLE_DEVICES=0 uv run --frozen --no-sync python "$CHECK" --suite model --root "$SMOKE_ROOT" --history-config "$HC" --updates 1
JAX_PLATFORMS=cpu uv run --frozen --no-sync python "$CHECK" --suite summary --root "$SMOKE_ROOT"
```

长阶段放入 detached tmux，计划会话名 `paired4-epoch6-original-smoke-20261020-a`，日志 `reports/smoke.log`；实际创建的会话与精确 PID 写入运行记录。使用 `PYTHONUNBUFFERED=1`、`set -o pipefail`、`tee` 和 `EXIT_CODE=` 保证日志与退出状态一致。只清理本次登记的进程。

实际使用上述唯一tmux，原尝试wrapper PID为350702，成功重试wrapper PID为352601、pane PID为352599、Torch PID为352652、模型PID为353235；全部已退出。运行脚本保留在产物根 `reports/run_gpu.sh`，使用首次起点1792549237和固定1790秒计算截止，另留10秒终止余量，重试不重置。美东22:30:40正常结束。Wan依赖同步使用以下命令级本地Git传输，固定远端URL/revision保持：

```bash
UV_CACHE_DIR=/scratch/hongze/.cache/uv \
UV_PROJECT_ENVIRONMENT="$PWD/v1-store/venvs/wan" \
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.file:///scratch/hongze/MotionJEPA.insteadOf \
GIT_CONFIG_VALUE_0=https://github.com/hongzefu/MotionJEPA \
uv sync --project scripts/dataset/wan --frozen
```

## 验收与停止点

必须分别记录 `RAW_CONTRACT`、`PAIRING`、`WINDOW_BUDGET`、`MASK_ORDER`、`SOURCE_FRESH`、`REGISTERED_SOURCE`、`ORIGINAL_IMPL_UNCHANGED`、`SCOPE`、`REAL_FORWARD`、`SAVE_LOAD`。`TRAIN_SMOKE=PASS` 只在所有必需项通过后成立，未运行项保留 `NOT_RUN`。

原始 RGB 与父 PKL 33/33 逐位一致；raw8 与正常 pack 读回、配对逆变换逐位一致；有效 motion 输入梯度、投影梯度均非零，至少一项 fp32 motion 参数更新。保存用原生 `save_state`，等待异步完成，按实际 EMA/live 导出参数严格回读原 dtype，固定 batch/RNG 调用原 `compute_loss` 的差异必须为零。该验收不证明完整训练状态可恢复。

完成后停止并留档。在线、eval、benchmark、全量建库、缓存迁移与正式训练均不在范围；未来 VAE 缓存兼容及最终 encoder 需另行核验。
