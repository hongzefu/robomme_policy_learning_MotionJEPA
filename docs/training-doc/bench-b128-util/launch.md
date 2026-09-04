# bench-b128-util — 起跑记录

## 目的

在环境 B（AWS 单机 8×A100-80GB）上，为正式 run `awsprod40k-b128-motion` 起跑前钉死三个未知数：

1. **per-device 32 是否 OOM**。global batch 128 / 4 卡 = per-device 32。官方唯一跑过 batch 128 的是
   `pi05_baseline`（`use_history=False`），本档是 history + motion，显存更高。`fsdp_devices=4` 在 4 卡上
   已是 mesh `(1, 4)` 全量分片、最省显存的配置，没有更省的档可退。
2. **worker 能否吃满 GPU**。环境 A 教训：`v1-e2e-b64`（8 核 / 4 worker）util **中位数 100% 但均值仅
   69–70%**、6.933 s/step；提到 16 核 / 8 worker 后 5.301 s/step；`v1-l0-gauge` 的 S0 锚在 w8c16 下
   4.756 s/step、util 99.718%。判读按 AGENTS.md 第 16 条：均值 / 0% 占比 / 分层均值，禁用中位数。
3. **wandb 凭据与 entity**（已先行完成，见下）。

## 环境

- 环境 B（AWS 单机）。仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。
- 8 × NVIDIA A100-SXM4-80GB，96 vCPU，1121 GB RAM。
- 存储介质：**AWS 本地 NVMe RAID（`/dev/md0`，6.9 T）**。本文所有吞吐数字仅在此介质口径下有效，
  **不得与环境 A 的 A40 + turbo NFS、或 RTX 6000 Ada 数字混比**（AGENTS.md 第 13 条）。

## commit

起跑 commit：`2b3f4c2`（clean HEAD，已 push 到 origin/v2-motionmem）。
- `abe7b6b` commitV6.13：新增 `mme_vla_suite_b128` 条目
- `2b3f4c2` fix：`project_name` 改为 `robomme-framesamp`

## 配置档位

config 条目 `mme_vla_suite_b128`（`src/mme_vla_suite/training/config.py`）实测生效值：

```
batch_size     = 128            (官方 mme_vla_suite = 64)
num_train_steps= 40000          (官方 80000)   ← 基准 run 用 --num-train-steps=300 覆盖
lr_schedule    = CosineDecaySchedule(warmup_steps=5000, peak_lr=1e-4, decay_steps=50000, decay_lr=1e-4)
官方 lr        = CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-5, decay_steps=100000, decay_lr=5e-5)
optimizer      = AdamW(b1=0.9, b2=0.95, eps=1e-08, weight_decay=1e-10, clip_gradient_norm=1.0)  ← 与官方逐项相同
num_workers    = 8    fsdp_devices = 4    ema_decay = 0.999    seed = 42
save/keep      = 5000 / 5000
project_name   = robomme-framesamp
assets         = AssetsConfig(assets_dir='v1-store/train-assets/mme_vla_suite/robomme-400ep', asset_id='robomme')
```

## 数据来源

- 数据集：`v1-store/datasets/4task-motion-400ep/framesamp`（400 ep / 123,044 timestep / 101,066 exec 样本，
  `status=verified`，manifest sha256 `92fa17e9…`）
- motion store：`v1-store/datasets/4task-motion-400ep/motion`，经 `MMEVLA_MOTION_STORE` 覆盖 YAML 里
  硬写的 40ep 路径。覆盖点 `src/mme_vla_suite/training/dataloader.py::_motion_gates`（env 优先）。
  漏设不会静默串数据——`datastore/motion_store.py::check_same_source` 比对 manifest sha256 后 fail-loud。
- norm_stats：`v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`
- 预训练权重：`v1-store/models/openpi-assets/checkpoints/pi05_base/params`（12 G，实体目录）

## B1 命令

tmux 会话 `bench-b1`（本轮起过的会话清单：`bench-b1`。用户自有会话 `0`、`1`、`claude-private` 一律不动）。

```bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-400ep/motion
uv run scripts/training/train.py mme_vla_suite_b128 \
  --exp-name=bench-b1-motion \
  --num-train-steps=300 \
  --log-interval=10 \
  --no-wandb-enabled \
  --checkpoint-base-dir=v1-store/train-runs \
  --dataset-path=v1-store/datasets/4task-motion-400ep/framesamp \
  --model.use_history \
  --model.history_config=perceptual-framesamp-context-motion.yaml 2>&1 | tee -a "$LOG"
EC=${PIPESTATUS[0]}; echo "EXIT_CODE=$EC" | tee -a "$LOG"
```

GPU util 采样（AGENTS.md 第 16 条，500 ms 密集采样）：

```bash
timeout 3000 nvidia-smi -lms 500 \
  --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits > records/b1_gpu_util.csv
```

## 输出路径

- 日志：`docs/training-doc/bench-b128-util/records/b1_run.txt`
- GPU 采样：`docs/training-doc/bench-b128-util/records/b1_gpu_util.csv`
- checkpoint（跑完即删）：`v1-store/train-runs/mme_vla_suite_b128/bench-b1-motion/`

## wandb 探针结论（先于 B1 完成）

判定行：`WANDB_PROBE=PASS entity=hongzefu-university-of-michigan project=robomme-framesamp`
（探针 run 建后即 `wandb.Api().run(...).delete()` 删除，未留残迹）

两条实测，均已写入 commit `2b3f4c2` 的 body：

1. **entity 必须写 team 名**。写 organization 名 `hongzefu-university-of-michigan-org` 被拒：
   `400 "you may not log runs directly to your organization, please try using your team entity"`。
2. **新格式 API key 在当前锁定版本下可用，但不能走 `wandb.login()`**。本次是 wandb 新格式 key
   （`wandb_v1_` 前缀，86 字符），而锁定的 wandb 0.19.11 里 `wandb.login()` 硬校验 40 字符会
   `ValueError: API key must be 40 characters long, yours was 86`。`scripts/training/train.py` 链路
   **不调用 `wandb.login()`**（已 grep 确认），仅靠 `WANDB_API_KEY` 环境变量取凭据，该路径对新格式
   key 正常工作。**因此不需要升级 wandb、不动 uv.lock。**

凭据落点 `v1-store/secrets/wandb.env`（权限 600，`v1-store/` 整体不进 git）。
**API key 不出现在本留档、commit、日志与 `records/` 任何文件中。**

## 判定门

`OOM=NONE`、`UTIL_MEAN>=95%`（或已确认瓶颈不在 worker）、`WANDB_PROBE=PASS`。
三条全过后**仍不得自行起正式 run**——须先向用户报告 util 三口径、s/step、40k step 预计总墙钟、
显存峰值余量，等用户确认。

## 起跑事故与修正（2026-09-04 17:56）

**首次启动失败**，`EXIT_CODE=1`，日志留存于 `records/b1_run_fail_openpi_data_home.txt`：

```
FileNotFoundError: File not found at ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
  src/openpi/shared/download.py:54 maybe_download
  ← src/openpi/training/weight_loaders.py:52 CheckpointWeightLoader.load
  ← scripts/training/train.py:244 init_train_state
```

**原因**：启动脚本只显式设了 `CUDA_VISIBLE_DEVICES` / `XLA_PYTHON_CLIENT_MEM_FRACTION` /
`MMEVLA_MOTION_STORE` / `PYTHONUNBUFFERED`，**漏设 `OPENPI_DATA_HOME`**。
`src/mme_vla_suite/training/config.py` 顶层的
`OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")` 于是取默认值，权重路径被解析到
`~/.cache/openpi/openpi-assets/...`。该目录在本机存在（`drwxrwxrwx`，建于 2026-08-07）但**不含
`openpi-assets/`**，属"目录在、内容不在"，因此不是被 symlink 兜住的情形，直接 fail。
实际权重在 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`。

**修正**：启动脚本改为 source 仓库唯一路径源 `scripts/dataset/paths.sh`，与
`scripts/training/prod/gl_train_prod.sbatch`（`export OPENPI_DATA_HOME="$STORE/models"`）、
`scripts/dataset/run_local.py`（`"OPENPI_DATA_HOME": str(V1_STORE / "models")`）同口径，
不再手拼环境变量。paths.sh 顶层只有 `export`（`OPENPI_DATA_HOME` / `XDG_CACHE_HOME` / `HF_HOME` /
`HF_HUB_OFFLINE` / `JAX_COMPILATION_CACHE_DIR` / `UV_LINK_MODE` / `PYTHONUNBUFFERED`），
`mkdir` 只在函数 `v1_prepare_dirs` 内，source 不触发；且 paths.sh 明确禁覆盖 `HOME`（AGENTS.md 第 14 条）。

修正后启动脚本首行落盘 `OPENPI_DATA_HOME=` 以便事后核对。

**清理**：失败 run 的 checkpoint 目录 `v1-store/train-runs/mme_vla_suite_b128/bench-b1-motion` 已删除；
tmux 会话 `bench-b1` 已自行退出（删前删后 `tmux ls` 均确认用户自有会话 `0`、`1`、`claude-private` 未受影响）。

**重启**：2026-09-04 17:58:54，tmux `bench-b1`，GPU 采样进程 PID 21576（`timeout 3600 nvidia-smi -lms 500`）。
