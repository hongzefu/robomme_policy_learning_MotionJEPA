# repro-4a100-fsdp4 — 起跑记录

## 目的

外部合作者在 **4 × A100-40GB、无 NVLink/P2P** 的机器上跑官方 MME-VLA 训练遇阻，怀疑 40G 显存不够。
本轮在环境 B（AWS 单机 8 × A100-80GB）上复现官方训练命令，回答两个问题：

1. 官方那条命令在 A100×4 + `--fsdp-devices=4` 下能否跑起来；
2. **40G 显存到底够不够**。

本轮**不产出模型**，只跑 20 步验证链路可用。

## 环境

- 环境 B（AWS 单机）。仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。
- 8 × NVIDIA A100-SXM4-80GB，`nvidia-smi topo -m` 全 `NV12`（两两 NVLink 全互联），96 vCPU，1121 GB RAM。
- 存储介质：**AWS 本地 NVMe RAID（`/dev/md0`，6.9 T）**。本文吞吐数字仅在此介质口径下有效，
  不得与环境 A 的 A40 + turbo NFS 数字混比（AGENTS.md 第 13 条）。

## 代码锚点

**实际执行的代码不是本分支 HEAD**，而是官方 dataloader 版本的 detached worktree：

| | |
|---|---|
| 执行代码 | `89efeaab461cc2b00ede344edf4283692e9c3ada`（README 引用的官方 dataloader 版本） |
| worktree | `v1-store/worktrees/official-89efeaab`（`git worktree add --detach`） |
| 脚本载体（本分支） | `ec1101a`，`v2-motionmem` |

### 为什么必须用 worktree

当前 `v2-motionmem` 已删掉官方那条数据链，无法跑官方 pkl 库：

- `src/mme_vla_suite/training/dataset.py`（`SampleDataset` / `RoboMMEDataset`）删于 `2699fa5` commitV4.1；
- `src/mme_vla_suite/shared/mem_buffer.py`（`MemoryBuffer` / `MemoryBufferRecurrent`）删于 `fc77bf0` commitV4.4；
- `dataloader.py` 的 `MMEVLA_DATA_BACKEND=legacy` 分派同在 `2699fa5` 压成无条件 packed `FrameSampDataset`；
- `scripts/train.py` 已挪到 `scripts/training/train.py`。

### venv 隔离（关键，否则静默跑成嵌合体）

主 checkout 的 `.venv` 是 editable 安装，`_editable_impl_openpi.pth` 内是**硬编码绝对路径**指向主 checkout
的 `src/`。worktree 若复用该 venv，`import mme_vla_suite` 会解析到 **HEAD 的代码**而非 89efeaab，
形成「官方 train.py + HEAD dataloader」的嵌合体，并在 `StoreMeta.load` 上报出完全不指向真实原因的错。

故 worktree 内单独 `uv sync` 建独立 venv（89efeaab 依赖是 HEAD 的严格子集，`uv.lock` 仅差
`flask` / `blinker` / `itsdangerous`，全部命中 `/scratch/hongze/.cache/uv` 热缓存）。起跑前实测判据：

```
$ cat <WT>/.venv/lib/python3.11/site-packages/_editable_impl_openpi.pth
/scratch/hongze/.../v1-store/worktrees/official-89efeaab/src        ← 指向 worktree ✓
$ uv run --no-sync python -c "import mme_vla_suite.training.dataset as d; print(d.__file__)"
.../official-89efeaab/src/mme_vla_suite/training/dataset.py         ← 解析到 89efeaab ✓
RoboMMEDataset True | SampleDataset True
```

两档并行前 `uv sync` 已单独跑完（exit 0），起跑一律用 `uv run --no-sync`，避免两进程争同一把 venv 锁。

## 数据

`v1-store/datasets/4task-motion-400ep/source` —— 官方 `robomme_pkl` 格式，逐项对得上官方 `RoboMMEDataset`：

| 官方 dataloader 要的 | 本机实际 |
|---|---|
| `data/{idx}.pkl` | 101,066 个，38 G |
| pkl 键 | `image`(256,256,3 uint8)、`wrist_image`、`state`(8)、`actions`(20,8)、`is_demo`、`exec_start_idx`、`step_idx`、`epis_idx`、`prompt`、`simple_subgoal`、`grounded_subgoal`、`*_online` —— 与 `SampleDataset.__getitem__` 逐键一致 |
| `features/episode_{i}/token_emb_{idx}.npy` | 400 个 episode，70 G；含 `image_emb_8x8/4x4/2x2`(bf16)、`pos_emb_*`(fp32)、`state_emb` |
| `meta/stats.json` | `execution_samples=101066`、`total_samples=123044`（`SampleDataset.__len__` 读前者） |

**与对方环境的唯一实质差异**：对方用 `data/robomme_preprocessed_PickXtimes`，本机没有 PickXtimes
（原始 H5 只下了 `ButtonUnmask` / `ButtonUnmaskSwap` / `VideoUnmask` / `VideoUnmaskSwap` 四个任务，85 G）。
格式、schema、dataloader 代码路径、batch 形状全同，差的只是样本内容与任务语义，
对「命令能否跑起来」「38G 够不够」两个判据无影响。

其余资产：`OPENPI_DATA_HOME=v1-store/models`（`openpi-assets/checkpoints/pi05_base/params`，12 G）、
`--assets-base-dir=v1-store/train-assets`（代码找 `mme_vla_suite/robomme/norm_stats.json`，
因 `repo_id="robomme"` → `asset_id="robomme"`）。

## 两档设计

8 卡同场次并行，GPU 不重叠、`exp_name` 不同、checkpoint 与 jax 编译缓存目录各自独立。

| 档 | GPU | `XLA_PYTHON_CLIENT_MEM_FRACTION` | `NCCL_P2P_DISABLE` | 作用 |
|---|---|---|---|---|
| `repro-4a100-fsdp4-40gsim` | 0,1,2,3 | `0.475`（≈38 G） | `1` | **主判据**：模拟对方 4×40G 无 P2P |
| `repro-4a100-fsdp4-80g` | 4,5,6,7 | `0.95`（≈76 G） | 不设 | **对照**：本机原生条件 |

**显存换算**：对方 40G 卡 × `0.95` ≈ 38 G 可用；本机 80G 卡 × `0.475` = 38 G，等效。

**为什么要带对照**：单跑 40G 一档时，OOM 分不清是「显存真不够」还是「命令/环境本身有问题」。
并行拿两个结果后判读确定：两档都通过 → 40G 够；40G 挂 OOM 而 80G 通过 → 确证显存边界；
两档都挂 → 与显存无关，按报错另查。

## 命令

脚本：`run_40gsim.sh`、`run_80g.sh`（同目录）。两份仅差档位三项（`EXP` / `CUDA_VISIBLE_DEVICES` /
`XLA_PYTHON_CLIENT_MEM_FRACTION` + `NCCL_P2P_DISABLE`），核心命令：

```bash
uv run --no-sync scripts/train.py mme_vla_suite \
  --exp-name="$EXP" \
  --batch-size=64 \
  --num-train-steps=20 \
  --num-workers=4 \
  --fsdp-devices=4 \
  --dataset-path="$REPO/v1-store/datasets/4task-motion-400ep/source" \
  --assets-base-dir="$REPO/v1-store/train-assets" \
  --checkpoint-base-dir="$REPO/v1-store/train-runs" \
  --model.use-history \
  --model.history-config=perceptual-framesamp-modul.yaml \
  --no-wandb-enabled \
  --overwrite
```

与对方原命令的差异（40gsim 档五处，80g 档四处），均为环境适配、不改训练语义：
`XLA_PYTHON_CLIENT_MEM_FRACTION` 0.95→0.475（仅 40gsim）、`--dataset-path`（本机 4task 库）、
`--assets-base-dir` / `--checkpoint-base-dir`（产物落 `v1-store`，AGENTS.md 第 14 条单一产物根）、
`--num-train-steps` 200→20（只验证能否跑起来）、`--overwrite`（允许重跑）。
`--batch-size=64`、`--num-workers=4`、`--fsdp-devices=4`、`--model.history-config`、
`--no-wandb-enabled`、`NCCL_IB_DISABLE=1` 逐字保持。

### 一处越界的处理

官方 `scripts/train.py` 硬编码
`jax.config.update("jax_compilation_cache_dir", str(epath.Path(f"~/.cache/jax_{config.exp_name}").expanduser()))`，
会往 `$HOME` 写几百 MB；而 AGENTS.md 第 14 条**禁止覆盖 `HOME`**。脚本起跑前把该路径预置成
指向 `/scratch/hongze/.cache/jax_<exp>` 的 symlink，官方代码照写不误，实际字节落在 scratch。

## tmux 会话名单（清理时的唯一依据）

本轮起过的会话**只有两个**：

- `repro-40gsim`
- `repro-80g`

`tmux ls` 中的 `0`、`1`、`claude-private` 是**用户自己的会话，一律不动**。
清理按 AGENTS.md 第 7 条红线：禁止 `kill-server` 及一切全局杀法，只逐个
`tmux kill-session -t <确切名>`，删前删后各 `tmux ls` 一次核对差集。

## 判据

| 判据 | 通过标志 |
|---|---|
| 数据链通 | 日志出现 `Initialized data loader:` 及其 batch 形状树 |
| 训练在跑 | `Progress on: 1it/20it`、`Step 0: ... loss=` |
| 正常收尾 | 末行 `EXIT_CODE=0` |
| 显存 | `records/gpu_all8.csv` 各卡 `memory.used` 峰值；40gsim 档（GPU 0-3）须 < 38 G 才算「40G 够」 |

GPU 采样：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -lms 500`
（AGENTS.md 第 16 条口径，采样间隔远小于步时；判读用均值 / 0% 占比 / 分层均值，不用中位数）。

## 起跑时刻

2026-09-04 19:53:41（两档同时）。结果见 `result.md`。
