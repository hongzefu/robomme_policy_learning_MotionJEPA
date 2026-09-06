# eval-3seed-context-vs-motion 起跑留档

## 一、目的

已有两轮在线评估各只跑了一个 policy 采样 seed（都是 42），四任务均值 官方 context 24.0% vs
我们的 awsprod40k-b128-motion 28.0%。`docs/training-doc/eval-official-framesamp-context/result.md`
的盲区清单已点名「单 seed、单 checkpoint……任何『谁更好』的结论都需要多 seed」。
本轮在本机复刻这两组评估，各跑 **3 个采样 seed（42 → 7 → 2024）**，取逐 episode 成功率，
判断 24.0% vs 28.0% 是否落在采样噪声内。

**seed 的语义（重要）**：这条链路上换 seed 只换 policy 采样噪声，换不了环境。
每集的物理初始布局固化在 benchmark 元数据里——`examples/robomme/env_runner.py` 的
`EnvRunner.__init__` 硬编码 `BenchmarkEnvBuilder(dataset="test")`，
`episode_config_resolver.py::make_env_for_episode` 从
`env_metadata/test/record_dataset_<task>_metadata.json` 读 `records[].seed` 传给 `gym.make(..., seed=...)`，
`env.reset()` 不传 seed。可调的 `SEED` 只走到 `serve_policy.py --seed` →
`src/mme_vla_suite/policies/policy.py::MME_VLA_Policy.reset()` 的 `jax.random.key(self._seed)`，
再由 `infer()` 的 `jax.random.split` 喂给 flow-matching 采样噪声；
`eval.py --args.model_seed` 更弱，只决定输出目录名 `seed<N>/`。
故本轮产出是「同一批 200 个固定环境实例上的 3 条采样噪声流」。

## 二、运行环境

**环境 A**（`AGENTS.md`「运行环境判定」），判定输出：

| 判据 | 实测 |
|---|---|
| 仓库根 | `/data/hongzefu/robomme_policy_learning_MotionJEPA`（环境 A） |
| `/nfs/turbo/coe-chaijy-unreplicated/hongzefu` | 存在（环境 A） |
| `/data/hongzefu` | 存在（环境 A） |
| `/scratch/hongze` | 不存在（环境 A） |
| `~/.ssh/config` | 存在（环境 A） |
| GPU | **2 × NVIDIA RTX 6000 Ada Generation，46,068 MiB/卡**（判据表写的是 A40，属第三套硬件） |
| CPU / 内存 | 32 核 / 377 GB |

GPU 一项与判据表不符，已交用户裁决，**拍板：按环境 A 走，GPU 一项按实测记为 2 × RTX 6000 Ada**。
本留档所有耗时数字均为该硬件实测，**与历史 A100（环境 B）/ A40 数字不得混比**（AGENTS 13）。

## 三、被评权重

| 组 | run 根 | CKPT_DIR | 加载分支 |
|---|---|---|---|
| 官方 context | `v1-store/models/official-mme-vla/perceptual-framesamp-context/` | 同目录 `/79999` | `history_config.txt` 旧兼容路径（非严格恢复），无 sidecar |
| awsprod40k motion | `v1-store/models/awsprod40k-b128-motion/` | 同目录 `/39999` | `history_config.resolved.yaml` 快照口径（严格恢复），自动拉起 motion sidecar |

两套本机原先都没有（`v1-store/models/` 只有 `big_vision / openpi-assets / pi05_vision_encoder`，
turbo 归档的 `train-runs/mme_vla_suite/` 只有 `v1-prod-*`），本轮现下：

- **官方 context**：`docs/training-doc/eval-3seed-context-vs-motion/download_official.sh`
  （与 `eval-official-framesamp-context/download.sh` 同源，仅把 `MAIN` 换成本机路径、显式传 `HF_TOKEN`）。
  源 `Yinpei/mme_vla_suite` @ `5db4d53ddb98c7f80cab08792dd53d985d712ab1` 的
  `perceptual-framesamp-context/79999.zip`，11,574,223,742 B，sha256 与
  `387d4bd500a29588d81c2675a83be351f84383b93bea4161b8f7fadbf91a8ae8` 三方核对。
- **awsprod40k**：`docs/training-doc/eval-3seed-context-vs-motion/download_awsprod40k.sh`。
  源私有 bucket `hf://buckets/HongzeFu/robomme-motionjepa-vla-v1`（上传留档
  `docs/dataset-build-doc/hf-export-awsprod40k-ckpt-v1/`，全量 166 文件 / 92,689,314,972 B / 8 个 step）。
  本轮只取 **27 文件 / 11,584,776,670 B** = `39999/**` 20 个 + run 根 7 个元文件
  （`history_config.resolved.yaml`、`history_config.resolved.sha256`、`history_config.txt`、
  `motion_provenance.json`、`wandb_id.txt`、`README.md`、`SHA256SUMS.pre.txt`），
  `--include` 清单已用 `--dry-run` 逐条核对命中（不用 `*.json` 泛匹配，否则会带下其余 7 个 step 的
  `norm_stats.json` 造出空壳目录）。

本机其余依赖已确认存在：sidecar venv `v1-store/venvs/wan/`、sidecar encoder
`v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt`、
robomme 解释器 `/home/hongzefu/micromamba/envs/robomme/bin/python`。
benchmark 元数据用 micromamba `robomme` env editable 装的那份
`/data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata`
（本仓库 `third_party/robomme_benchmark/` 是空目录，未 init 子模块）。

## 四、本轮代码改动

| # | 文件 | 改动 |
|---|---|---|
| 1 | `scripts/training/prod/eval_all_shards.sh` | 新增 `GPU_LIST`（默认 `0,1,…,7`，即历史的「worker 号 = 卡号」），`build_rows()` 的 stride 与 task 两个分支都按它取模分配卡号 |
| 2 | 同上 | `ENVS=` 补转发 `POLICY_MEM_FRACTION` 与 `ROBOMME_PY`（仅显式设置时才带；tmux 会话不继承调用方环境，不转发则本机会用 `eval_shard.sh` 里环境 B 的默认解释器路径 `/scratch/hongze/...` 而直接报错退出） |
| 3 | `scripts/training/prod/aggregate_seed_runs.py`（新增） | 跨 seed / 跨任务批次汇总：读各批分片 `progress.json` 与 `videos/` 文件名，出逐任务 × 逐 seed 成功率、均值/标准差、逐集跨 seed 稳定性与明细 |

**逐 episode 成功率不需要改 `eval.py`**：`progress.json` 早就是 `{task: {ep: true|false|"error"}}` 逐集落盘，
success/fail/timeout 三分从视频文件名反解（沿用 `merge_eval_shards.py` 的 `_VIDEO_RE`）。

三处都只动调度与汇总，不碰模型、数据、训练输入或推理语义，故不适用 AGENTS 18 的
「前后两张链路图 + 梯度一致性」（那条针对影响训练输入或训练语义的改动）。

改动验证（`DRY_RUN=1`，改动前后逐行对比）：

- 不带 `GPU_LIST` 的 8 卡 stride：`w0..w7` → `GPU=0..7`、`PORT=8031..8038`，与改动前一致。
- 不带 `GPU_LIST` 的 task 布局：8 片 → `GPU=0..7`，与改动前一致。
- 官方组档位 `WORKERS=4 GPU_LIST=0,0,1,1`：`w0/w1→GPU0`、`w2/w3→GPU1`，`EP_STRIDE=4 EP_START=0..3`。
- motion 组档位 `WORKERS=2 GPU_LIST=0,1`：`w0→GPU0`、`w1→GPU1`，`EP_STRIDE=2`。
- 两个新转发：显式设置时进命令，不设时命令里零出现。
- `aggregate_seed_runs.py` 用假数据冒烟（2 组 × 2 seed × 1 任务 × 2 worker，含 1 个 `error`、故意缺 2 集）：
  判定行 `SEED_AGG=INCOMPLETE ... episodes=38/40 errors=1`，缺口列出 `seed7 ButtonUnmask 缺 2 集: 6,8`，
  跨 seed 表与稳定性统计均正确。

## 五、分片配置（两条硬约束）

**约束一：单个 `eval.py` 进程最多建 27 个仿真环境**，第 28 次 `make_env` 必抛
`vk::createInstanceUnique: ErrorIncompatibleDriver`（`eval-official-framesamp-context/result.md`
「一个确定性上限」，w0/w1 两次精确复现）。故两组都按任务分批：一批只跑一个任务（50 集）。

**约束二：sidecar 同卡共享时各慢一倍**——本机 `docs/training-doc/motion-t3-open/result.md` 实测
单 sidecar 独占一卡 0.88 s/窗，两个同卡共享时增量 1.64 s/窗（util 100%）。故 motion 组一卡一 worker。

| | 官方 context | awsprod40k motion |
|---|---|---|
| 每批 | 1 个任务（50 集） | 1 个任务（50 集） |
| `MODE` / `WORKERS` / `GPU_LIST` | stride / 4 / `0,0,1,1` | stride / 2 / `0,1` |
| 每 worker 集数 | 13/13/12/12 | 25/25 |
| `POLICY_MEM_FRACTION` | 0.40（≈18.4 GB/policy，每卡 ≈38.4 GB） | 0.55（≈25.3 GB + sidecar 3.3 + 仿真 0.8 ≈ 29.4 GB） |
| 显存背书 | `motion-t3-open/result.md`：同款卡「每卡 2 policy(0.40) + 2 仿真」= GPU0 39.8 GB | 同留档「GPU1 = 2 policy + 2 sidecar + 2 仿真」= 42.0 GB，本轮每卡只放一半 |

## 六、执行顺序（用户拍板）

| 阶段 | 组 | seed |
|---|---|---|
| 1 | 官方 context | 42 → 7 |
| 2 | awsprod40k motion | 42 → 7 → 2024 |
| 3 | 官方 context | 2024 |

全程串行，一次只有一组在跑、两张卡都归它。每个 (组, seed) 跑完立刻汇总出表。

4 小时预算下的外推：阶段 1 每 seed 16–24 min、阶段 2 每 seed 54–74 min、阶段 3 16–24 min，
合计 240–324 min（含前置约 30 min）。**T+200 决策点**：阶段 2 的 seed 2024 若未起跑则跳过、
改跑阶段 3（换一个完整的官方三点）；若已在跑则让它跑完，阶段 3 顺延。

## 七、命令

公共：`ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python`，`PORT_BASE` 每批 +8。

阶段 1 / 3（官方组，`<Task>` 遍历 ButtonUnmask / VideoUnmask / ButtonUnmaskSwap / VideoUnmaskSwap）：

```bash
MODE=stride WORKERS=4 GPU_LIST=0,0,1,1 TASKS_ALL=<Task> \
  RUN_NAME=official-ctx-s<SEED>-<Task> CKPT_ID=79999 \
  CKPT_DIR=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-context/79999 \
  SEED=<SEED> LOG_PREFIX=evctx-s<SEED>-<task缩写> PORT_BASE=<每批+8> \
  POLICY_MEM_FRACTION=0.40 ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python \
  bash scripts/training/prod/eval_all_shards.sh
```

阶段 2（motion 组）：

```bash
MODE=stride WORKERS=2 GPU_LIST=0,1 TASKS_ALL=<Task> \
  RUN_NAME=awsprod40k-motion-s<SEED>-<Task> CKPT_ID=39999 \
  CKPT_DIR=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/awsprod40k-b128-motion/39999 \
  SEED=<SEED> LOG_PREFIX=evmot-s<SEED>-<task缩写> PORT_BASE=<每批+8> \
  POLICY_MEM_FRACTION=0.55 ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python \
  bash scripts/training/prod/eval_all_shards.sh
```

汇总：

```bash
UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/aggregate_seed_runs.py \
  --group official:official-ctx:79999:4 --group motion:awsprod40k-motion:39999:2 \
  --seeds <已完成的 seed 列表> --tasks ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap \
  --metadata-dir /data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata \
  --out-dir docs/training-doc/eval-3seed-context-vs-motion/records
```

`--metadata-dir` 必须显式给：默认值指向本仓库空的 submodule，不指过去则逐集 env seed / 难度全为 `None`。

## 八、产物路径

| 产物 | 路径 |
|---|---|
| 分片结果 | `v1-store/evaluation/<RUN_NAME>-w<k>/ckpt<79999\|39999>/seed<SEED>/{progress.json,log.json,videos/}` |
| 驱动日志 | `v1-store/logs/<LOG_PREFIX>-w<k>.log` / `.server.log` |
| 下载日志 | `v1-store/logs/dl-official-ctx.log`、`v1-store/logs/dl-awsprod40k.log` |
| 汇总 | `docs/training-doc/eval-3seed-context-vs-motion/records/{summary.txt,per_episode_by_seed.json}` |

## 九、盯盘项

- `Error saving final results`：`eval.py` 收尾 `sum(log_dict[task].values())` 遇到 `"error"` 会 TypeError，
  外层 `while not os.path.exists(log.json)` 会重跑整个 worker。看到即人工介入。
- `vk::createInstanceUnique: ErrorIncompatibleDriver`：本轮每进程 ≤25 集，理论上不触发；
  若触发，同参数重跑该片即续评（`eval.py` 按 `progress.json` 跳过已评集）。
- 阶段切换前确认上一阶段 server 全退出（`nvidia-smi` 回基线、`tmux ls` 无本阶段会话），
  否则新阶段 `XLA_PYTHON_CLIENT_MEM_FRACTION` 预占会撞上残留显存。

## 十、本轮 tmux 会话清单（AGENTS 7）

清理只允许 `tmux kill-session -t <确切会话名>`，一次一个、名字写全，删前删后各跑一次 `tmux ls` 比对差集。
**禁止 `tmux kill-server`、`kill-session -a`、通配与前缀匹配。**
`tmux ls` 里不在下表内的会话（本机长期存在 `unisite`、`v3site`、`site8042`、`site8044`、
`ensite8048`、`ensite8049`、`ensite8050` 等属于用户）一律不动。

| 会话名 | 用途 |
|---|---|
| `dl-official-ctx` | 官方 context 权重下载 |
| `dl-awsprod40k` | awsprod40k 39999 checkpoint 下载 |
| `evctx-s<SEED>-<task缩写>-w<k>` | 阶段 1 / 3 官方组评估分片 |
| `evmot-s<SEED>-<task缩写>-w<k>` | 阶段 2 motion 组评估分片 |

实际起过的完整会话名逐条记入 `result.md`。
