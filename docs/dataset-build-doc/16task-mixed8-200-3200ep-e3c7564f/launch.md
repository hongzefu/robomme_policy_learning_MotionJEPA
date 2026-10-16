# mixed8 十六任务 3200 集 VLA 新库启动说明

本文件是建库档案的第一段，在 SigLIP 起跑之前提交。SigLIP 起跑到 motion pack verify PASS 之间 policy 仓库 HEAD 冻结，任何 commit 都不允许（红线 14）；所以本档案必须在冻结前落盘。第二段 `result.md` 在冻结窗口结束、全部验收通过后补写。下文标「待补」的项一律在 result.md 补，本文件不预填数字。时间均为美东，判定行原文里的 UTC 不改。

## 结论与实施范围

按 [mixed8 VLA 计划](/scratch/hongze/MotionJEPA-backup/1016-mixed8-vla-plan.html)（`dev/mixed8-vla-plan @ 45c3f9a`）从 mixed8-200 源 H5 全新构建 VLA 训练库 `16task-mixed8-200-3200ep-e3c7564f`，覆盖 16 任务 × 200 集共 3200 条 primary。本轮交付包括：物理合并 H5、SigLIP source、4×4 与 8×8 两档 framesamp、本库独立的 norm_stats、Wan latent、e72 encoder 的 motion token 与 motion 表、独立 oracle 及对拍，最后用两条 4 卡 20 步试训兼作 4 卡短测。两条 80k 正式训练（`mixed8-200-m8x8-modul-b128-80k`、`mixed8-200-m8x8-modul-motion-b128-80k`）各自另建训练档案，不在本档案范围内。encoder 的 mp4 可视化（`m8v-viz`）和 encoder ckpt 上传归 MotionJEPA 侧档案；本档案只记录它们与建库共用 GPU 和闸门的部分。

截至本档案提交，A 段（下载、解包、合并）和 B1 起跑判据已经完成，具体如下：

- B1 判据和 encoder 拷贝 PASS。
- A3 两任务小样本 G1 PASS，smoke 目录已删。
- 全量 100 个包下载、校验、解包 PASS。
- 全量 merge 的 plan 已出 `PLAN_OK`，merge 和 verify 正在 tmux `m8v-merge` 中运行，结果待补。

SigLIP 及以后各阶段都还没有起跑。

## 版本与代码状态

建库消费的 policy 代码是 `v2-motionmem` 上的 commitV11.23 批次，依次三次合并：

| 提交 | 内容 | 合入时刻（美东） |
|---|---|---|
| `e944b69` | 合入 sub/A1-merge：新增 `scripts/dataset/fetch_mixed8_h5.py`；`merge_v2_h5.py` 适配 local-benchmark 来源、8 档难度和子集预检 | 2026-10-16 08:05 |
| `44fb642` | 合入 sub/A1-train：`scripts/training/tests/train_es_bound.py`、`eval_es_bound` 参数化、`mixed8_200_runner` 两条 80k runner、motion-mixed8 history yaml | 2026-10-16 08:06 |
| `aa29828` | 合入 sub/A1b：四个公开 HF bucket 的上传驱动（trainset 分批回读、encoder ckpt + eval-mp4、两条 VLA 80k ckpt） | 2026-10-16 08:20 |

再往前是开工令 handoff `71f0ee5`，内容是 `mixed8-vla-build` 登记为 running。三次合并的 PRE_MERGE_REVIEW 判定行都写在各自的 commit subject 里，可以用 `git log` 还原。

A 段的 CPU 阶段起跑时，HEAD 位于上表三个提交之间：

- `m8v-fetch-smoke` 约 08:06 起，当时 A1-merge 已合入。
- `m8v-fetch` 08:11 起，HEAD 为 `44fb642`。
- `m8v-merge` 09:09 起，HEAD 为 `aa29828`。

这几个会话的日志都没有打印 `START_HEAD`，所以逐会话的起跑 HEAD 只能按提交时刻推断。实际消费的代码已用 `git diff --stat e944b69 aa29828 -- scripts/dataset/fetch_mixed8_h5.py scripts/dataset/merge_v2_h5.py scripts/dataset/paths.sh src pyproject.toml uv.lock` 核对：区间内这三个脚本没有改动，唯一的差异是 `src` 下新增的 motion-mixed8 history yaml。下载、解包和合并都不读这份 yaml，所以 A 段各会话实际执行的代码相同。policy 主检出另有他人的 3 个未跟踪文件（`scripts/training/tests/{compare_context_gpu_scaling.py,context_gpu_scaling.py,run_context_gpu_scaling.sh}`），本轮不碰（红线 5）。

**HEAD 冻结窗口（红线 14）**：从 `m8v-siglip` 起跑到 `m8v-motion` 的 `VERIFY_MOTION=PASS`，policy 仓库不合并任何分支、不写 handoff commit、不回流规则，文档提交也不允许。原因有两点：

- SigLIP 各 shard 把 `git rev-parse HEAD` 写进 `meta/_shard*of*.json` 的 fingerprint，`finalize_checks.py check` 要求各分片的 git_commit 一致。
- `pack_motion_store.py` 要求各 worker 的 git_commit 唯一。

A1-train 已在 `44fb642` 合入，A1b 已在 `aa29828` 合入。本档案合入后，主检出 HEAD 就是窗口 HEAD。SigLIP 起跑前由主会话记下完整 40 位 sha，并核对 `git status --porcelain` 只含上述 3 个他人未跟踪文件，窗口 HEAD 待补。第二段 result.md 在窗口之后提交。

**encoder 来源**：MotionJEPA run `wan-mixed8-200-filter2-stop075-b88x8-72ep-a`，按口径 1 只用最后一个 ckpt `checkpoint_epoch_72.pt`，sha256 为 `bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04`。该 ckpt 的 `encoder` 键就是 EMA 权重（`ema_only: True`、`has_live_weights: False`），由 `E72_EMA_ONLY=PASS` 断言确认。`m8v-b1gate` 已把 ckpt 和同目录 `config.yaml` 拷入 `v1-store/external/motionjepa/wan-mixed8-200-filter2-stop075-b88x8-72ep-a/`，拷贝后 sha256 复核一致（G3）。sha 同时写入 `v1-store/m8v-scripts/e72.sha256`，供 encode 与 oracle 的 `--expected-ckpt-sha256` 读取。

B1 判据由 `m8v-b1gate` 于 2026-10-16 约 08:07 产出，日志为 `v1-store/logs/m8v-b1gate.log`，原文如下：

```text
E72_EMA_ONLY=PASS
B1_GATE=PASS exit=0 epochs=72 ckpt_sha=bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04 manifest_sha=e3c7564f tmux_gone=1 gpu_idle=1 ema_only=1
G3=PASS enc_copy_sha=bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04
EXIT_CODE=0
```

**计划内调整**：A1-merge 和 A1b 两个写入子代理曾卡死，时间约为 03:12–07:47。原因是主会话误调了 EnterWorktree，与代码内容无关。A 段因此顺延，按风险 8 处理：B1 的 mp4 照常起，SigLIP 等合并 H5 和 manifest 就绪后再起，B2 和 B4 整体顺延同等时长。另外，全量下载与 A3 小样本并行起跑。这一项经用户同意（「全量下载是否和 A3 并行 同意」），并通过了 `ADJUST_REVIEW=PASS scale=same`；`m8v-merge` 仍等 G1 PASS 后才起。

## 数据来源与排序

- **源**：公开 HF bucket `HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1`，内含 100 个 `record_dataset_<任务>_<难度>.h5.tar.xz` 包，每个包对应一个（任务, 难度）格。
- **MANIFEST**：`/scratch/hongze/robomme_benchmark_newtask-v3-MotionJepa1006/artifacts/mixed8/full200/snapshot/MANIFEST.json`，sha256 为 `e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11`，库名后缀取它的前 8 位。
- **来源 pin**：`merge_plan.json` 的 source_pin 记为 `source_kind=local-benchmark`、revision `f9f8c64edc6d5d6dea0dcb961903d09e243ba04c`、results_sha256 `9f6c5c5464e16a5be6c4ddefdfa477ed85bbc3e4f41ec7bb0e19d066b15e5137`、run_parameters_sha256 `525d07030a48937842e93043a45051b0928640f85b3af936f2ffd720b65b6ae3`。local-benchmark 来源没有 HF repo revision 可钉，身份只靠 MANIFEST 字节摘要。
- **规模**：16 任务 × 200 = 3200 条 primary，MANIFEST 中 primary 的 `timestep_count` 之和为 2,145,548。

**排序**：`sort_version=difficulty-episode-v1`。每个任务内部先按难度排，顺序为 easy、medium、hard、xhard12、xhard14、xhard16、xhard18、xhard20，难度相同再按原 episode 编号排，编为 `episode_0..199`。原 episode 号和 seed 不能单独作为全局身份；每行 episode map 都保留难度、member、src_group、h5_sha256 和原始身份。

下表由 plan 阶段写出的 16 份 `record_dataset_<任务>_episode_map.json` 汇总而成，16 任务合计 3200 行，timestep_count 之和 2,145,548，与 MANIFEST 一致。merge verify 完成后要再与 `MERGE_VERIFY` 的 timesteps 核对，结果待补。

| 任务 | 难度与条数（按 raw_ep_idx 递增） | timesteps 合计 | 单集最长 |
|---|---|---:|---:|
| BinFill | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 215622 | 1874 |
| ButtonUnmask | easy 50、medium 50、hard 50、xhard14 50 | 67865 | 579 |
| ButtonUnmaskSwap | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 121015 | 942 |
| InsertPeg | easy 67、medium 67、hard 66 | 99115 | 1445 |
| MoveCube | easy 67、medium 67、hard 66 | 81806 | 1446 |
| PatternLock | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 85692 | 784 |
| PickHighlight | easy 34、medium 34、hard 33、xhard12 33、xhard16 33、xhard18 33 | 123865 | 1074 |
| PickXtimes | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 201716 | 1989 |
| RouteStick | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 131100 | 1000 |
| StopCube | easy 29、medium 29、hard 29、xhard14 29、xhard16 28、xhard18 28、xhard20 28 | 72416 | 585 |
| SwingXtimes | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 131368 | 1096 |
| VideoPlaceButton | easy 40、medium 40、hard 40、xhard12 40、xhard14 40 | 212979 | 1369 |
| VideoPlaceOrder | easy 50、medium 50、hard 50、xhard12 50 | 236371 | 1448 |
| VideoRepick | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 194704 | 1400 |
| VideoUnmask | easy 50、medium 50、hard 50、xhard14 50 | 58223 | 506 |
| VideoUnmaskSwap | easy 25、medium 25、hard 25、xhard12 25、xhard14 25、xhard16 25、xhard18 25、xhard20 25 | 111691 | 899 |

`TASKS` 由 env.sh 从 MANIFEST 的 primary 任务名排序生成，共 16 个。下游 `canonical_order` 以实际 `episode_manifest.json` 为准，不按 CLI 任务顺序推断 global episode 区间；manifest 的 canonical_order 待补。

**窗口预算（建库前实算，主会话已完成）**：按本地 data-raw 帧数、scope.json 的 demo 帧数和 `dropped_tail_frames` 对 3200 集逐集计算。判定行如下：

```text
TRAIN_ES_BOUND_PRE=PASS episodes=3200 tasks=16 max_windows=123 budget=160 headroom=37
```

最坏的一集是 PickXtimes 输出索引 196，源文件 `PickXtimes_ep175_seed18500.h5`：demo 0 窗，执行 1989 步，共 123 窗。data-raw 里的 `PickXtimes_ep196` 是另一集（1163 帧），对拍时两种身份都要记录。上表 PickXtimes 单集最长 1989，与这一集一致。建库后用真实 manifest 复算 `TRAIN_ES_BOUND`，要求 max_windows≤123 且 headroom≥0，结果待补。

## 路径、环境与资源

| 用途 | 路径 |
|---|---|
| 源下载与解包 | `v1-store/staging/mixed8-200-src`（`snapshot/`、`control/`、`extracted/`） |
| 合并 H5 | `v1-store/raw-h5/16task-mixed8-200-20261014` |
| 库 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f` |
| norm_stats | `v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f/robomme/norm_stats.json` |
| encoder 拷贝 | `v1-store/external/motionjepa/wan-mixed8-200-filter2-stop075-b88x8-72ep-a/` |
| 脚本与日志 | `v1-store/m8v-scripts/`、`v1-store/logs/` |

`v1-store` 指 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store`。库、norm_stats 和合并 H5 都是本轮全新输出，写本档案时库目录与 norm_stats 目录尚不存在，不覆盖任何旧库、旧统计量或既有 run。

**删除时点（口径 24、红线 8）**：

- 解包件：由 `fetch_mixed8_h5.py cleanup` 核对 verify 日志里的 PASS 后统一删除。
- `source/features`：4×4、8×8、xgrid 都 PASS 后删除，只保留 store_meta 点名的抽样文件。
- 合并 H5：motion verify PASS 且 VLA 新库上传 PASS 之后才删。
- `wan-latents`、`motion-tokens`、`oracle`：随新库上传，上传 PASS 后本地删除。

每次删除前都要列清单交主会话核对。

**硬件与资源**：AWS `p4de.24xlarge`，8×A100-SXM4-80GB，96 核。存储为 `/dev/md0` XFS 本地 NVMe RAID，总容量 7,516,777,873,408 B。美东 09:11（全量 merge 进行中，此时 1.4 TB 解包件仍在盘）空闲 4,073,680,723,968 B，约 4.07 TB。磁盘账和应急阈值（0.4 TB）按计划第四节执行；B2 的 8×8 打包峰值按上沿估计空闲约 0.5 TB。

GPU 分配：

| 阶段 | GPU |
|---|---|
| mp4（`m8v-viz`） | 0–3 |
| SigLIP | 4–7，与 mp4 并行 |
| finalize | 7，结束即让出 |
| Wan、encode | 0–7 |
| oracle | encoder 段 7；vae 段 8 分片，每片一卡 |
| 两条 try20 | 0–3 与 4–7 |

CPU 阶段一律用 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu` 隔离；合并用 16 进程，verify 与打包用 48 进程。缓存全部放在 `/scratch/hongze` 下：`UV_CACHE_DIR=$V1_STORE/cache/uv`、`HF_HOME=/scratch/hongze/.cache/huggingface`。`.venv` 用主检出自己的。

## 命令与配置还原

`v1-store/m8v-scripts/` 不在 git 中，所以各阶段脚本的原文直接写在下面，不另存脚本拷贝。每段都是独立的 detached tmux 会话，会话前缀为 `m8v-`，按计划 runbook「启动方式」的模板启动：

```bash
S=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts
LOGS=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs
tmux new-session -d -s <会话名> "set -o pipefail; PYTHONUNBUFFERED=1 bash $S/<会话名>.sh 2>&1 | tee $LOGS/<会话名>.log; echo \"EXIT_CODE=\$?\" >> $LOGS/<会话名>.log"
```

各会话都由脚本首行 source 公共环境，主会话 shell 不 source `paths.sh`。公共环境 `$S/env.sh` 原文如下：

```bash
# m8v 公共环境（计划 1016-mixed8-vla-plan.html @ 110fcd7 runbook「公共环境」）；各 m8v-*.sh 首行 source 本文件
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh       # 定义 V1_STORE、OPENPI_DATA_HOME 等；会 set -euo pipefail，并 export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$V1_STORE/cache/uv" HF_HOME=/scratch/hongze/.cache/huggingface OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -a; source "$V1_STORE/secrets/hf.env"; set +a
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
MJ=/scratch/hongze/MotionJEPA
HF=/scratch/hongze/MotionJEPA/.venv/bin/hf
NAME=16task-mixed8-200-3200ep-e3c7564f
SRC=$V1_STORE/staging/mixed8-200-src
RAW=$V1_STORE/raw-h5/16task-mixed8-200-20261014
LIB=$V1_STORE/datasets/$NAME
LOGS=$V1_STORE/logs
R=wan-mixed8-200-filter2-stop075-b88x8-72ep-a
ENC=$V1_STORE/external/motionjepa/$R
SRC_MANIFEST=/scratch/hongze/robomme_benchmark_newtask-v3-MotionJepa1006/artifacts/mixed8/full200/snapshot/MANIFEST.json
MANI=$LIB/meta/episode_manifest.json; INMANI=$LIB/meta/input_manifest.json
TASKS=$(uv run --no-sync python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(",".join(sorted({e["task"] for e in m["episodes"] if e["role"]=="primary"})))' "$SRC_MANIFEST")
test "$(echo "$TASKS" | tr ',' '\n' | wc -l)" = 16
SHA=$(cat "$V1_STORE/m8v-scripts/e72.sha256" 2>/dev/null || true)   # B1 起跑判据通过后才有
export PYTHONUNBUFFERED=1
```

env.sh 首行注释写的是计划 `110fcd7`。经核对，计划之后到 `45c3f9a` 的修订没有改动「公共环境」一段，内容仍与 runbook 相同。

**env 覆盖项（可复现面）**：

| 变量 | 值 | 作用范围 |
|---|---|---|
| `UV_CACHE_DIR` | `$V1_STORE/cache/uv` | 全部会话 |
| `HF_HOME` | `/scratch/hongze/.cache/huggingface` | 全部会话 |
| `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | `1` | 全部会话 |
| `HF_HUB_OFFLINE` | paths.sh 设为 `1`；fetch 两段 `unset` | 下载段联网 |
| `PYTHONUNBUFFERED` | `1` | 全部会话 |
| `CUDA_VISIBLE_DEVICES` | 各段命令串内显式给出（见下） | 逐段 |
| `JAX_PLATFORMS` | `cpu`（CPU 打包链、norm、motion pack） | 逐段 |
| `MMEVLA_FRAMESAMP_SOURCE` / `MMEVLA_FRAMESAMP_MANIFEST` | `$LIB/source` / `$MANI` | 由 runner 与 preflight 设置 |
| `FSDP_DEVICES` / `NUM_WORKERS` / `XLA_PYTHON_CLIENT_PREALLOCATE` | `4` / `8` / `false` | 两条 try20 |

**已执行：A3 小样本 `m8v-fetch-smoke`**（闸门 G1，约 08:06 起）

```bash
source /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts/env.sh
# m8v-fetch-smoke（A3，闸门 G1）：InsertPeg + BinFill 两任务，独立 smoke 目录；计划 45c3f9a runbook「训练结束前（A 段）」
unset HF_HUB_OFFLINE
SM="$V1_STORE/staging/m8v-smoke-src"; SR="$V1_STORE/staging/m8v-smoke-raw"
uv run --no-sync python scripts/dataset/fetch_mixed8_h5.py fetch --hf "$HF" --bucket HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1 --src "$SM" --tasks InsertPeg,BinFill --jobs 8 --retries 3
uv run --no-sync python scripts/dataset/merge_v2_h5.py plan --snapshot "$SM/snapshot" --control "$SM/control" --extracted "$SM/extracted" --out "$SR" --tasks InsertPeg,BinFill --procs 16
uv run --no-sync python scripts/dataset/merge_v2_h5.py merge --out "$SR" --extracted "$SM/extracted" --procs 16 --check-source-sha256
uv run --no-sync python scripts/dataset/merge_v2_h5.py verify --out "$SR" --extracted "$SM/extracted" --level full --procs 16
echo "全部完成 m8v-fetch-smoke"
```

判定行如下，日志为 `v1-store/logs/m8v-fetch-smoke.log`：

```text
FETCH_VERIFY=PASS task=BinFill archives=8 episodes=200
FETCH_VERIFY=PASS task=InsertPeg archives=3 episodes=200
PLAN_OK tasks=2 selected=400 role=primary:400 max_timesteps=1989 prechecks=PASS precheck_files=400
MERGE_TASK=PASS task=InsertPeg episodes=200
MERGE_TASK=PASS task=BinFill episodes=200
MERGE_VERIFY=PASS level=full tasks=2 episodes=400 timesteps=314737 role=primary:400 mismatches=0
全部完成 m8v-fetch-smoke
EXIT_CODE=0
```

MANIFEST 中这两个任务的 timestep_count 之和同为 314737（BinFill 215622 + InsertPeg 99115）。G1 PASS 后，主会话核对清单，删除了 `staging/m8v-smoke-src` 与 `staging/m8v-smoke-raw`，共约 391 GB。

**已执行：全量下载 `m8v-fetch`**（08:11 起，与 A3 并行，约 09:09 完成）

```bash
source /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts/env.sh
# m8v-fetch（A4 全量下载解包；单包断点续传重试 3 次，仍失败即停点 4）；计划 45c3f9a runbook「训练结束前（A 段）」
# 用户 2026-10-16「全量下载是否和 A3 并行 同意」：与 m8v-fetch-smoke 并行起，m8v-merge 仍等 G1 PASS
unset HF_HUB_OFFLINE
uv run --no-sync python scripts/dataset/fetch_mixed8_h5.py fetch --hf "$HF" --bucket HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1 --src "$SRC" --tasks "$TASKS" --jobs 8 --retries 3
echo "全部完成 m8v-fetch"
```

判定行如下，日志为 `v1-store/logs/m8v-fetch.log`。16 行 archives 合计 100 个包，解包件约 1.4 TB：

```text
FETCH_VERIFY=PASS task=BinFill archives=8 episodes=200
FETCH_VERIFY=PASS task=ButtonUnmaskSwap archives=8 episodes=200
FETCH_VERIFY=PASS task=ButtonUnmask archives=4 episodes=200
FETCH_VERIFY=PASS task=MoveCube archives=3 episodes=200
FETCH_VERIFY=PASS task=InsertPeg archives=3 episodes=200
FETCH_VERIFY=PASS task=PatternLock archives=8 episodes=200
FETCH_VERIFY=PASS task=PickHighlight archives=6 episodes=200
FETCH_VERIFY=PASS task=PickXtimes archives=8 episodes=200
FETCH_VERIFY=PASS task=StopCube archives=7 episodes=200
FETCH_VERIFY=PASS task=RouteStick archives=8 episodes=200
FETCH_VERIFY=PASS task=SwingXtimes archives=8 episodes=200
FETCH_VERIFY=PASS task=VideoPlaceButton archives=5 episodes=200
FETCH_VERIFY=PASS task=VideoRepick archives=8 episodes=200
FETCH_VERIFY=PASS task=VideoUnmaskSwap archives=8 episodes=200
FETCH_VERIFY=PASS task=VideoPlaceOrder archives=4 episodes=200
FETCH_VERIFY=PASS task=VideoUnmask archives=4 episodes=200
全部完成 m8v-fetch
EXIT_CODE=0
```

**已执行：B1 判据 `m8v-b1gate`**（约 08:07；判定行见「版本与代码状态」）

```bash
source /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts/env.sh
# m8v-b1gate：B1 起跑判据（口径 26，用户授权的第 27 条例外）。只读训练会话的文件，不写；计划 1016-mixed8-vla-plan.html @ 45c3f9a runbook
L=/scratch/hongze/logs/$R.train.log
CK="$MJ/runs/$R/checkpoint_epoch_72.pt"
if [ ! -f "$L" ] || [ "$(tail -n 1 "$L")" != "EXIT_CODE=0" ]; then echo "B1_GATE=FAIL reason=train_exit_not_0"; exit 1; fi
N=$(tr '\r' '\n' < "$L" | grep -vE '%\|' | grep -c '^Epoch ' || true)
if [ "$N" != 72 ]; then echo "B1_GATE=FAIL reason=epochs_$N"; exit 1; fi
if tmux has-session -t '=tr-wan-mixed8-200-filter2-stop075-b88x8-72ep-a' 2>/dev/null; then echo "B1_GATE=FAIL reason=tmux_alive"; exit 1; fi
if ! G=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then echo "B1_GATE=FAIL reason=nvidia_smi_error"; exit 1; fi
if [ -n "$G" ]; then echo "B1_GATE=FAIL reason=gpu_busy"; exit 1; fi
if [ ! -f "$CK" ]; then echo "B1_GATE=FAIL reason=ckpt_missing"; exit 1; fi
S1=$(sha256sum "$CK" | cut -d' ' -f1)
S2=$(sha256sum "$CK" | cut -d' ' -f1)
if [ "$S1" != "$S2" ]; then echo "B1_GATE=FAIL reason=ckpt_sha_unstable"; exit 1; fi
if [ "$(sha256sum "$SRC_MANIFEST" | cut -d' ' -f1)" != e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11 ]; then echo "B1_GATE=FAIL reason=manifest_sha"; exit 1; fi
# E72_EMA_ONLY 断言必须在 B1_GATE=PASS 与写 e72.sha256 之前；FAIL → 停点 8
if ! uv run --project "$MJ" --no-sync python -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert c.get('ema_only') is True and c.get('has_live_weights') is False; print('E72_EMA_ONLY=PASS')" "$CK"; then echo "B1_GATE=FAIL reason=e72_not_ema_only"; exit 1; fi
echo "B1_GATE=PASS exit=0 epochs=72 ckpt_sha=$S1 manifest_sha=e3c7564f tmux_gone=1 gpu_idle=1 ema_only=1"
echo "$S1" > "$V1_STORE/m8v-scripts/e72.sha256"
mkdir -p "$ENC"; cp "$CK" "$MJ/runs/$R/config.yaml" "$ENC/"
if [ "$(sha256sum "$ENC/checkpoint_epoch_72.pt" | cut -d' ' -f1)" != "$S1" ]; then echo "G3=FAIL reason=enc_copy_sha_mismatch"; exit 1; fi
echo "G3=PASS enc_copy_sha=$S1"
```

**进行中：全量合并 `m8v-merge`**（09:09 起，等 G1 PASS 且 16 行 `FETCH_VERIFY=PASS` 齐全后才起）

```bash
source /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts/env.sh
# m8v-merge（A4 全量合并；G1 PASS 后起；须等 m8v-fetch 全部 FETCH_VERIFY=PASS）；计划 45c3f9a runbook「训练结束前（A 段）」
# 中断后重跑 merge 改带 --resume，不用 --force
uv run --no-sync python scripts/dataset/merge_v2_h5.py plan --snapshot "$SRC/snapshot" --control "$SRC/control" --extracted "$SRC/extracted" --out "$RAW" --tasks "$TASKS" --procs 16
uv run --no-sync python scripts/dataset/merge_v2_h5.py merge --out "$RAW" --extracted "$SRC/extracted" --procs 16 --check-source-sha256
uv run --no-sync python scripts/dataset/merge_v2_h5.py verify --out "$RAW" --extracted "$SRC/extracted" --level full --procs 16 | tee "$LOGS/m8v-merge-verify.log"
uv run --no-sync python scripts/dataset/fetch_mixed8_h5.py cleanup --src "$SRC" --verify-log "$LOGS/m8v-merge-verify.log"
echo "全部完成 m8v-merge"
```

截至写本档案，日志 `v1-store/logs/m8v-merge.log` 已出 plan 判定行：

```text
PLAN_OK tasks=16 selected=3200 role=primary:3200 max_timesteps=1989 prechecks=PASS precheck_files=3200
```

以下结果均待补：16 行 `MERGE_TASK=PASS`、`MERGE_VERIFY=PASS level=full tasks=16 episodes=3200 timesteps=<n> role=primary:3200 mismatches=0`（期望 timesteps=2,145,548）、cleanup 结果、耗时和合并 H5 体积。

**待跑：manifest 与输入哈希 `m8v-hash`，窗口预算复算 `m8v-esbound`**（m8v-merge 全部完成后起）

```bash
source /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts/env.sh
# m8v-hash：manifest 与输入哈希（m8v-merge 全部完成后）；随后 m8v-esbound 复算窗口预算；计划 45c3f9a runbook
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --tasks "$TASKS" --episodes-per-task 200 --num_shards 1 --out "$MANI"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
echo "全部完成 m8v-hash"
```

```bash
source /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/m8v-scripts/env.sh
# m8v-esbound：建库后窗口预算复算（G2a，headroom<0 停下问用户，只暂停 motion 相关阶段）
uv run --no-sync python scripts/training/tests/train_es_bound.py --manifest "$MANI" --budget 160
```

`m8v-esbound` 之后，主会话再跑一次参数化后的 `eval_es_bound.py`，输出 tee 到 `$LOGS/m8v-sim-esbound.log`。如果 16 任务 × 50 条 sim test 集不存在，就直接写一行 `ES_BOUND=SKIP reason=no_sim_testset`，不阻塞后续阶段。

**B2 后续阶段**：命令照抄计划 runbook B2 段，各段写成 `$S/<会话名>.sh`，首行 `source $S/env.sh`。这些命令在 policy 仓库根目录执行（env.sh 已 cd 到这里）。

```bash
# m8v-siglip（B1_GATE=PASS 且 m8v-hash 完成后起）。HEAD 冻结窗口从此开始（红线 14，口径 42）：起跑前确认 A1-train、建库档案 launch.md 已合入或提交（A1b 来不及时按红线 14 推迟），记下 policy 主检出 HEAD 作为窗口 HEAD
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
# m8v-finalize（结束即让出 GPU 7）
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
# m8v-wan（HEAD 冻结窗口已自 m8v-siglip 起跑开始，红线 14）。起跑前主会话确认：/scratch/hongze/logs/m8v-viz.log 含 DELIVERY=PASS 与 EXIT_CODE=0、m8v-finalize EXIT_CODE=0、GPU 0–7 无计算进程
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
# m8v-pack4 / m8v-pack8：与 m8v-wan 并行的 CPU 链
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-8x8" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp-8x8" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store "$LIB/framesamp-8x8"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/xgrid_pos_check.py --store-4x4 "$LIB/framesamp" --store-8x8 "$LIB/framesamp-8x8" --source "$LIB/source" --image-spot 512
# C1（主会话）：sha256sum "$V1_STORE/train-assets/mme_vla_suite/$NAME/robomme/norm_stats.json"；report 的 (steps, samples)；MANIFEST sha → DATA_IDENTITY=PASS …，写进建库档案
# B3-1（xgrid 两行 IMAGE_NPY_SPOT、MOTION_POS_XGRID 都 PASS 后）：删 source/features 中未被 store_meta 的 source_spot_sha256.entries 点名的文件，删前列清单
# 删完 features 即满足 latent 上传闸门 → 起 m8v-latent-upload（见下）

# m8v-encode（G2a 通过后）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --encoder-run-dir "$ENC" --expected-ckpt-sha256 "$SHA"
# m8v-oracle（必须用 MotionJEPA 环境；目录不得已存在）
ORACLE="$LIB/oracle/wan-mj"; test ! -e "$ORACLE"; mkdir -p "$ORACLE"
CUDA_VISIBLE_DEVICES=7 uv run --project "$MJ" --no-sync python "$P/scripts/dataset/wan/oracle_driver.py" --mj-repo "$MJ" encoder --manifest "$MANI" --latents "$LIB/wan-latents" --out "$ORACLE" --encoder-run-dir "$ENC" --expected-ckpt-sha256 "$SHA" --demo-min-real 17 --exec-min-real 33
# 分片退出码写进各分片自己的日志（照抄 4 任务 …-motion-demopad17/launch.md 的 oracle 段写法）
pids=()
for i in 0 1 2 3 4 5 6 7; do
  (
    set +e
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$i" uv run --project "$MJ" --no-sync python "$P/scripts/dataset/wan/oracle_driver.py" --mj-repo "$MJ" vae --manifest "$MANI" --raw-dir "$RAW" --latents "$LIB/wan-latents" --out "$ORACLE" --shard-idx "$i" --num-shards 8 --demo-min-real 17 --exec-min-real 33 --sample-spec 'padded:all,rest:0.10,seed:0' 2>&1 | tee "$LOGS/m8v-oracle-vae-$i.log"
    code=${PIPESTATUS[0]}
    printf 'SHARD_EXIT_CODE=%s shard=%s\n' "$code" "$i" | tee -a "$LOGS/m8v-oracle-vae-$i.log"
    exit "$code"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || true; done      # 不在这里退出：先让 8 片都写完退出码行，再统一判
OK=$(cat "$LOGS"/m8v-oracle-vae-[0-7].log | grep -c '^SHARD_EXIT_CODE=0 ' || true)
if [ "$OK" != 8 ]; then echo "错误: SHARD_EXIT_CODE=0 只有 $OK/8 片"; grep -h '^SHARD_EXIT_CODE=' "$LOGS"/m8v-oracle-vae-[0-7].log || true; exit 1; fi
uv run --project "$MJ" --no-sync python "$P/scripts/dataset/wan/oracle_driver.py" aggregate --manifest "$MANI" --out "$ORACLE" --num-shards 8 --kind vae --demo-min-real 17 --exec-min-real 33
# m8v-motion（pack + verify 后 HEAD 冻结窗口结束）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py pack --manifest "$MANI" --tokens "$LIB/motion-tokens" --latents "$LIB/wan-latents" --out "$LIB/motion" --encoder-run-dir "$ENC" --raw-dir "$RAW"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion" --resume
# 对拍（--expect-* 由主会话现场从 manifest 计算后填入；按 Codex 审查从元信息推导的参考值：rows 128342 = exec 81611 + demo 46731，padded 1800，episodes 3200）
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents "$LIB/wan-latents" --oracle "$ORACLE" --sampled "$ORACLE/sampled_windows.json" --sample-spec 'padded:all,rest:0.10,seed:0' --expect-padded 1800
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens --store "$LIB/motion" --oracle "$ORACLE"
uv run --no-sync python scripts/dataset/motion_checks.py a6set --manifest "$MANI" --framesamp "$LIB/framesamp-8x8" --motion "$LIB/motion"
uv run --no-sync python scripts/dataset/motion_checks.py a7 --manifest "$MANI" --latents "$LIB/wan-latents" --tokens "$LIB/motion-tokens" --motion "$LIB/motion"
uv run --no-sync python scripts/dataset/motion_checks.py a9set --manifest "$MANI" --latents "$LIB/wan-latents" --motion "$LIB/motion" --n 500 --cold-all
uv run --no-sync python scripts/dataset/motion_checks.py a10 --manifest "$MANI" --motion "$LIB/motion" --demo-min-real 17 --expect-rows 128342 --expect-exec 81611 --expect-demo 46731 --expect-episodes 3200
uv run --no-sync python scripts/dataset/motion_checks.py a11 --manifest "$MANI" --latents "$LIB/wan-latents" --demo-min-real 17 --expect-rows 128342 --expect-padded 1800 --expect-episodes 3200
```

motion_checks 和 compare_wan 的 `--expect-*` 目前是参考值，主会话须在现场从 manifest 算出实际值再替换；实际取值和 `$LIB/motion/meta/store_meta.json` 的 sha256 与 `layout` 均待补。B2 各阶段之间的起跑条件：

| 阶段 | 起跑条件 |
|---|---|
| SigLIP | B1_GATE=PASS 且 m8v-hash 完成 |
| Wan | DELIVERY=PASS、finalize EXIT_CODE=0、GPU 0–7 无计算进程 |
| encode / oracle / motion pack | G2a（TRAIN_ES_BOUND headroom≥0）PASS |
| 删 features | xgrid 两行都 PASS |

长尾上传有两项与本库直接相关：encoder latent 库（`m8v-latent-upload`）与 VLA 新库（`m8v-trainset-upload`）。两者的命令、闸门和判定行按计划 runbook 原文执行，两次大回读串行：先跑 latent 上传，`EXIT_CODE=0` 后删其 verify 目录，再起新库上传，新库上传按口径 38 分批回读。

## 内容与结构保证

- **下载**：`fetch_mixed8_h5.py fetch` 先下载 bucket 的 `SHA256SUMS` 与 `MANIFEST.json`，要求 `record_dataset_*.h5.tar.xz` 恰为 100 个，并与 MANIFEST 的 summary.cells 一一对应。随后逐包核对 sha256，解包到平铺的 `extracted/` 后删包。续跑时，新下载的 SHA256SUMS 必须与已落盘的逐字节相同，否则拒绝续跑。每个任务输出一行 `FETCH_VERIFY`。
- **合并**：merge 保持逐字拷贝（口径 34）。原单 episode H5 经 h5py `Group.copy` 物理拷入合并 H5，只改顶层 episode 名；`--check-source-sha256` 独立重算源 sha256 回写 map。plan 预检只覆盖 `--tasks` 所选任务的文件，manifest 全局身份和 3200 计数校验保留。
- **合并验证**：`verify --level full` 按链接遍历，逐对象核对形状、dtype 和值（浮点用 `equal_nan`），并核对 chunks、compression 等布局。它还要求 attrs 为空、名字集合一致，只接受 HardLink，拒绝软链、外链和硬链接别名。这里不承诺 H5 容器字节完全相同。
- **SigLIP 之后**：finalize 用 sha256 级输入指纹和 1024 帧抽样复算；4×4 与 8×8 各自 verify 零失配；xgrid 做 512 帧图像抽查，并核对 motion_pos 跨网格位置。
- **motion**：motion 表 verify 零失配；用 `compare_wan.py` 与独立 oracle 严格对拍 latent 与 token；motion_checks 五项（a6set、a7、a9set、a10、a11）覆盖同源、字节、索引集、行数和补帧。
- **窗口预算**：`TRAIN_ES_BOUND` 以 `motion_store.max_visible_count` 为准，预算 160 不改。
- **确定性对拍 FAIL 不重跑**：xgrid、compare_wan、motion_checks、DATA_IDENTITY、跨分片 git_commit 一致性检查任一 FAIL，就交证据并停下问用户（停点 9）。

## 用户决策与关联限制

编号行的用户原话从计划「一、已定口径」逐字摘录；「开工令」与「计划内调整」两行不在该表中，取自主会话的执行记录。右列是与本库相关的落地，按计划正文转述。

| 口径 | 用户原话 | 本库相关落地 |
|---|---|---|
| 开工令 | 「/scratch/hongze/MotionJEPA-backup/1016-mixed8-vla-plan.html 开工 有关键问题问用户但尽可能不要阻塞」 | 按 11 项停点执行，清单外不主动请示 |
| 1 | 「纯探索性 定一个high level的计划 训练完需要立刻启动两件事 encoder的mp4可视化 和 vla预处理的构建 评估可行性 以及磁盘的内容等问题 然后构建玩预处理直接开始512 和 512+motion训练 ckpt不再进行挑选 encoder的ckpt就挑最后一个」 | encoder 只用 `checkpoint_epoch_72.pt` |
| 2 | 「vla数据集还是用的encoder训练的那个数据集就是你相当于要重新做一次预处理」 | 从 16 任务 × 200 集源 H5 全新建库，不复用旧库 |
| 3 | 「我说的512和512+motion训练 是用的8帧8*8feature」「我也需要8乘8的」 | 两条训练读 `framesamp-8x8/`；motion 用 `…-8x8-motion-mixed8.yaml` |
| 5 | 「避免影响正在进行的训练 只要不报错都可以 可以暂时不考虑吞吐问题」 | A 段只用 CPU，merge `--procs 16` |
| 12 | 「VLA 训练集范围全部」 | 3200 集全部入库 |
| 13 | 「4×4 framesamp 打包同意」 | 4×4 `framesamp/` 打包并保留 |
| 14 | 「推荐先不删，盘够。如果盘不够了立刻删除」 | `data-raw/` 暂不删；应急时第一个删，前提是 DELIVERY=PASS、无读者、建库后 TRAIN_ES_BOUND 已 PASS |
| 16 | 「这个也是我现在需要重点检查的部分 如果还是在160以内就暂时不用管」「23全部同意」 | `TRAIN_ES_BOUND` 为硬闸门；仿真侧不阻塞 |
| 17 | 「23全部同意」 | 库名 `16task-mixed8-200-3200ep-e3c7564f`，raw 名 `16task-mixed8-200-20261014` |
| 24 | D1：「1同意」 | 合并 H5 在 motion verify PASS 且新库上传 PASS 之后才删 |
| 25 | D2：「2 encoder上传可以错开 往后推 飞出方案」；第二轮：「你直接给我这个prompt然后我直接输给他然后直接让他放弃做这件事情这件事情由我们这边做」 | latent 库上传转交本计划，原样运行其驱动；features 删除后起，两次大回读串行 |
| 26 | D3：「3 同意」 | B1 不等 encoder 任务 done，判据全部满足即起（第 27 条例外） |
| 27 | D4：「4 下载失败问我同意但是能做的继续做 上传失败pending decsion同意」 | 下载断点续传重试 3 次，仍失败停点 4；上传失败写 pending_decisions |
| 34 | D11：「11评估一下严重性如果没有特别需要改变的你可以继续你需要报告用户。否则的话就要停下来」 | merge 逐字拷贝不改；plan 预检 FAIL 涉及语义即停点 5，只涉及命名、dtype、路径则改下游并报告 |
| 35 | 第三轮：「3 c」（TRAIN_ES_BOUND headroom<0 怎么办） | headroom<0 停下问用户，只暂停 motion 相关阶段 |
| 36 | 第三轮：「4全部同意 但是A1 拆成两半，让下载先走。什么意思？」；第四轮：「同意a1拆分还是先全部回写入计划并且push」 | A1 拆为 A1-merge 与 A1-train；HEAD 冻结窗口（P4）；本档案名（P6） |
| 38 | 第四轮之后：「a. 分批回读（推荐） 同意」 | 新库上传分批回读，单批 ≤100 GiB |
| 40 | 「a 同意 用 .git/info/exclude」 | 训练在主检出起；3 个他人未跟踪文件进主检出 exclude，排除记录写入 512 的训练档案，不写本档案 |
| 42 | 「这些都是很细节的内容了你自己来决定 然后你直接自己修改不用再走子代理的修改和推送了。」 | HEAD 冻结窗口起点前移到 SigLIP；建库档案须在 SigLIP 前提交 |
| 43 | 「a同意」 | build 交接只留 try20 短测、DATA_IDENTITY 与建库档案；512 与 motion 的 preflight 都移到 train 开工前自检；build done 的 handoff 在 V11.24Beta 之前提交 |
| 计划内调整 | 「全量下载是否和 A3 并行 同意」 | `m8v-fetch` 与 `m8v-fetch-smoke` 并行；`ADJUST_REVIEW=PASS scale=same` |

**关联限制**：

- **停点**：本库涉及停点 2（headroom<0）、4（下载失败）、5（plan 预检语义 FAIL）、7（磁盘应急走完仍不足）、8（修复轮次耗尽、E72 断言 FAIL）和 9（确定性对拍 FAIL）。
- **GPU 被他人占用**：不抢卡、不杀进程，相关 GPU 阶段暂缓。
- **tmux 会话意外消失**：日志里没有 `EXIT_CODE=` 的，一律视为 FAIL。
- **冻结期间**：policy 仓库需要的改动推迟到窗口之后。
- **build done 的 handoff commit**：放在冻结窗口之后、V11.24Beta 之前提交（口径 43）。

## 归档与验收计划

**records/ 拟归档**（第二段补齐，只放 git 还原不出的东西）：

- 16 份 episode map、`merge_plan.json`、合并完成标记、source pin。
- `episode_manifest.json`、`input_manifest.json`。
- SigLIP 各 shard 的 fingerprint、finalize 结果。
- 4×4 与 8×8 的 `store_meta.json`、motion 的 `store_meta.json`。
- norm_stats 的 sha256。
- oracle 的汇总报告。
- 各阶段清洗后的 `*.summary.log`：用 `tr '\r' '\n' | grep -vE '%\|'` 清洗；oracle 分片按 `SHARD_EXIT_CODE` 数核验。
- 两条 try20 的判定行与显存采样峰值。
- `DATA_IDENTITY` 判定行。

不归档 H5、features、latent、权重或任何 .sh/.yaml 拷贝；命令原文已写在上文代码块里。

**验收判定行**（全部 PASS 才算建库通过；未执行的项不标通过）：

| 阶段 | 判定行 | 状态 |
|---|---|---|
| B1 | `B1_GATE=PASS …`、`E72_EMA_ONLY=PASS`、`G3=PASS …` | 已 PASS |
| 下载 | `FETCH_VERIFY=PASS` × 16，合计 100 包 | 已 PASS |
| 合并 | `PLAN_OK tasks=16 selected=3200 …`；`MERGE_TASK=PASS` × 16；`MERGE_VERIFY=PASS level=full tasks=16 episodes=3200 … mismatches=0`，timesteps=2,145,548 | PLAN_OK 已出，其余待补 |
| manifest | `episode_manifest.json`、`input_manifest.json` 产出 | 待补 |
| 窗口预算 | `TRAIN_ES_BOUND=PASS episodes=3200 tasks=16 max_windows=<n≤123> budget=160 headroom=<h≥0>`；`ES_BOUND=PASS|SKIP`（非阻塞） | 待补 |
| SigLIP | `STAGE_DONE stage=siglip workers=4 items=3200` | 待补 |
| finalize | `全部检查通过 PASS`、`FINALIZE_EXIT_CODE=0` | 待补 |
| 打包 | `VERIFY_PACK=PASS … mismatches=0`（4×4、8×8）、`IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、`MOTION_POS_XGRID=PASS` | 待补 |
| norm / C1 | `DATA_IDENTITY=PASS norm_sha=<sha> steps=<n> samples=<m> manifest_sha=e3c7564f…` @ `$LOGS/m8v-identity.log` | 待补 |
| Wan / encode | `STAGE_DONE stage=wan workers=8 …`、`STAGE_DONE stage=encode workers=8 …` | 待补 |
| oracle | `SHARD_EXIT_CODE=0` × 8，aggregate `EXIT_CODE=0` | 待补 |
| motion | `VERIFY_MOTION=PASS scanned=<n> mismatches=0`、`WAN_BITEXACT=PASS`、`ENCODER_BITEXACT=PASS`、`A6_SAMESOURCE=PASS`、`A7_BYTES=PASS`、`A9_INDEXSET=PASS`、`A10_ROWS=PASS`、`A11_PAD=PASS` | 待补 |
| try20 | 两次 `EXIT_CODE=0`，两行 `MOTION_4GPU_SMOKE=…` | 待补 |
| 上传 | latent 库与新库上传：`RESULT=PASS`、读回零差、公开、`全部完成`、`EXIT_CODE=0`；新库另要 `READBACK_PEAK_BYTES ≤ 107374182400` | 待补（长尾） |

第二段还要逐项写出：各阶段起止时刻（美东）与耗时、产物字节量、窗口 HEAD 的 40 位 sha、canonical_order、删除清单，以及冻结窗口内 git_commit 一致性的核对结果。

## 20 步训练可读性检查计划

两条 20 步试训同时作为 4 卡短测闸门，安排在 motion 对拍全部 PASS 之后、正式训练之前，在 policy 主检出用正式 runner 跑。runner 的用法是 `train-*.sh <run_name> [额外参数...]`，run 名之后的位置参数会原样追加到 train.py 参数末尾。命令照抄计划 runbook B4 段：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
tmux new-session -d -s m8v-try20-512 "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false bash scripts/dataset/hf_export/mixed8_200_runner/train-512-80k.sh m8v-try20-512 --num-train-steps 20 --log-interval 1 --no-wandb-enabled"
tmux new-session -d -s m8v-try20-motion "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false bash scripts/dataset/hf_export/mixed8_200_runner/train-motion-80k.sh m8v-try20-motion --num-train-steps 20 --log-interval 1 --no-wandb-enabled"
tmux new-session -d -s m8v-try20-512-mem "nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8v-try20-512.mem.csv"
tmux new-session -d -s m8v-try20-motion-mem "nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8v-try20-motion.mem.csv"
```

- **日志与采样会话**：runner 自写 `$LOGS/m8v-try20-{512,motion}.log` 与 `EXIT_CODE=`。两条结束后，分别用 `tmux kill-session -t '=m8v-try20-512-mem'` 和 `tmux kill-session -t '=m8v-try20-motion-mem'` 停掉采样会话，一次一个。
- **配置**：两条都用 `mme_vla_suite_b128_80k`，全局 batch 128，CLI 覆盖 `--fsdp-devices 4 --num-workers 8`，经环境变量透传。512 用 `perceptual-framesamp-modul-8frame-8x8.yaml`，motion 用 `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml`。数据读本库 `framesamp-8x8/`，统计量用本库 norm_stats。
- **判定行**：由主会话根据日志（`TRAIN_MESH` 行、20 行 step loss）和 `.mem.csv` 生成，两条各一行：`MOTION_4GPU_SMOKE=PASS|FAIL run=<512|motion> steps=20 exit=<c> finite_losses=<k>/20 peak_mem_gib=<x> limit_gib=72 mesh=<TRAIN_MESH 行原文>`。
- **FAIL 的认定**：只有 OOM 或运行报错（`EXIT_CODE≠0`、Traceback、out of memory / RESOURCE_EXHAUSTED、loss 非有限）才算 FAIL。单卡峰值超过 72 GiB（大于 73728 MiB）但没有 OOM，仍判 PASS，只多输出一行 `MOTION_4GPU_SMOKE_WARN run=<512|motion> peak_mem_gib=<x> limit_gib=72`。
- **FAIL 的处理**：某条 FAIL，就给该条补跑 8 卡 20 步（run 名 `m8v-try20-<512|motion>-8gpu`，不设 `FSDP_DEVICES` / `NUM_WORKERS`），并只把这一条改为 8 卡串行；两条都 FAIL 才全部串行。能定位为 CLI、数据、环境类的报错，要先修再测，不能用改 8 卡代替修复。
- **记录与清理**：结果记入本档案 records，之后删除两个 try20 run 目录。20 步覆盖不到 ckpt 保存和 EMA，这两项由正式训练的前 100 步复查。

本节结果全部待补。

## 本轮 tmux 会话清单

以下只列建库相关、名字以 `m8v-` 开头的会话。只清理本表中实际记录的完整会话名，一次一个，用 `tmux kill-session -t '=<完整名>'`；不按前缀清理，也不动其他任务的会话。

| 会话 | 阶段 | 起跑（美东 2026-10-16） | 状态 |
|---|---|---|---|
| `m8v-fetch-smoke` | A3 两任务小样本（G1） | 约 08:06 | 结束，`EXIT_CODE=0` |
| `m8v-b1gate` | B1 起跑判据与 encoder 拷贝 | 约 08:07 | 结束，`EXIT_CODE=0` |
| `m8v-viz` | B1 mp4，在 MotionJEPA-backup 运行，GPU 0–3 | 待补 | 运行中；`DELIVERY=PASS` 待补 |
| `m8v-fetch` | 全量下载解包 | 08:11 | 结束（约 09:09），`EXIT_CODE=0` |
| `m8v-merge` | 全量 plan、merge、verify、cleanup | 09:09 | 运行中，结果待补 |

后续拟起的会话：`m8v-hash`、`m8v-esbound`、`m8v-siglip`、`m8v-finalize`、`m8v-wan`、`m8v-pack4`、`m8v-pack8`、`m8v-encode`、`m8v-oracle`、`m8v-motion`、`m8v-try20-512`、`m8v-try20-motion`、`m8v-try20-512-mem`、`m8v-try20-motion-mem`，以及长尾的 `m8v-latent-upload-smoke`、`m8v-latent-upload`、`m8v-trainset-upload-smoke`、`m8v-trainset-upload`。实际名称、起跑时刻和启动 HEAD 在 result.md 补记。
