# 原版口径 80k：完整库 + 纯 counting 库，4 卡 + 4 卡并跑计划

创建日期 2026-09-25（America/New_York）。本文件即仓库根计划文件；尚未修改代码、配置或启动任何任务。
用户原话：「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」。
本轮澄清（AskUserQuestion）：完整 = 现有 `4task-v2-1600ep-604f16da` 库；history = `perceptual-framesamp-modul.yaml`；对拍 = 本仓库自洽 + 对上游两者都做；counting 数据 = 公开 H5 的四个 counting 任务新建库；counting 的 norm_stats 对其数据重算；磁盘由用户自己清理，计划只设闸门。

运行环境：**环境 B（AWS 单机）**。8×A100-80GB 全空；`/dev/md0` 起初只剩 402G（95% 已用），用户清理后复查为可用 **1.2T（84% 已用）**。分支 `v2-motionmem` 已与 origin 同步，远端改动由 `7e44706` 合并，磁盘清理记在 `2216672`。

## 第一部分（给人看）

### 1. 要跑什么：两个 run 都是「原版配置原样」，只换数据

**「原版」就是 `config.py::_CONFIGS` 的 `mme_vla_suite` 条目，本仓库一字未改。** 它的注释写明「= 上游 ecf086c 官方口径」；README 与 `scripts/training/finetune_mme_vla_suite.sh` 的原版命令也正是这一条目加 4 卡。固定值：batch 64、`num_train_steps=80_000`、`CosineDecaySchedule(warmup 10_000, peak 5e-5, decay_steps 100_000, decay 5e-5)`（peak 等于 decay，warmup 之后恒为 5e-5）、`AdamW(clip 1.0)`、EMA 0.999、seed 42、`fsdp_devices=4`、`num_workers=4`、`save_interval=keep_period=10_000`、`pi05_base` 初始化。history 用 `perceptual-framesamp-modul.yaml`：budget 512、每帧 16 token（4x4）、最多 32 帧、modulation、`memory_token_dim=1024`。`git diff ecf086c HEAD -- <该 yaml>` 为空，与上游逐字一致。

**CLI 只覆盖路径类参数，不动任何超参。** 这与 1600ep 建库档案里 20 步可读性检查（`docs/dataset-build-doc/4task-v2-1600ep-604f16da/launch.md` 末节）用过的写法相同：`--exp-name`、`--dataset-path <lib>/framesamp`、`--assets-base-dir`、`--data.assets.assets-dir`、`--data.assets.asset-id robomme`、`--checkpoint-base-dir v1-store/train-runs`、`--weight-loader.params-path v1-store/models/openpi-assets/checkpoints/pi05_base/params`（默认值指向 `~/.cache/openpi`，必须覆盖）、`--model.use-history --model.history-config perceptual-framesamp-modul.yaml`。另外设环境变量 `MMEVLA_FRAMESAMP_SOURCE`、`MMEVLA_FRAMESAMP_MANIFEST`。`project_name` 沿用条目默认值 `openpi`，W&B 开启，目录类变量指向 `v1-store/`。

| | 完整 run（GPU 0–3） | 纯 counting run（GPU 4–7） |
|---|---|---|
| 拟定 run_name | `v2-orig-1600ep-modul-b64-80k` | `v2-orig-counting400ep-modul-b64-80k` |
| 数据 | 现有 `4task-v2-1600ep-604f16da`（BinFill、RouteStick、VideoUnmaskSwap、VideoRepick 各 400 集，605611 执行样本） | 新建 `4task-counting-pub-400ep`（BinFill、PickXtimes、SwingXtimes、StopCube 各 100 集，公开 H5，189035 帧） |
| packed store | `framesamp`（`framesamp-4x4-v1`，verified/full，已存在） | 新建 `framesamp`（4x4） |
| norm_stats | 已有 `train-assets/mme_vla_suite/4task-v2-1600ep-604f16da` | 新算 `train-assets/mme_vla_suite/4task-counting-pub-400ep` |
| 其余 | `mme_vla_suite` 原样 | 同左 |

### 1.1 与原版的区别：训练超参零差异，差异全在数据侧

**训练超参一项都不改。** 下表的「原版」取自上游 `ecf086c` 的 `mme_vla_suite` 条目，以及 README / `finetune_mme_vla_suite.sh` 的原版命令。本轮两个 run 用的是当前 HEAD 的同名条目。三处比对都已实测：

- 条目本身：从 `git show ecf086c:src/mme_vla_suite/training/config.py` 抽出条目，与 HEAD 的条目去掉行尾空白后 `diff` 为空（`ENTRY_SAME`）。
- 默认值：`class TrainConfig` 的定义同样 `diff` 为空，所以 seed、log_interval 等继承的默认值也相同。
- history yaml：`git diff ecf086c HEAD -- perceptual-framesamp-modul.yaml` 为空。

CLI 只传路径类参数，不传任何超参覆盖。

| 训练参数 | 原版（ecf086c + README 命令） | 本轮两个 run | 是否相同 |
|---|---|---|---|
| 具名配置 | `mme_vla_suite` | `mme_vla_suite` | 相同（条目逐字同） |
| history yaml | `perceptual-framesamp-modul.yaml`（budget 512，16 token/帧，最多 32 帧，modulation，memory_token_dim 1024） | 同一文件 | 相同（diff 为空） |
| global batch | 64（`--batch-size=64`） | 64 | 相同 |
| GPU 数 / FSDP | 4 卡 / `--fsdp-devices=4`（每卡 16 样本） | 4 卡 / 4 | 相同；硬件由 4×A40-40GB 换成 4×A100-80GB，数值计算不受影响 |
| num_workers | 4 | 4 | 相同 |
| 总步数 | 80000 | 80000 | 相同 |
| 学习率 | Cosine：warmup 10000，peak 5e-5，decay_steps 100000，decay 5e-5（warmup 后恒为 5e-5） | 同左 | 相同 |
| 优化器 | AdamW，clip_gradient_norm 1.0，其余用 openpi 默认值 | 同左 | 相同 |
| EMA | 0.999 | 0.999 | 相同 |
| seed | 42（TrainConfig 默认） | 42 | 相同 |
| 冻结 | `HistoryPi0Config().get_freeze_filter()` | 同左 | 相同 |
| 初始化 | `pi05_base/params`（`CheckpointWeightLoader`） | 同一权重，路径改指 `v1-store/models/openpi-assets/…`，按 `external-assets-lock.md` 的 sha256 核对 | 权重相同，仅路径不同 |
| 模型 | pi05，action_horizon 20，use_history，bf16 | 同左 | 相同 |
| log / save / keep | 100 / 10000 / 10000 | 同左 | 相同 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.95 | 0.95 | 相同 |
| W&B project | `openpi`（默认） | `openpi` | 相同 |

**真正的差异在数据侧，共 4 项，都是你本轮拍板的选择或实现形式：**

| 差异项 | 原版 | 完整 run | 纯 counting run |
|---|---|---|---|
| ① 训练数据内容 | 公开 `Yinpei/robomme_data_h5`：16 任务 × 100 集 = 1600 集 | 私有新录 4 任务 × 400 集 = 1600 集（BinFill、RouteStick、VideoUnmaskSwap、VideoRepick） | 公开 H5 中的 4 个 counting 任务 × 100 集 = 400 集 |
| ② 样本量与 epoch 数（80k × 64 = 5.12M 样本） | 476857 个执行样本，约 **10.74 epoch** | 605611 个，约 **8.45 epoch** | 189035 个，约 **27.08 epoch** |
| ③ norm_stats | README 附带的 `assets/norm_stats.json`（sha `f332bbd3…`，按 16 任务统计） | 1600ep 库自算（sha `856c75ea…`） | 对 counting 库新算 |
| ④ 数据格式与加载器 | pkl + features（`source` 格式），上游 `dataset.py` 逐样本读取，collate 走 pickle | packed `framesamp` 4x4 store 加 `FrameSampDataset`，collate 走 torch 共享内存 | 同左 |

上表样本数由 16task-scan 与 1600ep 库的 manifest 按 `Σ(num_timesteps − exec_start_idx)` 实算；counting 任务没有 demo 段，所以样本数等于帧数。

- ①②③ 会改变训练结果。这是你的数据选择带来的预期差异，不属于实现偏差。由于 epoch 数不同，两个 run 与原版以及彼此之间都不能直接比较 loss。
- ④ 不应改变任何数。第 2 步 2a 的上游逐位对拍，就是要证明「上游代码读 `source`」与「本仓库读 packed store」在前 100 步逐位相同。

**训练入口与代码的实现差异（不改超参，要靠对拍证明结果等价）：**

- **入口**：上游 `scripts/train.py` 的 `__main__` 会先跑 12 步 tentative 预热，再用 `--overwrite` 重跑正式训练。本仓库 `scripts/training/train.py` 只跑一遍，没有这 12 步。`v1-upstream-eq` 曾证明两者正式轨迹逐位相同，但那次测的是 context 变体。
- **代码**：那次对拍（`f641f40`）之后，`history_pi0.py`、`percep_mem.py`、`framesamp_dataset.py`、`data_loader.py`、`train.py` 又经历了 9 个 commit，包括 V9.1 的 4x4/8x8 两档库、V9.10 的 collate 共享内存，以及 V10.0/V11.0 的 motion 接线与形制守卫（motion 在本轮关闭）。每个改动都有各自的对拍，但**当前 HEAD 在原版 modul 4x4 配置上从未直接对过上游**。2a 补上这一环：不通过就不起正式训练。
- **缓存与产物路径**：JAX 编译缓存、checkpoint、W&B 目录、pi05_base 路径都改到 `v1-store/` 下，与原版的 `~/.cache`、`runs/` 不同。这属于环境 B 的存放规定，不影响计算。

**两个 run 同时起跑，各占 4 卡，互不共享任何东西。** `train.py` 用 `make_mesh(fsdp_devices)` 在可见的 4 张卡上建 mesh `(1,4)`，每卡 16 个样本。历史上有 4 卡并跑先例（`repro-4a100-fsdp4` 在 GPU 0–3 和 4–7 各跑一档）。collate 用 torch 匿名共享内存，不会撞名。JAX 编译缓存、记录目录、日志和 checkpoint 根逐项分开。生产 runner（`run_modul*.sh`、`modul_launch_contract.py`）写死了 8 卡、b128 和 fsdp8，本轮**不用也不改**它们，另写一个 4 卡 runner。

**checkpoint 与耗时。** 按 10k 间隔保存，两个 run 各留 `10000…70000, 79999` 共 8 份，每份约 12G，合计约 192G。修复 collate 共享内存后，本机还没有 4 卡 b64 的步时实测：旧的 4 卡数字（2.34 s/步，b128，修复前）不能用，ETA 以第 2 节测速为准，预估每个 run 在 1 天量级。

### 2. 执行顺序：备好 → 建 counting 库 → 对拍 → 测速 → 正式 80k

**第 0 步：闸门检查。** 分支已与 origin 同步；开工前再确认 `git status -sb` 首行没有 ahead/behind。磁盘闸门：**建库前 `df` 可用空间 ≥ 450G，正式起跑前 ≥ 300G**，不满足就停下交给你清理。本计划不删任何已有产物。

**第 1 步：新建 counting 库（约 1 小时，沿用 1600ep 建库链路，只换输入）。** 原始 H5 在 `/scratch/hongze/robomme_data_h5/record_dataset_{BinFill,PickXtimes,SwingXtimes,StopCube}.h5`，共 126G。`finalize_checks.py hash-inputs` 会收录目录里全部 `*.h5`，所以先在 `/scratch/hongze/robomme_data_h5_counting4/` 为这四个文件建**硬链接**：同盘零额外占用。不用 symlink，因为主副本 `v1-store` 不许外链。之后依次执行：

1. `scan_manifest.py build --tasks BinFill,PickXtimes,SwingXtimes,StopCube --episodes-per-task 100`
2. `hash-inputs`
3. `run_local.py --stage siglip`（GPU 0–7）
4. `finalize_checks.py check --spot_check 1024`
5. `pack_framesamp_store.py pack`，默认 4x4 布局
6. `pack_framesamp_store.py verify`
7. `compute_norm_stats.py`（CPU）

不建 8x8，不做 motion。预计新增约 155G（source 按 1600ep 实测每帧约 0.75MB 推算约 142G，4x4 store 约 12G）。收尾时在同一库上跑 20 步可读性训练（原版配置，`--num-train-steps 20`），作为第 2 步的前置。留档写在 `docs/dataset-build-doc/4task-counting-pub-400ep/`。

**第 2 步：对拍，两块都在确定性档下逐位判定。** 所有对拍 run 都设 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`，配置就是正式的 `mme_vla_suite`（b64 / fsdp4 / workers 4 / seed 42 / modul），只把步数改成 100，另加 `--log-interval 1 --save-interval 50`。每侧记录 100 步的五项标量（loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm 的 `float.hex()`），以及第 50、99 步参数状态的 strict sha256 摘要。复用的量具是 `scripts/training/tests/entry_equiv.py`（run/judge）：它以前做过「上游 `ecf086c` vs 本仓库」1000 步逐位对拍（`v1-upstream-eq`，`ENTRY_EQ=PASS`），`save_state` 已替换成只写摘要、不落权重。

- **2a 对上游（两个库各一组）**：A 侧用 `git worktree add --detach v1-store/worktrees/orig-ecf086c ecf086c`，配独立 `.venv`，入口是上游 `scripts/train.py` 的官方 `__main__`（含 12 步 tentative 预热，加 `--overwrite`），读 `<lib>/source`，即上游 pkl + features 格式。B 侧是本仓库 HEAD 的 `scripts/training/train.py`，读 `<lib>/framesamp` packed store。两侧 argv 只允许 5 处差异：入口、exp-name、dataset-path、checkpoint 根，以及 A 侧的 `--overwrite`。完整库这组在 GPU 0–3、counting 库这组在 GPU 4–7 同时跑，每组内部 A、B 串行。判定行：`ENTRY_SCALARS steps=100 keys=5 hex_mismatch=0`、`ENTRY_STATE_DIGEST mismatch=0`、`ENTRY_RESOLVED_CFG mismatch=0`、`ENTRY_PROVENANCE=PASS`，合起来即 `ENTRY_EQ=PASS`。这些 run 没有历史锚点，anchor 打 SKIPPED。**这一块证明：我们的训练入口和 packed 数据链，与上游原版代码读原版格式数据，在前 100 步逐位相同。**
- **2b 本仓库自洽（4+4 并跑不改变任何数）**：只用 B 侧入口跑三次。S1 = 完整库单独跑在 GPU 0–3；S3 = counting 库单独跑在 GPU 4–7；S2 = 两者同时跑。用 `entry_equiv.py judge --expect-tentative-a 0` 比较 S1 与 S2 的完整侧、S3 与 S2 的 counting 侧，要求 `hex_mismatch=0`、状态摘要 `mismatch=0`。S1 和 S3 本身就是 2a 的 B 侧，可以直接复用，所以只需加跑一次 S2。**这一块证明：并跑、共享主机 CPU/内存/磁盘不会改变任一侧的数据顺序或计算结果。**

**第 3 步：正式环境测速（不开确定性档，4+4 同时各跑 300 步）。** 用正式 runner、正式环境变量，两个临时 run 同时跑。记录步时、samples/s，以及 GPU 利用率：`nvidia-smi -lms 500` 流式采样，稳态窗口内给出均值、0% 采样占比、慢步/非慢步分层均值，不以中位数当结论（AGENTS 第 16 条）。同时记录主机内存和 `/dev/shm` 峰值，由此估算两个 run 的 ETA。「完全参照原版」意味着 `num_workers=4` 不改。即使测出数据路径是瓶颈，也只报告给你，不自行调参。测完删除临时 run。

**第 4 步：正式 80k（clean HEAD，同时起跑两个 detached tmux）。** 会话名为 `orig80k-full-<UTC>` 与 `orig80k-count-<UTC>`，会话清单写进 launch.md，事后只按这个清单逐个 `kill-session`。每个会话都用 `PYTHONUNBUFFERED=1`、`set -o pipefail`、`tee` 输出日志，结束写 `EXIT_CODE=`；每份日志挂一个 Monitor，管道每一级都行缓冲。起跑前两个 run 各自提交 `docs/training-doc/<run>/launch.md`，写明 commit、完整命令、数据/资产 sha256、会话名。训练期间主副本锁定只读，开发工作转到 `-temp` 副本。

**完成验收（每个 run 独立判）**：唯一终态 `EXIT_CODE=0`；checkpoint 恰为 `10000,20000,…,70000,79999` 共 8 份；log100 共 800 条，五项标量全部有限；最终 `79999/params` 能真实加载，全部叶都是有限值。结果写进 `result.md`：全程步时、samples/s、GPU 利用率均值。策略评估不在本轮范围内。

### 3. 两条核心保证

**保证一：两个 run 的训练配置就是原版，没有夹带差异。** 分三层核：

- 配置文件层：history yaml 用 `git diff ecf086c HEAD` 证明为空；`mme_vla_suite` 条目用 `git diff ecf086c HEAD -- src/mme_vla_suite/training/config.py` 限定到该条目，同样证明未改。
- 实际命令层：2a 的 `ENTRY_RESOLVED_CFG mismatch=0` 证明，本仓库解析出的 resolved config 与上游在同一 argv 下逐字段相同，白名单只放行 exp_name、dataset_path、checkpoint 根和 overwrite。
- 正式起跑层：runner 在起跑前用 CPU 重新解析即将执行的同一份 argv，断言 batch 64、steps 80000、warmup 10000、lr 5e-5、fsdp 4、workers 4、seed 42、save/keep 10000、ema 0.999，并且可见卡数恰为 4。任一不符就拒绝起跑。

**保证二：数据链与原版逐位相同，并跑互不干扰。** 同样分三层：

- 输入层：新库须通过 `finalize_checks` 与 `pack verify` 的 PASS。
- 训练层：2a 上游对拍逐位相同，这同时证明 packed 4x4 store 与原版 source 格式在训练看到的内容上一致。
- 并跑层：2b 的 S1/S2/S3 逐位相同。

对拍不过（任一 mismatch ≠ 0）就停在对拍阶段，按四象限判读后交你裁决，不起正式训练。

## 第二部分（技术细节，供 agent 追踪）

### A. 代码改动（全部是新增或参数化，不改训练源码）
1. `scripts/training/tests/run_entry_equiv.sh`：用环境变量参数化，默认值保持历史行为逐字不变。要参数化的有 `HISTORY_YAML`、`BATCH`、`FSDP`、`GPUS`、`SAVE_INTERVAL`、`DATA_A`、`DATA_B`、`ASSETS_DIR`、`ANCHOR_SHA256`（可为空）、`FRAMESAMP_SOURCE/MANIFEST`（只注入 B 侧）、`MMEVLA_JAX_CACHE_DIR`（B 侧）。如果 `common_args` 的 `--assets-base-dir` 只给 base，要改成 `--data.assets.assets-dir` 加 asset-id。
2. 上游 A 侧的 JAX 缓存写死在 `~/.cache/jax_{exp_name}`（`ecf086c:scripts/train.py` 中 `jax_compilation_cache_dir`）。禁止覆盖 HOME，所以起跑前把 `~/.cache/jax_<expA>` 建成 symlink，指向 `v1-store/cache/jax/<expA>`；跑完删除 symlink 和缓存。
3. 新增 `scripts/training/prod/run_orig80k.sh <run_name> <gpus> <lib> <assets_dir>`，负责：
   - 设置 `CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`MMEVLA_JAX_CACHE_DIR=v1-store/cache/jax/<run>`、`WANDB_DIR/CACHE/CONFIG_DIR`、`UV_CACHE_DIR`、`MMEVLA_FRAMESAMP_*`；
   - 运行 preflight：复用 `scripts/training/preflight_train_launch.py` 的通用检查（主副本、clean HEAD 字面量、yaml/norm_stats sha、run-root 不存在）。如果它的 `--launch-mode prod` 绑定了 modul 8 卡契约，就不传该参数；
   - 用 CPU 解析 argv 做断言（保证一第三层）；
   - 断言指定的 4 卡空闲；
   - 执行 `uv run scripts/training/train.py mme_vla_suite …`，写 tee 日志并以 `EXIT_CODE=` 结尾。
4. 新增 `scripts/training/tests/check_orig80k_completion.py`：实现第 2 节第 4 步的完成验收，输出 `RUN_COMPLETED=PASS run=<name> checkpoints=8 final=79999`。
5. 最小单测：runner 参数断言和完成验收器的负例（缺 checkpoint、NaN 指标、卡数≠4），放 `scripts/training/tests/`，用 `uv run pytest` 跑，5 分钟内完成。

### B. 建库命令骨架（沿用 `docs/dataset-build-doc/4task-v2-1600ep-604f16da/launch.md` 的「命令与配置还原」）
```bash
source scripts/dataset/paths.sh; export UV_CACHE_DIR="$V1_STORE/cache/uv" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RAW=/scratch/hongze/robomme_data_h5_counting4   # 四个 H5 的硬链接目录（ln，非 symlink）
LIB=$V1_STORE/datasets/4task-counting-pub-400ep; TASKS=BinFill,PickXtimes,SwingXtimes,StopCube
MANI=$LIB/meta/episode_manifest.json; INMANI=$LIB/meta/input_manifest.json
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --tasks "$TASKS" --episodes-per-task 100 --num_shards 1 --out "$MANI"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
```
- 前置：公开 H5 的 schema 已被 `16task-scan` 用同一套 `scan_manifest` 扫通（1600 集，缺失 0）。每个阶段都在 detached tmux（前缀 `cnt-`）里跑，按阶段留档。
- 输出根起跑前先 `ls -ld`，确认不存在。本轮不用任何 `--force`。

### C. 对拍执行细节
- A 侧 worktree 建在 `v1-store/worktrees/orig-ecf086c`，`UV_LINK_MODE=copy`，`uv sync` 用 `UV_CACHE_DIR=v1-store/cache/uv`。run 侧带 `--expect-root <worktree>`，B 侧带 `--forbid-root <worktree>`，防止 A 侧误 import 到 B 侧代码而假通过。
- 判定：`entry_equiv.py judge --expect-steps 100 --expect-tentative-a 12 --expect-state-steps 50,99 --expect-head-a ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b --expect-head-b <实施 HEAD>`。2b 用 `--expect-tentative-a 0`，比较两个 B 侧记录。
- 首先做 P1 冒烟：A 侧 `STEPS=2`，确认上游读得了新库的 `source`，以及上游 `modul` 与 `framesamp` 的组合能跑通，之后才上 100 步。
- 留档写在 `docs/training-doc/orig80k-equiv-0925/`（launch/result/records，含判定行原文）。PASS 后清理两侧临时 ckpt、worktree（`git worktree remove --force` 加 `prune`）、`~/.cache/jax_*` symlink 与缓存；FAIL 时保留，目录改名为 `.failed-<n>`。

### D. 验证与提交
- 代码改动后：`uv run pytest` 跑新增单测，加上 `run_entry_equiv.sh` 默认参数的 dry 展开，与历史 argv 逐字比对。
- commit 按「代码 → 建库留档 → 对拍留档 → 测速留档 → 两个 launch.md → 两个 result.md」分批提交。subject 用 `commitV11.9Beta:` / `docs:`（`commitV11.8Beta` 已被 `d023b72` 占用），每批逐文件 `git add`，提交后立即裸 `git push`。
- 需要你确认的事项（批准本计划即视为确认）：两个 run_name；counting 库名 `4task-counting-pub-400ep`；对拍 100 步加确定性档；测速 300 步；GPU 分配为完整库 0–3、counting 库 4–7。
