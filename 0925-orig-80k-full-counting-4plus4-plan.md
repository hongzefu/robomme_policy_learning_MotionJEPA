# 原版口径 80k：公开 16 任务全集 + 纯 counting 子集，4 卡 + 4 卡并跑计划

创建日期 2026-09-25（America/New_York）。本文件即仓库根计划文件；尚未修改代码、配置或启动任何任务。
用户原话：「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」。
本轮澄清：完整 = 公开 `Yinpei/robomme_data_h5` 的 16 任务 × 100 集 = 1600 集，与原版同一份数据（用户纠正：「完整run跑公开 Yinpei/robomme_data_h5：16 任务 × 100 集 = 1600 集」）；history = `perceptual-framesamp-modul.yaml`；对拍 = 本仓库自洽 + 对上游两者都做；counting 数据 = 同一公开集合的子集，4 个 counting 任务 × 100 集（用户原话「纯counting跑4*100 公开集合的子集合」）；counting 的 norm_stats 对其数据重算；磁盘由用户自己清理，计划只设闸门。

修订授权：用户在对抗审查后回复「同意修改」，本轮只修改本计划，不实施下述代码、建库或训练。审查基线为 `90098d7922d7e504bb19d1ad4696743ae569021d`，上游固定为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`；后文所有“拟新增”接口与判定行都是实施要求，不代表已存在或已通过。已确认的数据口径、history 与超参沿用，正式执行授权和拟定 run_name 见 D 节。

运行环境：**环境 B（AWS 单机）**。本轮复核为 8×A100-80GB，`/scratch` 位于 `/dev/md0`（XFS，本地 NVMe RAID），旧 `/data/hongzefu`、NFS 与集群 SSH 配置不存在。早前清理后记录的可用 **1.2T（84% 已用）**和“GPU 全空”只是当时快照；起跑必须重测，不能代替第 0 步预算。远端改动由 `7e44706` 合并，磁盘清理记在 `2216672`。

## 第一部分（给人看）

### 1. 要跑什么：两个 run 沿用原版训练超参，完整 run 使用公开全集，counting run 使用其子集

**「原版」就是 `config.py::_CONFIGS` 的 `mme_vla_suite` 条目，本仓库一字未改。** 它的注释写明「= 上游 ecf086c 官方口径」；README 与 `scripts/training/finetune_mme_vla_suite.sh` 的原版命令也正是这一条目加 4 卡。固定值：batch 64、`num_train_steps=80_000`、`CosineDecaySchedule(warmup 10_000, peak 5e-5, decay_steps 100_000, decay 5e-5)`（peak 等于 decay，warmup 之后恒为 5e-5）、`AdamW(clip 1.0)`、EMA 0.999、seed 42、`fsdp_devices=4`、`num_workers=4`、`save_interval=keep_period=10_000`、`pi05_base` 初始化。history 用 `perceptual-framesamp-modul.yaml`：budget 512、每帧 16 token（4x4）、最多 32 帧、modulation、`memory_token_dim=1024`。`git diff ecf086c HEAD -- <该 yaml>` 为空，与上游逐字一致。

**正式 `prod` 模式只指定 run、路径、资产与已选 history，不覆盖上述训练超参。** CLI 包括 `--exp-name`、`--dataset-path <lib>/framesamp`、`--assets-base-dir`、`--data.assets.assets-dir`、`--data.assets.asset-id robomme`、`--checkpoint-base-dir`、`--weight-loader.params-path`、`--model.use-history --model.history-config perceptual-framesamp-modul.yaml`；路径全部传本仓库 `v1-store/` 下的绝对路径。`DataConfigFactory._load_norm_stats()` 实际读取 `<assets-dir>/<asset-id>/norm_stats.json`，所以完整 run 的 `assets-dir` 是 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite`，不能多加一层 `robomme`；counting 则是该目录下的 `4task-counting-pub-400ep`。`--weight-loader.params-path` 指向 `v1-store/models/openpi-assets/checkpoints/pi05_base/params` 的绝对路径。

runner 显式加载 `scripts/training/paths.sh`，设置 `OPENPI_DATA_HOME` 等资产/缓存变量以及 `MMEVLA_FRAMESAMP_SOURCE/MANIFEST`；不能依赖之前建库 shell 的环境。`project_name=openpi`、W&B 开启。`perf` 的 300 步、`smoke` 的 20 步及对拍的 100 步覆盖仅限对应验证入口，不修改全局默认或正式 `prod` 参数；对拍另覆盖 log/save 间隔，见第 2 步。

| | 完整 run（GPU 0–3） | 纯 counting run（GPU 4–7） |
|---|---|---|
| 拟定 run_name | `v2-orig-16task-pub1600ep-modul-b64-80k` | `v2-orig-counting-pub400ep-modul-b64-80k` |
| 数据 | 新建 `16task-pub-1600ep`：公开 `Yinpei/robomme_data_h5` 全部 16 任务 × 100 集 = 1600 集，768897 帧，476857 个执行样本 | 新建 `4task-counting-pub-400ep`：同一公开集合中 BinFill、PickXtimes、SwingXtimes、StopCube 各 100 集 = 400 集，是左列的严格子集，189035 帧 = 189035 个执行样本 |
| 原始 H5 | `/scratch/hongze/robomme_data_h5/` 下 16 个 `record_dataset_*.h5`（已下载并校验，见 `docs/dataset-build-doc/16task-h5-scan/`） | 同一目录中的 4 个 counting H5 |
| packed store | 新建 `framesamp`（`framesamp-4x4-v1`） | 新建 `framesamp`（4x4） |
| norm_stats | 原版 README 附带的 `assets/norm_stats.json`（sha `f332bbd3…`），本机 `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json` 已是同 sha 副本，直接用它；另对新库自算一份，只用来比对、不用于训练 | 对 counting 库新算，放 `train-assets/mme_vla_suite/4task-counting-pub-400ep` |
| 其余 | `mme_vla_suite` 原样 | 同左 |

### 1.1 与原版的区别：超参相同，原始数据同源，派生输入与训练轨迹按当前环境验证

**正式训练超参一项都不改。** 下表的「原版」取自上游 `ecf086c` 的 `mme_vla_suite` 条目，以及 README / `finetune_mme_vla_suite.sh` 的原版命令。本轮两个 run 用的是同名条目。以下是计划审查基线上的已有静态比对，实施后须对固定的 `<实施 HEAD>` 再检查：

- 条目本身：从 `git show ecf086c:src/mme_vla_suite/training/config.py` 抽出条目，与 HEAD 的条目去掉行尾空白后 `diff` 为空（`ENTRY_SAME`）。
- 默认值：`class TrainConfig` 的定义同样 `diff` 为空，所以 seed、log_interval 等继承的默认值也相同。
- history yaml：实际路径为 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml`，对上游与计划审查基线比较为空；不能拿仓库根下不存在的同名路径做 diff。

这里的超参不变指正式 `prod` 模式；短验证的覆盖项单独记录。

| 训练参数 | 原版（ecf086c + README 命令） | 本轮两个 run | 是否相同 |
|---|---|---|---|
| 具名配置 | `mme_vla_suite` | `mme_vla_suite` | 相同（条目逐字同） |
| history yaml | `perceptual-framesamp-modul.yaml`（budget 512，16 token/帧，最多 32 帧，modulation，memory_token_dim 1024） | 同一文件 | 相同（diff 为空） |
| global batch | 64（`--batch-size=64`） | 64 | 相同 |
| GPU 数 / FSDP | 4 卡 / `--fsdp-devices=4`（每卡 16 样本） | 4 卡 / 4 | 配置相同；历史 A40 与本轮 A100 硬件不同，不承诺跨架构逐位相同 |
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

**数据侧：完整 run 保持公开原始数据、样本量与原版 norm_stats；派生特征在本机重建，加载链也已改变。counting 另有子集分布、样本量与 norm_stats 三项预期差异。**

| 差异项 | 原版 | 完整 run | 纯 counting run |
|---|---|---|---|
| ① 原始数据内容 | 公开 `Yinpei/robomme_data_h5`：16 任务 × 100 集 = 1600 集 | 同一批 16 个公开 H5；须命中已归档的输入 SHA256 | 原版数据的子集：4 个 counting 任务 × 100 集 = 400 集 |
| ② 样本量与 epoch 数（80k × 64 = 5.12M 样本） | 476857 个执行样本，约 **10.74 epoch** | **相同**：476857 个，约 10.74 epoch | 189035 个，约 **27.08 epoch** |
| ③ norm_stats | README 附带的 `assets/norm_stats.json`（sha `f332bbd3…`，按 16 任务统计） | **相同**：用同一份文件 | 对 counting 子集新算 |
| ④ 数据格式与加载器 | pkl + features（`source` 格式），上游 `dataset.py` 逐样本读取，collate 走 pickle | packed `framesamp` 4x4 store 加 `FrameSampDataset`，collate 走 torch 共享内存 | 同左 |
| ⑤ 派生特征的计算环境 | 历史硬件与当次抽取环境 | AWS A100 上重新抽取，不据原始 H5 同源推断历史特征逐位相同 | 同左；本轮两库中的相同 episode 另做全量子集对拍 |

上表样本数以已提交的 `docs/dataset-build-doc/16task-h5-scan/records/episode_manifest.json` 为参考，按 `Σ(num_timesteps − exec_start_idx)` 实算。counting 任务没有 demo 段，所以样本数等于帧数。新库建好后会用各自的 manifest 复算一次，结果必须与上表一致，并双向核对 episode 集合、顺序及身份。

- 第 2 步先对输入，再对训练：两侧使用本轮相同的 A100、依赖、source 与 norm_stats，验证上游 source 链与当前 packed 链在规定输入集合及确定性档前 100 步内逐位相同。这不等于证明历史 A40 轨迹相同，也不外推为非确定性生产档全部 80k 步相同。
- counting run 的 ①②③ 会改变训练结果，这是「只训 counting 子集」带来的预期差异。它和完整 run 的 loss 不能直接比，因为二者的数据分布和 epoch 数都不同。
- 原版 norm_stats 与新库自算版的比较只记录、不作闸门：逐键报告最大绝对差，以及参考值非零位置的最大相对差；参考零值单列，避免除零。默认仍用原版文件，不因差异自行切换。

**训练入口与代码的实现差异（不改超参，要靠对拍证明结果等价）：**

- **入口**：上游 `scripts/train.py` 的 `__main__` 会先跑 12 步 tentative 预热，再用 `--overwrite` 重跑正式训练。本仓库 `scripts/training/train.py` 只跑一遍，没有这 12 步。`v1-upstream-eq` 曾证明两者正式轨迹逐位相同，但那次测的是 context 变体。
- **代码**：那次对拍（`f641f40`）之后，`history_pi0.py`、`percep_mem.py`、`framesamp_dataset.py`、`data_loader.py`、`train.py` 又经历了 9 个 commit，包括 V9.1 的 4x4/8x8 两档库、V9.10 的 collate 共享内存，以及 V10.0/V11.0 的 motion 接线与形制守卫（motion 在本轮关闭）。每个改动都有各自的对拍，但**当前 HEAD 在原版 modul 4x4 配置上从未直接对过上游**。2a 补上这一环：不通过就不起正式训练。
- **缓存与产物路径**：JAX 编译缓存、checkpoint、W&B 目录、pi05_base 路径都改到 `v1-store/` 下，与原版的 `~/.cache`、`runs/` 不同。这属于环境 B 的存放规定，不影响计算。

**两个正式 run 同时起跑，各占 4 卡，共享主机 CPU、内存、`/dev/shm` 和磁盘。** `train.py` 用 `make_mesh(fsdp_devices)` 在可见的 4 张卡上建 mesh `(1,4)`，每卡 16 个样本。历史上有 4 卡并跑先例（`repro-4a100-fsdp4` 在 GPU 0–3 和 4–7 各跑一档），仍须做本轮单跑/并跑对拍和资源实测。collate 用 torch 匿名共享内存；JAX 编译缓存、记录目录、日志和 checkpoint run 根逐项分开，模型资产可共同只读。生产 runner（`run_modul*.sh`、`modul_launch_contract.py`）绑定了其他档位，本轮**不用也不改**它们，另写一个 4 卡 runner。

**checkpoint 与耗时。** 按 10k 间隔保存，两个 run 各留 `10000…70000, 79999` 共 8 份；旧估算每份约 12G、合计约 192G，须用本轮 smoke/perf 真保存的字节量复核，并计入异步保存的临时占用。修复 collate 共享内存后，本机还没有 4 卡 b64 的步时实测：旧的 4 卡数字（2.34 s/步，b128，修复前）不能用，ETA 以第 3 步稳态测速加编译、保存耗时估算。

### 1.2 两条输入链：先比较交付内容，再比较训练

两侧消费本轮同一份 source；上游使用自己的 Dataset、采样和 transforms 作为参照，不能只用当前仓库参考类代替上游。拟补的非训练量具须在各自独立环境记录身份、shape、dtype 与字节摘要，再由第三个只读判定过程比较。

```text
上游 A：公开 H5 → 本轮 source（执行 pkl + 逐帧 npy）
        → 上游 RoboMMEDataset（选帧、padding）→ 上游 transforms
        → 上游 pickle collate（b64，workers4）→ JAX sharding → 模型

当前 B：同一 source → pack 4x4 + 全量 verify → FrameSampDataset
        （当前图像/动作仍读 source pkl，历史特征读 packed）
        → 当前 transforms → torch 共享内存 collate（b64，workers4）
        → JAX sharding → 模型
```

H5 到 source 的抽取是两侧共同前置；源图像每执行样本为两幅 `(256,256,3) uint8`，纯图像共 `393216 B`。packed 的历史图像每总帧为 `(16,2048) bf16 = 65536 B`，pos/state 另存；pack 对保留的 4x4 特征应逐位不改数。当前 B 装配后的历史图像每样本为 `(512,2048) bf16 = 2097152 B`，历史位置为 `(512,768) float32 = 1572864 B`，历史 state 归一化后为 `(512,8) float64 = 32768 B`，mask 为 `(512,) bool = 512 B`。当前两幅图像经 transforms 缩到 `(224,224,3) uint8`，在 Observation 构造中转到 `[-1,1] float32`；state/actions 经归一化、补维，再由禁用 x64 的 JAX 环境交付 `(64,32)` / `(64,20,32) float32`。这些已有转换本身会改数，要求的是对应节点之间的实现对拍，不能把整链写成不做数值转换。motion 四键保持 `None`。

**修订时发现的执行前阻断：上游短历史 padding 的 dtype 已有差异。** 固定上游的 `shared/data_utils.py::right_padding_token_emb()` 用未指定 dtype 的 `np.zeros`，短历史可能使 image/pos 提升为 float64，混合 batch 还会扩大影响；当前同函数及 `FrameSampDataset._pad()` 保留输入 dtype。上游对应数组的实际 dtype/字节量必须单独记录，不能套用上述 B 侧数字。此差异尚无本轮输入/训练实测；严格输入判据不自动豁免它，也不得临时 cast 真正训练输入或修改上游来制造 PASS。旁路统一 dtype 的数值诊断只能定位差异，不能代替原始摘要。若原始输入判据因此失败，应先形成差异报告，是否改变实现或接受另一等价口径另行明确，未解决前不进入正式训练。

输入判据：两库分别核对全量样本身份及 sampler 的前两个完整 epoch 顺序（含 `drop_last` 与重建 iterator 边界）；每集选执行首尾、demo/exec 交界及历史不足/达到/超过 32 帧的合法样本，比较原始样本和 transforms 后的全部模型输入；另比较正式 b64/workers4 的前 100 个 batch，并覆盖 epoch 换轮前后各两批。仅日志路径、运行身份及不进入模型的历史废弃键可列明排除，任何模型输入键不得忽略。数值、dtype、shape 和顺序均要求逐位一致，判定行拟为 `INPUT_EQ=PASS`，摘要按字段记录，失败打印首个样本身份与键名。

### 2. 执行顺序：环境与预算 → 构建冒烟 → 两个正式库 → 输入与训练对拍 → 测速 → 正式 80k

**第 0 步：先把环境、来源与磁盘闸门做实。** 确认主副本 clean、分支与既有 upstream 同步，固定实施提交。先串行准备上游 worktree 的独立 `.venv` 和本仓库环境，记录相同 Python 版本、依赖指纹、CUDA/JAX、资产摘要与模块导入来源。上游 `pyproject.toml` 的 workspace 声明含未入提交树的 `sandbox2/flash_attn_jax`，不能假定 fresh worktree 内直接 `uv sync` 就绪；C 节规定受控准备、恢复源码及来源验收，通过后才进入全量建库。

先重算 16 个 H5 的摘要，与已提交 `docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` 的 `files` 集合、逐文件 `size/sha256` 完全一致；counting 按四任务过滤同一参考。`raw_dir` 与新增 `count` 等记录字段不作字节等同要求。仅对照现场新生成的 input_manifest 不能证明公开源同源。拟判定 `INPUT_PIN=PASS files=16` / `files=4`，归档参考提交、双方文件摘要和判定行。

磁盘用 `df -B1 /scratch/hongze` 记录可用字节，统一 `1 GiB=2^30 B`。建库前要求 `可用字节 ≥ 剩余构建峰值预算 + 300 GiB`，预算包括两个 source/packed、构建冒烟、独立 venv/缓存、验证 checkpoint 和临时写入；正式前要求 `可用字节 ≥ max(300 GiB, 两个 run 全部 checkpoint 预算 + 保存临时峰值 + 日志缓存余量)`。下界不能作为最终预算：先用第 1 步冒烟量出 feature/pkl 的实际分配字节，结合全量 T/E 推算并至少加 10% 数据开销余量，其他项目单列。每阶段重测剩余预算，不足就停止交用户清理，不删除已有产物。

**第 1 步：先构建 16 任务小库，再建两个正式库。** 两库从 `/scratch/hongze/robomme_data_h5/` 的公开 H5 建，已有约 530G（含原始压缩档）只是历史体积记录。`16task-scan` 证明了 episode 元数据可扫描，未证明每任务图像、动作和 subgoal 等构造字段完整。先用 `scan_manifest.py build --episodes-per-task 1 --num_shards 1` 生成独立 16 集冒烟清单，再走 `run_local.py --stage siglip`、finalize、pack/verify；这仅覆盖各任务首集，不声称覆盖所有难度。预计超过 5 分钟时按正式诊断纪律 tmux 留档，保留摘要与结果后只清理本轮临时大产物。

- **1a `16task-pub-1600ep`**：scan 不传 `--tasks`，传 `--episodes-per-task 100`；`--raw_dir` 指向公开目录，hash 只收 `*.h5`，必须恰好命中已归档的 16 文件集合。
- **1b `4task-counting-pub-400ep`**：在全新的 `/scratch/hongze/robomme_data_h5_counting4/` 中为 BinFill、PickXtimes、SwingXtimes、StopCube 四个 H5 建硬链接；先核实同盘、目录不存在，建后核对设备/inode 与摘要。scan 加四任务名单及 `--episodes-per-task 100`。source 使用全局样本/集号，故仍独立建库，不引入切子集重编号的新生产链。

每库依次执行 scan → hash 与已归档输入绑定 → SigLIP（GPU 0–7）→ `finalize_checks.py check --input_level sha256 --spot_check 1024` → 默认 4x4 pack → 全量 verify → CPU `compute_norm_stats.py`。两个正式库串行构建；不建 8x8 packed，不做 motion。完整库的自算 norm_stats 仅供比较。每库收尾以 `smoke` 模式跑 20 步、真实保存和加载末步 `19/params`。正式库留档分别位于 `docs/dataset-build-doc/16task-pub-1600ep/` 与 `docs/dataset-build-doc/4task-counting-pub-400ep/`，起跑版本与命令先记录，结果后补。

**子集一致核对：全量输入内容都要比。** 按 `(h5_file, raw_ep_idx, step_idx)` 对齐，检查 counting 恰为 400 集、189035 个执行样本，集合无重复/遗漏；对应 feature 文件逐字节比较，pkl 解码后全键比较 shape、dtype、内容，包括 `image`、`wrist_image`、完整 `actions`、`state`、`prompt`、`is_demo`、`exec_start_idx` 和 `step_idx`。仅 `epis_idx` 允许根据两份清单做明确映射；不能直接比较含全局编号差异的 pickle 文件字节。拟判定 `SUBSET_EQ=PASS episodes=400 samples=189035 feature_mismatch=0 sample_mismatch=0`。图像错帧必须失败；若 SigLIP 两次抽取不逐位相同，记录最大绝对差并停止，不自行放宽。

**空间重算：按总帧 T 与执行样本 E 分开计价。** `mem_buffer.py::add_buffer/get_history_feats` 的 feature 纯数组为 `(64+16+4)×(2048×2+768×4)+8×4 = 602144 B/总帧`；`build_robomme_dataset.py::_process_episode` 的 pkl 两幅图片另占 `393216 B/执行样本`；packed 4x4 图像另占 `65536 B/总帧`。旧私有库 `605611/1192918≈50.8%` 是执行帧，counting 为 100%，故删除旧“780G”均摊估算。

| 数组载荷下界（B） | 完整库（T=768897，E=476857） | counting（T=E=189035） |
|---|---:|---:|
| source feature | 462986715168 | 113826291040 |
| pkl 两幅图片 | 187507802112 | 74331586560 |
| packed 图像 | 50390433792 | 12388597760 |
| 合计 | 700884951072 | 200546475360 |

两库上述载荷已达 `901431426432 B = 839.52 GiB`；尚未含 NPY/pickle 头、动作、packed pos/state、索引、文件系统分配与缓存。这是静态计算下界，不是体积实测；仅它与正式 300 GiB 余量就需 1139.52 GiB，最终闸门按第 0 步完整预算计算。耗时也在冒烟后重估，不沿用旧“约 1.5 小时”。

**第 2 步：先完成 §1.2 输入验证，再做两类训练对拍。** 两类均使用确定性档 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`，正式配置的 b64/fsdp4/workers4/seed42/modul 不变；验证入口仅覆盖 `num_train_steps=100`、`log_interval=1`、`save_interval=50`，保留原 warmup/lr/save keep 口径。每侧记录 step `0…99` 的五标量 `float.hex()`，以及 step `50,99` 的参数、EMA、优化器状态和 step 摘要。现有 `entry_equiv.py` 已可替换 `save_state` 为摘要器；拟补有限值检查，要求五标量及每次摘要的所有数值叶都有限，不能仅靠相等的 `nan/inf` 字符串或哈希通过。

- **2a 上游与当前入口**：A 固定上游完整提交，使用其官方 `scripts/train.py::__main__`（12 步 tentative，再 `--overwrite` 正式跑），读本轮 source；B 固定实施提交，用当前 `scripts/training/train.py` 读 packed。argv 仅允许入口、exp-name、dataset-path、checkpoint 根及 A 的 `--overwrite` 五处差异，路径都由运行清单约束到相应数据/资产。两组 A 可在 GPU 0–3 / 4–7 并行；等两组 A 全部退出，再先 B-full 独占运行、后 B-count 独占运行，期间停止本轮其他 GPU/建库/测速负载并记录主机其他负载。判定保留 scalars/state/config/provenance 四项，并新增 `ENTRY_FINITE=PASS`；总判 `ENTRY_EQ=PASS`。没有历史锚点时打 `anchor=SKIPPED`。
- **2b 同入口单跑与并跑**：复用刚才真正单跑的 B-full 为 S1、B-count 为 S3；再新起 B-full+B-count 作为 S2。拟新增 `judge --mode same-entry`，两侧必须同 root、同 HEAD、同入口和模块摘要、同依赖/数据/资产/配置；仅 run 身份、输出与缓存目录可不同。它与 `--mode upstream` 的异根防污染要求分开，不能直接删除原有同根拒绝。S1 对 S2-full、S3 对 S2-count 均要求标量与状态零失配且全部有限。记录逐步时间区间，并跑观测窗口定义为各自 step `10…99`，两窗口交集须覆盖各侧至少 50 个完整步骤；若编译错开导致不满足，则本次不能判并跑通过，在独立同配置预热后重跑，不能把同时起进程当成同时训练。

两类通过只支持“本轮 A100 确定性档、所验证输入与前 100 步”的结论。发生失配时先区分输入、配置/来源、非有限值和计算结果问题；保留证据并停止，不用没有历史锚点的“四象限”替代诊断，也不自动放宽阈值。

**第 3 步：正式环境测速，`perf` 模式 4+4 同时各跑 300 步。** 与 `prod` 共用同一训练配置和计算环境构造，配置仅步数与 run 身份不同；另有下述显式记录的计时观测差异。清除确定性档 `XLA_FLAGS`，保留 b64/fsdp4/workers4/log100/save10000/keep10000/W&B。拟新增进程内轻量观测 harness，通过 `runpy` 执行原入口，对训练调用、取数及真实 `save_state` 安装计时包装并原样转发，不能像对拍 harness 那样取消真保存。不启用会启动 profiler 的 `TRAIN_TIMING_STEPS`，也不直接套用固定 1000 步、8 卡/b128 的 `check_modul_speed.py`。预热 `0…99`，稳态 `100…299`；仅在窗口边界等待设备完成，末步训练完成后、保存前结束稳态窗口，初始化/JIT、常规 logging 与最终保存耗时分别说明，ETA 加回周期保存成本。单步记录含主机阶段耗时，其异步提交时间不冒充 GPU 计算时间；包装的开关对照纳入 20 步真实短测，验证输入、标量、状态不变并披露计时开销，prod/smoke 默认直调训练入口。

`nvidia-smi -lms 500` 采样按物理 GPU 和时间戳对齐上述窗口，报告 util 均值、0% 采样占比，以及慢步（主机端到端时长大于稳态中位步时 2 倍）/其他步骤的分层均值；中位数只用于分层定义，不作吞吐结论。报告每侧及合计 samples/s、两侧稳态实际重叠时间、主机内存和 `/dev/shm` 峰值，记录底层存储、batch、worker、窗口、采样间隔。300 步的末步 `299` 须真实保存、等待完成、加载并对本次 EMA 摘要验证；任一非有限值拒绝进入正式训练。预计超过 5 分钟，两个 perf run 各自从 clean HEAD 在 detached tmux 起跑并先建 launch 档，结果与测量记录归档后才清理本轮临时权重/缓存。即使发现瓶颈，也不自行修改 workers 或其他超参。

**第 4 步：正式 80k，从同一个 clean HEAD 启动两个 detached tmux。** 先完成两个 `docs/training-doc/<run>/launch.md` 的准备与提交，之后锁定共同 `<TRAIN_HEAD>`；最终命令使用该完整 SHA 字面量，避免先起一个 run 再提交另一个档案使版本漂移。档案区分代码提交与包含档案自身的提交，不把尚未创建的提交写成已知值；启动现场记录再保存真实 HEAD、完整 argv/env、数据/资产 SHA256 与会话名。会话为 `orig80k-full-<UTC>`、`orig80k-count-<UTC>`，只按本轮清单用 `tmux has-session -t '=完整名称'` 检查，必要清理同样逐个精确匹配并核对前后清单。

每个会话使用 `PYTHONUNBUFFERED=1`、`set -o pipefail` 和 `tee`，失败和正常退出均记录 `EXIT_CODE=`；同时保留训练进程及 tee 的退出状态，日志写失败不能当成功。用当前宿主可用的流式监听持续观察，每一级过滤都行缓冲，不依赖宿主不存在的 Monitor 工具。主副本代码与 `.venv` 训练期间锁定，开发转到 `-temp` 独立环境；允许计划内 run 产物写入各自 `v1-store/` 目录。

**完成验收（每个 run 独立判）**：终态 `EXIT_CODE=0`；checkpoint 集合恰为 `10000,20000,…,70000,79999`；结构化普通日志步集合恰为 `range(0,80000,100)`，共 800 条、无重复、五标量完整且有限。`train.py::main` 常规日志最后一条为 79900，所以 runner 必须同时启用已有 `TRAIN_RECORD_DIR` 与 `TRAIN_FINAL_RECORD_DIR`：复用 `final_record.finish()` 检查尾窗恰为 `79901…79999` 共 99 步、五标量全部有限，`state_step=80000`、`loop_step=checkpoint_step=79999`，并核对 run/HEAD/UUID 与 `checkpoint_wait_done.json` 的异步保存完成记录一致。真实加载 `79999/params`，全部叶有限且 dtype/shape/字节摘要等于本 run 末步 EMA 记录。结果归档全程及稳态耗时、samples/s、GPU util；策略评估不在范围内。

### 3. 两条核心保证及证据边界

**保证一：正式训练超参沿用原版，短验证覆盖项单列。** 配置层比对固定上游与实施提交的 `mme_vla_suite` 条目、`TrainConfig` 默认值和 history yaml 实际路径，不能用整个 config.py 的差异或空路径替代条目核验。命令层对 2a 的 resolved config 除 exp_name、dataset_path、checkpoint 根和 overwrite 外逐字段一致；路径白名单还须绑定实际文件内容，不能放行“双方一样指错”。正式入口用同一份 `TRAIN_ARGS` 在 CPU 解析，断言 b64、80000 步、warmup10000、peak/decay lr 都为 5e-5、fsdp4、workers4、seed42、log100/save10000/keep10000、EMA0.999、modul history；再在真实设备检查中确认恰好四卡、GPU 分组无重叠。`perf/smoke` 仅分别放行 300/20 步，不修改正式断言。

**保证二：原始输入有独立来源锚，当前环境的输入与短轨迹有分层证据。** 输入先通过已提交 H5 pin、episode 集合、finalize、packed 全量 verify、子集全字段对拍和 §1.2 的样本/batch 对拍；完整 run 使用原版 norm_stats，counting 使用其自算文件并记录摘要。训练层分别要求上游/当前与单跑/并跑比较的有限值、标量、状态、配置、来源全部通过，且并跑窗口确实重叠。这些证据不承诺历史异构 GPU 或生产 80k 的逐位轨迹；生产环境以 300 步真实保存加载和正式完成验收补充覆盖。任一必需判据失败就停在对应阶段，不起正式训练。

## 第二部分（技术细节，供 agent 追踪）

### A. 拟实施的工具改动（新增或参数化工具，不改训练源码、全局配置和超参）

1. **对拍驱动 `scripts/training/tests/run_entry_equiv.sh`**：参数化 `HISTORY_YAML`、`BATCH`、`FSDP`、`GPUS`、`SAVE_INTERVAL`、`DATA_A/B`、`ASSETS_DIR`、`ANCHOR_SHA256`（显式空值表示不传锚点）、记录根与两侧完整 HEAD。新增 `ASSETS_DIR` 设置时才传 `--data.assets.assets-dir` 和 asset-id，未设置时保留历史默认 argv；新增 dry 展开并验证历史默认 argv 逐字不变。B 独立注入 `MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 与 JAX cache；A 解释器由 C 节确定。judge 必须传 steps、tentative、state steps、HEAD 与比较模式，不能继承不适用的历史 1000 步锚点。
2. **判定器 `scripts/training/tests/entry_equiv.py`**：拟新增 `judge --mode upstream|same-entry`（默认 upstream）。upstream 保留异根与 forbid-root；same-entry 要求两次 entry/root/HEAD/模块 SHA 一致，并核对依赖、设备、数据、资产、history 与有效环境指纹。两种模式都要求唯一完整步集合、状态步集合、标量和状态全有限；记录器写 finite 结果，judge 缺记录即失败。same-entry 对 resolved config 只放行 run 身份与输出路径，不能沿用 upstream 的 dataset/overwrite 宽白名单。起跑和收尾分别记录 clean HEAD，避免只在 finally 取证却称为起跑版本。
3. **输入量具**：拟新增 `scripts/training/tests/check_orig80k_inputs.py` 的取证/判定能力，分别在 A/B 环境 import 该侧 Dataset、transforms 和真实 loader，按 §1.2 样本与 batch 计划取摘要，禁止 A 导入当前仓库参考实现。现有 `dump_fixture_samples.py` 的 `DTYPE_DUMP_IMPL=refnpy|packed`、`DTYPE_DUMP_MODE=both`、`DTYPE_DUMP_ARRAYS=0` 可作为当前 source→packed 的额外检查，正式取证不设 `DTYPE_DUMP_LIMIT`；其现有 batch 固定为 8、单进程 collate，不能冒称正式 b64/workers4 检查。拟参数化真实 sampler/collate 量具补足正式档位，原始 dtype 差异按 §1.2 阻断，不扩大排除键。
4. **runner `scripts/training/prod/run_orig80k.sh <prod|perf|smoke> <run_name> <gpus> <lib> <assets_dir>`**：拟新增三模式与统一 `TRAIN_ARGS`，分别固定 80000/300/20 步；不接受未登记的额外训练参数。显式 source 训练域 `paths.sh`，设置 `UV_CACHE_DIR`、`OPENPI_DATA_HOME`、`XDG_CACHE_HOME`、`HF_HOME`、每 run 独立的 `MMEVLA_JAX_CACHE_DIR/JAX_COMPILATION_CACHE_DIR`、`CUDA_CACHE_PATH`、`WANDB_DIR/CACHE_DIR/CONFIG_DIR/DATA_DIR`，以及 `MMEVLA_FRAMESAMP_SOURCE/MANIFEST`、`TRAIN_RECORD_DIR`、`TRAIN_FINAL_RECORD_DIR`。显式清除验证遗留 `XLA_FLAGS`、`TRAIN_TIMING_STEPS` 和非本模式观测补丁；保留真实环境快照。
5. **runner 的 preflight 与测速**：复用 `preflight_train_launch.py` 的通用检查，传完整 `TRAIN_HEAD` 字面量、history/norm_stats 期望 SHA、绝对数据/资产路径与由配置推导的 run-root。512/4x4 本轮不传绑定旧 modul 契约的 `--launch-mode`，由新 runner 实现 §3 的模式契约，不能因不传旧参数就跳过配置检查。先 CPU 解析最终 argv 并对原版配置做完整比对，再单独核对可见物理 GPU 恰四张且空闲、两组不重叠；CPU 解析子进程的 `JAX_PLATFORMS=cpu` 不得泄漏到正式训练。拟新增 `scripts/training/tests/check_orig80k_speed.py` 的进程内观测/汇总两入口，不开启 profiler；perf 由其观测入口接收原始 `TRAIN_ARGS` 后 `runpy` 执行训练，按第 3 步插入边界同步且保留真实保存。prod/smoke 默认直调 `uv run --no-sync scripts/training/train.py ...`，正式仅使用已有指标记录与外部资源采样。两种调用都保存实际入口/argv/观测项，不能声称 perf 与 prod 的包装完全相同；并发期间禁止同步或修改共享 `.venv`。
6. **完成器 `scripts/training/tests/check_orig80k_completion.py`**：拟新增按 mode 固定期望步骤的检查；prod 按第 4 步全部判据输出 `RUN_COMPLETED=PASS run=<name> checkpoints=8 final=79999`，perf/smoke 分别要求末步 `299/19`、`state_step=300/20`、对应普通日志与尾窗、真实保存加载同本次 EMA。不可用手造文件名或仅统计目录数通过；缺失/重复终态、非有限值、UUID/HEAD 不符、缺保存等待记录、权重摘要不符均失败。
7. **数据守卫**：拟新增 `scripts/dataset/check_orig80k_sources.py`，参数为 `--input-manifest`、`--reference-input`、`--manifest`、`--reference-manifest`、可选 `--tasks`，实现第 0 步 pin 与 episode 集合检查。拟新增 `scripts/dataset/check_subset_eq.py --full <lib> --subset <lib>`，实现 §2 的全覆盖全字段比较；既有 `compare_datasets.py` 只取 episode 交集且 pkl 用 `np.array_equal`，只能复用身份映射/读数逻辑，必须补集合、dtype、字节判据。
8. **最小验证**：工具实现后运行 D 节列出的正负例和两侧 P1 冒烟，新增测试位于 `scripts/training/tests/` 与 `scripts/dataset/`；单测尽量 5 分钟内。上述工具均尚未实施，本次修订不运行它们。

### B. 建库命令骨架（沿用 `docs/dataset-build-doc/4task-v2-1600ep-604f16da/launch.md` 的「命令与配置还原」）

以下骨架依赖 A 节拟新增的 `check_orig80k_sources.py`，实施及单测通过后才可使用。第 0 步环境、16 文件来源与预算闸门，以及第 1 步独立 16 集构建冒烟必须先通过。函数内各阶段应由本轮 detached tmux 驱动串行执行，逐阶段记完整命令、耗时、退出码与日志；不是当前可直接整段启动的命令。

```bash
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
build_lib() {   # 前两个参数为原始目录、库名，其余是可选 --tasks 参数
  local raw_dir="$1" lib_name="$2"
  shift 2
  local lib_dir="$V1_STORE/datasets/$lib_name"
  local mani="$lib_dir/meta/episode_manifest.json"
  local inmani="$lib_dir/meta/input_manifest.json"
  local reference="$REPO_ROOT/docs/dataset-build-doc/16task-h5-scan/records"
  local stats_dir="$V1_STORE/train-assets/mme_vla_suite/$lib_name"
  if [[ -e "$lib_dir" || -L "$lib_dir" || -e "$stats_dir" || -L "$stats_dir" ]]; then
    printf '拒绝复用库或统计量输出：%s / %s\n' "$lib_dir" "$stats_dir" >&2
    return 1
  fi
  uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$raw_dir" "$@" --episodes-per-task 100 --num_shards 1 --out "$mani" || return "$?"
  uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$raw_dir" --out "$inmani" || return "$?"
  uv run --no-sync python scripts/dataset/check_orig80k_sources.py --input-manifest "$inmani" --reference-input "$reference/input_manifest.json" --manifest "$mani" --reference-manifest "$reference/episode_manifest.json" "$@" || return "$?"
  uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$lib_dir" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$raw_dir" --require-free-mib 70000 || return "$?"
  CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$mani" --out "$lib_dir/source" --raw_dir "$raw_dir" --input_manifest "$inmani" --input_level sha256 --spot_check 1024 || return "$?"
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$lib_dir/source" --manifest "$mani" --out "$lib_dir/framesamp" --procs 48 || return "$?"
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$lib_dir/framesamp" --resume --procs 48 || return "$?"
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$stats_dir" --config-name mme_vla_suite --repo-id robomme --dataset-path "$lib_dir/source" || return "$?"
}
build_lib /scratch/hongze/robomme_data_h5 16task-pub-1600ep || exit "$?"
mkdir /scratch/hongze/robomme_data_h5_counting4 || exit "$?"
for task in BinFill PickXtimes SwingXtimes StopCube; do
  ln -- "/scratch/hongze/robomme_data_h5/record_dataset_$task.h5" /scratch/hongze/robomme_data_h5_counting4/ || exit "$?"
done
build_lib /scratch/hongze/robomme_data_h5_counting4 4task-counting-pub-400ep --tasks BinFill,PickXtimes,SwingXtimes,StopCube || exit "$?"
```

- 完整 run 的 `--data.assets.assets-dir` 为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite`；counting 为该目录下 `4task-counting-pub-400ep`。`compute_norm_stats.main()` 固定写 `<output-dir>/robomme/norm_stats.json`，不改变 asset-id，也不用完整库自算文件替代原版文件。launch 记录完整 64 位期望 SHA，文内 `f332bbd3…` 仅为阅读缩写。
- 新旧 episode manifest 按物理身份核对 `num_timesteps/exec_start_idx/exec_samples`、双向集合及总数；调度字段 `num_shards/shard_idx/shard_load_timesteps` 可以因本地分片安排不同而变化，不能要求整个 JSON 或整体 SHA 相等。完整库还核对规范 episode 顺序；counting 的全局编号允许重排，但物理身份与局部 step 必须一一对应。两份 manifest SHA 都保留。
- `check_subset_eq.py` 必须实现 §2 的全集和全键检查后才能使用；现有 `finalize` / `pack verify` 不覆盖 pkl 腕部图像，不能替代它。
- 每阶段都在 detached tmux 中跑，正式库分别使用 `pub16-`、`cnt-` 前缀并记录完整会话清单。输出根起跑前 `ls -ld` 检查实体父目录与目标不存在（含悬空 symlink），不使用覆盖选项；骨架若失败，不直接重放整库命令，先依据本轮阶段日志定位已完成与未完成部分。

### C. 对拍执行细节

- **P0 独立环境**：A worktree 放在全新 `v1-store/worktrees/orig-ecf086c`，固定上游完整 SHA。拟新增环境准备器，针对上游不存在且未列入锁文件 workspace 清单的 `sandbox2/flash_attn_jax`，只在安装期间对 `pyproject.toml` 做可逆补丁，移除这一成员；以同一 Python 版本、上游 `uv.lock` 和 `uv sync --frozen` 建独立 `.venv`，随后逐字恢复原文件。`UV_CACHE_DIR` 与 uv 管理的 Python 下载目录显式位于本仓库 `v1-store/`；`UV_LINK_MODE=copy`。准备前后核对原 pyproject/uv.lock SHA、clean HEAD、editable 项目模块来源与第三方包指纹；任何差异失败，不以空壳包或修改训练源码替代。该方案仍待实施和真实验证，失败就停在 P0。恢复缺失成员声明后，不假定 `uv run --no-sync` 可绕过 workspace 发现；A 用这份 uv 管理的 `<worktree>/.venv/bin/python` 执行 harness，B 用自身环境，绝不共享 `.venv`。
- **缓存与来源隔离**：上游入口写死 `~/.cache/jax_<exp_name>`；拟在 harness `run` 新增可选 `--jax-cache-dir <绝对路径>`，仅转接 `jax.config.update` 的 `jax_compilation_cache_dir` 设置到本轮 `v1-store/cache/jax/<run>`，其他配置调用原样转发，记录实际值并测试没有额外配置变化。该路径适配不改上游文件、不覆盖 HOME，也不创建 HOME 下的缓存链接。A/B 共用只读资产的绝对路径，各自分开记录、输出和编译缓存。A run 带 `--expect-root <worktree>`，B 带 `--expect-root <主仓库> --forbid-root <worktree>`，清除 `PYTHONPATH/PYTHONHOME`，所有导入来源要实际记录。
- **P1 入口冒烟**：两库分别在 A/B 跑 2 步，A 保留官方 tentative+正式双段，B 单段；A 的 2 步 tentative 判据为 2，而不是 100 步对拍时的 12。两段都会触发末步 `1` 的状态摘要，现有 harness 只拆 metrics，所以 P1 仅判入口 `ENTRY_RUN=OK/EXIT_CODE=0`、A 的 tentative/main 各 2 行、B 的 main 2 行、有限值与来源，保留两段状态记录，不套用要求状态 step 无重复的 100 步 judge，也不据 P1 宣称轨迹等价。先完成这项及 §1.2 的输入检查，再跑 100 步；上游 A 消费 source，不能描述为直接读取 packed。
- **拟判定命令**：`entry_equiv.py judge --mode upstream --expect-steps 100 --expect-tentative-a 12 --expect-state-steps 50,99 --expect-head-a ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b --expect-head-b <实施 HEAD> --a-dir <A记录> --b-dir <B记录>`；同入口比较用 `--mode same-entry --expect-tentative-a 0`，两侧 `--expect-head-*` 均为同一实施 SHA，其余 steps/state 判据保持。显式空锚点时完全不传 `--expect-sha256`，不能传空字符串。
- **调度与有效性**：严格执行 A 两组完成 → S1 → S3 → S2；记录 GPU UUID、进程、实际开始/结束和重叠窗口。S1/S3 的复用需同 HEAD、依赖、数据、资产和计算环境；如果期间改了实现，旧基线失效，两侧重跑。S2 若没有实际训练重叠则判无效，不能仅凭数值一致判并跑通过。
- **留档与清理**：`docs/training-doc/orig80k-equiv-0925/` 保存 launch/result/records、两侧环境、输入摘要、原始与清洗日志、每项判定及退出码。预计超过 5 分钟的输入验证同样留档，先提交 launch 再从 clean HEAD 运行。PASS 后先确认归档完整，再逐项清理本轮临时权重/缓存；worktree 先查 clean、无运行进程，使用精确路径的普通 `git worktree remove`，不默认 `--force` 或全局 prune。FAIL 时保留原目录和路径，不改名破坏 provenance 引用，不通配删除缓存。

### D. 验证与提交

**本次纯文档修订**只改本文件，使用 `git diff --check`、核对代码锚点/路径、B 节 shell 骨架语法及最终文件范围，不运行仓库脚本、依赖同步或训练。通过后独立 `docs:` commit，body 记录用户「同意修改」、修订范围、静态检查结果与尚待实施事项。授权仅为文档修改，不等于批准起跑。

**后续工具实施验证**：`uv run --no-sync pytest <本轮明确测试路径>`，测试路径随工具实现列入 commit body；对拍驱动增加 dry 展开，历史参数未设置时 argv 必须与旧默认逐字相同。最低覆盖如下：

- runner：prod/perf/smoke 分别解析出 80000/300/20，prod 拒绝步数覆盖；双层 `robomme`、相对资产路径、错 norm_stats SHA、卡数不为 4、重复 GPU、已有输出根均拒绝；CPU preflight 不污染真实设备环境。
- harness：upstream 同根拒绝、same-entry 合法同根通过、same-entry 不同 HEAD/模块/数据拒绝；两侧同为 NaN/Inf 仍失败；缺/重复 step、缺 finite 证据或状态步集合错误失败；JAX 路径适配只改变缓存目录；未重叠的 S2 不得判并跑有效。
- 数据：H5 同尺寸改内容、缺/多文件、episode 缺失/重复、仅 wrist 图像错帧、同值不同 dtype、错误 epis_idx 映射都失败；允许 input_manifest 记录字段和 manifest 分片字段的约定差异，禁止丢掉真实内容差异；任一建库阶段模拟失败，必须保留非零退出码且后续阶段不启动。
- 完成器：缺/多 checkpoint、普通日志重复/缺步、尾窗任一五标量非有限、`state_step` 错、run/HEAD/UUID 错、缺保存完成记录、真实加载权重与 EMA 摘要不一致都失败。真实 smoke/perf 保存加载验证补上合成负例无法覆盖的核心路径。

**实施阶段提交与档案**：代码及环境准备工具先提交；每阶段 launch 在该阶段起跑前完成提交，再运行并回写结果。顺序为环境 P0 → 16 集冒烟 → 两个正式库 → 输入/入口/训练对拍 → perf → 两个正式 launch 一并就绪 → 从同一 HEAD 起跑 → result。每份超过 5 分钟的记录保留，临时权重清理不删除归档。代码编号实施前查最新 `commitV*` 再确定，审查基线中 `commitV11.8Beta` 已被 `d023b72` 占用，不能假定未来 `11.9Beta` 仍空闲；按仓库提交约定逐文件暂存、同步既有 upstream，不改写历史。

**范围与决定**：公开 16×100 全集、四 counting×100 子集、modul、原版超参和 counting 重算 norm_stats 已由用户明确，沿用不重复询问。本轮「同意修改」仅授权修订方案。未来正式执行需明确批准本计划及表中的两个全新 run_name；批准执行时一并确认两个库名、完整 run 使用原版 norm_stats、100 步确定性对拍、300 步测速，以及 full=GPU0–3/counting=GPU4–7。§1.2 已知上游 padding dtype 差异若触发严格输入判据，必须先报告并解决，不把执行授权当成放宽判据或修改训练语义的授权。
