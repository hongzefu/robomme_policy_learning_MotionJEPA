# 8 帧 × 8×8 训练支持与前后对拍（实施过程档案）

> 本轮实施和验收已完成。现行数据接口与前后链路见[数据链路正本](docs/dataloader-restructure.md)，实际推理覆盖与边界见[训练/推理一致性正本](docs/train-infer-consistency.md)。下方保留实施前计划的组织与命令口径，完成结果以各run档案为准。

本文是 2026-09-13 细化版经对抗验证后的修订版。对抗验证锚定 HEAD `9c344fa1bf615847c0233be012420d21d8a7841b`（7 个只读审计 agent 与 Codex 独立审稿交叉核对，全部结论以代码原文为证；当时的计划修订未执行仓库脚本或训练，后续实施结果见文首链接）。结构仍按 `AGENTS.md` 第 2 条分两部分：**第一部分给人看**，分「数据预处理」「训练」「推理」三块；**第二部分供 agent 追踪**，列文件、函数、命令与判定行。环境判定为 B：仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，8 × A100-SXM4-80GB，无 turbo、无 GreatLakes，`/scratch` 余量 4.1 T（实测 `df -h /scratch` = 6.9T / 2.9T 已用 / 4.1T 可用）。

**授权与完成状态**：用户随后明确要求“开始实现”“一口气全做完”，本轮代码、两库构建、完整输入、12条1000次更新轨迹、四profile三batch全梯度，以及C8/M8七关和各48集闭环已完成。只使用物理GPU4–7；1000次参数更新没有用短测代替。详细结果见[训练总览](docs/training-doc/t8-training/result.md)、[全梯度](docs/training-doc/t8-gradient/result.md)、[C8推理](docs/training-doc/t8-infer-c8/result.md)和[M8推理](docs/training-doc/t8-infer-m8/result.md)。

**GPU 约束（2026-09-14 用户新增，最高优先级）**：本计划一切用 GPU 的命令**只能用 GPU 4、5、6、7**，GPU 0–3 一律不碰。落地三条：每条起 GPU 进程的命令都显式写 `CUDA_VISIBLE_DEVICES`，取值只能是 `{4,5,6,7}` 的子集，禁止不设而继承默认全卡，也禁止「照抄历史留档命令」（`tic-obs-model-40k/launch.md` 等历史命令写的是 `CUDA_VISIBLE_DEVICES=0`，照抄即违规）；motion sidecar 的卡号**不跟随** `CUDA_VISIBLE_DEVICES`——主线 `policies/policy_config.py::create_trained_policy` 直接读 yaml 快照的 `motion.online_gpu`（现值 1，绝对物理卡号）且无环境变量覆盖，`motion_client.py` 把它整体写进子进程的 `CUDA_VISIBLE_DEVICES`，所以凡起 sidecar 的路径必须走带显式卡号参数的入口（`scripts/motion-variance/serve_policy_mv.py --motion-gpu`、`compare_train_infer_obs.py --motion-gpu`、`compare_online_motion.py --gpu`），并把新 8×8 motion yaml 的 `online_gpu` 写成 4 作兜底；第二块 12 条轨迹只有两对卡可用，分两波跑（见第二块）。

**2026-09-14 用户裁决（对抗验证后的 9 项拍板，本文全部按此落实）**：

| # | 事项 | 裁决 |
|---|---|---|
| 1 | 数据口径 | 12 条轨迹用 **400ep** 库；norm_stats 显式指 `robomme-400ep`（与生产 40k run 同口径） |
| 2 | 第三块闭环驱动 | **改 motion-variance 三个脚本**（加 checkpoint 覆盖、motion 关闭态、`RUN_SUFFIX`），接受 48 集；**legacy-eval 那套一律禁用** |
| 3 | 12 条轨迹排卡 | **两波串行**，同 profile 同卡对，fsdp 2 |
| 4 | 关 4 已知 FAIL | bf16 `VT_FULL_VS_CACHED` 降为**观察项**；f32 `VT_FULL_VS_CACHED_F32` 升为**阻断判据** |
| 5 | `UV_CACHE_DIR` | `/scratch/hongze/.cache/uv`（全局约定） |
| 6 | run_name 与启动参数 | `t8-{c32,m32,c8,m8}-{a1,a2,b}` 共 12 个、排错 run 加 `-s100`；batch 8 / fsdp 2 / seed 42 / worker 4 / 1000 步 / `--log-interval 1` / `--save-interval 1` 全部作为启动覆盖参数，不改 `config.py` 默认值 |
| 7 | commit 版本 | `commitV9.0`（阶段 1）、`commitV9.1`、`commitV9.2`（阶段 2） |
| 8 | `-s100` 排错 run | **用户明确豁免第 17 条留档**：跑完清理、不留档、不作证据（本条为用户 2026-09-14 显式授权的例外） |
| 9 | C8/M8 结论范围 | **补独立手算边界样本**作为 8 帧语义证据（`hand_calc_8frame.py`），不接受缩水口径 |
| 10 | norm_stats验收修正 | 用户采用“文件 SHA＋解析后数组摘要双重检查”；文件SHA与数组摘要不跨域比较 |
| 11 | C8闭环前缀 | 用户同意“让评估脚本读取 checkpoint 保存的配置，自动算出正确长度”；serve_policy_mv按记忆预算和motion开关设置adapter.P，C8=1088、M8=1184 |
| 12 | 四个-s100的取证成本 | 用户采用“排错关闭完整状态摘要（推荐）”，只对这四个排错设置BENCH_CHECKSUM=0；100次更新、逐步标量及输入摘要保留，正式12条全部11份完整状态摘要保留 |

---

## 第一部分（给人看）

### 一句话

现在的训练把历史压成「最多 32 帧，每帧 4×4 = 16 个视觉 token」。本轮新增一种并列配置「最多 8 帧，每帧 8×8 = 64 个 token」。两种都是 512 个 token，模型一个参数不加、一条计算式不改，改动全在数据侧与在线记忆构造侧：**打一份新库**（第一块）、**让 Dataset 和训练入口认得这份新库并证明没改坏**（第二块）、**让在线记忆构造认得 8×8 并证明推理侧与训练侧交付一致**（第三块）。

### 第一块：数据预处理——把 8×8 特征打成一份新的 packed 库

#### 结论先行

- **不用重抽 SigLIP。** 源特征 npy 里三档 image/pos 键都在。对抗验证实开 5 个跨库跨 episode 的文件（400ep `episode_0/token_emb_0`、`episode_123/token_emb_7`、`episode_399/token_emb_50`；40ep `episode_0/token_emb_0`、`episode_39/token_emb_3`）确认键序、形状、dtype 完全一致，字节账 602,144 B 数据 + 807 B 头 = 602,951 B 与文件大小相符。
- **要做的只是新增一档库格式并打一份新库。** 现有 packed 库每帧只固化了 4×4 的 16 个 token。
- **规模**：400ep 新库约 30 GiB，40ep 约 3.4 GiB，磁盘余量 4.1 T。
- **不改旧库、不改旧格式、不动源库。** 新库落在各自库目录下的新子目录 `framesamp-8x8/`，与旧 `framesamp/` 并列。
- **motion 时间码跨网格一致这条前提已经被验过一次**：对抗验证用 numpy 直读 12 个 (episode, t) 组合，`pos_emb_4x4[0,0,:256]` 与 `pos_emb_8x8[0,0,:256]` 逐字节相同 12/12；根因是 `PosEmb3D` 的时间段（前 256 维）经 `einops.repeat` 跨 patch 复制、与 `spatial_size` 无关，且建库侧 `mem_buffer.py` 三档共用同一个 `temporal_pe` 对象。正式建库后仍按第一层检查全量跑一遍 `MOTION_POS_XGRID`，但它不再是未知数。

#### 完整数据链路与本轮新增的调用

现有链路的正式记录在 `docs/dataset-build-doc/4task-motion-400ep/launch.md`「命令序列」：8 个 Python 脚本加 `paths.sh`，独立调用 16 次（`scan_manifest` 1、`finalize_checks` 2、`run_local` 3、`pack_framesamp_store` 2、`compute_norm_stats` 1、`pack_motion_store` 2、`oracle_driver` 3、`compare_wan` 2），另有一行「附加检查」（`motion_checks.py` 六项、`dataloader_bench`、`test_padding_dtype.py`）。概要如下（`LIB=v1-store/datasets/4task-motion-400ep`，全部 `uv run --no-sync`；**这是历史记录，本轮一次都不重跑，其中的 `--gpus` 卡号不作为本轮口径**）：

```bash
source scripts/dataset/paths.sh; v1_prepare_dirs
# 1 清单 + 输入 sha
scan_manifest.py build … --episodes-per-task 100 --out $LIB/meta/episode_manifest.json
finalize_checks.py hash-inputs … --out $LIB/meta/input_manifest.json
# 2 SigLIP 建源库（写 7 键 npy + pkl）
run_local.py --stage siglip --lib $LIB …
finalize_checks.py check … --out $LIB/source --spot_check 1024
# 3 视觉 packed 库
pack_framesamp_store.py pack   --source $LIB/source --manifest … --out $LIB/framesamp --procs 48
pack_framesamp_store.py verify --store $LIB/framesamp --resume --procs 48
# 4 norm_stats
compute_norm_stats.py --dataset-path $LIB/source --output-dir v1-store/train-assets/mme_vla_suite/robomme-400ep
# 5 motion 支路
run_local.py --stage wan … ; run_local.py --stage encode …
pack_motion_store.py pack … --out $LIB/motion ; pack_motion_store.py verify --store $LIB/motion --resume
# 6 motion 旁证
oracle_driver.py vae / aggregate / encoder  →  compare_wan.py latents / tokens
```

训练时 `train.py --dataset-path $LIB/framesamp` 加 `MMEVLA_MOTION_STORE=$LIB/motion`，`FrameSampDataset` 把第 3 步和第 5 步的两张表拼起来。

**8×8 库只在第 3 步旁边多跑三条：**

```bash
# 3' 新增：同一个脚本、同一个源库，换键打一份新表（--procs 48 与旧库一致，96 核机器）
pack_framesamp_store.py pack   --layout framesamp-8x8-v1 --reader decode --source $LIB/source --manifest … --out $LIB/framesamp-8x8 --procs 48
pack_framesamp_store.py verify --store $LIB/framesamp-8x8 --resume --procs 48
# 3'' 新增：汇合点检查——两张 pos 表的 patch 0 时间码逐位相同，另对 image 行做不经规格表的 np.load 直读抽样
xgrid_pos_check.py --store-4x4 $LIB/framesamp --store-8x8 $LIB/framesamp-8x8 --source $LIB/source --image-spot 512
```

训练时只换 `--dataset-path $LIB/framesamp-8x8`，`MMEVLA_MOTION_STORE` 还是指原来的 `$LIB/motion`。

代码上加的东西对应这三条命令：`pack_framesamp_store.py` 加 `--layout` 参数并让 decode 守卫、写侧、verify、空间预检、meta 写入全部按规格表走；`framesamp_store.py` 加规格表让 `StoreMeta.load`、`run_fast_checks`、`run_full_checks` 和 reader 认 `framesamp-8x8-v1`；`framesamp_dataset.py` 放开白名单并核对库规格；`xgrid_pos_check.py` 是新脚本。第 1、2、4、5、6 步的脚本一行不改。

40ep 开发库同样的三条命令先跑一遍，`--out` 换成 `4task-motion-40ep/framesamp-8x8`。

#### 源数据长什么样

400ep 库 `v1-store/datasets/4task-motion-400ep/source/`，40ep 库同构：

| 项 | 400ep | 40ep |
|---|---:|---:|
| `features/episode_*` 目录数 | 400 | 40 |
| 帧总数（= `episode_manifest.json` 的 `totals.timesteps`） | 123,044 | 13,756 |
| 执行样本数（`totals.exec_samples`，= `data/*.pkl` 个数） | 101,066 | 11,530 |
| 最长 episode 帧数（决定 pos 表行数，两库相同） | 586 | 586 |
| `source/` 体积 | 107 G | 13 G |
| `exec_start_idx == 0` 的 episode 数（决定短样本 fixture 配额） | 200 | 20 |

每帧一个 `token_emb_{t}.npy`，是 `np.save` 存的 pickle dict（读取必须 `allow_pickle=True`），7 个键：

```
image_emb_8x8 (1,64,2048) bf16  262,144 B   ← 本轮新用
image_emb_4x4 (1,16,2048) bf16   65,536 B   ← 旧库用
image_emb_2x2 (1, 4,2048) bf16   16,384 B
pos_emb_8x8   (1,64, 768) f32   196,608 B   ← 本轮新用
pos_emb_4x4   (1,16, 768) f32    49,152 B   ← 旧库用
pos_emb_2x2   (1, 4, 768) f32    12,288 B
state_emb     (8,)        f32        32 B   ← 两档共用
```

写这个 dict 的是 `dataset_builder/mem_buffer.py::MemoryBuffer.add_buffer`：image 三档来自同一次 SigLIP 前向后 `pool_tokens_to_size` 到 64/16/4；**pos 三档不是池化来的**，是 `__init__` 里同一个 `PosEmb3D` 实例预算好的三张查表，`add_buffer` 按 `step_idx` 切行。落盘在 `dataset_builder/build_robomme_dataset.py::DatasetProcessor._process_episode`。40ep 与 400ep 是**两个独立建的库**（manifest sha 不同、inode 不同、取样口径 400ep 每任务 `raw_ep_idx 0..99`、40ep `0..9`），不是 subset 引用；但 40ep 的内容恰是 400ep 每任务前 10 集，所以「40ep 通过」不覆盖 400ep 的 `raw_ep_idx 10..99`。

#### 现有 packed 库为什么不能直接放 8×8

现有格式 `framesamp-4x4-v1` 由 `src/mme_vla_suite/datastore/framesamp_store.py` 顶部一组常量钉死，每一处都和 4×4 绑定：

- 形状与字节：`IMAGE_ROW_SHAPE=(16,2048)`、`POS_ROW_SHAPE=(16,768)`、`IMAGE_ROW_BYTES=16*2048*2`、`POS_ROW_BYTES=16*768*4`。
- 键名与文件名：`IMAGE_KEY="image_emb_4x4"`、`POS_KEY="pos_emb_4x4"`、`POS_TABLE_RELPATH="pos_emb_4x4.f32.bin"`、`IMAGE_PART_DIR="image_emb_4x4"`。
- 源 npy 固定偏移（slice 读档用）：`SOURCE_IMAGE_OFFSET=262595`、`SOURCE_POS_OFFSET=541352`、`SOURCE_STATE_OFFSET=602906`，只对 4×4 窗口成立。
- 读侧与校验侧引用这些常量的地方，**一个都不能漏**：`StoreMeta.load` 校验 `tables` 形制与 **parts 的 bytes 账（`p["bytes"] != p["num_rows"] * IMAGE_ROW_BYTES`）**；`run_fast_checks` 的小表大小账；**`run_full_checks` 硬取 `POS_KEY` / `POS_TABLE_RELPATH`**；`FrameSampStore.__init__` 的 pos 表 reshape；`read_image_rows` 的行字节与 reshape。
- 打包脚本 `scripts/dataset/pack_framesamp_store.py` 引用这些常量共 27 行，分布在 `read_frame_decode` 守卫、`read_frame_slice` / `_pin_slice_prefix`、`build_small_tables` / `_state_chunk_worker`、`_pack_part_worker`（写侧三重校验的宿主）、`cmd_pack`（空间预检、`IMAGE_PART_DIR`、`POS_TABLE_RELPATH`、meta 写入段）。CLI 没有网格参数。

所以「把 yaml 里 `token_per_image` 改成 64」在 Dataset 构造时就会被 `FrameSampDataset.__init__` 的硬相等守卫 `_req((budget, token_per_image, num_views) == (512, 16, 1))` 拦下；就算放开，旧库每帧也只有 16 个 token。

#### 新库怎么建

**规格表驱动。** 在 `framesamp_store.py` 里把上面那组散常量收成一张规格表，键是布局名，两档：

| 布局名 | 网格 | 每帧 token | image 键 / 行形状 / 行字节 | pos 键 / 行形状 / 行字节 | part 目录 | pos 表文件 |
|---|---|---:|---|---|---|---|
| `framesamp-4x4-v1`（不变） | 4×4 | 16 | `image_emb_4x4` / (16,2048) / 65,536 | `pos_emb_4x4` / (16,768) / 49,152 | `image_emb_4x4/` | `pos_emb_4x4.f32.bin` |
| `framesamp-8x8-v1`（新增） | 8×8 | 64 | `image_emb_8x8` / (64,2048) / 262,144 | `pos_emb_8x8` / (64,768) / 196,608 | `image_emb_8x8/` | `pos_emb_8x8.f32.bin` |

state 表两档相同：`state_emb` (8,) f32，32 B/行。`StoreMeta.load` 改成先读 meta 里的 `layout`，查表得规格，再校验 `tables` 三键与 **parts bytes 账**逐项相符；`run_fast_checks`、`run_full_checks`、reader、packer 的写侧与 verify 全部从同一张表取键名、形状、字节数。旧库 meta 无需迁移，`layout` 字段本来就有。**「不改」清单只剩真正与网格无关的东西**：`plan_parts`、锁协议、`atomic_write_bytes`、resume 逻辑、`_spot_entries`、preadv / fadvise / `_runs_of`、`__reduce__` 禁 pickle、`SOURCE_NPY_SIZE` / `SOURCE_STATE_OFFSET` / `TARGET_PARTS` / `STATE_*` / `ROW_DIGEST_*`。

**新库落点**（实体目录，不建外链）：

```
v1-store/datasets/4task-motion-40ep/framesamp-8x8/     开发库，先建
v1-store/datasets/4task-motion-400ep/framesamp-8x8/    正式库，后建
```

源 pkl、`episode_manifest.json`、motion store 通过 meta 里的 `source_dataset_root`、`manifest_path` 与显式路径复用，不复制。全仓库没有按目录名判 layout 的逻辑（只有 `framesamp_store.py` 与 `motion_store.py` 读 meta 的 `layout` 字段），`framesamp-8x8/` 这个目录名安全。

**打包流程与旧库完全同一套**，只是键换成 8×8：

1. `pack` 子命令加 `--layout` 参数（默认 `framesamp-4x4-v1`，旧行为不变），`--reader decode` 首跑（完整反序列化每个 npy 后取键，零布局假设；slice 档在 decode 建库并 verify 通过前不用）。
2. part 切分由 `plan_parts` 按 manifest 的 episode 边界贪心，只看帧数不看行宽，所以 **8×8 库的 part 边界与 4×4 库逐一相同**：400ep 31 个 part，40ep 22 个 part，只是每个 part 的字节数是原来的 4 倍（400ep 单 part 约 1.04 GB，旧库实测 part_0 264,437,760 B × 4）。
3. 写侧三重校验照旧：每帧 pos 与 `pos_table[t]` 逐字节相同（钉死「pos 只依赖帧号」）、每帧 state 与 state 表同行相同、每 episode slab 落盘后读回比对且 sha 取读回字节。
4. `verify` 全量档：每个 part 的每一帧重新 decode 源 npy，经**真实读 API**（`FrameSampStore.read_image_rows / pos_rows / state_rows`）逐字节比对三键，同时逐行算 blake2b-128 写 `meta/row_digests.blake2b.bin`。判据 `scanned == num_rows`、`mismatches == 0`，通过才回填 `status="verified"` 并释放 `pack.lock`。抽样档 `--sample N` 只作开发快检，不作交付判定。

**空间预检要改。** `cmd_pack` 的全量路径写死 `floor = 40 * 10**9`，对 8×8 的 32.4 GB 库只剩约 7.6 GB 余量。改成 `floor = max(40 GB, 1.25 × need)`，其中 `need = num_rows × image_row_bytes + num_rows × 32 + max_timesteps × pos_row_bytes` 按规格表现算（8×8 下 1.25 × need ≈ 40.5 GB，与 40 GB 几乎等价，改的意义是让公式随规格走）。临时 part 文件是同目录 rename，不额外占空间。起跑前照第 14 条先 `ls -ld <输出根>`。

**预计字节账**（公式推算，基数 `num_rows` = timesteps）：

| 表 | 400ep | 40ep |
|---|---:|---:|
| image（rows × 262,144） | 32,255,246,336 B ≈ 30.0 GiB | 3,606,052,864 B ≈ 3.36 GiB |
| pos（586 × 196,608） | 115,212,288 B | 同 |
| state（rows × 32） | 3,937,408 B | 440,192 B |
| row_digests（rows × 16） | 1,968,704 B | 220,096 B |

**耗时预期**：旧库 meta 记的 pack 5 s / verify 2 s（48 进程）是源库完全命中 page cache 时的数字（本机 1.12 TB RAM），不能当基准。decode 档下 `pack` 实际把每个 603 KB 的 npy **完整反序列化两遍**（`build_small_tables` 为取 32 B state 一遍、`_pack_part_worker` 取 image 一遍），`verify` 再一遍，源文件总读量约 3 × 74 GB ≈ 222 GB，瓶颈是 123,044 次 `allow_pickle` 反序列化而不是 `/dev/md0`（raid0 × 8 NVMe）带宽。冷缓存下按分钟到几十分钟量级估计，按第 7 条视作长任务：detached tmux、日志三件套、挂 Monitor。

#### 怎么证明新库没改数

分两层，都在不启动训练的前提下完成。

**第一层：库层。** `VERIFY_PACK=PASS scanned=<num_rows> mismatches=0` 证明库里每一行三键与源 npy 经规格表读出的键逐字节相同。**注意 pack 与 verify 共用同一张规格表与同一个 `read_frame_decode`，规格表里的对称错误两侧看不出来**，所以另加三条不经规格表的独立检查（都在 `xgrid_pos_check.py` 里）：

- **image 直读抽样**：随机 512 帧，用 numpy `np.load(allow_pickle=True)` 直接取 `["image_emb_8x8"]` 的 `.tobytes()`，与库 `read_image_rows` 逐字节比。判定行 `IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`。
- **pos 表跨 episode 同值**：随机若干 episode 的若干帧，源 npy 的 `pos_emb_8x8[t]` 与库 `pos_table[t]` 逐字节相同。
- **motion 时间码跨网格一致**：motion 路取的是 `store.pos_rows(f_m)[:, 0, :256]`，即第 0 个 patch 的前 256 维（`pos_dim` 由 yaml 给、`__init__` 强制等于 `pos.input_dim // 3` = 256）。对全部 586 个帧号比对两库 pos 表该切片的 raw 字节，另对源 npy 抽 64 帧比 `pos_emb_4x4[0,0,:256]` 与 `pos_emb_8x8[0,0,:256]`。判定行 `MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0`。**这一条不过，就不得让 8 帧 motion 配置起跑。**

**第二层：装配层（Dataset 输出对参考链）。** 8 帧从没跑过，没有现成参考，所以写一条**只用于验证的独立参考链** `scripts/training/tests/ref_npy_dataset.py::RefNpyFrameSampDataset`：直接读源 npy 的 `image_emb_8x8 / pos_emb_8x8 / state_emb`，选帧调 `shared/sampling.py::even_sampling_indices(step, 8)`，补零调 `shared/data_utils.py::right_padding_token_emb`（与 `dataset_builder/data_utils.py` 那份逐字相同，生产侧 import 的是 `shared/`；三处补零沿用输入 dtype、mask 固定 `np.bool_`，所以交付 image bf16、pos f32、state f32），然后 `reshape(-1, dim)`、`np.repeat(…, 64)`，state 归一化**照抄** `FrameSampDataset._normalize_state` 的 quantile 公式（`(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`，norm stats 是 f64 所以输出 f64；顺序与候选链一致：先补零、再 repeat、再归一化），pkl 身份互校同式。motion 四键复用 `datastore/motion_store.py::visible_motion_rows` 与 `MotionStore.rows`，`motion_pos` 取 npy `pos_emb_8x8[0, 0, :256]`，`mem_order` 调 `shared/sampling.py::memory_order(pad_times(frames, 8), 64, mtimes)`。这条链不进生产 loader、不作 fallback。它必须实现与 `FrameSampDataset` 同样的 spawn 生命周期契约（`__getstate__` 不带 fd / mmap / `MotionStore` 实例过进程，按 owner pid 懒重建），并在 `spawn_matrix.py` 上同样跑一遍 worker 矩阵。

**这条参考链只是半独立的，要说清它证明什么。** 它与候选链共用 `even_sampling_indices`、`memory_order`、`pad_times`、`visible_motion_rows`、`MotionStore.rows` 与同一张 motion 表，所以下表 7 行里「数据身份」「选帧」「motion_emb / motion_mask」「mem_order」四行是两侧调同一函数同一数据的同义反复，真正被检验的是：库 image 行 vs npy、库 pos 行 vs npy（含 `motion_pos` 切片）、`_pad` vs `right_padding_token_emb` 的字节等价、新守卫拒绝错配。8 帧参数下「选帧公式取的是不是想要的 8 帧」「`mem_order` 交错是不是设计想要的」这类语义问题，两侧会一起错。**按用户裁决第 9 项，另加一条不调仓库函数的独立手算证据**：`scripts/training/tests/hand_calc_8frame.py` 对十几个边界样本（step ∈ {6, 7, 8, 9}、首个 motion 窗前后 t ∈ {31, 32, 33}、episode 段边界 `exec_start_idx ∈ {0, 66, 114}` 各取一个）用脚本内手写公式算出期望的帧索引列表、`mem_order` 排列、可见 motion 窗起点，与候选 Dataset 输出逐项比，判定行 `HAND_CALC_8FRAME=PASS samples=<n> mismatches=0`。

参考链 vs 候选链（新 8×8 库上的 `FrameSampDataset`）的对拍项：

| 检查 | 方法 | 通过条件 | 独立性 |
|---|---|---|---|
| 数据身份 | 全部 episode 的 `exec_start_idx`、样本长度、pkl 内 `(epis_idx, step_idx)` 与清单推导一致 | 零差异 | 同义反复（同 manifest 同 pkl） |
| 选帧 | 断言 `_max_frames == 8`、`_tokens_per_frame == 64`；全部合法 step 比较 `even_sampling_indices(step, 8)`；`step < 8` 走 `range(step+1)`，`step ≥ 8` 走 `np.linspace(0, step, 8)` 向下取整（numpy 整数 dtype linspace 是 floor） | 索引列表逐项相同 | 同义反复（同函数） |
| 定点样本 | 短样本档 step ∈ {0,1,2,5,6}（补零分支），满长档 {7,8,9}（**7 仍走 range 分支**，8、9 走 linspace），**motion 档 {33,34,35}**（`exec_start_idx=0` 下首个 motion 窗 `u+32 ≤ t` 成立，k ≥ 1；不加这一档，前两档 1600 个样本的 motion 四键全是 padding）；每档 `PER_STEP` 个（400ep 200、40ep 20，由 `exec_start_idx==0` 的 episode 数决定并在判定行显式打出），另随机 1000 个；Dataset 输出与 transform 后所有键 raw 字节、dtype、shape、None 状态零差异 | 零差异 | 真检验 |
| 真实 batch | 200 个经真实 `transform_dataset` + `_collate_fn` 的 batch（全短 / 全满 / 混合 / 随机各 50，分档按 `max_frames` 推：8 帧下短 = step ≤ 6、满 = step ≥ 7） | 所有键零差异 | 真检验 |
| 抽样顺序 | 同 seed、同 worker 数下真实 loader 消费的 index 序列 | 前 8000 个逐项相同 | 真检验 |
| worker 生命周期 | 40ep 8×8 库上 `spawn_matrix.py --yaml <8×8 yaml>` w0/w1/w4/w16 各 2 个 epoch，packed 与 refnpy 各跑一遍 | 跑通，主进程 fd 数回到基线 | 真检验 |
| 拒绝错误输入 | 错 layout、8×8 meta 配 4×4 目录、未 verified、残留 `pack.lock`、错 manifest、错 motion 来源、配置 `token_per_image=16` 却指向 8×8 库 | 全部明确报错，不回退、不截断 | 真检验 |
| 手算边界 | `hand_calc_8frame.py` | `HAND_CALC_8FRAME=PASS` | 独立 |

比较一律先核键集与结构（含 None 键），再核 dtype、shape、raw 字节；**不继承**旧 dtype 重构允许的「4 个 batch raw 失配」（那是一次真实 dtype 变更留下的，现版本两侧补零 dtype 一致，8×8 下不会重现），也不用 canonical 口径掩盖差异。

#### 顺序与留档

1. 先在 40ep 库建 8×8 开发库，跑通 pack → verify → 上面全部检查（40ep 上 `PER_STEP=20`，判定行显式标注 `per_step=20`，只作开发快检）。
2. 再建 400ep 正式库，同样全量 verify + 全部检查（`PER_STEP=200`）。
3. tmux 会话名前缀 `p8-`（如 `p8-pack-40`、`p8-verify-400`、`p8-fx-*`）。**本轮全部 tmux 前缀集中列于第二部分第 6 节**，清理只按名单逐个 `tmux kill-session -t`。
4. 留档 `docs/dataset-build-doc/4task-motion-40ep-framesamp-8x8/` 与 `4task-motion-400ep-framesamp-8x8/`：commit、精确命令、判定行、`store_meta.json` 副本、耗时、体积、存储介质（AWS 本地 NVMe RAID `/dev/md0`）。

### 第二块：训练——让 Dataset 与训练入口接受 8 帧配置，并用 1000 步真实训练证明等价

#### 结论先行

- **模型侧一处不改。** 对 `src/mme_vla_suite/models/` 全目录静态核对（对抗验证复核）：无任何 `token_per_image` / `max_frames` / `num_frames` 引用；`HistoryPi0Config.inputs_spec` 的形状全部从 `history_config.budget`、`memory_feature.*.input_dim` 推出；`mem_order` 长度是 `budget + motion.budget` 算出的 608；`PerceptualMemory.__call__` 只检查 `static_image_emb.shape[1] == budget`；`FeatureEncoder` 只碰最后一维。16/32/512/608 只出现在注释与 Gemma 架构常量里。
- **改动集中在**：`FrameSampDataset.__init__` 的配置白名单与 store 规格匹配；两份新 yaml；训练入口的 jax 编译缓存落点；对拍量具（bench、gate、指纹、fixture 工具、单步梯度、手算脚本）。
- **验收是四个 profile 各三条轨迹**（参考自重复 A1/A2 + 候选 B），共 12 条 × 1000 次更新，逐步浮点位比对。

#### 配置怎么写

新增两份 yaml，与旧文件逐键相同、只改一个数：

| 文件 | 相对旧文件的差异 |
|---|---|
| `perceptual-framesamp-context-8frame-8x8.yaml` | `token_per_image: 16` → `64`，其余（含 `motion.enabled: false`、**`streaming_obs_horizon: 16`**）照抄 `perceptual-framesamp-context.yaml` |
| `perceptual-framesamp-context-8frame-8x8-motion.yaml` | 同上，照抄 `perceptual-framesamp-context-motion.yaml`（`motion.enabled: true`），另把 `motion.online_gpu: 1` 改为 `4`（GPU 约束兜底；训练不读此键，评估驱动仍须逐片显式给卡号） |

帧数在 yaml 里没有键，是派生量：`FrameSampDataset.__init__` 里 `_max_frames = budget // (token_per_image * num_views)` = 512 // 64 = **8**，`_tokens_per_frame` = 64。yaml 里没有 `max_frames` / `grid` / `pool_size` 之类与帧数或网格相关的键。**`streaming_obs_horizon: 16` 就在 yaml 第 4 行（motion 版第 6 行），必须照抄**，`train.py::main` 的交叉断言 `streaming_obs_horizon == 16 ⇒ action_horizon == 20` 读的就是它，漏抄即报错；`action_horizon=20` 在 `training/config.py` 的 `_CONFIGS` 条目里。motion 子键（`budget: 96`、`stride: 16`、`window_frames: 33`、`pos_dim: 256`、`store_path`）一个不改。

四个验收档位：

| 档位 | yaml | 帧上限 | 每帧 token | motion |
|---|---|---:|---:|---|
| C32 | `perceptual-framesamp-context.yaml` | 32 | 16 | 关 |
| M32 | `perceptual-framesamp-context-motion.yaml` | 32 | 16 | 开 |
| C8 | `perceptual-framesamp-context-8frame-8x8.yaml` | 8 | 64 | 关 |
| M8 | `perceptual-framesamp-context-8frame-8x8-motion.yaml` | 8 | 64 | 开 |

#### Dataset 改什么、不改什么

只改 `src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__init__` 里两行守卫：

- 硬相等 `(budget, token_per_image, num_views) == (512, 16, 1)` 改成白名单 `∈ {(512, 16, 1), (512, 64, 1)}`。
- 新增 store 规格匹配：从 `StoreMeta.spec` 取规格，要求 `spec.tokens_per_frame == token_per_image * num_views`。8×8 配置指向 4×4 库、或反过来，都在构造时报错，不静默取前 16 个 token、不插值。

其余逻辑本来就是通用的，逐项核过：`_pad` 用 `_max_frames` 做目标长度并按输入 dtype 预分配；reshape 是 `img.reshape(-1, img.shape[-1])`；state 与 mask 的 `np.repeat` 用 `_tokens_per_frame`；motion 块的 `pad_times(frames_arr, self._max_frames)` 与 `memory_order(ftimes, self._tokens_per_frame, mtimes)` 都读实例变量。`_normalize_state` 不动。交付形状在两档下相同：`static_image_emb (512,2048) bf16`、`static_pos_emb (512,768) f32`、`static_state_emb (512,8) f64`、`static_mask (512,) bool`；motion 开启态 `motion_emb (96,768) f32`、`motion_pos (96,256) f32`、`motion_mask (96,) bool`、`mem_order (608,) int32`，关闭态四键 None（`_NONE_KEYS` 实际是九键，含 `prompt`，不动）。同文件里三处写死 32 帧口径的注释（`_pad` 的「目标长度 `_max_frames(32)`」、`__getitem__` 的 `(n,16,2048)` 与 `(32,16,2048)→(512,2048)`）随代码一起改成按变量描述。

`mem_order` 的**值**跨配置会变：排序键是 `帧时刻 × 2 + 类型`，`np.argsort(kind="stable")`，帧路 token 从 32 组 × 16 个变成 8 组 × 64 个，motion 在混合序列里的位置随之移动。这是设计内的变化，只要求「按时刻稳定排序、同刻视觉在 motion 前、padding 落尾、是 0..607 的合法置换」，不要求跨配置相等；同 profile 内 A1/A2/B 三侧输入相同、输出逐位相同，与此不矛盾。

`dataloader.py::_create_framesamp_dataset` 与四道闸（`require_no_pack_lock` → `StoreMeta.load` → `require_verified` → `_motion_gates`）不改，四道闸全部在该函数体内。`_motion_gates` 的 `check_same_source` 比的是 framesamp meta 与 motion meta 的 `manifest_sha256` 加逐 episode index 身份，8×8 库与 4×4 库同源同 manifest，天然通过。**两份旧 yaml 的 `motion.store_path` 写死 `4task-motion-40ep/motion`**，与 400ep 库 manifest sha 不同，所以 **M32 与 M8 起跑都必须设 `MMEVLA_MOTION_STORE=$L/motion`**，否则 `check_same_source` 报串配。

#### 训练入口、缓存落点与资产口径

`scripts/training/train.py::main` 与 `scripts/training/tests/single_step_grad.py::main` 都硬编码 `jax.config.update("jax_compilation_cache_dir", "~/.cache/jax_<exp_name>")`。实测 `~/.cache/` 下已有 **8 个** `jax_*` 目录（`jax_awsprod40k-b128-motion`、`jax_bench-b1-motion`、`jax_bench-b2-w16`、`jax_bench-b3-w12`、`jax_bench-b4-8gpu-w16`、`jax_bench-b5-8gpu-w32`、`jax_repro-4a100-fsdp4-40gsim`、`jax_repro-4a100-fsdp4-80g`），违反第 13 条环境 B 红线；本轮不清理它们（另立任务），但本轮所有 run 不得再往里写。做法：两处入口改成 `os.environ.get("MMEVLA_JAX_CACHE_DIR") or "~/.cache/jax_<exp_name>"`，设了就用、没设保持旧行为。`bench_train_steps.py` 调的是 `train.main`，自动继承。启动时统一 `source scripts/training/paths.sh`（它设 `XDG_CACHE_HOME`、`HF_HOME`、`WANDB_*` 到 `v1-store/cache/`，不动 HOME；它是 `set -euo pipefail` 且大量 `readonly`，**同一 shell 只 source 一次，source 之后的「期望不存在」检查必须写成 `[ ! -e … ] || exit 1` 而不是裸 `ls`**），再显式 `export UV_CACHE_DIR=/scratch/hongze/.cache/uv MMEVLA_JAX_CACHE_DIR=$V1/cache/jax/<run_name>`（`V1` 一律取主树绝对路径，见阶段 5 命令）。`UV_CACHE_DIR` 落点按用户裁决第 5 项取全局约定。

**norm_stats 必须显式指 400ep（用户裁决第 1 项）。** 训练用的配置条目 `mme_vla_suite` 没有 `AssetsConfig`，默认会读 `train-assets/mme_vla_suite/robomme/norm_stats.json`（sha `f332bbd3…`，40ep 口径）；400ep 交付件是 `train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`（sha `750a8e9b…`），两文件大小与内容都不同。生产 40k run 用的是 `mme_vla_suite_b128` 条目，它把 400ep 路径写进了 `AssetsConfig`。本轮 12 条轨迹与全部 fixture / 单步梯度命令一律加 tyro 覆盖 `--data.assets.assets-dir v1-store/train-assets/mme_vla_suite/robomme-400ep --data.assets.asset-id robomme`（`DataConfigFactory.assets: AssetsConfig` 是 dataclass 字段，tyro 展开为该嵌套 flag；阶段 1 量具自检时先用 `--help` 核对拼写，若 tyro 不接受则改为在 `_CONFIGS` 新增 `mme_vla_suite_400ep` 条目、照抄 `mme_vla_suite` 只加 assets）。bench在 `run_meta.json` 里记录 `norm_stats_file_sha256`、`norm_stats_actual` 和 `norm_stats_expected`。gate要求文件SHA为750a8e9b…，且实际loader数组摘要 == 从该文件解析的数组摘要 == 指纹中的 `assets.norm_stats_arrays`，字典必须含8项；每个数组的dtype、shape和原始字节共同参与该项SHA。文件SHA和数组摘要属于不同哈希域，按用户确认不直接比较它们。

训练本身的 CLI 是 tyro 从 `TrainConfig` 生成的：位置参数选条目（`mme_vla_suite` 4 worker / b64，`mme_vla_suite_b128` 8 worker / b128），`--model.history-config=<yaml 文件名>` 按仓库根相对路径解析（`get_history_config` 用 cwd 相对路径，cwd 必须是仓库根或 worktree 根），`--dataset-path` 指 packed 库根。checkpoint 与 run 根的真实落点是 `checkpoint_base_dir / <配置条目名> / <exp_name>`（`TrainConfig.checkpoint_dir`），即 **`$V1/train-runs/<run>/mme_vla_suite/<run>/`**，`history_config.resolved.yaml` 等三件套与 `999/` 都在这一层。8 帧训练只需把 `--dataset-path` 指到 `framesamp-8x8/`、yaml 换成新文件名，没有别的开关。

#### 第二块验收：同机真实训练 1000 次更新

**这一节在干什么。** 第一块只证明了「喂进模型的数据一样」，但数据一样不等于训练一样，中间还有 transforms、collate、JAX 交付、前向、反向、优化器一整串。最稳的办法是真的训一遍，把每一步的数字拿出来逐位比。所以这里要跑 12 条正式 run，每条 1000 次参数更新，两两比对。

**总共跑几组。** 四个配置，每个配置三条，共 12 条：

| 配置 | motion | run 1（A1，参考） | run 2（A2，参考重跑） | run 3（B，候选） | 跑在哪对卡 |
|---|---|---|---|---|---|
| C32 | 关 | 旧代码 + 旧 4×4 库 | 同 A1 再跑一遍 | 新代码 + 旧 4×4 库 | GPU 4,5（第一波） |
| M32 | 开 | 旧代码 + 旧 4×4 库 | 同 A1 再跑一遍 | 新代码 + 旧 4×4 库 | GPU 6,7（第一波） |
| C8 | 关 | 新代码 + 直读源 npy 的验证用 Dataset | 同 A1 再跑一遍 | 新代码 + 新 8×8 库 | GPU 4,5（第二波） |
| M8 | 开 | 新代码 + 直读源 npy 的验证用 Dataset | 同 A1 再跑一遍 | 新代码 + 新 8×8 库 | GPU 6,7（第二波） |

「旧代码」是阶段 1 结束时冻结的参考 commit `REF`（量具、yaml、缓存落点已加，但库格式与 Dataset 还没改）；「新代码」是阶段 2 结束的候选 commit `CAND`。每条都是 1000 步、batch 8（全局，两卡各 4）、seed 42、2 卡 fsdp 2。

**A 侧怎么保证跑的真是 REF 的代码（对抗验证发现的最大坑）。** 主树 `.venv` 的 editable 安装是一条绝对路径 `.pth`（`_editable_impl_openpi.pth` → 主树 `src/`），worktree 的 `src/` 不会自动进 `sys.path`。REF↔CAND 的差异恰好全在 `src/`，不处理就会让 A1/A2/B 跑同一份代码、`PASS` 无信息量。落地：A 侧启动一律 `cd <REF worktree> && PYTHONPATH=<REF worktree>/src UV_PROJECT_ENVIRONMENT=<主树>/.venv uv run --no-sync …`（`aws-t2-ref-s100/launch.md` 的既有做法），bench 起跑时打印并记录 `importlib.util.find_spec("mme_vla_suite").origin`、`train.__file__`、`framesamp_dataset.__file__`、`framesamp_store.__file__` 四项到 `run_meta.json`，gate 要求 A 侧全部落在 REF worktree、B 侧全部落在主树，不符即 FAIL。

**比法。** 每一行里先比 run 1 和 run 2：这两条代码、数据、seed 全一样，如果它们自己都对不上，说明管线有随机抖动，后面的比较没有意义。run 1 == run 2 成立后，再比 run 1 和 run 3。行与行之间不比：32 帧和 8 帧本来就该不一样，motion 开和关也不该一样。

**「直读源 npy 的验证用 Dataset」和「新 8×8 库」是什么、有什么区别。** 数据是同一份，读法不同：

- **新 8×8 库**是第一块打出来的三张大表 `framesamp-8x8/`，训练时 `FrameSampDataset` 按行号从大表里读 8 帧。这是生产链路。它是本轮新写的代码加新打的库，两样都是待验证的东西。
- **直读源 npy 的验证用 Dataset**（`RefNpyFrameSampDataset`）不经过大表，直接打开源库里每帧的 `token_emb_{t}.npy`，把 8 帧的 `image_emb_8x8 / pos_emb_8x8 / state_emb` 拼起来。它绕开了「打包」和「新 Dataset」两个本轮改动的环节。它慢（每样本 `np.load` 8 个 603 KB 文件，与 packed 侧 `preadv` 8 × 256 KiB 不在一个量级），不进生产，只为对拍存在。它证明的是「誊抄没抄错」，不证明 8 帧语义本身（那由 `hand_calc_8frame.py` 与第三块承担）。32 帧配置不需要它，因为改动前的代码跑旧库本身就是正确答案。

参考侧的 8 帧 Dataset 通过 `bench_train_steps.py` 的测试专用开关 `BENCH_DATASET_IMPL=refnpy`（**新增**）注入：bench 已经在 monkeypatch `train.wandb`、`_checkpoints.save_state`、`init_train_state`、`TorchDataLoader.__iter__`，再替换 `dataloader._create_framesamp_dataset` 是同一类只读注入。**但 `train.py::main` 在建 loader 之前还会独立调一次 `init_history_config(..., framesamp_root=config.dataset_path)`，里面无条件 `StoreMeta.load`；`--dataset-path $L/source` 会在那里先 `FileNotFoundError`。** 所以 refnpy 模式下 bench 同时包一层 `train.init_history_config`，把 `framesamp_root` 改传 `None`（provenance 里记 `framesamp=refnpy` 与 `BENCH_REF_SOURCE` / `BENCH_REF_MANIFEST` 的 sha），并把收尾的 epoch 样本数改从 `BENCH_REF_MANIFEST` 的 `totals.exec_samples` 读，不再碰 `<dataset_path>/meta`。生产 loader 不加任何 fallback。

**每一步比什么。** 训练用现成的 `bench_train_steps.py` 起，它不改训练逻辑，只在旁边记录四样东西，两条轨迹逐位比：

1. 每一步的 `loss`、`grad_norm`、`llm_grad_norm`、`mem_enc_norm`、`param_norm` 五个标量，记的是浮点数的 `float.hex()`。
2. 在 11 个摘要点把完整 TrainState（params 55 叶 + AdamW 动量 66 叶 + EMA 55 叶 + step 1 叶 = 177 叶；motion 开启态 193 叶）逐叶 `sha256(dtype‖shape‖bytes)`。**摘要点按更新次数 `state_step` 定义**：0 是 `init_train_state` 之后的初态；正整数 k 对应循环步 `loop_step = k-1` 更新之后。`train.py` 的 save 分支条件是 `step % save_interval == 0 and step > start_step` 或末步，所以 **loop_step 0（state_step 1）永远轮不到记录器**，摘要点集合定为 `state_step ∈ {0, 2, 3, 25, 50, 100, 200, 400, 600, 800, 1000}`，对应 `loop_step ∈ {init, 1, 2, 24, 49, 99, 199, 399, 599, 799, 999}`。启动参数：`--save-interval 1`（让每步都轮到记录器）+ `BENCH_DIGEST_INTERVAL=1000`（间隔本身不命中）+ `BENCH_EXTRA_DIGEST_STEPS=1,2,24,49,99,199,399,599,799`（末步 999 由 gate 自动含）。记录写 `phase ∈ {init, post_update}`、`loop_step`、`state_step` 三字段。
3. 每个记录步喂进模型的那个 batch（collate 后 host 侧 numpy）逐键 raw sha，外加 `sample_indices`。batch 编号 idx 与 loop_step 同号，idx k 的 batch 产生 state_step k+1；记录 idx ∈ {0, 1, 2, 24, 49, 99, 199, 399, 599, 799, 999}。None 键显式记 `null`。
4. 全程主进程抽取的 index 序列。

判据一律逐位，另加两类对抗验证指出的空缺：

- `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`，并核 `scalars_hex.tsv` 表头恰为六列（该文件由 `project_scalars.py` 从 `metrics.jsonl` 投影，阶段 5 命令里显式跑这一步）。
- `STATE_DIGEST` 在 11 个摘要点上 `mismatch=0`，按 `state_step` 对齐。
- `BATCH_DIGEST` raw 口径在 11 个记录步上 `mismatch=0`，`sample_indices` 逐位相同，**键名集合按 profile 钉死**（关闭态 12 个数组叶 + 4 个 None 键；开启态 16 个数组叶），不只比计数。
- `INDEX_SEQ` 前 8000 个 index 逐项相同，且两侧 `n ≥ 8072`（1001 个 batch + 4 worker × prefetch 2 = 1009 × 8）。
- `n_leaves` 三侧相等，且按 profile 软核对 177（C）/ 193（M）；变了就 FAIL。
- **有限性**：五个标量全程 `isfinite`，全部摘要叶 `isfinite`（NaN 的 `float.hex()` 两侧恒等，没有这条 NaN 训练会全项 PASS）。
- **活性**：`loss` 首末不等且末值小于首值、`param_norm` 首末不等、`mem_enc_norm > 0`、state_step 0 与 1000 的 params 摘要不等。
- `IMPORT_ORIGIN`：A 侧四个 `__file__` 在 REF worktree、B 侧在主树。
- `NORM_STATS_ACTUAL`：实际资产文件SHA == 指纹文件SHA == `750a8e9b…`；另逐项比较 `norm_stats_actual == norm_stats_expected == fingerprint.assets.norm_stats_arrays`，共8个数组摘要。
- 基线自己不重复（A1 ≠ A2）就不能放宽阈值宣称等价。

**为什么是 1000 步、batch 8、2 卡。** 1000 步是用户定死的口径。batch 8 与 2 卡沿用 2026-09-04 `aws-t2-ref-s100` / `aws-t2-cand-s100` 的档位：小到两组能并行，大到覆盖 FSDP 分片（`--fsdp-devices 2`）。1009 × 8 = 8072 个样本不到 400ep 库一个 epoch（101,066），避开 epoch 边界的抽样分叉。确定性前提逐项核过：index 抽样在主进程（`_LoggingSampler` 包 `bs.sampler`），`worker_init_fn` 不播种、Dataset 与 transforms 无随机源，模型内 rng 由 `fold_in(seed, step)` 派生，dropout 为 0；环境 B 上有 100 步三对物理卡逐位相同的实测（`aws-t2-cand-s100`），1000 步尚无先例，第一波 A1 ≠ A2 即停。

**为什么要确定性 flag。** `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'` 关掉 GPU 上的非确定性归约与自动调优。生产训练默认档做不到这一点，所以对拍必须开。

**卡怎么排、要多久。** 同一个配置的三条在同一对卡上顺序跑。**理由不是数值**（三对卡实测逐位相同），**是环境指纹把 `CUDA_VISIBLE_DEVICES` 当硬一致项**，跨卡对 gate 会 FAIL（2026-09-04 实际发生过）。按用户裁决第 3 项维持两波：第一波 C32 在 4,5、M32 在 6,7 同时跑；两组都跑完后第二波 C8 在 4,5、M8 在 6,7。单条耗时按分解式估：`非摘要步 s/step × 1000 + 11 × 单次摘要耗时 + 启动编译`，本环境实测 `aws-t3-closed-s100` 1.51 s/step、`aws-t3-open-s100` 1.68 s/step、单次全树摘要约 169 s、启动约 90 s，得每条约 58–61 min，每对卡 6 条约 6 h，两波总墙钟约 6–7 h（refnpy 侧 dataloader 更慢，第二波以 `-s100` 实测的非摘要步时长重算）。**不引用环境 A 的数字**：那边同配置实测是 50–53 min 而非计划旧稿写的 2.5 h，且 Ada 单次摘要 89 s、A100 169 s，跨环境不可比。正式跑之前每个配置先跑一次 100 步短版本（run_name 加 `-s100`，摘要步列表换成 `BENCH_EXTRA_DIGEST_STEPS=1,2,24,49`）排错、估时；按用户裁决第 8 项，`-s100` 跑完清理、不留档、不作证据。

**环境指纹。** 起跑前用 `scripts/training/g0/check_baseline_env.py` 的指纹做 preflight，硬一致项：`uv.lock` sha、`packages`（现只采 torch / jax / jaxlib / numpy / ml_dtypes 五个，本轮加 flax 与 optax）、GPU 型号与驱动、`XLA_FLAGS`、x64 与 matmul precision、norm_stats sha（显式 `--norm-stats` 指 400ep 文件）、tokenizer、pi05 底座权重抽样、源库抽样（改对 `source_dataset_root` 抽样，现有代码对 packed 库根做 `scandir(dataset/"data")` 会崩）、**本库 manifest sha 与 motion store meta sha**（B 侧从 `framesamp-8x8/meta/store_meta.json` 读；A 侧 refnpy 没有 store meta，改记 `BENCH_REF_SOURCE` 的 `stats.json` sha 与 manifest sha，两侧的 manifest sha 必须相等）。指纹在 `dump` 后必须接 `manifest` 与 `check` 两步产出 `BASELINE_ENV=PASS`，三步都写进阶段 5 命令。允许差异逐字段白名单：`--exp-name`、`--checkpoint-base-dir`、C8/M8 的 `--dataset-path`、参考/候选 commit、`BENCH_DATASET_IMPL` 与 `BENCH_REF_*`、`BENCH_RECORD_DIR`、`MMEVLA_JAX_CACHE_DIR`、`dataset.store_meta_sha256`（A 侧为 None）、`PYTHONPATH`。`CUDA_VISIBLE_DEVICES` 不在白名单，同 profile 三条同卡对。

**补充单步全梯度对拍。** 全短、全满、混合三种真实 batch 各一个，用 `single_step_grad.py` 记完整梯度树逐叶 sha，定位可能被梯度范数掩盖的差异；同配置两侧逐位相等，比较器是 `compare_grad_summaries.py`，判定行 `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0`。`single_step_grad.py` 的 `_create_framesamp_dataset` 是模块级 from-import，monkeypatch 打不到，所以它自己加 `DTYPE_DUMP_IMPL={packed,refnpy}` 开关与两个新 yaml 的白名单；C32/M32 的 A 侧在 REF worktree 跑（带 `PYTHONPATH`），C8/M8 的 A 侧用 refnpy。

#### 交付判定与边界

- 只有第一块与第二块**全部**通过，才宣称对应 profile 的训练交付等价；第二块未通过只能报告已通过的输入检查。
- 交付报告分开写五件事：旧能力回归（C32/M32）、新 8 帧数据搬运等价（C8/M8 的 `GATE_8X8`）、**8 帧语义正确性**（`HAND_CALC_8FRAME` + 第三块在线一致）、motion 接口保护、实际资源表现（吞吐与显存在数值验收之外独立报告，用 `nvidia-smi -lms 500` 的稳态窗口均值 / 0% 占比 / 分层均值，排除 warmup，记录存储介质「AWS 本地 NVMe RAID（`/dev/md0`）」、batch、worker，不以中位数下结论，不承诺性能收益）。1000步×8=8000个实际更新样本，采样记录包含预取共8072项，对123,044行的覆盖远低于50%，`TRAIN_1000_EXACT` 只抓粗错，细错靠第一块 `VERIFY_PACK` 全量档兜底，留档写明。
- 在线推理放在第三块：训练验收通过不等于 8×8 能部署。
- motion 只做回归保护：不重抽 Wan、不重训 MotionJEPA、不做预算或结构消融。注意 M8 下帧路只剩 8 个时刻而 motion 窗数不变，`mem_order` 交错结构与 motion/frame 尺度必然与 M32 不同，属设计内，本轮不评价。

### 第三块：推理——让在线记忆构造认得 8×8，并证明推理侧与训练侧交付一致

#### 结论先行

- **模型侧、motion sidecar、IPC 协议不用改。** 评估侧没有 yaml 文件名白名单，`policies/policy_config.py::create_trained_policy` 只按 run 根的 `history_config.resolved.yaml` + `.sha256` + `motion_provenance.json` 建策略；`train.py::init_history_config` 会把这三样写进 run 根，8×8 的 run 自动被认。
- **要改的是 `policies/framesamp_memory.py::FrameSampMemory` 与 `policies/policy.py::_prepare_mem_buffer`**（构造函数现在没有 `token_per_image` 参数，必须由 policy 传入），外加全部 `FrameSampMemory(...)` 构造点与引用 `pos_emb_4x4` / `image_emb_4x4` 的工具脚本（清单见第二部分第 1 节），以及 motion-variance 三个驱动脚本（用户裁决第 2 项）。
- **没有训练收敛的 8×8 模型。** checkpoint 只能用第二块候选 B run（`t8-c8-b`、`t8-m8-b`）1000 步后的 EMA 参数，落在 `$V1/train-runs/t8-<p>-b/mme_vla_suite/t8-<p>-b/999`（bench 起跑带 `BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=999`，两者缺一不落盘）。这样的模型不会完成任务，闭环成功率**不作任何指标**；第三块交付的是「在线链路跑通」与「推理侧 obs 与训练侧 obs 逐字节一致（灌真值行口径）」两件事。这与 2026-09-04 `aws-t3-open-s100` 用 100 步 ckpt 跑 40 集、0% 成功、只证链路的做法同口径。
- **在线 4×4 不能被改坏。** `FrameSampMemory` 改成网格驱动后，先用现成的真模型（`awsprod40k-b128-motion/39999`）证明 4×4 在线行为逐位不变，再验 8×8。
- **legacy-eval 那套（`scripts/training/legacy-eval/*`）本轮一律禁用**（用户裁决第 2 项），不作备选。

#### 现在的推理测试怎么做

闭环评估是三个进程共一张卡、多张卡并行：

- **policy server**：视觉记忆由 `FrameSampMemory` 在线造：每步当前帧过 SigLIP（`HistoryPi0.vision_encode`，bf16 权重，输出 256 个 patch），`pool_tokens_to_size(output_emb, 16)` 池成 4×4（`shared/data_utils.py` 那份，与建库侧 `dataset_builder/data_utils.py` 那份逐字相同但不是同一对象），存进缓冲键 `image_emb_4x4`；位置表 `pos_embedder(ranges, 4)` 只算 4×4 一档，`(4096,16,768)` f32 = 192 MiB。推理时 `get_frame_sampling_indices` 由 `budget // (token_per_image * num_views)` 算出 32 再调 `even_sampling_indices`（**已是网格驱动，不用改**），`_prepare_frame_sampling` 的 `spatial_key` 也已由 `sqrt(token_per_image)` 推；motion 开启时 `_prepare_motion` 取 `pos_emb_4x4[f, 0, :pos_dim]` 当时间码，`mem_order` 与训练侧同一个 `memory_order`。
- **motion sidecar**：`policies/motion_client.py::MotionEncoderClient` 在策略构造时 `subprocess.Popen` 起常驻子进程 `scripts/dataset/wan/motion_sidecar.py`（跑在 `v1-store/venvs/wan`），Unix socketpair 通信，协议 `MMEMOT01`：父发 33 帧 256×256×3 uint8（6,488,064 B）加起点帧号，子回 768 维 f32（3,072 B），单窗约 1.4 s。握手时逐键比对 provenance 与离线 motion store，不等即 raise。
- **仿真客户端**：`examples/robomme/eval.py` 在 `/scratch/hongze/micromamba/envs/robomme/bin/python` 下跑，驱动 shell 给它带 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`。
- **驱动脚本（本轮唯一允许）**：`scripts/motion-variance/run_batch_mv.sh` → `eval_shard_mv.sh` → `serve_policy_mv.py`，自带 robomme 副本，每片一卡一 tmux 会话，`eval_shard_mv.sh` 把片的卡号经 `--motion-gpu` 传给 `serve_policy_mv.py`。最近一次正式评估 `mv-matrix-40k` 用的就是它。**它现在有三处硬假设本轮必须改**（见「改什么」）：checkpoint 由 `COND` 硬编码为 40k 模型且不读任何 env；`serve_policy_mv.py` 一见 motion 关闭态 checkpoint 就 `SystemExit`，`mv_prepare_history` 无条件读 motion 三键；`EP_COUNT` 是每 worker 窗口，4 worker 下每任务 12 集。

上一次 40k motion 模型的正式评估（`docs/training-doc/eval-awsprod40k-b128-motion/`）：200 集 38 min，成功率 28%，policy 单次 `infer` 69 ms，每卡显存约 37 GB（policy 32.9 + sidecar 3.3 + 仿真 0.8）。

#### 改什么

`FrameSampMemory` 改成从构造参数 `token_per_image` 推网格边长 `g = int(sqrt(token_per_image))`（要求 g² == token_per_image 且 g ∈ {2,4,8}，`PosEmb3D.__call__` 只接受这三档；否则构造时报错）：

| 位置 | 现在 | 改成 |
|---|---|---|
| `__init__` 签名 | 无 `token_per_image` | 新增必填 `token_per_image: int`；`policy.py::_prepare_mem_buffer` 从 `self.history_config.token_per_image` 传入；其余 5 个构造点同步（第二部分第 1 节） |
| `__init__` pos 表 | `self.pos_emb_4x4 = np.array(pos_embedder(ranges, 4))`，192 MiB | `self.pos_emb = np.array(pos_embedder(ranges, g))`；g=8 时 `(4096,64,768)` f32 = 768 MiB host 内存/进程，第三步 4 片共 3 GiB；构造瞬时设备侧约 1.5 GiB（repeat 中间量），`POLICY_MEM_FRACTION` 0.55 下无碍 |
| `add_buffer` | `pool_tokens_to_size(output_emb, 16)` | `pool_tokens_to_size(output_emb, token_per_image)`；g=8 时 `pool_size = sqrt(256 // 64) = 2`，2×2 平均池化，与建库侧同一算式 |
| `add_buffer` 写缓冲 | 键名写死 `"image_emb_4x4"` / `"pos_emb_4x4"` | `f"image_emb_{g}x{g}"` / `f"pos_emb_{g}x{g}"`，与 `_prepare_frame_sampling` 已有的 `spatial_key` 推导对上 |
| `_prepare_motion` | `self.pos_emb_4x4[f, 0, :pos_dim]` | `self.pos_emb[f, 0, :pos_dim]`；取值不变（第一块 `MOTION_POS_XGRID` 验的就是这一点） |
| docstring 与行内注释 | 「只算 4x4 一档 … 192 MiB」；`# (t, v, 64, 2048)`；`# (t, v, 16, 2048)` 等三处；docstring 里 `pos_emb_4x4[f, 0, :pos_dim]` | 按网格改写；`# (t, v, 64, 2048)` 本身就是错的（SigLIP `So400m/14` 输出 256 patch），顺手纠正 |

`policy.py::_prepare_history` 除传参外不用改：`max_frames = budget // (token_per_image * num_views)` 自动变 8，`memory_order(frame_times, token_per_image * num_views, motion_times)` 仍是 608 位置换。

引用旧属性名或键名的工具全部改成按 policy 的 `history_config.token_per_image` 推：`scripts/training/g0/serve_policy_probe.py`、`compare_online_memory.py`、`compare_train_infer_obs.py`、`compare_siglip_replay.py`、`scripts/training/tests/motion_gates_online.py`（还写死 `(512, 16, 1)` 与 `MAX_FRAMES=32`，改成按 `--yaml` 推）、**`scripts/training/tests/motion_gates_model.py`**（`load_lib_oracle` 硬读 `framesamp/pos_emb_4x4.f32.bin` 并 `reshape(-1,16,768)`，8×8 下会静默读 4×4 表而 assert 照样过，属假通过，必须改成按 store spec 读）、**`scripts/training/tests/eval_rhythm_gates.py`**（构造 `FrameSampMemory`）、**`scripts/training/g0/check_config_provenance.py`**（关 0 工具，写死 `framesamp/` 子目录）。`scripts/motion-variance/mv_common.py` 的 `608/512/96/1184` 在 8×8 下数值不变，只改注释口径；`shared/sampling.py::memory_order` docstring 的「同帧 16 位」同改。

motion-variance 三个驱动脚本（用户裁决第 2 项）：
- `run_batch_mv.sh` 与 `eval_shard_mv.sh`：新增 env `CKPT_OVERRIDE=<checkpoint 目录>`，设了就跳过 `COND` 的硬编码分支，`CKPT_ID` 取其 basename；`MV_MOTION_OFF=1` 时不检查 wan venv 与 encoder 权重、不给 `--motion-gpu`、改传 `--motion-off`；`RUN_SUFFIX` 照旧透传（本轮 `-t8c8` / `-t8m8`）。
- `serve_policy_mv.py`：新增 `--motion-off`：跳过「必须 motion 开启态」检查、`motion_factory` 返回 `ZeroMotion`、`mv_prepare_history` 在关闭态不读 `visible_motion_frames` 与 motion 三键（记 `motion=off`）。

#### 第三块验收

分三步，前一步不过不进下一步。

**第一步：4×4 在线回归（用真模型，不需要 8×8 checkpoint，阶段 2 代码改完即可跑）。** 目标是证明网格驱动的 `FrameSampMemory` 在 `token_per_image=16` 下与改前逐位相同。

- 三方对拍 `online_mem_regress.py`：**不用 worktree**（单进程里模块名唯一，跨进程又会被 `.pth` 拉回主树），沿用 `compare_online_memory.py` 的既有范式——`git show $REF:src/mme_vla_suite/policies/framesamp_memory.py` 抽到 scratch 目录、以 `framesamp_memory_ref` 之名 import，与主树的 `FrameSampMemory` 在同一进程共享同一个 jitted SigLIP，喂同一批真实帧，比 `prepare_frame_sampling` 交付的三键（`image_emb_4x4` / `pos_emb_4x4` / `state_emb` 装配后）与 `_prepare_motion` 输出的 `leaf_sha256`；起跑前断言两个模块的 `__file__` 不同、`find_spec("mme_vla_suite").origin` 在主树。判定行 `ONLINE_MEM_REGRESS=PASS frames=<n> keys=4 mismatches=0`。
- 同时用 `compare_online_memory.py` 本身（`--h5` 默认是环境 A 路径，改传本机 H5 或改用源 npy 帧）跑一遍 4×4，`ENC_LAYER=PASS` 与 `ASSEMBLY=PASS`。
- 在 `awsprod40k-b128-motion/39999` 上重跑 `scripts/training/g0/compare_train_infer_obs.py` 的 S 臂（帧特征灌训练库真值行），13 条阻断判据与 `docs/training-doc/tic-obs-model-40k/result.md` 逐条对照：12 条 PASS 同值；`VT_FULL_VS_CACHED` 要求「同为 FAIL 且 rel_fro 同量级（1e-3 级）」，因为它四次跑 1.86e-3～3.04e-3 本就不稳，「逐条同值」做不到；`VT_FULL_VS_CACHED_F32` 要求前缀 KV 逐位相同、rel_fro ≤ 1e-6（4×4 实测 1.35e-7）。命令用 `CUDA_VISIBLE_DEVICES=7`，不照抄留档里的 `0`。
- 用 `scripts/training/g0/compare_online_motion.py --gpu 7 --yaml perceptual-framesamp-context-motion.yaml --lib v1-store/datasets/4task-motion-40ep`（**没有 `--ckpt` 参数**）重跑 40ep 库 772 窗：`ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0`，`ONLINE_START_SET / ONLINE_POS / ONLINE_ORDER / PROVENANCE` 全 PASS。

**第二步：8×8 推理侧 obs 与训练侧 obs 对拍（用 `t8-c8-b` / `t8-m8-b` 的 999 目录）。** 七关分属三个工具，不是一个脚本：

| 关 | 工具 | 比什么 | 8×8 下的要求 |
|---|---|---|---|
| 0 配置与库同源 | `check_config_provenance.py`（加 `--store-subdir framesamp-8x8`） | norm stats、参数树叶、四张表指纹 vs run 记录 | 全同；`n_leaves_params` 与第二块 B run 的 params 子树叶数一致（55），不与全树 177 混比 |
| 1 输入键 | `compare_train_infer_obs.py`（加 `--store-subdir framesamp-8x8`、支持 motion 关闭态） | 在线 `_prepare_history` 八键（C8 四键）vs `FrameSampDataset`；sidecar 现算 motion vs 表 | S 臂逐字节相同。**S 臂把在线池化结果覆盖成库真值行，所以这一关检验的是「给定同一份帧特征，装配层交付一致」，不检验在线 2×2 池化本身**；在线池化的逐位证据由下面的 `ENC_LAYER_8X8` 给 |
| 1' 在线池化 | `compare_online_memory.py` 扩到 `image_emb_8x8` | 同一 jitted SigLIP 给建库域冻结副本 `MemoryBuffer` 与 `FrameSampMemory(token_per_image=64)`，比 64 token 池化输出 | `ENC_LAYER_8X8=PASS steps=<n> mismatch=0`（阻断）；B 臂 `MEM_S_VS_B` 的 rel_fro 照记为观察项，4×4 实测 0.33%，8×8 无判据 |
| 2 预处理后 | 同 1 | 两侧过完变换链的全部键 | 逐字节相同 |
| 3 模型内部 | 同 1 | 前缀 token（M8 1184 = 608 + 512 + 64；**C8 1088 = 512 + 512 + 64**，`want_mem` 按 `motion.enabled` 算）、attn mask、positions、18 层 K/V、`mem_order` 可逆（C8 无 motion 置换项） | 逐位相同 |
| 4 整段 vs 缓存 | 同 1 | `compute_loss` 整段前向 vs 前缀缓存单步 | 结构三项全等；**bf16 `VT_FULL_VS_CACHED` 降为观察项**（记 rel_fro / ulp_p99，只要求与 4×4 同量级）；**f32 `VT_FULL_VS_CACHED_F32` 升为阻断**：`--f32-diag-points-per-episode 3`，要求前缀 KV 逐位相同、rel_fro ≤ 1e-6（用户裁决第 4 项） |
| 5 最终动作 | 同 1 | 两侧 obs 经同一 `_sample_actions` 去噪 | 逐位相同 |
| 6 真仿真 | `serve_policy_probe.py` + `summarize_eval_probe.py --max-frames 8 --tokens-per-frame 64` | 在线不变量：窗数=公式值、`mem_order` 合法排列、prompt token 属训练集、pos 表 sha（在线 `(4096,64,768)` 表前 586 行 vs 库 `pos_emb_8x8.f32.bin`，工具已如此实现）、一一对应、无抛错 | 全过 |

**第三步：闭环冒烟（证明链路跑通，不看成功率）。** 每个 profile（C8、M8）用改过的 motion-variance 链路跑 4 片 4 卡（GPU 4–7），`EP_COUNT=10` 在 4 worker 下每任务 12 集、共 **48 集**（用户裁决第 2 项接受），M8 带 sidecar 与 policy 同卡，C8 走 `MV_MOTION_OFF=1`。判定只有四条：`errors=0`、每集推理次数 ≤ 82（这是 timeout 上界、近乎恒真，只作 sanity）、`mem_order` 逐步合法、M8 的 sidecar 窗数与公式值一致；成功率照实记但明确标「1000 步测试模型，无意义」。显存与 host 内存（pos 表 768 MiB/进程）在结果里报。`RUN_SUFFIX` 分别 `-t8c8` / `-t8m8`，避免会话名与结果目录互撞。

#### 顺序、耗时与边界

- 第一步在阶段 2 代码改完后立刻做（不依赖第二块），是第三块的准入。第二、三步排在第二块之后，因为要等 `t8-c8-b` / `t8-m8-b` 跑完。
- 第二步的主进程 jax 必须在 GPU 上（`motion-p5-online/launch.md`：CPU 算的 `PosEmb3D` 表与 GPU 生成表 22% 元素不等；`mv_common.py` 有同样的硬闸），用 GPU 7 一张卡，motion 侧 `--motion sidecar --motion-gpu 7`；第三步 48 集按 4 卡并行约 10–20 min 量级（8 卡 200 集 38 min 外推）。
- 边界：第三块证明的是「8×8 在线链路可跑、推理侧与训练侧在灌真值行口径下交付一致、在线 2×2 池化与建库池化逐位一致」，**不证明** 8×8 模型能完成任务；那需要一次正式训练，另立 run_name 另行授权。

### 执行阶段与 commit 节奏

按时间顺序 8 个阶段。规则三条：每个阶段结束一个功能 commit（`commitV9.x`）或文档 commit（`docs:`），commit 后立即 `git push`；所有正式 run 从 clean HEAD 起；**阶段 5 的 12 条轨迹期间主树冻结不 commit**，留档在 12 条全部跑完后一次提交。`CAND` 定义为阶段 2 末的代码 commit；阶段 3、4 的 `docs:` 留档 commit 会让阶段 5 起跑时的主树 HEAD 与 `CAND` 不同，留档同时记「候选代码 commit = CAND」与「起跑 HEAD」两项，gate 比 `CAND` 与起跑 HEAD 之间 `git diff --stat` 只含 `docs/` 路径。

| 阶段 | 做什么 | GPU | 产出 | commit |
|---|---|---|---|---|
| 0 preflight | 只读核对源库、旧库 meta、motion store、norm_stats 两份、权重、tokenizer、磁盘、环境变量干净、`--help` 核 tyro 嵌套 flag | 无 | 核对记录写进阶段 1 的 launch.md | 无 |
| 1 量具与参考 | 改 `train.py` / `single_step_grad.py` 缓存落点、`bench_train_steps.py` 开关与白名单与 refnpy 注入与 origin 记录、`check_baseline_env.py` 指纹、`_common.py` / `dump_fixture_samples.py` 配额与分档与路径、`single_step_grad.py` 开关、新建 `gate_8x8.py` / `compare_fixture_dumps.py` / `ref_npy_dataset.py` / `xgrid_pos_check.py` / `online_mem_regress.py` / `hand_calc_8frame.py`、两份 8×8 yaml；量具自检（含负例：候选源码误入 REF、缺 None 键、漏 state_step、错 norm_stats、NaN 标量、旧 checkpoint）；C32 **40ep** 库 100 步冒烟与 `aws-t2-ref-s100` 的 scalars sha `85b8fe37…` 比对（同其 launch.md 全部参数：40ep 库、默认 norm_stats、`--save-interval 1`、`BENCH_DIGEST_INTERVAL=25 BENCH_EXTRA_DIGEST_STEPS=99`，卡换 4,5，指纹 preflight 允许 `gpu` 差异并在留档写明基线 run_name 与 commit `c5925d9`） | 冒烟用 GPU 4,5 | 参考侧全部工具 | `commitV9.0`，push 后 `git rev-parse HEAD` 记为 **`REF`**，随后 `git worktree add --detach v1-store/worktrees/ref-8x8 $REF` |
| 2 两档格式与 Dataset | `framesamp_store.py` 规格表（含 parts bytes 账、`run_full_checks`）、`pack_framesamp_store.py --layout`（27 处常量全改）、`framesamp_dataset.py` 白名单与 spec 匹配与注释、`test_pack_guards.py` 两档用例；`FrameSampMemory` 网格驱动 + `policy.py` + 6 个构造点 + 全部工具 + `check_config_provenance.py` + motion-variance 三脚本；第三块第一步 | pytest 走 `JAX_PLATFORMS=cpu`；第三块第一步用 GPU 7 | 候选侧代码 | `commitV9.1`（库格式 + Dataset）、`commitV9.2`（在线记忆 + 工具 + mv 驱动）；`9.2` 的 HEAD 记为 **`CAND`** |
| 3 建库 | 40ep 先：pack → verify → report → `xgrid_pos_check`；再 400ep 同四步 | 无（纯 CPU，`--procs 48`） | `framesamp-8x8/` 两份，`VERIFY_PACK` / `IMAGE_NPY_SPOT` / `MOTION_POS_XGRID` 各两条 | `docs:` 建库留档 |
| 4 第一块验收 | 四个 profile 的 fixture dump（refnpy 与 packed）与比较、`spawn_matrix`（packed 与 refnpy）、错配拒绝用例、`hand_calc_8frame` | dump 走 CPU；超 5 分钟项进 tmux（前缀 `p8-fx-`） | `SAMPLE_RAW_EXACT` / `BATCH_RAW_EXACT` / `MATRIX` / `REJECT_INPUTS` / `HAND_CALC_8FRAME` 判定行 | `docs:` 留档 `docs/training-doc/t8-fixture/` |
| 5 第二块 | 先 4 条 100 步排错 run（每对卡顺序跑本波两个 profile 的 `-s100`，估时后清理、不留档）；再两波 12 条正式轨迹；每 profile 跑完立即 `gate_8x8` | 第一波 C32 在 4,5、M32 在 6,7；第二波 C8 在 4,5、M8 在 6,7 | 12 份 records、4 条 `GATE_8X8=PASS` | 期间**不 commit**；全部跑完后一次 `docs:` 留档 `docs/training-doc/t8-<profile>-{a1,a2,b}/` |
| 6 第三块 | 第一步已在阶段 2 末做完；第二步三个工具在 `t8-{c8,m8}-b/mme_vla_suite/t8-{c8,m8}-b/999` 上跑七关；第三步 48 集闭环冒烟 | 第二步 GPU 7；第三步 4 片 GPU 4–7 | 七关判定行两套、`EVAL_SMOKE=DONE` 两条 | `docs:` 留档 `docs/training-doc/t8-infer-regress-4x4/`、`t8-infer-{c8,m8}/` |
| 7 收官 | `docs/dataloader-restructure.md` 加「8 帧 × 8×8 档」节、`docs/train-infer-consistency.md` 加「8×8 档（测试模型）」节、按实测更新前后链路图；本文降为过程档案；删除 `ref-8x8` worktree | 无 | 正本更新 | `docs:` 收官 |

阶段 1 与 2 之间、2 与 3 之间可以穿插文档 commit（不影响 `REF` / `CAND` 的定义，两者各按阶段末代码 commit 记）；阶段 5 内不行。任何阶段的判定行不过即停在该阶段，把原始输出交用户，不改判据、不进下一阶段。**唯一例外**是第三块关 4 的 bf16 观察项，它不阻断（用户裁决第 4 项）。

---

## 第二部分（技术细节，供 agent 追踪）

### 1. 拟修改文件与函数

| 文件 / 锚点 | 改法 | 不改什么 |
|---|---|---|
| `src/mme_vla_suite/datastore/framesamp_store.py` 顶部常量段 | 新增 `@dataclass(frozen=True) class StoreSpec`（字段：`layout, grid, tokens_per_frame, image_key, pos_key, image_row_shape, pos_row_shape, image_row_bytes, pos_row_bytes, image_part_dir, pos_table_relpath, source_image_offset, source_pos_offset`）与 `SPECS: dict[str, StoreSpec]` 两档；旧常量保留为 4×4 档的别名以免破坏现有 import | `STATE_*`、`ROW_DIGEST_*`、`SOURCE_NPY_SIZE=602951`、`SOURCE_STATE_OFFSET=602906`、`TARGET_PARTS=32`、`row_of`、`build_exec_lookup`、锁与 meta 路径常量 |
| 同文件 `StoreMeta` | 新增字段 `spec: StoreSpec`；`load` 先取 `raw["layout"]`，不在 `SPECS` 即 `ValueError("layout 不符")`，再按 `spec` 校验 `tables` 三键的 `row_shape/dtype/row_bytes` **与 parts 的 `bytes == num_rows × spec.image_row_bytes`** | 其余校验项（parts 连续、status、manifest_scope）原样 |
| 同文件 `run_fast_checks` / **`run_full_checks`** / `FrameSampStore.__init__ / read_image_rows / pos_rows` | 小表大小账、`run_full_checks` 的 `(POS_KEY, POS_TABLE_RELPATH)` 元组、pos 表 reshape、image 行 reshape、`IMAGE_ROW_BYTES` 乘法全部改读 `meta.spec` | preadv 读法、`_runs_of`、fadvise、`__reduce__` 禁 pickle |
| `scripts/dataset/pack_framesamp_store.py` | `pack` 加 `--layout {framesamp-4x4-v1,framesamp-8x8-v1}`（默认 4×4）；全脚本 27 处网格常量引用（`read_frame_decode` 守卫、`read_frame_slice` / `_pin_slice_prefix` 偏移、`build_small_tables` / `_state_chunk_worker`、`_pack_part_worker` 的 `IMAGE_PART_DIR` / `IMAGE_ROW_BYTES` / `POS_ROW_BYTES` / `STATE_ROW_BYTES`、`cmd_pack` 的空间预检 / `IMAGE_PART_DIR` / `POS_TABLE_RELPATH` / meta 写入段）全部改读 spec；8×8 slice 偏移 387 / 344,682 / 602,906（对抗验证 5 文件 `raw.find` 恒定）写入规格表但首跑仍 decode；`cmd_pack` 的 `floor = max(40*10**9, int(1.25*need))`；`verify` 从 meta 取 spec | `plan_parts`、锁协议、`atomic_write_bytes`、写侧三重校验的**逻辑**、resume 逻辑、`_spot_entries` |
| `src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__init__` | `_req((budget, tpi, nv) in {(512,16,1),(512,64,1)})`；`_req(self._meta.spec.tokens_per_frame == tpi*nv)`；三处 32 帧口径注释改按变量 | `__getitem__`、`_pad`、`_normalize_state`、motion 块、`_NONE_KEYS`、pkl 互校 |
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context-8frame-8x8.yaml` 与 `-motion.yaml` | 新建，见第一部分（含 `streaming_obs_horizon: 16`） | 两份旧 yaml |
| `scripts/training/train.py::main`、`scripts/training/tests/single_step_grad.py::main` | `jax_compilation_cache_dir` 取 `os.environ.get("MMEVLA_JAX_CACHE_DIR") or 旧值` | 其余一行不动（`bench_train_steps.py` 的 `inspect.getsource` 护栏要求 `train.main` 仍含 `wandb.log(reduced_info`、`_checkpoints.save_state(`、`init_train_state(`、`create_data_loader(` 四个片段，`TorchDataLoader.__iter__` 仍含 `make_array_from_process_local_data` / `num_items` / `StopIteration`） |
| `scripts/training/tests/ref_npy_dataset.py`（新建） | `RefNpyFrameSampDataset(source_root, manifest_path, data_config, history_config, action_horizon, motion_root)`：装配逻辑见第一部分第一块「第二层」；接口与 `FrameSampDataset` 同；实现 `__getstate__` spawn 契约；只 import `shared/` 下的 `even_sampling_indices` / `right_padding_token_emb` / `memory_order` / `pad_times` | 不进 `src/`，不被生产代码 import |
| `scripts/training/tests/hand_calc_8frame.py`（新建） | 脚本内手写选帧 / `mem_order` / 可见窗公式（不 import `shared/sampling.py` 与 `motion_store.py`），对边界样本集与候选 Dataset 输出逐项比；判定行 `HAND_CALC_8FRAME=PASS samples=<n> mismatches=0` | |
| `scripts/training/g0/bench_train_steps.py` | `_EXPECTED_HISTORY_CONFIGS` 加两个新文件名；新增 env `BENCH_DATASET_IMPL={packed,refnpy}`（默认 packed）、`BENCH_REF_SOURCE`、`BENCH_REF_MANIFEST`、`BENCH_REF_MOTION`，refnpy 时 monkeypatch `dataloader._create_framesamp_dataset` 返回参考 Dataset、包 `train.init_history_config` 传 `framesamp_root=None`、epoch 样本数改从 `BENCH_REF_MANIFEST` 读；摘要记录写 `phase / loop_step / state_step` 三字段；`batch_digests` 对 None 键显式记 `null`；`run_meta.json` 新增 `import_origins`（四个 `__file__`）与 `norm_stats_file_sha256`、`norm_stats_actual`、`norm_stats_expected`；有限性与活性在记录侧不判、由 gate 判 | 哈希口径 `_leaf_sha256` / `_canonical_sha256`、`_WandbProxy` 行 schema、`index_sequence` 记法、`_make_digest_gate` 语义、`BENCH_SAVE_FINAL_CKPT` + `BENCH_FINAL_STEP` 语义、`_MAX_BENCH_STEPS=1200` |
| `scripts/training/tests/gate_8x8.py`（新建） | 参考 `g0_gate.py::_gate_t2` 八步重写为配置驱动：`--profile {c32,m32,c8,m8}`、`--run-a1 --run-a2 --run-b --log-a1 --log-a2 --log-b --env-a1 --env-a2 --env-b`（全部必填）、`--steps 1000 --batch-size 8`；argv 白名单与 env 白名单见第一部分；键名集合按 profile 钉死；`n_leaves` 三侧相等且按 profile 软核 177/193；摘要步按 `state_step` 对齐；表头六列与 `n ≥ 8072` 内建；有限性、活性、`IMPORT_ORIGIN`、`NORM_STATS_ACTUAL`、`git diff --stat CAND..起跑HEAD` 只含 `docs/`；`--self-test` 构造负例 fixture 逐个确认 FAIL；成功行 `GATE_8X8=PASS profile=<p> steps=1000 batch=8 state_steps=[…] digest_steps=[…]` | `g0_gate.py` 的 T1/T2 常量（`_EXPECT_RAW_MISMATCH=4` 等）一字不动 |
| `scripts/training/g0/check_baseline_env.py::collect_fingerprint` | `assets.norm_stats_sha256` 改为 CLI `--norm-stats` 显式路径；`packages` 加 `flax`、`optax`；新增 `dataset.store_meta_sha256`（`<dataset>/meta/store_meta.json`，不存在记 None）、`dataset.manifest_sha256`（store meta 或 `--manifest` 显式）、`motion.store_meta_sha256`；`dataset_spot` 改对 `source_dataset_root` 或 `--source` 抽样；tokenizer / pi05 路径改从 `--v1-store` 显式取，不再按脚本所在代码根推 | `uv_lock_sha256`、`gpu`、`xla`、`jax_config`、`_diff` 深比较、`dump / manifest / check` 三步与 `BASELINE_ENV=PASS/FAIL` 判定行 |
| `scripts/training/tests/_common.py` | `SHORT_STEPS / FULL_STEPS` 改为 `fixture_steps(max_frames)` 返回 `((0,1,2,mf-3,mf-2), (mf-1,mf,mf+1), (33,34,35))`（第三组为 motion 档，两档共用）；`PER_STEP` 改为按库内 `exec_start_idx==0` 的 episode 数取 `min(200, 该数)` 并在判定行打出；`build_fixture_batches` 的短/满分档改按 `max_frames`（短 = step ≤ mf-2，满 = step ≥ mf-1）；`test_padding_dtype.py` 随之改引用 | `MEMORY_KEYS`、`leaf_sha256`、`canonical_sha256`、位型容器格式 |
| `scripts/training/tests/dump_fixture_samples.py` | `manifest_path` 从硬编码根 `v1-store/episode_manifest.json` 改为 env `DTYPE_MANIFEST` 或 store meta 的 `manifest_path`；白名单加两个新 yaml；新增 env `DTYPE_DUMP_IMPL={packed,refnpy}`；`is_short` 改按 `max_frames` | 两层取证结构、raw+canonical 双口径记录 |
| `scripts/training/tests/compare_fixture_dumps.py`（新建） | 严格零差异比较器：先比键集（含 None）、再 dtype/shape/raw sha；判定行 `SOURCE_IDENTITY=PASS …`、`FRAME_INDEX_EXACT=PASS …`、`SAMPLE_RAW_EXACT=PASS samples=<n> per_step=<k> mismatches=0`、`BATCH_RAW_EXACT=PASS batches=200 mismatches=0` | `compare_dtype_fix.py` 不改 |
| `scripts/training/tests/single_step_grad.py` | 白名单加两个新 yaml；新增 env `DTYPE_DUMP_IMPL={packed,refnpy}`（自己构造参考 Dataset，不依赖 monkeypatch）；norm_stats 走同一 tyro 覆盖 | `train.train_step` 护栏、`_GRAD_BATCH_KINDS`、`GRAD_DONE` 行 |
| `scripts/training/tests/compare_grad_summaries.py` | 不改；判定行 `GRAD_EQ` 的真实出处 | |
| `scripts/training/tests/test_pack_guards.py` | `REF_SHARD` / `MANIFEST` 改为 env `MMEVLA_TEST_SOURCE` / `MMEVLA_TEST_MANIFEST`（指向 40ep 库），`test_g6a_exec_lookup_formula` 的样本数断言从 manifest 推导而非写死 395289；session fixture 对两档 layout 各打一份前缀 [0..2] 迷你库；新增用例：8×8 meta 配 4×4 目录名、`token_per_image=64` 配 4×4 库、`=16` 配 8×8 库、未知 layout、parts bytes 账错位均 raise；末尾打印 `REJECT_INPUTS=PASS cases=<n>` | 既有 G1–G14 门（20 个测试函数）逻辑 |
| `scripts/training/tests/spawn_matrix.py` | 加 `--yaml`（默认旧 yaml）与 `--impl {packed,refnpy}` | `MATRIX=PASS workers=… epochs=… batches_per_run=…` 判定行 |
| `scripts/training/tests/project_scalars.py`、`scripts/training/tests/dump_index_seq.py` | 不改；前者是 `scalars_hex.tsv` 的生产者，阶段 5 显式调用；后者默认 `4task-gl` 路径在环境 B 不存在，调用时显式传参 | |
| `scripts/dataset/xgrid_pos_check.py`（新建） | 读两个库的 pos 表（`np.fromfile` 后按各自 spec reshape），对 t ∈ [0,586) 比 `[t,0,:256].tobytes()`；对源 npy 抽 64 帧比 `pos_emb_4x4[0,0,:256]` 与 `pos_emb_8x8[0,0,:256]`；`--image-spot N` 对随机 N 帧 `np.load` 直读 `image_emb_8x8` 与库 `read_image_rows` 比；判定行 `MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0`、`IMAGE_NPY_SPOT=PASS frames=<N> mismatches=0` | |
| `src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory` | `__init__` 新增必填 `token_per_image`，推 `g`（`g*g == token_per_image` 且 `g ∈ {2,4,8}` 否则 raise），`self.pos_emb = pos_embedder(ranges, g)`；`add_buffer` 池到 `token_per_image`、缓冲键 `image_emb_{g}x{g}` / `pos_emb_{g}x{g}`；`_prepare_motion` 改读 `self.pos_emb`；docstring 与 5 处行内注释改正 | `_prepare_frame_sampling`、`get_frame_sampling_indices`、`_encode_ready_windows`、`visible_motion_frames`、motion 客户端调用 |
| `src/mme_vla_suite/policies/policy.py::_prepare_mem_buffer` | 构造 `FrameSampMemory` 时传 `token_per_image=self.history_config.token_per_image` | `_prepare_history` |
| `FrameSampMemory(...)` 其余构造点：`g0/compare_online_motion.py`、`g0/compare_online_memory.py`（两处）、`g0/compare_siglip_replay.py`、`tests/motion_gates_online.py`（5 处）、`tests/eval_rhythm_gates.py` | 同步传 `token_per_image` | |
| `scripts/training/g0/serve_policy_probe.py`、`compare_online_memory.py`、`compare_train_infer_obs.py`、`compare_siglip_replay.py` | `pos_emb_4x4` → `pos_emb`；`"image_emb_4x4"` / `ENC_KEYS` 按 policy 的 `history_config.token_per_image` 推键名；`compare_online_memory.py` 加 `--token-per-image` 与 `ENC_LAYER_8X8` 判定行，`--h5` 默认改本机路径或改用源 npy 帧；`compare_train_infer_obs.py` 加 `--store-subdir`（默认 `framesamp`）、`--train-config` 默认改 `mme_vla_suite`、支持 motion 关闭态（`KEYS8` 按 `motion_enabled` 裁成四键、`want_mem` 按 `motion.enabled` 算、跳过 `visible_motion_rows` / `_motion_client.close()`）、`VT_FULL_VS_CACHED` 改观察行、`VT_FULL_VS_CACHED_F32` 改阻断行（`rel_fro ≤ 1e-6` 且 `prefix_kv_max_abs == 0`） | 其余判据、`THR_REL_FRO=1e-3`、`THR_ULP_P99=4.0` 数值不动（只改阻断归属） |
| `scripts/training/g0/check_config_provenance.py` | 加 `--store-subdir`，四张表指纹按 spec 读 | 三项比对逻辑 |
| `scripts/training/tests/motion_gates_online.py` | `HIST_BUDGET, TOKEN_PER_IMAGE, NUM_VIEWS` 与 `MAX_FRAMES` 改为从 `--yaml` 读；`mem.pos_emb_4x4[…]` → `mem.pos_emb[…]` | 闸门逻辑 |
| `scripts/training/tests/motion_gates_model.py::load_lib_oracle` | 改按 store spec 读 pos 表（`--store-subdir` + `StoreMeta.load`） | 其余 |
| `scripts/motion-variance/run_batch_mv.sh`、`eval_shard_mv.sh` | 新增 env `CKPT_OVERRIDE`（设了则 `CKPT_DIR=$CKPT_OVERRIDE`、`CKPT_ID=$(basename)`，跳过 COND 分支）与 `MV_MOTION_OFF=1`（不检查 wan venv / encoder 权重、不传 `--motion-gpu`、传 `--motion-off`）；`MV_SHARD` 行打印实际 `ckpt=` | 分片公式、端口守卫、tmux 会话名规则 `mv-<t|v><seed>-<cond><SUFFIX>-w<k>` |
| `scripts/motion-variance/serve_policy_mv.py` | 新增 `--motion-off`：跳过 `policy.motion_enabled` 检查、`motion_factory` 返回 `ZeroMotion`、`mv_prepare_history` 关闭态不读 `visible_motion_frames` 与 motion 三键；`MV_SERVE_ENV` 行加 `motion=off/on` 与 `token_per_image` | `MVAdapter`、`_guard_model_source`（不护 `FrameSampMemory`，改它不触发护栏） |
| `scripts/motion-variance/mv_common.py`、`src/mme_vla_suite/shared/sampling.py` | 只改注释口径（`608/512/96/1184` 数值不变；「同帧 16 位」→ `tokens_per_frame` 位） | 数值 |
| `scripts/training/tests/online_mem_regress.py`（新建） | 三方对拍：`git show $REF:src/mme_vla_suite/policies/framesamp_memory.py` 抽到 scratch、改名 import 为 `framesamp_memory_ref`，与主树 `FrameSampMemory` 同进程共享一个 jitted SigLIP、同帧，比 `prepare_frame_sampling` 三键装配输出与 `_prepare_motion` 输出的 `leaf_sha256`；起跑断言两模块 `__file__` 不同且 `find_spec("mme_vla_suite").origin` 在主树；判定行 `ONLINE_MEM_REGRESS=PASS frames=<n> keys=4 mismatches=0` | |

### 2. 规格表（代码级草案）

```python
@dataclasses.dataclass(frozen=True)
class StoreSpec:
    layout: str; grid: int; tokens_per_frame: int
    image_key: str; pos_key: str
    image_row_shape: tuple[int, int]; pos_row_shape: tuple[int, int]
    image_row_bytes: int; pos_row_bytes: int
    image_part_dir: str; pos_table_relpath: str
    source_image_offset: int; source_pos_offset: int

SPECS = {
    "framesamp-4x4-v1": StoreSpec("framesamp-4x4-v1", 4, 16, "image_emb_4x4", "pos_emb_4x4",
        (16, 2048), (16, 768), 16*2048*2, 16*768*4, "image_emb_4x4", "pos_emb_4x4.f32.bin",
        262595, 541352),
    "framesamp-8x8-v1": StoreSpec("framesamp-8x8-v1", 8, 64, "image_emb_8x8", "pos_emb_8x8",
        (64, 2048), (64, 768), 64*2048*2, 64*768*4, "image_emb_8x8", "pos_emb_8x8.f32.bin",
        387, 344682),
}
```

`SOURCE_STATE_OFFSET=602906` 与 `SOURCE_NPY_SIZE=602951` 两档共用。slice 档的偏移必须在运行期由 `_pin_slice_prefix` 对首帧 decode 结果逐字节自证后才可用，本轮首跑一律 `--reader decode`。

### 3. 实施顺序与命令

每步的 Python 一律 `uv run --no-sync`，cwd 仓库根（A 侧为 REF worktree 根）；超过 5 分钟的放 detached tmux（`PYTHONUNBUFFERED=1`、`set -o pipefail`、`tee`、尾行 `EXIT_CODE=`，tmux 命令用 `bash -c '…'` 显式包 bash）并挂 Monitor（`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "PASS|FAIL|DONE|EXIT_CODE=|Error|Traceback"`）。

**阶段 0：preflight（只读，不 source `paths.sh`）**

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA && git status --short && git rev-parse HEAD
for L in 4task-motion-40ep 4task-motion-400ep; do
  jq '{layout,status,num_rows,num_pos_rows,manifest_sha256}' v1-store/datasets/$L/framesamp/meta/store_meta.json
  jq '{layout,status,num_rows}' v1-store/datasets/$L/motion/meta/store_meta.json
  [ ! -e v1-store/datasets/$L/framesamp-8x8 ] && echo "$L/framesamp-8x8 不存在 OK" || echo "错误: 已存在"
done
sha256sum v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json   # 期望 f332bbd3… / 750a8e9b…
ls -l v1-store/models/big_vision/paligemma_tokenizer.model; ls v1-store/models/openpi-assets/checkpoints/pi05_base/params | head -3
env | grep -E "MMEVLA_FRAMESAMP|MMEVLA_MOTION|PYTHONPATH" ; echo "(以上应为空)"
df -h /scratch | tail -1; ls -d ~/.cache/jax_* | wc -l   # 期望 8
uv run --no-sync python scripts/training/train.py mme_vla_suite --help | grep -E "data.assets.assets-dir|data.assets.asset-id"   # 核 tyro 嵌套 flag
```

**阶段 1：量具、参考链、缓存落点、yaml（冻结为参考 commit `REF`）**

改第 1 节表里的 `train.py`、`single_step_grad.py`、`bench_train_steps.py`、`gate_8x8.py`、`check_baseline_env.py`、`_common.py`、`dump_fixture_samples.py`、`compare_fixture_dumps.py`、`ref_npy_dataset.py`、`hand_calc_8frame.py`、`xgrid_pos_check.py`、`online_mem_regress.py`、`test_padding_dtype.py`、两份 yaml。**不碰** `framesamp_store.py`、`pack_framesamp_store.py`、`framesamp_dataset.py`、`framesamp_memory.py`。验证：

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest scripts/training/tests/test_padding_dtype.py -x -q   # 哈希口径未变（已随 fixture_steps 改引用）
uv run --no-sync python scripts/training/tests/gate_8x8.py --self-test   # 负例逐个 FAIL
# 旧链回归 100 步冒烟：完全复刻 aws-t2-ref-s100/launch.md 的参数（40ep 库、默认 norm_stats、--save-interval 1、
# BENCH_DIGEST_INTERVAL=25 BENCH_EXTRA_DIGEST_STEPS=99、seed 42、b8、fsdp 2、worker 4），只换 CUDA_VISIBLE_DEVICES=4,5 与 MMEVLA_JAX_CACHE_DIR；
# scalars_hex.tsv sha 须为 85b8fe37…；指纹 preflight 允许 gpu 差异，留档写明基线 run_name / commit c5925d9 / 比对结论
```

阶段 1 结束后 `git rev-parse HEAD` 记为 `REF`，写进留档，随后建 worktree。

**阶段 2：两档格式、Dataset 白名单、在线记忆、mv 驱动（候选 commit `CAND`）**

改 `framesamp_store.py`、`pack_framesamp_store.py`、`framesamp_dataset.py`、`test_pack_guards.py`（`commitV9.1`）；`framesamp_memory.py`、`policy.py`、6 个构造点、全部工具、`check_config_provenance.py`、`motion_gates_model.py`、mv 三脚本（`commitV9.2`）。验证（≤5 分钟）：

```bash
MMEVLA_TEST_SOURCE=v1-store/datasets/4task-motion-40ep/source \
MMEVLA_TEST_MANIFEST=v1-store/datasets/4task-motion-40ep/meta/episode_manifest.json \
JAX_PLATFORMS=cpu uv run --no-sync pytest scripts/training/tests/test_pack_guards.py -x -q   # 末行 REJECT_INPUTS=PASS cases=<n>
# 旧库仍可读：StoreMeta.load + run_fast_checks + run_full_checks 对两个旧 framesamp/ 通过
# 第三块第一步（阶段 2 末，GPU 7）：
CUDA_VISIBLE_DEVICES=7 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
uv run --no-sync python scripts/training/tests/online_mem_regress.py --ref-commit $REF --frames-from v1-store/datasets/4task-motion-40ep/source --n-frames 256
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/training/g0/compare_online_memory.py --token-per-image 16 --out v1-store/reports/t8-infer-regress-4x4/online_memory
# 真模型 S 臂回归：参数照 tic-obs-model-40k/launch.md，但 CUDA_VISIBLE_DEVICES=7、--motion sidecar --motion-gpu 7、--f32-diag-points-per-episode 3
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/training/g0/compare_online_motion.py --gpu 7 --yaml perceptual-framesamp-context-motion.yaml --lib v1-store/datasets/4task-motion-40ep --out v1-store/reports/t8-infer-regress-4x4/online_motion.json
```

**阶段 3：建库（40ep 先、400ep 后）**

```bash
source scripts/training/paths.sh; export UV_CACHE_DIR=/scratch/hongze/.cache/uv
L=v1-store/datasets/4task-motion-40ep   # 第二轮换 4task-motion-400ep
[ ! -e $L/framesamp-8x8 ] || { echo "错误: 输出根已存在"; exit 1; }
tmux new-session -d -s p8-pack-40 "bash -c 'cd $PWD && set -o pipefail && PYTHONUNBUFFERED=1 uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source $L/source --manifest $L/meta/episode_manifest.json --out $L/framesamp-8x8 --procs 48 2>&1 | tee v1-store/logs/p8-pack-40.log; echo EXIT_CODE=\${PIPESTATUS[0]} | tee -a v1-store/logs/p8-pack-40.log'"
# 期望 PACK_DONE=1；随后
tmux new-session -d -s p8-verify-40 "bash -c '… pack_framesamp_store.py verify --store $L/framesamp-8x8 --resume --procs 48 … p8-verify-40.log'"
# 期望 VERIFY_PACK=PASS scanned=13756 mismatches=0（400ep：scanned=123044）
uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store $L/framesamp-8x8
uv run --no-sync python scripts/dataset/xgrid_pos_check.py --store-4x4 $L/framesamp --store-8x8 $L/framesamp-8x8 --source $L/source --image-spot 512
```

400ep 会话名 `p8-pack-400`、`p8-verify-400`。留档到 `docs/dataset-build-doc/4task-motion-{40ep,400ep}-framesamp-8x8/`。

**阶段 4：第一块验收（装配层）**

```bash
V1=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store; L=$V1/datasets/4task-motion-400ep
ASSETS="--assets-base-dir $V1/train-assets --data.assets.assets-dir $V1/train-assets/mme_vla_suite/robomme-400ep --data.assets.asset-id robomme"
export DTYPE_MANIFEST=$L/meta/episode_manifest.json MMEVLA_MOTION_STORE=$L/motion
for IMPL in refnpy packed; do
  DTYPE_DUMP_DIR=$V1/fixtures/8x8/$IMPL-c8 DTYPE_DUMP_IMPL=$IMPL JAX_PLATFORMS=cpu \
  uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite --exp-name fx-c8-$IMPL $ASSETS \
    --dataset-path $L/$([ $IMPL = packed ] && echo framesamp-8x8 || echo source) \
    --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled
done   # M8 同法换 -motion.yaml
uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py $V1/fixtures/8x8/refnpy-c8 $V1/fixtures/8x8/packed-c8
# C32/M32：dump 目录换旧库 + 旧 yaml，参考侧在 REF worktree 跑 packed（cd worktree && PYTHONPATH=$PWD/src UV_PROJECT_ENVIRONMENT=<主树>/.venv）
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/spawn_matrix.py --store $L/framesamp-8x8 --source $L/source --manifest $L/meta/episode_manifest.json --yaml perceptual-framesamp-context-8frame-8x8.yaml --workers 0,1,4,16 --epochs 2   # 再 --impl refnpy 一遍
uv run --no-sync python scripts/training/tests/hand_calc_8frame.py --store $L/framesamp-8x8 --manifest $L/meta/episode_manifest.json --motion $L/motion
```

超过 5 分钟的项进 tmux（前缀 `p8-fx-`）并按第 17 条留档。

**阶段 5：第二块（12 条轨迹）**

两波、每波两组并行、组内串行；下面以 C8 组（第二波，GPU 4,5）为例。A1/A2 在 REF worktree 跑，B 在主树跑：

```bash
V1=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store; L=$V1/datasets/4task-motion-400ep; RUN=t8-c8-a1   # run_name 已由用户 2026-09-14 确认
REC=$V1/bench/8x8/$RUN; mkdir -p $REC
ASSETS="--assets-base-dir $V1/train-assets --data.assets.assets-dir $V1/train-assets/mme_vla_suite/robomme-400ep --data.assets.asset-id robomme"
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=4,5
export MMEVLA_JAX_CACHE_DIR=$V1/cache/jax/$RUN UV_CACHE_DIR=/scratch/hongze/.cache/uv WANDB_MODE=disabled
# A 侧：cd $V1/worktrees/ref-8x8 && export PYTHONPATH=$PWD/src UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv ；B 侧：cd 主树、不设 PYTHONPATH
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir $REC --v1-store $V1 --source $L/source --manifest $L/meta/episode_manifest.json --dataset $L/framesamp-8x8 --norm-stats $V1/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json   # A 侧 --dataset 省略（refnpy）
BENCH_RECORD_DIR=$REC BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000 BENCH_EXTRA_DIGEST_STEPS=1,2,24,49,99,199,399,599,799 \
BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=999 \
BENCH_DATASET_IMPL=refnpy BENCH_REF_SOURCE=$L/source BENCH_REF_MANIFEST=$L/meta/episode_manifest.json \
uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name $RUN $ASSETS \
  --checkpoint-base-dir $V1/train-runs/$RUN \
  --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 2 \
  --dataset-path $L/source \
  --weight-loader.params-path $V1/models/openpi-assets/checkpoints/pi05_base/params \
  --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled
uv run --no-sync python scripts/training/tests/project_scalars.py $REC/metrics.jsonl $REC/scalars_hex.tsv   # 产 scalars_hex.tsv
uv run --no-sync python scripts/training/g0/check_baseline_env.py manifest $REC && uv run --no-sync python scripts/training/g0/check_baseline_env.py check --base $V1/bench/8x8/t8-c8-a1 --record-dir $REC   # A1 自检；A2/B 对 A1 → BASELINE_ENV=PASS
# A2：RUN=t8-c8-a2，其余逐字相同。B：RUN=t8-c8-b，主树，去掉 BENCH_DATASET_IMPL/BENCH_REF_*，--dataset-path $L/framesamp-8x8
# C32/M32 三条：--dataset-path $L/framesamp 与旧 yaml，A 侧 REF worktree（带 PYTHONPATH）、B 侧主树，无 refnpy；M32/M8 加 MMEVLA_MOTION_STORE=$L/motion（M8 A 侧另加 BENCH_REF_MOTION=$L/motion）
uv run --no-sync python scripts/training/tests/gate_8x8.py --profile c8 --run-a1 $V1/bench/8x8/t8-c8-a1 --run-a2 …/t8-c8-a2 --run-b …/t8-c8-b --log-a1 v1-store/logs/t8-c8-a1.log … --env-a1 … --steps 1000 --batch-size 8
```

M8 组用 `CUDA_VISIBLE_DEVICES=6,7`，yaml 换成 `-motion.yaml`。每条轨迹放 detached tmux（会话名 = run_name，前缀 `t8-`），日志 `v1-store/logs/<run_name>.log`。`-s100`：`--num-train-steps 100`、run_name后缀 `-s100`、`BENCH_EXTRA_DIGEST_STEPS=1,2,24,49`、不带 `BENCH_SAVE_FINAL_CKPT`。实施时按用户追加决定，仅这四个排错设置 `BENCH_CHECKSUM=0`，保留100次更新、逐步标量和输入摘要；结束后已逐个核实归属并清理records、run根和日志，缓存保留，不作正式证据。阶段1与历史100步基线的完整回归记录不在此豁免中。

单步全梯度对拍（每 profile 三种 batch）：`single_step_grad.py` 走 `DTYPE_MANIFEST` / `DTYPE_DUMP_IMPL` 与同一 `$ASSETS`，两侧摘要经 `compare_grad_summaries.py`，判定行 `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0`。

**阶段 6：第三块（推理）**

第一步已在阶段 2 末完成。第二步（`t8-c8-b` / `t8-m8-b` 跑完之后）：

```bash
CK=$V1/train-runs/t8-m8-b/mme_vla_suite/t8-m8-b/999   # run 根 $(dirname $CK) 须含 history_config.resolved.yaml/.sha256 与 motion_provenance.json
ls $(dirname $CK)/history_config.resolved.yaml $(dirname $CK)/motion_provenance.json $CK/params
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/training/g0/check_config_provenance.py --ckpt $CK --lib $L --store-subdir framesamp-8x8
CUDA_VISIBLE_DEVICES=7 MMEVLA_MOTION_STORE=$L/motion \
uv run --no-sync python scripts/training/g0/compare_train_infer_obs.py --ckpt $CK --lib $L --store-subdir framesamp-8x8 --train-config mme_vla_suite --motion sidecar --motion-gpu 7 --f32-diag-points-per-episode 3 --out v1-store/reports/t8-infer-m8/tic.json   # --episodes 沿用默认 task:idx 列表
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/training/g0/compare_online_memory.py --token-per-image 64 --out v1-store/reports/t8-infer-m8/online_memory
# 关 6：serve_policy_probe.py --ckpt $CK --lib $L --config mme_vla_suite --motion-gpu 7 --probe-out … + 一集仿真 + summarize_eval_probe.py --max-frames 8 --tokens-per-frame 64 --probe … --progress … --eval-log … --server-log …
# C8 同法：CK=$V1/train-runs/t8-c8-b/mme_vla_suite/t8-c8-b/999，不带 MMEVLA_MOTION_STORE、--motion store、无 --motion-gpu
```

第三步（闭环冒烟，4 片 4 卡，tmux 会话由 `run_batch_mv.sh` 起、名 `mv-t42-normal-t8m8-w{0..3}`）：

```bash
COND=normal SPLIT=test SEED=42 WORKERS=4 GPU_LIST=4,5,6,7 EP_COUNT=10 RUN_SUFFIX=-t8m8 CKPT_OVERRIDE=$V1/train-runs/t8-m8-b/mme_vla_suite/t8-m8-b/999 bash scripts/motion-variance/run_batch_mv.sh
COND=normal SPLIT=test SEED=42 WORKERS=4 GPU_LIST=4,5,6,7 EP_COUNT=10 RUN_SUFFIX=-t8c8 MV_MOTION_OFF=1 CKPT_OVERRIDE=$V1/train-runs/t8-c8-b/mme_vla_suite/t8-c8-b/999 bash scripts/motion-variance/run_batch_mv.sh
# 期望 MV_BATCH=PASS … episodes=48/48；MV_SHARD 行 ckpt= 必须是 t8-*-b 路径
```

判定只看 `errors=0`、每集推理次数 ≤ 82、`mem_order` 合法、M8 窗数与公式一致；成功率照实记但标「测试模型，不作指标」。

### 4. 判定行清单（最终留档必须逐条出现，禁止一条笼统 PASS）

| 判定行 | 出处 | 通过条件 |
|---|---|---|
| `VERIFY_PACK=PASS scanned=<rows> mismatches=0` | pack_framesamp_store verify（40ep 13756、400ep 123044） | 两库各一条 |
| `IMAGE_NPY_SPOT=PASS frames=512 mismatches=0` | xgrid_pos_check | 两库各一条 |
| `MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0` | xgrid_pos_check | 两库各一条 |
| `SOURCE_IDENTITY=PASS episodes=<n> samples=<n>` | compare_fixture_dumps | 全部一致 |
| `FRAME_INDEX_EXACT=PASS steps=<n> max_frames=8` | compare_fixture_dumps | 全部合法 step |
| `SAMPLE_RAW_EXACT=PASS samples=<n> per_step=<k> mismatches=0` | compare_fixture_dumps | 四个 profile 各一条 |
| `BATCH_RAW_EXACT=PASS batches=200 mismatches=0` | compare_fixture_dumps | 四个 profile 各一条 |
| `HAND_CALC_8FRAME=PASS samples=<n> mismatches=0` | hand_calc_8frame | C8、M8 |
| `MATRIX=PASS workers=0,1,4,16 epochs=2 batches_per_run=<n>` | spawn_matrix | 40ep 8×8 库，packed 与 refnpy 各一条 |
| `REJECT_INPUTS=PASS cases=<n>` | test_pack_guards | 全部错配 raise |
| `BASELINE_ENV=PASS` | check_baseline_env check | 每条轨迹起跑前 |
| `BASELINE_REPEAT_EXACT=PASS profile=<p>` | gate_8x8（A1 vs A2） | 四个 profile |
| `TRAIN_1000_EXACT=PASS profile=<p>` | gate_8x8（A1 vs B，1000 步五标量 hex） | 四个 profile |
| `FINAL_STATE_EXACT=PASS profile=<p> state_steps=11` | gate_8x8（摘要步全树） | 四个 profile |
| `FINITE_ALIVE=PASS profile=<p>` | gate_8x8（有限性 + 活性） | 四个 profile |
| `IMPORT_ORIGIN=PASS profile=<p>` | gate_8x8 | A 侧 REF worktree、B 侧主树 |
| `NORM_STATS_ACTUAL=PASS sha=750a8e9b…` | gate_8x8 | 四个 profile |
| `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0` | compare_grad_summaries | 四个 profile |
| `GATE_8X8=PASS profile=<p> steps=1000 batch=8 …` | gate_8x8 总行 | 四个 profile |
| `ONLINE_MEM_REGRESS=PASS frames=<n> keys=4 mismatches=0` | online_mem_regress（4×4，真模型 SigLIP） | 第三块准入 |
| `ENC_LAYER=PASS … ASSEMBLY=PASS` | compare_online_memory（4×4） | 第三块准入 |
| `tic-obs-model-40k` 的 12 条 PASS 判据同值 + `VT_FULL_VS_CACHED` 同为 FAIL 且量级相当 + `VT_FULL_VS_CACHED_F32=PASS` | compare_train_infer_obs S 臂，`awsprod40k-b128-motion/39999` | 4×4 在线回归 |
| `ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0` 等五条 | compare_online_motion，40ep 库 | 4×4 在线回归 |
| 关 0 判定行 | check_config_provenance，`t8-{c8,m8}-b` | C8、M8 各一套 |
| 关 1–5 判定行（含 `VT_FULL_VS_CACHED_F32=PASS`，bf16 `VT_FULL_VS_CACHED` 只记数） | compare_train_infer_obs，`--store-subdir framesamp-8x8` | C8、M8 各一套 |
| `ENC_LAYER_8X8=PASS steps=<n> mismatch=0` | compare_online_memory（8×8） | 阻断 |
| 关 6 判定行（`EVAL_BACKEND` 等） | serve_policy_probe + summarize_eval_probe | C8、M8 各一套 |
| `EVAL_SMOKE=DONE profile=<p> episodes=48/48 errors=0 max_infer<=82 mem_order_ok=1` | 闭环冒烟汇总 | C8、M8 各一条；成功率不作指标 |

### 5. 前后链路图（实施时按实测数字更新后落档）

**图 A-before（32 帧 × 4×4，现行）**

```text
源库 source/features/episode_g/token_emb_t.npy + source/data/idx.pkl + meta/episode_manifest.json
  image_emb_4x4 (1,16,2048) bf16 65,536 B/帧；pos_emb_4x4 (1,16,768) f32 49,152 B/帧；state_emb (8,) f32 32 B
  │ pack_framesamp_store.py pack（decode）→ 31/22 个 part + 两张小表；不改数
  ▼
framesamp/ (framesamp-4x4-v1)：image 行 (16,2048) bf16；pos 表 586 行 (16,768) f32；state 行 (8,) f32
  │ FrameSampDataset：even_sampling_indices(step,32) → rows = row_base[g]+frames → preadv；不改数
  ▼
最多 32 帧 → _pad 同 dtype 右补零（补零位不改已有数）→ reshape/repeat：
  static_image_emb (512,2048) bf16 2,097,152 B；static_pos_emb (512,768) f32 1,572,864 B
  static_state_emb (512,8) f64 32,768 B（_normalize_state 改数，含补零位）；static_mask (512,) bool 512 B
  motion 开启：motion_emb (96,768) f32；motion_pos (96,256) f32 = pos_rows(f_m)[:,0,:256]；motion_mask (96,) bool；mem_order (608,) int32
  │ RepackTransform（13 键映射，未登记键丢弃）→ RoboMMEInputs → DeltaActions（改 actions）→ Normalize（只改 state/actions，不碰 static_state_emb）
  │ → InjectDefaultPrompt / ResizeImages(224) / TokenizePromptWithState / PadStatesAndActions（改图与 prompt）
  │ → _collate_fn np.stack（None 透传）→ make_array_from_process_local_data → HistAugObservation
  ▼
HistoryPi0：inputs_spec 从 budget 推形状；PerceptualMemory 取 budget；mem_order gather 608 位
```

**图 A-after（8 帧 × 8×8，目标）**

```text
同一源库、同一 manifest、同一 pkl
  image_emb_8x8 (1,64,2048) bf16 262,144 B/帧；pos_emb_8x8 (1,64,768) f32 196,608 B/帧；state_emb 同上
  │ pack_framesamp_store.py pack --layout framesamp-8x8-v1（decode）→ part 边界与 4×4 库相同、字节 ×4；不改数
  ▼
framesamp-8x8/ (framesamp-8x8-v1)：image 行 (64,2048) bf16；pos 表 586 行 (64,768) f32；state 行 (8,) f32
  │ FrameSampDataset（白名单 (512,64,1) + spec 匹配）：even_sampling_indices(step,8) → 同一行号规则；不改数
  ▼
最多 8 帧 → _pad 到 8 → reshape/repeat 64：
  static_* 四键形状、dtype、字节数与图 A-before 逐项相同；内容完全不同（8 帧 × 64 token vs 32 帧 × 16 token）
  motion 开启：motion_emb / motion_pos / motion_mask 逐位同 A-before（同源、同窗口，visible_motion_rows 不依赖 max_frames；时间码切片跨网格一致已实证）；
  mem_order 仍 (608,) int32 合法置换，但排列随 8×64 的帧路布局变化（设计内）
  │ transforms → collate → JAX 交付：与图 A-before 同一套代码、同一形制
  ▼
同一 HistoryPi0，无参数增减
```

每样本 image 读取仍是 2 MiB（8 行 × 256 KiB），pos 小表每 worker 从 28.8 MB 增到 115 MB（`np.fromfile` 整表进内存，4 worker 约 460 MB/run）。

### 6. 留档、tmux 前缀与 commit

- 阶段 1 `commitV9.0`、阶段 2 `commitV9.1` / `commitV9.2`，各自提交后 `git push`；阶段 1 结束 HEAD 记为 `REF`，`9.2` 记为 `CAND`，两者都写进每份 launch.md，起跑 HEAD 另记。
- **本轮 tmux 会话名前缀清单（清理只按此）**：`p8-pack-*`、`p8-verify-*`、`p8-fx-*`（第一块）；`t8-<profile>-{a1,a2,b}`、`t8-<profile>-s100`（第二块）；`mv-t42-normal-t8m8-w{0..3}`、`mv-t42-normal-t8c8-w{0..3}`（第三块）。`tmux ls` 里不在清单的会话一律不动（当前存在的用户会话：`0`、`1`、`claude-private`、`codex`、`codex-local4task-full1600-20260913-v825`、`codex-repo`）。
- 建库留档 `docs/dataset-build-doc/4task-motion-{40ep,400ep}-framesamp-8x8/{launch.md,result.md,records/}`，records 收 `store_meta.json`、`report` 输出、判定行、日志尾部。
- 训练留档 `docs/training-doc/t8-<profile>-{a1,a2,b}/`，每条轨迹 `launch.md`（REF / CAND / 起跑 HEAD、完整命令、env 含 `PYTHONPATH`、GPU 对、数据路径、norm_stats 路径与 sha）、`result.md`（判定行、耗时分解、吞吐、显存、存储介质）、`records/`（`scalars_hex.tsv`、`param_checksums.jsonl`、`batch_digests.jsonl`、`index_sequence.json`、`env.json`、`run_meta.json`）。不归档 checkpoint。`-s100` 不留档（用户裁决第 8 项）。
- 推理留档 `docs/training-doc/t8-infer-regress-4x4/`（第一步）、`t8-infer-{c8,m8}/`（第二、三步），各含 `launch.md`（代码 commit、checkpoint 完整路径与其来源 run、完整命令、GPU）、`result.md`（七关判定行、bf16 关 4 观察值、冒烟四条判定、显存与 host 内存、成功率并标注不作指标）、`records/`。
- 收官：`docs/dataloader-restructure.md` 加一节「8 帧 × 8×8 档」；`docs/train-infer-consistency.md` 加一节「8×8 档（测试模型）」；本文降为过程档案。

### 7. 已由用户拍板的项（2026-09-14，见文首表）

数据口径 400ep + 显式 norm_stats；第三块驱动改 mv 三脚本、禁 legacy、48 集；两波串行 fsdp 2；关 4 bf16 观察 / f32 阻断；`UV_CACHE_DIR=/scratch/hongze/.cache/uv`；12 个 run_name 与启动覆盖参数；`commitV9.0/9.1/9.2`；`-s100` 豁免留档；C8/M8 补手算边界证据。**实施前不再有待拍板项**；实施中若阶段 0 的 `--help` 核出 tyro 不接受 `--data.assets.*` 嵌套 flag，按第一部分所述改为新增 `_CONFIGS` 条目，属既定备选，不另请示。

### 8. 计划修订时的验证与提交（历史）

2026-09-14的计划修订轮仅改根目录 `8frame-8x8-training-plan.md`；随后用户已授权并完成上述实施。这段保留计划修订时的验证记录：`git diff --check`、核对无尾随空白与末尾换行、表格与代码围栏配对、引用的文件路径在 HEAD 上存在（新建文件已标「新建」）。提交主题 `docs: 8帧8×8计划按对抗验证结论与用户 9 项裁决修订`，逐文件 `git add`，commit 后 `git push` 到既有 upstream，`git status -sb` 无 ahead。


### 9. 实施落点与最终证据

源码分三次落地：REF `99faacb1319adfc63c0cf9a15187e24c34e38fd1`（commitV9.0），双布局 `236765fdb7b35f8fc96fb7d71133870b70b76271`（commitV9.1），在线与评估候选 `c08ec2060a544af1869c1e24f755e536150569ca`（commitV9.2）。具体run从各自启动档案中的clean HEAD执行，不能把候选功能提交误当成每次实际启动HEAD。

两库的全量行校验、512帧抽样和586行跨网格位置检查见[40ep](docs/dataset-build-doc/4task-motion-40ep-framesamp-8x8/result.md)与[400ep](docs/dataset-build-doc/4task-motion-400ep-framesamp-8x8/result.md)。输入、worker矩阵、29项拒绝和独立手算见[fixture档案](docs/training-doc/t8-fixture/result.md)。四组1000更新全部11状态点和三batch全梯度均逐位通过；推理按bf16观察/f32阻断的已确认口径验收。实际数值、命令、硬件、耗时及限制均在文首链接的结果档案中。开环与探针实际按C8用GPU7、M8用GPU6并行，各组内部始终同卡比较；48集仍按M8→C8顺序使用GPU4–7四卡，实际命令以各run启动档案为准。长期链路图明确标出host static_state_emb为f64、JAX x64关闭后交付f32的既有转换。
