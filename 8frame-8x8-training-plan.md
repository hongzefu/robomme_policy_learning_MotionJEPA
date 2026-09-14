# 8 帧 × 8×8 训练支持与前后对拍计划（细化版）

本文是 2026-09-08 版计划的细化，2026-09-13 按 `AGENTS.md` 第 2 条改成两部分结构：**第一部分给人看**，分「数据预处理」「训练」「推理」三块（推理块 2026-09-13 按用户要求补入：没有训练收敛的 8×8 模型，checkpoint 只能用第二块 1000 步测试 run 的产物，成功率不作指标）；**第二部分供 agent 追踪**，列文件、函数、命令与判定行。细化依据是对 HEAD `82c2ccef509d34800a397f547d7144e151cb8d4a` 的只读静态核实（未执行任何仓库脚本、未训练；唯一一次动态动作是用 `.venv/bin/python` 只读 `np.load` 打开了三个源 npy 核对键名）。环境判定为 B：仓库 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，8 × A100-SXM4-80GB，无 turbo、无 GreatLakes，`/scratch` 余量 4.1 T。

**授权状态不变**：当前只授权本文件。代码修改、建库、测试、训练一样都没开始，每一步都要另行授权。用户已定死的口径：最终真实训练对拍每侧 **1000 次参数更新**，不能用短测代替。

**GPU 约束（2026-09-14 用户新增，最高优先级）**：本计划一切用 GPU 的命令**只能用 GPU 4、5、6、7**，GPU 0–3 一律不碰。落地三条：每条起 GPU 进程的命令都显式写 `CUDA_VISIBLE_DEVICES`，取值只能是 `{4,5,6,7}` 的子集，禁止不设而继承默认全卡；motion sidecar 的卡号**不跟随** `CUDA_VISIBLE_DEVICES`——主线 `policies/policy_config.py::create_trained_policy` 直接读 yaml 快照的 `motion.online_gpu`（现值 1，绝对卡号）且无环境变量覆盖，所以凡起 sidecar 的路径必须走带显式卡号参数的入口（`scripts/motion-variance/serve_policy_mv.py --motion-gpu`、`compare_train_infer_obs.py --motion-gpu`、`compare_online_motion.py --gpu`，或 legacy-eval 副本的 `MMEVLA_MOTION_ONLINE_GPU`），并把新 8×8 motion yaml 的 `online_gpu` 写成 4 作兜底；第二块 12 条轨迹只有两对卡可用，分两波跑（见第二块）。

---

## 第一部分（给人看）

### 一句话

现在的训练把历史压成「最多 32 帧，每帧 4×4 = 16 个视觉 token」。本轮新增一种并列配置「最多 8 帧，每帧 8×8 = 64 个 token」。两种都是 512 个 token，模型一个参数不加、一条计算式不改，改动全在数据侧：**打一份新库**（第一块）、**让 Dataset 和训练入口认得这份新库并证明没改坏**（第二块）、**让在线记忆构造认得 8×8 并证明推理侧与训练侧交付一致**（第三块）。

### 第一块：数据预处理——把 8×8 特征打成一份新的 packed 库

#### 结论先行

- **不用重抽 SigLIP。** 源特征 npy 里三档 image/pos 键都在。实开 400ep 与 40ep 各一个文件（`features/episode_0/token_emb_0.npy`、`episode_399/token_emb_50.npy`）确认键名、形状、dtype 完全一致，字节账 602,144 B 数据 + 807 B 头 = 602,951 B 与文件大小相符。
- **要做的只是新增一档库格式并打一份新库。** 现有 packed 库每帧只固化了 4×4 的 16 个 token，8×8 的 64 个 token 从未进库。
- **规模**：400ep 新库约 30 GiB，40ep 约 3.4 GiB，磁盘余量 4.1 T，不成问题。
- **不改旧库、不改旧格式、不动源库。** 新库落在各自库目录下的新子目录 `framesamp-8x8/`，与旧 `framesamp/` 并列。

#### 完整数据链路与本轮新增的调用

现有链路一共跑 9 个脚本、12 次调用（`LIB=v1-store/datasets/4task-motion-400ep`，全部 `uv run --no-sync`，出处 `docs/dataset-build-doc/4task-motion-400ep/launch.md`「命令序列」）：

```bash
source scripts/dataset/paths.sh; v1_prepare_dirs
# 1 清单 + 输入 sha
scan_manifest.py build --raw_dir $RAW --tasks … --episodes-per-task 100 --out $LIB/meta/episode_manifest.json
finalize_checks.py hash-inputs --raw_dir $RAW --out $LIB/meta/input_manifest.json
# 2 SigLIP 建源库（写 7 键 npy + pkl）
run_local.py --stage siglip --lib $LIB --gpus 2,3,7 --raw-dir $RAW
finalize_checks.py check --manifest … --out $LIB/source --spot_check 1024
# 3 视觉 packed 库
pack_framesamp_store.py pack   --source $LIB/source --manifest … --out $LIB/framesamp --procs 48
pack_framesamp_store.py verify --store $LIB/framesamp --resume --procs 48
# 4 norm_stats
compute_norm_stats.py --dataset-path $LIB/source --output-dir v1-store/train-assets/mme_vla_suite/robomme-400ep
# 5 motion 支路
run_local.py --stage wan    --lib $LIB --gpus 2..7          # 切 33 帧窗 → Wan VAE → wan-latents/
run_local.py --stage encode --lib $LIB --gpus 0..7          # MotionJEPA encoder → motion-tokens/
pack_motion_store.py pack   --manifest … --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion
pack_motion_store.py verify --store $LIB/motion --resume
# 6 motion 旁证
oracle_driver.py vae / aggregate / encoder  →  compare_wan.py latents / tokens
```

训练时 `train.py --dataset-path $LIB/framesamp` 加 `MMEVLA_MOTION_STORE=$LIB/motion`，`FrameSampDataset` 把第 3 步和第 5 步的两张表拼起来。

**改完之后，这 12 次调用一次都不重跑。** 8×8 库只在第 3 步旁边多跑两条：

```bash
# 3' 新增：同一个脚本、同一个源库，换键打一份新表
pack_framesamp_store.py pack   --layout framesamp-8x8-v1 --reader decode --source $LIB/source --manifest … --out $LIB/framesamp-8x8 --procs 32
pack_framesamp_store.py verify --store $LIB/framesamp-8x8 --resume --procs 32
# 3'' 新增：汇合点检查，8×8 与 4×4 两张 pos 表的 patch 0 时间码逐位相同
xgrid_pos_check.py --store-4x4 $LIB/framesamp --store-8x8 $LIB/framesamp-8x8 --source $LIB/source
```

训练时只换 `--dataset-path $LIB/framesamp-8x8`，`MMEVLA_MOTION_STORE` 还是指原来的 `$LIB/motion`。

代码上加的东西对应这两条命令：`pack_framesamp_store.py` 加 `--layout` 参数并让 decode 守卫、verify、空间预检按规格表走；`framesamp_store.py` 加规格表让 `StoreMeta.load` 和 reader 认 `framesamp-8x8-v1`；`framesamp_dataset.py` 放开白名单并核对库规格；`xgrid_pos_check.py` 是新脚本。第 1、2、4、5、6 步的脚本一行不改。

40ep 开发库同样的两条命令先跑一遍，`--out` 换成 `4task-motion-40ep/framesamp-8x8`。

#### 源数据长什么样

400ep 库 `v1-store/datasets/4task-motion-400ep/source/`，40ep 库同构：

| 项 | 400ep | 40ep |
|---|---:|---:|
| `features/episode_*` 目录数 | 400 | 40 |
| 帧总数（= `episode_manifest.json` 的 `totals.timesteps`） | 123,044 | 13,756 |
| 执行样本数（`totals.exec_samples`，= `data/*.pkl` 个数） | 101,066 | 11,530 |
| 最长 episode 帧数（决定 pos 表行数） | 586 | 586 |
| `source/` 体积 | 107 G | 13 G |

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

写这个 dict 的是 `dataset_builder/mem_buffer.py::MemoryBuffer.add_buffer`（三档来自同一次 SigLIP 前向后 `pool_tokens_to_size` 到 64/16/4），落盘在 `dataset_builder/build_robomme_dataset.py::DatasetProcessor._process_episode`。40ep 与 400ep 是**两个独立建的库**（manifest sha 不同、文件 inode 不同、episode 取样口径不同：400ep 每任务 `raw_ep_idx 0..99`，40ep 每任务 `0..9`），不是 subset 引用。

#### 现有 packed 库为什么不能直接放 8×8

现有格式 `framesamp-4x4-v1` 由 `src/mme_vla_suite/datastore/framesamp_store.py` 顶部一组常量钉死，每一处都和 4×4 绑定：

- 形状与字节：`IMAGE_ROW_SHAPE=(16,2048)`、`POS_ROW_SHAPE=(16,768)`、`IMAGE_ROW_BYTES=16*2048*2`、`POS_ROW_BYTES=16*768*4`。
- 键名与文件名：`IMAGE_KEY="image_emb_4x4"`、`POS_KEY="pos_emb_4x4"`、`POS_TABLE_RELPATH="pos_emb_4x4.f32.bin"`、`IMAGE_PART_DIR="image_emb_4x4"`。
- 源 npy 固定偏移（slice 读档用）：`SOURCE_IMAGE_OFFSET=262595`、`SOURCE_POS_OFFSET=541352`、`SOURCE_STATE_OFFSET=602906`，只对 4×4 窗口成立。
- `StoreMeta.load` 校验 `tables` 的形制时拿的是这些常量，不是 meta 自述；`run_fast_checks` 的小表大小账、`FrameSampStore.__init__` 里 pos 表的 reshape 同理。
- 打包脚本 `scripts/dataset/pack_framesamp_store.py::read_frame_decode` 的形制守卫写死 `img.shape == (1,16,2048)`，CLI 没有网格参数。

所以「把 yaml 里 `token_per_image` 改成 64」在 Dataset 构造时就会被 `FrameSampDataset.__init__` 的硬相等断言 `(budget, token_per_image, num_views) == (512, 16, 1)` 拦下；就算放开断言，旧库每帧也只有 16 个 token，交付不出 64 个。

#### 新库怎么建

**规格表驱动。** 在 `framesamp_store.py` 里把上面那组散常量收成一张规格表，键是布局名，两档：

| 布局名 | 网格 | 每帧 token | image 键 / 行形状 / 行字节 | pos 键 / 行形状 / 行字节 | part 目录 | pos 表文件 |
|---|---|---:|---|---|---|---|
| `framesamp-4x4-v1`（不变） | 4×4 | 16 | `image_emb_4x4` / (16,2048) / 65,536 | `pos_emb_4x4` / (16,768) / 49,152 | `image_emb_4x4/` | `pos_emb_4x4.f32.bin` |
| `framesamp-8x8-v1`（新增） | 8×8 | 64 | `image_emb_8x8` / (64,2048) / 262,144 | `pos_emb_8x8` / (64,768) / 196,608 | `image_emb_8x8/` | `pos_emb_8x8.f32.bin` |

state 表两档相同：`state_emb` (8,) f32，32 B/行。`StoreMeta.load` 改成先读 meta 里的 `layout`，查表得规格，再校验 `tables` 与规格逐项相符；reader、packer、verify 全部从同一张表取键名、形状、字节数。旧库 meta 无需迁移，`layout` 字段本来就有。

**新库落点**（实体目录，不建外链）：

```
v1-store/datasets/4task-motion-40ep/framesamp-8x8/     开发库，先建
v1-store/datasets/4task-motion-400ep/framesamp-8x8/    正式库，后建
```

源 pkl、`episode_manifest.json`、motion store 通过 meta 里的 `source_dataset_root`、`manifest_path` 与显式路径复用，不复制。

**打包流程与旧库完全同一套**，只是键换成 8×8：

1. `pack` 子命令加 `--layout` 参数（默认 `framesamp-4x4-v1`，旧行为不变），`--reader decode` 首跑（完整反序列化每个 npy 后取键，零布局假设；slice 档在 decode 建库并 verify 通过前不用）。
2. part 切分由 `plan_parts` 按 manifest 的 episode 边界贪心，只看帧数不看行宽，所以 **8×8 库的 part 边界与 4×4 库逐一相同**：400ep 31 个 part，40ep 22 个 part，只是每个 part 的字节数是原来的 4 倍（400ep 单 part 约 1.0 GB）。
3. 写侧三重校验照旧：每帧 pos 与 `pos_table[t]` 逐字节相同（钉死「pos 只依赖帧号」）、每帧 state 与 state 表同行相同、每 episode slab 落盘后读回比对且 sha 取读回字节。
4. `verify` 全量档：每个 part 的每一帧重新 decode 源 npy，经**真实读 API**（`FrameSampStore.read_image_rows / pos_rows / state_rows`）逐字节比对三键，同时逐行算 blake2b-128 写 `meta/row_digests.blake2b.bin`。判据 `scanned == num_rows`、`mismatches == 0`，通过才回填 `status="verified"` 并释放 `pack.lock`。抽样档 `--sample N` 只作开发快检，不作交付判定。

**空间预检要改。** `cmd_pack` 的全量路径写死 `floor = 40 * 10**9`，对 4×4 的 8 GB 库绰绰有余，对 8×8 的 30 GiB 库只剩约 7.7 GB 余量，且没算临时 part 文件。改成 `floor = max(40 GB, 1.25 × need)`，其中 `need = num_rows × image_row_bytes + num_rows × 32 + max_timesteps × pos_row_bytes` 按规格表现算。起跑前照第 14 条先 `ls -ld <输出根>`。

**预计字节账**（公式推算，不是实测）：

| 表 | 400ep | 40ep |
|---|---:|---:|
| image（rows × 262,144） | 32,255,246,336 B ≈ 30.0 GiB | 3,606,052,864 B ≈ 3.36 GiB |
| pos（586 × 196,608） | 115,212,288 B | 同 |
| state（rows × 32） | 3,937,408 B | 440,192 B |
| row_digests（rows × 16） | 1,968,704 B | 220,096 B |

**耗时预期**：旧库 meta 记的 pack 5 s / verify 2 s（48 进程）是源库完全命中 page cache 时的数字，不能当基准。8×8 打包要完整读 123,044 × 603 KB ≈ 74 GB 源文件并写 30 GB，冷缓存下按 NVMe RAID 量级估计十几分钟到半小时，按第 7 条视作长任务：detached tmux、日志三件套、挂 Monitor。

#### 怎么证明新库没改数

分两层，都在不启动训练的前提下完成。

**第一层：库层（pack 与 verify 本身就是判据）。** `VERIFY_PACK=PASS scanned=<num_rows> mismatches=0` 证明库里每一行三键与源 npy 逐字节相同。另加两条 8×8 专属检查：

- pos 表跨 episode 同值：随机取若干 episode 的若干帧，源 npy 的 `pos_emb_8x8[t]` 与库 `pos_table[t]` 逐字节相同（写侧校验①已覆盖全部帧，这里是独立复核）。
- **motion 时间码跨网格一致**：motion 路取的是 `store.pos_rows(f_m)[:, 0, :256]`，即第 0 个 patch 的前 256 维（PosEmb3D 的时间段）。8×8 库的 patch 0 与 4×4 库的 patch 0 时间段理应逐位相同，但没人验过。对全部 586 个帧号比对两库 pos 表该切片的 raw 字节，判定行 `MOTION_POS_XGRID=PASS t=586 mismatches=0`。不过就不得让 8 帧 motion 配置起跑。

**第二层：装配层（Dataset 输出对参考链）。** 8 帧从没跑过，没有现成参考，所以写一条**只用于验证的独立参考链** `scripts/training/tests/ref_npy_dataset.py::RefNpyFrameSampDataset`：直接读源 npy 的 `image_emb_8x8 / pos_emb_8x8 / state_emb`，选帧调 `shared/sampling.py::even_sampling_indices(step, 8)`，补零调冻结副本 `dataset_builder/data_utils.py::right_padding_token_emb`（三处补零都显式沿用输入 dtype，所以交付 image bf16、pos f32、state f32），然后 `reshape(-1, dim)`、`np.repeat(…, 64)`，state 归一化**照抄** `FrameSampDataset._normalize_state` 的 quantile 公式（`(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`，norm stats 是 f64 所以输出 f64），pkl 身份互校同式。motion 四键复用 `datastore/motion_store.py::visible_motion_rows` 与 `MotionStore.rows`，`motion_pos` 取 npy `pos_emb_8x8[0, 0, :256]`，`mem_order` 调 `shared/sampling.py::memory_order(pad_times(frames, 8), 64, mtimes)`。这条链不进生产 loader、不作 fallback。

参考链 vs 候选链（新 8×8 库上的 `FrameSampDataset`）的对拍项：

| 检查 | 方法 | 通过条件 |
|---|---|---|
| 数据身份 | 全部 episode 的 `exec_start_idx`、样本长度、pkl 内 `(epis_idx, step_idx)` 与清单推导一致 | 零差异 |
| 选帧 | 全部合法 step 比较 `even_sampling_indices(step, 8)`；`step < 8` 时返回 `0..step` 共 `step+1` 帧再补零，`step ≥ 8` 时 `np.linspace(0, step, 8)` 取整 | 索引列表逐项相同 |
| 定点样本 | 短样本档 step ∈ {0,1,2,5,6}（补零分支），满长档 {7,8,9}（linspace 分支），每档 200 个，另随机 1000 个；Dataset 输出与 transform 后所有键 raw 字节、dtype、shape、None 状态零差异 | 零差异 |
| 真实 batch | 200 个经真实 `transform_dataset` + `_collate_fn` 的 batch（全短 / 全满 / 混合 / 随机各 50） | 所有键零差异 |
| 抽样顺序 | 同 seed、同 worker 数下真实 loader 消费的 index 序列 | 前 8000 个逐项相同 |
| worker 生命周期 | 40ep 8×8 库上 `spawn_matrix.py` w0/w1/w4/w16 各 2 个 epoch | 跑通，主进程 fd 数回到基线 |
| 拒绝错误输入 | 错 layout、8×8 meta 配 4×4 目录、未 verified、残留 `pack.lock`、错 manifest、错 motion 来源、配置 `token_per_image=16` 却指向 8×8 库 | 全部明确报错，不回退、不截断 |

比较一律先核键集与结构（含 None 键），再核 dtype、shape、raw 字节；**不继承**旧 dtype 重构允许的「4 个 batch raw 失配」，也不用 canonical 口径（升 f32 后哈希）掩盖差异。

#### 顺序与留档

1. 先在 40ep 库建 8×8 开发库，跑通 pack → verify → 上面全部检查。
2. 再建 400ep 正式库，同样全量 verify + 全部检查。
3. tmux 会话名前缀 `p8-`（如 `p8-pack-40`、`p8-verify-400`），本轮起过的会话名记进留档；清理只按名单逐个 `tmux kill-session -t`。
4. 留档 `docs/dataset-build-doc/4task-motion-40ep-framesamp-8x8/` 与 `4task-motion-400ep-framesamp-8x8/`：commit、精确命令、判定行、`store_meta.json` 副本、耗时、体积、存储介质（AWS 本地 NVMe RAID `/dev/md0`）。

### 第二块：训练——让 Dataset 与训练入口接受 8 帧配置，并用 1000 步真实训练证明等价

#### 结论先行

- **模型侧一处不改。** 对 `src/mme_vla_suite/models/` 全目录静态核对：`HistoryPi0Config.inputs_spec` 的形状全部从 `history_config.budget`、`memory_feature.*.input_dim` 推出；`mem_order` 长度是 `budget + motion.budget` 算出的 608，不是字面量；`PerceptualMemory.__call__` 只检查 `static_image_emb.shape[1] == budget`；`FeatureEncoder` 只碰最后一维。没有任何 16、32 或 608 写死。
- **改动集中在四处**：`FrameSampDataset.__init__` 的配置白名单与 store 规格匹配；两份新 yaml；训练入口的 jax 编译缓存落点；对拍量具。
- **验收是四个 profile 各三条轨迹**（参考自重复 A1/A2 + 候选 B），共 12 条 × 1000 次更新，逐步浮点位比对。

#### 配置怎么写

新增两份 yaml，与旧文件逐键相同、只改一个数：

| 文件 | 相对旧文件的差异 |
|---|---|
| `perceptual-framesamp-context-8frame-8x8.yaml` | `token_per_image: 16` → `64`，其余（含 `motion.enabled: false`）照抄 `perceptual-framesamp-context.yaml` |
| `perceptual-framesamp-context-8frame-8x8-motion.yaml` | 同上，照抄 `perceptual-framesamp-context-motion.yaml`（`motion.enabled: true`），另把 `motion.online_gpu: 1` 改为 `4`（GPU 约束兜底；训练不读此键，评估驱动仍须逐片显式给卡号） |

帧数在 yaml 里没有键，是派生量：`FrameSampDataset.__init__` 里 `_max_frames = budget // (token_per_image * num_views)` = 512 // 64 = **8**，`_tokens_per_frame` = 64。`streaming_obs_horizon: 16`、`action_horizon=20`（在 `training/config.py` 的 `_CONFIGS` 条目里，不在 yaml）不动，`train.py::main` 的交叉断言照常通过。motion 子键（`budget: 96`、`stride: 16`、`window_frames: 33`、`pos_dim: 256`、`store_path`）一个不改。

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
- 新增 store 规格匹配：从 `StoreMeta` 查规格表，要求 `spec.tokens_per_frame == token_per_image * num_views`。8×8 配置指向 4×4 库、或反过来，都在构造时报错，不静默取前 16 个 token、不插值。

其余逻辑本来就是通用的，逐项核过：`_pad` 用 `_max_frames` 做目标长度并按输入 dtype 预分配；reshape 是 `img.reshape(-1, img.shape[-1])`；state 与 mask 的 `np.repeat` 用 `_tokens_per_frame`；motion 块的 `pad_times(frames_arr, self._max_frames)` 与 `memory_order(ftimes, self._tokens_per_frame, mtimes)` 都读实例变量。`_normalize_state` 不动。交付形状在两档下相同：`static_image_emb (512,2048) bf16`、`static_pos_emb (512,768) f32`、`static_state_emb (512,8) f64`、`static_mask (512,) bool`；motion 开启态 `motion_emb (96,768) f32`、`motion_pos (96,256) f32`、`motion_mask (96,) bool`、`mem_order (608,) int32`，关闭态四键 None。

`mem_order` 的**值**跨配置会变：排序键是 `帧时刻 × 2 + 类型`，帧路 token 从 32 组 × 16 个变成 8 组 × 64 个，motion 在混合序列里的位置随之移动。这是设计内的变化，只要求「按时刻稳定排序、同刻视觉在 motion 前、padding 落尾、是 0..607 的合法置换」，不要求跨配置相等，也不据此要求跨配置 loss 相等。

`dataloader.py::_create_framesamp_dataset` 与四道闸（`require_no_pack_lock` → `StoreMeta.load` → `require_verified` → `_motion_gates`）不改。`_motion_gates` 的 `check_same_source` 比的是 framesamp meta 与 motion meta 的 `manifest_sha256`，8×8 库与 4×4 库同源同 manifest，天然通过。

#### 训练入口与缓存落点

`scripts/training/train.py::main` 与 `scripts/training/tests/single_step_grad.py::main` 都硬编码 `jax.config.update("jax_compilation_cache_dir", "~/.cache/jax_<exp_name>")`，这一句会覆盖环境变量 `JAX_COMPILATION_CACHE_DIR`。实测 `~/.cache/` 下已经躺着 6 个真实目录（`jax_awsprod40k-b128-motion`、`jax_bench-b1-motion` 等），是此前直接跑 `train.py` 留下的，违反第 13 条环境 B 红线；本轮不清理它们（另立任务），但本轮所有 run 不得再往里写。做法：两处入口改成 `os.environ.get("MMEVLA_JAX_CACHE_DIR") or "~/.cache/jax_<exp_name>"`，设了就用、没设保持旧行为，不改任何数值逻辑。`bench_train_steps.py` 调的是 `train.main`，自动继承。启动时统一 `source scripts/training/paths.sh`（它设 `XDG_CACHE_HOME`、`HF_HOME`、`WANDB_*` 到 `v1-store/cache/`，不动 HOME），再显式 `export UV_CACHE_DIR=/scratch/hongze/.cache/uv MMEVLA_JAX_CACHE_DIR=v1-store/cache/jax/<run_name>`。

训练本身的 CLI 是 tyro 从 `TrainConfig` 生成的：位置参数选条目（`mme_vla_suite` 4 worker / b64，`mme_vla_suite_b128` 8 worker / b128），`--model.history_config=<yaml 文件名>` 按仓库根相对路径解析（cwd 必须是仓库根），`--dataset-path` 指 packed 库根。8 帧训练只需把 `--dataset-path` 指到 `framesamp-8x8/`、yaml 换成新文件名，没有别的开关。

#### 第二块验收：同机真实训练 1000 次更新

**这一节在干什么。** 第一块只证明了「喂进模型的数据一样」，但数据一样不等于训练一样，中间还有 transforms、collate、JAX 交付、前向、反向、优化器一整串。最稳的办法是真的训一遍，把每一步的数字拿出来逐位比。所以这里要跑 12 条正式 run，每条 1000 次参数更新，两两比对。

**总共跑几组。** 四个配置，每个配置三条，共 12 条：

| 配置 | motion | run 1（A1，参考） | run 2（A2，参考重跑） | run 3（B，候选） | 跑在哪对卡 |
|---|---|---|---|---|---|
| C32 | 关 | 旧代码 + 旧 4×4 库 | 同 A1 再跑一遍 | 新代码 + 旧 4×4 库 | GPU 4,5（第一波） |
| M32 | 开 | 旧代码 + 旧 4×4 库 | 同 A1 再跑一遍 | 新代码 + 旧 4×4 库 | GPU 6,7（第一波） |
| C8 | 关 | 新代码 + 直读源 npy 的验证用 Dataset | 同 A1 再跑一遍 | 新代码 + 新 8×8 库 | GPU 4,5（第二波） |
| M8 | 开 | 新代码 + 直读源 npy 的验证用 Dataset | 同 A1 再跑一遍 | 新代码 + 新 8×8 库 | GPU 6,7（第二波） |

「旧代码」是阶段 1 结束时冻结的参考 commit `REF`（量具、yaml、缓存落点已加，但库格式与 Dataset 还没改）；「新代码」是阶段 2 结束的候选 commit `CAND`。每条都是 1000 步、batch 8、seed 42、2 卡。

**比法。** 每一行里先比 run 1 和 run 2：这两条代码、数据、seed 全一样，如果它们自己都对不上，说明管线有随机抖动，后面的比较没有意义。run 1 == run 2 成立后，再比 run 1 和 run 3，这才是「改动没改数」的证据。行与行之间不比：32 帧和 8 帧本来就该不一样，motion 开和关也不该一样。

**「直读源 npy 的验证用 Dataset」和「新 8×8 库」是什么、有什么区别。** 数据是同一份，读法不同：

- **新 8×8 库**是第一块打出来的三张大表 `framesamp-8x8/`，训练时 `FrameSampDataset` 按行号从大表里读 8 帧。这是生产链路，以后正式训练走的就是它。它是本轮新写的代码加新打的库，两样都是待验证的东西。
- **直读源 npy 的验证用 Dataset**（`scripts/training/tests/ref_npy_dataset.py::RefNpyFrameSampDataset`）不经过大表，直接打开源库里每帧的 `token_emb_{t}.npy`，把 8 帧的 `image_emb_8x8 / pos_emb_8x8 / state_emb` 拼起来。它绕开了「打包」和「新 Dataset」这两个本轮改动的环节，只用改动前就存在的函数：选帧 `even_sampling_indices`、补零 `right_padding_token_emb`、同一条归一化公式。所以它是正确答案，慢、不进生产、只为对拍存在。

打个比方：新库是把原始账本誊抄成表格，验证用 Dataset 是直接翻原始账本。两边训出来逐位一样，才能说誊抄没抄错。32 帧配置不需要这个，因为改动前的代码跑旧库本身就是正确答案。

参考侧的 8 帧 Dataset 通过 `bench_train_steps.py` 的测试专用开关 `BENCH_DATASET_IMPL=refnpy` 注入（bench 已经在 monkeypatch `train.wandb`、`_checkpoints.save_state`、`init_train_state`、`TorchDataLoader.__iter__`，再替换 `dataloader._create_framesamp_dataset` 是同一类只读注入），生产 loader 不加任何 fallback。

**每一步比什么。** 训练用现成的 `bench_train_steps.py` 起，它不改训练逻辑，只在旁边记录四样东西，两条轨迹逐位比：

1. 每一步的 `loss`、`grad_norm`、`llm_grad_norm`、`mem_enc_norm`、`param_norm` 五个标量，记的是浮点数的 `float.hex()`，不是四位小数。
2. 在第 0、1、2、25、50、100、200、400、600、800、1000 次更新时，把完整 TrainState（params + AdamW 动量 + EMA + step）逐叶 `sha256(dtype‖shape‖bytes)`。「第 k 次更新」按更新次数 `state_step` 定义：0 是 `init_train_state` 之后的初态，正整数 k 对应循环步 `loop_step = k-1` 更新之后。现有 bench 的 `param_checksums.step=0` 是初态、其他标签 s 是第 s 次循环后（已更新 s+1 次），两份记录的 step 字段不是同一时刻，量具要显式记 `phase` 与 `state_step`。
3. 每个记录步喂进模型的那个 batch（collate 后 host 侧 numpy）逐键 raw sha，外加它用了哪些样本编号 `sample_indices`。
4. 全程主进程抽取的 index 序列。

任何一跳改了一个字节，都会在标量或哈希里露出来。判据一律逐位：

- `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`，并另核 `scalars_hex.tsv` 表头恰为六列（现有比较器对缺键静默 `continue` 仍打 `keys=5`，必须人工补位）。
- `STATE_DIGEST` 在上面 11 个摘要点上 `mismatch=0`。
- `BATCH_DIGEST` raw 口径在记录步集合上 `mismatch=0`，`sample_indices` 逐位相同，键集完整（12 键，motion 关闭态四键记为 None 而不是缺键）。
- `INDEX_SEQ` 前 8000 个 index 逐项相同，且两侧 `n ≥ 8000`（现有比较器只比最短公共前缀，必须人工补位）。
- 参数叶数 `n_leaves` 要求 A1 = A2 = B，不再写死 177 / 193（那是下游 gate 的历史常量，模型没变所以数值应该不变，但判据不靠猜）。
- 基线自己不重复（A1 ≠ A2）就不能放宽阈值宣称等价。

**为什么是 1000 步、batch 8、2 卡。** 1000 步是用户定死的口径，100 步短测只用来排错和估时。batch 8 与 2 卡沿用 2026-09-04 `aws-t2-ref-s100` / `aws-t2-cand-s100` 的档位：小到四组能并行，大到覆盖 FSDP 分片（`--fsdp-devices 2`）。1000 × 8 = 8000 个样本不到 400ep 库一个 epoch（101,066），避开 epoch 边界的抽样分叉。其余启动参数：seed 42、4 worker、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`WANDB_MODE=disabled`。这些是本次启动覆盖参数，不改全局默认值，起跑前确认。

**为什么要确定性 flag。** `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'` 关掉 GPU 上的非确定性归约与自动调优，同样的输入两次跑出来才逐位相同。生产训练默认档做不到这一点，所以对拍必须开。

**卡怎么排、要多久。** 同一个配置的三条必须在同一对卡上顺序跑，因为不同卡对可能有极小的数值差异。只允许用 GPU 4–7，即两对卡：第一波 C32 在 4,5、M32 在 6,7 同时跑；第一波两组都跑完后第二波 C8 在 4,5、M8 在 6,7。每对卡上顺序跑 6 条，总墙钟约等于 6 条 1000 步轨迹的时间。单条耗时以先跑的 100 步排错 run 外推；环境 A 2×Ada 上 1000 步约 2.5 h 只作量级参考，不混比。正式跑之前每个配置先跑一次 100 步短版本（run_name 加 `-s100`）排错、估时，跑完清理，不算数。

**环境指纹。** 起跑前用 `scripts/training/g0/check_baseline_env.py` 的指纹做 preflight，硬一致项包括 `uv.lock` sha、jax 0.5.3 / jaxlib 0.5.3 / flax 0.10.2 / optax 0.2.4 / numpy 1.26.4 / ml_dtypes 0.4.1 / torch 2.7.1、GPU 型号与驱动、`XLA_FLAGS`、x64 与 matmul precision、norm_stats sha（400ep 交付件 `v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`，sha `750a8e9b…`）、tokenizer、pi05 底座权重抽样、源库抽样、**本库 manifest sha 与 motion store meta sha**。现有指纹只看根 `v1-store/episode_manifest.json`（环境 B 不存在，恒记 None），norm_stats 路径写死 `robomme/` 而非 `robomme-400ep/`，这两处要改。允许差异只有 `--exp-name`、`--checkpoint-base-dir`、C8/M8 的 `--dataset-path`、参考/候选 commit、`BENCH_DATASET_IMPL` 与 `BENCH_RECORD_DIR`，逐字段白名单。

**补充单步全梯度对拍。** 全短、全满、混合三种真实 batch 各一个，用 `single_step_grad.py` 记完整梯度树逐叶 sha，定位可能被梯度范数掩盖的差异；同配置两侧逐位相等。

#### 交付判定与边界

- 只有第一块与第二块**全部**通过，才宣称对应 profile 的训练交付等价；第二块未通过只能报告已通过的输入检查。
- 交付报告分开写四件事：旧能力回归（C32/M32）、新 8 帧训练支持（C8/M8）、motion 接口保护、实际资源表现（吞吐与显存在数值验收之外独立报告，用 `nvidia-smi -lms 500` 的均值 / 0% 占比 / 分层均值，不以中位数下结论，不承诺性能收益）。
- 在线推理放在第三块：训练验收通过不等于 8×8 能部署，第三块单独验。
- motion 只做回归保护：不重抽 Wan、不重训 MotionJEPA、不做预算或结构消融。

### 第三块：推理——让在线记忆构造认得 8×8，并证明推理侧与训练侧交付一致

#### 结论先行

- **模型侧、motion sidecar、IPC 协议、评估驱动脚本都不用改。** 评估侧没有 yaml 文件名白名单，`policies/policy_config.py::create_trained_policy` 只按 run 根的 `history_config.resolved.yaml` + `.sha256` + `motion_provenance.json` 建策略；`train.py::init_history_config` 会把这三样写进 run 根，8×8 的 run 自动被认。
- **要改的只有 `policies/framesamp_memory.py::FrameSampMemory` 一个文件的三处字面量**，外加 5 个只读引用它的工具脚本。
- **没有训练收敛的 8×8 模型。** checkpoint 只能用第二块候选 B run（`t8-c8-b`、`t8-m8-b`）1000 步后的 EMA 参数（bench 的 `BENCH_SAVE_FINAL_CKPT=1` 存到目录 `999`）。这样的模型不会完成任务，闭环成功率**不作任何指标**；第三块交付的是「在线链路跑通」与「推理侧 obs 与训练侧 obs 逐字节一致」两件事。这与 2026-09-04 `aws-t3-eval-obs` 用 100 步 ckpt 跑 40 集、0% 成功、只证链路的做法同口径。
- **在线 4×4 不能被改坏。** `FrameSampMemory` 改成网格驱动后，先用现成的真模型（`awsprod40k-b128-motion/39999`）证明 4×4 在线行为逐位不变，再验 8×8。

#### 现在的推理测试怎么做

闭环评估是三个进程共一张卡、八张卡并行：

- **policy server**：`uv run scripts/training/serve_policy.py policy:checkpoint --policy.dir=<ckpt>`。视觉记忆由 `FrameSampMemory` 在线造：每步当前帧过 SigLIP（`HistoryPi0.vision_encode`，bf16 权重，输出 256 个 patch），`pool_tokens_to_size(output_emb, 16)` 池成 4×4，存进缓冲键 `image_emb_4x4`；位置表 `pos_embedder(ranges, 4)` 只算 4×4 一档，`(4096,16,768)` f32 = 192 MiB。推理时 `even_sampling_indices(step, 32)` 选帧，交付与训练同形的四键；motion 开启时 `_prepare_motion` 取 `pos_emb_4x4[f, 0, :256]` 当时间码，`mem_order` 与训练侧同一个 `memory_order`。
- **motion sidecar**：`policies/motion_client.py::MotionEncoderClient` 在策略构造时 `subprocess.Popen` 起常驻子进程 `scripts/dataset/wan/motion_sidecar.py`（跑在 `v1-store/venvs/wan`），Unix socketpair 通信，协议 `MMEMOT01`：父发 33 帧 256×256×3 uint8（6,488,064 B）加起点帧号，子回 768 维 f32（3,072 B），单窗约 1.4 s。握手时逐键比对 provenance 与离线 motion store，不等即 raise。
- **仿真客户端**：`examples/robomme/eval.py` 在 `/scratch/hongze/micromamba/envs/robomme/bin/python` 下跑，带 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`。
- **驱动脚本**：`scripts/training/legacy-eval/eval_shard.{local,remote}.sh`（`.local` 八卡固定映射、`.remote` 少卡取模，用前要把副本拷回主线并 commit），或 `scripts/motion-variance/eval_shard_mv.sh`（自带 robomme 副本，无需拷回，最近一次正式评估 `mv-matrix-40k` 用的就是它）。

上一次 40k motion 模型的正式评估（`docs/training-doc/eval-awsprod40k-b128-motion/`）：200 集 38 min，成功率 28%，policy 单次 `infer` 69 ms，每卡显存约 37 GB（policy 32.9 + sidecar 3.3 + 仿真 0.8）。

#### 改什么

`FrameSampMemory` 里与 4×4 绑死的三处，改成从 `history_config.token_per_image` 推网格边长 `g = int(sqrt(token_per_image))`（要求 g² == token_per_image，否则构造时报错）：

| 位置 | 现在 | 改成 |
|---|---|---|
| `__init__` | `self.pos_emb_4x4 = np.array(pos_embedder(ranges, 4))`，192 MiB | `self.pos_emb = np.array(pos_embedder(ranges, g))`；g=8 时 `(4096,64,768)` f32 = 768 MiB host 内存，8 个 policy 进程共 6 GiB |
| `add_buffer` | `pool_tokens_to_size(output_emb, 16)` | `pool_tokens_to_size(output_emb, token_per_image)`；g=8 时 `pool_size = sqrt(256 // 64) = 2`，即 2×2 平均池化，与建库侧 `mem_buffer.py` 池 8×8 的算式同一个函数 |
| `add_buffer` 写缓冲 | 键名写死 `"image_emb_4x4"` / `"pos_emb_4x4"` | 键名 `f"image_emb_{g}x{g}"` / `f"pos_emb_{g}x{g}"`，与 `_prepare_frame_sampling` 已有的 `spatial_key` 推导对上 |
| `_prepare_motion` | `self.pos_emb_4x4[f, 0, :256]` | `self.pos_emb[f, 0, :256]`；取值不变，因为 `PosEmb3D` 的时间段跨 patch 复制，patch 0 前 256 维与网格无关（第一块 `MOTION_POS_XGRID` 验的就是这一点） |
| 类 docstring 与行内注释 | 「只算 4x4 一档 … 192 MiB」；`# (t, v, 64, 2048)` | 论证按网格改写；那条注释本身就是错的（SigLIP 输出 256 patch，不是 64），顺手纠正 |

`policy.py::_prepare_history` 不用改：`max_frames = budget // (token_per_image * num_views)` 自动变 8，`memory_order(frame_times, 64, motion_times)` 仍是 608 位置换。

引用 `pos_emb_4x4` 或 `image_emb_4x4` 的 5 个工具要认新属性名与键名：`scripts/training/g0/serve_policy_probe.py`、`compare_online_memory.py`、`compare_train_infer_obs.py`、`compare_siglip_replay.py`、`scripts/training/tests/motion_gates_online.py`（后者还写死 `(512, 16, 1)` 与 `MAX_FRAMES=32`，改成按 yaml 推）。`scripts/motion-variance/mv_common.py` 的 `608/512/96` 在 8×8 下数值不变，只改注释口径。

#### 第三块验收

分三步，前一步不过不进下一步。

**第一步：4×4 在线回归（用真模型，不需要 8×8 checkpoint，代码改完即可跑）。** 目标是证明网格驱动的 `FrameSampMemory` 在 `token_per_image=16` 下与改前逐位相同。

- 三方对拍：同一个 jitted SigLIP、同一批真实帧，分别喂改前（`REF` worktree）与改后的 `FrameSampMemory`，`get_history_feats` 全键 `leaf_sha256` 逐位相同；判定行 `ONLINE_MEM_REGRESS=PASS frames=<n> keys=3 mismatches=0`。
- 在 `awsprod40k-b128-motion/39999` 上重跑 `scripts/training/g0/compare_train_infer_obs.py` 的 S 臂（帧特征灌训练库真值行），13 条阻断判据与 `docs/training-doc/tic-obs-model-40k/result.md` 逐条同值（12 PASS + `VT_FULL_VS_CACHED` 维持已知 FAIL，其阈值待用户裁决，本轮不动）。
- 用 `scripts/training/g0/compare_online_motion.py` 重跑 40ep 库 772 窗：`ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0`，`ONLINE_START_SET / ONLINE_POS / ONLINE_ORDER / PROVENANCE` 全 PASS。

**第二步：8×8 推理侧 obs 与训练侧 obs 对拍（用 `t8-c8-b` / `t8-m8-b` 的 999 目录）。** 把 `compare_train_infer_obs.py` 的七关在 8×8 上跑一遍，`--dataset` 指 `framesamp-8x8/`，S 臂读 8×8 库真值行：

| 关 | 比什么 | 8×8 下的要求 |
|---|---|---|
| 0 配置与库同源 | norm stats、参数树叶、四张表指纹 vs run 记录 | 全同；`n_leaves` 与第二块 B run 一致 |
| 1 输入键 | 在线 `_prepare_history` 八键 vs `FrameSampDataset` 八键；sidecar 现算 motion vs 表 | S 臂逐字节相同。这一关直接检验在线 2×2 池化出的 64 token 与建库时池出的那份是否同一份 |
| 2 预处理后 | 两侧过完变换链的全部键 | 逐字节相同 |
| 3 模型内部 | 1184 前缀 token、attn mask、positions、18 层 K/V、`mem_order` 可逆 | 逐位相同 |
| 4 整段 vs 缓存 | `compute_loss` 整段前向 vs 前缀缓存单步 | 结构三项全等；数值项沿用现行 `THR_REL_FRO=1e-3 / THR_ULP_P99=4`，超阈值按现行口径记 FAIL 并标注「与 4×4 同一已知来源」，不现场放宽 |
| 5 最终动作 | 两侧 obs 经同一 `_sample_actions` 去噪 | 逐位相同 |
| 6 真仿真 | 在线不变量：窗数=公式值、`mem_order` 合法排列、prompt token 属训练集、pos 表 sha 同库表、一一对应、无抛错 | 全过；pos 表 sha 对的是 8×8 库的 `pos_emb_8x8.f32.bin` |

B 臂（在线 bf16 SigLIP vs 训练 f32 表）照旧只作观察项记 `rel_fro`，不作阻断；4×4 下实测 0.33%。

**第三步：闭环冒烟（证明链路跑通，不看成功率）。** 每个 profile（C8、M8）跑 4 任务 × 10 集 = 40 集，4 片 4 卡（GPU 4–7 每任务一片一卡），M8 带 sidecar，sidecar 与其 policy 同卡。判定只有四条：`errors=0`、每集推理次数 ≤ 82（评估节奏上限）、`mem_order` 逐步合法、M8 的 sidecar 窗数与公式值一致；成功率照实记但明确标「1000 步测试模型，无意义」。显存与 host 内存（pos 表 768 MiB/进程）在结果里报。

驱动用 `scripts/motion-variance/run_batch_mv.sh`（`COND=normal WORKERS=4 GPU_LIST=4,5,6,7 EP_COUNT=10`），它自带 robomme 副本、不必拷回 legacy-eval 副本，且 `eval_shard_mv.sh` 会把片的卡号经 `--motion-gpu` 传给 `serve_policy_mv.py`，sidecar 不会落到 yaml 快照写死的 GPU 1。若其对 40k 快照有硬假设跑不通，备选只能是 `legacy-eval/eval_shard.remote.sh`（支持 `GPU_LIST` 取模映射并设 `MMEVLA_MOTION_ONLINE_GPU`，须按其 README 拷回副本、commit 后起跑）；`.local` 那套固定 8 卡映射会碰 GPU 0–3，禁用。

#### 顺序、耗时与边界

- 第一步在阶段 2 代码改完后立刻做（不依赖第二块），是第三块的准入。第二、三步排在第二块之后，因为要等 `t8-c8-b` / `t8-m8-b` 跑完。
- 第二步的 `compare_train_infer_obs.py` 主进程 jax 必须在 GPU 上（CPU 上 PosEmb3D 表不逐位），用 GPU 7 一张卡，motion 侧 `--motion-gpu 7`；第三步 40 集按 4 卡并行约 10–20 min 量级（8 卡 200 集 38 min 外推）。
- 边界：第三块证明的是「8×8 在线链路可跑、推理侧与训练侧交付一致」，**不证明** 8×8 模型能完成任务；那需要一次正式训练，另立 run_name 另行授权。

### 执行阶段与 commit 节奏

按时间顺序 8 个阶段。规则三条：每个阶段结束一个功能 commit（`commitV<大>.<小>`）或文档 commit（`docs:`），commit 后立即 `git push`；所有正式 run 从 clean HEAD 起；**阶段 5 的 12 条轨迹期间主树冻结不 commit**（A 侧从 `REF` worktree 跑、B 侧从 `CAND` 主树跑，中途任何 commit 都会让后起的 B 与先起的 B 不同 HEAD），留档在 12 条全部跑完后一次提交。

| 阶段 | 做什么 | GPU | 产出 | commit |
|---|---|---|---|---|
| 0 preflight | 只读核对源库、旧库 meta、motion store、norm_stats、权重、tokenizer、磁盘、环境变量干净 | 无 | 核对记录写进阶段 1 的 launch.md | 无 |
| 1 量具与参考 | 改 `train.py` / `single_step_grad.py` 缓存落点、`bench_train_steps.py` 开关与白名单、`check_baseline_env.py` 指纹、`_common.py` / `dump_fixture_samples.py` 配额与路径、新建 `gate_8x8.py` / `compare_fixture_dumps.py` / `ref_npy_dataset.py` / `xgrid_pos_check.py` / `online_mem_regress.py`、两份 8×8 yaml；量具自检；C32 旧库 100 步冒烟与 `aws-t2-ref-s100` 的 scalars sha 比对 | 冒烟用 GPU 4,5 | 参考侧全部工具 | `commitV<x>.1`，push 后 `git rev-parse HEAD` 记为 **`REF`**，随后 `git worktree add --detach v1-store/worktrees/ref-8x8 $REF` |
| 2 两档格式与 Dataset | `framesamp_store.py` 规格表、`pack_framesamp_store.py --layout`、`framesamp_dataset.py` 白名单与 spec 匹配、`test_pack_guards.py` 两档用例；`FrameSampMemory` 网格驱动 + 5 个工具 + `motion_gates_online.py` | pytest 走 `JAX_PLATFORMS=cpu`；第三块第一步的 4×4 在线回归用 GPU 7 | 候选侧代码 | 两个 commit：`commitV<x>.2`（库格式 + Dataset）、`commitV<x>.3`（在线记忆 + 工具）；阶段末 HEAD 记为 **`CAND`** |
| 3 建库 | 40ep 先：pack → verify → report → `xgrid_pos_check`；再 400ep 同四步 | 无（纯 CPU，`--procs 32`） | `framesamp-8x8/` 两份，`VERIFY_PACK=PASS` 两条，`MOTION_POS_XGRID=PASS` 两条 | `docs:` 建库留档 `docs/dataset-build-doc/4task-motion-{40ep,400ep}-framesamp-8x8/` |
| 4 第一块验收 | 四个 profile 的 fixture dump（refnpy 与 packed）与比较、`spawn_matrix`、错配拒绝用例 | dump 走 CPU；超 5 分钟项进 tmux（前缀 `p8-fx-`） | `SAMPLE_RAW_EXACT` / `BATCH_RAW_EXACT` / `MATRIX` / `REJECT_INPUTS` 判定行 | `docs:` 留档 `docs/training-doc/t8-fixture/` |
| 5 第二块 | 先 4 条 100 步排错 run（每对卡一次跑两个 profile 的 `-s100`，估时后清理）；再两波 12 条正式轨迹；每 profile 跑完立即 `gate_8x8` | 第一波 C32 在 4,5、M32 在 6,7；第二波 C8 在 4,5、M8 在 6,7 | 12 份 records、4 条 `GATE_8X8=PASS` | 期间**不 commit**；全部跑完后一次 `docs:` 留档 `docs/training-doc/t8-<profile>-{a1,a2,b}/` |
| 6 第三块 | 第一步（4×4 在线回归）已在阶段 2 末做完；第二步 `compare_train_infer_obs.py` 在 `t8-{c8,m8}-b/999` 上跑七关；第三步 40 集闭环冒烟 | 第二步 GPU 7；第三步 4 片 GPU 4–7 | 七关判定行两套、`EVAL_SMOKE=DONE` 两条 | `docs:` 留档 `docs/training-doc/t8-infer-regress-4x4/`、`t8-infer-{c8,m8}/` |
| 7 收官 | `docs/dataloader-restructure.md` 加「8 帧 × 8×8 档」节、`docs/train-infer-consistency.md` 加「8×8 档（测试模型）」节、按实测更新前后链路图；本文降为过程档案；删除 `ref-8x8` worktree 与 `-s100` 临时 run | 无 | 正本更新 | `docs:` 收官 |

阶段 1 与 2 之间、2 与 3 之间可以穿插文档 commit（不影响 `REF` / `CAND` 的定义，两者各按阶段末 HEAD 记）；阶段 5 内不行。任何阶段的判定行不过即停在该阶段，把原始输出交用户，不改判据、不进下一阶段。

---

## 第二部分（技术细节，供 agent 追踪）

### 1. 拟修改文件与函数

| 文件 / 锚点 | 改法 | 不改什么 |
|---|---|---|
| `src/mme_vla_suite/datastore/framesamp_store.py` 顶部常量段 | 新增 `@dataclass(frozen=True) class StoreSpec`（字段：`layout, grid, tokens_per_frame, image_key, pos_key, image_row_shape, pos_row_shape, image_row_bytes, pos_row_bytes, image_part_dir, pos_table_relpath, source_image_offset, source_pos_offset`）与 `SPECS: dict[str, StoreSpec]` 两档；旧常量保留为 4×4 档的别名以免破坏现有 import | `STATE_*`、`ROW_DIGEST_*`、`SOURCE_NPY_SIZE=602951`、`SOURCE_STATE_OFFSET=602906`、`TARGET_PARTS=32`、`row_of`、`build_exec_lookup`、锁与 meta 路径常量 |
| 同文件 `StoreMeta` | 新增字段 `spec: StoreSpec`；`load` 先取 `raw["layout"]`，不在 `SPECS` 即 `ValueError("layout 不符")`，再按 `spec` 校验 `tables` 三键的 `row_shape/dtype/row_bytes` | 其余校验项（parts 连续、bytes 账、status、manifest_scope）原样 |
| 同文件 `run_fast_checks` / `FrameSampStore.__init__ / read_image_rows / pos_rows` | 小表大小账、pos 表 reshape、image 行 reshape、`IMAGE_ROW_BYTES` 乘法全部改读 `meta.spec` | preadv 读法、`_runs_of`、fadvise、`__reduce__` 禁 pickle |
| `scripts/dataset/pack_framesamp_store.py` | `pack` 加 `--layout {framesamp-4x4-v1,framesamp-8x8-v1}`（默认 4×4）；`read_frame_decode(path, spec)` 形制守卫改 `img.shape == (1,)+spec.image_row_shape`；`read_frame_slice` 与 `_pin_slice_prefix` 按 `spec.source_*_offset` 取窗（8×8 偏移 387 / 344,682 / 602,906 由 2026-09-13 实测 `raw.find` 得到，写入规格表但首跑仍 decode）；`cmd_pack` 的 `floor = max(40*10**9, int(1.25*need))`；meta 的 `layout`、`tables`、`packer.layout` 按 spec 写；`verify` 从 meta 取 spec | `plan_parts`、锁协议、`atomic_write_bytes`、写侧三重校验、resume 逻辑、`_spot_entries` |
| `src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset.__init__` | `_req((budget, tpi, nv) in {(512,16,1),(512,64,1)})`；`_req(self._meta.spec.tokens_per_frame == tpi*nv)` | `__getitem__`、`_pad`、`_normalize_state`、motion 块、`_NONE_KEYS`、pkl 互校 |
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context-8frame-8x8.yaml` 与 `-motion.yaml` | 新建，见第一部分 | 两份旧 yaml |
| `scripts/training/train.py::main`、`scripts/training/tests/single_step_grad.py::main` | `jax_compilation_cache_dir` 取 `os.environ.get("MMEVLA_JAX_CACHE_DIR") or 旧值` | 其余一行不动（`g0_gate` 的 `inspect.getsource` 护栏要求 `train.main` 仍含 `wandb.log(reduced_info`、`_checkpoints.save_state(`、`init_train_state(`、`create_data_loader(` 四个片段） |
| `scripts/training/tests/ref_npy_dataset.py`（新建） | `RefNpyFrameSampDataset(source_root, manifest_path, data_config, history_config, action_horizon, motion_root)`：装配逻辑见第一部分第一块「第二层」；接口与 `FrameSampDataset` 同（`__len__`、`__getitem__` 返回同键集） | 不进 `src/`，不被生产代码 import |
| `scripts/training/g0/bench_train_steps.py` | `_EXPECTED_HISTORY_CONFIGS` 加两个新文件名；新增 env `BENCH_DATASET_IMPL={packed,refnpy}`（默认 packed）、`BENCH_REF_SOURCE`、`BENCH_REF_MANIFEST`、`BENCH_REF_MOTION`，refnpy 时 monkeypatch `mme_vla_suite.training.dataloader._create_framesamp_dataset` 返回参考 Dataset；新增 env `BENCH_CHECKSUM_STATE_STEPS`（显式更新次数列表，如 `0,1,2,25,50,100,200,400,600,800,1000`），摘要记录写 `phase ∈ {init, post_update}`、`loop_step`、`state_step` 三字段；`batch_digests` 对 None 键显式记 `null` 而非省略 | 哈希口径 `_leaf_sha256` / `_canonical_sha256`、`_WandbProxy` 行 schema、`index_sequence` 记法、`_MAX_BENCH_STEPS=1200` |
| `scripts/training/tests/gate_8x8.py`（新建） | 参考 `g0_gate.py::_gate_t2` 八步重写为配置驱动：`--profile {c32,m32,c8,m8}`、`--run-a1 --run-a2 --run-b --log-*`、`--steps 1000 --batch-size 8`；argv 白名单 `{--exp-name, --checkpoint-base-dir}` ∪（c8/m8 时）`{--dataset-path}`；env 白名单 `{BENCH_DATASET_IMPL, BENCH_REF_*, BENCH_RECORD_DIR, MMEVLA_JAX_CACHE_DIR}`；`n_leaves`、`n_keys` 要求三侧相等且键集完整；摘要步按 `state_step` 对齐；表头六列与 `n ≥ steps×batch` 内建为判据；成功行 `GATE_8X8=PASS profile=<p> steps=1000 batch=8 state_steps=[…] digest_steps=[…]` | `g0_gate.py` 的 T1/T2 常量（`_EXPECT_RAW_MISMATCH=4` 等）一字不动，服务历史验证 |
| `scripts/training/g0/check_baseline_env.py::collect_fingerprint` | `assets.norm_stats_sha256` 改为 CLI `--norm-stats` 显式路径；新增 `dataset.store_meta_sha256`（`<dataset>/meta/store_meta.json`）、`dataset.manifest_sha256`（从 store meta 读 `manifest_sha256`）、`motion.store_meta_sha256`（`MMEVLA_MOTION_STORE` 或 yaml `store_path` 下的 `meta/store_meta.json`）；`dataset_spot` 改对 `source_dataset_root` 抽样 | `uv_lock_sha256`、`packages`、`gpu`、`xla`、`jax_config`、`_diff` 深比较、`BASELINE_ENV=PASS/FAIL` 判定行 |
| `scripts/training/tests/_common.py` | `SHORT_STEPS / FULL_STEPS` 改为 `fixture_steps(max_frames)` 返回 `((0,1,2,mf-3,mf-2), (mf-1,mf,mf+1))`（mf=32 得 `(0,1,2,29,30)/(31,32,33)` 与现值一致，mf=8 得 `(0,1,2,5,6)/(7,8,9)`）；`PER_STEP` 改为按库内 `exec_start_idx==0` 的 episode 数取 `min(200, 该数)`（400ep 为 200，40ep 为 20） | `MEMORY_KEYS`、`leaf_sha256`、`canonical_sha256`、位型容器格式 |
| `scripts/training/tests/dump_fixture_samples.py` | `manifest_path` 从硬编码根 `v1-store/episode_manifest.json` 改为 env `DTYPE_MANIFEST`（与 `single_step_grad.py` 同名）或 store meta 的 `manifest_path`；白名单加两个新 yaml；新增 env `DTYPE_DUMP_IMPL={packed,refnpy}` | 两层取证结构、raw+canonical 双口径记录 |
| `scripts/training/tests/compare_fixture_dumps.py`（新建） | 严格零差异比较器：先比键集（含 None）、再 dtype/shape/raw sha；判定行 `SAMPLE_RAW_EXACT=PASS samples=<n> mismatches=0` 与 `BATCH_RAW_EXACT=PASS batches=200 mismatches=0` | `compare_dtype_fix.py` 不改 |
| `scripts/training/tests/test_pack_guards.py` | `REF_SHARD` / `MANIFEST` 改为 env `MMEVLA_TEST_SOURCE` / `MMEVLA_TEST_MANIFEST`（指向 40ep 库 `source/` 与 `meta/episode_manifest.json`），session fixture 对两档 layout 各打一份前缀 [0..2] 迷你库；新增用例：8×8 meta 配 4×4 目录名、`token_per_image=64` 配 4×4 库、`=16` 配 8×8 库、未知 layout 均 raise | 既有 G1–G14 用例逻辑 |
| `scripts/training/tests/spawn_matrix.py`、`dump_index_seq.py` | 不改代码；前者 `--store/--source/--manifest` 已是必填；后者默认 `4task-gl` 路径在环境 B 不存在，调用时显式传参 | |
| `scripts/dataset/xgrid_pos_check.py`（新建） | 读两个库的 pos 表（`np.fromfile` 后按各自 spec reshape），对 t ∈ [0,586) 比 `[t,0,:256].tobytes()`；另对源 npy 抽 64 帧比 `pos_emb_4x4[0,0,:256]` 与 `pos_emb_8x8[0,0,:256]`；判定行 `MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0` | |
| `src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory` | `__init__` 从 `history_config.token_per_image` 推 `g`（`g*g == token_per_image` 否则 raise），`self.pos_emb = pos_embedder(ranges, g)`；`add_buffer` 池到 `token_per_image`、缓冲键 `image_emb_{g}x{g}` / `pos_emb_{g}x{g}`；`_prepare_motion` 改读 `self.pos_emb`；docstring 与 `# (t, v, 64, 2048)` 注释改正 | `_prepare_frame_sampling`、`get_frame_sampling_indices`、`_encode_ready_windows`、`visible_motion_frames`、motion 客户端调用 |
| `scripts/training/g0/serve_policy_probe.py`、`compare_online_memory.py`、`compare_train_infer_obs.py`、`compare_siglip_replay.py` | `pos_emb_4x4` → `pos_emb`；`"image_emb_4x4"` / `ENC_KEYS` 按 policy 的 `history_config.token_per_image` 推键名；`compare_train_infer_obs.py` 的 S 臂按 store spec 读行 | 七关判据、`THR_REL_FRO=1e-3`、`THR_ULP_P99=4.0` 不动 |
| `scripts/training/tests/motion_gates_online.py` | `HIST_BUDGET, TOKEN_PER_IMAGE, NUM_VIEWS` 与 `MAX_FRAMES` 改为从 `--yaml` 读；`mem.pos_emb_4x4[…]` → `mem.pos_emb[…]` | 闸门逻辑 |
| `scripts/motion-variance/mv_common.py` | 只改 `MEM_LEN, FRAME_SLOTS, MOTION_SLOTS = 608, 512, 96` 旁的注释口径 | 数值不变 |
| `scripts/training/tests/online_mem_regress.py`（新建） | 三方对拍：`REF` worktree 的 `FrameSampMemory` vs 主树的，同 jitted SigLIP、同帧，`get_history_feats` 全键 `leaf_sha256`；判定行 `ONLINE_MEM_REGRESS=PASS frames=<n> keys=3 mismatches=0` | |

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

每步的 Python 一律 `uv run --no-sync`，cwd 仓库根；超过 5 分钟的放 detached tmux（`PYTHONUNBUFFERED=1`、`set -o pipefail`、`tee`、尾行 `EXIT_CODE=`）并挂 Monitor（`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "PASS|FAIL|DONE|EXIT_CODE=|Error|Traceback"`）。

**阶段 0：preflight（只读）**

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA && git status --short && git rev-parse HEAD
for L in 4task-motion-40ep 4task-motion-400ep; do
  jq '{layout,status,num_rows,num_pos_rows,manifest_sha256}' v1-store/datasets/$L/framesamp/meta/store_meta.json
  jq '{layout,status,num_rows}' v1-store/datasets/$L/motion/meta/store_meta.json
  ls -ld v1-store/datasets/$L/framesamp-8x8 2>&1   # 期望：不存在
done
ls -l v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json v1-store/models/big_vision/paligemma_tokenizer.model
ls v1-store/models/openpi-assets/checkpoints/pi05_base/params | head -3
env | grep -E "MMEVLA_FRAMESAMP|MMEVLA_MOTION" ; echo "(以上应为空)"
df -h /scratch | tail -1
```

**阶段 1：量具、参考链、缓存落点、yaml（一个或几个 commit，冻结为参考 commit `REF`）**

改第 1 节表里的 `train.py`、`single_step_grad.py`、`bench_train_steps.py`、`gate_8x8.py`、`check_baseline_env.py`、`_common.py`、`dump_fixture_samples.py`、`compare_fixture_dumps.py`、`ref_npy_dataset.py`、`xgrid_pos_check.py`、两份 yaml。**不碰** `framesamp_store.py`、`pack_framesamp_store.py`、`framesamp_dataset.py`。验证：

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest scripts/training/tests/test_padding_dtype.py -x -q   # 哈希口径未变
# 量具自检：故意缺键 / 缺步 / 重复步 / 扰动一个字节，gate_8x8 必须 FAIL
uv run --no-sync python scripts/training/tests/gate_8x8.py --self-test
# 旧链回归 100 步冒烟（C32，旧库、旧 yaml），确认 bench 改动不改数：与 aws-t2-ref-s100 的 scalars sha 85b8fe37… 比对
```

阶段 1 结束后 `git rev-parse HEAD` 记为 `REF`，写进留档。

**阶段 2：两档格式与 Dataset 白名单（候选 commit `CAND`）**

改 `framesamp_store.py`、`pack_framesamp_store.py`、`framesamp_dataset.py`、`test_pack_guards.py`。验证（≤5 分钟）：

```bash
MMEVLA_TEST_SOURCE=v1-store/datasets/4task-motion-40ep/source \
MMEVLA_TEST_MANIFEST=v1-store/datasets/4task-motion-40ep/meta/episode_manifest.json \
JAX_PLATFORMS=cpu uv run --no-sync pytest scripts/training/tests/test_pack_guards.py -x -q
# 旧库仍可读：StoreMeta.load + run_fast_checks 对两个旧 framesamp/ 通过
```

**阶段 3：建库（40ep 先、400ep 后）**

```bash
source scripts/training/paths.sh; export UV_CACHE_DIR=/scratch/hongze/.cache/uv
L=v1-store/datasets/4task-motion-40ep   # 第二轮换 4task-motion-400ep
ls -ld $L/framesamp-8x8 2>&1            # 必须不存在
tmux new-session -d -s p8-pack-40 "cd $PWD && set -o pipefail && PYTHONUNBUFFERED=1 uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source $L/source --manifest $L/meta/episode_manifest.json --out $L/framesamp-8x8 --procs 32 2>&1 | tee v1-store/logs/p8-pack-40.log; echo EXIT_CODE=\${PIPESTATUS[0]} | tee -a v1-store/logs/p8-pack-40.log"
# 期望 PACK_DONE=1；随后
tmux new-session -d -s p8-verify-40 "… pack_framesamp_store.py verify --store $L/framesamp-8x8 --resume --procs 32 … p8-verify-40.log"
# 期望 VERIFY_PACK=PASS scanned=13756 mismatches=0（400ep：scanned=123044）
uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store $L/framesamp-8x8
uv run --no-sync python scripts/dataset/xgrid_pos_check.py --store-4x4 $L/framesamp --store-8x8 $L/framesamp-8x8 --source $L/source
```

400ep 会话名 `p8-pack-400`、`p8-verify-400`。留档到 `docs/dataset-build-doc/4task-motion-{40ep,400ep}-framesamp-8x8/`。

**阶段 4：第一块验收（装配层）**

```bash
export DTYPE_MANIFEST=$L/meta/episode_manifest.json MMEVLA_MOTION_STORE=$L/motion
for IMPL in refnpy packed; do
  DTYPE_DUMP_DIR=v1-store/fixtures/8x8/$IMPL-c8 DTYPE_DUMP_IMPL=$IMPL JAX_PLATFORMS=cpu \
  uv run --no-sync python scripts/training/tests/dump_fixture_samples.py mme_vla_suite --exp-name fx-c8-$IMPL \
    --dataset-path $L/$([ $IMPL = packed ] && echo framesamp-8x8 || echo source) \
    --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled
done   # M8 同法换 -motion.yaml
uv run --no-sync python scripts/training/tests/compare_fixture_dumps.py v1-store/fixtures/8x8/refnpy-c8 v1-store/fixtures/8x8/packed-c8
# C32/M32：dump 目录换旧库 + 旧 yaml，refnpy 侧改为 REF commit 的 packed 实现（worktree v1-store/worktrees/ref-8x8 @REF）
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/spawn_matrix.py --store $L/framesamp-8x8 --source $L/source --manifest $L/meta/episode_manifest.json --workers 0,1,4,16 --epochs 2
```

超过 5 分钟的项进 tmux（前缀 `p8-fx-`）并按第 17 条留档。

**阶段 5：第二块（12 条轨迹）**

两波、每波两组并行、组内串行（GPU 约束只允许 4–7）；下面以 C8 组（第二波，GPU 4,5）为例，A1/A2 在 `v1-store/worktrees/ref-8x8`（`git worktree add --detach … REF`，`.venv` 走 `UV_PROJECT_ENVIRONMENT=<主树>/.venv`），B 在主树 `CAND`：

```bash
V1=$PWD/v1-store; L=$V1/datasets/4task-motion-400ep; RUN=t8-c8-a1   # 全新 run_name，起跑前与用户确认
REC=$V1/bench/8x8/$RUN; mkdir -p $REC
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=4,5
export MMEVLA_JAX_CACHE_DIR=$V1/cache/jax/$RUN UV_CACHE_DIR=/scratch/hongze/.cache/uv WANDB_MODE=disabled
JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/check_baseline_env.py dump --record-dir $REC --dataset $L/framesamp-8x8 --norm-stats $V1/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json
BENCH_RECORD_DIR=$REC BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_CHECKSUM_STATE_STEPS=0,1,2,25,50,100,200,400,600,800,1000 BENCH_EXTRA_DIGEST_STEPS=0,1,2,24,49,99,199,399,599,799,999 \
BENCH_DATASET_IMPL=refnpy BENCH_REF_SOURCE=$L/source BENCH_REF_MANIFEST=$L/meta/episode_manifest.json \
uv run --no-sync python scripts/training/g0/bench_train_steps.py mme_vla_suite --exp-name $RUN \
  --assets-base-dir $V1/train-assets --checkpoint-base-dir $V1/train-runs/$RUN \
  --batch-size 8 --num-workers 4 --num-train-steps 1000 --log-interval 1 --save-interval 1000 --seed 42 --fsdp-devices 2 \
  --dataset-path $L/source \
  --weight-loader.params-path $V1/models/openpi-assets/checkpoints/pi05_base/params \
  --model.use-history --model.history-config perceptual-framesamp-context-8frame-8x8.yaml --no-wandb-enabled
# A2：RUN=t8-c8-a2，其余逐字相同。B：RUN=t8-c8-b，在主树跑，去掉 BENCH_DATASET_IMPL/BENCH_REF_*，--dataset-path $L/framesamp-8x8
# C32/M32 三条都用 --dataset-path $L/framesamp 与旧 yaml，A 侧在 REF worktree、B 侧在主树，无 refnpy；C32 用 CUDA_VISIBLE_DEVICES=4,5、M32 用 6,7（第一波）
uv run --no-sync python scripts/training/tests/gate_8x8.py --profile c8 --run-a1 $V1/bench/8x8/t8-c8-a1 --run-a2 …/t8-c8-a2 --run-b …/t8-c8-b --log-a1 v1-store/logs/t8-c8-a1.log … --steps 1000 --batch-size 8
```

M8 组用 `CUDA_VISIBLE_DEVICES=6,7`，yaml 换成 `-motion.yaml` 并加 `MMEVLA_MOTION_STORE=$L/motion`、`BENCH_REF_MOTION=$L/motion`。每条轨迹放 detached tmux（会话名 = run_name，前缀 `t8-`），日志 `v1-store/logs/<run_name>.log`。先各跑一次 100 步（`--num-train-steps 100`，run_name 后缀 `-s100`）排错与估时，跑完清理；100 步不作通过证据。

单步全梯度对拍（每 profile 三种 batch）：`single_step_grad.py` 走同样的 `DTYPE_MANIFEST` / impl 开关，判定行 `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0`。

**阶段 6：第三块（推理）**

第一步（阶段 2 之后即可，GPU 任一张）：

```bash
CUDA_VISIBLE_DEVICES=7 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
uv run --no-sync python scripts/training/tests/online_mem_regress.py --ref-worktree v1-store/worktrees/ref-8x8 --frames-from $L/source --n-frames 256
# 真模型 S 臂回归：命令照抄 docs/training-doc/tic-obs-model-40k/launch.md，只换代码版本，13 条判据逐条与该留档同值
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/training/g0/compare_online_motion.py --gpu 7 --yaml perceptual-framesamp-context-motion.yaml --lib v1-store/datasets/4task-motion-40ep --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999
```

第二步（`t8-c8-b` / `t8-m8-b` 跑完之后；bench 侧起跑时带 `BENCH_SAVE_FINAL_CKPT=1`，checkpoint 落 `$V1/train-runs/t8-m8-b/999`）：

```bash
CK=$V1/train-runs/t8-m8-b/999   # run 根 $V1/train-runs/t8-m8-b 须含 history_config.resolved.yaml/.sha256 与 motion_provenance.json
ls $(dirname $CK)/history_config.resolved.yaml $(dirname $CK)/motion_provenance.json
CUDA_VISIBLE_DEVICES=7 MMEVLA_MOTION_STORE=$L/motion \
uv run --no-sync python scripts/training/g0/compare_train_infer_obs.py --ckpt $CK --motion-gpu 7 --dataset $L/framesamp-8x8 --source $L/source --manifest $L/meta/episode_manifest.json --episodes 5 --out v1-store/reports/t8-infer-m8   # 其余参数照 tic-obs-model-40k launch.md
# C8 同法，--ckpt $V1/train-runs/t8-c8-b/999，不带 MMEVLA_MOTION_STORE
```

第三步（闭环冒烟，4 片 4 卡，tmux 会话由 `run_batch_mv.sh` 起、前缀 `mv-`）：

```bash
# motion-variance 链路，normal 条件，--ckpt 换成测试 run；4 任务 × 10 集，每任务一片一卡，sidecar 经 --motion-gpu 与 policy 同卡
COND=normal SPLIT=test SEED=42 WORKERS=4 GPU_LIST=4,5,6,7 EP_COUNT=10 CKPT_DIR=$V1/train-runs/t8-m8-b/999 bash scripts/motion-variance/run_batch_mv.sh
# C8 同法 CKPT_DIR=$V1/train-runs/t8-c8-b/999（关闭态不起 sidecar）
# 备选：legacy-eval/eval_shard.remote.sh + GPU_LIST=4,5,6,7（先按其 README 拷回副本并 commit）；.local 固定 8 卡映射禁用
uv run --no-sync python scripts/training/g0/summarize_eval_probe.py --max-frames 8 --tokens-per-frame 64 …
```

判定只看 `errors=0`、每集推理次数 ≤ 82、`mem_order` 合法、M8 窗数与公式一致；成功率照实记但标「测试模型，不作指标」。

### 4. 判定行清单（最终留档必须逐条出现，禁止一条笼统 PASS）

| 判定行 | 出处 | 通过条件 |
|---|---|---|
| `VERIFY_PACK=PASS scanned=<rows> mismatches=0` | pack_framesamp_store verify（40ep 13756、400ep 123044） | 两库各一条 |
| `MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0` | xgrid_pos_check | 两库各一条 |
| `SOURCE_IDENTITY=PASS episodes=<n> samples=<n>` | compare_fixture_dumps | 全部一致 |
| `FRAME_INDEX_EXACT=PASS steps=<n>` | compare_fixture_dumps | 全部合法 step |
| `SAMPLE_RAW_EXACT=PASS samples=<n> mismatches=0` | compare_fixture_dumps | 四个 profile 各一条 |
| `BATCH_RAW_EXACT=PASS batches=200 mismatches=0` | compare_fixture_dumps | 四个 profile 各一条 |
| `MATRIX=PASS workers=0,1,4,16 epochs=2` | spawn_matrix | 40ep 8×8 库 |
| `REJECT_INPUTS=PASS cases=<n>` | test_pack_guards | 全部错配 raise |
| `BASELINE_ENV=PASS` | check_baseline_env check | 每条轨迹起跑前 |
| `BASELINE_REPEAT_EXACT=PASS profile=<p>` | gate_8x8（A1 vs A2） | 四个 profile |
| `TRAIN_1000_EXACT=PASS profile=<p>` | gate_8x8（A1 vs B，1000 步五标量 hex） | 四个 profile |
| `FINAL_STATE_EXACT=PASS profile=<p> state_steps=11` | gate_8x8（摘要步全树） | 四个 profile |
| `GRAD_EQ=PASS kinds=3 leaves=<n> mismatches=0` | single_step_grad 对拍 | 四个 profile |
| `GATE_8X8=PASS profile=<p> steps=1000 batch=8 …` | gate_8x8 总行 | 四个 profile |
| `ONLINE_MEM_REGRESS=PASS frames=<n> keys=3 mismatches=0` | online_mem_regress（4×4，真模型 SigLIP） | 第三块准入 |
| `tic-obs-model-40k` 的 13 条阻断判据逐条同值 | compare_train_infer_obs S 臂，`awsprod40k-b128-motion/39999` | 4×4 在线回归 |
| `ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0` 等五条 | compare_online_motion，40ep 库 | 4×4 在线回归 |
| 七关判定行（关 0–6），`--dataset framesamp-8x8` | compare_train_infer_obs，`t8-{c8,m8}-b/999` | C8、M8 各一套；关 4 数值项按现行阈值如实记 |
| `EVAL_SMOKE=DONE profile=<p> episodes=40/40 errors=0 max_infer<=82 mem_order_ok=1` | 闭环冒烟汇总 | C8、M8 各一条；成功率不作指标 |

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
最多 32 帧 → _pad 同 dtype 右补零 → reshape/repeat：
  static_image_emb (512,2048) bf16 2,097,152 B；static_pos_emb (512,768) f32 1,572,864 B
  static_state_emb (512,8) f64 32,768 B（_normalize_state 改数）；static_mask (512,) bool 512 B
  motion 开启：motion_emb (96,768) f32；motion_pos (96,256) f32 = pos_rows(f_m)[:,0,:256]；motion_mask (96,) bool；mem_order (608,) int32
  │ RepackTransform（12 键）→ RoboMMEInputs → DeltaActions（改 actions）→ Normalize（改 state/actions）
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
  static_* 四键形状、dtype、字节数与图 A-before 逐项相同
  motion 开启：motion_emb / motion_pos / motion_mask 逐位同 A-before（同源、同窗口、时间码切片跨网格一致）；
  mem_order 仍 (608,) int32 合法置换，但排列随 8×64 的帧路布局变化（设计内）
  │ transforms → collate → JAX 交付：与图 A-before 同一套代码、同一形制
  ▼
同一 HistoryPi0，无参数增减
```

每样本 image 读取仍是 2 MiB（8 行 × 256 KiB），pos 小表每 worker 从 28.8 MB 增到 115 MB（`np.fromfile` 整表进内存）。

### 6. 留档与 commit

- 阶段 1、2 各自 `commitV<大>.<小>: …` 提交并 `git push`；阶段 1 结束 HEAD 记为 `REF`，阶段 2 结束记为 `CAND`，两者都写进每份 launch.md。
- 建库留档 `docs/dataset-build-doc/4task-motion-{40ep,400ep}-framesamp-8x8/{launch.md,result.md,records/}`，records 收 `store_meta.json`、`report` 输出、判定行、日志尾部。
- 训练留档 `docs/training-doc/t8-<profile>-{a1,a2,b}/`，每条轨迹 `launch.md`（commit、完整命令、env、GPU 对、数据路径）、`result.md`（判定行、耗时、吞吐、显存）、`records/`（`scalars_hex.tsv`、`param_checksums.jsonl`、`batch_digests.jsonl`、`index_sequence.json`、`env.json`、`run_meta.json`）。不归档 checkpoint。
- 推理留档 `docs/training-doc/t8-infer-regress-4x4/`（第一步）、`t8-infer-{c8,m8}/`（第二、三步），各含 `launch.md`（代码 commit、checkpoint 路径与其来源 run、完整命令、GPU）、`result.md`（七关判定行、冒烟四条判定、显存与 host 内存、成功率并标注不作指标）、`records/`（对拍报告 json、eval 汇总）。
- 收官：`docs/dataloader-restructure.md` 加一节「8 帧 × 8×8 档」引用上面留档；`docs/train-infer-consistency.md` 加一节「8×8 档（测试模型）」；本文降为过程档案。

### 7. 需用户拍板的项（实施前）

1. **数据口径**：第二块 12 条轨迹用 400ep 库（本文推荐，交付件就是它）还是 40ep 开发库。
2. **run_name**：`t8-{c32,m32,c8,m8}-{a1,a2,b}` 共 12 个，另 100 步排错 run 加 `-s100` 后缀。
3. **参考链注入方式**：本文推荐通过 `bench_train_steps.py` 的 `BENCH_DATASET_IMPL` 只读 monkeypatch，不改生产 `dataloader.py`；若用户要求生产侧开关，需另议。
4. **jax 缓存落点**：本文推荐入口读 `MMEVLA_JAX_CACHE_DIR`；`~/.cache/` 下已有的 6 个越界目录另立清理任务。
5. **磁盘**：400ep 8×8 库约 30 GiB，确认落 `v1-store/datasets/4task-motion-400ep/framesamp-8x8/`。
6. **推理用 checkpoint**：用第二块 `t8-c8-b` / `t8-m8-b` 1000 步后的 999 目录（本文推荐，唯一现成来源）；bench 起跑时须带 `BENCH_SAVE_FINAL_CKPT=1`。
7. **闭环冒烟规模**：每 profile 4 任务 × 10 集 = 40 集，4 片 GPU 4–7；驱动已由 GPU 约束定为 motion-variance 链路（`run_batch_mv.sh`），备选 `.remote`。
8. **两波顺序**：第一波 C32/M32、第二波 C8/M8（本文推荐，先证旧能力再验新配置）；或反过来先出 8 帧结果。

### 8. 本次文档改动的验证与提交

本轮只改根目录 `8frame-8x8-training-plan.md` 一个文件。验证：`git diff --check`、核对无尾随空白与末尾换行、表格与代码围栏配对、引用的文件路径在 HEAD 上存在（新建文件已标「新建」）。提交主题 `docs: 8帧8×8计划按两部分结构细化——数据预处理与训练分块`，逐文件 `git add`，commit 后 `git push` 到既有 upstream，`git status -sb` 无 ahead。
