# 新库 motion 表构建 + modulation 8×8 接入 motion 计划（v2-motion）

> **状态：方案已固化，八项口径经用户拍板，`run_name` 待确认；未实施。** 2026-09-16 起草；2026-09-17 重写第一部分；**2026-09-17 第二轮**：九路独立核验一份 32 条外部审计清单后整体修订；**2026-09-17 第三轮**：对修订版再派三路对抗审查（建库侧 / 训练验证侧 / 修订忠实度），共 52 条 finding，按结论二次修订——本轮新增 5 个 P1 的处置（encoder 换型的全仓连带面、V7/V8 摘要步口径、V5 测试模型、H 节关闭态图纠错、V-online 分片抽样）。环境判定：**环境 B（AWS 单机 8×A100-SXM4-80GB）**，主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`。基线 `v2-1600ep-m8x8-modul-b128-60k` 已于 `2026-09-16T20:53:31Z` 以 `EXIT_CODE=0` 跑完，主副本已解锁、开发副本 `-temp` 已删除（`5432ba2`），**8 卡全空**。无 turbo、无 GreatLakes。
>
> 本文按 `AGENTS.md` 第 2 条分两部分。正本 `docs/motion-memory.md` 写的是 context + 32 帧 4×4 + 旧四任务口径，本文只写**与其不同**的部分；实施后正本另立 `docs:` 更新。

**用户已拍板**

| 项 | 决定 | 日期 |
|---|---|---|
| 接入目标 | **modulation 8×8**，与已跑完的基线 `v2-1600ep-m8x8-modul-b128-60k` 同 YAML、只多 `motion` 节 | 09-16 |
| demo 段窗口 | **全覆盖 + 真实帧 ≥ 17**：起点 `s ∈ range(0, es, 16)` 且 `es − s ≥ 17`；窗口越过 `es − 1` 的部分用第 `es − 1` 帧重复填充凑满 33 帧 | 09-16 |
| exec 段窗口 | **不变**：尾端 ≤ 当前帧的完整 33 帧窗，不补帧 | 09-16 |
| 新库 motion 表 | **单独构建**（1600 集新库当前没有 motion 表） | 09-16 |
| 正式 run 步数 | **80k 步**（用户原话「一路做到开始 80k 训练」） | 09-17 |
| 80k 落点 | **新具名配置条目 `mme_vla_suite_b128_80k`**（`AGENTS.md` 第 10 条要求的落点确认）；`mme_vla_suite_b128_60k` 与官方条目一字不动 | 09-17 |
| `motion.budget` | **160**（训练零截断实证；1300 步评估口径下训练集外推最大 151 ≤ 160，**评估集本身不可预知，见「六、budget」**） | 09-17 |
| **encoder** | **换用 `wan-full1600-filter2-b176x4-72ep-a` 的 `checkpoint_epoch_72.pt`**（在本批 1600 集全库上训成），替换计划初稿的 `wan-v8-filter10-72ep-a` | 09-17 |
| **V-online 口径** | **按 episode 确定式抽 10%（160 集）+ 8 卡分片并行**；抽中集内**全部窗逐位比**（含其 160 个补帧窗），约 21–25 min | 09-17 |
| encode 续跑 bug | **本轮顺带修 + 加单测**（既有 bug，完成性判断恒 False，400ep 实测约 7.5× 冗余） | 09-17 |
| D2 抽样边界 | **只抽样 VAE 前向；段集合/行数/逐行 m·offset/帧 sha 四道检查保持全量** | 09-17 |
| 旧配置兼容 | **reader 侧默认值兼容**：两键同缺 → 按历史契约 `(33, "none")`；只缺一键 → raise；两键齐备 → 必须与表 spec 逐值相等 | 09-17 |
| 主比较锚点 | **80k run 的 `60000` vs 基线终点 `59999`**（相差 1 次优化步，报告须标明） | 09-17 |
| **V5 测试模型** | **在 GPU 上用真实配置跑**（真 1024 宽、真 YAML），不新增测试变体、不改任何生产模型代码 | 09-17 |
| `run_name` | **待确认**（本文建议 `v2-1600ep-m8x8-modul-motion-b128-80k`，起跑前按 `AGENTS.md` 第 6 条确认） | — |

---

## 第一部分（给人看）

### 这轮做什么

给 1600 集新库（BinFill / RouteStick / VideoRepick / VideoUnmaskSwap，605,611 个执行样本）建一张 motion token 表，然后用已跑完的 modulation 8×8 基线配置**只加一个 `motion` 节**，训练一条 80k 步的「+motion」模型，与基线成对可比。全文顺序：① demo 段补帧规则 → ② 建表六步 → ③ 新 run 与基线的逐项差异 → ④ 训练链路哪些不动、哪些改 → ⑤ 推理侧 → ⑥ budget → ⑦ 验证 → ⑧ commit 与执行顺序。代码级细节全部在第二部分。

### 一、demo 段怎么补：数轴

一个 motion 窗口 = 连续 33 帧，喂给冻结的 MotionJEPA encoder 得一个 token。起点钉在段内网格 0, 16, 32, … 上，demo 段和 exec 段各自算网格、不跨段。

**现行规则的问题只在 demo 段尾部**：整窗放不下就不编，demo 最后 0–15 帧的运动没有任何窗覆盖。以 RouteStick 一集 `es = 100`（demo 帧 0–99）为例：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
现行规则    [0 ══════════════ 32]                                        s=0   真 33
                     [16 ═════════════ 48]                               s=16  真 33
                              [32 ═════════════ 64]                      s=32  真 33
                                       [48 ═════════════ 80]             s=48  真 33
                                                [64 ═════════════ 96]    s=64  真 33
                                                         ✗ s=80: 80+32=112 > 99，整窗越段，不编
                                                            → 帧 97–99 的运动没有任何窗覆盖
```

**新规则：起点只要还剩 ≥ 17 帧真实帧就编一窗，越过段尾的部分用 demo 最后一帧重复填满**：

```
demo 帧号   0        16       32       48       64       80       96  99 │100 exec →
            ├────────┼────────┼────────┼────────┼────────┼────────┼──┤
新规则      前 5 窗与现行完全相同（字节不变）
                                                         [80 ═══════ 99│99 99 … 99]   s=80  真 20 + 补 13（第 99 帧重复）
                                                                  ✗ s=96: 真实帧只剩 4 < 17，不编
```

补的是静止帧。**这是对 encoder 输入的事实描述——「前 20 帧真实运动、后 13 帧定格在第 99 帧」；本文不断言 encoder 会把它编码成「动作结束」这一语义，那需要另行验证。**

**`es ≥ 17` 的集恰好多 1 窗，且 demo 尾帧一定被盖住**：旧规则接受 `s ≤ es − 33`，新规则接受 `s ≤ es − 17`，多出的区间宽 16 = 一个网格步，里面有且只有一个网格点；这个新窗右端 ≥ es，必然覆盖到最后一帧。解析证明：令 `es − 17 = 16q + r`（`0 ≤ r < 16`），末窗起点 `s_max = 16q`、真实帧数 `= 17 + r ∈ [17, 32] < 33` 必为补帧窗，倒数第二窗真实帧数 `= 33 + r ≥ 33` 必为满窗。**由此还得到一条被下游反复依赖的性质：补帧窗永远是 demo 段的最后一个 `m`，可由 `seg_len` 直接推出、不必查表。** `es ≤ 16` 的集新旧规则都不编 demo 窗、增量为 0，本库 `es` 最小 100、不存在该情形，但账目检查须把「`es ≥ 17` 的集数 == 1600」一并断言，防止换库时静默失配。

**exec 段一字不改**：窗口尾端不能超过当前帧，当前帧之后的帧根本不存在，没有段尾可补。

新库实算：demo 窗 34,313 → **35,913**，exec 35,403 不变，总行 **71,316**，补帧窗恰 **1,600**（= 集数）。**注意这是整表重建、不是增量补行**：demo 每段多 1 窗会把该 episode 的 exec `row_base` 以及其后所有 episode 的行号整体平移，旧表的任何一行都不能复用。

同一个网格公式在代码里被**独立重写了 8 遍**（训练侧契约 `motion_store`、Wan 抽取器 `wan_common`、oracle `oracle_driver::expected_segments`、在线 `framesamp_memory`、账目 `motion_checks::_independent_visible`、`extra_checks::visible`、模型测试 `motion_gates_model::oracle_visible`、评估探针 `summarize_eval_probe::motion_frames_formula`，另加手算参照 `hand_calc_8frame::expected_motion` 与节奏闸 `eval_rhythm_gates::predict_k`），**逐处裁决见第二部分 A1 的清单**；而**公式的调用点**更多——`visible_motion_rows` 全仓 ≥25 处、`max_visible_count` 5 处、`seg_num_grid` / `seg_num_chunks` 十余处，散在 **15 个文件**里。

### 二、建表：六步，哪步没做

建表链路照抄 400ep 库，**链路本身不改，只换口径与规模**，另加两步本轮新增的前置工作。现状：新库只有帧特征库，motion 相关的五样（latent、token、整表、oracle、留档）一样都没有；原始 H5、wan 子 venv 在位。

| 步 | 模块 | 改不改 | 状态 | 预计耗时（8 卡） |
|---|---|---|---|---|
| 0 | **encoder 换型** | 新 encoder 落 `v1-store/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/`，`ASSETS_LOCK.json` 换锚点，**全仓 7 处写死旧 run 名/目录逐项裁决**（见 A9） | **未做** | 分钟级（本地复制） |
| 1 | **输入重锚** | 建 motion 前重核四个 H5 的内容 sha256 == 帧库 finalize 时的 `input_manifest.json` | **未做** | ≈ 15 min（791.8 GB） |
| 2 | **改代码** | 见第二部分 A1–A9 | **未做** | CPU 单测 < 5 min |
| 3 | **Wan 抽取** | 逻辑不改，只是切出来的 demo 尾窗多了补帧 | **未跑** | ≈ 3.6 h（≈ 39 GiB latent） |
| 4 | **encode** | 数值核心不改；**只修续跑完成性判断的路径 bug** | **未跑** | 修复后理论 ≈ 3 min（400ep 实测 17.7 ms/窗，8 卡真并行 71,316/8 × 17.7 ms ≈ 158 s）；**按实测回填** |
| 5 | **pack / verify** | 拼表逻辑不改，只多写新布局名与新契约键 | **未跑** | 分钟级 |
| 6 | **oracle 与账目** | MotionJEPA 仓库自己的 VAE / encoder 独立重算逐位比：encoder 全量，VAE 抽样（补帧窗全部 + 其余 10%，**只抽 VAE 前向**）；账目六项 | **未跑** | ≈ 1.2 h |

第 3 → 4 步之间**不能 commit**（provenance 要求跨两阶段 `git_commit` 唯一，400ep 曾因此重抽全量）。新表布局名换成带 `demopad17` 的新名，训练侧按布局名查规格表，**40ep / 400ep 旧表照旧可读**。全部命令与判定行见第二部分 B 节。

**encoder 换型的连带面比初稿写的大得多**（第三轮审查的核心发现）。ckpt 本身完全兼容：新旧**架构一致**（`motion.dim 768` / encoder `depth 8, heads 12, dim_head 64` / dit `depth 4` / latent `16ch×32`、`patchify 2` / 同一 Wan2.1 VAE），差异只在训练超参与数据源（filter 阈值 `0.02711` vs `0.045993`、batch 176 vs 88、lr 6e-4 vs 3e-4、holdout 集不同、`save_live false` vs `true`）；已实测新 ckpt `arch = "wan-latent-v7"` 与 `ARCH_TAG` 相符、主键 `encoder`（EMA）含 77 个张量、affine buffer 有限，`load_encoder` 三条断言全过；477 MB vs 954 MB 的差异来自 `save_live=false`，不影响加载。**真正的坑是全仓 7 处写死了旧 run 名或旧 run 目录，其中 4 处在本轮必经路径上**——最致命的是 `pack_motion_store::gather_provenance` 把 `run_name` 硬编码成 `"wan-v8-filter10-72ep-a"`，不改的话新表 provenance 会带旧 run 名，而 `dataloader::_motion_gates` 拿它与新 YAML 的 `source_run` 逐字段比对 ⇒ **训练起跑必 raise，且会在建库全绿、留档写完、preflight 过后才炸**。逐项裁决见 A9。

### 三、新 run 与刚跑完的基线：逐项差异

基线 `v2-1600ep-m8x8-modul-b128-60k` 是 **modulation 8×8、motion 关闭**的 60k run。新 run 是同一条链路**只开 motion + 步数改 80k**。基线值全部取自 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/records/prod-runner.sh`、`records/launch.actual.json` 与 `config.py` 的 `mme_vla_suite_b128_60k` 实体，不是推算。

**只有这 8 处不同，其中 3 处随 `$RUN` 自动派生**（初稿写「5 处」并漏了 sha 与三条派生路径，与 G 节自相矛盾，已订正）：

| 项 | 基线（已跑完） | 新 run | 说明 |
|---|---|---|---|
| 具名配置 | `mme_vla_suite_b128_60k` | `mme_vla_suite_b128_80k` | 新条目复制 60k，**只改** `num_train_steps` 与 `decay_steps` 60,000 → 80,000 |
| history YAML | `perceptual-framesamp-modul-8frame-8x8.yaml` | `perceptual-framesamp-modul-8frame-8x8-motion.yaml` | 基线 YAML 一字不动，新 YAML = 基线 + 一个 `motion` 节 |
| `--history-config-sha256` | `5b5ac2f8…`（runner 第 2 个位置参数） | 新 YAML 的 sha | preflight 用；`store_path` 参与该 sha，改路径即改 sha |
| `run_name` / `--exp-name` | `v2-1600ep-m8x8-modul-b128-60k` | 待确认（建议 `v2-1600ep-m8x8-modul-motion-b128-80k`） | 第 6 条 |
| `--run-root` 子目录 | `…/train-runs/mme_vla_suite_b128_60k/<RUN>` | `…/train-runs/mme_vla_suite_b128_80k/<RUN>` | 随配置名 |
| tmux 会话 | `m8-prod` / `m8-prod-dense` | `mv2-prod` / **`mv2-dense`** | 后者故意不叫 `mv2-prod-dense`：`tmux -t` 做前缀匹配，`mv2-prod` 会误杀它（第 7 条红线） |
| `LOG` / `TRAIN_RECORD_DIR` | `v1-store/logs/$RUN.log` / `v1-store/bench/$RUN` | 同式 | **随 `$RUN` 自动派生**，照抄 runner 即生效；此处列出以免被误判为「一字不动」 |
| `MMEVLA_JAX_CACHE_DIR` | `$V1_STORE/cache/jax/$RUN` | 同式 | 同上；新 run = 冷编译缓存，与下面「步时必须重测」相关 |

**启动环境变量不在 runner 里，「照抄 runner」会漏三个**：基线的实际启动命令（`records/launch.actual.json::launch_command`）是 `env CUDA_CACHE_PATH=…/v1-store/cache/cuda WANDB_DATA_DIR=…/v1-store/cache/wandb-data XDG_DATA_HOME=…/v1-store/cache/xdg-data bash <runner> <TRAIN_HEAD> <HC_SHA>`，这三个 `env` 都**不在** `prod-runner.sh` 内（runner 里只有 `UV_CACHE_DIR`）。漏掉会让 CUDA JIT 缓存与 wandb 数据落到 `$HOME`，违反 `AGENTS.md` 第 14 条与环境 B 第一条红线。新 runner 要么照抄这个 `env` 前缀，要么把三个 `export` 并进 `body()`。

**其余一字不动**（逐项核对）：`batch_size=128`、`fsdp_devices=4`、`CUDA_VISIBLE_DEVICES=4,5,6,7`、`num_workers=8`、`ema_decay=0.999`、`optimizer=AdamW(clip_gradient_norm=1.0)`、`freeze_filter`、`weight_loader=pi05_base`、`save_interval=5_000`、`keep_period=5_000`、`project_name="robomme-framesamp"`、`seed=42`、`lr_schedule` 的 `warmup_steps=5_000 / peak_lr=5e-5 / decay_lr=5e-5`、`--assets-base-dir`、`--data.assets.assets-dir`（同一 `4task-v2-1600ep-604f16da`）、`--dataset-path .../framesamp-8x8`、`norm_stats` sha256 `856c75ea…`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`UV_CACHE_DIR`、`MMEVLA_FRAMESAMP_SOURCE` / `MMEVLA_FRAMESAMP_MANIFEST`、`MMEVLA_MOTION_STORE` 保持 unset、runner 外壳（`set -o pipefail` + `tee` + `PIPESTATUS` + `EXIT_CODE=`）与 15 s GPU 采样。

**lr 逐步完全相同，比初稿说的更强**：`peak_lr == decay_lr == 5e-5` ⇒ optax 的 `alpha = end/peak = 1` ⇒ 余弦段恒等于 5e-5，`decay_steps` 之后 clamp 也是 5e-5。所以 `decay_steps` 60k→80k **不改变任何一步的 lr**，不只是「前 60k 相同」。F 表用一条 `GUARD_LR` 逐步比两条 schedule 直接证明。

**数据顺序与 `num_train_steps` 无关**：`train.py` 调 `create_data_loader` 时不传 `num_batches`，索引序列只由 `seed=42`、`batch_size=128` 与数据集长度 605,611 决定；motion 开启只让 `__getitem__` 多返回 4 个键，不改索引序列。

**开 motion 带来的模型/数据差异**（这是本轮要测的东西本身，不是配置差异）：

| 维度 | 基线 | 新 run |
|---|---|---|
| 参数叶数 | 61 | **65**（多 `motion_pos_proj` kernel/bias + `motion_encoder_static` kernel/bias） |
| 新增参数量 | — | `256×768+768 + 1536×1024+1024 = 1,771,264` ≈ 1.77 M |
| MemoryAttention 的 `mem_len` | 512 | **672** = 512 + budget 160 |
| 每样本多交付的键 | 四键恒 `None` | `motion_emb (160,768) f32`、`motion_pos (160,256) f32`、`motion_mask (160,) bool`、`mem_order (672,) int32` ≈ **658 KB/样本**，b128 ≈ **84 MB/batch** |
| dataloader 常驻 | — | motion 表 `motion_token.f32.bin` 219 MB × 8 worker ≈ **1.75 GB** |

**步时必须重测，52 h 只是下界**：基线 4 卡 2.34033 s/步（`result.md` 的 step 100→59900 实测），80,000 × 2.34 s = **52.0 h**。但基线本身已是数据受限——全程 15 s 采样四卡 util 均值 **72.39%**、0% 采样占比 **25.39%**，起跑 500 ms 密采均值 70.14%。上面那 84 MB/batch、每样本一次 672 长稳定排序，以及 `MemoryAttention` 在 18 层里每层对整条 mem_seq 做 K/V 投影（512→672 即 +31%，且 remat 策略是 `nothing_saveable`、反向还要重算），大概率直接转成步时。执行顺序表为此设了硬节点：推进到 300 步后按 `AGENTS.md` 第 16 条口径（稳态窗口均值 + 0% 占比 + 慢/非慢分层，**禁用中位数**）重估 ETA 并回报；步时较基线上升 > 15% 时先停下评估 `num_workers` 是否要上调（属第 10 条超参，须另行确认落点）。

**为什么仍用 4 卡而不是 8 卡**：成对可比要求的是**卡数**（`fsdp_devices=4`、可见 4 张 → per-device batch 32）与 mesh 形状，物理卡号不影响任何数值（基线当初用 4–7 只因 0–3 在跑 MotionJEPA 训练，现在 8 卡全空）。改 8 卡会让 per-device batch 从 32 变 16、fsdp mesh 改变，与基线不再逐位可比——代价是另外 4 张 A100 闲置约 52 h。**若愿意牺牲成对可比换速度，这一条可以改，需用户明确。**

### 四、训练链路：哪些不动，哪些改

**不动的（及为什么）**
- **帧路的数据交付**（SigLIP 8×8 帧特征库 → 512 位记忆）：与 motion 无关，**基线 batch 的帧路内容逐字节不变**。这句话只在**数据交付层**成立——进了模型之后，`mem_order` 会把 motion token 按时刻插进帧 token 之间，帧 token 彼此的 RoPE 相对距离随之改变（见下）。
- **dataloader 的 motion 取样逻辑**（查哪些行、右填充、与帧路交错）：公式在契约层，取样代码只是调用它。
- **模型的两层 motion 投影与 `embed_memory`**：context 版已经写好，modulation 复用；参数只在 `motion.enabled` 时创建（1.77 M），关闭态模型与基线完全相同。
- **modulation 的消费路径**：`compute_loss` / `sample_actions` 的 modulation 分支早已写着「取 memory 序列交给 MemoryAttention」，出口维正好是 1024，不用动。
- **基线 YAML、60k 配置、官方配置、Wan VAE、`utils.get_config` 的变体表**：一字不动，基线可复现。
- **`norm_stats`**：`compute_norm_stats.py::_NONE_KEYS` 已含四个 motion 键，norm_stats 与 motion 完全解耦，沿用 `856c75ea…`；**不得为开启态重算**，否则破坏成对可比。

**要改的（及加什么）**
- **motion 表契约**（`motion_store.py`）：加规格表，让「demo 最少真帧数 / 尾部补帧方式」成为布局的属性；**`spec` 挂在 `IndexEntry` 与 `MotionMeta` 上**而不是加进各函数签名（见 A1，这样 25+ 个调用点零改动且自动对每张表用对口径）；新旧表共存。
- **dataloader 守卫**（`FrameSampDataset`）：把写死的 `budget == 96` 放开为 16 的倍数；新增两条三态核对（YAML 两键与表 spec 的关系，缺键按历史契约兜底）。**注意这段必须放在 `MotionMeta.load` 之后**，不能并进现有的 `_req` 块（那时 `self._motion_meta` 还是 `None`）。
- **模型闸**（`HistoryPi0.__init__`）：现在 motion 只放行 context，改成也放行 modulation。就这一处（全仓 grep 确认唯一）。
- **新 YAML**：基线 YAML + `motion` 节（`budget: 160`、补帧口径两键、新表路径、新 `source_run`）。
- **新具名配置 `mme_vla_suite_b128_80k`**：复制 60k 条目，只把 `num_train_steps` 与 `decay_steps` 改 80k。

**一条必须写进正本、且初稿写错了的语义**：modulation 的 `MemoryAttention` 用 `q_positions = arange(mem_len, mem_len + x_len)`、`k_positions = arange(mem_len)`，而 `_apply_rope` 是**纯相对位置**旋转。padding 全在尾部、真实 key 位置不随 budget 变，但 **`mem_len = 512 + budget` 一变，所有 query 的位置整体平移，真实 token 的有效注意力随之改变**。数值实验（真实形制 1024 宽 4 头、真实 token 内容一字不变、只改 budget）实测：144→160 时注意力分布总变差 **0.33**、该层输出相对 L2 变化 **0.76**；512→672 为 **0.87**。两个对照组把因果钉死：改成「padding 不占号」（context 的 `cumsum(input_mask)−1` 口径）或干脆不加 RoPE，同样的改动都是 `1e-17` 逐位不变；而**同一 budget 下只把 padding 位塞 1e3 量级垃圾，差异严格为 0**（`masked_logits = where(mem_mask, logits, -2.38e38)` 保证 padding 位 softmax 概率恰为 0）。

结论：**`motion.budget` 是模型语义参数，不是纯容量上限。** 训练、resume、评估、在线四侧必须逐值相同，任何一侧改 budget 都等同换模型。而且这条**不受参数树保护**——motion 四个参数叶的形状只依赖 `pos_dim/dim/pos.hidden_dim/memory_token_dim`，与 budget 无关，拿 budget=96 的 YAML 去加载 budget=160 训的 checkpoint，`PARAM_TREE_EXACT` 会照样 PASS 而语义已静默改变。唯一拦得住的是 `policy_config._load_resolved_snapshot` 的快照 sha 三方核对，所以验证里必须显式加一条 budget 一致性判据。

### 五、推理侧：哪些不动，哪些改

**不动的**：sidecar 进程本身、线协议（仍是 33 帧一包）、客户端、exec 段增量编码逻辑、policy 的记忆装配。理由：补帧发生在「凑齐 33 帧」这一步之前，sidecar 收到的永远是 33 帧。

**要改的**
- **`FrameSampMemory`**：它是唯一知道 demo 段在哪里的地方。demo 判据从「整窗放得下」改成「还剩 ≥ 17 帧」，并在凑窗时用 demo 最后一帧补齐。**这个谓词在该文件里出现 4 次（不是初稿写的 3 处）**，其中两次是同一条件写两遍，极易只改前者，见 E 节。
- **policy 层**：把 YAML 里的两个补帧口径键透传下去，并按「reader 侧默认值兼容」处理缺键。
- **encoder 目录的传递（初稿写「encoder 权重的加载方式不动」是错的）**：`motion_client.py` 的 sidecar 默认 encoder 目录写死旧 run，而两个构造点（`compare_online_motion.py` 与 `policy_config.py`）都只传 `expected_ckpt_sha256`（取自新表 provenance = 新 ckpt 的 sha）、**不传 `encoder_run_dir`** ⇒ 换 encoder 后 sidecar 必然握手失败，V-online 与将来所有 eval 都起不来。处置见 A9。
- **两处 stub 的帧号校验**（`motion_gates_online.py` 与 `motion_sidecar.py`）：它们硬要求 33 帧编号严格连续，而 `stub_decode` 把帧号写进像素，合法补帧窗的尾部会解出重复帧号 ⇒ 必被判协议错。**这是「sidecar 不改」的唯一例外**，且会改 `protocol_sha256`，握手断言与既有 provenance 记录须一并更新。
- **在线侧的 fail-loud 校验**：训练侧的两键核对在 `FrameSampDataset.__init__`，而 eval 根本不构造 dataset。若不在 `FrameSampMemory.__init__` 也加一道 raise，YAML 写错口径时会变成「训练 raise、在线照跑」，训推不一致而无人发现。

**怎么证明同源**：既有的在线对拍工具按 eval 节奏驱动 `FrameSampMemory` + 真 sidecar，逐窗与离线表比逐位。**全量需单卡 30–32 h**，用户已拍板改为「按 episode 确定式抽 10%（160 集）+ 8 卡分片」，见第七节与 F 节。

新任务的策略评估口径（仿真步数、episode 集合）本轮不定，另起计划。

### 六、budget：已拍板 160

按 ≥ 17 规则逐样本实算合法窗数（605,611 个）：均值 **43.9**，中位 39，P90 80，P95 93，P99 115，最大 **141**（BinFill ep367，es 1152，nt 2304 → demo 71 + exec 70）。

> **填充率定义**：`mean(min(k, budget)) / budget`（截断后实际占位）。初稿未写定义且整张表用了一个在 `δ < 32` 时多算一窗的 exec 公式（`len(range(0, max(0, δ−32)+1, 16))` 在 `δ<32` 时返回 1，正确值是 0），导致均值与填充率系统性偏高。正确 exec 可见窗数：`E(δ) = 0 if δ < 32 else (δ − 32)//16 + 1`，`δ = t − es`。受影响的恰是每集 exec 段前 32 个样本共 51,200 个，`51,200/605,611 = 0.08454` 与两版均值差 `43.99102 − 43.90648 = 0.08454` 吻合。**截断数、截断 episode 数、中位、P90/P95/P99、最大值不受影响。**

| budget | 超限样本 | 占比 | 超限 episode | 平均填充率 | 1300 步评估口径下末段 raise 的 episode |
|---:|---:|---:|---:|---:|---:|
| 96 | 24,366 | 4.02% | 112 | 45.2% | 951 |
| 112 | 7,817 | 1.29% | 43 | 39.1% | 271 |
| 128 | 488 | 0.08% | 6 | 34.3% | 112 |
| 144 | 0 | 0 | 0 | 30.5% | 6 |
| **160** | 0 | 0 | 0 | **27.4%** | **0** |

> 列名用「超限」而非「截断」：代码里 budget 超限**从不截断，一律 raise**（`FrameSampDataset.__init__` 的零截断契约预检、`__getitem__`、右填充三处各一道）。budget=96 时这个数据集根本构造不出来，不是「4.02% 的样本被截断」。

**最后一列度量的是训练集，不是评估集。** 它按 `k_eval = demo_num_grid(es) + E(τ_max − es)` 外推，`τ_max = es + 16·⌊max_steps/16⌋`（`max_steps=1300` → exec 项恒 80），训练集 `es` 最大 1152 ⇒ `k_eval` 最大 **151**；按正本红线在 16 任务全集口径复核，同为 **151**（`VideoPlaceOrder` ep10，es=1140）。

> 初稿 C 节写的 `τ_max = es + 16·⌊(max_steps − 4)/16⌋` 里那个「−4」不成立：`⌊(M−4)/16⌋ = ⌊M/16⌋` 当且仅当 `M mod 16 ≥ 4`，而 `1300 mod 16 = 4` 恰好卡在边界。反例：M=1296 正确 1296、该式给 1280；M=1312 正确 1312、该式给 1296；M=800 正确 800、该式给 784。代码里 `eval_rhythm_gates.py` 本来用的就是 `es + 1296`，与修正式一致。

**评估集不可预知（用户 09-17 口径）**：评估将走**新的 seed**，仅「1300 步」这一条固定。当前 `examples/robomme/env_runner.py` 硬编码 `dataset="test"`，`es` 由 `DemonstrationWrapper.reset` 现场 rollout 生成、`examples/robomme/eval.py` 现取，训练 manifest 约束不到它；现有 test split（4 任务 × 50 集，seed 540000+）与训练库 seed 4000–18500 完全不相交，且训练集里 `VideoRepick` 只有 easy/medium/xhard、没有 test split 里的 hard 档。因此：

- budget 160 在 1300 步下零截断的**充要条件是评估 `es ≤ 1296`**；
- 训练集实测上界 1152，裕度 144 帧（12.5%），唯一风险线是 BinFill hard（训练 es 578–1152）；
- `es ≥ 1297` 时在线装配直接 raise、该集记 error（raise 点唯一，在 `FrameSampMemory._prepare_motion`；`_encode_ready_windows` **不做上限检查**，所以超限的集会先白编一堆窗再在第一次 infer 时炸）；
- 评估口径或任务集变更时须按第二部分 C 节的式子重核。

**代价栏**（初稿写的「只是多 16 个被 mask 的 K/V 位」是错的，已删）：K/V 长度 512 → 672（+31%），`MemoryAttention` 在 18 层里每层都对整条 mem_seq 做 K/V 投影、`sample_actions` 每个去噪步再算一遍，且 remat 策略是 `nothing_saveable`（反向要重算）；此外 query 与最后一个真实 token 的 RoPE 间距 `gap = budget − k` 会逐样本在 **19..154** 间跳动（本库 `k` 最小 6、最大 141；实测 gap 只要非 0，分布总变差就已饱和在 ≈0.4），这是一个与内容无关的旁路信号，**budget 取得越大该 gap 的均值与方差越大**——所以 176 并非免费保险。

### 七、怎么证明没改坏

- **关闭态逐位不变**：dataset 交付（V1）、定点梯度（V6）、100 步守卫（V7），改前 vs 改后三项全逐位；任一不过不得宣称基线等价。
- **旧表 / 旧配置可读**（V2）：三条子判据——旧表 `MotionMeta.load` + `parse_index` 通过且解析出 `(33, "none")`；**未经修改**的旧 context YAML + 40ep 旧表构造 dataset 交付逐位不变；用被哈希冻结的历史 run 快照 `awsprod40k-b128-motion/5000` 真实走一遍 policy 构造（`motion_stub=True`，不起 sidecar、不占 GPU）。
- **新表正确**（V3）：输入重锚 + D3 encoder 全量逐位 + D2 VAE **抽样只缩前向、四道元数据检查全量** + 账目 a6set/a7/a9set/a10/a11。
- **开启态**（V4/V5/V8/V-online/V9）：交付行 == 表行且样本集**分层构造**（必须命中冷启动、首个 exec 窗、最大 141 窗、pad 的两个极端）、modulation 下 padding 内容不影响 loss 与**全部参数梯度**、A/A 100 步可复现、在线 vs 表逐位（抽样 160 集 × 8 卡分片）、4 卡 20 步 smoke + 参数树 65 叶精确匹配 + 真实 checkpoint→policy 加载 + budget 四侧一致。
- 判定行与工具全在第二部分 F 节，**每一行都已与工具实际输出对表两轮**。
- **按 `AGENTS.md` 第 18 条的两块划分**：**第一块（非训练轻量对拍）= V1 / V2 / V4 / V-online**（不启动训练，用 index 序列、逐样本/逐 batch 内容、dtype/shape 逐键比对证明新旧链路交付一致，判据逐位）；**第二块（本机训练梯度一致，最后检验）= V6 / V7 / V8**（真实训练，前 100 步逐步比 loss/梯度范数/参数摘要）。**第二块不通过不得宣称改动等价。** 两张链路图见 H 节。

**V-online 的口径（用户两次拍板后的最终形态）**：全量对拍需编码全部 71,316 窗（编码由 `feed → add_buffer → _encode_ready_windows` 触发，与「比不比这一行」无关，所以**窗级抽样一秒也省不下来**；而补帧窗恰好每集一个，「补帧窗全比」等价于跑满 1600 集）。最终口径改为**按 episode 确定式抽 10%（160 集）+ 8 卡分片并行**：

- 抽样规则 `sha256(f"{seed}:{global_episode_idx}")` 取模，与分片数、遍历顺序无关；
- 抽中的 160 集内**全部窗逐位比**，含其 160 个补帧窗；
- 8 片各占一张 GPU（各起一个 sidecar），`aggregate` 合并 `rows_seen` / `mismatches` / `per_ep`；
- 判据从 `covered == 71316` 改写为 `covered == 抽样集合窗数总和` 且 `padded_covered == 160`；
- 预期 ≈ 7,100 窗 × 1421.5 ms ≈ 2.8 h 单卡 ⇒ **8 卡约 21–25 min**。

**承诺口径据此下调并原样写进留档**：不再是「全部 71,316 窗逐位一致」，而是**「确定式抽中的 160 个 episode 内全部窗逐位一致（含其 160 个补帧窗）」**。

### 八、commit 与执行顺序：从现在到 80k 起跑

主副本上顺序执行，每个 commit 后立即 push。建库 8 卡全开；对拍、smoke、正式 run 用 GPU 4–7。

**本轮 tmux 会话名清单**（第 7 条要求先记清单、清理时只按全名逐个 kill；两两之间无前缀关系）：`mv2-wan`、`mv2-encode`、`mv2-pack`、`mv2-oracle`、`mv2-online`、`mv2-bench-base`、`mv2-bench-cand`、`mv2-aa`、`mv2-m4`、`mv2-smoke`、`mv2-prod`、`mv2-dense`。**注意没有 `mv2-prod-dense`**——它会被 `tmux -t mv2-prod` 前缀命中。

| 步 | 做什么 | commit |
|---|---|---|
| 0 | 用户确认 `run_name`；`df -h /scratch` ≥ 400 G 才起跑 | — |
| 1 | 本文入库 | `docs:` |
| **2a** | **在第 1 步 commit 之后的 clean HEAD 上取「改前」取证**：`BASE=$(git rev-parse HEAD)`（**不写死 sha**），`DTYPE_SOURCE_COMMIT=$BASE`；跑 V1 baseline dump、V6 baseline 定点梯度、V7 baseline 100 步 bench（tmux `mv2-bench-base`），产物落 `v1-store/bench/<RUN>/*-base/`。**V1/V6/V7 均 > 5 min，按第 17 条各自留档到 `docs/training-doc/<bench run_name>/`；V7 会真起训练进程，run 名记入清单、验完清理 run 产物、保留留档（第 6 条）** | — |
| **2b** | 实施第二部分 A–E 全部代码 + CPU 单测（含 `GUARD_LR`；此阶段工作区 dirty，**不跑任何带 clean-source 校验的工具**） | — |
| **2c** | `commitV10.0: demo 段补帧网格、modulation 接入 motion 与 80k 配置` + push；记 `CAND=$(git rev-parse HEAD)`，工作区再次干净 | `commitV10.0` |
| **2d** | 取「改后」取证并对拍：V1/V6/V7 三项 compare（tmux `mv2-bench-cand`）+ **V2 三条旧表/旧配置回归**。任一 FAIL → `git revert commitV10.0` + push，停下交用户 | — |
| 3 | encoder 换型落盘 + `ASSETS_LOCK.json` 更新 + `test_assets_lock.py` 通过 + 输入重锚（`INPUT_REANCHOR=PASS files=4`） | `fix:`（资产锚点） |
| 4 | 建库四步（tmux `mv2-wan` → `mv2-encode` → `mv2-pack` → `mv2-oracle`；**Wan 与 encode 之间零 commit**）+ 账目六项 | — |
| 5 | 建库留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/` | `docs:` |
| 6 | 开启态验证：V4 + V5（tmux `mv2-m4`，GPU 真实配置）+ V8（tmux `mv2-aa`）+ V-online（tmux `mv2-online`，8 卡分片约 21–25 min）+ 4 卡 20 步 smoke（tmux `mv2-smoke`，**基线同档实测 5 分 18 秒 > 5 min，按第 17 条以完整 run 留档**） | `docs:` 守卫留档 |
| 7 | 起跑留档 `docs/training-doc/<run_name>/launch.md` | `docs:`；此刻记 `TRAIN_HEAD` |
| 8 | 生成 runner（照抄基线 runner + `launch.actual.json` 的 `env` 前缀，只改第三节那 8 处）→ tmux `mv2-prod` → preflight PASS → 训练；Monitor 盯日志；**300 步处按第 16 条口径重估 ETA 并回报** | — |
| 9 | 训练期间主副本锁只读；需并行开发再建 `-temp` 副本，结束后删 | — |

**磁盘预算**：`/scratch` 当前 6.9 T 总、5.7 T 已用、**1.2 T 可用（84%）**。本轮新增约 **256 GiB** —— Wan latent 39.2 GiB + **oracle vae latent 39.2 GiB**（A4 规定「定长写零」⇒ 与被测同大，初稿按「只写抽中窗 ≈5 GiB」算，少记 34 GiB）+ motion-tokens 209 MiB + motion 表 209 MiB + oracle token 表 209 MiB + 80k run 的 **16 个** checkpoint ≈ 177 GiB（基线 12 个实测 133 G）。起跑前 `df -h /scratch` 判定 Avail ≥ 400 G；**compare 通过后可删 oracle 的 39 GiB**。

**红线**：不在 Wan → encode 之间 commit；不改 exec 规则、stride，不做消融；不改关闭态 YAML、既有 context YAML、run 内任何 `history_config.resolved.yaml`、三个既有配置条目、`train.py` 的保存条件与 `checkpoints.py` 的保留策略、`utils.get_config` 的变体表；tmux 只按全名逐个清理、禁 `kill-server`；正式 run 起跑前 `run_name` 必须经用户确认、run 根必须不存在。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 契约与公式

**A1 `src/mme_vla_suite/datastore/motion_store.py`**

三个新契约键：`demo_min_real_frames`、`exec_min_real_frames`、`demo_tail_pad`。**YAML 只承载 demo 两键**；`exec_min_real_frames` 由规格表固定为 33、不进 YAML、不参与 dataloader 的 `_req` 比对。

- **常量与规格表**：`LAYOUT = "motion-768-grid16-demopad17-v1"`；`LAYOUT_SPECS = {"motion-768-grid16-v1": LayoutSpec(demo_min_real=33, exec_min_real=33, demo_tail_pad="none"), "motion-768-grid16-demopad17-v1": LayoutSpec(17, 33, "repeat_last")}`。**`LayoutSpec` 必须是模块级 `@dataclass(frozen=True)`**——`IndexEntry` 是 frozen dataclass，其 `__eq__`/`__hash__` 逐字段递归，且 `FrameSampDataset` 持有 `MotionMeta`（含 entries）并定义了 `__getstate__`，8 个 worker 要 pickle 它。
- **导入守卫必须同改**（不改则模块 import 即 RuntimeError）：现有 `if not LAYOUT.endswith(f"-{LAYOUT_GRID_SUFFIX}-v1"): raise` 对新名求值为 False。改为遍历 `LAYOUT_SPECS`，对每个 key 断言 `f"-{LAYOUT_GRID_SUFFIX}-" in key and key.endswith("-v1")`，并断言 `LAYOUT in LAYOUT_SPECS`。
- **两处 layout 等值校验必须同改**（初稿只提了 `load`，漏了 `parse_index`；而 `MotionMeta.load` 内部正是调 `parse_index`，不改则 V2「旧表可读」必 FAIL）：`parse_index` 与 `MotionMeta.load` 里的 `if need("layout") != LAYOUT: raise` 一律改为 `raw["layout"] not in LAYOUT_SPECS` → raise。
- **`parse_index` 的逐段公式核对**（`int(s["num_chunks"]) != seg_num_chunks(L) or int(s["num_grid"]) != seg_num_grid(L)`）必须用**本表 layout 查出的 spec**，不得用模块默认值——不改这里，新库的 `motion_index` 一加载就 ValueError。
- **公式签名**：`seg_num_chunks(L, min_real=33)`、`seg_num_grid(L, min_real=33)`、`segment_grid_starts(L, min_real=33)`（该函数全仓无真实调用点，改它无收益但无害）。
- **`spec` 的传递方式 —— 挂在数据对象上，不加进各函数签名**：
  - `IndexEntry` 增 `spec: LayoutSpec` 字段，由两条产生路径各自填好——writer 走 `build_index_entries(manifest, spec)`（**必填位置参数，不给默认值**），reader 走 `parse_index`（从 `raw["layout"]` 查表写进每个 entry）。**`parse_index` 的返回值类型不变**（初稿写「spec 随 entries 一起返回」措辞会被理解成改返回签名，那样 4 个调用点全断）。
  - **`MotionMeta` 增 `spec: LayoutSpec` 字段**（初稿漏了，而 A6/C 三次引用 `meta.spec`），由 `MotionMeta.load` 从 `parse_index` 的结果填入。A5 里 `_independent_visible` 读的 `store_meta.demo_min_real_frames` 是**第三个口径**，须统一：`store_meta` 也写三新键，`MotionMeta.load` 的常量循环里校验它与 layout 查出的 spec 同值。
  - 于是 `visible_motion_rows(entry, t)` 与 `max_visible_count(entry)` **签名一字不改**，25+ 个调用点零改动且自动对每张表用对口径。
  - `visible_motion_rows`：demo 条件用 `s + (entry.spec.demo_min_real − 1) ≤ es − 1`；exec 条件不变。
  - **`pack_motion_store::dataclass_tuple` 同步加 `spec` 字段**——理由不是「防止漏掉规格差异」（`cmd_verify` 两侧 spec 同源、该比对对 spec 恒真），而是保持逐字段完备、避免日后加字段时漏。
- **`index_payload` 必须加 `spec` 与 `layout` 形参**（初稿只说「写三新键」）：现行它取模块常量 `LAYOUT`。A8 要求 `test_guards.py` 分「旧 spec 组 + 新 spec 组」，旧组用 `build_index_entries(m, OLD_SPEC)` 造 entries 而 `index_payload` 写新 `LAYOUT` + 新三键、`parse_index` 再按新 spec 校 `num_grid` ⇒ 必 `ValueError`。签名改为 `index_payload(manifest, entries, *, spec, layout, mj_repo_commit)`。
- **`INDEX_SCHEMA` / `META_SCHEMA` 一并升版**（A3 的 `METADATA_SCHEMA` 2→3、A2b 的 aggregate schema 2→3 都升了，契约层不升会导致旧解析器读新表时不报错只错读）。

**同一网格公式的 10 处独立重写 —— 逐处裁决**（初稿的「四处同改」低估）：

| # | 位置 | 本轮处置 |
|---|---|---|
| 1 | `motion_store.seg_num_grid` / `seg_num_chunks` | 参数化（本节） |
| 2 | `wan_common.seg_num_grid` / `seg_num_chunks` | 参数化（A2） |
| 3 | `oracle_driver.expected_segments` | 参数化，demo 17 / exec 33 分段取值（A4） |
| 4 | `framesamp_memory` 的 4 处 demo 谓词 | 同改（E） |
| 5 | `motion_checks._independent_visible` | **按段分支**改写，禁止 import 公式（A5） |
| 6 | `extra_checks.visible` | 本轮不跑 a8/a9enc，**显式标注「不适用新库、不跑」**，避免留下不一致的独立实现被误用 |
| 7 | `motion_gates_model.oracle_visible` | 按 demo `min_real` 参数化（F 表 V5 适配） |
| 8 | ★ `summarize_eval_probe.motion_frames_formula` | **必改**：它是阻断级判据 `EVAL_K_FORMULA` 的期望侧（文件自述「本文件内独立重写 demo+exec 起点公式」），不改则下一轮评估直接 FAIL。本轮不跑，但口径必须同步 |
| 9 | ★ `hand_calc_8frame.expected_motion` | 必改（dataloader 手算参照） |
| 10 | ★ `eval_rhythm_gates.predict_k` | 必改（1300 步评估口径与 budget 的唯一守门，见 E 节） |

**全仓调用点裁决**（`★` = 初稿未提及；总计 `visible_motion_rows` ≥25 处、`max_visible_count` 5 处、`seg_num_grid`/`seg_num_chunks` 十余处，散在 **15 个文件**）：

| 符号 | 调用点 | 处置 |
|---|---|---|
| `build_index_entries`（8 处） | `pack_motion_store::cmd_pack` | writer：传**目标** spec |
| | `pack_motion_store::cmd_verify` | reader：传 `meta.spec` |
| | `motion_checks::cmd_a7` | reader：传 `--motion` 表的 spec |
| | ★ `scan_16task_memory` | 无表可读，**必须显式传规格**；本轮不跑，但签名改后不传即 TypeError |
| | ★ `test_guards`（3）、★ `motion_gates_online::_entry` | 单测/用例：新旧两套规格各一组 |
| `visible_motion_rows`（≥25 处 / 13 文件） | `framesamp_dataset`、`motion_checks::cmd_a9set`、`motion_store::max_visible_count`、★`compare_online_motion`、★`compare_siglip_replay`、★`compare_train_infer_obs`（2）、★`ref_npy_dataset`、★`eval_rhythm_gates`（2）、★`motion_gates_model`、★`motion_gates_online`（2）、★`mv_points`、★`scan_16task_memory`、★`test_guards`（10） | 签名不变，**随 `entry.spec` 自动生效**；只有 `test_guards` 的硬编码期望值要按新旧两套分组 |
| `max_visible_count`（5 处） | `framesamp_dataset`、★`ref_npy_dataset`、★`scan_16task_memory`（2）、★`test_guards` | 同上 |
| `seg_num_grid` / `seg_num_chunks` | `motion_store` 内部、`wan_common`、`extract_wan`、★`scan_16task_memory`、★`compare_online_motion`、★`motion_gates_online`、★`test_guards` | **纯公式调用，一律显式传 `min_real`** |

**A2 `scripts/dataset/wan/wan_common.py`**（子 venv 独立副本）
- 同名常量与 `seg_num_chunks(L, min_real=33)` / `seg_num_grid(L, min_real=33)`。
- `list_segments`：demo 项 `min_real = 17`、exec 项 `33`，每项增 `min_real`；**`num_chunks` 必须用该项 `min_real` 计算**（初稿只提了 `num_grid`，漏了 `num_chunks` → 会写出 `num_chunks=68` 与 `num_grid=6` 这种单文件内部就不自洽的 metadata），并断言 `num_grid == len(range(0, num_chunks, 16))`。

**A2b `scripts/dataset/run_local.py`**（初稿完全未列此文件）
- `aggregate_segments` 的 payload `schema` 2 → **3**，增 `demo_min_real_frames` / `exec_min_real_frames` / `demo_tail_pad` 三键（取自 `wan_common` 常量）**与 `raw_dir`**（A6 的三方断言要用它，现在 payload 里根本没有这个字段），逐段 segs 增 `min_real`。
- 消费端 `oracle_driver::cmd_vae` 的 `schema != 2` 改为「`schema ∈ {2,3}`；为 3 时三键必须与 oracle 自身口径相等，为 2 时必须是旧口径」。
- `aggregate_segments` 补两条集合级检查：逐段 `len(rows) == num_grid`、`rows` 的 `m` 集合 `== range(num_grid)`（现在完全不查，原样搬运）。
- **汇总 8 个 worker 的 `skipped` 打进 `STAGE_DONE`**（现行格式是 `STAGE_DONE stage={} workers={} items={} elapsed={:.0f}s`，**没有 `skipped` 字段**；`skipped` 只在 `encode_motion` 的 per-worker `WORKER_DONE` 上，而 B 节 Monitor 正则不含 `WORKER_DONE`）。

**A3 `scripts/dataset/wan/extract_wan.py::process_segment`**
```python
off = wc.GRID_STRIDE * m
real = min(wc.WINDOW_FRAMES, item["seg_len"] - off)
if real < item["min_real"]:            # exec 段 min_real=33 → 与现行「窗口越段」raise 数学等价
    raise RuntimeError(...)
window = frames[off:off + real]
if real < wc.WINDOW_FRAMES:            # 只有 demo 段能进这里
    last = frames[item["seg_len"] - 1:item["seg_len"]]
    window = np.concatenate([window, np.repeat(last, wc.WINDOW_FRAMES - real, axis=0)])
window = np.ascontiguousarray(window)  # (33,256,256,3) uint8；后续 sha / encode_chunk 不变
```
- `frames` 是**段内切片**，故 `frames[seg_len-1]` 对 demo 段而言全域帧号正是 `es − 1`，与 `pad_source_frame` 定义一致。
- `metadata.json` 每行增 `real_frames` / `pad_frames` / `pad_source_frame`（demo 补帧行为全域帧号 `es − 1`，demo 未补帧行与 exec 段一律 `null`）；`input_frames_sha256` 仍对最终 33 帧算；`METADATA_SCHEMA` 升 3；写入前核对 `num_grid` 与 `num_chunks` 互洽。
- **resume 纪律**：`wan_common::segment_outputs_complete` 只查字节数 / sha / JSON 可解析，**不查 schema 与字段**。exec 段规则不变 ⇒ 残留的旧 schema 2 的 exec 产物字节完全匹配、会被当成「已完成」跳过，其 metadata 永远缺三新字段，一路带到 a11 才以 KeyError 形态炸。处置：给该函数增可选 `expect_schema` / `require_keys`，**且** launch.md 里写死「新库建在空 `wan-latents/` 上，起跑前 `ls wan-latents | wc -l` 必须为 0」。

**A3b `scripts/dataset/wan/encode_motion.py`（修既有 bug，用户已拍板本轮修）**
- 现状：调用 `wc.segment_outputs_complete(out_dir, key + ".f32", …)`，该函数内部拼 `m = out_dir / f"{key}.metadata.json"`，于是去找 `<key>.f32.metadata.json`；而实际写出的是 `<key>.metadata.json` ⇒ **函数恒返回 False**，`and` 右边那个写对了的检查永远短路不到。`extract_wan` 用裸 `key` 调同一函数，三个后缀全对，所以 bug 只在 encode。
- 实证（统一引 `v1-store/logs/p3-post.attempt1-mixedcommit.log` 一份日志）：8 个 worker 的 `WORKER_DONE` items 合计 **4,529** 段，而 400ep 全库 `totals.segments = 600`（≈7.5×）；单 worker 跑到 `windows=6,799` ≈ 全库 6,832 的 **99.5%**，即每个 worker 几乎重编了整个库。
- 改法：完成性检查分别接受 binary 与 metadata 两个路径（或给 `segment_outputs_complete` 增 `bin_stem` / `meta_stem` 两参）；**取得 claim 之后、`unlink` 三件套之前**再复核一次（放反了等于没做）；并把输入 `wan-latents/<key>.bin` 的 sha256 与 metadata 里的 `input_latent_sha256` 绑定核对——**该字段在现行 schema 1 里已经写了**（初稿担心的兼容问题不存在），只需在 skip 判据里加读取与比对，全库 39 GiB 重算 sha ≈ 30 s 量级。
- **stale claim**：`try_claim` 的 claim 文件只在 `finally: release_claim` 时删，worker 被 SIGKILL 会留下 stale claim，该段之后永远没人认领且当场无人报错（到 `aggregate_segments` 的残留 claim 检查才以整段缺产物的 SystemExit 暴露）。续跑前先 `ls motion-tokens/_claims | wc -l` 必须为 0。
- 单测：`scripts/dataset/test_guards.py` 加一条——造两段假产物，断言 skip 生效。**这条单测是 bug 已修的唯一证据**：建库是全新库，`skipped=0` 与「bug 没修」时的输出完全一样，判定行本身零信息量。

**A4 `scripts/dataset/wan/oracle_driver.py`**
- **`expected_segments` 是 vae / encoder / aggregate 三个子命令共用的唯一枚举函数**（初稿只说改 `vae`，照字面改会让 D3 在编码前就 SystemExit：`cmd_encoder` 用 `len(s["starts"]) * CHUNK_BYTES` 校 latent 字节，新 demo 段多一块 latent 即不符）。改为 `expected_segments(manifest, *, demo_min_real, exec_min_real)`，**demo 项用 17、exec 项用 33**（绝不能用单一全局 `min_real`，否则 exec 段被按 17 切、整个 exec 表全错），三个子命令共用同一签名。连带更新：`cmd_encoder` 的字节判据、`row_map` 行数、`aggregate` 的重排顺序（**口径错会让行序静默错乱，比 FAIL 更危险**）。
- **口径参数必须在命令行显式注入**：三个子命令各加 `--demo-min-real` / `--exec-min-real`（**不给默认值**，漏传即 argparse 报错，不静默用旧口径）。
- **独立实现补帧**：`vae` 子命令独立重算起点并**独立实现**补帧（不 import `extract_wan` / `wan_common` 的切窗函数）。
- **抽样协议（用户拍板：只抽 VAE 前向）**：
  - 新增 `--sample-spec "padded:all,rest:0.10,seed:0"`。抽样判据用**确定式** `sha256(f"{seed}:{key}:{m}")` 前 8 字节取模，与分片数、遍历顺序、并发度全部无关；补帧窗（demo 段最后一个 m，可由 `seg_len` 直接推）恒抽中。改 seed 或比例即换库版本。
  - **段文件仍为 `num_grid` 行定长**，未抽中的行写零；**零抽中的段仍产出 `.bin` 与 `.bin.sha256`**，使 `cmd_aggregate` 现有的「段集合全等 + 每段文件必在」两条断言不必放宽（按 `rest:0.10` 逐窗抽样，exec 段平均 22 窗，一段都不中的概率 `0.9^22 ≈ 9.8%` → 约 157 个 exec 段会完全没有文件，不这样做 aggregate 直接 SystemExit）。**代价：oracle latent 与被测同大（39.2 GiB），已计入磁盘预算。**
  - **分片产物必须带分片名**：`sampled_windows.shard<i>of<n>.json`（现行分片一律走 `_shard_name`；8 个进程写同一个 `sampled_windows.json` + `atomic_write` 的 `os.replace` ⇒ 最后一个赢、抽样集合丢 7/8）。`aggregate --kind vae` 合并成 `sampled_windows.json` 并断言 `len(merged) == report["windows"]`；各片报告加 `sample_spec`，aggregate 用现成的 `_same_across` 校各片同 spec。
  - **四道廉价检查保持全量遍历、不受抽样影响**：段集合双向相等、逐段 `num_grid`/`seg_len`/`len(rows)`、逐行 `m`/`seg_offset`/`start_global_frame`、逐窗 `input_frames_sha256`（读 h5 本来就是整段读，成本低）；另加新增的三键全量核对。`vae_report.json` 增 `metadata_rows_checked` / `windows_encoded` **与 `raw_dir`**（A6 三方断言要用），**并把这两个计数打进 `ORACLE_VAE=DONE` 的 print**（现行 print 没有 `metadata_rows=` 字段），`cmd_aggregate` 跨片求和。
- `encoder` 子命令不抽样、全量。
- **耗时提醒**：`cmd_vae` 是**整段读** h5（`read_frames(..., s["start"], s["len"])`），与抽了几个窗无关；本库 1,192,918 帧 ≈ 218 GiB 一帧不少，1.2 h 的估计取决于 h5py 读速，留档里按实测回填。

**A5 `scripts/dataset/motion_checks.py`**
- **`_independent_visible` 必须按段分支改写**（初稿写「换公式」，字面照做会出错）：现行 demo 与 exec **共用**谓词 `if f + 32 <= t`。旧规则下 demo 行恒真；**新规则下 demo 补帧行 `f = es − r`（`r ∈ [17,32]`）⇒ `f + 32 ∈ [es, es+15]`，冷启动 `t = es` 时除 `r = 32` 外全为假 → 独立实现漏掉补帧窗 → `a9set` 在冷启动样本上必 FAIL**。而若把 32 统一换成 `min_real − 1`，demo 侧碰巧对、**exec 侧被放宽成错的**。正确写法：demo 行判 `s ≤ es − spec.demo_min_real`（与 `t` 无关），exec 行判 `f + (EXEC_MIN_REAL − 1) ≤ t`。**禁止从 `motion_store` import 任何公式或常量**——该函数的价值就在独立性，`min_real` 只从 `store_meta.demo_min_real_frames` 读数值。
- **`a9set` 改分层采样**：现行 `--n 500` 纯随机，而 F 表 V4 要求「样本集必须含每个 episode 的 `t = es`」（冷启动是唯一能区分正确谓词与错误统一谓词的样本，全库占比仅 `1600/605611 = 0.264%`，随机 500 条有 26.6% 概率一个都打不中）。加 `--cold-all` 分层参数，判定行改为 `A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0`。
- **`a10` 改为真正独立**：新增 `--manifest` 与 `--demo-min-real`，`want` 从清单的 `num_timesteps` / `exec_start_idx` 现算（`demo: len(range(0, max(0, es−(min_real−1)), 16))`、`exec: len(range(0, max(0, T−es−32), 16))`），`min_real` 取**命令行参数**、一律不读 `store_meta`（不符时单独报一条失配）；**`--expect-*` 写死** `--expect-rows 71316 --expect-exec 35403 --expect-demo 35913 --expect-episodes 1600`，不得从 `motion_index.json` 的 totals 现算（那样等号两侧同源、该半个判据恒真）。
- **新增 `a11`（补帧账，集合级而非只算总数）**：从 `episode_manifest.json` 独立枚举完整窗口集合 `{(段 key, m)}`，与 `wan-latents/*.metadata.json` 的实际 rows 做**集合相等**（双向差集）；断言每段 `len(rows) == num_grid`、`m` 集合 `== range(num_grid)`、`seg_offset == 16m`、`start_global_frame == seg_start + 16m`；demo 逐行 `real_frames == min(33, es − 16m)`、`pad_frames == 33 − real_frames`、补帧行 `pad_source_frame == es − 1`、**未补帧行 `pad_source_frame is None`**；exec 逐行 `real_frames == 33 and pad_frames == 0 and pad_source_frame is None`；并断言 **`es ≥ 17` 的集数 == 1600**；聚合 `metadata.json` 与逐段文件一致。判定行 `A11_PAD=PASS segs=3200 rows=71316 padded=1600 set_diff=0`（**段数是 3200 = 1600 集 × 2 段**，初稿写 6400 是错的，且与 B 节自己的 `items=3200` 打架）。
- **`a5` 不跑**，真实理由是**对照物在本环境不可得**：其默认路径 `--raw-dir-400ep` 与 `--mj-data-raw` 都写死在 `/data/hongzefu/...`，环境 B 下不存在（第 13 条环境 B 段）。它留下的「像素同源」空位由 B 节的 `INPUT_REANCHOR` 填补。
- **`a6` 拆分、不整条砍掉**：前半「新清单 40 条 vs 旧 400ep 清单同身份」只对 40ep 库有意义、不适用；**后半适用且重要**——四方 `manifest_sha256` 绑定 + `check_index_against_manifest` 逐 episode 五字段互校，是**唯一的建库期**双库同源闸。新增子命令 `a6set`（或给 a6 加 `--skip-legacy`），判定行 `A6_SAMESOURCE=PASS`。

**A6 `scripts/dataset/pack_motion_store.py`**
- **`gather_provenance` 的 `run_name` 硬编码必须改（本轮最致命的一条）**：现行返回体里写死 `"encoder": {"run_name": "wan-v8-filter10-72ep-a", …}`，而 `dataloader::_motion_gates` 会把它与新 YAML 的 `source_run` 解析结果逐字段比对（`run_name` / `checkpoint_name` / `epoch` / `state_key`）⇒ 换 encoder 后**训练起跑必 raise，且在建库全绿、留档写完、preflight 过后才炸**。改为从 `args.encoder_run_dir`（或新增 `--encoder-run-name`）推导，并断言「推导出的 run_name == ckpt 路径所在目录名」。
- `cmd_pack` 写三新键与新 `layout`；`cmd_verify` 的 `build_index_entries` 传 `meta.spec`。
- **`gather_provenance` 增两个字段**：`raw_dir`（绝对路径）与 `input_manifest_sha256`，把输入重锚绑进新库 provenance。
- `cmd_pack` 新增三方断言：`wan-latents/metadata.json`（A2b 新加该字段）、`oracle/wan-mj/vae_report.json`（A4 新加该字段）记录的 `raw_dir` 与 `meta/input_manifest.json` 的 `raw_dir` 相同。现在 `run_local.py --raw-dir` 与 `oracle_driver vae --raw-dir` 是两处独立传入、**无任何一处断言二者相同**，一次手误就会让 Wan 与 oracle 读不同目录而对拍照样通过。判定行并入 `PACK_MOTION_DONE`。

**A7 `scripts/dataset/wan/compare_wan.py`（初稿未列此文件，但 B 节要跑它）**
- 现行 `cmd_latents` 三处硬编码「oracle 是全量」：段集合双向相等、`.bin` 大小必须 `== ng * CHUNK_F32`、`for m in range(ng)`，末行还要 `compared == rep["windows"]`。按 A4 的「定长写零」方案，前两条自动满足；仍需：新增 `--sampled <sampled_windows.json>`，`for m in` 改为遍历该段的抽样 m 列表，`compared` 改为按抽样集合计数。
- **比较器必须自己按同一确定式规则重算整张抽样表**，只用 `sampled_windows.json` 做交叉核对（而非当作真值）——否则 sampler 漏抽、分片文件互相覆盖，comparer 跟随同一缩水清单会一起 PASS。并**独立重算补帧窗集合**（demo 段最后一个 m 由 `seg_len` 直接推），断言恰好 1,600 个且各比一次。

**A8 测试（落点纠正）**
- **`scripts/dataset/test_guards.py` 是 motion 网格公式唯一的 CPU 单测，初稿全文未提**；而初稿点名的 `scripts/training/tests/test_pack_guards.py` 601 行里**没有任何 motion 用例**（全是 framesamp），规格表用例放在那里是放错了文件。
- `test_guards.py` 至少四组用例按旧口径写死，须分成「旧 spec 组（沿用现有期望值）+ 新 spec 组（新期望值）」两套：
  - `test_wan_common_constants_match_motion_store`：`for L in range(0,1300)` 逐 L 比 `wc` 与 `ms` 两份实现——这是 wan 子 venv 与主 venv **互为对照的唯一守卫**，参数化后必须 `for mr in (17, 33)` 各扫一遍，否则新的 17 口径零覆盖。锚点补 `seg_num_grid(100,17)==6`、`seg_num_grid(100,33)==5`、`seg_num_grid(16,17)==0`、`seg_num_grid(17,17)==1`。
  - `test_motion_index_roundtrip_and_totals`、`test_visible_motion_rows_boundaries`、`test_wan_common_list_segments_matches_index`：旧组显式传旧 spec 保留原断言，新组按新规则给新期望。
- **新增离线边界用例表**（纯公式，不依赖真实数据；本库 `es` 全部 ≥ 100，全量对拍覆盖不到短 demo 边界）：

  | es | demo 窗数 | 末窗 真/补 |
  |---:|---:|---|
  | 0、16 | 0 | 无 |
  | 17 | 1 | 17 / 16 |
  | 32 | 1 | 32 / 1 |
  | 33 | 2 | 17 / 16 |
  | 66 | 4 | 18 / 15 |
  | 114 | 7 | 18 / 15 |

  每档再按 `t − es ∈ {0, 16, 31, 32, 33, 48}` 核可见窗集合；**构造夹具时让 exec 首帧像素 ≠ demo 末帧像素**，以捕获「误取第 `es` 帧而非第 `es−1` 帧」这一类差一错。**这张表必须驱动上面清单里的全部 10 处独立实现**（初稿只说四处，把最易写错的 `_independent_visible` 排除在外——它是 a9set 的对照侧、A5 刚重写，且每档都要有 `t = es` 这一行）。
- `a11` 与 pack 三键要有 writer→parser→verify 三段各自的负例（三键缺一 / 三方不同值 / 旧布局文件带新键）。

**A9 encoder 换型（用户 09-17 拍板）**
- 落点：`v1-store/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/{checkpoint_epoch_72.pt,config.yaml}`，源 `/scratch/hongze/MotionJEPA/runs/wan-full1600-filter2-b176x4-72ep-a/`。
- 实测锚点：ckpt `477,432,433 B`、sha256 `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca`；config `2,008 B`、sha256 `4a505440b7c5ff0f1b0d7ec4680800e9296f5ede1df622767be3d5c3aee4c9b4`。
- **全仓写死旧 run 名 / 旧 run 目录的清单与逐项裁决**（本轮必经路径标 ⚠）：

  | # | 位置 | 裁决 |
  |---|---|---|
  | 1 | ⚠ `pack_motion_store::gather_provenance` 的 `run_name` | **必改**，见 A6（不改则训练起跑必 raise） |
  | 2 | ⚠ `run_local.py::ENCODER_RUN_DIR_DEFAULT` | **必改或 B 节显式传 `--encoder-run-dir $ENC`**（`--expected-ckpt-sha256` 会跟 ASSETS_LOCK 换新 sha，用旧目录的 ckpt 对新 sha ⇒ `load_encoder` SystemExit） |
  | 3 | ⚠ `oracle_driver.py::ENCODER_RUN_DIR_DEFAULT` | 同上，D3 起不来 |
  | 4 | ⚠ `motion_client.py` 的 sidecar 默认 `--encoder-run-dir` | **必改**：两个构造点（`compare_online_motion.py`、`policy_config.py`）都只传 `expected_ckpt_sha256`、不传目录 ⇒ V-online 与所有 eval 握手失败。建议给 `MotionEncoderClient` 加「从 YAML `source_run` 推导 run 目录」的路径 |
  | 5 | `paths.sh` 的 `readonly ENCODER_RUN_DIR` / `ENCODER_CKPT` / `v1_require_encoder` | 列入改动清单（B 节自定义 `ENC=` 绕开了它，不会立刻炸，但 `readonly` 让 runner 无法覆盖） |
  | 6 | `extra_checks.py`、`probe_wan.py` | 本轮不跑，显式标注「仍指旧 run，不适用新库」 |
  | 7 | `scripts/motion-variance/eval_shard_mv.sh` | 同上 |

- **`ASSETS_LOCK.json` 要改的字段**（初稿清单不全）：`motionjepa_ckpt` / `motionjepa_config` 两条的 `dest`、`bytes`、`sha256`、`headtail`、`related.run_name`、**`related.mj_train_commit`**（旧值 `7388a42` 是训练旧 ckpt 的 commit，新 ckpt 由不同 commit 训出）、**`source.filename`**（含旧 run 名），以及顶层 `sha256` 重算。
- **⚠ 换锚点会打挂一条现存单测**：`scripts/assets/test_assets_lock.py::test_lock_cross_checks_motion_store_provenance` 断言「40ep 库 `store_meta.json` 里记的 `encoder.checkpoint_sha256` == `expected_sha256("motionjepa_ckpt")`」，而 40ep 库记的是旧 ckpt 的 `bae96037…`，skipif 条件（40ep 库存在）在本机为真 ⇒ **该测试会跑且必 FAIL**。处置：把这条交叉互证改成「按 `related.run_name` 分档」或「对历史库用历史锚点」，并在步骤 3 的验证里点名跑一次 `test_assets_lock.py`。
- **HF 上传缺口**：`source` 段指向 HF 私有仓 `HongzeFu/MotionJEPA` 的 revision + tag，**新 ckpt 尚未上传**。本环境有本地原件不影响建库，但「异地无 NFS 机器从零复刻」路径会断；**另立一项补上传，不阻塞本轮**。同时注意旧 ckpt 一旦失去 lock 条目，所有仍指旧目录的消费者（上表 5/6/7）会同时失效——这些本轮都不跑，但要在留档里写明。
- 新 YAML 的 `source_run: wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder`。
- `scripts/dataset/wan/SOURCE_PIN.json` **不动**：它钉的是推理脚本 `wan_motion_infer.py` 的 sha（`af67fdd9…`），与 ckpt 无关；已核 MJ 活仓 HEAD 虽已前进到 `f43b38f`，该脚本 sha 仍与复制件逐字节相同，D3 的 `module_sha256` 比对不受影响。`paths.sh` 里 `readonly MJ_COMMIT` 指 `2a484ad9`（推理侧钉版），与活仓 HEAD 是两回事，留档里要区分开。

### B. 建库命令序列（主副本，clean HEAD）

每个 tmux 阶段都走同一个 runner 外壳（第 7 条）：`set -o pipefail` + `PYTHONUNBUFFERED=1` + `tee` + `rc=${PIPESTATUS[0]}` + 收尾 `printf 'END_UTC=%s\nEXIT_CODE=%s\n'`，runner 落 `v1-store/logs/mv2-<stage>-runner.sh`（不进 git）。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA; source scripts/dataset/paths.sh; v1_prepare_dirs
LIB=$V1_STORE/datasets/4task-v2-1600ep-604f16da; RAW=$V1_STORE/raw-h5/4task-20260912-v2; MJ=/scratch/hongze/MotionJEPA
ENC=$V1_STORE/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a
SHA=0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca
ls -ld $LIB $LIB/meta $RAW $ENC; df -h /scratch      # 实体目录；Avail ≥ 400 G
BUILD_HEAD=$(git rev-parse HEAD); git status --porcelain   # 必须为空
ls $LIB/wan-latents 2>/dev/null | wc -l                    # 必须为 0（A3 的 resume 纪律）
ls $LIB/motion-tokens/_claims 2>/dev/null | wc -l           # 必须为 0（A3b 的 stale claim）
```

**第 1 步 输入重锚**（≈15 min，791.8 GB）——runner 内用一次性 `uv run python -c` 调 `finalize_checks.check_inputs(manifest, raw_dir, input_manifest, level="sha256")`（纯函数、返回 errs、不写任何文件、不拉 jax），**绝不重跑 `finalize_checks.py check`**：后者在全通过时会**覆写** `<lib>/source/meta/provenance.json`（`git_commit` 被换成 motion 建库时的 commit，`finalize_host` / `jax` / `gpu_device_kind` 一并重写），等于用这次核验的指纹污染帧库的建库留档。判定行 `INPUT_REANCHOR=PASS files=4` 由该 wrapper 打印，不新增仓库文件。

```bash
# 第 2 步 Wan（tmux mv2-wan；日志 $LIB/logs/mv2-wan.log）
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1,2,3,4,5,6,7 --raw-dir $RAW
# 红线：wan 与 oracle vae 必须显式传 --raw-dir $RAW。paths.sh 的 RAW_H5_DIR 是普通赋值、没有 export，
#       子进程看不到；而 run_local.py 在环境 B 的缺省是 /scratch/hongze/robomme_data_h5 —— 该目录
#       **真实存在**（v1 旧库，同名不同内容的 16 任务 h5），误用不会「路径不存在」而是在 episode_{j}
#       KeyError 处失败。最终兜底是 A6 的 raw_dir 三方断言。
# 第 3 步 encode（tmux mv2-encode）——与 wan 之间零 commit
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB --gpus 0,1,2,3,4,5,6,7 \
  --encoder-run-dir $ENC --expected-ckpt-sha256 $SHA
# 第 4 步 pack / verify（tmux mv2-pack）
uv run --no-sync python scripts/dataset/pack_motion_store.py pack   --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion --encoder-run-dir $ENC --raw-dir $RAW
uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store $LIB/motion --resume
# 第 5 步 oracle（MotionJEPA venv，--mj-repo $MJ 是顶层参数、必须在子命令之前；tmux mv2-oracle）
#   三个子命令都必须显式传口径（--demo-min-real / --exec-min-real 无默认值，漏传即报错）
CUDA_VISIBLE_DEVICES=7 … oracle_driver.py --mj-repo $MJ encoder --manifest $LIB/meta/episode_manifest.json --latents $LIB/wan-latents --out $LIB/oracle/wan-mj \
  --encoder-run-dir $ENC --expected-ckpt-sha256 $SHA --demo-min-real 17 --exec-min-real 33
for i in 0..7: CUDA_VISIBLE_DEVICES=$i … oracle_driver.py --mj-repo $MJ vae --manifest … --raw-dir $RAW --latents $LIB/wan-latents --out $LIB/oracle/wan-mj \
  --shard-idx $i --num-shards 8 --demo-min-real 17 --exec-min-real 33 --sample-spec "padded:all,rest:0.10,seed:0"
… oracle_driver.py aggregate --manifest … --out $LIB/oracle/wan-mj --num-shards 8 --kind vae --demo-min-real 17 --exec-min-real 33
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents $LIB/wan-latents --oracle $LIB/oracle/wan-mj --sampled $LIB/oracle/wan-mj/sampled_windows.json
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens  --store $LIB/motion --oracle $LIB/oracle/wan-mj
# 账目 a6set / a7 / a9set / a10 / a11（a10 的 --expect-* 写死，不取 motion_index.json）
```

> B 节落地时必须展开成完整 runner（真实 interpreter、绝对路径、八片并发启动与逐片退出码收拢）：**全部分片成功才允许 aggregate，aggregate 成功才允许 compare**；不能只靠日志过滤器看到一个 PASS 就继续。

Monitor 过滤管道（每级行缓冲）：`tail -n +1 -F <日志> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E 'STAGE_DONE|STAGE_FAIL|WORKER_DONE|INPUT_REANCHOR=|PACK_MOTION_DONE|VERIFY_MOTION=|ORACLE_.*=DONE|BITEXACT=|A1[01]_|A[679]_|Traceback|out of memory|EXIT_CODE='`。

判定行：`INPUT_REANCHOR=PASS files=4`、`STAGE_DONE stage=wan workers=8 items=3200 skipped=0 elapsed=…s`、`STAGE_DONE stage=encode workers=8 items=3200 skipped=0 elapsed=…s`（`skipped=` 是 A2b 新加到 `STAGE_DONE` 的字段；**注意首次全新建库 `skipped=0` 与 bug 未修时输出相同，bug 已修的唯一证据是 A3b 的单测**）、`PACK_MOTION_DONE=1`、`VERIFY_MOTION=PASS scanned=71316 mismatches=0`、`ORACLE_ENCODER=DONE rows=71316`、`ENCODER_BITEXACT=PASS compared=71316 mismatches=0`、`ORACLE_VAE=DONE metadata_rows=71316 windows=<抽样数> frame_mismatches=0 metadata_mismatches=0`（`metadata_rows=` 是 A4 新加到 print 的字段）、`WAN_BITEXACT=PASS compared=<抽样数> padded_covered=1600 frame_mismatches=0 latent_mismatches=0`、`A6_SAMESOURCE=PASS`、`A7_BYTES=PASS`、`A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0`、`A10_ROWS=PASS rows=71316 exec=35403 demo=35913 episodes=1600`、`A11_PAD=PASS segs=3200 rows=71316 padded=1600 set_diff=0`。

留档 `docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/{launch.md,result.md,records/}`，`records/` 只放清洗后日志与判定行，不放 `.sh` / `.yaml`。

### C. dataloader / 配置

- **`src/mme_vla_suite/training/framesamp_dataset.py::__init__`**
  - `_req(int(mcfg.budget) % 16 == 0 and int(mcfg.budget) >= 16, …)`（160 = 10 × 16 通过）。
  - **两键的三态判定（用户拍板：reader 侧默认值兼容）**——必须用 `mcfg.get(k, None)`，**不能用 `mcfg[k]` 或属性式**：实测 omegaconf（非 struct 模式）上前者抛 `ConfigKeyError`、后者抛 `ConfigAttributeError`，都会在比对之前先炸。
    ```
    want = self._motion_meta.spec
    got_min, got_pad = mcfg.get("demo_min_real_frames", None), mcfg.get("demo_tail_pad", None)
    两键同缺 → 视作历史契约 (33, "none") 并与 want 比对   # 旧 YAML + 旧表照跑；旧 YAML + 新表必 raise
    只缺一键 → raise（配置半残）
    两键齐备 → 必须与 want 逐值相等，否则 raise
    ```
    **位置硬约束**：这段必须写在 `self._motion_meta = ms.MotionMeta.load(...)` **之后**，不能并进现有那段 `_req` 块（那时 `self._motion_meta` 还是 `None`，取 `.spec` 会 AttributeError）。若要严格区分「键不存在」与「键存在但值为 null」，用 `k in mcfg` 判存在性而非 `.get` 的默认值。
  - `__getitem__` 调 `ms.visible_motion_rows(entry, step)`（签名不变，spec 随 entry）；`__init__` 的零截断预检 `ms.max_visible_count(e)` 同理。
  - 三处「32 帧 / 96 / 608」注释改按变量描述。注意 `percep_mem.py` 里还有一个同名但不同义的 `self.config.budget`（帧路 512 位），注释里要区分这两个 budget。
- ★ **`scripts/training/tests/ref_npy_dataset.py`**：硬断言 `(self._motion_budget, self._motion_pos_dim, int(mc.stride), int(mc.window_frames)) != (96, 256, 16, 33) → raise`。它是 V1（`dump_fixture_samples` import 它）与 V8（`BENCH_DATASET_IMPL=refnpy`）的参考侧，**budget=160 会直接 ValueError**。**改法不能写成 `(mcfg.budget, 256, 16, 33)`**——`self._motion_budget` 本来就等于 `int(mc.budget)`，那样 budget 那项变成自比恒真、守卫失效。正确写法：`self._motion_budget % 16 == 0 and self._motion_budget >= 16`，其余三项 `(pos_dim, stride, window_frames) != (256, 16, 33)` 照旧 raise，与 `framesamp_dataset._req` 同口径。
- `src/mme_vla_suite/training/dataloader.py::_motion_gates`：逻辑不变。注意 `root = os.environ.get("MMEVLA_MOTION_STORE") or str(mcfg.store_path)`，相对路径按 `_REPO_ROOT`（不是 cwd）解析，所以 YAML 写相对路径安全；生产 runner 保持该变量 unset ⇒ **motion 表路径的唯一来源是新 YAML 的 `store_path`**，而 `store_path` 参与 `history_config.resolved.sha256`，改路径即改 YAML sha，preflight 的 `--history-config-sha256` 要同步更新。
- **`src/mme_vla_suite/training/config.py`**
  - `RepackTransform` 注释改 `(b, budget, …)` / `(b, 512 + budget)`。
  - `_CONFIGS` 新增 `mme_vla_suite_b128_80k` = 复制 `mme_vla_suite_b128_60k`，**只改** `num_train_steps=80_000`、`decay_steps=80_000`；注释写明唯一差异与「因 `peak_lr == decay_lr == 5e-5`，两条配置的 lr 在**任意 step** 逐步相同」。文件末尾的名字唯一性断言与 `tyro` 精确 subcommand 匹配保证新增条目不影响既有三条。
- **新 YAML** `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml` = 基线 `perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，一字不动）+ `motion` 节：`enabled: true`、`budget: 160`、`demo_min_real_frames: 17`、`demo_tail_pad: repeat_last`、`store_path: v1-store/datasets/4task-v2-1600ep-604f16da/motion`、`source_run: wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder`，其余（`dim 768` / `stride 16` / `window_frames 33` / `window_direction forward` / `grid_origin segment_start` / `pos_dim 256` / `frame_size 256` / `online_gpu`）照抄 `perceptual-framesamp-context-8frame-8x8-motion.yaml`。
  - ★ 备注：`scripts/training/tests/g0_gate.py` 对关闭态 YAML 的 `motion` 节做**精确集合相等**校验。本轮新 YAML 是开启态、不受影响；但日后若往关闭态 YAML 也补这两个新键，那条断言会挂。
- **budget 复核式（评估口径变更时用）**——初稿两处都错，修正后：
  ```
  τ_max = es + 16·⌊max_steps/16⌋                 # 不是 ⌊(max_steps−4)/16⌋
  E(δ)  = 0 if δ < 32 else (δ − 32)//16 + 1      # 等价于 seg_num_grid(δ+1, min_real=33)，直接复用 A1 的函数更稳
  k_eval = demo_num_grid(es, min_real=17) + E(τ_max − es)
  ```
  `max_steps = 1300` 时 `τ_max − es = 1296`、exec 项恒 80，新库 `k_eval` 最大 151 ≤ 160；16 任务全集口径复核同为 151。
- `scripts/training/compute_norm_stats.py::_NONE_KEYS` 不变；norm_stats 沿用新库 `856c75ea…`，**不得为开启态重算**。
- ★ **写死 `96` / `608` 的代码清单**（非注释，逐项裁决）：
  - **必改（本轮跑到）**：`motion_gates_model.py`（`MOTION_BUDGET = 96` 及 m3/m4 里的 32/16/96/608 字面量，V5 的工具本身）、`ref_npy_dataset.py`（见上）、`motion_gates_online.py`、`eval_rhythm_gates.py`（见 E）。
  - **必改（本轮不跑但含独立网格公式，口径不同步会让下一轮直接 FAIL）**：**`summarize_eval_probe.py`** —— 不只是 `--budget` 默认 96，它的 `motion_frames_formula` 是 `EVAL_K_FORMULA`（**阻断级判据**）的期望侧、内含旧 demo 谓词 `while s + (window−1) <= es−1`；**`hand_calc_8frame.py::expected_motion`** —— `[s for s in range(0, es, 16) if s + 32 < es]`，dataloader 手算参照。
  - **点名但本轮不动**：`mv_common.py`（`MEM_LEN, FRAME_SLOTS, MOTION_SLOTS = 608, 512, 96`，motion 利用率链路）、`scan_16task_memory.py`（`MOTION_BUDGET = 96`）。
  - **纯注释**：`history_observation.py`、`config.py`、`percep_mem.py`、`history_pi0.py`、★`robomme_policy.py`（四行 `(96,768)`/`(608,)` 注释）。

### D. 模型侧

- **`history_pi0.py::HistoryPi0.__init__`**：`if self.mem_encoder.motion_enabled and self.integration_type not in ("context", "modulation"): raise`。全仓 grep 确认这是 `integration_type` 与 `motion_enabled` 联用的**唯一**断言，「就这一处」成立。另一条相关守卫 `framesamp_dataset` 里已同时允许 `("modulation", 1024)`，无需改。
- `history_observation.py`、`percep_mem.py`、`robomme_policy.py` 只改注释。**特别点名**：`percep_mem.py` 里 `motion_encoder_static` 的中文注释写的是「1536 → 2048，W 1536×2048 + b 2048」——那是 context 版（`memory_token_dim: 2048`）的数字，modulation 下是 **1024**，本轮必须改。
- **motion 四个参数叶**（从模块定义推算，供 V9 断言）：`mem_encoder/motion_pos_proj/kernel (256,768)` + `bias (768,)`、`mem_encoder/motion_encoder_static/kernel (1536,1024)` + `bias (1024,)`，合计 `1,771,264`。基线实测 61 叶 + 4 = **65**。6 条 modulation 叶的来源已定位：`MemoryAttention` 的 `q_einsum_mem` / `kv_einsum_mem` / `out_einsum_mem` / `mem_rms_norm.scale` 共 4 条，加 `HistoryBlock` 里 `MemoryRMSNorm(name="mem_rms_norm_ffn")` 带 cond 分支产生的 `Dense(2*width)` kernel+bias 2 条 ⇒ `MEM_PARAMS` 从 `n=6` 变 `n=10`。
- `scripts/training/train.py::init_history_config`：`motion_provenance.json` 的字段按 `enabled` 填、与 integration 无关，**预期无需改**（已核无 modulation 专属漏写）。
- **两个参数树工具的开关名不同，别抄串**：`scripts/training/legacy-eval/check_ckpt_param_tree.py` 用 `--config`（默认 `mme_vla_suite`，须显式带 `mme_vla_suite_b128_80k`），只产出 `PARAM_TREE_EXACT`，且读的是**仓库** YAML、用文件名字符串建模型，完全绕开 run 内 `history_config.resolved.yaml` 快照 ⇒ 它证明不了「部署加载已验证」。走生产口径的是 `scripts/training/g0/check_config_provenance.py::gate_ckpt_param_tree`（用 `_load_resolved_snapshot` + `model.load(remove_extra_params=False)` + `_assert_param_tree_exact`），开关名是 **`--train-config`**，判定行 `CKPT_PARAM_TREE=PASS missing=0 extra=0 leaves=<n> model_leaves=<n> ckpt_has_motion=1`。**它的 `--ckpt/--lib/--train-config/--store-subdir/--neg-lib` 默认值全部指向旧库旧 run，V9 必须逐个显式传**（见 F 表）。
- **budget 不进参数树，参数树校验对 budget 完全失明**：四个 motion 叶的形状只依赖 `pos_dim/dim/pos.hidden_dim/memory_token_dim`，与 budget 无关 ⇒ 拿 budget=96 的 YAML 加载 budget=160 训的 ckpt，`PARAM_TREE_EXACT` 与 `_assert_param_tree_exact` 都会 PASS 而语义已静默改变。**V9 必须显式加一条 `BUDGET_CONSISTENT`**：从 `history_config.resolved.yaml` 读出的 `motion.budget` == 160 == dataloader 实际使用值 == 在线 `_motion_cfg["budget"]`。

### E. 在线侧

- **`src/mme_vla_suite/policies/framesamp_memory.py`**
  - `motion_cfg` 增 `demo_min_real_frames` / `demo_tail_pad`，读取一律用 `.get(k, 历史默认)`。
  - **demo 谓词 `+ (W−1) <= es − 1` 在本文件出现 4 次（不是初稿写的 3 处）**：`_encode_ready_windows` 的 demo `while`、其后 `keep_from` 收缩的 `if`、同一条件在列表推导里的**第二份拷贝**、`visible_motion_frames` 的 demo `while`。四处全改为 `+ (demo_min_real − 1) <= es − 1`，并把重复的两处抽成局部变量 `demo_unfinished` 消掉拷贝。exec 的两处谓词不动。
  - `_encode_window(f)`：对 `f < es`（等价于「这是 demo 起点」，因 demo 起点 `s ≤ es−17 < es`、exec 起点 `≥ es`）取 `[f, min(f+32, es−1)]` 并 `np.repeat` 第 `es−1` 帧补齐。边界已核：`es = 0` 时 demo 窗数为 0、分支永不进入；`es = 17` 时 `[0, 16]` 真 17 补 16；满窗时 `min(f+32, es−1) = f+32`、补 0 帧。
  - `_raw_frames` 清理语义**确认不变**：新规则下 demo 仍在首批一次编完 ⇒ `keep_from = es` ⇒ 删光 `k < es`，与旧规则同结果；补帧源帧 `es−1` 的存活性成立（`_encode_ready_windows` 在 `add_buffer` 写完 `_raw_frames` 之后才调，清理发生在编码之后）；`max_buf ≤ 32+16+1` 的上界不变。
  - ★ **新增 fail-loud 校验**：仿照该文件已有的 `window_direction/grid_origin` 守卫，对两键做同形式 raise（`demo_tail_pad` 必须 `repeat_last`；`demo_min_real_frames` 须是 `1 ≤ x ≤ window_frames` 的整数）。理由：训练侧的 `_req` 在 `FrameSampDataset.__init__`，而 **eval 根本不构造 dataset** —— 不加这道闸，YAML 写错口径会变成「训练 raise、在线照跑」，训推口径不一致而无人发现。
- **`src/mme_vla_suite/policies/policy.py::__init__`**：`_motion_cfg` 的 8 键字典推导增两键，且这两键用 `mcfg.get(k, 历史默认)`（旧 YAML 与被哈希冻结的历史 run 快照都缺它们，**快照不能补键**）。
- ★ **同型写死的 `motion_cfg` 生产者共三处（另有一处吃整节），必须同改**：`policy.py`、`scripts/training/g0/compare_online_motion.py`（同样的 8 键推导——所以初稿写的「脚本本身逻辑不变」是错的）、`scripts/training/tests/motion_gates_online.py::MOTION_CFG`；`scripts/training/tests/online_mem_regress.py` 直接吃旧 YAML 的整节，随 reader 侧默认值兼容自动可用。
- **两处 stub 帧号校验放宽（「sidecar 不改」的唯一例外）**：`motion_gates_online.py::_stub_enc_local` 与 `scripts/dataset/wan/motion_sidecar.py` 都断言 `ids == list(range(start, start+33))`；`stub_decode` 把全域帧号写进像素、逐帧解回，补帧窗尾部会解出重复帧号 ⇒ 前者 assert 失败、后者 `return 4` 退出，还会与 P1 的「错帧→rc=4」用例混同。放宽为「连续真实前缀 + 合法重复尾」：存在 `r ∈ [demo_min_real, 33]` 使 `ids[:r] == range(start, start+r)` 且 `ids[r:]` 全等于 `ids[r−1]`；**继续拒绝**内部错帧、`r < demo_min_real`、尾部重复的不是 `ids[r−1]`、以及 exec 段出现任何 `r < 33`。**这会改 `protocol_sha256`，P1 的握手断言与既有 provenance 记录须一并更新。** 注意 `scripts/training/tests/eval_rhythm_gates.py` 直接复用 `G._stub_enc_local`，会跟着生效。
- **`scripts/training/tests/motion_gates_online.py`**：`MOTION_CFG` 从 `--yaml` 的 `motion` 节整体读入（不再硬写字面量，与 `compare_online_motion` 同一份键列表）；所有 `96` / `608` / `(96,768)` 断言改用 `MOTION_CFG["budget"]` 与 `HIST_BUDGET + budget` 表达（8×8 档还须 `TOKEN_PER_IMAGE=64`，现文件写死 `512, 16, 1`）；`--yaml` 分支现在只更新帧路四个量、必须一并更新 `MOTION_CFG`。**超预算用例失效需重造**：原用例取 `es = 1600`，新规则下 demo 窗数 99（`⌊1583/16⌋+1`）< 160，`_prepare_motion` 不再 raise ⇒ 改用**专门的小 budget 实例**（如 budget=8 + es=200），不要靠拉长 es 去撞 budget。`assert mem2.motion_encode_calls == ms.seg_num_grid(es) > 96` 同时踩「公式默认值」与「96 字面量」两颗雷，一并改。P2/P4 用例加「es=114 首批编 7 窗、第 7 窗补 15 帧」断言。
- ★ **`scripts/training/tests/eval_rhythm_gates.py`（初稿完全没提这个文件）**：`predict_k` 里写死 `demo = ((es-33)//16 + 1) if es >= 33 else 0`，新规则应为 `((es-17)//16 + 1) if es >= 17 else 0`；`BUDGET = G.MOTION_CFG["budget"]` 随 budget 改 160 一起变，于是文件里写死的 `es=288/289` 边界失效——**新边界由计划自己的式子直接算出：`k > 160 ⇔ demo > 80 ⇔ es ≥ 1297`，故 `288/289` 改为 `1296/1297`**，与第六节「充要条件是评估 `es ≤ 1296`」自洽。该文件用的 `tau = es + 1296` 本来就与修正后的 `τ_max` 一致。**注意它对边界两侧各跑一遍完整 1300 步 eval 控制流，`es=1297` 意味着要缓冲 1297 帧 256×256×3 原始帧（≈255 MB），须先估内存。** 它是 1300 步评估口径与 budget 的唯一守门，必须和 budget=160 的结论一起重算。
- **`scripts/training/g0/compare_online_motion.py`**：读 YAML `motion` 节透传两键；`visible_motion_rows` 调用随 `entry.spec` 自动生效；`first_batch_windows` 的 `ms.seg_num_grid(es)` 必须显式传 `min_real=17`（否则静默少记 1 窗/集）；**必须传 `encoder_run_dir`**（见 A9 第 4 条）。另见 F 节 V-online 行的加固与分片改造。
- `motion_protocol.py`（除协议 sha 随 stub 变更）、`motion_client.py` 的**数值逻辑**不改，但其默认 encoder 目录必须改（A9）。

### F. 验证与判定行

> **判定行已两轮与工具实际输出对表。** 初稿有 7 个判定行名在代码里根本不存在：其中 `DS_EQUIV` / `M4_MASK` / `A8_ROWS` / `A9_SET` / `MPOS` 已分别改用工具真实输出的 `SAMPLE_RAW_EXACT` / `MASK_INVARIANCE` / `A10_ROWS` / `A9_INDEXSET` / `ONLINE_POS`；`AA_100` 与 `LEGACY_*` 保留为本轮新写的汇总行，**已在下表逐个指派产出者**。**第 18 条的两块划分**：第一块（非训练轻量对拍）= V1/V2/V4/V-online；第二块（本机训练梯度一致）= V6/V7/V8。

| 编号 | 内容 | 工具与关键参数 | 判定行（工具真实输出） | 顺序步 |
|---|---|---|---|---|
| V0 | lr 曲线逐步相同 | 新写数行 `uv run python -c`：`config.lr_schedule.create()` 在 `range(0, 80_000)` 上逐步比 60k 与 80k 两条配置（秒级，放 2b 的 CPU 单测里） | `GUARD_LR=PASS steps=80000 mismatches=0` | 2b |
| V1 | 关闭态 dataset 交付逐位（改前 vs 改后） | `dump_fixture_samples.py` ×2 + `compare_fixture_dumps.py`；**数据集必须用 `4task-motion-400ep/framesamp-8x8`**（新库 `exec_start_idx` 全 ≥100，`fixture_per_step` 找不到 `es=0` 的 episode 必 raise）；环境 `DTYPE_DUMP_DIR`（两侧各一空目录）/`DTYPE_DUMP_GIT_HEAD`/`DTYPE_DUMP_MODE=both`/`DTYPE_DUMP_ARRAYS=1`/**不设** `DTYPE_DUMP_LIMIT`/`JAX_PLATFORMS=cpu`；白名单见下 | 单侧 `DUMP_DONE samples=3200 batches=200 out=<dir>`；对拍 `SOURCE_IDENTITY=PASS episodes=400 samples=101066` / `FRAME_INDEX_EXACT=PASS … max_frames=8` / `SAMPLE_RAW_EXACT=PASS samples=3200 per_step=200 mismatches=0` / `BATCH_RAW_EXACT=PASS batches=200 mismatches=0`。失败是抛异常 + 退出码 1，无 FAIL 行。**该比较器不比较两侧 `DUMP_MANIFEST.json` 的 `git_head`**，须人工写进留档或 `jq` 断言 | 2a/2d |
| V2 | 旧表 + 旧 YAML + 历史快照三重回归 | **新写 `scripts/training/tests/legacy_regress.py`**（三条判据同一脚本产出）：① `MotionMeta.load` 两张旧表（40ep rows=772、400ep rows=6832）且解析出 `(33,"none")`；② **未经修改的** `perceptual-framesamp-context-motion.yaml` + 40ep 旧表构造 dataset，前 N 样本四键逐位不变；③ 历史快照 `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/5000` 走 `create_trained_policy(..., **motion_stub=True**)`（**不加这个参数会真去拉 sidecar、占 GPU、加载 40ep encoder**；该目录**不得改动任何文件**） | `LEGACY_LAYOUT=PASS` / `LEGACY_YAML=PASS` / `LEGACY_POLICY=PASS` | 2d |
| V6 | 关闭态定点梯度（改前 vs 改后） | `single_step_grad_fixed.py` ×2 + `compare_fixed_grad.py`；**`DTYPE_BATCH_FIXTURE_DIR=v1-store/fixtures/8x8/grad/c8-b`**（8 帧档、12 个数组键 + 4 个 `none` motion 键，满足工具「必须恰好 12 数组键」的断言；`m8-*` 是 16 键会 raise）；`--batch-size 8 --fsdp-devices 2 --seed 42 --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml`；`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`；`DTYPE_GRAD_DIR` 预先不存在；`unset DTYPE_BASELINE_CHECKSUMS`。**该工具白名单已放行 modul YAML，无需改** | 单侧 `FIXTURE=PASS kind=… keys=12 batch=8`（×3）/ `INIT_PARAMS=PASS leaves=<N>` / `GRAD_KIND=PASS kind=…`（×3）/ `GRAD_DONE kinds=3`；对拍 `INIT_EQ=PASS leaves=<N> mismatches=0` / `GRAD_EQ=PASS kinds=3 leaves=<N> mismatches=0` | 2a/2d |
| V7 | 关闭态 100 步守卫 | `bench_train_steps.py` **两侧各跑一次**（基线 60k run 的 `metrics.jsonl` 只有每 100 步一行，且无 `index_sequence.json`/`batch_digests.jsonl`/`param_checksums.jsonl`/`env.json`，**不能复用**）；**参数按正本 `docs/motion-memory.md` 的既有 100 步口径：`SAVE_INTERVAL=25` + `EXTRA_DIGEST_STEPS=99`**（初稿写的 `[0,1,2,24,49,99]` + `rows=6` 三重矛盾、任何参数下都产不出：state 族与 batch 族触发规则不同，batch 族恒含 `{1,2}`，且单一 interval 无法同时命中 24 与 49）；2 卡 b8 / fsdp 2 / seed 42 / WORKERS 4 / 确定性 flag；白名单见下；约 30–35 min × 2，tmux `mv2-bench-base` / `mv2-bench-cand`，**按第 17 条留档** | `BASELINE_ENV=PASS` / `SCALARS steps=100 keys=5 hex_mismatch_steps=0` / `INDEX_SEQ=PASS n=800` / **`STATE_DIGEST rows=5 mismatch=0`** / **`BATCH_DIGEST rows=7 mismatch=0`** / **`CANON_CHECK=PASS steps=7`** / `DET_CHECK=…`（摘要步 `{0,25,50,75,99}`、输入摘要步 `{0,1,2,25,50,75,99}`，与正本同口径），汇总 `GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5`（汇总行由本轮新写的 `finish_check.py` 拼出） | 2a/2d |
| V3 | 新表：输入重锚 / D3 全量 / D2 抽样 / a6set a7 a9set a10 a11 | B 节 | 见 B 节判定行清单 | 3（重锚）/ 4（其余） |
| V4 | 开启态 dataloader 交付 | `motion_checks.py a9set --cold-all` + 新增 `scripts/training/tests/dataloader_motion_check.py`。**样本集必须分层构造、不能纯随机 500**：实算命中概率——冷启动 `t==es` 与首个 exec 窗 `t==es+32` 各 0.264%（500 条打不中的概率 26.6%）、**`k≥140` 全库只有 32 个样本（0.0053%，97.4% 概率完全打不中）**，而 budget=160 的全部理由就是「最大 141」 | `A9_INDEXSET=PASS samples=2100 cold=1600 mismatches=0`；新工具 `V4_COVER=PASS cold=<n> first_exec=<n> kmax=141 pad_min=1 pad_max=16` | 6 |
| V5 | 模型侧 mask 正确性（modulation） | **用户拍板：在 GPU 上用真实配置跑**（真 1024 宽、真 YAML、真库），不新增测试变体、不改 `utils.get_config`。理由见下方「V5 为何不能用 CPU dummy」。tmux `mv2-m4`，占 1 张 GPU，**按第 17 条留档**；工具适配见下 | `MASK_INVARIANCE=PASS` / `GRAD_LEAK=PASS` / `ORDER_EFFECT=PASS` / **新增 `PAD_CONTENT_INVARIANCE=PASS`**；**`ZERO_MOTION_EQUIV` 在 modulation 下不适用，标 `LEN_EQUIV_NA` 并写明原因** | 6 |
| V8 | 开启态 A/A 100 步可复现 | 同 V7 工具链 ×2（同一侧跑两次），tmux `mv2-aa`，**按第 17 条留档** | `SCALARS steps=100 keys=5 hex_mismatch_steps=0` / `INDEX_SEQ=PASS n=800` / `STATE_DIGEST rows=5 mismatch=0`，汇总 `AA_100=PASS hex_mismatch_steps=0`（汇总行本轮新写） | 6 |
| V-online | 在线 vs 表逐位（**按 episode 抽 10% + 8 卡分片**） | `compare_online_motion.py`，tmux `mv2-online`，约 21–25 min。**新增 `--episode-sample 0.10 --sample-seed 0` 与 `--shard-idx/--num-shards` 与 `aggregate` 子命令**；抽样规则 `sha256(f"{seed}:{global_episode_idx}")` 取模、与分片数无关；8 片各占一张 GPU（`--gpu 0..7`，各起一个 sidecar）；**必须显式 `--out v1-store/reports/motion/p5_online_v2_1600ep.shard<i>of8.json`**（默认路径已有 40ep 历史结果）；主进程显式 `export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`（现在靠被 import 模块的 `setdefault` 生效，import 顺序一变就静默抢 GPU）；**必须传 `encoder_run_dir`**（A9 第 4 条） | 各片 `ONLINE_ENC_BITEXACT=PASS compared=<片内窗数> mismatches=0`；aggregate 后 `ONLINE_START_SET=PASS` / `ONLINE_POS=PASS` / `ONLINE_ORDER=PASS` / `PROVENANCE=PASS` / `YAML_STORE_PATH=PASS` / 汇总 `P5_ONLINE=PASS episodes=160 windows=<总窗数> padded_covered=160 stub=False`。**判据改写**：原 `all_rows = len(rows_seen) == total_rows`（硬要求 71,316）改为 `covered == 抽样集合窗数总和` 且 `padded_covered == 160` | 6 |
| V9 | 4 卡 b128 20 步 smoke（开启态） | 基线 `records/smoke-runner.sh` 同法，换 YAML / 配置名；tmux `mv2-smoke`；**基线同档实测 5 分 18 秒 > 5 min，按第 17 条以完整 run 留档**，run_name 形如 `smoke-m8x8-modul-motion-<UTC>`，验完删 run 产物、留档保留。`CKPT_PARAM_TREE` 用 `check_config_provenance.py` 并**逐个显式传** `--ckpt <smoke run>/19 --lib v1-store/datasets/4task-v2-1600ep-604f16da --train-config mme_vla_suite_b128_80k --store-subdir framesamp-8x8 --neg-lib v1-store/datasets/4task-motion-400ep --norm-stats <新库 norm_stats.json>`（其默认值全指向旧库旧 run） | `SMOKE20=PASS steps=20 finite=1 exit_code=0` / `PARAM_TREE_EXACT=PASS … n_model=65 n_ckpt=65 missing=0 extra=0 shape_mismatch=0` / `MEM_PARAMS=PASS n=10`（6 modulation 叶 + 4 motion 叶）/ `CKPT_PARAM_TREE=PASS … leaves=65 ckpt_has_motion=1` / `BUDGET_CONSISTENT=PASS budget=160` / `NORM_STATS=PASS sha256=856c75ea…`。**新验收器必须新写，不得沿用 `smoke-m8x8-modul-20260915T054007Z/records/finish_check.py`**（它硬编码 `motion_enabled is False` 与 `n_model=n_ckpt=61`） | 6 |
| V10 | 起跑 preflight | `preflight_train_launch.py`（runner 内，与 train 共用同一 `TRAIN_ARGS`）。**`n=25` 属实但前提是 runner 必须传 `--run-root`**（该项条件计入，不传就悄悄变 24）。**preflight 对 motion 开启态是盲区**（全文无 motion 检查），而本轮是第一次在正式 run 开 motion ⇒ 新增 **5** 项：`CHECK_MOTION_STORE_PATH`、`CHECK_MOTION_STORE_META_SHA256`、`CHECK_MOTION_LAYOUT`（== `motion-768-grid16-demopad17-v1`）、`CHECK_MOTION_ROWS`（== 71316）、`CHECK_MOTION_ENV_UNSET`（`MMEVLA_MOTION_STORE` 为 unset） | `PREFLIGHT=PASS n=30`（25 + 5；初稿写 29 少算了 unset 那条） | 8 |

**工具白名单（V1 / V7 / V8）**：`dump_fixture_samples.py` 与 `bench_train_steps.py` 各有一份 `_EXPECTED_HISTORY_CONFIGS`，当前只含 4 个 context YAML，对 modul YAML 会 raise。**两处各加两个**：`perceptual-framesamp-modul-8frame-8x8.yaml`（V1/V7 关闭态用）与本轮新增的 `perceptual-framesamp-modul-8frame-8x8-motion.yaml`（**V8 开启态用**）。不可理解成「`modul.yaml` + `modul-8frame-8x8.yaml`」，否则 V8 起跑即 raise。

**V5 为何不能用 CPU dummy（用户据此拍板改 GPU 真实配置）**：`MemoryAttention` 里 `num_heads, num_kv_heads, head_dim, width = (4, 1, 256, 1024)` 并 `assert mem_width == x_width == width`，其中 **`x_width` 来自最后一条 expert 流（action expert）的宽度**，不是 `memory_token_dim`；同时 `HistoryBlock` 断言 `all(config.depth == configs[0].depth)`。现有变体里 `dummy` = width 64 / depth 4，唯一 width=1024 的是 `gemma_300m` = depth 18。所以「只改 `hc.memory_token_dim`」不够，「只换单侧变体」也不够。三条路里，新增测试变体要动生产模型代码、两流都用真变体等于在 CPU 上跑 2.3B，**用户选择在 GPU 上跑真实配置**。

**V5 工具适配（`motion_gates_model.py`，初稿只写「新增 `--integration` 开关」，实际至少六层不兼容）**
1. `_make_models` 的 dummy 宽度问题（见上）：改为接受真实配置路径、在 GPU 上建真模型。
2. `cmd_m3`/`cmd_m4` 写死 32 帧 / 16 token 每帧 / 96 motion / 608 总长的**字面量**（连现有 8×8 context 口径的 `FRAME_BUDGET=8`、`TOKENS_PER_FRAME=64` 都已对不上），全部换成常量表达。
3. `_fixture_batch` 固定找 `(6,0)`/`(32,11)`/`(32,max)`，而 `(6,0)` 在新库**物理不可达**（新库 `exec_start_idx` 最小 100 ⇒ `k = min(t+1,32)` 恒为 32）；`(32,"max")` 在新库 m 最大 141 > 96 也会 raise。改为按数据集真实交付挑样本（`k = static_mask.sum()//tokens_per_frame`、`m = motion_mask.sum()`），spec 用边界语义：`m=0` / `m` 中位 / `m=max`（141）/ `m=budget`（人造满 160）。
4. `_t3_main` 把 YAML stem 写死成 `perceptual-framesamp-context…`，`--lib` 默认旧 40ep 库，`--store-subdir` 默认 `framesamp`：新增 `--integration {context,modulation}` 决定 stem，V5 显式传新库与 `framesamp-8x8`。
5. **`MOTION_BUDGET` 不在 `_t3_main` 的 `global` 列表里**，`--budget` 无从注入 ⇒ 必须把它纳入 global 并新增 `--motion-budget`。
6. `_synthetic_entry` 用旧公式造合成 index、且 `parse_index` 的 payload 写 `"layout": ms.LAYOUT` 但**不带三新键** ⇒ 换 `LAYOUT` 常量后该用例会撞 A1 的「新布局三键缺一即 raise」。改为显式写布局名与三键、`_synthetic_entry` 增 `min_real` 形参；其 oracle `oracle_visible` 的 demo 判据也要按 demo `min_real` 参数化。

**V5 判据的两处修正**
- `ZERO_MOTION_EQUIV`（开启态 motion 全 mask ⇒ 等价于关闭态）在 context 下成立（记忆进 prefix、位置走 `cumsum(input_mask)−1`，全 mask 的 motion 位不占号）；**modulation 下比的是 `mem_len=672` 与 `mem_len=512` 两个模型，q 位置差 160 位，实测输出相对 L2 变化 0.87，`tol=1e-4·|loss|` 不可能过，且 FAIL 不代表实现有错**。拆成两条不同承诺：`PAD_CONTENT_INVARIANCE`（**同一 budget** 下 padding 内容不影响结果）与 `LEN_EQUIV_NA`（不同 memory 长度的两模型不等价，显式不做该断言并写明 RoPE 原因）。
- 现有垃圾对拍**只比 loss 与 actions 的字节**，其后的梯度检查跑在 clean 输入上、只看零/非零性质，**从未比较垃圾前后的参数梯度**。`PAD_CONTENT_INVARIANCE` 的判据改为：固定模型参数、固定 rng key、固定 actions，分别算 clean 与 garbage 的 `jax.grad`，**逐叶比全部可训练参数**（不只 4 条 motion 叶）。明确「垃圾 = 有限数值（1e3 量级正态）」，NaN/Inf 不在同一承诺内。保留正向检查：真实 motion 内容改变必须改变 loss；4 条 motion 叶在有 motion 时梯度非零；**20 步 smoke 内这 4 叶的参数值确有更新**（现有 M4 只查梯度非零、不查更新）。

**V-online 比较器的加固**（现工具有两种「错误却通过」）
1. `verdict = start_ok and pos_ok and order_ok and (args.stub or (not tok_mism and all_rows))` —— `args.stub` 短路让**已记录的 `stub_token` 失配**永远不进判定，同时那一行还被打印成 `ONLINE_ENC_BITEXACT=SKIP(stub) … mismatches=N`，即 stub 档 token 全错仍 `P5_ONLINE=PASS`。改为**任何模式下已检测到的失配都必须清零**，`SKIP` 只允许出现在「真离线 token 对拍」这一项的打印上、不得进入 verdict。
2. 四处 `np.array_equal` 比浮点：`+0.0 == -0.0` 为真（放过符号零）、float32 vs float64 同值也为真（放过 dtype 漂移），而判定行却叫 `BITEXACT`。换成先核 dtype/shape 再按 `uint8` 视图比原始字节（`motion_gates_model.py` 里已有现成的 `_bytes_equal` 可抄）。
3. 改完必须**注入两个反向用例证明比较器会 FAIL**：把某个 stub token 改成错常数、把某行 `+0.0` 换成 `-0.0`，各跑一次确认非零退出。
4. ★ 另补一项**秒级 CPU 前置门**（现有 P 系列全部用 `MME_VLA_Policy.__new__` 手工赋 `_motion_cfg`，真 `__init__` 的键透传与真 `reset()`（**重建对象**，不是 `mem.clear()`）从未被跑到）：用真 YAML → `get_history_config` → 真 `MME_VLA_Policy.__init__`（注入 stub `motion_enc_fn`）构造 policy，断言 `_motion_cfg` 的键集合与值逐项正确；连跑两条 es 不同的 episode，中间调**真 `pol.reset()`**，断言新 `mem_buffer` 是新对象、补帧窗数正确。别让 20+ min 的 V-online 去当第一道发现者。
5. ★ V-online 会被 `_make_dataset` 设置的 `MMEVLA_MOTION_STORE` 隐式覆盖 YAML 路径（生产 runner 保持 unset），所以**对拍前先 `unset MMEVLA_MOTION_STORE` 用真 YAML 解析一次路径并断言 `realpath == $LIB/motion`**（判定行 `YAML_STORE_PATH=PASS`）。注意这是**验证覆盖缺口，不是数据正确性风险**：生产 unset 时路径写错会被 `check_same_source` fail-loud raise，不会静默用错库。

### G. commit 约束

- 第一部分第八节的表是唯一执行顺序；每个 commit 后立即 `git push`；只 `git add` 本轮明确文件，禁 `git add .` / `-A` / `-a`。
- **「改→验→commit」的顺序硬闸**：`single_step_grad_fixed.py` 要求 `git rev-parse HEAD` 逐字等于 `DTYPE_SOURCE_COMMIT` 且 `git status --porcelain` 为空（唯一例外是一条为历史树写死的单行 `??`，**不得拿来放行本轮改动**）。故拆成 2a（clean HEAD 取基线证，`BASE=$(git rev-parse HEAD)` **不写死 sha**）→ 2b（改码 + CPU 单测，dirty，不跑带校验的工具）→ 2c（commit + push，工作区再次干净）→ 2d（取候选证 + 对拍）。**绝不允许临时放宽 clean-source 校验。**
- ★ `compare_fixture_dumps.py` **不比较** `DUMP_MANIFEST.json` 里的 `git_head`（与 V6 的 `DTYPE_SOURCE_COMMIT` 形成反差）：V1 两侧的源码身份必须人工写进留档，或在对拍前 `jq` 断言两者等于预期 SHA。
- 步骤 4 的 Wan → encode 之间零 commit；步骤 2c 结束后到步骤 4 起跑前 `git status --porcelain` 必须为空；步骤 7 的 `TRAIN_HEAD` 记在 docs commit 之后、runner 生成之前。
- 正式 run 的 runner 与 preflight 参数照抄 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/records/prod-runner.sh` **加上 `records/launch.actual.json::launch_command` 的 `env` 前缀**（`CUDA_CACHE_PATH` / `WANDB_DATA_DIR` / `XDG_DATA_HOME` 三个不在 runner 里），差异只有第一部分第三节列的那 8 处；`MMEVLA_MOTION_STORE` 保持 unset。
- **不得修改**：`perceptual-framesamp-context-motion.yaml`、`perceptual-framesamp-context-8frame-8x8-motion.yaml`、任何 run 内 `history_config.resolved.yaml`、`train.py` 的保存条件、`checkpoints.py` 的保留策略、`utils.get_config` 的变体表。两键的向后兼容靠 reader 侧默认值实现，**不靠回填旧配置**。

### H. 重构前后链路图（第 18 条强制）

**关闭态（基线，本轮必须逐位不变）** —— 初稿此图画错了一跳（把 `memory_order` → `mem_order (512,)` 当成基线的一步），实际关闭态根本不调 `memory_order`：

```
episode_manifest.json ──► FrameSampDataset.__getitem__(g, step)
  framesamp-8x8 store ──► static_image_emb (512, 2048) bf16   [(8,64,2048) 只是 _pad 的中间量]
                          static_mask      (512,)      bool   [np.repeat(mask, 64)]
                          static_pos_emb   (512, 768)  f32
                          static_state_emb (512, 8)    f64
                          ftimes           (8,)        int64  [pad_times 返回 int64，仅内部中间量]
     ★ mem_order / motion_emb / motion_pos / motion_mask 四键恒为 None（模块级 _NONE_KEYS）
       —— memory_order() 只在 if self._motion_enabled: 块内调用，关闭态不执行
  ──► RepackTransform ──► HistAugObservation ──► PerceptualMemory.embed_memory  [关闭态早返回，不做 take_along_axis 重排]
  ──► mem_seq (b, 512, 1024) ──► MemoryAttention  k_pos=arange(512), q_pos=arange(512, 512+x_len)
```

**开启态（本轮新增，只多 motion 一路）**

```
motion 表 motion_token.f32.bin (71316, 768) f32  ← pack/verify 产出，整表 219 MB（208.9 MiB）
  └► visible_motion_rows(entry, step)   [entry.spec = (17, 33, repeat_last)]
       demo 行：s ≤ es − 17（含补帧窗）      exec 行：u + 32 ≤ t − es
  └► 取行 → motion_emb  (k, 768) f32 ──右填充──► (160, 768) f32  = 491,520 B  [填充位内容不影响结果，V5 判据]
            motion_pos  (k, 256) f32 ──右填充──► (160, 256) f32  = 163,840 B
            motion_mask (k,)     bool──右填充──► (160,)     bool =     160 B
            mtimes      (160,)   int64  [_pad_motion 右填充后长度是 budget]
  └► memory_order(ftimes, 64, mtimes) ──► mem_order (672,) int32 = 2,688 B
       ★ 交错：motion token 按时刻插进帧 token 之间 ⇒ 帧 token 彼此的 RoPE 相对距离**改变**
                                      合计每样本新增 658,208 B ≈ 658 KB；b128 ≈ 84 MB/batch
  ──► RepackTransform ──► HistAugObservation ──► PerceptualMemory.embed_memory
        motion_pos_proj      (256→768)    ┐ 新增 4 叶
        motion_encoder_static(1536→1024)  ┘ 1,771,264 参数
  ──► mem_seq (b, 672, 1024) ──► MemoryAttention  k_pos=arange(672), q_pos=arange(672, 672+x_len)
       ★ mem_len 512→672 ⇒ 全部 query 的 RoPE 位置整体 +160 ⇒ 真实 token 的有效注意力**改变**
```

**每一跳有没有改数**：取行、右填充、`memory_order` 置换、`RepackTransform` 全是搬运与重排，**不改数值**；唯一改数的是两层 Linear 投影与 `MemoryAttention` 本身。帧路在**数据交付层**逐字节不变，但在**模型内**因 `mem_order` 交错与 `mem_len` 变化而不再等价——这正是本轮要测的东西，不是回归。
