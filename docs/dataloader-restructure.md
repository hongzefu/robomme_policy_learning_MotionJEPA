# 训练特征库：从每帧一个小文件改成三张连续大表

> 训练时模型每一步要读 32 帧图像的 SigLIP 特征。原来每帧存一个小文件，一个样本要开三十几个文件、每个文件还要整包反序列化；现在改成三张连续大表，一个样本只开一个文件、按行直接读。本页只讲三件事：改之前和改之后的文件结构、这些文件怎么生成、怎么证明改前改后训练结果一模一样。
>
> 数字分两个环境、不混表：**环境 A**（GreatLakes 4×A40 + turbo NFS、本机 2×RTX 6000 Ada，2026-09-03 及以前）是历史；**环境 B**（AWS 单机 8×A100-80GB，本地 NVMe RAID，2026-09-04 起）是现行。根目录四份计划文件（`v1-`/`v2-framesamp-restructure-plan.md`、`v3-destructive-restructure-plan.md`、`v5.0-train-entry-restructure-plan.md`）保留为过程档案，冲突以本页为准。更细的 `store_meta.json` 逐字段表、锁协议、删旧链路的七次提交顺序、吞吐档位扫描，见本文件 git 历史 `3f4afb5` 版本。

## 一、结论

- **改了什么**：特征存法从「每帧一个 602,951 字节的 pickle 小文件」改成「三张裸字节大表」；读法从「每样本开 33 个文件、整包反序列化 32 次」改成「常驻文件句柄按行直接读、零反序列化」。前面的清单扫描与 SigLIP 建库两步一字未动，只在后面新增一步「打包」。
- **改前改后一致**：同一配置各跑 1000 步真实训练，逐步五个标量的浮点位串 sha256 全部相等，值为 `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`（基线两份加四个改动节点，共六份同值）。这是位级一致，不是近似。
- **现行速度**：8×A100、global batch 128、16 个 worker，稳态 2.656 s/step，40k 步约 30 h（`training-doc/bench-b128-util/result.md`）。

## 二、改之前的文件结构

```
源库 <lib>/{features,data,meta}/
├── meta/stats.json                          执行样本数 / 总帧数
├── features/episode_{g}/token_emb_{t}.npy   每帧一个，602,951 B，np.save 存的 pickle 字典，7 个键绑在一起：
│       image_emb_8x8 (1,64,2048) bf16  256 KiB
│       image_emb_4x4 (1,16,2048) bf16   64 KiB   ← 训练只用这三个键
│       image_emb_2x2 (1, 4,2048) bf16   16 KiB
│       pos_emb_8x8   (1,64, 768) f32   192 KiB
│       pos_emb_4x4   (1,16, 768) f32    48 KiB   ←
│       pos_emb_2x2   (1, 4, 768) f32    12 KiB
│       state_emb     (8,)        f32      32 B   ←    三个键合计 112 KiB，占每帧字节的 19%
└── data/{idx}.pkl                           每个执行样本一个，约 395 KB：两张原图 (256,256,3) u8、
                                             state (8,) f32、actions (20,8) f64、prompt 字符串、episode 与帧号
```

**两套编号。** `features/` 按全部帧存，包括 `Video*` 任务开头的演示帧；`data/*.pkl` 只按执行样本存，演示帧不出样本。两套编号靠清单 `episode_manifest.json` 里每集的 `exec_start_idx` 换算。漏掉它，`VideoUnmask`（演示帧恒 66 帧）和 `VideoUnmaskSwap`（114 到 216 帧）的每个样本都会错位 66 到 216 帧，而错位后读到的仍是合法帧，不报错。

**为什么慢。** 每个样本要开 1 个 pkl 加最多 32 个 npy；每个 npy 的 7 个键绑死，要拿 112 KiB 必须整包反序列化 589 KiB；`pos_emb_4x4` 只跟帧号有关、跨集逐字节相同，却每帧存一份。每个样本还新建一个最多 32 线程的线程池，用完即弃。

**环境 A 实测（b64）。** 每样本读盘 19.08 MB，真正用到 3.95 MB；每步读 1.22 GB、开 2,112 个文件；每样本热 25.4 ms、冷 132.4 ms，其中 32 次 open 本身就要 74.3 ms。worker 数 8 / 12 / 16 三档步时 5.301 / 5.319 / 5.327 s 几乎不变，说明调参数没用、必须改代码。这些数字属环境 A，不与环境 B 比；论证见 `archive/v1-nfs-bottleneck-analysis.md` 与 `archive/training-doc/v1-e2e-b64/`。

## 三、改之后的文件结构

```
<lib>/framesamp/                                     纯派生，源库原样保留
├── meta/store_meta.json          唯一契约：布局、每张表的形状与 dtype、part 边界、源库根与清单路径、
│                                 清单 sha256、每个文件的 sha256、状态（packed / verified）
├── meta/row_digests.blake2b.bin  逐行摘要（行数 × 16 B），verify 时产出
├── meta/pack.lock                打包或校验进行中的锁；存在时训练拒绝读
├── image_emb_4x4/part_000.bf16.bin … part_NNN.bf16.bin
│                                 每行一帧 (16,2048) bf16 = 65,536 B；按 episode 边界切成约 32 个文件
├── pos_emb_4x4.f32.bin           每行一个帧号 (16,768) f32 = 49,152 B；只存 586 行（帧号 0..585）
└── state_emb.f32.bin             每行一帧 (8,) f32 = 32 B
```

**行号规则。** `行号 = 该集的 total_sample_offset + 帧号`，写侧和读侧用同一个函数 `framesamp_store.py::row_of`，不可能分叉。part 只在 episode 边界切，所以一个样本要读的至多 32 帧一定落在同一个文件里，一次连续读完。pos 表按帧号索引、只存一份，不再每帧冗余。

**为什么用裸字节不用 `.npy`。** numpy 存 bf16 会写成 `V2` 类型，读回来丢 dtype。裸字节加 meta 里显式写明 dtype 是唯一可靠的落盘方式，读侧用 `uint16` 读进来再 `view` 成 bf16。

**没打包的。** `data/{idx}.pkl` 和里面的两张原图训练时仍从源库读。打包库根、源库根、清单三个位置全部写在 `store_meta.json` 里，不靠目录名做字符串推导。

两个环境各建过一个库：

| | 环境 A `4task-gl-framesamp`（4 任务 × 400 集） | 环境 B `4task-motion-400ep/framesamp`（4 任务 × 100 集） |
|---|---|---|
| 源库 → 打包库体积 | 678 GB → 31.7 GB | 107 GB → 7.6 GB |
| 总帧数 / 执行样本数 / pos 行数 | 483,291 / 395,289 / 586 | 123,044 / 101,066 / 586 |
| part 数 | 32 个（前 31 个约 1 GB，末个 621 MB） | 31 个（每个约 260 MB） |
| pack / verify 耗时 | 2,941 s / 1,061 s（16 进程，turbo NFS） | 5 s / 2 s（48 进程，本地 NVMe） |
| verify 结果 | 483,291 行全部比对，0 失配 | 123,044 行全部比对，0 失配 |
| `store_meta.json` sha256 | `3990165c9cebffdadaceb01cc88470645a3d62f9af65eabccf4331d3fcd5b556` | `dffdd47b09aad2812bc46201231e49cd120a4498828d836c9d2695311a815642` |
| 留档 | `dataset-build-doc/4task-gl-framesamp/` | `dataset-build-doc/4task-motion-400ep/` |

环境 B 另有一个开发用小库 `4task-motion-40ep/framesamp/`（13,756 帧、11,530 样本、22 个 part），CPU 级检查多跑在它上面。motion memory 接入后同一库根下还有第四张表 `motion/`，与这三张表共用同一份清单并互相核对同源，细节见 `motion-memory.md`。

## 四、生成链路

```
原始 H5 ──① scripts/dataset/scan_manifest.py──▶ episode_manifest.json   每集的帧数、执行起点、三个偏移，顶层带 sha256
        ──② SigLIP 建库（环境 B：run_local.py --stage siglip）──▶ 源库 features/ + data/     改前改后一字未动
        ──③ scripts/dataset/pack_framesamp_store.py pack──▶ framesamp/ 三张表 + store_meta.json（status=packed）
        ──④ scripts/dataset/pack_framesamp_store.py verify──▶ 回填 status=verified，删 pack.lock
```

**打包时做什么。** 先在 `meta/` 下建锁，两张小表由主进程独写；image 表按 part 分给 16 个进程并行，每个 part 只有一个写者。每个 episode 写进 part 的临时文件后立刻读回逐字节比对，整个 part 写完算 sha256 再改名落盘。写每一帧时顺手核两件事：这帧的 pos 与 pos 表同帧号那行逐字节相同，这帧的 state 与 state 表同一行逐字节相同。pos 表本身是从源库里凑齐帧号 0..585 抽出来的，不是重新生成的（CPU 后端重新生成与库里的值差约 7e-7，GPU 后端才一致）。

**verify 做什么。** 独立于 pack 再遍历全部（集，帧），重新完整解码源 npy，三个键各经训练实际用的读函数逐行比对，同时写逐行摘要。这是「一帧没漏、一帧没错」的唯一凭据；抽样档只供开发期快检，10% 抽样对单行错位的漏检率约 90%，不能作交付判定。环境 A 全量 pack 约 49 分钟、verify 约 17.7 分钟。

**训练读取时做什么。** Dataset 在 `src/mme_vla_suite/training/framesamp_dataset.py`，只服务 `perceptual-framesamp-context` 一种配置，配置形制不对直接报错。

```
主进程构造 Dataset：只读 meta、做静态检查、从清单算好「样本 → (集, 帧)」查表数组；不开任何文件句柄
   检查项：清单 sha256 现场重算比对、每个 part 存在且大小等于 meta 记录、两张小表大小、
           抽一个 part 首尾 1 MiB 摘要复验、抽一个源库文件摘要复验源库没动
每个 worker 第一次取数：打开全部 part 句柄，两张小表整个读进内存（环境 A 库口径每 worker 约 44 MB）
之后每个样本 idx：
  ① 查表得 (集 g, 帧 step)
  ② 读源库 data/{idx}.pkl；核对 pkl 里记的集号与帧号 == 查表结果，不符即报错（行号错位的最后一道闸）
  ③ 选帧：even_sampling_indices(step, 32)，与旧链路 import 同一个函数，选帧逐位不变
  ④ 行号 = total_sample_offset[g] + 帧号；按行号 preadv 读 image（≤32 行连续，通常一次读完）；
     pos、state 直接查内存里的小表
  ⑤ 不足 32 帧补零；补零区按输入 dtype 分配，不会把 bf16 提升成 f64
  ⑥ 交付 static_image_emb (512,2048) bf16、static_pos_emb (512,768) f32、
     static_state_emb (512,8)、static_mask (512,) bool，加上 pkl 里透传的图、state、actions、prompt
```

**训练拒绝读的三种情况。** `store_meta.json` 的状态不是 verified、`meta/pack.lock` 存在、上面任何一项检查不过，一律直接报错，没有回退到旧文件的路径（旧链路已整体删除）。开发期可用 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1` 放行未 verify 的库，放行会打 WARNING、跑出的 run 不算数。另有 `MMEVLA_FRAMESAMP_SOURCE` / `MMEVLA_FRAMESAMP_MANIFEST` 可覆盖源库与清单位置。正式 run 起跑前 `env | grep MMEVLA_FRAMESAMP` 必须为空：残留一个源库覆盖就会从另一个库读 pkl，而环境指纹检查看的是命令行参数，抓不到。

## 五、改前改后一致

**前提。** 给定 seed、batch 与同一份数据集，旧链路每步抽哪些样本、每个样本选哪 32 帧都是确定的，worker 数只影响交付时机不影响内容（跨 epoch 边界 w0 与 w>0 会分叉，这是 torch 既有行为，对拍只在同 worker 数下比）。正确性对拍统一跑确定性 XLA（`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`），这一档独立冷编译两次已证逐位一致；生产默认档不逐位。

**不跑训练的轻量对拍。**

| 核对什么 | 结果 |
|---|---|
| 同 seed 抽样序列：新旧 Dataset 装进同一个 loader，w0 / w4 / w8 三档各 dump 一段 index 序列 | 逐条相同 |
| 定点样本：约 8,200 个样本（每集首样本、step 边界全覆盖、随机抽）在 transform 之后逐键比 shape、dtype、位串，再加 200 个真实 batch 过 collate | 零差异 |
| 全量写读对拍：即第四节的 verify | 483,291 行零失配 |
| 真实 spawn loader 矩阵 w0 / w1 / w4 / w16 各 2 个 epoch | 跑通，主进程文件句柄数回到基线，无泄漏 |

**真实训练 1000 步对拍。** 全部是 1000 步、b8、seed 42、2 卡、4 worker、确定性档的真实训练。先把改动前的代码跑两次作黄金基线，之后每改一次链路再跑一次：

| 节点 | 改了什么 | 留档 | `scalars_hex.tsv` 的 sha256 |
|---|---|---|---|
| 基线（两轮） | 改动前 | `training-doc/v1-grad-baseline-g0b/` | `c799a0b2…105757` 两份 |
| 改 padding dtype 之后 | 补零段改成跟输入同 dtype，不再随 batch 组成摆动 | `archive/training-doc/v1-dtype-ab-post-r1/`、单步梯度 `training-doc/v1-dtype-p5-grad/` | 同值 |
| 改成三张表之后 | 本页第三、四节 | `training-doc/v1-framesamp-g2/` | 同值 |
| 删旧链路与冗余分支之后 | 删散文件读取器、删 recurrent / symbolic 分支、脚本目录收敛 | `training-doc/v1-postclean-g3/` | 同值 |
| `train.py` 改单跑之后 | 去掉「预热 + 正式」两遍 `main()` | `training-doc/v1-singlerun-g0/` | 同值 |

每个节点除逐步五标量外，还核对了 12 个摘要步的完整参数树逐叶 sha 和前 8,000 条抽样顺序，全部相同。

**边界，如实说。**

- 这条链全部产于环境 A。环境 B 没有这些固化产物（A100 与 Ada 的 bf16 归约不逐位），环境 B 的等价性只能「同机两侧各跑一次」再比。
- 1000 步 × b8 = 8,000 个样本只覆盖粗错：行号错位、选帧错误几十步内就会撞穿。「万分之一错帧」这种细错靠定点样本对拍与 verify 全量比对兜底。
- raw 口径的 batch 摘要有 4 步预期失配（`static_image_emb` / `static_pos_emb` 两键，改 padding dtype 前后补零段位型不同）。这是已知口径差，每轮对拍都必须与它逐字吻合，不吻合才是信号；判定用的是逐键升到 f32 再比位串的口径。

## 六、代码与留档在哪

| 什么 | 路径 |
|---|---|
| 三张表的格式常量、行号函数、读 API、启动检查 | `src/mme_vla_suite/datastore/framesamp_store.py` |
| 训练 Dataset（取数八步） | `src/mme_vla_suite/training/framesamp_dataset.py` |
| 分派与拒绝读的闸 | `src/mme_vla_suite/training/dataloader.py::_create_framesamp_dataset` |
| 选帧函数 | `src/mme_vla_suite/shared/sampling.py::even_sampling_indices` |
| 打包与 verify 工具 | `scripts/dataset/pack_framesamp_store.py`（子命令 `plan / pack / verify / report`） |
| 对拍工具 | `scripts/training/tests/`：`dump_index_seq.py`、`spawn_matrix.py`、`test_pack_guards.py`、`single_step_grad.py`、`g0_gate.py`；基线环境指纹 `scripts/training/g0/check_baseline_env.py` |
| 建库留档 | 环境 A `dataset-build-doc/4task-gl-framesamp/`；环境 B `dataset-build-doc/4task-motion-400ep/` |
| 环境 B 吞吐留档 | `training-doc/bench-b128-util/` |
| 第四张表 motion | `motion-memory.md` |
