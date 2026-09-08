# dataloader 四阶段重构（现行正本）

> **本文件是现行正本**：`perceptual-framesamp-context` 的训练数据链路从「原版 PKL + per-step NPY 逐样本读」改成「framesamp 三张连续大表 + 常驻 fd 直读」的全过程与现状都以本文为准。根目录的四份计划文件（[`v1-framesamp-restructure-plan.md`](../v1-framesamp-restructure-plan.md)、[`v2-framesamp-restructure-plan.md`](../v2-framesamp-restructure-plan.md)、[`v3-destructive-restructure-plan.md`](../v3-destructive-restructure-plan.md)、[`v5.0-train-entry-restructure-plan.md`](../v5.0-train-entry-restructure-plan.md)）保留为过程档案，**冲突以本文为准**。
>
> **环境分野（[`AGENTS.md`](../AGENTS.md) 第 13 条）**：本文的实测数字分两套，**永不混表**——
> **环境 A**（GreatLakes 4×A40 + turbo NFS / 本机 2×RTX 6000 Ada，2026-09-03 及以前）的吞吐、步时、util 与库规模数字，一律注明「环境 A 历史」；
> **环境 B**（AWS 单机 8×A100-80GB，本地 NVMe RAID `/dev/md0`，2026-09-04 起）是当前环境，其数字即最终指标。
> 环境 A 的中间留档已于 2026-09-07（commitV7.0）迁入 [`docs/archive/`](archive/README.md)，本文引用一律使用迁移后的新路径。
>
> **引用纪律**：代码只写 `文件::类/函数/配置键` 稳定锚点，不写行号（`AGENTS.md` 第 9 条）。

---

## 一、结论先行

### 1.1 改了什么：四个阶段

| 阶段 | 主题 | 落地 commit | 权威计划 | 一句话 |
|---|---|---|---|---|
| 阶段 1 | **dtype 统一** | `commitV2.4a`（验证工具）、`commitV2.4b`（三行修复） | [`v1-dtype-unify-plan.md`](../v1-dtype-unify-plan.md) | `right_padding_token_emb` 的三个 `np.zeros` 各加一个 `dtype=` 参数，消灭「dtype 随 batch 组成摆动」的双路径；报告见 [`v1-phase2-dtype-unify-report.md`](archive/v1-phase2-dtype-unify-report.md) |
| 阶段 2 | **IO 重构（packed 三表）** | `commitV3.0`–`commitV3.4`（S0'→S6 收官），后续 `V3.5`–`V3.7` 为验收资产 | [`v2-framesamp-restructure-plan.md`](../v2-framesamp-restructure-plan.md) | 把每帧一个 `token_emb_{t}.npy` 的散小文件压成三张连续大表，训练时常驻 fd `preadv` 直读，`FrameSampStore` + `FrameSampDataset` 成为新链路 |
| 阶段 3 | **破坏性单一化** | `commitV4.0`–`commitV4.6` | [`v3-destructive-restructure-plan.md`](../v3-destructive-restructure-plan.md) | 删 legacy 数据链与 `MMEVLA_DATA_BACKEND` 三态、删 recurrent/symbolic 分支、建库域自包含隔离、`scripts/` 收敛成 `training/` + `dataset/` 两域 |
| 阶段 4 | **训练入口单跑** | `commitV5.0` | [`v5.0-train-entry-restructure-plan.md`](../v5.0-train-entry-restructure-plan.md) | `train.py` 官方 `__main__` 由「tentative 预热 + 正式」两次 `main()` 改为单跑，加两条 fail-loud 护栏与可选 `TRAIN_RECORD_DIR` 记录器 |

四阶段之前还有一个前置：**G0 黄金基线**（`commitV2.1`–`commitV2.3.1`），把「同配置重跑逐位一致」做成可证伪的前提并固化一轮 1000 步产物，供后续每一阶段离线对拍——报告见 [`v1-phase1-gradient-baseline-report.md`](archive/v1-phase1-gradient-baseline-report.md)。

### 1.2 训练语义不变的证明链：一句话

**G0b → G1（dtype 后）→ G2（packed IO 后）→ G3（单一化后）→ `v1-singlerun-g0`（入口重构后）五个节点，各自 1000 步确定性档真实训练，逐步五标量的 IEEE 浮点位投影 `scalars_hex.tsv` 的 sha256 全部等于同一个值 `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`**（G0b 自身两轮各一份，故共六份；出处逐个见十一节表）——四阶段合起来对训练数值零影响，这是位级结论、不是量化近似。

### 1.3 环境 B 现行吞吐：一行

**8×A100 / global batch 128 / `fsdp_devices=4` / `num_workers=16`：稳态 2.656 s/step，40k step 外推 29.5 h（叠加编译与 8 次 checkpoint 约 30 h）**——出处 [`docs/training-doc/bench-b128-util/result.md`](training-doc/bench-b128-util/result.md) 档位扫描表 B4 档，介质 AWS 本地 NVMe RAID（`/dev/md0`）。

---

## 二、重构前链路：原版 PKL + per-step NPY

> 本节数字全部是**环境 A 历史**实测，出处见小节末；瓶颈判定的完整论证不在本文复述，只做指针。

### 2.1 生产侧两阶段（本次重构一字未动）

```
原始 H5（四任务各一个文件，环境 A 历史：本机原件永久保留于 /data/hongzefu/，
         turbo 暂存副本经逐文件 sha256 同源核对）
  │   内部：episode_{i}/timestep_{t}/obs/front_rgb (256,256,3) u8、wrist_rgb、
  │         joint_state (7,) f4、gripper_state (2,) f4、action/joint_action (8,) f8 …
  │         ⚠ 无 chunk 无压缩，全是逐 timestep 散小数组
  │ 阶段一 scan_manifest.py build（只读 metadata，规范序 sorted(*.h5) × sorted(episode)）
  ▼
episode_manifest.json ——「唯一真值源」
  每 episode 记 (h5_file, raw_ep_idx) 身份、num_timesteps、exec_start_idx、exec_samples，
  以及三个前缀和偏移 global_episode_idx / exec_sample_offset / total_sample_offset；
  顶层带 sha256（被改动即 fail-loud，下游引用时命名为 manifest_sha256）
  │ 阶段二 SigLIP 建库（环境 A 历史：GreatLakes 8×1GPU job array；环境 B：本机 run_local.py --stage siglip）
  │ 逐 timestep：front_rgb → resize_with_pad(256→224) → SigLIP So400m/14（bf16，256 token）
  │            → 池化出 8x8 / 4x4 / 2x2 三档 → PosEmb3D 按 step 切片 → 每帧存一个 npy
  ▼
源库 <lib>/{features,data,meta}/
├── meta/stats.json          execution_samples / total_samples
├── features/episode_{g}/token_emb_{t}.npy   每帧一个，602,951 B（真常量）
│     np.save 的 pickle dict，7 键绑死：
│       image_emb_8x8 (1,64,2048) bf16  256 KiB ┐
│       image_emb_4x4 (1,16,2048) bf16   64 KiB │←┐
│       image_emb_2x2 (1, 4,2048) bf16   16 KiB │  │ framesample 只用这三个键
│       pos_emb_8x8   (1,64, 768) f32   192 KiB │  │ 共 112 KiB（= 每帧字节的 19%）
│       pos_emb_4x4   (1,16, 768) f32    48 KiB │←┤
│       pos_emb_2x2   (1, 4, 768) f32    12 KiB │  │
│       state_emb     (8,)        f32      32 B ┘←┘
└── data/{idx}.pkl           每执行样本一个，约 395.4–395.6 KB（内嵌变长 prompt/subgoal 字符串，
      非定长，下界 395,440 B）：image / wrist_image (256,256,3) u8 两张原图（共 393 KiB）、
      state (8,) f32、actions (20,8) f64、prompt/subgoal 字符串、epis_idx / step_idx / exec_start_idx
```

两个必须记住的口径（出处 `v2-framesamp-restructure-plan.md` 1.2）：

- **features 按「全部 timestep」存**（含 `Video*` 任务的 demo 前缀帧），**`data/*.pkl` 只按「执行样本」存**（demo 前缀不出样本）。两套编号靠清单偏移字段换算，**必须带 `exec_start_idx`**——漏掉会让 `Video*` 任务的样本错位 66–216 帧（demo 前缀长度：`VideoUnmask` 恒 66 帧，`VideoUnmaskSwap` 实测取值 {114, 168, 216}）。
- 每帧 npy 是 `np.save` 的 **object dict（pickle）**，7 键绑在一起 **无法部分读取**——要拿 `image_emb_4x4`+`pos_emb_4x4`+`state_emb` 那 112 KiB，必须整包反序列化 589 KiB。
- `pos_emb_4x4` 实测是 **`step_idx` 的纯函数**（跨 episode 逐字节相同），却按帧冗余存了每一帧一份。

### 2.2 消费侧：训练时每个 step 的取数链（旧）

```
┌─ 主进程（jax，驱动多 GPU）────────────────────────────────────────────────┐
│ torch.Generator().manual_seed(seed) + shuffle + drop_last                 │
│   每 step 抽 batch 个样本 idx。同一迭代器生命周期内序列只由                │
│   (len, seed, batch, drop_last) 决定、与 num_workers 无关（跨 epoch 见 2.4）│
└──────┬────────────────────────────────────────────────────────────────────┘
       ▼  idx 分派给 spawn worker（persistent_workers，prefetch_factor 取 torch 默认 2）
┌─ RoboMMEDataset.__getitem__（已随 commitV4.1 删除，见 git 历史）──────────┐
│ ① pickle.load(data/{idx}.pkl)     395 KB，热约 2.7 ms（其中约 2.3 ms 是    │
│    NFS open 延迟）；取出 epis_idx=g、step_idx=t                            │
│ ② even_sampling_indices(t, 32)    纯确定选帧，零随机源                     │
│ ③ _gather_history_feat            ⚠ 每样本新建 ThreadPoolExecutor(≤32 线程)│
│    逐帧 np.load(token_emb_{f}.npy, allow_pickle).item() 全量反序列化       │
│    = ≤32 次 open+close + ≤32 次 589 KiB 整包 pickle（每次 4.0–4.9 ms）     │
│ ④ 拼装 (n,16,2048) bf16 / (n,16,768) f32 / (n,8) f32 / mask (n,)          │
│ ⑤ right_padding_token_emb         ← 阶段 1 修的就是这里的三个 np.zeros     │
│ ⑥ reshape → static_image_emb (512,2048) / static_pos_emb (512,768) /      │
│    static_state_emb (512,8)（use_state_emb=false，GPU 不消费）/ static_mask │
│ ⑦ transforms（与新链路完全相同，见九节）                                   │
│ ⑧ _collate_fn（np.stack）在 worker 内执行                                  │
└──────┬────────────────────────────────────────────────────────────────────┘
       ▼ 已合并的 batch 经 IPC 回主进程 → device_put → GPU jit embed_memory
   FeatureEncoder.encode_perceptual_memory：
     x = concat(static_image_emb, silu(pos_proj(static_pos_emb)))  (512, 2816)
     memory_tokens = encoder_static(x)                             (512, 2048)
   → 512 个 memory token 拼在 prefix 最前：[512 mem | 256 img | 256 wrist | 64 txt]
```

### 2.3 字节帐与耗时帐（环境 A 历史，b64 口径）

| 口径 | 旧链路数值 | 备注 |
|---|---|---|
| 每样本读盘 | 均值 **19.08 MB**（上界 19.69 MB） | 真正用到均值 3.95 MB（上界 4.07 MB），放大约 **4.8×**；单看 npy 是 589 KiB 只用 112 KiB（5.3×） |
| 每样本耗时（热 / 冷） | **25.4 / 132.4 ms** | gather 占 17.7 / 约 110 ms；32 次 open 本身 74.3 ms |
| 每 step 读盘（b64） | **1.22 GB** | 需求 256 MB/s，turbo NFS 供给 398–628 MB/s（带宽不是瓶颈） |
| 每 step 文件打开（b64） | **2,112 次** | 64 pkl + 64×32 npy |
| collate / IPC / device_put | 19 ms / 约 257 MB / 23 ms | collate 在 worker 内执行；257 MB = memory 三键 236 MB + 两张原图 19 MB |
| 步时（GreatLakes 4×A40，b64） | 中位 **6.933 s**（compute-only 下界 4.778 s，+45%） | util 均值 69.7%、0% 采样 27.8%、慢步占稳态墙钟 32.9% |

> 全表统一十进制 MB（1 MB = 10⁶ B），与旧留档里的 MiB 数字不可直接比。数字出处：`v2-framesamp-restructure-plan.md` 1.4 与 [`docs/archive/training-doc/v1-e2e-b64/`](archive/training-doc/v1-e2e-b64/)。

### 2.4 浪费在哪里（按影响排序）

1. **文件个数**：每样本 ≤33 次 NFS open——32 次 64 KiB 读若落在一个常开 fd 上只要 0.33 ms，open 却要 74.3 ms。
2. **整包 pickle 反序列化**：7 键绑死，读 5.3× 于所需字节。
3. **每样本新建 ≤32 线程的线程池**，用完即弃。
4. **`pos_emb` 冗余**：纯函数按帧存盘反复读，占必需读量 38%。
5. **worker 里的 JAX**：来源是 dataset 模块级导入链（`training/dataset.py` → `mem_buffer.py` → `openpi.shared.image_tools`，import 即 `import jax`），**不是 `transforms.ResizeImages`**（后者是 `openpi_client` 的 NumPy/PIL CPU 实现）。
6. **`static_state_emb` 白算白传**（`use_state_emb=false`）。

### 2.5 瓶颈判定结论：只做指针

「瓶颈在 dataloader worker 的 CPU/文件层、不在 NFS 带宽、且靠调 workers 参数解决不了」这一判定的完整论证与实据，全部在**环境 A 的只读报告**里，本文不复述：

- [`docs/v1-nfs-bottleneck-analysis.md`](archive/v1-nfs-bottleneck-analysis.md)——NFS 带宽与 IOPS 侧的排除论证。
- [`docs/v1-gl-resource-tier-bench.md`](archive/v1-gl-resource-tier-bench.md)——资源档位实测。
- [`docs/archive/training-doc/v1-e2e-b64/`](archive/training-doc/v1-e2e-b64/)——端到端 util 均值 69.7%（中位 100% 是假象，`AGENTS.md` 第 16 条的原始教训案例）。
- [`docs/archive/training-doc/v1-e2efix-w8c16/`](archive/training-doc/v1-e2efix-w8c16/)、[`w12c16`](archive/training-doc/v1-e2efix-w12c16/)、[`w16c16`](archive/training-doc/v1-e2efix-w16c16/)——workers 8/12/16 三档曲线完全平坦（5.301 / 5.319 / 5.327 s，util 71.2% / 70.6% / 67.1%），坐实纯参数调整无效、必须代码级重构。

**这些数字属环境 A，禁止与环境 B 的数字并列比较**（`AGENTS.md` 第 13 条）；所引 `v1-store/` 产物在环境 B 也已不可得。

### 2.6 旧链路的确定性（新链路继承的前提）

给定 seed / batch_size / fsdp_devices 与同一份数据集：

- **单个 epoch 内**每 step 的样本集合与 memory token 内容逐位可复现，`num_workers` 只影响交付时机不影响内容。
- **跨 epoch 边界与 `num_workers` 相关**（torch 既有语义，与本重构无关）：`_BaseDataLoaderIter.__init__` 每次构造迭代器都从同一 generator 抽一次 `_base_seed`，而 `persistent_workers`（w>0）跨 epoch 只 `_reset` 不重建、w0 每 epoch 重建——**同 seed 下 w0 与 w>0 从第 2 个 epoch 起排列分叉**。恒等链只要求「新旧链路在相同 `num_workers` 下序列相同」，不受影响。
- XLA 层：生产默认 autotune 档下同配置重跑非 bitwise；正确性对拍一律跑确定性档（`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`），该档下独立冷编译两次已实证逐位一致。

---

## 三、重构后：三张连续大表

### 3.1 一句话与目录布局

**把每帧一个 602,951 B 的小 npy 压成少数几个连续大文件，只保留 framesample 真正要的三张表，训练时用常驻 fd 直接 `preadv`。** 预处理前两阶段（清单、SigLIP 建库）与其产物**原样保留、一字不动**，只新增纯派生的「阶段三：打包」。

```
<lib>/framesamp/                                   （打包库根，纯派生）
├── meta/store_meta.json         唯一契约：布局 / 形状 / dtype / 字节序 / part 边界 /
│                                源库根与清单路径 / 源库指纹 / manifest_sha256（四节）
├── meta/pack_progress.jsonl     断点续跑记录（父进程单写）
├── meta/row_digests.blake2b.bin 逐行独立摘要（num_rows × 16 B，verify 时产出，必产出）
├── meta/pack.lock               排他锁；存在即读侧拒绝消费（四节）
├── image_emb_4x4/part_000.bf16.bin … part_NNN.bf16.bin
│       (rows,16,2048) bf16 裸字节；行号 = total_sample_offset[g] + t（写读共用同一函数）；
│       按 episode 边界切分 → 一个样本的 ≤32 帧必落在同一 part
├── pos_emb_4x4.f32.bin          (num_pos_rows,16,768) f32——pos 是 t 的纯函数，只存一份
└── state_emb.f32.bin            (num_rows,8) f32
```

`data/{idx}.pkl` 与两张原图**不打包**：训练时仍从源库 `<lib>/source/data/` 读。打包库根、源库根、清单三个位置经**显式双根契约**传递（六节 6.5），禁止从目录名做字符串变换推导。

### 3.2 三表的行布局、dtype 与每行字节量

格式常量的唯一实现在 [`src/mme_vla_suite/datastore/framesamp_store.py`](../src/mme_vla_suite/datastore/framesamp_store.py) 模块顶部（`LAYOUT` / `IMAGE_ROW_SHAPE` / … 一族），打包工具、`FrameSampDataset`、对拍工具一律从 `mme_vla_suite.datastore` import，**绝不复制**：

| 表 | 一行是什么 | `row_shape` | dtype | 每行字节 | 索引键 |
|---|---|---|---|---|---|
| `image_emb_4x4`（分 part） | 一帧的 16 个 4x4 池化 token | `(16, 2048)` | `ml_dtypes.bfloat16` | `IMAGE_ROW_BYTES` = 16×2048×2 = **65,536** | 全局行号 `row = total_sample_offset[g] + t` |
| `pos_emb_4x4`（单文件） | 一个 `step_idx` 的位置编码 | `(16, 768)` | `np.float32` | `POS_ROW_BYTES` = 16×768×4 = **49,152** | 帧号 `t`（全 timestep 域） |
| `state_emb`（单文件） | 一帧的状态嵌入 | `(8,)` | `np.float32` | `STATE_ROW_BYTES` = 8×4 = **32** | 全局行号 `row` |

其余格式常量：`BYTE_ORDER="little"`、`ARRAY_ORDER="C"`、`BF16_ENCODING="ml_dtypes.bfloat16 (1s+8e+7m)"`、`LAYOUT="framesamp-4x4-v1"`、`META_SCHEMA=1`。

**为什么是裸 `.bin` 而不是 `.npy`**（`v2-framesamp-restructure-plan.md` A.1 的定论，已实测）：`np.save` 对 ml_dtypes bf16 写出 `V2` descr，`np.load` 读回丢类型。裸字节 + meta 显式声明 dtype 是唯一可靠的落盘方式；读侧 `np.memmap(dtype=ml_dtypes.bfloat16)` 与 `frombuffer(uint16).view(bfloat16)` 两条路都实测可用，实际选的是后者（`FrameSampStore.read_image_rows` 里 `raw.view(IMAGE_DTYPE)`）。

### 3.3 part 切法

按 `global_episode_idx` 升序累积 `num_timesteps`，累计 ≥ `ceil(num_rows / TARGET_PARTS)` 即切，**切点必在 episode 边界**——这条保证「一个样本的 ≤32 帧必在同一 part」，从而单样本 gather 天然是一次连续读。`TARGET_PARTS = 32` 是切分阈值的目标值，不是产出 part 数的硬约束：实际 part 数由 episode 长度分布决定（下表两个库分别是 31 和 32）。单文件方案（与原子写 + 并行互斥）与「每 episode 一文件」方案（退化回每样本一次 open）均已否决。

### 3.4 两个环境的实际库规模（**分表，禁止混比**）

**环境 A 历史：`4task-gl-framesamp`**（源库 `4task-gl`，4 任务 × 400 episode）——出处 [`docs/dataset-build-doc/4task-gl-framesamp/README.md`](dataset-build-doc/4task-gl-framesamp/README.md)：

| 项 | 值 |
|---|---|
| 源库体积 / 派生库体积 | 678 GB → **31.7 GB**（`du --apparent-size` 30 GiB；特征侧体积的约 1/9） |
| `num_rows` / `num_exec_samples` / `num_pos_rows` | 483,291 / 395,289 / 586 |
| part 数与尺寸 | **32 个**，前 31 个约 990–1020 MB + 末 1 个 620,691,456 B（末 part 覆盖 episodes[1573..1599]、9,471 行） |
| 小表 | pos 28,803,072 B（= 586 × 49,152）、state 15,465,312 B（= 483,291 × 32） |
| pack / verify 耗时 | 2,941 s（约 49 min）/ 1,061 s（约 17.7 min），16 进程 |
| 判定 | `PACK_DONE=1`、`VERIFY_PACK=PASS scanned=483291 mismatches=0`、`EXIT_CODE=0` |
| `store_meta.json` sha256 | `3990165c9cebffdadaceb01cc88470645a3d62f9af65eabccf4331d3fcd5b556`（G2 / G3 / `v1-singlerun-g0` 三个 run 的 env.json 实测同值） |

**环境 B 现行：`4task-motion-400ep/framesamp/`**（4 任务 × 100 episode = 400 episode 的公开版数据）——出处 [`docs/dataset-build-doc/4task-motion-400ep/result.md`](dataset-build-doc/4task-motion-400ep/result.md) 与库内 `meta/store_meta.json`：

| 项 | 值 |
|---|---|
| 源库体积 / 派生库体积 | `source/` 107 GB → `framesamp/` **7.6 GB** |
| `num_rows` / `num_exec_samples` / `num_pos_rows` | 123,044 / 101,066 / 586 |
| part 数 | **31 个**（每个约 258–264 MB） |
| 小表 | pos 28,803,072 B（与环境 A 逐字节同尺寸——`num_pos_rows` 同为 586）、state 3,937,408 B（= 123,044 × 32） |
| 判定 | `PACK_DONE=1`、`VERIFY_PACK=PASS scanned=123044 mismatches=0`，`status="verified"` |
| `store_meta.json` sha256 | `dffdd47b09aad2812bc46201231e49cd120a4498828d836c9d2695311a815642` |
| 清单 sha256 | `92fa17e97fba9434ee75302de12556319d8ce6d3feeb3adb9a397e830f477223` |

另有开发/闸门用的小库 `4task-motion-40ep/framesamp/`（`num_rows=13756`、`num_exec_samples=11530`、22 个 part、`status="verified"`），环境 B 的多数 CPU 级闸门跑在它上面。

### 3.5 训练期每 step 的对比表（环境 A 历史，b64 口径）

| 维度 | 重构前 | 重构后 |
|---|---|---|
| 每样本文件打开 | ≤33 次（1 pkl + ≤32 npy） | **1 次**（特征走常驻 fd） |
| 每样本读盘 | 均值 19.08 MB（上界 19.69），只用 3.95 MB | 均值 **2.43 MB**（上界 2.49），几乎全用到 |
| 反序列化 | ≤32 次全量 pickle | **0 次**（裸字节直读） |
| 线程池 | 每样本新建 ≤32 线程 | **无** |
| padding dtype | 显式 bf16 / f32（阶段 1 之后） | 相同（阶段 2 无 dtype 变更） |
| collate 后 batch 载荷 | 约 257 MB/batch | 相同 |
| 每 step 读盘 / 打开（b64） | 1.22 GB / 2,112 次 | 均值 **155 MB**（上界 159.5）/ **64 次** |
| 单样本耗时（热 / 冷） | 25.4 / 132 ms | 约 7 / 15–40 ms（计划期预估，实测以吞吐 run 为准） |

### 3.6 第四张表：motion

motion memory 接入后，同一个库根下另有一张 `motion/` 表（`LAYOUT="motion-768-grid16-v1"`），由 `MotionStore` 提供、与 framesamp 三表**共用同一份 `episode_manifest.json`** 并经 `check_same_source` 硬闸互校（四节 4.5）。它的窗口口径、行数公式、建库链、闸门与评估结果**全部不在本文范围**，正本是 [`docs/motion-memory.md`](motion-memory.md)。本文只在读路径（六节）与契约（四节）里说明它与 framesamp 三表的交互点。

---

## 四、`store_meta.json` 契约与两阶段闸

### 4.1 字段分组

`store_meta.json` 是打包库的**唯一契约**，结构化视图与逐项 fail-loud 校验在 `framesamp_store.py::StoreMeta.load`：

| 组 | 字段 | 作用 |
|---|---|---|
| 布局与格式 | `layout`、`tables{image_emb_4x4,pos_emb_4x4,state_emb}` 各自的 `row_shape`/`dtype`/`row_bytes`、`byte_order`、`array_order`、`bf16_encoding`、`writer_versions{python,numpy,ml_dtypes,git_commit}` | 读侧逐项与模块常量比对，任一不符即 raise |
| 身份与双根 | `manifest_sha256`（须等于当前 `episode_manifest.json` 顶层 `sha256`）、`manifest_path`、`source_dataset_root`（绝对路径，运行期可被环境变量覆盖）、`source_provenance_sha256`、`source_spot_sha256`（16 个抽样源文件摘要，读侧启动抽验） | 钉死「这份表是从哪份清单、哪个源库派生的」 |
| 规模与校验 | `num_rows` / `num_exec_samples` / `num_pos_rows`、`parts[]`（每 part 的精确 `start_row`/`num_rows`/`bytes`/`sha256`/`head_tail_digest`/`full_covered`）、`packer`、`verify` | 行区间连续覆盖与逐 part 完整性 |
| 小表与逐行摘要 | `small_tables{pos_emb_4x4,state_emb}` 各记相对路径/shape/dtype/byte_count/sha256；`row_digests` 记相对路径、`algo:"blake2b-128"`、`covered_rows`、覆盖范围（image‖pos‖state 三键原始位串拼接）、byte_count 与文件自身 sha256 | 行级独立证据 |
| 迷你库 | `manifest_scope ∈ {"full","subset"}`；`subset` 时必带 `subset_episodes[]` 与 `mini_manifest_sha256`，且 `subset_episodes` **只允许是 `global_episode_idx` 的连续前缀 `[0..k]`** | 前缀保证全局行号即物理行号、pkl 编号与全量域一致、偏移原值可用 |

`head_tail_digest` 是 blake2b-128 覆盖首尾各 1 MiB（文件 ≤ 2 MiB 时覆盖全文件并标 `full_covered: true`），实现见 `framesamp_store.py::headtail_digest`，与 `scripts/training/g0/check_baseline_env.py` 的 `_headtail_digest` 同口径。

### 4.2 两阶段写与两阶段闸

meta **两阶段写，均为 tmp + fsync + `os.replace` 原子落盘**：

1. **阶段 1 `pack` 结束**写 meta，`status:"packed"`、`verify:null`；
2. **阶段 2 `verify` 通过后**原子回填，`status:"verified"`；`verify` 期间继续持有 `meta/pack.lock`，回填完成后才删锁。

对应两道读侧闸，**位置刻意分层**：

- `require_verified(meta)`：`status != "verified"` 即 raise；显式设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1` 可放行但必打 WARNING（仅迷你库/开发期）。
- `require_no_pack_lock(store_root)`：`meta/pack.lock` 存在即 raise（打包或 verify 进行中/异常残留）。

**两道闸只落在分派层** `dataloader.py::_create_framesamp_dataset`，`FrameSampStore` 本身不看 `status`、不看锁——否则 `verify` 子命令持锁期间读不了自己正在验的库。

### 4.3 打包期验什么（写侧，100% 覆盖）

打包工具是 [`scripts/dataset/pack_framesamp_store.py`](../scripts/dataset/pack_framesamp_store.py)，四个子命令 `plan | pack | verify | report`。写入路径内的逐帧校验四步（`v2-framesamp-restructure-plan.md` A.2）：

1. 该帧 `pos_emb_4x4` ≟ `pos_table[t]`（memcmp，**钉死 t、不钉 g**——pos 是 t 的纯函数，数学上分不出「同 t 不同 episode」的调包）；
2. `state_emb` ≟ state 表同一行（同源自证，防行内错乱）；
3. episode slab 写 `part_XXX.bf16.bin.tmp` 后 `os.pread` 读回 memcmp（read-after-write）；
4. part sha256 → `os.replace` → 目录 fd `fsync` → 汇报父进程记 `pack_progress.jsonl`。

事务协议：`O_CREAT|O_EXCL` 建 `meta/pack.lock`（记 build_uuid/host/pid/开始时间；同 host 且 pid 存活则拒跑，pid 不存活判残锁需 `--resume` 显式接管，**异 host 一律拒跑**，破锁只能 `--force-break-lock`）；两张小表由主进程在并行阶段之前独写；image part 用 `multiprocessing.Pool(min(16, cpu))`、每 part 唯一属主天然无锁；progress 只由父进程追加，`--resume` 前先 `ftruncate` 掉尾部半行。

源读取两档 `--reader`：`decode`（默认，逐帧 `np.load(allow_pickle).item()` 全量反序列化，零布局假设）；`slice`（已实测 npy 内部偏移恒定，常量固化在 `framesamp_store.py` 的 `SOURCE_NPY_SIZE=602951` / `SOURCE_IMAGE_OFFSET=262595` / `SOURCE_POS_OFFSET=541352` / `SOURCE_STATE_OFFSET=602906`，三重守卫护航——留作重跑加速档）。

**「g 级零遗漏」的唯一凭据是 `verify` 子命令**：独立于 pack 的后验遍历，覆盖**全部** `(g,t)`，重新完整 decode 源 npy，三键各经**真实读 API**（`FrameSampStore.read_image_rows` / `pos_rows` / `state_rows`）逐行 memcmp，同时逐行产出 blake2b-128 摘要写 `meta/row_digests.blake2b.bin`。判定行 `VERIFY_PACK=PASS scanned=<num_rows> mismatches=0`。`--sample N` 抽样档只供开发期快检（10% 抽样对单行错位的漏检率约 90%），**不得用于交付判定**。

`pos` 表的来源是**从源库抽取拼装**（用若干 episode 凑齐 `t=0..585`，逐位同源、零后端风险），主 pass 的 100% memcmp 即证明「只依赖 t」。`PosEmb3D` 现生成只作旁证，且已实测 **CPU 后端生成与库中值不逐位一致（max|diff| ≈ 7e-7）、GPU 后端一致**——走生成路径必须校验 `jax.devices()[0].platform == "gpu"`，实现在 `pack_framesamp_store.py::generate_pos_table_posemb3d`。

### 4.4 读取期验什么

| 时机 | 函数 | 覆盖 |
|---|---|---|
| 主进程构造 `FrameSampDataset.__init__` | `StoreMeta.load` + `run_fast_checks` | 结构契约（layout / byte_order / tables 形制 / parts 连续覆盖 `[0,num_rows)` / subset 前缀合法性）+ `manifest_sha256` 现场重算比对 + 每 part 存在且 `st_size == meta.bytes` + 两张小表尺寸 + 抽 1 个 part 首尾 1 MiB 复验 `head_tail_digest` + 抽 1 条 `source_spot_sha256` 复验源库未动 |
| 每个 worker 首次 `__getitem__` | `FrameSampStore.__init__` → 同一套 `run_fast_checks` | 同上（按 pid 轮转抽样点），**worker 内恒 fast 档** |
| 独立 preflight（可选） | `run_full_checks` | 全部 part + 两张小表完整 sha256 对 meta 比对（约整库一读），能抓「同尺寸中部翻转」 |

`MMEVLA_FRAMESAMP_VERIFY ∈ {fast, full}` 选档，默认 `fast`。**`full` 禁止在性能 allocation 内执行**——这一读会把整个 store 预热进 page cache，性能 run 就不再测真实供给。

**fail-open 禁令**：packed 模式下任何校验不过一律直接 raise，**绝不回退散 npy**（旧链路在阶段 3 已整体删除，回退路径物理上也不存在了）。

### 4.5 与 motion 表的同源闸：`check_same_source`

motion 开启时，`motion_store.py::check_same_source(frame_meta.manifest_sha256, motion_meta, manifest)` 在两处被调用，钉死「framesamp 三表与 motion 表出自同一份清单」：

- `dataloader.py::_motion_gates`（分派层，随 `require_no_pack_lock` / `MotionMeta.load` / `require_verified` / stride 一致性 / `source_run` 与 store provenance 绑定一并检查）；
- `framesamp_dataset.py::FrameSampDataset.__init__`（装配层，另做逐 episode 身份互校，并预检每 episode 最大合法起点数 ≤ `motion.budget`，超过即报 episode 身份、**禁止静默裁剪**）。

同时 motion 开启态**不支持 subset 迷你库**（`motion_index` 覆盖全清单），检出即 raise。

---

## 五、index 派生：样本 idx → (g, t) → 行号

### 5.1 两个域与一个公式

链路里同时存在两套编号，**必须靠清单换算，禁止靠目录序**：

- **执行样本域**：`idx ∈ [0, num_exec_samples)`，`data/{idx}.pkl` 的编号，demo 前缀不出样本。
- **全 timestep 域**：帧号 `t`，含 demo 前缀帧；三张表的行号建在这个域上。

行号公式在 `framesamp_store.py::row_of`，**写侧与读侧共用同一个函数，物理上不可分叉**：

```
row(g, t) = manifest.episodes[g].total_sample_offset + t        # t 为全 timestep 域帧号
```

### 5.2 `build_exec_lookup`：O(1) 查表数组

`framesamp_store.py::build_exec_lookup(manifest, num_episodes=None)` 从清单派生三个数组，一次性建好，取数期零解析：

```
_epis_of[idx] = g                                  # int32
_step_of[idx] = ep.exec_start_idx + k              # int32；k 是该 episode 内第 k 个执行样本
_row_base[g]  = ep.total_sample_offset             # int64
```

**`exec_start_idx` 是这里最容易漏、也最致命的一项**：漏掉它，`VideoUnmask`（demo 前缀恒 66 帧）与 `VideoUnmaskSwap`（114–216 帧）的每个样本都会错位 66–216 帧，而错位后的帧仍然是合法帧、读得出来、不报错——只有身份互校（5.3）能拦住。

函数内三条显式 raise 兜住清单自身的错乱（一律 `raise`、不用 `assert`，`PYTHONOPTIMIZE=1` 会剥离 assert）：

- `ep["global_episode_idx"] != g` → 「清单 episodes 序错乱」；
- `ep["exec_sample_offset"] != cursor` → 「清单 exec_sample_offset 不连续」；
- 累计 `cursor != totals.exec_samples` → 「清单 exec_samples 总和不符」。

`num_episodes` 参数只服务 subset 迷你库：只取 `episodes` 的**连续前缀** `[0..num_episodes)`。前缀（而非任意子集）是硬约束——由此全局行号即物理行号、pkl 编号与全量域一致、`state` 表偏移原值可用，Dataset 长度与合法 idx 区间也都能沿用全量公式。

`FrameSampDataset.__init__` 拿到三个数组后立刻做一次总数核对：`len(self._epis_of) != meta.num_exec_samples` 即 raise（清单与库不配套的第一道闸）。

### 5.3 身份互校：行号错位的最后一道闸

`FrameSampDataset.__getitem__` 读完 pkl 后立刻比对：

```
pkl 内 data["epis_idx"] / data["step_idx"]   ≟   清单推导的 (g, step)
```

不符即 `raise RuntimeError("身份互校失败: …行号错位或清单/源库不配套")`。这条闸是「双根契约把源库指错了」「清单换代了但库没换」「`exec_start_idx` 漏乘」三类错误的共同兜底，代价是每样本两次 `.item()`。

### 5.4 均值选帧数（读盘字节帐的现场推导）

`framesamp_store.py::mean_sampled_frames(manifest, max_frames=32)` 用分段闭式求和算出 `Σ min(step+1, 32) / N`（避免几十万次逐样本循环），供量具现场推导每样本读盘字节，替代硬编码：

- 真实清单实测 = **30.996**（`max_frames=32`）；
- 旧链路每样本读盘 = pkl + `mean_frames × 602,951`（整包 npy）；
- packed 链路 = pkl + `mean_frames × 65,536`（只有 image 行走盘；pos/state 走进程内小表）。

`SOURCE_PKL_BYTES_FLOOR = 395_440` 是 pkl 的字节下界常量（内嵌变长字符串，实测）。

---

## 六、读路径：`FrameSampDataset.__getitem__` 逐步

装配层唯一实现在 [`src/mme_vla_suite/training/framesamp_dataset.py`](../src/mme_vla_suite/training/framesamp_dataset.py)，**单一路径、无分支、只服务 `perceptual-framesamp-context`**。

### 6.1 形制断言即文档

`__init__` 开头一串 `_req(...)` 显式 raise（不用 assert），**必须能挡住同形的 `modul` 配置**：`representation_type == "perceptual"`、`perceptual_memory.type == "frame_sampling"`、`integration_type == "context"`、`memory_token_dim == 2048`、`(budget, token_per_image, num_views) == (512, 16, 1)`、`memory_feature.img.input_dim == 2048`、`memory_feature.pos.input_dim == 768`、`use_state_emb is False`。motion 开启时另有一组形制断言（`dim` / `budget=96` / `pos_dim == pos.input_dim // 3` / `stride` / `window_frames` / `window_direction` / `grid_origin` / `frame_size` 与 motion store 常量逐项对齐）。

由 budget 推出的两个内部常量：`_max_frames = budget // (token_per_image × num_views) = 32`、`_tokens_per_frame = token_per_image × num_views = 16`。

### 6.2 取数八步

```
__getitem__(idx):
 ⓪ idx 越界检查（显式 IndexError）
 ① store = self._ensure_store()          每进程懒构造 + _owner_pid 校验（6.4）
 ② g, step = _epis_of[idx], _step_of[idx]          O(1) 数组，不读目录
 ③ pickle.load(<source_root>/data/{idx}.pkl)       与旧路径同源同字节
    ＋ 身份互校（5.3，不符 raise）
    ＋ data["actions"] = data["actions"][:action_horizon]（与旧路径逐字相同）
    ＋ data.pop("simple_subgoal_online") / data.pop("grounded_subgoal_online")
 ④ frames = even_sampling_indices(step, 32)        同一函数 import，选帧逐位不变（九节）
    rows   = _row_base[g] + frames
 ⑤ img = store.read_image_rows(rows)   (n,16,2048) bf16 —— 0 open、0 线程池、0 pickle
    pos = store.pos_rows(frames)       (n,16,768) f32  —— 进程内小表
    stt = store.state_rows(rows)       (n,8)     f32  —— 进程内小表
 ⑥ img, pos, stt, mask = self._pad(img, pos, stt, n)   预分配 + 填充区清零，零 concatenate
 ⑦ 拼装四个交付键：
      static_image_emb = img.reshape(-1, 2048)                       (512,2048) bf16
      static_pos_emb   = pos.reshape(-1, 768)                        (512,768)  f32
      static_state_emb = _normalize_state(np.repeat(stt, 16, axis=0))(512,8)
      static_mask      = np.repeat(mask, 16)                         (512,)     bool
 ⑧ [motion 开启时] 另拼 motion_emb / motion_pos / motion_mask / mem_order（6.6）
    for key in _NONE_KEYS: data.setdefault(key, None)   与旧路径尾部补空键逐字一致
```

`_pad` 是**单一实现**：按最终形状一次性 `np.empty` 分配（img bf16 / pos f32 / stt f32），前 n 行填真值、后面清零，`mask` 前 n 位 True。目标长度 32 是内部常量，第四个参数 `n` 的语义是**实际帧数**；`n > 32` 即 raise（`even_sampling` 契约被破坏）。填充零的 bf16/f32 位型同为全零字节，故交付 dtype 与旧路径 `right_padding_token_emb` 逐键一致、数值逐位一致。

`_normalize_state` 与旧路径逐字同式；因 norm stats 的 `q01`/`q99` 是 f64，**该键在 dataset 侧输出恒为 f64**。

### 6.3 `FrameSampStore` 的三个读 API

| API | 做什么 | 关键实现 |
|---|---|---|
| `read_image_rows(rows, out=None)` | 按全局行号读 image 表，返回 `(n,16,2048)` bf16 | ① 行号越界即 `IndexError`；② `_runs_of` 把行号序列切成连续游程再按 part 边界细分；③ 对每段发 `posix_fadvise(WILLNEED)` 触发并发预读（`ENOSYS`/`EOPNOTSUPP` 打一次 WARNING 后本进程永久跳过）；④ 按段 `os.preadv` 直读进预分配的 uint8 缓冲；⑤ `raw.view(bfloat16).reshape(...)` 返回 |
| `pos_rows(frames)` | 按帧号 `t` 查 pos 小表，返回 `(n,16,768)` f32 | 进程内整表副本切片，无缺页 |
| `state_rows(rows)` | 按全局行号查 state 小表，返回 `(n,8)` f32 | 同上 |

**短读处理**（`_pread_exact`）：`preadv` 返回不足**不立即判损坏**，从已读偏移续读；**读到 0 字节（EOF）或请求区间越出 part 边界立即 raise**——这才是完整性判据。短样本的 ≤32 帧天然连续且必在同一 part，因此绝大多数样本只发 1 次 `preadv`。

**小表为什么不用 mmap**：实测同一 NFS 上「32×64 KiB 常开 fd `pread` 0.33 ms」对「`np.memmap` 切片 2.4–2.5 ms（走缺页/revalidate）」，热路径不留任何 NFS mmap。两张小表用 `np.fromfile` 整表读入进程内存（构造后就地设 `.shape`、不走 `reshape` 视图，保证 `.base is None`，即进程内拥有内存的副本、非映射非视图）。

### 6.4 spawn 生命周期契约

| 环节 | 行为 |
|---|---|
| `FrameSampDataset.__init__`（主进程） | 只读 meta + fail-loud 静态校验 + 清单派生查表数组；**不打开任何 part fd、不建任何 mmap** |
| `__getstate__` | 剔除 `_store` 与 `_mstore` 句柄字段（置 None）、复位 `_atexit_registered`——Dataset 被 pickle 进 spawn worker 时不携带任何内核资源 |
| `_ensure_store()` | 每进程首次 `__getitem__` 时懒构造：`os.open` 全部 part fd（`O_RDONLY|O_CLOEXEC`）+ 两张小表 `np.fromfile`，记 `_owner_pid`；此后每次取数校验 pid，**不符先显式 `close()` 旧句柄再重建**（fd 属内核资源，不允许靠 GC 兜底） |
| `close()` + `atexit` | Dataset 转发关闭 store 与 motion store，并注册 `atexit` 兜底；w0 场景同 pid 复用同一 store，不重复建 |
| `FrameSampStore.__reduce__` | 直接 `raise TypeError`——禁止 pickle 跨进程携带，把「懒加载」从约定升格为机制 |

这条契约就是「每样本 0 次 open」承诺的实现路径。验收工具是 `scripts/training/tests/spawn_matrix.py`（w0/w1/w4/w16 × 2 epoch 真实 spawn loader，收官检查主进程 `/proc/self/fd` 计数回到基线）。

### 6.5 双根契约与环境变量

分派在 `dataloader.py::_create_framesamp_dataset`，packed 模式下**三个位置全部显式解析**，禁止从打包库目录名做字符串变换推导源库：

| 位置 | 来源 | 环境变量覆盖 |
|---|---|---|
| 打包库根 | `dataset_path`（CLI `--dataset-path`） | — |
| 源库根（pkl 所在） | `store_meta.source_dataset_root` | `MMEVLA_FRAMESAMP_SOURCE` |
| 清单 | `store_meta.manifest_path` | `MMEVLA_FRAMESAMP_MANIFEST` |
| 校验档 | `fast` | `MMEVLA_FRAMESAMP_VERIFY` |
| motion 库根 | `history_config.motion.store_path` | `MMEVLA_MOTION_STORE` |

另有两个开发期放行阀，放行必打 WARNING、判据 run 出现即 run 无效：`MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1`、`MMEVLA_FRAMESAMP_ALLOW_SUBSET=1`。

> **起跑纪律**：正式对拍 run 起跑前必须 `env | grep MMEVLA_FRAMESAMP` 输出为空。残留一个 `MMEVLA_FRAMESAMP_SOURCE` 就会从另一个源库读 pkl，**而环境指纹 preflight 的 `dataset_spot` 锚在 `--dataset` 命令行参数上、根本不会失配**——指纹全绿、交付字节已变（`v3-destructive-restructure-plan.md` 6.2 第 1.5 层）。

### 6.6 交付键与 shape/dtype 表

**单样本（dataset 侧交付，`[推导]` 锚点 `framesamp_dataset.py::FrameSampDataset.__getitem__`）**：

| 键 | shape | dtype | 说明 |
|---|---|---|---|
| `static_image_emb` | `(512, 2048)` | bfloat16 | `(32,16,2048)` reshape，C-order 字节不变 |
| `static_pos_emb` | `(512, 768)` | float32 | 同上 |
| `static_state_emb` | `(512, 8)` | **float64** | `_normalize_state` 的 norm stats q01/q99 为 f64 |
| `static_mask` | `(512,)` | bool | 帧 mask 按 16 重复 |
| `motion_emb` / `motion_pos` / `motion_mask` / `mem_order` | `(96,768)` / `(96,256)` / `(96,)` / `(608,)` | f32 / f32 / bool / int32 | motion 关闭态恒 `None`；语义见 [`docs/motion-memory.md`](motion-memory.md) |
| pkl 透传键 | — | — | `image` / `wrist_image` / `state` / `actions`（截断到 `action_horizon`）/ `prompt` / `epis_idx` / `step_idx` … |

**host batch（collate 之后、`TorchDataLoader.__iter__` 输出，`[实测]` b8 口径）**——出处 [`docs/training-doc/v1-singlerun-g0/appendix-d-datapath.md`](training-doc/v1-singlerun-g0/appendix-d-datapath.md) 的实测回填表（来源为 `train.main` 首 batch 的 `array_tree_to_info` 打印，**环境 A 历史，b8 / 2 卡口径**；字节量 = `prod(shape) × itemsize`）：

| 键 | shape@dtype | per-batch 字节量（b8） |
|---|---|---|
| `[0].images['base_0_rgb']` / `['left_wrist_0_rgb']` | `(8,224,224,3)@float32` ×2 | 各 4,816,896 B ≈ 4.59 MiB |
| `[0].image_masks[...]` ×2 | `(8,)@bool` ×2 | 各 8 B |
| `[0].state` | `(8,32)@float32` | 1,024 B |
| `[0].tokenized_prompt` / `_mask` | `(8,64)@int32` / `@bool` | 2,048 B / 512 B |
| `[0].static_image_emb` | `(8,512,2048)@bfloat16` | 16,777,216 B = 16 MiB |
| `[0].static_mask` | `(8,512)@bool` | 4,096 B |
| `[0].static_pos_emb` | `(8,512,768)@float32` | 12,582,912 B = 12 MiB |
| `[0].static_state_emb` | `(8,512,8)@float32` | 131,072 B = 128 KiB |
| `[1]`（actions） | `(8,20,32)@float32` | 20,480 B = 20 KiB |

合计约 **38.9 MiB / batch**，键数 12（11 个观测键 + actions），与 `batch_digests.jsonl` 首行 `n_keys=12` 一致。

> ⚠ **一处口径提示（不下断言）**：`static_state_emb` 在 dataset 侧输出 f64（6.2），而上表实测记为 `float32`。上表的观测点在 `TorchDataLoader.__iter__` 的 `jax.make_array_from_process_local_data` 之后，而 JAX 默认关闭 x64；**降位发生在哪一跳本文不下断言**，需要精确结论时以量具在对应观测点的实测为准。

---

## 七、dtype 统一摘要（阶段 1）

> 完整报告 [`docs/v1-phase2-dtype-unify-report.md`](archive/v1-phase2-dtype-unify-report.md)（**环境 A 产物，只读历史存档**）；源计划 [`v1-dtype-unify-plan.md`](../v1-dtype-unify-plan.md)。本节只做摘要与指针，不复述其验证明细与性能数字。

**问题**：旧链路的 `right_padding_token_emb` 里，三段 padding 用的是不带 `dtype` 的 `np.zeros`（默认 f64）。满长样本（`step ≥ 31`）走纯切片分支、交付 bf16/f32；短样本走 padding 分支、`np.concatenate` 把整块提升成 f64。于是**同一个键的 dtype 随 batch 里有没有短样本而摆动**，XLA 因此要编译两份产物，host 侧还多一次降精度搬运。

**改动**：一个函数、三行——img / pos / state 三个 `np.zeros` 各加 `dtype=对应输入.dtype`。`mask` 的 padding 本就显式 `dtype=np.bool_`，不动；满长分支不动。**改动即行为，无运行时开关**，A/B 是跨 commit 的（`a0f76f8` = `commitV2.4b`）。

**等价性**：bf16 → f32 → f64 是**精确升位**，三种交付进投影层的张量逐位相同。这一论断在真实模型真实数据上有直接实证——单步定点梯度三档（`mixed1` 主判据、`allshort` 密度最大、`allfull` 阴性对照）各 32 个梯度叶子逐位相同、`loss_bitwise=True`，判定行 `COMPARE_GRAD=PASS kinds=3 mismatches=0`（[`docs/training-doc/v1-dtype-p5-grad/result.md`](training-doc/v1-dtype-p5-grad/result.md)）；1000 步训练轨迹与 G0b 逐位一致（G1，见十一节）。

**与本文其余部分的接口**：

- packed 链路的 `FrameSampDataset._pad` **继承了这个契约**——它按最终形状一次性 `np.empty(dtype=输入.dtype)` 分配、填充区清零，从设计上就不存在「padding 段 dtype 与真值段不同」的可能。阶段 2 因此**不含任何 dtype 变更**，A/B 只剩「字节从哪读」一个变量。
- G0 固化产物的 **raw 口径 `batch_digests` 与阶段 1 之后的 HEAD 存在已知失配**：`BATCH_DIGEST mismatch=4 first_bad_step=100 bad_keys=2（static_image_emb / static_pos_emb）`。这是 dtype 统一的固有口径差、**预期失配**，此后每一轮对拍都必须与它**逐字吻合**（不吻合才是信号）；判据一律走 **canonical 口径**（逐键升 f32 后按位视图哈希）。
- **验收范围边界**：两块验证只覆盖 `perceptual-framesamp-context` 训练路径；同一函数也被在线评估路径与 `modulation`/`expert` 变体调用，那些路径不在验收范围内。
- `static_state_emb` 那一处修复在交付键与梯度上**均不可观测**（该键经 `_normalize_state` 后恒 f64），其唯一有效证据是纯函数位型测试，在 `scripts/training/tests/test_padding_dtype.py`。

---

## 八、破坏性单一化（阶段 3）

> 权威计划 [`v3-destructive-restructure-plan.md`](../v3-destructive-restructure-plan.md) 三 / 六 / 七节。计划里的 commit 编号是 `V3.8`–`V3.14`，**实际落地编号为 `commitV4.0`–`commitV4.6`**（七刀一一对应，顺序不变），引用 git 历史时以后者为准。

### 8.1 七刀与顺序

| 落地 commit | 计划编号 | 做什么 | 为什么排这个位置 |
|---|---|---|---|
| `commitV4.0` | V3.8 | 建库域自包含隔离：四个共用文件复制入 `dataset_builder/`，建库脚本改 import 指向；另加建库输出 `--force` 闸 | 训练侧零改动；先落地，后面删共用文件时建库链已安全 |
| `commitV4.1` | V3.9 | **数据链单一化**：删 legacy 读取器 `training/dataset.py` 与 `MMEVLA_DATA_BACKEND` 三态；norm stats 脚本内联最小读取器；删 `pi05_baseline` 训练入口；失败 record 改保留 | 数据入口先收口，后面改 transforms 才只有一个消费者 |
| `commitV4.2` | V3.10 | transforms 瘦身：删 recurrent / symbolic 的键与分支 | 必须在模型改动前——先让废字段断流，模型侧删除就成纯死代码清理 |
| `commitV4.3` | V3.11 | 模型侧单一化：删 recurrent / symbolic 模型分支与 stats 返回链、删 `inputs_spec` 的 `use_history` 二分 | 依赖上一刀已断流；二分的唯一依赖方已在 V4.1 删除 |
| `commitV4.4` | V3.12 | 在线评估改造：新写 `policies/framesamp_memory.py::FrameSampMemory`，训练/在线脱离 `mem_buffer`；`shared/` 重划（`even_sampling_indices` 搬入新模块 `shared/sampling.py`） | 不影响训练数值，但动训练侧 import 路径，须在长跑前 |
| `commitV4.5` | V3.13 | 脚本与示例清理；`runs/` → `v1-store/` 收敛 + 裸 `python` 换 `uv run` | 纯外围，不进训练链路 |
| `commitV4.6` | V3.14 | `scripts/` 目录统一：幸存脚本搬入 `scripts/training/` 与 `scripts/dataset/`，**逐文件根解析修正** | 最后搬移，G3 在最终布局上验收整条链 |
| — | G3 | 全部冻结后 clean HEAD 跑 1000 步确定性档、对拍 G0 固化产物 | 一次覆盖七个提交 |

### 8.2 训练链单一化后的形态

- `create_data_loader`（[`src/mme_vla_suite/training/dataloader.py`](../src/mme_vla_suite/training/dataloader.py)）压成**无条件 packed**：`get_history_config` → `_create_framesamp_dataset` → `transform_dataset` → `TorchDataLoader` → `DataLoaderImpl`。三态分派与 `RoboMMEDataset` import 一并消失，`MMEVLA_DATA_BACKEND` 这个环境变量在现行仓库**不再存在**。
- `FrameSampDataset` 的 `_NONE_KEYS` 从旧路径的补空键列表缩到当前 9 项（四个 `static_*` + `prompt` + motion 四键），recurrent 四键与 subgoal 两键随 V4.2 删除。`None` 不是 pytree 叶子，因此 `n_keys` 仍为 12。
- `RepackTransform`（`training/config.py::RoboMMEDataConfig.create`）同步删 6 键；`RoboMMEInputs` 删对应 6 个 `data.get`。
- 此后仓库**没有「无 history」的训练路径**：`pi05_baseline` TrainConfig、其启动脚本、`inputs_spec` 的 `use_history` 二分、`policy.py` 的 `config is None` 分支一并消失。

### 8.3 建库域隔离的两条机器判据

「将来再跑建库链，代码行为与今天逐位相同」这条保证不靠行为测试猜，靠三层：

1. **源码同一性（`COPY_DIFF`）**：四个文件从 `shared/` 复制进 `dataset_builder/` 时，`git show $COPY_BASE:<shared 路径> | diff -u - <dataset_builder 路径>`，`COPY_BASE=732fae3b13e2ff5f485d7014473b99ed577de387`（**钉死、不随 HEAD 走**——V4.4 之后 `shared/` 侧源文件或删或改，用浮动 `HEAD` 会 `fatal: path does not exist`）。三个叶子文件零差异，`mem_buffer.py` 差异必须恰好 3 行且全是 import 指向替换。判定行 `COPY_DIFF=PASS files=4 nonimport_lines=0 base=732fae3b`。
   - **注释语言豁免**：四个源文件实测含中文行数为 0，逐字节复制会在仓库里新增全英文注释文件。本轮明确声明冻结副本**视同 `git mv` 产物**、不适用「新增注释必须中文」，逐字节同一性优先。
   - V4.4 之后 `COPY_DIFF` 退为「可复算的历史证明」，**在岗的接力是 `scripts/dataset/test_guards.py` 里四条 sha256 哨兵**（四个副本各钉一条，防止有人把训练侧改动「同步」回建库域副本）。
2. **依赖闭合（`IMPORT_ISOLATION`）**：静态 grep 双向（建库域不 import `shared`/`training`/`policies`/`models`/`datastore`；训练/在线/量具不 import `dataset_builder`）+ 动态断言（全新解释器 import 建库入口后 `sys.modules` 零训练侧模块泄漏，这一条抓静态 grep 抓不到的传递依赖）。判定行 `IMPORT_ISOLATION=PASS builder_leaks=0 train_leaks=0 online_leaks=0 whitelisted=1`——白名单唯一一项是 `compare_online_memory.py`，它必须 import `dataset_builder.mem_buffer` 才能做「副本 vs 原件」的三方逐位对拍。
3. **同架构行为抽查（`BUILDER_SPOT`）**：只允许 `import spot_check` 函数、只对本机同架构库跑（跨架构逐位不可得）。**它是 float64 数值零差、不是逐位比对**，抓不到「同值但 dtype 变了」「±0.0 位型差」「字节布局不同」三类变化，只作诊断。

**建库输出 `--force` 闸**：`dataset_builder/build_robomme_dataset.py::DatasetProcessor.__init__` 原本在任何校验之前无条件 `shutil.rmtree(self.dataset_path)`；改为新增 `force: bool = False` 形参，目标已存在且 `force` 为假时**显式 raise**。CLI 开关在 [`scripts/dataset/build_dataset.py`](../scripts/dataset/build_dataset.py)。它挡的是「重建时少写一层目录名、把输出根传成 `v1-store/datasets` 或 `v1-store`」这种最现实的事故，**不挡「显式加了 `--force` 又传错路径」**，也不做 canonical containment 检查。

### 8.4 `scripts/` 两域与根解析修正

`scripts/` 顶层收敛成两个域文件夹（现行布局）：

```
scripts/
  training/   train.py、eval.sh、serve_policy.py、compute_results.py、compute_norm_stats.py、
              download_pi05_base.py、unzip_ckpt.py、gl_submit.py、finetune_mme_vla_suite.sh、
              paths.sh、__init__.py
              ├ g0/         bench_train_steps.py、check_baseline_env.py、compare_baseline.py、
              │             run_2gpu_epoch_bench.sh、compare_online_memory.py … （G 链量具）
              ├ prod/       正式训练启动器与 sbatch
              ├ tests/      对拍与守卫工具（十一节表）
              ├ util/       analyze_gpu_util.py、analyze_util.py、downsample_util_csv.py（纯事后分析器）
              └ legacy-eval/  评估链路（主线不再维护，见仓库内该目录说明）
  dataset/    build_dataset.py、scan_manifest.py、build_shard.py、finalize_checks.py、
              compare_datasets.py、test_guards.py、pack_framesamp_store.py、pack_motion_store.py、
              motion_checks.py、run_local.py、paths.sh、tarxz_h5.py、unzip_data.py、
              fetch_h5_16task.py、scan_16task_memory.py …
              ├ wan/        ├ hf_export/
```

> `v3-destructive-restructure-plan.md` 7.4 曾规划 `dataset/gl/` 与 `dataset/pack/` 两个子目录；**现行仓库是扁平布局**（建库脚本与打包脚本直接放在 `scripts/dataset/` 下），以本节为准。

**V4.6 不是「纯搬移」**：仓库里所有脚本都靠「我在第几层、往上数 N 层」定位仓库根，搬深一层后全部错位。后果分两类，第二类才是真正理由——

- **响亮的**：`check_baseline_env.py` 的 `_REPO_ROOT / "uv.lock"` 会读到 `<repo>/scripts/uv.lock`，直接 `FileNotFoundError`，一眼看见就修。
- **静默的**：建库域 `paths.sh` 用 `${V1_SCRIPT_DIR}/../..` 求根，搬深一层后得 `<repo>/scripts`，而它原来的 fail-loud 只做前缀匹配、照常通过；于是 `V1_STORE=<repo>/scripts/v1-store`，`v1_prepare_dirs()` 一路 `mkdir -p` 造出整棵假目录树，建库产物全写进去，不报错、不告警，还因为根 `.gitignore` 只有 `/v1-store/`（前导 `/` 只锚仓库根）而直接出现在 `git status` 里。

做法是**逐文件改对层数 + 给每个入口加一句自证断言**——算出来的根必须存在 `pyproject.toml`，否则立刻 fail-loud 退出（现行写法可见 [`scripts/dataset/run_local.py`](../scripts/dataset/run_local.py) 顶部：`if not (_REPO_ROOT / "pyproject.toml").exists(): raise SystemExit(...)`）。验收三条判定行：

```
RELOCATION_REFS=PASS old_refs=0          # 排除 docs/、v1-store/ 与计划文件后，全仓无旧路径引用
RELOCATION_ROOT=PASS entries=<N> mismatch=0  # 每个入口实测算出的根 == 仓库根（唯一能抓住 paths.sh 静默错位的判据）
RELOCATION_COLLECT=PASS errors=0         # uv run pytest --collect-only -q src scripts packages
```

### 8.5 `v1-store/` 单一根与环境 B 的形态

除最初的全局原始 H5 外，派生数据、索引、缓存、模型、tokenizer、checkpoint、日志与 smoke 产物一律收敛到单一根 `v1-store/`（整体不进 git，`AGENTS.md` 第 14 条）。该根随仓库走：

- **环境 A 历史**：位于 `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/`，其中旧产物以**只读 symlink 逐项引用** turbo 归档（`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/...`），禁止穿透 symlink 写入。
- **环境 B（当前）**：位于 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/`，**下面没有任何 symlink 外链，全部是实体目录**；turbo 与旧本机路径在本环境不存在，凡依赖它们的对拍、基线复用口径一律视作失效。

两个环境共同的纪律：凡带 `--force` 或输出根参数的命令，起跑前先 `ls -ld <输出根>` 确认它是本环境的实体目录；禁止覆盖 `HOME`，改为逐项显式设置 `UV_CACHE_DIR` / `XDG_CACHE_HOME` / `HF_HOME` / `JAX_COMPILATION_CACHE_DIR` 等指向 `v1-store/cache/`（现行做法见 `run_local.py::base_env`）。

---

## 九、RepackTransform、`even_sampling_indices` 与 spawn 生命周期

本节回答一个问题：**训练侧比在线侧多出来的那些步骤，为什么不改数。**

### 9.1 transform 链的顺序与组成

`openpi/training/data_loader.py::transform_dataset` 把四组变换按固定顺序串起来（Dataset → `TransformedDataset`）：

```
① data_config.repack_transforms.inputs   → RepackTransform（键名映射，training/config.py::RoboMMEDataConfig.create）
② data_config.data_transforms.inputs     → RoboMMEInputs（policies/robomme_policy.py）→ DeltaActions
③ Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
④ data_config.model_transforms.inputs    → InjectDefaultPrompt → ResizeImages(224,224)
                                            → TokenizePromptWithState(PaligemmaTokenizer(max_token_len))
                                            → PadStatesAndActions(action_dim)
```

`ResizeImages` 走的是 `openpi_client` 的 NumPy/PIL **CPU** 实现（不是 JAX 版本，两者同名不同模块）。

### 9.2 `RepackTransform`：改键名不改数

`RoboMMEDataConfig.create` 里的 `RepackTransform` 是一张**纯键名映射表**：

```
"observation/image"      : "image"
"observation/wrist_image": "wrist_image"
"observation/state"      : "state"
"actions" / "prompt"     : 同名
"static_image_emb" / "static_pos_emb" / "static_state_emb" / "static_mask" : 同名
"motion_emb" / "motion_pos" / "motion_mask" / "mem_order"                  : 同名
```

三条要点：

1. **它只搬引用，不做任何数值运算、不改 dtype、不改 shape**——所以「训练侧比在线侧多一个改键名步骤」不构成数值差异来源。
2. **未登记的键会被静默丢弃**：结构表里没有的键在 repack 之后就不存在了。反过来，**表里登记了而上游没给的键会硬索引 KeyError**——这正是「删交付键必须同时改 `RepackTransform` 与 `inputs_spec` 两处硬阻断」的原因（`v1-post-restructure-roadmap.md` 项 3）。
3. motion 四键在关闭态是 `None` 透传（pytree 空节点，不是叶子），因此 `n_keys` 恒为 12，开关 motion 不改变 batch 的叶子数。

### 9.3 `even_sampling_indices`：纯函数、同一份 import

选帧函数在 [`src/mme_vla_suite/shared/sampling.py`](../src/mme_vla_suite/shared/sampling.py)：

```python
def even_sampling_indices(step_idx: int, token_budget: int) -> list[int]:
    if step_idx < token_budget:
        return list(range(step_idx + 1))
    else:
        return np.linspace(0, step_idx, token_budget, dtype=np.int32).tolist()
```

- **零随机源、纯确定**：给定 `(step_idx, 32)` 输出唯一。`step_idx < 32` 取 `[0..step_idx]` 全部；否则在 `[0, step_idx]` 上均匀取 32 个（`linspace` 截断成 int32，`step_idx` 较小时会出现重复索引，这是既有语义，新旧链路一致）。
- **采样域含 demo 前缀帧**；`32 = budget(512) ÷ token_per_image(16)`。
- **新旧链路 import 同一个函数、不重写**——这是「每样本选哪 32 帧逐位不变」的构造性保证（`v2-framesamp-restructure-plan.md` 三节恒等链第 ②）。
- 该模块**单独成文件的原因**：它是 dataloader worker 导入链上唯一的 `shared` 依赖，原先住在 `data_utils.py` 会连带拉起 flax/jax（`pool_tokens_to_size` 用 nnx）；搬出后 worker 只需 numpy。函数体一个字符未动（V4.4）。

同模块另有 motion 交错用的三件（`MEM_ORDER_SENTINEL`、`pad_times`、`memory_order`），训练侧与在线侧**必须 import 同一份**——两侧各写一份不会报错，只会静默让在线看到与训练不同的次序。`memory_order` 产出后显式校验「是 0..N-1 的合法置换」，因为 `jnp.take_along_axis` 默认 `mode="fill"`（float 越界填 NaN、bool 越界填 True、负索引静默回绕），「界内但非置换」只有这道校验能拦。

### 9.4 worker 生命周期与「热路径不留 NFS mmap」

- `TorchDataLoader` 在 `num_workers > 0` 时用 **spawn** context、`persistent_workers=True`、`drop_last=True`、`collate_fn=_collate_fn`、`generator` 由 `seed` 固定（`openpi/training/data_loader.py::TorchDataLoader`）。`jax.process_count() > 1` 直接 raise——全部内容仅覆盖单进程多 GPU。
- `_collate_fn` 是 `jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)`，**在 worker 内执行**，batch 内 dtype 一致、零提升（阶段 1 的直接收益）。
- Dataset 跨进程只带路径 / meta / 查表数组（6.4 的 `__getstate__`），fd 与两张小表在 worker 内首次取数时懒构造。**热路径上没有任何 mmap**：大表走常驻 fd `preadv`，小表是进程内 `np.fromfile` 副本（每 worker 约 44 MB：pos 28.8 MB + state 15.5 MB，按环境 A 库口径）。
- `_worker_init_fn` 在 worker 内设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` / `XLA_PYTHON_CLIENT_ALLOCATOR=platform`。

---

## 十、`train.py` 单跑（阶段 4）

> 权威计划 [`v5.0-train-entry-restructure-plan.md`](../v5.0-train-entry-restructure-plan.md) 三 / 四 / 六节；落地 `commitV5.0`（`f641f40`）。入口现址 [`scripts/training/train.py`](../scripts/training/train.py)。

### 10.1 改之前为什么没人走官方入口

原 `__main__` 是：

```
main(_config.cli(), tentative_run=True)   # 第一遍：预热，step > tentative_run_step(=10) 即 break
time.sleep(20)
main(_config.cli())                       # 第二遍：本该是正式训练
```

第一遍已经 `initialize_checkpoint_dir` 把目录建了出来，第二遍在 `overwrite=False, resume=False` 且目录已存在时**必然 `FileExistsError`**。所以现状是：本机对拍/测速绕道 `g0/bench_train_steps.py`，集群绕道薄启动器。这条双跑路径在环境 B 也被独立复现过——[`docs/training-doc/repro-4a100-fsdp4/result.md`](training-doc/repro-4a100-fsdp4/result.md) 记录了官方代码 worktree 下「日志出现两次完整启动、两次 `Step 0` 数值一模一样、中间隔 20 秒」的形态，并说明这是预期行为、不是崩溃。

### 10.2 改之后

`__main__` 单跑，无 sleep、无二次调用；`main()` 签名去掉 `tentative_run`，`tentative_run_step` 定义与循环内 break 块一并删除（这些行在 `tentative_run=False` 下本就是死代码）。另加三样：

1. **两条 fail-loud 护栏**（置于 `main()` 体最前、早于权重加载与 JIT）：
   - `config.overwrite or config.resume` → raise。理由：checkpoint 只存 EMA 权重、不存优化器状态，续跑会丢 AdamW 动量并把 warmup 从头再爬一遍。
   - `not config.model.use_history` → raise。这是**唯一的静默失败模式**：`HistoryPi0.__init__` 的 `else` 分支（注释 `# safe setting`）会把 `history_config`/`integration_type` 置空，关掉 `use_history` 一定跑得通、一定不报错，静默训出不含记忆分支的模型，而判读链路零告警。
2. **可选 metrics 记录器**：`TRAIN_RECORD_DIR` 设置时，`__main__` 路径把模块全局 `wandb` 换成透明代理，逐 log 追加 `metrics.jsonl`（`{"step":…, "wall_time":…, "<key>":{"dec":…, "hex":float.hex()}}`），`finally` 写 `run_meta.json` 并 `wandb.finish()`。未设时零行为差异。防覆盖护栏：**先 `mkdir(parents=True, exist_ok=True)`，再只按 `metrics.jsonl` 是否已存在判**，不按目录非空判（sbatch 会先建目录、写 `env.json`、起采样器，那时目录里已有多个文件）。
3. `batch % fsdp_devices` 不再单独设护栏：`main()` 体最前已有 `config.batch_size % jax.device_count() != 0` 检查，`openpi/training/sharding.py::make_mesh` 另有 `jax.device_count() % num_fsdp_devices != 0` 检查，由整除传递性两道同时通过即保证 `batch_size % fsdp_devices == 0`。

**主张边界**：本改动**只**消除双跑造成的 `FileExistsError`；官方 shell 与 README 的示例因另外两处独立阻断仍不可原样跑通，修它们不在阶段 4 范围。

### 10.3 `init_history_config`：run 根写三样

`train.py::init_history_config(config, resolved_history_config, framesamp_root)` 在 checkpoint 根写下训练侧的 provenance（`AGENTS.md` 第 12 条的机器化落地）：

| 文件 | 内容 |
|---|---|
| `history_config.txt` | 只作源文件名标签（旧口径，评估侧兼容路径仍读它） |
| `history_config.resolved.yaml` + `.sha256` | 训练**实际使用**的完整解析结果与其字节 sha256 |
| `motion_provenance.json` | `motion_enabled`、`framesamp_root`、`framesamp_manifest_sha256`、`framesamp_store_meta_sha256`；motion 开启时另填 motion 侧 manifest / index sha / store meta sha 与 VAE、encoder provenance，**关闭态对这些 motion-only 字段写 null、禁止省键** |

两侧 enabled 同源是硬要求：`main()` 只按 CLI 文件名解析一次 `get_history_config(history_config_name)`，同一个 `DictConfig` 对象既装入 model config 又直接传给 dataloader，**禁止两侧再次按文件名重读**——数据侧给了、模型侧不消费会让 `n_keys` 悄悄变而训练照跑。`init_history_config` 收到非字符串的 `history_config` 即 raise。

### 10.4 `mme_vla_suite_b128` 配置要点

现行正式训练配置条目在 `training/config.py::_CONFIGS` 的 `name="mme_vla_suite_b128"`：

| 键 | 值 |
|---|---|
| `model` | `HistoryPi0Config(pi05=True, action_horizon=20, use_history=True, history_config=None, discrete_state_input=False)`；`history_config` 由 CLI 传文件名 |
| `data` | `RoboMMEDataConfig(repo_id="robomme", assets=AssetsConfig(assets_dir="v1-store/train-assets/mme_vla_suite/robomme-400ep", asset_id="robomme"), base_config=DataConfig(prompt_from_task=True))` |
| `batch_size` | **128**（global） |
| `lr_schedule` | `CosineDecaySchedule(warmup_steps=5_000, peak_lr=1e-4, decay_steps=50_000, decay_lr=1e-4)` |
| `optimizer` | `AdamW(clip_gradient_norm=1.0)` |
| `num_train_steps` / `save_interval` / `keep_period` | 40_000 / 5_000 / 5_000 |
| `num_workers` | 8（**正式 run 由命令行覆盖为 16**，属启动脚本覆盖参数，`AGENTS.md` 第 10 条） |
| `ema_decay` | 0.999 |
| `fsdp_devices` | **4** |
| `weight_loader` | `CheckpointWeightLoader(<OPENPI_DATA_HOME>/openpi-assets/checkpoints/pi05_base/params)` |

**`fsdp_devices` 的语义**：它是 mesh 里 FSDP 轴的大小，不是「用几张卡」。8 卡可见 + `fsdp_devices=4` 得到 mesh `(2,4)`——数据并行 2 × FSDP 4。环境 B 选定档位就是这一组（十一节 11.3）。改动 `--dataset-path` 必须指打包库根（`<lib>/framesamp`），源库与清单走 `store_meta` 或环境变量覆盖（6.5）。

---

## 十一、对拍体系与吞吐（分环境）

### 11.1 梯度对拍链：只列名与指针

五个节点全部是 **1000 步、b8、seed 42、2 卡 fsdp2、workers 4、确定性档**（`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`）的真实训练，**环境 A 历史**；判据是与 G0 固化产物的四分项逐位对拍：

| 节点 | run_name / 留档 | 覆盖的阶段 | 收官 `sha256(scalars_hex.tsv)` |
|---|---|---|---|
| **G0**（黄金基线，链头） | [`docs/training-doc/v1-grad-baseline-g0b/`](training-doc/v1-grad-baseline-g0b/result.md)（r1/r2 两轮） | 重构前训练语义 | `c799a0b2…105757`（两份） |
| **G1** | dtype 修复后 1000 步 A/B（中间留档见 [`docs/archive/training-doc/v1-dtype-ab-post-r1/`](archive/training-doc/v1-dtype-ab-post-r1/)；单步梯度锚 [`v1-dtype-p5-grad`](training-doc/v1-dtype-p5-grad/result.md) 原地保留） | 阶段 1 | 同值 |
| **G2** | [`docs/training-doc/v1-framesamp-g2/`](training-doc/v1-framesamp-g2/result.md) | 阶段 2（packed IO） | 同值 |
| **G3** | [`docs/training-doc/v1-postclean-g3/`](training-doc/v1-postclean-g3/result.md) | 阶段 3（七刀合起来） | 同值 |
| **入口重锚** | [`docs/training-doc/v1-singlerun-g0/`](training-doc/v1-singlerun-g0/result.md) | 阶段 4 | 同值（第六份），总闸 `G0_EQ=PASS` |

四分项判定行（每个节点逐行核对，**退出码与总行 `DET_CHECK` 不具判据资格**——raw 口径的已知预期失配会拖累总行）：

```
SCALARS steps=1000 keys=5 hex_mismatch_steps=0        # 逐步五标量 IEEE 浮点位
STATE_DIGEST rows=12 mismatch=0                       # 12 摘要步完整 TrainState 逐叶 sha
BATCH_DIGEST_CANONICAL rows=14 mismatch=0 ; CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072                                 # 前 8000 条抽样顺序前缀
BATCH_DIGEST rows=14 mismatch=4 bad_keys=2            # ← 非判据，dtype 统一已知预期失配，须逐字吻合
```

三处必须人工/机器补位的 fail-open（`compare_baseline.py` 自带）：缺 scalar key 会静默 `continue` 但判定行仍打 `keys=5`；`INDEX_SEQ` 只比两侧最短公共前缀；`canonical` 与 `INDEX_SEQ` 不进最终 verdict。阶段 4 起由 `scripts/training/tests/g0_gate.py` 做 **fail-closed 外层总闸**，把判读收敛成唯一一行 `G0_EQ=PASS`（补检 `scalars_hex.tsv` 表头恰六列 + 行数恰 1001 + sha256 命中锚点、`batch_digests.jsonl` 首行 `n_keys=12`、本侧 index `n ≥ 8072`、`BASELINE_ENV=PASS`、raw `BATCH_DIGEST mismatch` 恰为 4 且失配键恰为那两个）。

**引用 G0 产物前必须过环境指纹 preflight**：`scripts/training/g0/check_baseline_env.py check --baseline <G0b-r1> …`，输出 `BASELINE_ENV=PASS|FAIL`，逐项比对 uv.lock 全量 sha、torch/jax/jaxlib/numpy/ml_dtypes 版本、GPU 型号与驱动、norm_stats/tokenizer/权重/数据集抽样指纹等数十项。**指纹不含仓库代码 sha**——PASS 只证「还有资格引用 G0」，不证「代码没改坏」。

### 11.2 非训练轻量对拍与守卫（工具名与判定行）

| 工具（`scripts/training/tests/` 除非另注） | 判定行 | 覆盖 |
|---|---|---|
| `dump_index_seq.py` | `INDEX_SEQ_EQ` | 两侧 Dataset 同 `TorchDataLoader` 同 seed dump index 序列，w0/w4/w8 三档 diff 为空（dump 步数 < 1 个 epoch，避开跨 epoch 既有分叉） |
| （阶段 2 的 `compare_batches.py`，A 侧消失后随 V4.1 删除） | `COMPARE_BATCH=PASS` | 约 8,200 个定点样本（step 边界全覆盖 + 每 episode 首样本 + 随机）在 **transform 之后**逐样本全键对拍 + 200 个真实 batch 过 `_collate_fn`；判据全键 shape/dtype/`view(uintN)` 逐位零容差 |
| `test_pack_guards.py` | pytest | Store 组（G1/G4/G5/G7/G11/G12/G14）+ Dataset 组（G2/G3/G6a/G8/G9/G10/G13）守卫，刻意制造失败断言亮红灯 |
| `spawn_matrix.py` | fd 计数回基线 | 迷你库真实 spawn loader 矩阵 w0/w1/w4/w16 × 2 epoch + fd 泄漏检查 |
| `single_step_grad.py` + `compare_grad_summaries.py` | `COMPARE_GRAD=PASS kinds=3 mismatches=0` | 单步定点梯度取证（固定初始 TrainState + 固定 batch），比 1000 步轨迹便宜两个数量级，能定位到具体哪个参数叶子先分叉 |
| `dump_fixture_samples.py` + `compare_dtype_fix.py`（驱动 `run_dtype_dump.sh` / `run_dtype_grad.sh`） | 四条零容差判据 | 阶段 1 的定点样本 / batch dump 与离线对拍 |
| `test_padding_dtype.py` | pytest | `right_padding_token_emb` 纯函数位型测试（`static_state_emb` 那处修复的唯一有效验证） |
| `project_scalars.py` | — | `metrics.jsonl → scalars_hex.tsv` 的**唯一规范投影**（两处共用，口径不一致则 sha256 必然不同） |
| `g0_gate.py` / `entry_equiv.py` | `G0_EQ=PASS` / `ENTRY_EQ=PASS` | 阶段 4 的 fail-closed 总闸与上游 main 分支对拍 harness |
| `closed_equiv.py` | `A13/A15/A17` | motion 关闭态轻量等价对拍（旧码 worktree vs HEAD） |
| `dataloader_bench.py` | 样本/s | dataloader-only 微基准（不建模型、不 device_put） |
| `scripts/dataset/test_guards.py` | 4 条 sha256 哨兵 + `--force` 负向用例 | 建库域冻结副本防发散（8.3） |
| `scripts/dataset/pack_framesamp_store.py verify` | `VERIFY_PACK=PASS scanned=<N> mismatches=0` | 全量写×读对拍，「g 级零遗漏」唯一凭据（四节 4.3） |

### 11.3 吞吐：环境 B（当前，最终指标）

介质：**AWS 本地 NVMe RAID（`/dev/md0`）**；库 `4task-motion-400ep`；global batch 128 / seed 42 / `--log-interval 10` / 300 步 / **非确定性 XLA**。出处 [`docs/training-doc/bench-b128-util/result.md`](training-doc/bench-b128-util/result.md)。

| 档 | 卡 | worker | 条件 | s/step | 中位步时 | util 均值 | 0% 占比 | 慢步 util | 非慢步 util | 40k 外推 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 4 (0-3) | 8 | 独占 | 3.831 | 3.643 | **94.75%** | 4.21% | 99.96% | 93.73% | 42.6 h |
| B2 | 4 (0-3) | 16 | 与 B3 并行 | 4.256 | 3.655 | 84.83% | 14.14% | 73.72% | 90.49% | 47.3 h |
| B3 | 4 (4-7) | 12 | 与 B2 并行 | 4.337 | 3.645 | 84.17% | 14.97% | 59.41% | 97.37% | 48.2 h |
| **B4（选定）** | **8** | **16** | 独占 | **2.656** | 2.030 | 70.25% | 27.49% | 47.25% | 91.81% | **29.5 h** |
| B5 | 8 | 32 | 独占 | 2.827 | **1.889** | 66.36% | 32.35% | 23.16% | 95.54% | 31.4 h |

四条结论（原文见留档）：① worker 数不是 4 卡档的瓶颈；② 两条 4 卡 run 并行的互扰幅度是 s/step +11%~+13%、util 94.75% → 84%；③ 变慢的机制是「慢步变多变重」而非「每步变慢」——**慢步 util 的方向反转**（独占 4 卡 99.96%「慢步在算」→ 8 卡 47.25%/23.16%「慢步在等数据」）是数据供给成为瓶颈的直接判据；④ 8 卡计算扩展性良好（中位步时 3.643 → 2.030，接近理想减半），瓶颈在 host→device 供给。**选定 B4**：util 是效率指标、不是目标函数，墙钟才是——8 卡即便空转 27.5% 仍比 4 卡满载快约 13 h。

供给侧对照（同环境，出处 [`docs/dataset-build-doc/4task-motion-400ep/result.md`](dataset-build-doc/4task-motion-400ep/result.md) 的 `dataloader_bench.py`，b64、warmup 5、measure 40，**与其他任务并行时测得、属下界**）：

| 档 | 样本/s | 每批 pickle 载荷 | `Pipe` 往返 |
|---|---|---|---|
| motion 关闭 w4 / w8 | 82.0 / 82.4 | 262.3 MB | — |
| motion 开启 w4 / w8 | 79.0 / 80.4 | 287.6 MB | 732.5 ms（392.6 MB/s 单向） |

B1 档需求 = 3.831 s/step × 128 = 33.4 样本/s，供给对需求约 2.4× 余量。

另一条环境 B 观测是 [`docs/training-doc/repro-4a100-fsdp4/result.md`](training-doc/repro-4a100-fsdp4/result.md)：官方原版代码（worktree `official-89efeaab`）在 4×A100 + `--fsdp-devices=4` 下跑通 20 步，两档（模拟 38 G 显存 + 关 P2P / 76 G + NVLink）`EXIT_CODE=0`，`Step 0` 数值一致（`param_norm=1815.5465`）。**它的 util 与步时不可作吞吐基准**（只跑 20 步、含编译期、两档同场次并行争 CPU 与 NVMe IO、采样窗口覆盖了两遍启动之间的 `sleep(20)`）；它的 batch 形状打印是**官方原版 legacy 链路 + modulation 变体**的口径（`static_image_emb (64,512,2048)@float32`），与本文六节 packed 链路的 bf16 交付不是同一条链，**不可混读**。

### 11.4 吞吐：环境 A（历史）

**本文不列环境 A 的吞吐数字**（`AGENTS.md` 第 13 条：turbo NFS / RTX 6000 Ada / A40 的数字与环境 B 不可混比，且所引 `v1-store` 产物在环境 B 不存在）。需要时见归档报告：

- [`docs/v1-nfs-bottleneck-analysis.md`](archive/v1-nfs-bottleneck-analysis.md)、[`docs/v1-gl-resource-tier-bench.md`](archive/v1-gl-resource-tier-bench.md)（原地保留）；
- [`docs/archive/training-doc/`](archive/README.md) 下的 A 组 15 项（`v1-e2e-b64`、`v1-e2efix-w{8,12,16}c16`、`v1-g0-speed-r2`、`v1-g1-speed`、`v1-framesamp-{dl,e2e,cmp}`、`v1-gl-dlbench`、`v1-2gpu-epoch-bench-b8`、`v1-coldcache-b8`、`v1-computeonly-b64`、`v1-prod-trend-10h` 等）；
- 两份阶段报告的性能节：[`v1-phase1-gradient-baseline-report.md`](archive/v1-phase1-gradient-baseline-report.md)、[`v1-phase2-dtype-unify-report.md`](archive/v1-phase2-dtype-unify-report.md)。

### 11.5 已裁定不立项的三个加速项

[`v1-post-restructure-roadmap.md`](../v1-post-restructure-roadmap.md) 的决策门在 2026-08-28 回填结论：**走第一条分支，util 已达 95%+，三个加速项全部不立项**（环境 A 生产口径双 seed 双节点 `E2E95_ACCEPT` 双 PASS）。三条具体裁定：

1. **项 1（worker JAX 导入链根因处置）不立项，且其现象前提已被证伪**——诊断实测集群上全程只有主进程一个 pid，「每 worker 在 GPU0 建 442 MiB CUDA context」不成立。
2. **项 2（`pos_emb` 移出 batch、GPU 侧生成）不立项**——性能动机消失。
3. **项 3（`static_state_emb` 白算白传清除）不立项**——同上。

关键归因：此前 80–90% 一族数字系**量具每步强制 `jax.device_get` 打断流水线**所致，非数据供应瓶颈；同配置同节点仅改日志粒度，util 88.1% → 99.7%。项 2/3 若将来因**可读性或正确性**理由重提，须按 roadmap 各自节内的迁移矩阵与前置条件独立立项，**不得援引当时的性能数据作依据**。项 4（`MemoryBuffer` 可读性拆分）的训练侧动机已随 `FrameSampDataset` 交付自动消失。

> 上述结论建立在**环境 A 的生产口径**上。环境 B 的形态不同（11.3 显示 8 卡档 0% 采样 27.5%、瓶颈在 host→device 供给），**若将来要在环境 B 重开加速项，必须在环境 B 重测后另行立项**，不得直接沿用上面的「不立项」结论。

---

## 十二、溯源

### 12.1 各节 → 计划文件 / 报告

| 本文节 | 主要来源 |
|---|---|
| 一、结论先行 | 四份计划的结论节 + 十一节各 run 留档 |
| 二、重构前链路 | `v2-framesamp-restructure-plan.md` 一节（1.1–1.6）；瓶颈判定指针见 2.5 |
| 三、重构后三张表 | `v2-framesamp-restructure-plan.md` 二节（2.1–2.4）与 A.1；`framesamp_store.py` 模块常量；两个库的 `meta/store_meta.json` 与两份 dataset-build-doc |
| 四、`store_meta.json` 契约与两阶段闸 | `v2-framesamp-restructure-plan.md` A.1 / A.2 / B.2；`framesamp_store.py::{StoreMeta,run_fast_checks,run_full_checks,require_verified,require_no_pack_lock}`；`pack_framesamp_store.py` |
| 五、index 派生 | `v2-framesamp-restructure-plan.md` A.1 / B.3；`framesamp_store.py::{row_of,build_exec_lookup,mean_sampled_frames}` |
| 六、读路径 | `v2-framesamp-restructure-plan.md` B.2 / B.3 / B.4；`framesamp_dataset.py::FrameSampDataset`；`dataloader.py::{_create_framesamp_dataset,_motion_gates}`；实测表出自 `docs/training-doc/v1-singlerun-g0/appendix-d-datapath.md` |
| 七、dtype 统一 | `docs/v1-phase2-dtype-unify-report.md`（指针，不复述）；`v1-dtype-unify-plan.md` |
| 八、破坏性单一化 | `v3-destructive-restructure-plan.md` 三 / 六 / 七节；现行 `scripts/` 布局实况 |
| 九、Repack / 选帧 / spawn | `training/config.py::RoboMMEDataConfig.create`；`shared/sampling.py`；`openpi/training/data_loader.py::{transform_dataset,TorchDataLoader,_collate_fn}` |
| 十、`train.py` 单跑 | `v5.0-train-entry-restructure-plan.md` 三 / 四 / 六节与附录 A；`scripts/training/train.py::{main,init_history_config}`；`training/config.py::_CONFIGS` 的 `mme_vla_suite_b128` |
| 十一、对拍与吞吐 | 五个 G 链 run 留档；`scripts/training/tests/`；`bench-b128-util` / `repro-4a100-fsdp4` / `4task-motion-400ep` 三份留档；`v1-post-restructure-roadmap.md` |

### 12.2 阶段 → commit

| 阶段 | commit 区间 | 关键 commit |
|---|---|---|
| 前置：G0 黄金基线 | `commitV2.1`–`commitV2.3.1` | `9eb2aaa`（G0 固化）、`570287f`（P1b 量具补遗 + canonical 口径） |
| 阶段 1：dtype 统一 | `commitV2.4a`–`commitV2.4b` | `a0f76f8`（三行修复） |
| 阶段 2：IO 重构 | `commitV3.0`–`commitV3.7` | `ad6559a`（格式层与打包工具）、`6ee7494`（`FrameSampDataset` + 三态接线）、`cf43e4c`（S5 第一块收官）、`2659ec1`（S6 / G2 收官） |
| 阶段 3：破坏性单一化 | `commitV4.0`–`commitV4.6` | `e5b238d`、`2699fa5`（数据链单一化）、`c17a928`、`07702f0`、`fc77bf0`、`056ffe4`、`b30be80`（目录统一） |
| 阶段 4：训练入口 | `commitV5.0` | `f641f40` |

其他钉死的锚点常量：`COPY_BASE=732fae3b13e2ff5f485d7014473b99ed577de387`（建库域副本 diff 的 A 侧锚）；`sha256(scalars_hex.tsv)=c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`（五节点六份同值）；环境 A 打包库 `store_meta` sha256 `3990165c…b556`；环境 B 打包库 `store_meta` sha256 `dffdd47b…5642`。

### 12.3 留档路径速查

| 类别 | 路径 |
|---|---|
| 梯度对拍链（环境 A，原地保留） | `docs/training-doc/{v1-grad-baseline-g0,v1-grad-baseline-g0b,v1-dtype-p5-grad,v1-framesamp-g2,v1-postclean-g3,v1-singlerun-g0,v1-l0-gauge}/` |
| 环境 A 吞吐 / 确定性 / dtype 中间留档 | `docs/archive/training-doc/`（清单与归档理由见 [`docs/archive/README.md`](archive/README.md)） |
| 打包库构建留档 | 环境 A：[`docs/dataset-build-doc/4task-gl-framesamp/`](dataset-build-doc/4task-gl-framesamp/README.md)；环境 B：[`docs/dataset-build-doc/4task-motion-400ep/`](dataset-build-doc/4task-motion-400ep/result.md) |
| 环境 B 吞吐 | [`docs/training-doc/bench-b128-util/`](training-doc/bench-b128-util/result.md)、[`docs/training-doc/repro-4a100-fsdp4/`](training-doc/repro-4a100-fsdp4/result.md) |
| 阶段报告（环境 A，只读） | [`docs/v1-phase1-gradient-baseline-report.md`](archive/v1-phase1-gradient-baseline-report.md)、[`docs/v1-phase2-dtype-unify-report.md`](archive/v1-phase2-dtype-unify-report.md) |
| motion 正本 | [`docs/motion-memory.md`](motion-memory.md) |

### 12.4 已知边界（如实声明）

1. **`v1-framesamp-restructure-plan.md` 已被 v2 取代**：2026-08-27 用户拍板彻底重构为独立可执行单一文档，该文件只作历史存档、不再更新；其 v4 版本已把 dtype 统一拆成前置计划、删掉 replica/f32 交付模式与 `MMEVLA_FRAMESAMP_DTYPE` 三态开关。**内容冲突时以 v2 为准，v2 与本文冲突时以本文为准。**
2. **1000 步 × b8 = 8,000 样本的覆盖范围有限**：只覆盖粗差（行号错位、选帧错误会在头几十步撞穿）；「万分之一错帧」类细差的覆盖责任在定点对拍与全量 `verify`。
3. **G 链的 bitwise 结论成立于本机受控确定性档**，且全部产于**环境 A**；环境 B 没有这些固化产物（A100 上 bf16 归约与 Ada 不逐位），环境 B 的等价性走的是「同机两侧各跑一次」的口径（`compare_grad_summaries.py` 文件头有说明）。
4. **`compute_norm_stats.py` 在阶段 3 内联了最小 pkl 读取器，但本轮无新旧输出对拍**（G 链全程只消费既有 `norm_stats.json`），口径差要到下次重算才显形。
5. **建库链的「GL 侧不重跑」**：8.3 的三层保证证到的是「代码逐字节同一 + 本机同架构复算逐位一致」，不是「在集群上重建一遍逐位相同」；后者是推论而非实测。
6. **`static_state_emb` 的 dtype 观测口径**见 6.6 末尾的提示，本文不下断言。
