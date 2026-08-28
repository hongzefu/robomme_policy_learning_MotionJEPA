# v2：framesamp 数据链路 IO 重构与基线对拍计划（独立可执行版）

> **本文件是自包含的单一权威文档**（2026-08-27 定稿，用户拍板彻底重构）：执行本计划的人和 agent **不需要、也不应该**再阅读任何其他计划 md——重构前后链路、验证判据、实施顺序、基线定义、登记簿全部内联在本文件。历史沿革详见 git 历史，不在此复述。
>
> **范围**：只兼容 `perceptual-framesamp-context` 一种 run。对拍与验收只关注五样：**G0**（黄金基线固化产物）、**G2**（packed IO 重构后）、**G0-speed**（本机速度锚点）、**G2-speed**（重构后本机速度）、**GreatLakes 吞吐验收**。
>
> **当前状态**：G0、G0-speed 已固化（登记簿 T8）；IO 重构 2026-08-27 拍板开工（编号 V3.0 始，原 V2.5–V2.9 作废）。**同日已完成：阶段 1（V3.0/V3.1）、阶段 2 S4（`VERIFY_PACK=PASS scanned=483291 mismatches=0`）、阶段 3 双过——S5 第一块（commitV3.3，零失配）与 S6 第二块 G2（登记簿 T8，bitwise 全过）——「IO 重构不改变训练语义」已按放行规则第 1 条正式成立。** 余下阶段 4（S7.5→S8a→S8b→S9，GL 验收与本机速度对账）。

## Context（为什么做这件事）

- GL 4×A40 端到端实测（留档 `docs/training-doc/v1-e2e-b64/`）：GPU util 均值仅 69.7%（中位 100% 是假象）、0% 采样占比 27.8%、慢步占稳态墙钟 32.9%；步时中位 6.933 s，而 compute-only 下界 4.778 s（+45%）。NFS 带宽已排除（供给 398–628 MB/s vs 需求 256 MB/s），坐实瓶颈在 dataloader worker 的 CPU/文件层。
- 纯参数调整不解决问题已有三档完整实据：w8c16 **5.301 s / util 71.2%**、w12c16 **5.319 s / 70.6%**、w16c16 **5.327 s / 67.1%**——workers 8/12/16 曲线完全平坦，三档均距下界约 11%。需要代码级 IO 重构。
- 硬性要求：**每个 step 拿到的 memory token 近乎一致、训练梯度差距极小**——本计划把目标提到「受控环境下逐位一致」，并用三块验证（三节）做成可证伪的梯子。整链方向性目标：**GPU 占用 100%**（north star，用户拍板；验收阈值见 D 节，不按字面 100% 定判据）。
- 基线锚点的意义：改动的等价性不靠「vs 自己改动前」的口头论证，而是对拍**一次跑定、固化进 git 的黄金基线 G0**——之后引用产物离线对拍，不必反复 checkout 旧 commit 重跑对照侧。

---

# 第一部分（给人看）

## 〇、符号总表（本文件唯一权威；一个 run 不得身兼两职）

**正确性族**证明「重构没改变训练结果」：带 TrainState 摘要 + batch 输入摘要 + 确定性 XLA 档（`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`）；其 util/步时仅留档参考，禁作性能结论。**性能族**回答「重构值不值得」：生产 XLA 档（不注入确定性 flags、autotune 默认开）、禁一切摘要。

| 符号 | run_name | 是什么 | 状态 |
|---|---|---|---|
| **G0** | `v1-grad-baseline-g0b-r{1,2}` | 重构前训练语义的黄金基线，本机 2 卡 b8 **1000 步**，两轮逐位自证，产物固化进 git（run_name 带 `g0b` 是历史命名，符号即 G0） | **已固化** |
| **G2** | `v1-framesamp-g2`（建议名，起跑前确认） | packed IO 后的正确性节点：packed 一轮 1000 步，离线对拍 G0 固化产物 | 待跑（S6） |
| **G0-speed** | `v1-g0-speed-r2` | 本机速度锚点：稳态中位 **1.152 s/step**（n=949）、均值 1.186、util 均值 86.5%、0% 采样 4.9%（1000 步口径） | **已固化** |
| **G2-speed** | `v1-g2-speed` | 重构后本机速度：vs G0-speed 的「重构前基线 → 重构后」合并对账，**不设阈值、只报数、不做单项归因** | 待跑（S9） |
| **GL 验收** | `v1-framesamp-dl-*` / `v1-framesamp-e2e-*` | GL 侧吞吐主判据（S8a/S8b），过/不过阈值在此判定 | 待跑 |

- **speed run 统一口径（权威定义）**：`bench_train_steps.py` 入口、本机 2×RTX 6000 Ada、b8、**1000 步**、seed 42、num_workers=4；不注入 `XLA_FLAGS`；`SAVE_INTERVAL=0`（驱动层自动联动 `BATCH_DIGESTS=0`）；`nvidia-smi -lms 500` 密集采样 + 15 s legacy 通道；报 util 稳态均值 / 0% 采样占比 / 慢步分层均值 / 步时中位与均值（AGENTS 16，禁中位数标题结论），标注「本机口径，不作最终吞吐结论」（AGENTS 13）。
- 每个 >5 min 的 run 按 AGENTS 17 留档 `docs/training-doc/<run_name>/`；run_name 起跑前按 AGENTS 6 逐个向用户确认，多轮场景一律 `-r<N>` 后缀。

## 一、重构前链路（从数据集预处理到每 step 的 memory token）

### 1.1 全景

```
   生产侧（一次性，已完成交付，本轮只读不动）        消费侧（训练期，每个 step 重复发生）
   ─────────────────────────────────────            ─────────────────────────────────────
   原始 H5（4 文件，321 GB）                         torch DataLoader 按 seed 抽 64 个样本 idx
     │  scan_manifest.py（只读 metadata）              │  分派给 spawn worker
     ▼                                                 ▼
   episode_manifest.json（唯一真值源）               RoboMMEDataset.__getitem__
     │  GL 8×1GPU job array（SigLIP 编码）             │  1 个 pkl + ≤32 个 npy + padding + transforms
     ▼                                                 │  → worker 内 collate（np.stack 合并成 batch）
   4task-gl 库（678 GB）────────────────────────►      ▼
                                                    batch 经 IPC 回主进程 → device_put → GPU jit embed_memory
                                                       ▼
                                                    每样本 512 个 memory token
```

### 1.2 阶段一与阶段二：数据集预处理（生产侧，本轮不动）

```
 原始 H5（本机 /data/hongzefu 原件永久保留；turbo 暂存副本经 sha256 同源核对）
 ├── record_dataset_ButtonUnmask.h5      400 ep   无 demo 前缀（exec_start_idx = 0）
 ├── record_dataset_ButtonUnmaskSwap.h5  400 ep   无 demo 前缀
 ├── record_dataset_VideoUnmask.h5       400 ep   demo 前缀恒 66 帧
 └── record_dataset_VideoUnmaskSwap.h5   400 ep   demo 前缀 114–216 帧（实测取值 {114,168,216}）
       内部：episode_{i}/timestep_{t}/obs/front_rgb (256,256,3) u8、wrist_rgb、
             joint_state (7,) f4、gripper_state (2,) f4、action/joint_action (8,) f8 …
       ⚠ 无 chunk 无压缩，全是逐 timestep 散小数组
        │
        │ scan_manifest.py build：规范序 sorted(*.h5) × sorted(episode)，只读 metadata
        ▼
 v1-store/episode_manifest.json ——「唯一真值源」
       每 episode 记：(h5_file, raw_ep_idx) 身份、num_timesteps、exec_start_idx、exec_samples、
       三个前缀和偏移 global_episode_idx / exec_sample_offset / total_sample_offset，
       整体带完整性字段 sha256（被改动即 fail-loud；下游引用时命名为 manifest_sha256）
        │
        │ GL 8×1GPU job array（build_shard.py，LPT 装箱分 8 片，跨片指纹同源断言）
        │ 逐 timestep：front_rgb → resize_with_pad(256→224) → SigLIP So400m/14（bf16，
        │ 256 token）→ 池化出 8x8/4x4/2x2 三档 → PosEmb3D 按 step 切片 → 每帧存一个 npy
        ▼
 v1-store/datasets/4task-gl/（678 GB；1600 episodes；483,291 帧；395,289 执行样本）
 ├── meta/stats.json                    execution_samples=395289, total_samples=483291
 ├── meta/provenance.json、_shard{0..7}of8.json
 ├── features/episode_{g}/              g = 0..1599（四任务拉平编号，归属查清单）
 │     ├── token_emb_{t}.npy            每帧一个，602,951 B（真常量，30+ 抽样全等）——
 │     │      np.save 的 pickle dict：
 │     │      image_emb_8x8 (1,64,2048) bf16   256 KiB ┐
 │     │      image_emb_4x4 (1,16,2048) bf16    64 KiB │←┐
 │     │      image_emb_2x2 (1, 4,2048) bf16    16 KiB │  │ framesample 只用
 │     │      pos_emb_8x8   (1,64, 768) f32    192 KiB │  │ 这三个键，共 112 KiB
 │     │      pos_emb_4x4   (1,16, 768) f32     48 KiB │←┤（= 每帧字节的 19%）
 │     │      pos_emb_2x2   (1, 4, 768) f32     12 KiB │  │
 │     │      state_emb     (8,)        f32       32 B ┘←┘
 │     └── kept_indices.json            token_dropping 用，framesample 路径完全不读
 └── data/{0..395288}.pkl               每执行样本一个，约 395.4–395.6 KB（内嵌变长
        prompt/subgoal 字符串，非定长；下界 395,440 B）：
        image / wrist_image (256,256,3) u8 两张原图（共 393 KiB）、state (8,) f32、
        actions (20,8) f64、prompt/subgoal 字符串、epis_idx / step_idx / exec_start_idx
```

要点：

- **features 按「全部 timestep」存**（含 Video* 任务 demo 前缀帧），**data/pkl 只按「执行样本」存**（demo 前缀不出样本）。两套编号靠清单偏移字段换算，**必须带 `exec_start_idx`**（漏掉会让 Video* 任务样本错位 66–216 帧，公式见 B.3）。
- 每帧 npy 是 `np.save` 的 **object dict（pickle）**：7 键绑在一起，**无法部分读取**——要拿 4x4 那 112 KiB 必须整包反序列化 589 KiB。
- `pos_emb_4x4` 实测是 **step_idx 的纯函数**（跨 episode 逐字节相同），却按帧冗余存了 483,291 份。

### 1.3 训练时每个 step 的取数链（消费侧，现状）

```
┌─ 主进程（jax，驱动多 GPU）────────────────────────────────────────────────────┐
│ torch.Generator().manual_seed(seed) + shuffle + drop_last                      │
│   └─ 每 step 抽 batch 个样本 idx。同一迭代器生命周期内（单个 epoch 内）序列只由 │
│      (len, seed, batch, drop_last) 决定、与 num_workers 无关；跨 epoch 见 1.6   │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       │ idx 分派
       ▼
┌─ spawn worker × N（persistent_workers，prefetch_factor=torch 默认 2）──────────┐
│ RoboMMEDataset.__getitem__(idx)                                                │
│  ① pickle.load(data/{idx}.pkl)         395 KB，热 ~2.7 ms（其中 ~2.3 ms 是      │
│     NFS open 延迟）；取出 epis_idx=g、step_idx=t                                │
│  ② even_sampling_indices(t, 32)        纯确定选帧，零随机源：                    │
│       t < 32 → [0..t] 全取；否则 linspace(0, t, 32) 均匀 32 帧                  │
│       ⚠ 采样域含 demo 前缀帧；32 = budget(512) ÷ token_per_image(16)            │
│  ③ _gather_history_feat                ⚠ 每样本新建 ThreadPoolExecutor(≤32 线程) │
│       逐帧 np.load(token_emb_{f}.npy, allow_pickle).item() 全量反序列化          │
│       = ≤32 次 NFS open+close（实测 32 次 open 本身 74.3 ms）                    │
│       + ≤32 次 589 KiB 整包 pickle（每次 4.0–4.9 ms，只用 112 KiB）              │
│  ④ 拼装 (n,16,2048) bf16 / (n,16,768) f32 / (n,8) f32 / mask (n,)              │
│  ⑤ right_padding_token_emb             np.zeros 显式 dtype：短样本（t<31，       │
│       占 6.27%）padding 后每键 dtype 与满长样本一致（image bf16/pos f32/stt f32）│
│  ⑥ reshape → static_image_emb (512,2048) / static_pos_emb (512,768) /          │
│       static_state_emb (512,8)（⚠ use_state_emb=false，GPU 不用，白算白传）     │
│       / static_mask (512,)                                                     │
│  ⑦ transforms：Repack → RoboMMEInputs（两张原图解析）→ DeltaActions →           │
│       Normalize(quantile) → ResizeImages(224)（openpi_client 的 NumPy/PIL       │
│       CPU 实现，无 JAX）→ PaligemmaTokenizer(64) → PadStatesAndActions          │
│     ⚠ worker 内另有 JAX：dataset 模块级导入链（training/dataset.py →            │
│       mem_buffer.py → openpi.shared.image_tools，import 即 import jax）使每个    │
│       spawn worker 初始化 JAX（8 s）并在 GPU0 建 442 MiB CUDA context——         │
│       与 ResizeImages 无关                                                      │
│  ⑧ _collate_fn（np.stack）在 worker 内执行：batch 内 dtype 一致、零提升          │
│       （memory 三键 ~236 MB，image 单键 134 MB）                                 │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       │ 已合并的 batch 经 IPC 回主进程（batch 载荷 ~257 MB：三键 236+原图 19）
       ▼
┌─ 主进程 交付 ──────────────────────────────────────────────────────────────────┐
│ jax.make_array_from_process_local_data：bf16/f32 直付，host 侧无降精度搬运；     │
│   100% batch 同一 dtype 组合，XLA 编译产物 1 份                                  │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       ▼
┌─ GPU（ptrain_step，jit）───────────────────────────────────────────────────────┐
│ embed_memory → FeatureEncoder.encode_perceptual_memory（nnx.Linear 显式 bf16，  │
│ promote_dtype 把输入统一转 bf16）：                                              │
│   x = concat(static_image_emb, silu(pos_proj(static_pos_emb)))    (512, 2816)   │
│   memory_tokens = encoder_static(x)                               (512, 2048)   │
│ → 512 个 memory token 拼 prefix 最前：[512 mem | 256 img | 256 wrist | 64 txt]  │
│   唯一可训练的 memory 参数：pos_proj 768→768、encoder_static 2816→2048           │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 字节帐与耗时帐（实测；数字来源见五节「探针脚本固化」与 A.2 的 probe_layout.py）

| 口径 | 现状数值 | 备注 |
|---|---|---|
| 每样本读盘 | 均值 19.08 MB（上界 19.69 MB） | 其中真正用到 均值 3.95 MB（上界 4.07 MB），放大 ≈4.8×；单看 npy 是 589 KiB 只用 112 KiB（5.3×） |
| 每样本耗时（热/冷） | 25.4 / 132.4 ms | gather 占 17.7 / ~110 ms；32 次 open 本身 74.3 ms |
| 每 step（b64）读盘 | 1.22 GB | 需求 256 MB/s，NFS 供给 398–628 MB/s（带宽不是瓶颈） |
| 每 step 文件打开 | 2,112 次 | 64 pkl + 64×32 npy |
| collate / IPC / device_put | 19 ms / ~257 MB / 23 ms | collate 在 worker 内执行（num_workers>0 时）；257 MB 为 batch 载荷（memory 三键 236 MB + 两张原图 19 MB） |
| 步时（GL b64） | 中位 6.933 s（compute-only 下界 4.778 s，+45%） | GPU util 均值 69.7%、0% 采样 27.8%、慢步墙钟 32.9%；历史口径实测、标注口径不重测（用户拍板） |

> 注：全表统一十进制 MB（1 MB = 10⁶ B），与既有留档中的 MiB 数字不可直接比。

### 1.5 浪费在哪里（按影响排序）

1. **文件个数**：每样本 ≤33 次 NFS open——32 次 64 KiB 读若落在一个常开 fd 上只要 0.33 ms，open 却要 74.3 ms。
2. **整包 pickle 反序列化**：7 键绑死，读 5.3× 于所需字节。
3. **每样本新建 ≤32 线程的线程池**（用完即弃）。
4. **pos_emb 冗余**：纯函数按帧存盘反复读，占必需读量 38%。
5. **worker 里的 JAX**：每 worker 初始化 8 s、GPU0 上 442 MiB CUDA context（16 workers ≈ 7 GB 显存 + 上下文抢占）。来源是 dataset 模块级导入链，**不是 ResizeImages**。本轮不修，bench 采样存证，作为后续优化的立项输入。
6. **state_emb 白算白传**（use_state_emb=false）。

### 1.6 现状的确定性

给定 seed / batch_size / fsdp_devices 与同一份数据集：

- **单个 epoch 内**（同一迭代器生命周期内），每 step 的样本集合与 memory token 内容逐位可复现，num_workers 只影响交付时机不影响内容。
- **跨 epoch 边界与 num_workers 相关**（torch 既有语义，与本重构无关）：`_BaseDataLoaderIter.__init__` 每次构造迭代器都从同一 generator 抽一次 `_base_seed`，而 `persistent_workers`（w>0）跨 epoch 只 `_reset` 不重建、w0 每 epoch 重建——两条路径消耗 generator 的节奏不同，**同 seed 下 w0 与 w>0 从第 2 个 epoch 起排列分叉**（torch 2.7.1 读码 + 双人独立脚本复现确认）。恒等链只需「新旧链路在相同 num_workers 下序列相同」，不受影响。
- XLA 层：生产默认 autotune 下同配置重跑非 bitwise 确定；正确性对拍一律跑在确定性档（四节），该档下独立冷编译两次已实证逐位一致。

## 二、重构后链路（与现状逐项对比）

### 2.1 一句话

**把 483,291 个 602 KB 小 npy 压成 32 个连续大文件（只含 framesample 真正要的三张表，共 31.7 GB），训练时用常开 fd 直接 pread。** 预处理前两阶段（清单、SigLIP 建库）与产物**原样保留、一字不动**，只新增纯派生的「阶段三：打包」。32 个 part 中前 31 个 ≈990–1020 MB，末 1 个 ≈621 MB（贪心切分的尾部效应）。

### 2.2 预处理步骤对比：旧两阶段不动，新增阶段三

```
 阶段一 scan_manifest.py ──► episode_manifest.json          ┐
 阶段二 GL 8×1GPU SigLIP 建库 ──► 4task-gl（678 GB）         ├─ 与现状完全相同，不重跑
                                    │                        ┘
                                    │  阶段三（新增）：打包派生
                                    │  pack_framesamp_store.py（本机 tmux，
                                    │  纯 CPU+NFS，16 进程，40–80 min）
                                    │  · 逐 episode 读全部 token_emb_*.npy
                                    │  · 只抽 image_emb_4x4 / pos_emb_4x4 / state_emb
                                    │  · 写侧逐帧校验（pos memcmp 钉死 t；见 A.2）
                                    │  · 写后 pread 读回校验 + 逐 part sha256 原子落盘
                                    │  · verify 子命令独立后验（16 进程）做「全量
                                    │    483,291 帧写×读对拍」——g 级零遗漏唯一凭据
                                    ▼
 v1-store/datasets/4task-gl-framesamp/            共 31.7 GB（特征侧体积的 1/9）
 ├── meta/store_meta.json         唯一契约：布局/形状/dtype/字节序/part 边界/源库根
 │                                与清单路径/源库指纹/manifest_sha256（读侧逐项
 │                                fail-loud，packed 模式绝不回退散 npy）
 ├── meta/pack_progress.jsonl     断点续跑记录（父进程单写）
 ├── meta/row_digests.blake2b.bin 逐行独立摘要（483,291 × 16 B，verify 时产出，必产出）
 ├── image_emb_4x4/part_000.bf16.bin … part_031.bf16.bin
 │       前 31 个 ≈990–1020 MB + 末 1 个 ≈621 MB，(rows,16,2048) bf16 裸字节；
 │       行号 = total_sample_offset[g] + t（写读共用同一函数）；
 │       按 episode 边界切分 → 一个样本的 32 帧必落在同一 part
 ├── pos_emb_4x4.f32.bin          (586,16,768) f32 = 28.8 MB，按 step_idx 查表
 └── state_emb.f32.bin            (483291,8) f32 = 15.5 MB，按全局行号查表
 （data/{idx}.pkl 与两张原图不动：训练时仍从源库 4task-gl/data/ 读——
   「打包库 + 源库 + 清单」三个位置经显式双根契约传递，见 B.4，禁止从目录名推导）
```

**预处理格式/步骤对比表**：

| 维度 | 现状 | 重构后 |
|---|---|---|
| 预处理阶段 | ① 清单 ② SigLIP 建库 | ①② 原样保留（产物不动）＋ ③ 打包派生（**新增**） |
| 是否重读 H5 / 重跑 SigLIP | — | 否。阶段三只读 4task-gl，纯 CPU，无 GPU、无集群 |
| 特征存储形态 | 每帧一个 npy 小文件（483,291 个；pickle dict 7 键，602,951 B，无法部分读取） | 32 个连续 `.bin` 大文件（只存所需 3 键；裸字节，dtype/字节序显式声明于 meta） |
| pos_emb 存法 | 每帧冗余存一份（38% 的必需读量） | 586 行小表一份（pos 是 step 的纯函数，实测证实） |
| state_emb 存法 | 混在每帧 npy 里 | 独立小表 |
| 原图/actions/prompt | data/{idx}.pkl | **不变**（仍读源库；位置由双根契约显式传递；若实测成为新瓶颈，Phase C 预留了同机制打包接口） |
| 磁盘体积 | 678 GB | 源库原样保留 ＋ 新增 31.7 GB |
| dtype 契约 | 隐式（pickle 内嵌） | 显式：meta 声明 shape/dtype/字节序（交付 dtype 与旧路径相同，无模式开关） |
| 完整性判据 | 建库期三层一致性验证（已完成） | 打包写侧逐帧校验 ＋ 写后读回 ＋ part sha256 ＋ **全量 verify 对拍（零遗漏）** ＋ 读侧 fail-loud |
| 身份来源 | episode_manifest.json | 同一清单；行号公式写读共用，禁止目录序 |

### 2.3 训练取数链对比

```
┌─ spawn worker × N ─────────────────────────────────────────────────────────────┐
│ FrameSampDataset.__getitem__(idx)      单一路径、无分支、只服务 framesamp+context │
│  ⓪ FrameSampStore 懒加载：Dataset 对象 pickle 进 worker 时不携带任何 fd/mmap，    │
│     首次 __getitem__ 按当前 pid 在 worker 内构造（fd/mmap 生命周期契约见 B.2）    │
│  ① 清单查表得 (g, t)                    O(1) 数组（含 exec_start_idx 换算），     │
│     不读目录                                                                     │
│  ② pickle.load(data/{idx}.pkl)          与旧路径同源同字节，~2.7 ms              │
│     ＋校验 pkl 内 epis_idx/step_idx == 清单推导值（不符显式 raise，行号错位闸）   │
│  ③ even_sampling_indices(t, 32)         同一个函数 import，选帧逐位不变          │
│  ④ gather：32 个常驻 fd 上 fadvise 预读 + 游程合并 preadv 直读进预分配数组        │
│     （短读循环补齐，EOF/越界才 raise）                                           │
│     0 次 open、0 线程池、0 pickle       热 0.3–0.5 ms / 冷 10–30 ms（待实测）    │
│     pos/state 从进程内常驻小表按行取（44 MB/worker，无 NFS 缺页）                │
│  ⑤ _pad：预分配 bf16/f32，填充区清零（dtype 行为与旧路径逐键一致）               │
│  ⑥⑦⑧ 拼装、transforms 与 worker 内 collate 与旧路径完全相同                     │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       ▼
  collate / device_put：与旧路径完全相同——batch 内 dtype 一致、bf16/f32 直付
       （memory 三键 ~236 MB/batch）
  GPU：同一段 jit 代码，输入张量与旧链路逐位相同 → memory token 逐位一致
       （有效性域见三节）；两侧 HLO 相同
```

**训练期每 step 对比表**（均值与上界分列，口径与 1.4 对齐）：

| 维度 | 现状 | 重构后 |
|---|---|---|
| 每样本文件打开 | ≤33 次（1 pkl + ≤32 npy） | 1 次（特征走常驻 fd） |
| 每样本读盘 | 均值 19.08 MB（上界 19.69 MB），只用 3.95 MB | 均值 2.43 MB（上界 2.49 MB），几乎全用到 |
| 反序列化 | ≤32 次全量 pickle | 0 次（裸字节直读） |
| 线程池 | 每样本新建 ≤32 线程 | 无 |
| padding dtype | 显式 bf16/f32 | 相同（本计划无 dtype 变更） |
| collate 后 batch 载荷（memory 三键＋两张原图） | ~257 MB/batch | 相同（~257 MB/batch） |
| 每 step 读盘 / 打开（b64） | 1.22 GB / 2,112 次 | 均值 155 MB（上界 159.5 MB）/ 64 次 |
| 单样本耗时（热/冷） | 25.4 / 132 ms | ≈7 / 15–40 ms（预估，S8 实测为准） |
| 供给余量（vs 计算需求） | 不足（GPU 空转 30%） | ≈15× |
| 预期步时 / epoch（GL b64） | 6.933 s / 11.9 h | ≈4.9–5.0 s / 8.4–8.6 h（下界 4.778 s / 8.2 h） |

### 2.4 接口与开关（给人看的版本；细节在 B.2/B.4）

- **backend 显式三态**：`MMEVLA_DATA_BACKEND ∈ {packed, legacy, auto}`。`packed`＝新链路，meta 缺失/损坏/指纹不符**直接报错，绝不静默回退**；`legacy`＝旧链路逐字不动；`auto`＝按 meta 存在性分派并打 WARNING，**仅限本机探索且必须显式设置才生效**。**环境变量未设置时默认 `legacy`——与现状行为逐字节相同，零静默切换。**正式 launcher 一律显式 `packed` 或 `legacy`——「按目录内容猜路径」被彻底禁止。
- **双根契约**：打包库、源库（pkl 所在）、清单三个位置全部显式：`store_meta.json` 记录 `source_dataset_root` 与 `manifest_path`（绝对路径），可被环境变量覆盖（节点本地盘缓存场景：store 在 `/tmp`，pkl 与清单仍在 NFS）。
- **store 生命周期**：常驻 fd 与两张小表**不跨进程携带**——Dataset 被 pickle 进 spawn worker 时剔除句柄，worker 内首次取数时按 pid 懒构造（小表 `np.fromfile` 全量读入进程内存，44 MB/worker）。这是「0 次 open」承诺的实现路径。

## 三、三块验证（「memory token 近乎一致」的实施结构）

memory token 由四个因素完全决定，重构后全部构造性不变：① 每 step 取哪些样本（不换 `TorchDataLoader`/generator/seed 语义、`len` 相同 395,289 → 同 workers 档位下序列逐位不变）；② 每样本选哪 32 帧（`even_sampling_indices` 纯函数、import 同一函数不重写、`step_idx` 与清单互校）；③ 每帧特征的字节（写侧逐帧校验 + 全量 verify 零遗漏钉死）；④ 交付 dtype（两侧相同，无模式开关）。**论证再严密也可能被没想到的环节推翻，所以配三块可证伪的验证，每块有硬判据**：

### 第一块：非训练轻量对拍（不启动训练，证明新旧链路交付内容逐位一致）

对应 C.1/C.2（实施步 S5）。两侧交付 dtype 相同，判据是**不折算的直接逐位零容差**：

1. **index 序列对拍**（C.1）：legacy 与 packed 两侧 Dataset 用同一 `TorchDataLoader` 同 seed dump 序列，w0/w4/w8 三档 diff 为空（dump 步数 < 1 个 epoch，防跨 epoch 既有分叉制造假阳性）；另有 `BENCH_DUMP_IDX` batch_sampler 层的真实链路旁证。
2. **样本/batch 内容对拍**（C.2）：约 8,200 个定点样本（step 边界全覆盖 + 每 episode 首样本 + 随机）在 **transform 之后**逐样本对拍全键，另加 200 个真实 batch 过 `_collate_fn` 对拍；判据全键 shape/dtype/`view(uintN)` 逐位零容差，判定行 `COMPARE_BATCH=PASS`。
3. 前置资产：守卫测试两组（C.5）与全量打包 verify（483,291 帧零遗漏，A.2）。

**本块不过，不开第二块。**

### 第二块：本机真实训练对拍——G2（正确性）+ G2-speed（本机性能对账）

- **G2（C.3，实施步 S6，本计划终局等价性检验）**：`MMEVLA_DATA_BACKEND=packed` 从 clean HEAD 起跑**一轮 1000 步**（本机 2 卡 b8 seed 42，确定性档，摘要步集与 G0 完全对齐：`SAVE_INTERVAL=100` + 附加摘要步 299），**离线对拍 G0 固化产物**（A 侧不重跑）。判据：
  - 主判据 bitwise：逐步五标量（loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm）hex 列 + 12 次摘要步 `state_digest` 与 G0 逐位相同；
  - 输入侧：**canonical 摘要**（逐键升 f32 后按位视图哈希）+ 全步 index 序列与 G0 逐位一致。**raw 输入摘要不计入判据**——G0 固化后交付 dtype 经过一次已验收的统一，G0 产物的 raw 摘要在 4 个摘要步的 `static_image_emb`/`static_pos_emb` 两键上与现行 HEAD 存在已知失配（canonical 同步一致），属预期差异、不再展开。
  - G2 起跑前必过 `BASELINE_ENV=PASS` preflight（vs G0 r1，四节）；EXP_NAME 独立（不与任何历史 run 共用编译缓存——确定性档本就关 autotune，共用无收益）；单 epoch 约束 1000×8=8,000 < 395,289 满足。
  - **失败处置**：分叉叶子在 `mem_enc*` → 回第一块查交付内容；在 LLM 主干而 mem 一致 → 疑非确定性，先跑 packed 自身重跑自证（`-r2`）；仍无法归因时**现场加跑一轮同 HEAD legacy** 与 packed 三方定位（legacy 轮只是定位手段，不是判据的一部分）。量化兜底（四节）**仅在证明分歧源于编译期非确定且 packed 自身重跑稳定后**可作评估参考，但本对拍无 dtype 差异，任何数值残差都指向 bug 或非确定性，**必须定位修复后重跑至 bitwise 通过，绝不「差不多就行」**。
- **G2-speed（S9）**：`v1-g2-speed` 一轮，speed 统一口径（〇节，1000 步），vs 锚点 `v1-g0-speed-r2`——「重构前基线 → 重构后」的本机合并对账，**不设阈值、只报数、不做单项归因**。

**覆盖范围声明**：1000 步 × b8 = 8,000 样本只覆盖粗差（行号错位、选帧错误会在头几十步撞穿）；「万分之一错帧」类细差的覆盖责任在第一块定点对拍与全量 verify。本块 bitwise 结论成立于本机受控确定性环境；「正式平台吞吐达标」归第三块。

### 第三块：GL 吞吐验收——north star「GPU 吃满」判据（D 节，实施步 S8a/S8b）

正式训练平台是 GL 4×A40，瓶颈基线与验收阈值全是 GL 实测；AGENTS 13 明文本机吞吐不作最终结论——**吞吐过/不过在本块判定**：

- **S8a**：GL dataloader-only 四档（w2/w4/w8/w16），fast 校验档 + 冷态自证 provenance。
- **S8b（全链收官测试）**：GL e2e 600 步 T1–T3（+条件档）+ cold-like/hot 双跑；主判据表 5 项机器判定 `E2E_ACCEPT=PASS|FAIL`（必达：步时中位 ≤5.00 s、util 稳态均值 ≥90%、0% 采样 ≤5%、慢步墙钟 ≤5%、epoch ≤8.6 h，全表见 D 节），并附「距 100% 的残差分解」。
- 秩序：**全程严格串行（用户 2026-08-27 拍板，E 节）**——全部 GL 验收在第二块 G2 bitwise 通过之后，按 S7.5 → S8a → S8b 顺序执行，不提前排队、不与本机验证并行（也避免 GL 侧与本机验证竞争读取同一份 NFS 数据）；每个超 GL 硬限的 job（4×A40 / 2–4 h）提交前逐个向用户做资源审批并在 `greatlakes.md` 留记录，run_name 确认不能替代资源审批。
- GL 侧不重复 bitwise 证明（与本机是两种硬件、GL 无稳定确定性基线）；GL 侧异常按量化判据思想兜底、量级由本块吞吐验收覆盖。

### 放行规则与回滚

1. 第一块 + 第二块 G2 bitwise 全过 → 允许宣称「IO 重构不改变训练语义」，回填登记簿（T8）；
2. G2-speed 与第三块产出性能结论（本机对账 + GL 主判据）；
3. 任何一块失败按上文处置定位修复重跑，无「差不多就行」路径；
4. 功能回滚 = launcher 里 `MMEVLA_DATA_BACKEND` 切回 `legacy` + `--dataset-path` 指回源库（两件事必须一起回退）；打包库保留作证据不删（不进 git，31.7 GB），只有确认彻底放弃方案时才删库目录。

## 四、G0 是什么与锁定方式

- **G0**：重构前训练语义在受控确定性档下的一轮真实训练（run `v1-grad-baseline-g0b-r{1,2}`）。口径：本机 2×RTX 6000 Ada、b8、1000 步、seed 42、`SAVE_INTERVAL=100` + 附加摘要步 299（步 0 与末步必记，共 12 次完整 TrainState 摘要）、`bench_train_steps.py` 入口。两轮千步逐位自证 PASS，产物固化进 git。**其 util/步时不作任何性能结论**（摘要停顿一次约 47–140 s + 确定性档污染性能口径）；性能基线由 `v1-g0-speed-r2` 承担。
- **「跑一次就不再重跑」的三前提（均已达成，引用时逐次核验第 3 条）**：
  1. **仪器完整**：逐步五标量 hex、完整 TrainState 摘要（params/opt_state/EMA/step 全叶子 sha256 + `state_digest`）、batch 输入摘要 raw+canonical 双口径、全步 index 序列、真实 argv 与编译缓存命中计数进 env.json；
  2. **确定性成立**：确定性档 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` 下，同配置重跑（共用编译缓存）与**独立冷编译两次**均实证逐位一致（后者是跨期充当 bitwise 判据一侧的唯一授权闸——未来对拍 run 计算图变了必然现场重编译）；生产默认档重跑噪声底已实测固化（loss rel median 2.7e-3 / max 4.6e-2），作量化判据 null 上界；
  3. **环境指纹不变**：每次引用 G0 产物前必过 `check_baseline_env.py`（`BASELINE_ENV=PASS|FAIL`，断言清单见 T5），任一失效条件触发即基线作废、必须重跑并在登记簿记版本。
- **git 锁定**：锚点 commit `55e6e5bf8ef38b780902d0e63257ea859a432a2c`；`<G0-HEAD>` = `570287f`，锚点到起跑 commit 的全部 diff 已过 G0_SCOPE 反向白名单断言（T1）。git 外指纹（sha256 单列）：`norm_stats.json`、`pi05_base` 初始权重、paligemma tokenizer、`episode_manifest.json` 顶层 `sha256`、数据集抽样指纹 `source_spot_sha256`。
- **TrainState 数组参照**（量化裁决用；固化产物只有 sha256、补算不出数值）：G0 r1 摘要步 @0/299/999 的完整 TrainState 数组存本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（sha 清单进 git）；失配步不在所存步时按独立冷编译授权重放补落。跨 commit 引用 G0 初始状态机器可证：同 seed/同 config 现场 `init_train_state` 的 177 个叶子摘要与 G0 r1 步 0 逐个相同（已实证），无需加载 45.4 GiB 的 `state_step_0.bin`。
- **量化判据（等价性检验形态；只作兜底评估与定位参考，G2 场景不作放行依据——见三节第二块）**：
  - `rel(a,b) = |a−b| / max(|a|,|b|,1e-8)` 逐步计算，五标量各统计 median / p95 / max；
  - null 对（噪声底）与被比场景同构优先：跨编译场景 → 确定性档独立冷编译两轮（实测逐位为零，退化为下限守卫）；不可得时 → 生产档两轮（上界，数字见上）；
  - 判据：A/B 的 rel 各统计档 ≤ null 对相应档 × 2；下限守卫：null 低于绝对下限（loss 1e-6 / 三个梯度范数 1e-5 / 末步 param_norm 1e-5）时以绝对下限为准；趋势主用包络（rel(t) 逐步 ≤ null 上包络 × 2），`log(rel)` 斜率仅作定位参考；
  - **TrainState 数值裁决**：`state_digest` 失配时仅凭五标量统计不足以判 PASS——必须输出逐叶数值统计（max-abs / max-rel / L2 / cosine，params/opt_state/EMA 全叶子，数值参照取上述本机数组）；拿不出即判 `INCONCLUSIVE`，不得写 PASS；
  - 判定行：`QUANT_EQUIV=PASS|FAIL scalars=5 null=<pair> margin=2.0`。

## 五、明确不做的事，与本重构解耦的既有问题

### 5.1 本轮明确不做

- **不打包 pkl**（每 step 仅 25.3 MB、2.7 ms/样本，打包无体积收益；按同一套 part 机制预留 Phase C 接口，实测成为新墙再单独一轮）。
- **不预烘焙 ResizeImages、不预 tokenize prompt**（tokenize 实测 24 µs/次；烘焙需 119 GB 存储；`transforms.ResizeImages` 实为 NumPy/PIL CPU 实现、无 JAX）。worker JAX 导入链（每 worker 442 MiB CUDA context）本轮只采样存证，作为后续优化的立项输入。
- **不重建数据集、不动源库、不动旧多分支 Dataset 代码**（symbolic/recurrent/token_drop 原地保留不惊动）。
- **不给 `TorchDataLoader` 加 `prefetch_factor` 形参**（与「不动 `src/openpi/**`」红线矛盾，列为未来备选）。
- **探针脚本固化**：历次标「已实测」的数字（npy 内部偏移、pos 纯函数等）原是一次性交互探针，S2 固化为 `scripts/data-pack-framesamp/probe_layout.py` 等小脚本，数字在构建留档附录记录命令与原始输出。

### 5.2 解耦的既有问题（本轮不修，如实声明，处置须用户单独拍板）

1. **`scripts/train.py` 正式入口双次 `main()`**：尾部先 `main(tentative_run=True)` 再正式 `main()`；`initialize_checkpoint_dir` 默认 `overwrite=False, resume=False` 且目录已存在即 `FileExistsError`——全新 run_name 下第二次 `main()` 必然报错（tentative 已建目录），除非显式传 `--overwrite`/`--resume`。全部验证与验收走 `bench_train_steps.py` 入口（单次 main）不触发。「用 `scripts/train.py` 起正式长训练」在该问题修复前不纳入本轮交付声明。
2. **checkpoint 只保存 `assets`/`params`**（`train_state` handler 被注释）：中断恢复不保证 optimizer/EMA/step 连续，不在本轮任何判据之内。
3. **`jax.process_count() > 1` 明确不支持**（`TorchDataLoader` 直接 raise）：本计划全部内容仅覆盖单进程多 GPU。

## 六、实施顺序一览（说人话版）

> 本节是 E 节的大白话版，只讲「先做什么、后做什么、每步过关标准是什么」；参数、commit 切分等细节以 E 节为准。**全程严格串行（用户 2026-08-27 拍板）：前一步判定过关，才开下一步；没有任何并行或提前排队的安排。**

**阶段 1：把代码和工具写好（本机开发，约半天）**

1. **S0' 补验证小工具**：让 bench 驱动认得 packed 库、能记录 index 序列。过关：5 步烟测跑通、idx 序列落盘（原「3 步」判据 2026-08-27 实施时发现结构性不可行——稳态统计剔除摘要步及其邻步后 3 步的样本集必为空，经用户拍板改 5 步）。
2. **S2 写格式层和打包工具**：先造一个小的「迷你库」，把打包 → 读取全流程走一遍。过关：Store 组守卫测试全绿。
3. **S3 写新 Dataset 和切换开关**：迷你库上起真实多进程 loader 跑两个 epoch。过关：Dataset 组守卫全绿、不崩、不漏文件句柄。

**阶段 2：造正式数据（本机 tmux，一两个小时）**

4. **S4 全量打包 + 全量校验**：把 678 GB 源库里 framesample 真正用到的三张表抽出来，压成 31.7 GB 打包库；然后 483,291 帧逐帧和源库对拍。过关：`VERIFY_PACK=PASS`、零失配。

**阶段 3：证明「没改数」（本机，核心验证）**

5. **S5 第一块·不训练对拍**：8,200 个定点样本 + 200 个真实 batch，新旧两条链的交付内容逐位比较。过关：`COMPARE_BATCH=PASS`、零失配。
6. **S6 第二块·真跑训练对拍（终局检验）**：packed 链路真跑 1000 步训练，逐步和固化的 G0 黄金基线逐位对拍。过关：五个训练标量 + 12 次完整模型状态摘要全部逐位相同。**这一步过了，才允许说「重构不改变训练」。**

**阶段 4：证明「变快了」（S6 过关后才开始）**

7. **S7.5 GL 验收资产参数化**：把 GL 侧提交脚本和分析脚本改成可切 packed（默认值保持现状）。过关：默认值跑通。
8. **S8a GL dataloader 单测四档**：不训练，只测新链路在 GL 上的取数吞吐（默认 w2/w4/w8/w16）。**提交 job 前先和用户确定跑哪几个档位**（2026-08-27 用户指定），每个超硬限 job 提交前再逐个找用户审批资源。
9. **S8b GL e2e 收官测试**：GL 4×A40 真跑 600 步端到端 + 冷/热双跑。过关：`E2E_ACCEPT=PASS`（步时中位 ≤5.00 s、util 均值 ≥90% 等五项必达，D 节）。
10. **S9 本机速度对账**：本机跑 `v1-g2-speed`，和锚点 `v1-g0-speed-r2`（1.152 s/step）对比报数，回填登记簿（T8）。

**当前位置**：2026-08-27 用户已拍板开工，第一个动作是 S0'（commit V3.0）。

---

# 第二部分（技术细节，供 agent 追踪）

## A. 打包特征库

### A.1 目录布局与 store_meta 契约

目录布局见 2.2 图。关键规则：

- **行号公式**（写读两侧共用同一函数，物理上不可分叉）：`row(g,t) = manifest.episodes[g].total_sample_offset + t`，t 是**全 timestep 域**帧号（含 demo 前缀）。执行样本 idx → (g,t) 换算见 B.3（必须带 `exec_start_idx`）。
- **part 切法**：按 `global_episode_idx` 升序累积 `num_timesteps`，累计 ≥ `ceil(483291/32)=15103` 即切，**切点必在 episode 边界**（一个样本的 32 帧必在同一 part）。真实 manifest 模拟切分恰得 32 个 part；末 part 覆盖 episodes[1573..1599]、9,471 行、620.7 MB（读侧按 meta 精确边界走）。单文件方案与「每 episode 一文件」方案均已否决（前者与原子写+并行互斥，后者退化回每样本一次 open）。
- **bf16 落盘定论**（已实测）：裸 `.bin` + meta 声明 dtype。**禁止 `.npy`**——`np.save` 对 ml_dtypes bf16 写出 `V2` descr，`np.load` 丢类型。读侧 `np.memmap(dtype=ml_dtypes.bfloat16)` 与 `frombuffer(uint16).view(bfloat16)` 均实测可用。
- `store_meta.json` 关键字段：
  - 布局与格式：`layout="framesamp-4x4-v1"`、三张表 shape/dtype/row_bytes、`byte_order="little"`、`array_order="C"`、`bf16_encoding="ml_dtypes.bfloat16 (1s+8e+7m)"`、`writer_versions{python,numpy,ml_dtypes,git_commit}`；
  - 身份与双根：`manifest_sha256`（须等于当前 `episode_manifest.json` 顶层 `sha256`）、`manifest_path`、`source_dataset_root`（绝对路径，运行期可被环境变量覆盖，见 B.4）、`source_provenance_sha256`、`source_spot_sha256`（16 个抽样源文件摘要，读侧启动抽验）；
  - 规模与校验：`num_rows/num_exec_samples/num_pos_rows`、`parts[]`（每 part 精确行边界、sha256、`head_tail_digest`——blake2b-128 覆盖首尾各 1 MiB，part 小于 2 MiB 时覆盖全文件并标 `full_covered: true`）、`packer`、`verify`（全量对拍的 seed/时间/结论）；
  - 小表与 row_digests 摘要契约：`small_tables{pos_emb_4x4, state_emb}` 各记相对路径、shape、dtype、byte_count、sha256（pack 阶段随 `status:"packed"` 写入）；`row_digests` 记相对路径、`algo:"blake2b-128"`、`covered_rows:483291`、摘要覆盖范围（image‖pos‖state 三键原始位串拼接）、byte_count 与文件自身 sha256（verify 通过后随 `status:"verified"` 同批原子回填）；
  - **meta 两阶段写，均为 tmp + fsync + replace 原子落盘**：阶段 1 `pack` 结束写 meta（`status:"packed"`、`verify:null`）；阶段 2 `verify` 通过后原子回填（`status:"verified"`）。verify 期间继续持有 `pack.lock`，回填完成后才删锁。**verify 闸只落在 `create_data_loader` 的 packed 分派层**（`status != "verified"` 且未显式设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1` 即 raise，放行必打 WARNING，见 R17/G14）；`FrameSampStore` 本身不看 `verify` 字段（否则打包工具读不了自己正在验的库）。provenance 统一记回填后的 `store_meta.json` sha256。
  - **迷你库契约（守卫/开发专用）**：`manifest_scope ∈ {"full","subset"}`；`subset` 时必带 `subset_episodes[]` 与 `mini_manifest_sha256`。**`subset_episodes` 只允许是 `global_episode_idx` 的连续前缀 `[0..k]`**——由此全局行号即物理行号、pkl 编号与全量域一致、偏移原值可用：Dataset 长度＝前缀内 `exec_samples` 之和、合法 idx＝`[0, 该和)`、state 表偏移不变，五项全部沿用全量公式；非前缀子集 packer 检出即 raise。fast 档覆盖校验改为「`parts` 连续覆盖前缀声明的行区间」，`manifest_sha256` 仍记全量值但只作溯源。**`manifest_scope=="subset"` 的库禁止用于 S5 及以上任何判据**（packed 分派检出即 raise）。**迷你库规格硬约束**：前缀内必须含至少一个 `num_timesteps ≥ 33` 的 episode（保证 step=30 短样本与 step=31 满长样本都存在，G2 边界守卫才可构造）。

### A.2 打包工具

新目录 `scripts/data-pack-framesamp/`：`pack_framesamp_store.py`（子命令 `plan | pack | verify | report`）+ `run_pack.sh`（tmux 驱动，PYTHONUNBUFFERED=1 + pipefail + tee + EXIT_CODE=）+ `probe_layout.py`（探针固化）+ `README.md`。

- **真值与复用**：`scripts/data-preprocess-GL/` 目录名含连字符、不是合法包名，跨目录 import 不成立。清单读取（`load_manifest`，sha256 fail-loud）在 `src/mme_vla_suite/datastore/manifest.py` 实现（schema 与 `scan_manifest.py` 一致，后者原样保留服务建库侧），格式常量与 `row_of()` 在 `datastore/framesamp_store.py`——打包工具、`FrameSampDataset`、对拍工具一律从 `mme_vla_suite.datastore` 包 import，**绝不复制**；训练包不反向依赖 scripts。
- **运行位置**：本机 detached tmux（纯 CPU+NFS，不占 GL GPU 配额；产物是确定性字节、由全量对拍背书，不属「本机吞吐结论」，不违反 AGENTS 13）。**源读取账（如实按双趟计）**：state 表阶段完整 decode 全部 483,291 个源文件（≈291 GB）；image part 并行阶段第二次完整 decode（又 ≈291 GB）——pack 合计 ≈582 GB 源读取（verify 是第三趟另计），预计 **40–80 min**（16 进程）。
- **并行与事务协议**：
  1. **排他锁**：启动时 `O_CREAT|O_EXCL` 建 `meta/pack.lock`（记 build_uuid/host/pid/开始时间）。锁已存在：同 host 且 pid 存活 → 拒跑；同 host 且 pid 不存活 → 判残锁，`--resume` 显式确认后接管换新 build_uuid；**异 host 一律拒跑**（跨 host 无法判活），破异 host 锁只能走 `--force-break-lock`（打印锁全文 + 交互确认）。**读侧闸**：packed 分派在 `status` 闸外增加「`meta/pack.lock` 存在即 raise」（闸放分派层而非 Store——verify 子命令持锁期间仍需经 Store 读库）。锁在全量 verify 回填 meta 后才释放。
  2. **小表先行、主进程独写**：两张小表由主进程在并行阶段之前单独构建、校验、原子落盘；worker 只读它们做逐帧比对——「谁写全局表」归属唯一。
  3. **image part 并行**：`multiprocessing.Pool(min(16, cpu))`，每 part 唯一属主，天然无锁。
  4. **progress 单写**：worker 完成一个 part 经队列汇报 `(idx, rows, sha256, elapsed)`，只有父进程追加 `pack_progress.jsonl`。读侧对尾部半行直接丢弃；写侧 `--resume` 在追加前先 seek 到最后一个换行符并 `ftruncate` 掉尾部半行（否则续跑 append 会把半行拼成文件中部畸形行）。
  5. **写序与持久化**：episode slab 写 `part_XXX.bf16.bin.tmp` → `os.fsync` → 全 part 完成后 sha256 → `os.replace` → 目录 fd `fsync`（NFS 重命名可见性）→ 汇报父进程记 progress。meta 最后写（同 tmp+fsync+replace）。
  6. **崩溃语义**：SIGKILL/断电/ENOSPC 后 `--resume`：按 progress 校验「存在+大小+sha256」跳过完好 part；`.tmp` 残留一律清除重做；写入前 `df` 预检余量 ≥ 40 GB。
- **源读取两档** `--reader`：`decode`（首跑默认，逐帧 `np.load(allow_pickle).item()` 全量 602,951 B，零布局假设）；`slice`（已实测 npy 内部偏移恒定：`image_emb_4x4`@262,595、`pos_emb_4x4`@541,352、`state_emb`@602,906，120/120 文件大小一致、60/60 memcmp 通过、独立复现 9/9；三重守卫：st_size==602,951、数据段前 64 B 前缀逐字节相同、逐帧 pos 窗口 100% memcmp——留作重跑加速档）。偏移常数是数据格式常量，由 `probe_layout.py` 可随时复核。
- **写侧逐帧校验（100% 覆盖，写入路径内；钉 t 不钉 g）**：① 该帧 `pos_emb_4x4` ≟ `pos_table[t]`（memcmp，钉死 t 与「pos 只依赖 t」；**不钉 g**——pos 是 t 的纯函数，数学上分不出「同 t 不同 episode」的调包）；② `state_emb` ≟ state 表同一行（同源自证，防行内错乱）；③ slab 写 `.tmp` 后 `os.pread` 读回 memcmp（read-after-write，多读 31.7 GB ≈ 80 s）；④ part sha256 → `os.replace` → progress。
- **g 级身份的唯一凭据：`verify` 子命令全量对拍**：独立于 pack 的后验遍历，覆盖**全部 483,291 个 (g,t)**——重新完整 decode 源 npy，三键各经**真实读 API** 对拍：image 经 `FrameSampStore.read_image_rows()`、pos 经 `pos_rows(t)`、state 经 `state_rows(row)`，逐行 memcmp；同时逐行产出 blake2b-128 摘要（覆盖 image‖pos‖state 三键原始位串拼接）。进程模型：与 pack 同一套 Pool、按 part 划分；worker 只返回（起始行号、摘要字节块、mismatch 列表），父进程按 part 序拼接后单写 `meta/row_digests.blake2b.bin`（同 tmp→fsync→replace）。总读 ≈291 GB（源）+ 31.7 GB（store），16 进程预计 20–40 min。判定行 `VERIFY_PACK=PASS scanned=483291 mismatches=0`。**「零遗漏」「逐位」只在全量 verify 通过后才允许宣称**；`--sample N` 抽样档仅供开发期快检（10% 抽样对单行错位漏检率约 90%，不得用于交付判定）。
- **pos 表来源（定论）**：主方案**从源库抽取拼装**（用若干 episode 凑齐 t=0..585，逐位同源、零后端风险），主 pass 的 100% memcmp 即证明「只依赖 t」。`PosEmb3D` 现生成仅作旁证：**已实测 CPU 后端生成与库中值不逐位一致（max|diff| ≈ 7e-7），GPU 后端一致**——走生成路径必须校验 `jax.devices()[0].platform == "gpu"`（不符 raise），守卫 G7 钉死。
- **留档**：`docs/dataset-build-doc/4task-gl-framesamp/README.md`（AGENTS 12：commit、命令、源库指纹、耗时、写侧校验与全量 verify 结果、探针输出附录）。

### A.3 pkl 侧：本轮不打包（定论）

量级不构成瓶颈（25.3 MB/step、2.7 ms/样本）；打包无体积收益（定宽后仍 156 GB）；源库 `data/` 依红线必须保留可读。**Phase C 预案**（不实施，仅预留）：同 part 机制加 `samples/part_XXX.bin` 定宽记录 + `strings.json` 字典编码（实测 prompt 全集封闭，400 样本仅 27 个不同值）；`actions` 保 f64、`state` 保 f32，否则 DeltaActions/Normalize 数值会变。

## B. 新 Dataset / 装配路径

### B.0 全局逐文件变更表（范围一表定死，超出即越界）

| 类别 | 文件 | 变更 |
|---|---|---|
| 新增·格式层 | `src/mme_vla_suite/datastore/__init__.py`、`datastore/framesamp_store.py`、`datastore/manifest.py`、`datastore/README.md` | 常量 + StoreMeta + FrameSampStore(只读) + row_of() + load_manifest（sha256 fail-loud）；不 import 任何 training/model 模块 |
| 新增·装配层 | `src/mme_vla_suite/training/framesamp_dataset.py` | FrameSampDataset |
| 新增·工具 | `scripts/data-pack-framesamp/{pack_framesamp_store.py, run_pack.sh, probe_layout.py, dump_index_seq.py, compare_batches.py, test_pack_guards.py, README.md}` | 打包/探针/对拍/守卫 |
| 修改·接线 | `src/mme_vla_suite/training/dataloader.py` | `create_data_loader` 内 backend 分派（约 15 行） |
| 修改·验证资产 | `scripts/smoke-local/run_2gpu_epoch_bench.sh` | S0'：preflight 兼容 packed 库（`stats.json` **或** `store_meta.json`）、env.json provenance 字段扩展（清单与 D 节一致）。既有能力（EXP_NAME/RUN_TAG、KEEP_JAX_CACHE、缓存软链、XLA_FLAGS 注入）已具备，不再改 |
| 修改·验证资产 | `scripts/smoke-local/bench_train_steps.py` | S0'：`BENCH_DUMP_IDX` batch_sampler 层记录（细节见 C.1）。既有 checksum recorder（五标量 hex、TrainState 摘要、batch 摘要 raw+canonical、index 序列）已具备，不再改 |
| 修改·验证资产 | `scripts/smoke-local/README.md` | 同步 S0' 用法 |
| 修改·验收资产 | `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | S7.5：`--dataset-path`/backend/`--save-interval`（默认 1000＝现状）/`--time` 参数化（默认值=现状）、新增 `COLDHOT=1` 双跑模式（同 allocation 先 C1 后 H1，`--time=04:00:00`）、meminfo 采样、provenance 扩展 |
| 修改·验收资产 | `scripts/bottleneck-bench/gl-dataloader/{gl_dlbench_single,gl_dataloader_bench}.sbatch` | S7.5：`DATASET_PATH`/`MMEVLA_DATA_BACKEND` 参数化（默认值＝现状 `4task-gl`+`legacy`），env.json 同步；`submit_split_jobs.sh` 的 `--export` 补项 |
| 修改·验收资产 | `scripts/bottleneck-bench/gl-dataloader/dataloader_bench.py` | S7.5：`_AVG_BYTES_PER_SAMPLE` 从 history_config+manifest 推导、`block_until_ready` 覆盖整个 (obs, actions) pytree、gather/pkl 分段计时 |
| 修改·验收资产 | `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | S7.5：主判据表 5 项机器判定输出 `E2E_ACCEPT=PASS|FAIL`（FAIL 非零退出）；每步读盘公式去硬编码 1.20 GB，从 history_config+manifest 现场推导；另加吃多个 record_dir 的附加判据汇总入口 |
| 新增·文档 | `docs/dataset-build-doc/4task-gl-framesamp/README.md`、`docs/v2-dataloader-restructure-report.md`、`docs/v1-framesamp-dataflow.md` | 留档与汇总 |
| **不动** | `scripts/train.py`、`src/openpi/**`、`src/mme_vla_suite/models/**`、`training/dataset.py`、`shared/**` | 硬红线（G 节 R2）；`prefetch_factor` 可选项已裁定本轮不实施 |

### B.1 依赖方向

格式层（datastore）不 import 任何 training/model 模块（单向依赖）。修改类改动全部是「默认行为逐字节不变」的加性参数化（bench/sbatch 默认值即现状值），验证/验收资产改动不影响训练语义。

### B.2 `FrameSampStore`（格式层）——含 spawn 生命周期契约

- **构造与 pickle 契约（定论：懒加载）**：
  - `FrameSampDataset.__init__`（主进程）只做：读 `store_meta.json` + 全部 fail-loud 静态校验（fast 档）+ 清单派生查表数组；**不打开任何 part fd、不建任何 mmap**。
  - `__getstate__` 剔除 `_store` 句柄字段（只序列化路径/meta/查表数组）——Dataset 被 pickle 进 spawn worker 时不携带任何内核资源。
  - **每进程首次 `__getitem__` 时懒构造** `FrameSampStore`：`os.open` 32 个 part fd（`O_RDONLY|O_CLOEXEC`）+ 两张小表 `np.fromfile` 全量读入，记 `_owner_pid = os.getpid()`；此后每次取数校验 pid，不符即丢弃重建。**关闭责任**：① pid 失配重建时旧 store 必须先显式 `close()` 再替换引用（32 个 fd 属内核资源，不允许靠 GC 兜底）；② `FrameSampDataset` 实现 `close()` 转发并注册 `atexit` 兜底；③ w0 场景重复创建 loader 时由 Dataset 复用同一 store（同 pid 不重建）。S3 的 fd 计数检查是对本契约的验收。w0 路径同样适用（主进程即 owner）。
- **校验档位** `MMEVLA_FRAMESAMP_VERIFY ∈ {fast, full}`：
  - `fast`（默认，每进程懒构造时执行）：layout / `manifest_sha256` 现场重算比对 / parts 连续覆盖 [0,num_rows)（subset 库按声明行区间）/ 每 part 存在且 `st_size == meta.bytes` / 抽 1 个 part 头尾 1 MiB 与 `head_tail_digest` 复验 / 抽 1 条 `source_spot_sha256` 复验源库未动。
  - `full`（**禁止在性能 allocation 内执行**）：全部 32 个 part + 两张小表完整 sha256 对 meta 比对（≈31.7 GB 读，~1–2 min），能抓「同尺寸中部翻转」。但这一读会把整个 store 预热进 page cache——性能 run 里跑会使 S8a 不再测真实 NFS 供给、C1 不再 cold-like。因此 full 只允许在独立 preflight job（或 S4 verify 同场次）执行、结论回填 `store_meta.verify`；**一切性能 run（S8a、S8b 含 C1/H1）一律 fast 档**，env.json 必记 `MMEVLA_FRAMESAMP_VERIFY=fast` 并以 provenance 硬字段声明「本 allocation 未执行 full 校验、未做本地复制预热」（D 节）。worker 内任何档位都只跑 fast + `fstat` 尺寸复核。
- **大表 pread、小表进程内常驻**（pos 28.8 MB + state 15.5 MB = 44.3 MB/worker，16 worker ≈ 700 MB；实测同一 NFS 上 32×64 KiB 常开 fd pread 0.33 ms、`np.memmap` 切片 2.4–2.5 ms 走缺页/revalidate——热路径不留任何 NFS mmap）：`read_image_rows(rows, out)` 先对全部行发 `posix_fadvise(POSIX_FADV_WILLNEED)` 触发并发预读（`ENOSYS`/`EOPNOTSUPP` 打一次 WARNING 后永久跳过），再按连续行游程合并 `os.preadv` 直读进预分配 bf16 数组（短样本 32 行天然连续 → 1 次调用）。
- **短读处理**：`preadv` 返回不足**不立即判损坏**——从已读偏移续读，连续 3 次零进展才 raise；**读到 0 字节（EOF）或请求区间越出 part 边界立即 raise**（这才是完整性判据）。本仓库 NFS4.2 `hard` 挂载实测 320 次 2 MB 单调用零短读，该循环是稳健性兜底。
- **fail-open 禁令**：packed 模式下任何校验不过直接 raise，绝不回退散 npy。

### B.3 `FrameSampDataset.__getitem__`（伪代码）

```python
def __init__(...):   # 形制断言即文档（必须能挡住同形的 modul 配置）：
                     #   representation_type=="perceptual"、perceptual_memory.type=="frame_sampling"、
                     #   integration_type=="context"、memory_token_dim==2048、
                     #   (budget,token_per_image,num_views)==(512,16,1)、
                     #   memory_feature.img.input_dim==2048、pos.input_dim==768、
                     #   use_state_emb is False —— 全部不符即 raise（不用 assert，见下）
                     # load_manifest fail-loud；清单派生 O(1) 查表（int32/int64 数组，禁止 os.listdir）：
                     #   for g, ep in enumerate(manifest.episodes):
                     #     for k in range(ep.exec_samples):
                     #       idx = ep.exec_sample_offset + k
                     #       _epis_of[idx] = g
                     #       _step_of[idx] = ep.exec_start_idx + k   # ⚠ 必须带 exec_start_idx：
                     #                                              # Video* 任务漏掉即错 66–216 帧
                     #   _row_base[g] = ep.total_sample_offset
                     # store 懒加载契约见 B.2（__init__ 不建 store）
def __getitem__(self, idx):
    store = self._ensure_store()                                  # 每进程懒构造 + _owner_pid 校验
    g, step = self._epis_of[idx], self._step_of[idx]
    data = pickle.load(open(f"{source_root}/data/{idx}.pkl"))     # source_root 来自双根契约（B.4）
    if data["epis_idx"].item() != g or data["step_idx"].item() != step:
        raise RuntimeError(...)     # 行号错位的最后一道闸；显式 raise，
                                    # 禁用 assert（PYTHONOPTIMIZE=1 下会被剥离）
    data["actions"] = data["actions"][:action_horizon]; data.pop(两个 *_online 键)   # 与旧路径逐字相同
    frames = even_sampling_indices(step, 32)                      # 复用同一函数，不重写
    rows = self._row_base[g] + np.asarray(frames, np.int64)
    img  = store.read_image_rows(rows)                            # (n,16,2048) bf16——0 open、0 线程池、0 pickle
    pos  = store.pos_rows(frames); stt = store.state_rows(rows)
    n = len(frames)                                               # 实际帧数（_pad 第四参语义 = 实际帧数，
                                                                  # 目标长度 32 是 _pad 内部常量）
    img, pos, stt, mask = self._pad(img, pos, stt, n)             # dtype 行为与旧路径逐键一致
    data["static_image_emb"] = img.reshape(-1,2048)               # (512,2048)
    data["static_pos_emb"]   = pos.reshape(-1,768)
    data["static_state_emb"] = self._normalize_state(np.repeat(stt,16,axis=0))  # 保留精确计算（f64，与现状同源同式）
    data["static_mask"]      = np.repeat(mask,16)
    for k in _NONE_KEYS: data.setdefault(k, None)                 # recur_* / subgoal 等下游会索引的空键
    return data
```

- **`_pad`（单一实现）**：按最终形状一次性 `np.empty` 分配（img bf16 / pos f32 / stt f32），填充区清零，全程零 concatenate——与旧路径 `right_padding_token_emb` 交付 dtype 逐键一致、数值逐位一致。
- **`_collate_fn` 一行不改**：逐样本 dtype 与旧路径一致，batch 内 dtype 一致无提升。
- **static_state_emb 定论：保留精确计算不置零**——置 None 改变 jit 输入 pytree 结构（多一份编译 + 与在线评估路径分叉），而精确计算成本 <0.05 ms、1–2 MB/batch。`_normalize_state` 因 norm stats q01/q99 为 f64，输出恒 f64，与现状逐位同。
- **交付帐**：batch 载荷 ~257 MB（memory 三键 236 MB、image 单键 134 MB、两张原图 19 MB）、collate 19 ms、device_put 23 ms、XLA 编译产物 1 份——两侧相同，不构成 A/B 变量。

### B.4 接线：backend 显式开关 + 双根契约

`create_data_loader`（`src/mme_vla_suite/training/dataloader.py`）按 **`MMEVLA_DATA_BACKEND ∈ {packed, legacy, auto}`** 分派：

- **`packed`**：`dataset_path` 必须是打包库根；`meta/store_meta.json` 缺失、损坏、指纹不符、未通过 verify（`status != "verified"` 且未设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1`）、`manifest_scope=="subset"`（未设 `MMEVLA_FRAMESAMP_ALLOW_SUBSET=1`——该开关为 S3 实施时新增的开发期放行阀，与 ALLOW_UNVERIFIED 同族、放行必打 WARNING，S5 及以上判据 run 出现即 run 无效）、`meta/pack.lock` 存在 → **直接 raise，绝不回退**。构造 `FrameSampDataset`。
- **`legacy`**：走原 `RoboMMEDataset`（旧路径逐字未动），`dataset_path` 语义不变。
- **`auto`**（必须显式设置才生效，仅本机探索）：按 `store_meta.json` 存在性分派并打 WARNING。**未设置默认 `legacy`——与现状逐字节相同，零静默切换。正式 launcher 一律显式 `packed` 或 `legacy`（R16）。**
- **双根契约**：packed 模式三个位置全部显式解析——打包库根＝`dataset_path`；源库根（pkl）＝`MMEVLA_FRAMESAMP_SOURCE`（未设取 `store_meta.source_dataset_root`）；清单＝`MMEVLA_FRAMESAMP_MANIFEST`（未设取 `store_meta.manifest_path`）。解析结果三者全部写进 env.json（provenance）。**禁止从打包库目录名做字符串变换推导源库。**
- 其余（`transform_dataset` + `TorchDataLoader` + `DataLoaderImpl`）一行不动——同 workers 档位下 index 序列逐位不变的构造性保证。
- **不新增 `_CONFIGS` 条目**（`assets_dirs = assets_base_dir / self.name`，换名会把 norm_stats 路径指飞）；**不新增 CLI 参数**（`bench_train_steps.py` 护栏零改动；backend/双根全走环境变量）。启动侧变化：sbatch 里 `--dataset-path …/4task-gl-framesamp` + `export MMEVLA_DATA_BACKEND=packed`。
- **节点本地盘拷贝开关** `MMEVLA_FRAMESAMP_LOCAL_CACHE`（默认关；**GL 节点 `/tmp` 属 NFS 路径规约例外，启用前须用户显式批准**）：sbatch 开头 `df` 守卫（≥40 GB）→ cp 31.7 GB 到 `/tmp/$SLURM_JOB_ID/`（~65 s 保守估计）→ 逐 part sha256 校验（~32 s，不过即回退 NFS 直读并 WARNING）→ `dataset_path` 指本地、pkl 与清单仍走 NFS → 退出 `trap` 清理。**前置**：先在一个 GL job 里 `df -h /tmp` 确认 spgpu 本地盘规格。若 NFS 直读已 util≥95% 则保持关闭。

## C. 验证细节（三节三块的机读版，全部有判定行）

### C.0 前置资产（S0'）

确定性环境已确立（四节；确定性档 XLA_FLAGS 内联于〇节，独立冷编译逐位一致已实证），bench 量具已具备（B.0 表「既有能力」）。本计划只自补三项验证资产（S0'）：

1. bench 驱动 preflight 从「必须存在 `meta/stats.json`」改为「`stats.json` **或** `store_meta.json` 二选一」（否则 packed 库过不了启动检查）。
2. `BENCH_DUMP_IDX` batch_sampler 层记录（见 C.1）。
3. `scripts/smoke-local/README.md` 同步。

### C.1 第一块之一：index 序列等价

已读码确认：同一迭代器生命周期内 torch index 序列只由 `(len, seed, batch_size, drop_last, shuffle)` 决定、与 num_workers 无关；跨 epoch 分叉是 torch 既有语义（1.6）。perceptual 路径不消耗 Python RNG。

- 新工具 `dump_index_seq.py`（`scripts/data-pack-framesamp/`）用探针数据集 + 同一 `TorchDataLoader` dump 序列，w0/w4/w8 三档 diff 为空。**约束：dump 步数必须 < 一个 epoch 的 batch 数**（探针数据集较小时尤其注意），否则 epoch 边界后的分叉制造假阳性。
- **端到端旁证**：`BENCH_DUMP_IDX=1` 时，bench 侧 monkeypatch `mme_vla_suite.training.dataloader.create_data_loader` 取得 loader，在**首次 `iter()` 之前**执行 `object.__setattr__(loader._data_loader.torch_loader, "batch_sampler", _IdxProbe(orig))`——必须绕过 `DataLoader.__setattr__` 赋值守卫（初始化后直接赋值 `ValueError`，torch 2.7.1 实测；`object.__setattr__` 绕道后 `_index_sampler` 正确返回包装器，`persistent_workers` 下跨 epoch 持续生效）。`_IdxProbe` 实现 `__iter__`（每 batch idx 追加写 `$BENCH_RECORD_DIR/idx_seq.jsonl` 后原样 yield）与 `__len__`。**判据**：batch_sampler 枚举比交付超前 `prefetch_factor × num_workers` 个 batch（w4/pf2 实测超前 8），故判据是「`idx_seq.jsonl` 前 N 条与第一块 dump 逐条相同，N＝实际消费步数；尾部允许至多 `prefetch_factor × num_workers` 条超前记录」。
- 失败定位：len 不等 → 误用 `total_samples`；序列恰从 epoch 边界开始不同 → torch 既有语义（非 Dataset 问题）；其余 → 查 sampler/drop_last。

### C.2 第一块之二：样本/batch 内容等价

新工具 `compare_batches.py`：**import 复用** `compare_datasets.py` 的 `metrics()/grid_metrics()/Agg/_raw_bits()` 统计口径资产（不改其本体）。**对拍层**：真实链路进 `_collate_fn` 前先经 `transform_dataset`（Repack → RoboMMEInputs → DeltaActions → Normalize → ResizeImages → Tokenizer → Pad）——只比底层 Dataset 证明不了「模型实际吃到的内容一致」。因此两侧各实例化 **`transform_dataset` 之后的 Dataset**（与 `create_data_loader` 同一变换栈构造，不起 worker），对定点 idx 列表逐样本对拍 transform 后全键；底层（transform 前）对拍保留为失配时的定位辅助（区分「store 交付错」与「transform 放大」），不单独作 PASS 依据。

- 定点集（由清单精确构造，~8,200 个）：step_idx∈{0,1,2,29,30}（触发 padding）各 200 + {31,32,33} 各 200 + 每 episode 首样本 1,600 + 固定 seed 均匀随机 5,000。不依赖 shuffle 撞边界。注：Video* 任务首样本 step_idx = exec_start_idx（66–216）恒为满长——padding 边界由 step 集合组（Button* 任务）负责，首样本组承担「每 episode 至少验一发」。
- 判据（单一模式）：全部键 shape/dtype/`view(uintN)` 零容差逐位——两侧交付 dtype 相同，无需 astype 折算。
- batch 级补充：用 C.1 dump 的真实序列前 200 个 batch（transform 后样本）过 `_collate_fn` 对拍（覆盖到「进 device_put 前的最后一层」）。
- 判定行 `COMPARE_BATCH=PASS samples=… batches=… mismatches=0`；失败输出首个失配 idx/键/元素 hex，配合守卫 G1–G3 缩小到 gather/padding/reshape/normalize 四段。
- 落盘用**位型容器**：每键一个 `.bin`（原始字节，C-order）+ 旁置 JSON 记 shape/逻辑 dtype（含 bfloat16）/字节序/键名，读回 `np.frombuffer` + `view`；禁 npy/npz（丢 bf16 类型）。写盘后立即读回断言逐位相同（round-trip 守卫）。

### C.3 第二块：G2 训练轨迹对拍（packed 1000 步 vs G0 固化产物）

- **run**：`MMEVLA_DATA_BACKEND=packed`（显式设置）、`--dataset-path …/4task-gl-framesamp`、本机 2 卡 b8、seed 42、num_workers=4、**1000 步**、`SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299`（摘要步集与 G0 完全对齐）、确定性档 XLA_FLAGS（〇节）、EXP_NAME 独立、`bench_train_steps.py` 入口，clean HEAD 起跑。预计 ~2–2.5 h（含摘要停顿，一次完整 TrainState 摘要约 95–140 s × 12 次）。
- **前置断言**：起跑前 `BASELINE_ENV=PASS`（vs G0 r1，T5）；env.json 的 backend 为显式设置（auto 推断即判 run 无效，R16）；`MMEVLA_FRAMESAMP_VERIFY` 至少 fast 且库 `status=="verified"`；单 epoch 约束 8,000 < 395,289。
- **判据（`compare_baseline.py` 离线对拍 G0 r1 records）**：
  1. 逐步五标量 hex 列 diff 为空（1000 步全对齐）；
  2. 12 次摘要步 `state_digest` diff 为空；
  3. 输入侧：canonical 摘要（14 个记录步）+ 全步 index 序列（8,072 条）与 G0 逐位一致；**raw 输入摘要不计入判据**（G0 产物 raw 口径与现行 HEAD 在 4 个摘要步的 `static_image_emb`/`static_pos_emb` 两键上存在已知的预期失配——基线固化后交付 dtype 经过一次已验收的统一；canonical 同步一致）。
- **失败处置**（三节第二块已给完整流程，此处机读要点）：参数 sha256 逐叶二分找首个分叉模块；`mem_enc*` 分叉 → 回 C.2；LLM 主干分叉而 mem 一致 → 先跑 packed `-r2` 自证重跑稳定；无法归因 → 现场加跑同 HEAD legacy 一轮三方定位（定位手段非判据）；量化判据（四节/T6）只作评估参考不作放行。TrainState 逐叶数值参照取本机 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（@0/299/999）。
- **工具已知缺口**：`compare_baseline.py` 总判定行 `DET_CHECK` 未区分 raw/canonical 口径——raw 预期失配会拖累总行 FAIL。**处置已定（2026-08-27 用户拍板）：不修工具本体**——G2 对拍时沿用分项判读（五标量 hex / state_digest / canonical 输入 / index 序列四分项逐项判定），并在留档写明 raw 预期失配的来源；总行 FAIL 不作为 G2 结论依据。

### C.5 守卫测试

新文件 `scripts/data-pack-framesamp/test_pack_guards.py`（不混进 `data-preprocess-GL/test_guards.py`），刻意制造失败断言亮红灯，`JAX_PLATFORMS=cpu` pytest 秒级。按依赖拆两组、分属两个实施步：

**Store 组（S2 交付，只依赖格式层与打包工具）**：
- G1 迷你库（ref-shard 派生连续前缀 subset，须含 ≥33 帧 episode）打包→读取逐位对拍；
- G4 meta 缺失 / manifest sha 不符 / offsets 不符三条各自 raise 且不回退散 npy；
- G5 blob 截短 1 字节启动即炸；同尺寸中部翻转由 full 校验档抓出；
- G7 CPU 后端生成 pos 表被拒；
- G11 构造「两个 episode 在同一 t 的帧互换」迷你库，断言写侧校验抓不到（预期，钉死「pos memcmp 不钉 g」边界认知）而 verify 全量对拍必须亮红灯；
- G12 mock 短读（首次返回一半字节），断言续读补齐且结果正确；mock EOF/越界断言 raise；
- G14 `meta.status != "verified"` 时 packed 分派必 raise；设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1` 放行且必打 WARNING。

**Dataset 组（S3 交付，依赖 FrameSampDataset）**：
- G2 dtype 边界钉死：step=30 短样本与 step=31 满长样本经 `_pad` 后各键 dtype 必须一致且为 image bf16 / pos f32 / stt f32（两个 (g,t) 由迷你库「须含 ≥33 帧 episode」硬约束保证存在）；
- G3 选帧重复索引必须重复输出不去重；
- G6a 换算公式单测——直接用全量清单构造查表数组（不构造 Dataset、不碰 store），断言 `len==395289` 且 VideoUnmask / VideoUnmaskSwap 各一个 episode 首样本 `_step_of == exec_start_idx`；G6b（S5 随第一块一起绿）在全量打包库上构造 `FrameSampDataset` 复验同两条；
- G8 mock 线程池抛错证明已彻底移除；
- G9 `use_state_emb is False` 前提钉死；
- G10 spawn 一个子进程消费 Dataset，断言子进程内 store 懒构造（`_owner_pid == 子进程 pid`）、fd 有效可读、两张小表 `.nbytes` 与 meta 一致且 `.base is None`（进程内副本非映射）、父进程无句柄泄漏；
- G13 喂 `perceptual-framesamp-modul.yaml`（integration_type=modulation、memory_token_dim=1024，其余同形），断言 `__init__` raise。

## D. GL 吞吐验收（第三块的机读版；全部 GL 验收按 E 节严格串行顺序执行）

- **MB/s 口径**：`dataloader_bench.py` 的 `_AVG_BYTES_PER_SAMPLE` 从 history_config + `episode_manifest.json` 现场推导（均值帧数 = Σ min(t+1,32)/395,289 = 30.996 → 均值 2.43 MB/样本；上界 2.49 MB；勿写死）；主判读换 mountstats `server_read`；新增 majflt 采样（页缺失口径，冷/热证据链辅助量）。`block_until_ready` 覆盖整个 `(obs, actions)` pytree（只 block actions 会低估 device_put 成本）。新增 gather/pkl 分段计时（每样本两段耗时直方图落 records——「谁是新瓶颈」的观测资产）。
- **dataloader-only 四档（S8a，单 GPU job）**：默认 w2/w4/w8/w16，seed 310–313（避开已用 42/200–205/210–212 防 page cache 串扰）；**档位在提交 job 前须与用户确认（2026-08-27 用户指定），与逐 job 资源审批分列、两者都不可省**。
- **e2e 600 步（S8b，`gl_e2e_fix.sbatch` 参数化后入口，4×A40/16C/96G）**：T1 w4（**最重要**：官方默认 workers 还需不需要调）→ T2 w8（直接对 v1-e2efix-w8c16）→ T3 w2（探底）；条件档 T4 w16、T5 w4c8（8C 直接对 v1-e2e-b64，「官方口径净收益」最干净对照）。sbatch 硬编码的 `--dataset-path` 两处（训练命令与 env.json）必须参数化，默认值保持现状。
- **对照组（历史 GL 实测，标注口径不重测）**：v1-e2e-b64（6.933 s / 69.7%）、w8c16 5.301 s / 71.2%、w12c16 5.319 s / 70.6%、w16c16 5.327 s / 67.1%（三档平坦，「只调参上限」＝5.301 s）、compute-only 4.778 s。
- **冷/热（证据口径 cold-like）**：31.7 GB 打包库一个 epoch 内即全驻 page cache，热态必然偏乐观；pkl 156 GB 仍是长期 NFS 流量来源。C1/H1 各 300 步（稳态窗口，与 T1–T3 的 600 步分开口径），同一 allocation 内串行（`COLDHOT=1`、`--time=04:00:00`；先 C1 后 H1，排除节点差异），共用同一冻结 index 序列（同 seed + C.1 dump 存证）；`/proc/meminfo` Cached 15 s 采样落 `meminfo.csv` + cgroup `memory.stat` 的 pgmajfault 同步采样。「冷」无法严格证明，结论一律称 **cold-like**，判据 `(C1稳态−H1稳态)/H1 ≤ 15%`。并行采 `nvidia-smi --query-compute-apps` 存证 worker CUDA context。
- **成功判据**（AGENTS 16 口径，禁中位数标题结论；主判据表 5 项由 `analyze_gpu_util.py` 机器判定输出单行 `E2E_ACCEPT=PASS|FAIL`，附加判据由吃多个 record_dir 的汇总脚本判定，人工只复核）：

| 指标 | v1 基线 | 必达 | 期望 | 下界 |
|---|---|---|---|---|
| 步时中位 | 6.933 s | ≤5.00 s | ≤4.95 s | 4.778 s |
| util 稳态均值 | 69.7% | ≥90% | ≥95% | — |
| 0% 采样占比 | 27.8% | ≤5% | ≤2% | — |
| 慢步(>8s)墙钟占比 | 32.9% | ≤5% | ≤2% | — |
| epoch(6,176 步) | 11.9 h | ≤8.6 h | ≤8.5 h | 8.2 h |

  > 方向性目标：GPU 占用 100%（north star）。阈值维持上表（字面 100% 物理不可达）；S8b 报告必附「距 100% 残差分解」——0% 采样与慢步来源归因（worker CUDA context、step 边界、H2D 等），作为后续优化立项输入。
  > 必达 ≤5.00 s 有意严于「只调参上限」5.301 s——低于它才证明重构有超出调参的净收益（对照基线为历史口径实测，差值不做单项归因）。附加判据：w4 与 w8 步时差 ≤3%（否则 CPU 侧仍未松绑）；NFS server_read（e2e，T1–T3）公式上界 ≈31–33 MB/s（155 MB/step ÷ 目标步时），历史实测普遍为公式口径 0.54–0.71 倍，稳态期望 ≈17–25 MB/s，**>65 MB/s（≈2× 公式上界）即视为读放大信号**，处置：image 大表 fd 以 `posix_fadvise(POSIX_FADV_RANDOM)` 替换 `WILLNEED` 对照重测；热态只剩 pkl 流量 25.3 MB/step ≈5 MB/s 属正常；dataloader-only（S8a）不套用本带，按 samples/s × 2.43 MB 现场折算；majflt 随 epoch 单调下降作辅助量。
- **GL 资源例外审批**：S8a/S8b 资源规格（e2e 4×A40、`--time` 2 h，COLDHOT 4 h）超出 `greatlakes.md` 默认硬限（≤2 GPU、≤30 min）——每个超限 job 提交前须用户显式批准 GPU 数与 walltime 并在 `greatlakes.md` 留放行记录；run_name 确认不能替代资源审批。
- **性能 allocation 冷态自证**：S8a 与 C1 的 env.json 必记「本 allocation 未执行 full 校验、未做本地复制预热」（`MMEVLA_FRAMESAMP_VERIFY=fast` + local-cache 关闭 + 开跑前 `/proc/meminfo` Cached 快照）。
- **provenance（所有 run，含本机 bench 与 dataloader-only）**：env.json 必记：resolved 后的 `dataset_path`/`source_dataset_root`/`manifest_path`、`store_meta.json` sha256（verify 回填后口径）、`manifest_sha256`、backend 及其来源（显式/auto——S5 及以上出现 auto 即判 run 无效）、`MMEVLA_FRAMESAMP_VERIFY` 档位与 full 结果、`MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED` 取值、local-cache 命中与 sha 校验结果、XLA_FLAGS、git HEAD。
- 结果分析一律 `analyze_gpu_util.py`；每个 >5 min run 留档 `docs/training-doc/<run_name>/`（records 含 env.json/metrics/gpu_util_dense/nfs_read/meminfo.csv/pgmajfault/分段计时/param_checksums）。本节吞吐 run 的 `--save-interval` 默认 1000，300/600 步内不触发 TrainState 摘要，性能口径不受摘要停顿污染。**与本机 speed 的分工**：本节是 GL 侧主判据（过/不过在此）；S9 的 G2-speed 是本机口径链式对账（不设阈值只报数），并存互不替代。

## E. 实施顺序、提交切分与留档

**总逻辑（先读这段再看表）**：整个实施分四个阶段——先在小数据上把全部新代码走通（**阶段 1 写代码**：迷你库全链路先行，代码没定型前不碰大数据，避免反复重造 31.7 GB）；再造一次正式数据（**阶段 2 造数据**：全量打包 + 全量校验）；然后证明「没改数」（**阶段 3 正确性**：先不训练逐位对拍，再真跑 1000 步对拍 G0 黄金基线）；最后证明「变快了」（**阶段 4 性能**：GL 验收 + 本机速度对账）。**全程严格串行（用户 2026-08-27 拍板）：前一步判定过关才开下一步，无任何并行或先行安排。** **唯一豁免（2026-08-27 用户批准）：S5/S6 本机对并行**——两 run 均不测速、判据与负载无关（S5 纯内容逐位；S6 确定性档 bitwise，util/步时本按 B5 禁作性能结论），资源不冲突（S5 纯 CPU / S6 两卡，NFS 合计需求远低于供给）；风险仅为 S5 若败则 S6 作废重跑，红线 B6 的「第一块不过不开第二块」在本对 run 上按此豁免执行、放行规则仍要求两块都过。说人话版见第一部分六节。

### 阶段 1：写代码（本机开发，迷你库全链路先行）

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| S0' | **补验证小工具**：preflight 兼容 packed、`BENCH_DUMP_IDX`、env.json provenance 扩展、README 同步。`compare_baseline.py` 不修（C.3 缺口处置已定：G2 分项判读） | 用户拍板开工 | STEPS=5 跑通（WARMUP_STEPS=0；原 3 步判据结构性不可行，用户拍板改 5）、idx_seq.jsonl 落盘 | ~30 min |
| S2 | **写格式层和打包工具**：+ Store 组守卫（G1/G4/G5/G7/G11/G12/G14），ref-shard 派生迷你库全流程（含迷你库全量 verify） | S0' | Store 组 pytest 全绿 | ~2 h 开发 |
| S3 | **写新 Dataset 和切换开关**：FrameSampDataset + backend 接线 + Dataset 组守卫（G2/G3/G6a/G8/G9/G10/G13）+ 迷你库真实 spawn loader 矩阵 w0/w1/w4/w16 × 2 epoch（fd 泄漏检查：前后 `ls /proc/<pid>/fd` 计数） | S2 | Dataset 组 pytest 全绿（G6 只跑 G6a）+ 矩阵无错无泄漏 | ~1.5 h |

### 阶段 2：造正式数据（本机 tmux）

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| S4 | **全量打包 + 全量校验**（decode 档，双趟源读 ≈582 GB；verify 16 进程三键对拍 + row_digests）+ 构建留档 | S3 + launch 预提交 | `VERIFY_PACK=PASS scanned=483291 mismatches=0` | 40–80 min + 20–40 min |

### 阶段 3：证明「没改数」（本机，核心验证）

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| S5 | **第一块·不训练对拍**：定点 8,200 样本 + 200 真实 batch + G6b（run_name 建议 `v1-framesamp-cmp`） | S4 | `COMPARE_BATCH=PASS` + G6b 绿 | 30–60 min |
| S6 | **第二块·G2 训练对拍（终局检验）**：packed 一轮 1000 步（clean HEAD 起跑、preflight 必过）→ `compare_baseline.py` 离线对拍 G0 固化产物 | S5（第一块通过后）| 五标量 hex 1000 步 + 12×state_digest diff 空 + canonical/index 一致 + 留档 | ~2–2.5 h + 对拍 |

### 阶段 4：证明「变快了」（S6 通过后才开始）

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| S7.5 | **GL 验收资产参数化**（gl_e2e_fix.sbatch、gl-dataloader 两个 sbatch、dataloader_bench.py、analyze_gpu_util.py，默认值＝现状） | S6 | 三个 launcher 默认值跑通、env.json 记到 backend/resolved 双根 | ~40 min |
| S8a | **GL dataloader 单测四档**（默认 w2/w4/w8/w16；**提交 job 前档位须与用户确认，2026-08-27 用户指定**，计划默认值不视为已授权） | S7.5 + launch 预提交 + 档位确认 + 超限 job 逐个资源审批 | 吞吐数据落档（backend==packed 显式、fast 档冷态自证） | 15 min×4 + 排队 |
| S8b | **GL e2e 收官测试**：600 步 T1–T3(+条件档) + cold-like/hot（COLDHOT 双跑各 300 步） | S8a + launch 预提交 + 逐 job 资源审批（4×A40 / 2–4 h 超硬限） | `E2E_ACCEPT=PASS` + 距 100% 残差分解 | 3×2 h + 1×4 h |
| S9 | **本机速度对账 G2-speed**：`v1-g2-speed` 一轮 **1000 步**（speed 统一口径，〇节），vs `v1-g0-speed-r2` 对比落档、回填登记簿 | S8b | 稳态统计 + 对比表落档 | ~40 min |

- **commit 切分**（沿用 `commitV<大>.<小>:` 中文体例，本计划从 **V3.0** 起（2026-08-27 用户指定，原 V2.5–V2.9 编号作废）；每个正式 run 拆「launch.md 预提交 → clean HEAD 起跑 → 结果留档提交」三段，兼顾 AGENTS 12「起跑前记录」与 clean HEAD；顺序与 S 步严格串行一致，每 commit 可独立回滚）：
  1. **V3.0**（阶段 1 前半，S0'+S2）：验证资产补齐 + 格式层 + 打包工具 + Store 组守卫（迷你库通过）；
  2. **V3.1**（阶段 1 后半，S3）：新 Dataset + backend 接线 + Dataset 组守卫 + spawn 矩阵；
  3. **docs**：S4 launch 预提交 → S4 全量打包+verify（clean HEAD 起跑）→ docs：S4 构建留档（`docs/dataset-build-doc/`）；
  4. **V3.2**（S5 对拍工具；2026-08-27 用户拍板提前占用本号，后续依次顺延）：`dump_index_seq.py` + `compare_batches.py`（clean HEAD 起跑要求工具先入库）；
  5. **docs**：S5/S6 launch 预提交 → S5/S6 运行（clean HEAD 起跑）；
  6. **V3.3**（S5 收官）：第一块通过（结果留档）；
  7. **V3.4**（S6 收官）：G2 对拍通过（结果留档 + 登记簿 T8 回填）；
  8. **V3.5**（S7.5）：GL 验收资产参数化；
  9. **docs**：S8a → S8b → S9 逐 run「launch 预提交（GL 资源审批记录随附）→ clean HEAD 起跑 → 结果留档」，按串行顺序逐个走完；
  10. **docs**：GL 验收汇总留档 + `docs/v1-framesamp-dataflow.md` 定稿。
- **run_name 建议**（起跑前逐个交用户确认，AGENTS 6）：`v1-framesamp-cmp`（S5）、`v1-framesamp-g2`（S6）、`v1-framesamp-dl-w{2,4,8,16}`、`v1-framesamp-e2e-w{4,8,2}c16`、`…-coldlike/-hot`、`v1-g2-speed`（S9）；打包库名 `4task-gl-framesamp`。
- **回滚策略**：功能回滚＝launcher 里 `MMEVLA_DATA_BACKEND` 切回 `legacy` + `--dataset-path` 指回源库（必须一起回退）；打包库保留作证据不删（不进 git，31.7 GB），确认彻底放弃方案时才删。
- **收官清理**：验证结束后清理 `v1-store/cache/jax/` 下各 EXP_NAME 缓存与 `~/.cache/jax_*` 软链，清理 S 步临时 run（AGENTS 6）。
- 汇总报告：新增 `docs/v2-dataloader-restructure-report.md`；`docs/v1-nfs-bottleneck-analysis.md` 只加指针不改结论。

## F. 风险 Top3 与规避

1. **行号错位（静默错帧，loss 只会慢慢变差不报错）**——防线分工：写侧 100% pos memcmp 钉死 **t** 与帧序（不钉 g）；写读共用 `row_of()` 排除公式分叉；**g 级身份唯一凭据是 verify 全量对拍（483,291 帧零遗漏）+ row_digests 逐行摘要**；运行时逐样本 pkl 身份校验（显式 raise）钉每次访问。G11 守卫保证「换帧攻击」在 verify 层必然亮红灯。
2. **交付 dtype 恒等前提被上游推翻**（flax promote_dtype 语义变更 / 有人改 `Pi0Config.dtype`）——G2 的 bitwise 是可证伪的硬验收；若因上游漂移失败，按 C.3 处置流程定位（preflight 的库版本断言也会先行报警），留档。
3. **page cache 假象与 pkl 新墙**——头条结论用 cold-like 口径（同 allocation 串行 + pgmajfault 证据链）；bench 分段打点 gather/pkl 各自耗时直接看谁是新瓶颈；pkl 若成墙走已预留的 Phase C；worker 在途内存 ~8 GB（batch 载荷口径 × 16 worker × prefetch 2）。

## G. 红线清单（实施期逐条自检，显式编号）

| # | 红线 |
|---|---|
| R1 | 训练循环/模型/超参/seed 零改动 |
| R2 | **`scripts/train.py`、`src/openpi/**`、`models/**`、`training/dataset.py`、`shared/**` 不动**（B.0 表为唯一授权范围；bench/sbatch/analyzer 等验证验收资产的参数化改动除外且默认值必须等价现状） |
| R3 | 同 workers 档位下 index 序列构造性不变 |
| R4 | `even_sampling_indices` 复用不重写 |
| R5 | 4task-gl 只读；新库旁路新增+原子写+provenance+fail-loud；packed 模式绝不回退散 npy |
| R6 | 身份只从 `episode_manifest.json`；换算必须带 `exec_start_idx`；身份校验用显式 raise 不用 assert |
| R7 | 旧分支代码不惊动；旧链路原地保留，不安排删除 legacy 的 commit |
| R8 | 禁复活 d951aef 的已弃用脚本 |
| R9 | uv 纪律（`uv run`、`UV_LINK_MODE=copy`） |
| R10 | >5 min 任务 tmux+tee+EXIT_CODE、Monitor 每级行缓冲 |
| R11 | GPU util 判读 AGENTS 16 口径（机器判定 `E2E_ACCEPT` 为准） |
| R12 | 正式 run clean HEAD 起跑+run_name 用户确认；凡预计或实际 >5 min 的诊断/基准/等价性 run（含 S6、S8a/S8b）一律留档 `docs/training-doc/<run_name>/`（AGENTS 17）；S4 全量打包走 `docs/dataset-build-doc/` |
| R13 | commit 逐文件 add、中文 body 详写过程 |
| R14 | 缓存类目录收敛 `v1-store/cache/`（含 jax 编译缓存软链）；禁止覆盖 `HOME`；收官清理 |
| R15 | GL 节点 `/tmp` 本地缓存属规约例外，启用前须用户显式批准；退出 trap 清理 |
| R16 | 正式 launcher（sbatch/bench 驱动/dataloader-only）必须**显式**设置 `MMEVLA_DATA_BACKEND`（未设置默认 legacy）；env.json 标明显式/auto，S5 及以上出现 auto 推断即判该 run 无效 |
| R17 | S5 及以上与全部 GL 验收禁止设置 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED` 与 `MMEVLA_FRAMESAMP_ALLOW_SUBSET`（后者为 S3 实施时新增的 subset 迷你库开发期放行阀）；两开关仅迷你库/开发期可用 |

## T. 基线机读细节

### T1 G0_SCOPE 断言

- 白名单 regex：`^(docs/|scripts/smoke-local/|scripts/dtype-unify/|scripts/data-pack-framesamp/|scripts/data-preprocess-GL/paths\.sh$|[^/]+\.md$)`（`scripts/data-pack-framesamp/` 为本计划新增验证资产目录：纯离线工具，不被训练进程 import）。**注意 `src/mme_vla_suite/datastore/` 与 `training/framesamp_dataset.py`、`training/dataloader.py` 不在白名单**——它们改变训练进程代码，G2 的「训练语义等价」正是由三块验证证明，而非白名单豁免；G2 起跑 commit 与锚点的 diff 中这些文件须在 launch.md 逐 hunk 说明「legacy 路径逐字未动」。
- 判定命令（进 launch.md，含原始输出）：`git diff --name-only 55e6e5bf8ef38b780902d0e63257ea859a432a2c HEAD | grep -Ev '<白名单>'`，输出须全部为 B.0 表内文件且逐 hunk 说明。附加：`git status --porcelain` 为空；submodule 指针与锚点一致；env.json `git_dirty == false`。

### T2 量具脚本职责（均在 `scripts/smoke-local/`）

- `check_baseline_env.py`（preflight）：读取目标基线 `env.json` 与 `BASELINE_MANIFEST.json`，按 T5 清单逐项断言，输出单行 `BASELINE_ENV=PASS|FAIL`（FAIL 非零退出 + 逐项差异清单）。
- `compare_baseline.py`（对拍）：输入两份 records 目录，产出逐步标量 hex diff、`state_digest` diff、`batch_digests` 逐键 diff（raw 与 canonical 双口径）、rel 分布与包络对比（T6 口径）；digest 失配时输出逐叶数值统计（max-abs/max-rel/L2/cosine，params/opt_state/EMA 全叶子；无数组可算即 `INCONCLUSIVE`）；先校验双方 `BASELINE_MANIFEST.json`（产物 sha256 不符即 fail-loud）。已知缺口与 G2 使用方式见 C.3。
- bench 驱动（`run_2gpu_epoch_bench.sh` + `bench_train_steps.py`）现行能力：EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE 与缓存软链进 `v1-store/cache/jax/`、XLA_FLAGS 外部注入、逐步五标量 hex、完整 TrainState 摘要（步 0 与末步必记）、batch 摘要 raw+canonical、全步 index 序列、真实 argv 与编译缓存命中计数进 env.json、`SAVE_INTERVAL=0` 联动 `BATCH_DIGESTS=0`、步数护栏 ≤1200、`EXTRA_DIGEST_STEPS`、`STATE_DUMP_STEPS`。

### T5 preflight 断言项（`check_baseline_env.py`）

1. `uv.lock` sha256；单列版本：torch、jax、jaxlib、numpy、ml_dtypes（`importlib.metadata` 现场取；torch 版本决定 `randperm` 排列，变了则样本序列变、一切失效）；
2. GPU 型号 + 驱动（`nvidia-smi --query-gpu=name,driver_version`）+ `jax.devices()` 数量/型号；CUDA_VISIBLE_DEVICES；
3. 四节全部 git 外指纹：`norm_stats.json`、`pi05_base/params`、tokenizer 模型、`episode_manifest.json` 顶层 `sha256`、数据集 `source_spot_sha256`（16 抽样）；
4. XLA_FLAGS 原文逐字比对；JAX 配置：`jax_enable_x64`、`jax_default_matmul_precision`、`XLA_PYTHON_CLIENT_MEM_FRACTION`、fsdp_devices；
5. 对拍 run 的 `steps × batch_size < 395,289` 单 epoch 约束；
6. 目标基线 `BASELINE_MANIFEST.json` 全部条目 sha256 复验。

### T6 量化判据参数（四节的机读版）

- `rel(a,b)=|a−b|/max(|a|,|b|,1e-8)`；统计档 median/p95/max；余量系数 2×。
- 绝对下限：loss 1e-6；grad_norm/llm_grad_norm/mem_enc_norm 1e-5；末步 param_norm 1e-5。
- null 对优先级：确定性档独立冷编译两轮（实测逐位为零 → 退化为下限守卫）→ 生产档两轮（上界：loss rel median 2.7e-3 / max 4.6e-2）。
- 包络：`rel_AB(t) ≤ 2 × max_null_envelope(t)` 逐步。
- 判定行：`QUANT_EQUIV=PASS|FAIL scalars=5 null=<pair> margin=2.0`。
- 适用限定：G2 场景不作放行依据（三节第二块）；仅作失败定位与评估参考。

### T7 G0 产物清单（records schema，固化于 `docs/training-doc/v1-grad-baseline-g0b/records/round{1,2}/`）

1. `metrics.jsonl`：逐步五标量十进制 + hex；
2. `param_checksums.jsonl`：12 次摘要步完整 TrainState 摘要（全叶子 sha256 + `state_digest`）；
3. `batch_digests.jsonl`：14 个记录步逐键 `sha256(dtype‖shape‖bytes)`（raw）+ canonical 字段（升 f32 后按位视图哈希）+ 全步 index 序列摘要；
4. `scalars_hex.tsv`：`metrics.jsonl` 规范化投影（剔除 wall_time）+ 其 sha256（`c799a0b2…`，两轮相同）——「两轨迹是否一致」退化为一次 sha256 比较；
5. `env.json`：环境指纹（真实 argv、库版本、GPU/驱动、XLA_FLAGS、编译缓存命中/编译计数）；
6. `BASELINE_MANIFEST.json`：逐产物 sha256 / 行数 / schema 版本；
7. util 采样原始数据与统计、launch.md、result.md（util/步时仅留档参考，禁作性能结论）。

### T8 基线链登记簿（唯一权威；实施时回填）

| 链节 | run_name | commit | 判据 | 结论 | 产物路径 |
|---|---|---|---|---|---|
| 确定性资产·生产档噪声底 | `v1-det-d0-r{1,2}` | `d9e509e` | 两轮重跑噪声底（非判据基线） | FAIL（预期）：loss rel median 2.7e-3 / max 4.6e-2，作 T6 null 上界 | `docs/training-doc/v1-det-d0-r{1,2}/` |
| 确定性资产·确定性档 | `v1-det-d2-r{1,2}`、`v1-det-d2cold-r{1,2}` | `d9e509e` | 两轮逐步 hex + state_digest + batch_digest diff 为空 | **双 PASS**（共用缓存与独立冷编译均逐位一致——跨期 bitwise 判据授权闸开） | `docs/training-doc/v1-det-*/` |
| **G0（链头）** | `v1-grad-baseline-g0b-r{1,2}` | `570287f`（`<G0-HEAD>`） | G0_SCOPE + r1/r2 千步自证 | **PASS**：1000 步标量 hex / 12×state_digest / 14×batch_digest（raw+canonical）/ index 8072 全逐位一致（scalars_hex sha256 `c799a0b2…`）；TrainState 数组 @0/299/999 存 `/data/hongzefu/v1-baselines/g0b-r1-state-dump/`（sha 清单进 git） | `docs/training-doc/v1-grad-baseline-g0b/` |
| **G2** | `v1-framesamp-g2` | `cf64ddd`（起跑 HEAD，clean） | vs G0 r1 固化产物：五标量 hex + 12×state_digest bitwise + canonical 输入 + index（C.3，分项判读） | **PASS**（2026-08-27）：1000 步五标量 hex 零失配、12×state_digest 零失配、canonical 14 步零失配、index 8072 逐个一致；`scalars_hex.tsv` sha256 与 G0 同值 `c799a0b2…`；raw 4 步×2 键预期失配（B3 不计入，V2.4b dtype 统一来源）；与 S5 并行（用户豁免） | `docs/training-doc/v1-framesamp-g2/` |
| **G0-speed（锚点）** | `v1-g0-speed-r2` | `570287f` | speed 链锚点（AGENTS 16 稳态统计，1000 步口径） | 稳态中位 **1.152 s/step**（n=949，p10 1.097/p90 1.276）、均值 1.186、util 均值 86.5%、0% 采样 4.9%、慢步 3、epoch 外推 15.82 h（本机口径，非最终吞吐结论） | `docs/training-doc/v1-g0-speed-r2/` |
| **G2-speed** | `v1-g2-speed` | 待回填 | vs `v1-g0-speed-r2`（合并对账，只报数） | 待回填 | 待回填 |
| **GL 验收** | `v1-framesamp-dl-*`、`v1-framesamp-e2e-*` | 待回填 | `E2E_ACCEPT`（D 节五项）+ 附加判据 | 待回填 | 待回填 |

### T9 基线侧红线（与 G 节并行自检）

| # | 红线 |
|---|---|
| B1 | 引用任何基线产物前必跑 `check_baseline_env.py`，`BASELINE_ENV=FAIL` 即停 |
| B2 | 跨期 bitwise 判据的唯一授权是独立冷编译两轮逐位一致（已 PASS，T8）；不得以保留缓存为由绕过 |
| B3 | raw 输入摘要不计入 G2 判据（C.3 已知预期失配）；输入侧判定只认 canonical + index |
| B4 | 登记簿数字只在本文档 T8 维护一份，其他任何文档只引用不复制 |
| B5 | 性能结论只取 speed 口径 run（〇节）与 GL 验收；带 TrainState 摘要 / batch_digests / 确定性 XLA 档的正确性 run，其 util/步时禁作任何性能结论 |
| B6 | 三块秩序：第一块不过不开第二块；S8b 在 G2 bitwise 通过后才跑；量化判据不作 G2 放行依据 |

## 附录：历史沿革（压缩）

本文件由此前的基线链规约与 IO 重构计划合并重写而成（2026-08-27 用户拍板）：黄金基线经一次 300→1000 步换代升级（旧 300 步版 records 已删、launch/result.md 留存证于 `docs/training-doc/v1-grad-baseline-g0/`）；速度锚点同步换代为 `v1-g0-speed-r2`（旧 `v1-g0-speed` 留档不再作对比对象）；基线固化后交付 dtype 经过一次已验收的统一（G2 判据中 raw 输入摘要的预期失配即源于此）。全部过程细节见 git 历史与 `docs/training-doc/` 各 run 留档，不在本文件复述。
