# framesample+context 数据链路彻底重构计划（v1-dataloader-Restructure）

> 本文件是计划文档，尚未实施。定稿于 2026-08-24。范围：只兼容 `perceptual-framesamp-context` 一种 run。

## Context（为什么做这件事）

- v1 端到端实测（`docs/training-doc/v1-e2e-b64/`）：GPU util 均值仅 69.7%（中位 100% 是假象）、0% 采样占比 27.8%、慢步占稳态墙钟 32.9%；步时中位 6.933 s，而 compute-only 下界 4.778 s（+45%）。NFS 带宽已排除（供给 398–628 MB/s vs 需求 251 MB/s），坐实瓶颈在 dataloader worker 的 CPU/文件层。
- 用户确认：16 CPU 档（v1-e2efix-w8c16 / w16c16 / w12c16）的纯参数调整也吃不满 GPU，需要代码级彻底重构。
- 本轮范围：**只兼容 `perceptual-framesamp-context` 一种 run**；硬性要求**每个 step 拿到的 memory token 近乎一致、训练梯度差距极小**（本计划实际把目标提到了「逐位一致」并给出证明梯子）；同时让整条流程更具可读性。
- 用户已拍板：正式训练交付 dtype 用 **native bf16 模式**；GL e2e 验收**尽快并行提交**（接受与在跑三档互相排队）。

---

# 第一部分（给人看）

## 一、现状链路（从数据集预处理到每 step 的 memory token，详细版）

### 1.1 全景

```
   生产侧（一次性，已完成交付，本轮只读不动）        消费侧（训练期，每个 step 重复发生）
   ─────────────────────────────────────            ─────────────────────────────────────
   原始 H5（4 文件，321 GB）                         torch DataLoader 按 seed 抽 64 个样本 idx
     │  scan_manifest.py（只读 metadata）              │  分派给 spawn worker
     ▼                                                 ▼
   episode_manifest.json（唯一真值源）               RoboMMEDataset.__getitem__
     │  GL 8×1GPU job array（SigLIP 编码）             │  1 个 pkl + ≤32 个 npy + padding + transforms
     ▼                                                 ▼
   4task-gl 库（678 GB）────────────────────────►  collate → device_put → GPU jit embed_memory
                                                       ▼
                                                    每样本 512 个 memory token
```

### 1.2 阶段一：数据集预处理（生产侧）

```
 原始 H5（本机 /data/hongzefu 原件永久保留；turbo 暂存副本经 sha256 同源核对）
 ├── record_dataset_ButtonUnmask.h5      400 ep   无 demo 前缀（exec_start_idx = 0）
 ├── record_dataset_ButtonUnmaskSwap.h5  400 ep   无 demo 前缀
 ├── record_dataset_VideoUnmask.h5       400 ep   demo 前缀恒 66 帧
 └── record_dataset_VideoUnmaskSwap.h5   400 ep   demo 前缀 114–216 帧
       内部：episode_{i}/timestep_{t}/obs/front_rgb (256,256,3) u8、wrist_rgb、
             joint_state (7,) f4、gripper_state (2,) f4、action/joint_action (8,) f8 …
       ⚠ 无 chunk 无压缩，全是逐 timestep 散小数组
        │
        │ scan_manifest.py build：规范序 sorted(*.h5) × sorted(episode)，只读 metadata
        ▼
 v1-store/episode_manifest.json ——「唯一真值源」
       每 episode 记：(h5_file, raw_ep_idx) 身份、num_timesteps、exec_start_idx、
       三个前缀和偏移 global_episode_idx / exec_sample_offset / total_sample_offset，
       整体带 manifest_sha256（被改动即 fail-loud）
        │
        │ GL 8×1GPU job array（build_shard.py，LPT 装箱分 8 片，跨片指纹同源断言）
        │ 逐 timestep：front_rgb → resize_with_pad(256→224) → SigLIP So400m/14（bf16，
        │ 256 token）→ 池化出 8x8/4x4/2x2 三档 → PosEmb3D 按 step 切片 → 每帧存一个 npy
        ▼
 v1-store/datasets/4task-gl/（678 GB；1600 episodes；483,291 帧；395,289 执行样本）
 ├── meta/stats.json                    execution_samples=395289, total_samples=483291
 ├── meta/provenance.json、_shard{0..7}of8.json
 ├── features/episode_{g}/              g = 0..1599（四任务拉平编号，归属查清单）
 │     ├── token_emb_{t}.npy            每帧一个，602,951 B——np.save 的 pickle dict：
 │     │      image_emb_8x8 (1,64,2048) bf16   256 KiB ┐
 │     │      image_emb_4x4 (1,16,2048) bf16    64 KiB │←┐
 │     │      image_emb_2x2 (1, 4,2048) bf16    16 KiB │  │ framesample 只用
 │     │      pos_emb_8x8   (1,64, 768) f32    192 KiB │  │ 这三个键，共 112 KiB
 │     │      pos_emb_4x4   (1,16, 768) f32     48 KiB │←┤（= 每帧字节的 19%）
 │     │      pos_emb_2x2   (1, 4, 768) f32     12 KiB │  │
 │     │      state_emb     (8,)        f32       32 B ┘←┘
 │     └── kept_indices.json            token_dropping 用，framesample 路径完全不读
 └── data/{0..395288}.pkl               每执行样本一个，395,440 B：
        image / wrist_image (256,256,3) u8 两张原图（共 393 KiB）、state (8,) f32、
        actions (20,8) f64、prompt/subgoal 字符串、epis_idx / step_idx / exec_start_idx
```

要点：

- **features 按「全部 timestep」存**（含 Video* 任务的 demo 前缀帧），**data/pkl 只按「执行样本」存**（demo 前缀不出样本）。两套编号靠清单的两个前缀和互相换算。
- 每帧 npy 是 `np.save` 的 **object dict（pickle）**：7 个键绑在一起，**无法部分读取**——要拿 4x4 那 112 KiB 必须整包反序列化 589 KiB。
- `pos_emb_4x4` 实测是 **step_idx 的纯函数**（跨 episode 逐字节相同），却按帧冗余存了 483,291 份。

### 1.3 阶段二：训练时每个 step 的取数链（消费侧）

```
┌─ 主进程（jax，驱动 4×A40）─────────────────────────────────────────────────────┐
│ torch.Generator().manual_seed(seed) + shuffle + drop_last                      │
│   └─ 每 step 抽 64 个样本 idx。序列只由 (len, seed, batch, drop_last) 决定，    │
│      与 num_workers 无关 —— 这是「每 step 拿到哪些样本」的唯一决定因素           │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       │ idx 分派
       ▼
┌─ spawn worker × N（persistent_workers，prefetch_factor=2 未显式设置）───────────┐
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
│       读 19.7 MB 用 3.6 MB              热 17.7 ms / 冷 ~110 ms                 │
│  ④ 拼装 (n,16,2048) bf16 / (n,16,768) f32 / (n,8) f32 / mask (n,)              │
│  ⑤ right_padding_token_emb             ⚠ np.zeros 未指定 dtype：                │
│       t < 31 的短样本（占 6.27%）整体提升 float64（2.1 MB → 8.4 MB）             │
│  ⑥ reshape → static_image_emb (512,2048) / static_pos_emb (512,768) /          │
│       static_state_emb (512,8)（⚠ use_state_emb=false，GPU 不用，白算白传）     │
│       / static_mask (512,)                                                     │
│  ⑦ transforms：Repack → RoboMMEInputs（两张原图解析）→ DeltaActions →           │
│       Normalize(quantile) → ResizeImages(224) ⚠ worker 内 jax.jit：每个 spawn   │
│       worker 独立初始化 JAX（8 s）+ 编译 + 在 GPU0 建 442 MiB CUDA context      │
│       → PaligemmaTokenizer(64) → PadStatesAndActions                           │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       │ 64 个样本 dict（IPC 回主进程，含短样本时 memory 三键 ~757 MB/batch）
       ▼
┌─ 主进程 collate + 交付 ────────────────────────────────────────────────────────┐
│ _collate_fn = np.stack：batch 内含任一短样本（b64 概率 98.4%）→ memory 键        │
│   整体提升 float64：仅 static_image_emb 一键就 537 MB/batch（collate ~52 ms）    │
│ jax.make_array_from_process_local_data：host 侧把 f64 降回 f32 再 H2D           │
│   （x64 关闭；已核实降精度发生在 host 侧——537 MB 的分配/搬运/astype 全是白费）    │
│   ⚠ 1.6% 的「整批满长」batch 以 bf16 交付 → dtype 随 batch 摆动，XLA 编译两份    │
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

### 1.4 字节帐与耗时帐（实测）

| 口径 | 现状数值 | 备注 |
|---|---|---|
| 每样本读盘 | 均值 18.7 MB（上界 19.7 MB） | 其中真正用到 ≈4.1 MB，放大 4.6×；单看 npy 是 589 KiB 只用 112 KiB（5.4×） |
| 每样本耗时（热/冷） | 25.4 / 132.4 ms | gather 占 17.7 / ~110 ms；32 次 open 本身 74.3 ms |
| 每 step（b64）读盘 | 1.20 GB | 需求 251 MB/s，NFS 供给 398–628 MB/s（带宽不是瓶颈） |
| 每 step 文件打开 | 2,112 次 | 64 pkl + 64×32 npy |
| collate / IPC / device_put | 52 ms / 757 MB / 73 ms | float64 提升的直接代价 |
| 步时 | 中位 6.933 s（compute-only 下界 4.778 s，+45%） | GPU util 均值 69.7%、0% 采样 27.8%、慢步墙钟 32.9% |

### 1.5 浪费在哪里（按影响排序）

1. **文件个数**：每样本 ≤33 次 NFS open——32 次 64 KiB 读若落在一个常开 fd 上只要 0.33 ms，open 却要 74.3 ms。
2. **整包 pickle 反序列化**：7 键绑死，读 5.4× 于所需字节。
3. **float64 提升**：padding 未指定 dtype → 98.4% 的 batch 的 memory 张量以 4× 体积在 worker→IPC→collate→device_put 全程白搬运。
4. **每样本新建 ≤32 线程的线程池**（用完即弃）。
5. **pos_emb 冗余**：纯函数按帧存盘反复读，占必需读量 38%。
6. **worker 里的 JAX**：每 worker 初始化 8 s、GPU0 上 442 MiB CUDA context（16 workers ≈ 7 GB 显存 + 上下文抢占）。
7. **state_emb 白算白传**（use_state_emb=false）。

### 1.6 现状的确定性

给定 seed / batch_size / fsdp_devices 与同一份数据集，**每 step 的样本集合与 memory token 内容逐位可复现**；num_workers 只影响交付时机不影响内容。唯一例外在 XLA 层：同配置重跑目前非 bitwise 确定（根因已定位，见第三节 3.3）。

## 二、重构后的链路（数据预处理格式/步骤，与现状逐项对比）

### 2.1 一句话

**把 483,291 个 602 KB 小 npy 压成 32 个约 990 MB 的连续大文件（只含 framesample 真正要的三张表，共 31.7 GB），训练时用常开 fd 直接 pread。** 预处理的前两个阶段（清单、SigLIP 建库）与产物**原样保留、一字不动**，只新增一个纯派生的「阶段三：打包」。

### 2.2 预处理步骤对比：旧两阶段不动，新增阶段三

```
 阶段一 scan_manifest.py ──► episode_manifest.json          ┐
 阶段二 GL 8×1GPU SigLIP 建库 ──► 4task-gl（678 GB）         ├─ 与现状完全相同，不重跑
                                    │                        ┘
                                    │  阶段三（新增）：打包派生
                                    │  pack_framesamp_store.py（本机 tmux，
                                    │  纯 CPU+NFS，16 进程，20–40 min）
                                    │  · 逐 episode 读全部 token_emb_*.npy
                                    │  · 只抽 image_emb_4x4 / pos_emb_4x4 / state_emb
                                    │  · 每帧 100% memcmp（pos 表比对同时钉死帧身份）
                                    │  · 写后 pread 读回校验 + 逐 part sha256 原子落盘
                                    │  · verify 子命令独立进程再抽 5 万帧「写×读」对拍
                                    ▼
 v1-store/datasets/4task-gl-framesamp/            共 31.7 GB（特征侧体积的 1/9）
 ├── meta/store_meta.json         唯一契约：布局/形状/dtype/part 边界/源库指纹/
 │                                manifest_sha256（读侧逐项 fail-loud，绝不回退散 npy）
 ├── meta/pack_progress.jsonl     断点续跑记录
 ├── image_emb_4x4/part_000.bf16.bin … part_031.bf16.bin
 │       32 个 ≈990 MB 连续文件，(rows,16,2048) bf16 裸字节；
 │       行号 = total_sample_offset[g] + t（写读共用同一函数）；
 │       按 episode 边界切分 → 一个样本的 32 帧必落在同一 part
 ├── pos_emb_4x4.f32.bin          (586,16,768) f32 = 28.8 MB，按 step_idx 查表
 └── state_emb.f32.bin            (483291,8) f32 = 15.5 MB，按全局行号查表
 （data/{idx}.pkl 与两张原图不动：训练时仍从源库 4task-gl/data/ 读）
```

**预处理格式/步骤对比表**：

| 维度 | 现状 | 重构后 |
|---|---|---|
| 预处理阶段 | ① 清单 ② SigLIP 建库 | ①② 原样保留（产物不动）＋ ③ 打包派生（**新增**） |
| 是否重读 H5 / 重跑 SigLIP | — | 否。阶段三只读 4task-gl，纯 CPU，无 GPU、无集群 |
| 特征存储形态 | 每帧一个 npy 小文件（483,291 个；pickle dict 7 键，602,951 B，无法部分读取） | 32 个连续 `.bin` 大文件（只存所需 3 键；裸字节，dtype 显式声明于 meta） |
| pos_emb 存法 | 每帧冗余存一份（38% 的必需读量） | 586 行小表一份（pos 是 step 的纯函数，实测证实） |
| state_emb 存法 | 混在每帧 npy 里 | 独立小表 |
| 原图/actions/prompt | data/{idx}.pkl | **不变**（仍读源库；若实测成为新瓶颈，Phase C 预留了同机制打包接口） |
| 磁盘体积 | 678 GB | 源库原样保留 ＋ 新增 31.7 GB |
| dtype 契约 | 隐式（pickle 内嵌；padding 未指定 dtype 引发 f64 事故） | 显式：meta 声明 + 交付模式三选一开关收敛在一处 |
| 完整性判据 | 建库期三层一致性验证（已完成） | 打包 100% 逐帧 memcmp ＋ 写后读回 ＋ part sha256 ＋ 独立 verify 5 万帧 ＋ 读侧 fail-loud |
| 身份来源 | episode_manifest.json | 同一清单；行号公式写读共用，禁止目录序 |

### 2.3 训练取数链对比

```
┌─ spawn worker × N ─────────────────────────────────────────────────────────────┐
│ FrameSampDataset.__getitem__(idx)      单一路径、无分支、只服务 framesamp+context │
│  ① 清单查表得 (g, t)                    O(1) 数组，不读目录                       │
│  ② pickle.load(data/{idx}.pkl)          与旧路径同源同字节，~2.7 ms              │
│     ＋断言 pkl 内 epis_idx/step_idx == 清单推导值（行号错位即炸）                 │
│  ③ even_sampling_indices(t, 32)         同一个函数 import，选帧逐位不变          │
│  ④ gather：32 个常驻 fd 上 fadvise 预读 + 游程合并 preadv 直读进预分配数组        │
│     0 次 open、0 线程池、0 pickle       热 0.3–0.5 ms / 冷 10–30 ms             │
│     pos/state 从 mmap 小表切片（跨 worker 共享 page cache，零副本）              │
│  ⑤ _pad_native：预分配 bf16/f32，填充区清零 —— 无 float64 提升                   │
│  ⑥⑦ 拼装与 transforms 与旧路径完全相同                                          │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       ▼
  collate：batch 内 dtype 一致，np.stack 不再提升（memory 键 134 MB/batch，降 4×）
  device_put：native bf16 直付，host 侧无降精度搬运（23 ms vs 73 ms）
  GPU：同一段 jit 代码，promote_dtype 到 bf16 后输入张量与旧链路逐位相同
       → memory token 逐位一致；XLA 编译产物从 2 份合为 1 份
```

**训练期每 step 对比表**：

| 维度 | 现状 | 重构后 |
|---|---|---|
| 每样本文件打开 | ≤33 次（1 pkl + ≤32 npy） | 1 次（特征走常驻 fd） |
| 每样本读盘 | 18.7 MB（只用 4.1 MB） | 2.4 MB（几乎全用到） |
| 反序列化 | ≤32 次全量 pickle | 0 次（裸字节直读） |
| 线程池 | 每样本新建 ≤32 线程 | 无 |
| padding dtype | 隐式 float64 提升 | 显式 bf16/f32（native 模式） |
| collate 后 memory 键 | 537 MB/batch（98.4% 的 batch） | 134 MB/batch |
| 每 step 读盘 / 打开 | 1.20 GB / 2,112 次 | 159.5 MB / 64 次 |
| 单样本耗时（热/冷） | 25.4 / 132 ms | ≈7 / 15–40 ms |
| 供给余量（vs 计算需求） | 不足（GPU 空转 30%） | ≈15× |
| 预期步时 / epoch | 6.933 s / 11.9 h | ≈4.9 s / ≈8.5 h（下界 4.778 s / 8.2 h） |

## 三、「memory token 近乎一致」怎么保证（结论：可以做到逐位一致）

### 3.1 恒等链：memory token 由四个因素完全决定，逐一钉死

每个 step 的 memory token 由且仅由四件事决定。前三件重构后**构造性不变**，第四件（dtype）有变化但**已实测证明不改数**：

1. **这个 step 取了哪 64 个样本。** torch 的 index 序列只由 `(len(dataset), seed, batch_size, drop_last, shuffle)` 决定，与 num_workers 无关（已读 torch 源码确认：worker base_seed 的抽取时机恒定，不额外消耗 generator）。重构不换 `TorchDataLoader`、不换 generator、不换 seed 语义，且 `len` 相同（395,289）→ 序列逐位不变。
2. **每个样本选了哪 32 帧。** `even_sampling_indices(step_idx, 32)` 是纯函数、零随机源；新链路 **import 同一个函数**而非重写；`step_idx` 取自 pkl 且与清单推导值互相断言（不一致即炸）。
3. **每帧特征的字节。** 打包库的每一行在写入时与源 npy 100% memcmp（pos 表比对同时钉死「这个文件确实是 (g,t)」），写后读回校验，另有独立 verify 进程抽 5 万帧走真实读路径对拍 → 新链路读进内存的数组与旧链路逐位相同。
4. **这些字节以什么 dtype 走到 GPU encoder 输入。** 这是唯一有变化的环节，论证链（全部已在真 GPU 实测）：
   - 存储值本身是 bf16（image）与 f32（pos）；bf16→f32→f64 都是**精确升位**，往返无损。
   - **现状交付本来就不统一**：98.4% 的 batch 因 f64 padding 事故以 float64 到 host、host 侧 cast 成 f32 上卡；1.6% 的「整批满长」batch 以 bf16 上卡——即现状自己就有两种 dtype 路径并存，模型对两者一视同仁。
   - 模型侧第一层 `nnx.Linear` 显式 `dtype=bfloat16`，flax 的 `promote_dtype` 在做任何算术之前把输入统一转成 bf16；bf16 的值经任何精确升位再转回 bf16 必然复原 → **三种交付（bf16/f32/f64）进 `pos_proj`/`encoder_static` 的实际张量逐位相同**。已用真实形状实测：三种输入的 memory token 输出全等（max 差 0.0）。
   - 因此 native bf16 **不是引入新行为，而是把现状 1.6% batch 的行为推广到 100%**，顺带把 XLA 的两份编译产物合成一份。

### 3.2 不靠论证靠梯子：四层验证，每层有硬判据

论证再严密也可能被没想到的环节推翻，所以每一段恒等链都配一层可证伪的验证（细节在第二部分 C 节）：

| 层 | 证明什么 | 怎么证 | 判据 |
|---|---|---|---|
| 第 0 层 | 恒等链 (1)：样本序列 | 新旧 loader 同 seed dump index 序列对拍（w0/w4/w8 三档 + 真实训练链路旁证） | diff 为空 |
| 第 1 层 | 恒等链 (2)(3)：交付内容 | 8,200 个**定点**样本（step∈{0,1,2,29,30,31,32,33} 边界全覆盖 + 每 episode 首样本 + 随机）逐键对拍，另加 200 个真实 batch 过 collate 对拍 | replica 模式全键逐位零容差；native 模式 astype(f32) 后逐位 |
| 第 2 层 | **IO 重构本身对训练零影响** | 旧链路 vs 新链路 replica 复刻模式：本机 2 卡 b8 跑 300 步，逐步比 loss/grad_norm 等五个量的 hex ＋ 参数逐叶子 sha256 | 全部 bitwise 相同 |
| 第 3 层 | **dtype 正规化不改数** | 先单步定点梯度对拍（专挑「整批满长」这种唯一有 dtype 差异的 batch），再 replica vs native 300 步 | 主判据 bitwise |

说明两点：

- **replica 复刻模式**是专为第 2 层设计的交付模式：精确复现现状的 f64 padding 与 collate 提升，使新旧链路的差异只剩「字节从哪读」——第 2 层通过即证明重构没改任何东西；第 3 层再单独隔离 dtype 这一个变量。
- **本机 b8 是比集群 b64 严 38 倍的检验**：dtype 差异只出现在「整批满长」的 batch，其占比 b8 下 59.6%、b64 下仅 1.6%——第 3 层的差异敞口在 b8 上被放大 38 倍，本机通过则集群更稳。

### 3.3 前置条件：先证明「同配置重跑本身可复现」

2026-08-24 实测：同 commit 同配置同 seed 重跑两轮，参数校验和逐步全不相同——**当前默认设置下训练不是 bitwise 确定的**。根因已定位：bench 驱动脚本 run 名写死（跑不了第二轮）、结尾删 jax 编译缓存、`train.py` 硬编码覆盖缓存目录 → 每轮空缓存 → XLA 每轮重新 autotune、可能选中不同 kernel。这个问题不修，任何 A/B 的差异都无法归因给 dataloader。

修法（已核实可行性）：驱动脚本拆分「实验名/轮次名」并保留缓存；两轮**共用同一份 jax 编译缓存**（jax 0.5.3 的持久编译缓存默认托管 XLA per-fusion autotune 缓存，共用目录即复用 autotune 结果）；必要时经 `XLA_FLAGS` 加 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`（六个相关 flag 已在本仓库 venv 的 XLA 插件二进制里逐个确认存在）。先跑 D0/D1/D2 三档「同配置重跑两轮」实验，**两轮逐步校验和完全一致后**，才开始 3.2 的任何 A/B。

### 3.4 任何一层失败怎么办（「梯度差距极小」的硬兜底）

- **定位手段**：第 0/1 层失败输出首个失配 idx/键/元素的 hex，配合守卫测试把问题缩小到 gather/padding/reshape/normalize 四段之一；第 2/3 层失败用参数 sha256 逐叶子二分找首个分叉模块——分叉在 `mem_enc*` 指向交付内容（回第 1 层），分叉在 LLM 主干而 mem 一致指向非确定性（回 3.3 重立前提）。
- **量化兜底**：若第 3 层 bitwise 失败，先证 native 自身重跑稳定，再启用量化判据——loss 逐步相对差 median ≤ 1e-6 / p95 ≤ 1e-5 / max ≤ 1e-4，梯度范数各松一档，且相对差对 step 的回归斜率不得为正（轨迹不允许发散，无论绝对值多小）。阈值比 bf16 的 1 ULP（≈0.39%）还保守两个数量级。
- **降级路径**：量化判据仍不达标，正式模式从 native 降级 f32 或 replica（三模式开关收敛在一处，随时可切），**绝不「差不多就行」**。

## 四、明确不做的事

- **不打包 pkl**（每 step 仅 25.3 MB、2.7 ms/样本，且打包无体积收益；按同一套 part 机制预留 Phase C 接口，若实测它成为新墙再单独一轮）。
- **不预烘焙 ResizeImages、不预 tokenize prompt**（tokenize 实测 24 µs/次；烘焙需 119 GB 存储且要先证 jax CPU/GPU resize 逐位一致）。但**记录一条重要观测**：每个 dataloader worker 会在 GPU 0 建 442 MiB CUDA context（16 workers ≈ 7 GB 显存 + 上下文抢占），本轮 bench 里采样存证，作为下一轮「worker 去 JAX 化」的立项依据。
- **不重建数据集、不动源库、不动旧多分支 Dataset 代码**（symbolic/recurrent/token_drop 原地保留不惊动）。

## 五、可读性产出

- `docs/v1-framesamp-dataflow.md`：一页式数据流图表，从 H5 → 清单 → 源库 → 打包库 → Store → Dataset → transforms → collate → device_put → memory token，每一跳标形状/dtype/字节数/「这一跳有没有改数」。
- 格式契约 README（datastore 层）、打包工具 README、构建留档、每个 run 的 launch/result 留档。
- 新模块的形制断言本身就是文档：读代码即知适用域，超出即炸。

---

# 第二部分（技术细节，供 agent 追踪）

## A. 打包特征库

### A.1 目录布局

```
v1-store/datasets/4task-gl-framesamp/
├── meta/
│   ├── store_meta.json          # 唯一契约（layout/形状/dtype/part 边界/源库指纹/manifest_sha256）
│   ├── pack_progress.jsonl      # 断点续跑：每 part 一行（idx, rows, sha256, elapsed, host, pid）
│   └── row_digests.blake2b.bin  # 可选逐行摘要（483,291 × 16 B）
├── image_emb_4x4/part_000.bf16.bin … part_031.bf16.bin   # 各 ≈990 MB，(rows,16,2048) bf16 裸字节
├── pos_emb_4x4.f32.bin          # (586,16,768) f32 = 28.8 MB，按 step_idx 索引
└── state_emb.f32.bin            # (483291,8) f32 = 15.5 MB，按全局行号索引
```

- **行号公式**（写读两侧共用同一函数，物理上不可分叉）：`row(g,t) = manifest.episodes[g].total_sample_offset + t`。
- **part 切法**：按 `global_episode_idx` 升序累积 `num_timesteps`，累计 ≥ `ceil(483291/32)` 即切，**切点必在 episode 边界**（一个样本的 32 帧必在同一 part）。边界写入 `store_meta.json.parts[i]`。单文件方案与「每 episode 一文件」方案均已否决（前者与原子写+并行互斥，后者退化回每样本一次 open）。
- **bf16 落盘定论**（已实测）：裸 `.bin` + meta 声明 dtype。**禁止 `.npy`**——`np.save` 对 ml_dtypes bf16 写出 `V2` descr，`np.load` 丢类型。读侧 `np.memmap(dtype=ml_dtypes.bfloat16)` 与 `frombuffer(uint16).view(bfloat16)` 均实测可用。
- `store_meta.json` 关键字段：`layout="framesamp-4x4-v1"`、`manifest_sha256`（须等于当前 `episode_manifest.json` 的 sha256）、`source_provenance_sha256`、`source_spot_sha256`（16 个抽样源文件摘要，读侧启动抽验）、`num_rows/num_exec_samples/num_pos_rows`、三张表的 shape/dtype/row_bytes、`parts[]`（含每 part sha256）、`packer`（git_commit/host/reader/校验覆盖率）。**meta 最后写**，它是「打包完成」的唯一标志。

### A.2 打包工具

新目录 `scripts/data-pack-framesamp/`：`pack_framesamp_store.py`（子命令 `plan | pack | verify | report`）+ `run_pack.sh`（tmux 驱动，PYTHONUNBUFFERED=1 + pipefail + tee + EXIT_CODE=）+ `README.md`。

- **真值与复用**：清单经 `scan_manifest.load_manifest()`（sha256 fail-loud）；格式常量与 `row_of()` 从 `src/mme_vla_suite/datastore/framesamp_store.py` import，**绝不复制**（沿用 `build_shard.py` 的既定做法）。
- **运行位置**：**本机 detached tmux**（纯 CPU+NFS，不占 GL 的 spgpu GPU 配额；产物是确定性字节、由 memcmp 背书，不属于「本机吞吐结论」，不违反 AGENTS 13）。`multiprocessing.Pool`（默认 `min(16, cpu)`），每进程独占若干 part，天然无锁。带宽受限（NFS 供给 398–628 MB/s），预计 **20–40 min**。
- **源读取两档** `--reader`：`decode`（首跑默认，逐帧 `np.load(allow_pickle).item()` 全量 602,951 B，零布局假设，总读 291 GB）；`slice`（已实测 npy 内部偏移恒定：`image_emb_4x4`@262,595、`pos_emb_4x4`@541,352、`state_emb`@602,906，120/120 文件大小一致、60/60 memcmp 通过；三重守卫：st_size==602,951、数据段前 64 B 前缀与基准逐字节相同、逐帧 pos 窗口 100% memcmp——留作重跑加速档）。
- **逐帧即校验（100% 覆盖，写入路径内）**：① 该帧 `pos_emb_4x4` ≟ `pos_table[t]`（memcmp，同时钉死「文件确实是 (g,t)」与「pos 只依赖 t」两条不变量）；② `state_emb` ≟ state 表同一行；③ episode slab 写入 `.tmp` 后 **`os.pread` 读回与内存 memcmp**（read-after-write，多读 31.7 GB ≈ 80 s）；④ part 完成 → sha256 → `os.replace` → 追加 progress。
- **pos 表来源（定论）**：**主方案从源库抽取拼装**（用若干 episode 凑齐 t=0..585 的 `pos_emb_4x4`，逐位同源、零后端风险），主 pass 的 100% memcmp 即证明「只依赖 t」。`PosEmb3D` 现生成仅作旁证：**已实测 CPU 后端生成与库中值不逐位一致（差 1e-7~1e-5），GPU 后端一致**——若走生成路径必须断言 `jax.devices()[0].platform == "gpu"`，守卫测试钉死（G7）。
- **断点续跑** `--resume`：按 progress 校验「存在+大小+sha256」跳过完好 part，否则删除重做；`.tmp` 残留一律清除（同 `episode_is_complete`/`purge_episode` 思路）。
- **`verify` 子命令（独立进程、后验）**：随机抽 **50,000 个 (g,t)**（≈10%，固定 seed 记入报告）重新完整解码源 npy，与 `FrameSampStore.read_image_rows()` 对拍——唯一能证明「写 × 读合成正确」的检查（≈30 GB 读、2 min）。判定行 `VERIFY_PACK=PASS … mismatches=0`。
- **留档**：`docs/dataset-build-doc/4task-gl-framesamp/README.md`（AGENTS 12：commit、命令、源库指纹、耗时、三层校验结果）。

### A.3 pkl 侧：本轮不打包（定论）

量级不构成瓶颈（25.3 MB/step、2.7 ms/样本）；打包无体积收益（定宽后仍 156 GB）；源库 `data/` 依红线必须保留可读。**Phase C 预案**（不实施，仅预留）：同 part 机制加 `samples/part_XXX.bin` 定宽记录 + `strings.json` 字典编码（实测 prompt 全集封闭，400 样本仅 27 个不同值）；`actions` 保 f64、`state` 保 f32，否则 DeltaActions/Normalize 数值会变。

## B. 新 Dataset / 装配路径

### B.1 新增文件

```
src/mme_vla_suite/datastore/__init__.py
src/mme_vla_suite/datastore/framesamp_store.py    # 格式层：常量 + StoreMeta + FrameSampStore(只读) + row_of()
src/mme_vla_suite/datastore/README.md             # 存储格式契约
src/mme_vla_suite/training/framesamp_dataset.py   # 装配层：FrameSampDataset
```
改动既有文件仅一处：`src/mme_vla_suite/training/dataloader.py`（`create_data_loader` 内加约 10 行分派）。**不动**：`scripts/train.py`、`src/openpi/**`、`src/mme_vla_suite/models/**`、`training/dataset.py`、`shared/**`。格式层不 import 任何 training/model 模块（单向依赖）。

### B.2 `FrameSampStore`（格式层）

- `__init__` 完成全部 fail-loud 校验（layout / manifest_sha256 现场重算比对 / parts 连续覆盖 [0,num_rows) / 每 part 存在且 st_size==meta.bytes（抽 1 个 part 头尾 1 MiB 复验）/ 抽 1 条 source_spot_sha256 复验源库未动），之后 `__getitem__` 路径零判断；**绝不 fallback 到散 npy**（fail-open 禁令）。
- **大表用 pread、小表用 mmap**（实测：32×64 KiB 常开 fd pread 0.33 ms，np.memmap 切片 2.4–2.5 ms 走 NFS 缺页/revalidate 路径）：32 个 part fd `os.open` 常驻；`read_image_rows(rows, out)` 先对全部行发 `posix_fadvise(POSIX_FADV_WILLNEED)` 触发内核并发预读，再按连续行游程合并 `os.preadv` 直读进预分配 bf16 数组（短样本 32 行天然连续 → 1 次调用），返回字节数不符即 raise。pos/state 表 `np.memmap(mode='r')`——跨 worker 由 page cache 共享，零副本。

### B.3 `FrameSampDataset.__getitem__`（伪代码）

```python
def __init__(...):   # 形制断言即文档：representation_type=="perceptual"、type=="frame_sampling"、
                     # (budget,token_per_image,num_views)==(512,16,1)；load_manifest fail-loud；
                     # 清单派生 O(1) 查表 _epis_of/_step_of/_row_base（int32 数组，禁止 os.listdir）；
                     # dtype_mode ∈ {"native","f32","replica"} → self._pad 三选一（唯一开关点）
def __getitem__(self, idx):
    g, step = self._epis_of[idx], self._step_of[idx]
    data = pickle.load(open(f"{source}/data/{idx}.pkl"))          # 图像/state/actions/prompt 与旧路径同源同字节
    assert data["epis_idx"].item()==g and data["step_idx"].item()==step   # 行号错位的最后一道闸（raise）
    data["actions"] = data["actions"][:action_horizon]; data.pop(两个 *_online 键)   # 与旧路径逐字相同
    frames = even_sampling_indices(step, 32)                      # 复用同一函数，不重写
    rows = self._row_base[g] + np.asarray(frames, np.int64)
    img  = store.read_image_rows(rows)                            # (n,16,2048) bf16——0 open、0 线程池、0 pickle
    pos  = store.pos_rows(frames); stt = store.state_rows(rows)
    img, pos, stt, mask = self._pad(img, pos, stt, n)             # dtype 三模式的唯一分叉点
    data["static_image_emb"] = img.reshape(-1,2048)               # (512,2048)
    data["static_pos_emb"]   = pos.reshape(-1,768)
    data["static_state_emb"] = self._normalize_state(np.repeat(stt,16,axis=0))  # 保留精确计算（f64，与现状同源同式）
    data["static_mask"]      = np.repeat(mask,16)
    for k in _NONE_KEYS: data.setdefault(k, None)                 # recur_* / subgoal 等下游会索引的空键
    return data
```

- **`_pad_native`**：按最终形状一次性 `np.empty` 分配（img bf16 / pos f32 / stt f32），填充区清零，全程零 concatenate。**`_pad_f32`**：同上 img 改 f32（精确升位）。**`_pad_replica`**：直接调用原 `right_padding_token_emb`（含 f64 提升），逐位复刻现状。
- **`_collate_fn` 一行不改**：replica 下逐样本 dtype 与现状一致，`np.stack` 提升行为自然复现；native/f32 下 batch 内 dtype 一致无提升。
- **static_state_emb 定论：保留精确计算不置零**——`HistAugObservation` 虽允许 None，但置 None 改变 jit 输入 pytree 结构（多一份编译 + 与在线评估路径分叉），而精确计算成本 <0.05 ms、1–2 MB/batch。`_normalize_state` 因 norm stats q01/q99 为 f64，输出恒 f64，与现状逐位同。
- **dtype 三模式帐**（实测）：host memory 张量 replica 757 MB/batch（同现状）→ native **257 MB** → f32 391 MB；collate 52→19/27 ms；device_put 73→23/38 ms（f64 降精度已核实发生在 host 侧，jaxlib `Squash64BitTypes`）；XLA 编译产物从 2 份（dtype 随 batch 摆动）→ 1 份。**正式默认 native（用户已拍板）**，replica 仅用于 A/B，f32 作保守回退。逐位相同论证链已实测：`nnx.Linear(dtype=bf16)` 的 `promote_dtype` 使三种交付进 `pos_proj`/`encoder_static` 的实际张量完全相同。

### B.4 接线

`create_data_loader`（`mme_vla_suite/training/dataloader.py`）内按 `dataset_path/meta/store_meta.json` 是否存在分派：存在 → `FrameSampDataset`（模式取环境变量 `MMEVLA_FRAMESAMP_DTYPE`，默认 native，非法值 raise），不存在 → 原 `RoboMMEDataset`（旧路径逐字未动）。其余（`transform_dataset` + `TorchDataLoader` + `DataLoaderImpl`）一行不动——index 序列逐位不变的构造性保证。

- **不新增 `_CONFIGS` 条目**（`assets_dirs = assets_base_dir / self.name`，换名会把 norm_stats 路径指飞）；**不新增 CLI 参数**（`bench_train_steps.py` 六道护栏与全部 sbatch 零改动）。启动侧唯一变化：sbatch 里 `--dataset-path …/4task-gl-framesamp`（+ 可选 `export MMEVLA_FRAMESAMP_DTYPE=replica`）。
- 可选加性改动：`TorchDataLoader` 加 `prefetch_factor: int | None = None` 形参（None 时不传 torch，默认行为逐字节不变）。
- **节点本地盘拷贝开关** `MMEVLA_FRAMESAMP_LOCAL_CACHE`（默认关）：sbatch 开头 df 守卫（≥40 GB）→ cp 31.7 GB 到 `/tmp/$SLURM_JOB_ID/`（~65 s）→ 逐 part sha256 校验（~32 s）→ dataset_path 指本地、pkl 仍走 NFS；容量不足 WARNING 回退。**前置**：先在一个 GL job 里 `df -h /tmp` 确认 spgpu 本地盘规格（greatlakes.md 未记载）。若 NFS 直读已 util≥95% 则保持关闭。

## C. 等价性验证（判据梯子，全部有判定行）

### C.0 前置：确定性前提确立（不过这关不开任何 A/B）

已核实根因链：`run_2gpu_epoch_bench.sh` run 名写死+记录目录存在即拒跑（跑不了第二轮）、结尾 `rm -rf ~/.cache/jax_*`、`scripts/train.py` 硬编码覆盖 `jax_compilation_cache_dir`——每轮空缓存 → XLA 每轮重新 autotune。已核实杠杆：jax 0.5.3 `jax_persistent_cache_enable_xla_caches` 默认值就是 `xla_gpu_per_fusion_autotune_cache_dir`，**共用编译缓存目录 = 同时复用编译产物与 autotune 结果**；六个 XLA flag（`--xla_gpu_deterministic_ops=true`、`--xla_gpu_exclude_nondeterministic_ops=true`、`--xla_gpu_autotune_level=0`、dump/load autotune、require_complete）已在 venv 的 XLA 插件二进制逐个确认存在，jax 0.5.3 无对应 config 开关，**只能走 `XLA_FLAGS` 环境变量**。

- **S0 脚本改动**（验证资产，非训练代码）：`run_2gpu_epoch_bench.sh` 拆 `EXP_NAME`（决定编译缓存目录，A/B 共用）与 `RUN_TAG`（决定记录目录，A/B 各异）；加 `KEEP_JAX_CACHE=1`；`XLA_FLAGS` 外部注入并写进 env.json；同步更新 `scripts/smoke-local/README.md` 第二节。
- **S1 三档实验**（各两轮相同 run，100 步，SAVE_INTERVAL=10）：D0=现状删缓存（预期 FAIL，复现对照）；D1=共用缓存；D2=D1+`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`（期望 PASS）。判定行：两轮 `global_digest` 与 `loss.hex` 逐步 diff 为空。取首个 PASS 档固定为全部 A/B 环境；D2 仍 FAIL 则加 exclude flag 并降 50 步二分（分清「第 1 步就不同」vs「第 k 步漂移」）。

### C.1 第 0 层：index 序列等价

已读码确认 torch index 序列只由 `(len, seed, batch_size, drop_last, shuffle)` 决定、与 num_workers 无关；perceptual 路径不消耗 Python RNG。新工具 `dump_index_seq.py`（并入 `scripts/data-pack-framesamp/`）用探针数据集 + 同一 `TorchDataLoader` dump 序列，w0/w4/w8 三档 diff 为空；端到端旁证：`bench_train_steps.py` 加第三处 monkeypatch（`BENCH_DUMP_IDX=1` 时 patch `_collate_fn`，记录 `_probe_idx` 后删键再交原 collate，交付内容不受影响）。失败定位：len 不等 → 误用 `total_samples`；序列不同 → 查 sampler/drop_last。

### C.2 第 1 层：样本/batch 内容等价

新工具 `compare_batches.py`：**import 复用** `compare_datasets.py` 的 `metrics()/grid_metrics()/Agg/_raw_bits()` 统计口径资产（**不改其本体**，其三层 --mode 语义不适用于 dataloader 对拍）。不走 create_data_loader，直接对**定点 idx 列表**逐样本对拍：

- 定点集（由清单精确构造，~8,200 个）：step_idx∈{0,1,2,29,30}（触发 f64）各 200 + {31,32,33} 各 200 + 每 episode 首样本 1,600 + 固定 seed 均匀随机 5,000。**不依赖 shuffle 撞边界**（98.4% 只保证「至少一个短样本」，不保证覆盖 step=0/1/30）。
- 判据：replica 模式全部键 shape/dtype/`view(uintN)` 零容差逐位；native/f32 模式 `astype(f32)` 后逐位相同 + dtype 差异逐键清单固定输出（image：短 f64→bf16、长 bf16→bf16；pos：短 f64→f32、长 f32→f32；state 恒 f64 不变；mask 与其余全部键位相同）。
- batch 级补充：用第 0 层 dump 的真实序列前 200 个 batch 过 `_collate_fn` 对拍（专验 collate 提升行为复刻）。
- 判定行 `COMPARE_BATCH=<mode> PASS samples=… batches=… mismatches=0`；失败输出首个失配 idx/键/元素 hex，配合守卫 G1–G3 缩小到 gather/padding/reshape/normalize 四段。

### C.3 第 2 层：300 步训练轨迹 bitwise（本机 2 卡 b8）

A=重构前 commit 旧链路，B=重构后 replica 模式。同 `EXP_NAME`（共用编译缓存；replica 下 dtype 序列相同 → HLO 相同 → 缓存命中）、同 XLA_FLAGS、同 seed 42、b8、300 步、save-interval 25。判定：`loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm` 五个 hex 列 + `global_digest` 逐步 diff 全空。失败定位：分叉叶子在 `mem_enc*` → 回第 1 层；在 LLM 主干而 mem 一致 → 非确定性回 C.0；再不然走 smoke-local README 第 3 级（固定 batch 单步逐元素梯度，固定 batch 直接用第 1 层落盘的 npz）。

### C.4 第 3 层：native 模式（主判据 bitwise）

先跑最便宜的**单步定点梯度对拍**（~5 min）：取「整批满长」batch（唯一有 dtype 差异的场景）与「含短样本」batch 各一，同一初始 state 各算一步逐元素比梯度——满长 batch 逐位相同即基本结案。再 300 步 B(replica) vs C(native)：独立 EXP_NAME（dtype 变 → 缓存 key 必变），主判据 `loss.hex`+`global_digest` 逐步 bitwise。**b8 是比 b64 严 38 倍的检验**（全长 batch 占比 59.6% vs 1.6%，dtype 差异点恰在全长 batch）。降级判据（仅当主判据失败且先证 C==C' 重跑稳定）：逐步相对差 loss median≤1e-6/p95≤1e-5/max≤1e-4，grad_norm 三量各松一档，末步 param_norm≤1e-5；**趋势判据**：相对差对 step 回归斜率 ≤0 或不显著，单调上升无论多小判 FAIL。若失败 → 正式模式降级 f32 或 replica，不「差不多就行」。

### C.5 守卫测试

新文件 `scripts/data-pack-framesamp/test_pack_guards.py`（**不混进** `data-preprocess-GL/test_guards.py`），照搬其「刻意制造失败断言亮红灯」风格，`JAX_PLATFORMS=cpu` pytest 秒级：G1 迷你库（ref-shard 派生 3 帧）新旧 gather 对拍逐位；G2 dtype 边界钉死（step=30→replica f64 / step=31→bf16）；G3 选帧重复索引必须重复输出不去重；G4 meta 缺失/manifest sha 不符/offsets 不符三条各自 raise 且不回退散 npy；G5 blob 截短 1 字节启动即炸；G6 `len(new_ds)==395289`；G7 CPU 后端生成 pos 表被拒（固化实测发现）；G8 mock 线程池抛错证明已彻底移除；G9 `use_state_emb is False` 前提钉死。

## D. 吞吐验收（GL，尽快并行提交——用户已拍板，接受与在跑三档排队）

- **MB/s 新口径**：公式从 18.73 → **2.389 MB/样本**（`dataloader_bench.py` 的 `_AVG_BYTES_PER_SAMPLE` 改为从 history_config 现场推导，勿再写死）；主判读换 mountstats `server_read`，新增 **majflt** 采样（mmap 缺页，`rchar` 对 mmap 无效）。`dataloader_bench.py` 其余零改动（只耦合 `_config.cli()` 与 `create_data_loader` 签名，均不变）。
- **dataloader-only 四档**（单 GPU job）：w2/w4/w8/w16，seed 310–313（避开已用 42/200–205/210–212 防 page cache 串扰）。
- **e2e 600 步**（v2 harness `gl_e2e_fix.sbatch` 零改动，4×A40/16C/96G）：T1 w4（**最重要**：官方默认 workers 还需不需要调）→ T2 w8（直接对 v1-e2efix-w8c16）→ T3 w2（探底）；条件档 T4 w16、T5 w4c8（8C 直接对 v1-e2e-b64，「官方口径净收益」最干净对照）。对照组：v1-e2e-b64（6.933 s / 69.7%）、三档 v1-e2efix（结果落地后补入对照，作「只调参上限」）、compute-only 4.778 s。
- **冷/热**（必测——31.7 GB 打包库一个 epoch 内即可全驻 page cache，热态数字会偏乐观；pkl 156 GB 仍是长期 NFS 流量来源）：C1 冷（新节点+新 seed，报「前 100 步」与「稳态」两窗口）、H1 热（同节点同 seed 紧接重跑）；判据 `(C1稳态−H1稳态)/H1 ≤ 15%`。sbatch 新增 15 s 一次 `/proc/meminfo` Cached 采样落 `meminfo.csv`（画「page cache 爬到 31.7 GB 后步时阶跃」证据图）。并行采 `nvidia-smi --query-compute-apps` 存证 worker CUDA context。
- **成功判据**（AGENTS 16 口径，禁中位数标题结论）：

| 指标 | v1 基线 | 必达 | 期望 | 下界 |
|---|---|---|---|---|
| 步时中位 | 6.933 s | ≤5.00 s | ≤4.95 s | 4.778 s |
| util 稳态均值 | 69.7% | ≥90% | ≥95% | — |
| 0% 采样占比 | 27.8% | ≤5% | ≤2% | — |
| 慢步(>8s)墙钟占比 | 32.9% | ≤5% | ≤2% | — |
| epoch(6,176 步) | 11.9 h | ≤8.6 h | ≤8.5 h | 8.2 h |

  附加判据：w4 与 w8 步时差 ≤3%（否则 CPU 侧仍未松绑）；NFS server_read 应落 60–160 MB/s 量级（仍 400+ 则 readahead 读放大，加 `madvise(MADV_RANDOM)` 重测）；majflt 随 epoch 单调下降。
- 结果分析一律 `analyze_gpu_util.py`；每个 >5 min run 留档 `docs/training-doc/<run_name>/`（records 含 env.json/metrics/gpu_util_dense/nfs_read/**meminfo.csv**/param_checksums）。

## E. 实施顺序、提交切分与留档

两条并行轨道，S4 汇合：

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| S0 | 修 bench 驱动（EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE、XLA_FLAGS 注入） | — | STEPS=3 连跑两次不拒跑、缓存未删 | ~20 min |
| S1 | 确定性前提 D0/D1/D2（各两轮 100 步） | S0 | 两轮 digest+loss.hex diff 空 | ~1 h |
| S2 | 格式层 + 打包工具 + 守卫测试，ref-shard 小库全流程 | — | pytest G1–G9 全绿 | ~1.5 h 开发 |
| S3 | 全量打包（本机 tmux，decode 档）+ verify 5 万帧 + 构建留档 | S2 | `VERIFY_PACK=PASS mismatches=0` | 20–40 min + 2 min |
| S4 | FrameSampDataset + dataloader 分派 | S3 | pytest + 小库对拍全绿 | ~1 h |
| S5 | 第 0/1 层（定点 8,200 样本 + 200 真实 batch） | S4 | `COMPARE_BATCH=* PASS` | 30–60 min |
| S6 | 第 2 层 replica vs 旧链路 300 步 bitwise | S1+S5 | 五 hex 列+digest diff 空 | ~30 min |
| S7 | 第 3 层 native（单步定点梯度 + 300 步 vs replica） | S6 | bitwise PASS（或降级判据） | ~35 min |
| S8 | GL 验收：dataloader-only 四档 → e2e T1–T3(+条件档) → 冷/热 | S4（提交不等三档 v1-e2efix，判读时结果落地即补对照） | D 节判据表 | 15 min + 3×2 h + 2×2 h |

- **commit 切分**（沿用 `commitV<大>.<小>:` 中文体例，dataloader 重构起 V2 系列）：V2.1 bench 驱动拆分解锁 A/B → V2.2 确定性前提确立 → V2.3 打包工具+守卫（小库通过）→ docs 打包留档 → V2.4 新 Dataset+接线（三模式）→ V2.5 第 0/1 层通过 → V2.6 第 2 层逐位一致 → V2.7 第 3 层 native 逐位一致 → docs GL 验收留档 + `docs/v1-framesamp-dataflow.md` 定稿。每 commit 可独立回滚；打包产物在 v1-store 不进 git，回滚即删目录；旧链路原地保留（红线 7），**不安排删除 legacy 的 commit**。
- **run_name 建议**（起跑前逐个交用户确认，AGENTS 6）：`v1-framesamp-det-d{0,1,2}-r{1,2}`、`v1-framesamp-ab-{old,replica,native}`、`v1-framesamp-dl-w{2,4,8,16}`、`v1-framesamp-e2e-w{4,8,2}c16`、`…-cold/-hot`；打包库名 `4task-gl-framesamp`。
- 汇总报告：新增 `docs/v2-dataloader-restructure-report.md`；`docs/v1-nfs-bottleneck-analysis.md` 只加指针不改结论。

## F. 风险 Top3 与规避

1. **行号错位（静默错帧，loss 只会慢慢变差不报错）**——四层防线：写侧 100% pos/state memcmp 钉死 (g,t)；写读共用 `row_of()`；verify 独立进程 5 万帧「写×读」合成对拍；运行时逐样本 pkl 身份断言。
2. **native 逐位结论被上游推翻**（flax promote_dtype 语义变更 / 有人改 `Pi0Config.dtype`）——梯子 C.3/C.4 是可证伪的硬验收而非「跑一下看看」；失败即降级 f32/replica，留档。
3. **page cache 假象与 pkl 新墙**——头条结论用冷缓存口径；bench 分段打点 gather/pkl 各自耗时直接看谁是新瓶颈；pkl 若成墙走已预留的 Phase C；worker 在途内存 native 模式从 ~24 GB 降到 ~8 GB（v1 OOM 的根源之一顺带缓解）。

## G. 红线清单（实施期逐条自检）

训练循环/模型/超参/seed 零改动；index 序列构造性不变；`even_sampling_indices` 复用不重写；4task-gl 只读、新库旁路新增+原子写+provenance+fail-loud；身份只从 `episode_manifest.json`；旧分支代码不惊动；禁复活 d951aef；uv 纪律（`UV_LINK_MODE=copy`）；>5 min 任务 tmux+tee+EXIT_CODE、Monitor 每级行缓冲；GPU util 判读 AGENTS 16；正式 run clean HEAD 起跑+留档+run_name 用户确认；commit 逐文件 add、中文 body 详写过程。
