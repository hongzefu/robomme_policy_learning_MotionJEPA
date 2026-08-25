# framesample+context 数据链路彻底重构计划（v1-dataloader-Restructure）

> 本文件是计划文档，尚未实施。v1 定稿于 2026-08-24；v2 于 2026-08-25 依据两轮独立对抗验证修订；本版为**定稿 v3（2026-08-25）**：
> ① 61-agent 对抗验证 workflow（报告 `v1-framesamp-restructure-adversarial-review.md`：确认问题 high 4 / medium 6 / low 10，驳回 3）；
> ② Codex 四路独立审计（8 阻断 / 16 高风险 / 8 规格缺口；其 file:line 断言已逐条读码复核属实）；
> ③ 定稿复核 workflow（全 opus，6 agent：修订落实 ×2 / 数字复算 / 设计自洽 / 残留扫描 + 裁决）：裁决清单必须修 M1–M13、建议修 S1–S9 已全部落实，驳回 R1–R6 维持原文。
> 全部修订对照见文末「修订记录」。范围：只兼容 `perceptual-framesamp-context` 一种 run。

## Context（为什么做这件事）

- v1 端到端实测（`docs/training-doc/v1-e2e-b64/`）：GPU util 均值仅 69.7%（中位 100% 是假象）、0% 采样占比 27.8%、慢步占稳态墙钟 32.9%；步时中位 6.933 s，而 compute-only 下界 4.778 s（+45%）。NFS 带宽已排除（供给 398–628 MB/s vs 需求 256 MB/s），坐实瓶颈在 dataloader worker 的 CPU/文件层。
- 16 CPU 档纯参数调整不解决问题已有**三档完整实据**：w8c16 **5.301 s / 71.2% / epoch ≈9.09 h**、w12c16 **5.319 s / 70.6% / ≈9.13 h**、w16c16 **5.327 s / 67.1% / ≈9.14 h**——workers 8/12/16 曲线完全平坦，三档均距 compute-only 下界 4.778 s 差约 11%、util 均值仍只有 67–71%、慢步墙钟 32–36%。需要代码级彻底重构。
- 本轮范围：**只兼容 `perceptual-framesamp-context` 一种 run**；硬性要求**每个 step 拿到的 memory token 近乎一致、训练梯度差距极小**（本计划把目标提到「受控环境下逐位一致」并给出证明梯子，有效性域见第三节）；同时让整条流程更具可读性。
- 用户已拍板：正式训练交付 dtype 用 **native bf16 模式**；GL e2e 验收**尽快并行提交**（接受与在跑档位互相排队）。

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
     ▼                                                 │  → worker 内 collate（np.stack 合并成 batch）
   4task-gl 库（678 GB）────────────────────────►      ▼
                                                    batch 经 IPC 回主进程 → device_put → GPU jit embed_memory
                                                       ▼
                                                    每样本 512 个 memory token
```

### 1.2 阶段一：数据集预处理（生产侧）

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
       整体带完整性字段 sha256（被改动即 fail-loud；下游产物引用该值时命名为 manifest_sha256）
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

- **features 按「全部 timestep」存**（含 Video* 任务的 demo 前缀帧），**data/pkl 只按「执行样本」存**（demo 前缀不出样本）。两套编号靠清单的偏移字段互相换算，**换算必须带 `exec_start_idx`**（精确公式见 B.3——漏掉它会让 Video* 任务样本错位 66–216 帧）。
- 每帧 npy 是 `np.save` 的 **object dict（pickle）**：7 个键绑在一起，**无法部分读取**——要拿 4x4 那 112 KiB 必须整包反序列化 589 KiB。
- `pos_emb_4x4` 实测是 **step_idx 的纯函数**（跨 episode 逐字节相同），却按帧冗余存了 483,291 份。

### 1.3 阶段二：训练时每个 step 的取数链（消费侧）

```
┌─ 主进程（jax，驱动 4×A40）─────────────────────────────────────────────────────┐
│ torch.Generator().manual_seed(seed) + shuffle + drop_last                      │
│   └─ 每 step 抽 64 个样本 idx。同一迭代器生命周期内（单个 epoch 内）序列只由    │
│      (len, seed, batch, drop_last) 决定、与 num_workers 无关；跨 epoch 见 1.6   │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       │ idx 分派
       ▼
┌─ spawn worker × N（persistent_workers，prefetch_factor 未显式设置=torch 默认 2）─┐
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
│       读 19.3 MB 用 3.7 MB（上界，只含 npy 不含 pkl 0.4 MB）热 17.7/冷 ~110 ms  │
│  ④ 拼装 (n,16,2048) bf16 / (n,16,768) f32 / (n,8) f32 / mask (n,)              │
│  ⑤ right_padding_token_emb             ⚠ np.zeros 未指定 dtype：                │
│       t < 31 的短样本（占 6.27%）整体提升 float64（2.1 MB → 8.4 MB）；           │
│       t ≥ 31 的满长样本走纯切片分支，不触发 padding、保持 bf16                   │
│  ⑥ reshape → static_image_emb (512,2048) / static_pos_emb (512,768) /          │
│       static_state_emb (512,8)（⚠ use_state_emb=false，GPU 不用，白算白传）     │
│       / static_mask (512,)                                                     │
│  ⑦ transforms：Repack → RoboMMEInputs（两张原图解析）→ DeltaActions →           │
│       Normalize(quantile) → ResizeImages(224) ⚠ worker 内 jax.jit：每个 spawn   │
│       worker 独立初始化 JAX（8 s）+ 编译 + 在 GPU0 建 442 MiB CUDA context      │
│       → PaligemmaTokenizer(64) → PadStatesAndActions                           │
│  ⑧ _collate_fn（np.stack）在 worker 内执行：batch 内含任一短样本（b64 概率      │
│       98.4%）→ memory 键整批提升 float64（仅 static_image_emb 一键 537 MB）      │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       │ 已合并的 batch 经 IPC 回主进程（含短样本时 batch 载荷 ~757 MB：三键 740+原图 19）
       ▼
┌─ 主进程 交付 ──────────────────────────────────────────────────────────────────┐
│ jax.make_array_from_process_local_data：host 侧把 f64 降回 f32 再 H2D           │
│   （x64 关闭；降精度发生在 host 侧，机制见 B.3——537 MB 的分配/IPC/astype 全是    │
│   白费）。⚠ 1.6% 的「整批满长」batch 以 bf16 交付 → dtype 随 batch 摆动，XLA     │
│   编译两份                                                                      │
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

### 1.4 字节帐与耗时帐（实测；数字来源与留档口径见第五节「探针脚本固化」与 A.2 的 probe_layout.py）

| 口径 | 现状数值 | 备注 |
|---|---|---|
| 每样本读盘 | 均值 19.08 MB（上界 19.69 MB） | 其中真正用到 均值 3.95 MB（上界 4.07 MB），放大 ≈4.8×；单看 npy 是 589 KiB 只用 112 KiB（5.3×） |
| 每样本耗时（热/冷） | 25.4 / 132.4 ms | gather 占 17.7 / ~110 ms；32 次 open 本身 74.3 ms |
| 每 step（b64）读盘 | 1.22 GB | 需求 256 MB/s，NFS 供给 398–628 MB/s（带宽不是瓶颈） |
| 每 step 文件打开 | 2,112 次 | 64 pkl + 64×32 npy |
| collate / IPC / device_put | 52 ms / 757 MB / 73 ms | float64 提升的直接代价；collate 在 worker 内执行（num_workers>0 时）；757 MB 为 batch 载荷（memory 三键 740 MB + 两张原图 19 MB） |
| 步时 | 中位 6.933 s（compute-only 下界 4.778 s，+45%） | GPU util 均值 69.7%、0% 采样 27.8%、慢步墙钟 32.9% |

> 注：全表统一十进制 MB（1 MB = 10⁶ B），与既有留档中的 MiB 数字不可直接比。

### 1.5 浪费在哪里（按影响排序）

1. **文件个数**：每样本 ≤33 次 NFS open——32 次 64 KiB 读若落在一个常开 fd 上只要 0.33 ms，open 却要 74.3 ms。
2. **整包 pickle 反序列化**：7 键绑死，读 5.3× 于所需字节。
3. **float64 提升**：padding 未指定 dtype → 98.4% 的 batch 的 memory 张量以 4× 体积在 worker→collate→IPC→device_put 全程白搬运。
4. **每样本新建 ≤32 线程的线程池**（用完即弃）。
5. **pos_emb 冗余**：纯函数按帧存盘反复读，占必需读量 38%。
6. **worker 里的 JAX**：每 worker 初始化 8 s、GPU0 上 442 MiB CUDA context（16 workers ≈ 7 GB 显存 + 上下文抢占）。
7. **state_emb 白算白传**（use_state_emb=false）。

### 1.6 现状的确定性

给定 seed / batch_size / fsdp_devices 与同一份数据集：

- **单个 epoch 内**（同一迭代器生命周期内），每 step 的样本集合与 memory token 内容逐位可复现，num_workers 只影响交付时机不影响内容。
- **跨 epoch 边界与 num_workers 相关**（torch 既有语义，与本重构无关）：`_BaseDataLoaderIter.__init__` 每次构造迭代器都从同一个 generator 抽一次 `_base_seed`，而 `persistent_workers`（w>0）跨 epoch 只 `_reset` 不重建、w0 每 epoch 重建——两条路径消耗 generator 的节奏不同，**同 seed 下 w0 与 w>0 从第 2 个 epoch 起排列分叉**（torch 2.7.1 读码 + 双人独立脚本复现确认）。恒等链只需「新旧链路在相同 num_workers 下序列相同」，不受影响，但一切「与 num_workers 无关」的表述以本条为准收窄。
- 唯一例外在 XLA 层：同配置重跑目前非 bitwise 确定（根因已定位，见第三节 3.3）。

## 二、重构后的链路（数据预处理格式/步骤，与现状逐项对比）

### 2.1 一句话

**把 483,291 个 602 KB 小 npy 压成 32 个连续大文件（只含 framesample 真正要的三张表，共 31.7 GB），训练时用常开 fd 直接 pread。** 预处理的前两个阶段（清单、SigLIP 建库）与产物**原样保留、一字不动**，只新增一个纯派生的「阶段三：打包」。32 个 part 中前 31 个 ≈990–1020 MB，末 1 个 ≈621 MB（贪心切分的尾部效应）。

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
| dtype 契约 | 隐式（pickle 内嵌；padding 未指定 dtype 引发 f64 事故） | 显式：meta 声明 + 交付模式三选一开关收敛在一处 |
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
│  ⑤ _pad_native：预分配 bf16/f32，填充区清零 —— 无 float64 提升                   │
│  ⑥⑦⑧ 拼装、transforms 与 worker 内 collate 与旧路径完全相同                     │
└──────┬─────────────────────────────────────────────────────────────────────────┘
       ▼
  collate：batch 内 dtype 一致，np.stack 不再提升（memory 三键 ~257 MB/batch，
       其中 image 单键 537→134 MB，降 4×）
  device_put：native bf16 直付，host 侧无降精度搬运（23 ms vs 73 ms）
  GPU：同一段 jit 代码，promote_dtype 到 bf16 后输入张量与旧链路逐位相同
       → memory token 逐位一致（有效性域见第三节）；XLA 编译产物从 2 份合为 1 份
```

**训练期每 step 对比表**（均值与上界分列，口径与 1.4 对齐）：

| 维度 | 现状 | 重构后 |
|---|---|---|
| 每样本文件打开 | ≤33 次（1 pkl + ≤32 npy） | 1 次（特征走常驻 fd） |
| 每样本读盘 | 均值 19.08 MB（上界 19.69 MB），只用 3.95 MB | 均值 2.43 MB（上界 2.49 MB），几乎全用到 |
| 反序列化 | ≤32 次全量 pickle | 0 次（裸字节直读） |
| 线程池 | 每样本新建 ≤32 线程 | 无 |
| padding dtype | 隐式 float64 提升 | 显式 bf16/f32（native 模式） |
| collate 后 batch 载荷（memory 三键＋两张原图） | ~757 MB/batch（98.4% 的 batch；三键 740 MB，image 单键 537 MB） | ~257 MB/batch（三键 236 MB，image 单键 134 MB） |
| 每 step 读盘 / 打开 | 1.22 GB / 2,112 次 | 均值 155 MB（上界 159.5 MB）/ 64 次 |
| 单样本耗时（热/冷） | 25.4 / 132 ms | ≈7 / 15–40 ms（预估，S8 实测为准） |
| 供给余量（vs 计算需求） | 不足（GPU 空转 30%） | ≈15× |
| 预期步时 / epoch | 6.933 s / 11.9 h | ≈4.9–5.0 s / 8.4–8.6 h（下界 4.778 s / 8.2 h） |

### 2.4 接口与开关（新增，给人看的版本；细节在 B.2/B.4）

- **backend 显式三态**：`MMEVLA_DATA_BACKEND ∈ {packed, legacy, auto}`。`packed`＝新链路，meta 缺失/损坏/指纹不符**直接报错，绝不静默回退**；`legacy`＝旧链路逐字不动；`auto`＝按 meta 存在性分派并打 WARNING，**仅限本机探索且必须显式设置才生效**。**环境变量未设置时默认 `legacy`——与现状行为逐字节相同，零静默切换。**正式 launcher 一律显式 `packed` 或 `legacy`——「按目录内容猜路径」被彻底禁止。
- **双根契约**：打包库、源库（pkl 所在）、清单三个位置全部显式：`store_meta.json` 记录 `source_dataset_root` 与 `manifest_path`（绝对路径），可被环境变量覆盖（节点本地盘缓存场景：store 在 `/tmp`，pkl 与清单仍在 NFS）。
- **store 生命周期**：常驻 fd 与两张小表**不跨进程携带**——Dataset 被 pickle 进 spawn worker 时剔除句柄，worker 内首次取数时按 pid 懒构造（小表 `np.fromfile` 全量读入进程内存，44 MB/worker）。这是「0 次 open」承诺的实现路径（v1 计划的空白，对抗验证 A2/Codex 阻断 2 命中）。

## 三、「memory token 近乎一致」怎么保证（结论：受控环境下可做到逐位一致）

> **有效性域**：本节的「逐位一致」成立于 C.0 固定的受控 XLA 环境（deterministic flags / 共用编译缓存）。生产默认 autotune 下的残差属 3.3 所述**既有** XLA 非确定性——旧链路自身重跑同样不 bitwise——其量级由 3.4 的量化判据兜底，不因本重构而变化。

### 3.1 恒等链：memory token 由四个因素完全决定，逐一钉死

每个 step 的 memory token 由且仅由四件事决定。前三件重构后**构造性不变**，第四件（dtype）有变化但**已实测证明不改数**：

1. **这个 step 取了哪 64 个样本。** 同一迭代器生命周期内，torch 的 index 序列只由 `(len(dataset), seed, batch_size, drop_last, shuffle)` 决定、与 num_workers 无关；跨 epoch 的 num_workers 相关性是 torch 既有语义（见 1.6），恒等链只依赖「新旧链路在相同 num_workers 下序列相同」。重构不换 `TorchDataLoader`、不换 generator、不换 seed 语义，且 `len` 相同（395,289）→ 同 workers 档位下序列逐位不变。
2. **每个样本选了哪 32 帧。** `even_sampling_indices(step_idx, 32)` 是纯函数、零随机源；新链路 **import 同一个函数**而非重写；`step_idx` 取自 pkl 且与清单推导值互相校验（不一致显式 raise）。
3. **每帧特征的字节。** 打包库的每一行：写侧与源 npy 逐帧校验 + 写后读回 + part sha256，**g 级身份由独立进程全量 verify（483,291 帧全部与源库对拍）零遗漏钉死**（见 A.2）→ 新链路读进内存的数组与旧链路逐位相同。
4. **这些字节以什么 dtype 走到 GPU encoder 输入。** 这是唯一有变化的环节，论证链（全部已在真 GPU 实测）：
   - 存储值本身是 bf16（image）与 f32（pos）；bf16→f32→f64 都是**精确升位**，往返无损（对全部 65,536 个 bf16 位型验证过位模式 100% 复原）。
   - **现状交付本来就不统一**：98.4% 的 batch 因 f64 padding 事故以 float64 到 host、host 侧 cast 成 f32 上卡；1.6% 的「整批满长」batch 以 bf16 上卡——即现状自己就有两种 dtype 路径并存，模型对两者一视同仁。
   - 模型侧第一层 `nnx.Linear` 显式 `dtype=bfloat16`，flax 的 `promote_dtype` 在做任何算术之前把输入统一转成 bf16；bf16 的值经任何精确升位再转回 bf16 必然复原 → **三种交付（bf16/f32/f64）进 `pos_proj`/`encoder_static` 的实际张量逐位相同**。已用真实形状实测：三种输入的 memory token 输出全等（max 差 0.0）。
   - 因此 native bf16 **不是引入新行为，而是把现状 1.6% batch 的行为推广到 100%**，顺带把 XLA 的两份编译产物合成一份。

### 3.2 不靠论证靠梯子：四层验证，每层有硬判据

论证再严密也可能被没想到的环节推翻，所以每一段恒等链都配一层可证伪的验证（细节在第二部分 C 节）：

| 层 | 证明什么 | 怎么证 | 判据 |
|---|---|---|---|
| 第 0 层 | 恒等链 (1)：样本序列 | 新旧 loader 同 seed dump index 序列对拍（w0/w4/w8 三档，dump 步数 < 1 个 epoch）+ 真实训练链路旁证（主进程 batch_sampler 层记录） | diff 为空 |
| 第 1 层 | 恒等链 (2)(3)：交付内容 | 8,200 个**定点**样本（step∈{0,1,2,29,30,31,32,33} 边界全覆盖 + 每 episode 首样本 + 随机）逐键对拍，另加 200 个真实 batch 过 collate 对拍 | replica 模式全键逐位零容差；native 模式 astype(f32) 后逐位 |
| 第 2 层 | **IO 重构本身对训练零影响** | 同一 clean HEAD 下 legacy backend vs packed+replica：本机 2 卡 b8 跑 300 步，逐步比 loss/grad_norm 等五个量的 hex ＋ 每 25 步完整 TrainState 摘要 | 全部 bitwise 相同 |
| 第 3 层 | **dtype 正规化不改数** | 先单步定点梯度对拍（**主判据取「含短样本」batch——唯一有 dtype 差异的场景；「整批满长」batch 作阴性对照**），再 replica vs native 300 步；另加 GL b64 100 步短程抽查 | 主判据 bitwise（本机 2 卡 b8）；GL b64 抽查只判 3.4 量化判据 |

说明三点：

- **replica 复刻模式**是专为第 2 层设计的交付模式：精确复现现状的 f64 padding 与 collate 提升，使新旧链路的差异只剩「字节从哪读」——第 2 层通过即证明重构没改任何东西；第 3 层再单独隔离 dtype 这一个变量。
- **dtype 差异场景与档位选择（v2 修订，方向纠正）**：replica 与 native 的交付 dtype 差异只出现在**含短样本**的 batch（replica f64 vs native bf16）；「整批满长」batch 两种模式本就同为 bf16，比对它零证伪力、只能当阴性对照。含短样本 batch 占比 **b64 = 98.4%、b8 = 40.4%**——单 batch 暴露率 **b64 比 b8 高约 2.4 倍**，b8 不构成更严的检验（v1 计划「b8 严 38 倍」为方向性错误，已废弃）。b8 的真实价值：本机 2 卡唯一可跑档位、迭代快，且 300 步内期望命中差异场景约 121 次，配合单步定点对拍足以证伪；b64 规模的直接证据由第 3 层新增的 GL 短程抽查补齐（C.4）。
- **本机 2×RTX 6000 Ada 与 GL 4×A40 是两种硬件**：第 2/3 层的 bitwise 结论在本机受控环境内自洽；GL 侧不重复 bitwise 证明（无稳定基线，见 3.3），由 GL b64 短程抽查 + 3.4 量化判据覆盖——**GL 抽查不承担 bitwise 举证，只做量级复核**。

### 3.3 前置条件：先证明「同配置重跑本身可复现」

2026-08-24 实测：同配置同 seed 重跑两轮（两轮间仅有 bench 记录层的 `_WandbProxy` 改动，不进计算图，见 `docs/training-doc/v1-2gpu-epoch-bench-b8/result.md`），参数校验和逐步全不相同——**当前默认设置下训练不是 bitwise 确定的**。根因已定位：bench 驱动脚本 run 名写死（跑不了第二轮）、结尾删 jax 编译缓存、`train.py` 硬编码覆盖缓存目录 → 每轮空缓存 → XLA 每轮重新 autotune、可能选中不同 kernel。这个问题不修，任何 A/B 的差异都无法归因给 dataloader。

修法（已核实可行性）：驱动脚本拆分「实验名/轮次名」并保留缓存；两轮**共用同一份 jax 编译缓存**（jax 0.5.3 的持久编译缓存默认托管 XLA per-fusion autotune 缓存，共用目录即复用 autotune 结果）；必要时经 `XLA_FLAGS` 加 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`（六个相关 flag 已在本仓库 venv 的 XLA 插件二进制里逐个确认存在，jax 0.5.3 无对应 config 开关，只能走 `XLA_FLAGS`）。**缓存目录按 AGENTS 14 收敛进 `v1-store/cache/jax/`**（软链方案，见 C.0，不动 `train.py` 也不覆盖 `HOME`），验证收官后统一清理。先跑 D0/D1/D2 三档「同配置重跑两轮」实验，**两轮逐步校验和完全一致后**，才开始 3.2 的任何 A/B。

### 3.4 任何一层失败怎么办（「梯度差距极小」的硬兜底）

- **定位手段**：第 0/1 层失败输出首个失配 idx/键/元素的 hex，配合守卫测试把问题缩小到 gather/padding/reshape/normalize 四段之一；第 0 层若差异恰从 epoch 边界开始 → 先检查是否跨 epoch（1.6 的 torch 既有语义），非 Dataset 问题。第 2/3 层失败用参数 sha256 逐叶子二分找首个分叉模块——分叉在 `mem_enc*` 指向交付内容（回第 1 层），分叉在 LLM 主干而 mem 一致指向非确定性（回 3.3 重立前提）。
- **量化兜底**（参数化判据全文见 C.4）：若第 3 层 bitwise 失败，先证 native 自身重跑稳定，再启用量化判据——loss 逐步相对差 median ≤ 1e-6 / p95 ≤ 1e-5 / max ≤ 1e-4，梯度范数三项 median/p95 各松一档、max 与 loss 同为 1e-4，且相对差对 step 的回归斜率不得显著为正（轨迹不允许发散，无论绝对值多小）。精度参照：bf16 在 [1,2) 区间 1 ULP = 2⁻⁷ ≈ 0.78%（半 ULP ≈ 0.39%）——median 档比 1 ULP 保守 2.9–3.9 个数量级、p95 档 1.9–2.9 个数量级、max 档约 1.9 个数量级（异常值兜底）。
- **降级路径**：量化判据仍不达标，正式模式从 native 降级 f32 或 replica（三模式开关收敛在一处，随时可切），**绝不「差不多就行」**。

## 四、明确不做的事，与本重构解耦的既有问题

### 4.1 本轮明确不做

- **不打包 pkl**（每 step 仅 25.3 MB、2.7 ms/样本，且打包无体积收益；按同一套 part 机制预留 Phase C 接口，若实测它成为新墙再单独一轮）。
- **不预烘焙 ResizeImages、不预 tokenize prompt**（tokenize 实测 24 µs/次；烘焙需 119 GB 存储且要先证 jax CPU/GPU resize 逐位一致）。但**记录一条重要观测**：每个 dataloader worker 会在 GPU 0 建 442 MiB CUDA context（16 workers ≈ 7 GB 显存 + 上下文抢占），本轮 bench 里采样存证，作为下一轮「worker 去 JAX 化」的立项依据。
- **不重建数据集、不动源库、不动旧多分支 Dataset 代码**（symbolic/recurrent/token_drop 原地保留不惊动）。
- **不给 `TorchDataLoader` 加 `prefetch_factor` 形参**（v1 计划的「可选加性改动」与「不动 `src/openpi/**`」红线矛盾，本轮裁定：红线保持、该改动不实施，列为未来备选）。

### 4.2 与本重构解耦的既有问题（本轮不修，如实声明，处置须用户单独拍板）

1. **`scripts/train.py` 正式入口双次 `main()`**：尾部先跑 `main(tentative_run=True)` 再跑正式 `main()`；`initialize_checkpoint_dir` 默认 `overwrite=False, resume=False` 且目录已存在即 `FileExistsError`——**全新 run_name 下第二次 `main()` 必然报错**（tentative 已建目录），除非启动方显式传 `--overwrite`/`--resume`（`finetune_mme_vla_suite.sh` 均未传）。v1 系列全部 e2e 走 `bench_train_steps.py`（单次 main）从未触发。**本轮交付口径**：吞吐验收与等价性验证继续走 bench 入口；「用 `scripts/train.py` 起正式长训练」在该问题修复前不纳入本轮交付声明，修复方案（改 `train.py` 或规约启动参数）单独一轮向用户拍板。
2. **checkpoint 只保存 `assets`/`params`**（`train_state` handler 被注释）：训练中断后恢复不保证 optimizer/EMA/step 状态连续，「中断恢复的轨迹连续性」不在本轮任何判据之内。
3. **`jax.process_count() > 1` 明确不支持**（`TorchDataLoader` 直接 raise）：本计划全部内容仅覆盖单进程多 GPU。

## 五、可读性产出

- `docs/v1-framesamp-dataflow.md`：一页式数据流图表，从 H5 → 清单 → 源库 → 打包库 → Store → Dataset → transforms → collate → device_put → memory token，每一跳标形状/dtype/字节数/「这一跳有没有改数」。
- 格式契约 README（datastore 层）、打包工具 README、构建留档、每个 run 的 launch/result 留档。
- **探针脚本固化**：v1 计划里标「已实测」的数字（npy 内部偏移、pos 纯函数、promote_dtype 逐位一致等）当时是一次性交互式探针、未留档（对抗验证已独立复现关键项：偏移 9/9 精确吻合、bf16 全位型往返无损）。S2 把这批探针固化为 `scripts/data-pack-framesamp/probe_layout.py` 等小脚本，数字在构建留档附录记录命令与原始输出。
- 新模块的形制断言本身就是文档：读代码即知适用域，超出即炸。

---

# 第二部分（技术细节，供 agent 追踪）

## A. 打包特征库

### A.1 目录布局

```
v1-store/datasets/4task-gl-framesamp/
├── meta/
│   ├── store_meta.json          # 唯一契约（见下）
│   ├── pack.lock                # 排他锁（打包期存在；记 build_uuid/host/pid）
│   ├── pack_progress.jsonl      # 断点续跑：每 part 一行（idx, rows, sha256, elapsed, host, pid）
│   └── row_digests.blake2b.bin  # 逐行独立摘要（483,291 × 16 B；verify 全量对拍时产出，必产出）
├── image_emb_4x4/part_000.bf16.bin … part_031.bf16.bin
│                                # 前 31 个 ≈990–1020 MB、末 1 个 ≈621 MB（9,471 行），
│                                # (rows,16,2048) bf16 裸字节
├── pos_emb_4x4.f32.bin          # (586,16,768) f32 = 28.8 MB，按 step_idx 索引
└── state_emb.f32.bin            # (483291,8) f32 = 15.5 MB，按全局行号索引
```

- **行号公式**（写读两侧共用同一函数，物理上不可分叉）：`row(g,t) = manifest.episodes[g].total_sample_offset + t`，其中 t 是**全 timestep 域**的帧号（含 demo 前缀）。执行样本 idx → (g,t) 的换算见 B.3（必须带 `exec_start_idx`）。
- **part 切法**：按 `global_episode_idx` 升序累积 `num_timesteps`，累计 ≥ `ceil(483291/32)=15103` 即切，**切点必在 episode 边界**（一个样本的 32 帧必在同一 part）。对真实 manifest 模拟切分恰得 32 个 part；末 part 覆盖 episodes[1573..1599]、9,471 行、620.7 MB（尾部效应，读侧按 meta 的精确边界走，不依赖「约 990 MB」这个描述）。边界写入 `store_meta.json.parts[i]`。单文件方案与「每 episode 一文件」方案均已否决（前者与原子写+并行互斥，后者退化回每样本一次 open）。
- **bf16 落盘定论**（已实测）：裸 `.bin` + meta 声明 dtype。**禁止 `.npy`**——`np.save` 对 ml_dtypes bf16 写出 `V2` descr，`np.load` 丢类型。读侧 `np.memmap(dtype=ml_dtypes.bfloat16)` 与 `frombuffer(uint16).view(bfloat16)` 均实测可用。
- `store_meta.json` 关键字段：
  - 布局与格式契约：`layout="framesamp-4x4-v1"`、三张表的 shape/dtype/row_bytes、**`byte_order="little"`、`array_order="C"`、`bf16_encoding="ml_dtypes.bfloat16 (1s+8e+7m)"`**、`writer_versions{python,numpy,ml_dtypes,git_commit}`；
  - 身份与双根：`manifest_sha256`（须等于当前 `episode_manifest.json` 顶层 `sha256`）、**`manifest_path`、`source_dataset_root`**（绝对路径，运行期可被环境变量覆盖，见 B.4）、`source_provenance_sha256`、`source_spot_sha256`（16 个抽样源文件摘要，读侧启动抽验）；
  - 规模与校验：`num_rows/num_exec_samples/num_pos_rows`、`parts[]`（含每 part 精确行边界、sha256 与 **`head_tail_digest`**——blake2b-128，覆盖该 part 首尾各 1 MiB；part 小于 2 MiB 时覆盖全文件并标 `full_covered: true`，供 fast 档不读全 part 复验）、`packer`（host/reader/校验覆盖率）、`verify`（全量对拍的 seed/时间/结论）。
  - **meta 两阶段写，均为 tmp + fsync + replace 原子落盘（v3 定稿）**：阶段 1 `pack` 结束写 meta（`status: "packed"`、`verify: null`）——这是「打包完成」的标志；阶段 2 `verify` 通过后原子回填（`status: "verified"`）。verify 期间**继续持有 `pack.lock`**，回填完成后才删锁。**verify 闸只落在 `create_data_loader` 的 packed 分派层**（`status != "verified"` 且未显式设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1` 即 raise，放行必打 WARNING，见 R17/G14）；`FrameSampStore` 本身不看 `verify` 字段——否则打包工具读不了自己正在验的库。provenance 统一记**回填后**的 `store_meta.json` sha256（该 sha 只在 verify 通过后才稳定）。
  - **迷你库契约（守卫/开发专用，v3 新增）**：`store_meta` 增 `manifest_scope ∈ {"full","subset"}`；`subset` 时必须带 `subset_episodes[]`（`global_episode_idx` 列表）与 `mini_manifest_sha256`，fast 档的覆盖校验相应改为「`parts` 连续覆盖 subset 声明的行区间」，`manifest_sha256` 仍记全量清单值但只作溯源、不作相等断言。**`manifest_scope=="subset"` 的库禁止用于 S5 及以上任何判据**（packed 分派检出即 raise）。

### A.2 打包工具

新目录 `scripts/data-pack-framesamp/`：`pack_framesamp_store.py`（子命令 `plan | pack | verify | report`）+ `run_pack.sh`（tmux 驱动，PYTHONUNBUFFERED=1 + pipefail + tee + EXIT_CODE=）+ `probe_layout.py`（探针固化）+ `README.md`。

- **真值与复用**：清单经 `scan_manifest.load_manifest()`（sha256 fail-loud）；格式常量与 `row_of()` 从 `src/mme_vla_suite/datastore/framesamp_store.py` import，**绝不复制**（沿用 `build_shard.py` 从 `scan_manifest.py` import 的既定做法）。
- **运行位置**：**本机 detached tmux**（纯 CPU+NFS，不占 GL 的 spgpu GPU 配额；产物是确定性字节、由全量对拍背书，不属于「本机吞吐结论」，不违反 AGENTS 13）。带宽受限（NFS 供给 398–628 MB/s、decode 总读 291 GB），预计 **20–40 min**。
- **并行与事务协议**（v2 补全，Codex 阻断 7）：
  1. **排他锁**：启动时 `O_CREAT|O_EXCL` 创建 `meta/pack.lock`（内容 build_uuid/host/pid/开始时间）。锁已存在时：pid 同 host 且存活 → 拒跑；否则视为残锁，`--resume` 显式确认后接管并换新 build_uuid。**双实例并发被此闸排除**；锁在全量 verify 回填 meta 后才释放（见 A.1 两阶段协议）。
  2. **小表先行、主进程独写**：`pos_emb_4x4.f32.bin` 与 `state_emb.f32.bin` 由主进程在并行阶段**之前**单独构建、校验、原子落盘（pos 表从源库抽取拼装，见下）；worker 只读它们做逐帧比对，全程无人再写小表——「谁写全局表」的归属唯一。
  3. **image part 并行**：`multiprocessing.Pool`（默认 `min(16, cpu)`），**每 part 唯一属主**（按 part 划分任务），天然无锁。
  4. **progress 单写**：worker 完成一个 part 后经 `multiprocessing` 队列把 `(idx, rows, sha256, elapsed)` 汇报给父进程，**只有父进程追加 `pack_progress.jsonl`**——排除多进程交错写坏行。读侧解析时对**尾部半行**（崩溃残留）直接丢弃，不视为损坏。
  5. **写序与持久化**：episode slab 写入 `part_XXX.bf16.bin.tmp` → `os.fsync(fd)` → 全 part 完成后 sha256 → `os.replace` → 对目录 fd `fsync`（NFS 上保证重命名可见性）→ 汇报父进程记 progress。**meta 最后写**（同样 tmp+fsync+replace）。
  6. **崩溃语义**：SIGKILL/断电/ENOSPC 后重跑 `--resume`：按 progress 校验「存在+大小+sha256」跳过完好 part；`.tmp` 残留一律清除重做；ENOSPC 在写入前有 `df` 预检（源目录所在文件系统余量 ≥ 40 GB）。
- **源读取两档** `--reader`：`decode`（首跑默认，逐帧 `np.load(allow_pickle).item()` 全量 602,951 B，零布局假设，总读 291 GB）；`slice`（已实测 npy 内部偏移恒定：`image_emb_4x4`@262,595、`pos_emb_4x4`@541,352、`state_emb`@602,906，120/120 文件大小一致、60/60 memcmp 通过，本次对抗验证又独立复现 9/9；三重守卫：st_size==602,951、数据段前 64 B 前缀与基准逐字节相同、逐帧 pos 窗口 100% memcmp——留作重跑加速档）。偏移常数是**数据格式常量**（非代码行号），由 `probe_layout.py` 可随时复核。
- **写侧逐帧校验（100% 覆盖，写入路径内；口径按 v2 修正——钉 t 不钉 g）**：① 该帧 `pos_emb_4x4` ≟ `pos_table[t]`（memcmp，**钉死 t 与「pos 只依赖 t」两条不变量；不钉 g**——pos 是 t 的纯函数，数学上不可能区分「同 t 不同 episode」的调包）；② `state_emb` ≟ state 表同一行（同源自证，防行内错乱，同样不构成独立 g 证据）；③ episode slab 写入 `.tmp` 后 `os.pread` 读回与内存 memcmp（read-after-write，多读 31.7 GB ≈ 80 s）；④ part 完成 → sha256 → `os.replace` → 汇报 progress。
- **g 级身份的唯一凭据：`verify` 子命令全量对拍（v2 从 5 万帧抽样升级为全量，Codex 阻断 6 / 对抗验证 A3）**：独立于 pack 的一次后验遍历，覆盖**全部 483,291 个 (g,t)**——重新完整 decode 源 npy，与 `FrameSampStore.read_image_rows()` 走真实读路径 memcmp，同时逐行产出 blake2b-128 摘要。**进程模型（v3 定稿）**：复用与 pack 同一套 `multiprocessing.Pool(min(16, cpu))`、按 part 划分任务；每个 worker 只返回本 part 的（起始行号、逐行摘要字节块、mismatch 列表），**由父进程按 part 序拼接后单写** `meta/row_digests.blake2b.bin`（同样 tmp → fsync → replace → 目录 fd fsync；worker 不写任何文件）。总读 ≈291 GB（源）+ 31.7 GB（store），**16 进程预计 20–40 min**（单进程口径按实测 decode 冷 12.3 ms/帧外推约 1.7 h，不采用）。判定行 `VERIFY_PACK=PASS scanned=483291 mismatches=0`。**「零遗漏」「逐位」只在全量 verify 通过后才允许宣称**；`--sample N` 抽样档仅供开发期快检（10% 抽样对单行错位漏检率约 90%，不得用于交付判定）。verify 通过后按 A.1 两阶段协议原子回填 `store_meta.json`。
- **pos 表来源（定论）**：**主方案从源库抽取拼装**（用若干 episode 凑齐 t=0..585 的 `pos_emb_4x4`，逐位同源、零后端风险），主 pass 的 100% memcmp 即证明「只依赖 t」。`PosEmb3D` 现生成仅作旁证：**已实测 CPU 后端生成与库中值不逐位一致（max|diff| ≈ 7e-7，落在 1e-7~1e-5 区间），GPU 后端一致**——若走生成路径必须校验 `jax.devices()[0].platform == "gpu"`（不符 raise），守卫测试钉死（G7）。
- **留档**：`docs/dataset-build-doc/4task-gl-framesamp/README.md`（AGENTS 12：commit、命令、源库指纹、耗时、写侧校验与全量 verify 结果、探针脚本输出附录）。

### A.3 pkl 侧：本轮不打包（定论）

量级不构成瓶颈（25.3 MB/step、2.7 ms/样本）；打包无体积收益（定宽后仍 156 GB）；源库 `data/` 依红线必须保留可读。**Phase C 预案**（不实施，仅预留）：同 part 机制加 `samples/part_XXX.bin` 定宽记录 + `strings.json` 字典编码（实测 prompt 全集封闭，400 样本仅 27 个不同值）；`actions` 保 f64、`state` 保 f32，否则 DeltaActions/Normalize 数值会变。

## B. 新 Dataset / 装配路径

### B.0 全局逐文件变更表（v2 新增，Codex 高 16：范围一表定死，超出即越界）

| 类别 | 文件 | 变更 |
|---|---|---|
| 新增·格式层 | `src/mme_vla_suite/datastore/__init__.py`、`datastore/framesamp_store.py`、`datastore/README.md` | 常量 + StoreMeta + FrameSampStore(只读) + row_of()；不 import 任何 training/model 模块 |
| 新增·装配层 | `src/mme_vla_suite/training/framesamp_dataset.py` | FrameSampDataset |
| 新增·工具 | `scripts/data-pack-framesamp/{pack_framesamp_store.py, run_pack.sh, probe_layout.py, dump_index_seq.py, compare_batches.py, test_pack_guards.py, README.md}` | 打包/探针/对拍/守卫 |
| 修改·接线 | `src/mme_vla_suite/training/dataloader.py` | `create_data_loader` 内 backend 分派（约 15 行） |
| 修改·验证资产 | `scripts/smoke-local/run_2gpu_epoch_bench.sh` | S0：EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE、缓存软链进 v1-store、XLA_FLAGS 注入、preflight 兼容 packed 库（`stats.json` **或** `store_meta.json`）、env.json provenance 字段扩展（清单与 D 节一致） |
| 修改·验证资产 | `scripts/smoke-local/bench_train_steps.py` | S0：checksum recorder 扩展（每 SAVE_INTERVAL 全 TrainState 摘要）；`BENCH_DUMP_IDX` 改 batch_sampler 层记录（monkeypatch `create_data_loader` 取 loader + `object.__setattr__` 安装包装器，首次 `iter()` 前——torch 禁止初始化后直接赋值） |
| 修改·验证资产 | `scripts/smoke-local/README.md` | 同步 S0 用法 |
| 修改·验收资产 | `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | S7.5：`--dataset-path`/backend/dtype/`--save-interval`（默认 1000＝现状，S7 抽查传 25）/`--time` 参数化（默认值=现状）、新增 `COLDHOT=1` 双跑模式（同 allocation 先 C1 后 H1，`--time=04:00:00`）、meminfo 采样、provenance 字段扩展 |
| 修改·验收资产 | `scripts/bottleneck-bench/gl-dataloader/{gl_dlbench_single,gl_dataloader_bench}.sbatch` | S7.5：`DATASET_PATH`/`MMEVLA_DATA_BACKEND`/dtype 参数化（默认值＝现状 `4task-gl`+`legacy`），env.json 同步；`submit_split_jobs.sh` 的 `--export` 同步补项 |
| 修改·验收资产 | `scripts/bottleneck-bench/gl-dataloader/dataloader_bench.py` | S7.5：`_AVG_BYTES_PER_SAMPLE` 从 history_config+manifest 推导、`block_until_ready` 覆盖整个 (obs, actions) pytree、gather/pkl 分段计时 |
| 修改·验收资产 | `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | S7.5：主判据表 5 项机器判定输出 `E2E_ACCEPT=PASS|FAIL`（FAIL 退出码非零）；每步读盘公式口径去掉硬编码 1.20 GB，改从 history_config+manifest 现场推导（与 `_AVG_BYTES_PER_SAMPLE` 同源）；另加吃多个 record_dir 的附加判据汇总入口 |
| 新增·文档 | `docs/dataset-build-doc/4task-gl-framesamp/README.md`、`docs/v2-dataloader-restructure-report.md`、`docs/v1-framesamp-dataflow.md` | 留档与汇总 |
| **不动** | `scripts/train.py`、`src/openpi/**`、`src/mme_vla_suite/models/**`、`training/dataset.py`、`shared/**` | 硬红线（G 节 R2）；v1 的 `prefetch_factor` 可选项已裁定本轮不实施 |

### B.1 依赖方向

格式层（datastore）不 import 任何 training/model 模块（单向依赖）。修改类改动全部是「默认行为逐字节不变」的加性参数化（bench/sbatch 的默认值即现状值），验证/验收资产的改动不影响训练语义。

### B.2 `FrameSampStore`（格式层）——含 spawn 生命周期契约（v2 补全）

- **构造与 pickle 契约（对抗验证 A2 / Codex 阻断 2 的修法，定论：懒加载）**：
  - `FrameSampDataset.__init__`（主进程）只做：读 `store_meta.json` + 全部 fail-loud 静态校验（fast 档）+ 清单派生查表数组；**不打开任何 part fd、不建任何 mmap**。
  - `FrameSampDataset.__getstate__` 剔除 `_store` 句柄字段（只序列化路径/meta/查表数组）——Dataset 被 pickle 进 spawn worker 时**不携带任何内核资源**。
  - **每进程首次 `__getitem__` 时懒构造** `FrameSampStore`：`os.open` 32 个 part fd（`O_RDONLY|O_CLOEXEC`）+ 两张小表 `np.fromfile` 全量读入，记录 `_owner_pid = os.getpid()`；此后每次取数校验 `os.getpid() == _owner_pid`，不符（异常的二次 fork 等）即丢弃重建。`close()` 幂等。w0（num_workers=0）路径同样适用（主进程即 owner）。
  - 该契约保证：worker 各自持有效 fd；两张小表为进程内常驻副本（44 MB/worker），热路径零 NFS 缺页。
- **校验档位** `MMEVLA_FRAMESAMP_VERIFY ∈ {fast, full}`（Codex 高 4）：
  - `fast`（默认，每进程懒构造时执行）：layout / `manifest_sha256` 现场重算比对 / parts 连续覆盖 [0,num_rows)（subset 库按声明行区间，见 A.1 迷你库契约）/ 每 part 存在且 `st_size == meta.bytes` / 抽 1 个 part 头尾 1 MiB 与 meta `parts[].head_tail_digest` 复验 / 抽 1 条 `source_spot_sha256` 复验源库未动。
  - `full`（正式 GL run 由 sbatch 显式设置）：主进程在构造 DataLoader **之前**额外做**全部 32 个 part + 两张小表的完整 sha256** 对 meta 比对（≈31.7 GB 读，~1–2 min），能抓「同尺寸中部翻转」；worker 内仍只跑 fast + `fstat` 尺寸复核（避免 N 个 worker 各读 31.7 GB）。校验结果写入 run 的 env.json（provenance，见 D 节）。
- **大表用 pread、两张小表在懒构造时一次性 `np.fromfile` 全量读入进程内存**（v3 修正：pos 28.8 MB + state 15.5 MB = 44.3 MB/worker，16 worker ≈ 700 MB，远小于 native 模式省下的在途内存 ~16 GB；实测同一 NFS 上 32×64 KiB 常开 fd pread 0.33 ms、`np.memmap` 切片 2.4–2.5 ms 走缺页/revalidate 路径——**热路径不留任何 NFS mmap**）：`read_image_rows(rows, out)` 先对全部行发 `posix_fadvise(POSIX_FADV_WILLNEED)` 触发内核并发预读（`ENOSYS`/`EOPNOTSUPP` 时打一次 WARNING 后永久跳过，纯性能 hint），再按连续行游程合并 `os.preadv` 直读进预分配 bf16 数组（短样本 32 行天然连续 → 1 次调用）。
- **短读处理（v2 修正，B3/Codex 高 6）**：`preadv` 返回字节数不足**不立即判损坏**——从已读偏移继续补读，连续 3 次零进展才 raise；**读到 0 字节（EOF）或请求区间越出 part 边界立即 raise**（这才是完整性判据）。本仓库 NFS4.2 `hard` 挂载实测 320 次 2 MB 单调用零短读，该循环是稳健性兜底而非常态路径。
- **fail-open 禁令**：packed 模式下任何校验不过**直接 raise，绝不回退散 npy**。

### B.3 `FrameSampDataset.__getitem__`（伪代码）

```python
def __init__(...):   # 形制断言即文档（v2 补全，Codex 高 3——必须能挡住同形的 modul 配置）：
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
                     # dtype_mode ∈ {"native","f32","replica"} → self._pad 三选一（唯一开关点）
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
    img, pos, stt, mask = self._pad(img, pos, stt, n)             # dtype 三模式的唯一分叉点
    data["static_image_emb"] = img.reshape(-1,2048)               # (512,2048)
    data["static_pos_emb"]   = pos.reshape(-1,768)
    data["static_state_emb"] = self._normalize_state(np.repeat(stt,16,axis=0))  # 保留精确计算（f64，与现状同源同式）
    data["static_mask"]      = np.repeat(mask,16)
    for k in _NONE_KEYS: data.setdefault(k, None)                 # recur_* / subgoal 等下游会索引的空键
    return data
```

- **`_pad_native`**：按最终形状一次性 `np.empty` 分配（img bf16 / pos f32 / stt f32），填充区清零，全程零 concatenate。**`_pad_f32`**：同上 img 改 f32（精确升位）。**`_pad_replica`**：直接调用原 `right_padding_token_emb`（短样本触发 f64 提升、满长纯切片保持 bf16——与现状分支行为逐位复刻，含 n==32 不走 concatenate 这一点）。
- **`_collate_fn` 一行不改**：replica 下逐样本 dtype 与现状一致，`np.stack` 提升行为自然复现；native/f32 下 batch 内 dtype 一致无提升。
- **static_state_emb 定论：保留精确计算不置零**——`HistAugObservation` 虽允许 None，但置 None 改变 jit 输入 pytree 结构（多一份编译 + 与在线评估路径分叉），而精确计算成本 <0.05 ms、1–2 MB/batch。`_normalize_state` 因 norm stats q01/q99 为 f64，输出恒 f64，与现状逐位同。
- **dtype 三模式帐**（实测；口径显式标注——三个总数是 **batch 载荷（memory 三键＋两张原图 19 MB）**）：replica ~757 MB/batch（三键 740 MB，image 单键 537 MB，同现状）→ native **~257 MB**（三键 236 MB，image 单键 134 MB）→ f32 ~391 MB（三键 370 MB）；collate 52→19/27 ms；device_put 73→23/38 ms。f64 降精度已核实发生在 host 侧——执行者是 `jax/_src/interpreters/xla.py::_canonicalize_ndarray_dtype`（`np.asarray(x, canonicalize_dtype(x.dtype))`，经 `pxla.shard_args` 调用，先于数据交给 PJRT；v1 计划所写「jaxlib `Squash64BitTypes`」为机制误标，已订正）；XLA 编译产物从 2 份（dtype 随 batch 摆动）→ 1 份。**正式默认 native（用户已拍板）**，replica 仅用于 A/B，f32 作保守回退。逐位相同论证链已实测：`nnx.Linear(dtype=bf16)` 的 `promote_dtype` 使三种交付进 `pos_proj`/`encoder_static` 的实际张量完全相同。

### B.4 接线（v2 重写：backend 显式开关 + 双根契约）

`create_data_loader`（`mme_vla_suite/training/dataloader.py`）按 **`MMEVLA_DATA_BACKEND ∈ {packed, legacy, auto}`** 分派（Codex 阻断 5——「按 meta 是否存在自动分派」与 fail-loud 红线矛盾，已废弃为唯一机制）：

- **`packed`**：`dataset_path` 必须是打包库根；`meta/store_meta.json` 缺失、损坏、指纹不符、未通过 verify（`status != "verified"` 且未设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1`，见 A.1/R17）、`manifest_scope=="subset"`（S5 及以上）→ **直接 raise，绝不回退**（与 G4/G14 守卫一致）。构造 `FrameSampDataset`。
- **`legacy`**：走原 `RoboMMEDataset`（旧路径逐字未动），`dataset_path` 语义不变。
- **`auto`**（**必须显式设置才生效**，仅为本机探索保留）：按 `store_meta.json` 存在性分派并打 WARNING 标明所选 backend。**环境变量未设置时默认 `legacy`——与现状行为逐字节相同，零静默切换。正式 launcher（sbatch/bench 驱动/dataloader-only）一律显式设置 `packed` 或 `legacy`，不许依赖 auto（R16）。**
- **双根契约**（Codex 阻断 3）：packed 模式下三个位置全部显式解析——打包库根＝`dataset_path`；源库根（pkl）＝`MMEVLA_FRAMESAMP_SOURCE`（未设则取 `store_meta.source_dataset_root`）；清单＝`MMEVLA_FRAMESAMP_MANIFEST`（未设则取 `store_meta.manifest_path`）。解析结果三者全部写进 run 的 env.json（provenance）。**禁止从打包库目录名删除 `-framesamp` 之类的字符串变换推导源库。**
- dtype 模式取环境变量 `MMEVLA_FRAMESAMP_DTYPE`（默认 native，非法值 raise）。其余（`transform_dataset` + `TorchDataLoader` + `DataLoaderImpl`）一行不动——同 workers 档位下 index 序列逐位不变的构造性保证。
- **不新增 `_CONFIGS` 条目**（`assets_dirs = assets_base_dir / self.name`，换名会把 norm_stats 路径指飞）；**不新增 CLI 参数**（`bench_train_steps.py` 护栏保持零改动；backend/dtype/双根全走环境变量）。启动侧变化：sbatch 里 `--dataset-path …/4task-gl-framesamp` + `export MMEVLA_DATA_BACKEND=packed`（+ 可选 `MMEVLA_FRAMESAMP_DTYPE=replica`）。
- **节点本地盘拷贝开关** `MMEVLA_FRAMESAMP_LOCAL_CACHE`（默认关；**在 GL 上使用节点 `/tmp` 属 NFS 路径规约的例外，启用前须用户显式批准**）：sbatch 开头 `df` 守卫（≥40 GB）→ cp 31.7 GB 到 `/tmp/$SLURM_JOB_ID/`（~65 s，按 NFS 供给中位 ≈513 MB/s 折算的保守估计——顺序大文件读通常优于该聚合值）→ 逐 part sha256 校验（~32 s，不过即回退 NFS 直读并 WARNING）→ `dataset_path` 指本地、**pkl 与清单仍走 NFS（双根契约天然支持）**→ 退出 `trap` 清理 `/tmp/$SLURM_JOB_ID`。**前置**：先在一个 GL job 里 `df -h /tmp` 确认 spgpu 本地盘规格（greatlakes.md 未记载）。若 NFS 直读已 util≥95% 则保持关闭。

## C. 等价性验证（判据梯子，全部有判定行）

### C.0 前置：确定性前提确立（不过这关不开任何 A/B）

已核实根因链：`run_2gpu_epoch_bench.sh` run 名写死+记录目录存在即拒跑（跑不了第二轮）、结尾 `rm -rf ~/.cache/jax_*`、`scripts/train.py` 硬编码覆盖 `jax_compilation_cache_dir`——每轮空缓存 → XLA 每轮重新 autotune。已核实杠杆：jax 0.5.3 `jax_persistent_cache_enable_xla_caches` 默认值就是 `xla_gpu_per_fusion_autotune_cache_dir`，**共用编译缓存目录 = 同时复用编译产物与 autotune 结果**；六个 XLA flag（`--xla_gpu_deterministic_ops`、`--xla_gpu_exclude_nondeterministic_ops`、`--xla_gpu_autotune_level`、dump/load autotune、require_complete）已在 venv 的 XLA 插件二进制里逐个确认存在，jax 0.5.3 无对应 config 开关，**只能走 `XLA_FLAGS` 环境变量**。

- **S0 脚本改动**（验证资产，非训练代码）：
  1. `run_2gpu_epoch_bench.sh` 拆 `EXP_NAME`（决定编译缓存目录，A/B 共用）与 `RUN_TAG`（决定记录目录，A/B 各异）；加 `KEEP_JAX_CACHE=1`；`XLA_FLAGS` 外部注入并写进 env.json。
  2. **缓存目录收敛（v2 新增，A9：与 AGENTS 14 对齐，不动 `train.py`、不覆盖 `HOME`）**：脚本启动前 `mkdir -p v1-store/cache/jax/$EXP_NAME && ln -sfn <该目录> ~/.cache/jax_$EXP_NAME`——`train.py` 硬编码路径经软链落进 `v1-store/cache/jax/`；E 节收官步骤统一清理软链与缓存目录。
  3. **摘要口径扩展（v2 新增，Codex 高 12）**：checksum recorder 除逐步标量 hex（loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm）外，每 `SAVE_INTERVAL` 记一次**完整 TrainState 摘要**——params、opt_state、EMA、step 全部叶子逐个 sha256 汇成 `state_digest`。判据措辞相应从「逐步 global_digest」改为「**逐步标量 hex + 每 25 步完整 TrainState 摘要**」。
  4. preflight 从「必须存在 `meta/stats.json`」改为「`meta/stats.json` **或** `meta/store_meta.json` 二选一」（否则 packed 库根本过不了启动检查——v1 计划漏改）。
  5. 同步更新 `scripts/smoke-local/README.md` 第二节。
- **S1 三档实验**（各两轮相同 run，100 步，SAVE_INTERVAL=10）：D0=现状删缓存（预期 FAIL，复现对照）；D1=共用缓存；D2=D1+`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`（期望 PASS）。判定行：两轮逐步标量 hex 与全部 `state_digest` diff 为空。取首个 PASS 档固定为 C.1–C.4 全部 A/B 的环境；D2 仍 FAIL 则加 exclude flag 并降 50 步二分（分清「第 1 步就不同」vs「第 k 步漂移」）。

### C.1 第 0 层：index 序列等价

已读码确认：**同一迭代器生命周期内** torch index 序列只由 `(len, seed, batch_size, drop_last, shuffle)` 决定、与 num_workers 无关；跨 epoch 因 `persistent_workers` 与 w0 消耗 generator 节奏不同而分叉（1.6，torch 既有语义）。perceptual 路径不消耗 Python RNG。

- 新工具 `dump_index_seq.py`（并入 `scripts/data-pack-framesamp/`）用探针数据集 + 同一 `TorchDataLoader` dump 序列，w0/w4/w8 三档 diff 为空。**约束：dump 步数必须 < 一个 epoch 的 batch 数**（探针数据集较小时尤其注意），否则 w0 与 w>0 在 epoch 边界后的分叉会制造假阳性。
- **端到端旁证（v3 按实测收紧实现与判据；原 v1「patch `_collate_fn` 记 `_probe_idx`」方案废弃——collate 在 worker 子进程执行、签名不含 idx）**：`BENCH_DUMP_IDX=1` 时，bench 侧参照 `scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py` 的做法 monkeypatch `mme_vla_suite.training.dataloader.create_data_loader` 取得 loader，在**首次 `iter()` 之前**执行 `object.__setattr__(loader._data_loader.torch_loader, "batch_sampler", _IdxProbe(orig))`——**必须绕过 `DataLoader.__setattr__` 的赋值守卫**（初始化后直接赋值会 `ValueError`，torch 2.7.1 实测；`object.__setattr__` 绕道后 `_index_sampler` 正确返回包装器，`persistent_workers` 下跨 epoch 持续生效）。`_IdxProbe` 需实现 `__iter__`（把每个 batch 的 idx 追加写 `$BENCH_RECORD_DIR/idx_seq.jsonl` 后原样 yield）与 `__len__`（`DataLoader.__len__` 走 `len(self._index_sampler)`）。交付内容零改动。**判据（v3 修正）**：batch_sampler 的枚举比交付**超前 `prefetch_factor × num_workers` 个 batch**（w4/pf2 实测超前 8），因此判据不是「diff 为空」，而是「`idx_seq.jsonl` 的**前 N 条**与第 0 层 dump 逐条相同，N＝实际消费步数；尾部允许至多 `prefetch_factor × num_workers` 条超前记录」。
- 失败定位：len 不等 → 误用 `total_samples`；序列不同且恰从 epoch 边界开始 → 跨 epoch 的 torch 既有语义（非 Dataset 问题）；其余 → 查 sampler/drop_last。

### C.2 第 1 层：样本/batch 内容等价

新工具 `compare_batches.py`：**import 复用** `compare_datasets.py` 的 `metrics()/grid_metrics()/Agg/_raw_bits()` 统计口径资产（**不改其本体**，其三层 --mode 语义不适用于 dataloader 对拍）。不走 create_data_loader，直接对**定点 idx 列表**逐样本对拍：

- 定点集（由清单精确构造，~8,200 个）：step_idx∈{0,1,2,29,30}（触发 f64）各 200 + {31,32,33} 各 200 + 每 episode 首样本 1,600 + 固定 seed 均匀随机 5,000。**不依赖 shuffle 撞边界**（98.4% 只保证「至少一个短样本」，不保证覆盖 step=0/1/30）。注：Video* 任务首样本的 step_idx = exec_start_idx（66–216），恒为满长——f64 边界由 step 集合那一组（从 Button* 任务构造）负责覆盖，首样本组承担的是「每 episode 至少验一发」的覆盖职责。
- 判据：replica 模式全部键 shape/dtype/`view(uintN)` 零容差逐位；native/f32 模式 `astype(f32)` 后逐位相同 + dtype 差异逐键清单固定输出（image：短 f64→bf16、长 bf16→bf16；pos：短 f64→f32、长 f32→f32；state 恒 f64 不变；mask 与其余全部键位相同）。
- batch 级补充：用第 0 层 dump 的真实序列前 200 个 batch 过 `_collate_fn` 对拍（专验 collate 提升行为复刻）。
- 判定行 `COMPARE_BATCH=<mode> PASS samples=… batches=… mismatches=0`；失败输出首个失配 idx/键/元素 hex，配合守卫 G1–G3 缩小到 gather/padding/reshape/normalize 四段。

### C.3 第 2 层：300 步训练轨迹 bitwise（本机 2 卡 b8）

**A/B 同一 clean HEAD（v2 修正，Codex 高 13——不再跨 commit 对比）**：旧链路原地保留使这可行。A = `MMEVLA_DATA_BACKEND=legacy`（旧链路），B = `MMEVLA_DATA_BACKEND=packed` + `MMEVLA_FRAMESAMP_DTYPE=replica`。同 `EXP_NAME`（共用编译缓存；replica 下 dtype 序列相同 → HLO 相同 → 缓存命中）、同 XLA_FLAGS、同 seed 42、同 num_workers、b8、300 步、save-interval 25。判定：五个标量 hex 列 + 每 25 步 `state_digest` 逐步 diff 全空；前置断言：A/B 两侧 env.json 的 backend/dtype 与预期一致且 backend 为显式设置（R16）。失败定位：分叉叶子在 `mem_enc*` → 回第 1 层；在 LLM 主干而 mem 一致 → 非确定性回 C.0；再不然走 smoke-local README 第 3 级（固定 batch 单步逐元素梯度，固定 batch 直接用第 1 层落盘的 npz）。

### C.4 第 3 层：native 模式（主判据 bitwise；v2 按 A1 方向纠正重写）

1. **单步定点梯度对拍**（~5 min，最便宜先跑）：同一初始 state 各算一步逐元素比梯度，三种 batch——
   - **主判据：「含短样本」batch**（唯一有 dtype 差异的场景：replica f64 vs native bf16）——逐位相同即基本结案；
   - **补充：「全短样本」batch**（差异密度最大化）；
   - **阴性对照：「整批满长」batch**（两模式本就同为 bf16，必须逐位相同；若不同说明 gather/pad 另有 bug，与 dtype 无关）。
2. **300 步 B(replica) vs C(native)**：独立 EXP_NAME（dtype 变 → 缓存 key 必变），主判据逐步标量 hex + 每 25 步 `state_digest` bitwise。b8 下含短样本 batch 占 40.4%，300 步期望命中约 121 次，覆盖充分（b64 暴露率 98.4% 更高，但本机 2 卡只能 b8——v1「b8 严 38 倍」的说法方向反了，已废弃）。
3. **GL b64 短程抽查（v3 定稿：补齐 b64 规模的量级证据，不承担 bitwise 举证）**：在 GL 4×A40 的**默认 autotune 环境**下跑 replica vs native 各 100 步（b64，同 seed/workers，走 `gl_e2e_fix.sbatch` 参数化入口，传 `--save-interval 25`）。**判据只用第 4 条的量化判据**（rel 三档阈值 + OLS 趋势），**不判 bitwise**——GL 侧无确定性基线（3.3），非 bitwise 不构成证伪。若量化判据 FAIL，先在同一个 job 内重跑 native 两轮自比给出 GL 自身噪声底，再判定是否 dtype 问题。轻量（每 job ~15 min 计算 + 排队），置于 S7 内、S8b 正式 e2e 之前。判定前置断言同 C.3（env.json backend/dtype 显式且符合预期）。
4. **降级判据（v2 参数化，Codex 高 11；仅当主判据失败且先证 C==C' 重跑稳定后启用）**：
   - 相对差定义：`rel(a,b) = |a−b| / max(|a|,|b|, 1e-8)`，逐步计算；
   - loss：median ≤ 1e-6 / p95 ≤ 1e-5 / max ≤ 1e-4；
   - grad_norm、llm_grad_norm、mem_enc_norm：median ≤ 1e-5 / p95 ≤ 1e-4 / **max ≤ 1e-4**（v3 收紧：原 1e-3 距 1 ULP 不足一个数量级）；
   - 末步 param_norm：rel ≤ 1e-5；
   - **趋势判据**：对 (step, rel_loss) 做 OLS 回归，斜率 β>0 且 p ≤ 0.05 即 FAIL（单调上升无论多小判 FAIL）；β≤0 或 p>0.05 通过；
   - 精度参照：bf16 在 [1,2) 区间 1 ULP = 2⁻⁷ ≈ 0.78%（半 ULP ≈ 0.39%）；median 档比 1 ULP 保守 2.9–3.9 个数量级、p95 档 1.9–2.9 个数量级、max 档约 1.9 个数量级（异常值兜底）。
5. 若失败 → 正式模式降级 f32 或 replica，不「差不多就行」。

### C.5 守卫测试

新文件 `scripts/data-pack-framesamp/test_pack_guards.py`（**不混进** `data-preprocess-GL/test_guards.py`），照搬其「刻意制造失败断言亮红灯」风格，`JAX_PLATFORMS=cpu` pytest 秒级。**按 v2 顺序拆成两组、分属两个实施步**（Codex 高 15——原 G1–G9 一次性全绿存在循环依赖：G6/G8/G9 依赖 S3 才实现的 Dataset）：

**Store 组（S2 交付，只依赖格式层与打包工具）**：
- G1 迷你库（ref-shard 派生 3 帧）打包→读取逐位对拍；
- G4 meta 缺失 / manifest sha 不符 / offsets 不符三条各自 raise 且不回退散 npy；
- G5 blob 截短 1 字节启动即炸；同尺寸中部翻转由 full 校验档抓出；
- G7 CPU 后端生成 pos 表被拒（固化实测发现）；
- G11（v2 新增，A3）：构造「两个 episode 在同一 t 的帧互换」的迷你库，断言写侧校验抓不到（预期行为，钉死「pos memcmp 不钉 g」的边界认知）而 verify 全量对拍必须亮红灯；
- G12（v2 新增，B3）：mock 短读（首次返回一半字节），断言续读补齐且结果正确；mock EOF/越界，断言 raise；
- G14（v3 新增，S2）：`meta.status != "verified"` 时 packed 分派必 raise；设 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED=1` 后放行且必须打 WARNING。

**Dataset 组（S3 交付，依赖 FrameSampDataset）**：
- G2 dtype 边界钉死（step=30→replica f64 / step=31→bf16；所用 (g,t) 必须落在迷你库 subset 覆盖内，迷你库构造时按此挑 episode）；
- G3 选帧重复索引必须重复输出不去重；
- G6a（S3 交付）：换算公式单测——直接用**全量清单**构造查表数组（不构造 Dataset、不碰 store），断言 `len==395289` 且 VideoUnmask / VideoUnmaskSwap 各一个 episode 首样本 `_step_of == exec_start_idx`；G6b（S5 随第 0/1 层一起绿）：在全量打包库上构造 `FrameSampDataset` 复验同两条；
- G8 mock 线程池抛错证明已彻底移除；
- G9 `use_state_emb is False` 前提钉死；
- G10（v2 新增，A2；断言随 M12 更新）：spawn 一个子进程消费 Dataset，断言子进程内 store 懒构造（`_owner_pid == 子进程 pid`）、fd 有效可读、两张小表数组 `.nbytes` 与 meta 声明一致且 `.base is None`（进程内副本而非映射）、父进程 Dataset 无句柄泄漏；
- G13（v2 新增，Codex 高 3）：喂 `perceptual-framesamp-modul.yaml`（integration_type=modulation、memory_token_dim=1024，其余同形），断言 `__init__` raise。

## D. 吞吐验收（GL；dataloader-only 可在 S4 后先行，正式 e2e 依赖 S7 定稿 dtype——v2 修正依赖）

- **MB/s 新口径**：`dataloader_bench.py` 的 `_AVG_BYTES_PER_SAMPLE` 改为**从 history_config + `episode_manifest.json` 现场推导**（均值帧数 = Σ min(t+1,32)/395,289 = 30.996 → 均值 2.43 MB/样本；上界 2.49 MB；勿再写死，也勿只从 history_config 推 32 帧上界）；主判读换 mountstats `server_read`，新增 **majflt** 采样（页缺失口径；v3 后两张小表已改进程内常驻，majflt 主要反映源库 pkl 读与页回收，保留作冷/热证据链辅助量）。**`block_until_ready` 覆盖整个 `(obs, actions)` pytree**（v2 修正，Codex 高 8——原来只 block actions，obs 里的大 memory 张量可能尚未完成 H2D，低估 device_put 成本）。**新增 gather/pkl 分段计时**（每样本两段耗时直方图落 records——F 节风险 3「谁是新瓶颈」的观测资产，v1 计划有承诺无资产）。
- **dataloader-only 四档**（单 GPU job）：w2/w4/w8/w16，seed 310–313（避开已用 42/200–205/210–212 防 page cache 串扰）。
- **e2e 600 步**（`gl_e2e_fix.sbatch` **参数化后**入口，4×A40/16C/96G；v1 计划「sbatch 零改动」的说法废弃——它硬编码 `--dataset-path …/4task-gl` 于训练命令与 env.json 两处，必须参数化，默认值保持现状）：T1 w4（**最重要**：官方默认 workers 还需不需要调）→ T2 w8（直接对 v1-e2efix-w8c16）→ T3 w2（探底）；条件档 T4 w16、T5 w4c8（8C 直接对 v1-e2e-b64，「官方口径净收益」最干净对照）。
- **对照组（v3 更新，三档已全部落地）**：v1-e2e-b64（6.933 s / 69.7%）、v1-e2efix 三档——w8c16 **5.301 s / 71.2% / ≈9.09 h**、w12c16 **5.319 s / 70.6% / ≈9.13 h**、w16c16 **5.327 s / 67.1% / ≈9.14 h**（workers 8/12/16 曲线完全平坦，「只调参上限」＝三档最优 5.301 s）、compute-only 4.778 s。
- **冷/热（v2 方法修正，Codex 高 9：证据口径降为 cold-like）**：31.7 GB 打包库一个 epoch 内即可全驻 page cache，热态数字必然偏乐观；pkl 156 GB 仍是长期 NFS 流量来源。C1/H1 **各 300 步**（只取稳态窗口，与 T1–T3 的 600 步分开口径），**在同一个 allocation 内串行执行**（`COLDHOT=1` 双跑模式、`--time=04:00:00`；先 C1 后 H1，排除节点差异），共用**同一冻结 index 序列**（同 seed + 第 0 层 dump 存证）；`/proc/meminfo` Cached 15 s 采样落 `meminfo.csv` + **cgroup `memory.stat` 的 pgmajfault** 同步采样。「冷」无法严格证明（新 allocation 的节点也可能带缓存），结论一律称 **cold-like**，判据 `(C1稳态−H1稳态)/H1 ≤ 15%`。并行采 `nvidia-smi --query-compute-apps` 存证 worker CUDA context。
- **成功判据**（AGENTS 16 口径，禁中位数标题结论；**主判据表 5 项由 `analyze_gpu_util.py` 机器判定并输出单行 `E2E_ACCEPT=PASS|FAIL`（FAIL 退出码非零）；附加判据（w4/w8 步时差、majflt 趋势、server_read 区间）由一个吃多个 record_dir 的汇总脚本判定**——人工只做复核）：

| 指标 | v1 基线 | 必达 | 期望 | 下界 |
|---|---|---|---|---|
| 步时中位 | 6.933 s | ≤5.00 s | ≤4.95 s | 4.778 s |
| util 稳态均值 | 69.7% | ≥90% | ≥95% | — |
| 0% 采样占比 | 27.8% | ≤5% | ≤2% | — |
| 慢步(>8s)墙钟占比 | 32.9% | ≤5% | ≤2% | — |
| epoch(6,176 步) | 11.9 h | ≤8.6 h | ≤8.5 h | 8.2 h |

  必达 ≤5.00 s 有意严于「只调参上限」5.301 s（w8c16 实测）——低于它才证明重构有超出调参的净收益。附加判据：w4 与 w8 步时差 ≤3%（否则 CPU 侧仍未松绑）；**NFS server_read（e2e，T1–T3；v3 重定标）**：公式上界 ≈31–33 MB/s（＝155 MB/step ÷ 目标步时 4.778–5.00 s），历史实测普遍为公式口径的 0.54–0.71 倍，稳态实测期望 ≈17–25 MB/s，**>65 MB/s（≈2× 公式上界）即视为读放大信号**，处置（Codex 高 10）：image 大表在 fd 上以 `posix_fadvise(POSIX_FADV_RANDOM)` 替换 `WILLNEED` 对照重测（pread 路径的 readahead 由 fadvise 管）；热态（打包库全驻 page cache 后）只剩 pkl 流量 25.3 MB/step ≈5 MB/s 属正常；**dataloader-only（S8a）不套用本带**，按 samples/s × 2.43 MB 现场折算；majflt 随 epoch 单调下降（无小表 mmap 贡献后作辅助量）。
- **provenance（v2 新增，Codex 缺口 4；v3 扩展到所有 run——含本机 bench 与 dataloader-only）**：env.json 必记：resolved 后的 `dataset_path`/`source_dataset_root`/`manifest_path`、`store_meta.json` 的 sha256（verify 回填后口径）、`manifest_sha256`、backend 及其来源（显式设置 / auto 推断——**S5 及以上任何 run 出现 auto 推断即判该 run 无效**，R16）、dtype 模式、`MMEVLA_FRAMESAMP_VERIFY` 档位与 full 校验结果、`MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED` 取值与是否生效、local-cache 是否命中及其 sha 校验结果、XLA_FLAGS、git HEAD。
- 结果分析一律 `analyze_gpu_util.py`；每个 >5 min run 留档 `docs/training-doc/<run_name>/`（records 含 env.json/metrics/gpu_util_dense/nfs_read/meminfo.csv/pgmajfault/分段计时/param_checksums）。

## E. 实施顺序、提交切分与留档（v2 重排：mini 全链路先行，全量打包后置）

| 步 | 内容 | 依赖 | 判定 | 预计 |
|---|---|---|---|---|
| S0 | 修 bench 驱动（EXP_NAME/RUN_TAG 拆分、KEEP_JAX_CACHE+缓存软链进 v1-store、XLA_FLAGS 注入、TrainState 摘要扩展、preflight 兼容 packed、BENCH_DUMP_IDX 改 batch_sampler 层、env.json provenance 扩展） | — | STEPS=3 连跑两次不拒跑、缓存未删且落 v1-store/cache/jax/ | ~30 min |
| S1 | 确定性前提 D0/D1/D2（各两轮 100 步） | S0 | 两轮标量 hex + state_digest diff 空 + `docs/training-doc/<run_name>/` 留档 | ~2 h |
| S2 | 格式层 + 打包工具 + **Store 组守卫（G1/G4/G5/G7/G11/G12/G14）**，ref-shard 派生迷你库（subset 契约）全流程（含迷你库全量 verify） | — | Store 组 pytest 全绿 | ~2 h 开发 |
| S3 | FrameSampDataset + backend 分派接线 + **Dataset 组守卫（G2/G3/G6a/G8/G9/G10/G13）** + **迷你库上真实 spawn loader 矩阵：w0/w1/w4/w16 × 2 epoch**（fd 泄漏检查：前后 `ls /proc/<pid>/fd` 计数） | S2 | Dataset 组 pytest 全绿（G6 只跑 G6a）+ 矩阵无错无泄漏 | ~1.5 h |
| S4 | **全量打包**（本机 tmux，decode 档）+ **全量 verify（16 进程，483,291 帧对拍 + row_digests）** + 构建留档 | S3 | `VERIFY_PACK=PASS scanned=483291 mismatches=0` | 20–40 min + 20–40 min（均 16 进程） |
| S5 | 第 0/1 层（定点 8,200 样本 + 200 真实 batch + G6b；run_name `v1-framesamp-cmp-{replica,native,f32}`，留档 `docs/training-doc/`） | S4 | `COMPARE_BATCH=* PASS` + G6b 绿 | 30–60 min |
| S6 | 第 2 层：同 HEAD legacy vs packed+replica 300 步 bitwise | S1+S5 | 标量 hex + state_digest diff 空 + 留档 | ~1 h |
| S7 | 第 3 层：单步定点（含短样本主判据/全短/满长阴性对照）+ 300 步 replica vs native + **GL b64 100 步抽查** | S6 | 本机 bitwise PASS（或 C.4.4 降级判据）+ GL 抽查量化判据 PASS + 留档 | ~1.5 h + GL 排队 |
| S7.5 | 验收资产参数化（gl_e2e_fix.sbatch、gl-dataloader 两个 sbatch、dataloader_bench.py、analyze_gpu_util.py，默认值＝现状） | S4 | 三个 launcher 用默认值跑通、env.json 记到 backend/dtype/resolved 双根 | ~40 min |
| S8a | GL dataloader-only 四档（w2/w4/w8/w16） | S7.5 | 吞吐数据落档（env.json backend==packed 且显式设置） | 15 min×4 + 排队 |
| S8b | GL e2e 600 步 T1–T3(+条件档) + cold-like/hot（COLDHOT 双跑各 300 步） | **S7**+S7.5（dtype 定稿后才跑正式 e2e——若 S7 降级，先前 e2e 数字不代表交付版本） | `E2E_ACCEPT=PASS` | 3×2 h + 1×4 h（C1+H1 同 job） |

> 工时注：摘要开销按实测 47.3 s/次（110 叶子，params+EMA）为基准，扩完整 TrainState（含 opt_state）后按 2–3× 估——S1/S6/S7 的预计已含该开销。

- **commit 切分**（沿用 `commitV<大>.<小>:` 中文体例，dataloader 重构起 V2 系列）：V2.1 bench 驱动改造解锁 A/B → V2.2 确定性前提确立 → V2.3 格式层+打包工具+Store 守卫（迷你库通过）→ V2.4 新 Dataset+backend 接线+Dataset 守卫+spawn 矩阵 → docs 打包留档（S4 后）→ V2.5 第 0/1 层通过 → V2.6 第 2 层逐位一致 → V2.7 第 3 层 native 定稿 → V2.8 验收资产参数化（S7.5）→ docs GL 验收留档 + `docs/v1-framesamp-dataflow.md` 定稿。每 commit 可独立回滚。
- **run_name 建议**（起跑前逐个交用户确认，AGENTS 6）：`v1-framesamp-det-d{0,1,2}-r{1,2}`、`v1-framesamp-cmp-{replica,native,f32}`（S5）、`v1-framesamp-ab-{legacy,replica,native}`（S6/S7）、`v1-framesamp-b64chk-{replica,native}`（S7 GL 抽查）、`v1-framesamp-dl-w{2,4,8,16}`、`v1-framesamp-e2e-w{4,8,2}c16`、`…-coldlike/-hot`；打包库名 `4task-gl-framesamp`。
- **回滚策略（v2 修正，Codex 缺口 5——不再是「删目录」）**：功能回滚＝launcher 里 `MMEVLA_DATA_BACKEND` 切回 `legacy` + `--dataset-path` 指回源库 + 撤 dtype 环境变量（三件事必须一起回退）；打包库保留作证据不删（不进 git，占 31.7 GB）。只有确认彻底放弃该方案时才删库目录。
- **收官清理**：验证全部结束后清理 `v1-store/cache/jax/` 下各 EXP_NAME 缓存与 `~/.cache/jax_*` 软链（A9 的对账义务），并清理 S 步产生的临时 run（AGENTS 6）。
- 汇总报告：新增 `docs/v2-dataloader-restructure-report.md`；`docs/v1-nfs-bottleneck-analysis.md` 只加指针不改结论。

## F. 风险 Top3 与规避（v2 按 A3 修正防线口径）

1. **行号错位（静默错帧，loss 只会慢慢变差不报错）**——防线分工如实陈述：写侧 100% pos memcmp 钉死 **t** 与帧序（不钉 g）；写读共用 `row_of()` 排除公式分叉；**g 级身份唯一凭据是 verify 全量对拍（483,291 帧零遗漏）+ row_digests 逐行摘要**；运行时逐样本 pkl 身份校验（显式 raise）钉每次访问。G11 守卫保证「换帧攻击」在 verify 层必然亮红灯。
2. **native 逐位结论被上游推翻**（flax promote_dtype 语义变更 / 有人改 `Pi0Config.dtype`）——梯子 C.3/C.4 是可证伪的硬验收而非「跑一下看看」；失败即降级 f32/replica，留档。
3. **page cache 假象与 pkl 新墙**——头条结论用 cold-like 口径（同 allocation 串行 + pgmajfault 证据链）；bench 分段打点 gather/pkl 各自耗时直接看谁是新瓶颈；pkl 若成墙走已预留的 Phase C；worker 在途内存 native 模式从 ~24 GB 降到 ~8 GB（batch 载荷口径 × 16 worker × prefetch 2；v1 OOM 的根源之一顺带缓解）。

## G. 红线清单（实施期逐条自检；v2 显式编号，杜绝「红线 N」引用漂移）

| # | 红线 |
|---|---|
| R1 | 训练循环/模型/超参/seed 零改动 |
| R2 | **`scripts/train.py`、`src/openpi/**`、`models/**`、`training/dataset.py`、`shared/**` 不动**（B.0 表为唯一授权范围；bench/sbatch/analyzer 等验证验收资产的参数化改动除外且默认值必须等价现状） |
| R3 | 同 workers 档位下 index 序列构造性不变 |
| R4 | `even_sampling_indices` 复用不重写 |
| R5 | 4task-gl 只读；新库旁路新增+原子写+provenance+fail-loud；packed 模式绝不回退散 npy |
| R6 | 身份只从 `episode_manifest.json`；换算必须带 `exec_start_idx`；身份校验用显式 raise 不用 assert |
| R7 | 旧分支代码不惊动；旧链路原地保留，不安排删除 legacy 的 commit |
| R8 | 禁复活 d951aef |
| R9 | uv 纪律（`uv run`、`UV_LINK_MODE=copy`） |
| R10 | >5 min 任务 tmux+tee+EXIT_CODE、Monitor 每级行缓冲 |
| R11 | GPU util 判读 AGENTS 16 口径（机器判定 `E2E_ACCEPT` 为准） |
| R12 | 正式 run clean HEAD 起跑+run_name 用户确认；**S1 起**凡预计或实际 >5 min 的诊断/基准/等价性 run（含 S1 六轮、S6/S7 全部 A/B、S8a/S8b）一律在 `docs/training-doc/<run_name>/` 留档（AGENTS 17）；S4 全量打包走 `docs/dataset-build-doc/` |
| R13 | commit 逐文件 add、中文 body 详写过程 |
| R14 | 缓存类目录收敛 `v1-store/cache/`（含 jax 编译缓存软链）；禁止覆盖 `HOME`；收官清理 |
| R15 | GL 节点 `/tmp` 本地缓存属规约例外，启用前须用户显式批准；退出 trap 清理 |
| R16 | 正式 launcher（sbatch/bench 驱动/dataloader-only）必须**显式**设置 `MMEVLA_DATA_BACKEND`（未设置默认 legacy）；env.json 标明该值是显式设置还是 auto 推断，S5 及以上任何 run 出现 auto 推断即判该 run 无效 |
| R17 | S5 及以上与全部 GL 验收禁止设置 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED`；该开关仅迷你库/开发期可用 |

---

## 修订记录（v1 → v2，2026-08-25）

依据：①对抗验证 workflow 报告 `v1-framesamp-restructure-adversarial-review.md`（编号 A1–A16、B2–B4、B6）；②Codex 审计（阻断 1–8、高 1–16、缺口 1–8）。两源重叠项合并列出。

| 修订 | 来源 | 落点 |
|---|---|---|
| dtype 差异场景方向纠正：差异在含短样本 batch（b64 98.4% / b8 40.4%），「b8 严 38 倍」废弃；C.4 主判据改含短样本 batch、满长作阴性对照；新增 GL b64 100 步抽查 | A1 / Codex 阻断 1 | 3.2、C.4、S7 |
| FrameSampStore 懒加载生命周期契约（`__getstate__` 剔句柄、per-pid 构造、`_owner_pid`、幂等 close）+ G10 守卫 + w0/w1/w4/w16×2 epoch spawn 矩阵 | A2 / Codex 阻断 2 | 2.4、B.2、C.5、S3 |
| 「pos memcmp 钉 (g,t)」降格为「钉 t 不钉 g」；verify 从 5 万帧抽样升级为全量 483,291 帧对拍 + row_digests 必产出 + G11 守卫；F 节防线口径改写 | A3 / Codex 阻断 6 | A.2、C.5、F |
| 「不动 openpi」与 `prefetch_factor` 矛盾裁定：红线保持、该改动本轮不实施；红线补进 G 节（R2）并加 B.0 变更总表 | A4 / Codex 高 16 | 4.1、B.0、G |
| 均值/上界口径分列（2.43/2.49 MB·样本，155/159.5 MB·step；memory 三键 757→257 MB 与 image 单键 537→134 MB 分开标注） | A5 / Codex 缺口 2 | 2.3、B.3 |
| index 序列「与 num_workers 无关」限定到单 epoch；C.1 加 dump<1 epoch 约束与 epoch 边界定位项 | A6 / Codex 高 1 | 1.6、3.1、C.1 |
| collate 位置修正（worker 内执行）；idx 旁证从 patch `_collate_fn` 改为主进程 batch_sampler 包装 | A7 / Codex 高 7 | 1.3、C.1 |
| w8c16 已落地数字（5.301 s / 71.2% / 9.09 h）写入 Context 与 D 节对照组；必达 ≤5.00 s 与之对照说明 | A8 | Context、D |
| jax 编译缓存经软链收敛 `v1-store/cache/jax/`，收官清理入 E 节，红线 R14 | A9 | C.0、E、G |
| S5 补 run_name（`v1-framesamp-cmp-*`）与 training-doc 留档 | A10 | E |
| pkl 大小改「约 395.4–395.6 KB 非定长」；manifest 顶层字段名订正为 `sha256`；5.4×→5.3×（两处）；`Squash64BitTypes`→`xla.py::_canonicalize_ndarray_dtype`；「同 commit 重跑」订正为「同配置（记录层改动不进计算图）」；「红线 7」引用错位由 G 节显式编号根治 | A11–A16 | 1.2、1.4、1.5、3.3、B.3、G |
| 「逐位一致」有效性域限定（受控 XLA 环境）；生产 autotune 残差归入既有非确定性、由量化判据兜底 | B2 | 三节引言、3.2 |
| preadv 短读循环补齐、EOF/越界才 raise + G12 守卫；fadvise 失败一次告警后跳过 | B3 / Codex 高 6 | B.2、C.5 |
| part 尺寸如实标注：31 个 ≈990–1020 MB + 末 1 个 ≈621 MB | B4 | 2.1、A.1 |
| 「已实测」数字固化：probe_layout.py + 构建留档附录 | B6 | 五、A.2 |
| 双根契约（packed/source/manifest 三位置显式，env 可覆盖，禁目录名推导）；local-cache 场景 pkl 走 NFS | Codex 阻断 3 | 2.4、B.4 |
| launcher 现状不兼容修复入表：bench preflight 兼容 packed、sbatch dataset_path/backend 参数化、「sbatch 零改动」说法废弃 | Codex 阻断 4 | B.0、C.0、D |
| backend 显式三态 `MMEVLA_DATA_BACKEND`（packed 严格 fail-loud / legacy / auto 仅本机），正式 launcher 禁用 auto | Codex 阻断 5 | 2.4、B.4 |
| pack 事务协议：pack.lock 排他 + build_uuid、小表主进程先行独写、progress 父进程单写+残行容忍、fsync→replace→目录 fsync、ENOSPC/崩溃/resume 语义 | Codex 阻断 7 | A.2 |
| train.py 双 main、checkpoint 仅存 params、process_count>1 三项列为「解耦的既有问题」，交付口径如实声明 | Codex 阻断 8、缺口 6、7 | 4.2 |
| idx→(g,t) 公式显式化（带 `exec_start_idx`）+ G6 扩展 Video* 首样本校验 | Codex 高 2 | B.3、C.5 |
| 形制断言补 `integration_type=="context"`、`memory_token_dim==2048`、feature dims + G13 守卫 | Codex 高 3 | B.3、C.5 |
| 校验档位 fast/full（正式 run 主进程全量 sha、worker fstat） | Codex 高 4 | B.2 |
| 身份校验 assert→显式 raise | Codex 高 5 | B.3、R6 |
| dataloader_bench：block 整个 pytree、均值帧数从 manifest 推导、gather/pkl 分段计时 | Codex 高 8、缺口 2、8 | D |
| cold/hot 改同 allocation 串行 + 冻结序列 + pgmajfault，结论称 cold-like | Codex 高 9 | D |
| readahead 处置改 image fd 的 `POSIX_FADV_RANDOM` 对照（madvise 只管 mmap 小表） | Codex 高 10 | D |
| 降级判据参数化（rel 定义、eps、各阈值、OLS 斜率+p 值、bf16 ULP 订正 0.78%） | Codex 高 11 | C.4、3.4 |
| 摘要口径扩展：逐步标量 hex + 每 25 步完整 TrainState（params/opt_state/EMA/step）摘要 | Codex 高 12 | C.0、C.3 |
| C.3 A/B 改同一 clean HEAD（legacy vs packed+replica），不跨 commit | Codex 高 13 | C.3 |
| S8 拆 S8a/S8b：正式 e2e 依赖 S7 定稿 dtype | Codex 高 14 | E |
| 守卫拆 Store 组（S2）/Dataset 组（S3），mini 全链路（含 spawn 矩阵）通过后才全量打包 | Codex 高 15 | C.5、E |
| bin 契约补字节序/C-order/bf16 编码/writer 版本 | Codex 缺口 1 | A.1 |
| local `/tmp` 例外须用户批准 + trap 清理 + 失败回退 NFS（R15） | Codex 缺口 3 | B.4、G |
| provenance 字段清单（resolved 路径、meta/manifest sha、backend、dtype、校验结果、local-cache 命中） | Codex 缺口 4 | D |
| 回滚改「切 backend 三件套、保留打包库」，不删目录 | Codex 缺口 5 | E |
| `E2E_ACCEPT=PASS|FAIL` 机器判定（analyzer 阈值代码化，FAIL 非零退出） | Codex 建议 9 | B.0、D |

### v2 → v3（2026-08-25，定稿）

依据：定稿复核 workflow（全 opus，6 agent；裁决清单必须修 M1–M13 / 建议修 S1–S9 均已落实；R1–R6 驳回维持原文）。

| 修订 | 落点 |
|---|---|
| M1 GL b64 抽查判据统一为量化判据（默认 autotune 环境、不承担 bitwise 举证、FAIL 先测 GL 噪声底）；save-interval/--time 参数化入 B.0 | 3.2、C.4、B.0、E |
| M2 idx 旁证实现收紧：monkeypatch `create_data_loader` 取 loader + `object.__setattr__` 绕 torch 赋值守卫；判据改「前 N 条一致 + 允许 prefetch_factor×num_workers 条超前」（实测超前 8） | C.1、B.0 |
| M3 迷你库 subset 契约（manifest_scope/subset_episodes/mini_manifest_sha256，禁用于 S5+）；G6 拆 G6a（S3，公式单测）/G6b（S5，全量库复验） | A.1、B.2、C.5、E |
| M4 全量 verify 进程模型定稿：16 进程 Pool 按 part 分任务、父进程按序单写 row_digests（单进程实测外推 1.7 h，不采用） | A.2、2.2、E |
| M5 留档下界改「S1 起」（AGENTS 17；S1 单轮实测口径 ≥12 min） | G(R12)、E |
| M6 legacy 列字节帐重算：均值 19.08 / 上界 19.69 MB、用 3.95/4.07 MB、放大 ≈4.8×、1.22 GB/step、需求 256 MB/s；1.3 图 gather 段 19.3/3.7 MB；全表统一十进制 MB 加注 | Context、1.3、1.4、2.3 |
| M7 server_read 判据重定标：公式上界 ≈31–33 MB/s、实测期望 17–25、>65 报警；热态 pkl ≈5 MB/s 属正常；S8a 不套用；analyzer 公式口径去硬编码 1.20 GB | D、B.0 |
| M8 B.0 补 gl-dataloader 两个 sbatch 行与 env.json provenance；新增 S7.5 步与 V2.8 commit；provenance 扩展到所有 run；C.3/C.4/S8a 补 backend/dtype 显式性前置断言 | B.0、E、D、C.3、C.4 |
| M9 w12c16（5.319 s/70.6%/9.13 h）与 w16c16（5.327 s/67.1%/9.14 h）已落地：Context 与 D 对照组三档齐全，「落地后补入」删除 | Context、D |
| M10 `auto` 不再是默认值：未设置默认 `legacy`（与现状逐字节相同）；R16 显式 backend 红线 + auto 推断判 run 无效 | 2.4、B.4、G、D |
| M11 量化判据 max 档收紧：梯度三项 max 1e-3→1e-4；「保守两个数量级」宣称改为分档如实表述 | 3.4、C.4 |
| M12 两张小表弃 mmap，懒构造时 `np.fromfile` 全量读入（44.3 MB/worker）；删「零副本/共享 page cache」表述；G10 断言与 majflt 口径同步更新 | 2.3、2.4、B.2、C.5、D |
| M13 cold-like 双跑定稿：C1/H1 各 300 步同 sbatch 串行（COLDHOT=1、--time=04:00:00）；S8b 预计改 3×2 h + 1×4 h | D、B.0、E |
| S1 S1/S6/S7 工时上调（checksum 实测 47.3 s/次，扩 TrainState 后 2–3×，E 表加注） | E |
| S2 `MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED` 开关补 R17 红线、G14 守卫、provenance 必记 | G、C.5、D、A.1 |
| S3 store_meta 两阶段写（status packed→verified 原子回填）、pack.lock 持有至回填、verify 闸只落分派层、provenance 记回填后 sha | A.1、A.2、B.4 |
| S4 `parts[].head_tail_digest` 字段定义（blake2b-128 首尾各 1 MiB，<2 MiB 全覆盖标注） | A.1、B.2 |
| S5 「机器执行全部阈值」收窄为主判据表 5 项；附加判据走多 record_dir 汇总脚本 | D、B.0 |
| S6 757/257/391 标签订正为 batch 载荷（三键 740/236/370 + 原图 19 MB）；F 节口径注明 | 1.3、1.4、2.3、B.3、F |
| S7 1.4 标题移除对审计报告条目编号「B.6」的正文交叉引用 | 1.4 |
| S8 文首 Codex 计数订正为 8 阻断 / 16 高风险 / 8 缺口 | 文首 |
| S9 2.3 预期步时改区间 ≈4.9–5.0 s / 8.4–8.6 h | 2.3 |

驳回维持原文：R1（GL 补确定性档）、R2（改 757/257/391 数值）、R3（verify 单进程选项）、R4（小表坚持 mmap）、R5（≈4.9 s 判不自洽）、R6（旁证「同样不可实施」论）。
