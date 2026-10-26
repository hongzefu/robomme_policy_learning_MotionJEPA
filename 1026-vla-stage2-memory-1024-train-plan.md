# 1026-vla-stage2-memory-1024-train-plan.md — 用阶段 III 最后 ckpt（EMA）把每窗 4×256 拼成 1 个 1024 维 memory token，窗 33／步长 32，训一条 mixed8 VLA 80k

> **权威与授权**：只规划不实施。本版 2026-10-26 美东按用户退回意见（「你这个计划是什么情况？为什么写成现在这么乱七八糟的？我在这个仓库里面有 agent markdown，要求了你要第一部分、第二部分」）推翻草稿结构、按 `AGENTS.md` 第 2 条重写为两部分；r2 按合并前审查 5 条 finding、用户新增裁决与 8 卡测速实测修订。用户要求 Markdown（「你写一个 Markdown 的计划，然后我确认了再开始。」），本次保留 `.md`。口径以本文为准。
> **代码锚点**：policy `v4-256concat1@3861154`（从 `v3-motionmem-pair4@8c319aa` 切出；用户：「然后这次的这个修改从branch v4-256concat1开始」），工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`；MotionJEPA `v12-3decoder@6c79ebb`，工作副本 `/scratch/hongze/MotionJEPA`（只新增一个模块＋一个 CLI 脚本，另开分支 `v12-3decoder-memexport`，不动 `v12-3decoder`／`main`）。
> **commit 体例**：两仓功能改动 `commitV<x>.<y>: <中文>`（policy 接续 `commitV11.37` → 本轮从 `commitV11.38` 起），文档 `docs:`，交接 `handoff:`；子代理提交 `sub/<编号>: <中文>`，经主会话 `git merge --no-ff <TIP sha>` 合入，合并提交占项目号。
> **外部依赖锚点**：Wan VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，`vae_state_sha256` 前缀 `9980d252`（恢复副本 metadata 原值）；恢复副本 `v1-store/restore/mixed8-200-vla-latents-20261021-a/`，来源 bucket `HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1`，`restore_receipt.json` PASS；encoder ckpt `/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`（sha256 在 G3 现场算并钉进 yaml）。
> **待用户裁决**：只剩开工令（只覆盖主 run；备选 run B 另取开工令）。bucket 名已定死（用户：「计划写在根目录，bucket 随便你定，以后不要再来问了。」）。开工令后按第二部分二节分配表派发；开工前先派一个 sonnet 审本计划自相矛盾（`PLAN_CONSISTENCY_REVIEW=`）。

# 第一部分（给人看）

## 一、要做什么与全部运行一览

**一句话**：用 MotionJEPA 阶段 III 最后一份 ckpt（ep105）的 EMA encoder，把 mixed8-200 每个 33 帧窗的 4×256 输出按源时间片 [0,3,5,8] 拼成 1 个 1024 维 token，步长从 16 改为 32、demo 尾窗补帧规则与原来一样、motion budget 96，离线建新 motion 库，训一条 8 卡 80k VLA（`mme_vla_suite_b128_80k`，测试直接取 60k）；边训边传 VLA ckpt，encoder／decoder ckpt 同期上传。不重过 Wan、不跑 4 token 对照臂；「帧路 1 帧＋motion」只作备选 run B 写进计划，不排进本次运行。

```
 开工 ─┬─ 改代码  S1(MJ 抽取) ∥ S2(库与 loader) ∥ S3(yaml/启动/导出) → S1 合并 push → S4(sidecar)；每块审→--no-ff 合→合后审→push
       ├─ S3 合入后即起（不占卡，与抽取并行）：encoder/decoder ckpt 上传 cc-vla-s2-mjup → G10
       ├─ 串行：抽取(GPU0, 读恢复副本 g 偶数窗) → G1 G2 G3 → 打包 → G5 G6 → 在线对拍 G4 → 短测 G7 → Beta commit
       └─ 正式 80k（cc-vla-s2-train，8 卡）∥ 增量上传（cc-vla-s2-inc，60k 一出即传）→ G8 → 全量回读 → G9
```

| 运行 | 内容 | 资源 | tmux | 判据 |
|---|---|---|---|---|
| 抽取＋打包 | MJ `scripts/stage2_memory_export.py` 读 4,995 段恢复副本只取偶数格窗；`pack_motion_store.py pack/verify` 写 `motion-stage2-1024/` | GPU0，约 6.4 万窗；打包 CPU | `cc-vla-s2-export`、`cc-vla-s2-pack` | G1–G3、G5、G6 |
| 短测 | 同配置 `MAX_STEPS=20` | 8 卡 ≤10 min | `cc-vla-s2-smoke` | G7 |
| 正式训练 | run `mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k` | 8 卡 fsdp 8、batch 128 | `cc-vla-s2-train` | G8 |
| VLA 上传 | `m8v-ckpt-inc.sh` → `HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1` | CPU/网络 | `cc-vla-s2-inc` | G9 |
| MJ 上传 | 阶段 II＋III ckpt → `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1` | CPU/网络 | `cc-vla-s2-mjup` | G10 |
| 备选 B（不在本次） | 帧路 1 帧／demo 任务 2 帧＋同一 motion 路，run `mixed8-200-m1x8x8-modul-motion-stage2-1024-b128-80k` | 8 卡 | — | G11 |

**测速实测（2026-10-26 美东 12:43 起跑，`cc-speed8-motion-b128-20261026-a`，现有 motion 160×768 配置，8 卡 fsdp 8、batch 128）**：编译约 2 min，第 12→111 步 99 步 95.3 s ＝ **0.96 s/步**，对 4 卡 1.73 s/步加速 1.80×（未到 2×，差 10%）；60k ≈ 16.0 h、80k ≈ 21.4 h。**开工令后 ETA（T0＝开工令）**：准备（改代码、抽取、打包、对拍、短测）3–4 h → 约 T0+4 h 起跑 → 约 T0+20 h 到 60k（测试取这一份）→ 约 T0+25.5 h 到 80k（稳态步时推算，不含每 5000 步存盘停顿与首次编译）。

**用户原话（逐字，2026-10-26）**：①「给出方案 用现在的最后一个ckpt训vla 4个256合并成1个token输入 然后窗口是32 不要再16重叠了 给出方案 然后注意 encoder inference 的时候别忘了 dropout 关掉 然后这个memory本身是不是不再做增强了 robomme官方policy repo怎么做的 然后再看一下wan是不是已经有缓存了」②「然后你一边训练，一边还要把现有的这个 encoder、decoder 的这个 checkpoint 是不是看一下是不是都上传了，然后边训练边上传 VIA 的 checkpoint。」③「你写一个 Markdown 的计划，然后我确认了再开始。」④「你写在 policy learning 的仓库里面，不要写在 motion Java 仓库，因为这个完全是 policy learning 仓库的事情。」⑤「然后这个 memory token 的预算是是九十六还是八十？然后我我然后我确认是用 EMA 权重，然后不跑这个不拼接的这个作为四个 token 的这么一个对照。」⑥「好的，先定为这个九十六，然后你要看一下现在这个 execution，就是 Robo M E 官方仓库的这个 evaluate 是最多是一千三百步，我记得是，然后这一千三百步会怎么影响我们的这个 evaluate 的情况？」⑦「然后你这个 padding 的这个工作还是要做的，就是 demo 的需要有 padding，还是要有一样的。」「你超过十六帧还是十七帧要有一个 padding 的这个，你也要写在这个计划里面。」⑧「计划写在根目录，bucket 随便你定，以后不要再来问了。」⑨「然后这次的这个修改从branch v4-256concat1开始」⑩「现在这个八十 K step 对应的是多少个小时？」「如果我只要这个六十 K 的这个位置，那么八卡会慢多少？然后四四卡会是什么情况？训训练我要训到八十 K，但是我测试我直接拿60k你什么时候能得到？」⑪「然后你可以现在就做一些简短的测试，然后把速度测测速的问题先告诉我，在 work tree 里面去实现。」⑫「然后这个巴卡的收益是多少？就是巴卡的收益达到两倍了吗？差了多少？」⑬「你再测一下，如果我同意把 batch size 拉满翻倍，就是会是什么样的情况？」「巴卡一二八现在这样可以解锁，你不用再测这个 batch 二五六了。然后这个再加一条 run，只作为备选，就先8卡跑。」「现在这个速度可以接受。」⑭「然后我之前说的这个 demo padding 的这个问题，就是填充填充帧的问题，你也要说一下，确保它实现了。我说的 fable 化的这个图是指的这个 memory token 没有满九十六的情况下，它是怎么影响计算的？」⑮「demo 补帧这个机制，不需要直接画，不需要画图，你只需要说一下它和原来是一样的就可以了。然后简单画一个数轴就可以，不用画 SVG」⑯「然后我还需要再加一个，再加一条跑，就是现在是八帧的八乘八的这个呃情况。然后我们相当于把这个八帧降到只有一帧。如果是 demo 的话，有 demo 的任务的话就是两帧。」

**已定口径**：
1. 权重 = ep105 的 `model` 键（`weight_type='ema'`），不用 `model_live`（原话⑤；2.1）。
2. 4×256 按源时间片 [0,3,5,8] 顺序拼接成 (1024,)，不平均；`memory_token_dim=1024` 不变（原话①；2.2）。
3. 窗 33 帧（锚点＋未来 32）、`GRID_STRIDE=32`、`segment_start`、`forward`，相邻窗只共享 1 帧（原话①；2.3）。
4. demo 末窗真实帧 ≥17 保留并 `repeat_last` 补到 33、≤16 丢弃，exec 33 帧全真实，与原来一样（原话⑦⑭⑮；2.3）。
5. motion budget 96（原话⑥；2.6）；不跑 4 token 对照臂（原话⑤）；不满 96 的补位不参与注意力（原话⑭；2.7）。
6. 抽取与 sidecar 一律 `eval()`＋`no_grad()`＋fp32（autocast 关），G1 证明 dropout 真关（原话①；2.4）。
7. memory 不增强，用干净 latent（等价 aug_id=0、`geometry_valid` 全 True）（原话①；2.4）。
8. 不重过 Wan VAE，直接读恢复副本 g 为偶数的窗（原话①；2.5）。
9. 训练沿用 `mme_vla_suite_b128_80k`（80k、batch 128、pi05_base、EMA 0.999、每 5000 步存 16 份），8 卡独占，不测 b256；训到 80k、测试取 60k（原话⑩⑬）。
10. 计划与全部 policy 改动在本仓库，MotionJEPA 只加抽取模块与 CLI（原话④）；bucket 名定死如上表（原话⑧）；边训边传 VLA，encoder／decoder 同期上传（原话②）。
11. 备选 run B（帧路 1 帧、有 demo 的任务 2 帧，motion 路相同）只写进计划，不在本次开工令内（原话⑬⑯；第二部分九节）。

## 二、现状与机制

**2.1 ckpt**：`runs/stage2-full144-1025-a/` 有 `checkpoint_epoch_{5..105 步长 5}.pt` 共 21 份，第 106 轮停训、无 `checkpoint_final`。ep105 顶层键 `arch`=`wan-latent-stage2-blockwise-pool4-three-decoder`、`fingerprint`、`weight_type='ema'`、`model`（EMA，185 张量，约 19.62M 参数，encoder＋三个 decoder）、`model_live`、`optimizer`、`config`、`epoch=105`、`global_step=7770`；EMA 与 live 确有差（`encoder.pos_t` 最大差 2.8e-4），必须显式取 `model`（MJ `src/motion_jepa/models/stage2_model.py:93-117`）。

**2.2 数据链路（逐跳形状／dtype／字节；是否改数）**

```
恢复副本 <seg>.bin 每窗 (9,16,32,32) fp32 = 589,824 B ── 只读不改数；新取 g=2k，旧取全部 g
  ▼ Stage2WanLatentMotionEncoder.forward(latents, geometry_valid=全 True (B,32,32))   [MJ wan_latent_encoder.py:418-422]
    ckpt 内 latents_mean/std affine → 576 patch(9×8×8) ＋ 4 readout(源时间片 [0,3,5,8] 32×32→4×4 掩码均值) = 580 token × 8 层
  → (B,4,256) fp32 = 4,096 B ── 冻结、eval、无梯度 → reshape(B,1024)（槽 0‖3‖5‖8 拼接，不改数）
新库行 (1,1024) f32 = 4,096 B，约 6.4 万行 ≈ 262 MB（旧库 (768,) 3,072 B×128,342 行 ≈ 380 MiB，grid16）
  ▼ FramesampDataset：motion_emb (96,1024) f32、motion_pos (96,256) f32、motion_mask (96,) bool
PerceptualMemory：Linear(256→768)+silu ‖ emb → Linear(1792→1024)，新参数 1,836,032（旧 1,573,888），从零训  [percep_mem.py:34-44,:85-87]
  ▼ memory (B, 512 帧路 + 96 motion = 608, 1024) 走 modulation（旧 512+160=672）   [history_pi0.py:326-361]
```

**2.3 窗口与 demo 补帧（与原来一样，只换步长）**：离线补帧在 policy `scripts/dataset/wan/extract_wan.py:60-99`：`real=min(33, seg_len−off)`，demo 且 real<33 时把段末帧 `frames[seg_len−1]` 重复补到 33 帧（像素层补、Wan 整窗编码），exec 补帧直接 raise，metadata 记 `real_frames`／`pad_frames`／`pad_source_frame`；在线 `policies/framesamp_memory.py:189-212 _encode_window` 用同式 `frame_ids=[f+j]+[es−1]*(W−real)`。可见条件 demo `s+16 ≤ es−1`、exec `u+32 ≤ t−es`（`motion_store.py:177-188`／`framesamp_memory.py:218,223`）。新方案只把 16m 换成 32m，补帧规则、补源帧、可见条件一字不改；{32k} 恰是 {16m} 的偶数子集，所以恢复副本偶数格就是新口径。

```
demo 段 len=100（帧 0..99），min_real=17 → 合法起点 < 84
 stride16: 0 16 32 48 64 80          80 窗真实 20 帧(80..99) → 补 13 帧（重复第 99 帧）
 stride32: 0    32    64             64 窗真实 33；96 窗只剩 4 帧 ≤16 → 不存在
exec 段 len=100，min_real=33 → 合法起点 < 68：stride32 = 0 32 64，三窗全真实，不补帧
```

⚠ 训练侧 `visible_motion_rows` 把步长写死在全局 `GRID_STRIDE`（`motion_store.py:55,162-185`），在线侧已读 yaml `motion.stride`（`framesamp_memory.py:84`）；只改 yaml 会让训练与在线可见窗不一致，必须按 layout 参数化（三节）。

**2.4 dropout 与增强**：encoder dropout 0.05 落在注意力 `to_out` 后与 SwiGLU 输出（MJ `wan_latent_encoder.py:123,135,149,152`，默认 `:303`），`eval()` 让其恒等；证明：同批窗 `eval()` 跑两遍逐位相等且 `train()` 一遍必须不等（参照 MJ `scripts/stage2_eval.py:51-54`）。增强只在 `preprocess_observation(train=True)` 对当前帧做（`src/openpi/models/model.py:144-190`），推理 `history_pi0.py:581` 传 `train=False`，官方 RoboMME policy 无 memory 路；本 fork 的 memory 一直离线从干净帧抽取，不增强、无 dropout（`history_gemma.py:117` 默认 0.0）。

**2.5 Wan 缓存可直接复用**：恢复副本 4,995 个 bin（exec 3,195，PatternLock 只有 195；demo 1,800），fp32、窗 33、grid16、demo `min_real 17 repeat_last`，共 75,699,191,808 B（约 70.5 GiB）。encoder 训练缓存用整 33 帧 fp16 编码，与 1+32 编码的 RGB 归一化 MSE 6.9e-8（MJ `1024-rgb-gtxyz-16views-stride2-plan.md:435`），fp16 落盘误差 mean(e²)≤8.96e-8（同文件 `:627`），G2 抽样重验。

**2.6 1300 步上限**：官方 eval 每集最多 1300 步（`examples/robomme/eval.py:30,88`），帧路 8 帧均匀采样不随步数长，只影响 motion 窗数：最坏 demo（VideoPlaceOrder 1247 帧）77→39、exec 1300 步 80→40，合计 157（旧预算 160）→ **79（预算 96，余量 17）**；训练集最坏（PickXtimes ep196）123→约 62。budget 80 只剩 1 窗余量；loader 零截断预检（`framesamp_dataset.py:227-233`）超窗即 raise，G6 复算。

**2.7 motion 不满 96 时补位怎么参与计算**：k 个真实窗之后的位零填充、mask 为 False、时刻记哨兵（`framesamp_dataset.py:320-341`，在线 `framesamp_memory.py:250-269` 同式）；96 位全部过两层 Linear（补位也算），再按时刻与帧路一起排序、补位落尾；action expert 的 Q 对 608 个 memory token 做注意力，`mask==False` 处 logit 置 −2.38e38，softmax 权重为 0、也不进分母（`history_gemma.py:97-99`）。结论：补位零贡献；序列长恒为 608、算力不随 k 变；k 小只是注意力在更少的 key 上分配。图：![](docs/figures/1026-motion-pad-modulation.svg)

## 三、我们这一侧要改什么

非改不可的原因：训练库 `LAYOUT`／`GRID_STRIDE=16` 是模块全局常量且布局名必须带 `grid16`（`motion_store.py:45-60,101-109`）；loader 合同只认 `(768,)` 旧路与 `adjacent_pair_v1` 四对（`motion_adapter.py:49-80`）；在线 sidecar 的 torch 子项目把 MotionJEPA 钉在 v10 提交 `ee719fb`、断言 v10 架构与 `(8,256)` 输出（`scripts/dataset/wan/wan_motion_infer.py:262-306`），客户端 `source_run` 正则只认 `#encoder`（`motion_client.py:51-58`）。

| 文件 | 锚点 | 改什么 | 关闭态（旧 yaml） | 开启态（新 yaml） |
|---|---|---|---|---|
| MJ `src/motion_jepa/export/stage2_memory.py`（新）＋`scripts/stage2_memory_export.py`（CLI） | `load_ema_encoder`、`encode_windows` | 读恢复副本偶数格窗 → EMA encoder → (n,1024) 按 pack 契约落盘 | 不存在 | 产出 `motion-tokens-stage2/` |
| `datastore/motion_store.py` | `LayoutSpec`、`visible_motion_rows` | 加 `grid_stride`；登记新 layout；网格公式读 spec 步长 | 旧三 layout 逐字节不变 | `motion-stage2-1x1024-grid32-demopad17-v1`，(1,1024) |
| `datastore/motion_adapter.py`、`training/framesamp_dataset.py`、`training/dataloader.py` | `motion_policy_spec`、`check_motion_contract`、stride 断言 | 新 readout `stage2_pool4_concat_v1`；stride 与 spec 比；绑 arch／state_key／sha | 不变 | 校验 stage2 身份 |
| `scripts/dataset/pack_motion_store.py` | `read_segment_tokens`、`gather_provenance` | 新 layout；latent 格与 token 格 2:1；`--check-padding` | 不变 | 只认偶数格 |
| 新 yaml＋启动／导出包装 | `motion:` 节 | yaml 见第二部分一节；脚本照 pair4／512 版改常量 | — | 新 run |
| `scripts/dataset/wan_stage2/`（新 torch 子项目）＋`motion_client.py`、`policy_config.py` | sidecar | 独立 `pyproject`/`uv.lock` 钉新 MJ 分支；`motion.encoder_kind` 选 sidecar；放宽 `#model` | 旧 v10 sidecar 原样 | 回 4,096 B |

`percep_mem.py`、服务端窗口枚举与客户端 `examples/robomme/utils.py:38 pack_buffer`（只打包帧）零改动，靠 yaml `stride: 32`。

## 四、验收

| 判定 | 查什么 | 怎么查 | 过了说明什么（为什么能成立） |
|---|---|---|---|
| G1 `DROPOUT_OFF` | dropout 真关 | 64 窗 eval 两遍、train 一遍 | eval 逐位相等＋train 不等 ⇒ 存在且被关；同卡同 kernel、fp32 TF32 关 ⇒ 可逐位 |
| G2 `WAN_EQUIV` | 恢复副本 = 训练口径 | 从 source 库 pkl 拼 64 窗原始帧（含 demo 尾窗）用 MJ `wan_fp16_full33.py` 重编比 | 归一化误差在 K4 阈内 ⇒ 无需重过 Wan |
| G3 `EXPORT_PARITY` | 导出函数 = MJ 评估路径 | 32 个 rgb-gtxyz val aug0 窗，两条路径同 encoder | 同权重同输入同精度 ⇒ `max_abs=0` |
| G4 `ONLINE_OFFLINE_PARITY` | 在线 = 离线 | 64 窗：编码段同 latent；端到端帧→VAE→encoder；核 frame_ids | 编码段同函数 B=1 ⇒ 逐位；端到端多一次 VAE ⇒ ≤1e-5 |
| G5 `VERIFY_MOTION` | 库完整、补帧与原来一样 | `pack verify --check-padding` | 行数＝偶数格窗数、全有限；`real+pad==33`、pad 只在 demo 末窗、补源帧＝段末帧 |
| G6 `TRAIN_ES_BOUND` | 预算够 | 训练集与评估 1300 上限 | 最坏 ≤96 ⇒ 零截断 |
| G7 `SMOKE_TRAIN` | 能训 | 8 卡 20 步 | loss 有限、无 OOM，s/步 与 0.96 对照 |
| G8 训练 `EXIT_CODE=0` | 训完 | 日志尾行＋16 个 step 目录 | 80k 全部存盘，60k（测试用）与 79999 都在 |
| G9／G10 上传 | 远端可恢复 | 全量 sha256 匿名回读 | 远端逐字节等于本地，此后才可写「已备份」 |

## 五、步骤

| 阶段 | 内容 | 判据 |
|---|---|---|
| 0 | 开工令；sonnet 审计划 | `PLAN_CONSISTENCY_REVIEW=PASS` |
| 1 | 派 S1、S2、S3 → 合 S1、push MJ 分支 → 派 S4 → 依次合 S2、S3、S4；S3 合入后即起 MJ ckpt 上传 | 每块 `PRE/POST_MERGE_REVIEW=PASS`；G10（长尾） |
| 2 | 抽取＋前三闸 → 打包＋校验＋预算 → 在线对拍 | G1–G3 → G5、G6 → G4 |
| 3 | 短测 → Beta commit → 正式起跑＋增量上传 | G7 → G8 |
| 4 | 60k 一传完即可交测试；80k 后正式驱动全量回读；留档；handoff | G9、`HANDOFF_WRITE=PASS` |

## 六、子代理分工与合并（简述）

拆四个写入块，各在独立 worktree 由 opus 写：S1 只在 MotionJEPA 新分支加抽取模块与 CLI；S2 管 policy 库格式、loader 合同与打包；S3 管新 yaml、启动脚本和两份导出包装；S4 管在线 sidecar，需把新 MJ 分支 sha 钉进 `uv.lock`，等 S1 合并 push 后才派。合并顺序 S1 → S2 → S3 → S4，每块合前派一个 sonnet 钉死 BASE/TIP 审 diff，合后跑定向短测再 push。抽取、打包、训练、上传由 opus 运行型代理在主检出起 tmux，运行代理不改代码。备选 run B 的 S5 不在本次。逐文件改动、分配表、闸门判定行全文、runbook、风险与盲区见第二部分。

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- R1 不重过 Wan VAE：一切 latent 只读恢复副本 `v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents/`（只读，不改不删）；G2 的 64 窗重编只为比对，结果只进 `records/`。
- R2 不改 encoder 权重：抽取与 sidecar 只读 `checkpoint_epoch_105.pt` 的 `model` 键，strict 加载、`requires_grad_(False)`、`eval()`、`torch.no_grad()`、`torch.autocast(enabled=False)`、`torch.backends.cuda.matmul.allow_tf32=False`、`cudnn.allow_tf32=False`；不写回任何 MJ `runs/` 文件。
- R3 memory 不增强：`geometry_valid` 恒为全 True `(B,32,32)` bool（等价 aug_id=0：MJ `dataset_paired_wan.py::derive_geometry_masks` 中恒等单应的 `common_valid` 全真）；不加噪声、不做帧增强。
- R4 不动 `main` 与 `v12-3decoder`：MJ 侧只在新分支 `v12-3decoder-memexport`（自 `6c79ebb`）提交并 push 该分支；policy 侧只在 `v4-256concat1` 及其 `worktree-agent-*` 子分支。
- R5 旧口径逐字节不变：旧 layout（`motion-768-grid16-v1`、`motion-768-grid16-demopad17-v1`、`motion-v10-8x256-grid16-demopad17-v1`）、旧 yaml、旧 v10 sidecar 子项目 `scripts/dataset/wan/`（含 `uv.lock`、`SOURCE_PIN.json`）零 diff；`motion_client.py` 的 `encoder_kind="v10"` 分支逐字不变。
- R6 训练期代码冻结与只读锁沿用 1016 做法：Beta commit 后到 G8 期间不 commit、不合并；被跟踪的 `src/`、`scripts/` 按 `v1-store/m8v-scripts/m8v-chmod-before.txt` 体例先记权限再 `chmod a-w`，训完按记录恢复；开发转开发副本（`AGENTS.md` 第 14 条覆盖项）。
- R7 tmux 只用前缀 `cc-vla-s2-`：`cc-vla-s2-export`、`cc-vla-s2-pack`、`cc-vla-s2-smoke`、`cc-vla-s2-train`、`cc-vla-s2-inc`、`cc-vla-s2-mjup`（测速用过的 `cc-vla-s2-speed8` 已退出）；清理只按此清单 `tmux kill-session -t '=名'` 逐个，前后各 `tmux ls`。
- R8 不进写集：`third_party/**`、根 `pyproject.toml`、根 `uv.lock`（0901 红线 10）、`scripts/dataset/wan/**`、`AGENTS.md`、`CLAUDE.md`、`greatlakes.md`、`.agentmetarules`、`v1-store/**`（`m8v-ckpt-inc.sh` 不改只换参数；注册副本目录由运行型代理按 runbook 建）。本仓库无第 21 条受保护目录。
- R9 不处理 rgb-gtxyz 上传失败：`/scratch/hongze/logs/rgb-gtxyz-full-20261024/upload.log` 末行 `HF_VERIFY=FAIL error=ValueError`、`EXIT_CODE=1`（进程 1675039、1678537 已不存在，MJ handoff 仍写 running），属 Codex 任务长尾，只报告。
- R10 新 run、新库、新 bucket 全新：run 目录、`motion-stage2-1024/`、两个 bucket 起跑前 `ls -ld`／`hf buckets list HongzeFu` 确认不存在；不带 `--force`、不 `overwrite`。
- R11 任一闸门 FAIL 停下交用户；不放宽阈值；崩溃不自动 resume。
- R12 子代理 worktree 内只跑 ≤5 分钟 CPU 测试，环境取法：policy `UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`（先打印 `mme_vla_suite.__file__` 确认在 worktree）；MJ `UV_PROJECT_ENVIRONMENT=/scratch/hongze/MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`（先打印 `motion_jepa.__file__`）。GPU 验收合并后由运行型代理串行跑。
- R13 不测 batch 256（用户「你不用再测这个 batch 二五六了」）；测速 run `cc-speed8-motion-b128-20261026-a` 在 `v1-store/train-runs/mme_vla_suite_b128_80k/cc-speed8-motion-b128-20261026-a/` 留有 step 119 一份 ckpt，收尾时按精确路径删除并记入留档。
- R14 备选 run B（九节）不在本次开工令内：S5 不派、G11 不跑，另取开工令。
- R15 `v1-store/` 下不建 symlink 外链（`AGENTS.md` 第 14 条覆盖项「主工作副本 v1-store/ 下没有任何 symlink 外链」）：sidecar 需要的 encoder 目录用注册副本（实体复制 + sha256 核对），不用软链。

## 一、逐文件改动清单

### A. MotionJEPA（分支 `v12-3decoder-memexport`，S1）

MJ `pyproject.toml` 用 setuptools `packages.find where=["src"]`，`scripts/` 不进包，所以可复用函数必须放进 `src/`。

**新模块 `src/motion_jepa/export/__init__.py`（空）＋`src/motion_jepa/export/stage2_memory.py`**（现有能力：`build_stage2_model_from_cfg(cfg, *, data_metadata)` 于 `src/motion_jepa/models/factory.py:150`、`load_stage2_checkpoint(model, checkpoint, *, weights="main")` 于 `stage2_model.py:137`；拟新增接口如下）

```python
def load_ema_encoder(ckpt_path: str, *, device: str, expected_sha256: str | None = None
                     ) -> tuple[Stage2WanLatentMotionEncoder, dict]:
    """整份 ckpt → build_stage2_model_from_cfg(ckpt['config'], data_metadata=<ckpt 内训练合同>)
    → load_stage2_checkpoint(weights='main') → 取 .encoder；断言 ckpt['weight_type']=='ema'、
    arch==STAGE2_ARCH_TAG、fingerprint 一致；eval()+requires_grad_(False)。info 含 checkpoint_sha256、
    fingerprint_sha256、epoch、global_step、state_key='model'、weight_type、precision='fp32'、tf32='off'。"""

@torch.no_grad()
def encode_windows(encoder, latents: torch.Tensor) -> torch.Tensor:
    """latents (B,9,16,32,32) float32 原始（未归一化）→ (B,1024) float32。
    geometry_valid = torch.ones(B,32,32,bool)；autocast 关；输出 (B,4,256).reshape(B,1024)，
    槽序即 config.source_times=[0,3,5,8]。断言 encoder.training is False、输出全有限。"""
```

**新 CLI `scripts/stage2_memory_export.py`**（只做参数解析与 IO，`from motion_jepa.export.stage2_memory import load_ema_encoder, encode_windows`）：
- CLI：`--ckpt <abs> --expected-sha256 <64hex> --latents <恢复副本 wan-latents> --manifest <mixed8 episode_manifest.json> --out <v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion-tokens-stage2> --grid-subsample 2 --batch 1 --device cuda:0 [--max-segments N] [--check-dropout 64] [--resume]`。`--batch` 固定 1（与 sidecar 同形，见 G4 逐位依据）；`--max-segments` 只冒烟。
- 输出（按 policy `pack_motion_store.py::read_segment_tokens` 现有契约，不另造 npz）：每段 `<key>.f32.bin`（`num_grid32 × 4096 B`）、`<key>.f32.bin.sha256`、`<key>.metadata.json`（`layout`、`row_shape=[1,1024]`、`dtype`、`token_dim=1024`、`token_bytes=4096`、`bytes`、`num_grid`、`sha256`、`source_grid=[0,2,4,…]`、逐行 `real_frames`／`pad_frames`／`pad_source_frame`（从 latent metadata 原样转抄）、`input_block_sha256[]`（逐窗 589,824 B 的 sha256）、`encoder{run_name, checkpoint_name, epoch, state_key, weight_type, arch, checkpoint_sha256, fingerprint_sha256}`、`numerics{fp32, tf32:off, batch:1, torch, cuda, gpu}`）；根下 `export_record.json`，末行 `STAGE2_EXPORT=PASS segments=<n> rows=<n>`。
- `--check-dropout 64` 产出 G1 判定行；不改任何既有 MJ 文件。
- 测试 `tests/test_stage2_memory_export.py`：CPU 上随机初始化小模型验证形状、槽序（`out[:,256*j:256*(j+1)] == enc_out[:,j]`）、`training=False` 断言、eval 两遍逐位相等、train 模式不等；`import motion_jepa.export.stage2_memory` 不依赖 `scripts/`。

### B. policy（分支 `v4-256concat1`）

**S2 `src/mme_vla_suite/datastore/motion_store.py`**
- `LayoutSpec` 增字段 `grid_stride: int = 16`（排在 `row_shape` 之后，旧三条登记值不变）；`LAYOUT_SPECS` 新增 `"motion-stage2-1x1024-grid32-demopad17-v1": LayoutSpec(17, 33, "repeat_last", (1, 1024), grid_stride=32)`。
- 常量 `STAGE2_LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"`；`LAYOUT` 保持旧值（R5）；`:106-109` 的布局名检查改为逐条 `f"-grid{spec.grid_stride}-" in name`。
- `seg_num_grid`、`segment_grid_starts`、`max_visible_count` 增 `stride` 参数（默认 `GRID_STRIDE`）；`visible_motion_rows` 的 `s = GRID_STRIDE*m`／`u = GRID_STRIDE*m` 改为 `entry.spec.grid_stride*m`，可见条件（`:177-188`）不动；`build_index_entries` 按 spec 步长算 num_grid。
- 测试 `tests/datastore/test_motion_store_stage2.py`（新）：旧三 layout 对 `BASE` 的 `visible_motion_rows` 在 1,000 组 (seg_len, t) 上逐元素相同；新 layout 的起点集合等于旧 grid16 偶数格子集；demo len=100 例出 {0,32,64}。

**S2 `src/mme_vla_suite/datastore/motion_adapter.py`**
- 新常量 `STAGE2_READOUT = "stage2_pool4_concat_v1"`、`STAGE2_SHAPE = (1, 1024)`；`motion_policy_spec` 在 `readout == STAGE2_READOUT` 时要求 `source_shape==[1,1024]`、`num_tokens==1`、`dim==1024`、`budget>0`，返回 `MotionPolicySpec((1,1024), STAGE2_READOUT, 1, 1024, budget, budget, (STAGE2_LAYOUT,))`；四对与旧分支逐字不变。

**S2 `src/mme_vla_suite/training/framesamp_dataset.py`、`training/dataloader.py`**
- `check_motion_contract`（`:68`）新增分支：`readout==STAGE2_READOUT` 时要求 `enc.arch=="wan-latent-stage2-blockwise-pool4-three-decoder"`、`enc.state_key=="model"`、`enc.weight_type=="ema"`、`enc.checkpoint_sha256==mcfg.checkpoint_sha256`。
- `framesamp_dataset.py:156` 与 `dataloader.py:51` 的 `int(mcfg.stride) == ms.GRID_STRIDE` 改为 `== meta.spec.grid_stride`。
- `__getitem__` 取行后 `(k,1,1024)` → `reshape(k,1024)` 再进 `_pad_motion`（`num_tokens==1` 不走重复分支），输出 `motion_emb (96,1024)`；`_pad_motion` 本身不改。
- `dataloader.py::_parse_source_run` 不改（`stage2-full144-1025-a/checkpoint_epoch_105.pt#model` 可解析）。
- 测试 `tests/training/test_framesamp_motion_stage2.py`（新）：3 episode 夹具库验证形状、mask、错 arch／错 sha／错 stride 均 raise；k<96 时 `motion_mask` 前 k 位 True、其余 False、补位 emb/pos 全 0；旧 768 与 pair4 夹具输出对 `BASE` 逐字节相同。

**S2 `scripts/dataset/pack_motion_store.py`**
- `read_segment_tokens`：`spec.row_shape == (1,1024)` 时套同一 contract 校验（layout 用 `STAGE2_LAYOUT`）。
- `gather_provenance`：新增参数 `latent_grid_subsample: int = 1`；为 2 时要求 `len(input_block_sha256)==_ng`、latent 文件 `st_size == latent_num_grid*CHUNK_BYTES` 且 token 第 j 行对应 latent 块 `2j`，逐块 sha256 现场比对；encoder 组取 token metadata 的 `encoder`，VAE 组沿用 `load_vae_compat`。CLI `pack` 新增 `--layout {legacy,stage2}`、`--latent-grid-subsample`；`verify` 逐行 memcmp 不变。
- `verify --check-padding`（新）：`pack` 把每行的 `real_frames`／`pad_frames`／`pad_source_frame` 从 latent metadata 原样写入 store metadata，`verify` 读 store metadata 逐行断言 `real_frames + pad_frames == 33`；`pad_frames > 0` 只出现在 demo 段末窗；`pad_source_frame == seg_start + seg_len − 1`；demo 行 `17 ≤ real ≤ 33`、exec 行 `real == 33`；输出并入 G5。

**S3 新 yaml `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml`**：帧路节与 `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml` 逐字相同（budget 512、token_per_image 64、modulation、memory_token_dim 1024、use_pos_emb true、use_state_emb false）；`motion:` 节：

```yaml
motion:
  enabled: true
  source_shape: [1, 1024]
  readout: stage2_pool4_concat_v1
  num_tokens: 1
  dim: 1024
  budget: 96
  pos_dim: 256
  demo_min_real_frames: 17
  demo_tail_pad: repeat_last
  stride: 32
  window_frames: 33
  window_direction: forward
  grid_origin: segment_start
  frame_size: 256
  store_path: v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion-stage2-1024
  source_run: stage2-full144-1025-a/checkpoint_epoch_105.pt#model
  checkpoint_sha256: <G3 现场算出的 64 位，S3 先写占位、主会话打包后单独 commit 填入>
  encoder_kind: stage2_pool4_concat
  online_gpu: 4
```

**S3 启动与导出包装**
- `scripts/training/run_mixed8_stage2_1024_80k.sh`（新，照 `run_mixed8_pair4_80k.sh` 88 行改 `HC=`、`RUN_NAME` 禁空、`TRAIN_HEAD` 字面量、`fsdp_devices=8`、`CUDA_VISIBLE_DEVICES=0-7`；保留 `TRAIN_CMD=` 回显；测速同款环境 `XLA_PYTHON_CLIENT_PREALLOCATE=false`）；`scripts/training/check_mixed8_stage2_1024_launch.py`（新，照 `check_mixed8_pair4_launch.py` 改 yaml 名、库路径、budget、readout）。
- `scripts/dataset/hf_export/run_mixed8_200_vla_stage2_1024_80k_ckpt_export.sh`（新，照 24 行的 `run_mixed8_200_vla_512_80k_ckpt_export.sh` 改 RUN 与 bucket）；`scripts/dataset/hf_export/run_stage2_pool4_encoder_ckpt_export.sh`（新，照 `run_mixed8_200_encoder_ckpt_export.sh` 改白名单：阶段 III 21 份 ckpt＋`config.yaml`＋`train_metrics_epoch.jsonl`＋`motion_ablation/`，阶段 II `stage2-scale-20261025-d` 的 `checkpoint_epoch_{4,8}.pt`、`checkpoint_final.pt`、`config.yaml`、`train_metrics_epoch.jsonl`；README 写明 `model`=EMA、`model_live`=live 与 fingerprint）。

**S4 在线 sidecar**
- 新 torch 子项目 `scripts/dataset/wan_stage2/`：`pyproject.toml` 依赖与 `scripts/dataset/wan/pyproject.toml` 相同的 torch 2.9.0／diffusers 0.39.0／numpy 2.4.4 等，`motion-jepa @ git+https://github.com/hongzefu/MotionJEPA@<S1 合并后 push 的 40 位 sha>`（git 依赖安装后 `motion_jepa.export.stage2_memory` 可 import）；独立 `uv.lock`；venv `UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2`。不改旧子项目（R5：旧锁文件钉 v10 `ee719fb`，换 pin 会改变旧 run 的 sidecar 环境；按全局 uv 规则独立依赖另建 `pyproject`，不用 workspace）。
- `scripts/dataset/wan_stage2/stage2_sidecar.py`（新）：协议沿用 `src/mme_vla_suite/policies/motion_protocol.py`（按文件加载）；VAE 段按文件加载 `scripts/dataset/wan/wan_motion_infer.py` 的 `load_vae`、`encode_chunk`（只读复用，不复制数值语句）；encoder 段 `from motion_jepa.export.stage2_memory import load_ema_encoder, encode_windows`，B=1；回包 1024 个 f32 = 4,096 B；`--stub` 返回 `np.full(1024, 起点)`。
- `src/mme_vla_suite/policies/motion_client.py`：`MotionEncoderClient.__init__` 增 `encoder_kind: str = "v10"`。现状 `:51-58` 用正则 `([A-Za-z0-9][A-Za-z0-9_.-]*)/(checkpoint_epoch_\d+\.pt)#encoder` 解析 `source_run`，并把 encoder 目录强制推成 `v1-store/external/motionjepa/<run>`。`"stage2_pool4_concat"` 分支：正则尾部改为接受 `#model`；目录同样推为 `v1-store/external/motionjepa/stage2-full144-1025-a`，该目录是注册副本（R15：实体复制 `checkpoint_epoch_105.pt` 与 `config.yaml`，sha256 与 MJ 原件一致，与现有 `wan-mixed8-200-filter2-stop075-b88x8-72ep-a/` 等目录同体例）；argv 用 `--project scripts/dataset/wan_stage2 … stage2_sidecar.py --ckpt <注册副本>/checkpoint_epoch_105.pt --expected-sha256 <sha>`，期望回包维度 1024。`"v10"` 分支逐字不变。
- `src/mme_vla_suite/policies/policy_config.py:132-148`：把 `mcfg.get("encoder_kind","v10")` 传给客户端；其余不变。`framesamp_memory.py` 不改（`:208` 已按 `motion_dim` 校验回包形状，`:189-212` 补帧同式）。
- 测试 `tests/policies/test_motion_client_stage2.py`（新）：stub 档走完整 IPC，起点帧号正确、回包 4,096 B；`#model` 只在 stage2 分支被接受，`v10` 分支收到 `#model` 仍 raise；`v10` 默认 argv 与 `BASE` 相同；注册目录不存在时报错信息含期望路径。

## 二、子代理分配表

`BASE` = policy `3861154`（S1 为 MJ `6c79ebb`）。派发前核对：全局与项目 `settings.json` 的 `worktree.baseRef=head`；`git check-ignore -q .claude/worktrees/probe`；`git worktree list` 存档；主检出 `git status --short --ignore-submodules=dirty` 记录。

| 子任务 | 目标 | 可写文件集合 | 禁触路径 | 接口契约与依赖 | 合并顺序 | 验收命令与判定行（在哪、什么环境） | 资源 | 共享文件归属 |
|---|---|---|---|---|---|---|---|---|
| S1（opus＋worktree，**MotionJEPA 仓库**新分支 `v12-3decoder-memexport`） | 抽取模块与 CLI | `src/motion_jepa/export/__init__.py`、`src/motion_jepa/export/stage2_memory.py`、`scripts/stage2_memory_export.py`、`tests/test_stage2_memory_export.py` | MJ 其余一切文件、`runs/**` | `load_ema_encoder`、`encode_windows`、token 文件契约（一节 A） | 1；合并后主会话 push 该分支并记 sha | MJ worktree，R12：`pytest tests/test_stage2_memory_export.py -q` passed | CPU | 无 |
| S2（opus＋worktree） | 库格式、loader 合同、打包 | `motion_store.py`、`motion_adapter.py`、`framesamp_dataset.py`、`dataloader.py`、`pack_motion_store.py`、`tests/datastore/test_motion_store_stage2.py`、`tests/training/test_framesamp_motion_stage2.py` | R8；其余 `src/**` | `STAGE2_LAYOUT`、`STAGE2_READOUT`、`LayoutSpec.grid_stride`、`--latent-grid-subsample`、`--check-padding` | 2 | policy worktree，R12：两份新测试＋`tests/datastore -q` 全 passed | CPU | `motion_store.py`、`framesamp_dataset.py` 唯一写者 |
| S3（opus＋worktree） | yaml、启动、导出包装 | 新 yaml、`run_mixed8_stage2_1024_80k.sh`、`check_mixed8_stage2_1024_launch.py`、两份导出包装 | R8；`src/**` 的 .py | 只按名字引用 S2 常量 | 3（在 S2 后） | `bash -n` 三脚本；S2 合入后主会话复跑：新 yaml 经 `motion_policy_spec` 解析通过 | CPU | yaml 唯一写者 |
| S4（opus＋worktree，S1 push 后派） | 在线 sidecar | `scripts/dataset/wan_stage2/**`（新）、`motion_client.py`、`policy_config.py`、`tests/policies/test_motion_client_stage2.py` | R8（含 `scripts/dataset/wan/**`） | `encoder_kind`；`#model`；回包 4,096 B；依赖 S1 sha | 4 | policy worktree：stub 测试 passed；`uv lock --project scripts/dataset/wan_stage2` 成功 | CPU | `policy_config.py`、`motion_client.py` 唯一写者 |
| S5（备选 run B，**不在本次开工令内**） | 帧路分段锚点选帧 | 见九节 | R8 | 见九节 | 另取开工令 | G11 | CPU | 与 S2 共享 `framesamp_dataset.py`，S2 合入后才派 |
| 审查 RV-S1…RV-S4（sonnet，只读） | 合前审 | 无 | 一切 | 钉 `REVIEW_BASE`/`REVIEW_TIP` | 各块合前 | `PRE_MERGE_REVIEW=PASS base= tip= files= commits= findings=` | — | — |
| 主会话自做 | yaml 填 sha、Beta commit、handoff | yaml `checkpoint_sha256` 一行、`docs/handoff/current.json`、留档 | — | 依赖 G3 | 打包后 | — | — | 主会话 |
| 运行型 X1（opus，主检出，不加 worktree） | 抽取＋G1–G4 对拍＋注册副本 | 只写 `motion-tokens-stage2/`、`v1-store/external/motionjepa/stage2-full144-1025-a/`、日志与 `records/` | 一切代码 | 命令照四节 2–4；起跑判据首段 `metadata.json` 落盘 | S2、S1 合入后（G4 等 S4） | `STAGE2_EXPORT=PASS`、G1–G4 行 | GPU0 | 账目归主会话 |
| 运行型 X2（opus，主检出） | 短测、正式训练＋增量上传 | run 目录、export 目录 | 一切代码 | 命令照四节 5–7；起跑判据日志出现 step 1 loss 与 `TRAIN_CMD=` | Beta 后 | 首个 loss 行有限 | GPU0–7 | 同上 |
| 运行型 X3（opus，主检出） | MJ ckpt 上传 | export 目录 | 一切代码 | 命令照四节 1 | S3 合入后即派，与 X1 并行 | `MJ_CKPT_UPLOAD=` 行 | CPU | 同上 |

合并：`git merge --no-ff <TIP sha> -F <scratchpad 消息>` → `POST_MERGE_REVIEW=PASS merge=<sha> tests=<…> files=<n>`（定向测试＋`git diff --name-only` 与上表核对＋R5 零 diff 检查 `git diff --quiet <BASE> HEAD -- scripts/dataset/wan third_party pyproject.toml uv.lock`）→ push → 清理 worktree（`git worktree remove`、`git branch -d`）。

## 三、闸门总表

| 闸门 | 命令 | 判定行与阈值 |
|---|---|---|
| G1 | `stage2_memory_export.py … --check-dropout 64`（X1，GPU0） | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0 train_vs_eval_max_abs=<x>`；要求前者 =0 且后者 >0 |
| G2 | X1（主会话授权）：从 source 库 `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/source/data/<idx>.pkl` 按 `(epis_idx, step_idx)` 拼 64 窗的 33 帧原始 `image`（uint8 256×256×3；每任务 4 窗，含 ≥16 个 demo 尾窗，按 latent metadata `pad_source_frame` 同规则补帧），用 MJ `scripts/dataset-build/wan_fp16_full33.py` 重编，与恢复副本对应 latent 按 `latents_std` 归一化比对 | `WAN_EQUIV=PASS windows=64 demo_tail=<n> worst_mse=<x> worst_max=<y>`；阈值 worst_mse ≤1e-6、worst_max ≤0.01（与 MJ 计划 K4 同阈） |
| G3 | X1：`/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1/index/windows.parquet` 过滤 `split=="val"`，固定 seed 0 取 32 窗；MJ `scripts/dataset-build/rgb_gtxyz_reader.py::read_pair(key, aug_id=0)` 读 RGB latent，`geometry_valid` 取自 `masks/`（断言全 True）；同一 encoder 分别走 `encode_windows` 与 `stage2_eval.py::evaluation_context` 下的 `model.encoder(rgb_raw.float(), geometry_valid)` | `EXPORT_PARITY=PASS windows=32 seed=0 max_abs=0 geometry_all_true=1 ckpt_sha256=<64>` |
| G4 | X1（S4 合入后）：64 窗 (a) 同 latent 经 sidecar 编码段 vs 库行；(b) 原始 33 帧经 sidecar 端到端 vs 库行；(c) 用 stub 编码器接住 `_encode_window` 送出的 33 帧窗（逐帧填唯一像素值反推 frame_ids），与 latent metadata 的 `real_frames`／`pad_source_frame` 一致 | `ONLINE_OFFLINE_PARITY=PASS windows=64 encoder_max_abs=0 e2e_max_abs=<x> frame_ids_mismatch=0`；e2e ≤1e-5 |
| G5 | `pack_motion_store.py verify --store … --check-padding` | `VERIFY_MOTION=PASS scanned=<rows> mismatches=0 nonfinite=0 real_plus_pad_ne_33=0 exec_padded=0 pad_not_last_demo=0 pad_source_mismatch=0 demo_tail_padded=<n> demo_tail_dropped=<n>` |
| G6 | loader 构造时的零截断预检（`framesamp_dataset.py:227-233`）打印训练集最大可见窗数＋主会话一次性脚本按 `ms.seg_num_grid(len, min_real, stride=32)` 复算 1300 步上限 | `TRAIN_ES_BOUND=PASS train_max=<n> eval_max=<n> budget=96`；任一 >96 FAIL |
| G7 | `MAX_STEPS=20 bash scripts/training/run_mixed8_stage2_1024_80k.sh` | `SMOKE_TRAIN=PASS steps=20 loss_finite=1 oom=0 sec_per_step=<x>`；记录与测速 0.96 s/步的差 |
| G8 | 正式训练日志 | 尾行 `EXIT_CODE=0`，且 `ls …/<RUN>/` 含 5000…75000、79999 共 16 个 step |
| G9 | `run_mixed8_200_vla_stage2_1024_80k_ckpt_export.sh`（`SMOKE=1` → 全量） | `CKPT_UPLOAD=PASS steps=16 files=<n> sha256_mismatch=0 public=1`；`INC_END … done=16` 只是增量结束，不是完成信号；60k 的 `.done` 出现即可交测试 |
| G10 | `run_stage2_pool4_encoder_ckpt_export.sh` | `MJ_CKPT_UPLOAD=PASS files=<n> bytes=<n> sha256_mismatch=0 public=1` |
| G11（备选 B） | 九节 | `FRAME_SELECT_PARITY=PASS episodes=<n> steps=<n> mismatch=0 nodemo_mask_ok=1` |
| 合并 | 见二节 | `PRE_MERGE_REVIEW=PASS`、`POST_MERGE_REVIEW=PASS` |

G1／G3／G4(a) 能要求逐位的理由：同一 A100、同一 torch 2.9.0、同一函数、B=1 固定 kernel 形状、fp32 且 TF32 关、`eval()` 后无随机算子；G4(b) 多一次 VAE 编码，沿用阈值。G2 不要求逐位：整 33 帧 fp16 与 1+32 fp32 本就是两种编码。G2 若发现 source 库 pkl 帧与 h5 原件不同源（帧 sha 对不上 latent metadata `input_frames_sha256`），先停下报告，不改用其他口径。

## 四、runbook（主会话；命令由运行型代理执行）

1. **MJ ckpt 上传**（S3 合入后即起，与 2–4 并行）：`tmux new-session -d -s cc-vla-s2-mjup "set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/dataset/hf_export/run_stage2_pool4_encoder_ckpt_export.sh 2>&1 | tee /scratch/hongze/logs/mj-ckpt-upload-20261026/upload.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/logs/mj-ckpt-upload-20261026/upload.log"`。上传前 `hf buckets list HongzeFu` 确认 bucket 不存在；结束后 MJ handoff `stage3-formal-training-v1` 的长尾由 MJ 会话更新（本会话只读，按第 28 条发消息）。
2. **抽取**（S1、S2 合入后）：先 `--max-segments 4` 冒烟（≤5 分钟，`run_in_background`），再 `tmux new-session -d -s cc-vla-s2-export "bash /scratch/hongze/logs/vla-s2-20261026/export.sh"`，脚本内：`cd /scratch/hongze/MotionJEPA && UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/stage2_memory_export.py --ckpt runs/stage2-full144-1025-a/checkpoint_epoch_105.pt --expected-sha256 <sha> --latents /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents --manifest <mixed8 episode_manifest.json> --out /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion-tokens-stage2 --grid-subsample 2 --batch 1 --check-dropout 64`，日志三件套。随后 G2、G3。
3. **打包**：`tmux new-session -d -s cc-vla-s2-pack "bash …/pack.sh"`，`uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --manifest … --tokens …/motion-tokens-stage2 --latents …/wan-latents --out …/motion-stage2-1024 --encoder-run-dir /scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a --raw-dir … --restore-receipt …/restore_receipt.json --expected-ckpt-sha256 <sha>`，再 `verify --store …/motion-stage2-1024 --check-padding` → G5；G6。主会话把 sha 填进 yaml 并 commit（`commitV11.<n>: …`）。
4. **注册副本与在线对拍**（S4 合入后）：`mkdir v1-store/external/motionjepa/stage2-full144-1025-a` 后 `cp` ckpt 与 `config.yaml`，`sha256sum` 与 MJ 原件一致（R15）；`UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2 uv sync --project scripts/dataset/wan_stage2`，起 sidecar 跑 64 窗 → G4。
5. **短测**：`RUN_NAME=cc-vla-s2-smoke-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<HEAD> bash scripts/training/run_mixed8_stage2_1024_80k.sh`（tmux `cc-vla-s2-smoke`）→ G7；跑完删该临时 run（第 6 条）。
6. **Beta 与正式起跑**：建 `docs/training-doc/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/launch.md`（①–⑥节、tmux 清单）→ `commitV11.<n>Beta: …`，`git status --short` 为空 → 写 `AGENTMETARULES_HOLD`（freeze＋sha）→ R6 只读锁 → `tmux new-session -d -s cc-vla-s2-train "set -o pipefail; RUN_NAME=mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k TRAIN_HEAD=<Beta sha> PYTHONUNBUFFERED=1 bash scripts/training/run_mixed8_stage2_1024_80k.sh 2>&1 | tee /scratch/hongze/logs/vla-s2-20261026/train.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/logs/vla-s2-20261026/train.log"`。
7. **增量上传**：起跑后立即 `tmux new-session -d -s cc-vla-s2-inc "set -o pipefail; bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1 mixed8-200-vla-stage2-1024-80k <allowlist 相对路径> cc-vla-s2-train 2>&1 | tee /scratch/hongze/logs/vla-s2-20261026/inc.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/logs/vla-s2-20261026/inc.log"`（allowlist 沿用 512 版驱动那份）。60000 的 `.done` 出现即通知用户可取 60k 测试（约 T0+20 h）。
8. **Monitor**（每份日志一个）：训练 `tail -n +1 -F …/train.log | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "EXIT_CODE=|Traceback|out of memory|CUDA out of memory|Killed|nan|NaN|Error|step [0-9]+000\b"`；增量 `-E "INC_END|EXIT_CODE=|FAIL|错误|Traceback|\.done"`；抽取 `-E "STAGE2_EXPORT=|DROPOUT_OFF=|EXIT_CODE=|Traceback|NaN|非有限"`；MJ 上传 `-E "MJ_CKPT_UPLOAD=|HF_VERIFY=|EXIT_CODE=|Traceback|FAIL"`。
9. **收尾**：G8 后解除只读锁、删 `AGENTMETARULES_HOLD`；跑 G9 正式驱动；写 `result.md`（吞吐、loss、判定行）→ `commitV11.<n>: …`；按 R13 删测速 run 目录；handoff。

## 五、风险登记

| 项 | 说明 | 处置 |
|---|---|---|
| encoder 域外 | 阶段 II/III 训练源是原版 16×100（val = 每任务 ep94–99），从未见过 mixed8 画面；源 episode 不同，无泄漏 | 只能由 VLA 结果判断；不在本计划内补训 encoder |
| `build_stage2_model_from_cfg` 需要 `data_metadata` | 抽取侧要从 ckpt 内还原训练合同（含 qstats 身份），缺键即构造失败 | S1 先在 CPU 上用 ep105 只建模型不前向；不行停下报主会话 |
| 新 MJ 分支未 push 前 S4 无法 lock | git 依赖要求远端可取 | 合并顺序钉死 S1 → push → S4 |
| 注册副本占盘 | ckpt 约 314 MB，`/scratch` 剩约 1.6 T | 可忽略；sha 核对 |
| 恢复副本 PatternLock 只有 195 段 exec | 5 集无 exec latent，上次建库同样处理（无 motion 行即不可见） | G5 记录缺段数，与旧库一致即可 |
| 只读锁误伤 | chmod 范围过大会挡住日志写 | 只锁被跟踪的 `src/`、`scripts/` 文件，记录清单 |
| 增量上传与训练争 IO | 上次 512／motion 两条 run 同期上传无异常 | 脚本自带 `ionice -c 3 nice -n 19` |
| 8 卡加速未到 2× | 测速 1.80× | 用户已接受（「现在这个速度可以接受。」），不调 batch |

## 六、盲区诚实清单

- 吞吐 0.96 s/步是现有 motion 160×768 配置的实测（99 步窗口），新配置（96×1024，memory 608）未实测，以 G7 为准；60k ≈ 16.0 h、80k ≈ 21.4 h 由此推算。
- `mem_encoder.py` 的 `FeatureEncoder.encoder_static` 为 Linear(2048+768→1024) 是读代码推断，未运行核对；motion 路 Linear(1792→1024) 同样未实例化验证，由 G7 的参数树打印核对。
- encoder 在 mixed8 上的表征质量未知（域外，见风险）。
- PatternLock 5 集无 exec latent 的根因未查（沿用上次建库口径）。
- 抽取耗时未实测：约 6.4 万窗、B=1，估十几分钟到一小时，IO 为主。
- G4(b) 端到端阈值 1e-5 沿用草稿，未在 stage2 encoder 上实测；若 FAIL 先查 VAE 段，不改阈值。
- G2 用 source 库 pkl 帧拼窗：已只读核实 `data/0.pkl` 含 `image (256,256,3) uint8`、`step_idx`、`epis_idx`、`is_demo`、`exec_start_idx`，但 pkl 帧与恢复副本当初编码所用 h5 帧逐字节同源未核，G2 用 `input_frames_sha256` 顺带验证。
- 补位调制：MemoryAttention 输出 `mem_mod_vec` 经 `MemoryRMSNorm`（`history_gemma.py:28-43`）作用于 action expert FFN 前后，紧随其后的 `pre_ffw_norm`（`:181-184`）再做一次 RMS 归一化，调制效果受二次归一化影响；这是既有设计，本计划不改、不实测。
- **草稿订正**：① qstats 是 state decoder 的分位数统计（MJ `dataset_paired_wan.py::fit_state_qstats`），encoder affine 用 ckpt 内 `latents_mean/std` buffer；② 客户端 `pack_buffer` 只打包帧，窗口枚举在服务端且已读 yaml，零改动；③ 6.9e-8 的出处是 `1024-rgb-gtxyz-16views-stride2-plan.md:435`，不是 `round4-fp16.md:63`；④ stage2 encoder `forward` 需要 `geometry_valid`，定为全 True；⑤（r2）抽取函数放 `src/motion_jepa/export/`，因 `scripts/` 不进包；⑥（r2）encoder 目录用注册副本而非软链（R15）。

## 七、补位 token 的计算细节（2.7 展开）

- **离线**：`framesamp_dataset.py:320-341 _pad_motion` 把 k 行 emb/pos 右填零到窗口容量 B = budget / num_tokens = 96，`mask[:k]=True`，`times = pad_times(frames, B)` 补位记哨兵（int32 max）；k > B 直接 raise（零截断）。在线 `framesamp_memory.py:250-269 _prepare_motion` 同式。
- **投影**：`percep_mem.py:70-87`：96 位全部过 `motion_pos_proj` Linear(256→768)+silu 与 `motion_encoder_static` Linear(1024+768→1024)，补位也被计算（零输入经 bias 得到非零向量），但只靠 mask 屏蔽，与帧路 padding 同构。
- **排序**：`history_pi0.py:326-361 embed_memory` 把 `input_mask = concat[static_mask, motion_mask]`，与 token 一起按 `mem_order` 重排；`shared/sampling.py:38-58 memory_order` 键 = 时刻×2+类型（帧 0、motion 1）稳定排序，哨兵时刻的补位落尾；序列长 512+96 = 608。
- **注意力**：`history_gemma.py:45-108 MemoryAttention`：Q 来自 action expert token x (B,T,1024)（4 头×256），K/V 来自 608 个 memory token（1 KV 头），带 RoPE；`:97-99` `jnp.where(mask, logits, -2.3819763e38)` 后 softmax，补位权重为 0、也不进分母；输出 `mem_mod_vec (B,T,1024)`。
- **调制**：`history_gemma.py:28-43 MemoryRMSNorm`：Dense(1024→2048) 切成 scale/shift，`normed*(1+scale)+shift`，在 action expert 每层 FFN 前使用（`:177-179`），每层参数独立（`nn.scan`，`:241-255`）。
- **结论**：补位零贡献；序列长固定 608，算力不随 k 变；k 小只是 softmax 在更少 key 上分配。配图由主会话放在 `docs/figures/1026-motion-pad-modulation.svg`（本计划不画）。

## 八、测速记录

| 项 | 值 |
|---|---|
| run / tmux / 日志 | `cc-speed8-motion-b128-20261026-a` / `cc-vla-s2-speed8`（已退出，`EXIT_CODE=0`）/ `v1-store/logs/cc-speed8-motion-b128-20261026-a.log` |
| 起跑 | 2026-10-26 12:43 美东 |
| 入口 | `scripts/dataset/hf_export/mixed8_200_runner/train-motion-80k.sh` + `--num-train-steps 120 --no-wandb-enabled`，`XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| 配置 | 8 卡 fsdp 8、batch 128、现有 motion 160×768 |
| 实测 | 编译约 2 min；第 12→111 步 99 步 95.3 s = 0.96 s/步；对 4 卡 1.73 s/步加速 1.80× |
| 推算 | 60k ≈ 16.0 h、80k ≈ 21.4 h |
| 残留 | `v1-store/train-runs/mme_vla_suite_b128_80k/cc-speed8-motion-b128-20261026-a/` 的 step 119 ckpt，收尾按 R13 删 |
| 用户裁决 | 8 卡 b128，不测 b256；训到 80k，测试取 60k |

## 九、备选 run B：帧路 1 帧／demo 任务 2 帧＋stage2 motion（不在本次开工令内）

**现状（已核实）**：`shared/sampling.py:11-16 even_sampling_indices`：`max_frames=1` → [0]（episode 第 0 帧，含 demo 前缀）、`=2` → [0, t]；step 为全域帧号（含 demo），没有按段选帧机制。`framesamp_dataset.py:122-141` 白名单只认 (512,16,1)/(512,64,1)/(1024|2048|4096,64,1)/ctx，budget 64 或 128 会在 `:137` 抛 ValueError；`:114-116` 要求 modulation↔1024。

**设计（主会话默认，用户可改）**：
- 新 yaml 字段 `frame_select: segment_anchor`（默认 `even`，保持旧行为）。规则：无 demo 段的任务取 [0]（即 exec 起点）；有 demo 段取 [0, es]（demo 首帧＋exec 首帧）。
- budget 统一 128（2 帧×64）；无 demo 任务第二帧位零填充、mask False（复用现有 `static_mask`）。
- 白名单加 (128,64,1)（modulation，可带 motion）。
- 在线 `framesamp_memory.py:273-275 get_frame_sampling_indices`／`policy.py:160-161` 走同一选帧函数。
- motion 路与主 run 完全相同（同库、同 yaml motion 节）。
- 新 yaml `perceptual-framesamp-modul-1frame-8x8-motion-stage2-mixed8.yaml`；run `mixed8-200-m1x8x8-modul-motion-stage2-1024-b128-80k`；bucket `HongzeFu/robomme-vla-mixed8-200-modul-1frame-motion-stage2-1024-80k-v1`。
- S5 可写集合：`src/mme_vla_suite/shared/sampling.py`、`training/framesamp_dataset.py`（白名单与选帧）、`policies/framesamp_memory.py`、`policies/policy.py`（只改选帧调用）、新 yaml、`tests/shared/test_frame_select.py`（新）；S2 合入后才派。
- G11 `FRAME_SELECT_PARITY=PASS`：离线与在线同 step 选帧一致，含无 demo 任务的 mask。
- 预估：memory token 从 608 降到 224，步时应 ≤ 主 run 的 0.96 s/步，未实测；用户「就先8卡跑」，主 run 先跑，B 另取开工令后排在其后。

## 十、交接顺序与状态机登记

- 任务 id：`mixed8-vla-stage2-1024-train`（policy `docs/handoff/current.json`）。登记前问用户一句「是否登记进状态机」；同意则计划 commit 时写入 `planned`。
- 交接前必须完成：G1–G8 判定行（出处 `/scratch/hongze/logs/vla-s2-20261026/*.log` 与 run 档案）。
- 长尾：G9（tmux `cc-vla-s2-inc`、`inc.log`、完成行 `CKPT_UPLOAD=PASS`）；G10（`cc-vla-s2-mjup`、`/scratch/hongze/logs/mj-ckpt-upload-20261026/upload.log`、`MJ_CKPT_UPLOAD=PASS`）。
- 依赖上游：MotionJEPA `stage3-formal-training-v1`（只读复核 ep105 存在且 sha 与 G3 一致）。
- 本任务 `done` 后可开工的下游：60k／80k 的 benchmark 评估（另立计划）、备选 run B（另取开工令）。

## 十一、留档与 commit 纪律

- S1–S4 各经 `--no-ff` 合并占一个项目号（policy `commitV11.38` 起；MJ 接续其 `commitV*` 最近号），子代理提交 `sub/S<n>: <中文>`、body 三项；合并提交 body 按第 11 条六项，含本计划一节全部用户原话。
- 训练留档 `docs/training-doc/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/{launch.md,result.md,records/}`；`records/` 只放 G1–G10 判定行、清洗后日志（`tr '\r' '\n' | grep -vE '%\|'`）、抽取 `export_record.json`、测速日志清洗版与八节测速表；不放 yaml／sh／ckpt。`launch.md` 写明测速 run 残留 ckpt 路径与删除时点。MJ 仓库只在 `docs/training-doc/stage2-full144-1025-a/` 留一行指向本档案的指针（MJ 会话做）。
- Beta 与正式 commit 同小版本号、两个独立 commit，禁止 squash/amend；所有 commit 立即 push，`git status -sb` 无 `ahead`。
- 本轮 tmux 清单（R7）写进 `launch.md`；结束按清单逐个清理。
