# 1026-vla-stage2-memory-1024-train-plan.md — 用阶段 III 最后 ckpt（EMA）把每窗 4×256 拼成 1 个 1024 维 memory token，窗 33／步长 32，训一条 mixed8 VLA 80k

> **权威与授权**：只规划不实施。本版 2026-10-26 美东按用户退回意见（「你这个计划是什么情况？为什么写成现在这么乱七八糟的？我在这个仓库里面有 agent markdown，要求了你要第一部分、第二部分」）推翻草稿结构、按 `AGENTS.md` 第 2 条重写为两部分；草稿里已核实的事实全部保留，另补核了 policy 侧现成建库／在线链路（`scripts/dataset/build_mixed8_v10_motion.py`、`scripts/dataset/wan/`、`policies/framesamp_memory.py`），据此订正草稿三处（见第二部分六节「草稿订正」）。用户要求 Markdown（「你写一个 Markdown 的计划，然后我确认了再开始。」），本次保留 `.md`。口径以本文为准。
> **代码锚点**：policy `v4-256concat1@3861154`（从 `v3-motionmem-pair4@8c319aa` 切出；用户：「然后这次的这个修改从branch v4-256concat1开始」），工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`；MotionJEPA `v12-3decoder@6c79ebb`，工作副本 `/scratch/hongze/MotionJEPA`（只新增一个脚本，另开分支 `v12-3decoder-memexport`，不动 `v12-3decoder`／`main`）。
> **commit 体例**：两仓功能改动 `commitV<x>.<y>: <中文>`（policy 接续 `commitV11.37` → 本轮从 `commitV11.38` 起），文档 `docs:`，交接 `handoff:`；子代理提交 `sub/<编号>: <中文>`，经主会话 `git merge --no-ff <TIP sha>` 合入，合并提交占项目号。
> **外部依赖锚点**：Wan VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，`vae_state_sha256` 前缀 `9980d252`（恢复副本 metadata 原值）；恢复副本 `v1-store/restore/mixed8-200-vla-latents-20261021-a/`，来源 bucket `HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1`，`restore_receipt.json` PASS；encoder ckpt `/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`（sha256 在 G3 现场算并钉进 yaml）。
> **待用户裁决**：只剩开工令。bucket 名已定死（用户：「计划写在根目录，bucket 随便你定，以后不要再来问了。」）。开工令后按第二部分二节分配表派发；开工前先派一个 sonnet 审本计划自相矛盾（`PLAN_CONSISTENCY_REVIEW=`）。

# 第一部分（给人看）

## 一、要做什么与全部运行一览

**一句话**：用 MotionJEPA 阶段 III 最后一份 ckpt（ep105）的 EMA encoder，把 mixed8-200 每个 33 帧窗的 4×256 输出按源时间片 [0,3,5,8] 拼成 1 个 1024 维 token，窗口步长从 16 改为 32、demo 尾窗 padding 规则不变、motion budget 96，离线建新 motion 库，训一条 8 卡 80k VLA（`mme_vla_suite_b128_80k`）；边训边传 VLA ckpt，同时把 encoder／decoder ckpt 传上 HF。不重过 Wan、不跑 4 token 对照臂。

```
 开工 ─┬─ 并行一：改代码  S1(MJ 抽取) ∥ S2(库与 loader) ∥ S3(yaml/启动/导出) → S1 合并 push → S4(sidecar)；每块审→--no-ff 合→合后审→push
       ├─ 并行二（不占卡、立即可跑）：encoder/decoder ckpt 上传 cc-vla-s2-mjup → G10
       ├─ 串行：抽取(GPU0, 读恢复副本 g 偶数窗) → G1 G2 G3 → 打包 → G5 G6 → 在线对拍 G4
       ├─ 短测 8 卡 20 步 → G7 → Beta commit（commitV11.<n>Beta）
       └─ 正式 80k（cc-vla-s2-train，8 卡）∥ 增量上传（cc-vla-s2-inc）→ G8 → 正式驱动全量回读 → G9
```

| 运行 | 内容 | 资源 | tmux | 判据 |
|---|---|---|---|---|
| 抽取＋打包 | MJ `scripts/stage2_memory_export.py` 读 4,995 段恢复副本只取偶数格窗；`pack_motion_store.py pack/verify` 写 `motion-stage2-1024/` | GPU0，约 6.4 万窗；打包 CPU | `cc-vla-s2-export`、`cc-vla-s2-pack` | G1–G3、G5、G6 |
| 短测 | 同配置 `MAX_STEPS=20` | 8 卡 ≤10 min | `cc-vla-s2-smoke` | G7 |
| 正式训练 | run `mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k` | 8 卡 fsdp 8，估 22–24 h | `cc-vla-s2-train` | G8 |
| VLA 上传 | `m8v-ckpt-inc.sh` → bucket `HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1` | CPU/网络 | `cc-vla-s2-inc` | G9 |
| MJ 上传 | 阶段 II＋III ckpt → bucket `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1` | CPU/网络 | `cc-vla-s2-mjup` | G10 |

**用户原话（逐字，2026-10-26）**：①「给出方案 用现在的最后一个ckpt训vla 4个256合并成1个token输入 然后窗口是32 不要再16重叠了 给出方案 然后注意 encoder inference 的时候别忘了 dropout 关掉 然后这个memory本身是不是不再做增强了 robomme官方policy repo怎么做的 然后再看一下wan是不是已经有缓存了」②「然后你一边训练，一边还要把现有的这个 encoder、decoder 的这个 checkpoint 是不是看一下是不是都上传了，然后边训练边上传 VIA 的 checkpoint。」③「你写一个 Markdown 的计划，然后我确认了再开始。」④「你写在 policy learning 的仓库里面，不要写在 motion Java 仓库，因为这个完全是 policy learning 仓库的事情。」⑤「然后这个 memory token 的预算是是九十六还是八十？然后我我然后我确认是用 EMA 权重，然后不跑这个不拼接的这个作为四个 token 的这么一个对照。」⑥「好的，先定为这个九十六，然后你要看一下现在这个 execution，就是 Robo M E 官方仓库的这个 evaluate 是最多是一千三百步，我记得是，然后这一千三百步会怎么影响我们的这个 evaluate 的情况？」⑦「然后你这个 padding 的这个工作还是要做的，就是 demo 的需要有 padding，还是要有一样的。」「你超过十六帧还是十七帧要有一个 padding 的这个，你也要写在这个计划里面。」⑧「计划写在根目录，bucket 随便你定，以后不要再来问了。」⑨「然后这次的这个修改从branch v4-256concat1开始」

**已定口径**：
1. 权重 = ep105 的 `model` 键（`weight_type='ema'`），不用 `model_live`（原话⑤；二节 2.1）。
2. 4×256 按源时间片 [0,3,5,8] 顺序拼接成 (1024,)，不平均；策略侧 `memory_token_dim=1024` 不变（原话①；2.2）。
3. 窗 33 帧（锚点＋未来 32）、`GRID_STRIDE=32`、`segment_start`、`forward`，相邻窗只共享 1 帧（原话①；2.3）。
4. demo 末窗真实帧 ≥17 保留并 `repeat_last` 补到 33，≤16 丢弃；exec 窗 33 帧全真实（原话⑦；2.3）。
5. motion budget 96（原话⑥；2.7）；不跑 4 token 对照臂（原话⑤）。
6. 抽取与在线 sidecar 一律 `model.eval()`＋`torch.no_grad()`＋fp32（autocast 关），G1 证明 dropout 真关（原话①；2.4）。
7. memory 不增强，用干净 latent（等价 aug_id=0、`geometry_valid` 全 True）（原话①；2.5）。
8. 不重过 Wan VAE，直接读恢复副本 g 为偶数的窗（原话①；2.6）。
9. 训练沿用 `mme_vla_suite_b128_80k`（80k、batch 128、pi05_base、EMA 0.999、每 5000 步存 16 份），一条 run 独占 8 卡。
10. 计划与全部 policy 改动在本仓库（原话④），MotionJEPA 只加 `scripts/stage2_memory_export.py`；bucket 名执行方定死如上表（原话⑧）；边训边传 VLA，encoder／decoder 现在就传（原话②）。

## 二、现状与机制

**2.1 ckpt**：`runs/stage2-full144-1025-a/` 有 `checkpoint_epoch_{5..105 步长 5}.pt` 共 21 份，第 106 轮用户停训、无 `checkpoint_final`。ep105 顶层键 `arch`=`wan-latent-stage2-blockwise-pool4-three-decoder`、`fingerprint`、`weight_type='ema'`、`model`（EMA，185 张量，约 19.62M 参数，encoder＋三个 decoder 同文件）、`model_live`、`optimizer`、`config`、`epoch=105`、`global_step=7770`；EMA 与 live 确有差（`encoder.pos_t` 最大差 2.8e-4），所以必须显式取 `model`（保存逻辑 MotionJEPA `src/motion_jepa/models/stage2_model.py:93-117`）。

**2.2 数据链路（逐跳形状／dtype／字节；「改数」指这一跳是否改变数值）**

```
恢复副本 <seg>.bin  每窗 (9,16,32,32) fp32 = 589,824 B   ── 只读、不改数；新旧共用
  │ 新：只取 g=2k 的窗（stride32 ⊂ stride16，见 2.3）       旧：取全部 g（stride16）
  ▼
Stage2WanLatentMotionEncoder.forward(latents, geometry_valid=全 True (B,32,32) bool)   [MJ wan_latent_encoder.py:418-422]
  affine(latents_mean/std, ckpt 内 EMA buffer) → 576 patch(9×8×8, 4×4×16=256 维→input_proj 256)
  ＋4 readout(源时间片 [0,3,5,8] 各 32×32→4×4 掩码均值 256 值→pool_proj 256)  = 580 token × 8 层 blockwise
  → output_proj(final_norm(x[:,576:580])) = (B,4,256) fp32 = 4,096 B/窗      ── 冻结、eval、无梯度
  ▼ reshape(B,1024)（行优先 = 槽 0‖3‖5‖8 顺序拼接，不改数）
新库 motion_token.f32.bin 行 (1,1024) f32 = 4,096 B，约 6.4 万行 ≈ 262 MB（旧库 (768,) 3,072 B×128,342 行 ≈ 380 MiB，grid16）
  ▼ FramesampDataset：motion_emb (96,1024) f32、motion_pos (96,256) f32（PE 前 256 维）、motion_mask (96,) bool
PerceptualMemory：pos→motion_pos_proj Linear(256→768)+silu ‖ emb → motion_encoder_static Linear(1792→1024)
  新可训练参数 1,836,032（旧 Linear(1536→1024) 为 1,573,888），从零初始化   [percep_mem.py:34-44, :85-87]
  ▼ memory (B, 512 帧路 + 96 motion, 1024) 走 modulation（旧 512+160=672）   [history_pi0.py:326-361]
```

**2.3 窗口与 demo padding**：窗口起点按段（demo、exec 各自从 0 起）枚举，`seg_num_chunks = len − (min_real − 1)`（`motion_store.py:136-140`），demo `min_real=17`、exec `min_real=33`。stride32 的起点集合 {32k} 恰是 stride16 起点集合 {16m} 中 m 为偶数的子集，判据（起点 < len − min_real + 1）不变，所以「取恢复副本偶数格」与「直接按 stride32 枚举」逐窗等价，demo 尾窗的 `repeat_last` 补帧已在恢复副本里按同一规则做好。

```
demo 段 len=100（帧 0..99），min_real=17 → 合法起点 < 84
 stride16: 0 16 32 48 64 80          80 窗真实 20 帧(80..99)≥17 → 补 13 帧 repeat_last
 stride32: 0    32    64             64 窗真实 33；96 窗只剩 4 帧 ≤16 → 不存在（丢弃）
exec 段 len=100，min_real=33 → 合法起点 < 68：stride32 = 0 32 64，三窗全真实 33 帧，不补帧
```

⚠ 训练侧 `visible_motion_rows` 现在把 `16m` 写死在全局 `GRID_STRIDE`（`motion_store.py:55,162-185`），在线侧 `framesamp_memory.py:84,218-247` 已读 yaml `motion.stride`；只改 yaml 不改库代码会让训练与在线可见窗不一致，必须按 layout 参数化（三节）。

**2.4 dropout**：encoder 内 dropout 0.05 落在注意力 `to_out` 后与 SwiGLU 输出（MJ `wan_latent_encoder.py:123,135,149,152`，构造默认 `:303`），无 drop_path；DiT 与 state decoder 不参与抽取。`eval()` 让 `nn.Dropout` 恒等；证明方法：同一批窗 `eval()` 下跑两遍逐位相等，且 `train()` 下跑一遍必须不等（证明 dropout 存在且被 eval 关掉，不是本来就没有）。参照 MJ 自己的评估口径 `scripts/stage2_eval.py:51-54`（`eval()`＋`no_grad`＋autocast 关）。

**2.5 memory 不增强的依据**：官方 OpenPI 增强只在 `preprocess_observation(train=True)` 对**当前帧**做（`src/openpi/models/model.py:144-190`：主相机 RandomCrop 0.95→Resize→Rotate ±5°→ColorJitter(0.3,0.4,0.5)，腕相机只 ColorJitter），推理 `history_pi0.py:581` 传 `train=False`；官方 RoboMME policy 无 memory 路。本 fork 的 memory 一直离线从干净帧抽取，不增强、无 dropout（`history_gemma.py:117` 默认 0.0）、无噪声，mask 只管 padding。

**2.6 Wan 缓存可直接复用**：恢复副本 `wan-latents/` 4,995 个 bin（exec 3,195，PatternLock 只有 195；demo 1,800），schema 3、fp32、窗 33、grid16、`segment_start`、demo `min_real 17 repeat_last`，每 bin = num_grid × 589,824 B，共 75,699,191,808 B（约 70.5 GiB），`latent_mode=mode`、tf32 off。阶段 II/III encoder 训练缓存用整 33 帧 fp16 编码，与标准 1+32 编码的 RGB 归一化 MSE 6.9e-8（MJ `1024-rgb-gtxyz-16views-stride2-plan.md:435`），fp16 落盘误差 mean(e²)≤8.96e-8（同文件 `:627`）；所以 fp32 的 1+32 恢复副本可直接喂 encoder，G2 抽样重验。其他缓存不适用：`dataset-rgb-gtxyz-16views-stride2-v1` 是原版 16×100、stride2；mixed8 主库下无 `wan_chunk_latents`。

**2.7 1300 步上限对预算的影响**：官方 eval 每集最多 1300 步（`examples/robomme/eval.py:30,88`，`env_runner.py:26`），超过判失败，对所有模型一样；帧路 8 帧均匀采样不随步数增长，受影响的只有 motion 窗数：

| 段 | 最坏长度 | stride16 窗数（旧） | stride32 窗数（新） |
|---|---|---|---|
| demo（VideoPlaceOrder） | 1247 帧 | 77 | 39 |
| exec（1300 步上限） | 1300 | 80 | 40 |
| 合计 | — | 157（旧预算 160） | **79（新预算 96，余量 17）** |

训练集最坏（PickXtimes ep196）stride16 为 123 窗，stride32 约 62（G6 实算）。budget 80 在评估侧只剩 1 窗余量，96 多 16 个 token、开销可忽略；loader 已有零截断预检（`framesamp_dataset.py:227-233` 超窗即 raise），G6 再按评估上限复算。

## 三、我们这一侧要改什么

非改不可的原因：现有链路三处写死旧口径——训练库 `LAYOUT`／`GRID_STRIDE=16` 是模块全局常量且布局名必须带 `grid16`（`motion_store.py:45-60,101-109`）；loader 合同只认 `(768,)` 旧路与 `adjacent_pair_v1` 四对（`motion_adapter.py:49-80`），且四对分支要求 arch=`wan-latent-v10-st`；在线 sidecar 的 torch 子项目把 MotionJEPA 钉在 v10 提交 `ee719fb`，`load_encoder` 断言 `ARCH_TAG=="wan-latent-v10-st"`、输出 `(8,256)`（`scripts/dataset/wan/wan_motion_infer.py:262-306`）。

| 文件 | 锚点 | 改什么 | 关闭态（旧 yaml） | 开启态（新 yaml） |
|---|---|---|---|---|
| MJ `scripts/stage2_memory_export.py`（新） | `encode_windows` | 读恢复副本偶数格窗 → EMA encoder → (n,1024) 按 pack 契约落盘 | 不存在 | 产出 `motion-tokens-stage2/` |
| `datastore/motion_store.py` | `LayoutSpec`、`LAYOUT_SPECS`、`visible_motion_rows` | `LayoutSpec` 加 `grid_stride`；登记新 layout；网格公式读 spec 步长 | 旧三 layout 行为逐字节不变 | `motion-stage2-1x1024-grid32-demopad17-v1`，(1,1024)，步长 32 |
| `datastore/motion_adapter.py` | `motion_policy_spec` | 新 readout `stage2_pool4_concat_v1`（source (1,1024)、1 token、dim 1024） | 旧两分支不变 | 新分支 |
| `training/framesamp_dataset.py`、`training/dataloader.py` | `check_motion_contract`、stride 断言 | stride 与 spec 比；新 readout 绑 arch／state_key／sha | 不变 | 校验 stage2 身份 |
| `scripts/dataset/pack_motion_store.py` | `read_segment_tokens`、`gather_provenance` | 支持新 layout；latent 格与 token 格 2:1 映射 | 不变 | 只认偶数格 |
| 新 yaml `perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml`＋启动／导出包装 `scripts/training/run_mixed8_stage2_1024_80k.sh` 等 | `motion:` 节 | yaml 见第二部分一节；脚本照 pair4／512 版复制改常量 | — | 新 run |
| `scripts/dataset/wan_stage2/`（新 torch 子项目）＋`policies/motion_client.py`、`policy_config.py` | sidecar | 独立 `pyproject`/`uv.lock` 钉新 MJ 分支；`motion.encoder_kind` 选 sidecar | 旧 v10 sidecar 原样 | 回 4,096 B |

`percep_mem.py` 与客户端 `examples/robomme/utils.py:38 pack_buffer`（只打包帧、不枚举窗）零改动；服务端窗口枚举已参数化，只靠 yaml `stride: 32`。

## 四、验收

| 判定 | 查什么 | 怎么查 | 过了说明什么（为什么能成立） |
|---|---|---|---|
| G1 `DROPOUT_OFF` | dropout 真关 | 64 窗 eval 两遍、train 一遍 | eval 逐位相等＋train 不等 ⇒ 存在且被关；同卡同 batch 同 kernel，fp32 TF32 关 ⇒ 可逐位 |
| G2 `WAN_EQUIV` | 恢复副本 = 训练口径 | 64 窗（含 demo 尾窗）用 MJ `wan_fp16_full33.py` 重编比 | 归一化 MSE 与 2.6 同量级 ⇒ 无需重过 Wan |
| G3 `EXPORT_PARITY` | 导出函数 = MJ 评估路径 | 32 个 aug0 验证窗，`encode_windows` vs `stage2_eval` 口径 encoder | 同权重同输入同精度 ⇒ `max_abs=0`；不为 0 即导出路径有偏 |
| G4 `ONLINE_OFFLINE_PARITY` | 在线 = 离线 | 64 窗：编码段同 latent 比；端到端帧→VAE→encoder 比 | 编码段同一函数 B=1 ⇒ 逐位；端到端多一次 VAE ⇒ 阈值 1e-5 |
| G5 `VERIFY_MOTION` | 库完整、padding 正确 | `pack verify` + 逐段 real_frames 断言 | 行数＝偶数格窗数、全有限、demo 尾窗 real∈[17,33]、exec real=33 |
| G6 `TRAIN_ES_BOUND` | 预算够 | 遍历训练集与评估 1300 上限 | 最坏 ≤96 ⇒ 零截断 |
| G7 `SMOKE_TRAIN` | 能训 | 8 卡 20 步 | loss 有限、无 OOM，记 s/步 |
| G8 训练 `EXIT_CODE=0` | 训完 | 日志尾行＋16 个 step 目录 | — |
| G9 `CKPT_UPLOAD` / G10 `MJ_CKPT_UPLOAD` | 远端可恢复 | 全量 sha256 回读 | 未 PASS 不写「已备份」 |

## 五、步骤

| 阶段 | 内容 | 判据 |
|---|---|---|
| 0 | 用户开工令；sonnet 审计划 | `PLAN_CONSISTENCY_REVIEW=PASS` |
| 1 | 起 MJ ckpt 上传（与 2–6 并行） | G10 |
| 2 | 派 S1、S2、S3 → 合 S1、push MJ 分支 → 派 S4 → 依次合 S2、S3、S4 | 每块 `PRE_MERGE_REVIEW=PASS`、`POST_MERGE_REVIEW=PASS` |
| 3–5 | 抽取＋前三闸 → 打包＋校验＋预算 → 在线 sidecar 对拍 | G1–G3 → G5、G6 → G4 |
| 6 | 短测 → Beta commit → 正式起跑＋增量上传 | G7 → G8 |
| 7 | 正式导出驱动全量回读；留档；handoff | G9、`HANDOFF_WRITE=PASS` |

## 六、子代理分工与合并（简述）

拆四个写入块，各在独立 worktree 由 opus 写：S1 只在 MotionJEPA 新分支加抽取脚本；S2 管 policy 库格式、loader 合同与打包；S3 管新 yaml、启动脚本和两份导出包装；S4 管在线 sidecar（新 torch 子项目＋客户端选择），因要把新 MJ 分支 sha 钉进 `uv.lock`，必须等 S1 合并 push 后才派。合并顺序 S1 → S2 → S3 → S4，每块合前派一个 sonnet 钉死 BASE/TIP 审 diff，合后跑定向短测再 push。抽取、打包、训练、上传由主会话或 opus 运行型代理在主检出起 tmux，运行代理不改代码。逐文件逐函数的改动、分配表、闸门判定行全文、runbook 命令、风险与盲区见第二部分。

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- R1 不重过 Wan VAE：一切 latent 只读恢复副本 `v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents/`（只读，不改不删）；G2 的 64 窗重编是比对用，结果只进 `records/`。
- R2 不改 encoder 权重：抽取与 sidecar 只读 `checkpoint_epoch_105.pt` 的 `model` 键，strict 加载、`requires_grad_(False)`、`eval()`、`torch.no_grad()`、`torch.autocast(enabled=False)`、`torch.backends.cuda.matmul.allow_tf32=False`、`cudnn.allow_tf32=False`；不写回任何 MJ `runs/` 文件。
- R3 memory 不增强：`geometry_valid` 恒为全 True `(B,32,32)` bool（等价 aug_id=0：MJ `dataset_paired_wan.py::derive_geometry_masks` 中恒等单应的 `common_valid` 全真）；不加噪声、不做帧增强。
- R4 不动 `main` 与 `v12-3decoder`：MJ 侧只在新分支 `v12-3decoder-memexport`（自 `6c79ebb`）提交，push 该分支；policy 侧只在 `v4-256concat1` 及其 `worktree-agent-*` 子分支。
- R5 旧口径逐字节不变：旧 layout（`motion-768-grid16-v1`、`motion-768-grid16-demopad17-v1`、`motion-v10-8x256-grid16-demopad17-v1`）、旧 yaml、旧 v10 sidecar 子项目 `scripts/dataset/wan/`（含其 `uv.lock`、`SOURCE_PIN.json`）零 diff。
- R6 训练期代码冻结与只读锁沿用 1016 做法：Beta commit 后到 G8 期间不 commit、不合并；被跟踪的 `src/`、`scripts/` 按 `v1-store/m8v-scripts/m8v-chmod-before.txt` 体例先记权限再 `chmod a-w`，训完按记录恢复；开发转开发副本（`AGENTS.md` 第 14 条覆盖项）。
- R7 tmux 只用前缀 `cc-vla-s2-`：`cc-vla-s2-export`、`cc-vla-s2-pack`、`cc-vla-s2-smoke`、`cc-vla-s2-train`、`cc-vla-s2-inc`、`cc-vla-s2-mjup`；清理只按此清单 `tmux kill-session -t '=名'` 逐个，前后各 `tmux ls`。
- R8 不进写集：`third_party/**`、根 `pyproject.toml`、根 `uv.lock`（0901 红线 10）、`scripts/dataset/wan/**`、`AGENTS.md`、`CLAUDE.md`、`greatlakes.md`、`.agentmetarules`、`v1-store/**`（`m8v-ckpt-inc.sh` 不改只换参数）。本仓库无第 21 条受保护目录。
- R9 不处理 rgb-gtxyz 上传失败：`/scratch/hongze/logs/rgb-gtxyz-full-20261024/upload.log` 末行 `HF_VERIFY=FAIL error=ValueError`、`EXIT_CODE=1`（进程 1675039、1678537 已不存在，MJ handoff 仍写 running），属 Codex 任务长尾，只在汇报里报告。
- R10 新 run、新库、新 bucket 全新：run 目录、`motion-stage2-1024/`、两个 bucket 起跑前 `ls -ld`／`hf buckets list HongzeFu` 确认不存在；不带 `--force`、不 `overwrite`。
- R11 任一闸门 FAIL 停下交用户；不放宽阈值；崩溃不自动 resume。
- R12 子代理 worktree 内只跑 ≤5 分钟 CPU 测试，环境取法：policy `UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`（先打印 `mme_vla_suite.__file__` 确认在 worktree）；MJ `UV_PROJECT_ENVIRONMENT=/scratch/hongze/MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`。GPU 验收合并后主会话串行跑。

## 一、逐文件改动清单

### A. MotionJEPA（分支 `v12-3decoder-memexport`，S1）

**新文件 `scripts/stage2_memory_export.py`**（现有能力：`build_stage2_model_from_cfg(cfg, *, data_metadata)` 于 `src/motion_jepa/models/factory.py:150`、`load_stage2_checkpoint(model, checkpoint, *, weights="main")` 于 `stage2_model.py:137`；拟新增接口如下）

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

- CLI：`--ckpt <abs> --expected-sha256 <64hex> --latents <恢复副本 wan-latents> --manifest <mixed8 episode_manifest.json> --out <v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion-tokens-stage2> --grid-subsample 2 --batch 1 --device cuda:0 [--max-segments N] [--check-dropout 64] [--resume]`。`--batch` 固定 1（与 sidecar 同形，见 G4 逐位依据）；`--max-segments` 只冒烟。
- 输出（按 policy `pack_motion_store.py::read_segment_tokens` 现有契约，不另造 npz）：每段 `<key>.f32.bin`（`num_grid32 × 4096 B`）、`<key>.f32.bin.sha256`、`<key>.metadata.json`（`layout`、`row_shape=[1,1024]`、`dtype`、`token_dim=1024`、`token_bytes=4096`、`bytes`、`num_grid`、`sha256`、`source_grid=[0,2,4,…]`、`real_frames`、`input_block_sha256[]`（逐窗 589,824 B 的 sha256）、`encoder{run_name, checkpoint_name, epoch, state_key, weight_type, arch, checkpoint_sha256, fingerprint_sha256}`、`numerics{fp32, tf32:off, batch:1, torch, cuda, gpu}`）；根下 `export_record.json`，末行 `STAGE2_EXPORT=PASS segments=<n> rows=<n>`。
- `--check-dropout 64` 产出 G1 判定行；不改任何既有 MJ 文件。测试 `tests/test_stage2_memory_export.py`：CPU 上随机初始化小 batch 验证形状、槽序（`out[:,256*j:256*(j+1)] == enc_out[:,j]`）、`training=False` 断言、eval 两遍逐位相等、train 模式不等。

### B. policy（分支 `v4-256concat1`）

**S2 `src/mme_vla_suite/datastore/motion_store.py`**
- `LayoutSpec` 增字段 `grid_stride: int = 16`（排在 `row_shape` 之后，旧三条登记值不变）；`LAYOUT_SPECS` 新增 `"motion-stage2-1x1024-grid32-demopad17-v1": LayoutSpec(17, 33, "repeat_last", (1, 1024), grid_stride=32)`。
- 常量 `STAGE2_LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"`；`LAYOUT` 保持旧值（R5）；`:106-109` 的布局名检查改为逐条 `f"-grid{spec.grid_stride}-" in name`。
- `seg_num_grid`、`segment_grid_starts`、`max_visible_count` 增 `stride` 参数（默认 `GRID_STRIDE`）；`visible_motion_rows` 的 `s = GRID_STRIDE*m`／`u = GRID_STRIDE*m` 改为 `entry.spec.grid_stride*m`；`build_index_entries` 按 spec 步长算 num_grid。
- 测试 `tests/datastore/test_motion_store_stage2.py`（新）：旧三 layout 对 `BASE` 的 `visible_motion_rows` 在 1,000 组 (seg_len, t) 上逐元素相同；新 layout 的起点集合等于旧 grid16 偶数格子集；demo len=100 例出 {0,32,64}。

**S2 `src/mme_vla_suite/datastore/motion_adapter.py`**
- 新常量 `STAGE2_READOUT = "stage2_pool4_concat_v1"`、`STAGE2_SHAPE = (1, 1024)`；`motion_policy_spec` 在 `readout == STAGE2_READOUT` 时要求 `source_shape==[1,1024]`、`num_tokens==1`、`dim==1024`、`budget>0`，返回 `MotionPolicySpec((1,1024), STAGE2_READOUT, 1, 1024, budget, budget, (STAGE2_LAYOUT,))`；四对与旧分支逐字不变。

**S2 `src/mme_vla_suite/training/framesamp_dataset.py`、`training/dataloader.py`**
- `check_motion_contract`（`:68`）新增分支：`readout==STAGE2_READOUT` 时要求 `enc.arch=="wan-latent-stage2-blockwise-pool4-three-decoder"`、`enc.state_key=="model"`、`enc.weight_type=="ema"`、`enc.checkpoint_sha256==mcfg.checkpoint_sha256`。
- `framesamp_dataset.py:156` 与 `dataloader.py:51` 的 `int(mcfg.stride) == ms.GRID_STRIDE` 改为 `== meta.spec.grid_stride`（loader 内已有 `self._motion_meta`）。
- `__getitem__` 取行后 `(k,1,1024)` → `reshape(k,1024)` 再进 `_pad_motion`（`num_tokens==1` 不走重复分支），输出 `motion_emb (96,1024)`。
- `dataloader.py::_parse_source_run` 不改（`stage2-full144-1025-a/checkpoint_epoch_105.pt#model` 可解析）。
- 测试：`tests/training/test_framesamp_motion_stage2.py`（新）用 3 episode 夹具库验证形状、mask、错 arch／错 sha／错 stride 均 raise；旧 768 与 pair4 夹具输出对 `BASE` 逐字节相同。

**S2 `scripts/dataset/pack_motion_store.py`**
- `read_segment_tokens`：`spec.row_shape == (1,1024)` 时套同一 contract 校验（layout 用 `STAGE2_LAYOUT`）。
- `gather_provenance`：新增参数 `latent_grid_subsample: int = 1`；为 2 时要求 `len(input_block_sha256)==_ng`、latent 文件 `st_size == latent_num_grid*CHUNK_BYTES` 且 token 第 j 行对应 latent 块 `2j`，逐块 sha256 现场比对；encoder 组取 token metadata 的 `encoder`，VAE 组沿用 `load_vae_compat`。CLI `pack` 新增 `--layout {legacy,stage2}`、`--latent-grid-subsample`；`verify` 不变（逐行 memcmp）。
- 新增 `verify` 附加检查 `--check-padding`：逐段从 latent metadata 取 `real_frames`，断言 demo 行 `17 ≤ real ≤ 33`、exec 行 `real == 33`、demo 段最后合法起点后无行；输出并入 G5。

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
- `scripts/training/run_mixed8_stage2_1024_80k.sh`（新，照 `run_mixed8_pair4_80k.sh` 88 行改 `HC=`、`RUN_NAME` 默认禁空、`TRAIN_HEAD` 字面量、`fsdp_devices=8`、`CUDA_VISIBLE_DEVICES=0-7`；`TRAIN_CMD=` 回显保留）；`scripts/training/check_mixed8_stage2_1024_launch.py`（新，照 `check_mixed8_pair4_launch.py` 改 yaml 名、库路径、budget、readout）。
- `scripts/dataset/hf_export/run_mixed8_200_vla_stage2_1024_80k_ckpt_export.sh`（新，照 24 行的 `run_mixed8_200_vla_512_80k_ckpt_export.sh` 改 RUN 与 bucket）；`scripts/dataset/hf_export/run_stage2_pool4_encoder_ckpt_export.sh`（新，照 `run_mixed8_200_encoder_ckpt_export.sh` 改白名单：阶段 III 21 份 ckpt＋`config.yaml`＋`train_metrics_epoch.jsonl`＋`motion_ablation/`，阶段 II `stage2-scale-20261025-d` 的 `checkpoint_epoch_{4,8}.pt`、`checkpoint_final.pt`、`config.yaml`、`train_metrics_epoch.jsonl`；README 写明 `model`=EMA、`model_live`=live 与 fingerprint）。

**S4 在线 sidecar**
- 新 torch 子项目 `scripts/dataset/wan_stage2/`：`pyproject.toml` 依赖与 `scripts/dataset/wan/pyproject.toml` 相同的 torch 2.9.0／diffusers 0.39.0／numpy 2.4.4 等，`motion-jepa @ git+https://github.com/hongzefu/MotionJEPA@<S1 合并后 push 的 40 位 sha>`；独立 `uv.lock`；venv `UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2`。不改旧子项目（R5，理由：旧锁文件钉 v10 `ee719fb`，换 pin 会改变旧 run 的 sidecar 环境；按全局 uv 规则，独立依赖另建 `pyproject`，不用 workspace）。
- `scripts/dataset/wan_stage2/stage2_sidecar.py`（新）：协议沿用 `src/mme_vla_suite/policies/motion_protocol.py`（按文件加载）；VAE 段按文件加载 `scripts/dataset/wan/wan_motion_infer.py` 的 `load_vae`、`encode_chunk`（只读复用，不复制数值语句）；encoder 段 `from scripts.stage2_memory_export import load_ema_encoder, encode_windows`（经 MJ 包路径），B=1；回包 1024 个 f32 = 4,096 B；`--stub` 返回 `np.full(1024, 起点)`。
- `src/mme_vla_suite/policies/motion_client.py`：`MotionEncoderClient.__init__` 增 `encoder_kind: str = "v10"`；`"stage2_pool4_concat"` 时 argv 用 `--project scripts/dataset/wan_stage2 … stage2_sidecar.py`、`--ckpt <MJ 绝对路径>`、`--expected-sha256`，期望回包维度 1024；`"v10"` 分支逐字不变。
- `src/mme_vla_suite/policies/policy_config.py:132-148`：把 `mcfg.get("encoder_kind","v10")` 传给客户端；其余不变。`framesamp_memory.py` 不改（`:208` 已按 `motion_dim` 校验回包形状）。
- 测试：`tests/policies/test_motion_client_stage2.py`（新）stub 档走完整 IPC：起点帧号正确、回包 4,096 B、旧 `encoder_kind` 默认 argv 与 `BASE` 相同。

## 二、子代理分配表

`BASE` = policy `3861154`（S1 为 MJ `6c79ebb`）。派发前核对：全局与项目 `settings.json` 的 `worktree.baseRef=head`；`git check-ignore -q .claude/worktrees/probe`；`git worktree list` 存档；主检出 `git status --short --ignore-submodules=dirty` 记录。

| 子任务 | 目标 | 可写文件集合 | 禁触路径 | 接口契约与依赖 | 合并顺序 | 验收命令与判定行（在哪、什么环境） | 资源 | 共享文件归属 |
|---|---|---|---|---|---|---|---|---|
| S1（opus＋worktree，**MotionJEPA 仓库**新分支 `v12-3decoder-memexport`） | 抽取脚本 | `scripts/stage2_memory_export.py`、`tests/test_stage2_memory_export.py` | MJ 其余一切文件、`runs/**` | `load_ema_encoder`、`encode_windows`、token 文件契约（一节 A） | 1；合并后主会话 push 该分支并记 sha | MJ worktree，R12 环境：`pytest tests/test_stage2_memory_export.py -q` passed | CPU | 无 |
| S2（opus＋worktree） | 库格式、loader 合同、打包 | `motion_store.py`、`motion_adapter.py`、`framesamp_dataset.py`、`dataloader.py`、`pack_motion_store.py`、`tests/datastore/test_motion_store_stage2.py`、`tests/training/test_framesamp_motion_stage2.py` | R8；其余 `src/**` | `STAGE2_LAYOUT`、`STAGE2_READOUT`、`LayoutSpec.grid_stride`、`--latent-grid-subsample`、`--check-padding` | 2 | policy worktree，R12：两份新测试＋`tests/datastore -q` 全 passed | CPU | `motion_store.py` 唯一写者 |
| S3（opus＋worktree） | yaml、启动、导出包装 | 新 yaml、`run_mixed8_stage2_1024_80k.sh`、`check_mixed8_stage2_1024_launch.py`、两份导出包装 | R8；`src/**` 的 .py | 读 S2 常量名（只按名字引用，不改 S2 文件） | 3（在 S2 后） | `bash -n` 三脚本；S2 合入后主会话复跑：新 yaml 经 `motion_policy_spec` 解析通过 | CPU | yaml 唯一写者 |
| S4（opus＋worktree，S1 push 后派） | 在线 sidecar | `scripts/dataset/wan_stage2/**`（新）、`motion_client.py`、`policy_config.py`、`tests/policies/test_motion_client_stage2.py` | R8（含 `scripts/dataset/wan/**`） | `encoder_kind`；回包 4,096 B；依赖 S1 sha | 4 | policy worktree：stub 测试 passed；`uv lock --project scripts/dataset/wan_stage2` 成功 | CPU | `policy_config.py` 唯一写者 |
| 审查 RV-S1…RV-S4（sonnet，只读） | 合前审 | 无 | 一切 | 钉 `REVIEW_BASE`/`REVIEW_TIP` | 各块合前 | `PRE_MERGE_REVIEW=PASS base= tip= files= commits= findings=` | — | — |
| 主会话自做 | yaml 填 sha、Beta commit、handoff | yaml `checkpoint_sha256` 一行、`docs/handoff/current.json`、留档 | — | 依赖 G3 | 打包后 | — | — | 主会话 |
| 运行型 X1（opus，主检出，不加 worktree） | 抽取 | 只写 `motion-tokens-stage2/` 与日志 | 一切代码 | 命令照四节 2；起跑判据首段 `metadata.json` 落盘 | 合并后 | `STAGE2_EXPORT=PASS`、G1 行 | GPU0 | 账目归主会话 |
| 运行型 X2（opus，主检出） | 正式训练＋增量上传 | run 目录、export 目录 | 一切代码 | 命令照四节 6–7；起跑判据日志出现 step 1 loss 与 `TRAIN_CMD=` | Beta 后 | 首个 loss 行有限 | GPU0–7 | 同上 |
| 运行型 X3（opus，主检出） | MJ ckpt 上传 | export 目录 | 一切代码 | 命令照四节 1 | 开工即派 | `MJ_CKPT_UPLOAD=` 行 | CPU | 同上 |

合并：`git merge --no-ff <TIP sha> -F <scratchpad 消息>` → `POST_MERGE_REVIEW=PASS merge=<sha> tests=<…> files=<n>`（定向测试＋`git diff --name-only` 与上表核对＋R5 零 diff 检查 `git diff --quiet <BASE> HEAD -- scripts/dataset/wan third_party pyproject.toml uv.lock`）→ push → 清理 worktree（`git worktree remove`、`git branch -d`）。

## 三、闸门总表

| 闸门 | 命令 | 判定行与阈值 |
|---|---|---|
| G1 | `stage2_memory_export.py … --check-dropout 64`（GPU0） | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0 train_vs_eval_max_abs=<x>`；要求前者 =0 且后者 >0 |
| G2 | MJ `scripts/dataset-build/wan_fp16_full33.py` 对 64 窗（每任务 4，含 ≥16 个 demo 尾窗）重编，按 `latents_std` 归一化 | `WAN_EQUIV=PASS windows=64 worst_mse=<x> worst_max=<y>`；阈值 worst_mse ≤1e-6、worst_max ≤0.01（与 MJ 计划 K4 同阈） |
| G3 | 32 个 rgb-gtxyz val aug0 窗：`encode_windows` vs `stage2_eval.py::evaluation_context` 下 `model.encoder(rgb_raw.float(), geometry_valid)`；同时断言该批 `geometry_valid` 全 True | `EXPORT_PARITY=PASS windows=32 max_abs=0 geometry_all_true=1 ckpt_sha256=<64>` |
| G4 | 64 窗：(a) 同 latent 经 sidecar 编码段 vs 库行；(b) 原始 33 帧经 sidecar 端到端 vs 库行 | `ONLINE_OFFLINE_PARITY=PASS windows=64 encoder_max_abs=0 e2e_max_abs=<x>`；e2e ≤1e-5 |
| G5 | `pack_motion_store.py verify --store … --check-padding` | `VERIFY_MOTION=PASS scanned=<rows> mismatches=0 nonfinite=0 demo_tail_padded=<n> demo_tail_dropped=<n> exec_padded=0` |
| G6 | loader 构造时的零截断预检（`framesamp_dataset.py:227-233`）打印训练集最大可见窗数＋主会话一次性脚本按 `ms.seg_num_grid(len, min_real, stride=32)` 复算 1300 步上限 | `TRAIN_ES_BOUND=PASS train_max=<n> eval_max=<n> budget=96`；任一 >96 FAIL |
| G7 | `MAX_STEPS=20 bash scripts/training/run_mixed8_stage2_1024_80k.sh` | `SMOKE_TRAIN=PASS steps=20 loss_finite=1 oom=0 sec_per_step=<x>` |
| G8 | 正式训练日志 | 尾行 `EXIT_CODE=0`，且 `ls …/<RUN>/` 含 5000…75000、79999 共 16 个 step |
| G9 | `run_mixed8_200_vla_stage2_1024_80k_ckpt_export.sh`（`SMOKE=1` → 全量） | `CKPT_UPLOAD=PASS steps=16 files=<n> sha256_mismatch=0 public=1`；`INC_END … done=16` 只是增量结束，不是完成信号 |
| G10 | `run_stage2_pool4_encoder_ckpt_export.sh` | `MJ_CKPT_UPLOAD=PASS files=<n> bytes=<n> sha256_mismatch=0 public=1` |
| 合并 | 见二节 | `PRE_MERGE_REVIEW=PASS`、`POST_MERGE_REVIEW=PASS` |

G1／G3／G4(a) 能要求逐位的理由：同一 A100、同一 torch 2.9.0、同一函数、B=1 固定 kernel 形状、fp32 且 TF32 关、`eval()` 后无随机算子；G4(b) 多一次 VAE 编码（帧→latent），沿用阈值。G2 不要求逐位，因整 33 帧 fp16 与 1+32 fp32 本就是两种编码。

## 四、runbook（主会话）

1. **MJ ckpt 上传**（开工即起，与下面并行）：`tmux new-session -d -s cc-vla-s2-mjup "set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/dataset/hf_export/run_stage2_pool4_encoder_ckpt_export.sh 2>&1 | tee /scratch/hongze/logs/mj-ckpt-upload-20261026/upload.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/logs/mj-ckpt-upload-20261026/upload.log"`（该脚本由 S3 交付；S3 未合入前不起）。上传前先 `hf buckets list HongzeFu` 确认 bucket 不存在，结束后 MJ handoff `stage3-formal-training-v1` 长尾「checkpoint 训练中增量上传 bucket」由 MJ 会话更新（本会话只读，按第 28 条发消息）。
2. **抽取**（S1、S2 合入后）：先 `--max-segments 4` 冒烟（≤5 分钟，`run_in_background`），再
   `tmux new-session -d -s cc-vla-s2-export "bash /scratch/hongze/logs/vla-s2-20261026/export.sh"`，脚本内：`cd /scratch/hongze/MotionJEPA && UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/stage2_memory_export.py --ckpt runs/stage2-full144-1025-a/checkpoint_epoch_105.pt --expected-sha256 <sha> --latents /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents --manifest <mixed8 episode_manifest.json> --out /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion-tokens-stage2 --grid-subsample 2 --batch 1 --check-dropout 64`，日志三件套。随后 G2、G3。
3. **打包**：`tmux new-session -d -s cc-vla-s2-pack "bash …/pack.sh"`，`uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --manifest … --tokens …/motion-tokens-stage2 --latents …/wan-latents --out …/motion-stage2-1024 --encoder-run-dir /scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a --raw-dir … --restore-receipt …/restore_receipt.json --expected-ckpt-sha256 <sha>`，再 `verify --store …/motion-stage2-1024 --check-padding` → G5；G6。主会话把 sha 填进 yaml 并 commit（`commitV11.<n>: …`）。
4. **在线对拍**（S4 合入后）：`UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan-stage2 uv sync --project scripts/dataset/wan_stage2`，起 sidecar 跑 64 窗 → G4。
5. **短测**：`RUN_NAME=cc-vla-s2-smoke-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<HEAD> bash scripts/training/run_mixed8_stage2_1024_80k.sh`（tmux `cc-vla-s2-smoke`）→ G7；跑完删该临时 run（第 6 条）。
6. **Beta 与正式起跑**：建 `docs/training-doc/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/launch.md`（①–⑥节、tmux 清单）→ `commitV11.<n>Beta: …`，`git status --short` 为空 → 写 `AGENTMETARULES_HOLD`（freeze＋sha）→ R6 只读锁 →
   `tmux new-session -d -s cc-vla-s2-train "set -o pipefail; RUN_NAME=mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k TRAIN_HEAD=<Beta sha> PYTHONUNBUFFERED=1 bash scripts/training/run_mixed8_stage2_1024_80k.sh 2>&1 | tee /scratch/hongze/logs/vla-s2-20261026/train.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/logs/vla-s2-20261026/train.log"`。
7. **增量上传**：起跑后立即 `tmux new-session -d -s cc-vla-s2-inc "set -o pipefail; bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1 mixed8-200-vla-stage2-1024-80k <allowlist 相对路径> cc-vla-s2-train 2>&1 | tee /scratch/hongze/logs/vla-s2-20261026/inc.log; echo \"EXIT_CODE=\$?\" >> /scratch/hongze/logs/vla-s2-20261026/inc.log"`（allowlist 沿用 512 版驱动所用那份）。
8. **Monitor**（每份日志一个）：训练 `tail -n +1 -F …/train.log | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "EXIT_CODE=|Traceback|out of memory|CUDA out of memory|Killed|nan|NaN|Error|step [0-9]+000\b"`；增量 `-E "INC_END|EXIT_CODE=|FAIL|错误|Traceback|\.done"`；抽取 `-E "STAGE2_EXPORT=|DROPOUT_OFF=|EXIT_CODE=|Traceback|NaN|非有限"`；MJ 上传 `-E "MJ_CKPT_UPLOAD=|HF_VERIFY=|EXIT_CODE=|Traceback|FAIL"`。
9. **收尾**：G8 后解除只读锁、删 `AGENTMETARULES_HOLD`；跑 G9 正式驱动；写 `result.md`（吞吐、loss、判定行）→ `commitV11.<n>: …`；handoff。

## 五、风险登记

| 项 | 说明 | 处置 |
|---|---|---|
| encoder 域外 | 阶段 II/III 训练源是原版 16×100（val = 每任务 ep94–99），从未见过 mixed8 画面；源 episode 不同，无泄漏 | 只能由 VLA 结果判断；不在本计划内补训 encoder |
| `build_stage2_model_from_cfg` 需要 `data_metadata` | 抽取侧要从 ckpt 内还原训练合同（含 qstats 身份），缺键即构造失败 | S1 先在 CPU 上用 ep105 只建模型不前向；不行停下报主会话 |
| 新 MJ 分支未 push 前 S4 无法 lock | git 依赖要求远端可取 | 合并顺序钉死 S1 → push → S4 |
| 恢复副本 PatternLock 只有 195 段 exec | 5 集无 exec latent，上次建库同样处理（无 motion 行即不可见） | G5 记录缺段数，与旧库一致即可 |
| 旧库与新库同源 | 新库行必须与 framesamp-8x8 同一 manifest | `ms.check_same_source` 已在 loader 内 |
| 只读锁误伤 | chmod 范围过大会挡住日志写 | 只锁被跟踪的 `src/`、`scripts/` 文件，记录清单 |
| 增量上传与训练争 IO | 上次 512／motion 两条 run 同期上传无异常 | 脚本自带 `ionice -c 3 nice -n 19` |

## 六、盲区诚实清单

- 吞吐 8 卡约 1.0 s/步、80k 约 22–24 h 是由上次 4 卡 1.718／1.7346 s/步推的估值，未实测；以 G7 为准。
- `mem_encoder.py` 的 `FeatureEncoder.encoder_static` 为 Linear(2048+768→1024) 是读代码推断，未运行核对；motion 路 Linear(1792→1024) 同样未实例化验证，由 G7 的参数树打印核对。
- encoder 在 mixed8 上的表征质量未知（域外，见风险）。
- PatternLock 5 集无 exec latent 的根因未查（沿用上次建库口径）。
- 抽取耗时未实测：约 6.4 万窗、B=1、19.62M 参数级模型，估十几分钟到一小时，IO 为主。
- G4(b) 端到端阈值 1e-5 沿用草稿，未在 stage2 encoder 上实测过；若 FAIL 先查 VAE 段，不改阈值。
- **草稿订正**（本版核实）：① 草稿写「affine 统计量取 qstats.json」——错，qstats 是 state decoder 的分位数统计（MJ `dataset_paired_wan.py::fit_state_qstats`），encoder affine 用 ckpt 内 `latents_mean/std` buffer；② 草稿写「eval 端 `pack_buffer` 的窗口枚举改 stride 32」——客户端 `pack_buffer` 只打包帧，窗口枚举在服务端且已读 yaml，零改动；③ 草稿写 6.9e-8 出自 `round4-fp16.md:63`——该行是另一组数字，正确出处是 `1024-rgb-gtxyz-16views-stride2-plan.md:435`。另：stage2 encoder `forward` 需要 `geometry_valid` 参数（草稿未提），本计划定为全 True。

## 七、交接顺序与状态机登记

- 任务 id：`mixed8-vla-stage2-1024-train`（policy `docs/handoff/current.json`）。登记前问用户一句「是否登记进状态机」；同意则计划 commit 时写入 `planned`。
- 交接前必须完成：G1–G8 判定行（出处 `/scratch/hongze/logs/vla-s2-20261026/*.log` 与 run 档案）。
- 长尾：G9（tmux `cc-vla-s2-inc`、`inc.log`、完成行 `CKPT_UPLOAD=PASS`）；G10（`cc-vla-s2-mjup`、`/scratch/hongze/logs/mj-ckpt-upload-20261026/upload.log`、`MJ_CKPT_UPLOAD=PASS`）。
- 依赖上游：MotionJEPA `stage3-formal-training-v1`（只读复核 ep105 存在且 sha 与 G3 一致）。
- 本任务 `done` 后可开工的下游：该 VLA 的 benchmark 评估（另立计划）。

## 八、留档与 commit 纪律

- S1–S4 各经 `--no-ff` 合并占一个项目号（policy `commitV11.38` 起；MJ 接续其 `commitV*` 最近号），子代理提交 `sub/S<n>: <中文>`、body 三项；合并提交 body 按第 11 条六项，含本计划一节全部用户原话。
- 训练留档 `docs/training-doc/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/{launch.md,result.md,records/}`；`records/` 只放 G1–G10 判定行、清洗后日志（`tr '\r' '\n' | grep -vE '%\|'`）、抽取 `export_record.json`；不放 yaml／sh／ckpt。MJ 仓库只在 `docs/training-doc/stage2-full144-1025-a/` 留一行指向本档案的指针（MJ 会话做）。
- Beta 与正式 commit 同小版本号、两个独立 commit，禁止 squash/amend；所有 commit 立即 push，`git status -sb` 无 `ahead`。
- 本轮 tmux 清单（R7）写进 `launch.md`；结束按清单逐个清理。
