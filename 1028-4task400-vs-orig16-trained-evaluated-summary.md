# 旧 4 任务×400 集 与 官方 16×100：已训练、已评估结果对照（留档）

> **本文是留档，不是计划。** 只收录已经训练完成、已经评估出结果的 encoder 与 VLA；尚未训练或尚未评估的计划（如 `1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md`）不在本文范围。
> 用户原话（2026-10-28）：「总结这个 encoder VLA 训练和数据集的这个差异……我只关心旧的这个训练的四个 task，每个 task 四百 episode 和这一次的官方的数据集的训练，帮我整合成一个文档写在这个根目录」「你说你现在已经训练好的和现在已经评估好的结果」。
> 时间一律美东。所有数字均转引自下文「出处」列的档案，本文没有重新运行任何训练或评估。

## 一、一句话结论

| | 旧：4 任务×400 集 | 新：官方 16 任务×100 集 |
|---|---|---|
| encoder | `wan-full1600-filter2-b176x4-72ep-a`（v7，1×768），已训练、已评估 | `stage2-full144-1025-a` ep105（blockwise＋pool4＋三路，4×256），已训练、已评估 |
| VLA（无 motion） | 512／1024／2048／4096 四条，已评估：32.95%／41.33%／45.66%／45.95% | full16 512 一条（另有 count4 子集），已评估：**44.62%**（官方权重 47.88%） |
| VLA（有 motion） | 512＋motion 一条，已评估：**56.36%** | **没有**：官方数据＋stage2 motion 的 VLA 尚未训练 |
| 两边能否直接比 | 不能：任务集、测试集、评估步数、seed 全不同（见第五节） | |

旧这一组是唯一一组 encoder 和 VLA 同源、并且做完 motion 对照评估的实验。新的官方数据集上，目前只有无 motion 的 VLA 基线和单独训练好的 stage2 encoder，两者还没有组合起来训练过。

## 二、数据集

| 项 | 旧：4 任务×400 集 | 新：官方 16 任务×100 集 |
|---|---|---|
| 来源 | 自生成 newtask-v2 交付集 `HongzeFu/robomme-4task-h5-20260912-v2`（本机 `v1-store/raw-h5/4task-20260912-v2`） | 公开 `Yinpei/robomme_data_h5`（本机 `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5`，约 512.6 GB） |
| 任务 | BinFill、RouteStick、VideoRepick、VideoUnmaskSwap | 官方全部 16 任务（count4 子集：BinFill、StopCube、PickXtimes、SwingXtimes） |
| 每任务集数与难度 | 400 集；分 easy／medium／hard／xhard（各任务分档不同，BinFill 无 xhard，VideoRepick 无 hard） | 100 集；easy 50／medium 25／hard 25 |
| BinFill 的 demo | 自生成时把整条成功轨迹复制一遍接在前面当假 demo（训练有 demo、官方评测没有，后来评测时补了 demo 前缀） | 官方原样 |
| VLA 训练库 | `4task-v2-1600ep-604f16da`：1600 集、1,192,918 帧行、605,611 个执行样本，帧表 8×8 | `16task-pub-1600ep`：1600 集、768,897 帧、476,857 个执行样本，帧表 4×4；count4 为 `4task-counting-pub-400ep`：400 集、189,035 样本 |
| encoder 训练库 | `dataset-local-a100-4task-full1600`：2,800 个变体、901,970 帧；stride 1 的 33 帧 chunk 812,370 个，过滤后 796,001 个；只有 RGB、无增强 | `dataset-rgb-gtxyz-16views-stride2-v1`：stride 2 的 33 帧源窗 330,657 个（训练 311,151 个×16 个增强版，验证 19,506 个只用原图），共 4,997,922 行；RGB 与 GT-XYZ 两路 latent，另有 state |
| encoder 留出集 | 每「任务×难度」5 集，共 70 集（125 个变体，32,502 个 chunk） | 每「任务×难度」最后 2 集，共 96 集（训练 1504 集） |
| encoder 与 VLA 是否同源 | 同源（同一组 4 个 h5） | encoder 与官方 VLA 库同源（同一组 16 个官方 h5）；但两者还没有组合训练过 |

## 三、encoder

| 项 | 旧：`wan-full1600-filter2-b176x4-72ep-a` | 新：`stage2-full144-1025-a` |
|---|---|---|
| 仓库与提交 | MotionJEPA `660cee1`（commitV8.29Beta） | MotionJEPA `v12-3decoder`，配置 `configs/stage2_pool4.yaml` |
| 输入 | 9 个 Wan latent 组 (16,32,32)，每组整块展平后 Linear 16384→768，得到 9 个 token | 同样 9 组 RGB latent；每组 4×4 patchify 得 576 个 patch，另把第 0、3、5、8 组按 8×8 掩码平均池化成 4 个读出 token，共 580 行 |
| 主干 | 8 层因果 Transformer，宽 768，12 头 | 8 层 blockwise 注意力，宽 256，4 头，3D RoPE |
| 输出 motion | 末位读出 **1×768** | **4×256**（训练和 SIGReg 时展平为 1024） |
| decoder 与 loss | 单路 RGB WanLatentDiT（4 层，宽 768），latent MSE ＋ SIGReg 1e-4 | 三路 decoder：RGB、GT-XYZ（各以自身 z₀ 为 anchor 预测残差）、未来 32 步 13 维 state；λ = 1/1/1，SIGReg 1e-4 |
| 卡数与 batch | 4×A100，每卡 176，梯度累积 2，有效 batch 1,408 | 8×A100，每卡 256，梯度累积 2，有效 batch 4,096 |
| 优化 | lr 6e-4，warmup 1 轮，EMA 0.999，bf16 | lr 3e-4 cosine，warmup 1 轮，梯度裁剪 0.1，EMA 0.99，bf16 |
| 运动过滤 | 阈值 0.0271（丢弃 2.015%） | 阈值 0.021994（训练集百分位，丢弃约 2%） |
| 训练长度 | 计划 72 轮，跑满；09-14 00:17 → 17:47，共 17.5 h | 计划 144 轮，用户在第 106 轮叫停（10-26 09:43）；交付 ep105 EMA，sha256 `27daa832…` |
| 训练末 loss | train latent 0.0013（ep72） | train：RGB 0.00989，XYZ 0.00400，state 0.00061（ep105） |
| 已有评估 | 2,779 个留出窗，36 个 checkpoint 逐一评：最好的是 ep70，归一化 latent MSE 2.2222e-3（ep72 为 2.2225e-3）；重建 RGB MSE 1.467e-3，VAE 自身底噪 1.084e-3 | 四臂消融（19,083 个留出窗，fp32 EMA）。RGB：normal 0.01159、zero 0.04482、shuffled 0.05862、donor 0.06646。XYZ：normal 0.00413、zero 0.1247。state：normal 0.00162、zero 0.0616。测试集重建视频：RGB PSNR 25.60（VAE 底噪 29.42），XYZ RMSE 25.6 mm |
| VLA 实际使用的 checkpoint | `checkpoint_epoch_72.pt`（不是评估选出的 ep70，两者差可忽略） | 未接入官方数据的 VLA |
| HF 备份 | `HongzeFu/motionjepa-wan-full1600-72ep-v1` | `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1` |
| 出处 | MotionJEPA `docs/training-doc/wan-full1600-filter2-b176x4-72ep-a/README.md`、`scripts/evaluteVLAencode0914/reports/vlaencode0914-full1600-best-a-r2/` | MotionJEPA `docs/training-doc/stage2-full144-1025-a/`（config、val_loss、motion_ablation_epoch105、rgb_recon_video） |

两个 encoder 的重建误差口径不同，不能按数值直接排高低。同一份官方测试集重建视频里，作为参照的 v7 是 mixed8 版 e72（PSNR 28.64），不是本文旧 encoder。

## 四、VLA 训练

| 项 | 旧：4 任务×400 的 5 条 run | 新：官方 16×100 的 2 条 run |
|---|---|---|
| run | `v2-1600ep-m8x8-modul-b128-60k`、`v2-1600ep-m8x8-modul-motion-b128-80k`、`v2-1600ep-m{16,32,64}x8x8-modul-b128-80k` | `v2-orig-16task-pub1600ep-modul-b64-80k`（full16）、`v2-orig-counting-pub400ep-modul-b64-80k`（count4） |
| 训练条目 | `mme_vla_suite_b128_60k`／`_b128_80k`：batch 128，warmup 5k，lr 5e-5 恒定 | 原版 `mme_vla_suite`：batch 64，warmup 10k，lr 5e-5，decay_steps 100k |
| 共同部分 | pi05，`action_horizon=20`，从 pi05_base 起训，图像编码器冻结，EMA 0.999，梯度裁剪 1.0，seed 42，modulation，`memory_token_dim 1024` | 同左 |
| 帧记忆 | 8 帧×64 token（8×8）= 512；另有 16／32／64 帧×64 = 1024／2048／4096 三档 | **32 帧×16 token（4×4）= 512**（`perceptual-framesamp-modul.yaml`，与官方权重同配置） |
| motion | 只有 m8x8-motion 一条开启：每窗 1×768（旧 encoder e72），stride 16，33 帧窗，budget 160，demo 至少 17 帧、末帧重复补齐；记忆总长 672 | 无 |
| 步数与卡数 | 60k 那条：4 卡 FSDP4，w8，2.34 s/步，39 h。80k 那几条：8 卡 FSDP8，w16，约 1.0 s/步，约 22 h | 80k，4 卡 FSDP4（full16 用 GPU0–3，count4 用 GPU4–7） |
| 训练时间 | 09-15 → 09-25 | 09-27 前后完成（`completion.json status=PASS final=79999`） |
| HF 备份 | `robomme-vla-modul-60k-v1`、`-modul-motion-80k-v1`、`-modul-{1024,2048,4096}-80k-v1` | `robomme-vla-orig-16task-1600ep-modul-b64-80k-v1`、`robomme-vla-orig-counting-400ep-modul-b64-80k-v1` |
| 出处 | policy `docs/training-doc/v2-1600ep-*/` | policy `docs/training-doc/v2-orig-*-modul-b64-80k/`（README 的「未起跑」是起跑前写的，没有回填） |

## 五、VLA 评估

| 项 | 旧 | 新 |
|---|---|---|
| 在哪跑 | sled 会话经 GreatLakes A40 单卡（分支 `v2-eval-0918-motion`） | AWS 本机 8 卡（`1af91f9`） |
| 测试集 | benchmark fork 的 test/primary 700 条，分 14 个「任务×难度」格子；BinFill 补 demo 前缀后剩 142 局，共 692 局 | 官方 test split，16 任务×50 = 800 局（count4 为 200 局） |
| 评估参数 | seed 7，`max_steps=2000`；9 局在环境 reset 时卡死，计为失败 | seed 42，`max_steps=1300`，errors 0 |
| 出处 | `v2-eval-0918-motion@ba52d79:docs/training-doc/binfilldemo-nomotion50k-gl/result.md` | `docs/training-doc/eval-orig80k-modul-vs-official/result.md`、`eval-orig40k-ckpt40000/result.md` |

**旧：4 任务 692 局**

| 模型 | @50000 | 最终 ckpt |
|---|---|---|
| 512（8 帧） | 30.49% | 32.95%（@59999） |
| 1024（16 帧） | 42.05% | 41.33% |
| 2048（32 帧） | 45.09% | 45.66% |
| 4096（64 帧） | 45.23% | 45.95% |
| **512＋motion** | 54.62% | **56.36%** |

按任务看（最终 ckpt，512 → 4096 → 512＋motion）：
- BinFill：54.9% → 59.2% → 76.8%
- RouteStick：39.0% → 82.0% → 83.5%
- VideoRepick：24.0% → 26.0% → 48.0%
- VideoUnmaskSwap：18.0% → 15.5% → 21.0%

档案自己的结论：上下文加到 2048 后就饱和；motion 的收益主要在 VideoRepick 和 BinFill，比饱和值还高 10 pp 以上。

**新：官方 test 800 局**

| 模型 | @40000 | @79999 | 分类别（@79999）：Counting／Persistent／Referential／Behavior |
|---|---|---|---|
| 官方 modul 80k 权重（参照） | — | 47.88% | 72.0／25.5／40.0／54.0 |
| full16（自训） | 40.62% | **44.62%** | 67.0／26.0／39.5／46.0 |
| count4（只评 4 个 counting 任务，200 局） | 67.0% | 70.0% | 70.0／—／—／— |

- full16 比官方权重低 3.25 pp，McNemar p=0.065，未检出显著差异。
- 差距集中在 Behavior 类（−8 pp，p=0.0052），主要来自 MoveCube 和 PatternLock，各低 16 pp。

## 六、哪些比较能做、哪些不能做

1. **旧组内部可以比**：5 条 run 用同一份 692 局清单、同一套评估参数，排序 512＋motion > 4096 ≈ 2048 > 1024 > 512 有效。唯一的不对等是 60k 那条的最终 ckpt 是 59999，不是 80k。
2. **新组内部可以比**：full16、count4 和官方权重用同一份 800 局清单、同一套评估参数。
3. **旧组和新组之间不能比数值**：任务集（4 个 vs 16 个，只有 BinFill 重叠，且两边的 BinFill 数据生成方式不同）、测试集、`max_steps`（2000 vs 1300）、seed（7 vs 42）全都不同。旧组的 32.95% 和新组的 44.62% 不能说明谁更好。
4. **motion 的收益目前只在旧组上验证过**。官方数据上，还没有「同源 encoder＋同一 VLA 库」的有 motion 对照。
5. **encoder 同样不能横向比**：旧 v7 encoder 和新 stage2 encoder 的结构、损失、数据、增强都不同。stage2 的有效性证据是四臂消融（normal 明显低于 zero／shuffled／donor），不是与 v7 比重建误差。

## 附：不属于两组但容易混淆的已评估结果

以下 VLA 的训练数据是 mixed8-200（16 任务×200 的本地自生成集），不是上面两组中的任何一组，只列出来以免混淆：
- `mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k`：encoder 用的是官方数据训练的 stage2 ep105，VLA 用的却是 mixed8 数据，违反同源约束 P-LINEAGE。官方 800 局 @60k 成绩 42.00%。
- mixed8 纯 512 @60k：37.6%。
- mixed8 v7 512＋motion：40.6%／42.5%／43.1%（@40k／60k／80k）。

出处：分支 `v4-eval-official-1026` 的 `ed31e97`、`2346484`，以及分支 `v2-motionmem-eval-test1017` 的 `d726934`。
