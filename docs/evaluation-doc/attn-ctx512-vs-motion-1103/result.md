# attn-ctx512-vs-motion-1103 第一阶段结果

## 一句话结论
ButtonUnmaskSwap 第 8 局（easy，seed 570800，交换 2 次）两模型都成功；在「揭开」那次推理（第 18 次，step 288，发出 step 288～303 的动作，视频第 288～304 帧容器被抬起、第 304 帧露出绿色方块）上，ctx512+motion 在帧流里落在裸露窗口 [0,32) 的占比 17.0%（均匀期望 12.5%），基线只有 3.6%；motion 流落在交换区间 [64,164) 的占比 61.0%（均匀期望 55.6%）。但两模型帧流总质量都很小（基线 0.90%、motion 1.35%，motion 流 0.26%），注意力主体在腕部图像、文本与动作 token 上；这是单局单次推理的「看哪里」观察，不是因果结论。


> **口径更正（2026-11-03，X3）**：本节及下方「前后 5 次推理」表中 motion 流交换区间占比用的是旧口径「窗口与区间沾边即整窗计入」，偏高。按「重叠帧数/33」计权后：#15 92.2%→21.7%（均匀 45.0%）、#16 85.5%→（SWAP2 44.1%，均匀 19.7%，唯一明显高于均匀）、#17 69.2%→22.1%（39.4%）、#18 61.0%→30.5%（35.0%）、#19 57.5%→36%；即抬起前后 motion 流对交换区间并无偏好。帧流口径不受影响。详见 analysis/x3/README.md。

## 判定行（原文）
- OFFICIAL_CTX_ASSET=PASS sha256=387d4bd5…8ae8 bytes=11574223742 files=17
- LOCAL_ASSETS=PASS cc59999_files=38 cc59999_bytes=11976241540 encoder_sha256=27daa832
- LOCAL_ENVS：.venv（jax 0.5.3、torch 2.7.1+cu126，mme_vla_suite → 本仓库 src）；official-sim（torch 2.9.1+cu128，robomme → 本仓库 third_party，dist-info 与 turbo 原环境逐项一致 DIST_INFO_IDENTICAL）；stage2（torch 2.9.0+cu128，STAGE2_DIST_INFO_IDENTICAL）
- CFG_RESOLVE：基线 mme_vla_suite_b128 + history_config.txt=perceptual-framesamp-context.yaml，norm_stats 取自 79999/assets/robomme；motion mme_vla_suite_orig16_b128_60k（src 同步 5fce5d3 后注册），norm_stats 取自 59999/assets/robomme；P=1088／1184，motion_enabled False／True
- ATTN_SELFTEST=PASS f32_maxabs=1.19e-07 bf16_relfro=0.00321 rowsum_err=3.54e-07 uniform_ok=1 P=1088 MEM=512 motion=off
- ATTN_SELFTEST=PASS f32_maxabs=5.96e-08 bf16_relfro=0.00329 rowsum_err=3.02e-07 uniform_ok=1 P=1184 MEM=608 motion=Zero1024
- SRC_EQ_5FCE5D3=PASS（src 按用户裁决与 release 5fce5d3 一致）；examples/ third_party/ scripts/training/ scripts/evaluation/ 相对 b613836 零改动
- ATTN_SMOKE=PASS task=ButtonUnmaskSwap ep=4 model=ctx infers=20 npz=20 P=1088 rowsum_f16_err=3.52e-04
- ATTN_SMOKE=PASS task=ButtonUnmaskSwap ep=8 model=ctx infers=20 npz=20 P=1088 rowsum_f16_err=3.94e-04
- ATTN_SMOKE=PASS task=ButtonUnmaskSwap ep=8 model=mot infers=20 npz=20 P=1184 rowsum_f16_err=3.78e-04
- STAGE1_PAIR=PASS task=ButtonUnmaskSwap ep=8 ctx=success motion=success tries=2
- SIM_EEF_POSE=PASS infers=20（ctx、mot 各一次，CPU 物理后端）
- KEY_STEP=task=ButtonUnmaskSwap ep=8 model=history infer=18 vis_share=0.0357 vis_uniform=0.1250 swap_share=na swap_uniform=na
- KEY_STEP=task=ButtonUnmaskSwap ep=8 model=stage2_pool4_concat infer=18 vis_share=0.1705 vis_uniform=0.1250 swap_share=0.6100 swap_uniform=0.5556
- ATTN_AGG=PASS episodes=2 pad_mass=0.000e+00 stream_sum_err=1.532e-06
- ATTN_PLOT=PASS figs=1（主会话目视复查通过，见下）

## 试局记录
| 尝试 | 局 | 模型 | 结局 | 说明 |
|---|---|---|---|---|
| 1 | 4 | 基线 | infra（不计） | 本地仿真环境缺 websockets／msgpack／dm-tree，客户端未发出推理；补齐后重跑 |
| 2 | 4 | 基线 | fail | 317 步、20 次推理；按规则不跑 concat |
| 3 | 8 | 基线 | success | 20 次推理 |
| 4 | 8 | concat | infra（不计） | 本分支 src 不支持 published-cache 来源；用户裁决同步 5fce5d3 |
| 5 | 8 | concat | infra（不计） | 本机 HF 缓存缺 Wan VAE（sidecar 离线）；按钉死 revision 0fad780a 下载 |
| 6 | 8 | concat | success | 20 次推理 |
infra 尝试均未产生任何推理（npz=0），不属于为成功重试。

## 「揭开」推理的选定理由
两模型逐次推理轨迹一致：step 32～208 夹爪闭合、两次下压（按两个按钮）；step 208～272 张开；272～288 闭合并下降到 z≈0.065；288～304 抬升到 0.15～0.17。视频拼图（帧 256～304 每 8 帧）显示两模型都在第 288～304 帧抬起目标容器，第 304 帧露出绿色方块。第 18 次推理发出 step 288～303 的动作，覆盖抬起帧，定为关键推理。自动候选只命中 motion 的第 18 次；基线的闭合点恰落在第 17、18 次两块交界，块内规则漏检（S3 已知局限）。

## 图（figures/fig_key_step.png）目视复查
两列两排版式、坐标轴、角标、中文字体正常，无重叠与截断；左右纵轴各自缩放，柱高不可跨列直接比；motion 窗宽 33 半透明叠放。基线帧流质量几乎全落在最后一帧（当前时刻），motion 模型在全部 32 帧上较均匀、早期 4 帧略高。

## 五流质量（第 18 次推理，全步全层全头全动作 query 平均）
| 模型 | 帧流 | motion 流 | 当前 base 图 | 当前 wrist 图 | 文本 | 动作 |
|---|---|---|---|---|---|---|
| ctx512 基线 | 0.90% | — | 10.30% | 25.22% | 33.18% | 30.40% |
| ctx512+motion | 1.35% | 0.26% | 8.89% | 27.76% | 32.11% | 29.63% |

## 盲区
单局、单次推理、seed 42；attention 不等于因果；两模型训练口径不同（官方 80k vs 自训 60k + motion）；展开版与 scan 版 bf16 不逐位（rel_fro≈3.2e-3），执行局动作来自展开版。

## 产物
原始 npz（约 4.9 GB，含第 4 局基线）与视频在本机 v1-store/evaluation/interpret-1103/stage1/，不进 git；records/ 为汇总 JSON。

## 「揭开」前后 5 次推理（用户追问「前后呢 往下移动到这个位置并且抬起到」）
同一份原始 npz 按 `aggregate_attn.py --key-infer <k>` 逐次计算，未重跑仿真。动作阶段按 TCP z 与夹爪：#15/#16 张开靠近，#17 下降到 z≈0.065 并闭合，#18 抬起（揭开），#19 抬起后。

| 推理 | step | 模型 | 帧流总 | motion 总 | 帧流裸露窗占比/均匀 | motion 交换区间占比/均匀 |
|---|---|---|---|---|---|---|
| #15 | 240 | 基线 | 0.23% | — | 14.2% / 15.6% | — |
| #15 | 240 | motion | 1.68% | 0.90% | 15.3% / 15.6% | 92.2% / 71.4% |
| #16 | 256 | 基线 | 0.35% | — | 8.5% / 12.5% | — |
| #16 | 256 | motion | 1.10% | 0.58% | 21.0% / 12.5% | 85.5% / 62.5% |
| #17 | 272 | 基线 | 0.64% | — | 4.4% / 12.5% | — |
| #17 | 272 | motion | 1.05% | 0.32% | 17.0% / 12.5% | 69.2% / 62.5% |
| #18 | 288 | 基线 | 0.90% | — | 3.6% / 12.5% | — |
| #18 | 288 | motion | 1.35% | 0.26% | 17.0% / 12.5% | 61.0% / 55.6% |
| #19 | 304 | 基线 | 0.83% | — | 4.2% / 12.5% | — |
| #19 | 304 | motion | 1.93% | 0.33% | 15.5% / 12.5% | 57.5% / 55.6% |

读法：
- 基线的帧流在整段靠近→下降→抬起中都几乎只看当前帧，裸露窗占比从 #15 的 14% 一路降到 3～4%，低于均匀期望；早期帧在基线里没有被「回看」。帧流里另有两处小隆起在全域帧约 90～100 与 185～195，对应两次按按钮（step 80～96、176～192）。
- motion 模型从 #16 起帧流裸露窗占比稳定在 15～21%，高于均匀期望 12.5%，在下降与抬起（#17/#18）期间保持约 17%。
- motion 流对交换区间的偏好在靠近阶段最强（#15 92%、#16 85%，明显高于均匀），到下降、抬起时回落到接近均匀（#17 69% vs 62.5%，#18 61% vs 55.6%，#19 57.5% vs 55.6%）；motion 流总质量也从 0.90% 降到 0.26～0.33%。即「盯交换」更像发生在决定去抓哪个容器的靠近阶段，而不是抬起那一刻。
- 口径提醒：交换区间按窗口 [t,t+33) 与 [64,164) 相交计，#17/#18 的峰值窗起点约 160（主体在交换之后，只重叠 4 帧）也被算入，占比会被高估；左右纵轴各自缩放；单局单次、非因果。

图：figures/fig_key_neighbors_trend.png（5 次推理三项指标趋势，黄底 #18）、fig_key_step_infer17.png、fig_key_step.png（#18）、fig_key_step_infer19.png。三张均经主会话渲染后目视复查，无重叠、截断、字体缺失。

## 可解释性深挖（X1–X4，围绕裸露窗与交换区间）
用户原话：「就对于这个任务 更加详细的挖掘可解释性 用多个subagent来 不同worktree 全部opus」「我关注的还是这几个区间」。四个 opus 子代理各在独立 worktree 只读已存 npz 与视频，逐个审查、目视复查后 --no-ff 合入。明细见 analysis/x1～x4/README.md。

**区间核实（X3）**：视频帧号 = 全域帧号（reset 1 帧 + 每动作 1 帧）。裸露窗 [0,32)：reset 起三色方块就露在外面，第 32 帧容器瞬间落回盖住（视频里看不到抬起过程）；交换 [64,114)、[114,164) 与源码一致，第 165 帧起静止；基线 ep4 只交换 1 次 [64,114)；ep8 约第 302 帧抬起后绿方块重新出现。

**基线 ctx512（不会回看裸露窗）**
- 整局自 #4 起帧流裸露窗占比约为均匀的 0.3～0.9 倍、始终 <1，成功局 ep8 与失败局 ep4 一样（X2）；去 sink 后仍低于均匀（X4）。
- 注意力几乎只看最近几帧：最大记忆头 L15H1 质量 0.759，99.8% 落在最近 4 帧（X1）；基线 #18 帧流 63～64% 在当前帧 t=288（X4）。
- 交换期间（#4～#13）偏向「交换 1」那批帧，主要是近因——那时它们就是最新的帧（X2/X3）。
- 失败 ep4 与成功 ep8 的差别出现在张开靠近 #14～#17：ep8 仍让交换 1 高于均匀（r=1.24），ep4 降到 0.77；抬起 #17～#19 差别消失（X2，单对观察）。两局头结构几乎相同（r=0.996，top-10 10/10，X1）。

**ctx512+motion（帧流稳定偏向裸露窗，motion 流不盯交换）**
- 帧流裸露窗：动作 query 整局多数推理高于均匀，#16～#19 为 15.5～21.0%（均匀 12.5%），窗内 4 帧大致均分（X3）；去 sink 后不变（17.2%，均匀 13.1%，1.3 倍，X4），这是四个方向里最稳的一条正向信号。帧流总质量约为基线 3 倍。
- 由少数头承担：L16H0（帧流质量 0.464、VIS 富集 1.81，但逐次波动大）与 L3H0（0.110、3.65 倍）合计约 69% 的正 VIS 超额；与基线回看 VIS 的头几乎不重合（X1）。
- 回看偏去噪前半段：#18 前 3 步 19.8%、后 3 步 13.0%（X3）。
- motion 流对交换区间：按重叠/33 计权后几乎都低于均匀，第 1 次交换基本不看（2.5～9.4%，均匀 17.5～22.5%）；只有张开靠近的 #16 对第 2 次交换明显偏高（44.1%，均匀 19.7%）（X3）；无稳定的 SWAP 专门头（X1）。
- 文本「green」query 的裸露窗偏高主要是 sink：t=0 帧占帧流 25.9～39.8%、其中单 token 占 73.4～98.8%，去 sink 后 37.8%→20.8%，#15/#18/#19 回到均匀附近（X4）。

**画面位置与文字（X4）**：文本 query 看记忆只占 5.4～8.0%，8 个词之间看不出分工；动作 query 在被看的历史帧里 4×4 近乎均匀（最大块 1.07～1.6 倍），没有对准绿方块或容器；当前图像里 wrist 视角占 prefix 36～48%，集中在被抓容器边缘与指尖。

**总体读法**：两个模型的注意力主体都在当前图像（尤其 wrist）、文本与动作 token 上，记忆区只占约 1～3%。在这一小块里，motion 模型稳定地多看开局方块裸露的那几帧，基线则几乎只看最近帧；两者都没有表现出「在抬起时回看交换过程」的模式，motion 流对交换的偏好只在靠近阶段的 #16 出现过一次。

**局限**：基线 2 局（含 1 局失败）、motion 1 局，单 seed；attention 只说明「看哪里」，不是因果；均值掩盖步间与头间差异，Gemma 多 query 头共享 KV；sink 只做了「剔除每帧最大 token」的粗对照；motion 窗按 [t,t+33) 均匀计权是近似。
