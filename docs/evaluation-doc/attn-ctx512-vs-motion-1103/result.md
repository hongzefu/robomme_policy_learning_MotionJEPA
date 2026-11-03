# attn-ctx512-vs-motion-1103 第一阶段结果

## 一句话结论
ButtonUnmaskSwap 第 8 局（easy，seed 570800，交换 2 次）两模型都成功；在「揭开」那次推理（第 18 次，step 288，发出 step 288～303 的动作，视频第 288～304 帧容器被抬起、第 304 帧露出绿色方块）上，ctx512+motion 在帧流里落在裸露窗口 [0,32) 的占比 17.0%（均匀期望 12.5%），基线只有 3.6%；motion 流落在交换区间 [64,164) 的占比 61.0%（均匀期望 55.6%）。但两模型帧流总质量都很小（基线 0.90%、motion 1.35%，motion 流 0.26%），注意力主体在腕部图像、文本与动作 token 上；这是单局单次推理的「看哪里」观察，不是因果结论。

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
