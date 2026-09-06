# siglip-ab-replay-40k — 结果

**环境 B（AWS 单机 8×A100-SXM4-80GB，本地 NVMe RAID `/dev/md0`）**。执行 commit：权重核对 `c01e45d`、五臂重放 `034c18f`
（分支 `v2-motionmem-sgab`，基于 `v2-motionmem` `b12f1a0`）。2026-09-06 00:56 – 01:18。

## 结论先行

审计 D-A1（训练用离线 f32 SigLIP 权重建库、在线用 checkpoint 内 bf16 `PaliGemma.img`）**成立但可忽略**：

1. **两份权重同源**：`siglip_params.pkl` 与 `pi05_base` 的 `PaliGemma.img` 子树 23 叶逐元素相等（max_abs = 0）；训练 checkpoint `39999`
   的 img 子树逐叶 == `bf16(pi05_base.img)`（EMA 与 4 万步训练都没动过它）。在线 img 就是训练前向看到的那份，差别只在
   离线建库用的是同一权重的 f32 原件。
2. **记忆 token 的训练/推理真实差距（S vs B）**：相对 Frobenius 误差 **0.34%**，逐 token cosine 最低 0.999988，
   78% 元素不同但中位差 2 个 bf16 ULP（p99 = 86 ULP）。
3. **差距由两个同量级、互不叠加的来源构成**：权重 dtype（A vs B，0.32%）与**批形状**（S vs A，0.30%——建库逐帧 batch=1、在线成批
   67/16 帧，XLA 在不同 M 维下 bf16 GEMM 归约序不同；审计 D-A2 由此从「无法静态判定」变为**实测成立**）。
   诊断臂 C（A 权重 cast bf16 后成批）与 B **227/227 逐位相等**，证明除 dtype 与批形状外没有第三个来源。
4. **传到动作上**：固定噪声下 S vs B 的动作 RMS = **0.0003**（关节弧度），是同一记忆换采样噪声 seed 的 RMS（0.0133）的 **2.3%**，
   是动作 std 均值（0.205）的 **0.15%**；归一化空间最大逐元素差 0.0039（≈ 1 个 bf16 ULP @1）。
5. **对评估的含义**：按上游原样起评即可，不需要为 D-A1 改在线侧或重建库；评估留档写明「记忆 token 与训练存在约 0.3% 的
   bf16 舍入 + 批形状差，动作影响为采样噪声的 2%」即可。

## 判定行（`records/replay/summary.txt`、`records/weights.txt`）

```
PARAM_SAME_ENC=PASS n_leaves=23 max_abs=0.000e+00
CKPT_IMG_FROZEN=PASS mismatched_leaves=0
PKL_VS_CKPT max_abs=5.948e-01 max_rel=3.891e-03 rel_le_2^-8=True
MEM_A1_VS_STORE=PASS frames=227 mismatches=0
MEM_S_VS_TRAINSET=PASS points=11 key_mismatches=none
MEM_C_VS_B=PASS frames=227 mismatches=0
MEM_A_VS_STORE_BITEXACT=0/227 A_static_image_emb_vs_trainset_bitexact_points=0/11
MEM_S_VS_B  frames=227 max_abs=1 mean_abs=0.007261 rel_fro=0.003358 cos_min=0.999988 cos_mean=0.999995 ulp_p50=2 ulp_p99=86   ulp_max=500 frac_nonzero=0.780
MEM_S_VS_A  frames=227 max_abs=1 mean_abs=0.006122 rel_fro=0.003011 cos_min=0.999988 cos_mean=0.999996 ulp_p50=1 ulp_p99=73.2 ulp_max=505 frac_nonzero=0.752
MEM_A_VS_B  frames=227 max_abs=1 mean_abs=0.007177 rel_fro=0.003188 cos_min=0.999986 cos_mean=0.999996 ulp_p50=2 ulp_p99=85   ulp_max=505 frac_nonzero=0.779
MEM_S_VS_A1 frames=227 max_abs=0 … frac_nonzero=0.000
MEM_C_VS_B_NUM frames=227 max_abs=0 … frac_nonzero=0.000
ACT_S_VS_B points=11 rms_norm=0.0004307 max_abs_norm=0.003906 rms_unnorm=0.0003    | NOISE_S rms_norm=0.009616 rms_unnorm=0.01327 | ratio_norm=0.04479 ratio_unnorm=0.02261 | act_std_mean=0.2047 rms_unnorm/act_std=0.001466
ACT_S_VS_A points=11 rms_norm=0.0004247 max_abs_norm=0.00293  rms_unnorm=0.0002942 | … ratio_unnorm=0.02217
ACT_A_VS_B points=11 rms_norm=0.0004223 max_abs_norm=0.003906 rms_unnorm=0.0002873 | … ratio_unnorm=0.02165
ACT_C_VS_B_BITEXACT=PASS
ACT_DETERMINISM=PASS
SIGLIP_AB_REPLAY=DONE episodes=1 points=11
```

臂定义：S = 训练库 `image_emb_4x4` 行（训练真值）；A1 = 离线 f32 tokenizer 逐帧 batch=1（只走帧路）；A = 同一 f32 编码器按 eval 节奏成批；
B = 在线 `policy._vision_encode`（checkpoint bf16 img，成批）；C = A 权重 `astype(bf16)` 成批。预处理五臂逐字同一
（`FrameSampMemory.add_buffer`）。运动路各臂固定为训练库真 motion 行。

## 逐时刻（VideoUnmask ep0，es=66，T=239，11 个 infer 点）

| t | 新帧 | 帧 rel_fro S-B / S-A / A-B | A1==S | C==B | S 装配==训练样本 | 动作 RMS S-B / S-A / A-B（弧度） | 换噪声 RMS |
|---|---|---|---|---|---|---|---|
| 66 | 67 | 0.00339 / 0.00305 / 0.00313 | 67/67 | 67/67 | ✓ | 0.000273 / 0.000258 / 0.000264 | 0.00157 |
| 82 | 16 | 0.00378 / 0.00336 / 0.00318 | ✓ | ✓ | ✓ | 0.000327 / 0.000315 / 0.000289 | 0.00473 |
| 98 | 16 | 0.00378 / 0.00336 / 0.00318 | ✓ | ✓ | ✓ | 0.000306 / 0.000348 / 0.000331 | 0.00471 |
| 114 | 16 | 0.00333 / 0.00295 / 0.00326 | ✓ | ✓ | ✓ | 0.000268 / 0.000249 / 0.000272 | 0.00451 |
| 130 | 16 | 0.00317 / 0.00284 / 0.00329 | ✓ | ✓ | ✓ | 0.000287 / 0.000287 / 0.000261 | 0.00155 |
| 146 | 16 | 0.00309 / 0.00290 / 0.00319 | ✓ | ✓ | ✓ | 0.000318 / 0.000309 / 0.000292 | 0.00287 |
| 162 | 16 | 0.00321 / 0.00295 / 0.00317 | ✓ | ✓ | ✓ | 0.000350 / 0.000290 / 0.000298 | 0.106 |
| 178 | 16 | 0.00322 / 0.00290 / 0.00317 | ✓ | ✓ | ✓ | 0.000313 / 0.000326 / 0.000270 | 0.0102 |
| 194 | 16 | 0.00316 / 0.00294 / 0.00318 | ✓ | ✓ | ✓ | 0.000264 / 0.000253 / 0.000270 | 0.00153 |
| 210 | 16 | 0.00341 / 0.00290 / 0.00325 | ✓ | ✓ | ✓ | 0.000305 / 0.000357 / 0.000326 | 0.00629 |
| 226 | 16 | 0.00328 / 0.00287 / 0.00326 | ✓ | ✓ | ✓ | 0.000290 / 0.000245 / 0.000288 | 0.00178 |

动作差在 11 个点上稳定在 2.5–3.5e-4 弧度，与记忆差的量级一致、不随 t 增长；换噪声 RMS 在 1.5e-3 到 0.106 之间波动
（t=162 一对噪声给出了差别很大的两条轨迹），比值取均值为 2.3%，逐点最差（t=194，噪声 RMS 最小处）也只有 17%。

## 读法与边界

- **S 臂装配八键（含 motion 四键与 mem_order）与训练样本 11/11 逐位**：这是在**生产 config（`mme_vla_suite`）+ 真 checkpoint 快照 yaml + 400ep 真库**上
  第一次把「在线 `_prepare_history` == 训练 `FrameSampDataset.__getitem__`」跑通，比审计静态结论更强，也覆盖了审计盲区 7.2 第 4 条。
- **批形状效应是新发现**：审计把 D-A2 列为「无法静态判定」；实测 f32 同权重、只换喂法（逐帧 vs 成批）就让 227/227 帧不逐位、rel_fro 0.30%。
  这意味着即便在线改用 f32 tokenizer，也无法与训练库逐位——除非在线也逐帧喂（每帧一次 jit 调用，首批 67 次）。
- **单集、单 seed、开环**：数字只证「差多少」，不证「差异对成功率的影响」；后者只能靠闭环评估，而闭环里这 0.0003 弧度会被环境动力学放大或吸收，
  本 run 不做外推。VideoUnmask ep0 是 train split 录制集，不代表 test 分布，但对拍的是同一输入下两条链路的差，与分布无关。
- 第一轮三臂版本（`records/replay-v1-3arm/`）已给出 C==B 与动作差同一数字，因缺 S/A1 臂无法拆分来源，作过程证据保留。
- 未测：demo 段之外 es=0 的 Button 任务（首批只有 1 帧、批形状效应更小）；多 episode 统计。脚本支持 `--episodes` 列表，需要时几分钟可补。

## 与审计报告的对照

| 审计条目 | 审计结论 | 本 run |
|---|---|---|
| D-A1 权重两份、dtype 两种 | 已确认（confidence 0.90） | 成立；同源（`PARAM_SAME_ENC=PASS`），差 = 一次 f32→bf16 舍入 |
| D-A1b 同源性 | 需执行 | **PASS**，D-A1 退化为纯 dtype 问题 |
| D-A2 批形状 | 需执行 | **成立**：同权重逐帧 vs 成批 0/227 逐位，rel_fro 0.30% |
| 影响面 | 静态判不了 | 记忆 token 0.34%、动作 = 采样噪声的 2.3% / 动作 std 的 0.15% |

## 产物

- `records/weights.{txt,json}`：权重核对（CPU，逐叶表）。
- `records/replay.txt`、`records/replay/{summary.txt,per_point.json}`：五臂重放（含每点两臂 seed0 的反归一化动作全量）。
- `records/replay-v1-3arm*`：第一轮三臂版本。
- `run_weights.sh` / `run_replay.sh`：可复跑命令。
