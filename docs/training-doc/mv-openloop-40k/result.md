# mv-openloop-40k — 结果

**环境 B（AWS 单机 8×A100-SXM4-80GB）**。执行 commit `26ff2b4`（clean HEAD；代码 commitV8.0 `5620f66`），被评 `awsprod40k-b128-motion/39999`，GPU 7（与 mask 校准批 w7 共卡，0.3 显存份额），墙钟 `04:34:28 → 04:51`（**993 s**，含编译）。

## 结论先行

```
OL_ACT_DELTA points=149 episodes=6 noise_seeds=5 k_mean=15.93 k_range=0..34 mask/noise=9.877 swap/noise=6.358 mask/noise_median=7.523 swap/noise_median=4.344 mask_unnorm/act_std=0.7262 swap_unnorm/act_std=0.4267
OL_ZEROK_NULL=PASS zerok_points=4 mask_bitexact=4/4 swap_bitexact=4/4
LAYER_ATTN points=24 enrich=[…] top3_enrich_layers=[15, 16, 12]
LAYER_ALLMASK_VS_STAGE0 points=24 allmask_layered_rms=0.08423 stage0_mask_rms=0.07761 ratio=1.085
MV_OPEN_LOOP=PASS points=149 layer_points=24 stages=[0, 1] wall_s=993 blocking=4/4 observe=11
```
（判定行全文 `records/open_loop.summary.txt`；逐点数据 `records/open_loop.json`。）

**模型对 motion token 通路的敏感性很高，且读取的是内容而不只是槽位存在。** 屏蔽通路（mask）使最终动作变化 = noise seed 间动作差的 **9.9 倍（中位 7.5）**，反归一化后 = 动作 std 的 73%；换成同任务异集内容（swap）= 6.4 倍（中位 4.3）、动作 std 的 43%。按预注册措辞，「所测点动作敏感性较低」一行排除。

### 阶段 0 分层（mask/noise · swap/noise）

| 分层 | n | mask/noise | swap/noise |
|---|---|---|---|
| 全部 | 149 | 9.88（中位 7.52） | 6.36（中位 4.34） |
| k=0 | 4 | 0.000（逐位） | 0.000（逐位） |
| k 1–4 | 11 | 2.41 | 2.26 |
| k 5–12 | 39 | 9.48 | 6.25 |
| k >12 | 95 | 11.11 | 7.01 |
| cold（无 exec 窗） | 12 | 3.37 | 1.02 |
| steady | 137 | 10.27 | 6.68 |
| ButtonUnmask | 29 | 8.73 | 2.69 |
| VideoUnmask | 21 | 10.57 | 7.64 |
| ButtonUnmaskSwap | 35 | 8.49 | 6.46 |
| VideoUnmaskSwap | 64 | 11.02 | 7.86 |

敏感性随有效窗数 k 单调上升，四任务一致；swap 在 ButtonUnmask 上明显小于其它三任务（2.7 vs 6.5–7.9）。开环 donor 覆盖 `exact=9342/11865 fallback=2523 cycle=0 cross_seg=0`。

### 阶段 1（24 点 = 6 集 × cold/early/mid/late）

**(i) 去噪 action query 的 motion 富集度**（份额 / 逐 query 合法-key 均匀基准 0.0125）：第 0–14 层全 < 1（`2.8e-5 … 0.52`），**第 15 层 6.17、第 16 层 4.07**，第 17 层 0.18；第 15 层单 head 最大份额 0.31。cold 点第 16 层 15.9、steady 点第 15 层 6.5。prefill 侧 frame query 对 motion 的富集在**第 0/1 层 36.5/33.7**，其余层 5–23；文本 query 第 1 层 39.4，之后多 <1。

**(ii) 只动第 l 层的最终动作差 / 噪声**（`noise_layered=0.0112`）：

| scope | 最大三层（层:倍数） | 其余层 |
|---|---|---|
| `both`（该层 prefill+去噪都挡） | **1: 2.92，0: 1.20**，16: 0.26 | 0.03–0.10 |
| `step_only`（只挡 action 直读） | 16: 0.25，15: 0.09，10: 0.03 | 0.03 |
| `kv_vzero`（该层 motion V 置零） | 16: 0.18，15: 0.11，10: 0.03 | 0.03 |
| `kv_donor`（该层 motion K/V 换 donor 内容） | **10: 0.32**，16: 0.07，15: 0.05 | 0.03 |

18 层全挡 = 7.5 倍（与阶段 0 一致，ratio 1.085）。没有任何单层直读能解释总效应：**主通路是第 0–1 层 motion 信息被帧/文本 token 吸收后间接传播**（与 (i) 的 prefill 富集一致）；action 直读集中在 15–16 层；对**内容**敏感的读取点在第 10 层。

**(iii) 固定-loss 逐层入口梯度**（motion 列 / frame 列逐 token RMS 比）：第 0 层 137× / 105× / 68.5×（t=0.1/0.5/0.9），第 16 层 23×，其余 2–6×；`LAYER_GRAD_PAD_ZERO` 三档全 PASS；`∂L/∂motion_emb` 有效位 RMS 2e-6–9e-6。

## 盲区诚实清单

- 开环、训练集 episode（6 集），只回答「读不读、在哪读」，不回答成功率方向；swap 的 donor 是同分布训练集 episode（开环映射排除自身）。
- 阶段 1 在 18 层展开路径内自洽比较；展开与 `nn.scan` 在 bf16 下不逐位（`UNROLL_VS_SCAN_BF16 rel_fro≈3.3e-3`），f32+highest 下逐位相同（`UNROLL_VS_SCAN_F32` / `LAYER_SELFCHECK_F32` rel_fro=0，`records/adapter_selftest.txt`）。
- 逐层干预的 0.03 倍「地板」是该层 attention 少量 motion 贡献被移除的真实小效应，不是零。
- 与 mask 校准批共卡只影响墙钟。

## 产物

- `records/open_loop.summary.txt`、`records/open_loop.json`（去掉逐 head 逐 step 的 `share_step_full`，全量在 `v1-store/reports/motion-variance/open_loop.{json,npz}`，不进 git）、`records/adapter_selftest.txt`、`records/open_loop.smoke.summary.txt`。
- 图：`docs/motion-utilization/figures/fig4-noise-facet.*`、`fig5-layers.*`（矩阵跑完统一出图）。
