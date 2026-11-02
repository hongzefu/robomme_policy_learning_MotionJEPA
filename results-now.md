# 当前结果汇总（2026-11-02，美东）

本文件汇总当前的评估结果，以及 context 注入长度对训练耗时的影响，供画论文图直接取数。标有「实测」的是已有运行的记录；标有「FLOPs 推算」「拟合推测」的是计算结果，没有经过 GPU 实跑验证。

## 1. 官方 16 任务评估：context concat 1×1024（512 + 96 motion）

训练与评估都使用 RoboMME 官方口径：官方 16 任务 × 100 集训练库；官方 test split 16 任务 × 50 集 = 800 局；`max_steps=1300`；官方 seed 7、42、0。被评的是 checkpoint 59999。

| 模型 | seed 42 | seed 7 | seed 0 | 三 seed 均值 ± std | 论文 Table 3 |
|---|---|---|---|---|---|
| context concat 1×1024（512+96） | 38.00% | 36.88% | 37.13% | 37.33 ± 0.59 | FrameSamp + Context 30.68 ± 1.17 |

- run：`orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k`。帧路 32 帧 × 16 token = 512；motion 每窗 1 个 1024 维 token，预算 96（评估回显 `readout=stage2_pool4_concat_v1 budget=96`）。
- std 为样本标准差（ddof=1），与官方 `compute_results.py` 同口径。论文还对 3 个 checkpoint（60000、70000、79999）取平均，本轮只评 59999，所以两边的 ± 不完全可比。
- 来源：[docs/evaluation-doc/o16-3seed-1031/result.md](docs/evaluation-doc/o16-3seed-1031/result.md)。同一文件还记录了 context split 4×256 为 38.42 ± 1.25，modulation split 为 43.42 ± 0.52（论文 FrameSamp + Modul 44.51 ± 0.77）。

## 2. context 注入长度与训练耗时：按 FLOPs 推算

### 2.1 计算口径

- 主干：Gemma 2B，18 层，宽 2048，MLP 16384，8 头 × 256，1 个 KV 头；非嵌入参数 1.98 B（`src/openpi/models/gemma.py` 的 `gemma_2b`）。
- 动作专家：Gemma 300M，18 层，宽 1024，MLP 4096；参数 311 M（`gemma_300m`）。
- 序列长度：前缀 = context 长度 n + 两路当前图像 512 + 文本 64；动作后缀 20。即总长度 n + 596，与 `docs/context-gpu-scaling-readonly-audit-20260928.md` 及 `ctx-stock-scaling-20260928/result.md` 的口径一致。
- 线性层的训练 FLOPs 按 6 × 参数 × token 计。注意力按稠密计算 QKᵀ 与 AV，掩码不减少计算量，训练开销取前向的 3 倍。SigLIP 冻结，只计前向 2 × 0.4 B × 512 token。
- 每样本训练 FLOPs：

  ```text
  F(n) = 6·N_llm·P + 6·N_exp·S + 12·D·A·(P·T + S·T) + 2·N_sig·512
  P = n + 576（前缀），S = 20（动作），T = P + S，D = 18，A = 2048
  ```

- 换算成步时：以 context 512 的实测步时为基准，乘以 FLOPs 比，即 `步时(n) = 步时(512) × F(n)/F(512)`。8 卡基准为 1.727 s/步，4 卡基准为 3.249 s/步，条件均为 A100-80GB、global batch 128。

### 2.2 结果（FLOPs 推算）

| context 长度 n | 每样本训练 FLOPs | 每步 FLOPs（×128） | 相对 512 | 8 卡步时（FLOPs 推算） | 4 卡步时（FLOPs 推算） | 8 卡 80k 步 | 4 卡 80k 步 |
|---|---|---|---|---|---|---|---|
| 256 | 10.66 TFLOP | 1.36 PFLOP | 0.765 | 1.32 s | 2.49 s | 29 h | 55 h |
| 512 | 13.93 TFLOP | 1.78 PFLOP | 1.000 | 1.73 s（实测） | 3.25 s（实测） | 38 h | 72 h |
| 1024 | 20.63 TFLOP | 2.64 PFLOP | 1.481 | 2.56 s | 4.81 s | 57 h | 107 h |
| 2048 | 34.74 TFLOP | 4.45 PFLOP | 2.494 | 4.31 s | 8.10 s | 96 h | 180 h |

按 FLOPs 算，长度从 512 翻倍到 1024，计算量约 ×1.48；翻四倍到 2048，约 ×2.49。80k 步总时长只算纯步时，不含初始化、编译与保存（约再加 1%）。

### 2.3 FLOPs 推算与实测对比

| 配置（总长度 n） | FLOPs 比 | 实测步时比 | 实测步时 | 8 卡实测 MFU（A100 bf16 峰值 312 TFLOPS） |
|---|---|---|---|---|
| 512 | 1.000 | 1.000 | 8 卡 1.727 s，4 卡 3.249 s | 41.4% |
| 512+motion160（672） | 1.149 | 8 卡 1.146，4 卡 1.186 | 8 卡 1.979 s，4 卡 3.854 s | 41.5% |
| 512+motion384（896） | 1.360 | 8 卡 1.407 | 8 卡 2.43 s | 40.0% |
| 2048 | 2.494 | 4 卡 3.106 | 4 卡 10.09 s | — |

- 在 896 以内，FLOPs 比与实测吻合得很好，误差不超过 3.5%，MFU 稳定在 40%–41%。说明这一段训练基本受计算量限制。
- 到 2048 时，实测比 FLOPs 推算慢约 25%（10.09 s 对 8.10 s）。原因推测是这一档显存已到 97%（每卡峰值 79445 MiB），注意力 logits 按 float32 物化（逻辑量 `16×8×(n+596)²×4` 字节），访存开始占主导；这一点没有专门测量过。因此 1024 和 2048 的 FLOPs 推算应视为**下限**。

## 3. 拟合推测（上一轮的估算，保留作对照）

用 4 卡实测的 512、672、2048 三点（3.249、3.854、10.09 s/步）拟合二次曲线 `步时 ≈ 1.481 + 3.203e-3·n + 4.888e-7·n²`；8 卡按实测的 4 卡/8 卡比约 1.9 换算。在 896 处交叉检查：曲线换算得 2.50 s，实测 2.43 s。

| context 长度 n | 4 卡步时（拟合推测） | 8 卡步时（拟合推测） | 4 卡 80k 步 | 8 卡 80k 步 | 相对 512 |
|---|---|---|---|---|---|
| 256 | ≈2.3 s（外推） | ≈1.2 s | ≈52 h | ≈27 h | ×0.72 |
| 512 | 3.25 s（实测） | 1.73 s（实测） | 72 h | 38 h | ×1 |
| 1024 | ≈5.3 s（内插） | ≈2.8 s | ≈117 h | ≈62 h | ×1.6 |
| 2048 | 10.09 s（实测） | ≈5.3 s | 224 h | ≈118 h | ×3.1 |

**两种推测的关系**：FLOPs 推算只算计算量，不含 2048 档的访存开销，是下限；拟合曲线把 2048 的实测慢化带进来了，所以 1024 的结果偏高。**1024 的真实步时预计在 8 卡 2.56–2.8 s、4 卡 4.8–5.3 s 之间；256 预计在 8 卡 1.2–1.3 s、4 卡 2.3–2.5 s 之间。**

## 4. 实测数据来源

| 配置 | 卡数 | 稳态步时 | 来源 |
|---|---|---|---|
| context 512 | 8 卡 | 1.727 s/步 | [docs/training-doc/ctx-stock-scaling-20260928/result.md](docs/training-doc/ctx-stock-scaling-20260928/result.md) |
| context 512 | 4 卡（两个 4 卡任务同机并行） | 3.247 / 3.251 s/步 | 同上 |
| context 512+motion160 | 8 卡 / 4 卡 | 1.979 / 3.845–3.863 s/步 | 同上 |
| context 512+motion384 | 8 卡 | 2.43 s/步 | [1101-gl-ablation-encoders-ctx-b64-80k-plan.md](1101-gl-ablation-encoders-ctx-b64-80k-plan.md) |
| context 2048 | 4 卡，每卡 batch 32 | 10.09 s/步 | [docs/training-doc/mixed8-200-m32x8x8-context-b128-80k/result.md](docs/training-doc/mixed8-200-m32x8x8-context-b128-80k/result.md) |
| 对照：modulation 512 / 2048 | 4 卡 | 1.7183 / 1.8559 s/步 | `mixed8-200-m8x8-modul-b128-80k`、`mixed8-200-m32x8x8-modul-b128-80k` 的 result.md |

以上实测都在 A100-80GB、global batch 128 下完成。4 卡数字都是两个 4 卡任务在同一台机器上并行时测得，单独占用 4 卡可能略快。GreatLakes 4×A40 使用 global batch 64，不适用本表。context 256 与 1024 没有实测，现有读取守卫能否放行这两档也尚未核实。

## 5. 供画图用的数据（CSV）

```csv
context_len,flops_per_sample_tflop,flops_ratio_vs_512,step_s_8gpu_flops,step_s_4gpu_flops,step_s_8gpu_fit,step_s_4gpu_fit,step_s_8gpu_measured,step_s_4gpu_measured
256,10.66,0.765,1.32,2.49,1.23,2.33,,
512,13.93,1.000,1.73,3.25,1.73,3.25,1.727,3.249
672,16.00,1.149,1.98,3.73,,,1.979,3.854
896,18.94,1.360,2.35,4.42,2.50,4.74,2.43,
1024,20.63,1.481,2.56,4.81,2.77,5.27,,
2048,34.74,2.494,4.31,8.10,5.31,10.09,,10.09
```

```csv
model,seed42,seed7,seed0,mean,std,paper_row,paper_mean,paper_std
context_concat_1x1024_512+96,38.00,36.88,37.13,37.33,0.59,FrameSamp+Context,30.68,1.17
```
