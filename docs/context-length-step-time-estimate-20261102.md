# context 注入长度与训练步时的增长估算（2026-11-02）

## 结论

**仓库里没有 context 256 与 1024 的实测步时。** 已有实测点是 context 512、672（512+motion160）、896（512+motion384）和 2048。下表中 256 与 1024 的步时由这些实测点拟合得出，属于估算，没有经过 GPU 实跑验证。

条件：A100-80GB，global batch 128。

| context 长度 | 4 卡步时 | 8 卡步时 | 4 卡 80k 步 | 8 卡 80k 步 | 相对 512 |
|---|---|---|---|---|---|
| 256 | ≈2.3 s（外推） | ≈1.2 s | ≈52 h | ≈27 h | ×0.72 |
| **512** | **3.25 s（实测）** | **1.73 s（实测）** | 72 h | 38 h | ×1 |
| 1024 | ≈5.3 s（内插） | ≈2.8 s | ≈117 h | ≈62 h | ×1.6 |
| **2048** | **10.09 s（实测）** | 未测，≈5.3 s | 224 h | ≈118 h | ×3.1 |

增长规律：长度从 512 翻倍到 1024，步时约 ×1.6；再翻倍到 2048，又约 ×1.9。增长比线性快，主要原因是 attention 计算量随序列长度平方增长。实际进入主干的序列除 context 外还有约 596 个固定 token（两路图像、文本 64、动作 20），见 `ctx-stock-scaling-20260928/result.md` 中的 logits 逻辑量公式 `16×8×(budget+596)²×4`。

## 实测依据

| 配置 | 卡数 | 稳态步时 | 来源 |
|---|---|---|---|
| context 512（纯视觉） | 8 卡 | 1.727 s/步（前后两次独立运行均值，漂移 +0.0045%） | [ctx-stock-scaling-20260928/result.md](training-doc/ctx-stock-scaling-20260928/result.md) |
| context 512（纯视觉） | 4 卡（两个 4 卡任务同机并行） | 3.247 / 3.251 s/步 | 同上 |
| context 512+motion160（共 672） | 8 卡 | 1.979 s/步 | 同上 |
| context 512+motion160（共 672） | 4 卡（两个 4 卡任务同机并行） | 3.845 / 3.863 s/步 | 同上 |
| context 512+motion384（共 896） | 8 卡 | 2.43 s/步（80k 步共 40 h 34 min） | [1101-gl-ablation-encoders-ctx-b64-80k-plan.md](../1101-gl-ablation-encoders-ctx-b64-80k-plan.md) 硬件对照表 |
| context 2048（32 帧 × 8×8） | 4 卡，每卡 batch 32 | 10.09 s/步（step 100→400，300 步 3027 s），每卡显存峰值 79445 MiB（约 97%）；该 run 已叫停 | [mixed8-200-m32x8x8-context-b128-80k/result.md](training-doc/mixed8-200-m32x8x8-context-b128-80k/result.md) |
| 对照：modulation 512 | 4 卡 | 1.7183 s/步（5000→79999 平均） | [mixed8-200-m8x8-modul-b128-80k/result.md](training-doc/mixed8-200-m8x8-modul-b128-80k/result.md) |
| 对照：modulation 2048 | 4 卡 | 1.8559 s/步（5000→79999 平均） | [mixed8-200-m32x8x8-modul-b128-80k/result.md](training-doc/mixed8-200-m32x8x8-modul-b128-80k/result.md) |

modulation 从 512 到 2048 步时只增加约 8%，基本不受长度影响；同为 2048 时 context 比 modulation 慢约 5.4 倍。

## 估算方法

- **4 卡曲线**：用 4 卡的 512、672、2048 三个实测点（3.249、3.854、10.09 s/步）拟合二次曲线：

  ```text
  步时(s) ≈ 1.481 + 3.203e-3·n + 4.888e-7·n²    （n 为 context 长度）
  ```

- **8 卡换算**：按实测的「4 卡 ÷ 8 卡」比值（512 为 1.88，672 为 1.95）取约 1.9 换算。直接用 8 卡三点（512、672、896）拟合的曲线外推到 2048 会得到 6.56 s，因为它在 896 以上没有实测点约束，所以不采用。
- **交叉检查**：4 卡曲线在 896 处给出 4.74 s，除以 1.9 得 2.50 s，与 8 卡实测 2.43 s 相差约 3%。
- **80k 总时长**：只按纯步时 × 80000 计算，不含初始化、编译与 16 次 checkpoint 保存；按 modulation 正式训练的经验，这部分大约再增加 1%。

## 误差与适用边界

- **1024** 位于实测点之间，属于内插，可信度较高；**256** 在最小实测点以下，属于外推，误差可能较大；**2048 的 8 卡步时**没有实测。
- 4 卡实测都是两个 4 卡任务在同一台 8 卡机上并行时测得（context 2048 当时与 GPU 4–7 上的 modulation 2048 并行）；单独占用 4 卡可能略快。
- 本表只适用于 A100-80GB、global batch 128。**GreatLakes 4×A40 使用 global batch 64，不能套用**：历史 `v1-prod-trend-10h`（4×A40、b64、context 512、无 motion）为 4.81 s/步；正在运行的 `gl-abl-encoders-train`（context 896）step 0→100 为 6.14 / 6.39 s/步。
- 2026-09-28 时原生产读取守卫只放行 context 512 档，context 2048、4096、8192 在 `FrameSampDataset.__init__` 被形制断言拒绝（`ctx-stock-scaling-20260928/result.md`，`ORIGINAL_CONTEXT_GUARDS=PASS`）；之后 2048 档已在 `mixed8-200-m32x8x8-context-b128-80k` 中实际训练。context 256 与 1024 能否通过现有守卫尚未核实。
- 如需精确数字，可以对 256 与 1024 各做一次 400 步短测速，取 step 100→300 稳态窗口，口径与 `ctx-stock-scaling-20260928` 一致。
