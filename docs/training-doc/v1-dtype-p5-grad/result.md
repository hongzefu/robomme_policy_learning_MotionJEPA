# v1-dtype-p5-grad 结果

```
COMPARE_GRAD=PASS kinds=3 mismatches=0
```

`EXIT_CODE=0`。**三个定点 batch 的单步梯度在修复前后逐位相同**，零失配。

## 一、判定明细

| batch | 交付 dtype（A → B，image/pos） | 梯度叶子 | 失配 | 单步 loss (hex) A / B |
|---|---|---|---|---|
| `mixed1`（主判据） | float64/float64 → **bfloat16/float32** | 32 | **0** | `0x1.0d48f00000000p-1` / 同 |
| `allshort`（密度最大） | float64/float64 → **bfloat16/float32** | 32 | **0** | `0x1.0f06060000000p-2` / 同 |
| `allfull`（阴性对照） | bfloat16/float32 → bfloat16/float32（前后同） | 32 | **0** | `0x1.37890e0000000p-1` / 同 |

三项 `loss_bitwise=True`。**主判据与密度最大档的输入 dtype 确实变了**（f64 → bf16/f32），
而梯度与 loss 一位不差——这正是计划的核心论断「bf16→f32→f64 是精确升位，三种交付进
投影层的张量逐位相同」在真实模型、真实数据上的直接实证。

**阴性对照 `allfull` 同样零失配**，说明本次改动没有越出 padding 填充块的 dtype 这一
范围（若改动误伤了满长切片分支或其他路径，它会第一个炸）。

## 二、初始 state 同源

`[state] 同源校验 PASS（177 叶子）`——同 seed / 同 config 现场 `init_train_state`
产出的完整 TrainState，与 G0b r1 `param_checksums.jsonl` 步 0 的 `per_leaf` 逐条相同。
本轮在**修复后**的 commit 上再次通过，与 P3（修复前）结果一致，因此 A/B 两侧初始
状态与 G0b 三者完全相同——梯度若有差异，就只可能来自输入 dtype 这一个变量，归因干净。

## 三、产物

| 路径 | 内容 | 体积 | 处置 |
|---|---|---|---|
| `v1-store/fixtures/dtype-unify-v1/` | **修复后**三个定点 batch 的位型容器 | 64 MB | **长期保留**，sha 清单进 git；按基线计划五节升格为常规回归闸 |
| `v1-store/dtype-unify/v1-dtype-p5-grad-grad/grad_summary.json` | B 侧逐叶梯度摘要与统计 | — | 留档进 git |
| `/data/hongzefu/v1-baselines/dtype-p5-grad-post/` | B 侧梯度数组本体 | 33 GB | P6 验收通过后删除，只留 sha 清单与统计 |
| `v1-store/dtype-unify/p5-grad-report.json` | 对拍报告 | — | 留档进 git |

## 四、这套 fixture 的后续用途

固定初始 state（由 seed 42 现场重建、以 G0b 步 0 摘要校验）+ 三个固定 batch，构成
一个约 2 分钟即可重跑的回归闸：后续任何 commit 只要重算一次单步梯度并与本轮的逐叶
sha256 比对，就能立刻判断它是否改变了训练语义，无须重跑 1.5 h 的千步轨迹
（基线计划五节「单步 fixture 常规回归闸」）。
