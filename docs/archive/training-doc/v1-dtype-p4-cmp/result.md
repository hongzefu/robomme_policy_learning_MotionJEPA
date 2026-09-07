# v1-dtype-p4-cmp 结果

```
COMPARE_DTYPE=PASS samples=2600 batches=200 mismatches=0
```

`EXIT_CODE=0`。第一块（非训练轻量化测试）**全部通过**，2,600 个样本 × 全键 +
200 个 batch × 全键零容差一致，零失配。

## 一、修复后的 dtype（逐样本，2,600 个）

**短样本与满长样本的交付 dtype 完全一致，双路径消失**：

| 键 | 短样本 1,068 | 满长 1,532 | 修复前（P3） |
|---|---|---|---|
| `static_image_emb` | bfloat16 | bfloat16 | 短 float64 / 满长 bfloat16 |
| `static_pos_emb` | float32 | float32 | 短 float64 / 满长 float32 |
| `static_state_emb` | float64 | float64 | float64（前后不变） |
| `static_mask` | bool | bool | bool（前后不变） |

`static_state_emb` 恒 f64 是预期内的——它经 `_normalize_state`（norm stats q01/q99
为 f64）被抬到 f64，与 padding 是否发生无关。第三处 `np.zeros` 的修复在这个交付键上
不可观测，其唯一有效证据是本轮同时跑过的归一化前纯函数位型测试（见第三节）。

## 二、判据 4：batch dtype 不再随组成摆动（200 个 batch）

| batch 组成 | 修复前 `static_image_emb` | 修复后 |
|---|---|---|
| `allfull`（全满长） | bfloat16 ×50 | bfloat16 ×50 |
| `allshort`（全短样本） | float64 ×50 | **bfloat16 ×50** |
| `mixed1`（1 短 + 7 满长） | float64 ×50 | **bfloat16 ×50** |
| `random`（随机组成） | **bfloat16 ×28 / float64 ×22** | **bfloat16 ×50** |

`static_pos_emb` 同形（float32 ×200）、`static_state_emb` 恒 float64 ×200。
修复前 `random` 档 28/22 的分裂——即「XLA 被迫编译两份产物」的成因——已完全消失。

## 三、归一化前纯函数位型测试

`compare_dtype_fix.py` 内置该项，判定并入上面的总判定行；同一组断言另由
`uv run pytest scripts/dtype-unify/ -q` 独立覆盖：修复前 **7 passed / 7 skipped**
（skip 的正是这 7 条，测试自动探测「修复未落地」并说明原因），修复后
**14 passed / 0 skipped**。覆盖 t ∈ {1,2,3,30,31,32,33} 七档，逐档断言：
三键输出 dtype 跟随各自输入、非填充区逐字节不变、填充区全零、mask 恒 bool 且
有效区全 True / 填充区全 False。

## 四、产物

| 路径 | 内容 | 体积 | 处置 |
|---|---|---|---|
| `v1-store/dtype-unify/v1-dtype-p4-cmp/` | 修复后 2,600 样本 + 200 batch | 6.2 GB | 收官清理 |
| `v1-store/dtype-unify/p4-compare-report.json` | 对拍报告（判定行 + 零失配） | — | 留档进 git |

修复后 dump 比修复前小 0.9 GB（7.1 → 6.2 GB），差值即短样本 memory 键从 f64 降回
bf16/f32 省下的字节——这也是修复在 IO 侧收益的一个侧证。
