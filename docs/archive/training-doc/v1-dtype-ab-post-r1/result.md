# v1-dtype-ab-post-r1 结果（G1 vs G0b）

## 一、判定行原文（如实照录）

```
OK manifest 复验通过: docs/training-doc/v1-grad-baseline-g0b/records/r1/BASELINE_MANIFEST.json（10 条）
SCALARS steps=1000 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
REL key=loss           median=0.000e+00 p95=0.000e+00 max=0.000e+00
REL key=grad_norm      median=0.000e+00 p95=0.000e+00 max=0.000e+00
REL key=llm_grad_norm  median=0.000e+00 p95=0.000e+00 max=0.000e+00
REL key=mem_enc_norm   median=0.000e+00 p95=0.000e+00 max=0.000e+00
REL key=param_norm     median=0.000e+00 p95=0.000e+00 max=0.000e+00
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST rows=14 mismatch=4 first_bad_step=100 bad_keys=2 首个: ["['static_image_emb']"]
BATCH_DIGEST_CANONICAL rows=14 mismatch=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致, steps≈1000）
DET_CHECK=FAIL tier=g1-vs-g0b steps=1000 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=4
```

`compare_baseline.py` 退出码 1；训练本体 `BENCH_PASS`、`EXIT_CODE=0`。

## 二、`DET_CHECK=FAIL` 的唯一成因，与按计划口径的实质判定

**`DET_CHECK` 的三个分项里，两个为 0，第三个是 raw `batch_digest`**：

| 分项 | 值 | 说明 |
|---|---|---|
| `scalar_hex_diff` | **0** | 1000 步 × 5 标量逐位全等 |
| `state_digest_diff` | **0** | 12 次完整 TrainState 摘要（177 叶）逐位全等 |
| `batch_digest_diff` | 4 | **raw 口径**，跨 dtype 必然失配 |

那 4 处失配已逐条核到键级——**步 100 / 299 / 400 / 999，每步恰好两个键**：

```
step 100  raw 失配键 = ["['static_image_emb']", "['static_pos_emb']"]   同步 canonical 失配 = 0
step 299  raw 失配键 = ["['static_image_emb']", "['static_pos_emb']"]   同步 canonical 失配 = 0
step 400  raw 失配键 = ["['static_image_emb']", "['static_pos_emb']"]   同步 canonical 失配 = 0
step 999  raw 失配键 = ["['static_image_emb']", "['static_pos_emb']"]   同步 canonical 失配 = 0
```

这两个键正是本修复改变 dtype 的键（f64 → bf16 / f32），而 raw 摘要的哈希域是
`dtype‖shape‖bytes`——**dtype 变了，raw 必然失配**。其余 10 个摘要步的 batch 恰好整批
满长（两侧本就同为 bf16/f32），raw 也相同；4/14 ≈ 29%，与 b8 下含短样本 batch 的理论
概率 40.4% 同量级，失配来源指向 dtype 而非数值。

**计划早已就此定过口径**：`v1-gradient-baseline.md` 三节「口径限定」与五节对拍矩阵
G1 行均明确「跨 dtype 场景（G1 vs G0）禁用 raw `batch_digests`，须改用 canonical 数值
口径 + 全步 index 序列」；`v1-dtype-unify-plan.md` T4 同调。`compare_baseline.py` 的
总判定行把 raw 一并计入，因此它的 FAIL 是**工具总判定未区分口径**所致。

**按计划规定的判据口径，G1 vs G0b 的实质判定是 PASS**：

| 计划规定的判据 | 结果 |
|---|---|
| 五标量逐步 hex | **0 失配**（rel median/p95/max 全为 `0.000e+00`） |
| 12 次完整 TrainState `state_digest` | **0 失配** |
| 输入侧 canonical `batch_digests` | **`CANON_CHECK=PASS steps=14`** |
| 输入侧 index 全序列 | **`INDEX_SEQ=PASS n=8072`** |
| 输入侧 raw `batch_digests` | 4 处失配——**按口径不计入判定** |

量化兜底判据（`QUANT_EQUIV`）**未启用也不需要**：它的前置是「bitwise 失败」，而本轮
五标量与 TrainState 摘要都是零差异，rel 全部恰为 0。

## 三、最硬的单一证据：`scalars_hex.tsv` 整文件 sha256 相同

```
c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757  G1  (v1-dtype-ab-post-r1)
c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757  G0b r1
c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757  G0b r2
```

1000 步 × 5 个标量的规范化投影（剔除 wall_time 等易变字段）整个文件逐字节相同——
「两条轨迹是否同一条」退化为一次 sha256 比较，人和机器都不会看错。

## 四、运行统计（**仅留档参考，禁作性能结论**——红线 B7 / D4）

本 run 带 12 次完整 TrainState 摘要（单次中位 89.4 s）+ 14 次输入摘要 + 确定性 XLA 档，
util 与步时均被污染。性能结论一律取 speed 链（P7 `v1-g1-speed` vs `v1-g0-speed-r2`）。

以 `scripts/dtype-unify/analyze_util.py` 对两侧**同法**重算（稳态窗步 50–999，剔摘要步
及其下一步，慢步阈值 1.5× 中位）：

| run | 步时中位 | 步时均值 | p10 / p90 | n | 慢步 | util 均值 | 0% 采样 |
|---|---|---|---|---|---|---|---|
| G1（本 run） | 1.9693 s | 1.9497 s | 1.8631 / 1.9932 | 930 | 0 | 96.32% | 0.46% |
| G0b r1（对照） | 1.9676 s | 1.9511 s | 1.8649 / 2.0113 | 930 | 0 | 95.57% | 0.31% |

两者步时均值差 0.07%，在同口径噪声内。bench 自报 `RESULT batch=8 稳态=1.969s/step
(n=929, p10=1.863, p90=1.993)`、`epoch估算 ≈ 27.03 小时`——该外推为确定性档 + 摘要停顿
下的数字，**禁止引用**（生产口径见 `v1-g0-speed-r2` / `v1-g1-speed`）。

## 五、产物

`records/`：`metrics.jsonl`(1000) / `param_checksums.jsonl`(12) / `batch_digests.jsonl`(14)
/ `index_sequence.json`(n=8072) / `scalars_hex.tsv`(1001) / `env.json` / `run_meta.json`
/ `util-lms500.csv` / `util-legacy15s.csv` / `BASELINE_MANIFEST.json`(9 条)。

## 六、结论

**第二块（本机训练梯度一致）通过。** 与第一块（`COMPARE_DTYPE=PASS`）、单步定点梯度
（`COMPARE_GRAD=PASS`，32 叶逐位相同）合并，dtype 修复的等价性在三个层面得证：
数据交付、单步梯度、千步轨迹。按 AGENTS 18 的两块要求，**修复可宣称等价**。
