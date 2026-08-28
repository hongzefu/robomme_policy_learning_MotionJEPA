# v1-framesamp-g2（S6 第二块·G2 训练对拍）result

launch 见同目录 [`launch.md`](launch.md)。**G2 判定：四分项全 PASS（第二块通过）**，
对拍 `compare_baseline.py` vs G0 r1 固化产物（判定行原文
`records/compare_vs_g0_r1.txt`）：

```
SCALARS steps=1000 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST_CANONICAL rows=14 mismatch=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致, steps≈1000）
```

- **一行收官**：G2 `scalars_hex.tsv` sha256 = G0 r1/r2 同值
  `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`——
  「两轨迹是否一致」退化为一次 sha256 比较并通过（T7 设计意图兑现）。
- **raw 口径（不计入判据，B3）**：`BATCH_DIGEST mismatch=4 first_bad_step=100
  bad_keys=2（static_image_emb / static_pos_emb）`——与 C.3 预记的已知预期失配
  **逐字吻合**（G0 固化后交付 dtype 经 commitV2.4b 已验收统一；canonical 同步一致）。
  总行 `DET_CHECK=FAIL` 即已拍板不修的工具缺口（raw 拖累），分项判读为准
  （2026-08-27 用户拍板）。
- 五标量 REL 各统计档全为 0.000e+00（bitwise 的量化侧写）。

## 口径与过程

- **起跑 HEAD**：`cf64dddf04bb5147fdb4a2fb9ebd77bff04bd3ce`（launch 预提交 commit，
  clean；env.json `git_dirty=False` 实证）
- **run**：本机 2×RTX 6000 Ada、b8、seed 42、workers 4、1000 步、确定性档
  `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`、
  `MMEVLA_DATA_BACKEND=packed`（env.json `backend_source=explicit`）、库
  `status=verified`（store_meta sha `3990165c…`）、`ALLOW_UNVERIFIED`/`ALLOW_SUBSET`
  均未设、摘要步集与 G0 完全对齐（0/每 100/299/999 共 12 次）、EXP_NAME 独立
- **preflight**：起跑前 `BASELINE_ENV=PASS`（vs G0 r1）；T1 白名单审计与逐项说明见
  launch.md
- **与 S5 并行**（用户批准豁免，launch.md 声明）：S5 已先行三判据全过
  （commitV3.3）——豁免的作废风险未兑现
- **过程监控**：12 次摘要步逐个即时对拍 G0 r1，全程零分叉（步 0 起点锚定 →
  步 999 末点相同）；`BENCH_PASS`、`EXIT_CODE=0`
- **步时（仅留档参考，禁作性能结论——B5：确定性档 + TrainState 摘要 + 与 S5 并行
  三重污染）**：稳态中位 1.960 s/step（n=929，p10 1.850 / p90 2.028）
- **records/**：metrics / param_checksums / batch_digests / index_sequence /
  scalars_hex.tsv / env / run_meta + `BASELINE_MANIFEST.json`（7 产物）+
  `compare_vs_g0_r1.txt`

## 结论

**第一块（S5，commitV3.3）+ 第二块（本 run）双过——按放行规则第 1 条，正式宣称：
「packed IO 重构不改变训练语义」**（受控确定性环境下 1000 步逐位等价；正式平台
吞吐归第三块 GL 验收）。登记簿 T8 已回填。
