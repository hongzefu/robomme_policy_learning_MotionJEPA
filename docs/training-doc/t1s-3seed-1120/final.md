# t1s-3seed-1120 最终汇总

TABLE1_3SEED=PASS rows=12 seeds=3 ckpt_rows=4 ckpts=3 runs_new=32

每次 692 局；新 seed / 新 ckpt 全在 AWS 8×A100 上评估；标 GL 的 seed 7 / 50k 为既有 GL 评估栈结果（K2 不考虑跨硬件）；前缀库：新评估用本机库（与 GL 原库有 40 局 BinFill 前缀不同，用户确认不处理）。

## 三 seed（seed 7 / 0 / 42，ckpt 50k；数字为成功局数，括号为 %）

| 行 | seed 7 | seed 0 | seed 42 | 均值 ± 样本标准差（%） |
|---|---|---|---|---|
| MotionMEM 512+160 modulation（seed 7 GL） | 378 (54.62) | 387 (55.92) | 375 (54.19) | 54.91 ± 0.90 |
| E2E-Enc 512+160 modulation | 337 (48.70) | 344 (49.71) | 336 (48.55) | 48.99 ± 0.63 |
| Wan-1 512+160 modulation | 298 (43.06) | 281 (40.61) | 286 (41.33) | 41.67 ± 1.26 |
| MotionMEM 512+160 context | 254 (36.71) | 245 (35.40) | 247 (35.69) | 35.93 ± 0.68 |
| Wan-9 512+720 modulation | 262 (37.86) | 258 (37.28) | 269 (38.87) | 38.01 ± 0.80 |
| NativeMEM | 290 (41.91) | 291 (42.05) | 278 (40.17) | 41.38 ± 1.05 |
| FrameSamp 512 modulation（seed 7 GL） | 211 (30.49) | 231 (33.38) | 223 (32.23) | 32.03 ± 1.45 |
| FrameSamp 1024 modulation（seed 7 GL） | 291 (42.05) | 279 (40.32) | 279 (40.32) | 40.90 ± 1.00 |
| FrameSamp 2048 modulation（seed 7 GL） | 312 (45.09) | 303 (43.79) | 316 (45.66) | 44.85 ± 0.96 |
| FrameSamp 4096 modulation（seed 7 GL） | 313 (45.23) | 323 (46.68) | 301 (43.50) | 45.13 ± 1.59 |
| FrameSamp 512 context（seed 7 GL） | 158 (22.83) | 158 (22.83) | 146 (21.10) | 22.25 ± 1.00 |
| FrameSamp 1024 context（seed 7 GL） | 160 (23.12) | 148 (21.39) | 152 (21.97) | 22.16 ± 0.88 |

## 三 ckpt（FrameSamp modulation，seed 7；40k / 45k / 50k）

| 行 | 40k | 45k | 50k | 均值 ± 样本标准差（%） |
|---|---|---|---|---|
| FrameSamp 512 modulation | 211 (30.49) | 218 (31.50) | 211 (30.49) | 30.83 ± 0.58 |
| FrameSamp 1024 modulation | 278 (40.17) | 292 (42.20) | 291 (42.05) | 41.47 ± 1.13 |
| FrameSamp 2048 modulation | 320 (46.24) | 300 (43.35) | 312 (45.09) | 44.89 ± 1.45 |
| FrameSamp 4096 modulation | 322 (46.53) | 313 (45.23) | 313 (45.23) | 45.66 ± 0.75 |

数据：`table1_3seed.json`；逐局：`../t1s-*/records/merged/progress.json`。
