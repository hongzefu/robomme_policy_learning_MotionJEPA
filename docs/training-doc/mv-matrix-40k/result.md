# mv-matrix-40k — 结果

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0`）**。执行 commit `49f68c7`（clean HEAD；校准两批在 `26ff2b4`，代码同为 commitV8.0 `5620f66`）。矩阵墙钟 `05:04:11 → 10:47:05`（**5.71 h**，30 批）+ 校准两批 49 min + 跨卡 5 min ≈ **6.6 h**。被评：motion `awsprod40k-b128-motion/39999` 与官方 `perceptual-framesamp-context/79999`。

## 结论先行

```
MV_GRID=DONE cells=32/32 episodes=6400/6400 errors=0 missing=0 workers=8
SPLIT_SEED_MATCH=PASS split=test n=3200 mismatch=0 unlogged=0 / split=val n=3200 mismatch=0
MV_PAIRING=PASS cells=32 episodes=400 missing_pairs=0
MV_STRUCT_IDENTICAL=PASS ×8（test/val × 4 seed）first_infer_sha_match=200/200
DONOR_COVER_ACTUAL windows=756823 exact=55.8% fallback=15.9% cycle=28.3% cross_seg=0
MV_PAIRED pool=pool400 pair=normal-mask     dmean=+5.00pp t_ci=[+3.44,+6.56] boot_ci=[+2.31,+7.62] bonf_t=[+2.36,+7.64] mcnemar_b=155 c=75 verdict=POSITIVE
MV_PAIRED pool=pool400 pair=normal-swap     dmean=+2.44pp t_ci=[+0.16,+4.72] boot_ci=[-0.62,+5.62] verdict=NOT_DETECTED
MV_PAIRED pool=pool400 pair=mask-swap       dmean=-2.56pp t_ci=[-3.44,-1.68] boot_ci=[-5.56,+0.38] verdict=NOT_DETECTED
MV_PAIRED pool=pool400 pair=normal-official dmean=-0.81pp t_ci=[-2.63,+1.01] boot_ci=[-4.31,+2.69] verdict=NOT_DETECTED
MV_XGPU=BITEXACT episodes=4 infers=125 act_sha_match=125/125 outcome_match=4/4 gpu_a=0 gpu_b=3
MV_MATRIX=PASS batches=32/32 wall_h=5.71 ; MV_SUMMARY=PASS cells=32 blocking=13/13 observe=83
```
（全文 `records/summary.txt`，逐格数值 `records/paired_stats.json`，逐集 `records/per_episode_by_cond.json`。）

**按预注册措辞（正本第六节）：**
1. **motion token 通路带来收益**：pooled 400 集 `normal − mask` = **+5.00pp**，t 区间（df=3）[+3.44, +6.56]、分层 bootstrap [+2.31, +7.62]、Bonferroni 同时区间 [+2.36, +7.64] 三者全在 0 以上；四 seed 逐一为 +5.25 / +6.25 / +3.25 / +5.25，McNemar 155:75。test 与 val 各自也成立（+4.25 [+2.36, +6.14]；+5.75 [+3.86, +7.64]）。
2. **「内容被读取」在成功率层面尚未检出**：`mask − swap` = −2.56pp（异集内容反而略好于屏蔽），t 区间 [−3.44, −1.68] 全负但 bootstrap [−5.56, +0.38] 含 0 → NOT_DETECTED；`normal − swap` = +2.44 [t +0.16, +4.72；boot −0.62, +5.62] 同为 NOT_DETECTED。结合阶段 0（换内容的动作差 6.4× 噪声）：**动作层面内容确实被读，但成功率收益主要来自通路存在 / 位置信号，正确内容相对异集内容的增量尚未分辨**。
3. **自训模型 vs 官方 baseline 尚未检出差异，也不判等价**：`normal − official` = −0.81pp，t [−2.63, +1.01] 下界跨出 ±2pp；test +1.50、val −3.12（val 上 t 区间 [−4.32, −1.93] 全负但 bootstrap [−8.12, +1.75] 含 0）。
4. **行为后果**：屏蔽通路使 **27.0% 的集走满 1300 步 timeout**（normal 1.6%、official 0.4%、swap 10.1%），mask 的步数均值 560 vs normal 232——没有 motion 通路时策略明显「不收敛 / 不做终止动作」。

## 成功率（pooled 400 集，4 seed 均值 ± sd；逐 seed）

| 条件 | pooled 400 | test 200 | val 200 | timeout | 步数 median / mean |
|---|---|---|---|---|---|
| official | 25.25 ± 1.10（25.75 / 26.5 / 24.75 / 24.0） | 23.37 | 27.12 | 0.4% | 219 / 216 |
| normal | 24.44 ± 0.52（23.75 / 25.0 / 24.5 / 24.5） | 24.87 | 24.00 | 1.6% | 217 / 232 |
| mask | **19.44 ± 0.97**（18.5 / 18.75 / 20.5 / 20.0） | 20.62 | 18.25 | **27.0%** | 360 / 560 |
| swap | 22.00 ± 1.24（21.5 / 20.5 / 23.25 / 22.75） | 22.38 | 21.62 | 10.1% | 219 / 331 |

## 分层配对差（pooled 400；t 区间 / bootstrap 区间；verdict）

`normal − mask`：ButtonUnmask −0.5 [t −9.7,+8.7]（ND）；VideoUnmask **+4.0** [+2.2,+5.8]/[+0.8,+8.0]（POS）；ButtonUnmaskSwap **+9.75** [+3.6,+15.9]/[+4.3,+15.8]（POS）；VideoUnmaskSwap **+6.75** [+1.5,+12.0]/[+1.5,+12.5]（POS）。难度 easy +6.0（POS）、medium +1.3（ND）、hard +6.5（POS）。启动空窗 k=0 +4.6（t [−1.6,+10.9] ND）、k>0 +5.4（POS）。steps 四分位（取 normal seed 42，描述性）Q1 +1.9 → Q4 **+9.95**（POS）：集越长收益越大。

`mask − swap`：ButtonUnmaskSwap **−6.5** [−9.3,−3.7]/[−11.5,−1.5]（NEGATIVE：异集内容好于屏蔽），其余三任务 ND；donor 档 exact −1.25（ND）、fallback −5.5（ND）、cycle −5.7（NEGATIVE）。
`normal − swap`：四任务全 ND；donor=fallback +12.3（POS，n=59）、exact +1.0（ND，n=280）、cycle −0.4（ND）。

图：`docs/motion-utilization/figures/fig2-rates.*`（成功率）、`fig3-forest.*`（森林图）、`fig6-donor.*`（donor 覆盖）。

## 校准批、跨卡校验与耗时

| 项 | 结果 |
|---|---|
| 校准 normal（seed 42 test） | 21 min，200/200，timeout 4，成功 47 |
| 校准 mask（seed 42 test） | 28 min（w7 与开环共卡；矩阵内同条件批 9–20 min），200/200，timeout 54，成功 40 |
| 跨卡 `MV_XGPU` | **BITEXACT**：cond=mask seed 42 test 4 集（`EP_COUNT=2` 在 stride 8 下每任务只取 ep 0），GPU 0 vs 3 逐次推理动作 sha 125/125 一致、成败 4/4 一致 |
| 各条件单批墙钟均值 | official 6.5 / swap 7.9 / mask 14.9 / normal 17.9 min（`records/mv-matrix.txt`） |
| 矩阵 | 30 批 5.71 h（校准两批 `SKIP_DONE` 复用）；估算 8.5 h 偏保守——official/swap 比预估快、mask 的 timeout 拖长被 stride 均衡吸收 |

## 盲区诚实清单

- 结论限定于 checkpoint 39999、固定 400 集面板（test+val）、三种干预；不能替代「带/不带 motion 训练」的因果对照。
- donor 是训练集专家演示 motion（分布差异，固定称「异集专家 motion」）；长集后段回填（实测 exact 55.8% / fallback 15.9% / cycle 28.3%），donor 档分层 n 小（59 / 61）。
- `mask−swap` 与 `normal−swap` 的 t 区间与 bootstrap 区间结论不一致（t 更窄），按预注册规则以两者一致为准。
- 跨卡校验只 4 集（计划写 8 集，`EP_COUNT` 语义所致）；矩阵 `w_k → GPU k` 固定映射使跨卡差异不进主比较。
- 审计 D-A1 / B-A2 原样保留；test ep 0–9 在 MotionJEPA encoder 训练集内；`UNROLL_VS_SCAN` bf16 不逐位（只影响阶段 1，见 `mv-openloop-40k/result.md`）。
- timeout 计失败；error 0 例；视频未人工抽看。

## tmux 会话清单（本轮起过的，已全部自动退出）

`mv-calib`、`mv-calib-smi`、`mv-openloop`、`mv-matrix`；分片会话 `mv-<t|v><seed>-<cond>-w<k>`（4 seed × 2 split × 4 条件 × 8 = 256 个，含校准两批）；smoke 期的 `mv-t42-mask-bsmoke-w0/w1`。用户会话 `0`、`1`、`claude-private`、`hf-ds400ep-export` 未动。

## 产物

- `records/{summary.txt, per_episode_by_cond.json, paired_stats.json, xgpu.txt, mv-matrix.txt, mv-matrix.driver.txt, mv-calib.driver.txt, val_seeds.json, test_seeds.json, param_tree_*.json}`
- 分片结果 `v1-store/evaluation/mv-<cond>-s<seed>-<split>-w<k>/…`（视频 ~4.8 GB 不进 git）；日志 `v1-store/logs/mv-*`；GPU 采样 `v1-store/logs/mv-calib.smi.csv`
