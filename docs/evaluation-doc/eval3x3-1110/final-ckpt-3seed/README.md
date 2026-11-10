# 最后一个 checkpoint × 3 个模型 seed：五个模型的逐局结果归档

2026-11-10 美东，用户决定不再补 3 checkpoint × 3 seed，改为每个模型只报最后一个 checkpoint 的 3 个模型 seed（42/7/0）。用户原话：「给出方案把我们也全部改成三个run 就不用跑九次了。」「先按照现在的这个五个模型跑三次的这个现状，呃，你你直接归档就就可以了。这个其他情况下都不需要这个，呃，就就不跑三个 checkpoint 了。」

口径：官方 test 16 任务 × 50 局 = 800 局（清单签封 `991c3496…`，环境 seed 不变），官方 profile，1300 步上限，正常失败不重试，基础设施错误每局最多重试 1 次；标准差为三次 All 的样本标准差（ddof=1）。

| 模型（总表 `model` 列） | run | checkpoint | seed 42 / 7 / 0 成功局数 | All 均值 ± 标准差 | 评测平台 |
|---|---|---|---|---|---|
| MotionMem | `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k` | 59999 | 304 / 295 / 297 | 37.33 ± 0.59 | GreatLakes |
| FrozenWanPooling | `orig100-ctx32x4x4-wan-timepool8x8-b128-60k` | 59999 | 291 / 280 / 280 | 35.46 ± 0.79 | AWS 本机 |
| wo_depth_dec | `gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k` | 79999 | 276 / 299 / 308 | 36.79 ± 2.06 | GreatLakes |
| wo_state_dec | `gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k` | 79999 | 263 / 291 / 274 | 34.50 ± 1.76 | GreatLakes |
| FrameSamp1024 | `orig100-ctx-framesamp1024-b64-80k-8gpu` | 79999 | 256 / 255 / 261 | 32.17 ± 0.40 | AWS 本机 |

## 文件

- `all-models-final-ckpt-3seed.episodes.csv`：总表，12000 行（5 模型 × 3 seed × 800 局），列 `model,checkpoint,model_seed,task,episode_id,success,outcome,platform`，按 `(task, episode_id)` 排序；sha256 `26fdbdc449e711fa98e5d92f6e05599cfd8ce3783c947020b8e6099aaa91e353`。合成时核对：15 份的局号集合完全相同（`ALIGN=PASS models=5 seeds=3 episodes=800 rows=12000`），每份成功数与上表一致。
- 来源文件原样保留：
  - MotionMem 与两个消融：GreatLakes 原始记录导出，来自策略仓分支 `v13-gl-episode-records@b219c1a`，来源路径、run_identity 与 sha256 见 `gl-README.md`。
  - FrozenWanPooling：本机 1107 评测 `v1-store/evaluation/ctx-wan-timepool1024-3seed-1107/seed{42,7,0}/run800`，执行代码 `ffb307cccdbe`，经 `scripts/evaluation/export_episodes.py` 导出；run 身份见 `wanpool-59999-seed*.identity.json`。
  - FrameSamp1024：本机本轮评测，执行代码 `f0149d8`，见上级 `result.md` 与 `records/`。

## 说明

- 本表只换模型采样 seed，三次共用同一批 800 局，逐局可配对；不含 checkpoint 维度的变化。
- 评测平台不同（GreatLakes A40/A6000 与 AWS A100），表注需写明。
- 本轮已跑但不进主表的额外结果：FrameSamp1024 70000 seed 42 = 228/800（28.50%），见上级 `result.md`，可作附录中的 checkpoint 敏感性参考。
