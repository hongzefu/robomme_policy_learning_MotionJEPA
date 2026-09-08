# scripts/motion-variance — motion memory 利用率评估

回答一个问题：`awsprod40k-b128-motion/39999` 每步多喂的 96 个 motion token，**模型在推理时读不读、读到的内容对动作与闭环成功率有没有影响**。冻结 checkpoint，只评估不训练。计划正本：`docs/motion-utilization.md`；留档 `docs/training-doc/mv-openloop-40k/`、`docs/training-doc/mv-matrix-40k/`。

## 三阶段

| 阶段 | 做什么 | 入口 | 产物 |
|---|---|---|---|
| 0 开环动作差 | 训练集决策点上，同一观测同一噪声，比较 none / mask / swap 的最终动作，标尺 = noise seed 间动作差 | `run_open_loop.py --stage 0` | `open_loop.json` `OL_ACT_DELTA` `OL_ZEROK_NULL` |
| 1 逐层机制 | 18 层手工展开：attention 富集度、逐层门控 / KV 干预后的**最终动作**差、固定-loss 逐层输入梯度 | `run_open_loop.py --stage 1` | `LAYER_ATTN` `LAYER_ACT_DELTA` `LAYER_GRAD` |
| 2 闭环矩阵 | 4 条件 × 4 policy seed × (test 200 + val 200) = 6400 集，8 worker stride 分片，pooled 400 配对统计 | `run_matrix_mv.sh` → `summarize_mv.py` → `plot_mv.py` | `MV_PAIRED` `MV_SUMMARY` 六组图 |

## 四条件的精确定义

| 条件 | checkpoint | 改了什么 | 没改什么 |
|---|---|---|---|
| `official` | 官方 `perceptual-framesamp-context/79999` | —（没有 motion 路） | — |
| `normal` | motion 39999 | — | — |
| `mask` | motion 39999 | prefill + 去噪两处 attention mask 的 **motion key 列** 置 False（非 motion 的 query 读不到） | `motion_emb/motion_pos/motion_mask/mem_order`、`positions=cumsum(input_mask)-1` 全部原样 |
| `swap` | motion 39999 | 有效槽 `motion_emb` 换成同任务**另一训练集 episode** 的内容（主 donor + 兜底链，见 `mv_donor_bank.py`） | `motion_pos/motion_mask/mem_order` 来自接收方 |

**绝不用改 `motion_mask` 的方式屏蔽**——那会让其后所有 token（含动作）的 RoPE 位置整体前移。比较语义：`normal−mask` 是整个 motion token 通路（内容 + 位置投影）的效应；`mask−swap` / `normal−swap` 才涉及内容。

## 文件

| 文件 | 职责 |
|---|---|
| `mv_common.py` | 常量、`leaf_sha256`、`Verdict`、源码护栏（空白规范化）、`episode_plan`（eval / serve / 验收三方唯一遍历序：stride 8 → 28/28/24×6）、motion 列算法 |
| `mv_model_adapter.py` | 复刻 `sample_actions` + key 列门控（`f_sample`）；18 层展开含完整 10 步去噪（`f_sample_layered`）；固定-loss 梯度；`--selftest` |
| `mv_points.py` | 决策点驱动：按真实评估节奏喂 h5 帧，产出生产口径 `HistAugObservation` + 训练真值动作 |
| `mv_donor_bank.py` / `build_donor_bank.py` | donor bank（library 源）：主 donor（hash 固定、四 seed 共用）+ 同任务兜底链 + 超全库上限循环 |
| `serve_policy_mv.py` | normal / mask / swap 三条件 policy server（实例属性遮蔽，`src/` 零改动；不起 stub 子进程） |
| `robomme/` | `legacy-eval/robomme-local` 的评估客户端副本，`eval.py` 改 4 处（身份键、`MV_EP_START/DONE`、续评守卫、docstring） |
| `eval_shard_mv.sh` / `run_batch_mv.sh` / `run_matrix_mv.sh` | 单分片 / 一批 8 worker / 32 批 seed-major 矩阵 |
| `summarize_mv.py` | 合并、核对（`MV_GRID` `SPLIT_SEED_MATCH` `MV_PAIRING` `MV_STRUCT_IDENTICAL` `DONOR_COVER_ACTUAL`）、pooled 400 配对统计、`--xgpu` 跨卡校验 |
| `plot_mv.py` | 六组图 + `PLOT_CELL_CHECK` |

## 一次完整跑法

```bash
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'   # 全部 python 入口硬闸
# 0. 自检（≤10 min，1 卡）
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/motion-variance/mv_model_adapter.py --selftest --unroll --grads --f32-check --max-points 1
# 1. donor bank（CPU，<1 min）
uv run --no-sync python scripts/motion-variance/build_donor_bank.py --source library --out v1-store/reports/motion-variance/bank-lib
# 2. 开环阶段 0+1（1 卡，~1 h）
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/motion-variance/run_open_loop.py --stage 0,1 --out v1-store/reports/motion-variance/open_loop.json
# 3. 闭环矩阵（8 卡，clean HEAD；detached tmux `mv-matrix`）
SKIP_DONE=1 bash scripts/motion-variance/run_matrix_mv.sh
# 4. 汇总与出图
uv run --no-sync python scripts/motion-variance/summarize_mv.py --out-dir docs/training-doc/mv-matrix-40k/records
uv run --no-sync python scripts/motion-variance/plot_mv.py
```

## 判定行（阻断 / 观察）

阻断：`MV_NOOP_BITEXACT`（复刻 + 门控算子 vs 生产逐位）、`MV_PAD_INVARIANT`、`MV_MASK_CONTENT_NULL`、`UNROLL_VS_SCAN_F32`、`LAYER_SELFCHECK_F32`、`ATTN_UNIFORM_SELFCHECK`、`GRAD_SELFCHECK`、`DONOR_BANK`、`DONOR_CROSS_SEG`、`OL_ZEROK_NULL`、`LAYER_GRAD_PAD_ZERO`、`MV_GRID`、`SPLIT_SEED_MATCH`、`MV_PAIRING`、`MV_STRUCT_IDENTICAL`、`MV_BATCH`、`MV_MATRIX`。
观察：`UNROLL_VS_SCAN_BF16`（展开与 nn.scan 是两份 XLA 编译产物，bf16 下 rel_fro≈3e-3 不逐位，与仓库 `VT_FULL_VS_CACHED` 同性质；语义等价由 f32 闸判定）、`DONOR_COVER_EST/ACTUAL`、`OL_ACT_DELTA`、`LAYER_*`、`MV_PAIRED`、`MV_RATE`、`MV_XGPU`。

## 与主线的边界

`src/`、`examples/`、`third_party/`、`scripts/training/` 零改动；`src/mme_vla_suite/policies/` 下不得出现 `MotionStore` / `MMEVLA_MOTION_STORE`（`check_config_provenance.py` 静态闸），查表逻辑全在本目录。`MV_ALLOW_DIRTY=1` 只供 smoke，矩阵一律拒绝。tmux 会话前缀 `mv-`，清理只允许 `tmux kill-session -t <确切名>`。

## 已知限制

donor 是训练集专家演示的 motion、接收方是策略 rollout（分布差异，报告固定称「异集专家 motion」）；库 exec 窗上限 BU 27 / BUS 33 / VU 19 / VUS 22，长集后段必然回填（比例实测报并分层）；本轮不能替代「带/不带 motion 训练」的因果对照；跨卡一致性只做 8 集校验（`MV_XGPU`），矩阵 `w_k → GPU k` 固定映射使同一集四条件永远同卡。
