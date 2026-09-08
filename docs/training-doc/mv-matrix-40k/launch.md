# mv-matrix-40k — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。正式评估（AGENTS 12），motion 利用率评估阶段 2：4 条件 × 4 policy seed × (test 200 + val 200) = 6400 集闭环矩阵 + 校准批 + 跨卡校验。正本 `docs/motion-utilization.md`，工具 `scripts/motion-variance/`（commitV8.0）。

起跑时间：2026-09-08 04:07:20（校准批起跑；矩阵起跑时间见 result.md）
起跑 commit：代码 `5620f662968ff68d278add6339a818c0be443490`（commitV8.0）；本留档在起跑前提交，工作区 clean（`git status --porcelain` 为空）
被评对象：
- motion：`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`（`PARAM_TREE_EXACT=PASS n_model=59 n_ckpt=59`，`records/param_tree_motion.json`）
- 官方：`v1-store/models/official-mme-vla/perceptual-framesamp-context/79999`（`PARAM_TREE_EXACT=PASS n_model=55 n_ckpt=55`，`records/param_tree_official.json`）
数据来源：benchmark `env_metadata/{test,val}/`，每任务 50 集；donor 库 `v1-store/datasets/4task-motion-400ep/motion`（bank `v1-store/reports/motion-variance/bank-lib`，sha `6e70604da518c156`）

## 口径

- 四条件：`official`（官方，无 motion 路）/ `normal`（真 sidecar）/ `mask`（prefill + 去噪 attention 的 motion key 列门控，位置结构原样，零向量 stub）/ `swap`（有效槽 `motion_emb` 换同任务另一训练集 episode，主 donor + 兜底链）。定义与比较语义见正本第四节。
- 每批 = 一个 (cond, split, seed)，8 worker stride 分片（worker k 评每任务 ep ∈ {k, k+8, …}，即 28/28/24/24/24/24/24/24 集），worker k 固定 GPU k，端口 `9300 + (批号 % 16) × 8 + k`。seed-major：42 → 7 → 2024 → 17，每 seed 内 test → val，每 split 内 official → normal → mask → swap。
- 全部 policy server 加 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`（official 也加）；`XLA_PYTHON_CLIENT_MEM_FRACTION` normal 0.55、其余 0.40；eval 端 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`（单进程 28 集 ≤ 实测 30 上限）。
- policy seed 只换 flow-matching 采样噪声（`MME_VLA_Policy.reset()` 里 `jax.random.key(seed)`），环境实例由 benchmark 元数据 seed 固定；`eval.py --args.model_seed` 只拼目录名。
- 前置闸门（`records/`）：`TEST_SEED_DISJOINT=PASS test=50x4 train_h5=100x4 overlap=0`、`VAL_SEED_DISJOINT=PASS val=50x4 train_h5=100x4 overlap=0`（本轮首次核 val）、两 checkpoint `PARAM_TREE_EXACT=PASS`、adapter 自检（`mv-openloop-40k/records/`）。
- 统计（`summarize_mv.py`）：pooled 400 集主估计；逐 seed 配对差 `mean ± 3.182·sd/√4`；episode bootstrap 10000 次按 split×task 分层、条件与 seed 共享索引；McNemar；等价界 ±2pp（用户 2026-09-08 拍板）。
- 已知污点：donor 为专家演示 motion（分布差异，固定称「异集专家 motion」）；长集后段 donor 回填（实测比例分层报）；审计 D-A1 / B-A2 未消除；test ep 0–9 在 encoder 训练集内；`UNROLL_VS_SCAN` bf16 不逐位、改 f32 语义闸（与本矩阵无关，只影响阶段 1）。

## 校准批与跨卡校验（正式，先于矩阵）

| 步 | 命令 | 记录 |
|---|---|---|
| 校准 normal | `COND=normal SPLIT=test SEED=42 bash scripts/motion-variance/run_batch_mv.sh`（端口 9300） | wall_min=21（200/200，timeout 4，成功 47） |
| 校准 mask | `COND=mask SPLIT=test SEED=42 PORT_BASE=9308 bash scripts/motion-variance/run_batch_mv.sh` | wall_min=28（w7 与开环共卡；排除后约 20–24；200/200，timeout 54，成功 40） |
| 跨卡 | `COND=mask SPLIT=test SEED=42 K=0 GPU=0 PORT=9390 EP_COUNT=2 WORKERS=8 RUN_SUFFIX=-xgpu0 LOG_PREFIX=mv-xgpu0 bash scripts/motion-variance/eval_shard_mv.sh` 与 `GPU=3 PORT=9391 RUN_SUFFIX=-xgpu3 LOG_PREFIX=mv-xgpu3`（4 任务 × ep 0 = 4 集）；`summarize_mv.py --xgpu mv-mask-s42-test-xgpu0 mv-mask-s42-test-xgpu3 mv-xgpu0 mv-xgpu3 0 3` | `MV_XGPU=BITEXACT episodes=4 infers=125 act_sha_match=125/125 outcome_match=4/4`（`EP_COUNT=2` 在 stride 8 下每任务只取 ep 0，实为 4 集而非计划的 8 集） |

校准批结果直接计入矩阵（`run_matrix_mv.sh SKIP_DONE=1` 按 `mv-matrix.log` 跳过）。

## 启动命令

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
tmux new-session -d -s mv-matrix -c "$PWD" \
  "set -o pipefail; PYTHONUNBUFFERED=1 SKIP_DONE=1 bash scripts/motion-variance/run_matrix_mv.sh 2>&1 | tee v1-store/logs/mv-matrix.driver.log"
# 汇总
UV_LINK_MODE=copy uv run --no-sync python scripts/motion-variance/summarize_mv.py --out-dir docs/training-doc/mv-matrix-40k/records [--partial]
UV_LINK_MODE=copy uv run --no-sync python scripts/motion-variance/plot_mv.py
```

## tmux 会话清单（清理时的唯一依据）

本轮起过的会话只有：`mv-matrix`（矩阵驱动）与每批 8 个分片会话 `mv-<t|v><seed>-<cond>-w<k>`（`t42`/`v42`/`t7`/`v7`/`t2024`/`v2024`/`t17`/`v17` × `official|normal|mask|swap` × `w0..w7`，共 256 个，跑完自动退出），以及跨卡校验的两个直接前台运行（无 tmux）。用户既有会话一律不动。清理只允许 `tmux kill-session -t <确切会话名>`。

## 输出

- 分片结果：`v1-store/evaluation/mv-<cond>-s<seed>-<split>-w<k>/ckpt<id>/<seed<S>|val-seed<S>>/{progress.json,log.json,videos/}`（视频不进 git）
- 日志：`v1-store/logs/mv-<t|v><seed>-<cond>-w<k>.{log,server.log,eval.log,probe.jsonl}`、`v1-store/logs/mv-matrix.log`（逐批 `MV_BATCH=`）、`mv-matrix.driver.log`
- 留档：`records/{summary.txt,per_episode_by_cond.json,paired_stats.json,xgpu.txt,mv-matrix.log}`；图 `docs/motion-utilization/figures/`
