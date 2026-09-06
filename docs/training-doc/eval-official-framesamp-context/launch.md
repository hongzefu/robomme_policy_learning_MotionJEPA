# eval-official-framesamp-context — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。正式评估，按 AGENTS 12 留档。

起跑时间：2026-09-06（预检与起跑时刻见本文末「时间线」）
起跑 commit：代码 `f210b40`（commitV5.4，stride 交错分片 + 参数树核对脚本）；本留档在起跑前提交，工作区 clean（`eval_shard.sh` 断言 `git status --porcelain` 为空）
被评 checkpoint：**官方 MME-VLA `perceptual-framesamp-context` 80k 步权重**（HF 合集仓 `Yinpei/mme_vla_suite` @ `5db4d53ddb98c7f80cab08792dd53d985d712ab1`，
文件 `perceptual-framesamp-context/79999.zip` 11,574,223,742 B，sha256 `387d4bd500a29588d81c2675a83be351f84383b93bea4161b8f7fadbf91a8ae8`，本地 / HF LFS / 钉死值三方一致），
解压落 `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999/{params,assets/robomme/norm_stats.json}`，run 根只有 `history_config.txt`（内容 `perceptual-framesamp-context.yaml`，无换行）。
下载 / 校验 / 解压全记录见 `records/ckpt_download.txt`（脚本 `records/download.sh`，tmux 会话 `evoffctx-dl`，160 s 下载完，`DOWNLOAD=PASS`）。
HF 单模型仓只发布了 `perceptual-framesamp-modul`；context 只在合集仓里有，用户拍板只评 context。

## 为什么评它、与我们 run 的可比性边界

`eval-awsprod40k-b128-motion/result.md` 给出我们自己 run（batch 128 / 40k 步 / lr 1e-4 / 带 motion memory）的 28.0%，但没有同口径的官方参照。本轮在**完全相同的评估口径**
（同 benchmark test split、同 `eval.py` 代码、同 policy 采样 seed 42、同 8 卡脚本）下评官方发布的同架构（context 集成）权重。
**只能并列、不能归因**：两者模型权重、训练配置（官方 batch 64 / 80k 步 / lr 5e-5）、是否带 motion 全不同。

## 加载口径（官方 checkpoint 走旧兼容分支）

- `serve_policy.py --policy.config=mme_vla_suite --policy.dir=<…>/79999`：`create_trained_policy` 发现 run 根无 `history_config.resolved.yaml`、有 `history_config.txt` → 旧兼容分支：
  读 txt 原文当配置文件名，`get_history_config` 按 cwd 相对路径加载 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml`；server 日志应有
  ` == You are using None, changing to perceptual-framesamp-context.yaml ==`。norm stats 从 `79999/assets/robomme/norm_stats.json` 读。
- **HEAD 的该 yaml 相对官方 commit `89efeaab` 只多 `motion:` 块 13 行、`enabled: false`**（`git show 89efeaab:… | diff -`，全文见 `records/ckpt_download.txt`）；关闭态下 `_motion_enabled` 为假、
  两个 motion 模块根本不创建，**不起 sidecar**。
- 旧兼容分支是非严格恢复（`model.load(params)` 默认 `remove_extra_params=True`，多出参数会被静默裁掉）。为掀开这层遮蔽，起跑前用 `scripts/training/prod/check_ckpt_param_tree.py`
  在 CPU 上把官方 params 与 HEAD 按该 yaml 建的模型逐路径比对（`records/param_tree.json`）：
  `PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-context.yaml yaml_sha256=38db33fa3c48ec8a n_model=55 n_ckpt=55 missing=0 extra=0 shape_mismatch=0`。
  同脚本对我们的 `awsprod40k-b128-motion/39999`（motion 开启）实测 `n_model=59 n_ckpt=59` PASS——多出的 4 叶正是两个 motion 模块，脚本能分辨两种参数树。

## benchmark 口径（与 eval-awsprod40k-b128-motion 逐项相同）

- `examples/robomme/eval.py` 默认 `max_steps=1300`、`obs_horizon=16`；`env_runner.py` 硬编码 `dataset="test"`，每任务 50 集。
- policy 采样 seed 42（`serve_policy.py --seed=42`）；`--args.model_seed=42` 只决定结果路径。单 seed、单 checkpoint。
- test seed 核对 `records/test_seeds.json`：`TEST_SEED_DISJOINT=PASS test=50x4 train_h5=100x4 overlap=0 test_episodes_contiguous=True`、`TRAIN_H5_IN_TRAIN_SPLIT=PASS missing=0`（与上一轮相同）。

## 分片：8 个 worker，stride 交错（负载均衡）

上一轮按「任务 × 两半」切片，8 片墙钟 9.4 / 11.3 / 15.5 / 20.5 / 15.2 / 16.5 / 36.4 / 25.4 min——总工作量 150 min、理想均分 18.8 min/卡，实际由最慢片 36.4 min 决定（超额 1.94×）。
主因是任务级差异：Video 类每集约 7 个 16 步窗即终止（server 日志 `add_buffer` 计数 150–203/片），Button 类 20–30 窗（387–718/片）。
本轮 `MODE=stride`：worker w 跑**全部 4 任务**，每任务取 ep ∈ {w, w+8, …}（`eval.py --args.episode_start=w --args.episode_stride=8 --args.max_episodes=0`），任务级差异被均摊到每个 worker，
残余不均衡只剩「28 vs 24 集」与集内方差。覆盖自证 `STRIDE_COVER=PASS counts=[7, 7, 6, 6, 6, 6, 6, 6]`（并集 = 0..49）。

| 会话 | worker / GPU | 端口 | 集号（每任务） | 集数 | 结果目录（`v1-store/evaluation/`） |
|---|---|---|---|---|---|
| `evoffctx-w0` | 0 | 8041 | 0,8,16,24,32,40,48 | 7×4=28 | `official-framesamp-context-w0/ckpt79999/seed42/` |
| `evoffctx-w1` | 1 | 8042 | 1,9,…,49 | 28 | `…-w1/…` |
| `evoffctx-w2` | 2 | 8043 | 2,10,…,42 | 24 | `…-w2/…` |
| `evoffctx-w3` | 3 | 8044 | 3,11,…,43 | 24 | `…-w3/…` |
| `evoffctx-w4` | 4 | 8045 | 4,12,…,44 | 24 | `…-w4/…` |
| `evoffctx-w5` | 5 | 8046 | 5,13,…,45 | 24 | `…-w5/…` |
| `evoffctx-w6` | 6 | 8047 | 6,14,…,46 | 24 | `…-w6/…` |
| `evoffctx-w7` | 7 | 8048 | 7,15,…,47 | 24 | `…-w7/…` |

每卡 2 个进程：policy server（uv venv，`CUDA_VISIBLE_DEVICES=<GPU>`，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` ≈ 33 GB）+ RoboMME 仿真（micromamba `robomme`，≈0.8 GB）。无 sidecar。
预期：官方模型无 sidecar，上一轮每窗 1.46 s 的 `add_buffer` 几乎全是 sidecar，去掉后每窗只剩帧路 SigLIP + `infer` 69 ms + 16 步仿真；每集预计 5–15 s，每 worker 4–8 min + 起服务约 4 min，总墙钟 8–12 min。

## 命令

```bash
# 预检（独立 RUN_NAME：eval.py 跑完集号区间即写 log.json，若用正式名会让正式 w0 空转退出）
MODE=stride ONLY=w0 EP_COUNT=9 RUN_NAME=official-framesamp-context-pre CKPT_ID=79999 \
  CKPT_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-context/79999 \
  SEED=42 LOG_PREFIX=evoffctxpre PORT_BASE=8051 bash scripts/training/prod/eval_all_shards.sh     # 每任务评 ep 0、8，共 8 集
# 正式 8 worker
MODE=stride RUN_NAME=official-framesamp-context CKPT_ID=79999 \
  CKPT_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-context/79999 \
  SEED=42 LOG_PREFIX=evoffctx PORT_BASE=8041 bash scripts/training/prod/eval_all_shards.sh
# 合并
UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/merge_eval_shards.py --layout stride --shards 8 \
  --run-name official-framesamp-context --ckpt-id 79999 --seed 42 --log-prefix evoffctx --tag EVAL_OFFICIAL_CTX \
  --out-dir docs/training-doc/eval-official-framesamp-context/records
```

盯盘项：驱动日志 `v1-store/logs/evoffctx-w<k>.log` 出现 `Error saving final results`（`eval.py` 收尾对含 `"error"` 值的 `sum()` 会 TypeError 并重跑整个 worker）即人工介入；`EXIT_CODE=0` 为完成。
**实测新增（见 result.md）**：单个 eval.py 进程连续建 27 个仿真环境后第 28 次 `make_env` 必抛 `vk::createInstanceUnique: ErrorIncompatibleDriver`，w0/w1（28 集）均中招；
处置是 `ONLY=w<k>` 重跑同 worker 续评。后续分片每进程 ≤27 集。

## tmux 会话清单（清理时的唯一依据）

本轮起过：`evoffctx-dl`（下载，已退出）、`evoffctxpre-w0`（预检，脚本结束自动退出）、`evoffctx-w0` … `evoffctx-w7`（正式 8 个，脚本结束自动退出）。
用户会话 `0`、`1`、`claude-private` 不动。清理只允许 `tmux kill-session -t <确切会话名>`。

## 输出

- 分片日志：`v1-store/logs/evoffctx-w<k>.log`（驱动 + eval 客户端，起跑行 `=== EVAL_SHARD shard=w<k> … tasks=… ep_start=<k> ep_stride=8 ep_count=0 … ===`，结束 `EXIT_CODE=`）、`…server.log`（policy server，含 TIMING 行）
- 分片结果：上表目录下 `progress.json`（4 任务各 6–7 集的 True/False）、`log.json`（该 worker 4 任务成功率）、`videos/`（每集 mp4，不进 git）
- 合并：`v1-store/evaluation/official-framesamp-context/ckpt79999/seed42/{progress.json,log.json,shards.json}`；`records/{summary.txt,per_episode.json,evoffctx-w*.txt,shard_timing_summary.txt}`

## 时间线

- 05:55:18 → 05:58:26 下载（tmux `evoffctx-dl`，`DOWNLOAD=PASS`）；05:59 解压 42 s；06:00 参数树核对 `PARAM_TREE_EXACT=PASS`。
- 06:03:02 → 06:08:36 预检 `evoffctxpre-w0`（8 集，`EXIT_CODE=0`，`records/preflight-w0.txt`）。
- 06:09:08 正式 8 worker 起跑，06:09:36 八个端口全部就绪。
- 06:14:08–06:15:44 w2/w4/w5/w6/w3/w7 依次 `EXIT_CODE=0`；06:15:23 / 06:15:55 w0 / w1 在第 28 集建环境时 Vulkan 崩溃（`EVAL_RC=1`，进度 27 集、0 error）。
- 06:16:10 / 06:16:22 `ONLY=w0` / `ONLY=w1` 续评重起；06:17:17 / 06:17:29 各补 1 集 `EXIT_CODE=0`。
- 06:18 合并 `EVAL_OFFICIAL_CTX=DONE … mean_rate=0.2400`。
