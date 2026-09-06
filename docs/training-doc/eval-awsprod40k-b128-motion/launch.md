# eval-awsprod40k-b128-motion — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。正式评估，按 AGENTS 12 留档。

起跑时间：2026-09-06 01:43（训练于 00:46 落 `39999`、`EXIT_CODE=0`）
起跑 commit：代码 `7867dcd`（commitV5.3）；本留档在起跑前提交，工作区 clean（`eval_shard.sh` 断言 `git status --porcelain` 为空）
被评 checkpoint：`v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`（EMA 树，bf16 restore；训练留档 `docs/training-doc/awsprod40k-b128-motion/`，
run 快照 `history_config.resolved.sha256 = 94d8660…927fb`，`create_trained_policy` 起跑时核 sha 与 `motion_provenance.json`）

## 口径

- **benchmark 上游原样**：`examples/robomme/eval.py` 默认 `max_steps=1300`、`obs_horizon=16`；`env_runner.py` 硬编码 `dataset="test"`，每任务 50 集。
  只加了 `episode_start`（分片用，默认 0 不改行为）。
- **policy 采样 seed 42**（`serve_policy.py --seed=42`，此前拍板）；`eval.py --args.model_seed=42` 只决定结果路径。单 seed、单 checkpoint。
- **test seed 核对**（`scripts/training/prod/check_test_seeds.py`，`records/test_seeds.json`）：
  `TEST_SEED_DISJOINT=PASS test=50x4 train_h5=100x4 overlap=0 test_episodes_contiguous=True`、`TRAIN_H5_IN_TRAIN_SPLIT=PASS missing=0`——
  test seed 55xxxx–58xxxx，训练 H5 seed 5000–17900 且全部属 `env_metadata/train`；每任务难度 easy 26 / medium 12 / hard 12。
- **审计闭合项**（`v1-store/reports/audit/train-infer-consistency-b12f1a0.md`）：D-A1 记忆帧 SigLIP f32/bf16 → 实测记忆 token 差 0.34%、动作差为采样噪声的 2.3%，
  **不改在线侧、不重建库**（`docs/training-doc/siglip-ab-replay-40k/result.md`）；B-A2 在线 τ 可达 1296 > 训练 ≤585 → 是 `max_steps=1300` 的固有属性，**保留 1300**，结果里 timeout 集与之共现。
- 已知污点：test ep 0–9 在 MotionJEPA encoder 训练集内（holdout 90–99）；训练 50.6 epoch 无验证集；无 nomotion 对照。

## 分片与资源：8 个 worker，每个独占一张卡，policy + sidecar + 仿真同卡

| 会话 | 任务 | 集 | GPU | 端口 | 结果目录（`v1-store/evaluation/`） |
|---|---|---|---|---|---|
| `ev40k-ButtonUnmask-0` | ButtonUnmask | 0–24 | 0 | 8031 | `awsprod40k-b128-motion-ButtonUnmask-0/ckpt39999/seed42/` |
| `ev40k-ButtonUnmask-1` | ButtonUnmask | 25–49 | 1 | 8032 | `…-ButtonUnmask-1/…` |
| `ev40k-VideoUnmask-0` | VideoUnmask | 0–24 | 2 | 8033 | `…-VideoUnmask-0/…` |
| `ev40k-VideoUnmask-1` | VideoUnmask | 25–49 | 3 | 8034 | `…-VideoUnmask-1/…` |
| `ev40k-ButtonUnmaskSwap-0` | ButtonUnmaskSwap | 0–24 | 4 | 8035 | `…-ButtonUnmaskSwap-0/…` |
| `ev40k-ButtonUnmaskSwap-1` | ButtonUnmaskSwap | 25–49 | 5 | 8036 | `…-ButtonUnmaskSwap-1/…` |
| `ev40k-VideoUnmaskSwap-0` | VideoUnmaskSwap | 0–24 | 6 | 8037 | `…-VideoUnmaskSwap-0/…` |
| `ev40k-VideoUnmaskSwap-1` | VideoUnmaskSwap | 25–49 | 7 | 8038 | `…-VideoUnmaskSwap-1/…` |

每个 worker 三个进程同一张卡：policy server（uv venv，`CUDA_VISIBLE_DEVICES=<GPU>`，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`）、motion sidecar
（`v1-store/venvs/wan`，卡号由 `MMEVLA_MOTION_ONLINE_GPU=<GPU>` 覆盖快照的 `motion.online_gpu: 1`）、RoboMME 仿真（micromamba `robomme`，
ManiSkill 渲染 `sapien.Device("cuda")` = 可见集第 0 张，PhysX 走 CPU）。理由：一集是「仿真 16 步 → add_buffer（帧路 SigLIP + sidecar 一窗）→ infer」的串行链，
三进程互相等待、不并发，同卡不损吞吐；历史实测 sidecar 独占卡 1.42 s/窗（P5），两 sidecar + 两仿真挤一张卡 3.5 s/窗（`aws-t3-open-s100/result.md` 四节）。
不做 16 worker：sidecar 是 GPU 算力瓶颈，两个同卡各慢一倍，墙钟不降。

预期：每片 25 集 × ≤2.7 min（1300 步全超时的最坏情况：81 个 chunk × (1.42 + 0.2 + 仿真)）+ 起服务约 4 min ≈ 70 min；成功集提前终止只会更短。

## 启动命令

```bash
ONLY=ButtonUnmask-0 bash scripts/training/prod/eval_all_shards.sh   # 预检片：等端口就绪 + 第 0 集起跑，nvidia-smi 核对 GPU 0 上恰 3 个进程
bash scripts/training/prod/eval_all_shards.sh                        # 其余 7 片（已存在的会话跳过）
UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/merge_eval_shards.py --out-dir docs/training-doc/eval-awsprod40k-b128-motion/records   # 全部 EXIT_CODE=0 后合并
```

## tmux 会话清单（清理时的唯一依据）

本轮起过的会话仅上表 8 个（前缀 `ev40k-`），脚本结束自动退出；用户会话 `0`、`1`、`claude-private` 不动。清理只允许 `tmux kill-session -t <确切会话名>`。

## 输出

- 分片日志：`v1-store/logs/ev40k-<Task>-<k>.log`（驱动 + eval 客户端，结束 `EXIT_CODE=`）、`…server.log`（policy server + sidecar，含 TIMING 行）
- 分片结果：上表目录下 `progress.json`（逐集 True/False）、`log.json`（该片成功率）、`videos/`（每集 mp4，约 8 MB/集，不进 git）
- 合并：`v1-store/evaluation/awsprod40k-b128-motion/ckpt39999/seed42/{progress.json,log.json,shards.json}`；`records/{summary.txt,per_episode.json,test_seeds.json,ev40k-*.txt}`
