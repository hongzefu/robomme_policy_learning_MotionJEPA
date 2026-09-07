# tic-eval-probe-40k —— 起跑记录（launch）

> 目的：第 6 关闭环：探针版 policy server（`serve_policy_probe.py`，实例属性遮蔽 reset/_prepare_history/_input_transform，逐次 infer 记完整数组）+ 主线 `examples/robomme/eval.py` 跑 4 任务 × 6 集 = 24 集 test split 仿真（单次运行、全新 save_dir、不续评），`summarize_eval_probe.py` 核在线不变量。
> 计划：`~/.claude-personal/plans/hashed-petting-garden.md`（训练/推理一致性验证，Codex 审计修订版）第二部分 2.5 与 S3 组 E；正本文档 `docs/train-infer-consistency.md`（S4 补写）。
> 起跑 commit：`6f029ec` 之后的 launch 预提交 HEAD（clean HEAD，工具版本 commitV7.1 a8cfa17）。起跑日期 2026-09-07。

## 环境与介质

- **环境 B**（AGENTS「运行环境判定」）：仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`；`/nfs/turbo/coe-chaijy-unreplicated/hongzefu`、`/data/hongzefu`、`~/.ssh/config` 均不存在；`nvidia-smi` 8 × `NVIDIA A100-SXM4-80GB`。
- 存储介质：AWS 本地 NVMe RAID（`/dev/md0`，raid0 × 8 NVMe，6.9 T）。
- Python：`UV_LINK_MODE=copy uv run --no-sync python …`；tmux detached + `PYTHONUNBUFFERED=1` + `set -o pipefail` + `tee`，结尾 `EXIT_CODE=`。

## 输入指纹（生产口径）

| 项 | 值 |
|---|---|
| 训练配置 | `mme_vla_suite_b128`（`data.assets` → `v1-store/train-assets/mme_vla_suite/robomme-400ep`） |
| checkpoint | `v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999`（`params/` + `assets/robomme/norm_stats.json`；run 快照 `history_config.resolved.sha256` = `94d86603…`） |
| 库 | `v1-store/datasets/4task-motion-400ep`：`framesamp_manifest_sha256=92fa17e9…`、`framesamp_store_meta_sha256=dffdd47b…`、`motion_index_sha256=74185921…`、`motion_store_meta_sha256=e19fb5fd…`、`motion_table_sha256=6e70604d…`（出自 run 的 `motion_provenance.json`，由 `LIB_PROVENANCE_MATCH` 现算核对） |
| motion 表覆盖 | `MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-400ep/motion`（快照 `store_path` 记的是 40ep 路径，不覆盖时 `check_same_source` 直接 raise） |
| sidecar 权重 | `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt`（sha256 `bae96037…c15a`，`ASSETS_LOCK.json`）；VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| 原始 h5 | `/scratch/hongze/robomme_data_h5/`（`docs/dataset-build-doc/16task-h5-scan/`） |

## 命令（tmux `tic-evalprobe`，policy + 仿真 GPU 0，sidecar 按快照 `motion.online_gpu: 1` 落 GPU 1，端口 8123）

```bash
bash v1-store/reports/tic/launch_wave1.sh E    # 日志 v1-store/reports/tic/E/eval-probe.log（驱动）、server.log（探针 server + sidecar）、eval.log（eval.py 全量 stdout/stderr）、probe.jsonl（逐次 infer 记录）
# E1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run --no-sync python scripts/training/g0/serve_policy_probe.py --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 --lib v1-store/datasets/4task-motion-400ep --port 8123 --seed 42 --probe-out v1-store/reports/tic/E/probe.jsonl
# E2 cd examples/robomme && CUDA_VISIBLE_DEVICES=0 /scratch/hongze/micromamba/envs/robomme/bin/python eval.py --args.port=8123 --args.model_seed=42 --args.policy_name=tic-eval-probe --args.model_ckpt_id=39999 --args.only_tasks=ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap --args.max_episodes=6 --args.save_dir=v1-store/evaluation/tic-eval-probe-40k
# E4 uv run --no-sync python scripts/training/g0/summarize_eval_probe.py --probe … --progress v1-store/evaluation/tic-eval-probe-40k/tic-eval-probe/ckpt39999/seed42/progress.json --eval-log … --server-log … --lib v1-store/datasets/4task-motion-400ep --expect-episodes 24 --expect-tasks 4
```

- 主线 `eval.py` 为 4b7a710 版（`max_steps=1300`、`EnvRunner` 写死 `dataset="test"`）；24 < 27，不触发单进程第 28 次 `make_env` 崩溃（根因与修法见 `tic-vulkan-makeenv/`，本组不依赖修法）。
- **不续评**：`save_dir` 起跑前必须不存在（起跑器检查），续评会把 `"error"` 条目删掉重评而掩盖问题。
- 探针只遮蔽实例属性，`src/` 与 `examples/` 零改动；`--motion-gpu` 不传即生产路径（`create_trained_policy` 自动拉起 sidecar）。

## 判据（七阻断 + 一观察，唯一成功行 `TIC_EVAL=PASS episodes=24 tasks=4 …`）

`EVAL_EPISODE_MAP`（探针 episode_seq 与 eval 日志「顺序 + exec_start_idx + task_goal」三重一一对应、progress.json 24 条、无 `already evaluated, skipping`）、`EVAL_TAU_K`、`EVAL_K_FORMULA`（汇总器独立重写公式）、`EVAL_ORDER_LEGAL`（汇总器独立重排序 608 位）、`EVAL_BACKEND`（在线 pos 表 sha == 库 pos 表前 586 行 sha）、`EVAL_PROMPT`（test split 的 prompt token 必须在训练集 4 任务 token 集合内——**按实测判，可能 FAIL**）、`EVAL_NO_RAISE`（按块判 Traceback，排除端口探测握手噪声；timeout 由「82 次 infer」判并与视频文件名交叉核）；观察 `EVAL_DIST_OBS`。任一 FAIL 停在该行交用户，不放宽。预计 24 集 ≤ 50 min。

## tmux 会话清单

- `tic-evalprobe`（内含探针 server 后台进程，由驱动脚本自身 kill 其 PID，不涉及任何 tmux 全局操作）。
