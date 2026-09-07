# tic-l0-rhythm-40k —— 起跑记录（launch）

> 目的：第 0 关「配置与库同源」（`check_config_provenance.py`）、评估真节奏 CPU 复刻与上限边界（`eval_rhythm_gates.py`）、A19 期望按库重算后在 400 ep 库重跑 M1（`motion_gates_model.py --gate m1`）。
> 计划：`~/.claude-personal/plans/hashed-petting-garden.md`（训练/推理一致性验证，Codex 审计修订版）第二部分 2.2 / 2.3 / 2.1(A) 与 S3 组 A；正本文档 `docs/train-infer-consistency.md`（S4 补写）。
> 起跑 commit：`a8cfa17`（clean HEAD，含 commitV7.1 的对拍工具）。起跑日期 2026-09-07。

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

## 命令（tmux `tic-l0rhythm`，不占 GPU，`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`）

```bash
bash v1-store/reports/tic/launch_wave1.sh A    # 三条顺序执行，日志 v1-store/reports/tic/A/l0-rhythm-m1.log
# A1  uv run --no-sync python scripts/training/g0/check_config_provenance.py --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 --lib v1-store/datasets/4task-motion-400ep --train-config mme_vla_suite_b128 --out v1-store/reports/tic/A/l0.json
# A2  uv run --no-sync python scripts/training/tests/eval_rhythm_gates.py --lib v1-store/datasets/4task-motion-400ep --out v1-store/reports/tic/A/rhythm.json
# A3  MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-400ep/motion uv run --no-sync python scripts/training/tests/motion_gates_model.py --gate m1 --lib v1-store/datasets/4task-motion-400ep --dataset v1-store/datasets/4task-motion-400ep/framesamp --out v1-store/reports/tic/A/m1-400ep.json
```

## 判据（全部阻断，除 `CKPT_DTYPE_PROFILE` 观察）

`NORM_STATS_SAME`、`LIB_PROVENANCE_MATCH`、`MOTION_STORE_PATH`、`CKPT_PARAM_TREE`（叶数实测，预期 59）、`CKPT_DTYPE_PROFILE`（观察）→ `TIC_L0=PASS`；`RHYTHM_EQ`、`EVAL_TERMINATION`、`TAU_LONG`、`ES_BOUNDARY` → `TIC_RHYTHM=PASS`；`A19_VALID_DIST`（期望由 400 ep 清单重算、实测取 `motion_mask.sum()`）、`MOTION_DELIVERY` → 400 ep 库 M1 PASS。任一 FAIL 停在该行，原始输出交用户，不放宽。

## tmux 会话清单

- `tic-l0rhythm`（本 run 唯一会话；结束自行退出）。
