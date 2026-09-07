# tic-sidecar-40k —— 起跑记录（launch）

> 目的：第 1–5 关再跑一遍，motion 换真 sidecar（`MotionEncoderClient(online_gpu=1)`）现算，多比一条 `MOTION_S_VS_SIDECAR`（现算 vs 表逐位）；其余判据与 `tic-obs-model-40k` 同。
> 计划：`~/.claude-personal/plans/hashed-petting-garden.md`（训练/推理一致性验证，Codex 审计修订版）第二部分 2.4 与 S3 组 C；正本文档 `docs/train-infer-consistency.md`（S4 补写）。
> 起跑 commit：`fbdaf25`（clean HEAD，工具版本 commitV7.1 a8cfa17；组 B 起跑于 9b3b95f）。起跑日期 2026-09-07。

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

## 命令（tmux `tic-sidecar`，GPU 0 + sidecar GPU 1）

```bash
bash v1-store/reports/tic/launch_wave1.sh C    # 日志 v1-store/reports/tic/C/obsmodel-sidecar.log
# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-400ep/motion uv run --no-sync python scripts/training/g0/compare_train_infer_obs.py --lib v1-store/datasets/4task-motion-400ep --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 --train-config mme_vla_suite_b128 --motion sidecar --motion-gpu 1 --noise-seeds 0,1,2 --out v1-store/reports/tic/C/obsmodel-sidecar.json
```

sidecar 权重：`v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt`（sha256 `bae96037…c15a`），VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，与 run `motion_provenance.json` 的 `encoder` / `vae` 字段同源（`provenance_keys_equal` 在判定行内核对）。episode 与时刻清单同 `tic-obs-model-40k`（5 集 120 点 140 窗）。

## 判据

阻断新增 `MOTION_S_VS_SIDECAR=PASS windows=<m> mismatches=0 provenance_keys_equal=1`；其余 13 条阻断与观察行同 `tic-obs-model-40k/launch.md`（`VT_FULL_VS_CACHED` 预期同样超事先阈值，不放宽，交用户裁决；`ACT_CKPT_DTYPE` 仅 store 档有）。唯一成功行 `TIC_SIDECAR`。组 B 实测 15 min，本组预计 15–25 min。

## tmux 会话清单

- `tic-sidecar`。
