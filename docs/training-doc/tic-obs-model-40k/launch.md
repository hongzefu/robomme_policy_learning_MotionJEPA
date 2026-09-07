# tic-obs-model-40k —— 起跑记录（launch）

> 目的：第 1–5 关：输入键 → 预处理后 → 模型内部（前缀 / mask / 位置 / KV）→ 整段前向 vs 缓存分步 → 最终动作；5 条训练集 episode 按评估节奏重放，motion 从库查表（`--motion store`），S 臂灌训练表帧特征真值，B/A 臂只出观察行。
> 计划：`~/.claude-personal/plans/hashed-petting-garden.md`（训练/推理一致性验证，Codex 审计修订版）第二部分 2.4 与 S3 组 B；正本文档 `docs/train-infer-consistency.md`（S4 补写）。
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

## 命令（tmux `tic-obsmodel`，GPU 0）

```bash
bash v1-store/reports/tic/launch_wave1.sh B    # 日志 v1-store/reports/tic/B/obsmodel-store.log
# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-400ep/motion uv run --no-sync python scripts/training/g0/compare_train_infer_obs.py --lib v1-store/datasets/4task-motion-400ep --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 --train-config mme_vla_suite_b128 --motion store --noise-seeds 0,1,2 --out v1-store/reports/tic/B/obsmodel-store.json
```

episode（默认 5 项，覆盖 es ∈ {0,66,114,168,216} 各取最长一集）：`ButtonUnmaskSwap:3`(g103, es0, T555)、`VideoUnmask:3`(g203, 66, 399)、`VideoUnmaskSwap:5`(g305, 114, 446)、`VideoUnmaskSwap:31`(g331, 168, 461)、`VideoUnmaskSwap:3`(g303, 216, 586)；决策时刻清单由脚本按 `eval.py` 控制流独立生成（预期 120 点 / 140 窗，`c_mod16` 逐集打印）。

## 判据

阻断：`RAW_OBS_VS_PKL`（图像/状态；prompt 子键只记录）、`MEM_S_VS_TRAINSET`、`OBS_T_VS_I`、`OBS_PROMPT`（token 不等即 FAIL 并停 L3+）、`TRANSFORMS_CHAIN`、`PREFIX_SHAPE`、`PREFIX_T_VS_I`、`PREFIX_POSITIONS`、`MEM_INVPERM`、`KV_T_VS_I`、`FULL_VS_CACHED_STRUCT`、`VT_FULL_VS_CACHED`（阈值事先定死 rel_fro ≤ 1e-3 且 bf16 ulp_p99 ≤ 4，**开发跑已实测超阈值（rel_fro ≈ 3e-3、ulp_p99 22–31），正式跑预期同样 FAIL，不现场放宽，交用户裁决**）、`ACT_T_VS_I`。观察：`MEM_S_VS_B`/`MEM_S_VS_A`/`MEM_A_VS_B`、`MEM_SCALE_OBS`、`VT_FULL_VS_CACHED_F32`（f32 权重下同口径诊断）、`ACT_S_VS_B`/`NOISE_S`、`AUG_EFFECT_OBS`、`ACT_CKPT_DTYPE`。唯一成功行 `TIC_OBS_MODEL`。预计 85–100 min。

## tmux 会话清单

- `tic-obsmodel`。
