# tic-vulkan-makeenv —— 起跑记录（launch）

> 目的：「单进程第 28 次 `make_env` 必崩（Vulkan `ErrorIncompatibleDriver`）」根因排查的正式复现与修法验证：基线循环 35 轮（预期第 28 轮崩）、修法 1 钉住 RenderSystem、修法 2 `GLIBC_TUNABLES` 抬静态 TLS 余量，各 35 轮不崩即通过。
> 计划：`~/.claude-personal/plans/hashed-petting-garden.md`（训练/推理一致性验证，Codex 审计修订版）S3 组 F；正本文档 `docs/train-infer-consistency.md`（S4 补写）。
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

## 命令（tmux `tic-vulkan`，GPU 7，micromamba `robomme` 环境，不起 policy）

```bash
bash v1-store/reports/tic/launch_wave1.sh F    # 日志 v1-store/reports/tic/F/vulkan.log；逐轮 json 落 v1-store/reports/tic-vulkan/probe-tic-*.json
# F1 CUDA_VISIBLE_DEVICES=7 /scratch/hongze/micromamba/envs/robomme/bin/python scripts/training/legacy-eval/probe_vulkan_makeenv.py --tag tic-baseline --rounds 35 --no-reset
# F2 … --tag tic-pin --rounds 35 --no-reset --pin-renderer
# F3 GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192 … --tag tic-tls8192 --rounds 35 --no-reset
```

排查阶段（同脚本、未提交版本，2026-09-07 GPU 7）的全部证据（基线 / `--gc --renderer-release` / pin / tls8192 / `VK_LOADER_DEBUG` 实例计数 / C 层最小复现 `vk_instance_limit.c`、`vk_device_limit.c`、`dlopen_icd_limit.c` / 六档 TLS 扫描）在 `v1-store/reports/tic-vulkan/`，结论与逐轮数据表转录进 result.md。

## 判据

`CRASH at round=28`（F1，复现）；F2、F3 各 `rounds_done=35 crash_round=-1`；`VULKAN_ROOTCAUSE=… repro_crash_at=28 fix=… fixed_runs=35`。主线 `examples/robomme/` 与 `src/` 零改动，两份 patch（`fix-a-glibc-tunables.patch`、`fix-b-pin-renderer.patch`）只作文本产出、不应用，落点建议 legacy-eval。

## tmux 会话清单

- `tic-vulkan`。
