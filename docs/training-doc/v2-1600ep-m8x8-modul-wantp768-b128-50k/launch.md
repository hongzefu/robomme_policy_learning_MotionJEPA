# v2-1600ep-m8x8-modul-wantp768-b128-50k 起跑档案

本页在起跑那一刻写入（版本与启动配置），训练结束后另写 result.md。计划见仓库根目录 [1108-4task-modul-wantp768-50k-plan.html](../../../1108-4task-modul-wantp768-50k-plan.html)。

## ① 目的

在 4task 原训练库 `4task-v2-1600ep-604f16da`（BinFill、RouteStick、VideoRepick、VideoUnmaskSwap 各 400 局，共 1600 局）上，复刻上次 512+motion modulation run（`v2-1600ep-m8x8-modul-motion-b128-80k`），只把 motion 来源从学习型 encoder（`wan-full1600-filter2-b176x4-72ep-a`，1×768）换成同一批 stride 16 重叠窗的 Wan VAE raw latent 零参数池化：丢第 0 个 latent、后 8 个时间均值、空间自适应 6×8（高 32→6、宽 32→8）、C→H→W 展平 1×768。训到 50k，以 @50000 对比上次 @50000 的 54.62%（跨环境历史参考）。

## ② 版本与代码状态

- 分支 `v11-4task-modul-wantp768-50k`；起跑锚点为本档案所在的 Beta 提交（`commitV11.88Beta`），父提交 `a0bded4`（commitV11.87）。
- 代码来源：S1 `commitV11.84`（池化模块、导出、G3、motion store/训练闸/快照）、S2 `commitV11.85`（sidecar、客户端、policy_config、YAML、runner、G4 工具）、S3 `commitV11.86`（评估侧驱动、白名单、分片器、前缀失败率、ckpt 上传）、S2-fix1 `commitV11.87`（池化分支 provenance 比对排除 encoder 源码摘要）。
- 依赖：主 `.venv` 按本分支 `uv.lock` 同步（`uv sync --frozen`）；sidecar 用 `v1-store/venvs/wan`（torch 2.9.0+cu128、diffusers 0.39.0）。

## ③ 启动与配置还原

- TrainConfig：`git show <Beta>:src/mme_vla_suite/training/config.py` 中的 `mme_vla_suite_b128_80k`（batch 128、8 卡 FSDP8、workers 16、seed 42、EMA 0.999、AdamW 裁剪 1.0、warmup 5000、peak_lr = decay_lr = 5e-5、保存间隔 5000）。
- 启动脚本覆盖（`git show <Beta>:scripts/training/run_4task_wantp768_50k.sh`）：`--exp-name v2-1600ep-m8x8-modul-wantp768-b128-50k`、`--model.history-config perceptual-framesamp-modul-8frame-8x8-wantp768.yaml`、`--num_train_steps=50001`（最后一份 ckpt 为 50000，与上次评估 ckpt 同名），其余覆盖项与上次 `mv2-prod-runner.sh` 相同（主会话逐 flag 核对，仅少 preflight 的 4 个 `--motion-*`，改由 runner 内置 G0 环 4 同等检查）。
- history YAML 文件 sha256 `e049ed6243c7a226796a1e0f1ed346eff2cd586e3348c87e4a91b644ba2af5eb`，resolved_sha256 `4e70e93373d5b20061a4067c911021100375d2bc28cff276acaf7559b2441910`。
- 启动命令（tmux `w768-train`，主检出，HEAD 等于 Beta 且工作区 clean）：

```bash
tmux new-session -d -s w768-train "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/training/run_4task_wantp768_50k.sh <Beta 全 sha> 2>&1 | tee v1-store/logs/w768-train.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/w768-train.log"
```

- ckpt 上传长尾（tmux `w768-ckptup`，逐份等落盘即上传并本机读回）：`bash scripts/dataset/hf_export/run_wantp768_ckpt_export.sh`，bucket `HongzeFu/robomme-vla-4task-modul-wantp768-50k-v1`，完成行 `HF_VERIFY=PASS ckpts=10`。

## ④ 数据与划分

- VLA 库：`v1-store/datasets/4task-v2-1600ep-604f16da`，自 bucket `HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1` 恢复：`RESTORE_SHA=PASS objects=168 members=615212 runtime=PASS mismatches=0`（pkl 605611、latent 段 3200、framesamp-8x8 store_meta 与上次相同 `f7677e69…`）；norm_stats `856c75ea…`（与上次相同）。
- motion 表：`motion-wan-pool768`，`EXPORT_POOL768=PASS rows=71316 demo=35913 exec=35403 padded=1600 segments=3200 table_sha256=d039fb8d258b63d2cee4477e58d50e5d5bdc416d5d027477e3c8267afade75e8`。
- canonical manifest identity `4cd5a170…bb918`；input_manifest 文件 sha256 `655e64b4…`。

## ⑤ 起跑前闸门（全部 PASS）

| 闸门 | 判定行 |
|---|---|
| G1 | `RESTORE_SHA=PASS objects=168 members=615212 runtime=PASS mismatches=0` |
| G0 | `DATA_LINEAGE=PASS encoder=wanvae-9980d252+pool768 encoder_source_pin=41f4ca55… vla_input_pin=41f4ca55… files=4 mismatches=0 binding=input_manifest_sha256:655e64b4 enc_record=motion-wan-pool768.provenance+motion.provenance`（信任点：新表记录的清单摘要等于 VLA 库清单文件摘要；VLA 侧无逐文件 H5 sha，是现有数据的固有上限） |
| G2 | `POOL768_CONTRACT=PASS cases=24 max_abs=2.907e-07 bins=6x8` |
| G3 | `POOL768_TABLE=PASS rows=71316 demo=35913 exec=35403 padded=1600 index_mismatch=0 max_abs=9.437e-07` |
| G4 | `ONLINE_POOL768=PASS windows=64 padded=8 bitexact=64`（首跑因 encoder 源码摘要比对失败，修复后重跑） |
| G5 | `VLA_PARAM_EQ=PASS params=3443670288 params_old=3443670288 paths=65 only_old=0 only_new=0 shape_mismatch=0`；`TRAIN_CONFIG_EQ=PASS diffs=3` |
| G6 | `SMOKE20=PASS steps=20 save_load=PASS`（Step 0 loss 0.0887） |
| D1.5 | 冒烟快照 resolved_sha256 = 复算值 = 白名单 `WANTP768_RESOLVED_SHA256` = `4e70e933…1910` |
| G6b | `EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=78.3`（单片 497 s，同卡 2 片墙钟 731 s；首跑因同一 provenance 原因失败） |
| E0 | `DEMO_PREFIX_SRC=PASS code_same=1 benchmark=b4e97f2 shards_ok=16/16 ok=142 failed=8 diff_vs_old=40`（成功集合与旧清单相同，40 条 D 差 1–9 步，按用户裁决只记录） |

## ⑥ 硬件与资源

AWS 8×A100-SXM4-80GB，本地 NVMe RAID `/dev/md0`；起跑前 `/scratch` 剩 606 GB。上次 80k 稳态约 0.99 s/步，预计 50001 步约 14 h。

## 本轮 tmux 会话清单（截至起跑）

已结束：`w768-restore`、`w768-evalenv`、`w768-demoprobe`、`w768-export`、`w768-g3`、`w768-g4prep`、`w768-demo`、`w768-smoke`、`w768-g6b`、`w768-g4`、`w768-g4-r2`、`w768-g6b-r2`。起跑：`w768-train`、`w768-ckptup`。

## 盯盘项

Monitor 过滤 `Step [0-9]+:|nan|NaN|CUDA out of memory|Killed|Traceback|TRAIN_FINAL_RECORD|^EXIT_CODE=`；训练中途报错退出、OOM、被杀、NaN 一律停下等用户（用户裁决「停下等你」）。
