# mixed8-200 8×8 modulation 512（无 motion）80k 正式训练

本页是 commitV11.24Beta 起跑前档案（AGENTS 第 12 条两段式第一段）；跑完后结果回写同号 commitV11.24 与 result.md。起跑 HEAD 即本 Beta 提交，由 preflight 的 `REPO_HEAD` 与 runner 打印的 `GIT_HEAD=` 固定，本页不预填自身 sha。

实施依据：MotionJEPA-backup `1016-mixed8-vla-plan.html` @ `45c3f9a`（进展登记至 `952b8db`）。前置证据：[建库档案](../../dataset-build-doc/16task-mixed8-200-3200ep-e3c7564f/launch.md) 与 [判定行汇总](../../dataset-build-doc/16task-mixed8-200-3200ep-e3c7564f/records/verdicts-b1-b4.md)。用户开工令：「/scratch/hongze/MotionJEPA-backup/1016-mixed8-vla-plan.html 开工 有关键问题问用户但尽可能不要阻塞」。

## 版本与代码状态

代码为 commitV11.23 三次合入（`e944b69` A1-merge、`44fb642` A1-train、`aa29828` A1b）之后的主检出，训练路径只新增 runner 与 motion-mixed8 yaml，`src/mme_vla_suite/training/config.py` 不改。uv.lock sha256 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`，`uv run --no-sync`，不同步依赖。配置还原：`git show <Beta>:src/mme_vla_suite/training/config.py` 中的 `mme_vla_suite_b128_80k` 条目，加下述 CLI 覆盖。

代码冻结（红线 18，用户口径 41）：从本 Beta 起到两条训练都出现 `EXIT_CODE=`，主检出不合入改动 `src`、`pyproject.toml`、`uv.lock` 的提交；第一条训练起跑后对被跟踪的 `src`/`scripts`/`packages` 文件上只读锁（原权限记入 `v1-store/m8v-scripts/m8v-chmod-before.txt`），全部结束后按原权限恢复。

## 配置与启动覆盖

- run 名 `mixed8-200-m8x8-modul-b128-80k`；config `mme_vla_suite_b128_80k`（80k 步、decay 80k、warmup 5000、peak_lr=decay_lr=5e-5、AdamW clip 1.0、EMA 0.999、每 5000 步保存、batch 128、seed 42、pi05_base 初始化）。512 由 60k 改 80k 为用户裁决：「两次训练本轮的512和512加motion全部改为80K的训练」。
- history yaml `perceptual-framesamp-modul-8frame-8x8.yaml`，sha256 `5b5ac2f85302d4d87cf102c71e02729d0caad74162df0afc9b2d4380e450caec`（budget 512 = 64 token × 8 帧，无 motion）。
- 4 卡：`CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8`，runner 透传 `--fsdp-devices 4 --num-workers 8`；全局 batch 128 不变（每卡 32）。用户原话：「2 应该可以4卡来跑 当时用8卡是为了加速，其他的总batch size应该都能保持一致 但是4卡跑」。4 卡与 8 卡不保证逐位一致。
- CLI 另覆盖 `--data.assets.assets-dir v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f --data.assets.asset-id robomme`，dataset-path `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/framesamp-8x8`，`MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 指向新库 source 与 `meta/episode_manifest.json`。

```bash
# preflight（不传 --launch-mode：prod 合约要求 fsdp8/w16 与 4 任务 sha，不适用于本轮，口径 36 P2）
bash v1-store/m8v-scripts/m8v-preflight-512.sh
# 起训
tmux new-session -d -s m8v-train-512 "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/mixed8_200_runner/train-512-80k.sh mixed8-200-m8x8-modul-b128-80k"
```

## 数据与身份

| 对象 | 值 |
|---|---|
| 库 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f`（16 任务 × 200 = 3200 集） |
| MANIFEST（源） | `e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11` |
| canonical manifest | `d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad` |
| framesamp-8x8 | steps 2,145,548 / exec samples 1,383,846；store_meta `ace85246419836d343b925ac45591e3fd75eb2f054c16eaafafb3b35d6ca9b79` |
| norm_stats | `36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394` |

`DATA_IDENTITY=PASS norm_sha=36ee9acd… steps=2145548 samples=1383846 manifest_sha=e3c7564f…`。checkpoint 根 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-b128-80k`，预期 5000…75000 每 5000 一份加最终 79999；日志 `v1-store/logs/mixed8-200-m8x8-modul-b128-80k.log`（runner 拒绝覆盖已有日志）；wandb 项目 robomme-framesamp。

## 主检出 exclude 记录（口径 40）

起训位置为 policy 主检出（worktree 的 `.venv` 软链与 editable 安装会让 preflight 的 `SYS_PREFIX`/`PKG_*` FAIL）。他人的 3 个未跟踪文件以精确路径追加进主检出 `.git/info/exclude`（另 1 行注释），追加前确认存在且未被跟踪，追加后 `git status --porcelain` 为空（`EXCLUDE_OK`，美东 2026-10-17 约 01:00）：

- `scripts/training/tests/compare_context_gpu_scaling.py`
- `scripts/training/tests/context_gpu_scaling.py`
- `scripts/training/tests/run_context_gpu_scaling.sh`

不删除、不移动、不暂存、不提交这 3 个文件，不改 preflight。用户原话：「a 同意 用 .git/info/exclude」；训练结束后保留这 3 条路径行（用户口径 41「4 保留」）。两条训练之间工作区复查：起 motion 前在 V11.25Beta 档案记录复查时刻与 `git status --porcelain` 结果；非空即按停点 8 停下交证据，不扩大 exclude。

## 起跑条件与监控

环境 B，8×A100-SXM4-80GB，`/dev/md0`。20 步短测（与 motion 同时 4+4）：`MOTION_4GPU_SMOKE=PASS run=512 steps=20 exit=0 finite_losses=20/20 peak_mem_gib=39.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8`；稳态约 1.84 s/步，80k 预计约 41 h（含 4+4 互扰）。起跑前：主检出 HEAD = 本 Beta、`git status --porcelain` 为空、GPU 0–3 无计算进程、preflight 判定行全 PASS。监控：主会话对日志挂 Monitor，过滤 `EXIT_CODE=|Traceback|nan|NaN|out of memory|RESOURCE_EXHAUSTED|Killed`，前 100 步再核 loss 有限、无报错（20 步覆盖不到 EMA 与周期保存）。

已知机器状况：本机 nvidia 中断计数冻结、JAX 阻塞同步固定约 100 ms（SigLIP 因此改 32 worker，计划进展登记 21）；本训练短测步时未见明显拖累，正式训练以实测步时为准。

## 用户决定与结论边界

运行期崩溃、NaN、OOM、掉卡一律停下问用户，不 resume、不自动重跑（口径 32「80k训练崩了停下来问我」）。全部 16 份 ckpt、config、日志训练结束后上传 `HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1`（公开，不含 wandb/，口径 29、30、44）。本轮不做 VLA 评测、不做纯 encoder 对照（口径 15）。
