# mixed8-200 8×8 modulation 512+motion 80k 正式训练

本页是 commitV11.25Beta 起跑前档案（AGENTS 第 12 条两段式第一段）；跑完回写同号 commitV11.25 与 result.md。起跑 HEAD 即本 Beta 提交，由 preflight 的 `REPO_HEAD` 与 runner 打印的 `GIT_HEAD=` 固定。与 512 同批起跑（4+4 并行），512 档案见 [mixed8-200-m8x8-modul-b128-80k/launch.md](../mixed8-200-m8x8-modul-b128-80k/launch.md)（commitV11.24Beta `62a8ef15b4be267b56612d50562fc445160d041f`，美东 2026-10-17 01:00 起跑），版本、代码冻结、exclude、数据身份、监控与用户决定各节与之相同，本页只写差异。

## 与 512 的差异

- run 名 `mixed8-200-m8x8-modul-motion-b128-80k`；GPU 4–7（`CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8`），全局 batch 128、每卡 32。9 月 8 卡 motion run 每卡 16，本轮每卡样本翻倍、吞吐约减半。
- history yaml `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml`，sha256 `213bbe809baa286e951e873a49f6fc153a0517f939ac263cefcfc45c02380794`：复制自 `…-8x8-motion.yaml`，只改 `motion.store_path: v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion` 与 `motion.source_run: wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt#encoder`（budget 160、stride 16、window 33、demo 至少 17 帧 repeat_last）。
- motion 表：`v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion`，rows 128342（exec 81611 + demo 46731），补帧窗 1800；store_meta `c9f8275fe12d6395ea7cc29191e39a1fe5df7b569c579b5acdfea07a71bdde2e`，layout `motion-768-grid16-demopad17-v1`。encoder `checkpoint_epoch_72.pt`（EMA-only）sha256 `bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04`。窗口预算 `TRAIN_ES_BOUND=PASS max_windows=123 budget=160 headroom=37`。
- 对拍：`VERIFY_MOTION=PASS scanned=128342`、`WAN_BITEXACT=PASS compared=14480`、`ENCODER_BITEXACT=PASS compared=128342`、A6/A7/A9/A10/A11 全 PASS（[判定行汇总](../../dataset-build-doc/16task-mixed8-200-3200ep-e3c7564f/records/verdicts-b1-b4.md)）。
- 20 步短测：`MOTION_4GPU_SMOKE=PASS run=motion steps=20 exit=0 finite_losses=20/20 peak_mem_gib=38.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8`；稳态约 1.90 s/步，80k 预计约 42 h（含 4+4 互扰）。
- checkpoint 根 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-b128-80k`；日志 `v1-store/logs/mixed8-200-m8x8-modul-motion-b128-80k.log`；训练后上传 `HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1`（公开，不含 wandb/）。

```bash
bash v1-store/m8v-scripts/m8v-preflight-motion.sh      # 不传 --launch-mode；四个 --motion-* 一起给
tmux new-session -d -s m8v-train-motion "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/mixed8_200_runner/train-motion-80k.sh mixed8-200-m8x8-modul-motion-b128-80k"
```

## 两条训练之间工作区复查（口径 40 模板）

2026-10-17 01:01 美东，git status --porcelain：空（512 已于美东 01:00 起跑后复查）。exclude 仍只含口径 40 授权的 3 个路径与 1 行注释，未扩大。本 Beta 只新增本档案文件；起 motion 前另断言 `git diff --stat 62a8ef15b4be267b56612d50562fc445160d041f..HEAD -- src pyproject.toml uv.lock` 为空（红线 18）。
