# v2-1600ep-m8x8-ctx-motion-b128-50k 起跑档案

本页是 1115 计划（根目录 `1115-4task-ctx-motion160-v7-50k-plan.html`）的正式训练档案。按 AGENTS.md 第 12 条两段式：本次起跑前写「版本与代码状态」「启动与配置还原」两节，训练与评估结束后在 `result.md` 补其余各节。

## 一句话

把 54.62% 那条 run `v2-1600ep-m8x8-modul-motion-b128-80k`（TRAIN_HEAD `2f10473`，GL A40 692 局 @50000 54.62%）的 history YAML 只改 `integration_type: modulation → context`、`memory_token_dim: 1024 → 2048`，库、motion 表、encoder（`wan-full1600-filter2-b176x4-72ep-a` ep72）、norm、TrainConfig、8 卡 b128 全部照旧，训 50001 步（最后一份 ckpt 为 step 50000），再在本机评估 692 局（seed 7）。

用户原话（2026-11-15 美东）：「给出方案 54.62% 那条 学习型 encoder wan-full1600-filter2-b176x4-72ep-a（旧 4task v7 encoder，ep72），每窗输出 1×768 训练配置 512 帧记忆 + motion，modulation，50k 给出方案再训一个 但是用context模式其他保持一致」；开工令：「同意开工」。其余裁决见计划 §1、§8。

## ① 版本与代码状态

- 分支 `v17-4task-ctx-motion160-v7-50k`，从 `v11-4task-modul-wantp768-50k@c55a6bc` 切出（与原 TRAIN_HEAD `2f10473` 相比 `src/mme_vla_suite/models` 零改动、`mme_vla_suite_b128_80k` 零差异）。
- 计划：第一版 `5f55b5e`，第二版 `fe646be`，开工前自洽审查补丁 `79edd32`（`PLAN_CONSISTENCY_REVIEW=FAIL base=fe646be findings=4 blocking=1`，四条均已补入）。
- 代码合入：
  - `55ddfac` commitV17.1：S2 评估侧（v15 `c0826be` 每卡 4 份评估栈、`check_checkpoint.py` 按 run 登记 integration、`--g6b-capacity` 压测）。`PRE_MERGE_REVIEW=PASS base=79edd32 tip=ddf4c46 files=5 commits=2 findings=2`；`POST_MERGE_REVIEW=PASS merge=55ddfac tests=8passed files=5`。
  - `a085c88` commitV17.2：S1 训练侧（B1 YAML、四环血缘脚本、runner、上传脚本）。`PRE_MERGE_REVIEW=PASS base=79edd32 tip=283ec58 files=5 commits=2 findings=1`；`POST_MERGE_REVIEW=PASS merge=a085c88 tests=17passed files=5`。
  - `2527364` commitV17.3：回填 `CTXMOTION160_RESOLVED_SHA256=21791b2964c868119784d086f5ce2334405596d8fa20586e217c4703f018d0e3`。
  - `45180ac`：规则子模块同步到正本 `b91c20d`（`repo_guard.py fix`，切分支后规则文件回到 v11 时代旧版，故同步）。
- TRAIN_HEAD：本页所在的 `commitV17.4Beta`（启动命令的完整 sha 参数、`--check-only` 的 HEAD 检查与 runner 记录共同固定；本页不预填自身 sha）。
- 评估编码环境：`v1-store/venvs/wan` 于 11-15 美东约 21:50 按 `c55a6bc:scripts/dataset/wan/uv.lock` 原地修复，motion-jepa 由 `ee719fb` 换回 `2a484ad`（tmux `v17-wanenv`，`EXIT_CODE=0`）；修复后 `wan_latent_encoder.py` sha `d00267e1…` 等于离线表 `provenance.vae.encoder_src_sha256`。修复前清单 `v1-store/logs/v17-wanenv-before.txt`、`v17-wanenv-before.direct_url.json`。评估后按 `b5bdccf` 锁文件还原（计划第 13 道）。

## ② 启动与配置还原

- TrainConfig：`git show <TRAIN_HEAD>:src/mme_vla_suite/training/config.py` 的 `mme_vla_suite_b128_80k`（batch 128、warmup 5000、lr 恒定 5e-5、EMA 0.999、clip 1.0、FSDP8、workers 16、save/keep 5000、pi05_base）。唯一启动覆盖 `--num_train_steps=50001`。
- History YAML：`git show <TRAIN_HEAD>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-8frame-8x8-motion-v7ep72.yaml`，文件 sha256 `2e98e191db566c2a0f5c107e28a871493f2207cd65d00600b88ae5eab7b6d2d3`；相对 `2f10473` 原 YAML 键差集恰为 `integration_type`、`memory_token_dim`。
- 入口：`bash scripts/training/run_4task_ctxmotion160_50k.sh <TRAIN_HEAD>`；完整 TRAIN_ARGS 以 `--dry-run` 打印为准（`--exp-name v2-1600ep-m8x8-ctx-motion-b128-50k … --dataset-path v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8 --model.history-config perceptual-framesamp-ctx-8frame-8x8-motion-v7ep72.yaml --num_train_steps=50001`）。
- 输入身份（runner 第 1–5 道起跑前检查逐项核对）：motion 表 `03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5`、`store_meta.json` `d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268`、norm `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`、encoder ckpt `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca`、canonical manifest `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`。
- 数据血缘：合入前主会话实测 `DATA_LINEAGE=PASS mode=v7-4ring encoder_ckpt=0c198629 encoder_source_pin=df992cdf vla_input_pin=655e64b4 files=4 members=1600 mismatches=0`。
- 产物：ckpt 根 `v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-ctx-motion-b128-50k`（预期 5000…50000 共 10 份）；日志 `v1-store/logs/v17-train.log`；公开 bucket `HongzeFu/robomme-vla-4task-m8x8-ctx-motion160-v7-50k-v1`；评估目录 `v1-store/evaluation/local692-ctx-motion160-50k`。
- 本轮 tmux 会话清单（只按此清单清理）：`v17-wanenv`（已结束）、`v17-smoke`、`v17-g6b`、`v17-cap4`、`v17-cap3`（仅 4 份压测不过时）、`v17-train`、`v17-ckptup`、`v17-eval`、`v17-wanrestore`。
- 硬件：AWS p4de，8×A100-SXM4-80GB，`/dev/md0` NVMe RAID。0928 同组合测速 1.979 s/步、每卡峰值约 41.8 GB，预计 50k 约 28–29 h。
