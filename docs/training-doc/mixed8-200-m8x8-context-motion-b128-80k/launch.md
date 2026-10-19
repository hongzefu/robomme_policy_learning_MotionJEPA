# mixed8-200 8×8 context 512+motion160（全部走 context）80k 正式训练

本页是 commitV11.30Beta 起跑前档案（AGENTS 第 12 条两段式第一段）；跑完后结果回写同号 commitV11.30 与 result.md。起跑 HEAD 即本 Beta 提交，由 preflight 的 `REPO_HEAD` 与 runner 打印的 `GIT_HEAD=` 固定，本页不预填自身 sha。

实施依据：根目录 [`1019-mixed8-ctx512-motion-plan.md`](../../../1019-mixed8-ctx512-motion-plan.md)（开工审查 `PLAN_CONSISTENCY_REVIEW=PASS base=222e08e findings=12 blocking=0`，修订 `92437d0`）。用户原话（美东 2026-10-19）：「gpu0-3跑context 512+motion 写计划 直接用plan模式的md 不用再写html 但是要留在根目录」「不是！」「我的意思是全部走conetxxt」「现在已经实现过了」「1234都按照推荐的」「再审查一遍 没问题就开始」。

## 版本与代码状态

代码为 commitV11.29（合并 `34a7bad`，S1 `99bd38f`）之后的主检出：只新增 5 个文件（新 yaml、runner `train-ctx-motion-80k.sh`、上传驱动、allowlist、bucket README），不改任何已跟踪文件；模型代码自 0928 原入口快照 `6db0e0a` 起未改。`uv.lock` sha256 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`，`uv run --no-sync`。配置还原：`git show <Beta>:src/mme_vla_suite/training/config.py` 中 `mme_vla_suite_b128_80k` 条目，加下述 CLI 覆盖。

代码冻结：沿用 `mixed8-2048-train` 的只读锁（被跟踪 `src`/`scripts`/`packages` 已 `chmod a-w`，原权限 `v1-store/m8v-scripts/m2k-chmod-before.txt`）；本轮新增 5 文件合并后同样 `a-w`，原权限 `v1-store/m8v-scripts/cmo-chmod-added.txt`（`LOCK_ADDED=OK files=5`）。按用户 Q3 裁决，2048 收尾不恢复权限（其 owner 已在 handoff `b573f7e` 移交），由本 run 结束后按两份清单一次恢复。

## 配置与启动覆盖

- run 名 `mixed8-200-m8x8-context-motion-b128-80k`；config `mme_vla_suite_b128_80k`（80k 步、warmup 5000、lr 5e-5、AdamW clip 1.0、EMA 0.999、每 5000 步保存、batch 128、seed 42、`pi05_base` 初始化）。
- history yaml `perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml`，sha256 `48c04077cd1dc00040cf183653ea62367b75c2788961b6a998e9c9dc80ff6cfe`：复制 V11.25 的 `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml`（`213bbe80…`），只改 `integration_type: context`、`memory_token_dim: 2048`（`YAML_DIFF=PASS lines=2`）。帧 512 + motion 160 concat 成 672 位、按 `mem_order` 交错后拼进 PaliGemma 主干前缀；59 参数叶，不能热启动。
- 4 卡：`CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8`，全局 batch 128（每卡 32）；GPU 4–7 为在跑的 `m2k-train-modul`，不占用。
- CLI 覆盖（runner `_train_common.sh`）：`--assets-base-dir v1-store/train-assets --data.assets.assets-dir v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f --data.assets.asset-id robomme --checkpoint-base-dir v1-store/train-runs --dataset-path v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/framesamp-8x8`；`MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 指向新库；wandb 由 `v1-store/secrets/wandb.env` 开启。

```bash
# 起跑前自检（不传 --launch-mode；--train-head 为本 Beta 全 sha 字面量）
bash v1-store/m8v-scripts/cmo-preflight.sh 2>&1 | tee v1-store/logs/cmo-preflight.log
# 起训
tmux new-session -d -s cmo-train "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/mixed8_200_runner/train-ctx-motion-80k.sh mixed8-200-m8x8-context-motion-b128-80k"
# G6 之后、run 目录出现后：增量上传
tmux new-session -d -s cmo-inc "cd /scratch/hongze/robomme_policy_learning_MotionJEPA; set -o pipefail; bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m8x8-context-motion-b128-80k HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1 hf-ckpt-mixed8-200-vla-ctx-motion-80k scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_ctx_motion_80k.json cmo-train 2>&1 | tee v1-store/logs/cmo-inc.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/cmo-inc.log"
```

## 数据与身份

| 对象 | 值 |
|---|---|
| 库 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f`（16 任务 × 200 = 3200 集），与 V11.24/V11.25 同一份 |
| framesamp-8x8 | steps 2,145,548 / exec samples 1,383,846；store_meta `ace85246419836d343b925ac45591e3fd75eb2f054c16eaafafb3b35d6ca9b79` |
| motion | 128342 行（exec 81611 + demo 46731），store_meta `c9f8275fe12d6395ea7cc29191e39a1fe5df7b569c579b5acdfea07a71bdde2e`，layout `motion-768-grid16-demopad17-v1`；encoder `wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt#encoder` |
| norm_stats | `36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394` |

checkpoint 根 `v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-context-motion-b128-80k`，预期 5000…75000 每 5000 一份加 79999 共 16 份；日志 `v1-store/logs/mixed8-200-m8x8-context-motion-b128-80k.log`（runner 拒绝覆盖已有日志）；训练中增量上传到公开 bucket `HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1`，结束后正式驱动 `run_mixed8_200_vla_ctx_motion_80k_ckpt_export.sh`（先 `SMOKE=1` 再全量）。

## 起跑前闸门（主检出实测，美东 2026-10-19 00:40–00:49）

| 闸门 | 判定行 |
|---|---|
| S1 静态 | `S1_STATIC=PASS yaml_diff=2 bash_n=0 json=0 stale=0` |
| 合并审查 | `PRE_MERGE_REVIEW=PASS base=d943547 tip=99bd38f files=5 commits=1 findings=1 blocking=0`；`POST_MERGE_REVIEW=PASS merge=34a7bad tests=pytest25passed,G1,G2 files=5 blocking=0` |
| G1 | `DRY_RUN_CMO=PASS`（`HC=perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml`、`GIT_HEAD=34a7bad`、`--fsdp-devices 4 --num-workers 8`） |
| G2 | `YAML_DIFF=PASS lines=2` |
| G3 | `DATASET_BUILD=PASS samples=1383846 motion=1`（经训练同一工厂 `dataloader.py::_create_framesamp_dataset` 构造；首尾样本八键形状对，末样本 motion 有效 53、帧有效 512） |
| G4 | `SMOKE20=PASS oom=0 loss_finite=1 peak_mib=71983 steady_s_per_step=7.500`；run `cmo-smoke-20261019T044242Z`，美东 00:42:42–00:48:46，`EXIT_CODE=0`，Step 0 `loss=0.5753 grad_norm=22.2819`；NVML 峰值 GPU0 71983 / GPU1–3 71967 MiB |

⚠ G4 的 `steady_s_per_step=7.500` 不是稳态：20 步日志无可做差分的 tqdm 点，汇总脚本退回取末条 rate（含首次编译）。稳态以正式训练前几百步为准；0928 同配置 4 任务库四卡实测 3.845–3.863 s/步，外推 80k 约 85 h。短测 ckpt 目录（11G）已删。

## 本轮 tmux 会话清单

| 会话名 | 用途 | 状态 |
|---|---|---|
| `cmo-smoke` | G4 20 步短测 | 已自然退出（`EXIT_CODE=0`） |
| `cmo-smoke-nvml` | G4 NVML 采样（GPU0–3） | 已由主会话 `kill-session -t '=cmo-smoke-nvml'`，删前删后 `tmux ls` 差集仅此一个 |
| `cmo-train` | 80k 正式训练 | 待起 |
| `cmo-inc` | 训练中增量上传 | 待起（run 目录出现后） |
| `cmo-upload-smoke` / `cmo-upload` | 正式上传驱动 | 训练 `EXIT_CODE=0` 且 `INC_END` 后 |

## 起跑条件与监控

起跑前：主检出 HEAD = 本 Beta、`git status --porcelain` 为空、GPU 0–3 无计算进程、`PREFLIGHT=PASS`。监控：训练日志挂 Monitor，过滤 `EXIT_CODE=|Traceback|RESOURCE_EXHAUSTED|out of memory|nan|NaN|Killed`；inc 日志过滤 `INC_UPLOADED|INC_END|UPLOAD_FAIL|HYGIENE=FAIL|错误|Traceback|EXIT_CODE=`。

## 用户决定与结论边界

OOM 即停下报错，不换卡数、不改 mask；步时慢不停；80k 中途崩溃、NaN、掉卡停下问用户，不 resume、不自动重跑（计划口径 8，沿用 2048 计划裁决 D）。训练崩溃时公开 bucket 停在部分 step 的处置交用户裁决。本轮不做评估，评估另立计划。
