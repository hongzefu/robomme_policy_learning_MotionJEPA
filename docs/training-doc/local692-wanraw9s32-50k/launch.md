# local692-wanraw9s32-50k 评估起跑档案

本页记录本机 692 局评估的起跑口径，结果见 [result.md](result.md)。计划见仓库根目录 [1112-4task-modul-wanraw9-s32-50k-plan.html](../../../1112-4task-modul-wanraw9-s32-50k-plan.html) §4、闸 G8；对照 v11 档案 [local692-wantp768-50k](../local692-wantp768-50k/launch.md)。

## 被评对象

- checkpoint：`v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-wanraw9s32-b128-50k/50000`（训练档案 [v2-1600ep-m8x8-modul-wanraw9s32-b128-50k](../v2-1600ep-m8x8-modul-wanraw9s32-b128-50k/result.md)，`CKPT_LOAD=PASS step=50000 leaves=65 resolved_sha256=819a001c…3062`）；history YAML `perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml`（motion：wan_raw9、9 token/窗、stride 32、budget 80 窗）。
- 代码：分支 `v15-4task-modul-wanraw9-s32-50k` HEAD `66eb506`（`commitV15.18`，合入 sub/EVAL32 `0e63c49`；与训练 Beta `3addd7c` 只差评估分片可配置化，评估判据未改）；benchmark 子模块 `b4e97f22fe007078e297205898c07c1acbc69165`。
- 评估侧闸（s5 首跑日志原文）：`BUDGET_CONSISTENT=PASS budget=80 motion_kind=wan_raw9 readout=wan_raw9_flat16384_s32_v1 tokens_per_window=9`、`PARAM_TREE_EXACT=PASS n_model=65 n_ckpt=65 missing=0 extra=0 shape_mismatch=0 motion_leaves=4`、`HISTORY_CONFIG=PASS NORM_STATS=PASS MOTION_ENABLED=PASS config=mme_vla_suite_b128_80k step=50000`。

## 口径（与 v11 43.06% 同口径，只改分片与每卡并发）

- 候选：benchmark `b4e97f2` 内 `artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl`（identity `3b4de03a0b46`）；BinFill 用本机 demo 前缀库 `v1-store/demo-prefix/binfill-3b4de03a0b46`（与 v11 同一份，`E0_GATE=PASS DEMO_PREFIX_SRC=PASS code_same=1 benchmark=b4e97f2 shards_ok=16/16 ok=142 failed=8 diff_vs_old=40`）。
- 692 = 550（三个非 BinFill 任务，不注入）+ 142（BinFill，注入 demo 前缀）；seed 7；`max_steps=2000`；`EPISODE_WALL_S=2400`；`EVAL_TIMEOUT=14400`。
- 环境变量（`EVAL_ENV` 行原文）：`MMEVLA_MOTION_OVERFLOW=resample EXPECT_MOTION_STATS=1 EXPECT_DEMO_INJECT=1 CHUNK_EPISODES=20 MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count … LOCAL_GPU_SHARE=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 POLICY=perceptual-framesamp-modul-8frame-8x8-wanraw9s32 POLICY_CONFIG=mme_vla_suite_b128_80k EXPECT_CKPT_ID=50000 SHARDS=32 PER_GPU=4 MEM_FRACTION=0.2`。
- **与 v11 的差异（用户指令）**：v11 为 10 片、每卡 ≤2 片、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`；本次 32 片、每卡恰 4 片（s0–s31 按 `s<i> → GPU i mod 8`）、显存占比 0.2。用户原话「实现每卡4worker worker现存等以后都不需要保持一致」，分片数与每卡并发、显存占比不作口径要求；评估判据（`check_shard`、merge、PRESEED、`EVAL692`）不变。
- 分片：`PLAN_OK shards=32 total=692 per_shard=21 disjoint=True groups=14`；组分布 BinFill easy/medium/hard 49/49/44，其余 13 组各 50。
- 历史 9 局 reset 卡死（VideoRepick 7、VideoUnmaskSwap 2）按身份重定位预标 `"error"`：`PRESEED=PASS episodes=9 placed=9 duplicates=0`，计入分母按失败；其他新 error 只重跑一次。

## 启动

链脚本 `chain_wanraw9s32_train_eval.sh` 原定训完自动起 tmux `r9-eval`（10 片）。用户改为每卡 4 片后，链脚本在 `r9-train` 中运行不可改，主会话于美东 11-13 10:42 预放占位 `v1-store/logs/r9-eval.log`（内容「PLACEHOLDER … 正式评估日志改用 r9-eval32.log」），链脚本在训练闸通过后以 `CHAIN=STOP reason=eval_log_exists` 停下（预期）。随后主会话手动起 tmux `r9-eval32`，`LOCAL692_START … 2026-11-13T17:30:31Z`（美东 12:30:31）。启动脚本为主会话临时文件（未入库），内容如下（等价于 `local_692_pool.sh` 正式模式）：

```bash
# tmux r9-eval32：set -o pipefail; bash <launch_eval32.sh> 2>&1 | tee v1-store/logs/r9-eval32.log; echo "EXIT_CODE=$?" >> v1-store/logs/r9-eval32.log
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$MAIN"
export POLICY=perceptual-framesamp-modul-8frame-8x8-wanraw9s32 RUN_NAME=local692-wanraw9s32-50k \
       EXPECT_MOTION_STRIDE=32 EXPECT_MOTION_BUDGET=80 STORE="$MAIN/v1-store/demo-prefix/binfill-3b4de03a0b46" \
       EVAL_SHARDS=32 EVAL_PER_GPU=4 EVAL_MEM_FRACTION=0.2 PYTHONUNBUFFERED=1
bash scripts/evaluation/local_692_pool.sh "$MAIN/v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-wanraw9s32-b128-50k/50000" 7
```

**s5 补跑**（见 result ⑩）：s5 首跑在同卡 4 片并发启动时被进程归属检查误判退出（`EXIT_CODE=2`，0 局，日志改名 `s5.try1.log`），主会话于美东 12:41:57 起 tmux `r9-eval32-s5`，环境与正式模式逐项一致，`ALLOW_RESUME=1`，在 `gpu-slots/gpu5` 以 flock 先登记 `s5 pending` 再换成 PID，直接调 `run_shard.sh`：

```bash
# tmux r9-eval32-s5：set -o pipefail; bash <rerun_s5.sh> 2>&1 | tee v1-store/logs/r9-eval32-s5.log; echo "EXIT_CODE=$?" >> …
source scripts/evaluation/env.sh
export MMEVLA_MOTION_OVERFLOW=resample EXPECT_MOTION_STATS=1 EXPECT_DEMO_INJECT=1 CHUNK_EPISODES=20 \
       MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 \
       LOCAL_GPU_SHARE=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 ALLOW_RESUME=1 \
       POLICY=perceptual-framesamp-modul-8frame-8x8-wanraw9s32 POLICY_CONFIG=mme_vla_suite_b128_80k \
       EXPECT_CKPT_ID=50000 DEMO_PREFIX_STORE=$MAIN/v1-store/demo-prefix/binfill-3b4de03a0b46 \
       EXPECT_MOTION_STRIDE=32 EXPECT_MOTION_BUDGET=80
unset ROBOMME_GPU_RASTER SAPIEN_DISABLE_RAY_TRACING
env CUDA_VISIBLE_DEVICES=5 SHARD_INDEX=5 LOCAL_GPU_SLOT_FILE=$R/gpu-slots/gpu5 \
    bash scripts/evaluation/run_shard.sh local692-wanraw9s32-50k "$CKPT" $R/plans/shard5.json 7 s5 > $R/logs/s5.log 2>&1
```

另起看守 tmux `r9-eval32-guard`：若其余分片先于 s5 完成，则 SIGSTOP 驱动主进程、防止在 s5 缺结果时合并；最终未触发（`GUARD_DONE s5done=1 others_done=18 stopped=0 13:30`）。

评估目录 `v1-store/evaluation/local692-wanraw9s32-50k`（视频与逐分片产物留在 v1-store，不进 git）。

## 本轮 tmux 会话清单

`r9-eval32`（驱动，`EXIT_CODE=0`）、`r9-eval32-s5`（s5 补跑，`EXIT_CODE=0`）、`r9-eval32-guard`（看守，`EXIT_CODE=0`）；链脚本原定的 `r9-eval` 未启动。

## 盯盘项

主会话以 Monitor 盯 `v1-store/logs/r9-eval32.log`（分片退出、扫描、合并、`EVAL692=` 判定行与报错行）及 `r9-eval32-s5.log`；每卡 4 片的 OOM 与同卡并发启动风险（`PRE_MERGE_REVIEW` 非阻塞提示）首跑重点盯。
