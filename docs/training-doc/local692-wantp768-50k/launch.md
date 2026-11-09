# local692-wantp768-50k 评估起跑档案

本页记录本机 692 局评估的起跑口径，结果见 [result.md](result.md)。计划见仓库根目录 [1108-4task-modul-wantp768-50k-plan.html](../../../1108-4task-modul-wantp768-50k-plan.html) §4、§5 E0/E1。

## 被评对象

- checkpoint：`v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-wantp768-b128-50k/50000`（训练档案 [v2-1600ep-m8x8-modul-wantp768-b128-50k](../v2-1600ep-m8x8-modul-wantp768-b128-50k/result.md)）；history YAML `perceptual-framesamp-modul-8frame-8x8-wantp768.yaml`，resolved_sha256 `4e70e933…1910`（`check_checkpoint.py` 白名单一致）。
- 代码：分支 `v11-4task-modul-wantp768-50k` HEAD `dfc73df`（与训练 Beta `f7dfe16` 只差训练结论文档）；benchmark 子模块 `b4e97f22fe007078e297205898c07c1acbc69165`。

## 口径（与上次 GreatLakes 692 局同口径，环境不同）

- 候选：benchmark `b4e97f2` 内 `artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl`（identity `3b4de03a0b46`）；BinFill 用本机重生成的 demo 前缀库 `v1-store/demo-prefix/binfill-3b4de03a0b46`（`DEMO_PREFIX_SRC=PASS code_same=1 benchmark=b4e97f2 shards_ok=16/16 ok=142 failed=8 diff_vs_old=40`：成功集合与旧清单相同，40 条 D 差 1–9 步，按用户裁决只记录）。
- 692 = 550（三个非 BinFill 任务，不注入）+ 142（BinFill，注入 demo 前缀）；seed 7；`max_steps=2000`；`EPISODE_WALL_S=2400`；`EVAL_TIMEOUT=14400`。
- 环境变量：`MMEVLA_MOTION_OVERFLOW=resample`、`EXPECT_MOTION_STATS=1`、`EXPECT_DEMO_INJECT=1`、`CHUNK_EPISODES=20`、`MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count`、`LOCAL_GPU_SHARE=2`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`。
- 分片：保留历史 10 个逻辑分片（`PLAN_OK shards=10 total=692 per_shard=69 disjoint=True groups=14`），s0–s7 → GPU 0–7，s8 → GPU 0、s9 → GPU 1（每卡 ≤2 片，flock 预占卡位，以 run_shard.sh PID 为进程树根登记）。
- 历史 9 局 reset 卡死（VideoRepick 7、VideoUnmaskSwap 2）按身份重定位预标 `"error"`：`PRESEED=PASS episodes=9 placed=9 duplicates=0`，计入分母按失败；其他新 error 只重跑一次。
- 54.62%（上次 encoder motion @50000）只作跨环境历史参考（上次 GreatLakes A40 + 历史前缀，本次 AWS A100 + 本机重生成前缀），不写成严格同口径。

## 启动

tmux `w768-eval`，2026-11-09T08:01:24Z（美东 03:01:24），训练 `TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10` 后按状态机 `kind: go` 自动开工：

```bash
tmux new-session -d -s w768-eval "cd /scratch/hongze/robomme_policy_learning_MotionJEPA && set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/evaluation/local_692_pool.sh /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-wantp768-b128-50k/50000 7 2>&1 | tee v1-store/logs/w768-eval.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/w768-eval.log"
```

评估目录 `v1-store/evaluation/local692-wantp768-50k`（视频与逐局产物留在 v1-store，不进 git）。
