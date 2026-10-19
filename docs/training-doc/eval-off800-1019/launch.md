# eval-off800-1019 — 起跑记录

**评估在 GreatLakes（spgpu，单卡 A40）执行，编排在 sled-aspen 与 GL 登录节点 `gl-login2`。** 按 AGENTS 第 12 条留档，本文件是第一段（「版本与代码状态」「启动与配置还原」）；`result.md` 在两组评估完成后补写。时间一律美东，日志原样 UTC 时间戳括注。

## 一、目的

在官方 eval seed（历史「官方 800 局」口径：16 任务 × 官方 test metadata 的 ep0–49，策略 seed 42，max_steps 1300）上评自训 512+motion run 的两个 checkpoint：先 step 79999，再 step 40000。用于与历史 `eval-orig80k-modul-vs-official`、`eval-orig40k-ckpt40000` 的 800 局结果对照。

## 二、用户决策原话

1. 2026-10-19：「给出方案 跑完之后跑带motion的80k  40k注意优先跑80k 跑完了再跑40k 在官方的eval seed上 查看有什么阻塞的地方」
2. 2026-10-19（对方案口径 (a)「seed 42、1300 步，与历史 800 局可比」）：「a 同意开工」
3. 沿用 2026-10-18：「站位的卡可以停了。」「跑完了就可以停。」——本轮复用 1017 的 10 个占位 job，两组跑完后逐个释放。

## 三、版本与代码状态

| 版本 | commit | 内容 |
|---|---|---|
| commitV11.29 | `d27b5cd` | 1017 testx 40k 两模型结果回写（本轮前置） |
| commitV11.30 | `1a9477c` | `sub/OFF800` `3bdd0f0`：`gl_hold_pool.sh` 新增 `EVAL_SEED`（默认 7）／`EVAL_MAX_STEPS`（默认 2000），`run_shard.sh` 用 `EVAL_MAX_STEPS` 并导出 `EXPECT_MAX_STEPS`，`gl_hold_queue.sh` 白名单透传，`make_manifest.py export-official`；`PRE_MERGE_REVIEW=PASS base=f98a988 tip=3bdd0f0 files=6 commits=1 findings=0` |
| commitV11.31Beta | 本 commit | 起跑锚点；worker 的 `EXPECTED_GIT_HEAD` 即此 sha |

子模块 `third_party/robomme_benchmark` 锁 benchmark `943ddf32`：test metadata 已追加 ep50 起的 xhard 记录，但 ep0–49 纯追加未改（benchmark 侧 `PROTECTED_APPEND_ONLY=PASS`），seed 与官方原版一致。

## 四、被评对象

| 顺序 | run_name | checkpoint | policy / config | 身份核验 |
|---|---|---|---|---|
| 1 | `off800-512m-80k` | `v1-store/models/robomme-vla-mixed8-200-modul-motion-80k-v1/79999` | `mixed8-512-motion` / `mme_vla_suite_b128_80k` | 已在 1017 80k 轮核过（`CKPT_SHA=PASS`），单元起跑时 `check_checkpoint` 再核 |
| 2 | `off800-512m-40k` | 同 run `/40000` | 同上 | 已在 1017 40k 轮核过，同上 |

Motion encoder 同 1017：`wan-mixed8-200-filter2-stop075-b88x8-72ep-a`。

## 五、评估清单

- `make_manifest.py export-official v1-store/evaluation/official800` → `eval-official-800.jsonl`：每任务恰 ep0..49，共 800 行，只含 easy／medium／hard（easy 416、medium 192、hard 192）；子代理在 scratch 实跑 `MANIFEST_OFFICIAL=PASS rows=800 tasks=16 sha256=702028a2…`，首条 BinFill ep0 seed 540000。
- 每 run 10 片 × 80 局（`make_shard_plan.py --shards 10 --expect-per-shard 80`，`rows[i::10]` 发牌）。
- ⚠ 与历史口径的差异：历史 800 局用 `scripts/training/legacy-eval/robomme-local/` 的旧 eval.py（`--args.dataset/episode_start`），本轮用现行 `examples/robomme/eval.py` + `episode_plan`；episode、env seed、策略 seed、max_steps 四项一致，循环实现不同，对比时注明。

## 六、启动与配置还原

- **运行参数**：策略 seed 42（`EVAL_SEED=42`）、`max_steps` 1300（`EVAL_MAX_STEPS=1300`，`check_shard` 的 `EXPECT_MAX_STEPS` 同值）；其余与 1017 相同，写死在 `gl_hold_pool.sh`：`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、`CHUNK_EPISODES=20`、`PROV_RELAX` 四键、`EXPECT_DEMO_INJECT=0`；policy 名以 `-motion` 结尾自动设 `MMEVLA_MOTION_OVERFLOW=resample`、`EXPECT_MOTION_STATS=1`。不设 `XHARD_METADATA_DIR`（无 xhard 行）、不设 `RESET_CHECK`。
- **GL 资源**：复用 1017 的 10 个占位 job 64243171–64243180（规格 `--account=chaijy2 --partition=spgpu --gres=gpu:1 --gpu_cmode=shared --cpus-per-task=4 --mem=24G --time=48:00:00 --wrap='sleep infinity'`，清单 `v1-store/logs/hold-jobs-testx80k.txt`，scancel 唯一依据）。
- **manifest 与队列**：`v1-store/evaluation/official800/manifest.tsv`（20 行，80k 10 行在前、40k 10 行在后），队列 `v1-store/evaluation/official800/queue`。
- **worker**：`gl-login2` 上每个占位 job 一个 tmux 会话 `xh-off800-pool-<jobid>`：

```bash
tmux new-session -d -s xh-off800-pool-<jobid> "set -o pipefail; cd $R && MANIFEST=$R/v1-store/evaluation/official800/manifest.tsv QUEUE_DIR=$R/v1-store/evaluation/official800/queue RUN_SUFFIX=-gl EVAL_SEED=42 EVAL_MAX_STEPS=1300 MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count,driver bash scripts/evaluation/gl_hold_pool.sh <jobid> 2>&1 | tee -a $R/v1-store/logs/policy-eval/off800-pool-<jobid>.log; echo \"EXIT_CODE=\$?\" >> $R/v1-store/logs/policy-eval/off800-pool-<jobid>.log"
```

（`R=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA`）

- **起跑判据**：`WORKER_START … seed=42 max_steps=1300`；`CLAIM item=off800-512m-80k-s<k>`；每片 `EVAL_PARAMS=PASS max_steps=1300`。
- **结果路径**：`v1-store/evaluation/off800-512m-{80k,40k}/s<k>/mixed8-512-motion/ckpt<step>/seed42/`。
- **本轮 tmux 会话清单**：`xh-off800-pool-64243171` … `xh-off800-pool-64243180`（gl-login2）。
