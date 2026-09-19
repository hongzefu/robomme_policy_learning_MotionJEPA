# primary700-nomotion50k-gl 起跑留档（无 motion 50000，与 primary700-motion50k-gl 同一套调用）

**环境 A**（仓库工作副本 turbo，分支 `v2-eval-0918-motion`）。本 run 与 [`primary700-motion50k-gl`](../primary700-motion50k-gl/launch.md)
成对：同一 HEAD、同一套分片计划、同一作业、同一张卡、同一节点、前后紧挨，只差被评权重与结果目录名。

## 用户指令原话

「跑无motion和带motion的50k ckpt全部700个eval 在集群6张卡上跑 尽可能保持一致的调用」。
AskUserQuestion 拍板：无 motion 侧 = bucket `HongzeFu/robomme-vla-modul-60k-v1` 的 **`50000/`** 步（与 motion 同步数对照，
不是已有基线 `primary700-gl` 用的 59999）；run_name `primary700-nomotion50k-gl`。
计划正本：`~/.claude/plans/motion-eval-https-github-com-hongzefu-r-cozy-ullman.md`（2026-09-18 批准）。

## 被评权重

- bucket `HongzeFu/robomme-vla-modul-60k-v1` 的 `50000/`（run `v2-1600ep-m8x8-modul-b128-60k`，motion 关闭）。本地落点
  `v1-store/models/robomme-vla-modul-60k-v1/50000/`，由 `download.sh` **增量模式**补下（落点已有 59999 与根文件，本次只取 `50000/` 的
  22 个文件），按本地已有的 `SHA256SUMS.pre.txt` 逐文件校验：`CHECKPOINT_DOWNLOAD_PASS mode=sha256 files=22`（本机 tmux `ev-ab-dl-60k50000`，2026-09-18 21:55:22Z → 22:05:53Z，10 min 31 s；`records/dl-vla-60k-50000.txt`，校验清单 `records/vla-60k-50000-SHA256SUMS.selected.txt`）；59999 与根文件字节未动。
- run 快照 `history_config.resolved.yaml` sha256 `804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92`（与 59999 同一份，
  无 `motion` 节，`motion_provenance.json.motion_enabled=false`）；`norm_stats.json` sha256 `856c75ea…`（与 59999、与 motion 50000 三者同一份）。
- `check_checkpoint.py`（服务端环境、CPU）对 50000：`PARAM_TREE_EXACT=PASS n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0 motion_leaves=0`、
  `HISTORY_CONFIG=PASS NORM_STATS=PASS MOTION_DISABLED=PASS config=mme_vla_suite step=50000`（`records/checkpoint-50000.txt`）

## 与 motion 轮共用的口径（两轮逐字相同）

驱动 `scripts/evaluation/gl_hold_ab.sh <jobid> <shard>...`：对每个分片先跑无 motion、紧接着跑 motion，两轮各自走
`gl_hold_queue.sh → srun --jobid=<既有作业> --overlap --exact --gpu_cmode=shared --cpus-per-task=1 --gpus-per-node=1 --time=12:00:00
→ gl_eval_shard.sbatch → run_shard.sh`。共用：HEAD、`primary700-gl/plans/shard{0..9}.json`（身份 sha `3b4de03a…`，复制到各自 run 的
`plans/`）、策略 seed 7、`max_steps=2000`、`EXPECT_CKPT_ID=50000`、`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、`CHUNK_EPISODES=20`、
`MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count`（本轮无 sidecar、无人读取，只为环境逐字相同）、benchmark gitlink `b4e97f2`、
1×A40 / 1 CPU / 24G。只差四个值：

| | 本 run（无 motion） | motion 轮 |
|---|---|---|
| `RUN_NAME` | `primary700-nomotion50k-gl` | `primary700-motion50k-gl` |
| `CKPT` | `v1-store/models/robomme-vla-modul-60k-v1/50000` | `v1-store/models/robomme-vla-modul-motion-80k-v1/50000` |
| `POLICY` | `perceptual-framesamp-modul-8frame-8x8` | `perceptual-framesamp-modul-8frame-8x8-motion` |
| `POLICY_CONFIG` | `mme_vla_suite` | `mme_vla_suite_b128_80k` |

与老基线 `primary700-gl`（59999，sbatch array、2 CPU / 32G、`EPISODE_WALL_S` 默认 900）相比，本轮两侧互相一致，但资源档位与墙钟
与老基线不同；对照表里三列并排时以此为注。

## 动态分配：8 个作业抢 20 个单元 + 看门狗 + 续跑队列（gpu-hold-01、02 用户保留不用；09、10 为用户中途追加）

用户在计划批准后追加「61512013 gpu-hold-09 … 61512014 gpu-hold-10 … 这两个你也可以用」与「能否使用动态分配的机制？重新制定计划 使用动态的分配」，
AskUserQuestion 拍板工作单元粒度 = 20 个 (模型, 分片)。机制（`scripts/evaluation/gl_hold_pool.sh`，commitV10.4）：队列目录
`v1-store/evaluation/primary700-ab50k-gl/queue/` 的 `items.txt` 按「长任务先、短任务填尾」列 `motion 0`…`motion 9`、`nomotion 0`…`nomotion 9`；
每个作业在登录节点起一个 worker（tmux `ev-ab-h03`…`ev-ab-h10`），靠 NFS 原子 `mkdir claims/<单元>` 抢单、跑完写 `done/<单元>`（rc、起止、作业、节点），
扫完退出。同一分片的两轮不保证同节点同卡（用户已选此粒度）；所有节点均为 A40 + 驱动 595.71.05，调用逐字相同。
本机两进程并发干跑（`DRY_RUN=1`）：20 单元各被抢恰一次、两 worker 各 10。

| 作业 | jobid | 节点 | 起跑时状态 |
|---|---|---|---|
| gpu-hold-03 | 61495431 | gl1517 | RUNNING，剩 1 天 21.5 h |
| gpu-hold-04 | 61495432 | gl1523 | RUNNING，剩 1 天 21.5 h |
| gpu-hold-05 | 61495575 | gl1526 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-06 | 61495576 | gl1522 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-07 | 61495577 | gl1522 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-08 | 61495578 | gl1523 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-09 | 61512013 | gl1524 | RUNNING，剩 1 天 23.9 h |
| gpu-hold-10 | 61512014 | gl1511 | 计划时 PENDING，起跑前已 RUNNING |

**起跑后补上的两层（用户 2026-09-19 授权「自动标 error + KILL 客户端」）**，脚本本体在 `v1-store/evaluation/primary700-ab50k-gl/watchdog.sh`
（不进 git，副本见 `records/watchdog.sh.txt`）：

- **看门狗**（登录节点 tmux `ev-ab-watchdog`，每 180 s 扫一次）：VideoRepick / VideoUnmaskSwap 的 9 条 episode 会卡死在 `make_env`/reset 的
  native 调用里（与老基线 `primary700-gl` 的 9 条 error 完全同一批），`eval.py` 的 `SIGALRM` 单集墙钟打不断，只能等 4 h 的 `EVAL_TIMEOUT`，
  且 `run_shard.sh` 有 `set -e`、客户端一被砍整片 rc≠0 退出。判据：`client.log` 静默 ≥ 12 min 且最后一行是 `env for <group> episode <ep> setup finished`。
  处置：先把该集写成 `"error"`（与 `False`/fail 严格区分，`check_shard.py` 只接受 `{True, False, "error"}`），同时追加到 `queue/hang_marks.txt`
  以区分「看门狗标的 error」与「评测自身抛出的 error」；再经 `srun --jobid --overlap` 在计算节点只 `pkill -KILL` 该分片的客户端 python
  （`[e]val.py` 括号技巧防自匹配，不碰 server / sidecar / 其他分片）。
- **续跑队列** `queue-resume/`：看门狗把主队列 rc≠0 的单元追加进 `items.txt`；续跑再失败（同一分片含多条坏集）则把 `claims/`、`done/` 归档到
  `history/` 允许再抢，每单元最多 4 次尝试。续跑 worker 为 `ALLOW_RESUME=1 QUEUE_DIR=…/queue-resume gl_hold_pool.sh <jobid>`
  （tmux `ev-ab-r0K` 一次性 + `ev-ab-q0K` 循环直到 `queue-resume/STOP`），走同一条调用链，`run_shard.sh` 按 `progress.json` 跳过已评与已标 error 的集。

实际抢单结果（单元 → 作业 / 节点 / 起止 / rc）与看门狗标记表见 `result.md`。

## 冒烟（gpu-hold-08，两轮各 2 集）

HEAD `41f116f`（clean）。登录节点 tmux `ev-ab-smoke-gl` 内 `RUN_SUFFIX=-smoke-gl CHUNK_EPISODES=1 QUEUE_DIR=…/primary700-ab50k-smoke/queue
gl_hold_pool.sh 61495578`，队列两项 `nomotion smoke`、`motion smoke`（计划 = RouteStick/xhard 115 + BinFill/hard 153）。
`WORKER_START … node=gl1523 22:12:15Z` → `WORKER_DONE job=61495578 ran=2 failed=0 22:22:37Z`（10 min 22 s）。日志 `records/smoke-gl-worker.txt`，
两轮 `check.json` / `progress.json` / server 摘录 / `done` 记录均在 `records/smoke-gl-*`。

| 轮 | 起止（UTC） | 结果 | `MOTION_PROV_RELAXED` | server 进程树 | 首批 `add_buffer`（451 帧） | 首次 `infer` | 16 帧 `add_buffer` 稳态 median | `infer` 稳态 median |
|---|---|---|---|---|---|---|---|---|
| 无 motion 50000 | 22:12:15 → 22:15:25（3 min 10 s） | `SHARD_PASS evaluated=2 successes=0 errors=[]` | 0 行（无 sidecar） | 2 级 | 3,562 ms | 3,663 ms | **62 ms**（n=45） | 124 ms |
| motion 50000 | 22:15:25 → 22:22:37（7 min 12 s） | `SHARD_PASS evaluated=2 successes=1 errors=[]`（RouteStick 成功） | 恰 6 行（gpu_name / compute_cap / sm_count × vae / encoder） | 4 级（含 sidecar） | 49,044 ms | 3,823 ms | **1,711 ms**（n=93） | 126 ms |

两轮 `EVAL_PARAMS` 除 `policy` / `config` 外逐字相同（`ckpt=50000 seed=7 max_steps=2000 wall=2400 online_gpu=0 prov_relax=gpu_name,compute_cap,sm_count`）。
首次 `infer` 从上一轮冒烟的 43 s 降到 3.8 s：JAX 编译缓存（`v1-store/cache/policy-eval/jax`）已热。冒烟产物（两个 `*-smoke-gl` 运行目录与冒烟队列）已删除。

## 正式起跑

HEAD `d33c0ba`（clean，冒烟记录 docs commit 之后）。2026-09-18 **22:24:27Z** 在登录节点一次性起八个 detached tmux `ev-ab-h03`…`ev-ab-h10`，
每个：`bash $REPO/scripts/evaluation/gl_hold_pool.sh <jobid> 2>&1 | tee -a $REPO/v1-store/logs/policy-eval/primary700-ab50k-gl.h<K>.log`
（作业号 03:61495431 04:61495432 05:61495575 06:61495576 07:61495577 08:61495578 09:61512013 10:61512014；起跑时 gpu-hold-10 已 RUNNING 于 gl1511）。
队列 `v1-store/evaluation/primary700-ab50k-gl/queue/items.txt`（20 行）：

```
motion 0 … motion 9
nomotion 0 … nomotion 9
```

本机每份 `h<K>.log` 各挂一个 Monitor（过滤 `WORKER_START|CLAIM|SHARD_PASS|SHARD_FAIL|ITEM_DONE|WORKER_DONE|EXIT_CODE=|Traceback|…`）。
实际抢单结果（单元 → 作业 / 节点 / 起止 / rc）见 `result.md`。

## tmux 会话清单（清理唯一依据）

本机：`ev-ab-dl-60k50000`（下载）；登录节点：`ev-ab-smoke-gl`、`ev-ab-h03`…`ev-ab-h10`。
