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

## 动态分配：8 个作业抢 20 个单元（gpu-hold-01、02 用户保留不用；09、10 为用户中途追加）

用户在计划批准后追加「61512013 gpu-hold-09 … 61512014 gpu-hold-10 … 这两个你也可以用」与「能否使用动态分配的机制？重新制定计划 使用动态的分配」，
AskUserQuestion 拍板工作单元粒度 = 20 个 (模型, 分片)。机制（`scripts/evaluation/gl_hold_pool.sh`）：队列目录
`v1-store/evaluation/primary700-ab50k-gl/queue/` 的 `items.txt` 按「长任务先、短任务填尾」列 `motion 0`…`motion 9`、`nomotion 0`…`nomotion 9`；
每个作业在登录节点起一个 worker（tmux `ev-ab-h03`…`ev-ab-h10`），靠 NFS 原子 `mkdir claims/<单元>` 抢单、跑完写 `done/<单元>`（rc、起止、作业、节点），
扫完退出；起跑时仍 PENDING 的 gpu-hold-10 先起 worker、自己等 RUNNING 再加入。同一分片的两轮不保证同节点同卡（用户已选此粒度）；
所有节点均为 A40 + 驱动 595.71.05，调用逐字相同。本机两进程并发干跑（`DRY_RUN=1`）：20 单元各被抢恰一次、两 worker 各 10。

| 作业 | jobid | 节点 | 起跑时状态 |
|---|---|---|---|
| gpu-hold-03 | 61495431 | gl1517 | RUNNING，剩 1 天 21.5 h |
| gpu-hold-04 | 61495432 | gl1523 | RUNNING，剩 1 天 21.5 h |
| gpu-hold-05 | 61495575 | gl1526 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-06 | 61495576 | gl1522 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-07 | 61495577 | gl1522 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-08 | 61495578 | gl1523 | RUNNING，剩 1 天 21.7 h |
| gpu-hold-09 | 61512013 | gl1524 | RUNNING，剩 1 天 23.9 h |
| gpu-hold-10 | 61512014 | — | PENDING |

实际抢单结果（单元 → 作业/节点/起止/rc）见 `result.md`。

## 冒烟（gpu-hold-08，两轮各 2 集）

（待补：冒烟完成后填）

## 正式起跑

（待补：正式起跑后填 HEAD、items.txt、八条 tmux 命令与起跑时刻）

## tmux 会话清单（清理唯一依据）

本机：`ev-ab-dl-60k50000`（下载）；登录节点：`ev-ab-smoke-gl`、`ev-ab-h03`…`ev-ab-h10`。
