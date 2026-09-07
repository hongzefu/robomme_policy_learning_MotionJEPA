# eval-hard-patternlock-routestick —— 起跑记录

官方 `perceptual-framesamp-context` ckpt 79999 在 **PatternLock / RouteStick 的 hard 难度档**上的评估，
**val 与 test 两个 split 都跑**，policy 采样 seed 单跑 42。规模 2 任务 × 2 split × 12 集 = **48 集**。

本仓库首次评估这两个任务——此前 `v1-store/evaluation/` 下 96 个结果目录全部只含 Permanence suite 的四个任务。

## 一、起跑环境

| 项 | 值 |
|---|---|
| 环境判定 | 环境 A（路径五项判据全符；GPU 判据表笔误已由 commit `ff94539` 修正） |
| 仓库根 | `/data/hongzefu/robomme_policy_learning_MotionJEPA`（分支 `v2-motionmem`） |
| 起跑 commit | `ff94539`（clean HEAD，`eval_shard.sh` 有硬门禁） |
| GPU | 2 × NVIDIA RTX 6000 Ada Generation，各 46068 MiB |
| CPU / 内存 | 32 核 / 377 GB（可用 331 GB） |
| 存储介质 | 本机 NVMe（`/dev/nvme1n1p1`，余 2.9 T） |
| 被评 ckpt | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999`（本地实体目录） |
| 加载口径 | `run_kind=legacy(history_config.txt=perceptual-framesamp-context.yaml)` |
| 仿真解释器 | `/home/hongzefu/micromamba/envs/robomme/bin/python` |
| benchmark 元数据 | `/data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata`（本仓库 submodule 为空） |

## 二、口径

### hard 档怎么取

hard 是**逐集属性**（元数据 `records[].difficulty`），不是 env 参数，没有任何 `--difficulty` CLI 开关。
实读四个「任务 × split」组合，hard 集号完全一致，各 12 集：

```
[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]      即 episode % 4 == 3
（每 split 50 集 = easy 26 / medium 12 / hard 12）
```

选集**零代码改动**：`eval_all_shards.sh` 的 stride 布局按 worker 号生成 `EP_STRIDE=WORKERS`、`EP_START=k`，
故 `MODE=stride WORKERS=4 ONLY=w3` 恰好给出 `EP_START=3 EP_STRIDE=4 EP_COUNT=0`
→ `range(3, 50, 4)` = 全部 12 个 hard 集，零 easy/medium 污染。

不经过 `eval_seed_sweep.sh`：其批后判据写死「本批实评集数 == 50」，跑 12 集会被误判失败。

### val / test 是环境 split，不是采样 seed

每集的环境初始状态 seed 固化在元数据里不可调。实读 seed 基址（`BASE + 100 × episode`）：

| 任务 | val 基址 | test 基址 |
|---|---|---|
| PatternLock | 1150000 | 650000 |
| RouteStick | 1160000 | 660000 |

两 split 零交集。`serve_policy.py --seed=42` 只换 flow-matching 采样噪声，换不了环境布局。

split 选择由 commit `7fb5206` 新增（`SPLIT` 经 `eval_all_shards.sh` → `eval_shard.sh` → `--args.dataset_split`
→ `EnvRunner(dataset=…)` → `BenchmarkEnvBuilder`），四层默认值均为 `test`，不传时行为与改动前逐字节相同。

## 三、预检（2 集，用后即删）

正式起评前先探「官方 ckpt 是否认识这两个任务」——其 `norm_stats.json` 只有全局 8 维统计、不分任务，
无法从 ckpt 内部证实训练覆盖。

```bash
MODE=stride WORKERS=4 ONLY=w3 EP_COUNT=8 TASKS_ALL=PatternLock SPLIT=test \
  GPU_LIST=0,0,0,0 PORT_BASE=9240 RUN_NAME=evhard-pre-pl-test CKPT_ID=79999 \
  CKPT_DIR=<ckpt> SEED=42 LOG_PREFIX=evhardpre-pl-test POLICY_MEM_FRACTION=0.40 \
  ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python \
  bash scripts/training/prod/eval_all_shards.sh
```

结果：`EVAL_RC=0`，14:10:03 → 14:11:07（64 秒），实评 2 集 `[3, 7]`，两集全败。
视频名后缀均为 `_hard.mp4`（难度自证），日志自证行 `split=test`。

**窗数异常排查**：预检每集只有 2 次 `infer`（32 步即终止，而 `max_steps=1300`）。
起 1 集 ButtonUnmask 对照（同一条改过的链路，`SPLIT=test` ep 3）：

| | infer 次数 | demo 前缀 |
|---|---|---|
| ButtonUnmask ep3 | **14** | `首批>16帧 n=0`（无 demo） |
| PatternLock ep3/ep7 | **各 2** | `frames=199 / 93`（有 demo） |

两处交叉印证链路正常：demo 前缀的有无恰好对应 `utils.py::TASK_WITH_VIDEO_DEMO` 的成员关系
（PatternLock 在其中、ButtonUnmask 不在）。2 窗是 PatternLock 的任务特性——描错一笔即判失败、
环境主动终止（`stop_flag` 路径，非 `timeout` 路径）。

对照集 ButtonUnmask ep3 也是 fail，**与既有留档一致**：`eval-official-framesamp-context` 记录
ButtonUnmask 的 `hard 0/12`（hard 档 12 集全败）。改过的链路在已知任务的已知集号上重现了已知结果。

预检与对照产物均已删除（`v1-store/evaluation/evhard-pre-*`、对应日志）。

## 四、正式起跑命令（4 单元并行）

全部共用：`MODE=stride WORKERS=4 ONLY=w3 CKPT_ID=79999 SEED=42 POLICY_MEM_FRACTION=0.40`
`ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python`
`CKPT_DIR=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-context/79999`

| TASKS_ALL | SPLIT | GPU_LIST | PORT_BASE | 端口 | RUN_NAME = LOG_PREFIX | tmux 会话 |
|---|---|---|---|---|---|---|
| PatternLock | test | `0,0,0,0` | 9200 | 9203 | `evhard-pl-test` | `evhard-pl-test-w3` |
| PatternLock | val | `0,0,0,0` | 9210 | 9213 | `evhard-pl-val` | `evhard-pl-val-w3` |
| RouteStick | test | `1,1,1,1` | 9220 | 9223 | `evhard-rs-test` | `evhard-rs-test-w3` |
| RouteStick | val | `1,1,1,1` | 9230 | 9233 | `evhard-rs-val` | `evhard-rs-val-w3` |

`GPU_LIST` 填四个相同卡号，是因为 `ONLY=w3` 取 `GPU_ARR[3 % 4]`——填满四位即把该单元钉在指定卡上。
每卡 2 个 worker，`POLICY_MEM_FRACTION=0.40`（既有实测每卡 2 worker 约占 39.6 GB / 46 GB）。

起跑时刻：2026-09-06 14:14:08。

### 端口红线

8040 段整段避开——`ss -ltnp` 实测 8042/8044/8045/8047/8048/8049/8050 被用户自己的 `python3` 服务占用
（对应用户 tmux 会话 `site8042`、`site8044`、`ensite8048/8049/8050`）。commit `baa93fa` 记录的
「静默跑 0 集」正是撞在这里。本轮全部用 9200 段。

### tmux 会话清单（AGENTS 第 7 条）

本轮**只起**上表 4 个 `evhard-*-w3` 会话。用户原有的 7 个会话
（`ensite8048`、`ensite8049`、`ensite8050`、`site8042`、`site8044`、`unisite`、`v3site`）**一个都不碰**。
清理时逐个 `tmux kill-session -t <确切会话名>`，删前删后各 `tmux ls` 核对差集。

## 五、汇总命令

```bash
UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/summarize_hard_eval.py \
  --units evhard-pl-test:PatternLock:test --units evhard-pl-val:PatternLock:val \
  --units evhard-rs-test:RouteStick:test --units evhard-rs-val:RouteStick:val \
  --ckpt-id 79999 --seed 42 \
  --metadata-dir /data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata \
  --out-dir docs/training-doc/eval-hard-patternlock-routestick/records
```

## 六、判据

| 判定行 | 含义 |
|---|---|
| `HARD_ONLY` | 48 集的视频名后缀全为 `_hard`（难度取自 `env.unwrapped.difficulty` 回读值） |
| `EP_SET` | 每 unit 实评集号恰为 `{3,7,…,47}` |
| `SPLIT_SEED` | split 真的生效：①日志自证行 split 相符 ②同 task 跨 split 的 seed 两两不同 ③同名视频数（观察项） |
| `NO_REGRESSION` | 已在 commit `7fb5206` 验过：不设 `SPLIT` 时 stride/task 两种布局命令行与改动前逐字节一致 |

`SPLIT_SEED` 是最关键的一条——若 split 没接通，val 会静默跑成 test 的环境，成功率数字看起来完全正常，
但两组其实是同一批环境实例，对照整个是假的。该判据五个场景已实测（正常 / 假 val / 传参丢失 /
日志缺字段 / 单 split），均符合预期。
