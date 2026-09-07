# eval-binfill-pickxtimes —— 起跑记录

官方 `perceptual-framesamp-context` 与 `perceptual-framesamp-modul` 两个 ckpt 79999 在
**BinFill / PickXtimes 的 hard 与 medium 两档**上的评估，val + test 两 split，单 seed 42。
每变体每档 2 任务 × 2 split × 12 集 = 48 集，全轮共 **16 单元 / 192 集**。

## 与前两轮的关系

前两轮（`../eval-hard-patternlock-routestick/`、`../eval-medium-patternlock-routestick/`）测的是
**Imitation / Procedural memory suite** 的 PatternLock、RouteStick。本轮换成
**Counting / Temporal memory suite** 的 BinFill、PickXtimes，其余口径（split、seed、ckpt、
分片方式、资源档位）完全一致，用于检验 context 的劣势是特定于 Imitation suite 还是普遍现象。

**本轮一个目录收两个难度**（records 下按 `{hard,medium}/{context,modul}` 两级），与前两轮
hard/medium 分目录的体例不同——那两轮是 hard 先跑完、medium 后追加，本轮两档一起跑，分开反而割裂对照。

一处结构性差异：BinFill / PickXtimes **不在** `examples/robomme/utils.py::TASK_WITH_VIDEO_DEMO` 里，
故无视频演示前缀（日志里 `add_buffer(首批>16帧) n=0`），策略从第一帧就开始执行；
PatternLock / RouteStick 则要先灌 199 / 93 帧 demo。

## 集号核对（起跑前实读元数据）

| split | task | easy | medium | hard | hard 集号 == w3 | medium 集号 == w2 |
|---|---|---|---|---|---|---|
| test | BinFill | 26 | 12 | 12 | True | True |
| test | PickXtimes | 26 | 12 | 12 | True | True |
| val | BinFill | 26 | 12 | 12 | True | True |
| val | PickXtimes | 26 | 12 | 12 | True | True |

四个组合均无例外（`StopCube` 的 val 曾出现 hard 13 集的例外，故每轮必须实读核对）。
选集仍零代码改动：`ONLY=w3` → hard `[3,7,…,47]`，`ONLY=w2` → medium `[2,6,…,46]`。

seed 基址：BinFill test 540000 / val 1040000；PickXtimes test 510000 / val 1010000。

## 一次显存事故与编排修正

**首次起跑（16:03）四个单元里两个 BinFill 单元双双失败**：`EVAL_RC=139`（SIGSEGV）与 `EVAL_RC=1`，
`infer n=0`，报 `CUDA error at .../sapien-vulkan-2/src/core/buffer.cpp 251: out of memory`。

排查后确认根因**不是** Counting 任务的仿真需求大，而是**卡上另有他人任务**：进程表显示 16 个
约 740 MiB 的进程来自另一仓库 `/data/hongzefu/robomme_benchmark_MotionJEPA` 的
`scripts/data-generation-newSeed/generate_dataset_newseed.py`（用户自己的数据生成任务，17 个子进程，
两卡各约 6 GB），恰在本轮起跑前 12 秒启动。叠加本轮 2 × policy(18.6 GB) = 37.3 GB 后 GPU1 达
44341 / 46068 MiB，仅剩 1.7 GB，BinFill 遂被挤爆。

处置：立即停掉评估单元让出显存（GPU1 降至 24.8 GB），清空全部半截产物，等用户任务结束后重跑。
用户另有一个 `plrs-test` 会话不在本轮清单内，按 AGENTS 第 7 条未作任何处理。

**编排修正两处**：
1. 起跑前必须先 `nvidia-smi` 查实际占用再定并发，不照搬上一轮的档位（本次重跑前实测 GPU0 1017 MiB、
   GPU1 9 MiB，仅两个他人常驻小进程）。
2. 两个 BinFill 单元**拆到不同卡**，每卡配一个 PickXtimes——首次事故中 BinFill 两个同压 GPU0 双双崩溃、
   而 GPU1 的两个 PickXtimes 存活，提示 BinFill 仿真更吃显存。

## 起跑命令

共用：`MODE=stride WORKERS=4 CKPT_ID=79999 SEED=42 POLICY_MEM_FRACTION=0.40`
`ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python`；hard 用 `ONLY=w3`、medium 用 `ONLY=w2`。

| 批 | 变体 | 难度 | RUN_NAME = LOG_PREFIX | GPU | 端口 |
|---|---|---|---|---|---|
| 1 | context | hard | `evbp-h-{bf,px}-{test,val}` | bf-test/px-test→0，bf-val/px-val→1 | 9403 / 9423 / 9413 / 9433 |
| 2 | context | medium | `evbp-m-{bf,px}-{test,val}` | 同上 | 9442 / 9462 / 9452 / 9472 |
| 3 | modul | hard | `evbp-h-mod-{bf,px}-{test,val}` | 同上 | 9483 / 9503 / 9493 / 9513 |
| 4 | modul | medium | `evbp-m-mod-{bf,px}-{test,val}` | 同上 | 9522 / 9542 / 9532 / 9552 |

会话名 = `<LOG_PREFIX>-w3`（hard）或 `-w2`（medium）。**四批串行**：每 policy 约 18.6 GB，
2 × 46 GB 的卡最多容纳 4 个 worker。

批 1 起跑 commit `f612792` / 16:21:43；批 2 同 commit / 16:31:22；批 3、4 见 result.md。

## 汇总命令

```bash
UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/summarize_hard_eval.py \
  --units <run>:<Task>:<split> …（四个） \
  --shard w3 --expect-difficulty hard \        # medium 改 --shard w2 --expect-episodes 2,6,…,46 --expect-difficulty medium
  --ckpt-id 79999 --seed 42 \
  --metadata-dir /data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata \
  --out-dir docs/training-doc/eval-binfill-pickxtimes/records/<hard|medium>/<context|modul>
```

## tmux 会话清单（AGENTS 第 7 条）

本轮只起上表 16 个 `evbp-*` 会话。用户原有会话（`ensite8048/8049/8050`、`site8042`、`site8044`、
`unisite`、`v3site`，以及一度出现的 `plrs-test`）一个都不碰。8040 段端口整段避开。
