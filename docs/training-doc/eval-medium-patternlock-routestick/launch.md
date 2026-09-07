# eval-medium-patternlock-routestick —— 起跑记录

官方 `perceptual-framesamp-context` 与 `perceptual-framesamp-modul` 两个 ckpt 79999 在
**PatternLock / RouteStick 的 medium 难度档**上的评估，val 与 test 两个 split，单 seed 42。
每变体 2 任务 × 2 split × 12 集 = 48 集。

本轮是 `docs/training-doc/eval-hard-patternlock-routestick/` 的**难度档延伸**：那轮测 hard，
本轮换 medium，其余口径（任务、split、seed、ckpt、分片方式、资源档位）完全一致，便于对照。
单独建目录是因为原目录名写死了 `hard`，塞 medium 结果名不副实（用户 2026-09-06 拍板）。

## 与 hard 轮的唯一差别：集号与难度档

medium 是逐集属性 `difficulty == "medium"`，实读四个「任务 × split」组合完全一致，各 12 集：

```
[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]      即 episode % 4 == 2
```

选集仍**零代码改动**：`eval_all_shards.sh` 的 stride 布局按 worker 号生成 `EP_START=k`、
`EP_STRIDE=WORKERS`，故 `MODE=stride WORKERS=4 ONLY=w2` 恰好给出 `EP_START=2 EP_STRIDE=4 EP_COUNT=0`
→ `range(2, 50, 4)` = 全部 12 个 medium 集。（hard 轮用的是 `ONLY=w3`。）

其余环境、ckpt、口径与 hard 轮相同，见 `../eval-hard-patternlock-routestick/launch.md`。

## 汇总脚本的一处改动

`scripts/training/prod/summarize_hard_eval.py` 原先把期望难度写死为 `hard`（`_HARD = "hard"` 常量），
跑 medium 时 48 集全部被判"非 hard"、`HARD_ONLY=FAIL`。本轮将其参数化：

- 新增 `--expect-difficulty {easy,medium,hard}`，默认 `hard`（向后兼容）。
- 判定行 `HARD_ONLY` 更名为 `DIFFICULTY_ONLY`，输出含 `expect=<档>`——原名在检查 medium 时会误导。

回归验证：用新脚本重跑 hard 轮，`per_episode.json` 与已提交留档**逐字节一致**，`summary.txt`
唯一差异是该判定行改名（`HARD_ONLY=PASS checked=48 non_hard=0` → `DIFFICULTY_ONLY=PASS expect=hard
checked=48 mismatched=0`），数字全同。hard 轮的已提交留档**保持原样不改**——它是当时的真实输出。

## 起跑命令

共用：`MODE=stride WORKERS=4 ONLY=w2 CKPT_ID=79999 SEED=42 POLICY_MEM_FRACTION=0.40`
`ROBOMME_PY=/home/hongzefu/micromamba/envs/robomme/bin/python`

### context 组（CKPT_DIR = …/official-mme-vla/perceptual-framesamp-context/79999）

| TASKS_ALL | SPLIT | GPU_LIST | PORT_BASE | 端口 | RUN_NAME = LOG_PREFIX | tmux 会话 |
|---|---|---|---|---|---|---|
| PatternLock | test | `0,0,0,0` | 9300 | 9302 | `evmed-pl-test` | `evmed-pl-test-w2` |
| PatternLock | val | `0,0,0,0` | 9310 | 9312 | `evmed-pl-val` | `evmed-pl-val-w2` |
| RouteStick | test | `1,1,1,1` | 9320 | 9322 | `evmed-rs-test` | `evmed-rs-test-w2` |
| RouteStick | val | `1,1,1,1` | 9330 | 9332 | `evmed-rs-val` | `evmed-rs-val-w2` |

起跑 commit `643133b`，起跑时刻 2026-09-06 15:41:39，四单元并行，全部 `EXIT_CODE=0`。

### modul 组（CKPT_DIR = …/official-mme-vla/perceptual-framesamp-modul/79999）

| TASKS_ALL | SPLIT | GPU_LIST | PORT_BASE | 端口 | RUN_NAME = LOG_PREFIX | tmux 会话 |
|---|---|---|---|---|---|---|
| PatternLock | test | `0,0,0,0` | 9340 | 9342 | `evmed-mod-pl-test` | `evmed-mod-pl-test-w2` |
| PatternLock | val | `0,0,0,0` | 9350 | 9352 | `evmed-mod-pl-val` | `evmed-mod-pl-val-w2` |
| RouteStick | test | `1,1,1,1` | 9360 | 9362 | `evmed-mod-rs-test` | `evmed-mod-rs-test-w2` |
| RouteStick | val | `1,1,1,1` | 9370 | 9372 | `evmed-mod-rs-val` | `evmed-mod-rs-val-w2` |

**两组必须串行**：每个 policy 占约 18.6 GB，2 × 46 GB 的卡最多容纳 4 个 worker。

## 汇总命令

```bash
UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/summarize_hard_eval.py \
  --units <run>:<Task>:<split> …（四个） \
  --shard w2 --expect-episodes 2,6,10,14,18,22,26,30,34,38,42,46 --expect-difficulty medium \
  --ckpt-id 79999 --seed 42 \
  --metadata-dir /data/hongzefu/robomme_policy_learning-vqa-test/third_party/robomme_benchmark/src/robomme/env_metadata \
  --out-dir docs/training-doc/eval-medium-patternlock-routestick/records/<context|modul>
```

## tmux 会话清单（AGENTS 第 7 条）

本轮只起上两表共 8 个 `evmed-*-w2` 会话。用户原有的 7 个会话（`ensite8048`、`ensite8049`、
`ensite8050`、`site8042`、`site8044`、`unisite`、`v3site`）一个都不碰。8040 段端口整段避开。
