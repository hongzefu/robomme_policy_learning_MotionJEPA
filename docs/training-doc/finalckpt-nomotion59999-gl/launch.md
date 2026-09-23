# finalckpt-* 起跑留档（三个最终 ckpt × 692 条，三个 run 共用）

结果见 [`../binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)（两档合并的权威结果页）。

## 用户指令原话

1.「跑robomme-vla-modul-motion-80k-v1  79999 / robomme-vla-modul-2048-80k-v1 79999 /
   robomme-vla-modul-60k-v1 59999 全部重新测」
2.「我的完整结果需要呈现现在的和最终ckpt的情况」

## 被评权重

| run_name | checkpoint | POLICY | POLICY_CONFIG |
|---|---|---|---|
| `finalckpt-nomotion59999-gl` | `robomme-vla-modul-60k-v1/59999` | `perceptual-framesamp-modul-8frame-8x8` | `mme_vla_suite` |
| `finalckpt-motion79999-gl` | `robomme-vla-modul-motion-80k-v1/79999` | `perceptual-framesamp-modul-8frame-8x8-motion` | `mme_vla_suite_b128_80k` |
| `finalckpt-modul32frame79999-gl` | `robomme-vla-modul-2048-80k-v1/79999` | `perceptual-framesamp-modul-32frame-8x8` | `mme_vla_suite_b128_80k` |

**60k run 没有 80k**：它训到 60000 步为止，最终 ckpt 是 59999；另两条是 80k run。
所以「最终 ckpt」这一档不是等步数比较。

2048 的 79999 本地原先没有，本轮用 `download.sh` 增量模式拉取：19 个文件、14 G，
逐文件 sha256 校验全部 `OK`。另两个 ckpt 本地已有。

## 口径

与前几轮**逐字相同**：seed 7、`max_steps=2000`、`EPISODE_WALL_S=2400`、`EVAL_TIMEOUT=14400`、
1 卡 / 1 CPU / 24G、benchmark gitlink `b4e97f2`、候选库 identity `3b4de03a0b46…`。

本轮特有：
- `DEMO_PREFIX_STORE` 指向 `v1-store/demo-prefix/binfill-3b4de03a0b46`（BinFill 142 条注入 demo）
- `EXPECT_DEMO_INJECT=1`（开启 `DEMO_INJECT` 闸）
- `ALLOW_RESUME=1`（因为预标了 error，见下）
- `CHUNK_EPISODES=20`
- motion 那条另加 `MMEVLA_MOTION_OVERFLOW=resample` 与 `EXPECT_MOTION_STATS=1`
  （由 `gl_hold_pool.sh` 按 policy 名是否以 `-motion` 结尾自动分叉）
- **`MMEVLA_ENC_CHUNK` 不设**（= 不分批，与改动前逐字等价；用户 2026-09-22 决定）

起跑 HEAD：`f3e12e5`（clean）。

## 692 条混合计划

`make_shard_plan.py shards --demo-store <库根> --shards 10`：BinFill 按前缀库过滤（只收
`demo_status=ok` 的 142 条），其余三任务原样全收 550 条。

    PLAN_OK shards=10 total=692 per_shard=69 disjoint=True groups=14
      组分布: BinFill/easy=49 BinFill/hard=44 BinFill/medium=49 其余 11 组各 50

为跑这份混合计划改了代码（commitV10.9）：原来的两道守卫遇到非 BinFill 组直接 raise，
改成「BinFill 注入、其余不注入」并配三道显式检查——非 BinFill 打 `DEMO_PREFIX_DISABLED`、
起跑前硬校验每条 BinFill 都在 store 里、`check_shard.py` 新增 `DEMO_INJECT` 闸逐条核对注入边界。

## 预标的 9 条卡死 episode

`VideoRepick/easy 189`、`VideoRepick/medium 155 / 167 / 172`、`VideoRepick/xhard 168 / 197 / 201`、
`VideoUnmaskSwap/xhard 139 / 164`——前几轮逐条相同，卡在环境 `make_env`/reset 的 native 调用里，
`eval.py` 的 `SIGALRM` 单集墙钟打不断。起跑前按新分片（s0/s1/s4/s6/s7）写进各自 `progress.json` 为 `"error"`，
配 `ALLOW_RESUME=1`。不预标会白烧 3 个 run × 9 条 × 40 分钟墙钟。清单见 `records/preseeded-errors.json`。

## 执行

8 个既有 gpu-hold 作业（`61728418`–`61728421`、`61730613`–`61730616`）动态抢单，
30 行 manifest（3 run × 10 片）走 `gl_hold_pool.sh` 的 MANIFEST 模式。

**30/30 单元 `rc=0`**、30 个 `SHARD_PASS`、3 个 `MERGE_OK episodes=692 errors=9`。
墙钟约 4 小时 20 分钟（19:49 起、04:06 全部完成）。

## 验收

| run | 分片 | 注入合计 | DEMO_INJECT | MOTION_WINDOWS |
|---|---|---|---|---|
| `finalckpt-nomotion59999-gl` | 10 | 142 | 全 PASS | SKIP（非 motion） |
| `finalckpt-motion79999-gl` | 10 | 142 | 全 PASS | 全 PASS |
| `finalckpt-modul32frame79999-gl` | 10 | 142 | 全 PASS | SKIP（非 motion） |

三个 run 的注入数都恰好等于 BinFill 全集 142，非 BinFill 集一条都没注入——
这正是新增 `DEMO_INJECT` 闸要防的失效模式（静默注错或漏注）。

motion 十片的 `k_max` 为 98–175，其中一片 175 触发降级 15 次，闭式预测与服务端实测精确相等；
该片同时 `DEMO_INJECT=PASS`，两道闸在同一片上同时生效。

## 意外与处置

1. **原实现不支持混合计划**：两道守卫会拒绝非 BinFill 组。改成按任务分叉 + 三道显式检查（commitV10.9）。
2. **`cmd_shards` 的均匀性断言过严**：带 `--demo-store` 时 BinFill 三组被剔成 49/49/44，
   「每组条数相同」必然失败，放宽为「最多差 1 条」；不带该参数时严格判据原样保留。
3. **2048 的 79999 本地没有**，先用 `download.sh` 增量模式拉了 14 G。

## 产物

`records/merged/{log,progress}.json`（本 run）；另两个 run 在各自目录，motion 那条另有
`merged/motion_stats.json`。本目录另存 `manifest.tsv`、`preseeded-errors.json`、
`shard-checks.json`（30 个分片的两道闸验收汇总）。

编排脚本（不进 git）：`v1-store/evaluation/final-ckpt-queue/{pool-run.sh,preseed.py,manifest.tsv,plans/}`。
