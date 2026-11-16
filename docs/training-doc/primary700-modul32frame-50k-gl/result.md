# primary700-modul32frame-50k-gl — 结果（32 帧 / 2048，modulation，motion 关，50000 步）

> **⚠ 本页的 BinFill 部分已作废，总成绩已被取代。** BinFill 是四任务里唯一「训练有 demo 段、评测没有」的，
> 本页那 150 条是在训练时从未出现过的输入分布上打的分。补齐 demo 前缀重测后，
> 本模型（2048（32 帧 / budget 2048））的总成绩由 **34.86% 更新为 45.09%**。
> 现行权威结果见 [`binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。
>
> 本页其余部分（非 BinFill 的 11 组共 550 条、执行记录、推理开销、error 清单）**仍然有效且未重跑**，
> 新结果直接沿用。

`MERGE_OK shards=10 episodes=700 successes=244 errors=9`，**宏平均 = 微平均 = 34.86%**（每组恰 50 集，两者必然相等）。
起跑与口径见 [`launch.md`](launch.md)。下面是与同为 50000 步的另两条 run 的三方对照，**三条全是 modulation 注入**，
差别只在感知记忆规模与 motion 通道开关。

## 结论先行

1. **把感知上下文从 8 帧 / 512 提到 32 帧 / 2048，拿到 +8.86 pp（26.00% → 34.86%），但只有 motion 收益的六成**
   （motion 是 +14.71 pp → 40.71%）。更大的感知窗口有效，但顶不上 motion memory。
2. **收益全在难题**：easy 三者持平（42.5 / 43.0 / 46.5%），medium 起才拉开；hard 与 xhard 上本 run 把
   「8 帧 → motion」的差距走完了约一半。
3. **BinFill 上本 run 复现了 motion 的退化**（10.7% vs motion 12.0%，8 帧档是 29.3%）。两条机制完全不同的 run 撞进同一个坑，
   说明 BinFill 掉分**不是 motion token 引起的**，而是「感知记忆一变强就变差」的共性问题——这是本轮最值得跟进的发现。
4. **代价几乎为零**：`infer` 124 → 132 ms（+6%），`add_buffer` 与显存峰完全不变，单片墙钟反而略短于 8 帧档。
   motion 那条路每 16 步要多付约 1.7 秒的 sidecar 开销。

| 简称 | run_name | 感知记忆（frame sampling） | motion 通道 | 宏=微平均 |
|---|---|---|---|---|
| 8帧/512 | `primary700-nomotion50k-gl` | 8 帧 × 64 token = 512 | 关 | **26.00%**（182/700） |
| **32帧/2048** | `primary700-modul32frame-50k-gl` | **32 帧 × 64 token = 2048** | 关 | **34.86%**（244/700） |
| 512+motion | `primary700-motion50k-gl` | 8 帧 × 64 token = 512 | **开（motion budget 160）** | **40.71%**（285/700） |

## 14 组逐格

| 任务 | 难度 | 8帧/512 | **32帧/2048** | 512+motion | 2048 − 512 | 2048 − motion |
|---|---|---|---|---|---|---|
| BinFill | easy | 46% | **20%** | 26% | -26 pp | -6 pp |
| BinFill | medium | 34% | **12%** | 10% | -22 pp | +2 pp |
| BinFill | hard | 8% | **0%** | 0% | -8 pp | +0 pp |
| RouteStick | easy | 82% | **100%** | 100% | +18 pp | +0 pp |
| RouteStick | medium | 40% | **98%** | 94% | +58 pp | +4 pp |
| RouteStick | hard | 16% | **70%** | 82% | +54 pp | -12 pp |
| RouteStick | xhard | 2% | **34%** | 58% | +32 pp | -24 pp |
| VideoRepick | easy | 26% | **34%** | 44% | +8 pp | -10 pp |
| VideoRepick | medium | 32% | **32%** | 38% | +0 pp | -6 pp |
| VideoRepick | xhard | 20% | **26%** | 46% | +6 pp | -20 pp |
| VideoUnmaskSwap | easy | 16% | **18%** | 16% | +2 pp | +2 pp |
| VideoUnmaskSwap | medium | 22% | **26%** | 34% | +4 pp | -8 pp |
| VideoUnmaskSwap | hard | 4% | **4%** | 14% | +0 pp | -10 pp |
| VideoUnmaskSwap | xhard | 16% | **14%** | 8% | -2 pp | +6 pp |
| **全部** | | **26.00%** | **34.86%** | **40.71%** | **+8.86 pp** | **-5.86 pp** |

## 按任务

| 任务 | n | 8帧/512 | **32帧/2048** | 512+motion | 2048 − 512 | 2048 − motion |
|---|---|---|---|---|---|---|
| BinFill | 150 | 29.3% | **10.7%** | 12.0% | -18.7 pp | -1.3 pp |
| RouteStick | 200 | 35.0% | **75.5%** | 83.5% | +40.5 pp | -8.0 pp |
| VideoRepick | 150 | 26.0% | **30.7%** | 42.7% | +4.7 pp | -12.0 pp |
| VideoUnmaskSwap | 200 | 14.5% | **15.5%** | 18.0% | +1.0 pp | -2.5 pp |

## 按难度

| 难度 | n | 8帧/512 | **32帧/2048** | 512+motion | 2048 − 512 | 2048 − motion |
|---|---|---|---|---|---|---|
| easy | 200 | 42.5% | **43.0%** | 46.5% | +0.5 pp | -3.5 pp |
| medium | 200 | 32.0% | **42.0%** | 44.0% | +10.0 pp | -2.0 pp |
| hard | 150 | 9.3% | **24.7%** | 32.0% | +15.3 pp | -7.3 pp |
| xhard | 150 | 12.7% | **24.7%** | 37.3% | +12.0 pp | -12.7 pp |

## 逐集配对（剔除任一侧为 error 的集）

| 对照 | 配对数 | A 成功 | B 成功 | 都成 | 仅 A 成 | 仅 B 成 | 都败 |
|---|---|---|---|---|---|---|---|
| 32帧/2048 (A) vs 8帧/512 (B) | 691 | 244 | 182 | 111 | **133** | **71** | 376 |
| 32帧/2048 (A) vs 512+motion (B) | 691 | 244 | 285 | 176 | **68** | **109** | 338 |
| 512+motion (A) vs 8帧/512 (B) | 691 | 285 | 182 | 112 | **173** | **70** | 336 |

## 推理开销（10 片范围，A40 / 1 CPU / 24G）

| 项 | 8帧/512 | **32帧/2048** | 512+motion |
|---|---|---|---|
| 16 帧 `add_buffer` median | 62–64 ms | **61–62 ms** | 1,651–1,742 ms |
| `infer` median | 124–125 ms | **132–133 ms** | 124–127 ms |
| 单片 70 集墙钟 | 33–38 min | **26–36 min** | 94–107 min |
| GPU 显存峰 | 32,771 MiB | **32,771–32,773 MiB** | 35,762 MiB |
| GPU util 均值 | 14–16% | **14–17%** | 65–71% |
## error 条目

三条 run 各 9 条、逐条相同：`VideoRepick/easy 189`、`VideoRepick/medium 155 / 167 / 172`、
`VideoRepick/xhard 168 / 197 / 201`、`VideoUnmaskSwap/xhard 139 / 164`。本 run 的 9 条是**起跑前预先标记**的
（见 `launch.md`「预先标记的 9 条卡死 episode」与 `records/preseeded-errors.json`），另两条是当轮现场由看门狗标记的。
三者都计入分母、不计成功，配对表已逐对剔除。补跑计划在 `records/merged/retry_plan.json`，本轮不补跑。

## 盲区诚实清单

- 单 seed（7）、单 checkpoint（三条各 50000）。
- 三条 run 不是严格的单变量对照：本 run 与 motion run 同属 b128 / 80k 训练配置，8 帧 512 那条是 b128 / 60k，
  训练总步数与 checkpoint 间隔不同；结论只在「50000 步这一横截面」上成立。
- BinFill 的退化未做视频归因，只知两条不同机制的 run 同向退化。
- VideoUnmaskSwap 三条都在 14.5–18.0%，各格差值在 ±6 集以内，不作单格结论。
- 成功率来自环境 `status` 判定，本轮未人工抽看视频。
- 9 条 error 是环境侧 reset 卡死（策略尚未动作），三条 run 对称。

## 产物

`records/merged/{log,progress,errors,shards,retry_plan}.json`、`records/shards/s*-*`、`records/preseeded-errors.json`、
`records/manifest.tsv`。视频与 server 全量日志留在 `v1-store/evaluation/primary700-modul32frame-50k-gl/`（不进 git）。
