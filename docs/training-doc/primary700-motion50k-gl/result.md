# primary700-motion50k-gl — 结果（motion 50000，700 条；与 primary700-nomotion50k-gl 成对）

> **⚠ 本页的 BinFill 部分已作废，总成绩已被取代。** BinFill 是四任务里唯一「训练有 demo 段、评测没有」的，
> 本页那 150 条是在训练时从未出现过的输入分布上打的分。补齐 demo 前缀重测后，
> 本模型（512+motion）的总成绩由 **40.71% 更新为 54.62%**。
> 现行权威结果见 [`binfilldemo-nomotion50k-gl/result.md`](../binfilldemo-nomotion50k-gl/result.md)。
>
> 本页其余部分（非 BinFill 的 11 组共 550 条、执行记录、推理开销、error 清单）**仍然有效且未重跑**，
> 新结果直接沿用。

**环境 A**（GreatLakes A40，gpu-hold-03…10，1×A40 / 1 CPU / 24G）。HEAD `d33c0ba`，被评 checkpoint bucket `HongzeFu/robomme-vla-modul-motion-80k-v1` 的 `50000`
（`motion.enabled=true`、`budget=160`、stride 16、window 33），策略 seed 7，`max_steps=2000`，同一套 700 条 test/primary 计划。
**两轮对照的完整结果、配对 2×2、执行记录、看门狗标记、资源统计都在配对 run 的
[`../primary700-nomotion50k-gl/result.md`](../primary700-nomotion50k-gl/result.md)**，本文件只放 motion 侧的判定行、总表与 motion 专属条目。

## 结论先行

```
MERGE_OK shards=10 episodes=700 successes=285 errors=9  macro_success_rate=0.4071  micro_success_rate=0.4071
```

对照：无 motion 50000 = 26.00%（182/700），老基线 59999 = 27.57%（193/700）。motion **+14.7 pp**；收益集中在 RouteStick（84% vs 35%）与
hard / xhard（+23 / +25 pp）；BinFill 退化（12% vs 29%）；VideoRepick +17 pp；VideoUnmaskSwap 持平（+4 pp）。

## 14 组成功率（三列：无 motion 50000 / motion 50000 / 老基线 59999）

| 任务 | 难度 | 无 motion 50000 | motion 50000 | 差（motion − 无） | 老基线 59999 |
|---|---|---|---|---|---|
| BinFill | easy | 23/50（46%） | 13/50（26%） | -10 | 25/50（50%） |
| BinFill | medium | 17/50（34%） | 5/50（10%） | -12 | 16/50（32%） |
| BinFill | hard | 4/50（8%） | 0/50（0%） | -4 | 2/50（4%） |
| RouteStick | easy | 41/50（82%） | 50/50（100%） | +9 | 44/50（88%） |
| RouteStick | medium | 20/50（40%） | 47/50（94%） | +27 | 23/50（46%） |
| RouteStick | hard | 8/50（16%） | 41/50（82%） | +33 | 10/50（20%） |
| RouteStick | xhard | 1/50（2%） | 29/50（58%） | +28 | 1/50（2%） |
| VideoRepick | easy | 13/50（26%） | 22/50（44%） | +9 | 9/50（18%） |
| VideoRepick | medium | 16/50（32%） | 19/50（38%） | +3 | 14/50（28%） |
| VideoRepick | xhard | 10/50（20%） | 23/50（46%） | +13 | 13/50（26%） |
| VideoUnmaskSwap | easy | 8/50（16%） | 8/50（16%） | +0 | 13/50（26%） |
| VideoUnmaskSwap | medium | 11/50（22%） | 17/50（34%） | +6 | 13/50（26%） |
| VideoUnmaskSwap | hard | 2/50（4%） | 7/50（14%） | +5 | 3/50（6%） |
| VideoUnmaskSwap | xhard | 8/50（16%） | 4/50（8%） | -4 | 7/50（14%） |
| **微平均** | | **182/700（26%）** | **285/700（41%）** | +103 | 193/700（28%） |
| **宏平均（14 组均值）** | | 26.00% | 40.71% | +14.71 pp | 27.57% |
| error 条目 | | 9 | 9 | | 9 |

### 按任务

| 任务 | 无 motion 50000 | motion 50000 | 差 | 老基线 59999 |
|---|---|---|---|---|
| BinFill | 44/150（29%） | 18/150（12%） | -17 pp | 43/150（29%） |
| RouteStick | 70/200（35%） | 167/200（84%） | +48 pp | 78/200（39%） |
| VideoRepick | 39/150（26%） | 64/150（43%） | +17 pp | 36/150（24%） |
| VideoUnmaskSwap | 29/200（14%） | 36/200（18%） | +4 pp | 36/200（18%） |

### 按难度

| 难度 | 无 motion 50000 | motion 50000 | 差 | 老基线 59999 |
|---|---|---|---|---|
| easy | 85/200（42%） | 93/200（46%） | +4 pp | 91/200（46%） |
| medium | 64/200（32%） | 88/200（44%） | +12 pp | 66/200（33%） |
| hard | 14/150（9%） | 48/150（32%） | +23 pp | 15/150（10%） |
| xhard | 19/150（13%） | 56/150（37%） | +25 pp | 21/150（14%） |

## motion 专属

- **sidecar 与 provenance**：每次 server 启动 `MOTION_PROV_RELAXED` 恰 6 行（`gpu_name 'NVIDIA A40' vs 'NVIDIA A100-SXM4-80GB'`、`compute_cap 8.6 vs 8.0`、
  `sm_count 84 vs 108`，vae / encoder 各一组），driver 595.71.05、torch 2.9.0+cu128、cuda 12.8、cudnn 91002、diffusers 0.39.0、VAE state sha、encoder ckpt sha
  `0c198629…` 全部与训练侧相等（`records/shards/s*-server.excerpt.txt`）。零截断契约（k ≤ 160）全程未触发。
- **计时（A40，10 片）**：16 帧 `add_buffer` 稳态 median 1,651–1,742 ms（sidecar 一窗 ≈1.64 s），`infer` 124–127 ms，首批 demo 段 17–33 s；
  无卡死的片 70 集净墙钟 94–107 min（无 motion 33–38 min）。
- **资源**：cgroup anon 峰 9.6–18.2 GiB（max 18.24，作业上限 24G），GPU 显存峰 35,762 MiB / 46,068，无卡死片 GPU util 均值 65–71%、0% 采样 15–19%，
  CPU 核当量 0.93–0.97（1 CPU 贴满）。
- **error 9 条**：VideoRepick/easy 189、medium 155 / 167 / 172、xhard 168 / 197 / 201，VideoUnmaskSwap/xhard 139 / 164——全部是看门狗标记的
  reset 卡死集，与无 motion 侧、与老基线逐条相同；motion s4 分片独占其中 4 条，续跑了 4 次才补齐。

## 产物

`records/merged/*.json`（合并结果）、`records/shards/s*-{check.json,checkpoint.txt,server.excerpt.txt,driver.excerpt.txt}`；
队列 / worker / 看门狗记录统一放在 `../primary700-nomotion50k-gl/records/`。视频与 server 全量日志留在 `v1-store/`。
