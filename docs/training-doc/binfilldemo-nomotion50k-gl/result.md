# 三模型完整结果：50000 与最终 ckpt 两档（BinFill 补 demo 后；旧 BinFill 因训练/推理不一致作废）

本页是三个模型的**现行权威结果**，同时呈现 50000 步与各自最终 ckpt 两档。
起跑口径见 [`launch.md`](launch.md)（50000 档）与
[`finalckpt-nomotion59999-gl/launch.md`](../finalckpt-nomotion59999-gl/launch.md)（最终 ckpt 档），
计划见根目录 [`0922-binfill-demo-prefix-plan.md`](../../../0922-binfill-demo-prefix-plan.md)。

## 结论先行

1. **旧的 BinFill 评测结果已作废**（训练/推理不一致：BinFill 是四任务里唯一「训练有 demo 段、评测没有」的）。
   本页所有 BinFill 数字均为补齐 demo 前缀后重测所得。
2. **三条 run 在 50k 之后基本收敛**：训到各自最终 ckpt 只带来 +0.6 ~ +2.5 pp，
   排序与量级都没有变化。
3. 三者排序在两档都一致：**512+motion > 2048 > 512**，与补 demo 前 BinFill 上的颠倒排序相反。

| 模型 | @50000 | 最终 ckpt | 变化 |
|---|---|---|---|
| 512（8 帧 / budget 512） | 211/692 = **30.49%** | 228/692 = **32.95%**（@59999） | +2.46 pp |
| 2048（32 帧 / budget 2048） | 312/692 = **45.09%** | 316/692 = **45.66%**（@79999） | +0.58 pp |
| 512+motion | 378/692 = **54.62%** | 390/692 = **56.36%**（@79999） | +1.73 pp |

macro 分别 0.3050→0.3295、0.4498→0.4566、0.5470→0.5635。error 每档每模型均为 9 条
（三个 run 完全相同的 reset 卡死集，全在 VideoRepick / VideoUnmaskSwap，计入分母按失败处理，故为下界）。

**注意 60k run 没有 80k**：它训到 60000 步为止，最终 ckpt 是 59999；另两条是 80k run，最终 ckpt 79999。
所以「最终 ckpt」这一列不是同一个训练步数，不能横向当作等步数比较。

## 按任务（@50k → 最终 ckpt）

| 任务 | 512 | 2048 | 512+motion |
|---|---|---|---|
| **BinFill** | 51.4% → 54.9% | 59.2% → 59.9% | 78.2% → 76.8% |
| RouteStick | 35.0% → 39.0% | 75.5% → 80.5% | 83.5% → 83.5% |
| VideoRepick | 26.0% → 24.0% | 30.7% → 28.0% | 42.7% → 48.0% |
| VideoUnmaskSwap | 14.5% → 18.0% | 15.5% → 14.0% | 18.0% → 21.0% |

逐任务有升有降，幅度多在 ±5 pp 内——这与「50k 后已收敛、剩下的是噪声级波动」一致，
不宜逐格解读成某个任务变好或变坏。

## 按难度（@50k → 最终 ckpt）

| 难度 | 512 | 2048 | 512+motion |
|---|---|---|---|
| easy | 50.3% → 53.8% | 58.3% → 54.3% | 62.3% → 65.8% |
| medium | 36.7% → 39.2% | 56.3% → 53.8% | 63.8% → 64.3% |
| hard | 13.2% → 15.3% | 32.6% → 42.4% | 49.3% → 46.5% |
| xhard | 12.7% → 14.0% | 24.7% → 26.7% | 37.3% → 42.7% |

## 14 组逐格（最终 ckpt 档）

| 组 | 512 @59999 | 2048 @79999 | 512+motion @79999 |
|---|---|---|---|
| **BinFill/easy** | 41/49 = 83.7% | 39/49 = 79.6% | 45/49 = 91.8% |
| **BinFill/medium** | 28/49 = 57.1% | 30/49 = 61.2% | 45/49 = 91.8% |
| **BinFill/hard** | 9/44 = 20.5% | 16/44 = 36.4% | 19/44 = 43.2% |
| RouteStick/easy | 44/50 = 88.0% | 50/50 = 100.0% | 50/50 = 100.0% |
| RouteStick/medium | 23/50 = 46.0% | 47/50 = 94.0% | 47/50 = 94.0% |
| RouteStick/hard | 10/50 = 20.0% | 42/50 = 84.0% | 39/50 = 78.0% |
| RouteStick/xhard | 1/50 = 2.0% | 22/50 = 44.0% | 31/50 = 62.0% |
| VideoRepick/easy | 9/50 = 18.0%（err 1） | 12/50 = 24.0%（err 1） | 25/50 = 50.0%（err 1） |
| VideoRepick/medium | 14/50 = 28.0%（err 3） | 16/50 = 32.0%（err 3） | 19/50 = 38.0%（err 3） |
| VideoRepick/xhard | 13/50 = 26.0%（err 3） | 14/50 = 28.0%（err 3） | 28/50 = 56.0%（err 3） |
| VideoUnmaskSwap/easy | 13/50 = 26.0% | 7/50 = 14.0% | 11/50 = 22.0% |
| VideoUnmaskSwap/medium | 13/50 = 26.0% | 14/50 = 28.0% | 17/50 = 34.0% |
| VideoUnmaskSwap/hard | 3/50 = 6.0% | 3/50 = 6.0% | 9/50 = 18.0% |
| VideoUnmaskSwap/xhard | 7/50 = 14.0%（err 2） | 4/50 = 8.0%（err 2） | 5/50 = 10.0%（err 2） |

50000 档的 14 组逐格见本页历史版本与三个 `binfilldemo-*` / `primary700-*` 目录。

## 为什么旧 BinFill 作废

BinFill 是四个评测任务里**唯一**「训练有 demo 段、评测没有」的：

- 训练侧 `generate_dataset_newseed.py::_binfill_demo_deliverable`（2026-09-11 路线 B）把整条成功轨迹
  复制一遍接在自己前面，400/400 满足 `num_timesteps == 2 × exec_start_idx`，
  `exec_start_idx` min 264 / mean 642 / max 1152；
- 评测侧 `BinFill.py::_initialize_episode` 的 `task_list` 里 `demonstration` 三处全 `False`，
  `reset()` 只返回 1 帧，`exec_start_idx = 0`。

实测佐证：三个 700 条 run 的 `client.log` 各 691 行 `exec_start_idx` + 9 条 error = 700，
其中**恰有 150 行为 0、全部是 BinFill**。

其它三个任务的训练 es 区间与评测 es 区间完全对齐（RouteStick 100–500/100–500、
VideoRepick 252–508/254–502、VideoUnmaskSwap 114–318/114–318），所以只有 BinFill 需要作废重测。

## BinFill 补 demo 前后（@50000 档，同一批 142 条）

两侧分母都是 142（planner 演不出来的 8 条两侧同时剔除）。无 demo 侧不重跑，
直接从已留档的 700 条结果里按同一子集重新统计。

| 模型 | 难度 | 无 demo（作废） | 补 demo | 变化 |
|---|---|---|---|---|
| 512 | easy | 22/49 = 44.9% | 38/49 = 77.6% | +32.7 pp |
| | medium | 17/49 = 34.7% | 26/49 = 53.1% | +18.4 pp |
| | hard | 4/44 = 9.1% | 9/44 = 20.5% | +11.4 pp |
| **512** | **合计** | **43/142 = 30.3%** | **73/142 = 51.4%** | **+21.1 pp（1.70x）** |
| 2048 | easy | 10/49 = 20.4% | 40/49 = 81.6% | +61.2 pp |
| | medium | 6/49 = 12.2% | 34/49 = 69.4% | +57.1 pp |
| | hard | 0/44 = 0.0% | 10/44 = 22.7% | +22.7 pp |
| **2048** | **合计** | **16/142 = 11.3%** | **84/142 = 59.2%** | **+47.9 pp（5.25x）** |
| 512+motion | easy | 13/49 = 26.5% | 44/49 = 89.8% | +63.3 pp |
| | medium | 5/49 = 10.2% | 44/49 = 89.8% | +79.6 pp |
| | hard | 0/44 = 0.0% | 23/44 = 52.3% | +52.3 pp |
| **512+motion** | **合计** | **18/142 = 12.7%** | **111/142 = 78.2%** | **+65.5 pp（6.17x）** |

## 排序恢复：作废旧结果的最直接依据

| 口径 | 排序 |
|---|---|
| 其它三个任务（550 条，@50000） | motion 48.5% > 2048 41.5% > 512 25.1% |
| BinFill **无 demo**（作废） | 512 30.3% > motion 12.7% > 2048 11.3%（**颠倒**） |
| BinFill **补 demo** @50000 | motion 78.2% > 2048 59.2% > 512 51.4%（**恢复一致**） |
| BinFill **补 demo** 最终 ckpt | motion 76.8% > 2048 59.9% > 512 54.9%（保持一致） |

补 demo 之前，BinFill 是唯一一个让 motion 与 2048 输给 8 帧基线的任务；补上之后这个反常消失，
且在最终 ckpt 档依然成立。

「记忆通道越宽、被 demo 缺失掏空的比例越大、补回后回升越多」这条预测写在跑之前的计划里
（`0922-binfill-demo-prefix-plan.md` Context 一节），不是事后解释。实测回升倍数与通道宽度单调对应：
8 帧感知 1.70x、32 帧感知 5.25x、8 帧 + 160 motion 窗 6.17x。

## 这些数字不能怎么用

1. **BinFill 补 demo 前后两列不是「同条件下的改进」。** 补 demo 把喂给模型的帧序列整个换了：
   `even_sampling_indices` 现在在 `[0, es+count]` 上取点，8 帧档里有 4–7 帧、32 帧档里有 16–28 帧
   落进 demo 段（此前是 0 帧）；motion 路的 demo 窗从 0 变成常驻 16–69 个。
   **帧路的变化量级远大于 motion 降级**。作废旧结果的理由是它口径错了，不是因为新结果更好看。
2. **「最终 ckpt」列不是等步数比较**：512 是 59999（60k run 的终点），另两条是 79999。
3. **hard 的分母缩得最多**（剔除 8 条里 hard 占 6 条），不能把 hard 的提升读成「hard 变简单了」。
4. **即使补上 demo，也仍与训练不完全一致**：训练侧 demo 段与 exec 段在 h5 里逐位相同
   （整条成功轨迹复制而来），模型能直接「抄」自己待会儿要走的轨迹；评测侧 demo 只能是专家轨迹。
   这是训练数据构造本身的性质，评测侧修不掉。
5. **逐任务 / 逐难度的两档差异多在 ±5 pp 内**，与「50k 后已收敛」一致，不宜逐格解读。

## 生成的 demo 与训练同分布（实证）

| | 本轮评测侧生成（142 条） | 训练侧（1600 集实测） |
|---|---|---|
| D min | 261 | 264 |
| D mean | 631.8 | 642 |
| D max | 1107 | 1152 |
| k_demo max | **69** | **71** |

`DEMO_WINDOW_GUARD max_D=1107 max_k_demo=69 cap=80 train_observed_max=71 over_train=0`。

另有更强的一条：**demo 段第 0 帧与干净 env 的 reset 帧逐位相同**（`max_abs=0`），
两条集各验一次（`easy/156`、`hard/202`）——「同 seed / 同 spec 另起一个 env」产出的初始态
与正式评测 env 没有任何渲染漂移。

## 注入与 motion 窗的验收

最终 ckpt 这一轮跑的是 **692 条混合计划**（BinFill 142 注入 demo + 其余三任务 550 条不注入），
为此新增 `DEMO_INJECT` 闸从 `client.log` 逐条核对注入边界：

| run | 分片 | 注入合计 | DEMO_INJECT | MOTION_WINDOWS |
|---|---|---|---|---|
| `finalckpt-nomotion59999-gl` | 10 | **142** | 全 PASS | SKIP（非 motion） |
| `finalckpt-motion79999-gl` | 10 | **142** | 全 PASS | 全 PASS |
| `finalckpt-modul32frame79999-gl` | 10 | **142** | 全 PASS | SKIP（非 motion） |

三个 run 的注入数都恰好等于 BinFill 全集 142，且非 BinFill 集一条都没注入。

**降级在两档复现得完全一致**：都只触发在 `BinFill/medium/195`（es=829、k_demo=51、跑满 2001 步 timeout），
`k_max=175`、闭式与实测同为 15 次、首次触发点 `t_rel=1776`、`kept_demo=51` 全程一个没丢。
这说明降级行为**只取决于 es 与步数，与模型权重无关**。

其余分片 `k_max` 在 98–148 之间，都够不到 160——成功的集在 1200 步左右结束，活不到超预算。
**motion budget 160 在补 demo 后的实际评测中够用，降级是一道几乎用不到的保险。**

分层成功率（降级组 0/1 vs 未降级组 390/682）**只作描述、不作因果证据**：
触发降级要求该集活到很晚，早早成功的集天然进不了触发组——选择偏倚，不是处理效应。配对实验记为待办。

## 剔除的 8 条

planner 演不出来、四道正确性校验全部未通过，**全部是确定性失败**（重试两次 D 逐字相同）：

| 集 | D | used_rrt |
|---|---|---|
| BinFill/easy/186 | 479 | False |
| BinFill/medium/189 | 631 | False |
| BinFill/hard/159 | 639 | False |
| BinFill/hard/166 | 894 | False |
| BinFill/hard/181 | 929 | False |
| BinFill/hard/189 | 891 | **True** |
| BinFill/hard/190 | 1063 | **True** |
| BinFill/hard/196 | 660 | False |

失败率 5.3%（8/150）超过计划定的 5% 闸，已按计划停下来交用户裁决，
用户 2026-09-22 决定「跑 142 条，对照同子集重算」。这 8 条里三个模型**总共只成功过 1 条**
（512 在 `easy/186` 上），剔除后三家分别 +1.0 / +0.6 / +0.7 pp，影响方向一致。

**重试无效**：8 条全部两次 D 逐字相同，含两条走过 RRT\* 的——env seed 固定时 RRT\* 那条路也是确定性的。
`--retries` 下次直接设 0。

## 盲区诚实清单

- **补 demo 与帧路变化无法分离**：本轮同时改变了「有没有 demo 前缀」和「帧采样落在哪里」，没有做单因素消融。
- **「最终 ckpt」不是等步数**：512 只有 59999，另两条是 79999。
- **S7：两条评测 prompt 训练里没见过**：`hard/202` 的 goal 是
  `put one red cube and four blue cubes into the bin, then press the button to stop`，正是其一。单独立项。
- **S2：`max_steps=2000` 仍超训练分布**：`t−es > 1136` 之后 exec 窗数就超过训练全集上限 70。
- **S4：motion sidecar 离线表烧在 A100、评测在 A40**，靠 `MMEVLA_MOTION_PROV_RELAX` 放行，幅度从未测过。
- **`MMEVLA_ENC_CHUNK` 未启用**（用户 2026-09-22 决定）：分批实测全局 rel_fro 0.090%、cos_min 0.9999917，
  两项同口径都优于既有 f32/bf16 差异的 0.33% / 0.99998，但逐帧最大 rel_fro 0.41% 无同口径基线可比。
  最坏 1108 帧首批实测 179.9 秒、**不 OOM**，所以整轮不分批、与改动前逐字等价。
- **9 条 error 未补跑**（三个 run、两档均相同），计入分母按失败处理，成功率是下界。
- **50000 档的非 BinFill 550 条未重跑**，沿用 700 条那轮；最终 ckpt 档是 692 条一次跑完。

## 产物

**50000 档**：`records/`（本目录）与 `binfilldemo-{motion50k,modul32frame-50k}-gl/records/`；
未重跑的 550 条原始记录在 `primary700-*` 三个目录（那三页的 BinFill 部分已作废）。

**最终 ckpt 档**：`finalckpt-{nomotion59999,motion79999,modul32frame79999}-gl/records/`，
其中 `finalckpt-nomotion59999-gl/records/` 另存 `manifest.tsv`（30 行）、
`preseeded-errors.json`（9 条卡死集的预标清单）与 `shard-checks.json`（30 个分片的两道闸验收汇总）。

**demo 前缀库**：`v1-store/demo-prefix/binfill-3b4de03a0b46`（11.2 GB，不进 git），
`index.json` sha256 `3ef6b165cc999c6a…`，清单见 `records/demo-prefix-{manifest,index}.json`。
