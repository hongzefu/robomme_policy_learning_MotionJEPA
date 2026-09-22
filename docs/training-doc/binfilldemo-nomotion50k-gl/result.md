# binfilldemo-nomotion50k-gl — 结果（BinFill 补 demo 前缀，三模型主对照表）

本轮三个 run 共用一份主结果表，放在这里；[`binfilldemo-motion50k-gl/result.md`](../binfilldemo-motion50k-gl/result.md)
与 [`binfilldemo-modul32frame-50k-gl/result.md`](../binfilldemo-modul32frame-50k-gl/result.md) 回链本页。
起跑口径见 [`launch.md`](launch.md)，计划见根目录 [`0922-binfill-demo-prefix-plan.md`](../../../0922-binfill-demo-prefix-plan.md)。

## 结论先行

**BinFill 此前的反常退化是评测输入缺了 demo 前缀段造成的，不是模型能力问题。** 补上之后：

- 三个模型全部大幅回升，且**记忆通道越宽回升越多**（1.70x / 5.25x / 6.17x）；
- 三者的**相对排序从颠倒变回与其它三个任务一致**（motion > 2048 > 512）。

**这不是「同条件下的性能改进」。** 补 demo 同时改变了喂给模型的整个输入结构，新旧两列是**两种输入条件**，
不能读成「优化带来 +65 pp」。详见下面「这些数字不能怎么用」。

## 主表：同一批 142 条，有 demo vs 无 demo

两侧分母都是 142（planner 演不出来的 8 条两侧同时剔除，见「剔除的 8 条」）。
无 demo 侧不重跑，直接从已留档的 700 条结果里按同一子集重新统计。

| 模型 | 难度 | 无 demo | 补 demo | 变化 |
|---|---|---|---|---|
| 512（8 帧 / budget 512） | easy | 22/49 = 44.9% | 38/49 = 77.6% | +32.7 pp |
| | medium | 17/49 = 34.7% | 26/49 = 53.1% | +18.4 pp |
| | hard | 4/44 = 9.1% | 9/44 = 20.5% | +11.4 pp |
| **512** | **合计** | **43/142 = 30.3%** | **73/142 = 51.4%** | **+21.1 pp（1.70x）** |
| 2048（32 帧 / budget 2048） | easy | 10/49 = 20.4% | 40/49 = 81.6% | +61.2 pp |
| | medium | 6/49 = 12.2% | 34/49 = 69.4% | +57.1 pp |
| | hard | 0/44 = 0.0% | 10/44 = 22.7% | +22.7 pp |
| **2048** | **合计** | **16/142 = 11.3%** | **84/142 = 59.2%** | **+47.9 pp（5.25x）** |
| 512+motion | easy | 13/49 = 26.5% | 44/49 = 89.8% | +63.3 pp |
| | medium | 5/49 = 10.2% | 44/49 = 89.8% | +79.6 pp |
| | hard | 0/44 = 0.0% | 23/44 = 52.3% | +52.3 pp |
| **512+motion** | **合计** | **18/142 = 12.7%** | **111/142 = 78.2%** | **+65.5 pp（6.17x）** |

三个 run 各 142 条、**零 error**，30 个分片单元全部 `rc=0`、全部 `SHARD_PASS`。

## 排序恢复：本轮最硬的一条证据

| 口径 | 排序 |
|---|---|
| 其它三个任务（RouteStick / VideoRepick / VideoUnmaskSwap，700 条留档） | motion 40.71% > 2048 34.86% > 512 26.00% |
| BinFill **无 demo** | 512 30.3% > motion 12.7% > 2048 11.3%（**颠倒**） |
| BinFill **补 demo** | motion 78.2% > 2048 59.2% > 512 51.4%（**恢复一致**） |

补 demo 之前，BinFill 是唯一一个让 motion 与 2048 输给 8 帧基线的任务；补上之后这个反常消失了。

「记忆通道越宽、被 demo 缺失掏空的比例越大、补回后回升越多」这条预测写在跑之前的计划里
（`0922-binfill-demo-prefix-plan.md` Context 一节），不是事后解释。实测回升倍数与通道宽度单调对应：
8 帧感知 1.70x、32 帧感知 5.25x、8 帧 + 160 motion 窗 6.17x。

## 这些数字不能怎么用

1. **不是同条件下的改进。** 补 demo 把喂给模型的帧序列整个换了：`even_sampling_indices` 现在在
   `[0, es+count]` 上取点，8 帧档里有 4–7 帧、32 帧档里有 16–28 帧落进 demo 段（此前是 0 帧）；
   motion 路更彻底——此前 `while s+16 <= es-1 = -1` 一次都不转，现在 demo 窗常驻 16–69 个。
   **帧路的变化量级远大于 motion 降级**，三个模型都受影响。新旧两列是两种输入条件，不是一次优化的前后。
2. **hard 的分母缩得最多。** 剔除的 8 条里 hard 占 6 条（44/50），不能把 hard 的提升读成「hard 变简单了」。
3. **即使补上 demo，也仍与训练不完全一致。** 训练侧 demo 段与 exec 段在 h5 里逐位相同
   （`_binfill_demo_deliverable` 把整条成功轨迹复制一遍），模型能直接「抄」自己待会儿要走的轨迹；
   评测侧 demo 只能是专家轨迹、exec 是策略自己的。这是训练数据构造本身的性质，评测侧修不掉。
4. **降级的分层成功率不能当因果证据。** 见下。

## 生成的 demo 与训练同分布（实证）

| | 本轮评测侧生成（142 条） | 训练侧（1600 集实测） |
|---|---|---|
| D min | 261 | 264 |
| D mean | 631.8 | 642 |
| D max | 1107 | 1152 |
| k_demo max | **69** | **71** |

`DEMO_WINDOW_GUARD max_D=1107 max_k_demo=69 cap=80 train_observed_max=71 over_train=0`——
没有任何一条超出训练见过的 demo 窗数上限。这是降级口径「demo 段全保、超额全部来自 exec 段」的前提，
本轮由实测坐实，不再是分布推断。

另有一条更强的一致性证据：**demo 段第 0 帧与干净 env 的 reset 帧逐位相同**（`max_abs=0`），
两条集各验一次（`easy/156`、`hard/202`）。说明「同 seed / 同 spec 另起一个 env」产出的初始态
与正式评测 env 没有任何渲染漂移，对应训练侧「demo 段第 0 帧 == exec 段第 0 帧」的逐位恒等。

## motion 窗与降级（只有 512+motion 这条 run 涉及）

十片全部 `MOTION_WINDOWS=PASS`。合并统计：`episodes=142 k_max=175 es_max=1107 downsampled_episodes=1`。

**真实评测里降级只触发在一条集上**：`BinFill/medium/195`（es=829、k_demo=51、跑满 2001 步 timeout）。

| | 客户端闭式 | 服务端实测 |
|---|---|---|
| k_max | 175 | 175 |
| 首次超预算 | 第 112 次推理，t_rel=1776 | `MOTION_DOWNSAMPLE step=2605` ⇒ t_rel = 2605−829 = **1776** |
| 降级次数 | 15 | 15 |

`kept_demo=51` 从头到尾一个没丢，15 次降级全部只动 exec 段（110→109 ... 124→109）。

**为什么只有一条**：触发降级要求该集活过 `16×(162−k_demo)` 步，而成功的集都在 1200 步左右结束。
窗数最多的几条（`hard/179` es=1068 k_max=140、`hard/177` k_max=130、`hard/202` k_max=128）都够不到 160。
**结论：补 demo 后 motion budget 160 在实际评测中是够用的，降级是一道几乎用不到的保险。**

**分层成功率只作描述，不作因果验收**：降级组 0/1 vs 未降级组 111/141。触发降级需要该集活到很晚，
而早早成功的集天然进不了触发组——两组本来就不可比，这是选择偏倚不是处理效应。
想真判降级有没有害，只能在同一批集上跑 `raise` 与 `resample` 两次做配对比较，而 `raise` 那侧会大量 error。
**本轮不做，记为待办。**

顺带说明降级带来的实际收益：`medium/195` 在原来的 `raise` 口径下会在第 1776 步抛异常、整集记成 `error`，
连「它失败了」这个信息都留不下；现在它完整跑到 2001 步、正常记为失败。

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

失败率 5.3%（8/150），超过计划定的 5% 闸，已按计划停下来交用户裁决，用户 2026-09-22 决定
「跑 142 条，对照同子集重算」。

剔除对对照的扰动很小：这 8 条里三个模型**总共只成功过 1 条**（512 在 `easy/186` 上），
剔除后三家分别 +1.0 / +0.6 / +0.7 pp，影响方向一致。

**重试无效**（8 条全部两次 D 逐字相同，含两条走过 RRT\* 的）：`--retries` 下次直接设 0，
本轮的 1 让每条失败集白花一倍时间。Codex 审计 P2-5 担心的是「用 easy 集测不出 RRT\* 分支」，
本轮拿到了 `hard/189`（D=891）与 `hard/190`（D=1063）两条走过 RRT\* 的失败病例，重试同样无效——
env seed 固定时 RRT\* 那条路也是确定性的。

## 盲区诚实清单

- **补 demo 与帧路变化无法分离。** 本轮同时改变了「有没有 demo 前缀」和「帧采样落在哪里」，
  没有做只改其一的消融。要归因到具体机制需要另设实验。
- **S7：两条评测 prompt 训练里没见过。** `hard/202` 的 goal 是
  `put one red cube and four blue cubes into the bin, then press the button to stop`，
  正是计划记的那两条之一。本轮未处理，单独立项。
- **S2：`max_steps=2000` 仍超训练分布。** `t−es > 1136` 之后 exec 窗数就超过训练全集上限 70，
  这对现有 550 条旧结果同样成立；降级只是把这个已存在的偏离又推远一点，不是降级引入的新问题。
- **S4：motion sidecar 离线表烧在 A100、评测在 A40**，靠 `MMEVLA_MOTION_PROV_RELAX` 放行，幅度从未测过。
- **`MMEVLA_ENC_CHUNK` 未启用**（用户 2026-09-22 决定）。分批实测全局 rel_fro 0.090%、cos_min 0.9999917，
  两项同口径都优于既有「训练 f32 逐帧编 vs 推理 bf16 整批编」的 0.33% / 0.99998
  （`docs/train-infer-consistency.md` 第七节 1），但逐帧最大 rel_fro 0.41% 无同口径基线可比。
  最坏 1108 帧首批实测 179.9 秒、**不 OOM**，所以整轮不分批、与改动前逐字等价。

## 产物

- `records/merged/{log.json,progress.json}`——本 run 合并结果；另两个 run 在各自目录下。
- `records/manifest.tsv`——30 行六列 manifest（3 run × 10 片）。
- `records/demo-prefix-{manifest,index}.json`——demo 前缀库的身份与逐集清单
  （`index.json` sha256 `3ef6b165cc999c6a…`，库根 `v1-store/demo-prefix/binfill-3b4de03a0b46`，11.2 GB，不进 git）。
- `records/enc_chunk_cmp.json`——SigLIP 分批对照实测。
- `../binfilldemo-motion50k-gl/records/merged/motion_stats.json`——142 条逐集 motion 窗统计。
- `../binfilldemo-motion50k-gl/records/maxd-inject-hard202.json`——最大 D 集的 OOM 与端到端降级验证。
