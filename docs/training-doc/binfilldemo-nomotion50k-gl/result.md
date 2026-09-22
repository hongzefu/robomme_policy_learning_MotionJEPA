# 三模型完整结果（BinFill 补 demo 后；旧 BinFill 因训练/推理不一致作废）

本页是三个 50000 步模型的**现行权威结果**。起跑口径见 [`launch.md`](launch.md)，
计划见根目录 [`0922-binfill-demo-prefix-plan.md`](../../../0922-binfill-demo-prefix-plan.md)。
另两个 run 的独有部分见 [`binfilldemo-motion50k-gl/result.md`](../binfilldemo-motion50k-gl/result.md)
与 [`binfilldemo-modul32frame-50k-gl/result.md`](../binfilldemo-modul32frame-50k-gl/result.md)。

## 结论先行

**旧的 BinFill 评测结果（150 条，`primary700-*` 三个 run 里的 BinFill 三组）已作废**，
原因是训练/推理不一致：BinFill 是四个任务里唯一「训练有 demo 段、评测没有」的，
模型在一个它训练时从未见过的输入分布上被打分。补齐 demo 前缀重测后，
**本页数字取代此前留档的 26.00% / 34.86% / 40.71%**。

| 模型 | 旧总成绩（含作废的 BinFill） | **现行总成绩（692 条）** |
|---|---|---|
| 512（8 帧 / budget 512） | 26.00%（182/700） | **30.49%**（211/692） |
| 2048（32 帧 / budget 2048） | 34.86%（244/700） | **45.09%**（312/692） |
| 512+motion | 40.71%（285/700） | **54.62%**（378/692） |

macro 分别 0.3050 / 0.4498 / 0.5470，error 各 9 条（全在 VideoRepick / VideoUnmaskSwap，
与 BinFill 无关，三个 run 完全相同）。

非 BinFill 的 11 组（550 条）**两轮共用、一条未重跑、数字未变**：512 是 138/550 = 25.1%、
2048 是 228/550 = 41.5%、motion 是 267/550 = 48.5%。总成绩的变化全部来自 BinFill 三组。

## 总表

| 口径 | 512 | 2048 | 512+motion |
|---|---|---|---|
| **micro（成功 / 692）** | **211/692 = 30.49%** | **312/692 = 45.09%** | **378/692 = 54.62%** |
| macro（14 组均值） | 0.3050 | 0.4498 | 0.5470 |
| error | 9 | 9 | 9 |

## 14 组逐格

BinFill 三组（**粗体**）为补 demo 后的 142 条重测结果，其余 11 组沿用 700 条那轮、未重跑。

| 组 | 512 | 2048 | 512+motion |
|---|---|---|---|
| **BinFill/easy** | 38/49 = 77.6% | 40/49 = 81.6% | 44/49 = 89.8% |
| **BinFill/medium** | 26/49 = 53.1% | 34/49 = 69.4% | 44/49 = 89.8% |
| **BinFill/hard** | 9/44 = 20.5% | 10/44 = 22.7% | 23/44 = 52.3% |
| RouteStick/easy | 41/50 = 82.0% | 50/50 = 100.0% | 50/50 = 100.0% |
| RouteStick/medium | 20/50 = 40.0% | 49/50 = 98.0% | 47/50 = 94.0% |
| RouteStick/hard | 8/50 = 16.0% | 35/50 = 70.0% | 41/50 = 82.0% |
| RouteStick/xhard | 1/50 = 2.0% | 17/50 = 34.0% | 29/50 = 58.0% |
| VideoRepick/easy | 13/50 = 26.0%（err 1） | 17/50 = 34.0%（err 1） | 22/50 = 44.0%（err 1） |
| VideoRepick/medium | 16/50 = 32.0%（err 3） | 16/50 = 32.0%（err 3） | 19/50 = 38.0%（err 3） |
| VideoRepick/xhard | 10/50 = 20.0%（err 3） | 13/50 = 26.0%（err 3） | 23/50 = 46.0%（err 3） |
| VideoUnmaskSwap/easy | 8/50 = 16.0% | 9/50 = 18.0% | 8/50 = 16.0% |
| VideoUnmaskSwap/medium | 11/50 = 22.0% | 13/50 = 26.0% | 17/50 = 34.0% |
| VideoUnmaskSwap/hard | 2/50 = 4.0% | 2/50 = 4.0% | 7/50 = 14.0% |
| VideoUnmaskSwap/xhard | 8/50 = 16.0%（err 2） | 7/50 = 14.0%（err 2） | 4/50 = 8.0%（err 2） |

## 按任务

| 任务 | 512 | 2048 | 512+motion |
|---|---|---|---|
| **BinFill** | 73/142 = 51.4% | 84/142 = 59.2% | 111/142 = 78.2% |
| RouteStick | 70/200 = 35.0% | 151/200 = 75.5% | 167/200 = 83.5% |
| VideoRepick | 39/150 = 26.0% | 46/150 = 30.7% | 64/150 = 42.7% |
| VideoUnmaskSwap | 29/200 = 14.5% | 31/200 = 15.5% | 36/200 = 18.0% |

## 按难度

| 难度 | 512 | 2048 | 512+motion |
|---|---|---|---|
| easy | 100/199 = 50.3% | 116/199 = 58.3% | 124/199 = 62.3% |
| medium | 73/199 = 36.7% | 112/199 = 56.3% | 127/199 = 63.8% |
| hard | 19/144 = 13.2% | 47/144 = 32.6% | 71/144 = 49.3% |
| xhard | 19/150 = 12.7% | 37/150 = 24.7% | 56/150 = 37.3% |

## 为什么旧 BinFill 作废

BinFill 是四个评测任务里**唯一**「训练有 demo 段、评测没有」的：

- 训练侧 `generate_dataset_newseed.py::_binfill_demo_deliverable`（2026-09-11 路线 B）把整条成功轨迹
  复制一遍接在自己前面，400/400 满足 `num_timesteps == 2 × exec_start_idx`，
  `exec_start_idx` min 264 / mean 642 / max 1152；
- 评测侧 `BinFill.py::_initialize_episode` 的 `task_list` 里 `demonstration` 三处全 `False`，
  `reset()` 只返回 1 帧，`exec_start_idx = 0`。

实测佐证：三个 run 的 `client.log` 各 691 行 `exec_start_idx` + 9 条 error = 700 条计划集数，
其中**恰有 150 行为 0、全部是 BinFill**。

后果是模型在一个训练时从未出现过的输入分布上被打分，而且**记忆通道越宽、被掏空的比例越大**：
8 帧档此前 8 帧全落在 exec 段（训练时有 4–7 帧在 demo 段）、32 帧档 32 帧全落在 exec 段（训练时 16–28 帧在 demo 段）、
motion 路最彻底——`while s+16 <= es-1 = -1` 一次都不转，训练时常驻的 16–69 个 demo 窗在评测时一个都没有。

其它三个任务的训练 es 区间与评测 es 区间完全对齐（RouteStick 100–500/100–500、
VideoRepick 252–508/254–502、VideoUnmaskSwap 114–318/114–318），所以只有 BinFill 需要作废重测。

## BinFill 前后对照（同一批 142 条）

两侧分母都是 142（planner 演不出来的 8 条两侧同时剔除）。无 demo 侧不重跑，
直接从已留档的 700 条结果里按同一子集重新统计。

| 模型 | 难度 | 无 demo（作废） | 补 demo（现行） | 变化 |
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

三个 run 各 142 条、**零 error**，30 个分片单元全部 `rc=0`、全部 `SHARD_PASS`。

## 排序恢复：作废旧结果的最直接依据

| 口径 | 排序 |
|---|---|
| 其它三个任务（550 条，未重跑） | motion 48.5% > 2048 41.5% > 512 25.1% |
| BinFill **无 demo**（作废） | 512 30.3% > motion 12.7% > 2048 11.3%（**颠倒**） |
| BinFill **补 demo**（现行） | motion 78.2% > 2048 59.2% > 512 51.4%（**恢复一致**） |

补 demo 之前，BinFill 是唯一一个让 motion 与 2048 输给 8 帧基线的任务；补上之后这个反常消失。

「记忆通道越宽、被 demo 缺失掏空的比例越大、补回后回升越多」这条预测写在跑之前的计划里
（`0922-binfill-demo-prefix-plan.md` Context 一节），不是事后解释。实测回升倍数与通道宽度单调对应：
8 帧感知 1.70x、32 帧感知 5.25x、8 帧 + 160 motion 窗 6.17x。

## 这些数字不能怎么用

1. **BinFill 前后两列不是「同条件下的改进」。** 补 demo 把喂给模型的帧序列整个换了：
   `even_sampling_indices` 现在在 `[0, es+count]` 上取点，8 帧档里有 4–7 帧、32 帧档里有 16–28 帧
   落进 demo 段（此前是 0 帧）；motion 路的 demo 窗从 0 变成常驻 16–69 个。
   **帧路的变化量级远大于 motion 降级**。新旧两列是两种输入条件，不是一次优化的前后。
   作废旧结果的理由是它口径错了，不是因为新结果更好看。
2. **hard 的分母缩得最多。** 剔除的 8 条里 hard 占 6 条（44/50），不能把 hard 的提升读成「hard 变简单了」。
3. **即使补上 demo，也仍与训练不完全一致。** 训练侧 demo 段与 exec 段在 h5 里逐位相同
   （整条成功轨迹复制而来），模型能直接「抄」自己待会儿要走的轨迹；评测侧 demo 只能是专家轨迹、
   exec 是策略自己的。这是训练数据构造本身的性质，评测侧修不掉。
4. **跨任务比较仍受 692 vs 700 的分母差影响**（BinFill 少 8 条）。按任务/按难度表里已按实际分母标注。

## 生成的 demo 与训练同分布（实证）

| | 本轮评测侧生成（142 条） | 训练侧（1600 集实测） |
|---|---|---|
| D min | 261 | 264 |
| D mean | 631.8 | 642 |
| D max | 1107 | 1152 |
| k_demo max | **69** | **71** |

`DEMO_WINDOW_GUARD max_D=1107 max_k_demo=69 cap=80 train_observed_max=71 over_train=0`——
没有任何一条超出训练见过的 demo 窗数上限。

另有更强的一条：**demo 段第 0 帧与干净 env 的 reset 帧逐位相同**（`max_abs=0`），
两条集各验一次（`easy/156`、`hard/202`）。说明「同 seed / 同 spec 另起一个 env」产出的初始态
与正式评测 env 没有任何渲染漂移，对应训练侧「demo 段第 0 帧 == exec 段第 0 帧」的逐位恒等。

## motion 窗与降级（只有 512+motion 涉及）

十片全部 `MOTION_WINDOWS=PASS`。合并：`episodes=142 k_max=175 es_max=1107 downsampled_episodes=1`。

真实评测里降级只触发在 `BinFill/medium/195`（es=829、k_demo=51、跑满 2001 步 timeout）：

| | 客户端闭式 | 服务端实测 |
|---|---|---|
| k_max | 175 | 175 |
| 首次超预算 | 第 112 次推理，t_rel=1776 | `MOTION_DOWNSAMPLE step=2605` ⇒ t_rel = 2605−829 = **1776** |
| 降级次数 | 15 | 15 |

`kept_demo=51` 从头到尾一个没丢，15 次降级全部只动 exec 段。

窗数最多的几条（`hard/179` k_max=140、`hard/177` 130、`hard/202` 128）都够不到 160——
成功的集都在 1200 步左右结束，活不到超预算。
**补 demo 后 motion budget 160 在实际评测中够用，降级是一道几乎用不到的保险。**

分层成功率（降级组 0/1 vs 未降级组 111/141）**只作描述、不作因果证据**：
触发降级要求该集活到很晚，早早成功的集天然进不了触发组——这是选择偏倚不是处理效应。
配对实验（同一批集跑 `raise` 与 `resample`）记为待办。

降级带来的实际收益：`medium/195` 在原来的 `raise` 口径下会在第 1776 步抛异常、整集记成 `error`，
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

失败率 5.3%（8/150）超过计划定的 5% 闸，已按计划停下来交用户裁决，
用户 2026-09-22 决定「跑 142 条，对照同子集重算」。

剔除对对照的扰动很小：这 8 条里三个模型**总共只成功过 1 条**（512 在 `easy/186` 上），
剔除后三家分别 +1.0 / +0.6 / +0.7 pp，影响方向一致。

**重试无效**：8 条全部两次 D 逐字相同，含两条走过 RRT\* 的（`hard/189` D=891、`hard/190` D=1063）——
env seed 固定时 RRT\* 那条路也是确定性的。`--retries` 下次直接设 0。

## 盲区诚实清单

- **补 demo 与帧路变化无法分离。** 本轮同时改变了「有没有 demo 前缀」和「帧采样落在哪里」，
  没有做只改其一的消融。要归因到具体机制需要另设实验。
- **S7：两条评测 prompt 训练里没见过。** `hard/202` 的 goal 是
  `put one red cube and four blue cubes into the bin, then press the button to stop`，正是其一。单独立项。
- **S2：`max_steps=2000` 仍超训练分布。** `t−es > 1136` 之后 exec 窗数就超过训练全集上限 70，
  这对未重跑的 550 条同样成立；降级只是把这个已存在的偏离又推远一点。
- **S4：motion sidecar 离线表烧在 A100、评测在 A40**，靠 `MMEVLA_MOTION_PROV_RELAX` 放行，幅度从未测过。
- **`MMEVLA_ENC_CHUNK` 未启用**（用户 2026-09-22 决定）。分批实测全局 rel_fro 0.090%、cos_min 0.9999917，
  两项同口径都优于既有 f32/bf16 差异的 0.33% / 0.99998（`docs/train-infer-consistency.md` 第七节 1），
  但逐帧最大 rel_fro 0.41% 无同口径基线可比。最坏 1108 帧首批实测 179.9 秒、**不 OOM**，
  所以整轮不分批、与改动前逐字等价。
- **9 条 error 未补跑**（三个 run 相同，全在 VideoRepick / VideoUnmaskSwap 的 reset 卡死集），
  计入 692 的分母、按失败处理，是成功率的下界。

## 产物

- `records/merged/{log.json,progress.json}`——本 run 的 BinFill 142 条；另两个 run 在各自目录。
- `records/manifest.tsv`——30 行六列 manifest（3 run × 10 片）。
- `records/demo-prefix-{manifest,index}.json`——demo 前缀库的身份与逐集清单
  （`index.json` sha256 `3ef6b165cc999c6a…`，库根 `v1-store/demo-prefix/binfill-3b4de03a0b46`，11.2 GB，不进 git）。
- `records/enc_chunk_cmp.json`——SigLIP 分批对照实测。
- `../binfilldemo-motion50k-gl/records/merged/motion_stats.json`——142 条逐集 motion 窗统计。
- `../binfilldemo-motion50k-gl/records/maxd-inject-hard202.json`——最大 D 集的 OOM 与端到端降级验证。
- 未重跑的 550 条原始记录在 `primary700-{nomotion50k,motion50k,modul32frame-50k}-gl/`，
  那三页的 **BinFill 部分已作废**，总成绩以本页为准。
