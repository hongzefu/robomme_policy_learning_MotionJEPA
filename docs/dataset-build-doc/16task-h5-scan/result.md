# 16task-h5-scan 结果

**结论：三条验收全部 PASS。** 16 任务 × 100 集的逐集 `(num_timesteps, exec_start_idx)` 已扫出，
运动记忆指标按官方数据集 train split 口径换算完成。

## 运行

| 阶段 | 会话 | 日志 | 结果 |
|---|---|---|---|
| 1 · 补下 12 个 H5 | `h5dl16-all` | `v1-store/logs/h5dl16.log` | 全部完成 tasks=16 新取=12 用时 **823 s**（13.7 min） |
| 2 · 只读扫描 1600 集 | `h5scan16-all` | `v1-store/logs/h5scan16.log` | `EXIT_CODE=0`，秒级（只读 metadata，不解帧） |
| 3 · 换算记忆指标 | 前台 | — | 秒级 |

`/scratch/hongze/robomme_data_h5/` 现有 16 个 `.h5` + 16 个 `.tar.xz`，共 **530 GB**；`/scratch` 仍余 6.1 T。

## 三条验收

1. **4 个已有任务逐条相同 —— PASS。** 新清单与 `v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json`
   的 400 条按 `(h5_file, raw_ep_idx)` 对齐后 **缺失 0、不一致 0**，4 任务帧数合计 `123044` 与旧 `totals.timesteps` 相同。
2. **逐任务单集最大窗口数 —— PASS，15 项逐个吻合** `motion-memory-plan.md` 第 2.3 节（环境 A 全集实测）：
   VideoPlaceOrder 85、VideoPlaceButton 65、BinFill 64、PickXtimes 63、VideoRepick 61、RouteStick 40、
   PickHighlight 39、SwingXtimes 36、StopCube 35、InsertPeg 34、VideoUnmaskSwap 34、ButtonUnmaskSwap 33、
   PatternLock 32、ButtonUnmask 27、VideoUnmask 22。（MoveCube 30 未在 2.3 节列出，该节只列了超 32 的 12 项与 v1 四任务。）
3. **全集规模 —— PASS。** `1600 episode`、`exec 样本 476857`、`窗口行 44328（demo 16944 + exec 27384）`，
   与 2.3 节的 1600 / 476,857 / 44,328 完全一致。

**同源结论**：公开集 `Yinpei/robomme_data_h5` 与环境 A 的 `/data/hongzefu/robomme_data_h5` 是同一份数据。

## 16 任务中位集（按 `num_timesteps` 排序取中位那一集，t = 该集最后一帧）

| 组 | 任务 | demo | 中位集帧数 | es | motion token | budget | Δ 采样间隔 | 32 帧落 demo | 该档单集最大 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Counting | BinFill | 无 | 622 | 0 | 37 | 38.5% | 20.03 | 0 | 64 |
| Counting | PickXtimes | 无 | 546 | 0 | 33 | 34.4% | 17.58 | 0 | 63 |
| Counting | SwingXtimes | 无 | 445 | 0 | 26 | 27.1% | 14.32 | 0 | 36 |
| Counting | StopCube | 无 | 311 | 0 | 18 | 18.8% | 10.0 | 0 | 35 |
| Persistent | ButtonUnmask | 无 | 230 | 0 | 13 | 13.5% | 7.39 | 0 | 27 |
| Persistent | VideoUnmask | 有 | 178 | 66 | 8 | 8.3% | 5.71 | 12 | 22 |
| Persistent | VideoUnmaskSwap | 有 | 370 | 114 | 20 | 20.8% | 11.9 | 10 | 34 |
| Persistent | ButtonUnmaskSwap | 无 | 445 | 0 | 26 | 27.1% | 14.32 | 0 | 33 |
| Referential | PickHighlight | 无 | 343 | 0 | 20 | 20.8% | 11.03 | 0 | 39 |
| Referential | VideoRepick | 有 | 686 | 314 | 40 | 41.7% | 22.1 | 15 | 61 |
| Referential | VideoPlaceButton | 有 | 960 | 765 | 57 | 59.4% | 30.94 | 25 | 65 |
| Referential | VideoPlaceOrder | 有 | 1129 | 921 | 67 | 69.8% | 36.39 | 26 | 85 |
| Behavior | MoveCube | 有 | 441 | 258 | 25 | 26.0% | 14.19 | 19 | 30 |
| Behavior | InsertPeg | 有 | 477 | 240 | 26 | 27.1% | 15.35 | 16 | 34 |
| Behavior | PatternLock | 有 | 192 | 96 | 8 | 8.3% | 6.16 | 16 | 32 |
| Behavior | RouteStick | 有 | 400 | 200 | 22 | 22.9% | 12.87 | 16 | 40 |

## 四条结论

**① budget 96 在 16 任务尺度上是紧的，不是宽裕的。** v1 只看 4 个任务时最大只到 34 窗（35%），
容易得出「budget 从未用到一半」的印象；全集实况是 **VideoPlaceOrder 单集最大 85 窗 = 88.5%**，
VideoPlaceButton 65、BinFill 64、PickXtimes 63 紧随。零截断契约（`k > B` 直接 raise）的余量只剩 11 窗。

**② demo 段可以占到一集的八成。** VideoPlaceOrder 中位集 1129 帧里 **921 帧是 demo**（81.6%），
VideoPlaceButton 765/960（79.7%）。这两个任务的记忆几乎全部来自示范回看，而不是自己的执行过程。

**③ 帧路采样间隔跨任务差 6.4 倍。** Δ = t/31 由轨迹长度决定：VideoUnmask 5.7 帧最密，
VideoPlaceOrder 36.4 帧最疏。32 帧预算固定，长任务的时间分辨率成比例退化——
VideoPlaceOrder 的 32 个采样帧里还有 26 个落在 demo 段，真正看自己执行的只剩 6 帧。

**④ 两条路的稀疏方向相反。** 运动路 stride 固定 16、任务越长窗口越多（覆盖不掉队）；
帧路预算固定 32、任务越长采样越稀（覆盖在退化）。所以运动记忆的相对价值在长任务上最高，
而长任务恰是帧路最看不清的地方。

## hard 难度档（每任务 25 集）

官方每任务 100 集固定切成 **easy 50 / medium 25 / hard 25**（读 `episode_<i>/setup/difficulty`，
全集 800 / 400 / 400）。难度直接决定轨迹长度，因而决定采样间隔与窗口数。

| 组 | 任务 | demo | 中位集帧数 | es | motion token | budget | Δ 采样间隔 | 32 帧落 demo | 该档单集最大 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Counting | BinFill | 无 | 868 | 0 | 53 | 55.2% | 27.97 | 0 | 64 |
| Counting | PickXtimes | 无 | 812 | 0 | 49 | 51.0% | 26.16 | 0 | 63 |
| Counting | SwingXtimes | 无 | 488 | 0 | 29 | 30.2% | 15.71 | 0 | 34 |
| Counting | StopCube | 无 | 309 | 0 | 18 | 18.8% | 9.94 | 0 | 27 |
| Persistent | ButtonUnmask | 无 | 373 | 0 | 22 | 22.9% | 12.0 | 0 | 27 |
| Persistent | VideoUnmask | 有 | 329 | 66 | 18 | 18.8% | 10.58 | 7 | 22 |
| Persistent | VideoUnmaskSwap | 有 | 457 | 216 | 26 | 27.1% | 14.71 | 15 | 34 |
| Persistent | ButtonUnmaskSwap | 无 | 461 | 0 | 27 | 28.1% | 14.84 | 0 | 33 |
| Referential | PickHighlight | 无 | 539 | 0 | 32 | 33.3% | 17.35 | 0 | 39 |
| Referential | VideoRepick | 有 | 543 | 169 | 31 | 32.3% | 17.48 | 10 | 61 |
| Referential | VideoPlaceButton | 有 | 961 | 757 | 57 | 59.4% | 30.97 | 25 | 62 |
| Referential | VideoPlaceOrder | 有 | 1115 | 923 | 66 | 68.8% | 35.94 | 26 | 85 |
| Behavior | MoveCube | 有 | 416 | 238 | 23 | 24.0% | 13.39 | 18 | 29 |
| Behavior | InsertPeg | 有 | 464 | 232 | 26 | 27.1% | 14.94 | 16 | 32 |
| Behavior | PatternLock | 有 | 324 | 162 | 18 | 18.8% | 10.42 | 16 | 32 |
| Behavior | RouteStick | 有 | 500 | 250 | 28 | 29.2% | 16.1 | 16 | 40 |

**hard 与全集中位的对照**：

| 任务 | 全部 100 集中位帧数 | hard 25 集中位帧数 | 变化 | token(all→hard) | Δ(all→hard) |
|---|---:|---:|---:|---:|---:|
| BinFill | 622 | 868 | +40% | 37 → 53 | 20.03 → 27.97 |
| PickXtimes | 546 | 812 | +49% | 33 → 49 | 17.58 → 26.16 |
| SwingXtimes | 445 | 488 | +10% | 26 → 29 | 14.32 → 15.71 |
| StopCube | 311 | 309 | -1% | 18 → 18 | 10.0 → 9.94 |
| ButtonUnmask | 230 | 373 | +62% | 13 → 22 | 7.39 → 12.0 |
| VideoUnmask | 178 | 329 | +85% | 8 → 18 | 5.71 → 10.58 |
| VideoUnmaskSwap | 370 | 457 | +24% | 20 → 26 | 11.9 → 14.71 |
| ButtonUnmaskSwap | 445 | 461 | +4% | 26 → 27 | 14.32 → 14.84 |
| PickHighlight | 343 | 539 | +57% | 20 → 32 | 11.03 → 17.35 |
| VideoRepick | 686 | 543 | -21% | 40 → 31 | 22.1 → 17.48 |
| VideoPlaceButton | 960 | 961 | +0% | 57 → 57 | 30.94 → 30.97 |
| VideoPlaceOrder | 1129 | 1115 | -1% | 67 → 66 | 36.39 → 35.94 |
| MoveCube | 441 | 416 | -6% | 25 → 23 | 14.19 → 13.39 |
| InsertPeg | 477 | 464 | -3% | 26 → 26 | 15.35 → 14.94 |
| PatternLock | 192 | 324 | +69% | 8 → 18 | 6.16 → 10.42 |
| RouteStick | 400 | 500 | +25% | 22 → 28 | 12.87 → 16.1 |

三点：

- **hard 普遍更长，但不是全部。** 涨幅最大的是 VideoUnmask（178 → 329，+85%）与 PatternLock（192 → 324，+69%）；
  而 VideoPlaceOrder 反而略降（1129 → 1115），因为它的长度主要由 demo 段决定、demo 不随难度线性变长。
- **hard 档 budget 占用上限 85/96**（VideoPlaceOrder），仍未触顶，但 BinFill 中位集就已到 53 窗（55.2%）——
  全集口径下它的中位只有 37 窗。
- **采样间隔在 hard 档整体变疏**：BinFill 20.0 → 28.0 帧、PickXtimes 17.6 → 26.2 帧。
  帧路预算固定 32，难度越高时间分辨率越差，正是最需要看清的时候。

## 帧路预算 32 vs 8，与 dataset 的 subgoal 覆盖

数据集每帧带 `info/is_subgoal_boundary` 与 `simple_subgoal`，据此把一集切成 subgoal 段
（demo 段的 subgoal 恒为 `static`；实测 demo/exec 边界本身也是一个 subgoal 边界）。
把现行 **32 帧**预算（`budget 512 // (16 × 1)`）与 **8 帧**预算（`128 // (16 × 1)`）的采样点
分别落到这些段上，看每段能不能被看到。下表为 hard 档中位集：

| 组 | 任务 | subgoal 段数 | 最短段 | Δ32 | Δ8 | 32 帧漏段 | 8 帧漏段 |
|---|---|---:|---:|---:|---:|---:|---:|
| Counting | BinFill | 10 | 42 | 27.97 | 123.86 | 0 | **2** |
| Counting | PickXtimes | 12 | 36 | 26.16 | 115.86 | 0 | **4** |
| Counting | SwingXtimes | 10 | 29 | 15.71 | 69.57 | 0 | **3** |
| Counting | StopCube | 6 | 27 | 9.94 | 44.0 | 0 | 0 |
| Persistent | ButtonUnmask | 5 | 11 | 12.0 | 53.14 | 0 | **1** |
| Persistent | VideoUnmask | 5 | 8 | 10.58 | 46.86 | 0 | 0 |
| Persistent | VideoUnmaskSwap | 5 | 10 | 14.71 | 65.14 | 0 | 0 |
| Persistent | ButtonUnmaskSwap | 6 | 12 | 14.84 | 65.71 | 0 | 0 |
| Referential | PickHighlight | 7 | 15 | 17.35 | 76.86 | 0 | 0 |
| Referential | VideoRepick | 8 | 36 | 17.48 | 77.43 | 0 | **1** |
| Referential | VideoPlaceButton | 12 | 18 | 30.97 | 137.14 | 0 | **4** |
| Referential | VideoPlaceOrder | 14 | 13 | 35.94 | 159.14 | 0 | **6** |
| Behavior | MoveCube | 6 | 12 | 13.39 | 59.29 | 0 | 0 |
| Behavior | InsertPeg | 5 | 12 | 14.94 | 66.14 | 0 | 0 |
| Behavior | PatternLock | 11 | 10 | 10.42 | 46.14 | 0 | **3** |
| Behavior | RouteStick | 11 | 7 | 16.1 | 71.29 | 0 | **3** |
| **合计** | 16 个中位集 | **133** | — | — | — | **0（0%）** | **27（20%）** |

**段的含义**：`simple_subgoal` 原文由 `short_label()` 压成「动作＋颜色/序数/方向」的短标签
（如 `pick up the container that hides the green cube` → `抓罩绿`、
`move to the nearest right target by circling around the stick clockwise` → `绕右顺`），
同一集内重名的按次序补 `·2 ·3`（PatternLock 这类反复同一动作的任务靠它区分）。
四档中位集共 80 种原文 → 65 种标签，**兜底规则 0 次命中**，原文仍保留在 `text` 字段。

**结论：32 帧一段不漏，8 帧漏掉五分之一。**
hard 档 16 个中位集共 133 个 subgoal 段，32 帧预算下**每段至少采到 1 帧（漏 0 段）**；
降到 8 帧后有 **27 段（20%）完全没有采样点**——该 subgoal 在视觉输入里没有任何证据。

原因是量级不匹配：8 帧的采样间隔 Δ = t/7 落在 44–159 帧，而 subgoal 段长中位只有几十帧。
最严重的是 VideoPlaceOrder（14 段漏 6 段，Δ8 = 159.1 帧）、PickXtimes 与 VideoPlaceButton（各漏 4 段）。
PatternLock 与 RouteStick 的段长本身就短（最短 10 / 7 帧、RouteStick 每段恒 50 帧），
Δ8 分别是 46.1 / 71.3 帧，平均要跨过 1–1.4 个完整 subgoal 才采一帧。

反过来看 32 帧预算的余量并不大：PatternLock 的 Δ32 = 10.4 帧对最短段 10 帧，
VideoPlaceOrder 的 Δ32 = 35.9 帧对最短段 13 帧——后者已经是"靠运气正好落进去"，
再压预算就会开始漏段。

## 与在线 rollout 口径的差异（不可混比）

| 任务 | 数据集示范（本轮） | eval rollout（`eval-official-framesamp-context`） |
|---|---|---|
| ButtonUnmask | 230 帧 / 13 token / Δ 7.4 | 224 步 / 13 token / Δ 7.2 |
| VideoUnmask | 178 帧 / 8 token / Δ 5.7 | 162 步 / 8 token / Δ 5.2 |
| ButtonUnmaskSwap | 445 帧 / 26 token / Δ 14.3 | 320 步 / 19 token / Δ 10.3 |
| VideoUnmaskSwap | 370 帧 / 20 token / Δ 11.9 | 210 步 / 11 token / Δ 6.8 |

示范是专家轨迹、跑到任务结束；rollout 由策略决定，成功或失败都会提前停，所以两个 Swap 任务差得最多。
**demo 前缀两边同源**：H5 的 `info/is_video_demo` 与在线的 `len(image_buffer)-1` 等价，
在 VideoUnmask（全 66）、VideoUnmaskSwap（114/168/216 三档）、ButtonUnmask（全 0）上逐值验证过。

## 产物

- `v1-store/datasets/16task-scan/episode_manifest.json` — 1600 集逐集 `(num_timesteps, exec_start_idx)`
- `v1-store/datasets/16task-scan/memory_axis_16task.json` — 16 任务记忆指标（中位集 + 分布）
- `v1-store/datasets/16task-scan/input_manifest.json` — 16 个 H5 的 size + sha256
- `v1-store/datasets/16task-scan/difficulty.json` — 1600 集的 easy/medium/hard 标签（读自 `setup/difficulty`）
- 上述前两份的副本在本目录 `records/`
- 时序数轴页面：<https://claude.ai/code/artifact/f512d233-6c6c-41b9-96f2-41f222401d50>
  （4 任务 rollout 口径的旧图仍在 <https://claude.ai/code/artifact/d48feefc-20e0-4365-b6ed-afd9b5d70916>）
