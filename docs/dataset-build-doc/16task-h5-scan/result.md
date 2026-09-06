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

| 组 | 任务 | demo | 中位集帧数 | es | motion token | budget | Δ 采样间隔 | 32 帧落 demo | 全集单集最大 token |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Counting | BinFill | 无 | 622 | 0 | 37 | 38.5% | 20.0 | 0 | 64 |
| Counting | PickXtimes | 无 | 546 | 0 | 33 | 34.4% | 17.6 | 0 | 63 |
| Counting | SwingXtimes | 无 | 445 | 0 | 26 | 27.1% | 14.3 | 0 | 36 |
| Counting | StopCube | 无 | 311 | 0 | 18 | 18.8% | 10.0 | 0 | 35 |
| Persistent | ButtonUnmask | 无 | 230 | 0 | 13 | 13.5% | 7.4 | 0 | 27 |
| Persistent | VideoUnmask | 有 | 178 | 66 | 8 | 8.3% | 5.7 | 12 | 22 |
| Persistent | VideoUnmaskSwap | 有 | 370 | 114 | 20 | 20.8% | 11.9 | 10 | 34 |
| Persistent | ButtonUnmaskSwap | 无 | 445 | 0 | 26 | 27.1% | 14.3 | 0 | 33 |
| Referential | PickHighlight | 无 | 343 | 0 | 20 | 20.8% | 11.0 | 0 | 39 |
| Referential | VideoRepick | 有 | 686 | 314 | 40 | 41.7% | 22.1 | 15 | 61 |
| Referential | VideoPlaceButton | 有 | 960 | 765 | 57 | 59.4% | 30.9 | 25 | 65 |
| Referential | VideoPlaceOrder | 有 | 1129 | 921 | 67 | 69.8% | 36.4 | 26 | 85 |
| Behavior | MoveCube | 有 | 441 | 258 | 25 | 26.0% | 14.2 | 19 | 30 |
| Behavior | InsertPeg | 有 | 477 | 240 | 26 | 27.1% | 15.3 | 16 | 34 |
| Behavior | PatternLock | 有 | 192 | 96 | 8 | 8.3% | 6.2 | 16 | 32 |
| Behavior | RouteStick | 有 | 400 | 200 | 22 | 22.9% | 12.9 | 16 | 40 |

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
- 上述前两份的副本在本目录 `records/`
- 时序数轴页面：<https://claude.ai/code/artifact/f512d233-6c6c-41b9-96f2-41f222401d50>
  （4 任务 rollout 口径的旧图仍在 <https://claude.ai/code/artifact/d48feefc-20e0-4365-b6ed-afd9b5d70916>）
