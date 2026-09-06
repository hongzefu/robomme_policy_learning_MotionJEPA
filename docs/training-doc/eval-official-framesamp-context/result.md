# eval-official-framesamp-context — 结果

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。执行 commit `ce2635e`（clean HEAD，代码同 `f210b40`），被评 checkpoint
**官方 MME-VLA `perceptual-framesamp-context` 80k 步权重**（HF `Yinpei/mme_vla_suite` @ `5db4d53d`，`79999.zip` sha256 `387d4bd5…`），policy 采样 seed 42，
benchmark 上游原样（test split、每任务 50 集、`max_steps=1300`）。墙钟 `2026-09-06 06:09:08 → 06:17:29`（**8.4 min**，8 worker stride 交错分片，含 2 个 worker 的一次崩溃续评）。

## 结论先行

```
EVAL_OFFICIAL_CTX=DONE tasks=4 episodes=200/200 errors=0 ButtonUnmask=17/50 VideoUnmask=17/50 ButtonUnmaskSwap=4/50 VideoUnmaskSwap=10/50 mean_rate=0.2400 timeout=0/200 fail=152/200
```

| 任务 | 官方 context 成功 / 50 | 成功率 | easy（26） | medium（12） | hard（12） | fail | timeout | 对照：我们的 `awsprod40k-b128-motion` |
|---|---|---|---|---|---|---|---|---|
| VideoUnmask | 17 | **34.0%** | 10 | 4 | 3 | 33 | 0 | 14（28.0%） |
| ButtonUnmask | 17 | **34.0%** | 11 | 6 | 0 | 33 | 0 | 13（26.0%） |
| VideoUnmaskSwap | 10 | **20.0%** | 4 | 6 | 0 | 40 | 0 | 12（24.0%） |
| ButtonUnmaskSwap | 4 | **8.0%** | 2 | 1 | 1 | 46 | 0 | 17（34.0%） |
| **四任务均值** | 48 / 200 | **24.0%** | 27/104（26.0%） | 17/48（35.4%） | 4/48（8.3%） | 152 | 0 | 56 / 200（**28.0%**） |

- **官方 context 权重在同口径下总成功率 24.0%**，我们的 40k motion run 是 28.0%。**只能并列、不能归因**：两者权重、训练配置（官方 batch 64 / 80k 步 / lr 5e-5 vs 我们 128 / 40k / 1e-4）、
  是否带 motion memory 全不同，且单 seed 下 ±4 个百分点在 200 集的采样噪声范围内（p=0.25 时 200 集标准差约 3.1 个百分点）。
- **任务分布明显不同**：官方在两个非 Swap 任务上各 34%，两个 Swap 任务只有 20% / 8%；我们的 run 四任务在 24–34% 之间较均匀，ButtonUnmaskSwap 反而最高（34%）。
  逐集交集：ButtonUnmaskSwap 我们成功的 17 集里官方只成功 2 集；VideoUnmask 两者同时成功 11 集（官方 17、我们 14）。
- **失败全是「做错」**：152 集 fail 均为环境判定的错误终止动作，**0 集 timeout**（我们的 run 有 2 集 timeout）。官方策略同样果断、不磨蹭。
- 难度：hard 4/48（8.3%）与我们的 5/48（10.4%）同量级；样本小（每格 12–26 集），不作难度趋势结论。

## 负载均衡实测（本轮目标之一）

| | 上一轮（任务 × 两半，8 片） | 本轮（stride 交错，8 worker） |
|---|---|---|
| 每片工作量 | 25 集，但 Video 类 150–203 窗 / Button 类 387–718 窗 | 24–28 集，每 worker **283–375 窗**（4 任务混合） |
| 各片墙钟 | 9.4 / 11.3 / 15.5 / 20.5 / 15.2 / 16.5 / 36.4 / 25.4 min | w2/w4/w5/w6 5.0 min，w3 6.4，w7 6.6，w0 6.3（27 集）+1.1（续评），w1 6.8（27 集）+1.1 |
| 最慢 / 均值 | 36.4 / 18.8 = **1.94** | 6.8 / 5.7 ≈ **1.2**（首段；含续评段 7.9 / 6.0 ≈ 1.3） |

stride 交错把任务级差异均摊掉了：每 worker 的窗数从上一轮的 150–718（4.8×）收窄到 283–375（1.3×），残余差异来自集内方差（w3 / w7 / w0 / w1 各多一两集长 fail）与 w0/w1 多 4 集。
本轮总墙钟 8.4 min 里起服务只占 28 s（无 sidecar、无 jit 预热等待），若无下述崩溃约 7 min。

## 一个确定性上限：单进程建 27 个仿真环境后第 28 次必崩

w0、w1（各 28 集）都在第 28 次 `make_env` 时抛 `RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`（SAPIEN/Vulkan 实例创建失败，`svulkan2` 先报
`Your GPU driver does not support Vulkan`），前 27 次正常；24 集的 6 个 worker 全部正常；上一轮每片 25 集、预检 8 集也正常。**两次都精确发生在第 28 次，判定为单进程内 Vulkan 资源泄漏累积到上限，
不是偶发**。处置：`eval.py` 每集落 `progress.json`，崩溃集未写入（0 error），用 `ONLY=w0` / `ONLY=w1` 重跑同一 worker，新进程各补最后 1 集（各 1.1 min，含起服务），结果不受影响。
**后续分片设计红线：每个 eval.py 进程分到的集数 ≤ 27**（8 worker × 200 集下 w0/w1 恰好 28，改成 `WORKERS=8` 且把 4 任务拆两批各起一次、或直接 10 worker 均 ≤20 即可）。
本轮 launch.md 的「盯盘项」已补此条。

## 耗时与资源（无 sidecar）

| 环节 | 本轮（官方 context，无 sidecar） | 上一轮（我们的 motion run，sidecar 同卡） |
|---|---|---|
| `add_buffer`（≤16 帧）median | **37 ms** | 1462 ms |
| 首批（demo 段整段）mean | 391–587 ms | 4.5–11.9 s |
| policy `infer` median | **66 ms** | 69 ms |
| 单 worker 起服务到端口就绪 | 28 s（8 个并行） | 约 2–4 min |
| 每卡显存 | policy 32.9 GB + 仿真 0.5–0.8 GB | policy 32.9 + sidecar 3.3 + 仿真 0.8 |

每窗从约 1.5 s 降到约 0.1 s 后，单集耗时由 16 步仿真 + 渲染主导（约 12–15 s/集）。

## 盲区诚实清单

- 单 seed、单 checkpoint（官方只发布 79999）。
- 官方 checkpoint 训练数据口径（16 任务全集 / 4 任务）未核实；本仓库 HEAD 代码与官方 `89efeaab` 的模型代码差异只核了 yaml 与参数树（`PARAM_TREE_EXACT=PASS`），
  没有做前向数值逐位对拍——若 HEAD 关闭态前向与官方代码有数值差，会体现为官方权重「吃亏」。
- 成功率来自环境 `status` 判定，未人工抽看视频（视频在各 worker 目录 `videos/`，140 MB，不进 git）。
- 与我们 run 的差异在采样噪声内，任何「谁更好」的结论都需要多 seed。

## 产物

- `records/summary.txt`（判定行 + 三张表 + 逐集 200 行）、`records/per_episode.json`（逐集 seed / 难度 / 结果 / worker + 各 worker TIMING 与墙钟——注意 w0/w1 的 `wall` 只算续评段，完整起止见下）。
- `records/evoffctx-w<k>.txt` × 8：驱动 + eval 客户端日志（w0/w1 含首段 `EVAL_RC=1` 崩溃栈与续评段 `EXIT_CODE=0`）；`records/shard_timing_summary.txt`：各 worker 起跑行、`EVAL_RC`、`TIMING_SUMMARY`。
- `records/{ckpt_download.txt,param_tree.json,test_seeds.json,preflight-w0.txt,download.sh}`（起跑前留档）。
- 合并结果 `v1-store/evaluation/official-framesamp-context/ckpt79999/seed42/{progress.json,log.json,shards.json}`；预检产物 `official-framesamp-context-pre-w0/`（不参与合并）。
- server 全量日志 `v1-store/logs/evoffctx-w<k>.server.log`（不进 git）。

## 逐集（编号 = test 元数据 episode；seed / 难度见 `records/test_seeds.json`）

- **VideoUnmask**：success 17 集 = `[3, 4, 5, 6, 9, 15, 16, 19, 20, 21, 22, 25, 29, 30, 34, 37, 45]`；其余 fail。
- **ButtonUnmask**：success 17 集 = `[2, 5, 6, 13, 14, 16, 18, 20, 24, 28, 29, 30, 33, 36, 37, 40, 42]`；其余 fail。
- **VideoUnmaskSwap**：success 10 集 = `[6, 8, 10, 12, 13, 26, 28, 30, 34, 46]`；其余 fail。
- **ButtonUnmaskSwap**：success 4 集 = `[8, 12, 18, 47]`；其余 fail。
