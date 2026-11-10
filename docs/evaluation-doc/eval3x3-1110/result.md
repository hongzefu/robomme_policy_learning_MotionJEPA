# 3 checkpoint × 3 模型 seed 官方 800 局评测：逐项结果

协议见论文仓 `motionjepa-paper-CoRLWorkshop/docs/1109-eval-3ckpt-3seed-protocol-todo.md`：官方 test 16 任务 × 50 局 = 800 局（清单签封 `991c3496…`），官方 profile，1300 步上限，正常失败不重试，基础设施错误每局最多重试 1 次。本机 AWS 8×A100-80GB，每卡 4 worker（MEM_FRACTION 0.18），执行副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-eval3x3-f0149d89`，执行代码 `f0149d897ebf5c56a7e674e5caf747f6e0830897`（分支 `v13-eval-3ckpt-3seed-1110`）。

本文件随评测进度逐项追加；每项的逐局结果、逐任务成功率与 run 身份在 `records/<model>-<step>-seed<s>.{episodes.csv,per_task.csv,identity.json}`。

## FS1024（FrameSamp@1024）79999 · seed 42

- **结果：256/800 = 32.00%**，800 局全部完成，errors=0，retries=0（`EVAL_DONE item=fs1024-79999-seed42 success=256/800 errors=0 retries=0 status=COMPUTE_DONE completed=800`）。
- 起止：首局完成 2026-11-10 15:02:58 美东，末局完成 15:45:40 美东（队列 15:01 开跑，约 45 分钟）。
- 被评模型：run `orig100-ctx-framesamp1024-b64-80k-8gpu`（b64、80k、8 卡 DP2×FSDP4，策略配置 `mme_vla_suite`，history `perceptual-framesamp-context-1024.yaml`，无 motion encoder），checkpoint `v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp1024-b64-80k-8gpu/79999`。
- run 身份：`run_identity_sha256=da2193870bf9c98764d6b7ce3bc5a87c4f4dd1ccdd5abc0476b84000bbb99060`，`spec_sha256=e81a2fab97bfffa2e5196e0d124f458327cd1e6ee492d5b41f65754ca374b506`。
- 逐局 CSV：`records/fs1024-79999-seed42.episodes.csv`（800 行，sha256 `023a9423d60e847ce94d5676836e64c3f49bff0b2c005304d3c2a6d42d687a67`），列 `model,checkpoint,model_seed,task,episode_id,success,outcome,retries`，`episode_id` 为官方清单原始局号，可与 GL 导出（分支 `v13-gl-episode-records`）逐局对齐。

| 套件 | 任务 | 成功/50 | 成功率 |
|---|---|---|---|
| Counting | BinFill | 9 | 18.0 |
| Counting | PickXtimes | 27 | 54.0 |
| Counting | SwingXtimes | 39 | 78.0 |
| Counting | StopCube | 24 | 48.0 |
| Permanence | VideoUnmask | 11 | 22.0 |
| Permanence | ButtonUnmask | 15 | 30.0 |
| Permanence | VideoUnmaskSwap | 15 | 30.0 |
| Permanence | ButtonUnmaskSwap | 10 | 20.0 |
| Reference | PickHighlight | 9 | 18.0 |
| Reference | VideoRepick | 10 | 20.0 |
| Reference | VideoPlaceButton | 13 | 26.0 |
| Reference | VideoPlaceOrder | 14 | 28.0 |
| Imitation | MoveCube | 39 | 78.0 |
| Imitation | InsertPeg | 2 | 4.0 |
| Imitation | PatternLock | 11 | 22.0 |
| Imitation | RouteStick | 8 | 16.0 |

套件均值（逐任务等权）：Counting 49.5、Permanence 25.5、Reference 23.0、Imitation 30.0；All 32.00。

口径提醒：这是单个 checkpoint、单个模型 seed 的一次评测，不能与其他模型的 3 seed 或 3×3 均值直接比较；主表数字要等 9 次跑完后按协议取均值并做配对 bootstrap。

## fs1024 79999 · seed 7

- **结果：255/800 = 31.88%**，accepted=800，errors=0，retries=0。
- 起止：首局完成 2026-11-10 15:40:23 美东，末局完成 16:21:53 美东。
- checkpoint `/scratch/hongze/robomme_policy_learning_MotionJEPA-eval3x3-f0149d89/v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp1024-b64-80k-8gpu/79999`；`run_identity_sha256=c48054496a94b09a7f0c5b94a0ef3f93cfb0c5b08e66ae585552b98316b7351a`；执行代码 `f0149d897ebf`。
- 逐局 CSV：`records/fs1024-79999-seed7.episodes.csv`（sha256 `c38f14f905176c826ea9b070f279a258bc91cdea007d449a37beee7f3aeaf796`）。

| 套件 | 任务 | 成功/50 | 成功率 |
|---|---|---|---|
| Counting | BinFill | 10 | 20.0 |
| Counting | PickXtimes | 29 | 58.0 |
| Counting | SwingXtimes | 42 | 84.0 |
| Counting | StopCube | 18 | 36.0 |
| Permanence | VideoUnmask | 15 | 30.0 |
| Permanence | ButtonUnmask | 15 | 30.0 |
| Permanence | VideoUnmaskSwap | 11 | 22.0 |
| Permanence | ButtonUnmaskSwap | 9 | 18.0 |
| Reference | PickHighlight | 11 | 22.0 |
| Reference | VideoRepick | 8 | 16.0 |
| Reference | VideoPlaceButton | 13 | 26.0 |
| Reference | VideoPlaceOrder | 12 | 24.0 |
| Imitation | MoveCube | 42 | 84.0 |
| Imitation | InsertPeg | 0 | 0.0 |
| Imitation | PatternLock | 12 | 24.0 |
| Imitation | RouteStick | 8 | 16.0 |

套件均值（逐任务等权）：Counting 49.5、Permanence 25.0、Reference 22.0、Imitation 31.0；All 31.88。

## fs1024 79999 · seed 0

- **结果：261/800 = 32.62%**，accepted=800，errors=0，retries=0。
- 起止：首局完成 2026-11-10 16:18:57 美东，末局完成 17:00:38 美东。
- checkpoint `/scratch/hongze/robomme_policy_learning_MotionJEPA-eval3x3-f0149d89/v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp1024-b64-80k-8gpu/79999`；`run_identity_sha256=e4bb0dc6c232f323f28d616cd55e0953c4a606dbf19830b70334bc3fd2c2f015`；执行代码 `f0149d897ebf`。
- 逐局 CSV：`records/fs1024-79999-seed0.episodes.csv`（sha256 `ff2ef75f4647838ae485c632c8ece761d73a55a42fbe61284d495acd1a82e152`）。

| 套件 | 任务 | 成功/50 | 成功率 |
|---|---|---|---|
| Counting | BinFill | 12 | 24.0 |
| Counting | PickXtimes | 29 | 58.0 |
| Counting | SwingXtimes | 39 | 78.0 |
| Counting | StopCube | 21 | 42.0 |
| Permanence | VideoUnmask | 13 | 26.0 |
| Permanence | ButtonUnmask | 14 | 28.0 |
| Permanence | VideoUnmaskSwap | 13 | 26.0 |
| Permanence | ButtonUnmaskSwap | 11 | 22.0 |
| Reference | PickHighlight | 12 | 24.0 |
| Reference | VideoRepick | 10 | 20.0 |
| Reference | VideoPlaceButton | 12 | 24.0 |
| Reference | VideoPlaceOrder | 15 | 30.0 |
| Imitation | MoveCube | 42 | 84.0 |
| Imitation | InsertPeg | 1 | 2.0 |
| Imitation | PatternLock | 10 | 20.0 |
| Imitation | RouteStick | 7 | 14.0 |

套件均值（逐任务等权）：Counting 50.5、Permanence 25.5、Reference 24.5、Imitation 30.0；All 32.62。

## 汇总：FS1024 79999 三个模型 seed（42/7/0）

- All：32.00 / 31.88 / 32.62，均值 **32.17 ± 0.40**（样本标准差 ddof=1）。
- 套件均值（三 seed 平均）：Counting 49.8、Permanence 25.3、Reference 23.2、Imitation 30.3。
- 这是单个 checkpoint 的 3 seed；主表口径是 3 checkpoint × 3 seed 的 9 次均值，60000/70000 跑完后再汇总。

## fs1024 70000 · seed 42

- **结果：228/800 = 28.50%**，accepted=800，errors=0，retries=0。
- 起止：首局完成 2026-11-10 16:56:29 美东，末局完成 17:39:06 美东。
- checkpoint `/scratch/hongze/robomme_policy_learning_MotionJEPA-eval3x3-f0149d89/v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp1024-b64-80k-8gpu/70000`；`run_identity_sha256=b4f34751fe61acf8540c571f43304c322d946ff9b22550ac4723fcab5e3b6596`；执行代码 `f0149d897ebf`。
- 逐局 CSV：`records/fs1024-70000-seed42.episodes.csv`（sha256 `39dc964b6ae45c39cd4ee5664cadf230653aeecab51f3594f16b748ea0478cd0`）。

| 套件 | 任务 | 成功/50 | 成功率 |
|---|---|---|---|
| Counting | BinFill | 10 | 20.0 |
| Counting | PickXtimes | 28 | 56.0 |
| Counting | SwingXtimes | 38 | 76.0 |
| Counting | StopCube | 18 | 36.0 |
| Permanence | VideoUnmask | 12 | 24.0 |
| Permanence | ButtonUnmask | 12 | 24.0 |
| Permanence | VideoUnmaskSwap | 12 | 24.0 |
| Permanence | ButtonUnmaskSwap | 10 | 20.0 |
| Reference | PickHighlight | 8 | 16.0 |
| Reference | VideoRepick | 10 | 20.0 |
| Reference | VideoPlaceButton | 8 | 16.0 |
| Reference | VideoPlaceOrder | 13 | 26.0 |
| Imitation | MoveCube | 39 | 78.0 |
| Imitation | InsertPeg | 0 | 0.0 |
| Imitation | PatternLock | 6 | 12.0 |
| Imitation | RouteStick | 4 | 8.0 |

套件均值（逐任务等权）：Counting 47.0、Permanence 23.0、Reference 19.5、Imitation 24.5；All 28.50。

## 协议变更（2026-11-10 美东）：停在最后 checkpoint × 3 seed

用户决定不再补 3 checkpoint × 3 seed（原话：「给出方案把我们也全部改成三个run 就不用跑九次了。」「先按照现在的这个五个模型跑三次的这个现状，呃，你你直接归档就就可以了。这个其他情况下都不需要这个，呃，就就不跑三个 checkpoint 了。」）。队列于 FS1024 70000 seed 7 运行中停止（`eval_queue.py --stop`：`STOP_DONE sent=97 remaining=[]`），70000 seed 7 未完成、不计入。五个模型最后 checkpoint × 3 seed 的逐局结果集中在 `final-ckpt-3seed/`，汇总见其 `README.md`。
