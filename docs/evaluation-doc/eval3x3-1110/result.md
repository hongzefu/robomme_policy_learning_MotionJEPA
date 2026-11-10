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
