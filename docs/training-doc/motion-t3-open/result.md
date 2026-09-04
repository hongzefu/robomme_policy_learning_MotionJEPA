# motion-t3-open — 结果（T3 开启态 1000 步；本文件同时归档 T3 跨侧判定与在线观察）

## 一、run 本体

- **起跑**：2026-09-03 16:47:45 → 17:40:56（53 min），HEAD `702009c`（clean），2×RTX 6000 Ada，b8，1000 步，`perceptual-framesamp-context-motion.yaml`（开启态），
  `DATASET_PATH=v1-store/datasets/4task-motion-40ep/framesamp`（本机 NVMe），确定性档；`BENCH_PASS`、`EXIT_CODE=0`；稳态 `2.112s/step (n=929, p10=2.018, p90=2.288)`（closed 1.991；仅记录）。
- **最终 checkpoint**：run 目录 `v1-store/train-runs/motion-t3-open/mme_vla_suite/motion-t3-open/999`（EMA，11 GB），由修补后的驱动保留（`保留 run 目录（BENCH_SAVE_FINAL_CKPT=1 …）`）。
- **records/**：`metrics.jsonl`、`param_checksums.jsonl`、`batch_digests.jsonl`、`index_sequence.json`、`run_meta.json`、`env.json`、`final_checkpoint.json`、`driver_summary.log`、
  `t3_mechanism.json` / `t3_mechanism.txt`、`t3_phase.json`（回填）、`eval/`（回填）。

## 二、T3 硬闸（`scripts/training/tests/motion_gates_model.py`）

| 闸 | 判定行 | 备注 |
|---|---|---|
| `T3_COMMON_INIT` | `T3_COMMON_INIT=PASS common_mismatches=0 open_only_params=4 open_only_ema=4 open_only_opt=8 closed_only=0 n_leaves_closed=177 n_leaves_open=193` | 起跑前，两态同进程初始化；reference `v1-store/reports/motion/t3_common_init_reference.json` |
| `T3_INIT_MATCH` | `T3_INIT_MATCH=PASS`（closed 177 / open 193 叶命中 reference） | 两侧 step-0 记录 |
| `T3_SMOKE` | `T3_SMOKE=PASS steps=1000 nan=0 motion_params_updated=4 n_keys=16/12 n_leaves=193/177` | open 侧 4 个 motion params 叶及其 ema / opt 共 16 叶初末态 sha 均变 |
| `T3_TRACE_PREFLIGHT` | `T3_TRACE_PREFLIGHT=PASS samples=112 empty=8 k_ge2=104 video=True` | 14 摘要步 × 8 样本覆盖空 / k≥2 / Video |
| `T3_TOKEN_TRACE` | `T3_TOKEN_TRACE=PASS steps=14 samples=112 keys=4 mismatches=0` | open 四键由 M1 oracle 重建逐位；公共 12 叶两侧同；open-only 恰四叶；前 8,000 index 相同 |
| `T3_MOTION_CAUSAL` | `T3_MOTION_CAUSAL=PASS pad_bitexact=1 emb_effect=1 pos_effect=1` | step 0 batch `[6556, 671, 8452, 3987, 10070, 3804, 8928, 2595]`，base loss 0.701339 |
| `T3_MECHANISM` | `T3_MECHANISM=PASS step=0 input_grad_ok=1 group_norms_ok=1` | 分组梯度范数：W2_content 4.21e+01 / W2_pos 5.07e+00 / W1 5.28e+00 / b1 4.59e-01 / b2 1.55e+00 / ∂motion_emb 有效位 9.01e-01 / ∂motion_pos 有效位 1.46e-01；padding 位输入梯度逐位 0 |
| `T3_PHASE_REPORT` | `T3_PHASE_REPORT samples=11530 phase0_n=738 … empty_n=640 nonempty_n=10890`（完整性：phase0 = 冷 80 + 稳 658、phase0 + other = 11530、empty + nonempty = 11530、加权均值闭合，均过） | run `motion-t3-phase`（GPU1，两侧最终 EMA ckpt 严格恢复，全部 11,530 样本、`fold_in(20260903, idx)` 固定 RNG、`compute_loss(train=False)`；`records/t3_phase.json`） |

**t3mechanism 的两处修补（`b364789`）**：① 双卡 fsdp=2 下第二次 `value_and_grad` OOM——脚本同时持有完整 TrainState 与整树梯度，改为初态校验后释放 ema / opt、
base 梯度取回四个 motion 叶即释放；② 首次 `pad_bitexact=0`，诊断发现 loss 逐位相同、59 叶只有 `['PaliGemma']['img']['embedding']['kernel']` 一叶变化，
且三档垃圾尺度（1e3 / 1 / 1e-3）与「同一 obs 连算两次」结论相同——SigLIP patch-embedding conv 的 wgrad 在 GPU 上不确定；该叶训练中被 `freeze_filter` 冻结、
`train_step` 的 `nnx.DiffState(trainable_filter)` 从不求它的梯度（T1 / T2 的 1000 步逐位从未触及）。摘要改为只覆盖 `trainable_filter` 叶（36 / 59）后 PASS。
**这不是 motion 路泄漏**：motion 相关叶与 trainable 叶在 padding 垃圾下逐位不变，loss 逐位不变。

## 三、描述性观察（无 PASS / FAIL；单 seed；ep0–9 在 encoder 训练集内，不得升级为泛化结论）

- **T3_EFFECT_OBS**：末 200 步 loss open 0.0304 / closed 0.0316（Δ −0.0012）；首步 0.5010 / 0.5920；末步 0.0300 / 0.0283；末 200 步 `mem_enc_norm` 均值 open 0.457 / closed 0.519。
  A20 观察项：open 首步 loss 低于 closed 是随机初始化的两个新层带来的初值差异，不作效果解读。
- **T3_PHASE_REPORT 均值**（20 步 loss 均值，open / closed）：phase0 0.0781 / 0.0686（冷 τ<32：0.0734 / 0.0683，n=80；稳态：0.0786 / 0.0686，n=658）；other 0.0772 / 0.0706（n=10792）；
  空运动路 0.1044 / 0.1203（n=640）；非空 0.0757 / 0.0675（n=10890）。方向：除空运动路样本外 open 侧 eval-loss 均高于 closed（1000 步、单 seed，不作结论）。
- **T3_EVAL_OBS open=0.0 closed=0.0 episodes=40/40**（用户 2026-09-03 改为每任务 10 集、共 40 集；`records/eval/t3_eval_obs.json`）。
  逐任务（成功 / 集数）两侧均为 ButtonUnmask 0/10、VideoUnmask 0/10、ButtonUnmaskSwap 0/10、VideoUnmaskSwap 0/10，error 0。
  两侧 checkpoint 均只训 1000 步 × b8 = 8,000 样本（< 1 epoch），0% 是预期内的欠训练结果；本观察的信息量在「在线链路在真实 checkpoint 上跑通」与「端到端耗时口径」，不在成功率。
  执行方式：每侧按任务拆两片并行（`run_t3_eval_obs.sh` 的 `RUN_SUFFIX=-a/-b`，`MAX_EPISODES=10`，`OVERWRITE=1`），四个 policy server 同时起——closed 两片 policy 在 GPU1（各 0.38）、
  open 两片 policy 在 GPU0（各 0.40）、两个 sidecar 在 GPU1、四个仿真在 GPU0（GPU0 39.8 GB / GPU1 42.0 GB，两卡均用满）；HEAD `2e1b328`，21:04–22:2x。
  之前按 50 集/任务起的两侧评估（closed 已评 110 集、open 56 集，成功均 0）已停，日志留 `v1-store/logs/motion-t3-{closed,open}-eval.*.50ep-partial.log`，不计入。
- **端到端耗时（server 端挂钟，`records/eval/timing-*.txt`；四进程并行、仿真与 sidecar 争用下的数字，只作量级）**：

| 侧 / 分片 | add_buffer ≤16 帧（每次推理前固定开销）mean / median / p90 | 首批（整段 pre_traj）mean / max | infer（除首次 jit）mean / median / p90 |
|---|---|---|---|
| closed-a | 100 / 107 / 146 ms | 447 / 1243 ms | 292 / 309 / 387 ms |
| closed-b | 98 / 106 / 145 ms | 3944 / 12556 ms | 290 / 304 / 387 ms |
| open-a | 1741 / 1812 / 2391 ms | 4595 / 5408 ms | 98 / 79 / 156 ms |
| open-b | 1724 / 1810 / 2387 ms | 12385 / 18616 ms | 98 / 80 / 155 ms |

  读法：open 侧每批 16 帧的 add_buffer 比 closed 多约 1.64 s，即一次 sidecar 窗编码（P5 单 sidecar 独占 GPU1 时 0.88 s/窗；此处两个 sidecar 共享 GPU1、util 100%，约翻倍），
  与计划 2.6「每次 infer 前固定 +1.57 s」同量级；首批多出的时间 = demo 段窗口数 × 单窗（Swap 任务 demo 更长）。closed 侧 infer 约 290 ms 高于 open 侧 98 ms，
  是因为 closed 的两个 policy 与两个 sidecar 同在 GPU1 上争用（P5 期间无争用的单 policy infer 为 69 ms 量级），不是模型差异。

## 四、盲区诚实清单（八节）

- 单 seed、1000 步、b8：`T3_EFFECT_OBS` / `T3_PHASE_REPORT` / `T3_EVAL_OBS` 三个描述性数字都不能升级为「motion memory 有 / 无效」的结论；open 侧比 closed 多 3.3M 参数、
  首步 loss 更低是初始化差异。
- ep0–9 在 MotionJEPA encoder 的训练集内（holdout 90–99），40 ep 库与在线评估都含这些 episode。
- 在线耗时口径受并行干扰：四进程 + 两 sidecar + 四仿真同机，绝对值只作量级；无争用口径见 P5（单窗 0.88 s）与 P5 期间的单 policy infer。
- closed 侧最终 checkpoint 由看门狗硬链接保出（驱动脚本旧版会 rm）；文件内容与 orbax 落盘一致（params / assets / `_CHECKPOINT_METADATA` 齐全），但未与「未被 rm」的原目录做逐字节比对（原目录已不存在）。
- `T3_MECHANISM` 的梯度摘要只覆盖 trainable 叶；冻结叶（SigLIP patch-embedding conv）的 wgrad 在 GPU 上不确定，这是脚本对全参数求导才暴露的现象，训练不求该叶梯度。
- 帧路的在线一致性由 `compare_online_memory.py`（S2 之前）保证，P5 帧路用零特征，本轮未在开启态重跑帧路对拍。
