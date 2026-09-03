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
| `T3_PHASE_REPORT` | （回填） | run `motion-t3-phase` |

**t3mechanism 的两处修补（`b364789`）**：① 双卡 fsdp=2 下第二次 `value_and_grad` OOM——脚本同时持有完整 TrainState 与整树梯度，改为初态校验后释放 ema / opt、
base 梯度取回四个 motion 叶即释放；② 首次 `pad_bitexact=0`，诊断发现 loss 逐位相同、59 叶只有 `['PaliGemma']['img']['embedding']['kernel']` 一叶变化，
且三档垃圾尺度（1e3 / 1 / 1e-3）与「同一 obs 连算两次」结论相同——SigLIP patch-embedding conv 的 wgrad 在 GPU 上不确定；该叶训练中被 `freeze_filter` 冻结、
`train_step` 的 `nnx.DiffState(trainable_filter)` 从不求它的梯度（T1 / T2 的 1000 步逐位从未触及）。摘要改为只覆盖 `trainable_filter` 叶（36 / 59）后 PASS。
**这不是 motion 路泄漏**：motion 相关叶与 trainable 叶在 padding 垃圾下逐位不变，loss 逐位不变。

## 三、描述性观察（无 PASS / FAIL；单 seed；ep0–9 在 encoder 训练集内，不得升级为泛化结论）

- **T3_EFFECT_OBS**：末 200 步 loss open 0.0304 / closed 0.0316（Δ −0.0012）；首步 0.5010 / 0.5920；末步 0.0300 / 0.0283；末 200 步 `mem_enc_norm` 均值 open 0.457 / closed 0.519。
  A20 观察项：open 首步 loss 低于 closed 是随机初始化的两个新层带来的初值差异，不作效果解读。
- **T3_PHASE_REPORT 均值**：（回填）
- **T3_EVAL_OBS**：（回填；两侧 checkpoint 均只训 1000 步 × b8 = 8,000 样本 < 1 epoch，成功率预期接近 0，观察只用于确认在线链路跑通与耗时口径）
