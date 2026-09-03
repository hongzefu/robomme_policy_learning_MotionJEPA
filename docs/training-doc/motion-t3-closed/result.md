# motion-t3-closed — 结果（T3 关闭态 1000 步；跨侧判定只归档在 `motion-t3-open/result.md`）

- **起跑**：2026-09-03 15:54:14 → 16:45（约 51 min），HEAD `25e066c`（clean），2×RTX 6000 Ada，b8，1000 步，`perceptual-framesamp-context.yaml`（关闭态），
  `DATASET_PATH=v1-store/datasets/4task-motion-40ep/framesamp`（本机 NVMe），确定性档 `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`。
  `BENCH_PASS 基准完成`、`EXIT_CODE=0`；吞吐 `RESULT batch=8 稳态=1.991s/step (n=929, p10=1.882, p90=2.013)`（仅记录）。
- **T3_INIT_MATCH=PASS**（`motion_gates_model.py --gate t3verifyinit --closed-records …`）：step-0 `param_checksums` 177 叶逐叶命中 `t3_common_init_reference.json` 的 closed 侧。
- **T3_SMOKE（closed 侧项）**：1000 步 loss 全有限（nan/inf 0；首步 0.5920、末步 0.0283、末 200 步均值 0.0316）；`n_keys=12`、`n_leaves=177`；
  摘要步 {0,100,…,900,299,999} 共 12、输入摘要 14 步齐全；index 序列 8072 个（sha `6b51e7dd…`）。closed 前 300 步五标量与 T2 candidate（同库同配置同 seed）逐位相同（第 1 / 100 / 200 / 299 步人工核对）。
- **T3_TRACE_PREFLIGHT=PASS samples=112 empty=8 k_ge2=104 video=True**（用 closed 的 index_sequence 预检 14 个摘要步的 112 个样本覆盖了空运动路、k≥2 与 Video 段）。
- **最终 checkpoint**：目录 999（EMA，`final_checkpoint.json`: checkpoint_id=999, state_step=1000, param_kind=ema），11 GB，与四份 run 根快照一起保存在
  `v1-store/train-runs/motion-t3-closed-final/`。**意外**：`run_2gpu_epoch_bench.sh` 收尾会无条件 `rm -rf` run 目录（S2 期间该路径未走过 BENCH_SAVE_FINAL_CKPT=1），
  run 中途发现后挂看门狗在 `final_checkpoint.json` 落盘的下一秒 `cp -al` 硬链接保出目录 999 与快照（硬链接不受随后的 rm 影响，params/assets/_CHECKPOINT_METADATA 齐全）；
  脚本修补 `702009c` 对之后的 run（含 `motion-t3-open`）生效。
- **records/**：`metrics.jsonl`、`param_checksums.jsonl`、`batch_digests.jsonl`、`index_sequence.json`、`run_meta.json`、`env.json`、`final_checkpoint.json`、`driver_summary.log`。
- 跨侧项（`T3_TOKEN_TRACE` / `T3_MECHANISM` / `T3_PHASE_REPORT` / `T3_EFFECT_OBS` / `T3_EVAL_OBS`）见 `motion-t3-open/result.md`。
