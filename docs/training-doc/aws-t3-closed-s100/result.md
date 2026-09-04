# aws-t3-closed-s100 — 结果（T3 关闭态 100 步）

- **起跑**：2026-09-04 06:23:15 → 06:41:30（18 min），HEAD `8093ebd`（clean），GPU4,5，`perceptual-framesamp-context.yaml`，b8 / 100 步 / seed 42 / fsdp 2 / 确定性档，
  `DATASET_PATH=v1-store/datasets/4task-motion-40ep/framesamp`（AWS 本地 NVMe RAID）。`BENCH_PASS`、`EXIT_CODE=0`。
- **驱动摘要**：`OK TrainState 摘要 5 次 @steps=[0, 25, 50, 75, 99], 末值 state=d26af894d2cae735…, 单次耗时中位 167.2s`；
  `OK 输入摘要 7 次 @steps=[0, 1, 2, 25, 50, 75, 99], raw 末值 18c0cae86094aa44…, canonical 末值 04ab679aa5ef99d1…, index 序列 872 个 sha=f8bd8d5a9720a61b…`；
  `OK loss n=100 min=0.0948 max=0.7339 末值=0.0970`；`RESULT batch=8 稳态=1.510s/step (n=72, p10=1.507, p90=1.587)`（确定性档、与三条 run 并行，**只记录**）。
- **最终 checkpoint**：`v1-store/train-runs/aws-t3-closed-s100/mme_vla_suite/aws-t3-closed-s100/999`（`final_checkpoint.json`：`checkpoint_id=999, state_step=100, param_kind=ema, n_leaves_params=59`）。
- **records/**：`metrics.jsonl`、`param_checksums.jsonl`（177 叶）、`batch_digests.jsonl`（`n_keys=12`）、`index_sequence.json`、`run_meta.json`、`env.json`、`final_checkpoint.json`。
- **与 T2 两侧逐位同**：5 次 `state_digest` 与 `aws-t2-ref-s100`（旧码）/ `aws-t2-cand-s100`（HEAD 直跑）逐值相同，loss 首 0.5807 / 末 0.0970 同值——关闭态经驱动脚本与直跑同数。
- 跨侧 T3 判定（t3verifyinit / T3_SMOKE / t3trace / t3mechanism / t3phase）统一写在 `../aws-t3-open-s100/result.md`。
