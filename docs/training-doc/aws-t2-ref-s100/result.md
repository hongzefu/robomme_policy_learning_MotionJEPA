# aws-t2-ref-s100 — 结果（T2 reference，S2_BASE 旧码 c5925d9，环境 B 100 步）

- **起跑**：2026-09-04 06:39:49 → 06:58:00（18 min，其中 5 次 TrainState 摘要各 ≈167–174 s），worktree `v1-store/worktrees/s2-base` HEAD `c5925d96305f771058e2206ae89461269af9d97c`（clean），
  `PYTHONPATH=<worktree>/src`，GPU0,1，b8 / 100 步 / seed 42 / fsdp 2 / 确定性档，`DATASET_PATH=v1-store/datasets/4task-motion-40ep/framesamp`（AWS 本地 NVMe RAID）。
- **判据全过**：`EXIT_CODE=0`；`metrics.jsonl` 100 行（loss 首 0.580677 / 末 0.096956 / min 0.094782 / max 0.733888）；`param_checksums.jsonl` 步集 {0,25,50,75,99}、`n_leaves=177`，
  末值 `state=d26af894d2cae735…`；`batch_digests.jsonl` 步集 {0,1,2,25,50,75,99}、`n_keys=12`；`index_sequence.json` 872 条（sha `f8bd8d5a9720a61b…`）；
  `scalars_hex.tsv` sha256 **`85b8fe376729259cf25bb3f56c409eaa55806b0b7497e6a2955cf6d2f05b9e34`**；`check_baseline_env.py manifest` → `OK BASELINE_MANIFEST.json: 8 个产物`；`check` → **`BASELINE_ENV=PASS`**。
- **records/**：`metrics.jsonl`、`param_checksums.jsonl`、`batch_digests.jsonl`、`index_sequence.json`、`run_meta.json`、`env.json`（含 fingerprint）、`scalars_hex.tsv`、`t2_reference_manifest.json`、`BASELINE_MANIFEST.json`。
- **意外**：旧码（c5925d9）的 `bench_train_steps.py` 写的 `run_meta.json` 没有 `epoch_samples` 键（该键是 S2 后 plan 2.8 加的），收尾脚本首版按 HEAD 口径 `rm["epoch_samples"]` 报 `KeyError`；
  改为从 `framesamp/meta/store_meta.json.num_exec_samples`（11530）取，训练本体不受影响，manifest / check 补做后 PASS。
- **同机三方一致**：本 run 的 5 次 `state_digest` 与 `aws-t2-cand-s100`（HEAD 关闭态，GPU2,3）以及 `aws-t3-closed-s100`（HEAD 关闭态经驱动脚本，GPU4,5）**逐值相同**
  （`b8be3453… / ecbe15aa… / c2a1e6b8… / 019e437f… / d26af894…`）——旧码 / 新码、直跑 / 驱动脚本、三对不同的物理 GPU，100 步 TrainState 逐位同。
- T2 gate 判定见 `../aws-t2-cand-s100/result.md`。
