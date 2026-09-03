# motion-t2-ref result — reference 冻结（S2_BASE = c5925d9）

- **起跑 HEAD**：`c5925d96305f771058e2206ae89461269af9d97c`（porcelain 空；S2 改码前的最后一个 commit，即 `S2_BASE`）。
- **执行**：2026-09-03 14:3x–14:5x，2×RTX 6000 Ada，b8，300 步，`bench_train_steps.py` 直跑（argv 全文在 `records/run_meta.json`），
  数据根 `v1-store/datasets/4task-motion-40ep/framesamp`（本机 NVMe，`num_exec_samples=11530`）；日志 `v1-store/logs/motion-t2-ref.log` 尾行唯一 `EXIT_CODE=0`。
- **生命周期按 plan 5.2**：`env.json` 原子空壳 → `check_baseline_env.py dump --dataset <lib>/source`（指纹并入）→ tmux 直跑 →
  `project_scalars.py`（`PROJECTED rows=300 sha256=3aee70eb00da002b96d1aadb7567e131b7b8f8ca8654c52fb204717c3483fff1`）→ `jq -n` 原子写
  `t2_reference_manifest.json` → checker `manifest`（8 个产物）→ `check --baseline <records> --dataset <lib>/source` → **`BASELINE_ENV=PASS`**（`records/check1.log`）。
- **记录步集**：`param_checksums.jsonl` {0,100,200,299}、`n_leaves=177`；`batch_digests.jsonl` {0,1,2,100,200,299}、`n_keys=12`；`metrics.jsonl` 300 行；
  `index_sequence.json` n=2472（≥ 300×8 = 2400）。
- **manifest 关键字段**：`S2_BASE=c5925d9…`、源 YAML `perceptual-framesamp-context.yaml` 原始字节 sha256 `26c791751f56d789…`（`git show S2_BASE:<path>` 现算）、
  `store_meta_sha256`、`manifest_sha256 fee2777f…`、`scalars_sha256 3aee70eb…`、训练语义 argv、环境指纹（`uv.lock` sha `02cbc3ba…`、数据集抽样 digest `b1a0a186…`）。
- **消费者**：`g0_gate.py --profile t2 --reference-manifest records/t2_reference_manifest.json`（S2 后 candidate `motion-t2-cand`）；candidate 起跑前与 gate 前
  各再 `check` 一次，任一无 PASS 即 reference 失效。
- 数字口径：本机 NVMe，与 G0b（turbo NFS）不同介质，不作吞吐结论。
