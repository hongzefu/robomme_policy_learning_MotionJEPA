# motion-t2-cand — 结果（T2 candidate vs reference，严格 profile 逐位）

- **判定：`T2_EQ=PASS steps=300 batch=8 record_steps=[0, 100, 200, 299] digest_steps=[0, 1, 2, 100, 200, 299]`**（`records/g0_gate_t2.txt`，`g0_gate.py --profile t2`）。
  `scalars_hex.tsv` sha256 = `3aee70eb00da002b96d1aadb7567e131b7b8f8ca8654c52fb204717c3483fff1`，与 reference `motion-t2-ref` 的 `t2_reference_manifest.json` 记录相同；
  reference 源 YAML 由 `git show S2_BASE:<yaml>` 现场核 sha 后与 candidate 解析比较（只允许新增 `motion: enabled: false` 节）；规范化 argv 相同；
  `param_checksums` 逐叶 / `batch_digests` `per_key` / index 序列前缀（n=2472）逐位；两份日志各唯一一行 `EXIT_CODE=0`。
- **起跑**：2026-09-03 14:59:26 → 15:14:20（15 min），HEAD `7ff0a17`（clean；commitV6.5 之后、S3 合入之前——训练链路代码与 T1 的 `3b02f18` 相同，其间只有纯文档提交），
  2×RTX 6000 Ada，b8，300 步，`--dataset-path v1-store/datasets/4task-motion-40ep/framesamp`（本机 NVMe）、`perceptual-framesamp-context.yaml`（关闭态）。
- **BASELINE_ENV 三次 check 全 PASS**：reference 冻结时一次、candidate 起跑前一次（`records/precheck2.log`）、gate 前一次（`records/precheck3.log`）。
- **关闭态形制**：`n_keys=12`、`n_leaves=177`。中途抽查：前 7 步五标量 hex 与 reference 逐位相同。
- **意外**：`g0_gate.py --profile t2` 首次执行在 `git show` 处 `NameError: _REPO_ROOT`（t2 分支新加时漏定义、也漏 `import subprocess`，S2 期间该 profile 只在 worktree 内做过参数解析级验证），
  `fix:` 补上 `_REPO_ROOT = Path(__file__).resolve().parents[3]` 后重跑得 PASS；判定逻辑未改。
- **records/**：`g0_gate_t2.txt`、`scalars_hex.sha256`、`precheck2.log`、`precheck3.log`、`run_meta.json`、`env.json`、`param_checksums.jsonl`、`batch_digests.jsonl`。
