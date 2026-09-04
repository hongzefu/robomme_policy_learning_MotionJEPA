# aws-t2-cand-s100 — 结果（T2 candidate，HEAD 关闭态；含 GPU0,1 重跑与 T2 gate）

## 一、两次 candidate run

| run | 树 / commit | GPU | 时间 | `scalars_hex.tsv` sha256 | 5 次 `state_digest` |
|---|---|---|---|---|---|
| `aws-t2-cand-s100`（计划内） | HEAD `8093ebd`（clean） | 2,3 | 06:27:18 → 06:45:19 | `85b8fe376729259cf25bb3f56c409eaa55806b0b7497e6a2955cf6d2f05b9e34` | `b8be3453… / ecbe15aa… / c2a1e6b8… / 019e437f… / d26af894…` |
| `aws-t2-cand-s100-gpu01`（计划外重跑，见二节） | 树 `cbf24e9`（训练代码 == `8093ebd`；工作区有未提交**文档**，无代码改动） | 0,1 | 07:29:39 → 07:47:46 | 同上 `85b8fe37…` | 同上 |

两次 run 与 reference（`aws-t2-ref-s100`，旧码 c5925d9，GPU0,1）三者：`metrics.jsonl` 100 步 loss / grad_norm hex 逐步相同、`param_checksums.jsonl` 5 步 177 叶逐叶相同、
`batch_digests.jsonl` 7 步 per_key / sample_indices 相同、`index_sequence.json` 872 条相同（离线逐字段比对，见 `../aws-t2-ref-s100/result.md`）。

## 二、T2 gate

- **第一次**（ref GPU0,1 vs cand GPU2,3；`records/t2_gate.attempt1-gpu23.txt`）：`T2_GATE_FAIL reason=环境指纹不同: ['gpu']` → `T2_EQ=FAIL reasons=1`。
  两侧 `env.json.fingerprint` 只有 `gpu.CUDA_VISIBLE_DEVICES`（`0,1` vs `2,3`）不同，`schema / uv_lock_sha256 / packages / assets / xla / jax_config` 与 `gpu.nvidia_smi` 全同。
  计划把 ref / cand 放在不同两对卡上并行，与 gate 的「指纹逐键相等」天然冲突；**不改 gate、不改指纹采集**，在 GPU0,1 重跑一次 candidate（run 名加后缀 `-gpu01`，
  AGENTS 第 6 条的 run_name 事前确认在此为计划外补跑，事后报用户）。
- **第二次**（ref GPU0,1 vs cand-gpu01 GPU0,1；`records/t2_gate.txt`）：

```
T2_EQ=PASS steps=100 batch=8 record_steps=[0, 25, 50, 75, 99] digest_steps=[0, 1, 2, 25, 50, 75, 99]
```

  `--env-out` 用 ref 的 `check_baseline_env.py check` 输出（`BASELINE_ENV=PASS`）。

## 三、结论

环境 B（A100）上，motion 接线后的 HEAD 关闭态与接线前旧码 S2_BASE 在 40 ep 库上 100 步 × b8 训练**逐位相同**（scalars sha、5 次 TrainState、7 次输入摘要、index 前缀），
`T2_EQ=PASS`；与 `aws-a22-grad`（单步梯度 32 叶逐位）互为印证。关闭态等价的证据链在本机闭合，替代环境 A 的 T1 / A21（其锚点在本机不可得）。

## records/

`metrics.jsonl / param_checksums.jsonl / batch_digests.jsonl / index_sequence.json / run_meta.json / env.json / scalars_hex.tsv`（第一次，GPU2,3）；`rerun-gpu01/` 同一套（第二次）；
`t2_gate.attempt1-gpu23.txt`、`t2_gate.txt`。
