# v42-smoke5（闸门 N2 SMOKE5）result — 全绿 PASS

对拍输出（`records/compare_vs_g0b_r1.txt`，逐行核对）：

```
SCALARS steps=5 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=1 mismatch=0
BATCH_DIGEST rows=3 mismatch=0
BATCH_DIGEST_CANONICAL rows=3 mismatch=0
CANON_CHECK=PASS steps=3
INDEX_SEQ=PASS n=112（共同前缀逐个一致, steps≈5）
```

- 五标量（loss/grad_norm/llm_grad_norm/mem_enc_norm/param_norm）5 步 IEEE 浮点位逐位与 G0b 一致；`metrics.jsonl` 首行五键 hex 齐全（`mem_enc_norm` 在岗，R17 补位核对通过）。
- 步 0 完整 TrainState 逐叶摘要与 G0b `param_checksums.jsonl` 首行一致——初始参数树未被 V4.0–V4.2 的删除改动扰动。
- `batch_digests.jsonl` 首行 `n_keys=12`、键集无 `recur_*`/subgoal——删恒 None 键与删 subgoal「Repack 去键 + 去 pop」成对落地后，交付键集与 G0b 完全一致（None 非叶子零贡献的实证）。
- `INDEX_SEQ` n=112 ≥ 40（steps×batch），抽取顺序前缀逐个一致。

**结论**：commitV4.0–V4.2（建库域隔离 + 数据链单一化 + transforms 瘦身）合计对训练交付面与前 5 步训练标量为位级恒等变换。N2 PASS，放行 V4.3（模型侧单一化）。

附：闸门 N1（commitV4.1 tip，RUN_TAG=v41-smoke，同口径同判据）亦全绿——`SCALARS steps=5 hex_mismatch_steps=0`、`INDEX_SEQ=PASS n=112`、`STATE_DIGEST rows=1 mismatch=0`、`BATCH_DIGEST rows=3 mismatch=0`、`CANON_CHECK=PASS steps=3`，n_keys=12。N1 不在计划九节留档表内，判定行随本档与 commitV4.2 body 记录。
