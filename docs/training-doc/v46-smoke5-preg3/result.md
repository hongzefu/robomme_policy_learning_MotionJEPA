# v46-smoke5（预 G3 闸）result — 全绿 PASS

```
SCALARS steps=5 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=1 mismatch=0
BATCH_DIGEST rows=3 mismatch=0
BATCH_DIGEST_CANONICAL rows=3 mismatch=0
CANON_CHECK=PASS steps=3
INDEX_SEQ=PASS n=112（共同前缀逐个一致, steps≈5）
```

- `batch_digests.jsonl` 首行 n_keys=12；metrics 五标量 hex（含 mem_enc_norm）齐全。
- packed 库本体锚点与 G2 一致：store_meta_sha256 =3990165c…、manifest_sha256
  =20da0dfe…（前置门审计新增第 2 条——G2 与 G3 之间库未被重建）。
- 三条迁移判定行（随 commitV4.6 body 落档）：RELOCATION_REFS=PASS old_refs=0、
  RELOCATION_ROOT=PASS entries=20 mismatch=0、RELOCATION_COLLECT=PASS（135
  collected 零 error）。
- **结论**：最终布局上七刀合计对训练前 5 步位级恒等。预 G3 闸 PASS，放行 G3。
