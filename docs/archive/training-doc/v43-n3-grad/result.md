# v43-n3（闸门 N3 GRAD_FIXTURE）result — PASS

```
GRAD_PASS run_tag=v43-n3 records=<REPO_ROOT>/v1-store/dtype-unify/v43-n3-grad
COMPARE_GRAD=PASS kinds=3 mismatches=0
```

- 三定点 batch（mixed1 / allshort / allfull）单步 loss 浮点位、输入 batch canonical、
  32 个可训练梯度叶 sha 与固化 A 侧（v1-dtype-p5-grad）全部逐位一致。
- 结论：`has_aux=True→False` 与 stats 链整删（commitV4.3）未改变单步梯度的任何
  字节；jaxpr 变更仅为形态（二返回），数值路径逐位保持。
- 附前置：同 tip 的 v43-smoke（确定性档 STEPS=5）亦全绿——SCALARS 5 步 0 失配、
  STATE_DIGEST rows=1 mismatch=0、BATCH_DIGEST rows=3 mismatch=0、CANON_CHECK=PASS、
  INDEX_SEQ=PASS n=112，五标量 hex 键（含 mem_enc_norm）齐全。V4.3 改 jaxpr 触发
  现场重编译，跨重编译 bitwise 成立（与 D2-cold 结论一致）。
- records/：B 侧 grad_summary.json 与对拍报告 compare_vs_p5.json。
