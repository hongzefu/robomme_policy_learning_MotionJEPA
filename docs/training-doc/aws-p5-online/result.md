# aws-p5-online — 结果

- **起跑**：2026-09-04 06:20:11 → 06:39:xx，HEAD `8093ebd`（clean），主进程 GPU0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`）+ sidecar GPU1（util 100%），与 t3common / 两条 T3 run 并行。
- **判定行**（`records/p5_driver.log`；报告 `records/p5_online.json`）：

```
ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0 rows_total=772 covered=772
ONLINE_START_SET=PASS steps=738
ONLINE_POS=PASS
ONLINE_ORDER=PASS steps=738
PROVENANCE=PASS
P5_ONLINE=PASS episodes=40 stub=False
```

- **A100 数字（只记录）**：sidecar 模型就绪 19.9 s（warmup 3.51 s）；单窗 ≈ **1.42–1.44 s**（`[p5] g=… 每窗 1421–1438 ms`，后续批均 1325–1395 ms），与建库探针 A2 的 1413.5 ms/窗一致；
  环境 A（Ada）同项 0.88 s/窗——不同架构不混比。
- **结论**：在线 sidecar 路（fp32 / 关 TF32 / B=1）在 A100 上与离线 8 卡建的 motion 表 772 窗逐位相同，起点集合 / pos / 交错次序 / provenance 全同；
  与 D2 / D3（离线表 vs MotionJEPA 原版）合起来构成「在线 = 离线 = 原版」三方逐位闭合。
