# motion-p5-online — 结果（P5：真编码器在线链 vs 离线 motion 表）

- **判定：`P5_ONLINE=PASS episodes=40 stub=False`**（`records/run.log`、`records/p5_online.json`）：

```
ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0 rows_total=772 covered=772
ONLINE_START_SET=PASS steps=738
ONLINE_POS=PASS
ONLINE_ORDER=PASS steps=738
PROVENANCE=PASS
ENC_MS_PER_WINDOW mean=880.7 FIRST_BATCH_MS max=11145 LATER_BATCH_MS mean=826.2
```

  772 个窗口（40 集 = 20 条 Button es=0 + 10 条 es=66 + 3 条 es=114 + 6 条 es=168 + 1 条 es=216）在线 sidecar 逐窗 `(768,)` 与离线表对应行 `np.array_equal`，
  全表 772 行覆盖、零失配；738 个推理时刻的起点集合、`motion_pos`（vs `FrameSampStore.pos_rows([f])[0,0,:256]`）、四键（vs `FrameSampDataset.__getitem__`）均逐位，
  `mem_order` 全为合法置换；sidecar 握手 provenance 与 `store_meta.provenance` 按打包器 `same_keys_*` 逐键相等（客户端构造期核过，不等即 raise）。
- **起跑**：2026-09-03 15:16:57 → 15:29:03（12 min），HEAD `aef40c6`（S3 合入后，clean）；sidecar 独占 GPU1（RTX 6000 Ada，fp32 VAE / 关 TF32 / B=1 / 33 帧一次喂，
  `HF_HUB_OFFLINE=1`），主进程 jax 在 GPU0（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`，只算 4x4 pos 表；帧路零特征）；sidecar 就绪 7.0 s（含预热 1.43 s）。
- **三笔耗时（本机、不经 websocket）**：
  - 每窗（客户端夹 send / recv，含 6.49 MB 请求）：mean 880.7 ms，min 811 / max 891 ms（S0 A2 探针 0.85 s/窗同量级）；
  - 首批 demo（Video 首批 add_buffer 挂钟 = demo 段窗口全部同步编完）：es=66 → 3 窗 2.7 s、es=114 → 6 窗 5.5 s、es=168 → 9 窗 8.2 s、es=216 → 12 窗 11.1 s（≈ 0.92 s/窗线性）；
  - 每次推理前固定开销（后续每批 16 帧的 add_buffer 挂钟，恰含一窗编码 + 帧路零特征）：mean 826 ms、max 923 ms——与计划 2.6「每次 infer 前固定 +1.57 s」相比更低，
    因本脚本帧路为零特征、不含 SigLIP 编码与 `jax.device_get`；含帧路的 server 端 `add_buffer_time_ms` 在 `T3_EVAL_OBS` 时另记。
- **意外**：无。P1–P4 在合入后的主树重跑仍全 PASS。
- **records/**：`p5_online.json`（判定行、逐 episode 窗数 / 耗时、失配清单为空、sidecar provenance 全文）、`run.log`。
